"""Source-only body clarification for the untouched remaining first attempts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
from uuid import uuid4

import correct_body_quote_batch as body

native, protocol = body.native, body.protocol
VERSION = "body-only-clarified-cohort-v1"
ROLE = "body_only_first_attempt"
SOURCE_COUNT, HISTORY_COUNT, TOTAL_COUNT, REMAINING_GROUPS = 610, 620, 1200, 59
engine = body.isolated("body_cohort_execution_", "run_codex_transport_recovery.py", "465d5ee77a5c9d1ef33a8791d0eb125852e543b0a5ac4aff9142d9e5015ac5c2")
engine.VERSION, engine.INSTRUCTION_SHA256, engine.clarification = VERSION, body.INSTRUCTION_SHA256, body.correction
old = body.isolated("body_cohort_history_", "run_codex_transport_recovery.py", "465d5ee77a5c9d1ef33a8791d0eb125852e543b0a5ac4aff9142d9e5015ac5c2")


def entries(path):
    return protocol.parse_jsonl(Path(path).read_bytes()) if Path(path).exists() else []


def inventory(output):
    output = Path(output).resolve()
    bound = {}
    for event in entries(output / "progress.jsonl"):
        if event.get("type") != "completed" or type(event.get("accepted")) is not bool:
            raise ValueError("Unknown durable body-cohort event")
        path = native.contained_file(output, event["receipt"])
        if str(path) in bound or protocol.sha256(path.read_bytes()) != event["receipt_sha256"]:
            raise ValueError("Duplicated or changed body-cohort receipt")
        receipt = native.read_json(path)
        if any(receipt.get(key) != event.get(key) for key in ("stream", "batch", "record_ids", "accepted")):
            raise ValueError("Body-cohort progress/receipt identity differs")
        bound[str(path)] = (event, receipt)
    actual = {str((attempt / "receipt.json").resolve()) for stream in native.STREAMS
        for attempt in (output / stream).glob("batch-*/attempt-*") if attempt.is_dir()}
    if actual != set(bound):
        raise ValueError("Undurable or interrupted body-cohort attempt requires review")
    return bound


def unique_usage(values):
    seen = set()
    for value in values:
        if not value["execution_id"] or value["execution_id"] in seen:
            raise ValueError("Native execution usage duplicated or lacks an identity")
        seen.add(value["execution_id"])
    return values


def bound_usage(values, receipts):
    """Every native attempt has exactly one matching measured-usage row."""
    by_id = {value["execution_id"]: value for value in unique_usage(values)}
    expected = {receipt["execution_id"]: receipt for receipt in receipts}
    if len(expected) != len(receipts) or set(by_id) != set(expected):
        raise ValueError("Native usage must bind exactly the complete receipt execution-ID set")
    for identifier, receipt in expected.items():
        if any(by_id[identifier].get(key) != receipt.get(key) for key in ("usage", "stream", "record_ids")):
            raise ValueError("Native usage/stream/record IDs differ from their own receipt")
    return values


def pair_usage(pair, certificate):
    receipts = []
    if set(certificate["receipts"]) != set(native.STREAMS):
        raise ValueError("Both corrective receipt streams must exist")
    for stream, proof in certificate["receipts"].items():
        path = Path(proof["path"])
        if not path.resolve().is_relative_to(Path(pair).resolve()) or protocol.sha256(path.read_bytes()) != proof["sha256"]:
            raise ValueError("Corrective usage receipt lineage changed")
        receipt = native.read_json(path)
        if receipt["stream"] != stream or receipt["record_ids"] != certificate["record_ids"] or len(receipt["record_ids"]) != 10:
            raise ValueError("Corrective usage needs the exact two complete ten-record receipts")
        receipts.append(receipt)
    return bound_usage(entries(Path(pair) / "execution_usage.jsonl"), receipts)


def load_history(source_run, correction_dir, packets):
    source, pair = Path(source_run).resolve(), Path(correction_dir).resolve()
    context = body.source_context(source)
    manifest = native.read_json(source / "manifest.json")
    spec = manifest["specification"]
    if packets != native.read_json(source / "packets.json"):
        raise ValueError("Frozen body-cohort canonical packets differ")
    own = inventory(source)
    failed = {str(Path(value["path"]).resolve()) for value in context["rejected_receipts"].values()}
    if {path for path, (event, receipt) in own.items() if not event["accepted"]} != failed:
        raise ValueError("Unknown stopped source failure cannot be imported")
    records = {stream: old.completed_records(source, stream, packets) for stream in native.STREAMS}
    sources = {stream: old.record_sources(source, stream) for stream in native.STREAMS}
    if any(len(value) != SOURCE_COUNT for value in records.values()) or set(records["a"]) != set(records["b"]):
        raise ValueError("Both unchanged accepted historical streams must be complete")
    certificate = native.read_json(pair / "certificate.json")
    pair_context = native.read_json(pair / "context.json")
    if Path(pair_context["source_run"]).resolve() != source:
        raise ValueError("Corrective pair belongs to another stopped source")
    for stream in native.STREAMS:
        values = body.completed_correction_records(pair, stream)
        if set(values) != set(context["record_ids"]) or set(values) & set(records[stream]):
            raise ValueError("Only the explicitly failed whole group may be overlaid")
        records[stream].update(values)
        proof = certificate["receipts"][stream]
        receipt = native.read_json(proof["path"])
        for rid in values:
            sources[stream][rid] = dict(receipt=proof["path"], receipt_sha256=proof["sha256"],
                execution_id=receipt["execution_id"], stdin_sha256=receipt["full_stdin_sha256"], source_run=str(pair),
                execution_protocol=body.VERSION, raw_encoding=body.codec.VERSION, annotation_role=body.ROLE,
                original_first_attempt=False, correction_certificate=str(pair / "certificate.json"),
                correction_certificate_sha256=protocol.sha256((pair / "certificate.json").read_bytes()))
    ids = spec["eligible_record_ids"]
    if (len(ids) != TOTAL_COUNT or any(len(value) != HISTORY_COUNT or set(value) != set(ids[:HISTORY_COUNT])
            for value in records.values())):
        raise ValueError("The exact paired historical source-order prefix must be retained")
    source_history = native.read_json(source / "history.json")
    own_usage = entries(source / "execution_usage.jsonl")
    bound_usage(own_usage, [receipt for event, receipt in own.values()])
    usage = unique_usage(source_history["execution_usage_records"] + own_usage + pair_usage(pair, certificate))
    protected = dict(context["protected_sha256"])
    for directory in (source, pair):
        protected.update({str(path): protocol.sha256(path.read_bytes()) for path in directory.rglob("*") if path.is_file()})
    protected[str(Path(body.__file__).resolve())] = protocol.sha256(Path(body.__file__).read_bytes())
    native.protected_unchanged(protected)
    return dict(source_run=str(source), correction_dir=str(pair), record_ids=ids[:HISTORY_COUNT], records=records,
        sources=sources, original_specification=spec, execution_usage_records=usage, protected_sha256=protected,
        original_failures_retained=True, body_corrective_records=context["record_ids"],
        body_only_first_attempt_population_separate=True), records


def future_groups(history, packets):
    groups = []
    done = set(history["record_ids"])
    for group in history["original_specification"]["groups"]:
        overlap = done.intersection(group["record_ids"])
        if overlap and overlap != set(group["record_ids"]):
            raise ValueError("Historical whole source group is only partly covered")
        if overlap:
            continue
        prepared = body.correction.prepare_source_batch([packets[rid]["payload"] for rid in group["record_ids"]])
        if prepared["actual_wire_schema_sha256"] != group["schema_sha256"] or prepared["codec"]["mapping_sha256"] != group["mapping_sha256"]:
            raise ValueError("Original complete source/schema/grouping changed")
        groups.append({**group, "full_stdin_sha256": prepared["actual_wire_stdin_sha256"],
            "full_payload_bytes": len(prepared["stdin"]) + len(protocol.canonical_bytes(prepared["schema"])),
            "annotation_role": ROLE, "original_first_attempt": True})
    if (len(groups) != REMAINING_GROUPS or [rid for group in groups for rid in group["record_ids"]]
            != history["original_specification"]["eligible_record_ids"][HISTORY_COUNT:]):
        raise ValueError("Untouched remaining source order/group coverage changed")
    return groups


def prepare(root, pool_dir, source_run, correction_dir):
    pool, rows, questions, protected, bindings = old.wire.previous.load_pool(Path(root), pool_dir)
    source = Path(source_run).resolve()
    if not source.is_relative_to(pool / "cohort_annotations_transport_recovery"):
        raise ValueError("Stopped source belongs to another candidate pool")
    packets = native.read_json(source / "packets.json")
    history, records = load_history(source, correction_dir, packets)
    original = history["original_specification"]
    if original["bindings"] != bindings or original["eligible_record_ids"] != [row["record_id"] for row in rows]:
        raise ValueError("Frozen source bank/isolation metadata changed")
    groups = future_groups(history, packets)
    history["future_groups_sha256"] = protocol.sha256(protocol.canonical_bytes(groups))
    protected.update(history["protected_sha256"])
    protected[str(Path(__file__).resolve())] = protocol.sha256(Path(__file__).read_bytes())
    output = pool / "cohort_annotations_body_only" / ("run-" + uuid4().hex[:10])
    spec = {**original, "version": VERSION, "groups": groups, "source_run": str(source),
        "correction_dir": str(Path(correction_dir).resolve()), "format_instruction_sha256": body.INSTRUCTION_SHA256,
        "historical_records_per_stream": HISTORY_COUNT, "first_attempt_population_size": TOTAL_COUNT - HISTORY_COUNT,
        "annotation_role": ROLE, "native_calls_prepared": 0, "no_old_failure_reclassification": True}
    values = {"packets.json": packets, "history.json": history, "groups.json": groups,
        "protected_inputs.json": protected, "instruction.txt": body.correction.CORRECTION_INSTRUCTION.encode(),
        "runner_script.py": Path(__file__).read_bytes()}
    for name, value in values.items():
        native.write_new(output / name, value)
    native.write_new(output / "manifest.json", dict(specification=spec,
        immutable_sha256={name: protocol.sha256((output / name).read_bytes()) for name in values}))
    return output, records


def verify_prepared(output):
    output = Path(output).resolve()
    manifest = native.read_json(output / "manifest.json")
    protected = dict(native.read_json(output / "protected_inputs.json"))
    protected.update({str(native.contained_file(output, name)): digest for name, digest in manifest["immutable_sha256"].items()})
    native.protected_unchanged(protected)
    history = native.read_json(output / "history.json")
    native.protected_unchanged(history["protected_sha256"])
    spec, original = manifest["specification"], history["original_specification"]
    original_path = Path(history["source_run"]) / "manifest.json"
    if (spec["version"] != VERSION or spec["format_instruction_sha256"] != body.INSTRUCTION_SHA256
            or spec["groups"] != native.read_json(output / "groups.json") or spec["historical_records_per_stream"] != HISTORY_COUNT
            or protocol.sha256(original_path.read_bytes()) != body.MANIFEST_SHA256
            or native.read_json(original_path)["specification"] != original
            or protocol.sha256(protocol.canonical_bytes(spec["groups"])) != history["future_groups_sha256"]
            or spec["first_attempt_population_size"] != TOTAL_COUNT - HISTORY_COUNT
            or any(spec[key] != original[key] for key in ("requested_model", "native_cli_sha256", "disabled_mcp_names", "native_command_identity", "bindings"))
            or any(set(history["records"][stream]) != set(original["eligible_record_ids"][:HISTORY_COUNT]) for stream in native.STREAMS)):
        raise ValueError("Sealed body-cohort identity or history changed")
    protected[str(output / "manifest.json")] = protocol.sha256((output / "manifest.json").read_bytes())
    return manifest, history, protected


def verify_attempt(attempt, receipt, packets):
    return engine.verify_attempt(attempt, receipt, packets)


def completed_records(output, stream, packets):
    manifest, history, protected = verify_prepared(output)
    result = dict(history["records"][stream])
    for path, (event, receipt) in inventory(output).items():
        if not event["accepted"] or event["stream"] != stream:
            continue
        values = verify_attempt(Path(path).parent, receipt, packets)
        if set(values) & set(result):
            raise ValueError("An accepted historical/body-only record was repeated")
        result.update(values)
    return result


def record_sources(output, stream):
    manifest, history, protected = verify_prepared(output)
    merged_usage(output, history)
    sources = dict(history["sources"][stream])
    for path, (event, receipt) in inventory(output).items():
        if not event["accepted"] or event["stream"] != stream:
            continue
        for rid in receipt["record_ids"]:
            if rid in sources:
                raise ValueError("A body-only source record was repeated")
            sources[rid] = dict(receipt=path, receipt_sha256=event["receipt_sha256"], execution_id=receipt["execution_id"],
                stdin_sha256=receipt["full_stdin_sha256"], source_run=str(Path(output).resolve()), execution_protocol=VERSION,
                raw_encoding=body.codec.VERSION, mapping_sha256=receipt["mapping_sha256"], annotation_role=ROLE, original_first_attempt=True)
    return sources


def merged_usage(output, history=None):
    if history is None:
        history = native.read_json(Path(output) / "history.json")
    receipts = [receipt for event, receipt in inventory(output).values()]
    own_usage = bound_usage(entries(Path(output) / "execution_usage.jsonl"), receipts)
    values = unique_usage(history["execution_usage_records"] + own_usage)
    totals, known = {}, 0
    for value in values:
        if value.get("usage") is not None:
            known += 1
            for key, count in value["usage"].items():
                if type(count) is int and count >= 0:
                    totals[key] = totals.get(key, 0) + count
    return dict(executions=len(values), known_usage_executions=known, native_token_totals=totals,
        subscription_dollar_cost=None, counting_unit="Unique original/failed/corrective/body-only native execution IDs")


def run(output, *, max_native_contexts, executor=native.invoke_cli, timeout_seconds=900):
    output = Path(output).resolve()
    manifest, history, protected = verify_prepared(output)
    own = inventory(output)
    merged_usage(output, history)
    if any(not event["accepted"] for event, receipt in own.values()):
        raise ValueError("Body-cohort failure retained; stop without automatic retry")
    packets = native.read_json(output / "packets.json")
    completed = {stream: completed_records(output, stream, packets) for stream in native.STREAMS}
    jobs = old.previous.pending_jobs(manifest["specification"]["groups"], completed, max_native_contexts)
    binary, environment = native.discover_native_cli(native.DEFAULT_CLI), native.child_environment()
    mcp = native.configured_mcp_names(environment)
    spec = manifest["specification"]
    if protocol.sha256(binary.read_bytes()) != spec["native_cli_sha256"] or mcp != spec["disabled_mcp_names"]:
        raise ValueError("Native binary/tool isolation changed")
    failures, calls = [], 0
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending, index = {}, 0
        while pending or index < len(jobs):
            while not failures and not (output / "STOP_REQUESTED").exists() and index < len(jobs) and len(pending) < 2:
                stream, group = jobs[index]
                pending[workers.submit(engine.execute_attempt, output, stream, group, packets, binary, environment, mcp,
                    protected, spec["requested_model"], timeout_seconds, executor)] = (stream, group)
                index += 1
                calls += 1
            if not pending:
                break
            ready, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in ready:
                stream, group = pending.pop(future)
                result = future.result()
                receipt, attempt = result["receipt"], result["attempt"]
                native.append_progress(output / "execution_usage.jsonl", result["usage"])
                native.append_progress(output / "progress.jsonl", dict(type="completed", accepted=receipt["accepted"], stream=stream,
                    batch=group["batch"], record_ids=group["record_ids"], receipt=attempt.relative_to(output).as_posix() + "/receipt.json",
                    receipt_sha256=protocol.sha256((attempt / "receipt.json").read_bytes())))
                if receipt["accepted"]:
                    completed[stream].update(result["envelopes"])
                else:
                    failures.append(receipt)
                print(f"Body-only {stream.upper()} {group['batch']}: accepted={receipt['accepted']}", flush=True)
    native.protected_unchanged(protected)
    completed = {stream: completed_records(output, stream, packets) for stream in native.STREAMS}
    future_ids = [rid for group in spec["groups"] for rid in group["record_ids"]]
    agreement = protocol.raw_agreement({rid: packets[rid] for rid in future_ids},
        [value for rid, value in completed["a"].items() if rid in future_ids],
        [value for rid, value in completed["b"].items() if rid in future_ids])
    agreement.update(agreement_population="body_only_first_attempts_only", planned_records=TOTAL_COUNT - HISTORY_COUNT,
        excludes_original_and_corrective_history=True)
    native.write_new(output / ("body_first_attempt_agreement-" + uuid4().hex[:10] + ".json"), agreement)
    summary = dict(output=str(output), accepted_counts={stream: len(value) for stream, value in completed.items()},
        new_native_invocations=calls, failures=failures, historical_records_reused_per_stream=HISTORY_COUNT,
        status="selected_body_calls_complete" if not failures else "stopped_without_automatic_retry",
        subscription_usage=merged_usage(output, history), subscription_dollar_cost=None, human_gold=False)
    native.write_new(output / ("summary-" + uuid4().hex[:10] + ".json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("prepare")
    for name in ("root", "pool-dir", "source-run", "correction-dir"):
        before.add_argument("--" + name, type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--prepared-directory", type=Path, required=True)
    execute.add_argument("--max-native-contexts", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        directory, records = prepare(args.root, args.pool_dir, args.source_run, args.correction_dir)
        print(json.dumps(dict(directory=str(directory), accepted_counts={s: len(v) for s, v in records.items()}, native_calls=0)))
    else:
        summary = run(args.prepared_directory, max_native_contexts=args.max_native_contexts)
        raise SystemExit(1 if summary["failures"] else 0)
