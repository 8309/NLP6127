"""Explicit composite continuation: historical raw pairs plus one corrective pair.

The stopped v4 source never receives replacement progress events. Exactly the
declared ten records use the newly verified equal-input corrective pair; all
other future groups retain the original sealed source order and protocol.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
from uuid import uuid4

import correct_codex_wire_batch as correction
import revalidate_codex_corrective_pair as metadata
import run_codex_annotation as native
import run_codex_cohort_wire as wire
from oil_evidence import machine_evaluation as protocol


VERSION = "codex-top5-candidate-cohort-corrective-composite-v1"


def load_history(source_run, corrective_pair, packets):
    source_run, corrective_pair = Path(source_run).resolve(), Path(corrective_pair).resolve()
    if not corrective_pair.is_relative_to(source_run / "correct"):
        raise ValueError("Use only the separately registered correction inside its original source run")
    manifest = native.read_json(source_run / "manifest.json")
    if manifest["specification"]["version"] != wire.VERSION:
        raise ValueError("Composite history requires the unchanged v4 source")
    for name, expected in manifest["immutable_sha256"].items():
        if protocol.sha256(native.contained_file(source_run, name).read_bytes()) != expected:
            raise ValueError("Original immutable v4 source changed")
    if packets != native.read_json(source_run / "packets.json"):
        raise ValueError("Composite canonical packets differ from the frozen original source bank")
    original = {stream: wire.completed_records(source_run, stream, packets) for stream in native.STREAMS}
    corrective = {stream: metadata.completed_correction_records(corrective_pair, stream) for stream in native.STREAMS}
    if any(protocol.validate_independent_output(value, packets[identifier]) for stream in corrective.values() for identifier, value in stream.items()):
        raise ValueError("Corrective final view differs from original canonical semantic packets")
    certificate = native.read_json(corrective_pair / "metadata_certificate.json")["verification"]
    ids = certificate["record_ids"]
    group = next((g for g in manifest["specification"]["groups"] if g["batch"] == "batch-0019"), None)
    if (not group or group["record_ids"] != ids or set(corrective["a"]) != set(ids)
            or set(corrective["b"]) != set(ids) or not set(ids) <= set(original["a"])
            or set(ids) & set(original["b"])):
        raise ValueError("Only the exact rejected original B0019 batch may receive the declared new pair")
    agreement = protocol.raw_agreement(packets, list(original["a"].values()), list(original["b"].values()))
    agreement.update(agreement_population="original_accepted_first_attempt_pairs_only",
        corrective_pair_excluded_record_ids=ids, rejected_native_batches=2,
        missing_rejected_outputs_not_imputed=True, subsequent_correction_is_not_original_first_attempt=True)
    baseline_path = source_run / ("first_attempt_agreement-" + protocol.sha256((source_run / "progress.jsonl").read_bytes())[:12] + ".json")
    baseline_raw = protocol.canonical_bytes(agreement)
    if baseline_path.exists():
        if baseline_path.read_bytes() != baseline_raw:
            raise ValueError("The retained original first-attempt agreement changed")
    else:
        native.write_new(baseline_path, baseline_raw)
    sources = {stream: {**wire.record_sources(source_run, stream), **metadata.correction_record_sources(corrective_pair, stream)} for stream in native.STREAMS}
    records = {stream: {**original[stream], **corrective[stream]} for stream in native.STREAMS}
    if set(records["a"]) != set(records["b"]):
        raise ValueError("Declared history still contains unmatched source records")
    ordered_ids = [identifier for identifier in manifest["specification"]["eligible_record_ids"] if identifier in records["a"]]
    protected = dict(native.read_json(source_run / "protected_inputs.json"))
    for path in source_run.rglob("*"):
        if path.is_file() and not path.is_relative_to(source_run / "correct") and not path.is_relative_to(source_run / "controlled_recoveries"):
            protected[str(path)] = protocol.sha256(path.read_bytes())
    for path in corrective_pair.rglob("*"):
        if path.is_file():
            protected[str(path)] = protocol.sha256(path.read_bytes())
    history = native.read_json(source_run / "history.json")
    protected.update(history["protected_sha256"])
    native.protected_unchanged(protected)
    return dict(source_run=str(source_run), corrective_pair=str(corrective_pair), record_ids=ordered_ids,
        corrective_record_ids=ids, original_specification=manifest["specification"], sources=sources,
        protected_sha256=protected, original_accepted_counts={stream: len(value) for stream, value in original.items()},
        original_accepted_A_preserved_but_explicitly_superseded_for_ten=ids,
        first_attempt_agreement_artifact=dict(path=str(baseline_path), sha256=protocol.sha256(baseline_raw),
            agreement_population="original_accepted_first_attempt_pairs_only"),
        corrective_pair_certificate_sha256=protocol.sha256((corrective_pair / "metadata_certificate.json").read_bytes()),
        corrective_validation_certificate=dict(path=str(corrective_pair / "metadata_certificate.json"),
            sha256=protocol.sha256((corrective_pair / "metadata_certificate.json").read_bytes()),
            validation_protocol=metadata.VERSION, validation_status="source_valid_pending_adjudication"),
        original_failures_retained=True, original_progress_rewritten=False), records


def verify_attempt(attempt, receipt, packets):
    attempt = Path(attempt).resolve()
    if receipt.get("execution_protocol") == correction.VERSION:
        directory = attempt.parent.parent
        stream = receipt["stream"]
        if attempt != directory / stream / "execution":
            raise ValueError("Corrective source attempt location changed")
        records = metadata.completed_correction_records(directory, stream)
        verified_receipt = native.read_json(attempt / "receipt.json")
        if receipt != verified_receipt or any(protocol.validate_independent_output(value, packets[key]) for key, value in records.items()):
            raise ValueError("Corrective receipt or canonical packet differs")
        return records
    return wire.verify_attempt(attempt, receipt, packets)


def completed_records(output, stream, packets):
    output = Path(output).resolve()
    history = native.read_json(output / "history.json")
    native.protected_unchanged(history["protected_sha256"])
    rebuilt, records = load_history(history["source_run"], history["corrective_pair"], packets)
    if rebuilt != history:
        raise ValueError("Composite retained original/corrective history changed")
    result = dict(records[stream])
    path = output / "progress.jsonl"
    for event in protocol.parse_jsonl(path.read_bytes(), "composite source progress") if path.exists() else []:
        if event.get("type") != "completed" or event.get("accepted") is not True or event["stream"] != stream:
            continue
        receipt_path = native.contained_file(output, event["receipt"])
        if protocol.sha256(receipt_path.read_bytes()) != event["receipt_sha256"]:
            raise ValueError("Composite native receipt changed")
        receipt = native.read_json(receipt_path)
        correction.retry.verify_raw_artifacts(receipt_path, receipt)
        if event.get("execution_protocol") != wire.VERSION:
            raise ValueError("Future composite native protocol must remain unchanged v4")
        values = verify_attempt(receipt_path.parent, receipt, packets)
        if set(values) & set(result):
            raise ValueError("An accepted historical/corrective/new record was rerun")
        result.update(values)
    return result


def record_sources(output, stream):
    output = Path(output).resolve()
    history = native.read_json(output / "history.json")
    native.protected_unchanged(history["protected_sha256"])
    sources = dict(history["sources"][stream])
    path = output / "progress.jsonl"
    for event in protocol.parse_jsonl(path.read_bytes(), "composite source progress") if path.exists() else []:
        if event.get("type") != "completed" or event.get("accepted") is not True or event["stream"] != stream:
            continue
        receipt_path = native.contained_file(output, event["receipt"])
        if protocol.sha256(receipt_path.read_bytes()) != event["receipt_sha256"]:
            raise ValueError("Composite source receipt changed")
        receipt = native.read_json(receipt_path)
        for identifier in receipt["record_ids"]:
            if identifier in sources:
                raise ValueError("An existing composite source was duplicated")
            sources[identifier] = dict(receipt=str(receipt_path), receipt_sha256=event["receipt_sha256"],
                execution_id=receipt["execution_id"], stdin_sha256=receipt["full_stdin_sha256"],
                source_run=str(output), migrated_unchanged=False, raw_encoding=wire.codec.VERSION,
                execution_protocol=wire.VERSION, mapping_sha256=receipt["mapping_sha256"])
    return sources


def pending_jobs(groups, records, limit):
    selected = []
    for group in groups:
        for stream in native.STREAMS:
            overlap = set(group["record_ids"]) & set(records[stream])
            if overlap == set(group["record_ids"]):
                continue
            if overlap:
                raise ValueError("Partially accepted sealed group cannot be regrouped")
            selected.append((stream, group))
    if type(limit) is not int or limit <= 0:
        raise ValueError("Use a positive explicit remaining-native-context bound")
    return selected[:limit]


def merged_usage(output):
    output = Path(output).resolve()
    history = native.read_json(output / "history.json")
    source = Path(history["source_run"])
    old = native.read_json(source / "history.json")
    directories = [Path(old["older_source_run"]), Path(old["source_run"]), source,
                   Path(history["corrective_pair"]), output]
    seen, totals, known = set(), {}, 0
    for directory in directories:
        path = directory / "execution_usage.jsonl"
        for value in protocol.parse_jsonl(path.read_bytes(), "actual unique original/corrective/new usage") if path.exists() else []:
            identifier = value["execution_id"]
            if not identifier or identifier in seen:
                raise ValueError("Actual execution usage is missing an ID or was duplicated")
            seen.add(identifier)
            usage = value.get("usage")
            if usage is not None:
                known += 1
                for key, count in usage.items():
                    if type(count) is int and count >= 0:
                        totals[key] = totals.get(key, 0) + count
    return dict(executions=len(seen), known_usage_executions=known, native_token_totals=totals,
        counting_unit="Unique original + failed + corrective + new execution ID; never label count",
        original_accepted_A_and_failed_B_usage_retained=True, subscription_dollar_cost=None)


def prepare(root, pool_dir, source_run, corrective_pair):
    root = Path(root).resolve()
    pool, rows, questions, protected, bindings = wire.previous.load_pool(root, pool_dir)
    source_run = Path(source_run).resolve()
    if not source_run.is_relative_to(pool / "cohort_annotations_wire"):
        raise ValueError("Composite original annotation source escapes this candidate pool")
    packets = native.read_json(source_run / "packets.json")
    history, records = load_history(source_run, corrective_pair, packets)
    original = history["original_specification"]
    if original["bindings"] != bindings or original["eligible_record_ids"] != [row["record_id"] for row in rows]:
        raise ValueError("Composite source bank, order or retrieval pool changed")
    protected.update(history["protected_sha256"])
    for path in (__file__, correction.__file__, metadata.__file__):
        protected[str(Path(path).resolve())] = protocol.sha256(Path(path).read_bytes())
    output = pool / "cohort_annotations_corrected" / ("run-" + uuid4().hex[:10])
    output.mkdir(parents=True, exist_ok=False)
    spec = {**original, "version": VERSION, "source_run": str(source_run), "corrective_pair": str(Path(corrective_pair).resolve()),
        "history_sha256": protocol.sha256(protocol.canonical_bytes(history)),
        "corrective_record_ids": history["corrective_record_ids"], "historical_labels_unchanged": True,
        "final_view_overlay_explicit": True, "original_first_attempt_agreement_separate": True,
        "implementation_sha256": {**original["implementation_sha256"], str(Path(__file__).resolve()): protocol.sha256(Path(__file__).read_bytes())}}
    values = {"history.json": history, "packets.json": packets, "groups.json": original["groups"],
        "protocol.txt": (source_run / "protocol.txt").read_bytes(), "wire_protocol.txt": (source_run / "wire_protocol.txt").read_bytes(),
        "protected_inputs.json": protected, "runner_script.py": Path(__file__).read_bytes()}
    for name, value in values.items():
        native.write_new(output / name, value)
    native.write_new(output / "manifest.json", dict(specification=spec,
        immutable_sha256={name: protocol.sha256((output / name).read_bytes()) for name in values}, provenance="machine_only_corrective_lineage"))
    return output, records


def run(output, *, max_native_contexts, executor=native.invoke_cli, timeout_seconds=900):
    output = Path(output).resolve()
    manifest = native.read_json(output / "manifest.json")
    if manifest["specification"]["version"] != VERSION:
        raise ValueError("Explicit composite continuation manifest is required")
    protected = dict(native.read_json(output / "protected_inputs.json"))
    for name, digest in manifest["immutable_sha256"].items():
        protected[str(native.contained_file(output, name))] = digest
    protected[str(output / "manifest.json")] = protocol.sha256((output / "manifest.json").read_bytes())
    native.protected_unchanged(protected)
    packets = native.read_json(output / "packets.json")
    completed = {stream: completed_records(output, stream, packets) for stream in native.STREAMS}
    groups = native.read_json(output / "groups.json")
    jobs = pending_jobs(groups, completed, max_native_contexts)
    if (output / "progress.jsonl").exists() and any(e.get("accepted") is False or e.get("type") == "failed"
            for e in protocol.parse_jsonl((output / "progress.jsonl").read_bytes())):
        raise ValueError("Composite failure retained; no automatic retry or corrective authority expansion")
    binary, environment = native.discover_native_cli(native.DEFAULT_CLI), native.child_environment()
    mcp = native.configured_mcp_names(environment)
    spec = manifest["specification"]
    if protocol.sha256(binary.read_bytes()) != spec["native_cli_sha256"] or mcp != spec["disabled_mcp_names"]:
        raise ValueError("Composite continuation native identity or tool isolation changed")
    failures, calls = [], 0
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending, index = {}, 0
        while pending or index < len(jobs):
            while not failures and not (output / "STOP_REQUESTED").exists() and index < len(jobs) and len(pending) < 2:
                stream, group = jobs[index]
                future = workers.submit(wire.execute_attempt, output, stream, group,
                    [packets[key]["payload"] for key in group["record_ids"]], binary, environment, mcp,
                    protected, spec["requested_model"], timeout_seconds, executor)
                pending[future] = (stream, group)
                calls += 1
                index += 1
            if not pending:
                break
            ready, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in ready:
                stream, group = pending.pop(future)
                result = future.result()
                attempt, receipt = result["attempt"], result["receipt"]
                if result["usage"] is not None:
                    native.append_progress(output / "execution_usage.jsonl", result["usage"])
                if (attempt / "receipt.json").exists():
                    native.append_progress(output / "progress.jsonl", dict(type="completed", accepted=receipt["accepted"],
                        stream=stream, batch=group["batch"], record_ids=group["record_ids"], execution_protocol=wire.VERSION,
                        receipt=attempt.relative_to(output).as_posix() + "/receipt.json",
                        receipt_sha256=protocol.sha256((attempt / "receipt.json").read_bytes())))
                else:
                    native.append_progress(output / "progress.jsonl", dict(type="failed", execution_protocol=wire.VERSION, **receipt))
                if receipt["accepted"]:
                    completed[stream].update(result["envelopes"])
                else:
                    failures.append(receipt)
                print(f"Composite {stream.upper()} {group['batch']}: accepted={receipt['accepted']}", flush=True)
    native.protected_unchanged(protected)
    summary = dict(output=str(output), accepted_counts={stream: len(values) for stream, values in completed.items()},
        new_native_invocations=calls, failures=failures, original_failed_batches_retained=2,
        corrected_records=len(spec["corrective_record_ids"]), original_first_attempt_agreement_separate=True,
        status="selected_composite_calls_complete" if not failures else "stopped_without_automatic_retry",
        subscription_usage=merged_usage(output),
        human_gold=False, external_paid_api_calls=0, subscription_dollar_cost=None)
    native.write_new(output / ("summary-" + uuid4().hex[:10] + ".json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("prepare")
    for name in ("canonical-root", "pool-dir", "source-run", "corrective-pair"):
        before.add_argument("--" + name, type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--max-native-contexts", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        directory, records = prepare(args.canonical_root, args.pool_dir, args.source_run, args.corrective_pair)
        print(json.dumps(dict(output=str(directory), accepted_counts={s: len(v) for s, v in records.items()}, native_calls=0)))
    else:
        result = run(args.output, max_native_contexts=args.max_native_contexts)
        raise SystemExit(1 if result["failures"] else 0)
