"""Resume unstarted body-first attempts after one explicit zero-call admission."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
from uuid import uuid4

import run_body_quote_cohort as previous

native, protocol = previous.native, previous.protocol
VERSION = "body-only-revalidated-continuation-v1"
MANIFEST_SHA256 = "81b7a9f16066245a7fb8323aaca82fa493210dd0af1de60ddf5d271b4b85109f"
PROGRESS_SHA256 = "e476508991e48c4eec8cdb32e2e1f1e60d67450e938867ac98414b405977ee0c"
RECEIPT_SHA256 = "3d3eb1746772754f9098c17ed8698e511a6d3494745b6ddbd497258a63df915c"
COUNTS = {"a": 746, "b": 740}
GROUP_COUNT, PENDING_CONTEXTS = 47, 93
TOTAL_COUNT = 1200
reader = previous.body.isolated("revalidated_body_reader_", "run_body_quote_cohort.py", "22756d7207c6a0e688b978a0d8ff4abdb04cccee07e93dc89762a33dcf42121d")
engine = previous.body.isolated("revalidated_body_execution_", "run_codex_transport_recovery.py", "465d5ee77a5c9d1ef33a8791d0eb125852e543b0a5ac4aff9142d9e5015ac5c2")
engine.VERSION, engine.INSTRUCTION_SHA256, engine.clarification = VERSION, previous.body.INSTRUCTION_SHA256, previous.body.correction


def load_history(source_run, certificate_dir, packets):
    import completed_503_admission as admission
    source, certificate_dir = Path(source_run).resolve(), Path(certificate_dir).resolve()
    if (protocol.sha256((source / "manifest.json").read_bytes()) != MANIFEST_SHA256
            or protocol.sha256((source / "progress.jsonl").read_bytes()) != PROGRESS_SHA256):
        raise ValueError("Only the exactly pinned stopped body run may resume")
    manifest, old_history, guards = previous.verify_prepared(source)
    if packets != native.read_json(source / "packets.json"):
        raise ValueError("Frozen source/body question packets changed")
    inventory = previous.inventory(source)
    failed = [(path, event, receipt) for path, (event, receipt) in inventory.items() if not event["accepted"]]
    if len(failed) != 1 or failed[0][1]["stream"] != "b" or failed[0][1]["batch"] != "batch-0038" or protocol.sha256(Path(failed[0][0]).read_bytes()) != RECEIPT_SHA256:
        raise ValueError("Unknown failures cannot be imported by the one registered admission")
    cert, admitted, cert_guards = admission.load("annotation_b38", certificate_dir)
    if set(admitted) != set(failed[0][2]["record_ids"]) or len(admitted) != 10:
        raise ValueError("The exact original whole B38 batch must be mechanically admitted")
    records = {stream: previous.completed_records(source, stream, packets) for stream in native.STREAMS}
    sources = {stream: previous.record_sources(source, stream) for stream in native.STREAMS}
    if len(records["a"]) != COUNTS["a"] or len(records["b"]) != COUNTS["b"] - len(admitted) or set(admitted) & set(records["b"]):
        raise ValueError("Original accepted asymmetric body coverage changed")
    records["b"].update(admitted)
    receipt = failed[0][2]
    for rid in admitted:
        sources["b"][rid] = dict(receipt=failed[0][0], receipt_sha256=RECEIPT_SHA256,
            execution_id=receipt["execution_id"], stdin_sha256=receipt["full_stdin_sha256"], source_run=str(source),
            execution_protocol=previous.VERSION, raw_encoding=previous.body.codec.VERSION,
            mapping_sha256=receipt["mapping_sha256"], annotation_role="body_only_first_attempt", original_first_attempt=True,
            validation_class="completed_output_mechanically_revalidated_with_transient_503_events",
            mechanical_certificate=str(certificate_dir / "certificate.json"),
            mechanical_certificate_sha256=protocol.sha256((certificate_dir / "certificate.json").read_bytes()),
            operational_receipt_accepted=False)
    spec = manifest["specification"]
    ids = spec["eligible_record_ids"]
    if any(len(records[stream]) != count or set(records[stream]) != set(ids[:count]) for stream, count in COUNTS.items()):
        raise ValueError("Paired740 and the unchanged six-record A-only prefix must be exact")
    usage = previous.unique_usage(old_history["execution_usage_records"] + previous.entries(source / "execution_usage.jsonl"))
    previous.merged_usage(source, old_history)
    guards.update(cert_guards)
    guards.update({str(path): protocol.sha256(path.read_bytes()) for path in source.rglob("*") if path.is_file()})
    native.protected_unchanged(guards)
    return dict(source_run=str(source), certificate_dir=str(certificate_dir), original_specification=spec,
        records=records, sources=sources, historical_counts=COUNTS, history_common_paired_ids=ids[:COUNTS["b"]],
        one_sided_record_ids={"a": ids[COUNTS["b"]:COUNTS["a"]], "b": []},
        record_ids={stream: ids[:count] for stream, count in COUNTS.items()},
        execution_usage_records=usage, protected_sha256=guards, original_false_receipt_retained=True,
        certified_original_invocation_usage_counted_once=True), records


def prepare(root, pool_dir, source_run, certificate_dir):
    pool, rows, questions, guards, bindings = previous.old.wire.previous.load_pool(Path(root), pool_dir)
    source = Path(source_run).resolve()
    if not source.is_relative_to(pool / "cohort_annotations_body_only"):
        raise ValueError("Stopped source belongs to another candidate pool")
    packets = native.read_json(source / "packets.json")
    history, records = load_history(source, certificate_dir, packets)
    original = history["original_specification"]
    if original["bindings"] != bindings or original["eligible_record_ids"] != [row["record_id"] for row in rows]:
        raise ValueError("Original full source bank/isolation changed")
    common = set(history["history_common_paired_ids"])
    groups = [group for group in original["groups"] if not set(group["record_ids"]) <= common]
    if (len(groups) != GROUP_COUNT or [rid for group in groups for rid in group["record_ids"]] != original["eligible_record_ids"][COUNTS["b"]:]
            or len(previous.old.previous.pending_jobs(groups, records, 10000)) != PENDING_CONTEXTS):
        raise ValueError("Only the exact still-unstarted per-stream whole groups may continue")
    if any(group["annotation_role"] != "body_only_first_attempt" or group["original_first_attempt"] is not True for group in groups):
        raise ValueError("Remaining contexts must be unchanged body-only first attempts")
    history["groups_sha256"] = protocol.sha256(protocol.canonical_bytes(groups))
    guards.update(history["protected_sha256"])
    guards[str(Path(__file__).resolve())] = protocol.sha256(Path(__file__).read_bytes())
    output = pool / "cohort_annotations_body_revalidated" / ("run-" + uuid4().hex[:10])
    spec = {**original, "version": VERSION, "groups": groups, "source_run": str(source),
        "certificate_dir": str(Path(certificate_dir).resolve()), "historical_counts": COUNTS,
        "format_instruction_sha256": previous.body.INSTRUCTION_SHA256, "pending_native_contexts_at_prepare": PENDING_CONTEXTS,
        "native_calls_prepared": 0, "old_receipt_still_false": True}
    values = {"history.json": history, "packets.json": packets, "groups.json": groups,
        "protected_inputs.json": guards, "runner_script.py": Path(__file__).read_bytes()}
    for name, value in values.items():
        native.write_new(output / name, value)
    native.write_new(output / "manifest.json", dict(specification=spec,
        immutable_sha256={name: protocol.sha256((output / name).read_bytes()) for name in values}))
    return output, records


def verify_prepared(output):
    output = Path(output).resolve()
    manifest = native.read_json(output / "manifest.json")
    guards = dict(native.read_json(output / "protected_inputs.json"))
    guards.update({str(native.contained_file(output, name)): digest for name, digest in manifest["immutable_sha256"].items()})
    native.protected_unchanged(guards)
    history = native.read_json(output / "history.json")
    native.protected_unchanged(history["protected_sha256"])
    spec, original = manifest["specification"], history["original_specification"]
    source_manifest = Path(history["source_run"]) / "manifest.json"
    if (protocol.sha256(source_manifest.read_bytes()) != MANIFEST_SHA256 or native.read_json(source_manifest)["specification"] != original
            or spec["version"] != VERSION or spec["historical_counts"] != COUNTS
            or spec["format_instruction_sha256"] != previous.body.INSTRUCTION_SHA256
            or spec["groups"] != native.read_json(output / "groups.json")
            or protocol.sha256(protocol.canonical_bytes(spec["groups"])) != history["groups_sha256"]
            or any(spec[key] != original[key] for key in ("requested_model", "native_cli_sha256", "disabled_mcp_names", "native_command_identity", "bindings"))
            or any(set(history["records"][stream]) != set(original["eligible_record_ids"][:count]) for stream, count in COUNTS.items())):
        raise ValueError("Revalidated continuation source/history/model/whole groups changed")
    guards[str(output / "manifest.json")] = protocol.sha256((output / "manifest.json").read_bytes())
    return manifest, history, guards


reader.verify_prepared = verify_prepared
reader.verify_attempt = engine.verify_attempt
reader.VERSION = VERSION
completed_records, record_sources, merged_usage = reader.completed_records, reader.record_sources, reader.merged_usage
verify_attempt = engine.verify_attempt


def run(output, *, max_native_contexts, executor=native.invoke_cli, timeout_seconds=900):
    output = Path(output).resolve()
    manifest, history, guards = verify_prepared(output)
    own = previous.inventory(output)
    merged_usage(output, history)
    if any(not event["accepted"] for event, receipt in own.values()):
        raise ValueError("New failure retained; no automatic revalidation or retry")
    packets = native.read_json(output / "packets.json")
    done = {stream: completed_records(output, stream, packets) for stream in native.STREAMS}
    jobs = previous.old.previous.pending_jobs(manifest["specification"]["groups"], done, max_native_contexts)
    binary, environment = native.discover_native_cli(native.DEFAULT_CLI), native.child_environment()
    mcp, spec = native.configured_mcp_names(environment), manifest["specification"]
    if protocol.sha256(binary.read_bytes()) != spec["native_cli_sha256"] or mcp != spec["disabled_mcp_names"]:
        raise ValueError("Native binary/isolation changed")
    failures, calls = [], 0
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending, index = {}, 0
        while pending or index < len(jobs):
            while not failures and not (output / "STOP_REQUESTED").exists() and index < len(jobs) and len(pending) < 2:
                stream, group = jobs[index]
                pending[workers.submit(engine.execute_attempt, output, stream, group, packets, binary, environment, mcp,
                    guards, spec["requested_model"], timeout_seconds, executor)] = (stream, group)
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
                if not receipt["accepted"]:
                    failures.append(receipt)
                print(f"Revalidated continuation {stream.upper()} {group['batch']}: accepted={receipt['accepted']}", flush=True)
    native.protected_unchanged(guards)
    done = {stream: completed_records(output, stream, packets) for stream in native.STREAMS}
    ids = [rid for group in spec["groups"] for rid in group["record_ids"]]
    agreement = protocol.raw_agreement({rid: packets[rid] for rid in ids},
        [value for rid, value in done["a"].items() if rid in ids], [value for rid, value in done["b"].items() if rid in ids])
    agreement.update(agreement_population="unchanged_body_first_attempt_continuation_groups",
        reused_one_sided_a_records=len(history["one_sided_record_ids"]["a"]), excludes_common_paired_history=True)
    native.write_new(output / ("body_first_attempt_agreement-" + uuid4().hex[:10] + ".json"), agreement)
    summary = dict(output=str(output), accepted_counts={stream: len(values) for stream, values in done.items()},
        new_native_invocations=calls, failures=failures, historical_records_reused_per_stream=history["historical_counts"],
        status="selected_revalidated_calls_complete" if not failures else "stopped_without_automatic_retry",
        original_receipt_still_false=True, subscription_usage=merged_usage(output, history), subscription_dollar_cost=None, human_gold=False)
    native.write_new(output / ("summary-" + uuid4().hex[:10] + ".json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("prepare")
    for name in ("root", "pool-dir", "source-run", "certificate-dir"):
        before.add_argument("--" + name, type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--prepared-directory", type=Path, required=True)
    execute.add_argument("--max-native-contexts", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        directory, records = prepare(args.root, args.pool_dir, args.source_run, args.certificate_dir)
        print(json.dumps(dict(directory=str(directory), accepted_counts={s: len(v) for s, v in records.items()}, native_calls=0)))
    else:
        result = run(args.prepared_directory, max_native_contexts=args.max_native_contexts)
        raise SystemExit(1 if result["failures"] else 0)
