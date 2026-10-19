"""Explicit wire-coded cohort driver; historical executions keep original hashes.

Only new runs may use structural aliases. This file first provides immutable
historical-code snapshots; no native wire execution is authorized by preparation.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import subprocess
import tempfile
from time import perf_counter

import run_codex_annotation as native
import run_codex_batch_annotations as batch
import run_codex_cohort_annotations as previous
import run_codex_cohort_throughput as throughput
from oil_evidence import machine_evaluation as protocol
from oil_evidence import wire_codec as codec


VERSION = "codex-top5-candidate-cohort-wire-v4"


def snapshot_historical_implementations(root, source_run):
    root, source = Path(root).resolve(), Path(source_run).resolve()
    manifest = native.read_json(source / "manifest.json")
    migration = native.read_json(source / "migration.json")
    older = Path(migration["source_run"])
    older_manifest = native.read_json(older / "manifest.json")
    implementations = {}
    for value in (manifest, older_manifest):
        for name, expected in value["specification"]["implementation_sha256"].items():
            if name in implementations and implementations[name] != expected:
                raise ValueError("Historical execution implementation hashes conflict")
            implementations[name] = expected
    raw = {name: Path(name).read_bytes() for name in implementations}
    if any(protocol.sha256(raw[name]) != expected for name, expected in implementations.items()):
        raise ValueError("Historical implementation changed before exact pre-edit snapshot")
    output = root / "data/machine_runs/annotation-migrations" / (native.utc_now().replace(":", "-") + "-" + uuid4().hex[:10])
    output.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for index, name in enumerate(sorted(raw)):
        relative = f"historical_code/{index:02d}-{Path(name).name}"
        native.write_new(output / relative, raw[name])
        snapshots[name] = dict(snapshot=relative, sha256=implementations[name])
    for prefix, run in (("v3", source), ("v2", older)):
        for name in ("manifest.json", "protected_inputs.json", "runner_script.py", "protocol.txt"):
            native.write_new(output / prefix / name, (run / name).read_bytes())
    receipt = dict(created_utc=native.utc_now(), status="historical_execution_code_snapshotted_before_core_change",
        source_annotation_run=str(source), earlier_annotation_run=str(older), code_snapshots=snapshots,
        original_execution_hashes_preserved=True, edited_implementation_paths_not_rebound=True,
        source_manifest_sha256=protocol.sha256((source / "manifest.json").read_bytes()),
        earlier_manifest_sha256=protocol.sha256((older / "manifest.json").read_bytes()),
        artifacts_sha256={path.relative_to(output).as_posix(): protocol.sha256(path.read_bytes())
                          for path in output.rglob("*") if path.is_file()},
        native_calls=0, semantic_labels_created=0)
    native.write_new(output / "receipt.json", receipt)
    return output, receipt


def load_history(source_run, snapshot_dir, packets):
    source, snapshots = Path(source_run).resolve(), Path(snapshot_dir).resolve()
    receipt = native.read_json(snapshots / "receipt.json")
    if receipt.get("source_annotation_run") != str(source) or receipt.get("original_execution_hashes_preserved") is not True:
        raise ValueError("Use the exact pre-edit historical snapshot for this stopped source")
    protected = {str(snapshots / "receipt.json"): protocol.sha256((snapshots / "receipt.json").read_bytes())}
    for name, expected in receipt["artifacts_sha256"].items():
        path = native.contained_file(snapshots, name)
        protected[str(path)] = expected
    native.protected_unchanged(protected)
    manifest = native.read_json(source / "manifest.json")
    if (protocol.sha256((source / "manifest.json").read_bytes()) != receipt["source_manifest_sha256"]
            or manifest["specification"]["version"] != throughput.VERSION
            or not (source / "STOP_REQUESTED").exists()):
        raise ValueError("Source run changed or is not a stopped v3 annotation run")
    for name, expected in manifest["specification"]["implementation_sha256"].items():
        if receipt["code_snapshots"].get(name, {}).get("sha256") != expected:
            raise ValueError("Historical execution code was rebound rather than snapshotted")
    historical = receipt["code_snapshots"]
    for name, expected in native.read_json(source / "protected_inputs.json").items():
        if name in historical:
            if historical[name]["sha256"] != expected:
                raise ValueError("Historical protected code hash conflicts with its immutable snapshot")
        else:
            protected[name] = expected
    for name, expected in manifest["immutable_sha256"].items():
        protected[str(native.contained_file(source, name))] = expected
    for path in (source / "manifest.json", source / "progress.jsonl", source / "execution_usage.jsonl", source / "STOP_REQUESTED", *source.glob("summary-*.json")):
        protected[str(path)] = protocol.sha256(path.read_bytes())
    migration = native.read_json(source / "migration.json")
    older = Path(migration["source_run"])
    protected.update(migration["source_protected_sha256"])
    native.protected_unchanged(protected)
    records = {}
    sources = {}
    for stream in native.STREAMS:
        old = batch.completed_records(older, stream, packets)
        current = batch.completed_records(source, stream, packets)
        if set(old) & set(current):
            raise ValueError("Historical accepted source records were rerun")
        records[stream] = {**old, **current}
        sources[stream] = throughput.record_sources(source, stream)
        for value in sources[stream].values():
            path = Path(value["receipt"])
            protected[str(path)] = value["receipt_sha256"]
            for name, expected in native.read_json(path)["artifacts_sha256"].items():
                protected[str(native.contained_file(path.parent, name))] = expected
    ids = manifest["specification"]["eligible_record_ids"][:len(records["a"])]
    if set(records["a"]) != set(records["b"]) or set(ids) != set(records["a"]):
        raise ValueError("Historical source is unpaired or not the original stable prefix")
    if not any(value.get("status") == "stopped_without_automatic_retry" and not value.get("failures")
               and value.get("accepted_counts") == {"a": len(ids), "b": len(ids)}
               for value in (native.read_json(path) for path in source.glob("summary-*.json"))):
        raise ValueError("Historical source has no clean final stopped summary")
    native.protected_unchanged(protected)
    return dict(source_run=str(source), older_source_run=str(older), snapshot_dir=str(snapshots),
        record_ids=ids, sources=sources, protected_sha256=protected,
        original_specification=manifest["specification"], original_code_snapshots=historical,
        historical_execution_code_hashes_not_rebound=True), records


def build_groups(rows, questions, migrated_ids, case_limit=10, payload_limit=262144):
    if type(case_limit) is not int or not 1 <= case_limit <= 10 or type(payload_limit) is not int or payload_limit <= 0:
        raise ValueError("Use an explicit max10 complete-case and positive full byte bound")
    splits = {row["record_id"]: row["split"] for row in questions}
    if ([row["record_id"] for row in rows[:len(migrated_ids)]] != migrated_ids
            or any(splits[value] != "development" for value in migrated_ids)):
        raise ValueError("Only the unchanged pre-test development prefix may migrate")
    groups, offset = [], len(migrated_ids)
    while offset < len(rows):
        split = splits[rows[offset]["record_id"]]
        part = []
        for row in rows[offset:offset + case_limit]:
            if splits[row["record_id"]] != split:
                break
            prepared = codec.prepare_batch(part + [row], batch.BATCH_PROMPT, native.output_schema)
            actual = len(prepared["stdin"]) + len(protocol.canonical_bytes(prepared["schema"]))
            if actual > payload_limit:
                if not part:
                    raise ValueError("Single full wire case exceeds the explicit byte bound")
                break
            part.append(row)
        prepared = codec.prepare_batch(part, batch.BATCH_PROMPT, native.output_schema)
        codec.verify_codec(prepared["codec"], part, prepared["wire_payload"])
        groups.append(dict(batch=f"batch-{len(groups) + 1:04d}", split=split,
            record_ids=[row["record_id"] for row in part], full_stdin_sha256=prepared["actual_wire_stdin_sha256"],
            schema_sha256=prepared["actual_wire_schema_sha256"], codec_sidecar_sha256=protocol.sha256(protocol.canonical_bytes(prepared["codec"])),
            mapping_sha256=prepared["codec"]["mapping_sha256"],
            full_payload_bytes=len(prepared["stdin"]) + len(protocol.canonical_bytes(prepared["schema"]))))
        offset += len(part)
    return groups


def verify_attempt(attempt, receipt, packets):
    ids = receipt["record_ids"]
    rows = [packets[identifier]["payload"] for identifier in ids]
    prepared = codec.prepare_batch(rows, batch.BATCH_PROMPT, native.output_schema)
    saved_codec = native.read_json(attempt / "codec.json")
    wire = native.read_json(attempt / "wire_payload.json")
    codec.verify_codec(saved_codec, rows, wire)
    if ((attempt / "stdin.txt").read_bytes() != prepared["stdin"]
            or (attempt / "schema.json").read_bytes() != protocol.canonical_bytes(prepared["schema"])
            or native.read_json(attempt / "canonical_semantic_packets.json") != prepared["canonical_semantic_packets"]
            or receipt["full_stdin_sha256"] != prepared["actual_wire_stdin_sha256"]
            or receipt["wire_schema_sha256"] != prepared["actual_wire_schema_sha256"]
            or receipt["mapping_sha256"] != saved_codec["mapping_sha256"]):
        raise ValueError("Actual complete wire input/schema/map differs from canonical source")
    raw = (attempt / "raw_final.json").read_bytes()
    decoded = codec.decode_batch(raw, saved_codec, rows)["decoded_records"]
    envelopes = native.read_json(attempt / "envelopes.json")
    if set(envelopes) != set(ids):
        raise ValueError("Wire envelope coverage differs")
    for identifier, value in envelopes.items():
        if (value["raw_output"].encode() != raw or value.get("raw_codec") != saved_codec
                or value["execution_input_sha256"] != receipt["full_stdin_sha256"]
                or value["label"] != decoded[identifier]["label"]
                or protocol.validate_independent_output(value, packets[identifier])):
            raise ValueError("Wire envelope no longer binds actual raw and canonical evidence")
    return envelopes


def completed_records(output, stream, packets):
    output = Path(output)
    history = native.read_json(output / "history.json")
    native.protected_unchanged(history["protected_sha256"])
    old = batch.completed_records(Path(history["older_source_run"]), stream, packets)
    records = {**old, **batch.completed_records(Path(history["source_run"]), stream, packets)}
    if set(records) != set(history["record_ids"]):
        raise ValueError("Historical accepted record membership changed")
    path = output / "progress.jsonl"
    events = protocol.parse_jsonl(path.read_bytes(), "wire progress") if path.exists() else []
    for event in events:
        if event.get("type") != "completed" or event.get("accepted") is not True or event["stream"] != stream:
            continue
        receipt_path = native.contained_file(output, event["receipt"])
        if protocol.sha256(receipt_path.read_bytes()) != event["receipt_sha256"]:
            raise ValueError("Wire receipt changed from durable progress")
        receipt = native.read_json(receipt_path)
        for name, expected in receipt["artifacts_sha256"].items():
            if protocol.sha256(native.contained_file(receipt_path.parent, name).read_bytes()) != expected:
                raise ValueError("Wire raw artifact changed")
        values = verify_attempt(receipt_path.parent, receipt, packets)
        if set(values) & set(records):
            raise ValueError("Accepted historical/wire records were rerun or duplicated")
        records.update(values)
    return records


def record_sources(output, stream):
    output = Path(output)
    history = native.read_json(output / "history.json")
    native.protected_unchanged(history["protected_sha256"])
    sources = dict(history["sources"][stream])
    path = output / "progress.jsonl"
    for event in protocol.parse_jsonl(path.read_bytes(), "wire progress") if path.exists() else []:
        if event.get("type") != "completed" or event.get("accepted") is not True or event["stream"] != stream:
            continue
        path = native.contained_file(output, event["receipt"])
        if protocol.sha256(path.read_bytes()) != event["receipt_sha256"]:
            raise ValueError("Wire source receipt changed")
        receipt = native.read_json(path)
        for identifier in receipt["record_ids"]:
            if identifier in sources:
                raise ValueError("Duplicate historical/wire source record")
            sources[identifier] = dict(receipt=str(path), receipt_sha256=event["receipt_sha256"],
                execution_id=receipt["execution_id"], stdin_sha256=receipt["full_stdin_sha256"], source_run=str(output),
                migrated_unchanged=False, raw_encoding=codec.VERSION, mapping_sha256=receipt["mapping_sha256"])
    return sources


def merged_usage(output):
    output = Path(output)
    history = native.read_json(output / "history.json")
    native.protected_unchanged(history["protected_sha256"])
    executions = {}
    for run_dir in (Path(history["older_source_run"]), Path(history["source_run"]), output):
        path = run_dir / "execution_usage.jsonl"
        for value in protocol.parse_jsonl(path.read_bytes(), "unique execution usage") if path.exists() else []:
            if value["execution_id"] in executions:
                raise ValueError("Historical native usage was duplicated/rebound")
            executions[value["execution_id"]] = value
    totals = {}
    for value in executions.values():
        for key, count in (value.get("usage") or {}).items():
            if type(count) is int and count >= 0:
                totals[key] = totals.get(key, 0) + count
    return dict(executions=len(executions), native_token_totals=totals,
        known_usage_executions=sum(value.get("usage") is not None for value in executions.values()),
        counting_unit="Original unique execution ID across v2, v3 and v4, never extracted record count",
        subscription_dollar_cost=None)


def execute_attempt(output, stream, group, rows, binary, environment, mcp_names, protected,
                    requested_model, timeout_seconds, executor):
    native.protected_unchanged(protected)
    prepared = codec.prepare_batch(rows, batch.BATCH_PROMPT, native.output_schema)
    codec.verify_codec(prepared["codec"], rows, prepared["wire_payload"])
    if (prepared["actual_wire_stdin_sha256"] != group["full_stdin_sha256"]
            or prepared["actual_wire_schema_sha256"] != group["schema_sha256"]
            or protocol.sha256(protocol.canonical_bytes(prepared["codec"])) != group["codec_sidecar_sha256"]):
        raise ValueError("Sealed wire group changed")
    attempt = Path(output) / stream / group["batch"] / ("attempt-" + uuid4().hex[:12])
    attempt.mkdir(parents=True, exist_ok=False)
    context = Path(tempfile.mkdtemp(prefix="oil-codex-wire-"))
    for name, value in (("stdin.txt", prepared["stdin"]), ("schema.json", prepared["schema"]),
            ("codec.json", prepared["codec"]), ("wire_payload.json", prepared["wire_payload"]),
            ("canonical_semantic_packets.json", prepared["canonical_semantic_packets"])):
        native.write_new(attempt / name, value)
    native.write_new(context / "packets.json", prepared["wire_payload"])
    native.write_new(context / "protocol.txt", prepared["prompt"].encode())
    native.write_new(context / "schema.json", prepared["schema"])
    final = context / "final.json"
    command = native.cli_command(binary, context, context / "schema.json", final, requested_model, mcp_names)
    native.write_new(attempt / "command.json", command)
    tick = perf_counter()
    try:
        status = executor(command, prepared["stdin"], context, attempt, environment, timeout_seconds)
        if final.exists():
            native.write_new(attempt / "raw_final.json", final.read_bytes())
        events = native.parse_execution_events((attempt / "events.jsonl").read_bytes())
        execution_id = events["thread_id"] or "unavailable-" + uuid4().hex
        identity = dict(service="codex_cli_chatgpt_subscription", requested_model=requested_model or "cli_default_unresolved",
            observed_model=events["observed_model"], execution_id=execution_id,
            observation_source="execution_metadata" if events["observed_model"] else "unavailable",
            identity_note="Native execution metadata only, never assistant self-report")
        errors = list(events["errors"]) + (["Native process returned nonzero status"] if status else [])
        envelopes = {}
        try:
            raw = (attempt / "raw_final.json").read_bytes().decode("utf-8")
            decoded = codec.decode_batch(raw, prepared["codec"], rows)["decoded_records"]
            for alias, binding in prepared["codec"]["records"].items():
                identifier = binding["record_id"]
                value = protocol.wrap_independent_output(prepared["canonical_semantic_packets"][identifier], raw,
                    identity, execution_id, dict(cohort_protocol=VERSION, requested_model=requested_model,
                        actual_wire_prompt_sha256=protocol.sha256(prepared["prompt"].encode()),
                        wire_schema_sha256=prepared["actual_wire_schema_sha256"], mapping_sha256=prepared["codec"]["mapping_sha256"],
                        other_settings="Unchanged inherited CLI configuration, not independently observed"), usage=None,
                    execution_input_sha256=prepared["actual_wire_stdin_sha256"], raw_label_path=[alias, "label"],
                    raw_record_id_path=[alias, "record_id"], raw_codec=prepared["codec"])
                if value["label"] != decoded[identifier]["label"]:
                    raise ValueError("Core normalized label differs from full verified codec decoding")
                value["usage_reference_execution_id"] = execution_id
                envelopes[identifier] = value
                errors.extend(f"{identifier}: {error}" for error in value["validation_errors"])
        except (ValueError, TypeError, OSError) as error:
            errors.append(str(error))
        native.write_new(attempt / "envelopes.json", envelopes)
        native.protected_unchanged(protected)
        receipt = dict(accepted=not errors, stream=stream, batch=group["batch"], record_ids=group["record_ids"], errors=errors,
            process_exit_code=status, elapsed_seconds=perf_counter() - tick, execution_id=execution_id, usage=events["usage"],
            full_stdin_sha256=prepared["actual_wire_stdin_sha256"], wire_schema_sha256=prepared["actual_wire_schema_sha256"],
            mapping_sha256=prepared["codec"]["mapping_sha256"], full_payload_bytes=group["full_payload_bytes"],
            raw_encoding=codec.VERSION, canonical_output_is_raw_model_output=False,
            artifacts_sha256={p.name: protocol.sha256(p.read_bytes()) for p in attempt.iterdir() if p.is_file()},
            human_gold=False, external_paid_api_calls=0, subscription_dollar_cost=None)
        native.write_new(attempt / "receipt.json", receipt)
        return dict(attempt=attempt, receipt=receipt, envelopes=envelopes,
            usage=dict(execution_id=execution_id, stream=stream, record_ids=group["record_ids"], usage=events["usage"],
                       native_events_sha256=protocol.sha256((attempt / "events.jsonl").read_bytes())))
    except Exception as error:
        failure = dict(accepted=False, stream=stream, batch=group["batch"], record_ids=group["record_ids"], errors=[str(error)])
        native.write_new(attempt / "failure.json", failure)
        return dict(attempt=attempt, receipt=failure, envelopes={}, usage=None)


def run(root, pool_dir, source_run, snapshot_dir, max_records, *, resume=None, requested_model=None,
        case_limit=10, payload_limit=262144, timeout_seconds=900, executor=native.invoke_cli):
    pool, rows, questions, protected, bindings = previous.load_pool(root, pool_dir)
    if (not Path(source_run).resolve().is_relative_to((pool / "cohort_annotations_throughput").resolve())
            or not Path(snapshot_dir).resolve().is_relative_to((Path(root) / "data/machine_runs/annotation-migrations").resolve())):
        raise ValueError("Historical source or pre-edit snapshot escapes its exact candidate/canonical run directory")
    packets = {row["record_id"]: protocol.clean_packet(row, prompt=batch.BATCH_PROMPT) for row in rows}
    history, old = load_history(source_run, snapshot_dir, packets)
    if (history["original_specification"]["bindings"] != bindings
            or history["original_specification"]["eligible_record_ids"] != list(packets)):
        raise ValueError("Historical annotation and frozen candidate pool source bindings differ")
    groups = build_groups(rows, questions, history["record_ids"], case_limit, payload_limit)
    selected = throughput.selected_groups(groups, len(history["record_ids"]), max_records, len(rows))
    binary, environment = native.discover_native_cli(native.DEFAULT_CLI), native.child_environment()
    mcp = native.configured_mcp_names(environment)
    version = subprocess.run([str(binary), "--version"], capture_output=True, env=environment, check=True).stdout.decode().strip()
    login = subprocess.run([str(binary), "login", "status"], capture_output=True, env=environment, check=True)
    if (login.stdout + login.stderr).decode().strip() != "Logged in using ChatGPT":
        raise ValueError("Existing ChatGPT subscription login required; no API fallback")
    old_spec = history["original_specification"]
    if (old_spec["requested_model"] != requested_model or old_spec["native_cli_sha256"] != protocol.sha256(binary.read_bytes())
            or old_spec["cli_version"] != version or old_spec["disabled_mcp_names"] != mcp):
        raise ValueError("Native identity request/binary/tool isolation changed during migration")
    protected.update(history["protected_sha256"])
    implementation = {str(Path(path).resolve()): protocol.sha256(Path(path).read_bytes()) for path in
                      (__file__, previous.__file__, throughput.__file__, batch.__file__, native.__file__, protocol.__file__, codec.__file__)}
    protected.update(implementation)
    spec = dict(version=VERSION, bindings=bindings, partition="all_current", groups=groups,
        case_limit=case_limit, full_payload_byte_limit=payload_limit, eligible_record_ids=list(packets),
        eligible_packets_sha256=protocol.sha256(protocol.canonical_bytes(packets)),
        batch_prompt_sha256=protocol.sha256(batch.BATCH_PROMPT.encode()), wire_encoding=codec.VERSION,
        wire_instructions_sha256=protocol.sha256(codec.WIRE_INSTRUCTIONS.encode()),
        history_sha256=protocol.sha256(protocol.canonical_bytes(history)), source_run=str(Path(source_run).resolve()),
        snapshot_dir=str(Path(snapshot_dir).resolve()), implementation_sha256=implementation, requested_model=requested_model,
        native_cli_sha256=protocol.sha256(binary.read_bytes()), cli_version=version, disabled_mcp_names=mcp,
        historical_labels_unchanged=True, canonical_semantic_hash_is_actual_wire_hash=False, human_gold=False,
        independent_model_diversity_claimed=False, formal_test_frozen=False)
    if resume:
        output = Path(resume).resolve()
        if not output.is_relative_to((pool / "cohort_annotations_wire").resolve()):
            raise ValueError("Wire resume escapes its distinct annotation directory")
        native.verify_resume(output, spec)
    else:
        output = pool / "cohort_annotations_wire" / (native.utc_now().replace(":", "-") + "-" + uuid4().hex[:10])
        output.mkdir(parents=True, exist_ok=False)
        for name, value in (("packets.json", packets), ("protocol.txt", batch.BATCH_PROMPT.encode()),
            ("wire_protocol.txt", (batch.BATCH_PROMPT + codec.WIRE_INSTRUCTIONS).encode()), ("groups.json", groups),
            ("history.json", history), ("protected_inputs.json", protected), ("runner_script.py", Path(__file__).read_bytes())):
            native.write_new(output / name, value)
        names = ("packets.json", "protocol.txt", "wire_protocol.txt", "groups.json", "history.json", "protected_inputs.json", "runner_script.py")
        native.write_new(output / "manifest.json", dict(specification=spec,
            immutable_sha256={name: protocol.sha256((output / name).read_bytes()) for name in names}, provenance="machine_only"))
    manifest = native.read_json(output / "manifest.json")
    protected.update({str(native.contained_file(output, name)): expected for name, expected in manifest["immutable_sha256"].items()})
    protected[str(output / "manifest.json")] = protocol.sha256((output / "manifest.json").read_bytes())
    completed = {stream: completed_records(output, stream, packets) for stream in native.STREAMS}
    by_id = {row["record_id"]: row for row in rows}
    jobs = []
    for group in selected:
        for stream in native.STREAMS:
            overlap = set(group["record_ids"]) & set(completed[stream])
            if overlap == set(group["record_ids"]):
                continue
            if overlap:
                raise ValueError("Partially accepted wire group cannot be regrouped")
            jobs.append((stream, group))
    failures, calls = [], 0
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending, index = {}, 0
        while pending or index < len(jobs):
            while not failures and not (output / "STOP_REQUESTED").exists() and index < len(jobs) and len(pending) < 2:
                stream, group = jobs[index]
                future = workers.submit(execute_attempt, output, stream, group, [by_id[key] for key in group["record_ids"]],
                    binary, environment, mcp, protected, requested_model, timeout_seconds, executor)
                pending[future] = (stream, group)
                index += 1
                calls += 1
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
                        stream=stream, batch=group["batch"], record_ids=group["record_ids"],
                        receipt=attempt.relative_to(output).as_posix() + "/receipt.json",
                        receipt_sha256=protocol.sha256((attempt / "receipt.json").read_bytes())))
                else:
                    native.append_progress(output / "progress.jsonl", dict(type="failed", **receipt))
                if receipt["accepted"]:
                    completed[stream].update(result["envelopes"])
                else:
                    failures.append(receipt)
                print(f"Wire {stream.upper()} {group['batch']}: accepted={receipt['accepted']}, records={len(group['record_ids'])}", flush=True)
    native.protected_unchanged(protected)
    paired = previous.validate_paired_inputs(output)
    for stream in native.STREAMS:
        native.write_new(output / (stream + "_outputs-" + uuid4().hex[:10] + ".jsonl"),
            b"".join(protocol.canonical_bytes(completed[stream][key]) + b"\n" for key in packets if key in completed[stream]))
    summary = dict(output=str(output), accepted_counts={stream: len(values) for stream, values in completed.items()},
        records_requested_per_stream=max_records, migrated_unchanged_records_per_stream=len(history["record_ids"]),
        new_native_invocations=calls, failures=failures, new_exact_paired_groups=paired,
        status="selected_paired_machine_annotations_complete" if not failures and not (output / "STOP_REQUESTED").exists()
            else "stopped_without_automatic_retry", human_gold=False, human_labels_created=0,
        formal_test_frozen=False, external_paid_api_calls=0, subscription_dollar_cost=None,
        subscription_usage=merged_usage(output),
        raw_alias_output_retained=True, canonical_semantic_hash_is_actual_wire_hash=False)
    native.write_new(output / ("summary-" + uuid4().hex[:10] + ".json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return output, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("canonical-root", "pool-dir", "source-run", "snapshot-dir"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--max-records", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    _, result = run(args.canonical_root, args.pool_dir, args.source_run, args.snapshot_dir, args.max_records, resume=args.resume)
    raise SystemExit(1 if result["failures"] else 0)
