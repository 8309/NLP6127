"""Bounded complete-case throughput migration with immutable old A/B contexts.

The annotation prompt, schema, evidence packets and per-record checks are unchanged.
Only the explicitly recorded grouping policy changes. No old output is rebound to
different stdin bytes, copied into a new execution, or charged as a new invocation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from uuid import uuid4

import run_codex_cohort_annotations as previous
import run_codex_annotation as native
import run_codex_batch_annotations as batch
from oil_evidence import machine_evaluation as protocol

VERSION = "codex-top5-candidate-cohort-throughput-v3"


def read_migration(source_run, packets, bindings):
    """Verify a stopped old run and retain exact receipt paths, not edited copies."""
    source = Path(source_run).resolve()
    manifest = native.read_json(source / "manifest.json")
    spec = manifest["specification"]
    if (spec.get("version") != previous.VERSION or spec.get("partition") != "all_current"
            or spec.get("batch_size") != 5 or spec.get("bindings") != bindings
            or spec.get("eligible_record_ids") != list(packets)
            or spec.get("eligible_packets_sha256") != protocol.sha256(protocol.canonical_bytes(packets))
            or spec.get("batch_prompt_sha256") != protocol.sha256(batch.BATCH_PROMPT.encode())
            or not (source / "STOP_REQUESTED").is_file()):
        raise ValueError("Migration requires the stopped exact-pool all-current five-record run")
    protected = {str(source / "manifest.json"): protocol.sha256((source / "manifest.json").read_bytes())}
    for name, expected in manifest["immutable_sha256"].items():
        path = native.contained_file(source, name)
        if protocol.sha256(path.read_bytes()) != expected:
            raise ValueError("Migration source immutable artifact changed")
        protected[str(path)] = expected
    records = {stream: batch.completed_records(source, stream, packets) for stream in native.STREAMS}
    if set(records["a"]) != set(records["b"]):
        raise ValueError("Finish any old unpaired batch with its original stdin before migration")
    prefix = list(packets)[:len(records["a"])]
    if set(prefix) != set(records["a"]) or not prefix or len(prefix) % 5:
        raise ValueError("Migration must preserve a complete stable five-record source prefix")
    summaries = [native.read_json(path) for path in source.glob("summary-*.json")]
    if not any(value.get("status") == "stopped_without_automatic_retry" and not value.get("failures")
               and value.get("accepted_counts") == {"a": len(prefix), "b": len(prefix)} for value in summaries):
        raise ValueError("Source run has not sealed a clean stopped summary")
    if previous.validate_paired_inputs(source) != len(prefix) // 5:
        raise ValueError("Migration source does not contain every exact A/B stdin pair")
    sources = {stream: {} for stream in native.STREAMS}
    progress = protocol.parse_jsonl((source / "progress.jsonl").read_bytes(), "migration source progress")
    for event in progress:
        if event.get("type") != "completed" or event.get("accepted") is not True:
            raise ValueError("A failed or unfinished source execution cannot be silently migrated")
        path = native.contained_file(source, event["receipt"])
        receipt = native.read_json(path)
        if len(receipt["record_ids"]) != 5:
            raise ValueError("Source grouping no longer matches the five-record protocol")
        protected[str(path)] = event["receipt_sha256"]
        for name, expected in receipt["artifacts_sha256"].items():
            protected[str(native.contained_file(path.parent, name))] = expected
        for identifier in receipt["record_ids"]:
            sources[event["stream"]][identifier] = dict(receipt=str(path), receipt_sha256=event["receipt_sha256"],
                execution_id=receipt["execution_id"], stdin_sha256=protocol.sha256((path.parent / "stdin.txt").read_bytes()),
                source_run=str(source), migrated_unchanged=True)
    for path in (source / "progress.jsonl", source / "execution_usage.jsonl", source / "STOP_REQUESTED", *source.glob("summary-*.json")):
        protected[str(path)] = protocol.sha256(path.read_bytes())
    native.protected_unchanged(protected)
    return dict(source_run=str(source), record_ids=prefix, sources=sources,
                source_protected_sha256=protected, original_specification=spec,
                usage_counting="Original unique execution IDs only; migration creates no native usage"), records


def build_groups(rows, questions, migrated_ids, case_limit, payload_limit):
    """Fixed complete groups cannot cross DEV/TEST or exceed actual full bytes."""
    if type(case_limit) is not int or not 1 <= case_limit <= 10:
        raise ValueError("Choose an explicit complete-case limit from one through ten")
    if type(payload_limit) is not int or payload_limit < 1:
        raise ValueError("Choose a positive full stdin plus schema byte limit")
    queries = {row["record_id"]: row for row in questions}
    if [row["record_id"] for row in rows[:len(migrated_ids)]] != migrated_ids:
        raise ValueError("Migrated records are not the exact locked source-order prefix")
    if any(queries[identifier]["split"] != "development" for identifier in migrated_ids):
        raise ValueError("Only pre-test development contexts may have the old grouping policy")
    groups = []
    remaining = rows[len(migrated_ids):]
    offset = 0
    while offset < len(remaining):
        split = queries[remaining[offset]["record_id"]]["split"]
        if split not in {"development", "test"}:
            raise ValueError("Unknown locked question split")
        part = []
        for row in remaining[offset:offset + case_limit]:
            if queries[row["record_id"]]["split"] != split:
                break
            candidate = part + [row]
            _, candidate_stdin, candidate_schema = batch.prepare_batch(candidate)
            candidate_bytes = len(candidate_stdin) + len(protocol.canonical_bytes(candidate_schema))
            if candidate_bytes > payload_limit:
                if not part:
                    raise ValueError(f"Single complete {split} case requires {candidate_bytes} full bytes; explicit bound is {payload_limit}")
                break
            part = candidate
        packets, stdin, schema = batch.prepare_batch(part)
        schema_bytes = protocol.canonical_bytes(schema)
        actual = len(stdin) + len(schema_bytes)
        if actual > payload_limit:
            raise ValueError(f"Complete {split} group requires {actual} full bytes; explicit bound is {payload_limit}")
        groups.append(dict(batch=f"batch-{len(groups) + 1:04d}", split=split,
            record_ids=list(packets), actual_stdin_bytes=len(stdin), actual_schema_bytes=len(schema_bytes),
            full_payload_bytes=actual, full_stdin_sha256=protocol.sha256(stdin),
            schema_sha256=protocol.sha256(schema_bytes)))
        offset += len(part)
    if [identifier for group in groups for identifier in group["record_ids"]] != [row["record_id"] for row in remaining]:
        raise ValueError("Locked cohort order interleaves splits; do not silently reorder annotations")
    return groups


def selected_groups(groups, migrated_count, max_records, total):
    if type(max_records) is not int or max_records <= migrated_count or max_records > total:
        raise ValueError("Explicit total record limit must include some new complete cases")
    count, selected = migrated_count, []
    for group in groups:
        if count == max_records:
            break
        count += len(group["record_ids"])
        if count > max_records:
            raise ValueError("Record limit cuts a sealed complete throughput group")
        selected.append(group)
    if count != max_records:
        raise ValueError("Record limit does not match sealed complete groups")
    return selected


def completed_records(output, stream, packets):
    """Return exact migrated and local envelopes with separate actual receipt lineage."""
    output = Path(output)
    migration = native.read_json(output / "migration.json")
    native.protected_unchanged(migration["source_protected_sha256"])
    old = batch.completed_records(Path(migration["source_run"]), stream, packets)
    if set(old) != set(migration["record_ids"]):
        raise ValueError("Migration accepted record membership changed")
    new = batch.completed_records(output, stream, packets)
    if set(old) & set(new):
        raise ValueError("Accepted source records were rerun or rebound in the throughput run")
    return {**old, **new}


def record_sources(output, stream):
    output = Path(output)
    migration = native.read_json(output / "migration.json")
    native.protected_unchanged(migration["source_protected_sha256"])
    sources = dict(migration["sources"][stream])
    path = output / "progress.jsonl"
    progress = protocol.parse_jsonl(path.read_bytes(), "throughput progress") if path.exists() else []
    for event in progress:
        if event.get("type") != "completed" or event.get("accepted") is not True or event["stream"] != stream:
            continue
        receipt_path = native.contained_file(output, event["receipt"])
        if protocol.sha256(receipt_path.read_bytes()) != event["receipt_sha256"]:
            raise ValueError("Throughput receipt differs from progress binding")
        receipt = native.read_json(receipt_path)
        for identifier in receipt["record_ids"]:
            if identifier in sources:
                raise ValueError("Duplicate migrated/local record lineage")
            sources[identifier] = dict(receipt=str(receipt_path), receipt_sha256=event["receipt_sha256"],
                execution_id=receipt["execution_id"], stdin_sha256=receipt["full_stdin_sha256"],
                source_run=str(output), migrated_unchanged=False)
    return sources


def merged_usage(output):
    migration = native.read_json(Path(output) / "migration.json")
    native.protected_unchanged(migration["source_protected_sha256"])
    unique = {}
    for run_dir in (Path(migration["source_run"]), Path(output)):
        path = run_dir / "execution_usage.jsonl"
        for row in protocol.parse_jsonl(path.read_bytes(), "actual execution usage") if path.exists() else []:
            identifier = row["execution_id"]
            if identifier in unique:
                raise ValueError("Native execution usage duplicated across migration versions")
            unique[identifier] = row
    totals = {}
    for row in unique.values():
        for key, value in (row.get("usage") or {}).items():
            if type(value) is int and value >= 0:
                totals[key] = totals.get(key, 0) + value
    return dict(executions=len(unique), known_usage_executions=sum(row.get("usage") is not None for row in unique.values()),
        native_token_totals=totals, counting_unit="unique original execution_id across source and new run",
        subscription_dollar_cost=None, credit_conversion="Not measured")


def execute_attempt(output, stream, group, rows, binary, environment, mcp_names,
                    protected, requested_model, timeout_seconds, executor):
    native.protected_unchanged(protected)
    packets, stdin, schema = batch.prepare_batch(rows)
    schema_bytes = protocol.canonical_bytes(schema)
    if (list(packets) != group["record_ids"] or protocol.sha256(stdin) != group["full_stdin_sha256"]
            or protocol.sha256(schema_bytes) != group["schema_sha256"]
            or len(stdin) + len(schema_bytes) != group["full_payload_bytes"]):
        raise ValueError("Sealed complete group bytes changed")
    attempt = Path(output) / stream / group["batch"] / ("attempt-" + uuid4().hex[:12])
    attempt.mkdir(parents=True, exist_ok=False)
    context = Path(tempfile.mkdtemp(prefix="oil-codex-throughput-"))
    native.write_new(attempt / "stdin.txt", stdin)
    native.write_new(attempt / "schema.json", schema_bytes)
    native.write_new(context / "packets.json", {key: value["payload"] for key, value in packets.items()})
    native.write_new(context / "protocol.txt", batch.BATCH_PROMPT.encode())
    native.write_new(context / "schema.json", schema_bytes)
    final_path = context / "final.json"
    command = native.cli_command(binary, context, context / "schema.json", final_path, requested_model, mcp_names)
    native.write_new(attempt / "command.json", command)
    tick = perf_counter()
    try:
        exit_code = executor(command, stdin, context, attempt, environment, timeout_seconds)
        if final_path.exists():
            native.write_new(attempt / "raw_final.json", final_path.read_bytes())
        raw = (attempt / "raw_final.json").read_bytes() if (attempt / "raw_final.json").exists() else b""
        events = native.parse_execution_events((attempt / "events.jsonl").read_bytes())
        execution_id = events["thread_id"] or "unavailable-" + uuid4().hex
        identity = dict(service="codex_cli_chatgpt_subscription", requested_model=requested_model or "cli_default_unresolved",
            observed_model=events["observed_model"], execution_id=execution_id,
            observation_source="execution_metadata" if events["observed_model"] else "unavailable",
            identity_note="Native execution metadata only, never assistant self-report.")
        errors = list(events["errors"])
        if exit_code:
            errors.append("Native Codex process returned nonzero status")
        try:
            envelopes = batch.extract_outputs(raw.decode(), packets, identity, events["usage"], stdin,
                dict(batch_context_version=batch.BATCH_VERSION, cohort_protocol=VERSION,
                    requested_model=requested_model, other_settings="Unchanged inherited CLI configuration, not independently observed"))
            errors.extend(f"{key}: {error}" for key, value in envelopes.items() for error in value["validation_errors"])
        except (ValueError, TypeError, UnicodeError) as error:
            envelopes, errors = {}, errors + [str(error)]
        native.write_new(attempt / "envelopes.json", envelopes)
        native.protected_unchanged(protected)
        receipt = dict(accepted=not errors, stream=stream, batch=group["batch"], record_ids=list(packets),
            errors=errors, process_exit_code=exit_code, elapsed_seconds=perf_counter() - tick,
            execution_id=execution_id, usage=events["usage"], full_stdin_sha256=protocol.sha256(stdin),
            full_payload_bytes=group["full_payload_bytes"], grouping_protocol=VERSION,
            artifacts_sha256={path.name: protocol.sha256(path.read_bytes()) for path in attempt.iterdir() if path.is_file()},
            original_inputs_unchanged=True, human_gold=False, subscription_dollar_cost=None, external_paid_api_calls=0)
        native.write_new(attempt / "receipt.json", receipt)
        return dict(attempt=attempt, receipt=receipt, envelopes=envelopes,
            usage=dict(execution_id=execution_id, stream=stream, record_ids=list(packets), usage=events["usage"],
                native_events_sha256=protocol.sha256((attempt / "events.jsonl").read_bytes())))
    except Exception as error:
        failure = dict(accepted=False, stream=stream, batch=group["batch"], record_ids=list(packets),
            error_type=type(error).__name__, errors=[str(error)])
        native.write_new(attempt / "failure.json", failure)
        return dict(attempt=attempt, receipt=failure, envelopes={}, usage=None)


def run(root, pool_dir, source_run, max_records, *, case_limit, payload_limit,
        requested_model=None, cli=native.DEFAULT_CLI, resume=None, timeout_seconds=900, executor=native.invoke_cli):
    root = Path(root).resolve()
    pool_dir, rows, questions, protected, bindings = previous.load_pool(root, pool_dir)
    packets = {row["record_id"]: protocol.clean_packet(row, prompt=batch.BATCH_PROMPT) for row in rows}
    migration, old = read_migration(source_run, packets, bindings)
    groups = build_groups(rows, questions, migration["record_ids"], case_limit, payload_limit)
    selected = selected_groups(groups, len(migration["record_ids"]), max_records, len(rows))
    binary, environment = native.discover_native_cli(cli), native.child_environment()
    mcp_names = native.configured_mcp_names(environment)
    version = subprocess.run([str(binary), "--version"], capture_output=True, env=environment, check=True).stdout.decode().strip()
    login = subprocess.run([str(binary), "login", "status"], capture_output=True, env=environment, check=True)
    if (login.stdout + login.stderr).decode().strip() != "Logged in using ChatGPT":
        raise ValueError("Existing ChatGPT subscription authentication required; no API fallback")
    old_spec = migration["original_specification"]
    if (old_spec["requested_model"] != requested_model or old_spec["native_cli_sha256"] != protocol.sha256(binary.read_bytes())
            or old_spec["disabled_mcp_names"] != mcp_names or old_spec["cli_version"] != version):
        raise ValueError("Model request, native binary or tool isolation changed during migration")
    implementation = {str(path.resolve()): protocol.sha256(path.read_bytes()) for path in
        (Path(__file__), Path(previous.__file__), Path(batch.__file__), Path(native.__file__), Path(protocol.__file__))}
    protected.update(implementation)
    protected.update(migration["source_protected_sha256"])
    specification = dict(version=VERSION, bindings=bindings, partition="all_current", case_limit=case_limit,
        full_payload_byte_limit=payload_limit, eligible_record_ids=list(packets), groups=groups,
        eligible_packets_sha256=protocol.sha256(protocol.canonical_bytes(packets)),
        batch_prompt_sha256=protocol.sha256(batch.BATCH_PROMPT.encode()), requested_model=requested_model,
        native_cli_sha256=protocol.sha256(binary.read_bytes()), cli_version=version, disabled_mcp_names=mcp_names,
        implementation_sha256=implementation, source_run=str(Path(source_run).resolve()),
        migration_sha256=protocol.sha256(protocol.canonical_bytes(migration)), migrated_record_ids=migration["record_ids"],
        human_gold=False, independent_model_diversity_claimed=False, formal_test_frozen=False,
        grouping_policy="Migrated original DEV5 contexts unchanged; remaining DEV and all TEST use the same sealed max-case/byte-bound greedy policy",
        accepted_records_rerun=False, old_protocol_outputs_rebound=False, annotation_semantic_protocol_unchanged=True)
    if resume:
        output = Path(resume).resolve()
        if not output.is_relative_to((pool_dir / "cohort_annotations_throughput").resolve()):
            raise ValueError("Throughput resume escapes its distinct annotation directory")
        native.verify_resume(output, specification)
    else:
        output = pool_dir / "cohort_annotations_throughput" / (native.utc_now().replace(":", "-") + "-" + uuid4().hex[:10])
        output.mkdir(parents=True, exist_ok=False)
        for name, value in (("packets.json", packets), ("protocol.txt", batch.BATCH_PROMPT.encode()),
                            ("protected_inputs.json", protected), ("runner_script.py", Path(__file__).read_bytes()),
                            ("migration.json", migration), ("groups.json", groups)):
            native.write_new(output / name, value)
        names = ("packets.json", "protocol.txt", "protected_inputs.json", "runner_script.py", "migration.json", "groups.json")
        native.write_new(output / "manifest.json", dict(specification=specification,
            immutable_sha256={name: protocol.sha256((output / name).read_bytes()) for name in names},
            provenance="machine_only", human_gold=False, actual_labels_created=0,
            context_policy="Fresh native A/B contexts, identical actual full inputs; unchanged raw migrated contexts"))
    own_manifest = native.read_json(output / "manifest.json")
    for name, expected in own_manifest["immutable_sha256"].items():
        protected[str(native.contained_file(output, name))] = expected
    protected[str(output / "manifest.json")] = protocol.sha256((output / "manifest.json").read_bytes())
    completed = {stream: completed_records(output, stream, packets) for stream in native.STREAMS}
    by_id = {row["record_id"]: row for row in rows}
    jobs = []
    for group in selected:
        ids = set(group["record_ids"])
        for stream in native.STREAMS:
            done = ids & set(completed[stream])
            if done == ids:
                continue
            if done:
                raise ValueError("Partially accepted complete group cannot be regrouped")
            jobs.append((stream, group))
    failures, invocations = [], 0
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending, cursor = {}, 0
        while pending or cursor < len(jobs):
            stopped = bool(failures) or (output / "STOP_REQUESTED").exists()
            while not stopped and cursor < len(jobs) and len(pending) < 2:
                stream, group = jobs[cursor]
                future = workers.submit(execute_attempt, output, stream, group,
                    [by_id[key] for key in group["record_ids"]], binary, environment, mcp_names,
                    protected, requested_model, timeout_seconds, executor)
                pending[future] = (stream, group)
                cursor += 1
                invocations += 1
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
                        stream=stream, batch=group["batch"], record_ids=receipt["record_ids"],
                        receipt=attempt.relative_to(output).as_posix() + "/receipt.json",
                        receipt_sha256=protocol.sha256((attempt / "receipt.json").read_bytes())))
                else:
                    native.append_progress(output / "progress.jsonl", dict(type="failed", **receipt))
                if not receipt["accepted"]:
                    failures.append(receipt)
                else:
                    completed[stream].update(result["envelopes"])
                print(f"Throughput {stream.upper()} {group['batch']}: accepted={receipt['accepted']}, records={len(receipt['record_ids'])}", flush=True)
    native.protected_unchanged(protected)
    paired_new = previous.validate_paired_inputs(output)
    for stream in native.STREAMS:
        native.write_new(output / (stream + "_outputs-" + uuid4().hex[:10] + ".jsonl"),
            b"".join(protocol.canonical_bytes(completed[stream][key]) + b"\n" for key in packets if key in completed[stream]))
    summary = dict(output=str(output), status="selected_paired_machine_annotations_complete" if not failures
        and not (output / "STOP_REQUESTED").exists() else "stopped_without_automatic_retry",
        records_requested_per_stream=max_records, accepted_counts={key: len(value) for key, value in completed.items()},
        migrated_unchanged_records_per_stream=len(migration["record_ids"]), paired_new_group_count=paired_new,
        case_limit=case_limit, full_payload_byte_limit=payload_limit, new_native_invocations=invocations,
        failures=failures, subscription_usage=merged_usage(output), original_inputs_unchanged=True,
        external_paid_api_calls=0, human_labels_created=0, human_gold=False, formal_test_frozen=False,
        grouping_variance_confined_to_development=True, accepted_records_rerun=False,
        limitations=["Larger shared contexts can correlate machine errors; grouping is explicitly bound.",
                     "Same configured model repeated is not model diversity or human review.",
                     "Mechanical acceptance is not semantic gold or an accepted 1000-question formal test freeze."])
    native.write_new(output / ("summary-" + uuid4().hex[:10] + ".json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return output, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--max-records", type=int, required=True)
    parser.add_argument("--case-limit", type=int, required=True)
    parser.add_argument("--payload-limit", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--requested-model")
    args = parser.parse_args()
    _, summary = run(args.canonical_root, args.pool_dir, args.source_run, args.max_records,
        case_limit=args.case_limit, payload_limit=args.payload_limit, resume=args.resume,
        requested_model=args.requested_model)
    raise SystemExit(1 if summary["failures"] else 0)
