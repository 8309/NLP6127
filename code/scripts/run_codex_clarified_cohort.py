"""Predeclared format clarification for genuinely unlabelled first attempts.

Reuse all verified historical pairs unchanged. Only remaining sources receive
the already reviewed generic citation-quote requirement, identically in A/B.
Question/evidence/schema/label standards and native model request do not change.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import tempfile
from time import perf_counter
from uuid import uuid4

import run_codex_corrected_cohort as previous
import correct_codex_wire_batch as clarification
import run_codex_annotation as native
import run_codex_cohort_wire as wire
from oil_evidence import machine_evaluation as protocol


VERSION = "codex-format-clarified-first-attempt-cohort-v1"
INSTRUCTION_SHA256 = "a9a5cd8573e1328e37c49fb9e3b5cbb93712275db4ccbecd92db616cffbb449b"


def build_groups(rows, questions, completed_ids):
    if protocol.sha256(clarification.CORRECTION_INSTRUCTION.encode()) != INSTRUCTION_SHA256:
        raise ValueError("Only the exact already reviewed format instruction is allowed")
    ids = [row["record_id"] for row in rows]
    if ids[:len(completed_ids)] != completed_ids:
        raise ValueError("Verified historical records must remain the unchanged source-order prefix")
    splits = {row["record_id"]: row["split"] for row in questions}
    groups, offset = [], len(completed_ids)
    while offset < len(rows):
        split = splits[rows[offset]["record_id"]]
        part = []
        for row in rows[offset:offset + 10]:
            if splits[row["record_id"]] != split:
                break
            try:
                prepared = clarification.prepare_source_batch(part + [row])
            except ValueError as error:
                if "byte bound" not in str(error) or not part:
                    raise
                break
            part.append(row)
        prepared = clarification.prepare_source_batch(part)
        groups.append(dict(batch=f"batch-{len(groups) + 1:04d}", split=split,
            record_ids=[row["record_id"] for row in part], full_stdin_sha256=prepared["actual_wire_stdin_sha256"],
            schema_sha256=prepared["actual_wire_schema_sha256"], mapping_sha256=prepared["codec"]["mapping_sha256"],
            full_payload_bytes=len(prepared["stdin"]) + len(protocol.canonical_bytes(prepared["schema"])),
            format_instruction_sha256=INSTRUCTION_SHA256))
        offset += len(part)
    if [identifier for group in groups for identifier in group["record_ids"]] != ids[len(completed_ids):]:
        raise ValueError("Neutral byte-bound grouping changed source order or coverage")
    return groups


def load_history(source_run, corrective_pair, packets):
    source_run = Path(source_run).resolve()
    manifest = native.read_json(source_run / "manifest.json")
    if manifest["specification"]["version"] != previous.VERSION:
        raise ValueError("Use the explicit verified composite history, never a raw failed source")
    base = native.read_json(source_run / "history.json")
    if str(Path(corrective_pair).resolve()) != base["corrective_pair"]:
        raise ValueError("Corrective-pair lineage changed")
    for name, expected in manifest["immutable_sha256"].items():
        if protocol.sha256(native.contained_file(source_run, name).read_bytes()) != expected:
            raise ValueError("Immutable composite history changed")
    if packets != native.read_json(source_run / "packets.json"):
        raise ValueError("Clarified canonical question/source packets changed")
    records = {stream: previous.completed_records(source_run, stream, packets) for stream in native.STREAMS}
    if set(records["a"]) != set(records["b"]):
        raise ValueError("Do not expose unpaired historical records to a new first-attempt protocol")
    ids = [identifier for identifier in manifest["specification"]["eligible_record_ids"] if identifier in records["a"]]
    protected = dict(native.read_json(source_run / "protected_inputs.json"))
    for path in source_run.rglob("*"):
        if path.is_file():
            protected[str(path)] = protocol.sha256(path.read_bytes())
    native.protected_unchanged(protected)
    return dict(source_run=str(source_run), corrective_pair=str(Path(corrective_pair).resolve()), record_ids=ids,
        sources={stream: previous.record_sources(source_run, stream) for stream in native.STREAMS},
        protected_sha256=protected, original_specification=manifest["specification"],
        corrective_record_ids=base["corrective_record_ids"],
        first_attempt_agreement_artifact=base["first_attempt_agreement_artifact"],
        original350_and_corrective10_retained_separately=True), records


def verify_attempt(attempt, receipt, packets):
    attempt = Path(attempt).resolve()
    if receipt.get("execution_protocol") != VERSION:
        return previous.verify_attempt(attempt, receipt, packets)
    rows = [packets[identifier]["payload"] for identifier in receipt["record_ids"]]
    prepared = clarification.prepare_source_batch(rows)
    source_output = Path(receipt["annotation_run"]).resolve()
    source_manifest = native.read_json(source_output / "manifest.json")
    source_spec = source_manifest["specification"]
    if source_spec["version"] != VERSION or protocol.sha256((source_output / "manifest.json").read_bytes()) != receipt["annotation_manifest_sha256"]:
        raise ValueError("Clarified native source manifest changed")
    group = next((g for g in source_spec["groups"] if g["batch"] == receipt["batch"]), None)
    actual_bytes = len(prepared["stdin"]) + len(protocol.canonical_bytes(prepared["schema"]))
    if (not group or group["record_ids"] != receipt["record_ids"] or group["full_stdin_sha256"] != prepared["actual_wire_stdin_sha256"]
            or group["schema_sha256"] != prepared["actual_wire_schema_sha256"] or group["mapping_sha256"] != prepared["codec"]["mapping_sha256"]
            or group["full_payload_bytes"] != actual_bytes or receipt.get("full_payload_bytes") != actual_bytes or actual_bytes > 262144):
        raise ValueError("Clarified native whole-case group or exact byte bound changed")
    clarification.retry.verify_raw_artifacts(attempt / "receipt.json", receipt)
    for name, expected in (("stdin.txt", prepared["stdin"]), ("schema.json", protocol.canonical_bytes(prepared["schema"])),
            ("codec.json", protocol.canonical_bytes(prepared["codec"])), ("wire_payload.json", protocol.canonical_bytes(prepared["wire_payload"])),
            ("canonical_semantic_packets.json", protocol.canonical_bytes(prepared["canonical_semantic_packets"])),
            ("format_instruction.txt", clarification.CORRECTION_INSTRUCTION.encode()), ("prompt.txt", prepared["prompt"].encode())):
        if (attempt / name).read_bytes() != expected:
            raise ValueError("Full clarified native input/schema/map/source differs")
    if (receipt.get("accepted") is not True or receipt.get("errors") or receipt.get("process_exit_code") != 0
            or receipt.get("annotation_role") != "format_clarified_first_attempt" or receipt.get("original_first_attempt") is not True
            or receipt.get("full_stdin_sha256") != prepared["actual_wire_stdin_sha256"]
            or receipt.get("schema_sha256") != prepared["actual_wire_schema_sha256"]
            or receipt.get("mapping_sha256") != prepared["codec"]["mapping_sha256"]
            or receipt.get("actual_prompt_sha256") != prepared["corrective_prompt_sha256"]
            or receipt.get("format_instruction_sha256") != INSTRUCTION_SHA256):
        raise ValueError("Clarified receipt does not pass unchanged source/label guards")
    events = native.parse_execution_events((attempt / "events.jsonl").read_bytes())
    if events["errors"] or events["thread_id"] != receipt["execution_id"] or events["usage"] != receipt["usage"]:
        raise ValueError("Clarified native execution or measured usage differs")
    actual_request = clarification.retry.requested_model_from_command(attempt / "command.json") or "cli_default_unresolved"
    if actual_request != (source_spec["requested_model"] or "cli_default_unresolved"):
        raise ValueError("Clarified actual model request differs from the frozen request")
    if clarification.normalized_command(native.read_json(attempt / "command.json")) != source_spec["native_command_identity"]:
        raise ValueError("Clarified native isolation command differs from the frozen settings")
    sent = native.read_json(attempt / "sent_input.json")
    if (sent.get("stdin_sha256") != prepared["actual_wire_stdin_sha256"] or sent.get("stdin_bytes") != len(prepared["stdin"])
            or sent.get("schema_sha256") != prepared["actual_wire_schema_sha256"]
            or sent.get("context_protocol_sha256") != prepared["corrective_prompt_sha256"]
            or sent.get("context_payload_sha256") != protocol.sha256(protocol.canonical_bytes(prepared["wire_payload"]))):
        raise ValueError("Clarified actual sent-input execution evidence differs")
    raw = (attempt / "raw_final.json").read_bytes()
    decoded = clarification.codec.decode_batch(raw, prepared["codec"], rows)["decoded_records"]
    envelopes = native.read_json(attempt / "envelopes.json")
    if set(envelopes) != set(receipt["record_ids"]):
        raise ValueError("Whole clarified native batch coverage differs")
    for identifier, value in envelopes.items():
        if (value.get("annotation_role") != "format_clarified_first_attempt" or value.get("original_first_attempt") is not True
                or value["raw_output"].encode() != raw or value["label"] != decoded[identifier]["label"]
                or value["execution_input_sha256"] != prepared["actual_wire_stdin_sha256"]
                or value["context_id"] != receipt["execution_id"] or value["identity"]["observed_model"] != events["observed_model"]
                or value["identity"]["execution_id"] != receipt["execution_id"]
                or value["identity"]["requested_model"] != actual_request
                or value["decoding"].get("actual_format_prompt_sha256") != prepared["corrective_prompt_sha256"]
                or value["decoding"].get("format_instruction_sha256") != INSTRUCTION_SHA256
                or protocol.validate_independent_output(value, packets[identifier])):
            raise ValueError("Clarified original-first-attempt raw envelope differs")
    return envelopes


def completed_records(output, stream, packets):
    output = Path(output).resolve()
    history = native.read_json(output / "history.json")
    native.protected_unchanged(history["protected_sha256"])
    rebuilt, records = load_history(history["source_run"], history["corrective_pair"], packets)
    if rebuilt != history:
        raise ValueError("Clarified source-history lineage changed")
    result = dict(records[stream])
    path = output / "progress.jsonl"
    for event in protocol.parse_jsonl(path.read_bytes(), "clarified source progress") if path.exists() else []:
        if event.get("type") != "completed" or event.get("accepted") is not True or event["stream"] != stream:
            continue
        receipt_path = native.contained_file(output, event["receipt"])
        if protocol.sha256(receipt_path.read_bytes()) != event["receipt_sha256"]:
            raise ValueError("Clarified durable native receipt changed")
        values = verify_attempt(receipt_path.parent, native.read_json(receipt_path), packets)
        if set(values) & set(result):
            raise ValueError("Historical/corrective/already accepted first attempts must not rerun")
        result.update(values)
    return result


def record_sources(output, stream):
    output = Path(output).resolve()
    history = native.read_json(output / "history.json")
    native.protected_unchanged(history["protected_sha256"])
    sources = dict(history["sources"][stream])
    path = output / "progress.jsonl"
    for event in protocol.parse_jsonl(path.read_bytes(), "clarified source progress") if path.exists() else []:
        if event.get("type") != "completed" or event.get("accepted") is not True or event["stream"] != stream:
            continue
        receipt_path = native.contained_file(output, event["receipt"])
        if protocol.sha256(receipt_path.read_bytes()) != event["receipt_sha256"]:
            raise ValueError("Clarified source receipt changed")
        receipt = native.read_json(receipt_path)
        for identifier in receipt["record_ids"]:
            if identifier in sources:
                raise ValueError("Clarified source record was repeated")
            sources[identifier] = dict(receipt=str(receipt_path), receipt_sha256=event["receipt_sha256"],
                execution_id=receipt["execution_id"], stdin_sha256=receipt["full_stdin_sha256"], source_run=str(output),
                migrated_unchanged=False, raw_encoding=clarification.codec.VERSION, execution_protocol=VERSION,
                mapping_sha256=receipt["mapping_sha256"], annotation_role="format_clarified_first_attempt")
    return sources


def merged_usage(output):
    output = Path(output).resolve()
    history = native.read_json(output / "history.json")
    composite = Path(history["source_run"])
    composite_history = native.read_json(composite / "history.json")
    original = Path(composite_history["source_run"])
    original_history = native.read_json(original / "history.json")
    directories = (Path(original_history["older_source_run"]), Path(original_history["source_run"]),
                   original, Path(composite_history["corrective_pair"]), composite, output)
    seen, totals, known = set(), {}, 0
    for directory in directories:
        path = directory / "execution_usage.jsonl"
        for value in protocol.parse_jsonl(path.read_bytes(), "unique annotation execution usage") if path.exists() else []:
            identifier = value["execution_id"]
            if not identifier or identifier in seen:
                raise ValueError("Original/corrective/clarified execution usage repeated or lacks an ID")
            seen.add(identifier)
            if value.get("usage") is not None:
                known += 1
                for key, count in value["usage"].items():
                    if type(count) is int and count >= 0:
                        totals[key] = totals.get(key, 0) + count
    return dict(executions=len(seen), known_usage_executions=known, native_token_totals=totals,
        counting_unit="Unique original, failed, corrective and clarified invocation IDs; not label count",
        unknown_execution_metadata_ids=sorted(value for value in seen if value.startswith("unavailable-")),
        scope="Annotation workflow only; other authoring/adjudication/experiment runs are separate",
        subscription_dollar_cost=None)


def prepare(root, pool_dir, source_run, corrective_pair):
    pool, rows, questions, protected, bindings = wire.previous.load_pool(Path(root), pool_dir)
    source_run = Path(source_run).resolve()
    if not source_run.is_relative_to(pool / "cohort_annotations_corrected"):
        raise ValueError("Clarified history must belong to this candidate pool")
    packets = native.read_json(source_run / "packets.json")
    history, records = load_history(source_run, corrective_pair, packets)
    original = history["original_specification"]
    ids = original["eligible_record_ids"]
    if original["bindings"] != bindings or ids != [row["record_id"] for row in rows]:
        raise ValueError("Frozen question/source bank or isolation metadata changed")
    groups = build_groups([packets[identifier]["payload"] for identifier in ids], questions, history["record_ids"])
    counterpart = native.read_json(Path(corrective_pair) / "manifest.json")["specification"]["counterpart_a"]["path"]
    command_identity = clarification.normalized_command(native.read_json(Path(counterpart).parent / "command.json"))
    protected.update(history["protected_sha256"])
    protected[str(Path(__file__).resolve())] = protocol.sha256(Path(__file__).read_bytes())
    output = pool / "cohort_annotations_clarified" / ("run-" + uuid4().hex[:10])
    output.mkdir(parents=True, exist_ok=False)
    spec = {**original, "version": VERSION, "source_run": str(source_run), "corrective_pair": history["corrective_pair"],
        "groups": groups, "history_sha256": protocol.sha256(protocol.canonical_bytes(history)),
        "format_instruction_sha256": INSTRUCTION_SHA256, "new_role": "format_clarified_first_attempt",
        "native_command_identity": command_identity,
        "format_change_predeclared_before_future_labels": True, "case_limit": 10, "full_payload_byte_limit": 262144,
        "semantic_standards_and_schema_unchanged": True, "future_original_IAA_population_separate": True,
        "implementation_sha256": {**original["implementation_sha256"], str(Path(__file__).resolve()): protocol.sha256(Path(__file__).read_bytes())}}
    values = {"packets.json": packets, "history.json": history, "groups.json": groups,
        "protocol.txt": wire.batch.BATCH_PROMPT.encode(), "format_instruction.txt": clarification.CORRECTION_INSTRUCTION.encode(),
        "protected_inputs.json": protected, "runner_script.py": Path(__file__).read_bytes()}
    for name, value in values.items():
        native.write_new(output / name, value)
    native.write_new(output / "manifest.json", dict(specification=spec,
        immutable_sha256={name: protocol.sha256((output / name).read_bytes()) for name in values}, provenance="machine_only_predeclared_format_clarification"))
    return output, records


def execute_attempt(output, stream, group, packets, binary, environment, mcp, protected, requested_model,
                    timeout_seconds, executor):
    if stream not in native.STREAMS:
        raise ValueError("Only the two declared fresh streams may execute")
    native.protected_unchanged(protected)
    rows = [packets[identifier]["payload"] for identifier in group["record_ids"]]
    prepared = clarification.prepare_source_batch(rows)
    if prepared["actual_wire_stdin_sha256"] != group["full_stdin_sha256"] or prepared["actual_wire_schema_sha256"] != group["schema_sha256"]:
        raise ValueError("Predeclared neutral group actual input/schema changed")
    attempt = Path(output) / stream / group["batch"] / ("attempt-" + uuid4().hex[:12])
    attempt.mkdir(parents=True, exist_ok=False)
    for name, value in (("stdin.txt", prepared["stdin"]), ("schema.json", prepared["schema"]), ("codec.json", prepared["codec"]),
            ("wire_payload.json", prepared["wire_payload"]), ("canonical_semantic_packets.json", prepared["canonical_semantic_packets"]),
            ("format_instruction.txt", clarification.CORRECTION_INSTRUCTION.encode()), ("prompt.txt", prepared["prompt"].encode())):
        native.write_new(attempt / name, value)
    context = Path(tempfile.mkdtemp(prefix="oil-clarified-first-"))
    for name, value in (("packets.json", prepared["wire_payload"]), ("schema.json", prepared["schema"]), ("protocol.txt", prepared["prompt"].encode())):
        native.write_new(context / name, value)
    final = context / "final.json"
    command = native.cli_command(binary, context, context / "schema.json", final, requested_model, mcp)
    manifest = native.read_json(Path(output) / "manifest.json")
    spec = manifest["specification"]
    if (clarification.normalized_command(command) != spec["native_command_identity"]
            or requested_model != spec["requested_model"]):
        raise ValueError("Native command/model settings differ from the predeclared future protocol")
    native.write_new(attempt / "command.json", command)
    native.write_new(attempt / "sent_input.json", dict(stdin_sha256=protocol.sha256(prepared["stdin"]), stdin_bytes=len(prepared["stdin"]),
        schema_sha256=protocol.sha256((context / "schema.json").read_bytes()),
        context_protocol_sha256=protocol.sha256((context / "protocol.txt").read_bytes()),
        context_payload_sha256=protocol.sha256((context / "packets.json").read_bytes()),
        source="Exact stdin byte argument passed to isolated native executor"))
    start = perf_counter()
    errors = []
    try:
        status = executor(command, prepared["stdin"], context, attempt, environment, timeout_seconds)
    except Exception as error:
        status = None
        errors.append("Native invocation failed: " + str(error))
    if final.exists():
        native.write_new(attempt / "raw_final.json", final.read_bytes())
    for name in ("events.jsonl", "stderr.txt"):
        if not (attempt / name).exists():
            native.write_new(attempt / name, b"")
    events = native.parse_execution_events((attempt / "events.jsonl").read_bytes())
    errors += list(events["errors"]) + (["Native process returned nonzero or unavailable status"] if status != 0 else [])
    if not events["thread_id"]:
        errors.append("Actual native fresh-context execution ID unavailable")
    execution_id = events["thread_id"] or "unavailable-" + uuid4().hex
    identity = dict(service="codex_cli_chatgpt_subscription", requested_model=requested_model or "cli_default_unresolved",
        observed_model=events["observed_model"], execution_id=execution_id,
        observation_source="execution_metadata" if events["observed_model"] else "unavailable",
        identity_note="Native execution metadata only, never assistant self-report")
    envelopes = {}
    try:
        raw = (attempt / "raw_final.json").read_bytes().decode("utf-8")
        decoded = clarification.codec.decode_batch(raw, prepared["codec"], rows)["decoded_records"]
        for alias, binding in prepared["codec"]["records"].items():
            identifier = binding["record_id"]
            value = protocol.wrap_independent_output(packets[identifier], raw, identity, execution_id,
                dict(cohort_protocol=VERSION, actual_format_prompt_sha256=prepared["corrective_prompt_sha256"],
                    format_instruction_sha256=INSTRUCTION_SHA256,
                    canonical_hash_scope="Baseline semantic source, not actual clarified model input"), usage=None,
                execution_input_sha256=prepared["actual_wire_stdin_sha256"], raw_label_path=[alias, "label"],
                raw_record_id_path=[alias, "record_id"], raw_codec=prepared["codec"])
            if value["label"] != decoded[identifier]["label"]:
                raise ValueError("Retained native label differs from whole verified decoding")
            value.update(annotation_role="format_clarified_first_attempt", original_first_attempt=True,
                         usage_reference_execution_id=execution_id)
            envelopes[identifier] = value
            errors.extend(value["validation_errors"])
    except (OSError, ValueError, TypeError) as error:
        errors.append(str(error))
    native.write_new(attempt / "envelopes.json", envelopes)
    native.protected_unchanged(protected)
    receipt = dict(accepted=not errors, errors=errors, stream=stream, batch=group["batch"], record_ids=group["record_ids"],
        process_exit_code=status, execution_id=execution_id, usage=events["usage"], elapsed_seconds=perf_counter() - start,
        execution_protocol=VERSION, annotation_role="format_clarified_first_attempt", original_first_attempt=True,
        annotation_run=str(Path(output).resolve()), annotation_manifest_sha256=protocol.sha256((Path(output) / "manifest.json").read_bytes()),
        full_stdin_sha256=prepared["actual_wire_stdin_sha256"], schema_sha256=prepared["actual_wire_schema_sha256"],
        mapping_sha256=prepared["codec"]["mapping_sha256"], format_instruction_sha256=INSTRUCTION_SHA256,
        actual_prompt_sha256=prepared["corrective_prompt_sha256"], full_payload_bytes=group["full_payload_bytes"],
        raw_encoding=clarification.codec.VERSION, human_gold=False, external_paid_api_calls=0,
        artifacts_sha256={p.name: protocol.sha256(p.read_bytes()) for p in attempt.iterdir() if p.is_file()})
    native.write_new(attempt / "receipt.json", receipt)
    return dict(attempt=attempt, receipt=receipt, envelopes=envelopes,
        usage=dict(execution_id=execution_id, stream=stream, usage=events["usage"], record_ids=group["record_ids"]))


def run(output, *, max_native_contexts, executor=native.invoke_cli, timeout_seconds=900):
    output = Path(output).resolve()
    manifest = native.read_json(output / "manifest.json")
    spec = manifest["specification"]
    if spec["version"] != VERSION or spec["format_instruction_sha256"] != INSTRUCTION_SHA256:
        raise ValueError("Explicit predeclared clarification manifest is required")
    protected = dict(native.read_json(output / "protected_inputs.json"))
    protected.update({str(native.contained_file(output, name)): digest for name, digest in manifest["immutable_sha256"].items()})
    protected[str(output / "manifest.json")] = protocol.sha256((output / "manifest.json").read_bytes())
    native.protected_unchanged(protected)
    path = output / "progress.jsonl"
    if path.exists() and any(e.get("accepted") is False or e.get("type") == "failed" for e in protocol.parse_jsonl(path.read_bytes())):
        raise ValueError("Clarified failure retained; stop without automatic retry")
    packets = native.read_json(output / "packets.json")
    completed = {stream: completed_records(output, stream, packets) for stream in native.STREAMS}
    jobs = previous.pending_jobs(native.read_json(output / "groups.json"), completed, max_native_contexts)
    binary, environment = native.discover_native_cli(native.DEFAULT_CLI), native.child_environment()
    mcp = native.configured_mcp_names(environment)
    if protocol.sha256(binary.read_bytes()) != spec["native_cli_sha256"] or mcp != spec["disabled_mcp_names"]:
        raise ValueError("Clarified native identity or tool isolation changed")
    failures, calls = [], 0
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending, index = {}, 0
        while pending or index < len(jobs):
            while not failures and not (output / "STOP_REQUESTED").exists() and index < len(jobs) and len(pending) < 2:
                stream, group = jobs[index]
                future = workers.submit(execute_attempt, output, stream, group, packets, binary, environment, mcp,
                    protected, spec["requested_model"], timeout_seconds, executor)
                pending[future] = (stream, group)
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
                native.append_progress(path, dict(type="completed", accepted=receipt["accepted"], stream=stream,
                    batch=group["batch"], record_ids=group["record_ids"], execution_protocol=VERSION,
                    receipt=attempt.relative_to(output).as_posix() + "/receipt.json",
                    receipt_sha256=protocol.sha256((attempt / "receipt.json").read_bytes())))
                if receipt["accepted"]:
                    completed[stream].update(result["envelopes"])
                else:
                    failures.append(receipt)
                print(f"Clarified {stream.upper()} {group['batch']}: accepted={receipt['accepted']}", flush=True)
    native.protected_unchanged(protected)
    future_ids = [identifier for group in spec["groups"] for identifier in group["record_ids"]]
    future_packets = {key: packets[key] for key in future_ids}
    future_agreement = protocol.raw_agreement(future_packets,
        [value for key, value in completed["a"].items() if key in future_packets],
        [value for key, value in completed["b"].items() if key in future_packets])
    future_agreement.update(agreement_population="format_clarified_future_first_attempts_only", excludes_historical350_and_corrective10=True)
    native.write_new(output / ("future_first_attempt_agreement-" + uuid4().hex[:10] + ".json"), future_agreement)
    summary = dict(output=str(output), accepted_counts={stream: len(values) for stream, values in completed.items()},
        new_native_invocations=calls, failures=failures, historical_records_reused_per_stream=360,
        status="selected_clarified_calls_complete" if not failures and not (output / "STOP_REQUESTED").exists() else "stopped_without_automatic_retry",
        original350_corrective10_futureIAA_reported_separately=True, human_gold=False, external_paid_api_calls=0,
        subscription_usage=merged_usage(output),
        subscription_dollar_cost=None)
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
