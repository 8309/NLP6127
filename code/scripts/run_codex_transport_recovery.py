"""Declared unchanged-input transport recovery; old failures remain immutable.

Only batch24 is retried after the user explicitly requests continuation.
The same sources, schemas, prompts, model request and isolation are retained.
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
import run_codex_clarified_cohort as clarified
import correct_codex_wire_batch as clarification
import run_codex_annotation as native
import run_codex_cohort_wire as wire
from oil_evidence import machine_evaluation as protocol


VERSION = "codex-unchanged-input-transport-recovery-v1"
HISTORY_COUNT = 590
EXPECTED_TOTAL = 1200
FAILED_BATCH = "batch-0024"
STOP_SUMMARY_SHA256 = "297cc1a52ebdc46ebb4e54d9d1dfdd1db6e00deddf31db777534a92a4ccc2520"
STOP_PROGRESS_SHA256 = "ec7d6c8599d2df3512facd504065210071ee1e69fef3d960713ad83138ab4efb"
STOP_MANIFEST_SHA256 = "165eed9c11bc986db77db0ccec79dd9cea2719110ae917f2b2c307ea399c75dc"
INSTRUCTION_SHA256 = "a9a5cd8573e1328e37c49fb9e3b5cbb93712275db4ccbecd92db616cffbb449b"


def transport_failures(source_run, manifest, packets):
    """Accept exactly the two pinned no-output DNS failures, never quota/labels."""
    source_run = Path(source_run).resolve()
    progress = source_run / "progress.jsonl"
    if protocol.sha256(progress.read_bytes()) != STOP_PROGRESS_SHA256:
        raise ValueError("Reviewed stopped progress changed")
    events = protocol.parse_jsonl(progress.read_bytes(), "retained transport failure progress")
    failed = [event for event in events if event.get("accepted") is False or event.get("type") == "failed"]
    if len(failed) != 2 or {event.get("stream") for event in failed} != {"a", "b"}:
        raise ValueError("Only the exact two declared transport failures may recover")
    group = next((g for g in manifest["specification"]["groups"] if g["batch"] == FAILED_BATCH), None)
    if group is None:
        raise ValueError("Declared failed source group is missing")
    rows = [packets[key]["payload"] for key in group["record_ids"]]
    prepared = clarification.prepare_source_batch(rows)
    result, execution_ids = {}, set()
    for event in failed:
        path = native.contained_file(source_run, event["receipt"])
        receipt = native.read_json(path)
        if (protocol.sha256(path.read_bytes()) != event["receipt_sha256"]
                or event.get("batch") != FAILED_BATCH or event["record_ids"] != group["record_ids"]
                or receipt.get("accepted") is not False or receipt.get("batch") != FAILED_BATCH
                or receipt["record_ids"] != group["record_ids"] or receipt["stream"] != event["stream"]
                or receipt.get("execution_protocol") != clarified.VERSION
                or receipt.get("process_exit_code") is not None or receipt.get("usage") is not None
                or receipt.get("annotation_manifest_sha256") != protocol.sha256((source_run / "manifest.json").read_bytes())
                or (path.parent / "raw_final.json").exists()
                or native.read_json(path.parent / "envelopes.json") != {}):
            raise ValueError("Failure is not the declared no-output transport-only attempt")
        clarification.retry.verify_raw_artifacts(path, receipt)
        actual = native.parse_execution_events((path.parent / "events.jsonl").read_bytes())
        raw_events = protocol.parse_jsonl((path.parent / "events.jsonl").read_bytes(), "retained native transport events")
        if (not actual["thread_id"] or actual["thread_id"] in execution_ids or actual["usage"] is not None
                or any(e.get("type") == "turn.completed" or e.get("item", {}).get("type") == "agent_message" for e in raw_events)
                or actual["thread_id"] != receipt["execution_id"]):
            raise ValueError("A prior final label/completed turn or usage cannot be retried as transport-only")
        messages = [str(e.get("message") or e.get("error") or e.get("item", {}).get("message", ""))
                    for e in raw_events if e.get("type") == "error" or e.get("item", {}).get("type") == "error"]
        low = "\n".join(messages).lower()
        if (not messages or "11001" not in low or any(word in low for word in
                ("quota", "rate limit", "usage limit", "invalid schema", "authentication", "unauthorized"))):
            raise ValueError("A quota, semantic, authentication or non-DNS failure requires separate review")
        if ((path.parent / "stdin.txt").read_bytes() != prepared["stdin"]
                or (path.parent / "schema.json").read_bytes() != protocol.canonical_bytes(prepared["schema"])
                or receipt["full_stdin_sha256"] != group["full_stdin_sha256"]
                or receipt["schema_sha256"] != group["schema_sha256"]
                or clarification.normalized_command(native.read_json(path.parent / "command.json"))
                    != manifest["specification"]["native_command_identity"]):
            raise ValueError("Failed source stdin/schema/model/isolation changed")
        execution_ids.add(actual["thread_id"])
        result[event["stream"]] = dict(path=str(path), sha256=event["receipt_sha256"],
            execution_id=actual["thread_id"], actual_stdin_sha256=receipt["full_stdin_sha256"],
            schema_sha256=receipt["schema_sha256"], usage=None, prior_final_output=False)
    return group, result


def historical_usage(source_run):
    """Read each original/failed/corrective execution ledger exactly once."""
    source_run = Path(source_run)
    first = native.read_json(source_run / "history.json")
    composite = Path(first["source_run"])
    second = native.read_json(composite / "history.json")
    original = Path(second["source_run"])
    third = native.read_json(original / "history.json")
    directories = (Path(third["older_source_run"]), Path(third["source_run"]), original,
                   Path(second["corrective_pair"]), composite, source_run)
    records, seen = [], set()
    for directory in directories:
        path = directory / "execution_usage.jsonl"
        for value in protocol.parse_jsonl(path.read_bytes(), "preserved annotation usage") if path.exists() else []:
            if not value["execution_id"] or value["execution_id"] in seen:
                raise ValueError("Historical invocation usage repeated")
            seen.add(value["execution_id"])
            records.append(value)
    return records


def load_history(source_run, packets):
    source_run = Path(source_run).resolve()
    manifest = native.read_json(source_run / "manifest.json")
    if protocol.sha256((source_run / "manifest.json").read_bytes()) != STOP_MANIFEST_SHA256:
        raise ValueError("Use the exact reviewed stopped source manifest")
    source_guard = dict(native.read_json(source_run / "protected_inputs.json"))
    source_guard[str(source_run / "manifest.json")] = STOP_MANIFEST_SHA256
    source_guard[str(source_run / "progress.jsonl")] = STOP_PROGRESS_SHA256
    for path in source_run.rglob("*"):
        if path.is_file() and str(path) not in source_guard:
            source_guard[str(path)] = protocol.sha256(path.read_bytes())
    native.protected_unchanged(source_guard)
    if manifest["specification"]["version"] != clarified.VERSION:
        raise ValueError("Only the reviewed failed clarified source may recover")
    for name, expected in manifest["immutable_sha256"].items():
        if protocol.sha256(native.contained_file(source_run, name).read_bytes()) != expected:
            raise ValueError("Original stopped immutable source changed")
    if packets != native.read_json(source_run / "packets.json"):
        raise ValueError("Original frozen canonical sources/questions changed")
    summaries = list(source_run.glob("summary-*.json"))
    if len(summaries) != 1 or protocol.sha256(summaries[0].read_bytes()) != STOP_SUMMARY_SHA256:
        raise ValueError("Use the exact reviewed transport-stop summary")
    summary = native.read_json(summaries[0])
    if summary["status"] != "stopped_without_automatic_retry" or summary["accepted_counts"] != {"a": HISTORY_COUNT, "b": HISTORY_COUNT}:
        raise ValueError("Reviewed stopped source coverage changed")
    group, failed = transport_failures(source_run, manifest, packets)
    records = {stream: clarified.completed_records(source_run, stream, packets) for stream in native.STREAMS}
    if any(len(values) != HISTORY_COUNT for values in records.values()) or set(records["a"]) != set(records["b"]):
        raise ValueError("All preserved accepted pairs must be complete and unchanged")
    ids = manifest["specification"]["eligible_record_ids"]
    accepted_ids = [key for key in ids if key in records["a"]]
    if ids[:HISTORY_COUNT] != accepted_ids or len(ids) != EXPECTED_TOTAL:
        raise ValueError("Preserved source order/coverage changed")
    sources = {stream: clarified.record_sources(source_run, stream) for stream in native.STREAMS}
    protected = source_guard
    old = native.read_json(source_run / "history.json")
    if any(name in protected and protected[name] != expected for name, expected in old["protected_sha256"].items()):
        raise ValueError("Historical source has conflicting original guards")
    protected.update(old["protected_sha256"])
    native.protected_unchanged(protected)
    history = {**old, "source_run": str(source_run), "record_ids": accepted_ids, "records": records,
        "sources": sources, "protected_sha256": protected, "original_specification": manifest["specification"],
        "failed_transport_pair": failed, "failed_group": group, "execution_usage_records": historical_usage(source_run),
        "future_original_first_attempt_agreement_artifact": dict(path=str(next(source_run.glob("future_first_attempt_agreement-*.json"))),
            sha256=protocol.sha256(next(source_run.glob("future_first_attempt_agreement-*.json")).read_bytes())),
        "original_failures_retained": True, "original_progress_rewritten": False,
        "first_complete_labels_after_transport_retry_are_not_original_invocations": True}
    return history, records


def verify_attempt(attempt, receipt, packets):
    attempt = Path(attempt).resolve()
    if receipt.get("execution_protocol") != VERSION:
        return clarified.verify_attempt(attempt, receipt, packets)
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
            or receipt.get("annotation_role") != group["annotation_role"] or receipt.get("original_first_attempt") != group["original_first_attempt"]
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
        if (value.get("annotation_role") != group["annotation_role"] or value.get("original_first_attempt") != group["original_first_attempt"]
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
    """Use a once-verified immutable history; verify each new raw attempt."""
    output = Path(output).resolve()
    manifest = native.read_json(output / "manifest.json")
    history = native.read_json(output / "history.json")
    if protocol.sha256((output / "history.json").read_bytes()) != manifest["immutable_sha256"]["history.json"]:
        raise ValueError("Immutable recovery history/cache changed")
    native.protected_unchanged(history["protected_sha256"])
    result = dict(history["records"][stream])
    if set(result) != set(history["record_ids"]):
        raise ValueError("Preserved historical label cache coverage changed")
    path = output / "progress.jsonl"
    for event in protocol.parse_jsonl(path.read_bytes(), "transport recovery progress") if path.exists() else []:
        if event.get("type") != "completed" or event.get("accepted") is not True or event["stream"] != stream:
            continue
        receipt_path = native.contained_file(output, event["receipt"])
        if protocol.sha256(receipt_path.read_bytes()) != event["receipt_sha256"]:
            raise ValueError("Recovery durable native receipt changed")
        values = verify_attempt(receipt_path.parent, native.read_json(receipt_path), packets)
        if set(values) & set(result):
            raise ValueError("An accepted original or recovery record was repeated")
        result.update(values)
    return result


def record_sources(output, stream):
    output = Path(output).resolve()
    history = native.read_json(output / "history.json")
    if protocol.sha256((output / "history.json").read_bytes()) != native.read_json(output / "manifest.json")["immutable_sha256"]["history.json"]:
        raise ValueError("Immutable recovery lineage changed")
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
                mapping_sha256=receipt["mapping_sha256"], annotation_role=receipt["annotation_role"], original_first_attempt=receipt["original_first_attempt"])
    return sources


def merged_usage(output):
    output = Path(output).resolve()
    history = native.read_json(output / "history.json")
    path = output / "execution_usage.jsonl"
    entries = history["execution_usage_records"] + (protocol.parse_jsonl(path.read_bytes()) if path.exists() else [])
    seen, totals, known = set(), {}, 0
    for value in entries:
        identifier = value["execution_id"]
        if not identifier or identifier in seen:
            raise ValueError("Original/failed/recovery execution usage duplicated")
        seen.add(identifier)
        if value.get("usage") is not None:
            known += 1
            for key, count in value["usage"].items():
                if type(count) is int and count >= 0:
                    totals[key] = totals.get(key, 0) + count
    return dict(executions=len(seen), known_usage_executions=known, native_token_totals=totals,
        counting_unit="Unique original, failed, corrective and transport-recovery invocation IDs, not labels",
        scope="Annotation workflow only; authoring/adjudication/answer experiments are separate",
        subscription_dollar_cost=None)


def prepare(root, pool_dir, source_run, *, authorize_retry=False):
    if authorize_retry is not True:
        raise ValueError("Explicit user-authorized unchanged-input transport recovery required")
    pool, rows, questions, protected, bindings = wire.previous.load_pool(Path(root), pool_dir)
    source_run = Path(source_run).resolve()
    if not source_run.is_relative_to(pool / "cohort_annotations_clarified"):
        raise ValueError("Stopped source must belong to this exact candidate pool")
    packets = native.read_json(source_run / "packets.json")
    history, records = load_history(source_run, packets)
    original = history["original_specification"]
    if original["bindings"] != bindings or original["eligible_record_ids"] != [row["record_id"] for row in rows]:
        raise ValueError("Frozen original question/source bank changed")
    covered = set(history["record_ids"])
    groups = []
    for group in original["groups"]:
        overlap = covered.intersection(group["record_ids"])
        if overlap and overlap != set(group["record_ids"]):
            raise ValueError("A preserved whole source group is only partly covered")
        if overlap:
            continue
        retry = group["batch"] == FAILED_BATCH
        groups.append({**group, "annotation_role": "transport_retry_without_prior_label" if retry else "format_clarified_first_attempt",
                       "original_first_attempt": not retry, "previous_transport_failure_bound": retry})
    if not groups or groups[0]["batch"] != FAILED_BATCH or [key for g in groups for key in g["record_ids"]] != original["eligible_record_ids"][HISTORY_COUNT:]:
        raise ValueError("Only the exact failed pair followed by the untouched remaining groups may execute")
    protected.update(history["protected_sha256"])
    protected[str(Path(__file__).resolve())] = protocol.sha256(Path(__file__).read_bytes())
    output = pool / "cohort_annotations_transport_recovery" / ("run-" + uuid4().hex[:10])
    spec = {**original, "version": VERSION, "source_run": str(source_run), "groups": groups,
        "history_sha256": protocol.sha256(protocol.canonical_bytes(history)),
        "new_role": "declared_transport_retry_and_unchanged_future_first_attempts",
        "explicit_user_authorized_unchanged_input_retry": True, "controlled_retry_contexts": 2,
        "failed_record_ids": history["failed_group"]["record_ids"], "old_source_coverage": HISTORY_COUNT,
        "failed_usage_is_unknown_not_zero": True, "old_failures_retained": True,
        "implementation_sha256": {**original["implementation_sha256"], str(Path(__file__).resolve()): protocol.sha256(Path(__file__).read_bytes())}}
    values = {"packets.json": packets, "history.json": history, "groups.json": groups,
        "protocol.txt": wire.batch.BATCH_PROMPT.encode(), "format_instruction.txt": clarification.CORRECTION_INSTRUCTION.encode(),
        "protected_inputs.json": protected, "runner_script.py": Path(__file__).read_bytes()}
    for name, value in values.items():
        native.write_new(output / name, value)
    native.write_new(output / "manifest.json", dict(specification=spec,
        immutable_sha256={name: protocol.sha256((output / name).read_bytes()) for name in values},
        provenance="machine_only_declared_unchanged_input_transport_recovery"))
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
            value.update(annotation_role=group["annotation_role"], original_first_attempt=group["original_first_attempt"],
                         usage_reference_execution_id=execution_id)
            envelopes[identifier] = value
            errors.extend(value["validation_errors"])
    except (OSError, ValueError, TypeError) as error:
        errors.append(str(error))
    native.write_new(attempt / "envelopes.json", envelopes)
    native.protected_unchanged(protected)
    receipt = dict(accepted=not errors, errors=errors, stream=stream, batch=group["batch"], record_ids=group["record_ids"],
        process_exit_code=status, execution_id=execution_id, usage=events["usage"], elapsed_seconds=perf_counter() - start,
        execution_protocol=VERSION, annotation_role=group["annotation_role"], original_first_attempt=group["original_first_attempt"],
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
    history = native.read_json(output / "history.json")
    original_manifest = Path(history["source_run"]) / "manifest.json"
    if protocol.sha256(original_manifest.read_bytes()) != STOP_MANIFEST_SHA256:
        raise ValueError("Pinned original source manifest changed")
    original = native.read_json(original_manifest)["specification"]
    if original != history["original_specification"]:
        raise ValueError("Recovery cached original specification changed")
    for key in ("requested_model", "native_cli_sha256", "disabled_mcp_names", "native_command_identity", "bindings"):
        if spec[key] != original[key]:
            raise ValueError("Recovery model/binary/isolation/source identity changed")
    expected_groups = []
    covered = set(history["record_ids"])
    if history["record_ids"] != original["eligible_record_ids"][:HISTORY_COUNT]:
        raise ValueError("Recovery history must retain the original exact accepted prefix")
    for group in original["groups"]:
        if set(group["record_ids"]) <= covered:
            continue
        retry = group["batch"] == FAILED_BATCH
        expected_groups.append({**group,
            "annotation_role": "transport_retry_without_prior_label" if retry else "format_clarified_first_attempt",
            "original_first_attempt": not retry, "previous_transport_failure_bound": retry})
    if spec["groups"] != expected_groups or native.read_json(output / "groups.json") != expected_groups:
        raise ValueError("Sealed recovery whole-source groups or roles changed")
    original_retry = next(group for group in original["groups"] if group["batch"] == FAILED_BATCH)
    if (spec["failed_record_ids"] != original_retry["record_ids"]
            or history["failed_group"]["record_ids"] != original_retry["record_ids"]
            or spec.get("controlled_retry_contexts") != 2):
        raise ValueError("Retry barrier and agreement exclusion must bind exactly the declared failed pair")
    path = output / "progress.jsonl"
    if path.exists() and any(e.get("accepted") is False or e.get("type") == "failed" for e in protocol.parse_jsonl(path.read_bytes())):
        raise ValueError("Clarified failure retained; stop without automatic retry")
    progress = protocol.parse_jsonl(path.read_bytes()) if path.exists() else []
    bound_attempts = {}
    for event in progress:
        if event.get("type") != "completed":
            raise ValueError("Unknown durable recovery progress event")
        receipt_path = native.contained_file(output, event["receipt"])
        if str(receipt_path) in bound_attempts or protocol.sha256(receipt_path.read_bytes()) != event["receipt_sha256"]:
            raise ValueError("Recovery progress has duplicated or changed attempt receipt")
        bound_attempts[str(receipt_path)] = event["receipt_sha256"]
    own_attempts = {str((attempt / "receipt.json").resolve()) for stream in native.STREAMS
                    for attempt in (output / stream).glob("batch-*/attempt-*") if attempt.is_dir()}
    if own_attempts != set(bound_attempts):
        raise ValueError("Undurable or interrupted recovery attempt retained; explicit owner review required")
    packets = native.read_json(output / "packets.json")
    completed = {stream: completed_records(output, stream, packets) for stream in native.STREAMS}
    groups = native.read_json(output / "groups.json")
    retry_ids = set(spec["failed_record_ids"])
    if not all(retry_ids <= set(values) for values in completed.values()):
        groups = [group for group in groups if group["batch"] == FAILED_BATCH]
    jobs = previous.pending_jobs(groups, completed, max_native_contexts)
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
    history = native.read_json(output / "history.json")
    failed_ids = set(spec["failed_record_ids"])
    original_groups = history["original_specification"]["groups"]
    future_ids = [identifier for group in original_groups for identifier in group["record_ids"] if identifier not in failed_ids]
    future_packets = {key: packets[key] for key in future_ids}
    future_agreement = protocol.raw_agreement(future_packets,
        [value for key, value in completed["a"].items() if key in future_packets],
        [value for key, value in completed["b"].items() if key in future_packets])
    future_agreement.update(agreement_population="format_clarified_original_first_attempts_excluding_transport_retry",
        excludes_historical350_and_corrective10=True, excluded_transport_failure_record_ids=sorted(failed_ids))
    native.write_new(output / ("future_first_attempt_agreement-" + uuid4().hex[:10] + ".json"), future_agreement)
    retry_packets = {key: packets[key] for key in failed_ids}
    retry_agreement = protocol.raw_agreement(retry_packets,
        [v for key, v in completed["a"].items() if key in failed_ids],
        [v for key, v in completed["b"].items() if key in failed_ids])
    retry_agreement.update(agreement_population="declared_transport_retry_without_prior_output_only", original_first_attempt=False)
    native.write_new(output / ("transport_retry_agreement-" + uuid4().hex[:10] + ".json"), retry_agreement)
    summary = dict(output=str(output), accepted_counts={stream: len(values) for stream, values in completed.items()},
        new_native_invocations=calls, failures=failures, historical_records_reused_per_stream=HISTORY_COUNT,
        status="selected_transport_recovery_calls_complete" if not failures and not (output / "STOP_REQUESTED").exists() else "stopped_without_automatic_retry",
        original350_corrective10_futureIAA_reported_separately=True, transport_retry_agreement_separate=True, human_gold=False, external_paid_api_calls=0,
        subscription_usage=merged_usage(output),
        subscription_dollar_cost=None)
    native.write_new(output / ("summary-" + uuid4().hex[:10] + ".json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("prepare")
    for name in ("canonical-root", "pool-dir", "source-run"):
        before.add_argument("--" + name, type=Path, required=True)
    before.add_argument("--authorize-transport-retry", action="store_true")
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--max-native-contexts", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        directory, records = prepare(args.canonical_root, args.pool_dir, args.source_run, authorize_retry=args.authorize_transport_retry)
        print(json.dumps(dict(output=str(directory), accepted_counts={s: len(v) for s, v in records.items()}, native_calls=0)))
    else:
        result = run(args.output, max_native_contexts=args.max_native_contexts)
        raise SystemExit(1 if result["failures"] else 0)


