"""One declared source-only corrective pair; never rewrite blinded history.

Preparation is model-free. Native execution requires a separate reviewed owner
authorization bound to the complete prepared manifest. Corrective executions
are ineligible for original-first-attempt agreement. Equal-input new pair
outputs can pass the unchanged source/label/pair standards with explicit lineage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from time import perf_counter
from uuid import uuid4

import run_codex_annotation as native
import run_codex_batch_annotations as batch
import run_codex_cohort_wire as wire
import record_codex_wire_retry as retry
from oil_evidence import machine_evaluation as protocol
from oil_evidence import wire_codec as codec


VERSION = "source-only-every-citation-corrective-pair-v1"
AUTHORIZED_FAILURE_HASHES = (
    "4fa3df9e6cdf271b9fc764f462a23c3a17e346d3b472cf20eb09f0e466c8a2ee",
    "f5fbeb76639163bd7baf4b7742155e61d5fcad7c318fe2f5016fa5211ebcae11",
)
SOURCE_ERROR = "Decoded canonical label invalid: Every answer point needs citations with exact eligible quotes"
CORRECTION_INSTRUCTION = """

EXPLICIT CITATION-QUOTE COVERAGE REQUIREMENT
Independently judge the supplied questions and full evidence pools afresh. You
have not been given prior answers, labels, disagreements or reference answers.
For EVERY passage ID listed in ANY answer_points[].citations, include at least
one quotes[] entry for that SAME passage ID. Its text must be an exact substring
of that own-record, cutoff-eligible source and must support the cited claim.
Multiple cited passages require corresponding exact quotes for EACH passage;
a quote from one passage does not satisfy another passage's citation.
Do not manufacture or paraphrase a quote, borrow another record's evidence, or
force a particular answerability, relevance grade or event status. If the pool
does not support a requested claim, state the limitation or evidence gap under
the unchanged annotation rules. Return the same complete six-field labels for
every record, using only the supplied source data.
"""


def prepare_source_batch(rows):
    baseline = codec.prepare_batch(rows, batch.BATCH_PROMPT, native.output_schema)
    prompt = baseline["prompt"] + CORRECTION_INSTRUCTION
    stdin = prompt.encode("utf-8") + b"\n\nUNTRUSTED_RECORD_PACKETS_JSON\n" + protocol.canonical_bytes(baseline["wire_payload"]) + b"\n"
    if len(stdin) + len(protocol.canonical_bytes(baseline["schema"])) > 262144:
        raise ValueError("Full corrective stdin plus schema exceeds the unchanged byte bound")
    return {**baseline, "prompt": prompt, "stdin": stdin,
        "baseline_wire_stdin_sha256": baseline["actual_wire_stdin_sha256"],
        "actual_wire_stdin_sha256": protocol.sha256(stdin),
        "corrective_prompt_sha256": protocol.sha256(prompt.encode("utf-8")),
        "corrective_instruction_sha256": protocol.sha256(CORRECTION_INSTRUCTION.encode("utf-8")),
        "canonical_packet_hash_scope": "Unchanged baseline semantic source packet, NOT actual corrective model input"}


def source_context(annotation_run, failure_paths):
    output = Path(annotation_run).resolve()
    if len(failure_paths) != 2:
        raise ValueError("Preserve the original rejection and the one failed exact-input retry")
    manifest = native.read_json(output / "manifest.json")
    protected = dict(native.read_json(output / "protected_inputs.json"))
    for name, expected in manifest["immutable_sha256"].items():
        protected[str(native.contained_file(output, name))] = expected
    protected[str(output / "manifest.json")] = protocol.sha256((output / "manifest.json").read_bytes())
    native.protected_unchanged(protected)
    progress = (output / "progress.jsonl").read_bytes()
    events = protocol.parse_jsonl(progress, "unchanged rejected source history")
    accepted = retry.accepted_receipts(output, events)
    if "b:batch-0019" in accepted:
        raise ValueError("An accepted blinded batch must never be corrected or replaced")
    failed = []
    for index, value in enumerate(failure_paths):
        path = Path(value).resolve()
        if not path.is_relative_to(output) or path.name != "receipt.json":
            raise ValueError("Rejected receipts must stay in their original annotation run")
        digest = protocol.sha256(path.read_bytes())
        if digest != AUTHORIZED_FAILURE_HASHES[index]:
            raise ValueError("This declared correction applies only to the reviewed two failures")
        receipt = native.read_json(path)
        retry.verify_raw_artifacts(path, receipt)
        execution = native.parse_execution_events((path.parent / "events.jsonl").read_bytes())
        if execution["errors"] or execution["thread_id"] != receipt["execution_id"] or execution["usage"] != receipt["usage"]:
            raise ValueError("Original native/quota execution evidence is not eligible for format correction")
        if (receipt.get("accepted") is not False or receipt.get("stream") != "b"
                or receipt.get("batch") != "batch-0019" or receipt.get("process_exit_code") != 0
                or receipt.get("errors") != [SOURCE_ERROR]):
            raise ValueError("Unknown/native/quota failure is not covered by this correction route")
        bound = [event for event in events if event.get("receipt") == path.relative_to(output).as_posix()]
        if len(bound) != 1 or bound[0].get("accepted") is not False or bound[0]["receipt_sha256"] != digest:
            raise ValueError("Rejected receipt is not uniquely bound to original progress")
        failed.append(dict(path=str(path), sha256=digest, execution_id=receipt["execution_id"],
                           record_ids=receipt["record_ids"]))
        protected[str(path)] = digest
        protected.update({str(native.contained_file(path.parent, name)): expected for name, expected in receipt["artifacts_sha256"].items()})
    counterpart = Path(accepted["a:batch-0019"]["receipt"])
    other = native.read_json(counterpart)
    retry.verify_raw_artifacts(counterpart, other)
    other_execution = native.parse_execution_events((counterpart.parent / "events.jsonl").read_bytes())
    if (other.get("accepted") is not True or other.get("process_exit_code") != 0 or other.get("errors")
            or other_execution["errors"] or other_execution["thread_id"] != other["execution_id"]
            or other_execution["usage"] != other["usage"]):
        raise ValueError("Original accepted A execution evidence changed or contains native errors")
    ids = failed[0]["record_ids"]
    if len(ids) != 10 or len(set(ids)) != 10 or failed[1]["record_ids"] != ids or other["record_ids"] != ids:
        raise ValueError("Correction must preserve the whole original ten-record batch")
    if len({other["execution_id"], *(value["execution_id"] for value in failed)}) != 3:
        raise ValueError("Original A and the two failures must remain distinct executions")
    original_hashes = retry.artifact_hashes(Path(failed[0]["path"]).parent)
    if any(retry.artifact_hashes(Path(value["path"]).parent) != original_hashes for value in failed[1:]) or retry.artifact_hashes(counterpart.parent) != original_hashes:
        raise ValueError("Original A and both rejected physical inputs differ")
    packets = native.read_json(output / "packets.json")
    rows = [packets[identifier]["payload"] for identifier in ids]
    prepared = codec.prepare_batch(rows, batch.BATCH_PROMPT, native.output_schema)
    original = Path(failed[0]["path"]).parent
    if (original.joinpath("stdin.txt").read_bytes() != prepared["stdin"]
            or original.joinpath("schema.json").read_bytes() != protocol.canonical_bytes(prepared["schema"])
            or original.joinpath("codec.json").read_bytes() != protocol.canonical_bytes(prepared["codec"])
            or original.joinpath("wire_payload.json").read_bytes() != protocol.canonical_bytes(prepared["wire_payload"])
            or original.joinpath("canonical_semantic_packets.json").read_bytes() != protocol.canonical_bytes(prepared["canonical_semantic_packets"])):
        raise ValueError("Original complete physical input no longer reconstructs from the frozen source pool")
    requested = manifest["specification"]["requested_model"]
    if any(retry.requested_model_from_command(Path(value["path"]).parent / "command.json") != requested for value in failed) or retry.requested_model_from_command(counterpart.parent / "command.json") != requested:
        raise ValueError("Actual original native model request differs")
    command_identity = normalized_command(native.read_json(counterpart.parent / "command.json"))
    if any(normalized_command(native.read_json(Path(value["path"]).parent / "command.json")) != command_identity for value in failed):
        raise ValueError("Original failed and accepted native isolation commands differ")
    wire.verify_attempt(counterpart.parent, other, packets)
    protected[str(counterpart)] = protocol.sha256(counterpart.read_bytes())
    protected.update({str(native.contained_file(counterpart.parent, name)): expected for name, expected in other["artifacts_sha256"].items()})
    protected.update({str(Path(value["receipt"])): value["receipt_sha256"] for value in accepted.values()})
    for path in (__file__, retry.__file__):
        protected[str(Path(path).resolve())] = protocol.sha256(Path(path).read_bytes())
    return dict(annotation_run=str(output), record_ids=ids, rejected_receipts=failed,
        counterpart_a=dict(path=str(counterpart), sha256=protocol.sha256(counterpart.read_bytes()), execution_id=other["execution_id"]),
        accepted_receipts_before=accepted, requested_model=requested, native_specification=manifest["specification"],
        protected_sha256=protected, rows=rows, progress_prefix=progress,
        usage_prefix=(output / "execution_usage.jsonl").read_bytes())


def prepare(annotation_run, failure_paths):
    context = source_context(annotation_run, failure_paths)
    prepared = prepare_source_batch(context["rows"])
    output = Path(annotation_run).resolve() / "correct" / uuid4().hex[:10]
    output.mkdir(parents=True, exist_ok=False)
    values = {"stdin.txt": prepared["stdin"], "prompt.txt": prepared["prompt"].encode("utf-8"),
        "instruction.txt": CORRECTION_INSTRUCTION.encode("utf-8"), "schema.json": prepared["schema"],
        "codec.json": prepared["codec"], "wire_payload.json": prepared["wire_payload"],
        "canonical_semantic_packets.json": prepared["canonical_semantic_packets"],
        "original_progress_prefix.jsonl": context["progress_prefix"], "original_usage_prefix.jsonl": context["usage_prefix"],
        "protected_inputs.json": context["protected_sha256"], "runner_script.py": Path(__file__).read_bytes()}
    for name, value in values.items():
        native.write_new(output / name, value)
    spec = dict(version=VERSION, created_utc=native.utc_now(), annotation_run=context["annotation_run"],
        record_ids=context["record_ids"], rejected_receipts=context["rejected_receipts"], counterpart_a=context["counterpart_a"],
        accepted_receipts_before=context["accepted_receipts_before"], requested_model=context["requested_model"],
        baseline_native_specification=context["native_specification"], baseline_semantic_prompt_sha256=protocol.sha256(batch.BATCH_PROMPT.encode()),
        baseline_wire_stdin_sha256=prepared["baseline_wire_stdin_sha256"], full_stdin_sha256=prepared["actual_wire_stdin_sha256"],
        corrective_prompt_sha256=prepared["corrective_prompt_sha256"], corrective_instruction_sha256=prepared["corrective_instruction_sha256"],
        schema_sha256=prepared["actual_wire_schema_sha256"], mapping_sha256=prepared["codec"]["mapping_sha256"],
        canonical_packet_hash_scope=prepared["canonical_packet_hash_scope"], native_calls_prepared=0, native_calls_allowed_after_review=2,
        execution_role="source_only_corrective_pair", original_blinded_annotation=False,
        old_failures_reclassified_as_valid=False, no_prior_labels_or_errors_in_model_input=True,
        original_agreement_substitution_allowed=False, equal_input_corrective_pair_required=True,
        unchanged_formal_freeze_compatible=True,
        corrected_labels_still_require_source_adjudication=True, human_gold=False)
    manifest = dict(specification=spec, immutable_sha256={name: protocol.sha256((output / name).read_bytes()) for name in values})
    native.write_new(output / "manifest.json", manifest)
    return output, manifest


def verify_prepared(directory):
    directory = Path(directory).resolve()
    manifest = native.read_json(directory / "manifest.json")
    for name, expected in manifest["immutable_sha256"].items():
        if protocol.sha256(native.contained_file(directory, name).read_bytes()) != expected:
            raise ValueError("Prepared corrective artifact changed")
    native.protected_unchanged(native.read_json(directory / "protected_inputs.json"))
    if (directory / "runner_script.py").read_bytes() != Path(__file__).read_bytes():
        raise ValueError("Corrective adapter implementation changed after preparation")
    spec = manifest["specification"]
    context = source_context(spec["annotation_run"], [value["path"] for value in spec["rejected_receipts"]])
    prepared = prepare_source_batch(context["rows"])
    expected = {"stdin.txt": prepared["stdin"], "prompt.txt": prepared["prompt"].encode(),
        "instruction.txt": CORRECTION_INSTRUCTION.encode(), "schema.json": protocol.canonical_bytes(prepared["schema"]),
        "codec.json": protocol.canonical_bytes(prepared["codec"]), "wire_payload.json": protocol.canonical_bytes(prepared["wire_payload"]),
        "canonical_semantic_packets.json": protocol.canonical_bytes(prepared["canonical_semantic_packets"])}
    if any((directory / name).read_bytes() != value for name, value in expected.items()):
        raise ValueError("Corrective input does not exactly reconstruct from unchanged sources plus fixed instruction")
    if (spec["full_stdin_sha256"] != prepared["actual_wire_stdin_sha256"]
            or spec["corrective_instruction_sha256"] != prepared["corrective_instruction_sha256"]
            or spec["original_blinded_annotation"] is not False or spec["original_agreement_substitution_allowed"] is not False):
        raise ValueError("Corrective execution identity or explicit provenance changed")
    for name, current in (("original_progress_prefix.jsonl", context["progress_prefix"]), ("original_usage_prefix.jsonl", context["usage_prefix"])):
        if not current.startswith((directory / name).read_bytes()):
            raise ValueError("Original failed execution history was rewritten")
    return manifest, context, prepared


def normalized_command(command):
    destinations = {"--cd", "--output-schema", "--output-last-message"}
    result, index = [], 0
    while index < len(command):
        item = command[index]
        result.append(item)
        if item in destinations:
            if index + 1 >= len(command):
                raise ValueError("Incomplete native command destination")
            result.append("FRESH_CONTEXT_DESTINATION")
            index += 2
        else:
            index += 1
    return result


def validate_authorization(directory, authorization_path):
    directory = Path(directory)
    path = Path(authorization_path).resolve()
    value = native.read_json(path)
    if (value.get("root_authorized") is not True or value.get("native_calls_allowed") != 2
            or type(value.get("native_calls_allowed")) is not int or value.get("no_label_feedback") is not True
            or value.get("manifest_sha256") != protocol.sha256((directory / "manifest.json").read_bytes())
            or value.get("corrective_instruction_sha256") != protocol.sha256(CORRECTION_INSTRUCTION.encode())
            or value.get("route") != VERSION):
        raise ValueError("A separately reviewed two-context corrective-pair authorization is required")
    return path, value


def execute_stream_once(directory, authorization_path, stream, *, executor=native.invoke_cli, native_setup=None, timeout_seconds=900):
    if stream not in native.STREAMS:
        raise ValueError("Only the two declared fresh corrective streams may execute")
    directory = Path(directory).resolve()
    manifest, source, prepared = verify_prepared(directory)
    authorization_path, authorization = validate_authorization(directory, authorization_path)
    attempt = directory / stream / "execution"
    attempt.mkdir(parents=True, exist_ok=False)  # Durable once-only boundary, even for rejection.
    native.write_new(attempt / "authorization.json", authorization)
    for name in ("stdin.txt", "schema.json", "codec.json", "wire_payload.json", "canonical_semantic_packets.json", "prompt.txt", "instruction.txt"):
        native.write_new(attempt / name, (directory / name).read_bytes())
    if native_setup is None:
        binary = native.discover_native_cli(native.DEFAULT_CLI)
        environment = native.child_environment()
        mcp_names = native.configured_mcp_names(environment)
    else:
        binary, environment, mcp_names = native_setup
    original = source["native_specification"]
    if protocol.sha256(Path(binary).read_bytes()) != original["native_cli_sha256"] or mcp_names != original["disabled_mcp_names"]:
        raise ValueError("Corrective native binary or isolated tool configuration changed")
    context = Path(tempfile.mkdtemp(prefix="oil-source-correction-"))
    native.write_new(context / "packets.json", prepared["wire_payload"])
    native.write_new(context / "protocol.txt", prepared["prompt"].encode())
    native.write_new(context / "schema.json", prepared["schema"])
    final = context / "final.json"
    command = native.cli_command(binary, context, context / "schema.json", final, source["requested_model"], mcp_names)
    original_command = native.read_json(Path(source["counterpart_a"]["path"]).parent / "command.json")
    if normalized_command(command) != normalized_command(original_command):
        raise ValueError("Corrective native isolation or non-ephemeral command setting changed")
    native.write_new(attempt / "command.json", command)
    native.write_new(attempt / "sent_input.json", dict(stdin_sha256=protocol.sha256(prepared["stdin"]),
        stdin_bytes=len(prepared["stdin"]), schema_sha256=protocol.sha256((context / "schema.json").read_bytes()),
        context_protocol_sha256=protocol.sha256((context / "protocol.txt").read_bytes()),
        context_payload_sha256=protocol.sha256((context / "packets.json").read_bytes()),
        source="Exact byte argument passed to isolated native executor; not canonical semantic hash"))
    start = perf_counter()
    execution_errors = []
    try:
        status = executor(command, prepared["stdin"], context, attempt, environment, timeout_seconds)
    except Exception as error:
        status = None
        execution_errors.append("Native invocation failed; retained raw events: " + str(error))
    if final.exists():
        native.write_new(attempt / "raw_final.json", final.read_bytes())
    if not (attempt / "events.jsonl").exists():
        native.write_new(attempt / "events.jsonl", b"")
    if not (attempt / "stderr.txt").exists():
        native.write_new(attempt / "stderr.txt", b"")
    events = native.parse_execution_events((attempt / "events.jsonl").read_bytes())
    errors = execution_errors + list(events["errors"]) + (["Native process returned nonzero status"] if status not in (0, None) else [])
    execution_id = events["thread_id"]
    if not execution_id or execution_id in {source["counterpart_a"]["execution_id"], *(v["execution_id"] for v in source["rejected_receipts"])}:
        errors.append("One distinct observed fresh execution ID is required")
    identity = dict(service="codex_cli_chatgpt_subscription", requested_model=source["requested_model"] or "cli_default_unresolved",
        observed_model=events["observed_model"], execution_id=execution_id or "unavailable-" + uuid4().hex,
        observation_source="execution_metadata" if events["observed_model"] else "unavailable",
        identity_note="Native execution metadata only; never assistant self-report")
    envelopes = {}
    try:
        raw = (attempt / "raw_final.json").read_bytes().decode("utf-8")
        decoded = codec.decode_batch(raw, prepared["codec"], source["rows"])["decoded_records"]
        for alias, binding in prepared["codec"]["records"].items():
            identifier = binding["record_id"]
            value = protocol.wrap_independent_output(prepared["canonical_semantic_packets"][identifier], raw, identity, identity["execution_id"],
                dict(annotation_role="source_only_corrective_pair", corrective_protocol=VERSION,
                    corrective_instruction_sha256=prepared["corrective_instruction_sha256"],
                    actual_corrective_prompt_sha256=prepared["corrective_prompt_sha256"],
                    canonical_packet_hash_scope=prepared["canonical_packet_hash_scope"]), usage=None,
                execution_input_sha256=prepared["actual_wire_stdin_sha256"], raw_label_path=[alias, "label"],
                raw_record_id_path=[alias, "record_id"], raw_codec=prepared["codec"])
            if value["label"] != decoded[identifier]["label"]:
                raise ValueError("Corrective raw label differs from exact decoder result")
            value.update(annotation_role="source_only_corrective_pair", original_blinded_annotation=False,
                excluded_from_original_agreement=True, usage_reference_execution_id=identity["execution_id"],
                corrective_stream=stream, fresh_blinded_corrective_context=True,
                correction_manifest_sha256=protocol.sha256((directory / "manifest.json").read_bytes()))
            envelopes[identifier] = value
            errors.extend(value["validation_errors"])
    except (OSError, ValueError, TypeError) as error:
        errors.append(str(error))
    native.write_new(attempt / "envelopes.json", envelopes)
    native.protected_unchanged(source["protected_sha256"])
    if protocol.sha256(authorization_path.read_bytes()) != protocol.sha256((attempt / "authorization.json").read_bytes()):
        errors.append("Reviewed corrective authorization changed during execution")
    receipt = dict(accepted=not errors, errors=errors, execution_id=execution_id, usage=events["usage"], stream=stream,
        record_ids=source["record_ids"], process_exit_code=status, elapsed_seconds=perf_counter() - start,
        execution_role="source_only_corrective_pair", execution_protocol=VERSION, original_blinded_annotation=False,
        full_stdin_sha256=prepared["actual_wire_stdin_sha256"], corrective_prompt_sha256=prepared["corrective_prompt_sha256"],
        corrective_instruction_sha256=prepared["corrective_instruction_sha256"], schema_sha256=prepared["actual_wire_schema_sha256"],
        mapping_sha256=prepared["codec"]["mapping_sha256"],
        manifest_sha256=protocol.sha256((directory / "manifest.json").read_bytes()),
        authorization_sha256=protocol.sha256(authorization_path.read_bytes()),
        original_failures_retained=True, original_rejected_event_count=2, whole_batch_atomic=True,
        native_calls_actual=1, external_paid_api_calls=0, human_gold=False,
        artifacts_sha256={path.name: protocol.sha256(path.read_bytes()) for path in attempt.iterdir() if path.is_file()})
    native.write_new(attempt / "receipt.json", receipt)
    return receipt


def verify_corrective_attempt(directory, stream):
    directory = Path(directory).resolve()
    manifest, source, prepared = verify_prepared(directory)
    attempt = directory / stream / "execution"
    receipt = native.read_json(attempt / "receipt.json")
    retry.verify_raw_artifacts(attempt / "receipt.json", receipt)
    if (receipt.get("accepted") is not True or receipt.get("errors") or receipt.get("process_exit_code") != 0
            or receipt.get("stream") != stream or receipt.get("record_ids") != source["record_ids"]
            or receipt.get("execution_protocol") != VERSION or receipt.get("original_blinded_annotation") is not False
            or receipt.get("manifest_sha256") != protocol.sha256((directory / "manifest.json").read_bytes())
            or receipt.get("full_stdin_sha256") != prepared["actual_wire_stdin_sha256"]
            or receipt.get("schema_sha256") != prepared["actual_wire_schema_sha256"]
            or receipt.get("mapping_sha256") != prepared["codec"]["mapping_sha256"]
            or receipt.get("corrective_prompt_sha256") != prepared["corrective_prompt_sha256"]
            or receipt.get("corrective_instruction_sha256") != prepared["corrective_instruction_sha256"]):
        raise ValueError("Corrective execution is not a complete accepted declared source-only stream")
    events = native.parse_execution_events((attempt / "events.jsonl").read_bytes())
    if events["errors"] or events["thread_id"] != receipt["execution_id"] or events["usage"] != receipt["usage"]:
        raise ValueError("Corrective native events contain errors or a different execution")
    if receipt["execution_id"] in {source["counterpart_a"]["execution_id"], *(v["execution_id"] for v in source["rejected_receipts"])}:
        raise ValueError("Corrective context reused an old execution identity")
    for name in ("stdin.txt", "schema.json", "codec.json", "wire_payload.json", "canonical_semantic_packets.json", "prompt.txt", "instruction.txt"):
        if (attempt / name).read_bytes() != (directory / name).read_bytes():
            raise ValueError("Actual corrective attempt input artifact differs from its prepared full input")
    sent = native.read_json(attempt / "sent_input.json")
    if (sent.get("stdin_sha256") != prepared["actual_wire_stdin_sha256"] or sent.get("stdin_bytes") != len(prepared["stdin"])
            or sent.get("schema_sha256") != prepared["actual_wire_schema_sha256"]
            or sent.get("context_protocol_sha256") != prepared["corrective_prompt_sha256"]
            or sent.get("context_payload_sha256") != protocol.sha256(protocol.canonical_bytes(prepared["wire_payload"]))):
        raise ValueError("Corrective sent-input execution evidence differs")
    if retry.requested_model_from_command(attempt / "command.json") != source["requested_model"]:
        raise ValueError("Corrective actual native model request differs")
    if normalized_command(native.read_json(attempt / "command.json")) != normalized_command(native.read_json(Path(source["counterpart_a"]["path"]).parent / "command.json")):
        raise ValueError("Corrective actual isolation command differs from the frozen source protocol")
    authorization_path, _ = validate_authorization(directory, attempt / "authorization.json")
    if protocol.sha256(authorization_path.read_bytes()) != receipt["authorization_sha256"]:
        raise ValueError("Corrective receipt authorization hash differs")
    raw = (attempt / "raw_final.json").read_bytes()
    decoded = codec.decode_batch(raw, prepared["codec"], source["rows"])["decoded_records"]
    envelopes = native.read_json(attempt / "envelopes.json")
    if set(envelopes) != set(source["record_ids"]):
        raise ValueError("Corrective envelope coverage differs from the whole original batch")
    for identifier, value in envelopes.items():
        if (value.get("annotation_role") != "source_only_corrective_pair"
                or value.get("original_blinded_annotation") is not False
                or value.get("excluded_from_original_agreement") is not True
                or value.get("corrective_stream") != stream or value["raw_output"].encode() != raw
                or value["label"] != decoded[identifier]["label"]
                or value["context_id"] != receipt["execution_id"] or value["identity"]["execution_id"] != receipt["execution_id"]
                or value["identity"]["observed_model"] != events["observed_model"]
                or value["identity"]["requested_model"] != (source["requested_model"] or "cli_default_unresolved")
                or value["decoding"].get("actual_corrective_prompt_sha256") != prepared["corrective_prompt_sha256"]
                or value["decoding"].get("corrective_instruction_sha256") != prepared["corrective_instruction_sha256"]
                or value["execution_input_sha256"] != prepared["actual_wire_stdin_sha256"]
                or protocol.validate_independent_output(value, prepared["canonical_semantic_packets"][identifier])):
            raise ValueError("Corrective envelope does not retain its exact new raw/source/role binding")
    return receipt, envelopes


def execute_pair_once(directory, authorization_path, *, executor=native.invoke_cli, native_setup=None, timeout_seconds=900):
    directory = Path(directory).resolve()
    _, source, _ = verify_prepared(directory)
    validate_authorization(directory, authorization_path)
    if any((directory / stream / "execution").exists() for stream in native.STREAMS):
        raise ValueError("Corrective pair execution is once-only; partial or rejected calls cannot resume")
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = {stream: workers.submit(execute_stream_once, directory, authorization_path, stream,
            executor=executor, native_setup=native_setup, timeout_seconds=timeout_seconds) for stream in native.STREAMS}
        receipts = {stream: future.result() for stream, future in futures.items()}
    accepted = all(value["accepted"] for value in receipts.values())
    agreement = None
    if accepted:
        records = {stream: verify_corrective_attempt(directory, stream)[1] for stream in native.STREAMS}
        packets = native.read_json(directory / "canonical_semantic_packets.json")
        agreement = protocol.raw_agreement(packets, list(records["a"].values()), list(records["b"].values()))
        if agreement["paired_records"] != len(source["record_ids"]):
            raise ValueError("Corrective pair actual inputs or fresh contexts do not match")
        agreement.update(agreement_population="new_source_only_corrective_pair",
                         original_first_attempt_agreement=False, original_failures_reclassified=False)
        native.write_new(directory / "corrective_pair_agreement.json", agreement)
    native.write_new(directory / "execution_usage.jsonl", b"".join(protocol.canonical_bytes(dict(
        execution_id=value["execution_id"], usage=value["usage"], stream=stream,
        counting_unit="one actual corrective execution, not ten extracted records")) + b"\n" for stream, value in receipts.items()))
    certificate = dict(version=VERSION, created_utc=native.utc_now(), status="source_only_corrective_pair_valid_pending_adjudication" if accepted else "source_only_corrective_pair_failed_stop",
        correction_directory=str(directory), manifest_sha256=protocol.sha256((directory / "manifest.json").read_bytes()),
        authorization_path=str(Path(authorization_path).resolve()), authorization_sha256=protocol.sha256(Path(authorization_path).read_bytes()),
        corrective_receipts={stream: dict(path=str(directory / stream / "execution/receipt.json"),
            sha256=protocol.sha256((directory / stream / "execution/receipt.json").read_bytes()),
            execution_id=value["execution_id"]) for stream, value in receipts.items()},
        rejected_receipts=source["rejected_receipts"], counterpart_a=source["counterpart_a"],
        record_ids=source["record_ids"], correction_accepted=accepted,
        original_rejected_event_count=2, original_blinded_annotation=False,
        original_agreement_substitution_allowed=False, unchanged_formal_freeze_compatible=accepted,
        corrective_pair_agreement_sha256=protocol.sha256((directory / "corrective_pair_agreement.json").read_bytes()) if agreement else None,
        original_progress_sha256=protocol.sha256((directory / "original_progress_prefix.jsonl").read_bytes()),
        corrected_labels_still_require_source_adjudication=True, native_calls_actual=2, human_gold=False)
    native.write_new(directory / "certificate.json", certificate)
    return receipts, certificate


def completed_correction_records(directory, stream):
    directory = Path(directory).resolve()
    manifest, source, _ = verify_prepared(directory)
    certificate = native.read_json(directory / "certificate.json")
    if (certificate.get("correction_accepted") is not True or certificate.get("native_calls_actual") != 2
            or certificate.get("manifest_sha256") != protocol.sha256((directory / "manifest.json").read_bytes())
            or certificate.get("rejected_receipts") != source["rejected_receipts"]
            or certificate.get("counterpart_a") != source["counterpart_a"]
            or certificate.get("record_ids") != source["record_ids"]
            or certificate.get("original_rejected_event_count") != 2
            or certificate.get("original_blinded_annotation") is not False
            or certificate.get("original_agreement_substitution_allowed") is not False):
        raise ValueError("Both whole corrective streams must pass before any replacement is eligible")
    values = {}
    for name in native.STREAMS:
        info = certificate["corrective_receipts"][name]
        path = Path(info["path"])
        if path != directory / name / "execution/receipt.json" or protocol.sha256(path.read_bytes()) != info["sha256"]:
            raise ValueError("Corrective certificate receipt lineage changed")
        receipt, records = verify_corrective_attempt(directory, name)
        if receipt["execution_id"] != info["execution_id"]:
            raise ValueError("Corrective certificate execution lineage changed")
        values[name] = records
    packets = native.read_json(directory / "canonical_semantic_packets.json")
    agreement = protocol.raw_agreement(packets, list(values["a"].values()), list(values["b"].values()))
    if agreement["paired_records"] != len(packets):
        raise ValueError("Corrective pair is not fresh and equal-input")
    saved_agreement = directory / "corrective_pair_agreement.json"
    if protocol.sha256(saved_agreement.read_bytes()) != certificate["corrective_pair_agreement_sha256"]:
        raise ValueError("Corrective-only agreement artifact changed")
    agreement.update(agreement_population="new_source_only_corrective_pair", original_first_attempt_agreement=False,
                     original_failures_reclassified=False)
    if protocol.canonical_bytes(native.read_json(saved_agreement)) != protocol.canonical_bytes(agreement):
        raise ValueError("Corrective-only agreement differs from exact retained pair")
    if certificate.get("original_progress_sha256") != protocol.sha256((directory / "original_progress_prefix.jsonl").read_bytes()):
        raise ValueError("Corrective certificate no longer binds the original failed history")
    return values[stream]


def correction_record_sources(directory, stream):
    directory = Path(directory).resolve()
    records = completed_correction_records(directory, stream)
    receipt = directory / stream / "execution/receipt.json"
    value = native.read_json(receipt)
    return {identifier: dict(receipt=str(receipt), receipt_sha256=protocol.sha256(receipt.read_bytes()),
        execution_id=value["execution_id"], stdin_sha256=value["full_stdin_sha256"], source_run=str(directory),
        migrated_unchanged=False, raw_encoding=codec.VERSION, execution_protocol=VERSION,
        correction_certificate=str(directory / "certificate.json"),
        correction_certificate_sha256=protocol.sha256((directory / "certificate.json").read_bytes())) for identifier in records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("prepare")
    before.add_argument("--annotation-run", type=Path, required=True)
    before.add_argument("--failure-receipt", type=Path, action="append", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--prepared-directory", type=Path, required=True)
    execute.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        directory, manifest = prepare(args.annotation_run, args.failure_receipt)
        print(json.dumps(dict(directory=str(directory), native_calls=0, manifest_sha256=protocol.sha256((directory / "manifest.json").read_bytes())), indent=2))
    else:
        receipts, certificate = execute_pair_once(args.prepared_directory, args.authorization)
        print(json.dumps(certificate, indent=2))
        raise SystemExit(0 if all(value["accepted"] for value in receipts.values()) else 1)
