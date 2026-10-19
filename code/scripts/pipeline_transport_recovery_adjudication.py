"""Declared, one-shot transport recovery with unchanged source-review inputs.

Completed reviews remain beside their original native executions. The old stop
and rejected invocation remain immutable; no failure is projected as accepted.
Only the registered transport-recovery producer is eligible for new exports.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import pipeline_wire_cohort_adjudication as frozen

native, protocol, legacy = frozen.native, frozen.protocol, frozen.legacy
VERSION = "source-review-identical-input-transport-recovery-v1"
PRODUCER_VERSION = "codex-unchanged-input-transport-recovery-v1"
RETRY_ROLE = "transport_retry_without_prior_label"
FIRST_ROLE = "format_clarified_first_attempt"
FAILED_RECEIPT_SHA256 = "70be80ff85187ec60e52f07843ec7521d16da4fa447b77b05e2872fe4f713e5e"
FAILED_PIPELINE_SHA256 = "c12b4099566817cc211b7fd866b0d3a49ac6c00263a9047f9974456ab63a43f5"
STOP_CHECKPOINT_SHA256 = "d78cc05168f671e37217bbdc57fe8bb69ea842f97552fba03dbad836539d5e20"
STOP_MANIFEST_SHA256 = "d57715b34534e92a097a45b6a9b00045c92a83a69b8889a6fb0064df7f09a425"


def merge_guards(guards, additional):
    if any(name in guards and guards[name] != digest for name, digest in additional.items()):
        raise ValueError("Recovery source guard has conflicting original hashes")
    guards.update(additional)


def guard_file(path, guards, expected=None):
    path = Path(path).resolve()
    digest = protocol.sha256(path.read_bytes())
    if expected is not None and digest != expected:
        raise ValueError("Recovery artifact differs from its original immutable hash")
    if str(path) in guards and guards[str(path)] != digest:
        raise ValueError("Recovery guard has conflicting hashes")
    guards[str(path)] = digest
    return digest


def receipt_artifacts(path, guards):
    path = Path(path).resolve()
    guard_file(path, guards)
    receipt = native.read_json(path)
    if Path(receipt["artifact_directory"]).resolve() != path.parent:
        raise ValueError("Native receipt no longer owns its actual execution directory")
    for name, digest in receipt["artifacts_sha256"].items():
        guard_file(native.contained_file(path.parent, name), guards, digest)
    return receipt


def verify_stopped_review(previous_run, *, receipt_sha256=FAILED_RECEIPT_SHA256,
                          failure_sha256=FAILED_PIPELINE_SHA256,
                          checkpoint_sha256=STOP_CHECKPOINT_SHA256, expected_last_chunk=28):
    """Admit only the specifically bound no-answer routing failure, not quota."""
    previous = Path(previous_run).resolve()
    guards = {}
    failure = previous / "failure.json"
    guard_file(failure, guards, failure_sha256)
    checkpoint_path = previous / "routing_stop_checkpoint.json"
    guard_file(checkpoint_path, guards, checkpoint_sha256)
    checkpoint = native.read_json(checkpoint_path)
    if (checkpoint.get("failure_message") != "workspace routing discovery failed"
            or checkpoint.get("quota_error_observed") is not False
            or checkpoint.get("model_output_from_failed_call") is not None
            or checkpoint.get("failed_usage_status") != "unknown_not_zero"
            or checkpoint.get("automatic_retry") is not False
            or checkpoint.get("original_outputs_and_failure_bytes_preserved") is not True
            or checkpoint.get("failed_receipt_sha256") != receipt_sha256
            or checkpoint.get("original_failure_sha256") != failure_sha256):
        raise ValueError("Unknown, quota, answered or unbound failure cannot receive transport recovery")
    if (checkpoint.get("last_complete_chunk") != expected_last_chunk
            or checkpoint.get("first_failed_chunk") != expected_last_chunk + 1
            or checkpoint.get("reviewed_records") != 20 * expected_last_chunk):
        raise ValueError("Original stopped source-review scope changed")
    path = native.contained_file(previous, checkpoint["failed_receipt"])
    guard_file(path, guards, receipt_sha256)
    receipt = receipt_artifacts(path, guards)
    events = native.parse_execution_events((path.parent / "events.jsonl").read_bytes())
    messages = [str(row.get("message", row.get("error", ""))) for row in events["events"]
                if row.get("type") in {"error", "turn.failed"}]
    if (receipt.get("accepted") is not False or receipt.get("role") != "adjudicator"
            or receipt.get("process_exit_code") != 1 or receipt.get("codex_cli_invocations") != 1
            or receipt.get("usage") is not None or receipt.get("external_paid_api_calls") != 0
            or receipt.get("model_tools_disabled") is not True
            or receipt.get("execpolicy_rules_not_disabled") is not True
            or receipt.get("authentication") != "existing_ChatGPT_subscription"
            or receipt.get("execution_id") != checkpoint.get("failed_execution_id")
            or events["thread_id"] != receipt["execution_id"]
            or any(row.get("type") == "turn.completed" for row in events["events"])
            or not messages or any("workspace routing discovery failed" not in message for message in messages)
            or (path.parent / "raw_final.json").exists()):
        raise ValueError("The original rejected native call is not the declared unanswered transport failure")
    if (receipt["artifacts_sha256"]["stdin.txt"] != checkpoint["failed_actual_stdin_sha256"]
            or receipt["artifacts_sha256"]["schema.json"] != checkpoint["failed_schema_sha256"]):
        raise ValueError("Original failed native input changed")
    parent = path.parent.parent.parent
    spec_path = parent / "inputs.json"
    guard_file(spec_path, guards)
    failed_spec = native.read_json(spec_path)
    native.protected_unchanged(failed_spec["input_sha256"])
    merge_guards(guards, failed_spec["input_sha256"])
    if any((parent / "retained_batches").glob("*.json")):
        raise ValueError("Declared failed first group unexpectedly contains accepted review rows")
    return dict(previous_run=str(previous), checkpoint=checkpoint, failed_receipt=str(path),
        failed_receipt_sha256=receipt_sha256, failed_pipeline_sha256=failure_sha256,
        failed_native_receipt=receipt, failed_group=failed_spec["groups"][0],
        failed_chunk=checkpoint["first_failed_chunk"], protected_sha256=guards,
        failed_usage=None, failed_usage_status="unknown_not_zero", automatic_retry=False)


def verify_retained_prefix(previous_run, stopped, *, manifest_sha256=STOP_MANIFEST_SHA256):
    """Verify each whole compact execution once, preserving historical ownership."""
    previous = Path(previous_run).resolve()
    guards = dict(stopped["protected_sha256"])
    guard_file(previous / "manifest.json", guards, manifest_sha256)
    for name in ("progress.jsonl", "source_inputs_sha256.json"):
        guard_file(previous / name, guards)
    manifest = native.read_json(previous / "manifest.json")
    if manifest.get("version") != frozen.VERSION:
        raise ValueError("Stopped review is not the registered frozen wrapper version")
    native.protected_unchanged(manifest["implementation_sha256"])
    merge_guards(guards, manifest["implementation_sha256"])
    inherited = manifest.get("original_reviews_cross_source_reuse", {})
    native.protected_unchanged(inherited.get("sha256", {}))
    merge_guards(guards, inherited.get("sha256", {}))
    progress = protocol.parse_jsonl((previous / "progress.jsonl").read_bytes())
    if any(row.get("type") != "chunk_retained" for row in progress):
        raise ValueError("Stopped review progress contains an unknown source outcome")
    chunks, executions, fresh_executions = {}, {}, set()
    source_ids = manifest["source_order_record_ids"]
    last = stopped["checkpoint"]["last_complete_chunk"]
    paths = sorted((previous / "retained").glob("chunk-*.json"))
    if [path.name for path in paths] != [f"chunk-{number:04d}.json" for number in range(1, last + 1)]:
        raise ValueError("Stopped reviews are not the exact complete fixed-order prefix")
    for number, path in enumerate(paths, 1):
        ids = source_ids[(number - 1) * 20:number * 20]
        bound = [row for row in progress if row.get("number") == number]
        digest = guard_file(path, guards)
        if (len(bound) != 1 or bound[0]["record_ids"] != ids or bound[0]["result_path"] != str(path)
                or bound[0]["result_sha256"] != digest):
            raise ValueError("Original retained result lacks its unique durable progress binding")
        chunk = previous / "chunks" / f"chunk-{number:04d}"
        export_path = chunk / "export_receipt.json"
        guard_file(export_path, guards)
        export = native.read_json(export_path)
        if export["record_ids"] != ids:
            raise ValueError("Original source export lost pre-fixed question membership")
        source_progress = Path(export["source_annotation_run"]) / "progress.jsonl"
        durable_prefix = source_progress.read_bytes()[:export["source_progress_prefix_bytes"]] if source_progress.exists() else b""
        if protocol.sha256(durable_prefix) != export["source_progress_prefix_sha256"]:
            raise ValueError("Original source export durable progress prefix changed")
        native.protected_unchanged(export["source_bindings"])
        merge_guards(guards, export["source_bindings"])
        for name, expected in export["artifacts_sha256"].items():
            guard_file(native.contained_file(chunk, name), guards, expected)
        result = native.read_json(path)
        decisions_path = Path(result["adjudications_path"]).resolve()
        guard_file(decisions_path, guards, result["adjudications_sha256"])
        decisions = protocol.parse_jsonl(decisions_path.read_bytes())
        if ([row["record_id"] for row in decisions] != ids or result["records"] != len(ids)
                or result["accepted"] != sum(row["status"] == "accepted" for row in decisions)
                or result["unresolved"] != sum(row["status"] == "unresolved" for row in decisions)):
            raise ValueError("Original result order, complete coverage or outcome count changed")
        historical = inherited.get("chunks", {}).get(str(number))
        if not decisions_path.is_relative_to(previous / "adjudications"):
            if (not historical or result != historical["result"]
                    or ids != historical["record_ids"] or historical["source_chunk"] == str(chunk)):
                raise ValueError("External historical decisions lack their exact unchanged prior proof")
        packets = native.read_json(chunk / "packets.json")
        left, right = (legacy.full_adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b"))
        grouped = {}
        for row in decisions:
            review = row["adjudication"]
            receipt_path = Path(review["execution_receipt_path"]).resolve()
            raw_path = Path(review["raw_response_path"]).resolve()
            if (not receipt_path.is_relative_to(decisions_path.parent / "executions")
                    or not raw_path.is_relative_to(decisions_path.parent / "executions")
                    or receipt_path.parent != raw_path.parent or raw_path.name != "raw_final.json"):
                raise ValueError("Original decision lost its native raw execution ownership")
            grouped.setdefault(receipt_path, []).append(row)
        for receipt_path, rows in grouped.items():
            if receipt_path not in executions:
                receipt = receipt_artifacts(receipt_path, guards)
                events = native.parse_execution_events((receipt_path.parent / "events.jsonl").read_bytes())
                if (receipt.get("accepted") is not True or events["errors"]
                        or events["thread_id"] != receipt["execution_id"]
                        or ("usage" in receipt and events["usage"] != receipt["usage"])):
                    raise ValueError("Previously accepted native source review no longer has clean retained events")
                executions[receipt_path] = receipt
            receipt = executions[receipt_path]
            raw = native.read_json(receipt_path.parent / "raw_final.json")
            if any(row["adjudication"]["raw_response_sha256"] != receipt["artifacts_sha256"]["raw_final.json"] for row in rows):
                raise ValueError("Original decisions no longer bind whole native raw bytes")
            if historical:
                # The inherited proof already reconstructed these historical
                # full/compact responses under their original driver snapshots.
                continue
            group_ids = [row["record_id"] for row in rows]
            payload, schema = legacy.compact.prepare_batch(packets, left, right, group_ids)
            from oil_evidence.codex_runtime import ISOLATION_PROMPT
            expected_stdin = (ISOLATION_PROMPT + "\n" + legacy.compact.SYSTEM + "\nUNTRUSTED_PAYLOAD_JSON\n").encode()
            expected_stdin += protocol.canonical_bytes(payload)
            if ((receipt_path.parent / "stdin.txt").read_bytes() != expected_stdin
                    or (receipt_path.parent / "schema.json").read_bytes() != protocol.canonical_bytes(schema)):
                raise ValueError("Original compact review physical input or schema changed")
            rebuilt = legacy.compact.retain_compact_decisions({key: raw[key] for key in group_ids},
                packets, left, right, group_ids, receipt)
            if rebuilt != rows:
                raise ValueError("Original compact decisions differ from their exact native raw selections")
            fresh_executions.add(receipt["execution_id"])
        chunks[str(number)] = dict(result_path=str(path), source_chunk=str(chunk), result=result, record_ids=ids)
    records = sum(value["result"]["records"] for value in chunks.values())
    checkpoint = stopped["checkpoint"]
    if (records != checkpoint["reviewed_records"] or len(fresh_executions) != checkpoint["successful_fresh_native_contexts"]
            or len(executions) - len(fresh_executions) != checkpoint["historical_native_contexts_retained"]):
        raise ValueError("Original source-review records or unique native invocation counts changed")
    native.protected_unchanged(guards)
    usage_records = [dict(execution_id=value["execution_id"], receipt=str(path),
        receipt_sha256=guards[str(path)], usage=value.get("usage"), usage_counted_once=True,
        origin="successful_fresh" if value["execution_id"] in fresh_executions else "historical_retained",
        subscription_dollar_cost=value.get("subscription_dollar_cost")) for path, value in executions.items()]
    usage_records.append(dict(execution_id=stopped["failed_native_receipt"]["execution_id"],
        receipt=stopped["failed_receipt"], receipt_sha256=stopped["failed_receipt_sha256"],
        usage=None, usage_status="unknown_not_zero", origin="failed_fresh", usage_counted_once=True,
        subscription_dollar_cost=None))
    if len({row["execution_id"] for row in usage_records}) != len(usage_records):
        raise ValueError("Historical source invocation identities were repeated")
    return dict(output=str(previous), records=records, chunks=chunks, sha256=guards,
        original_decision_files_preserved=True, native_calls_repeated=0,
        successful_fresh_native_contexts=len(fresh_executions), historical_native_contexts=len(executions) - len(fresh_executions),
        failed_fresh_native_contexts=1, failed_usage=None, failed_usage_status="unknown_not_zero",
        execution_usage_records=usage_records)


def producer_module():
    import run_codex_transport_recovery as producer
    if producer.VERSION != PRODUCER_VERSION:
        raise ValueError("Unregistered transport-recovery producer protocol")
    return producer


def prepare(root, pool_dir, annotation_run, chunk_size=20):
    """Validate registered producer history once, then hash-bound caches only."""
    if type(chunk_size) is not int or chunk_size != 20:
        raise ValueError("Recovery preserves the original fixed twenty-case source chunks")
    producer = producer_module()
    root, pool, run = (Path(value).resolve() for value in (root, pool_dir, annotation_run))
    loaded_pool, rows, questions, protected, bindings = frozen.wire.previous.load_pool(root, pool)
    if loaded_pool != pool or not run.is_relative_to(pool / "cohort_annotations_transport_recovery"):
        raise ValueError("Transport annotation source escapes its registered candidate pool")
    manifest = native.read_json(run / "manifest.json")
    spec = manifest["specification"]
    ids = [row["record_id"] for row in rows]
    if (spec["version"] != PRODUCER_VERSION or spec["partition"] != "all_current"
            or spec["bindings"] != bindings or spec["eligible_record_ids"] != ids
            or len(set(ids)) != len(ids) or set(ids) != {row["record_id"] for row in questions}
            or not 1 <= spec["case_limit"] <= 10 or not 1 <= spec["full_payload_byte_limit"] <= 262144
            or spec["historical_labels_unchanged"] is not True
            or spec["canonical_semantic_hash_is_actual_wire_hash"] is not False):
        raise ValueError("Recovery source order, whole-case policy or unchanged historical labels differ")
    for name, digest in manifest["immutable_sha256"].items():
        guard_file(native.contained_file(run, name), protected, digest)
    guard_file(run / "manifest.json", protected)
    source_guards = native.read_json(run / "protected_inputs.json")
    native.protected_unchanged(source_guards)
    frozen.bind_run_sources(dict(protected=protected), source_guards)
    prompt = (run / "protocol.txt").read_bytes()
    if (prompt != frozen.wire.batch.BATCH_PROMPT.encode() or protocol.sha256(prompt) != spec["batch_prompt_sha256"]
            or spec["wire_encoding"] != frozen.wire_codec.VERSION
            or spec["wire_instructions_sha256"] != protocol.sha256(frozen.wire_codec.WIRE_INSTRUCTIONS.encode())):
        raise ValueError("Unchanged recovery semantic and wire protocol bytes differ")
    packets = {row["record_id"]: protocol.clean_packet(row, prompt=prompt.decode()) for row in rows}
    if (native.read_json(run / "packets.json") != packets
            or spec["eligible_packets_sha256"] != protocol.sha256(protocol.canonical_bytes(packets))):
        raise ValueError("Recovery canonical source packets changed")
    history, records = producer.load_history(spec["source_run"], packets)
    if (native.read_json(run / "history.json") != history
            or spec["history_sha256"] != protocol.sha256(protocol.canonical_bytes(history))
            or native.read_json(run / "groups.json") != spec["groups"]):
        raise ValueError("Registered transport migration or sealed groups changed")
    original_groups = history["original_specification"]["groups"]
    role_keys = {"annotation_role", "original_first_attempt", "previous_transport_failure_bound"}
    baseline = [{key: value for key, value in group.items() if key not in role_keys} for group in spec["groups"]]
    if baseline != original_groups[23:] or spec["groups"][0]["batch"] != "batch-0024":
        raise ValueError("Transport continuation must preserve the exact original remaining groups")
    for index, group in enumerate(spec["groups"]):
        if (group.get("annotation_role") != (RETRY_ROLE if index == 0 else FIRST_ROLE)
                or group.get("original_first_attempt") is not (index != 0)
                or group.get("previous_transport_failure_bound") is not (index == 0)):
            raise ValueError("Recovery native group retry or original-first-attempt role is not predeclared")
    questions_by_id = {row["record_id"]: protocol.clean_question(row) for row in questions}
    context = dict(root=root, pool=pool, run=run, rows=rows, packets=packets, ids=ids,
        questions=[questions_by_id[key] for key in ids], chunks=[ids[offset:offset + 20] for offset in range(0, len(ids), 20)],
        prompt=prompt, protected=protected, corpus=root / "data/unified_runs/semantic-page-v2-buffer/stage/passages.jsonl",
        bindings=bindings, history=history, source_spec=spec, source_module=producer, corrected=True, clarified=False)
    frozen.seal_history_cache(context, records, history["sources"])
    return context


def validate_export_roles(context, chunk):
    """Retry pairs remain outside every original-first-attempt population."""
    packets = native.read_json(chunk / "packets.json")
    retry_ids = set(context["source_spec"]["groups"][0]["record_ids"])
    history_ids = set(context["history"]["record_ids"])
    streams = {side: legacy.full_adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b")}
    first_ids, retries = [], []
    for key in packets:
        if key in history_ids:
            for side in streams:
                if streams[side][key] != context["history_cache"]["records"][side][key]:
                    raise ValueError("Recovery changed a previously accepted whole source envelope")
        else:
            role, first = (RETRY_ROLE, False) if key in retry_ids else (FIRST_ROLE, True)
            if any(value[key].get("annotation_role") != role or value[key].get("original_first_attempt") is not first for value in streams.values()):
                raise ValueError("Transport retry or future first-attempt role changed")
        if streams["a"][key].get("annotation_role") == RETRY_ROLE:
            if any(value[key].get("original_first_attempt") is not False for value in streams.values()):
                raise ValueError("Transport retry cannot enter original-first-attempt agreement")
            retries.append(key)
        elif streams["a"][key].get("annotation_role") == FIRST_ROLE:
            first_ids.append(key)
    return dict(retry_record_ids=retries, future_original_first_attempt_record_ids=first_ids,
                original_failures_reclassified=False, human_gold=False)


def bounded_spec(base, chunk):
    packets = native.read_json(chunk / "packets.json")
    ids = list(packets)
    left, right = (base.adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b"))
    splits_path = chunk / "grouping_metadata.json"
    splits = native.read_json(splits_path)
    groups = base.adjudication_groups(packets, left, right, ids, splits=splits)
    protected = {str(path): protocol.sha256(path.read_bytes()) for path in
        (chunk / "packets.json", chunk / "independent_a.jsonl", chunk / "independent_b.jsonl", splits_path,
         Path(base.__file__), Path(base.adjudicator.__file__), Path(protocol.__file__), Path(base.runtime.__file__))}
    spec = dict(version=base.VERSION, groups=groups, max_cases=base.MAX_CASES,
        max_full_input_and_schema_bytes=base.MAX_CALL_BYTES, input_sha256=protected,
        protocol_sha256=protocol.sha256(base.adjudicator.SYSTEM.encode()),
        isolation_sha256=protocol.sha256(base.runtime.ISOLATION_PROMPT.encode()), human_gold=False,
        grouping_policy="Greedy complete cases in source order with private contiguous split boundaries")
    return spec, packets, left, right


def check_retry_identity(receipt, original):
    fields = ("requested_model", "authentication", "native_cli_sha256", "cli_version",
              "disabled_configured_mcp_servers", "model_tools_disabled", "execpolicy_rules_not_disabled", "role")
    if any(receipt.get(key) != original.get(key) for key in fields):
        raise ValueError("Transport retry changed native binary, model request or isolation policy")
    if receipt.get("execution_id") == original.get("execution_id"):
        raise ValueError("Transport retry must use a fresh independent context")
    if ("source_snapshots_sha256" in original
            and receipt.get("source_snapshots_sha256") != original["source_snapshots_sha256"]):
        raise ValueError("Transport retry changed guarded native runtime implementation bytes")


def one_retry(base, chunk, output_parent, stopped, authorization, *, caller=None):
    """Exactly one authorized first-group retry; keep full-chunk resume contract."""
    authorization = Path(authorization).resolve()
    auth = native.read_json(authorization)
    if (auth.get("version") != "source-review-transport-retry-authorization-v1"
            or auth.get("failed_receipt_sha256") != stopped["failed_receipt_sha256"]
            or auth.get("original_failure_sha256") != stopped["failed_pipeline_sha256"]
            or auth.get("max_retry_native_calls") != 1 or auth.get("same_physical_input") is not True
            or auth.get("same_model_request") is not True or auth.get("new_failures_stop") is not True):
        raise ValueError("One controlled exact-input source retry requires explicit bound root authorization")
    if Path(auth.get("recovery_output_parent", "")).resolve() != Path(output_parent).resolve():
        raise ValueError("Once-only source retry authorization is bound to a different recovery output")
    spec, packets, left, right = bounded_spec(base, chunk)
    group = spec["groups"][0]
    if group != stopped["failed_group"]:
        raise ValueError("Retry first group differs from the originally failed fixed source membership")
    payload, schema = base.adjudicator.prepare_batch(packets, left, right, group)
    stdin, schema_bytes = base.actual_adjudication_bytes(payload, schema)
    original_dir = Path(stopped["failed_receipt"]).parent
    if stdin != (original_dir / "stdin.txt").read_bytes() or schema_bytes != (original_dir / "schema.json").read_bytes():
        raise ValueError("Retry must preserve every physical source input and schema byte")
    output = Path(output_parent) / "bounded-run"
    if (output / "retry_request.json").exists():
        raise ValueError("The once-only source transport retry was already planned; do not invoke again")
    native.protected_unchanged(stopped["protected_sha256"])
    native.protected_unchanged(spec["input_sha256"])
    if (output / "inputs.json").exists():
        if native.read_json(output / "inputs.json") != spec:
            raise ValueError("Recovery full-chunk bounded protocol changed")
    else:
        native.write_new(output / "inputs.json", spec)
    request = dict(version=VERSION, original_failed_receipt=stopped["failed_receipt"],
        original_failed_receipt_sha256=stopped["failed_receipt_sha256"],
        original_failure_sha256=stopped["failed_pipeline_sha256"], record_ids=group,
        authorization=str(authorization), authorization_sha256=protocol.sha256(authorization.read_bytes()),
        stdin_sha256=protocol.sha256(stdin), schema_sha256=protocol.sha256(schema_bytes),
        max_new_native_calls=1, original_failed_usage=None, original_failed_usage_status="unknown_not_zero")
    native.write_new(output / "retry_request.json", request)
    execute = caller or base.runtime.call_structured_json
    try:
        raw, receipt = execute(base.adjudicator.SYSTEM, payload, schema, output / "executions", role="adjudicator")
        directory = Path(receipt["artifact_directory"]).resolve()
        if not directory.is_relative_to((output / "executions").resolve()):
            raise ValueError("Retry native response escapes its own new execution directory")
        guards = {}
        actual = receipt_artifacts(directory / "receipt.json", guards)
        if actual != receipt or receipt.get("accepted") is not True:
            raise ValueError("The single transport retry was not operationally accepted")
        if (directory / "stdin.txt").read_bytes() != stdin or (directory / "schema.json").read_bytes() != schema_bytes:
            raise ValueError("Actual retry changed the original full source input or schema")
        check_retry_identity(receipt, stopped["failed_native_receipt"])
        events = native.parse_execution_events((directory / "events.jsonl").read_bytes())
        if events["errors"] or events["thread_id"] != receipt["execution_id"] or events["observed_model"] != receipt.get("observed_model"):
            raise ValueError("Retry native event metadata is not clean and independently identifiable")
        if protocol.canonical_bytes(raw) != protocol.canonical_bytes(native.read_json(directory / "raw_final.json")):
            raise ValueError("Retry caller result differs from its whole retained actual raw response")
        decisions = base.adjudicator.retain_decisions(raw, packets, left, right, group, receipt)
        native.protected_unchanged(stopped["protected_sha256"])
        native.protected_unchanged(spec["input_sha256"])
        if protocol.sha256(authorization.read_bytes()) != request["authorization_sha256"]:
            raise ValueError("Reviewed source retry authorization changed during execution")
        binding = dict(record_ids=group, decisions=decisions,
            receipt_path=(directory / "receipt.json").relative_to(output).as_posix(),
            receipt_sha256=protocol.sha256((directory / "receipt.json").read_bytes()),
            actual_full_input_and_schema_bytes=len(stdin) + len(schema_bytes), human_gold=False)
        native.write_new(output / "retained_batches/batch-0001.json", binding)
        result = dict(version=VERSION, validation_status="one_exact_input_retry_retained_pending_full_chunk",
            original_failure_retained=True, new_native_calls=1, records=len(group),
            accepted=sum(row["status"] == "accepted" for row in decisions),
            unresolved=sum(row["status"] == "unresolved" for row in decisions),
            execution_id=receipt["execution_id"], usage=receipt.get("usage"),
            receipt_path=str(directory / "receipt.json"), receipt_sha256=binding["receipt_sha256"],
            record_ids=group, retry_request_sha256=protocol.sha256((output / "retry_request.json").read_bytes()),
            first_batch_binding_sha256=protocol.sha256((output / "retained_batches/batch-0001.json").read_bytes()),
            requested_model=receipt.get("requested_model"), observed_model=receipt.get("observed_model"),
            exact_full_stdin_equal=True, exact_schema_equal=True, complete_chunk=False, human_gold=False)
        native.write_new(output / "retry_gate.json", result)
        return result
    except Exception as error:
        native.write_new(output / "retry_failure.json", dict(version=VERSION, error=str(error),
            automatic_retry=False, original_failure_retained=True, new_native_calls_at_most=1,
            all_attempts_preserved=True, human_gold=False))
        raise


def verify_retry_gate(base, chunk, output_parent, stopped):
    """Rebuild the entire one-call gate before any ordinary continuation call."""
    output = Path(output_parent) / "bounded-run"
    if (output / "retry_failure.json").exists():
        raise ValueError("A new recovery failure requires a new explicit owner decision")
    guards = {}
    gate_path, request_path = output / "retry_gate.json", output / "retry_request.json"
    binding_path = output / "retained_batches/batch-0001.json"
    for path in (gate_path, request_path, binding_path, output / "inputs.json"):
        guard_file(path, guards)
    gate, request, binding = map(native.read_json, (gate_path, request_path, binding_path))
    spec, packets, left, right = bounded_spec(base, chunk)
    group = spec["groups"][0]
    if native.read_json(output / "inputs.json") != spec or group != stopped["failed_group"]:
        raise ValueError("Retained recovery gate changed its complete source-chunk protocol or first group")
    if (gate.get("version") != VERSION or gate.get("new_native_calls") != 1
            or gate.get("validation_status") != "one_exact_input_retry_retained_pending_full_chunk"
            or gate.get("original_failure_retained") is not True or gate.get("complete_chunk") is not False
            or gate.get("record_ids") != group or binding.get("record_ids") != group
            or gate.get("retry_request_sha256") != guards[str(request_path.resolve())]
            or gate.get("first_batch_binding_sha256") != guards[str(binding_path.resolve())]
            or request.get("original_failed_receipt") != stopped["failed_receipt"]
            or request.get("original_failed_receipt_sha256") != stopped["failed_receipt_sha256"]
            or request.get("original_failure_sha256") != stopped["failed_pipeline_sha256"]
            or request.get("record_ids") != group or request.get("max_new_native_calls") != 1):
        raise ValueError("One-shot recovery gate lost its exact original failure, request or retained batch binding")
    authorization = Path(request["authorization"]).resolve()
    guard_file(authorization, guards, request["authorization_sha256"])
    auth = native.read_json(authorization)
    if (auth.get("version") != "source-review-transport-retry-authorization-v1"
            or auth.get("failed_receipt_sha256") != stopped["failed_receipt_sha256"]
            or auth.get("original_failure_sha256") != stopped["failed_pipeline_sha256"]
            or auth.get("max_retry_native_calls") != 1 or auth.get("same_physical_input") is not True
            or auth.get("same_model_request") is not True or auth.get("new_failures_stop") is not True):
        raise ValueError("Retained one-shot retry no longer has its original bound reviewed authorization")
    if Path(auth.get("recovery_output_parent", "")).resolve() != Path(output_parent).resolve():
        raise ValueError("Retained one-shot authorization was rebound to another output")
    path = native.contained_file(output, binding["receipt_path"])
    guard_file(path, guards, binding["receipt_sha256"])
    if str(path) != gate["receipt_path"] or binding["receipt_sha256"] != gate["receipt_sha256"]:
        raise ValueError("Gate native receipt ownership changed")
    receipt = receipt_artifacts(path, guards)
    if receipt.get("accepted") is not True:
        raise ValueError("Retained one-shot retry was not natively accepted")
    check_retry_identity(receipt, stopped["failed_native_receipt"])
    payload, schema = base.adjudicator.prepare_batch(packets, left, right, group)
    stdin, schema_bytes = base.actual_adjudication_bytes(payload, schema)
    original = Path(stopped["failed_receipt"]).parent
    if (stdin != (original / "stdin.txt").read_bytes() or schema_bytes != (original / "schema.json").read_bytes()
            or stdin != (path.parent / "stdin.txt").read_bytes()
            or schema_bytes != (path.parent / "schema.json").read_bytes()
            or request["stdin_sha256"] != protocol.sha256(stdin)
            or request["schema_sha256"] != protocol.sha256(schema_bytes)):
        raise ValueError("Retained source recovery no longer has identical actual input and schema bytes")
    events = native.parse_execution_events((path.parent / "events.jsonl").read_bytes())
    if (events["errors"] or events["thread_id"] != receipt["execution_id"]
            or events["observed_model"] != receipt.get("observed_model")
            or gate["execution_id"] != receipt["execution_id"] or gate["usage"] != receipt.get("usage")):
        raise ValueError("Retained source retry event identity or usage changed")
    raw = native.read_json(path.parent / "raw_final.json")
    decisions = base.adjudicator.retain_decisions(raw, packets, left, right, group, receipt)
    if (binding["decisions"] != decisions or gate["records"] != len(group)
            or gate["accepted"] != sum(row["status"] == "accepted" for row in decisions)
            or gate["unresolved"] != sum(row["status"] == "unresolved" for row in decisions)):
        raise ValueError("Retained retry decisions differ from whole actual compact native raw output")
    bound = {native.read_json(item)["receipt_path"] for item in (output / "retained_batches").glob("*.json")}
    actual = {item.relative_to(output).as_posix() for item in (output / "executions").glob("*/receipt.json")}
    if bound != actual:
        raise ValueError("Unbound or failed additional source context cannot pass the one-shot recovery gate")
    for name in actual:
        if native.read_json(native.contained_file(output, name)).get("accepted") is not True:
            raise ValueError("A new source native failure stops continuation without another retry")
    native.protected_unchanged(stopped["protected_sha256"])
    native.protected_unchanged(guards)
    return gate, guards


def configure(base, reused, stopped):
    """Install only private callbacks; all frozen helper source bytes stay intact."""
    frozen.configure(base, dict(output=None, records=0, chunks={}, sha256={}))
    base.VERSION = VERSION
    holder = {}
    original_specification, original_create = base.specification, base.create_pipeline
    original_verify, original_adjudicate = base.verify_retained, base.adjudicate_chunk

    def prepared(root, pool, run, chunk_size=20):
        context = prepare(root, pool, run, chunk_size)
        merge_guards(context["protected"], reused["sha256"])
        merge_guards(context["protected"], stopped["protected_sha256"])
        for value in reused["chunks"].values():
            source = Path(value["source_chunk"])
            if native.read_json(source / "packets.json") != {key: context["packets"][key] for key in value["record_ids"]}:
                raise ValueError("Retained review packets differ from unchanged recovery source")
            for side in ("a", "b"):
                old = legacy.full_adjudicator.read_stream(source / f"independent_{side}.jsonl")
                if old != {key: context["history_cache"]["records"][side][key] for key in value["record_ids"]}:
                    raise ValueError("Retained reviews cannot be rebound to different whole original A/B envelopes")
        holder["context"] = context
        return context

    def specification(context, chunk_size):
        result = original_specification(context, chunk_size)
        result.update(version=VERSION, source_transport_protocol=PRODUCER_VERSION,
            original_source_review_reuse=reused, original_source_review_stop=stopped,
            retry_annotation_role=RETRY_ROLE, retry_records_excluded_from_first_attempt_agreement=True,
            future_annotation_role=FIRST_ROLE, all_original_failure_bytes_preserved=True)
        result["implementation_sha256"].update({str(Path(path).resolve()): protocol.sha256(Path(path).read_bytes())
            for path in (__file__, context["source_module"].__file__)})
        return result

    def create(context, chunk_size=20, resume=None):
        output = original_create(context, chunk_size, resume)
        if resume is None:
            for key, entry in reused["chunks"].items():
                number = int(key)
                result_path = output / "retained" / f"chunk-{number:04d}.json"
                native.write_new(result_path, Path(entry["result_path"]).read_bytes())
                native.write_new(output / "chunks" / f"chunk-{number:04d}" / "reused_export_receipt.json",
                    dict(version=VERSION, original_source_chunk=entry["source_chunk"], record_ids=entry["record_ids"],
                         original_export_sha256=protocol.sha256((Path(entry["source_chunk"]) / "export_receipt.json").read_bytes()),
                         original_decision_path=entry["result"]["adjudications_path"], native_calls_repeated=0))
                native.append_progress(output / "progress.jsonl", dict(type="chunk_retained", number=number,
                    record_ids=entry["record_ids"], result_path=str(result_path),
                    result_sha256=protocol.sha256(result_path.read_bytes()),
                    original_review_run=reused["output"], original_decision_path=entry["result"]["adjudications_path"],
                    native_calls_repeated=0))
        return output

    def export(context, output, number, prefix, entries):
        key = str(number)
        if key in reused["chunks"]:
            descriptor = output / "chunks" / f"chunk-{number:04d}" / "reused_export_receipt.json"
            expected = reused["chunks"][key]
            value = native.read_json(descriptor)
            if (value["original_source_chunk"] != expected["source_chunk"] or value["record_ids"] != expected["record_ids"]
                    or value["original_decision_path"] != expected["result"]["adjudications_path"]
                    or value["original_export_sha256"] != protocol.sha256((Path(expected["source_chunk"]) / "export_receipt.json").read_bytes())):
                raise ValueError("Explicit historical source export ownership changed")
            return Path(expected["source_chunk"])
        selected = set(context["chunks"][number - 1])
        for _, event in entries:
            if (event.get("type") == "completed" and event.get("accepted") is True
                    and selected.intersection(event["record_ids"])):
                source = native.contained_file(context["run"], event["receipt"])
                receipt = native.read_json(source)
                if (Path(receipt.get("annotation_run", "")).resolve() != context["run"]
                        or receipt.get("execution_protocol") != PRODUCER_VERSION):
                    raise ValueError("New durable native receipt is not owned by the exact registered recovery producer")
        chunk = frozen.snapshot_chunk(context, output, number, prefix, entries)
        if chunk is not None:
            population = validate_export_roles(context, chunk)
            path = chunk / "transport_recovery_population.json"
            if path.exists():
                if native.read_json(path) != population:
                    raise ValueError("Recovery agreement population or original-first-attempt role changed")
            else:
                native.write_new(path, population)
            frozen.bind_run_sources(context, {str(path): protocol.sha256(path.read_bytes())})
        return chunk

    def verify(output, chunk, ids, result_path):
        key = str(int(result_path.stem.split("-")[-1]))
        if key not in reused["chunks"]:
            return original_verify(output, chunk, ids, result_path)
        entry = reused["chunks"][key]
        if ids != entry["record_ids"] or result_path.read_bytes() != Path(entry["result_path"]).read_bytes():
            raise ValueError("Reused original review result bytes or membership changed")
        bound = [row for row in protocol.parse_jsonl((output / "progress.jsonl").read_bytes())
                 if row.get("type") == "chunk_retained" and row.get("number") == int(key)]
        if (len(bound) != 1 or bound[0].get("original_review_run") != reused["output"]
                or bound[0]["result_sha256"] != protocol.sha256(result_path.read_bytes())
                or bound[0].get("original_decision_path") != entry["result"]["adjudications_path"]):
            raise ValueError("Reused review lost its explicit original native ownership")
        return entry["result"]

    def adjudicate(chunk, directory, **kwargs):
        context = holder["context"]
        frozen.verify_history_cache(context)
        native.protected_unchanged(context["protected"])
        if "retry_gate_guards" in holder:
            native.protected_unchanged(holder["retry_gate_guards"])
        result = original_adjudicate(chunk, directory, **kwargs)
        frozen.verify_history_cache(context)
        native.protected_unchanged(context["protected"])
        if "retry_gate_guards" in holder:
            native.protected_unchanged(holder["retry_gate_guards"])
        return result

    base.prepare, base.specification, base.create_pipeline = prepared, specification, create
    base.export_chunk, base.verify_retained, base.adjudicate_chunk = export, verify, adjudicate
    return base, holder


def run(root, pool_dir, annotation_run, *, previous_review_run, resume=None,
        max_idle_seconds=21600, poll_seconds=5, export_only=False, retry_authorization=None, caller=None):
    stopped = verify_stopped_review(previous_review_run)
    reused = verify_retained_prefix(previous_review_run, stopped)
    base, holder = configure(legacy.isolated_base(), reused, stopped)
    if retry_authorization:
        context = base.prepare(root, pool_dir, annotation_run)
        output = base.create_pipeline(context, 20, resume)
        prefix, entries = base.progress_snapshot(context["run"])
        number = stopped["failed_chunk"]
        chunk = base.export_chunk(context, output, number, prefix, entries)
        if chunk is None:
            raise ValueError("Exact failed source chunk does not yet have durable unchanged paired inputs")
        result = one_retry(base, chunk, output / "adjudications" / f"chunk-{number:04d}",
            stopped, retry_authorization, caller=caller)
        native.protected_unchanged(context["protected"])
        return output, result
    if not export_only:
        if resume is None:
            raise ValueError("Source continuation requires the separately reviewed one-call recovery gate")
        chunk = Path(resume) / "chunks" / f"chunk-{stopped['failed_chunk']:04d}"
        _, gate_guards = verify_retry_gate(base, chunk,
            Path(resume) / "adjudications" / f"chunk-{stopped['failed_chunk']:04d}", stopped)
        holder["retry_gate_guards"] = gate_guards
    output, result = base.run(root, pool_dir, annotation_run, resume=resume, max_idle_seconds=max_idle_seconds,
        poll_seconds=poll_seconds, adjudicate=not export_only, caller=caller)
    result = dict(result, reused_exported_chunks=len(reused["chunks"]),
        original_reviewed_records=reused["records"], original_successful_native_calls=reused["successful_fresh_native_contexts"],
        original_failed_native_calls=1, original_failed_usage=None, original_failed_usage_status="unknown_not_zero")
    native.write_new(output / ("transport-summary-" + uuid4().hex[:8] + ".json"), result)
    return output, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-root", "pool-dir", "annotation-run", "previous-review-run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--retry-authorization", type=Path)
    parser.add_argument("--max-idle-seconds", type=int, default=21600)
    args = parser.parse_args()
    _, summary = run(args.canonical_root, args.pool_dir, args.annotation_run,
        previous_review_run=args.previous_review_run, resume=args.resume, export_only=args.export_only,
        retry_authorization=args.retry_authorization, max_idle_seconds=args.max_idle_seconds)
    print(json.dumps(summary, indent=2))
