"""Durable wire-source exports with explicit, unchanged historical review reuse.

Fixed twenty-case membership follows the annotation manifest, not label outcomes.
Whole native groups and alias sidecars are retained before selecting per-record
envelopes. Historical decision files stay beside their original execution folders;
they are never rewritten into a cross-run merged decision file.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pipeline_compact_cohort_adjudication as legacy
import run_codex_cohort_wire as wire
from oil_evidence import wire_codec

native, protocol = legacy.native, legacy.protocol
VERSION = "wire-source-compact-adjudication-clarified-cache-v4"
CLARIFIED_VERSION = "codex-format-clarified-first-attempt-cohort-v1"
CACHE_VERSION = "once-verified-immutable-source-history-v1"
COMPOSITE_VERSION = "codex-top5-candidate-cohort-corrective-composite-v1"
CORRECTION_VERSION = "source-only-every-citation-corrective-pair-v1"
METADATA_VALIDATION_VERSION = "metadata-revalidated-source-only-corrective-pair-v1"
AUTHORIZATION_COPY_ERROR = "Reviewed corrective authorization changed during execution"
AUTHORIZED_FAILURE_SHA256 = "4fa3df9e6cdf271b9fc764f462a23c3a17e346d3b472cf20eb09f0e466c8a2ee"
AUTHORIZED_SOURCE_ERROR = "Decoded canonical label invalid: Every answer point needs citations with exact eligible quotes"


def seal_history_cache(context, records, sources):
    """Bind whole already-verified envelopes, lineage and every original guard."""
    ids = context["history"]["record_ids"]
    records = {side: deepcopy(records.get(side, {})) for side in ("a", "b")}
    sources = {side: deepcopy(sources.get(side, {})) for side in ("a", "b")}
    if any(set(records[side]) != set(ids) or set(sources[side]) != set(ids) for side in records):
        raise ValueError("Historical cache needs complete already-verified paired envelopes and lineage")
    protected = dict(context["history"]["protected_sha256"])
    history_path = context["run"] / "history.json"
    protected[str(history_path)] = protocol.sha256(history_path.read_bytes())
    for side in sources:
        for value in sources[side].values():
            path = Path(value["receipt"]).resolve()
            receipt = native.read_json(path)
            if protocol.sha256(path.read_bytes()) != value["receipt_sha256"]:
                raise ValueError("Historical cache source receipt changed")
            protected[str(path)] = value["receipt_sha256"]
            for name, digest in receipt["artifacts_sha256"].items():
                artifact = native.contained_file(path.parent, name)
                if str(artifact) in protected and protected[str(artifact)] != digest:
                    raise ValueError("Historical cache has conflicting raw source guards")
                protected[str(artifact)] = digest
    native.protected_unchanged(protected)
    cache = dict(version=CACHE_VERSION, history=deepcopy(context["history"]), record_ids=list(ids),
        records=records, sources=sources, protected_sha256=protected,
        original_envelopes_unchanged=True, source_validation_completed_before_cache=True)
    context["history_cache"] = cache
    context["history_cache_sha256"] = protocol.sha256(protocol.canonical_bytes(cache))
    context["protected"].update(protected)
    return cache


def verify_history_cache(context):
    cache = context["history_cache"]
    if (protocol.sha256(protocol.canonical_bytes(cache)) != context["history_cache_sha256"]
            or cache["history"] != context["history"]
            or cache["record_ids"] != context["history"]["record_ids"]):
        raise ValueError("Historical cache bytes, labels or source-history lineage changed")
    native.protected_unchanged(cache["protected_sha256"])
    path = context.get("history_cache_artifact")
    if path and protocol.sha256(Path(path).read_bytes()) != context["history_cache_sha256"]:
        raise ValueError("Immutable historical cache artifact changed")
    if context.get("history_cache_receipt_guard"):
        native.protected_unchanged(context["history_cache_receipt_guard"])
    return cache


def verified_native_group(context, copied_attempt, source_receipt, receipt):
    """A group spanning fixed chunks is decoded once, with raw guards each use."""
    groups = context.setdefault("verified_native_groups", {})
    key = str(source_receipt)
    raw_digest = protocol.sha256(source_receipt.read_bytes())
    if key in groups:
        saved = groups[key]
        if (saved["receipt_sha256"] != raw_digest
                or protocol.sha256(protocol.canonical_bytes(saved["records"])) != saved["records_sha256"]):
            raise ValueError("Previously verified whole native group cache changed")
        native.protected_unchanged(saved["protected_sha256"])
        bind_run_sources(context, saved["protected_sha256"])
        return deepcopy(saved["records"])
    values = context.get("source_module", wire).verify_attempt(copied_attempt, receipt, context["packets"])
    protected = {str(source_receipt): raw_digest}
    protected.update({str(native.contained_file(source_receipt.parent, name)): digest
                      for name, digest in receipt["artifacts_sha256"].items()})
    native.protected_unchanged(protected)
    bind_run_sources(context, protected)
    groups[key] = dict(records=deepcopy(values), records_sha256=protocol.sha256(protocol.canonical_bytes(values)),
        receipt_sha256=raw_digest, protected_sha256=protected)
    return deepcopy(values)


def bind_run_sources(context, protected):
    """Every used original raw source stays guarded through final completion."""
    guards = context.setdefault("protected", {})
    if any(name in guards and guards[name] != digest for name, digest in protected.items()):
        raise ValueError("Used source raw artifact has conflicting run-wide guards")
    guards.update(protected)


def bound_source_receipt(run, path, digest):
    path = Path(path)
    path = (path if path.is_absolute() else Path(run) / path).resolve()
    if not path.is_relative_to(Path(run).resolve()) or not path.is_file() or path.name != "receipt.json":
        raise ValueError("Resolution receipt escapes its original annotation run")
    if protocol.sha256(path.read_bytes()) != digest:
        raise ValueError("Resolution source receipt bytes changed")
    value = native.read_json(path)
    bindings = {str(path): digest}
    for name, expected in value["artifacts_sha256"].items():
        artifact = native.contained_file(path.parent, name)
        if protocol.sha256(artifact.read_bytes()) != expected:
            raise ValueError("Resolution original/retry raw artifact changed")
        bindings[str(artifact)] = expected
    return path, value, bindings


def command_identity(command):
    """Only ephemeral context/schema/output destinations may differ."""
    ephemeral = {"--cd", "--output-schema", "--output-last-message"}
    result, index = [], 0
    while index < len(command):
        value = command[index]
        result.append(value)
        if value in ephemeral:
            if index + 1 >= len(command):
                raise ValueError("Incomplete native command metadata")
            result.append("EPHEMERAL_CONTEXT_DESTINATION")
            index += 2
        else:
            index += 1
    return result


def validate_resolution(context, resolution_path):
    """Admit ONLY the one explicitly authorized rejected B0019 replacement."""
    run = context["run"]
    resolution_path = Path(resolution_path).resolve()
    if not resolution_path.is_relative_to(run) or resolution_path.name != "resolution.json":
        raise ValueError("Resolution must remain inside its original annotation run")
    value = native.read_json(resolution_path)
    directory = resolution_path.parent
    authorization_path, request_path = directory / "authorization.json", directory / "request.json"
    authorization = native.read_json(authorization_path)
    if (value["annotation_run"] != str(run) or value["stream"] != "b" or value["batch"] != "batch-0019"
            or value["original_failure_receipt_sha256"] != AUTHORIZED_FAILURE_SHA256
            or value["original_failure_retained"] is not True or value["retry_accepted"] is not True
            or value["retries_authorized"] != 1 or type(value["retries_authorized"]) is not int
            or value["native_calls_actual"] != 1 or value["no_label_feedback"] is not True
            or authorization.get("root_authorized") is not True
            or protocol.sha256(authorization_path.read_bytes()) != value["authorization_sha256"]
            or protocol.sha256(request_path.read_bytes()) != value["request_sha256"]):
        raise ValueError("No explicit authorization for this exact source-validation resolution")
    paths, receipts, protected = {}, {}, {str(resolution_path): protocol.sha256(resolution_path.read_bytes()),
        str(authorization_path): value["authorization_sha256"], str(request_path): value["request_sha256"]}
    for name in ("original_failure", "counterpart_a", "retry"):
        paths[name], receipts[name], bindings = bound_source_receipt(run, value[name + "_receipt"], value[name + "_receipt_sha256"])
        protected.update(bindings)
    old, other, new = (receipts[name] for name in ("original_failure", "counterpart_a", "retry"))
    ids = value["record_ids"]
    group = next((row for row in context["source_spec"]["groups"] if row["batch"] == "batch-0019"), None)
    if (not group or ids != group["record_ids"] or old["record_ids"] != ids or other["record_ids"] != ids or new["record_ids"] != ids
            or old.get("accepted") is not False or old.get("process_exit_code") != 0
            or old.get("errors") != [AUTHORIZED_SOURCE_ERROR]
            or any(row.get("accepted") is not True or row.get("process_exit_code") != 0 or row.get("errors") for row in (other, new))
            or old["stream"] != "b" or new["stream"] != "b" or other["stream"] != "a"
            or any(row["batch"] != "batch-0019" for row in (old, other, new))
            or len({row["execution_id"] for row in (old, other, new)}) != 3
            or value["original_execution_id"] != old["execution_id"] or value["retry_execution_id"] != new["execution_id"]):
        raise ValueError("Resolution is not one rejected non-quota whole batch with a fresh valid counterpart")
    physical = ("stdin.txt", "schema.json", "codec.json", "wire_payload.json", "canonical_semantic_packets.json")
    for name in physical:
        blobs = [paths[key].parent.joinpath(name).read_bytes() for key in paths]
        if len(set(blobs)) != 1 or value["physical_equal_artifact_hashes"].get(name) != protocol.sha256(blobs[0]):
            raise ValueError("Resolution changed the actual complete input/schema/codec bytes")
    commands = [command_identity(native.read_json(paths[key].parent / "command.json")) for key in paths]
    if any(command != commands[0] for command in commands[1:]) or value["same_requested_model"] is not True:
        raise ValueError("Resolution changed the native identity/model request or isolation settings")
    for key, row in receipts.items():
        observed = native.parse_execution_events((paths[key].parent / "events.jsonl").read_bytes())
        if observed["errors"] or observed["thread_id"] != row["execution_id"]:
            raise ValueError("Quota/native errors cannot be resolved by this source-validation authorization")
    for key in ("counterpart_a", "retry"):
        wire.verify_attempt(paths[key].parent, receipts[key], context["packets"])
    prefix_info = value["original_progress_prefix"]
    prefix_path = native.contained_file(directory, prefix_info["path"])
    prefix = prefix_path.read_bytes()
    if len(prefix) != prefix_info["byte_count"] or protocol.sha256(prefix) != prefix_info["sha256"] or not prefix.endswith(b"\n"):
        raise ValueError("Original failed progress prefix changed")
    protected[str(prefix_path)] = prefix_info["sha256"]
    rows = protocol.parse_jsonl(prefix)
    failed = [row for row in rows if row.get("type") == "failed" or (row.get("type") == "completed" and row.get("accepted") is False)]
    if (len(failed) != 1 or failed[0].get("receipt_sha256") != AUTHORIZED_FAILURE_SHA256
            or native.contained_file(run, failed[0]["receipt"]) != paths["original_failure"]):
        raise ValueError("Resolution cannot suppress an unknown, additional or quota failure")
    # Preserve every previously accepted receipt; none may be rerun or replaced.
    expected_other = {}
    for row in rows:
        if row.get("type") != "completed" or row.get("accepted") is not True:
            continue
        receipt_path, _, bindings = bound_source_receipt(run, row["receipt"], row["receipt_sha256"])
        expected_other[str(receipt_path)] = row["receipt_sha256"]
        protected.update(bindings)
    if value["unchanged_A_and_other_accepted_receipts"] != expected_other:
        raise ValueError("Resolution changed previously accepted A or other original calls")
    return dict(path=str(resolution_path), sha256=protocol.sha256(resolution_path.read_bytes()),
        protected=protected, original_progress_prefix=prefix,
        failed_receipt=str(paths["original_failure"]), failed_receipt_sha256=AUTHORIZED_FAILURE_SHA256,
        retry_receipt=str(paths["retry"]), retry_receipt_sha256=value["retry_receipt_sha256"],
        stream="b", batch="batch-0019", record_ids=ids, failure_reclassified_as_accepted=False)


def resolved_progress_snapshot(run, resolution=None):
    if resolution is None:
        return legacy.isolated_base().progress_snapshot(run)
    native.protected_unchanged(resolution["protected"])
    raw = (Path(run) / "progress.jsonl").read_bytes()
    prefix = raw[:raw.rfind(b"\n") + 1]
    if not prefix.startswith(resolution["original_progress_prefix"]):
        raise ValueError("Original failed/accepted progress prefix was rewritten")
    rows = [(line, json.loads(line, object_pairs_hook=protocol._unique_object, parse_constant=protocol._invalid_constant))
        for line in prefix.splitlines(keepends=True) if line.strip()]
    projected, old_failure, accepted_retry, successes = [], 0, 0, set()
    for line, row in rows:
        failed = row.get("type") == "failed" or (row.get("type") == "completed" and row.get("accepted") is False)
        if failed:
            if (row.get("receipt_sha256") != resolution["failed_receipt_sha256"]
                    or str(native.contained_file(Path(run), row["receipt"])) != resolution["failed_receipt"]):
                raise ValueError("Unresolved/new/native/quota source failure stops without retry")
            old_failure += 1
            continue  # Original raw prefix is retained; only the verified replacement is eligible.
        if row.get("type") == "completed" and row.get("accepted") is True:
            key = (row["stream"], row["batch"])
            if key in successes:
                raise ValueError("An already accepted source batch was rerun during resolution")
            successes.add(key)
            if key == (resolution["stream"], resolution["batch"]):
                if (row["receipt_sha256"] != resolution["retry_receipt_sha256"]
                        or str(native.contained_file(Path(run), row["receipt"])) != resolution["retry_receipt"]
                        or row["record_ids"] != resolution["record_ids"]):
                    raise ValueError("Durable retry differs from its exact authorized resolution")
                accepted_retry += 1
        projected.append((line, row))
    if old_failure != 1 or accepted_retry != 1:
        raise ValueError("Resolution lacks one original failure and one durable valid replacement")
    return prefix, projected


def prepare(root, pool_dir, annotation_run, chunk_size=20):
    if chunk_size != 20 or type(chunk_size) is not int:
        raise ValueError("Use the pre-fixed twenty-case source chunk policy")
    root, pool_dir, annotation_run = (Path(value).resolve() for value in (root, pool_dir, annotation_run))
    pool, rows, questions, protected, bindings = wire.previous.load_pool(root, pool_dir)
    is_clarified = annotation_run.is_relative_to(pool / "cohort_annotations_clarified")
    is_corrected = annotation_run.is_relative_to(pool / "cohort_annotations_corrected")
    if not (is_clarified or is_corrected or annotation_run.is_relative_to(pool / "cohort_annotations_wire")):
        raise ValueError("Wire annotation source escapes its candidate pool")
    if is_clarified:
        import run_codex_clarified_cohort as source_module
        if source_module.VERSION != CLARIFIED_VERSION:
            raise ValueError("Unknown predeclared clarified first-attempt annotation protocol")
    elif is_corrected:
        import run_codex_corrected_cohort as source_module
        if source_module.VERSION != COMPOSITE_VERSION:
            raise ValueError("Unknown corrective-composite annotation protocol")
    else:
        source_module = wire
    manifest = native.read_json(annotation_run / "manifest.json")
    spec = manifest["specification"]
    ids = [row["record_id"] for row in rows]
    if (spec["version"] != source_module.VERSION or spec["partition"] != "all_current"
            or spec["bindings"] != bindings or spec["eligible_record_ids"] != ids
            or len(set(ids)) != len(ids) or set(ids) != {row["record_id"] for row in questions}
            or not 1 <= spec["case_limit"] <= 10 or not 1 <= spec["full_payload_byte_limit"] <= 262144
            or spec["historical_labels_unchanged"] is not True
            or spec["canonical_semantic_hash_is_actual_wire_hash"] is not False):
        raise ValueError("Wire source order, coverage or complete-case policy changed")
    for name, digest in manifest["immutable_sha256"].items():
        path = native.contained_file(annotation_run, name)
        if protocol.sha256(path.read_bytes()) != digest:
            raise ValueError("Immutable wire annotation artifact changed")
        protected[str(path)] = digest
    protected[str(annotation_run / "manifest.json")] = protocol.sha256((annotation_run / "manifest.json").read_bytes())
    source_guard = native.read_json(annotation_run / "protected_inputs.json")
    native.protected_unchanged(source_guard)
    for path, digest in source_guard.items():
        if path in protected and protected[path] != digest:
            raise ValueError("Conflicting canonical and wire source guards")
        protected[path] = digest
    prompt = (annotation_run / "protocol.txt").read_bytes()
    if (prompt != wire.batch.BATCH_PROMPT.encode()
            or protocol.sha256(prompt) != spec["batch_prompt_sha256"]
            or spec["wire_encoding"] != wire_codec.VERSION
            or spec["wire_instructions_sha256"] != protocol.sha256(wire_codec.WIRE_INSTRUCTIONS.encode())):
        raise ValueError("Actual semantic and wire protocols differ from their sealed bytes")
    if is_clarified:
        instruction = source_module.clarification.CORRECTION_INSTRUCTION.encode()
        if ((annotation_run / "format_instruction.txt").read_bytes() != instruction
                or protocol.sha256(instruction) != source_module.INSTRUCTION_SHA256
                or spec.get("format_instruction_sha256") != source_module.INSTRUCTION_SHA256
                or spec.get("format_change_predeclared_before_future_labels") is not True
                or spec.get("semantic_standards_and_schema_unchanged") is not True
                or spec.get("new_role") != "format_clarified_first_attempt"
                or spec.get("future_original_IAA_population_separate") is not True):
            raise ValueError("Future format clarification differs from its predeclared exact bytes and role")
    elif (annotation_run / "wire_protocol.txt").read_bytes() != prompt + wire_codec.WIRE_INSTRUCTIONS.encode():
        raise ValueError("Actual wire protocol changed")
    packets = {row["record_id"]: protocol.clean_packet(row, prompt=prompt.decode()) for row in rows}
    if (native.read_json(annotation_run / "packets.json") != packets
            or spec["eligible_packets_sha256"] != protocol.sha256(protocol.canonical_bytes(packets))):
        raise ValueError("Canonical source packets changed")
    if is_corrected or is_clarified:
        history, records = source_module.load_history(spec["source_run"], spec["corrective_pair"], packets)
        # The sealed source history now includes TEST questions. It must not be
        # regrouped or passed through the DEV-only v4 migration policy.
        original_groups = history["original_specification"]["groups"]
        groups = (source_module.build_groups([packets[key]["payload"] for key in ids], questions, history["record_ids"])
                  if is_clarified else original_groups)
        corrected_ids = history["corrective_record_ids"]
        selected = [group for group in original_groups if group["batch"] == "batch-0019"]
        if len(corrected_ids) != 10 or len(set(corrected_ids)) != 10 or len(selected) != 1 or selected[0]["record_ids"] != corrected_ids:
            raise ValueError("Only the registered whole B0019 batch may receive a corrective pair")
    else:
        history, records = wire.load_history(spec["source_run"], spec["snapshot_dir"], packets)
        groups = wire.build_groups(rows, questions, history["record_ids"], spec["case_limit"], spec["full_payload_byte_limit"])
    if (native.read_json(annotation_run / "history.json") != history
            or protocol.sha256(protocol.canonical_bytes(history)) != spec["history_sha256"]
            or groups != spec["groups"] or groups != native.read_json(annotation_run / "groups.json")):
        raise ValueError("Wire migration or sealed whole-case groups changed")
    questions_by_id = {row["record_id"]: protocol.clean_question(row) for row in questions}
    context = dict(root=root, pool=pool, run=annotation_run, rows=rows, packets=packets, ids=ids,
        questions=[questions_by_id[key] for key in ids], chunks=[ids[i:i + 20] for i in range(0, len(ids), 20)],
        prompt=prompt, protected=protected, corpus=root / "data/unified_runs/semantic-page-v2-buffer/stage/passages.jsonl",
        bindings=bindings, history=history, source_spec=spec, source_module=source_module,
        corrected=is_corrected or is_clarified, clarified=is_clarified)
    seal_history_cache(context, records, history["sources"])
    return context


def snapshot_chunk(context, output, number, prefix, entries):
    """Only complete durable lines enter a snapshot; full native groups remain whole."""
    ids = context["chunks"][number - 1]
    selected = set(ids)
    native.protected_unchanged(context.get("protected", {}))
    if "history_cache" not in context:
        # Explicit synthetic contexts use the same one-time strict source API.
        module = context.get("source_module", wire)
        records = {side: module.completed_records(context["run"], side, context["packets"]) for side in ("a", "b")}
        sources = {side: module.record_sources(context["run"], side) for side in ("a", "b")}
        context.setdefault("protected", {})
        seal_history_cache(context, records, sources)
    cache = verify_history_cache(context)
    chunk = output / "chunks" / f"chunk-{number:04d}"
    if chunk.exists():
        receipt = native.read_json(chunk / "export_receipt.json")
        if receipt["record_ids"] != ids or receipt["source_annotation_run"] != str(context["run"]):
            raise ValueError("Existing wire chunk source binding changed")
        for name, digest in receipt["artifacts_sha256"].items():
            if protocol.sha256(native.contained_file(chunk, name).read_bytes()) != digest:
                raise ValueError("Immutable wire chunk changed")
        native.protected_unchanged(receipt["source_bindings"])
        bind_run_sources(context, receipt["source_bindings"])
        if protocol.sha256(prefix[:receipt["source_progress_prefix_bytes"]]) != receipt["source_progress_prefix_sha256"]:
            raise ValueError("Durable wire progress prefix changed")
        verify_history_cache(context)
        native.protected_unchanged(context.get("protected", {}))
        return chunk
    events, covered = [], {side: set(context["history"]["record_ids"]) for side in ("a", "b")}
    sealed_groups = {row["batch"]: row["record_ids"] for row in context["source_spec"]["groups"]}
    seen = set()
    for line, event in entries:
        if event.get("type") != "completed" or event.get("accepted") is not True or not selected.intersection(event["record_ids"]):
            continue
        side, batch = event["stream"], event["batch"]
        if side not in covered or sealed_groups.get(batch) != event["record_ids"] or (side, batch) in seen:
            raise ValueError("Durable wire event has unknown, repeated or reordered native membership")
        seen.add((side, batch))
        covered[side].update(event["record_ids"])
        events.append((line, event))
    if any(not selected <= values for values in covered.values()):
        return None
    chunk.mkdir(parents=True, exist_ok=False)
    snapshot = chunk / "durable_snapshot"
    source_bindings = dict(context["history"]["protected_sha256"])
    source_bindings.update(cache["protected_sha256"])
    if context.get("history_cache_artifact"):
        source_bindings[str(context["history_cache_artifact"])] = context["history_cache_sha256"]
    source_bindings.update(context.get("history_cache_receipt_guard", {}))
    source_bindings.update(context.get("resolution_bindings", {}))
    native.write_new(snapshot / "history.json", (context["run"] / "history.json").read_bytes())
    if events:
        native.write_new(snapshot / "progress.jsonl", b"".join(line for line, _ in events))
    source_module = context.get("source_module", wire)
    originals = deepcopy(cache["records"])
    lineage = deepcopy(cache["sources"])
    for _, event in events:
        source = native.contained_file(context["run"], event["receipt"])
        raw = source.read_bytes()
        receipt = native.read_json(source)
        if (protocol.sha256(raw) != event["receipt_sha256"] or receipt.get("accepted") is not True
                or receipt.get("errors") or receipt.get("process_exit_code") != 0
                or receipt["record_ids"] != event["record_ids"] or receipt["stream"] != event["stream"]
                or receipt["batch"] != event["batch"]):
            raise ValueError("Durable whole wire native receipt changed or failed")
        target = snapshot / Path(event["receipt"]).parent
        for name, digest in receipt["artifacts_sha256"].items():
            path = native.contained_file(source.parent, name)
            artifact = path.read_bytes()
            if protocol.sha256(artifact) != digest:
                raise ValueError("Wire raw response, codec or input sidecar changed")
            native.write_new(target / name, artifact)
            source_bindings[str(path)] = digest
        native.write_new(target / "receipt.json", raw)
        source_bindings[str(source)] = event["receipt_sha256"]
        execution = native.parse_execution_events((target / "events.jsonl").read_bytes())
        if execution["errors"] or execution["thread_id"] != receipt["execution_id"]:
            raise ValueError("Wire source execution metadata was not cleanly accepted")
        # Validate the ENTIRE durable native group, not a selected twenty-case
        # subset. Already verified historical contexts are never decoded again.
        values = verified_native_group(context, target, source, receipt)
        side = event["stream"]
        if set(values) != set(event["record_ids"]) or set(values) & set(originals[side]):
            raise ValueError("Whole native coverage changed or historical/accepted records were repeated")
        if context.get("clarified") and (receipt.get("execution_protocol") != CLARIFIED_VERSION
                or any(value.get("annotation_role") != "format_clarified_first_attempt"
                       or value.get("original_first_attempt") is not True for value in values.values())):
            raise ValueError("Future sources must retain their predeclared original-first-attempt role")
        originals[side].update(values)
        for key in values:
            lineage[side][key] = dict(receipt=str(source), receipt_sha256=event["receipt_sha256"],
                exported_receipt=str(target / "receipt.json"), execution_id=receipt["execution_id"],
                stdin_sha256=receipt["full_stdin_sha256"], source_run=str(context["run"]), migrated_unchanged=False,
                raw_encoding=receipt.get("raw_encoding", wire_codec.VERSION), execution_protocol=receipt.get("execution_protocol", wire.VERSION),
                mapping_sha256=receipt["mapping_sha256"])
            if context.get("clarified"):
                lineage[side][key]["annotation_role"] = "format_clarified_first_attempt"
    for key in ids:
        left, right = originals["a"][key], originals["b"][key]
        if context.get("corrected") and key in context["history"]["corrective_record_ids"]:
            if any(value.get("annotation_role") != "source_only_corrective_pair"
                   or value.get("excluded_from_original_agreement") is not True for value in (left, right)):
                raise ValueError("Registered corrective records must retain their new explicit role")
        legacy.full_adjudicator.validate_pair(context["packets"][key], left, right)
        if left["execution_input_sha256"] != right["execution_input_sha256"]:
            raise ValueError("Paired original native full input bytes differ")
        left_path, right_path = (Path(lineage[side][key]["receipt"]).parent for side in ("a", "b"))
        if (left_path / "stdin.txt").read_bytes() != (right_path / "stdin.txt").read_bytes():
            raise ValueError("Actual paired original stdin bytes differ")
        if (left_path / "schema.json").read_bytes() != (right_path / "schema.json").read_bytes():
            raise ValueError("Actual paired original schema bytes differ")
    packets = {key: context["packets"][key] for key in ids}
    native.write_new(chunk / "packets.json", json.dumps(packets, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
    splits = {row["record_id"]: row["split"] for row in context["questions"]}
    native.write_new(chunk / "grouping_metadata.json", {key: splits[key] for key in ids})
    for side in originals:
        native.write_new(chunk / f"independent_{side}.jsonl", b"".join(protocol.canonical_bytes(originals[side][key]) + b"\n" for key in ids))
    native.write_new(chunk / "original_record_sources.json", {side: {key: lineage[side][key] for key in ids} for side in lineage})
    if context.get("corrected"):
        copied = set()
        for side in ("a", "b"):
            for key in ids:
                value = lineage[side][key]
                if key not in context["history"]["corrective_record_ids"]:
                    continue
                if value.get("execution_protocol") != CORRECTION_VERSION:
                    raise ValueError("Unknown or missing registered corrective execution protocol")
                receipt_path = Path(value["receipt"]).resolve()
                if receipt_path in copied:
                    continue
                copied.add(receipt_path)
                receipt = native.read_json(receipt_path)
                if (receipt.get("execution_protocol") != CORRECTION_VERSION
                        or receipt.get("record_ids") != context["history"]["corrective_record_ids"]):
                    raise ValueError("Corrective source is not the exact registered complete paired batch")
                if receipt.get("accepted") is not True:
                    certificate_path = Path(value.get("validation_certificate", "")).resolve()
                    pair_root = Path(context["history"]["corrective_pair"]).resolve()
                    if (receipt.get("accepted") is not False or receipt.get("process_exit_code") != 0
                            or receipt.get("errors") != [AUTHORIZATION_COPY_ERROR]
                            or value.get("validation_protocol") != METADATA_VALIDATION_VERSION
                            or value.get("source_validation_status") != "source_valid_pending_adjudication"
                            or value.get("native_receipt_accepted") is not False
                            or value.get("software_false_alarm") is not True
                            or not certificate_path.is_relative_to(pair_root)
                            or certificate_path.name != "metadata_certificate.json"
                            or protocol.sha256(certificate_path.read_bytes()) != value.get("validation_certificate_sha256")):
                        raise ValueError("A false native receipt needs an exact separate zero-call metadata certificate")
                    certificate = native.read_json(certificate_path)
                    verification = certificate.get("verification", {})
                    if (verification.get("version") != METADATA_VALIDATION_VERSION
                            or verification.get("validation_status") != "source_valid_pending_adjudication"
                            or verification.get("new_native_calls") != 0
                            or verification.get("native_receipts_remain_accepted_false") is not True):
                        raise ValueError("Metadata validation cannot rewrite native acceptance or add hidden calls")
                    source_bindings[str(certificate_path)] = value["validation_certificate_sha256"]
                    target_certificate = chunk / "metadata_revalidation" / certificate_path.name
                    if target_certificate.exists():
                        if target_certificate.read_bytes() != certificate_path.read_bytes():
                            raise ValueError("The two corrective streams have different metadata certificates")
                    else:
                        native.write_new(target_certificate, certificate_path.read_bytes())
                # The one-time history loader already ran the allowlisted
                # metadata/source verifier. Every original file remains guarded.
                verified = {identifier: cache["records"][side][identifier] for identifier in receipt["record_ids"]}
                if verified != {identifier: originals[side][identifier] for identifier in receipt["record_ids"]}:
                    raise ValueError("Corrective validation changed original envelopes, raw bytes or labels")
                target = chunk / "corrective_raw" / side
                for name, digest in receipt["artifacts_sha256"].items():
                    path = native.contained_file(receipt_path.parent, name)
                    blob = path.read_bytes()
                    if protocol.sha256(blob) != digest:
                        raise ValueError("Corrective raw/input/quote instruction artifact changed")
                    native.write_new(target / name, blob)
                    source_bindings[str(path)] = digest
                native.write_new(target / "receipt.json", receipt_path.read_bytes())
                source_bindings[str(receipt_path)] = value["receipt_sha256"]
                if not originals[side][key].get("excluded_from_original_agreement"):
                    raise ValueError("Corrective labels cannot masquerade as original first attempts")
        if copied:
            pair = Path(context["history"]["corrective_pair"])
            proof = native.read_json(pair / "manifest.json")
            names = {**proof["immutable_sha256"], "manifest.json": protocol.sha256((pair / "manifest.json").read_bytes()),
                     "certificate.json": protocol.sha256((pair / "certificate.json").read_bytes())}
            for name, digest in names.items():
                path = native.contained_file(pair, name)
                blob = path.read_bytes()
                if protocol.sha256(blob) != digest:
                    raise ValueError("Corrective prepared manifest, full input or certificate changed")
                native.write_new(chunk / "corrective_proof" / name, blob)
                source_bindings[str(path)] = digest
    agreement = protocol.raw_agreement(packets, [originals["a"][key] for key in ids], [originals["b"][key] for key in ids])
    if agreement["paired_records"] != len(ids) or agreement["pooled_passages_without_paired_judgment"]:
        raise ValueError("Fixed wire chunk lacks complete raw paired judgments")
    native.write_new(chunk / "raw_machine_agreement.json", agreement)
    if context.get("clarified"):
        future_ids = [key for key in ids if originals["a"][key].get("annotation_role") == "format_clarified_first_attempt"]
        if future_ids:
            separate = protocol.raw_agreement({key: packets[key] for key in future_ids},
                [originals["a"][key] for key in future_ids], [originals["b"][key] for key in future_ids])
            separate.update(agreement_population="predeclared_format_clarified_original_first_attempt_pairs_only",
                            original_baseline_and_corrective_pairs_excluded=True)
            native.write_new(chunk / "future_first_attempt_raw_machine_agreement.json", separate)
    if context.get("corrected"):
        native.write_new(chunk / "agreement_population.json", dict(
            agreement_population="accepted_pairs_including_explicit_corrective_pair",
            original_first_attempt_agreement=False,
            corrective_record_ids=[key for key in ids if key in context["history"]["corrective_record_ids"]],
            original_failed_attempts_reclassified=False,
            first_attempt_agreement_artifact=context["history"]["first_attempt_agreement_artifact"],
            future_format_clarified_first_attempt_record_ids=[key for key in ids
                if originals["a"][key].get("annotation_role") == "format_clarified_first_attempt"],
            original_baseline_corrective_and_future_first_attempt_populations_separate=True))
    verify_history_cache(context)
    native.protected_unchanged(context.get("protected", {}))
    native.protected_unchanged(source_bindings)
    native.write_new(chunk / "export_receipt.json", dict(protocol=VERSION, records=len(ids), record_ids=ids,
        source_annotation_run=str(context["run"]), source_progress_prefix_bytes=len(prefix),
        source_progress_prefix_sha256=protocol.sha256(prefix), source_bindings=source_bindings,
        historical_cache_sha256=context["history_cache_sha256"], historical_source_decoded_once=True,
        human_gold=False, raw_agreement_computed_before_adjudication=True, whole_native_groups_preserved=True,
        artifacts_sha256={path.relative_to(chunk).as_posix(): protocol.sha256(path.read_bytes()) for path in chunk.rglob("*") if path.is_file()}))
    return chunk


def verify_legacy(root, pool, previous_run, snapshot):
    """Reconstruct reviews in their own original context, not a wire manifest."""
    previous_run = Path(previous_run).resolve()
    allowed = Path(root).resolve() / "data/machine_runs/cohort_pipeline"
    if not previous_run.is_relative_to(allowed) or previous_run == allowed:
        raise ValueError("Previous source-review run escapes its original private root")
    manifest = native.read_json(previous_run / "manifest.json")
    if manifest.get("version") != legacy.VERSION or (previous_run / "failure.json").exists():
        raise ValueError("Use a completed guarded legacy review prefix without a failure")
    old_base, base, guards = legacy.bases(root, snapshot)
    context = base.prepare(root, pool, manifest["source_annotation_run"], 20)
    prior = legacy.verify_previous(context, manifest["previous_full_response"]["output"], old_base)
    partial = manifest.get("partial_compact_recovery")
    if partial:
        native.protected_unchanged(partial["sha256"])
    legacy.configure(base, old_base, prior, guards, partial)
    if manifest != base.specification(context, 20):
        raise ValueError("Legacy review source or implementation changed before cross-source reuse")
    prefix, entries = base.progress_snapshot(context["run"])
    chunks, bindings = {}, {}
    for number, path in enumerate(sorted((previous_run / "retained").glob("chunk-*.json")), 1):
        if path.name != f"chunk-{number:04d}.json":
            raise ValueError("Legacy reviews are not a fixed source-order prefix")
        source_chunk = base.export_chunk(context, previous_run, number, prefix, entries)
        result = base.verify_retained(previous_run, source_chunk, context["chunks"][number - 1], path)
        decisions = Path(result["adjudications_path"])
        # The final assembler's strict ownership rule stays unchanged.
        for row in protocol.parse_jsonl(decisions.read_bytes()):
            for field in ("raw_response_path", "execution_receipt_path"):
                if not Path(row["adjudication"][field]).resolve().is_relative_to(decisions.parent.resolve() / "executions"):
                    raise ValueError("Historical decisions lost original raw execution ownership")
        chunks[str(number)] = dict(result_path=str(path), source_chunk=str(source_chunk), result=result,
            record_ids=context["chunks"][number - 1])
        for directory in (source_chunk, decisions.parent):
            for artifact in directory.rglob("*"):
                if artifact.is_file():
                    bindings[str(artifact)] = protocol.sha256(artifact.read_bytes())
        bindings[str(path)] = protocol.sha256(path.read_bytes())
    for name in ("manifest.json", "progress.jsonl", "source_inputs_sha256.json"):
        path = previous_run / name
        bindings[str(path)] = protocol.sha256(path.read_bytes())
    native.protected_unchanged(bindings)
    return dict(output=str(previous_run), records=sum(value["result"]["records"] for value in chunks.values()),
        chunks=chunks, sha256=bindings, original_decision_files_preserved=True, native_calls_repeated=0)


def configure(base, reused, resolution_path=None):
    empty = dict(output=None, records=0, chunks={}, sha256={})
    legacy.configure(base, legacy.isolated_base(), empty)
    base.VERSION = VERSION
    resolution_holder = {}
    context_holder = {}
    def prepared(root, pool, source, chunk_size=20):
        context = prepare(root, pool, source, chunk_size)
        if resolution_path:
            resolution = validate_resolution(context, resolution_path)
            context["protected"].update(resolution["protected"])
            context["resolution_bindings"] = resolution["protected"]
            context["resolution"] = {key: value for key, value in resolution.items() if key != "original_progress_prefix"}
            resolution_holder["value"] = resolution
        context_holder["value"] = context
        return context
    base.prepare = prepared
    base.progress_snapshot = lambda run: resolved_progress_snapshot(run, resolution_holder.get("value"))
    base.export_chunk = snapshot_chunk
    original_spec, original_create, original_verify = base.specification, base.create_pipeline, base.verify_retained

    def specification(context, chunk_size):
        value = original_spec(context, chunk_size)
        value["implementation_sha256"].update({str(Path(path).resolve()): protocol.sha256(Path(path).read_bytes())
            for path in (__file__, wire.__file__, wire_codec.__file__)})
        if context.get("corrected"):
            module_path = Path(context["source_module"].__file__).resolve()
            value["implementation_sha256"][str(module_path)] = protocol.sha256(module_path.read_bytes())
            value["agreement_population"] = "accepted_pairs_after_explicit_source_only_corrective_pair"
            value["original_first_attempt_agreement_substitution"] = False
        value["historical_source_cache"] = dict(version=CACHE_VERSION,
            artifact="source_inputs/historical_source_cache.json", sha256=context["history_cache_sha256"],
            record_ids=context["history"]["record_ids"], original_guards_bound=True,
            already_verified_native_contexts_redecoded_per_chunk=False)
        if context.get("clarified"):
            value["future_annotation_role"] = "format_clarified_first_attempt"
            value["future_format_instruction_sha256"] = context["source_spec"]["format_instruction_sha256"]
        value.update(version=VERSION, original_reviews_cross_source_reuse=reused,
            whole_wire_native_artifacts_preserved=True, reused_decision_files_keep_original_execution_ownership=True)
        if context.get("resolution"):
            value["explicit_source_resolution"] = context["resolution"]
        return value

    def compare_originals(context, output, prefix, entries, key, entry):
        number = int(key)
        if entry["record_ids"] != context["chunks"][number - 1]:
            raise ValueError("Historical review membership differs from fixed wire source order")
        chunk = snapshot_chunk(context, output, number, prefix, entries)
        if chunk is None:
            raise ValueError("Historical review no longer has complete paired wire source lineage")
        source = Path(entry["source_chunk"])
        if native.read_json(chunk / "packets.json") != native.read_json(source / "packets.json"):
            raise ValueError("Historical canonical packets were rebound to new wire inputs")
        for side in ("a", "b"):
            if (chunk / f"independent_{side}.jsonl").read_bytes() != (source / f"independent_{side}.jsonl").read_bytes():
                raise ValueError("Historical whole original A/B envelopes were changed during wire reuse")
        return chunk

    def create(context, chunk_size=20, resume=None):
        output = original_create(context, chunk_size, resume)
        cache_path = output / "source_inputs/historical_source_cache.json"
        hashes_path = output / "source_inputs_sha256.json"
        expected_receipt = dict(version=CACHE_VERSION, artifact=cache_path.relative_to(output).as_posix(),
            sha256=context["history_cache_sha256"], original_input_receipt_sha256=protocol.sha256(hashes_path.read_bytes()),
            original_source_validated_once_before_cache=True, native_calls=0)
        if resume is None:
            native.write_new(cache_path, protocol.canonical_bytes(context["history_cache"]))
            # The existing receipt was just created in this new run; create a
            # separate append-only cache receipt instead of overwriting it.
            native.write_new(output / "historical_source_cache_receipt.json", expected_receipt)
        else:
            if (output / "historical_source_cache_receipt.json").read_bytes() != protocol.canonical_bytes(expected_receipt):
                raise ValueError("Historical cache resume binding changed")
        context["history_cache_artifact"] = cache_path
        receipt_path = output / "historical_source_cache_receipt.json"
        context["history_cache_receipt_guard"] = {str(receipt_path): protocol.sha256(receipt_path.read_bytes())}
        verify_history_cache(context)
        prefix, entries = base.progress_snapshot(context["run"])
        for key, entry in reused["chunks"].items():
            compare_originals(context, output, prefix, entries, key, entry)
            if resume is None:
                path = output / "retained" / f"chunk-{int(key):04d}.json"
                native.write_new(path, Path(entry["result_path"]).read_bytes())
                native.append_progress(output / "progress.jsonl", dict(type="chunk_retained", number=int(key),
                    record_ids=entry["record_ids"], result_path=str(path), result_sha256=protocol.sha256(path.read_bytes()),
                    original_review_run=reused["output"], original_decision_path=entry["result"]["adjudications_path"],
                    native_calls_repeated=0))
        native.protected_unchanged(reused["sha256"])
        return output

    def verify(output, chunk, ids, result_path):
        key = str(int(result_path.stem.split("-")[-1]))
        if key not in reused["chunks"]:
            return original_verify(output, chunk, ids, result_path)
        entry = reused["chunks"][key]
        if ids != entry["record_ids"] or result_path.read_bytes() != Path(entry["result_path"]).read_bytes():
            raise ValueError("Reused historical result bytes or coverage changed")
        bound = [row for row in protocol.parse_jsonl((output / "progress.jsonl").read_bytes())
            if row.get("type") == "chunk_retained" and row.get("number") == int(key)]
        if (len(bound) != 1 or bound[0].get("original_review_run") != reused["output"]
                or bound[0]["result_sha256"] != protocol.sha256(result_path.read_bytes())
                or bound[0].get("original_decision_path") != entry["result"]["adjudications_path"]):
            raise ValueError("Historical reuse lost its declared unchanged original ownership")
        native.protected_unchanged(reused["sha256"])
        return entry["result"]

    base.specification, base.create_pipeline, base.verify_retained = specification, create, verify
    original_adjudicate = base.adjudicate_chunk
    def adjudicate(chunk, directory, **kwargs):
        context = context_holder.get("value")
        if context:
            verify_history_cache(context)
            native.protected_unchanged(context["protected"])
        result = original_adjudicate(chunk, directory, **kwargs)
        if context:
            verify_history_cache(context)
            native.protected_unchanged(context["protected"])
        return result
    original_complete = base.export_complete_sources
    def complete_sources(context, output, chunks):
        verify_history_cache(context)
        native.protected_unchanged(context["protected"])
        result = original_complete(context, output, chunks)
        verify_history_cache(context)
        native.protected_unchanged(context["protected"])
        return result
    base.adjudicate_chunk, base.export_complete_sources = adjudicate, complete_sources
    return base


def run(root, pool_dir, annotation_run, *, previous_review_run, historical_guard_snapshot,
        resume=None, max_idle_seconds=600, poll_seconds=5, adjudicate=True, caller=None, resolution_path=None):
    reused = verify_legacy(root, pool_dir, previous_review_run, historical_guard_snapshot)
    base = configure(legacy.isolated_base(), reused, resolution_path)
    return base.run(root, pool_dir, annotation_run, chunk_size=20, resume=resume,
        max_idle_seconds=max_idle_seconds, poll_seconds=poll_seconds, adjudicate=adjudicate, caller=caller)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-root", "pool-dir", "annotation-run", "previous-review-run", "historical-guard-snapshot"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-idle-seconds", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--resolution-receipt", type=Path)
    args = parser.parse_args()
    _, result = run(args.canonical_root, args.pool_dir, args.annotation_run,
        previous_review_run=args.previous_review_run, historical_guard_snapshot=args.historical_guard_snapshot,
        resume=args.resume, max_idle_seconds=args.max_idle_seconds, poll_seconds=args.poll_seconds,
        adjudicate=not args.export_only, resolution_path=args.resolution_receipt)
    print(json.dumps(result, indent=2))
