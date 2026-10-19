"""Fixed-order durable cohort exports and serial source adjudication.

Annotation workers remain untouched. Only receipt bytes bound by a complete
progress line enter a private snapshot, then the original batch verifier checks
that snapshot. Every fixed chunk retains both whole raw responses and its raw
agreement before adjudication. No reserve substitution, gold claim or auto retry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

import adjudicate_machine_annotations as adjudicator
import run_codex_cohort_annotations as cohort
import run_codex_cohort_throughput as throughput
import run_codex_annotation as native
import run_codex_batch_annotations as batch
from oil_evidence import machine_evaluation as protocol
from oil_evidence import codex_runtime as runtime

VERSION = "fixed-order-durable-cohort-adjudication-v1"
MAX_CASES = 10
MAX_CALL_BYTES = 262144


def progress_snapshot(run):
    path = run / "progress.jsonl"
    raw = path.read_bytes() if path.exists() else b""
    end = raw.rfind(b"\n") + 1
    prefix = raw[:end]
    rows = [(line, json.loads(line, object_pairs_hook=protocol._unique_object,
                             parse_constant=protocol._invalid_constant))
            for line in prefix.splitlines(keepends=True) if line.strip()]
    for _, row in rows:
        if row.get("type") == "failed" or (row.get("type") == "completed" and row.get("accepted") is False):
            raise ValueError("Annotation failure retained; pipeline stops without retry")
    return prefix, rows


def prepare(root, pool_dir, annotation_run, chunk_size=20):
    if type(chunk_size) is not int or chunk_size <= 0 or chunk_size % 5:
        raise ValueError("Fixed chunk size must be a positive multiple of five")
    root, pool_dir, annotation_run = map(lambda value: Path(value).resolve(), (root, pool_dir, annotation_run))
    pool_dir, rows, questions, protected, bindings = cohort.load_pool(root, pool_dir)
    if not (annotation_run.is_relative_to(pool_dir / "cohort_annotations")
            or annotation_run.is_relative_to(pool_dir / "cohort_annotations_throughput")):
        raise ValueError("Annotation source run escapes its candidate pool")
    manifest = native.read_json(annotation_run / "manifest.json")
    spec = manifest["specification"]
    ids = spec["eligible_record_ids"]
    is_throughput = spec["version"] == throughput.VERSION
    valid_mode = (spec["case_limit"] <= MAX_CASES and spec["full_payload_byte_limit"] <= MAX_CALL_BYTES
                  and spec["accepted_records_rerun"] is False and spec["old_protocol_outputs_rebound"] is False
                  and spec["annotation_semantic_protocol_unchanged"] is True) if is_throughput else (
                      spec["version"] == cohort.VERSION and spec["batch_size"] == 5 and spec["old_protocol_outputs_reused"] is False)
    if (not valid_mode or spec["partition"] != "all_current" or spec["bindings"] != bindings
            or ids != [row["record_id"] for row in rows]
            or len(set(ids)) != len(ids) or set(ids) != {row["record_id"] for row in questions}):
        raise ValueError("Use the complete fresh all-current cohort in its manifested source order")
    for name, expected in manifest["immutable_sha256"].items():
        path = native.contained_file(annotation_run, name)
        if protocol.sha256(path.read_bytes()) != expected:
            raise ValueError("Immutable annotation source changed")
        protected[str(path)] = expected
    protected[str(annotation_run / "manifest.json")] = protocol.sha256((annotation_run / "manifest.json").read_bytes())
    source_protected = native.read_json(annotation_run / "protected_inputs.json")
    native.protected_unchanged(source_protected)
    for path, digest in source_protected.items():
        if path in protected and protected[path] != digest:
            raise ValueError("Conflicting original source bindings")
        protected[path] = digest
    prompt = (annotation_run / "protocol.txt").read_bytes()
    if prompt != batch.BATCH_PROMPT.encode("utf-8") or protocol.sha256(prompt) != spec["batch_prompt_sha256"]:
        raise ValueError("Bind the actual unchanged batch annotation protocol")
    packets = native.read_json(annotation_run / "packets.json")
    expected = {row["record_id"]: protocol.clean_packet(row, prompt=prompt.decode("utf-8")) for row in rows}
    if packets != expected or protocol.sha256(protocol.canonical_bytes(packets)) != spec["eligible_packets_sha256"]:
        raise ValueError("Original clean packets differ from candidate source evidence")
    packets = expected  # Actual model bytes are canonical; retain outer source order for grouping APIs.
    migration = None
    if is_throughput:
        migration = native.read_json(annotation_run / "migration.json")
        rebuilt, _ = throughput.read_migration(spec["source_run"], packets, bindings)
        groups = throughput.build_groups(rows, questions, migration["record_ids"], spec["case_limit"], spec["full_payload_byte_limit"])
        if (migration != rebuilt or groups != spec["groups"] or groups != native.read_json(annotation_run / "groups.json")
                or protocol.sha256(protocol.canonical_bytes(migration)) != spec["migration_sha256"]
                or migration["record_ids"] != spec["migrated_record_ids"]):
            raise ValueError("Sealed migration or complete-case group policy changed")
    by_question = {row["record_id"]: protocol.clean_question(row) for row in questions}
    chunks = [ids[offset:offset + chunk_size] for offset in range(0, len(ids), chunk_size)]
    return dict(root=root, pool=pool_dir, run=annotation_run, rows=rows, packets=packets, ids=ids,
                questions=[by_question[key] for key in ids], chunks=chunks, prompt=prompt, protected=protected,
                corpus=root / "data/unified_runs/semantic-page-v2-buffer/stage/passages.jsonl", bindings=bindings,
                throughput=is_throughput, migration=migration, source_spec=spec)


def specification(context, chunk_size):
    return dict(version=VERSION, source_annotation_run=str(context["run"]), chunk_size=chunk_size,
        source_order_record_ids=context["ids"], chunks=context["chunks"],
        source_bindings=context["protected"], actual_annotation_prompt_sha256=protocol.sha256(context["prompt"]),
        implementation_sha256={str(Path(path).resolve()): protocol.sha256(Path(path).read_bytes()) for path in
            (__file__, adjudicator.__file__, cohort.__file__, throughput.__file__, batch.__file__, native.__file__, protocol.__file__, runtime.__file__)},
        human_gold=False, formal_dataset_frozen=False, old_protocol_outputs_reused=False)


def create_pipeline(context, chunk_size=20, resume=None):
    spec = specification(context, chunk_size)
    output_parent = context["root"] / "data/machine_runs/cohort_pipeline"
    if resume:
        output = Path(resume).resolve()
        if not output.is_relative_to(output_parent):
            raise ValueError("Pipeline resume escapes the declared pool")
        if native.read_json(output / "manifest.json") != spec:
            raise ValueError("Fixed source order, protocol or pipeline code changed")
        if (output / "failure.json").exists():
            raise ValueError("Retained failure requires explicit owner recovery, never automatic retry")
        for name, expected in native.read_json(output / "source_inputs_sha256.json").items():
            if protocol.sha256(native.contained_file(output / "source_inputs", name).read_bytes()) != expected:
                raise ValueError("Immutable pipeline source input changed")
        return output
    # Keep private nested raw execution paths below Windows' legacy path limit.
    output = output_parent / ("run-" + uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    native.write_new(output / "manifest.json", spec)
    source = output / "source_inputs"
    native.write_new(source / "corpus.jsonl", context["corpus"].read_bytes())
    native.write_new(source / "questions.jsonl", b"".join(protocol.canonical_bytes(row) + b"\n" for row in context["questions"]))
    native.write_new(source / "evidence_pool.jsonl", b"".join(protocol.canonical_bytes(dict(record_id=row["record_id"],
        evidence_ids=[passage["passage_id"] for passage in row["evidence"]])) + b"\n" for row in context["rows"]))
    native.write_new(source / "annotation_prompt.txt", context["prompt"])
    native.write_new(source / "packets.json", (context["run"] / "packets.json").read_bytes())
    native.write_new(output / "source_inputs_sha256.json", {
        path.name: protocol.sha256(path.read_bytes()) for path in source.iterdir() if path.is_file()})
    return output


def accepted_for_chunk(context, entries, ids):
    if context.get("throughput"):
        return accepted_throughput_chunk(context, entries, ids)
    expected_batches = {}
    for offset in range(0, len(context["ids"]), 5):
        group = context["ids"][offset:offset + 5]
        if set(group) & set(ids):
            if not set(group) <= set(ids):
                raise ValueError("Fixed chunk cuts a stable source annotation batch")
            expected_batches[f"batch-{offset // 5 + 1:04d}"] = group
    accepted = {}
    for line, event in entries:
        if event.get("type") != "completed" or event.get("accepted") is not True:
            continue
        key = (event["stream"], event["batch"])
        if event["batch"] not in expected_batches:
            continue
        if key in accepted:
            raise ValueError("Duplicate successful source batch cannot be silently selected")
        if event["stream"] not in {"a", "b"} or event["record_ids"] != expected_batches[event["batch"]]:
            raise ValueError("Durable source batch IDs differ from the pre-fixed question order")
        path = native.contained_file(context["run"], event["receipt"])
        raw = path.read_bytes()
        receipt = json.loads(raw)
        if (protocol.sha256(raw) != event["receipt_sha256"] or receipt.get("accepted") is not True
                or receipt.get("process_exit_code") != 0 or receipt.get("errors")
                or receipt["record_ids"] != expected_batches[event["batch"]]
                or receipt["stream"] != event["stream"] or receipt["batch"] != event["batch"]):
            raise ValueError("Durable annotation receipt changed or is not operationally accepted")
        accepted[key] = (line, event, path, receipt)
    if set(accepted) != {(stream, name) for name in expected_batches for stream in ("a", "b")}:
        return None
    return accepted


def accepted_throughput_chunk(context, entries, ids):
    """Retain full native groups even if a pre-fixed twenty-case chunk cuts one."""
    selected = set(ids)
    accepted = {}
    migration = context["migration"]
    old_prefix, old_entries = progress_snapshot(Path(migration["source_run"]))
    groups = {group["batch"]: group["record_ids"] for group in context["source_spec"]["groups"]}
    sources = {side: {} for side in ("a", "b")}
    for migrated, events, source_run in ((True, old_entries, Path(migration["source_run"])),
                                          (False, entries, context["run"])):
        for line, event in events:
            if event.get("type") != "completed" or event.get("accepted") is not True:
                continue
            record_ids = event["record_ids"]
            if not set(record_ids) & selected:
                continue
            side = event["stream"]
            if side not in sources:
                raise ValueError("Unknown source annotation stream")
            if not migrated and groups.get(event["batch"]) != record_ids:
                raise ValueError("Durable native group differs from its sealed source order")
            path = native.contained_file(source_run, event["receipt"])
            raw, receipt = path.read_bytes(), native.read_json(path)
            if (protocol.sha256(raw) != event["receipt_sha256"] or receipt.get("accepted") is not True
                    or receipt.get("errors") or receipt.get("process_exit_code") != 0
                    or receipt["record_ids"] != record_ids or receipt["stream"] != side or receipt["batch"] != event["batch"]):
                raise ValueError("Source native receipt changed or lacks durable acceptance")
            for key in record_ids:
                if key not in context["packets"] or key in sources[side]:
                    raise ValueError("Duplicate or unknown original source label")
                if migrated:
                    lineage = migration["sources"][side].get(key)
                    if (not lineage or lineage["receipt"] != str(path)
                            or lineage["receipt_sha256"] != event["receipt_sha256"]):
                        raise ValueError("Migrated envelope receipt lineage changed")
                sources[side][key] = path
            accepted[(side, "old-" + event["batch"] if migrated else event["batch"])] = (line, event, path, receipt)
    if any(not selected <= set(sources[side]) for side in sources):
        return None
    return accepted


def export_chunk(context, output, number, prefix, entries):
    ids = context["chunks"][number - 1]
    accepted = accepted_for_chunk(context, entries, ids)
    if accepted is None:
        return None
    chunk = output / "chunks" / f"chunk-{number:04d}"
    if chunk.exists():
        receipt = native.read_json(chunk / "export_receipt.json")
        if receipt["record_ids"] != ids:
            raise ValueError("Existing chunk source order changed")
        for name, expected in receipt["artifacts_sha256"].items():
            if protocol.sha256(native.contained_file(chunk, name).read_bytes()) != expected:
                raise ValueError("Immutable exported chunk changed")
        native.protected_unchanged(receipt["source_bindings"])
        saved_prefix = prefix[:receipt["source_progress_prefix_bytes"]]
        if protocol.sha256(saved_prefix) != receipt["source_progress_prefix_sha256"]:
            raise ValueError("Durable annotation progress prefix changed across resume")
        return chunk
    chunk.mkdir(parents=True, exist_ok=False)
    snapshot = chunk / "durable_snapshot"
    source_bindings = {}
    copied_lines = []
    for (stream, name), (line, event, path, receipt) in accepted.items():
        migrated = context.get("throughput") and not path.is_relative_to(context["run"])
        target = (chunk / "migrated_raw" if migrated else snapshot) / Path(event["receipt"]).parent
        for filename, expected in receipt["artifacts_sha256"].items():
            source = native.contained_file(path.parent, filename)
            raw = source.read_bytes()
            if protocol.sha256(raw) != expected:
                raise ValueError("Original raw batch artifact changed")
            native.write_new(target / filename, raw)
            source_bindings[str(source)] = expected
        native.write_new(target / "receipt.json", path.read_bytes())
        source_bindings[str(path)] = event["receipt_sha256"]
        if not migrated:
            copied_lines.append(line)
        rows = [row for row in context["rows"] if row["record_id"] in set(receipt["record_ids"])]
        _, expected_stdin, expected_schema = batch.prepare_batch(rows)
        if ((target / "stdin.txt").read_bytes() != expected_stdin
                or (target / "schema.json").read_bytes() != protocol.canonical_bytes(expected_schema)):
            raise ValueError("Source batch full stdin/schema differs from original pool and protocol")
        events = native.parse_execution_events((target / "events.jsonl").read_bytes())
        if events["errors"] or events["thread_id"] != receipt["execution_id"]:
            raise ValueError("Accepted receipt does not bind a clean native execution")
    if copied_lines or not context.get("throughput"):
        native.write_new(snapshot / "progress.jsonl", b"".join(copied_lines))
    if context.get("throughput"):
        native.write_new(snapshot / "migration.json", (context["run"] / "migration.json").read_bytes())
        originals = {side: throughput.completed_records(snapshot, side, context["packets"]) for side in ("a", "b")}
        lineage = {side: throughput.record_sources(snapshot, side) for side in ("a", "b")}
        for side in lineage:
            for key, value in lineage[side].items():
                if not value["migrated_unchanged"]:
                    copied_path = Path(value["receipt"])
                    value["exported_receipt"] = str(copied_path)
                    value["receipt"] = str(context["run"] / copied_path.relative_to(snapshot))
                    value["source_run"] = str(context["run"])
        native.write_new(chunk / "original_record_sources.json", {side: {key: lineage[side][key] for key in ids} for side in lineage})
    else:
        originals = {side: batch.completed_records(snapshot, side, context["packets"]) for side in ("a", "b")}
    originals = {side: {key: values[key] for key in ids} for side, values in originals.items()}
    packets = {record_id: context["packets"][record_id] for record_id in ids}
    for record_id in ids:
        adjudicator.validate_pair(packets[record_id], originals["a"][record_id], originals["b"][record_id])
    # Preserve source order in the outer JSON object for the unchanged adjudicator.
    native.write_new(chunk / "packets.json", json.dumps(packets, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
    splits = {row["record_id"]: row["split"] for row in context["questions"]}
    native.write_new(chunk / "grouping_metadata.json", {key: splits[key] for key in ids})
    for side in ("a", "b"):
        native.write_new(chunk / f"independent_{side}.jsonl", b"".join(
            protocol.canonical_bytes(originals[side][record_id]) + b"\n" for record_id in ids))
    agreement = protocol.raw_agreement(packets, list(originals["a"].values()), list(originals["b"].values()))
    if agreement["paired_records"] != len(ids) or agreement["pooled_passages_without_paired_judgment"]:
        raise ValueError("Fixed exported chunk lacks complete paired judgments")
    native.write_new(chunk / "raw_machine_agreement.json", agreement)
    native.protected_unchanged(source_bindings)
    native.write_new(chunk / "export_receipt.json", dict(protocol=VERSION, record_ids=ids, records=len(ids),
        source_annotation_run=str(context["run"]), source_progress_prefix_sha256=protocol.sha256(prefix),
        source_progress_prefix_bytes=len(prefix), source_bindings=source_bindings, human_gold=False,
        raw_agreement_computed_before_adjudication=True, original_inputs_unchanged=True,
        artifacts_sha256={path.relative_to(chunk).as_posix(): protocol.sha256(path.read_bytes())
                         for path in chunk.rglob("*") if path.is_file()}))
    return chunk


def verify_retained(output, chunk, ids, result_path):
    """Rebuild retained decisions from unchanged raw executions, without calls."""
    progress = output / "progress.jsonl"
    bound = [event for event in (protocol.parse_jsonl(progress.read_bytes()) if progress.exists() else [])
             if event.get("type") == "chunk_retained" and event.get("result_path") == str(result_path)]
    if len(bound) != 1 or bound[0]["record_ids"] != ids or bound[0]["result_sha256"] != protocol.sha256(result_path.read_bytes()):
        raise ValueError("Retained result lacks its unique append-only progress binding")
    result = native.read_json(result_path)
    path = Path(result["adjudications_path"]).resolve()
    if not path.is_relative_to(output / "adjudications") or protocol.sha256(path.read_bytes()) != result["adjudications_sha256"]:
        raise ValueError("Retained adjudication result changed or escapes the pipeline")
    packets = native.read_json(chunk / "packets.json")
    left, right = (adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b"))
    decisions = protocol.parse_jsonl(path.read_bytes())
    if [row["record_id"] for row in decisions] != ids or result["records"] != len(ids):
        raise ValueError("Retained adjudication lost the fixed source order or questions")
    for decision in decisions:
        review = decision["adjudication"]
        raw_path, receipt_path = map(lambda value: Path(value).resolve(),
                                    (review["raw_response_path"], review["execution_receipt_path"]))
        if (not raw_path.is_relative_to(path.parent / "executions")
                or not receipt_path.is_relative_to(path.parent / "executions") or raw_path.parent != receipt_path.parent):
            raise ValueError("Retained raw adjudication escapes its original execution")
        receipt = native.read_json(receipt_path)
        if receipt.get("accepted") is not True:
            raise ValueError("Retained native adjudication was not operationally accepted")
        for name, expected in receipt["artifacts_sha256"].items():
            if protocol.sha256(native.contained_file(receipt_path.parent, name).read_bytes()) != expected:
                raise ValueError("Retained adjudication execution artifact changed")
        raw = raw_path.read_bytes()
        if protocol.sha256(raw) != review["raw_response_sha256"] or protocol.sha256(raw) != receipt["artifacts_sha256"]["raw_final.json"]:
            raise ValueError("Retained adjudication raw response changed")
        parsed = json.loads(raw, object_pairs_hook=protocol._unique_object, parse_constant=protocol._invalid_constant)
        key = decision["record_id"]
        rebuilt = adjudicator.retain_decisions({key: parsed[key]}, packets, left, right, [key], receipt)[0]
        if protocol.canonical_bytes(rebuilt) != protocol.canonical_bytes(decision):
            raise ValueError("Retained adjudication differs from its exact raw model decision")
    if (result["accepted"] != sum(row["status"] == "accepted" for row in decisions)
            or result["unresolved"] != sum(row["status"] == "unresolved" for row in decisions)):
        raise ValueError("Retained result status counts changed")
    return result


def actual_adjudication_bytes(payload, schema):
    stdin = (runtime.ISOLATION_PROMPT + "\n" + adjudicator.SYSTEM + "\nUNTRUSTED_PAYLOAD_JSON\n").encode("utf-8")
    stdin += protocol.canonical_bytes(payload)
    return stdin, protocol.canonical_bytes(schema)


def adjudication_groups(packets, left, right, ids, max_cases=MAX_CASES, max_bytes=MAX_CALL_BYTES, splits=None):
    """Fixed-order whole cases only; reject even one oversized case, no truncation."""
    if type(max_cases) is not int or not 1 <= max_cases <= MAX_CASES or type(max_bytes) is not int or not 1 <= max_bytes <= MAX_CALL_BYTES:
        raise ValueError("Adjudication exceeds the max-ten or hard full-input byte policy")
    groups, current = [], []
    for key in ids:
        if splits is not None:
            if splits.get(key) not in {"development", "test"}:
                raise ValueError("Unknown private grouping split")
            if current and splits[current[0]] != splits[key]:
                groups.append(current)
                current = []
        payload, schema = adjudicator.prepare_batch(packets, left, right, current + [key])
        stdin, schema_bytes = actual_adjudication_bytes(payload, schema)
        if len(current) < max_cases and len(stdin) + len(schema_bytes) <= max_bytes:
            current.append(key)
            continue
        if current:
            groups.append(current)
        payload, schema = adjudicator.prepare_batch(packets, left, right, [key])
        stdin, schema_bytes = actual_adjudication_bytes(payload, schema)
        if len(stdin) + len(schema_bytes) > max_bytes:
            raise ValueError("One complete adjudication case exceeds the hard byte bound; retain it without truncation")
        current = [key]
    if current:
        groups.append(current)
    return groups


def adjudicate_chunk(chunk, output_parent, caller=None):
    """Separate bounded adapter; the existing five-case adjudicator is untouched."""
    packets = native.read_json(chunk / "packets.json")
    ids = list(packets)
    left, right = (adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b"))
    if set(left) != set(packets) or set(right) != set(packets):
        raise ValueError("Complete original paired coverage is required")
    metadata = chunk / "grouping_metadata.json"
    splits = native.read_json(metadata) if metadata.exists() else None
    groups = adjudication_groups(packets, left, right, ids, splits=splits)
    protected = {str(path): protocol.sha256(path.read_bytes()) for path in
                 (chunk / "packets.json", chunk / "independent_a.jsonl", chunk / "independent_b.jsonl",
                  Path(__file__), Path(adjudicator.__file__), Path(protocol.__file__), Path(runtime.__file__))}
    if splits is not None:
        protected[str(metadata)] = protocol.sha256(metadata.read_bytes())
    output = Path(output_parent) / "bounded-run"
    spec = dict(version=VERSION, groups=groups, max_cases=MAX_CASES, max_full_input_and_schema_bytes=MAX_CALL_BYTES,
                input_sha256=protected, protocol_sha256=protocol.sha256(adjudicator.SYSTEM.encode()),
                isolation_sha256=protocol.sha256(runtime.ISOLATION_PROMPT.encode()), human_gold=False,
                grouping_policy="Greedy complete cases in source order with private contiguous split boundaries")
    if output.exists():
        if native.read_json(output / "inputs.json") != spec:
            raise ValueError("Bounded adjudication resume protocol, groups or inputs changed")
        # Never automatically retry an incomplete or rejected native attempt.
        for attempt in (output / "executions").iterdir() if (output / "executions").exists() else []:
            if not (attempt / "receipt.json").exists() or native.read_json(attempt / "receipt.json").get("accepted") is not True:
                raise ValueError("Retained interrupted or failed native attempt needs explicit owner recovery")
    else:
        native.write_new(output / "inputs.json", spec)
    caller = caller or runtime.call_structured_json
    original_contexts = {stream[key]["identity"]["execution_id"] for stream in (left, right) for key in ids}
    completed, execution_ids, new_calls = {}, set(), 0
    for number, group in enumerate(groups, 1):
        payload, schema = adjudicator.prepare_batch(packets, left, right, group)
        stdin, schema_bytes = actual_adjudication_bytes(payload, schema)
        path = output / "retained_batches" / f"batch-{number:04d}.json"
        if path.exists():
            binding = native.read_json(path)
            receipt_path = native.contained_file(output, binding["receipt_path"])
            if protocol.sha256(receipt_path.read_bytes()) != binding["receipt_sha256"]:
                raise ValueError("Retained bounded native receipt changed")
            receipt = native.read_json(receipt_path)
            parsed = native.read_json(receipt_path.parent / "raw_final.json")
        else:
            # Unbound accepted native calls are not automatically replayed.
            bound_receipts = {native.read_json(saved)["receipt_path"] for saved in (output / "retained_batches").glob("*.json")}
            actual_receipts = {saved.relative_to(output).as_posix() for saved in (output / "executions").glob("*/receipt.json")}
            if actual_receipts != bound_receipts:
                raise ValueError("Unbound retained native call needs explicit owner recovery, not replay")
            native.protected_unchanged(protected)
            parsed, receipt = caller(adjudicator.SYSTEM, payload, schema, output / "executions", role="adjudicator")
            new_calls += 1
            receipt_path = Path(receipt["artifact_directory"]) / "receipt.json"
        directory = receipt_path.parent.resolve()
        if not directory.is_relative_to((output / "executions").resolve()) or receipt.get("accepted") is not True:
            raise ValueError("Bounded native receipt escapes its run or was not accepted")
        if native.read_json(receipt_path) != receipt:
            raise ValueError("Actual native receipt differs from caller metadata")
        for name, expected in receipt["artifacts_sha256"].items():
            if protocol.sha256(native.contained_file(directory, name).read_bytes()) != expected:
                raise ValueError("Bounded native raw artifact changed")
        if (directory / "stdin.txt").read_bytes() != stdin or (directory / "schema.json").read_bytes() != schema_bytes:
            raise ValueError("Actual native full stdin/schema differs from the bounded plan")
        events = native.parse_execution_events((directory / "events.jsonl").read_bytes())
        if (events["errors"] or events["thread_id"] != receipt["execution_id"]
                or events["observed_model"] != receipt["observed_model"]):
            raise ValueError("Bounded native execution metadata differs")
        execution_id = receipt["execution_id"]
        if not execution_id or execution_id in execution_ids or execution_id in original_contexts:
            raise ValueError("Source adjudication requires distinct fresh execution contexts")
        execution_ids.add(execution_id)
        if protocol.canonical_bytes(parsed) != protocol.canonical_bytes(native.read_json(directory / "raw_final.json")):
            raise ValueError("Caller JSON differs from the entire retained raw response")
        decisions = adjudicator.retain_decisions(parsed, packets, left, right, group, receipt)
        if path.exists():
            if binding["record_ids"] != group or binding["decisions"] != decisions:
                raise ValueError("Retained bounded decisions differ from exact raw response")
        else:
            native.write_new(path, dict(record_ids=group, decisions=decisions,
                receipt_path=receipt_path.relative_to(output).as_posix(), receipt_sha256=protocol.sha256(receipt_path.read_bytes()),
                actual_full_input_and_schema_bytes=len(stdin) + len(schema_bytes), human_gold=False))
        completed.update({row["record_id"]: row for row in decisions})
        native.protected_unchanged(protected)
        print(f"Bounded source adjudication retained {len(completed)}/{len(ids)} records", flush=True)
    if set(completed) != set(ids):
        raise ValueError("Bounded source adjudication lost questions")
    result_path = output / ("adjudications-" + uuid4().hex[:8] + ".jsonl")
    native.write_new(result_path, b"".join(protocol.canonical_bytes(completed[key]) + b"\n" for key in ids))
    result = dict(output=str(output), adjudications_path=str(result_path), records=len(ids),
        accepted=sum(row["status"] == "accepted" for row in completed.values()),
        unresolved=sum(row["status"] == "unresolved" for row in completed.values()), new_native_calls=new_calls,
        retained_native_calls=len(execution_ids), human_gold=False, original_inputs_unchanged=True,
        external_paid_api_calls=0, adjudications_sha256=protocol.sha256(result_path.read_bytes()))
    native.write_new(output / ("result-" + uuid4().hex[:8] + ".json"), result)
    return result


def export_complete_sources(context, output, chunks):
    """Join the six original freeze inputs only after every paired record exists."""
    if len(chunks) != len(context["chunks"]):
        return None
    target = output / "complete_source_inputs"
    if target.exists():
        receipt = native.read_json(target / "source_receipt.json")
        for name, digest in receipt["artifacts_sha256"].items():
            if protocol.sha256(native.contained_file(target, name).read_bytes()) != digest:
                raise ValueError("Complete source export changed")
        native.protected_unchanged(receipt["source_sha256"])
        return target
    protected, raw_streams = {}, {side: [] for side in ("a", "b")}
    for name in ("corpus.jsonl", "questions.jsonl", "evidence_pool.jsonl", "annotation_prompt.txt"):
        path = output / "source_inputs" / name
        protected[str(path)] = protocol.sha256(path.read_bytes())
        native.write_new(target / name, path.read_bytes())
    for chunk in chunks:
        for side in ("a", "b"):
            path = chunk / f"independent_{side}.jsonl"
            raw = path.read_bytes()
            protected[str(path)] = protocol.sha256(raw)
            raw_streams[side].extend(protocol.parse_jsonl(raw))
    for side in ("a", "b"):
        if [row["record_id"] for row in raw_streams[side]] != context["ids"]:
            raise ValueError("Complete source export differs from pre-fixed question coverage/order")
        native.write_new(target / f"independent_{side}.jsonl", b"".join(
            protocol.canonical_bytes(row) + b"\n" for row in raw_streams[side]))
    agreement = protocol.raw_agreement(context["packets"], raw_streams["a"], raw_streams["b"])
    if agreement["paired_records"] != len(context["ids"]) or agreement["pooled_passages_without_paired_judgment"]:
        raise ValueError("Complete raw agreement coverage is missing")
    native.write_new(target / "raw_machine_agreement.json", agreement)
    native.protected_unchanged(protected)
    native.write_new(target / "source_receipt.json", dict(records=len(context["ids"]),
        record_ids=context["ids"], original_inputs_unchanged=True, human_gold=False, formal_dataset_frozen=False,
        source_sha256=protected, artifacts_sha256={path.name: protocol.sha256(path.read_bytes())
                                                for path in target.iterdir() if path.is_file()}))
    return target


def run(root, pool_dir, annotation_run, *, chunk_size=20, resume=None, max_idle_seconds=600,
        poll_seconds=5, adjudicate=True, caller=None):
    if type(max_idle_seconds) is not int or max_idle_seconds < 0 or not 1 <= poll_seconds <= 30:
        raise ValueError("Use bounded idle waits and polling of one through thirty seconds")
    context = prepare(root, pool_dir, annotation_run, chunk_size)
    output = create_pipeline(context, chunk_size, resume)
    prior_prefix, idle_started = b"", monotonic()
    retained, current_calls, exported = {}, 0, {}
    try:
        while True:
            prefix, entries = progress_snapshot(context["run"])
            if not prefix.startswith(prior_prefix):
                raise ValueError("Source annotation progress was rewritten instead of appended")
            prior_prefix = prefix
            moved = False
            for number, ids in enumerate(context["chunks"], 1):
                result_path = output / "retained" / f"chunk-{number:04d}.json"
                chunk = export_chunk(context, output, number, prefix, entries)
                if chunk is None:
                    if result_path.exists():
                        raise ValueError("Retained result has no durable complete paired originals")
                    break
                exported[number] = chunk
                if result_path.exists():
                    if number not in retained:
                        retained[number] = verify_retained(output, chunk, ids, result_path)
                    continue
                if not adjudicate:
                    continue
                # One synchronous caller means at most one native adjudication
                # context, alongside the annotation runner's two workers.
                result = adjudicate_chunk(chunk, output / "adjudications" / f"chunk-{number:04d}", caller=caller)
                if result["records"] != len(ids):
                    raise ValueError("Adjudication did not retain every fixed question")
                current_calls += result["new_native_calls"]
                native.write_new(result_path, result)
                retained[number] = result
                native.append_progress(output / "progress.jsonl", dict(type="chunk_retained", number=number,
                    record_ids=ids, result_path=str(result_path), result_sha256=protocol.sha256(result_path.read_bytes())))
                print(f"Source adjudication chunk{number}: {result['accepted']} accepted, {result['unresolved']} unresolved", flush=True)
                moved = True
            if len(retained) == len(context["chunks"]) or not adjudicate:
                break
            if moved:
                idle_started = monotonic()
            if monotonic() - idle_started >= max_idle_seconds:
                break
            sleep(poll_seconds)
        complete_sources = export_complete_sources(context, output, list(exported.values()))
        native.protected_unchanged(context["protected"])
        native.protected_unchanged(native.read_json(output / "manifest.json")["implementation_sha256"])
        result = dict(output=str(output), expected_records=len(context["ids"]),
            exported_chunks=len(list((output / "chunks").glob("*/export_receipt.json"))),
            retained_chunks=len(retained), retained_records=sum(row["records"] for row in retained.values()),
            accepted=sum(row["accepted"] for row in retained.values()), unresolved=sum(row["unresolved"] for row in retained.values()),
            complete=len(retained) == len(context["chunks"]), human_gold=False, formal_dataset_frozen=False,
            new_model_calls=current_calls,
            retained_model_calls=sum(row.get("retained_native_calls", row["new_native_calls"]) for row in retained.values()),
            complete_source_inputs=str(complete_sources) if complete_sources else None,
            adjudication_files=[row["adjudications_path"] for _, row in sorted(retained.items())])
        native.write_new(output / ("summary-" + uuid4().hex[:8] + ".json"), result)
        return output, result
    except Exception as error:
        native.write_new(output / "failure.json", dict(type=type(error).__name__, error=str(error),
            retained_chunks=sorted(retained), all_attempts_preserved=True, automatic_retry=False, human_gold=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("canonical-root", "pool-dir", "annotation-run"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-idle-seconds", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    _, summary = run(args.canonical_root, args.pool_dir, args.annotation_run, chunk_size=args.chunk_size,
        resume=args.resume, max_idle_seconds=args.max_idle_seconds, poll_seconds=args.poll_seconds,
        adjudicate=not args.export_only)
    print(json.dumps(summary, indent=2))
