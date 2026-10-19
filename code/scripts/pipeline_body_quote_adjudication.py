"""Fixed-order body-only source reviews with byte-exact prior ownership reuse.

The new producer's already-validated paired history is cached once. Original
560 decisions and the partial nine-case gate retain their own raw executions.
No rejected transport/title-only quote output is projected as an accepted label.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import pipeline_transport_recovery_adjudication as previous

native, protocol, frozen, legacy = previous.native, previous.protocol, previous.frozen, previous.legacy
VERSION = "fixed-order-body-only-source-adjudication-v1"
PRODUCER_VERSION = "body-only-clarified-cohort-v1"
PRODUCER_SHA256 = "22756d7207c6a0e688b978a0d8ff4abdb04cccee07e93dc89762a33dcf42121d"
CORRECTIVE_VERSION = "source-only-body-quote-corrective-pair-v1"
CORRECTIVE_ROLE = "body_quote_corrective_pair"
FIRST_ROLE = "body_only_first_attempt"
PRIOR_MANIFEST_SHA256 = "35504bded78c74846645563abe48d35884c46e32ee19cfbc27373aa3d8c42559"
GATE_SHA256 = "6158e95d60e0a09d566977bd5f55b01b9ce41ba6b01adc0884c057c9dbbcce55"


def producer_module():
    import run_body_quote_cohort as producer
    if (producer.VERSION != PRODUCER_VERSION or producer.ROLE != FIRST_ROLE
            or producer.body.VERSION != CORRECTIVE_VERSION or producer.body.ROLE != CORRECTIVE_ROLE
            or protocol.sha256(Path(producer.__file__).read_bytes()) != PRODUCER_SHA256):
        raise ValueError("Unregistered or changed body-only annotation producer/corrective protocol")
    return producer


def prior_reviews(review_run):
    """Use the certified immutable prefix proof, never decode old history again."""
    review = Path(review_run).resolve()
    guards = {}
    previous.guard_file(review / "manifest.json", guards, PRIOR_MANIFEST_SHA256)
    manifest = native.read_json(review / "manifest.json")
    if manifest["version"] != previous.VERSION:
        raise ValueError("Previous body source reuse does not match the certified recovery wrapper")
    reused, stopped = manifest["original_source_review_reuse"], manifest["original_source_review_stop"]
    if reused["records"] != 560 or len(reused["chunks"]) != 28 or stopped["failed_chunk"] != 29:
        raise ValueError("Certified original source-review prefix or failed owner changed")
    previous.merge_guards(guards, manifest["implementation_sha256"])
    previous.merge_guards(guards, reused["sha256"])
    previous.merge_guards(guards, stopped["protected_sha256"])
    pending = review / "adjudications/chunk-0029/bounded-run"
    previous.guard_file(pending / "retry_gate.json", guards, GATE_SHA256)
    pending_base = legacy.isolated_base()
    legacy.configure(pending_base, legacy.isolated_base(), dict(output=None, records=0, chunks={}, sha256={}))
    pending_base.VERSION = previous.VERSION
    _, gate_guards = previous.verify_retry_gate(pending_base, review / "chunks/chunk-0029",
        pending.parent, stopped)
    previous.merge_guards(guards, gate_guards)
    export_path = review / "chunks/chunk-0029/export_receipt.json"
    previous.guard_file(export_path, guards)
    export = native.read_json(export_path)
    for name, digest in export["artifacts_sha256"].items():
        previous.guard_file(native.contained_file(export_path.parent, name), guards, digest)
    previous.merge_guards(guards, export["source_bindings"])
    native.protected_unchanged(guards)
    return reused, stopped, dict(review_run=str(review), source_chunk=str(export_path.parent),
        output_parent=str(pending.parent), bounded_output=str(pending), gate_records=9,
        original_native_ownership_preserved=True, protected_sha256=guards), pending_base


def prepare(root, pool_dir, annotation_run, chunk_size=20):
    if type(chunk_size) is not int or chunk_size != 20:
        raise ValueError("Body source review preserves fixed twenty-question chunk membership")
    producer = producer_module()
    root, pool, run = (Path(value).resolve() for value in (root, pool_dir, annotation_run))
    loaded, rows, questions, guards, bindings = frozen.wire.previous.load_pool(root, pool)
    if loaded != pool or not run.is_relative_to(pool / "cohort_annotations_body_only"):
        raise ValueError("Body-only producer escapes the declared candidate pool")
    # This reads the producer's sealed cache; no nested cold load_history call.
    manifest, history, source_guards = producer.verify_prepared(run)
    previous.merge_guards(guards, source_guards)
    previous.merge_guards(guards, history["protected_sha256"])
    spec = manifest["specification"]
    ids = [row["record_id"] for row in rows]
    if (spec["version"] != PRODUCER_VERSION or spec["bindings"] != bindings
            or spec["eligible_record_ids"] != ids or len(ids) != 1200 or len(set(ids)) != 1200
            or set(ids) != {row["record_id"] for row in questions}
            or history["record_ids"] != ids[:620] or spec["historical_records_per_stream"] != 620
            or spec["annotation_role"] != FIRST_ROLE or spec["no_old_failure_reclassification"] is not True):
        raise ValueError("Body source order, complete history or declared first-attempt role changed")
    prompt = frozen.wire.batch.BATCH_PROMPT.encode()
    packets = {row["record_id"]: protocol.clean_packet(row, prompt=prompt.decode()) for row in rows}
    if (packets != native.read_json(run / "packets.json")
            or spec["batch_prompt_sha256"] != protocol.sha256(prompt)
            or spec["eligible_packets_sha256"] != protocol.sha256(protocol.canonical_bytes(packets))
            or spec["format_instruction_sha256"] != producer.body.INSTRUCTION_SHA256
            or (run / "instruction.txt").read_bytes() != producer.body.correction.CORRECTION_INSTRUCTION.encode()):
        raise ValueError("Body-only canonical sources, semantic protocol or exact clarification changed")
    if ([key for group in spec["groups"] for key in group["record_ids"]] != ids[620:]
            or len(spec["groups"]) != 59 or spec["groups"][0]["batch"] != "batch-0027"
            or spec["groups"][-1]["batch"] != "batch-0085"
            or any(group.get("annotation_role") != FIRST_ROLE or group.get("original_first_attempt") is not True
                   or not 1 <= len(group["record_ids"]) <= 10 or group["full_payload_bytes"] > 262144 for group in spec["groups"])):
        raise ValueError("Untouched body first-attempt groups or complete-case byte policy changed")
    records = history["records"]
    if any(set(records[side]) != set(ids[:620]) or set(history["sources"][side]) != set(ids[:620]) for side in ("a", "b")):
        raise ValueError("Body cached paired history or native source ownership is incomplete")
    corrected = set(history["body_corrective_records"])
    if corrected != set(ids[610:620]):
        raise ValueError("Only the registered whole rejected title-quote batch may receive new corrective labels")
    for key in corrected:
        if any(records[side][key].get("annotation_role") != CORRECTIVE_ROLE
               or records[side][key].get("original_first_attempt") is not False
               or history["sources"][side][key].get("execution_protocol") != CORRECTIVE_VERSION for side in ("a", "b")):
            raise ValueError("Body corrective labels cannot masquerade as original first attempts")
    by_question = {row["record_id"]: protocol.clean_question(row) for row in questions}
    context = dict(root=root, pool=pool, run=run, rows=rows, packets=packets, ids=ids,
        questions=[by_question[key] for key in ids], chunks=[ids[offset:offset + 20] for offset in range(0, 1200, 20)],
        prompt=prompt, protected=guards, corpus=root / "data/unified_runs/semantic-page-v2-buffer/stage/passages.jsonl",
        bindings=bindings, history=history, source_spec=spec, source_module=producer, corrected=False, clarified=False)
    frozen.seal_history_cache(context, records, history["sources"])
    return context


def extra_source_proofs(context, chunk):
    """Retain whole cached native artifacts before per-record selection."""
    cache = frozen.verify_history_cache(context)
    ids = list(native.read_json(chunk / "packets.json"))
    guards, copied = {}, set()
    for side in ("a", "b"):
        for key in ids:
            if key not in cache["records"][side]:
                continue
            value = cache["sources"][side][key]
            receipt_path = Path(value["receipt"]).resolve()
            if receipt_path in copied:
                continue
            copied.add(receipt_path)
            previous.guard_file(receipt_path, guards, value["receipt_sha256"])
            receipt = native.read_json(receipt_path)
            target = chunk / "cached_native_proofs" / side / value["receipt_sha256"][:12]
            for name, digest in receipt["artifacts_sha256"].items():
                source = native.contained_file(receipt_path.parent, name)
                previous.guard_file(source, guards, digest)
                path = target / name
                if path.exists():
                    if path.read_bytes() != source.read_bytes():
                        raise ValueError("Cached whole native source proof changed")
                else:
                    native.write_new(path, source.read_bytes())
            copied_receipt = target / "receipt.json"
            if copied_receipt.exists() and copied_receipt.read_bytes() != receipt_path.read_bytes():
                raise ValueError("Cached original source receipt copy changed")
            if not copied_receipt.exists():
                native.write_new(copied_receipt, receipt_path.read_bytes())
            for field in ("correction_certificate", "validation_certificate"):
                if field in value:
                    source = Path(value[field]).resolve()
                    digest = value[field + "_sha256"]
                    previous.guard_file(source, guards, digest)
                    path = target / source.name
                    if path.exists() and path.read_bytes() != source.read_bytes():
                        raise ValueError("Cached corrective certificate changed")
                    if not path.exists():
                        native.write_new(path, source.read_bytes())
    frozen.bind_run_sources(context, guards)
    return guards


def validate_roles(context, chunk):
    ids = list(native.read_json(chunk / "packets.json"))
    originals = {side: legacy.full_adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b")}
    body_first, body_corrective, transport_retry = [], [], []
    historical = set(context["history"]["record_ids"])
    for key in ids:
        if key in historical:
            if any(originals[side][key] != context["history_cache"]["records"][side][key] for side in originals):
                raise ValueError("Body export changed a whole accepted/corrective historical source envelope")
        elif any(originals[side][key].get("annotation_role") != FIRST_ROLE
                 or originals[side][key].get("original_first_attempt") is not True for side in originals):
            raise ValueError("New body source must preserve its declared first-attempt role")
        role = originals["a"][key].get("annotation_role")
        if role == FIRST_ROLE:
            body_first.append(key)
        elif role == CORRECTIVE_ROLE:
            body_corrective.append(key)
        elif role == previous.RETRY_ROLE:
            transport_retry.append(key)
        if role in {CORRECTIVE_ROLE, previous.RETRY_ROLE} and any(originals[side][key].get("original_first_attempt") is not False for side in originals):
            raise ValueError("Corrective or transport retry output cannot enter original-first-attempt agreement")
    return dict(body_only_first_attempt_record_ids=body_first, body_corrective_record_ids=body_corrective,
        transport_retry_record_ids=transport_retry, original_failed_labels_reclassified=False,
        all_populations_separate=True, human_gold=False)


def verify_pending_result(pending, chunk, ids, path):
    result = native.read_json(path)
    decisions_path = Path(result["adjudications_path"]).resolve()
    owner = Path(pending["bounded_output"]).resolve()
    if decisions_path.parent != owner or protocol.sha256(decisions_path.read_bytes()) != result["adjudications_sha256"]:
        raise ValueError("Completed pending chunk lost its unchanged original native ownership")
    packets = native.read_json(chunk / "packets.json")
    left, right = (legacy.full_adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b"))
    decisions = protocol.parse_jsonl(decisions_path.read_bytes())
    if [row["record_id"] for row in decisions] != ids or result["records"] != 20:
        raise ValueError("Pending whole chunk lost its fixed complete source order")
    groups = {}
    for row in decisions:
        receipt_path = Path(row["adjudication"]["execution_receipt_path"]).resolve()
        raw_path = Path(row["adjudication"]["raw_response_path"]).resolve()
        if (not receipt_path.is_relative_to(owner / "executions") or raw_path.parent != receipt_path.parent
                or raw_path.name != "raw_final.json"):
            raise ValueError("Pending decisions were moved across raw execution owners")
        groups.setdefault(receipt_path, []).append(row)
    for receipt_path, rows in groups.items():
        guards = {}
        receipt = previous.receipt_artifacts(receipt_path, guards)
        raw = native.read_json(receipt_path.parent / "raw_final.json")
        keys = [row["record_id"] for row in rows]
        if legacy.compact.retain_compact_decisions({key: raw[key] for key in keys}, packets, left, right, keys, receipt) != rows:
            raise ValueError("Pending whole chunk differs from its original actual compact raw response")
    if result["accepted"] != sum(row["status"] == "accepted" for row in decisions) or result["unresolved"] != sum(row["status"] == "unresolved" for row in decisions):
        raise ValueError("Pending whole-chunk outcome counts changed")
    return result


def configure(base, reused, stopped, pending, pending_base):
    _, previous_holder = previous.configure(base, reused, stopped)
    base.VERSION = VERSION
    holder = {}
    inherited_spec, inherited_export = base.specification, base.export_chunk
    inherited_verify, inherited_adjudicate = base.verify_retained, base.adjudicate_chunk
    inherited_progress = base.progress_snapshot

    def prepared(root, pool, run, chunk_size=20):
        context = prepare(root, pool, run, chunk_size)
        previous.merge_guards(context["protected"], reused["sha256"])
        previous.merge_guards(context["protected"], pending["protected_sha256"])
        for entry in list(reused["chunks"].values()) + [dict(source_chunk=pending["source_chunk"], record_ids=context["chunks"][28])]:
            chunk = Path(entry["source_chunk"])
            if native.read_json(chunk / "packets.json") != {key: context["packets"][key] for key in entry["record_ids"]}:
                raise ValueError("Body history rebound original canonical review packets")
            for side in ("a", "b"):
                if legacy.full_adjudicator.read_stream(chunk / f"independent_{side}.jsonl") != {
                        key: context["history_cache"]["records"][side][key] for key in entry["record_ids"]}:
                    raise ValueError("Body history rebound accepted original A/B source bytes")
        holder["context"] = context
        previous_holder["context"] = context
        return context

    def spec(context, chunk_size):
        result = inherited_spec(context, chunk_size)
        result.pop("source_transport_protocol", None)
        result.update(version=VERSION, body_source_protocol=PRODUCER_VERSION,
            body_corrective_protocol=CORRECTIVE_VERSION, body_corrective_records=context["history"]["body_corrective_records"],
            pending_original_owned_chunk=pending, body_history_decoded_once=True,
            body_original_first_attempt_role=FIRST_ROLE, future_annotation_role=FIRST_ROLE,
            historical_transport_source_protocol=previous.PRODUCER_VERSION, all_unknown_failed_usage_preserved=True)
        result["implementation_sha256"].update({str(Path(path).resolve()): protocol.sha256(Path(path).read_bytes())
            for path in (__file__, context["source_module"].__file__, context["source_module"].body.__file__)})
        return result

    def export(context, output, number, prefix, entries):
        if str(number) in reused["chunks"]:
            return inherited_export(context, output, number, prefix, entries)
        if number == 29:
            native.protected_unchanged(pending["protected_sha256"])
            return Path(pending["source_chunk"])
        selected = set(context["chunks"][number - 1])
        for _, event in entries:
            if event.get("type") == "completed" and event.get("accepted") is True and selected.intersection(event["record_ids"]):
                source = native.contained_file(context["run"], event["receipt"])
                receipt = native.read_json(source)
                if (Path(receipt.get("annotation_run", "")).resolve() != context["run"]
                        or receipt.get("execution_protocol") != PRODUCER_VERSION):
                    raise ValueError("Body source receipt is not owned by its registered actual producer")
        chunk = frozen.snapshot_chunk(context, output, number, prefix, entries)
        if chunk is not None:
            path = chunk / "body_export_receipt.json"
            if path.exists():
                receipt = native.read_json(path)
                if receipt.get("version") != VERSION or receipt.get("record_ids") != context["chunks"][number - 1]:
                    raise ValueError("Existing body source export protocol or fixed membership changed")
                for name, digest in receipt["artifacts_sha256"].items():
                    if protocol.sha256(native.contained_file(chunk, name).read_bytes()) != digest:
                        raise ValueError("Body native/corrective proof or agreement population changed")
                native.protected_unchanged(receipt["source_bindings"])
                frozen.bind_run_sources(context, receipt["source_bindings"])
            else:
                population = validate_roles(context, chunk)
                native.write_new(chunk / "body_agreement_population.json", population)
                guards = extra_source_proofs(context, chunk)
                native.write_new(path, dict(version=VERSION, record_ids=context["chunks"][number - 1], source_bindings=guards,
                    artifacts_sha256={item.relative_to(chunk).as_posix(): protocol.sha256(item.read_bytes()) for item in chunk.rglob("*") if item.is_file()}))
            if native.read_json(chunk / "body_agreement_population.json") != validate_roles(context, chunk):
                raise ValueError("Body source export population differs from unchanged original envelope roles")
            sidecar = native.read_json(path)
            guards = {str(path.resolve()): protocol.sha256(path.read_bytes())}
            guards.update({str(native.contained_file(chunk, name)): digest for name, digest in sidecar["artifacts_sha256"].items()})
            native.protected_unchanged(guards)
            frozen.bind_run_sources(context, guards)
        return chunk

    def verify(output, chunk, ids, result_path):
        if result_path.name != "chunk-0029.json":
            return inherited_verify(output, chunk, ids, result_path)
        bound = [row for row in protocol.parse_jsonl((output / "progress.jsonl").read_bytes())
                 if row.get("type") == "chunk_retained" and row.get("number") == 29]
        if len(bound) != 1 or bound[0]["record_ids"] != ids or bound[0]["result_sha256"] != protocol.sha256(result_path.read_bytes()):
            raise ValueError("Completed pending source chunk lacks its unique new-run progress binding")
        return verify_pending_result(pending, chunk, ids, result_path)

    def adjudicate(chunk, directory, **kwargs):
        context = holder["context"]
        if any((context["run"] / name).exists() for name in ("STOP_REQUESTED", "failure.json")):
            raise ValueError("Current body producer stop/failure marker requires owner review")
        frozen.verify_history_cache(context)
        native.protected_unchanged(context["protected"])
        if Path(chunk).resolve() == Path(pending["source_chunk"]).resolve():
            result = pending_base.adjudicate_chunk(chunk, Path(pending["output_parent"]), **kwargs)
        else:
            result = inherited_adjudicate(chunk, directory, **kwargs)
        frozen.verify_history_cache(context)
        native.protected_unchanged(context["protected"])
        return result

    def progress(run):
        run = Path(run).resolve()
        if any((run / name).exists() for name in ("STOP_REQUESTED", "failure.json")):
            raise ValueError("Current body producer stop/failure marker requires owner review")
        return inherited_progress(run)

    base.prepare, base.specification, base.export_chunk = prepared, spec, export
    base.verify_retained, base.adjudicate_chunk = verify, adjudicate
    base.progress_snapshot = progress
    return base


def run(root, pool_dir, annotation_run, *, previous_review_run, resume=None, export_only=False,
        max_idle_seconds=21600, caller=None):
    reused, stopped, pending, pending_base = prior_reviews(previous_review_run)
    base = configure(legacy.isolated_base(), reused, stopped, pending, pending_base)
    output, result = base.run(root, pool_dir, annotation_run, resume=resume, max_idle_seconds=max_idle_seconds,
                            adjudicate=not export_only, caller=caller)
    result = dict(result, version=VERSION, body_source_protocol=PRODUCER_VERSION,
        reused_exported_chunks=28, pending_original_owned_exported_chunks=1,
        total_exported_chunks=result["exported_chunks"] + 29,
        original_successful_native_calls=reused["successful_fresh_native_contexts"],
        original_historical_native_calls=reused["historical_native_contexts"],
        original_failed_native_calls=1, original_failed_usage=None, original_failed_usage_status="unknown_not_zero",
        pending_gate_native_calls_retained_once=1, pending_gate_records_not_added_to_complete_prefix=9)
    native.write_new(output / ("body-summary-" + uuid4().hex[:8] + ".json"), result)
    return output, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-root", "pool-dir", "annotation-run", "previous-review-run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--max-idle-seconds", type=int, default=21600)
    args = parser.parse_args()
    _, result = run(args.canonical_root, args.pool_dir, args.annotation_run, previous_review_run=args.previous_review_run,
        resume=args.resume, export_only=args.export_only, max_idle_seconds=args.max_idle_seconds)
    print(json.dumps(result, indent=2))
