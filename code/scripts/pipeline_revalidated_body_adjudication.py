"""Thin fixed-order continuation with two exact certificate exceptions.

No original module or receipt is modified. A private, byte-pinned snapshot bridge
changes only initial coverage to each stream's verified immutable cache. This
preserves the original six A-only records while the missing B context completes.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pipeline_body_quote_adjudication as body
import finish_revalidated_source_chunk as finish

native, protocol, recovery, legacy = body.native, body.protocol, body.previous, body.legacy
VERSION = "fixed-order-certificate-revalidated-body-source-v1"
PRODUCER_VERSION = "body-only-revalidated-continuation-v1"
PRODUCER_SHA256 = "48a182e837788f21e6d2057c5f0cb29563bd77b91226b3699da26a690ff2f569"
WIRE_SHA256 = "cd78ce3e4bdd104865b7cf1f18794fdd25e10c034fe7704ee45791c261660784"
RECOVERY_SHA256 = "0d3ed9f3a767ff79d0a23a97ee3f4a94e9e62b91f259e6b3430d74c5c99d3119"
FINISH_SHA256 = "43399565472366538fd1b972087292a9e0ee6ca70ea8d29a5323f6136748d0d4"
COMPLETION_SHA256 = "a3e8585da877db11d7a072d50e14dc51e721c595c9e187f8f5ad22277d7276e5"
CACHE_VERSION = "explicit-paired740-preserved-a-only6-cache-v1"


def clone_file(path, expected):
    path = Path(path).resolve()
    raw = path.read_bytes()
    if protocol.sha256(raw) != expected:
        raise ValueError("Pinned private source bridge dependency changed")
    module = ModuleType("revalidated_source_private_" + uuid4().hex)
    module.__file__, module.__package__ = str(path), ""
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def private_wire_bridge():
    module = clone_file(body.frozen.__file__, WIRE_SHA256)
    tree = ast.parse(Path(module.__file__).read_bytes(), filename=module.__file__)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "snapshot_chunk")
    old = ast.parse('{side: set(context["history"]["record_ids"]) for side in ("a", "b")}', mode="eval").body
    replacement = ast.parse('{side: set(cache["records"][side]) for side in ("a", "b")}', mode="eval").body
    matches = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 2
        and ast.dump(node.value.elts[1]) == ast.dump(old)]
    if len(matches) != 1:
        raise ValueError("The declared single initial-coverage bridge no longer matches frozen source")
    matches[0].value.elts[1] = replacement
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), module.__file__, "exec"), module.__dict__)
    module.verify_history_cache = verify_cache
    module.seal_history_cache = seal_cache
    return module


def producer_module():
    import run_revalidated_body_cohort as producer
    if producer.VERSION != PRODUCER_VERSION or protocol.sha256(Path(producer.__file__).read_bytes()) != PRODUCER_SHA256:
        raise ValueError("Unknown or changed revalidated body producer")
    return producer


def admitted_history(history):
    import completed_503_admission
    certificate, records, guards = completed_503_admission.load("annotation_b38", history["certificate_dir"])
    if len(records) != 10 or any(history["records"]["b"].get(key) != value for key, value in records.items()):
        raise ValueError("Only exact original B38 envelopes may use its mechanical certificate")
    selected = set()
    for side in history["sources"]:
        for key, value in history["sources"][side].items():
            if value.get("operational_receipt_accepted") is False:
                if (side != "b" or key not in records
                        or value.get("receipt_sha256") != "3d3eb1746772754f9098c17ed8698e511a6d3494745b6ddbd497258a63df915c"
                        or value.get("mechanical_certificate_sha256") != "71fdd5e5a5e0ab36b29771c1f19208887b89c672d2f7016f3f7cde3ba0fdc56c"
                        or Path(value["receipt"]).resolve() != Path(certificate["original_receipt"]).resolve()):
                    raise ValueError("A certificate cannot admit another false receipt or stream")
                selected.add(key)
    if selected != set(records):
        raise ValueError("The exact complete B38 certificate source lineage is required")
    return guards


def seal_cache(context, records, sources, *, counts=None):
    expected = {"a": 746, "b": 740} if counts is None else counts
    if set(expected) != {"a", "b"} or expected["a"] - expected["b"] != 6:
        raise ValueError("Only the declared six-record A-only asymmetry is supported")
    ids = context["ids"]
    for side in expected:
        if set(records[side]) != set(ids[:expected[side]]) or set(sources[side]) != set(records[side]):
            raise ValueError("Exact paired history and A-only original-source coverage changed")
    protected, seen = dict(context["protected"]), set()
    for side in sources:
        for value in sources[side].values():
            path = Path(value["receipt"]).resolve()
            recovery.guard_file(path, protected, value["receipt_sha256"])
            receipt = native.read_json(path)
            if receipt["execution_id"] != value["execution_id"]:
                raise ValueError("Cached original source execution identity changed")
            if path in seen:
                continue
            seen.add(path)
            for name, digest in receipt["artifacts_sha256"].items():
                recovery.guard_file(native.contained_file(path.parent, name), protected, digest)
    context["protected"].update(protected)
    cache = dict(version=CACHE_VERSION, history=deepcopy(context["history"]), record_ids=ids[:expected["b"]],
        per_stream_record_ids={side: ids[:count] for side, count in expected.items()},
        records=deepcopy(records), sources=deepcopy(sources), protected_sha256=protected,
        original_envelopes_unchanged=True, source_validation_completed_before_cache=True)
    context["history_cache"], context["history_cache_sha256"] = cache, protocol.sha256(protocol.canonical_bytes(cache))
    context["expected_cache_counts"] = dict(expected)
    return verify_cache(context)


def verify_cache(context):
    cache = context["history_cache"]
    expected = context["expected_cache_counts"]
    if (cache.get("version") != CACHE_VERSION or expected["a"] - expected["b"] != 6
            or protocol.sha256(protocol.canonical_bytes(cache)) != context["history_cache_sha256"]
            or cache["history"] != context["history"]
            or cache["record_ids"] != context["ids"][:expected["b"]]
            or any(cache["per_stream_record_ids"][side] != context["ids"][:count]
                   or set(cache["records"][side]) != set(context["ids"][:count])
                   or set(cache["sources"][side]) != set(cache["records"][side]) for side, count in expected.items())):
        raise ValueError("Declared asymmetric cached records, labels or source lineage changed")
    native.protected_unchanged(cache["protected_sha256"])
    if context.get("history_cache_artifact"):
        if protocol.sha256(Path(context["history_cache_artifact"]).read_bytes()) != context["history_cache_sha256"]:
            raise ValueError("Immutable asymmetric cache artifact changed")
    native.protected_unchanged(context.get("history_cache_receipt_guard", {}))
    return cache


def prepare(root, pool_dir, annotation_run, chunk_size=20):
    if type(chunk_size) is not int or chunk_size != 20:
        raise ValueError("Source continuation keeps fixed twenty-case chunks")
    root, pool, run = map(lambda value: Path(value).resolve(), (root, pool_dir, annotation_run))
    producer = producer_module()
    loaded, rows, questions, guards, bindings = recovery.frozen.wire.previous.load_pool(root, pool)
    if loaded != pool or not run.is_relative_to(pool / "cohort_annotations_body_revalidated"):
        raise ValueError("Revalidated source run escapes its registered pool")
    manifest, original_history, producer_guards = producer.verify_prepared(run)
    spec, ids = manifest["specification"], [row["record_id"] for row in rows]
    prompt = recovery.frozen.wire.batch.BATCH_PROMPT.encode()
    packets = {row["record_id"]: protocol.clean_packet(row, prompt=prompt.decode()) for row in rows}
    if (spec["version"] != PRODUCER_VERSION or spec["bindings"] != bindings or spec["eligible_record_ids"] != ids
            or len(ids) != 1200 or len(set(ids)) != 1200 or set(ids) != {row["record_id"] for row in questions}
            or packets != native.read_json(run / "packets.json") or spec["historical_counts"] != {"a": 746, "b": 740}
            or original_history["record_ids"] != {"a": ids[:746], "b": ids[:740]}
            or original_history["one_sided_record_ids"] != {"a": ids[740:746], "b": []}
            or spec["format_instruction_sha256"] != producer.previous.body.INSTRUCTION_SHA256
            or any(group["annotation_role"] != body.FIRST_ROLE or group["original_first_attempt"] is not True for group in spec["groups"])):
        raise ValueError("Full source order, original semantic bytes or exact A746/B740 history changed")
    recovery.merge_guards(guards, producer_guards)
    recovery.merge_guards(guards, original_history["protected_sha256"])
    recovery.merge_guards(guards, admitted_history(original_history))
    records, sources = original_history["records"], original_history["sources"]
    usage = original_history["execution_usage_records"]
    by_usage = {row["execution_id"]: row for row in usage}
    if len(by_usage) != len(usage) or any(value["execution_id"] not in by_usage for side in sources for value in sources[side].values()):
        raise ValueError("Every preserved source context needs a unique original usage entry")
    for side in sources:
        for value in sources[side].values():
            receipt = native.read_json(Path(value["receipt"]))
            if "usage" not in by_usage[value["execution_id"]] or by_usage[value["execution_id"]]["usage"] != receipt.get("usage"):
                raise ValueError("Cached native measured usage differs from its unique ledger entry")
    # The prepared producer cached exact native histories once; no cold loader.
    history = deepcopy(original_history)
    history["producer_record_ids"] = history["record_ids"]
    history["record_ids"] = ids[:740]
    question_map = {row["record_id"]: protocol.clean_question(row) for row in questions}
    context = dict(root=root, pool=pool, run=run, rows=rows, packets=packets, ids=ids,
        questions=[question_map[key] for key in ids], chunks=[ids[n:n + 20] for n in range(0, 1200, 20)],
        prompt=prompt, protected=guards, corpus=root / "data/unified_runs/semantic-page-v2-buffer/stage/passages.jsonl",
        bindings=bindings, history=history, source_spec=spec, source_module=source_dispatch(producer), corrected=False, clarified=False)
    seal_cache(context, records, sources)
    return context


def source_dispatch(producer):
    def verify(attempt, receipt, packets):
        if receipt.get("execution_protocol") != PRODUCER_VERSION:
            raise ValueError("Only registered new body contexts enter future native verification")
        values = producer.verify_attempt(attempt, receipt, packets)
        ledger = producer.previous.entries(Path(receipt["annotation_run"]) / "execution_usage.jsonl")
        selected = [row for row in ledger if row["execution_id"] == receipt["execution_id"]]
        if len(selected) != 1 or selected[0].get("usage") != receipt.get("usage"):
            raise ValueError("A durable future source context lacks its exact once-only measured usage")
        return values
    return SimpleNamespace(__file__=producer.__file__, VERSION=PRODUCER_VERSION, verify_attempt=verify)


def reuse_completed(review_run, completion_directory):
    """One explicit completion proof, not directory discovery or native replay."""
    review, completion_directory = Path(review_run).resolve(), Path(completion_directory).resolve()
    guards = {}
    recovery.guard_file(review / "manifest.json", guards, finish.ANCHORS["manifest"])
    recovery.guard_file(Path(finish.__file__), guards, FINISH_SHA256)
    completion_path = completion_directory / "completion.json"
    recovery.guard_file(completion_path, guards, COMPLETION_SHA256)
    intent = native.read_json(completion_directory / "intent.json")
    result = native.read_json(completion_path)
    sidecar = native.read_json(Path(result["certificate_admission_path"]))
    certificate_directory = Path(sidecar["certificate_directory"])
    checked = finish.run(review, certificate_directory, intent["proof"]["root_authorization"], completion_directory,
        caller=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Completed source reviews cannot invoke a model")))
    if checked != dict(result, new_native_calls=0) or result["records"] != 20 or result["retained_native_calls"] != 3:
        raise ValueError("Registered original-owned complete twenty-case proof changed")
    manifest = native.read_json(review / "manifest.json")
    reused, stopped = deepcopy(manifest["original_source_review_reuse"]), manifest["original_source_review_stop"]
    if reused["records"] != 560 or len(reused["chunks"]) != 28:
        raise ValueError("Only the exact original twenty-eight complete chunks may be reused")
    recovery.merge_guards(guards, reused["sha256"])
    recovery.merge_guards(guards, stopped["protected_sha256"])
    for path, digest in result["completion_artifact_sha256"].items():
        recovery.guard_file(path, guards, digest)
    for name in ("intent.json", "third_batch_binding.json"):
        recovery.guard_file(completion_directory / name, guards)
    recovery.merge_guards(guards, intent["protected_sha256"])
    recovery.merge_guards(guards, intent["original_owner_sha256"])
    chunk = review / "chunks/chunk-0029"
    reused["chunks"]["29"] = dict(source_chunk=str(chunk), record_ids=manifest["chunks"][28],
        result_path=str(completion_path), result=result)
    reused.update(records=580, sha256={**reused["sha256"], **guards},
        certified_original_false_source_calls=1, source_completion_native_calls_retained_once=3,
        source_certificate_admission_path=result["certificate_admission_path"])
    native.protected_unchanged(reused["sha256"])
    return reused, stopped


def configure(reused, stopped):
    wire = private_wire_bridge()
    private = clone_file(recovery.__file__, RECOVERY_SHA256)
    private.frozen, private.prepare, private.validate_export_roles = wire, prepare, body.validate_roles
    private.VERSION, private.PRODUCER_VERSION = VERSION, PRODUCER_VERSION
    base = legacy.isolated_base()
    base, holder = private.configure(base, reused, stopped)
    inherited_spec, inherited_create, inherited_export = base.specification, base.create_pipeline, base.export_chunk
    inherited_adjudicate = base.adjudicate_chunk

    def specification(context, size):
        value = inherited_spec(context, size)
        value.update(version=VERSION, history_coverage_bridge=CACHE_VERSION, frozen_snapshot_source_sha256=WIRE_SHA256,
            snapshot_change="Initial covered IDs are each side's verified immutable cache records only",
            historical_counts={"a": 746, "b": 740}, a_only_six_preserved=True,
            certified_source_admission_path=reused["source_certificate_admission_path"],
            native_acceptance_flags_never_rewritten=True)
        value["implementation_sha256"].update({str(Path(path).resolve()): protocol.sha256(Path(path).read_bytes())
            for path in (__file__, finish.__file__, context["source_module"].__file__)})
        return value

    def create(context, size=20, resume=None):
        output = inherited_create(context, size, resume)
        cache_path = output / "source_inputs/asymmetric_history_cache.json"
        cache_raw = protocol.canonical_bytes(context["history_cache"])
        if resume is None:
            native.write_new(cache_path, cache_raw)
        elif cache_path.read_bytes() != cache_raw:
            raise ValueError("Resumed asymmetric history cache changed")
        context["history_cache_artifact"] = cache_path
        return output

    def export(context, output, number, prefix, entries):
        if str(number) in reused["chunks"]:
            return inherited_export(context, output, number, prefix, entries)
        chunk = inherited_export(context, output, number, prefix, entries)
        if chunk is None:
            return None
        sidecar = chunk / "revalidated_source_receipt.json"
        if not sidecar.exists():
            # Existing whole-history copier retains false native bytes too.
            private_body = clone_file(body.__file__, "14bc5e33cee8bb3ce7a563977e17209726ac3a32ba9a78fdb19ade42d10bed86")
            private_body.frozen = wire
            guards = private_body.extra_source_proofs(context, chunk)
            lineage = native.read_json(chunk / "original_record_sources.json")
            false_sources = []
            for side in lineage:
                for value in lineage[side].values():
                    if value.get("operational_receipt_accepted") is False:
                        if value.get("mechanical_certificate_sha256") != "71fdd5e5a5e0ab36b29771c1f19208887b89c672d2f7016f3f7cde3ba0fdc56c":
                            raise ValueError("Unknown false source receipt cannot use the B38 certificate")
                        false_sources.append(dict(receipt=value["receipt"], certificate=value["mechanical_certificate"],
                            certificate_sha256=value["mechanical_certificate_sha256"], operational_receipt_accepted=False))
                        cert_path = Path(value["mechanical_certificate"]).resolve()
                        target = chunk / "completed_503_proof"
                        for name, digest in (("certificate.json", value["mechanical_certificate_sha256"]),
                            ("derived_records.json", "f2ee0fdf6513ae8f2e0607296a23ad18afad7a508abec6a5a8771e0861060c67")):
                            source = cert_path.parent / name
                            recovery.guard_file(source, guards, digest)
                            if (target / name).exists():
                                if (target / name).read_bytes() != source.read_bytes():
                                    raise ValueError("Copied exact B38 mechanical proof changed")
                            else:
                                native.write_new(target / name, source.read_bytes())
            native.write_new(sidecar, dict(version=VERSION, record_ids=context["chunks"][number - 1],
                source_bindings=guards, false_native_sources=false_sources, native_flags_unchanged=True,
                artifacts_sha256={path.relative_to(chunk).as_posix(): protocol.sha256(path.read_bytes())
                    for path in chunk.rglob("*") if path.is_file()}))
        value = native.read_json(sidecar)
        if value.get("version") != VERSION or value.get("record_ids") != context["chunks"][number - 1]:
            raise ValueError("Revalidated source chunk protocol or fixed membership changed")
        guards = {str(sidecar): protocol.sha256(sidecar.read_bytes())}
        recovery.merge_guards(guards, value["source_bindings"])
        for name, digest in value["artifacts_sha256"].items():
            recovery.guard_file(native.contained_file(chunk, name), guards, digest)
        wire.bind_run_sources(context, guards)
        native.protected_unchanged(guards)
        return chunk

    inherited_progress = base.progress_snapshot
    def progress(run):
        if any((Path(run) / name).exists() for name in ("STOP_REQUESTED", "failure.json")):
            raise ValueError("New producer stop or failure requires owner review")
        return inherited_progress(run)
    def adjudicate(chunk, directory, **kwargs):
        context = holder["context"]
        if any((context["run"] / name).exists() for name in ("STOP_REQUESTED", "failure.json")):
            raise ValueError("New producer stop or failure requires owner review")
        result = inherited_adjudicate(chunk, directory, **kwargs)
        if any((context["run"] / name).exists() for name in ("STOP_REQUESTED", "failure.json")):
            raise ValueError("New producer stopped during review; all attempted outputs remain retained")
        return result
    base.specification, base.create_pipeline, base.export_chunk, base.progress_snapshot = specification, create, export, progress
    base.adjudicate_chunk = adjudicate
    return base, holder


def run(root, pool_dir, annotation_run, *, previous_review_run, source_completion_directory,
        resume=None, export_only=False, max_idle_seconds=21600, caller=None):
    reused, stopped = reuse_completed(previous_review_run, source_completion_directory)
    base, _ = configure(reused, stopped)
    output, result = base.run(root, pool_dir, annotation_run, resume=resume, max_idle_seconds=max_idle_seconds,
        adjudicate=not export_only, caller=caller)
    result.update(version=VERSION, reused_complete_chunks=29, reused_complete_records=580,
        total_exported_chunks=result["exported_chunks"] + 29,
        original_successful_source_native_calls=reused["successful_fresh_native_contexts"],
        original_historical_source_native_calls=reused["historical_native_contexts"],
        original_unknown_failed_source_native_calls=1, original_unknown_failed_source_usage=None,
        certified_original_false_source_calls=1, source_completion_native_calls_retained_once=3,
        subscription_dollar_cost=None, native_receipt_flags_unchanged=True)
    native.write_new(output / ("revalidated-source-summary-" + uuid4().hex[:8] + ".json"), result)
    return output, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-root", "pool-dir", "annotation-run", "previous-review-run", "source-completion-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--max-idle-seconds", type=int, default=21600)
    args = parser.parse_args()
    _, summary = run(args.canonical_root, args.pool_dir, args.annotation_run, previous_review_run=args.previous_review_run,
        source_completion_directory=args.source_completion_directory, resume=args.resume,
        export_only=args.export_only, max_idle_seconds=args.max_idle_seconds)
    print(protocol.canonical_bytes(summary).decode())
