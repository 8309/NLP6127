"""Versioned compact pipeline with byte-exact historical full-review reuse.

The original pipeline and all native/core helpers remain unchanged. Isolated
module instances reuse its durable export/bounding checks with a declared compact
codec; accepted full-response historical chunks retain their original raw paths.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
from uuid import uuid4

import adjudicate_machine_annotations as full_adjudicator
import compact_machine_adjudication as compact
import run_codex_annotation as native
from oil_evidence import machine_evaluation as protocol

VERSION = "declared-historical-compact-source-pipeline-v4"


def isolated_base():
    path = Path(__file__).with_name("pipeline_cohort_adjudication.py")
    spec = importlib.util.spec_from_file_location("compact_pipeline_base_" + uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HistoricalGuards:
    """Archived code proves historical hashes; live execution hashes stay separate."""
    def __init__(self, root, directory):
        self.directory = Path(directory).resolve()
        allowed = Path(root).resolve() / "data/machine_runs/guard-snapshots"
        if not self.directory.is_relative_to(allowed) or self.directory == allowed:
            raise ValueError("Historical guard snapshot escapes its declared private root")
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = native.read_json(self.manifest_path)
        if self.manifest.get("copy_bytes_verified_against_original_at_capture") is not True:
            raise ValueError("Historical guard snapshot lacks its byte-equality capture proof")
        self.paths, self.hashes = {}, {}
        self.bindings = {str(self.manifest_path): protocol.sha256(self.manifest_path.read_bytes())}
        public_root = Path(__file__).parents[1].resolve()
        for name, expected in self.manifest["artifacts_sha256"].items():
            source = Path(self.manifest["original_source_paths"][name]).resolve()
            if not source.is_relative_to(public_root):
                raise ValueError("Historical implementation path escapes the public project")
            path = native.contained_file(self.directory, name)
            if protocol.sha256(path.read_bytes()) != expected:
                raise ValueError("Historical implementation snapshot bytes changed")
            self.paths[str(source)], self.hashes[str(source)] = path, expected
            self.bindings[str(path)] = expected
        # Only already-declared changed code can use archived proof. Unchanged
        # live helpers remain guarded against edits during a native invocation.
        self.changed_at_start = {source for source, expected in self.hashes.items()
            if not Path(source).is_file() or protocol.sha256(Path(source).read_bytes()) != expected}
        source = str(Path(protocol.__file__).resolve())
        if source not in self.paths:
            raise ValueError("Historical snapshot does not contain the required original label guard")
        self.protocol = ModuleType("historical_machine_guard_" + uuid4().hex)
        self.protocol.__file__ = source
        self.protocol.__package__ = "oil_evidence"
        exec(compile(self.paths[source].read_bytes(), source, "exec"), self.protocol.__dict__)

    def check_inputs(self, values):
        native.protected_unchanged(self.bindings)
        live = {}
        for source, expected in values.items():
            normalized = str(Path(source).resolve())
            if normalized in self.changed_at_start and self.hashes.get(normalized) == expected:
                if protocol.sha256(self.paths[normalized].read_bytes()) != expected:
                    raise ValueError("Historical code proof changed")
            else:
                live[source] = expected
        native.protected_unchanged(live)

    def install(self, base, historical=False):
        """Use private module instances; never mutate the core/native module."""
        proxy = SimpleNamespace(**{name: value for name, value in native.__dict__.items() if not name.startswith("__")})
        proxy.__file__ = native.__file__
        proxy.protected_unchanged = self.check_inputs
        def clone(module):
            value = ModuleType("historical_binding_module_" + uuid4().hex)
            value.__file__ = module.__file__
            value.__package__ = module.__package__
            source = str(Path(module.__file__).resolve())
            raw = self.paths[source].read_bytes() if historical and source in self.paths else Path(source).read_bytes()
            exec(compile(raw, source, "exec"), value.__dict__)
            value.native = proxy
            value.protocol = self.protocol if historical else protocol
            return value
        base.native = proxy
        base.cohort = clone(base.cohort)
        base.throughput = clone(base.throughput)
        base.throughput.previous = base.cohort
        base.protocol = self.protocol if historical else protocol
        if historical:
            base.adjudicator = clone(base.adjudicator)
            original_specification = base.specification
            def specification(context, chunk_size):
                spec = original_specification(context, chunk_size)
                spec["implementation_sha256"] = {source: self.hashes.get(str(Path(source).resolve()), expected)
                                                 for source, expected in spec["implementation_sha256"].items()}
                return spec
            base.specification = specification
        return base

    def receipt(self):
        return dict(directory=str(self.directory), manifest_sha256=self.bindings[str(self.manifest_path)],
                    artifacts_sha256=self.bindings, historical_code_hash_maps_preserved=True,
                    changed_code_verified_against_archived_bytes=sorted(self.changed_at_start),
                    current_execution_code_bound_separately=True)


def bases(root, historical_guard_snapshot=None):
    old_base, base = isolated_base(), isolated_base()
    guards = HistoricalGuards(root, historical_guard_snapshot) if historical_guard_snapshot else None
    if guards:
        old_base, base = guards.install(old_base, historical=True), guards.install(base)
    return old_base, base, guards


def verify_previous(context, previous_run, old_base):
    if previous_run is None:
        return dict(output=None, records=0, chunks={}, sha256={})
    previous_run = Path(previous_run).resolve()
    allowed = context["root"] / "data/machine_runs/cohort_pipeline"
    if not previous_run.is_relative_to(allowed) or previous_run == allowed:
        raise ValueError("Previous full-response pipeline escapes its private output root")
    if native.read_json(previous_run / "manifest.json") != old_base.specification(context, 20):
        raise ValueError("Previous full-response source, protocol or implementation binding differs")
    if (previous_run / "failure.json").exists():
        raise ValueError("Failed previous pipeline requires explicit owner recovery")
    stop = native.read_json(previous_run / "controlled_stop_receipt.json")
    if stop.get("status") not in {"owned_parent_stopped_after_two_durable_full_response_chunks",
                                  "owned_parent_stopped_after_durable_full_response_chunks"} or stop.get("all_original_attempts_preserved") is not True:
        raise ValueError("Previous full-response run lacks its controlled-stop audit")
    progress = previous_run / "progress.jsonl"
    if protocol.sha256(progress.read_bytes()) != stop["progress_sha256"]:
        raise ValueError("Previous stopped pipeline progress changed")
    prefix, entries = old_base.progress_snapshot(context["run"])
    chunks, bindings = {}, {}
    result_files = sorted((previous_run / "retained").glob("chunk-*.json"))
    for number, result_path in enumerate(result_files, 1):
        if result_path.name != f"chunk-{number:04d}.json":
            raise ValueError("Previous reusable full-response chunks are not a fixed source prefix")
        chunk = previous_run / "chunks" / f"chunk-{number:04d}"
        if not (chunk / "export_receipt.json").exists():
            raise ValueError("Previous full-response chunk lacks its immutable source export")
        verified_chunk = old_base.export_chunk(context, previous_run, number, prefix, entries)
        result = old_base.verify_retained(previous_run, verified_chunk, context["chunks"][number - 1], result_path)
        packets = native.read_json(chunk / "packets.json")
        if packets != {key: context["packets"][key] for key in context["chunks"][number - 1]}:
            raise ValueError("Previous full-response packets differ from current original inputs")
        chunks[str(number)] = dict(result_path=str(result_path), source_chunk=str(chunk), result=result,
                              record_ids=context["chunks"][number - 1])
        for parent in (chunk, Path(result["output"])):
            for path in parent.rglob("*"):
                if path.is_file():
                    bindings[str(path)] = protocol.sha256(path.read_bytes())
        bindings[str(result_path)] = protocol.sha256(result_path.read_bytes())
    records = sum(entry["result"]["records"] for entry in chunks.values())
    if records != stop["retained_records"] or not chunks:
        raise ValueError("Previous controlled stop and retained complete coverage differ")
    for name in ("manifest.json", "progress.jsonl", "controlled_stop_receipt.json"):
        path = previous_run / name
        bindings[str(path)] = protocol.sha256(path.read_bytes())
    native.protected_unchanged(bindings)
    return dict(output=str(previous_run), records=records, chunks=chunks, sha256=bindings)


def verify_partial(context, directory):
    if directory is None:
        return None
    directory = Path(directory).resolve()
    allowed = context["root"] / "data/machine_runs/cohort_pipeline"
    if not directory.is_relative_to(allowed) or directory == allowed:
        raise ValueError("Partial compact context escapes its declared private root")
    manifest = native.read_json(directory / "manifest.json")
    failure = native.read_json(directory / "outer_failure_receipt.json")
    archived = directory / failure["historical_driver_snapshot"]
    if (manifest.get("source_annotation_run") != str(context["run"])
            or manifest.get("source_order_record_ids") != context["ids"]
            or manifest.get("raw_encoding") != compact.VERSION
            or failure.get("status") != "native_compact_development_pilot_accepted_outer_final_audit_failed"
            or protocol.sha256(archived.read_bytes()) != failure["historical_driver_sha256"]
            or manifest["implementation_sha256"].get(str(Path(__file__).resolve())) != failure["historical_driver_sha256"]):
        raise ValueError("Partial compact historical driver/outcome/source binding differs")
    parents = sorted((directory / "adjudications").glob("chunk-*/bounded-run"))
    if len(parents) != 1:
        raise ValueError("Expected one partially completed compact source chunk")
    parent = parents[0]
    number = int(parent.parent.name.split("-")[-1])
    chunk = directory / "chunks" / f"chunk-{number:04d}"
    if native.read_json(chunk / "packets.json") != {key: context["packets"][key] for key in context["chunks"][number - 1]}:
        raise ValueError("Partial compact packets differ from original source evidence")
    batches = sorted((parent / "retained_batches").glob("batch-*.json"))
    if len(batches) != 1:
        raise ValueError("Pilot recovery must retain exactly one existing native batch")
    binding = native.read_json(batches[0])
    receipt_path = native.contained_file(parent, binding["receipt_path"])
    if protocol.sha256(receipt_path.read_bytes()) != binding["receipt_sha256"]:
        raise ValueError("Partial compact native receipt changed")
    receipt = native.read_json(receipt_path)
    packets = native.read_json(chunk / "packets.json")
    left, right = (full_adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b"))
    raw = native.read_json(receipt_path.parent / "raw_final.json")
    rebuilt = compact.retain_compact_decisions(raw, packets, left, right, binding["record_ids"], receipt)
    if rebuilt != binding["decisions"] or len(rebuilt) != failure["native_records"]:
        raise ValueError("Partial compact normalized decisions differ from actual raw source selection")
    protected = {str(path): protocol.sha256(path.read_bytes()) for path in directory.rglob("*") if path.is_file()}
    return dict(output=str(directory), number=number, source_chunk=str(chunk), parent=str(parent.parent),
                bounded_version=native.read_json(parent / "inputs.json")["version"],
                pilot_records=len(rebuilt), pilot_record_ids=binding["record_ids"], sha256=protected)


def configure(base, old_base, previous, guards=None, partial=None):
    base.VERSION = VERSION
    base.adjudicator = SimpleNamespace(__file__=compact.__file__, SYSTEM=compact.SYSTEM,
        prepare_batch=compact.prepare_batch, retain_decisions=compact.retain_compact_decisions,
        validate_pair=full_adjudicator.validate_pair, read_stream=full_adjudicator.read_stream)
    original_specification, original_create, original_verify = base.specification, base.create_pipeline, base.verify_retained
    original_adjudicate = base.adjudicate_chunk

    def specification(context, chunk_size):
        value = original_specification(context, chunk_size)
        value["implementation_sha256"].update({str(Path(path).resolve()): protocol.sha256(Path(path).read_bytes())
            for path in (__file__, compact.__file__, full_adjudicator.__file__)})
        value.update(raw_encoding=compact.VERSION, full_source_and_both_original_labels_visible=True,
                     compact_selection_not_automatic_consensus=True, previous_full_response=previous)
        if guards:
            value["historical_guard_snapshot"] = guards.receipt()
        if partial:
            value["partial_compact_recovery"] = partial
        return value

    def create(context, chunk_size=20, resume=None):
        output = original_create(context, chunk_size, resume)
        if resume is None:
            prefix, entries = base.progress_snapshot(context["run"])
            for key, entry in previous["chunks"].items():
                number = int(key)
                chunk = base.export_chunk(context, output, number, prefix, entries)
                if chunk is None:
                    raise ValueError("Reusable full-response chunk no longer has paired durable originals")
                for side in ("a", "b"):
                    original_rows = full_adjudicator.read_stream(Path(entry["source_chunk"]) / f"independent_{side}.jsonl")
                    new_rows = full_adjudicator.read_stream(chunk / f"independent_{side}.jsonl")
                    if original_rows != new_rows:
                        raise ValueError("Previous whole raw original envelopes were rebound during reuse")
                path = output / "retained" / f"chunk-{number:04d}.json"
                native.write_new(path, Path(entry["result_path"]).read_bytes())
                native.append_progress(output / "progress.jsonl", dict(type="chunk_retained", number=number,
                    record_ids=entry["record_ids"], result_path=str(path), result_sha256=protocol.sha256(path.read_bytes()),
                    previous_full_response_run=previous["output"], native_calls_repeated=0))
            native.write_new(output / "previous_full_response_reuse.json", previous)
        native.protected_unchanged(previous["sha256"])
        return output

    def verify(output, chunk, ids, result_path):
        number = int(result_path.stem.split("-")[-1])
        if partial and number == partial["number"]:
            progress = protocol.parse_jsonl((output / "progress.jsonl").read_bytes())
            bound = [row for row in progress if row.get("type") == "chunk_retained" and row.get("number") == number]
            if len(bound) != 1 or bound[0]["result_sha256"] != protocol.sha256(result_path.read_bytes()):
                raise ValueError("Recovered compact result lacks its new-run progress binding")
            result = native.read_json(result_path)
            path = Path(result["adjudications_path"]).resolve()
            expected_root = Path(partial["parent"]) / "bounded-run"
            if not path.is_relative_to(expected_root) or protocol.sha256(path.read_bytes()) != result["adjudications_sha256"]:
                raise ValueError("Recovered compact result escapes its declared original context")
            decisions = protocol.parse_jsonl(path.read_bytes())
            if [row["record_id"] for row in decisions] != ids:
                raise ValueError("Recovered compact result lost original source order/coverage")
            packets = native.read_json(chunk / "packets.json")
            left, right = (full_adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b"))
            for row in decisions:
                receipt_path = Path(row["adjudication"]["execution_receipt_path"]).resolve()
                if not receipt_path.is_relative_to(expected_root / "executions"):
                    raise ValueError("Recovered compact native context escapes its declared source run")
                receipt = native.read_json(receipt_path)
                raw = native.read_json(receipt_path.parent / "raw_final.json")
                rebuilt = compact.retain_compact_decisions({row["record_id"]: raw[row["record_id"]]}, packets, left, right, [row["record_id"]], receipt)[0]
                if rebuilt != row:
                    raise ValueError("Recovered compact decision differs from its unchanged actual raw bytes")
            native.protected_unchanged(partial["sha256"])
            return result
        if str(number) not in previous["chunks"]:
            return original_verify(output, chunk, ids, result_path)
        entry = previous["chunks"][str(number)]
        if result_path.read_bytes() != Path(entry["result_path"]).read_bytes() or ids != entry["record_ids"]:
            raise ValueError("Previous accepted full-response result was changed or reordered")
        progress = protocol.parse_jsonl((output / "progress.jsonl").read_bytes())
        bindings = [row for row in progress if row.get("type") == "chunk_retained" and row.get("number") == number]
        if (len(bindings) != 1 or bindings[0]["result_sha256"] != protocol.sha256(result_path.read_bytes())
                or bindings[0].get("previous_full_response_run") != previous["output"]):
            raise ValueError("Reusable full-response result lost its declared original provenance")
        native.protected_unchanged(previous["sha256"])
        return old_base.verify_retained(Path(previous["output"]), Path(entry["source_chunk"]), ids, Path(entry["result_path"]))

    def adjudicate(chunk, parent, caller=None):
        number = int(chunk.name.split("-")[-1])
        if not partial or number != partial["number"]:
            return original_adjudicate(chunk, parent, caller=caller)
        original_chunk = Path(partial["source_chunk"])
        for name in ("packets.json", "independent_a.jsonl", "independent_b.jsonl", "grouping_metadata.json"):
            differs = (native.read_json(chunk / name) != native.read_json(original_chunk / name)) if name.endswith(".json") else (
                (chunk / name).read_bytes() != (original_chunk / name).read_bytes())
            if differs:
                raise ValueError("Partial compact original input bytes were rebound in new run")
        native.protected_unchanged(partial["sha256"])
        version = base.VERSION
        try:
            # Continue the original fully bound source-chunk context. Its input
            # manifest/raw batch remains unchanged; new context attempts append.
            base.VERSION = partial["bounded_version"]
            return original_adjudicate(original_chunk, Path(partial["parent"]), caller=caller)
        finally:
            base.VERSION = version

    base.specification, base.create_pipeline, base.verify_retained, base.adjudicate_chunk = specification, create, verify, adjudicate
    return base


def run(root, pool_dir, annotation_run, *, previous_full_response_run=None, resume=None,
        max_idle_seconds=600, poll_seconds=5, adjudicate=True, caller=None, historical_guard_snapshot=None,
        partial_compact_run=None):
    old_base, base, guards = bases(root, historical_guard_snapshot)
    context = base.prepare(root, pool_dir, annotation_run, 20)
    previous = verify_previous(context, previous_full_response_run, old_base)
    if resume and partial_compact_run:
        partial = native.read_json(Path(resume) / "manifest.json").get("partial_compact_recovery")
        if not partial or partial["output"] != str(Path(partial_compact_run).resolve()):
            raise ValueError("Recovered compact resume differs from its immutable original pilot plan")
        native.protected_unchanged(partial["sha256"])
        if native.read_json(Path(partial["source_chunk"]) / "packets.json") != {
                key: context["packets"][key] for key in context["chunks"][partial["number"] - 1]}:
            raise ValueError("Recovered compact source packets changed across resume")
    else:
        partial = verify_partial(context, partial_compact_run)
    base = configure(base, old_base, previous, guards, partial)
    output, result = base.run(root, pool_dir, annotation_run, chunk_size=20, resume=resume,
        max_idle_seconds=max_idle_seconds, poll_seconds=poll_seconds, adjudicate=adjudicate, caller=caller)
    native.protected_unchanged(previous["sha256"])
    if partial:
        native.protected_unchanged(partial["sha256"])
    return output, result


class PilotBoundary(RuntimeError):
    """Declared local stop before a second native call; not an execution retry."""


def run_pilot(root, pool_dir, annotation_run, *, previous_full_response_run=None, caller=None, historical_guard_snapshot=None):
    """Seal one development context within the exact future full-chunk plan."""
    old_base, base, guards = bases(root, historical_guard_snapshot)
    context = base.prepare(root, pool_dir, annotation_run, 20)
    previous = verify_previous(context, previous_full_response_run, old_base)
    base = configure(base, old_base, previous, guards)
    output = base.create_pipeline(context, 20)
    number = len(previous["chunks"]) + 1
    prefix, entries = base.progress_snapshot(context["run"])
    chunk = base.export_chunk(context, output, number, prefix, entries)
    if chunk is None:
        raise ValueError("Next fixed source chunk lacks complete paired durable inputs for its pilot")
    packets = native.read_json(chunk / "packets.json")
    left, right = (full_adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b"))
    splits = native.read_json(chunk / "grouping_metadata.json")
    groups = base.adjudication_groups(packets, left, right, list(packets), splits=splits)
    if len(groups[0]) > 10 or any(splits[key] != "development" for key in groups[0]):
        raise ValueError("The first compact pilot must be at most ten whole development cases in source order")
    calls = 0
    from oil_evidence.codex_runtime import call_structured_json
    backend = caller or call_structured_json
    def one_call(*args, **kwargs):
        nonlocal calls
        if calls:
            raise PilotBoundary("Declared compact pilot boundary reached before a second native invocation")
        calls += 1
        return backend(*args, **kwargs)
    parent = output / "adjudications" / f"chunk-{number:04d}"
    completed_result = None
    try:
        completed_result = base.adjudicate_chunk(chunk, parent, caller=one_call)
    except PilotBoundary:
        pass
    if completed_result is not None:
        result_path = output / "retained" / f"chunk-{number:04d}.json"
        native.write_new(result_path, completed_result)
        native.append_progress(output / "progress.jsonl", dict(type="chunk_retained", number=number,
            record_ids=list(packets), result_path=str(result_path), result_sha256=protocol.sha256(result_path.read_bytes())))
    binding_path = parent / "bounded-run/retained_batches/batch-0001.json"
    binding = native.read_json(binding_path)
    if binding["record_ids"] != groups[0] or len(binding["decisions"]) != len(groups[0]):
        raise ValueError("Compact pilot differs from its exact future full-chunk grouping")
    result = dict(output=str(output), status="one_compact_development_context_retained_not_full_cohort_complete",
        pilot_records=len(groups[0]), record_ids=groups[0], native_calls=calls,
        accepted=sum(row["status"] == "accepted" for row in binding["decisions"]),
        unresolved=sum(row["status"] == "unresolved" for row in binding["decisions"]),
        pilot_binding_path=str(binding_path), pilot_binding_sha256=protocol.sha256(binding_path.read_bytes()),
        previous_full_response_records_reused=previous["records"], expected_records=len(context["ids"]),
        raw_encoding=compact.VERSION, human_gold=False, formal_dataset_frozen=False,
        resume_policy="Same full fixed-chunk inputs and groups; completed first native batch is revalidated, never rerun")
    base.native.protected_unchanged(context["protected"])
    native.protected_unchanged(previous["sha256"])
    native.write_new(output / ("pilot-summary-" + uuid4().hex[:8] + ".json"), result)
    return output, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("canonical-root", "pool-dir", "annotation-run"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--previous-full-response-run", type=Path)
    parser.add_argument("--historical-guard-snapshot", type=Path)
    parser.add_argument("--partial-compact-run", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-idle-seconds", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    if args.pilot_only:
        if args.resume or args.export_only or args.partial_compact_run:
            parser.error("Pilot-only creates one new declared run; use ordinary resume afterward")
        _, result = run_pilot(args.canonical_root, args.pool_dir, args.annotation_run,
            previous_full_response_run=args.previous_full_response_run, historical_guard_snapshot=args.historical_guard_snapshot)
    else:
        _, result = run(args.canonical_root, args.pool_dir, args.annotation_run,
            previous_full_response_run=args.previous_full_response_run, resume=args.resume,
            max_idle_seconds=args.max_idle_seconds, poll_seconds=args.poll_seconds, adjudicate=not args.export_only,
            historical_guard_snapshot=args.historical_guard_snapshot, partial_compact_run=args.partial_compact_run)
    print(json.dumps(result, indent=2))
