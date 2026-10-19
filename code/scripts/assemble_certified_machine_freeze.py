"""One scoped source-certificate bridge around the unchanged strict assembler."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import completed_503_admission as admission
import compact_machine_adjudication as compact_original
import run_codex_annotation as native
from oil_evidence import machine_evaluation as protocol

VERSION = "scoped-source-503-certificate-aware-assembly-v1"
ASSEMBLER_SHA256 = "925eb722f190e806055276f196b116b5ff8d269c5e318060da3e53acecb1f88b"
COMPACT_SHA256 = "1fbca15ec78e885c3800fb09e31a6546f2d5286d76aac3bab8a8bf16ae99b91d"
PINS = dict(decisions="b721fda5a6557e0298a8183de5db08299448765af001d10347ea4679f06d1ad9",
    sidecar="b65d57b682990d464063f20ee9e8678fc617ae5eee036128391e7c43049be501",
    completion="a3e8585da877db11d7a072d50e14dc51e721c595c9e187f8f5ad22277d7276e5")


def guard(path, expected, guards):
    path = Path(path).resolve()
    if str(path) in guards and guards[str(path)] != expected:
        raise ValueError("A sealed source digest cannot be overwritten by a fresh digest")
    if protocol.sha256(path.read_bytes()) != expected:
        raise ValueError("Scoped source certificate bridge input changed")
    guards[str(path)] = expected
    return path


def scoped_context(decisions_path, certificate_directory, completion_path):
    certificate, derived, guards = admission.load("source_group2", certificate_directory)
    decisions_path = guard(decisions_path, PINS["decisions"], guards)
    completion_path = guard(completion_path, PINS["completion"], guards)
    completion = native.read_json(completion_path)
    sidecar_path = guard(completion["certificate_admission_path"], PINS["sidecar"], guards)
    sidecar = native.read_json(sidecar_path)
    if (Path(completion["adjudications_path"]).resolve() != decisions_path or completion.get("complete_chunk") is not True
            or completion.get("records") != 20 or completion.get("original_second_receipt_accepted") is not False
            or completion["adjudications_sha256"] != PINS["decisions"]
            or Path(sidecar["adjudications_path"]).resolve() != decisions_path
            or Path(sidecar["certificate_directory"]).resolve() != Path(certificate_directory).resolve()
            or sidecar.get("operational_receipt_accepted") is not False
            or sidecar.get("actual_argv_recorded") is not False or sidecar.get("actual_command_bytes_unavailable") is not True):
        raise ValueError("Only the exact completed original-owned twenty-row chunk may bridge")
    for path, digest in completion["completion_artifact_sha256"].items():
        guard(path, digest, guards)
    receipt_path = Path(certificate["original_receipt"]).resolve()
    receipt = native.read_json(receipt_path)
    rows = {row["record_id"]: row for row in derived["decisions"]}
    all_rows = {row["record_id"]: row for row in protocol.parse_jsonl(decisions_path.read_bytes())}
    if (len(rows) != 9 or list(rows) != sidecar["certified_record_ids"] or sidecar["original_native_receipt"] != str(receipt_path)
            or receipt.get("accepted") is not False or any(all_rows.get(key) != value for key, value in rows.items())
            or sidecar["certificate_sha256"] != protocol.sha256((Path(certificate_directory) / "certificate.json").read_bytes())
            or sidecar["derived_records_sha256"] != certificate["derived_records_sha256"]
            or decisions_path.parent != receipt_path.parent.parent.parent
            or not receipt_path.is_relative_to(decisions_path.parent / "executions")):
        raise ValueError("Certified rows or original false-receipt execution ownership changed")
    native.protected_unchanged(guards)
    return dict(decisions_path=decisions_path, receipt_path=receipt_path, receipt=receipt,
        raw_path=receipt_path.parent / "raw_final.json", raw=(receipt_path.parent / "raw_final.json").read_bytes(),
        rows=rows, certificate=certificate, sidecar_path=sidecar_path, completion_path=completion_path,
        certificate_directory=Path(certificate_directory).resolve(), guards=guards)


def certified_retained(context, receipt, raw, record_ids):
    native.protected_unchanged(context["guards"])
    parsed = native.read_json(context["raw_path"])
    if (receipt != context["receipt"] or native.read_json(context["receipt_path"]) != receipt
            or receipt.get("accepted") is not False or not record_ids or len(record_ids) != len(set(record_ids))
            or any(key not in context["rows"] or raw.get(key) != parsed.get(key) for key in record_ids)
            or set(raw) != set(record_ids) or context["raw_path"].read_bytes() != context["raw"]):
        raise ValueError("Only exact retained certified source selections may reference the false receipt")
    return context["raw_path"], context["receipt_path"]


def private_assembler(context):
    scripts = Path(__file__).parent
    assembler_path = guard(scripts / "assemble_machine_freeze.py", ASSEMBLER_SHA256, context["guards"])
    compact_path = guard(scripts / "compact_machine_adjudication.py", COMPACT_SHA256, context["guards"])
    compact = ModuleType("scoped_compact_" + uuid4().hex)
    compact.__file__ = str(compact_path)
    exec(compile(compact_path.read_bytes(), str(compact_path), "exec"), compact.__dict__)
    original_retained = compact._retained_raw
    compact._retained_raw = lambda receipt, raw, ids: (original_retained(receipt, raw, ids)
        if receipt.get("accepted") is True else certified_retained(context, receipt, raw, ids))
    raw_source = assembler_path.read_text(encoding="utf-8")
    predicate = 'receipt.get("accepted") is not True'
    replacement = 'not _native_admitted(receipt, raw_output, review, identifier, path, receipt_path, packets, left, right)'
    if raw_source.count(predicate) != 1:
        raise ValueError("The frozen assembler admission boundary changed")
    assembler = ModuleType("scoped_assembler_" + uuid4().hex)
    assembler.__file__ = str(assembler_path)
    exec(compile(raw_source.replace(predicate, replacement), str(assembler_path), "exec"), assembler.__dict__)
    assembler.compact = compact
    def admitted(receipt, raw, review, identifier, path, receipt_path, packets, left, right):
        if receipt.get("accepted") is True:
            return True
        if (Path(path).resolve() != context["decisions_path"] or Path(receipt_path).resolve() != context["receipt_path"]
                or identifier not in context["rows"] or receipt != context["receipt"] or raw != context["raw"]
                or review != context["rows"][identifier]["adjudication"]):
            raise ValueError("An unknown false native receipt cannot use the scoped certificate")
        native.protected_unchanged(context["guards"])
        if context["raw_path"].read_bytes() != context["raw"]:
            raise ValueError("Certified retained raw output changed")
        parsed = native.read_json(context["raw_path"])
        rebuilt = compact.retain_compact_decisions({identifier: parsed[identifier]}, packets, left, right, [identifier], receipt)[0]
        if protocol.canonical_bytes(rebuilt) != protocol.canonical_bytes(context["rows"][identifier]):
            raise ValueError("Certified selected label/full source/pair hashes changed")
        return True
    assembler._native_admitted = admitted
    original_write = assembler.write_new
    def write(path, raw):
        native.protected_unchanged(context["guards"])
        if Path(path).name == "assembly_receipt.json":
            receipt = json.loads(raw)
            receipt.update(assembly_bridge_version=VERSION, certified_source_rows=9, original_false_source_receipt_retained=True,
                certificate_directory=str(context["certificate_directory"]), certificate_sha256=protocol.sha256((context["certificate_directory"] / "certificate.json").read_bytes()),
                certificate_aware_completion_path=str(context["completion_path"]), actual_source_argv_recorded=False,
                actual_source_command_bytes_unavailable=True, new_native_calls=0, semantic_accuracy_claimed=False)
            receipt["source_sha256"].update(context["guards"])
            raw = protocol.canonical_bytes(receipt) + b"\n"
        original_write(path, raw)
        native.protected_unchanged(context["guards"])
    assembler.write_new = write
    return assembler


def assemble(source, adjudication_files, runtime_map, output_parent, certificate_directory, completion_path, certified_decisions_path):
    context = scoped_context(certified_decisions_path, certificate_directory, completion_path)
    if context["decisions_path"] not in {Path(path).resolve() for path in adjudication_files}:
        raise ValueError("The registered completed decision file must be included explicitly")
    guard(Path(__file__), protocol.sha256(Path(__file__).read_bytes()), context["guards"])
    assembler = private_assembler(context)
    source = Path(source).resolve()
    for name in [name for key, name in assembler.NAMES.items() if key != "adjudications"]:
        guard(source / name, protocol.sha256((source / name).read_bytes()), context["guards"])
    for path in adjudication_files:
        path = Path(path).resolve()
        guard(path, protocol.sha256(path.read_bytes()), context["guards"])
        for row in protocol.parse_jsonl(path.read_bytes()):
            review = row.get("adjudication", {})
            raw_path, receipt_path = (Path(review.get(key, "")).resolve() for key in ("raw_response_path", "execution_receipt_path"))
            if (not raw_path.is_relative_to(path.parent / "executions") or not receipt_path.is_relative_to(path.parent / "executions")
                    or raw_path.parent != receipt_path.parent):
                raise ValueError("Explicit assembly execution paths escape their original owner")
            for retained in (raw_path, receipt_path):
                guard(retained, protocol.sha256(retained.read_bytes()), context["guards"])
    runtime_map = Path(runtime_map).resolve()
    guard(runtime_map, protocol.sha256(runtime_map.read_bytes()), context["guards"])
    runtime_paths = native.read_json(runtime_map)
    if not isinstance(runtime_paths, dict) or set(runtime_paths) != protocol.RUNTIME_KEYS:
        raise ValueError("Six explicit actual runtime artifacts are required")
    for path in runtime_paths.values():
        if not isinstance(path, str) or not Path(path).resolve().is_relative_to(runtime_map.parent / "runtime"):
            raise ValueError("Runtime artifacts escape their original experiment owner")
        guard(path, protocol.sha256(Path(path).read_bytes()), context["guards"])
    output, ignored = assembler.assemble(source, adjudication_files, runtime_map, output_parent)
    native.protected_unchanged(context["guards"])
    return output, native.read_json(output / "assembly_receipt.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "runtime-artifacts", "output-parent", "certificate-directory", "completion", "certified-decisions"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, action="append", required=True)
    args = parser.parse_args()
    output, receipt = assemble(args.source, args.adjudications, args.runtime_artifacts, args.output_parent,
        args.certificate_directory, args.completion, args.certified_decisions)
    print(protocol.canonical_bytes(dict(output=str(output), records=receipt["records"], certified_source_rows=9)).decode())
