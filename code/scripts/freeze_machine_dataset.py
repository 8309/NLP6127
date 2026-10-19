"""Publish a new immutable local machine-only dataset version, never human gold.

Inputs are seven original protocol artifacts plus a runtime-artifact path map.
Existing versions and all unsuccessful/unresolved adjudications are preserved.
No model call, credential access, index update or public upload is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oil_evidence import machine_evaluation as protocol
from report_machine_retrieval import NAMES


def freeze(source, output_parent):
    source, output_parent = Path(source).resolve(), Path(output_parent).resolve()
    artifacts = {key: (source / name).read_bytes() for key, name in NAMES.items()}
    raw_paths = (source / "runtime_artifacts.json").read_bytes()
    paths = json.loads(raw_paths)
    if not isinstance(paths, dict) or set(paths) != protocol.RUNTIME_KEYS:
        raise ValueError("Bind all six comparison runtime artifacts before freezing")
    # Callers stage the six auditable runtime snapshots inside this input run.
    # A path supplied in an artifact never authorizes reading credentials or
    # arbitrary host files; symlinks are resolved before the containment check.
    if any(not isinstance(path, str) or not Path(path).resolve().is_relative_to(source)
           for path in paths.values()):
        raise ValueError("Stage runtime snapshots inside the source run")
    runtime_bytes = {key: Path(path).read_bytes() for key, path in paths.items()}
    runtime_hashes = {key: protocol.sha256(raw) for key, raw in runtime_bytes.items()}
    manifest = protocol.build_manifest(artifacts, runtime_hashes)
    original_hashes = {str(source / name): protocol.sha256(artifacts[key]) for key, name in NAMES.items()}
    original_hashes[str(source / "runtime_artifacts.json")] = protocol.sha256(raw_paths)
    original_hashes.update({str(Path(paths[key]).resolve()): digest for key, digest in runtime_hashes.items()})
    output = output_parent / manifest["version"]
    if output.exists():
        raise ValueError("Frozen version already exists; never overwrite it")
    output.mkdir(parents=True, exist_ok=False)
    for key, name in NAMES.items():
        with (output / name).open("xb") as target:
            target.write(artifacts[key])
    snapshots = {}
    for key, raw in runtime_bytes.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError("Runtime artifact keys must be safe identifiers")
        path = output / "runtime" / (key + ".artifact")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as target:
            target.write(raw)
        snapshots[key] = str(path)
    write_new(output / "runtime_artifacts.json", snapshots)
    write_new(output / "manifest.json", manifest)
    by_id = {row["record_id"]: row for row in protocol.parse_jsonl(artifacts["questions"])}
    exported = {split: [] for split in ("development", "test")}
    for decision in protocol.parse_jsonl(artifacts["adjudications"]):
        question = by_id[decision["record_id"]]
        if decision["status"] == "accepted":
            exported[question["split"]].append(dict(question, **decision))
    for split, rows in exported.items():
        with (output / (split + "_machine_accepted.jsonl")).open("xb") as target:
            target.write(b"".join(protocol.canonical_bytes(row) + b"\n" for row in rows))
    for path, expected in original_hashes.items():
        if protocol.sha256(Path(path).read_bytes()) != expected:
            raise ValueError("Input changed while freezing; new output is retained but not certified")
    protocol.verify_manifest(manifest, {key: (output / name).read_bytes() for key, name in NAMES.items()},
                             {key: protocol.sha256(Path(path).read_bytes()) for key, path in snapshots.items()})
    receipt = dict(status="machine_only_version_frozen", source=str(source), output=str(output),
                   version=manifest["version"], human_gold=False, source_artifacts_unchanged=True,
                   source_sha256=original_hashes,
                   artifacts_sha256={path.relative_to(output).as_posix(): protocol.sha256(path.read_bytes())
                                     for path in output.rglob("*") if path.is_file()},
                   split_status_counts=manifest["split_status_counts"],
                   formal_test_ready=manifest["split_status_counts"]["test"].get("accepted", 0) >= 1000
                       and not manifest["split_status_counts"]["test"].get("unresolved", 0),
                   note="Frozen machine consensus/adjudication is not certified human ground truth.")
    write_new(output / "freeze_receipt.json", receipt)
    return output, receipt


def write_new(path, value):
    with Path(path).open("xb") as target:
        target.write(protocol.canonical_bytes(value) + b"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    output, receipt = freeze(args.source, args.output_parent)
    print(json.dumps(dict(output=str(output), **{key: receipt[key] for key in
                    ("version", "split_status_counts", "formal_test_ready", "human_gold")}), indent=2))
