"""Assemble immutable source, adjudication and actual runtime artifacts locally.

This joins existing decisions without invoking models, changing labels, dropping
unresolved records, modifying prior runs or pretending a development subset is a
held-out test set. The separate freezer revalidates the complete joined protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from oil_evidence import machine_evaluation as protocol
from adjudicate_machine_annotations import retain_decisions
import compact_machine_adjudication as compact
from report_machine_retrieval import NAMES


def write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(raw)


def assemble(source, adjudication_files, runtime_map, output_parent):
    source, runtime_map = Path(source).resolve(), Path(runtime_map).resolve()
    paths = [Path(path).resolve() for path in adjudication_files]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Supply unique retained adjudication files")
    artifacts = {key: (source / name).read_bytes() for key, name in NAMES.items() if key != "adjudications"}
    questions = protocol.parse_jsonl(artifacts["questions"])
    packets = protocol.prepare_inputs(questions, protocol.parse_jsonl(artifacts["evidence_pool"]),
                                      protocol.parse_jsonl(artifacts["corpus"]),
                                      artifacts["annotation_prompt"].decode("utf-8"))
    left, right = ({row["record_id"]: row for row in protocol.parse_jsonl(artifacts[key])}
                   for key in ("independent_a", "independent_b"))
    expected = [row["record_id"] for row in questions]
    decisions, original_hashes = {}, {str(source / NAMES[key]): protocol.sha256(raw) for key, raw in artifacts.items()}
    for path in paths:
        raw = path.read_bytes()
        original_hashes[str(path)] = protocol.sha256(raw)
        for decision in protocol.parse_jsonl(raw):
            identifier = decision.get("record_id")
            if identifier not in expected or identifier in decisions:
                raise ValueError("Unknown or duplicate retained adjudication")
            review = decision.get("adjudication", {})
            raw_path, receipt_path = Path(review.get("raw_response_path", "")).resolve(), Path(review.get("execution_receipt_path", "")).resolve()
            if (not raw_path.is_relative_to(path.parent / "executions")
                    or not receipt_path.is_relative_to(path.parent / "executions")
                    or raw_path.parent != receipt_path.parent):
                raise ValueError("Adjudication raw execution does not belong to its original run")
            raw_output = raw_path.read_bytes()
            receipt = json.loads(receipt_path.read_bytes())
            if Path(receipt.get("artifact_directory", "")).resolve() != raw_path.parent:
                raise ValueError("Execution artifact directory differs from its retained local raw path")
            if (receipt.get("accepted") is not True
                    or protocol.sha256(raw_output) != review.get("raw_response_sha256")
                    or protocol.sha256(raw_output) != receipt["artifacts_sha256"].get("raw_final.json")):
                raise ValueError("Retained adjudication execution changed")
            parsed = json.loads(raw_output, object_pairs_hook=protocol._unique_object,
                                parse_constant=protocol._invalid_constant)
            final = parsed.get(identifier)
            if (not isinstance(final, dict) or final.get("status") != decision.get("raw_model_status")
                    or final.get("reason") != review.get("reason")):
                raise ValueError("Adjudication differs from its retained exact raw decision")
            encoding = review.get("raw_encoding")
            if encoding is None:
                rebuilt = retain_decisions({identifier: final}, packets, left, right, [identifier], receipt)[0]
            elif encoding == compact.VERSION:
                rebuilt = compact.retain_compact_decisions(
                    {identifier: final}, packets, left, right, [identifier], receipt)[0]
            else:
                raise ValueError("Unknown raw adjudication encoding cannot be interpreted")
            if protocol.canonical_bytes(rebuilt) != protocol.canonical_bytes(decision):
                raise ValueError("Adjudication differs from its complete retained exact raw decision")
            original_hashes.update({str(raw_path): protocol.sha256(raw_output), str(receipt_path): protocol.sha256(receipt_path.read_bytes())})
            decisions[identifier] = decision
    if set(decisions) != set(expected):
        raise ValueError("Every frozen question requires an explicit adjudication, including unresolved ones")
    artifacts["adjudications"] = b"".join(protocol.canonical_bytes(decisions[identifier]) + b"\n" for identifier in expected)
    runtime_paths = json.loads(runtime_map.read_bytes())
    if not isinstance(runtime_paths, dict) or set(runtime_paths) != protocol.RUNTIME_KEYS:
        raise ValueError("Supply the six actual runtime artifacts from the experiment run")
    if any(not isinstance(path, str) or not Path(path).resolve().is_relative_to(runtime_map.parent / "runtime")
           for path in runtime_paths.values()):
        raise ValueError("Actual runtime snapshots must belong to their own experiment run")
    runtime_bytes = {key: Path(path).read_bytes() for key, path in runtime_paths.items()}
    runtime_hashes = {key: protocol.sha256(raw) for key, raw in runtime_bytes.items()}
    original_hashes[str(runtime_map)] = protocol.sha256(runtime_map.read_bytes())
    original_hashes.update({str(Path(runtime_paths[key]).resolve()): expected_hash for key, expected_hash in runtime_hashes.items()})
    manifest = protocol.build_manifest(artifacts, runtime_hashes)
    output = Path(output_parent).resolve() / (datetime.now(timezone.utc).isoformat().replace(":", "-") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    for key, name in NAMES.items():
        write_new(output / name, artifacts[key])
    new_paths = {}
    for key, raw in runtime_bytes.items():
        target = output / "runtime" / (key + ".artifact")
        write_new(target, raw)
        new_paths[key] = str(target)
    write_new(output / "runtime_artifacts.json", protocol.canonical_bytes(new_paths) + b"\n")
    for path, expected_hash in original_hashes.items():
        if protocol.sha256(Path(path).read_bytes()) != expected_hash:
            raise ValueError("Original input changed during assembly; output retained but uncertified")
    receipt = dict(status="complete_machine_freeze_inputs_assembled_not_yet_frozen", records=len(expected),
                   split_status_counts=manifest["split_status_counts"], human_gold=False,
                   source_artifacts_unchanged=True, source_sha256=original_hashes,
                   artifacts_sha256={path.relative_to(output).as_posix(): protocol.sha256(path.read_bytes())
                                     for path in output.rglob("*") if path.is_file()})
    write_new(output / "assembly_receipt.json", protocol.canonical_bytes(receipt) + b"\n")
    return output, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--adjudications", required=True, type=Path, action="append")
    parser.add_argument("--runtime-artifacts", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    output, receipt = assemble(args.source, args.adjudications, args.runtime_artifacts, args.output_parent)
    print(json.dumps(dict(output=str(output), records=receipt["records"], split_status_counts=receipt["split_status_counts"]), indent=2))
