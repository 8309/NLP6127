"""Evaluate frozen machine-only judgments without changing human freeze rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oil_evidence import machine_evaluation as protocol
from oil_evidence.machine_metrics import evaluate_rankings


NAMES = dict(corpus="corpus.jsonl", questions="questions.jsonl", evidence_pool="evidence_pool.jsonl",
             annotation_prompt="annotation_prompt.txt", independent_a="independent_a.jsonl",
             independent_b="independent_b.jsonl", adjudications="adjudications.jsonl")


def evaluate(frozen_dir, rankings_path, output, split="test", required_test_records=1000, k=10):
    frozen_dir, rankings_path, output = map(lambda p: Path(p).resolve(), (frozen_dir, rankings_path, output))
    if type(k) is not int or k <= 0:
        raise ValueError("Retrieval depth k must be a positive integer")
    if split not in {"test", "development"}:
        raise ValueError("Explicit test or development split is required")
    artifacts = {key: (frozen_dir / name).read_bytes() for key, name in NAMES.items()}
    manifest_bytes = (frozen_dir / "manifest.json").read_bytes()
    runtime_map_bytes = (frozen_dir / "runtime_artifacts.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    paths = json.loads(runtime_map_bytes)
    if set(paths) != set(manifest.get("runtime_sha256", {})):
        raise ValueError("Runtime artifact map differs from frozen bindings")
    if any(not isinstance(path, str) or not Path(path).resolve().is_relative_to(frozen_dir / "runtime")
           for path in paths.values()):
        raise ValueError("Runtime snapshots must remain inside this frozen version")
    runtime = {key: protocol.sha256(Path(path).read_bytes()) for key, path in paths.items()}
    protocol.verify_manifest(manifest, artifacts, runtime)
    if split == "test":
        if required_test_records != 1000:
            raise ValueError("Formal test coverage cannot lower the 1000-record requirement")
        protocol.preflight_experiments(manifest, 1000)
    questions = protocol.parse_jsonl(artifacts["questions"])
    selected = [row["record_id"] for row in questions if row["split"] == split]
    decisions = [row for row in protocol.parse_jsonl(artifacts["adjudications"]) if row["record_id"] in set(selected)]
    raw_rankings = rankings_path.read_bytes()
    rankings = protocol.parse_jsonl(raw_rankings)
    by_question = {row["record_id"]: row for row in questions}
    corpus = {row["passage_id"]: row for row in protocol.parse_jsonl(artifacts["corpus"])}
    for row in rankings:
        identifier = row.get("record_id", row.get("question_id"))
        question = by_question.get(identifier)
        if (question is None or row.get("question") != question["question"]
                or row.get("as_of") != question["as_of"]):
            raise ValueError("Ranking does not bind its frozen question and cutoff")
        if not isinstance(row.get("evidence"), list):
            raise ValueError("Ranking evidence must be a list")
        for passage in row["evidence"]:
            original = corpus.get(passage.get("passage_id")) if isinstance(passage, dict) else None
            if original is None or any(passage.get(key) != original.get(key)
                                      for key in ("document_id", "text", "available_by")):
                raise ValueError("Ranked hit does not match the frozen corpus evidence")
    # Selecting a named split is explicit, unlike selecting convenient records.
    rankings = [row for row in rankings if row.get("record_id", row.get("question_id")) in set(selected)]
    report = evaluate_rankings(rankings, decisions, selected, k=k)
    bound_paths = {name: frozen_dir / name for name in NAMES.values()}
    bound_paths.update({"manifest.json": frozen_dir / "manifest.json",
                       "runtime_artifacts.json": frozen_dir / "runtime_artifacts.json",
                       "rankings": rankings_path, **{"runtime:" + key: Path(value) for key, value in paths.items()}})
    expected_sha256 = {NAMES[key]: protocol.sha256(raw) for key, raw in artifacts.items()}
    expected_sha256.update({"manifest.json": protocol.sha256(manifest_bytes),
                           "runtime_artifacts.json": protocol.sha256(runtime_map_bytes),
                           "rankings": protocol.sha256(raw_rankings), **{"runtime:" + key: value for key, value in runtime.items()}})
    if any(protocol.sha256(path.read_bytes()) != expected_sha256[key] for key, path in bound_paths.items()):
        raise ValueError("Frozen evaluation input changed during derived metric calculation")
    report.update(split=split, formal_heldout_test=split == "test", dataset_version=manifest["version"],
                  frozen_manifest_sha256=protocol.sha256(manifest_bytes),
                  rankings_sha256=protocol.sha256(raw_rankings),
                  verified_input_sha256=expected_sha256, original_inputs_unchanged=True,
                  implementation_sha256=protocol.sha256(Path(__file__).read_bytes()))
    if output.exists():
        raise ValueError("Preserve earlier evaluation; choose a new output file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target:
        target.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "development"), default="test")
    parser.add_argument("--k", type=int, default=10, help="Ranking depth for metrics (default: 10)")
    args = parser.parse_args()
    result = evaluate(args.frozen_dir, args.rankings, args.output, args.split, k=args.k)
    print(json.dumps({key: result[key] for key in ("questions", "ranked_runs", "k", "split", "formal_heldout_test", "summary")}, indent=2))
