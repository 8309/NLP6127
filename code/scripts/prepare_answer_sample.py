"""Seal a uniform100 TEST answer sample before TEST annotation or scoring.

Selection reads only the locked question IDs/order and a metadata-only zero-TEST
annotation-attempt audit. Seed and counts are fixed; no answer, label, relevance,
retrieval rank, future eligibility or question-quality filter enters the draw.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
from uuid import uuid4

from oil_evidence import machine_evaluation as protocol
from oil_evidence.store import now

SEED, TEST_COUNT, ANSWER_COUNT = 6127, 1000, 100
VERSION = "uniform-test100-pre-label-v1"


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=protocol._unique_object,
                      parse_constant=protocol._invalid_constant)


def file_hash(path):
    return protocol.sha256(Path(path).read_bytes())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(value if isinstance(value, bytes) else protocol.canonical_bytes(value)+b"\n")


def sample_ids(ordered_test_ids):
    if len(ordered_test_ids) != TEST_COUNT or len(set(ordered_test_ids)) != TEST_COUNT:
        raise ValueError("Exactly1000 unique locked TEST IDs are required")
    draw_order = random.Random(SEED).sample(ordered_test_ids, ANSWER_COUNT)
    selected = set(draw_order)
    return draw_order, [qid for qid in ordered_test_ids if qid in selected]


def prepare(root, cohort, status_path, status_sha256, progress_path):
    root, cohort, status_path, progress_path = map(lambda p:Path(p).resolve(),
        (root, cohort, status_path, progress_path))
    if not cohort.is_relative_to(root / "data/machine_runs/question-cohorts"):
        raise ValueError("Use the original locked canonical question cohort")
    if not status_path.is_relative_to(root / "data/machine_runs/candidate-retrieval") or not progress_path.is_relative_to(status_path.parent):
        raise ValueError("Use the retained canonical metadata-only pre-label audit")
    if file_hash(status_path) != status_sha256:
        raise ValueError("Pre-label audit bytes changed")
    receipt, status = read_json(cohort / "receipt.json"), read_json(status_path)
    if (receipt.get("question_selection_locked") is not True or receipt.get("actual_counts") != {"development":200,"test":1000}
            or receipt.get("qa_annotations_created") != 0 or receipt.get("human_gold") is not False):
        raise ValueError("Question selection is not the fixed unlabelled200/1000 cohort")
    protected = {str(cohort / "receipt.json"):file_hash(cohort / "receipt.json"), str(status_path):status_sha256}
    for name in ("questions.jsonl","test_questions.jsonl"):
        path = cohort / name
        expected = receipt["artifact_sha256"][name]
        if file_hash(path) != expected:
            raise ValueError("Locked source questions changed")
        protected[str(path)] = expected
    questions = protocol.parse_jsonl((cohort / "questions.jsonl").read_bytes())
    originals = protocol._validate_questions(questions)
    tests = protocol.parse_jsonl((cohort / "test_questions.jsonl").read_bytes())
    if tests != [q for q in questions if q["split"] == "test"] or len(tests) != TEST_COUNT:
        raise ValueError("The exact original1000 TEST file/order is required")
    if (Path(status["locked_cohort"]).resolve() != cohort
            or status.get("questions_sha256") != protected[str(cohort / "questions.jsonl")]
            or status.get("test_questions_sha256") != protected[str(cohort / "test_questions.jsonl")]
            or status.get("accepted_test_labels_by_stream") != {"a":0,"b":0}
            or status.get("test_native_attempts_started") != 0
            or status.get("no_label_or_answer_bodies_used_for_sampling") is not True):
        raise ValueError("Draw must precede all TEST labels and TEST native attempts")
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(status["created_utc"])).total_seconds()
    if not 0 <= age <= 900:
        raise ValueError("Refresh the metadata-only zero-TEST audit before drawing")
    if file_hash(progress_path) != status["progress_sha256_at_snapshot"]:
        raise ValueError("Retained append-only audit prefix differs")
    protected[str(progress_path)] = file_hash(progress_path)
    for ids in status.get("accepted_record_ids_by_stream", {}).values():
        if any(qid not in originals or originals[qid]["split"] != "development" for qid in ids):
            raise ValueError("The pre-label audit contains a TEST/unknown accepted record")
    for attempt in status.get("native_attempts_started", []):
        if attempt.get("split") != "development" or any(originals.get(qid,{}).get("split") != "development"
                                                        for qid in attempt.get("record_ids", [])):
            raise ValueError("TEST annotation attempt already started")
    progress = protocol.parse_jsonl(progress_path.read_bytes(), "pre-label progress prefix")
    if any(originals.get(qid,{}).get("split") != "development" for row in progress for qid in row.get("record_ids", [])):
        raise ValueError("Retained progress prefix includes TEST annotation")
    ids = [q["record_id"] for q in tests]
    draw, ordered_sample = sample_ids(ids)
    selected_rows = [q for q in tests if q["record_id"] in set(ordered_sample)]
    output = root / "data/machine_runs/answer-samples" / (now().replace(":", "-")+"-"+uuid4().hex[:8])
    plan = dict(schema_version=VERSION, created_utc=now(), source_cohort=str(cohort),
        source_questions_sha256=protected[str(cohort / "questions.jsonl")],
        source_test_questions_sha256=protected[str(cohort / "test_questions.jsonl")],
        source_selection_receipt_sha256=protected[str(cohort / "receipt.json")],
        original_ordered_test_record_ids=ids, original_test_count=TEST_COUNT, answer_sample_count=ANSWER_COUNT,
        sample_seed=SEED, selection="python_random.Random.sample_without_replacement_from_original_ID_order",
        draw_order_record_ids=draw, answer_record_ids=ordered_sample, execution_order="original_test_file_order",
        selection_uses_only_IDs_and_order=True, labels_answers_ranks_scores_used=False,
        selected_before_test_labels=True, selected_before_test_native_attempts=True,
        pre_label_audit_sha256=status_sha256, pre_label_progress_prefix_sha256=file_hash(progress_path),
        original_status_observed_utc=status["created_utc"], human_gold=False,
        retrieval_evaluation_count=TEST_COUNT, answer_variant_results=ANSWER_COUNT*6,
        python=platform.python_version(), uncertainty_note="100 sampled questions, not1000 generative answers; machine-label and cluster biases remain.")
    write_new(output / "sample_plan.json", plan)
    write_new(output / "sample_questions.jsonl", b"".join(protocol.canonical_bytes(row)+b"\n" for row in selected_rows))
    write_new(output / "pre_label_status.json", status_path.read_bytes())
    write_new(output / "pre_label_progress.jsonl", progress_path.read_bytes())
    for path, expected in protected.items():
        if file_hash(path) != expected:
            raise ValueError("Original sampling input changed; retain new output uncertified")
    result = dict(status="immutable_uniform100_TEST_answer_sample_sealed_before_TEST_labels",
        sample_plan_sha256=file_hash(output / "sample_plan.json"), selected_questions=100, population_test_questions=1000,
        seed=6127, original_inputs_unchanged=True, original_input_sha256=protected,
        no_native_calls=True, labels_created=0, scores_created=0, human_gold=False,
        implementation_sha256=file_hash(Path(__file__)),
        artifact_sha256={p.name:file_hash(p) for p in output.iterdir() if p.is_file()})
    write_new(output / "receipt.json", result)
    return output, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--question-cohort", required=True, type=Path)
    parser.add_argument("--pre-label-status", required=True, type=Path)
    parser.add_argument("--status-sha256", required=True)
    parser.add_argument("--progress-prefix", required=True, type=Path)
    args = parser.parse_args()
    output, receipt = prepare(args.canonical_root, args.question_cohort, args.pre_label_status, args.status_sha256, args.progress_prefix)
    print(json.dumps(dict(output=str(output), **receipt), indent=2))
