"""Synthetic pre-label sampling checks; no annotation or quality measurements."""
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import prepare_answer_sample as module


def test_uniform_fixed_seed_has_exact100_ids_and_original_execution_order():
    ids = [f"fixture-test-{index}" for index in range(1000)]
    draw, ordered = module.sample_ids(ids)
    assert len(draw) == len(set(draw)) == len(ordered) == 100
    assert (draw,ordered) == module.sample_ids(ids)
    assert ordered == [qid for qid in ids if qid in set(draw)]
    assert module.SEED == 6127
    with pytest.raises(ValueError, match="Exactly1000"):
        module.sample_ids(ids[:999])


def sampling_fixture(tmp_path):
    root = tmp_path / "canonical"
    cohort = root / "data/machine_runs/question-cohorts/fixture"
    status_dir = root / "data/machine_runs/candidate-retrieval/fixture/annotation"
    cohort.mkdir(parents=True)
    status_dir.mkdir(parents=True)
    rows = [dict(record_id=f"fixture-{split}-{index}", question=f"Synthetic {split} question {index}?",
        split=split, as_of="2024-01-01", event_group=f"fixture-{split}-event-{index}")
        for split,count in (("development",200),("test",1000)) for index in range(count)]
    tests = rows[200:]
    for name, values in (("questions.jsonl",rows),("test_questions.jsonl",tests)):
        (cohort / name).write_bytes(b"".join(module.protocol.canonical_bytes(row)+b"\n" for row in values))
    receipt = dict(question_selection_locked=True, actual_counts=dict(development=200,test=1000),
        qa_annotations_created=0, human_gold=False,
        artifact_sha256={name:module.file_hash(cohort / name) for name in ("questions.jsonl","test_questions.jsonl")})
    (cohort / "receipt.json").write_bytes(module.protocol.canonical_bytes(receipt))
    progress = status_dir / "progress-prefix.jsonl"
    progress.write_bytes(module.protocol.canonical_bytes(dict(type="completed", record_ids=[rows[0]["record_id"]]))+b"\n")
    status = dict(locked_cohort=str(cohort), questions_sha256=receipt["artifact_sha256"]["questions.jsonl"],
        test_questions_sha256=receipt["artifact_sha256"]["test_questions.jsonl"],
        accepted_test_labels_by_stream=dict(a=0,b=0), test_native_attempts_started=0,
        no_label_or_answer_bodies_used_for_sampling=True, created_utc=datetime.now(timezone.utc).isoformat(),
        progress_sha256_at_snapshot=module.file_hash(progress), accepted_record_ids_by_stream=dict(a=[rows[0]["record_id"]],b=[]),
        native_attempts_started=[dict(split="development", record_ids=[rows[0]["record_id"]])])
    status_path = status_dir / "status.json"
    status_path.write_bytes(module.protocol.canonical_bytes(status))
    return root, cohort, status_path, progress


def test_sealed_sample_preserves_population_and_observations(tmp_path):
    root, cohort, status, progress = sampling_fixture(tmp_path)
    before = {p:p.read_bytes() for p in (cohort / "questions.jsonl",cohort / "test_questions.jsonl",status,progress)}
    output, receipt = module.prepare(root,cohort,status,module.file_hash(status),progress)
    assert receipt["selected_questions"] == 100 and receipt["population_test_questions"] == 1000
    assert all(p.read_bytes() == raw for p,raw in before.items())
    plan = module.read_json(output / "sample_plan.json")
    assert plan["labels_answers_ranks_scores_used"] is False and plan["answer_variant_results"] == 600
    assert len(module.protocol.parse_jsonl((output / "sample_questions.jsonl").read_bytes())) == 100
    assert receipt["sample_plan_sha256"] == module.file_hash(output / "sample_plan.json")


@pytest.mark.parametrize("change", ["test_labels","test_attempt","test_progress","wrong_hash","old_audit"])
def test_late_or_tampered_sampling_is_rejected_before_draw(tmp_path,change):
    root,cohort,status_path,progress = sampling_fixture(tmp_path)
    status = module.read_json(status_path)
    if change == "test_labels":
        status["accepted_test_labels_by_stream"]["a"] = 1
    elif change == "test_attempt":
        status["test_native_attempts_started"] = 1
    elif change == "test_progress":
        progress.write_bytes(module.protocol.canonical_bytes(dict(record_ids=["fixture-test-0"]))+b"\n")
        status["progress_sha256_at_snapshot"] = module.file_hash(progress)
    elif change == "old_audit":
        status["created_utc"] = "2000-01-01T00:00:00+00:00"
    status_path.write_bytes(module.protocol.canonical_bytes(status))
    expected = "0"*64 if change == "wrong_hash" else module.file_hash(status_path)
    with pytest.raises(ValueError):
        module.prepare(root,cohort,status_path,expected,progress)
    assert not (root / "data/machine_runs/answer-samples").exists()
