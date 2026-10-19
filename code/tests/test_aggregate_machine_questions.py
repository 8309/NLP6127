"""Synthetic admission checks; fixtures never count as real QA questions."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("question_aggregation_test", scripts / "aggregate_machine_questions.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def candidate(index, text, split="development", group=None, document=None):
    question = dict(record_id=f"synthetic-{index:03d}", question=text, as_of="2024-03-01",
                    event_group=group or f"synthetic-family-{index}", split=split)
    audit = dict(record_id=question["record_id"], packet_id=f"synthetic-packet-{index}", human_gold=False,
                 anchors=[dict(document_id=document or f"synthetic-document-{index}")])
    return question, audit, "synthetic-execution"


def test_actual_counts_not_source_packet_or_target_counts(module):
    result = module.aggregate_records([candidate(1, "How did Libya's crude oil production change?")])
    assert result["actual_counts"] == {"development": 1}
    assert result["targets_met"] is False
    assert result["target_shortfall"] == {"development": 199, "test": 1000}
    assert result["qa_annotations_created"] == 0
    assert result["human_gold"] is False


def test_nonnumeric_cross_attempt_duplicates_retained_not_counted(module):
    result = module.aggregate_records([
        candidate(1, "What was Libya's crude output in 2023?"),
        candidate(2, "What was Libya's crude output in 2024?", "test")])
    assert len(result["questions"]) == 1
    assert result["held_candidate_count"] == 1
    assert "nonnumeric_question_template_duplicate" in result["decisions"][1]["reasons"]


def test_prior_question_duplicate_never_reintroduced(module):
    text = "How did crude oil exports change?"
    result = module.aggregate_records([candidate(1, text)], [module.preparation.normalized(text)])
    assert not result["questions"]
    assert "normalized_question_duplicate" in result["decisions"][0]["reasons"]


def test_scope_holds_do_not_delete_raw_or_become_gold(module):
    item = candidate(1, "How did sustainable aviation fuel consumption change?")
    original = deepcopy(item)
    result = module.aggregate_records([item])
    assert not result["questions"]
    assert result["decisions"][0]["status"] == "held_question_candidate_not_label"
    assert item == original


@pytest.mark.parametrize("extra", ["scenario_intent", "answer", "qrels", "anchor_spans", "rank"])
def test_author_and_label_hints_rejected_in_query(module, extra):
    item = candidate(1, "How did crude oil exports change?")
    item[0][extra] = None
    with pytest.raises(ValueError, match="hints"):
        module.aggregate_records([item])


@pytest.mark.parametrize("shared", ["group", "document"])
def test_source_or_event_cross_split_fails_closed(module, shared):
    args = {shared: "synthetic-shared"}
    with pytest.raises(ValueError, match="crosses split"):
        module.aggregate_records([
            candidate(1, "How did crude exports change?", **args),
            candidate(2, "Which oil fields restarted production?", "test", **args)])


def test_original_development_is_separate_truthful_provenance(module):
    original = candidate(1, "How did crude oil production change?")[:2]
    result = module.aggregate_records([candidate(2, "Which pipeline outage reduced exports?", "test")], original=[original])
    assert result["original_development_count"] == 1
    assert result["actual_counts"] == {"development": 1, "test": 1}
    assert result["semantic_independence_established"] is False


def test_two_admitted_questions_total_per_original_source_packet(module):
    rows = [candidate(1, "What changed crude oil exports?"),
            candidate(2, "Which terminals resumed oil shipments?"),
            candidate(3, "How did refinery throughput compare with imports?")]
    for _, audit, _ in rows:
        audit["packet_id"] = "synthetic-shared-source-packet"
    result = module.aggregate_records(rows)
    assert len(result["questions"]) == 2
    assert "two_admitted_questions_per_native_source_packet_limit" in result["decisions"][2]["reasons"]


def test_exact_core_selection_by_prior_order_not_gold_and_reserves_retained(module):
    rows = [candidate(1, "How did crude exports change?"), candidate(2, "Which terminal closed?"),
            candidate(3, "Which fields restarted?", "test"), candidate(4, "How did refinery feedstocks change?", "test")]
    selected, audits, reserves = module.select_question_core([row[0] for row in rows], [row[1] for row in rows],
                                                            targets={"development": 1, "test": 1})
    assert [row["record_id"] for row in selected] == ["synthetic-001", "synthetic-003"]
    assert [row["record_id"] for row in reserves] == ["synthetic-002", "synthetic-004"]
    assert all(row["human_gold"] is False for row in audits)


def test_question_core_target_cannot_be_padded_or_lowered(module):
    item = candidate(1, "How did crude oil exports change?")
    with pytest.raises(ValueError, match="targets"):
        module.select_question_core([item[0]], [item[1]])
