"""Synthetic distinct-task augmentation checks; never real question counts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("source_augmentation_test", scripts / "augment_machine_questions.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def packet():
    text = ("Synthetic crude oil exports increased while pipeline capacity remained constrained in March 2024. "
            "Crude oil production at the northern fields restarted after repair work, but pipeline throughput was limited. "
            "The original paragraph separately describes export flows, field operation status and the transport bottleneck; "
            "these are not interchangeable measures or evidence of a later capacity expansion.")
    return dict(packet_id="synthetic-packet", event_family_id="synthetic-family", candidate_split="test",
                evidence=[dict(passage_id="synthetic-passage", document_id="synthetic-document", source_page_id="synthetic-page",
                               text=text, char_start=0, char_end=len(text), available_by="2024-03-01")],
                provenance="accepted_source_packet_not_question_or_label", human_gold=False, questions_created=0,
                uncertainty_flags=["semantic_independence_unverified"], earlier_questions=["How did crude oil exports change?"],
                prior_admitted_question_count=1)


def row(packet):
    return dict(local_id="q001", packet_id=packet["packet_id"], question="Which transport bottleneck limited pipeline throughput after northern crude oil fields restarted?",
                as_of="2024-03-01", scenario_intent="fact", design_reason="New transport-mechanism task, not a flow change question.",
                anchor_spans=[dict(passage_id="synthetic-passage", text="pipeline throughput was limited", role="support_context")],
                semantic_duplicate=False, new_fact_focus="The transport constraint after field operations resumed.",
                distinctness_reason="Earlier question asks about export-flow change; this asks about a different transport constraint.")


def raw(records):
    return json.dumps(dict(records=records)).encode()


def test_explicit_selfcheck_and_exact_span_not_answer_gold(module, packet):
    result = module.validate_candidates(raw([row(packet)]), [packet])
    assert result["actual_validated_questions"] == 1
    audit = result["authoring"][0]
    assert audit["semantic_duplicate_author_selfcheck"] is False
    assert audit["semantic_distinctness_independently_verified"] is False
    assert audit["human_gold"] is False
    assert "qrels" not in result["questions"][0]


@pytest.mark.parametrize("mutation", ["semantic_duplicate", "missing_focus", "missing_why", "wrong_quote", "no_answer_control", "future_control"])
def test_rejected_selfchecks_or_padding_controls_retained(module, packet, mutation):
    value = row(packet)
    if mutation == "semantic_duplicate":
        value["semantic_duplicate"] = True
    elif mutation == "missing_focus":
        value["new_fact_focus"] = ""
    elif mutation == "missing_why":
        value["distinctness_reason"] = ""
    elif mutation == "wrong_quote":
        value["anchor_spans"][0]["text"] = "Invented source"
    elif mutation == "no_answer_control":
        value["scenario_intent"] = "evidence_sufficiency"
    else:
        value["anchor_spans"][0]["role"] = "future_leakage_control"
    result = module.validate_candidates(raw([value]), [packet])
    assert not result["questions"]
    assert len(result["rejected"]) == 1


def test_repeat_number_date_template_is_not_distinct_task(module, packet):
    value = row(packet)
    value["question"] = "What was crude oil output in 2024?"
    result = module.validate_candidates(raw([value]), [packet], [module.preparation.normalized("What was crude oil output in 2023?")])
    assert not result["questions"]
    assert "duplicate" in result["rejected"][0]["reasons"][0].lower()


def test_one_new_task_limit_per_batch_packet(module, packet):
    with pytest.raises(ValueError, match="one"):
        module.validate_candidates(raw([row(packet), row(packet)]), [packet])


def test_earlier_strings_from_same_packet_only_and_cap_two(module, packet):
    sibling = deepcopy(packet)
    sibling["packet_id"] = "synthetic-sibling"
    questions = [dict(record_id="q001", question="Earlier export task."), dict(record_id="q002", question="Earlier transport task.")]
    audits = [dict(record_id="q001", packet_id=packet["packet_id"]), dict(record_id="q002", packet_id=packet["packet_id"])]
    eligible = module.eligible_packets([packet, sibling], questions, audits, "test")
    assert len(eligible) == 1 and eligible[0]["packet_id"] == sibling["packet_id"]
    assert eligible[0]["earlier_questions"] == []
    assert "Earlier export task." not in json.dumps(eligible)


def test_no_answers_labels_rankings_or_prior_other_packet_prompts(module):
    assert "ONLY earlier question strings" in module.INSTRUCTIONS
    assert "Do not paraphrase" in module.INSTRUCTIONS
    assert "NOT answerability labels" in module.INSTRUCTIONS
