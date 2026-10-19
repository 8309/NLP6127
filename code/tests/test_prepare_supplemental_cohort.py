"""Synthetic supplemental lineage checks; no sources or questions manufactured."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("supplement_preparation_test", scripts / "prepare_supplemental_cohort.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def packets():
    def packet(index):
        return dict(packet_id=f"synthetic-packet-{index}", event_family_id="synthetic-family", candidate_split="test",
                    evidence=[dict(passage_id=f"synthetic-passage-{index}", document_id="synthetic-document")],
                    provenance="accepted_source_packet_not_question_or_label", human_gold=False, questions_created=0)
    original = [packet(index) for index in range(3)]
    supplemental = [packet(index) for index in range(3, 6)]
    families = {"synthetic-family": dict(document_ids=["synthetic-document"], reserved_due_to_prior_exposure=False)}
    return supplemental, original, families


def test_source_packets_are_not_questions_or_qa_yield(module):
    summary = module.validate_supplement(*packets())
    assert summary["source_packets"] == 3
    assert summary["actual_qa_questions"] == summary["actual_qa_labels"] == 0
    assert summary["source_packet_capacity_is_not_qa_yield"]
    assert summary["event_partitions_unchanged"]


@pytest.mark.parametrize("mutation", ["split", "family", "prior_exposed", "human", "duplicate_source", "duplicate_packet"])
def test_never_weaken_exposure_or_source_isolation(module, mutation):
    supplemental, original, families = deepcopy(packets())
    if mutation == "split":
        supplemental[0]["candidate_split"] = "development"
    elif mutation == "family":
        supplemental[0]["event_family_id"] = "manufactured-family"
    elif mutation == "prior_exposed":
        families["synthetic-family"]["reserved_due_to_prior_exposure"] = True
    elif mutation == "human":
        supplemental[0]["human_gold"] = True
    elif mutation == "duplicate_source":
        supplemental[0]["evidence"][0]["passage_id"] = original[0]["evidence"][0]["passage_id"]
    else:
        supplemental[1]["packet_id"] = supplemental[0]["packet_id"]
    with pytest.raises(ValueError):
        module.validate_supplement(supplemental, original, families)


def test_six_paragraph_total_includes_original_three(module):
    supplemental, original, families = packets()
    extra = deepcopy(supplemental[0])
    extra["packet_id"] = "synthetic-extra-packet"
    extra["evidence"][0]["passage_id"] = "synthetic-extra-passage"
    with pytest.raises(ValueError, match="six"):
        module.validate_supplement(supplemental + [extra], original, families)
