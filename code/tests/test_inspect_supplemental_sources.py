"""Synthetic source cap and unchanged exposure-reservation checks."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("supplement_sources_test", scripts / "inspect_supplemental_sources.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(module, count=8, reserved=False):
    rows, pages = [], {}
    for index in range(count):
        text = (f"Synthetic crude oil production increased at field site {index} in March. "
                "Observed production remained affected by transport constraints and crude export availability. "
                "The report states a distinct location and reported production period for this original paragraph. "
                "Crude volumes were measured consistently, and this synthetic example preserves units and the observed event dates.")
        page = f"synthetic-page-{index}"
        rows.append(dict(passage_id=f"synthetic-passage-{index}", document_id="synthetic-document",
                         source_page_id=page, text=text, char_start=0, char_end=len(text),
                         source="SYNTHETIC", available_by="2024-03-01", page=1))
        pages[page] = dict(text=text)
    original = [dict(packet_id=f"synthetic-packet-{index}", event_family_id="synthetic-family", candidate_split="test",
                     evidence=[row]) for index, row in enumerate(rows[:3])]
    families = {"synthetic-family": dict(document_ids=["synthetic-document"],
                reserved_due_to_prior_exposure=reserved, uncertainty_flags=["semantic_independence_unverified"])}
    return rows, pages, original, families


def test_only_native_new_paragraphs_under_same_family_split(module):
    rows, pages, original, families = fixture(module)
    packets, summary = module.supplemental_packets(rows, pages, original, families)
    assert len(packets) == 3
    assert {packet["candidate_split"] for packet in packets} == {"test"}
    assert {packet["event_family_id"] for packet in packets} == {"synthetic-family"}
    original_ids = {row["passage_id"] for packet in original for row in packet["evidence"]}
    assert not original_ids & {row["passage_id"] for packet in packets for row in packet["evidence"]}
    assert summary["actual_new_questions"] == 0
    assert summary["source_corpus_resplit"] is False
    assert summary["family_reservations_weakened"] is False
    assert summary["rejection_counts"]["six_native_paragraph_cap"] == 2


def test_reserved_family_never_released_to_hit_counts(module):
    packets, summary = module.supplemental_packets(*fixture(module, reserved=True))
    assert packets == []
    assert summary["rejection_counts"]["unchanged_prior_exposure_family_reservation"] == 5


def test_duplicate_native_text_not_counted_again(module):
    rows, pages, original, families = fixture(module, count=4)
    rows[3] = dict(rows[3], text=rows[0]["text"], char_end=len(rows[0]["text"]))
    pages[rows[3]["source_page_id"]] = dict(text=rows[3]["text"])
    packets, summary = module.supplemental_packets(rows, pages, original, families)
    assert not packets
    assert summary["rejection_counts"]["repeated_native_paragraph_text"] == 1


@pytest.mark.parametrize("cap", [3, 5, 7, True])
def test_total_selection_cap_is_explicit_six_not_count_manufacture(module, cap):
    with pytest.raises(ValueError, match="cap"):
        module.supplemental_packets(*fixture(module), max_per_document=cap)


def test_exact_source_offset_failure_does_not_create_packet(module):
    rows, pages, original, families = fixture(module, count=4)
    pages[rows[3]["source_page_id"]]["text"] = "Altered source"
    with pytest.raises(ValueError):
        module.supplemental_packets(rows, pages, original, families)
