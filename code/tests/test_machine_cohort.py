"""Synthetic source-cohort guards; these fixtures are not real QA coverage."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def module():
    path = Path(__file__).parents[1] / "scripts/prepare_machine_cohort.py"
    spec = importlib.util.spec_from_file_location("isolated_machine_cohort", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def source():
    documents, passages, pages = [], [], {}
    for number in range(6):
        document_id = f"synthetic-document-{number}"
        passage_id = f"synthetic-passage-{number}"
        page_id = f"synthetic-page-{number}"
        text = (f"Synthetic country {number} reported that crude oil exports increased to three million barrels per day. "
                "The petroleum transport report describes exports and imports separately, preserves the stated reporting period, "
                "and distinguishes port capacity from actual observed flows. It does not establish a causal production effect, "
                "and supplies no information about future operational disruptions or their duration.")
        prefix = "Synthetic source header. "
        documents.append(dict(document_id=document_id, source="SYNTHETIC", title=f"Synthetic crude transport report {number}",
                              available_by="2024-03-01", url=f"https://fixture.invalid/report/{number}",
                              raw_sha256=f"{number + 1:064x}"))
        passages.append(dict(passage_id=passage_id, document_id=document_id, source_page_id=page_id,
                             text=text, available_by="2024-03-01", source="SYNTHETIC", title=documents[-1]["title"],
                             url=documents[-1]["url"], char_start=len(prefix), char_end=len(prefix) + len(text), page=1))
        pages[page_id] = dict(source_page_id=page_id, text=prefix + text + " Synthetic footer.")
    inventory = dict(blocked_document_ids=["synthetic-document-0"], blocked_urls=[], blocked_raw_sha256=[])
    return dict(documents=documents, passages=passages, pages=pages, inventories=[inventory])


def generated(packet, local_id="q01", question=None, **overrides):
    passage = packet["evidence"][0]
    return dict(local_id=local_id, packet_id=packet["packet_id"],
                question=question or "What crude oil export development was reported for synthetic country 2 in March 2024?",
                as_of="2024-03-31", scenario_intent="fact", design_reason="Synthetic standalone source-first task.",
                anchor_spans=[dict(passage_id=passage["passage_id"], text="crude oil exports increased",
                                  role="support_context")]) | overrides


def test_no_model_or_index_client_is_imported(module):
    for name in ("ModelClient", "Retriever", "embedder", "DATA", "sqlite3", "httpx"):
        assert not hasattr(module, name)


def test_alias_closure_is_transitive_and_ignores_empty_raw_hash(module, source):
    source["documents"][1]["url"] = source["documents"][0]["url"] + "#fragment"
    source["documents"][2]["raw_sha256"] = source["documents"][1]["raw_sha256"]
    source["documents"][3]["raw_sha256"] = ""
    source["documents"][4]["raw_sha256"] = ""
    blocked, urls, raw = module.alias_closure(source["documents"], source["inventories"])
    assert blocked == {"synthetic-document-0", "synthetic-document-1", "synthetic-document-2"}
    assert source["documents"][0]["url"] in urls
    assert "" not in raw


def test_final_url_alias_is_reserved(module, source):
    source["documents"][1]["final_url"] = source["documents"][0]["url"]
    blocked, _, _ = module.alias_closure(source["documents"], source["inventories"])
    assert "synthetic-document-1" in blocked


def test_named_hurricane_siblings_are_one_reserved_family(module, source):
    source["documents"][0]["title"] = "Synthetic Hurricane Michael update"
    source["documents"][1]["title"] = "Synthetic Tropical Storm Michael final activity report"
    families, assignment = module.grouped_documents(source["documents"], source["passages"], {"synthetic-document-0"})
    assert assignment["synthetic-document-0"] == assignment["synthetic-document-1"]
    family = families[assignment["synthetic-document-1"]]
    assert family["reserved_due_to_prior_exposure"] is True
    assert "storm_year_and_sibling_report_review_pending" in family["uncertainty_flags"]


def test_policy_license_versions_and_orders_link_related_reports(module, source):
    source["documents"][1]["title"] = "Synthetic Venezuela oil General License 44"
    source["documents"][2]["title"] = "Synthetic General License 44A under Executive Order 13850"
    source["documents"][3]["title"] = "Synthetic Executive Order 13850 oil amendment"
    _, assignment = module.grouped_documents(source["documents"], source["passages"], set())
    assert len({assignment[f"synthetic-document-{i}"] for i in (1, 2, 3)}) == 1


def test_company_policy_alias_is_not_fabricated_event_diversity(module, source):
    source["documents"][1]["title"] = "Synthetic Rosneft Trading oil sanctions notice"
    source["documents"][2]["title"] = "Synthetic Rosneft Trading petroleum designation notice"
    families, assignment = module.grouped_documents(source["documents"], source["passages"], set())
    assert assignment["synthetic-document-1"] == assignment["synthetic-document-2"]
    assert families[assignment["synthetic-document-1"]]["grouping_provenance"].endswith("not_verified_event_gold")


def test_recurring_same_quarter_report_is_not_split_into_daily_event_groups(module, source):
    source["documents"][1].update(source="DALLAS-OIL", available_by="2024-03-01")
    source["documents"][2].update(source="DALLAS-OIL", available_by="2024-03-15")
    source["documents"][3].update(source="DALLAS-OIL", available_by="2024-06-30")
    _, assignment = module.grouped_documents(source["documents"], source["passages"], set())
    assert assignment["synthetic-document-1"] == assignment["synthetic-document-2"]
    assert assignment["synthetic-document-1"] != assignment["synthetic-document-3"]


def test_source_packets_do_not_count_as_questions_or_labels(module, source):
    packets, families, stats = module.select_source_packets(**source)
    assert len(packets) == 5
    assert stats["actual_qa_questions"] == stats["actual_qa_labels"] == 0
    assert stats["source_packet_capacity_is_not_qa_yield"] is True
    assert stats["rejection_counts"]["prior_document_or_alias_exposure"] == 1
    for packet in packets:
        assert packet["questions_created"] == 0
        assert packet["human_gold"] is False
        assert packet["provenance"] == "accepted_source_packet_not_question_or_label"
        assert not module.FORBIDDEN_LABEL_FIELDS.intersection(set(packet) - {"human_gold"})
        assert not families[packet["event_family_id"]]["reserved_due_to_prior_exposure"]


def test_round_robin_packet_order_preserves_ids_and_does_not_collapse_to_one_stratum(module):
    packets = [dict(packet_id=f"a-{i}", source="A", publication_year="2024", primary_domain="production",
                    candidate_split="test") for i in range(3)]
    packets.append(dict(packet_id="b-0", source="B", publication_year="2024", primary_domain="production",
                        candidate_split="test"))
    ordered = module.balanced_packets(packets)
    assert [packet["packet_id"] for packet in ordered] == ["a-0", "b-0", "a-1", "a-2"]
    assert module.balanced_packets(list(reversed(packets))) == ordered


def test_held_related_family_passages_are_excluded_not_just_exact_documents(module, source):
    source["documents"][0]["title"] = "Synthetic Hurricane Ida activity"
    source["documents"][1]["title"] = "Synthetic Hurricane Ida final report"
    packets, _, stats = module.select_source_packets(**source)
    assert len(packets) == 4
    assert stats["rejection_counts"]["related_event_family_exposure_reservation"] == 1


def test_family_split_is_deterministic_and_whole_family(module, source):
    source["documents"][1]["title"] = "Synthetic Hurricane Ian update"
    source["documents"][2]["title"] = "Synthetic Hurricane Ian final report"
    packets, _, _ = module.select_source_packets(**source)
    reverse = dict(source, documents=list(reversed(source["documents"])), passages=list(reversed(source["passages"])))
    other, _, _ = module.select_source_packets(**reverse)
    assert packets == other
    groups = {}
    for packet in packets:
        groups.setdefault(packet["event_family_id"], set()).add(packet["candidate_split"])
    assert all(len(splits) == 1 for splits in groups.values())


def test_multiple_intact_same_page_paragraphs_are_allowed_without_splitting_sources(module, source):
    original = source["passages"][1]
    passage = deepcopy(original)
    passage["passage_id"] = "synthetic-extra-intact-paragraph"
    passage["text"] = original["text"].replace("three million", "four million")
    page = source["pages"][original["source_page_id"]]
    passage["char_start"] = len(page["text"]) + 1
    page["text"] += " " + passage["text"]
    passage["char_end"] = len(page["text"])
    source["passages"].append(passage)
    packets, _, _ = module.select_source_packets(**source)
    siblings = [packet for packet in packets if packet["evidence"][0]["document_id"] == original["document_id"]]
    assert len(siblings) == 2
    assert len({packet["event_family_id"] for packet in siblings}) == 1
    assert all(len(packet["evidence"]) == 1 for packet in siblings)
    assert all("same_document_focal_paragraphs_require_nonredundant_question_review" in packet["uncertainty_flags"] for packet in siblings)


def test_unknown_non_day_availability_is_held_not_assumed_valid(module, source):
    source["passages"][1]["available_by"] = "2024-03"
    _, _, stats = module.select_source_packets(**source)
    assert stats["rejection_counts"]["unknown_or_out_of_window_availability"] == 1


def test_scope_and_availability_holds_are_recorded(module, source):
    source["passages"][1]["available_by"] = None
    source["passages"][2]["text"] = "Natural gas information without standalone crude oil evidence."
    _, _, stats = module.select_source_packets(**source)
    assert stats["rejection_counts"]["unknown_or_out_of_window_availability"] == 1
    assert stats["rejection_counts"]["insufficient_standalone_oil_task_context"] == 1


def test_source_offset_mismatch_fails_without_repairing_source(module, source):
    source["passages"][1]["char_end"] -= 1
    with pytest.raises(ValueError, match="exact source-page"):
        module.select_source_packets(**source)


@pytest.mark.parametrize("limit", [0, 6, True, 1.5])
def test_no_unbounded_packet_proliferation(module, source, limit):
    with pytest.raises(ValueError, match="five bounded"):
        module.select_source_packets(**source, max_per_document=limit)


def test_actual_generated_question_and_author_spans_are_separate(module, source):
    packets, _, _ = module.select_source_packets(**source)
    questions, authoring = module.validate_generated_batch({"records": [generated(packets[0])]}, packets, expected_count=1)
    assert len(questions) == len(authoring) == 1
    assert set(questions[0]) == {"record_id", "question", "as_of", "event_group", "split"}
    assert not module.FORBIDDEN_LABEL_FIELDS.intersection(questions[0])
    assert "scenario_intent" not in questions[0]
    assert authoring[0]["human_gold"] is False
    anchor = authoring[0]["anchors"][0]
    page = source["pages"][anchor["source_page_id"]]
    assert page["text"][anchor["source_char_start"]:anchor["source_char_end"]] == anchor["text"]


@pytest.mark.parametrize("field", sorted(module_field for module_field in
                                        {"answerability", "answer_points", "qrels", "reference_answer", "event_status", "human_gold"}))
def test_generated_answers_and_labels_cannot_enter_question_cohort(module, source, field):
    packets, _, _ = module.select_source_packets(**source)
    row = generated(packets[0], **{field: "forbidden"})
    with pytest.raises(ValueError, match="labels|unknown"):
        module.validate_generated_batch({"records": [row]}, packets)


def test_missing_generated_questions_are_not_manufactured(module, source):
    packets, _, _ = module.select_source_packets(**source)
    with pytest.raises(ValueError, match="count"):
        module.validate_generated_batch({"records": []}, packets, expected_count=20)


def test_legacy_and_same_batch_duplicate_questions_fail(module, source):
    packets, _, _ = module.select_source_packets(**source)
    row = generated(packets[0])
    with pytest.raises(ValueError, match="duplicate|legacy"):
        module.validate_generated_batch({"records": [row]}, packets, existing_questions=[dict(question=row["question"].upper())])
    with pytest.raises(ValueError, match="duplicate|legacy"):
        module.validate_generated_batch({"records": [row, dict(row, local_id="q02")]}, packets)
    with pytest.raises(ValueError, match="duplicate|legacy"):
        module.validate_generated_batch({"records": [row]}, packets, existing_signatures=[module.normalized(row["question"])])


@pytest.mark.parametrize("question", ["What does this passage state about crude oil?",
                                      "What numerical oil claims appear in this evidence?",
                                      "Oil supply?"])
def test_generic_templates_are_not_formal_questions(module, source, question):
    packets, _, _ = module.select_source_packets(**source)
    with pytest.raises(ValueError, match="standalone|generic"):
        module.validate_generated_batch({"records": [generated(packets[0], question=question)]}, packets)


@pytest.mark.parametrize("case", ["bad_cutoff", "future_support", "wrong_quote", "unknown_packet", "invalid_intent", "no_anchor", "wrong_future_role"])
def test_generated_source_grounding_fails_closed(module, source, case):
    packets, _, _ = module.select_source_packets(**source)
    row = generated(packets[0])
    if case == "bad_cutoff":
        row["as_of"] = "2030-01-01"
    elif case == "future_support":
        row["as_of"] = "2024-02-29"
    elif case == "wrong_quote":
        row["anchor_spans"][0]["text"] = "Not in original text"
    elif case == "unknown_packet":
        row["packet_id"] = "unknown"
    elif case == "invalid_intent":
        row["scenario_intent"] = "sufficient"
    elif case == "no_anchor":
        row["anchor_spans"] = []
    else:
        row["anchor_spans"][0]["role"] = "future_leakage_control"
    with pytest.raises(ValueError):
        module.validate_generated_batch({"records": [row]}, packets)


def test_evidence_sufficiency_intent_is_not_an_insufficient_gold_label(module, source):
    packets, _, _ = module.select_source_packets(**source)
    row = generated(packets[0], as_of="2024-02-29", scenario_intent="evidence_sufficiency")
    row["anchor_spans"][0]["role"] = "future_leakage_control"
    questions, authoring = module.validate_generated_batch({"records": [row]}, packets)
    assert "answerability" not in questions[0]
    assert "answerability" not in authoring[0]
    assert authoring[0]["scenario_intent"] == "evidence_sufficiency"


def test_final_actual_counts_and_family_isolation_are_required(module):
    questions = [dict(record_id="q1", question="What crude export event happened in synthetic region one?",
                      split="development", event_group="family-a"),
                 dict(record_id="q2", question="What crude export event happened in synthetic region two?",
                      split="test", event_group="family-b")]
    authoring = [dict(record_id=row["record_id"]) for row in questions]
    counts = {"development": 1, "test": 1}
    result = module.validate_final_cohort(questions, authoring, required_counts=counts)
    assert result["actual_questions"] == 2
    assert result["final_annotation_freeze_ready"] is False
    with pytest.raises(ValueError, match="counts"):
        module.validate_final_cohort(questions, authoring)
    questions[1]["event_group"] = "family-a"
    with pytest.raises(ValueError, match="crosses"):
        module.validate_final_cohort(questions, authoring, required_counts=counts)


@pytest.mark.parametrize("name", ["../outside.json", "a/../../outside.json", "C:/outside.json", "a\\b.json", "/outside.json", "missing.json"])
def test_receipt_paths_cannot_escape_or_be_missing(module, tmp_path, name):
    with pytest.raises(ValueError):
        module.relative_file(tmp_path, name)


def test_original_hash_guard_refuses_changed_input_or_external_path(module, tmp_path):
    root = tmp_path / "canonical"
    root.mkdir()
    file = root / "input.txt"
    file.write_text("old immutable input", encoding="utf-8")
    guard = {str(file): module.file_hash(file)}
    module.assert_unchanged(guard, root)
    file.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        module.assert_unchanged(guard, root)
    outsider = tmp_path / "outside.txt"
    outsider.write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError, match="escaped"):
        module.assert_unchanged({str(outsider): module.file_hash(outsider)}, root)
