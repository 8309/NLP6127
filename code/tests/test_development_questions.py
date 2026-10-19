"""Development-question guards on synthetic fixtures, not annotation accuracy.

Only pure validation helpers are exercised. No live corpus, retrieval, encoder,
hosted model, annotation pipeline, acquisition or Git operation is invoked.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def validator(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Pure validation must not run pipeline operations")

    monkeypatch.setitem(sys.modules, "unify_corpus", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "build_staged_index", SimpleNamespace(
        assert_unchanged=forbidden, file_hashes=forbidden))
    path = Path(__file__).parents[1] / "scripts/prepare_development_questions.py"
    spec = importlib.util.spec_from_file_location("isolated_development_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scenario():
    passages, pages, documents, drafts = [], [], [], []
    types = ("fact", "temporal", "comparison", "insufficient")
    prefix = "Private synthetic header. "
    for number, kind in enumerate(types, 1):
        text = f"Synthetic oil source {number} states that crude exports were stable."
        document_id = f"fixture-document-{number}"
        passage_id = f"fixture-passage-{number}"
        page_id = f"fixture-source-page-{number}"
        passages.append(dict(passage_id=passage_id, document_id=document_id,
                             source_page_id=page_id, source="SYNTHETIC-OIL", page=1,
                             text=text, available_by="2024-01-01",
                             char_start=len(prefix), char_end=len(prefix) + len(text)))
        pages.append(dict(source_page_id=page_id, text=prefix + text + " Synthetic footer."))
        documents.append(dict(document_id=document_id, source="SYNTHETIC-OIL",
                              url=f"https://fixture.invalid/oil/{number}",
                              raw_sha256=f"{number:064x}"))
        drafts.append(dict(local_id=f"fixture{number:02d}", type=kind,
                           question=f"What does synthetic oil report {number} say about crude exports?",
                           question_zh=f"How does synthetic oil report {number} describe crude exports?",
                           as_of="2024-06-30", event_group=f"fixture-oil-episode-{number}",
                           topic="synthetic crude exports",
                           difficulty_reason_zh="Synthetic fixture for development-question protocol checks only.",
                           anchors=[dict(passage_id=passage_id,
                                         quote="crude exports were stable.",
                                         role="support_context")]))
    return dict(drafts=drafts, passages=passages, pages=pages, documents=documents,
                inventory=dict(blocked_document_ids=[], blocked_urls=[], blocked_raw_sha256=[]),
                old_questions=[], expected_count=4, expected_types={kind: 1 for kind in types})


def test_valid_distribution_returns_queries_without_labels_or_author_hints(validator, scenario):
    questions, authoring, stats = validator.validate(**scenario)
    assert len(questions) == len(authoring) == 4
    assert stats == dict(questions=4, scenario_counts=scenario["expected_types"],
                         event_groups=4, authoring_source_documents=4,
                         source_counts={"SYNTHETIC-OIL": 4}, authoring_anchors=4,
                         future_leakage_controls=0, exact_duplicate_questions=0,
                         legacy_document_overlap=0)
    for question in questions:
        assert set(question) == {"question_id", "question", "question_zh", "as_of",
                                 "event_group", "split", "type", "topic"}
        assert question["split"] == "development"
        assert not validator.LABEL_KEYS.intersection(question)
        assert not {"anchors", "difficulty_reason_zh", "selection_notes"}.intersection(question)
    for record in authoring:
        assert record["human_gold"] is False
        assert record["provenance"] == "codex_source_first_question_authoring_not_qrels"
        anchor = record["anchors"][0]
        passage = next(p for p in scenario["passages"] if p["passage_id"] == anchor["passage_id"])
        source = next(p for p in scenario["pages"] if p["source_page_id"] == anchor["source_page_id"])
        assert passage["text"][anchor["passage_char_start"]:
                               anchor["passage_char_start"] + len(anchor["quote"])] == anchor["quote"]
        assert source["text"][anchor["source_char_start"]:anchor["source_char_end"]] == anchor["quote"]


@pytest.mark.parametrize("mutation", ["wrong_count", "wrong_distribution"])
def test_count_and_type_distribution_fail_closed(validator, scenario, mutation):
    if mutation == "wrong_count":
        scenario["drafts"].pop()
    else:
        scenario["drafts"][0]["type"] = "temporal"
    with pytest.raises(ValueError, match="count or scenario distribution"):
        validator.validate(**scenario)


@pytest.mark.parametrize("language", ["question", "question_zh"])
def test_normalized_duplicate_questions_rejected(validator, scenario, language):
    scenario["drafts"][1][language] = scenario["drafts"][0][language].upper() + "  !!!"
    with pytest.raises(ValueError, match="duplicate/legacy question"):
        validator.validate(**scenario)


@pytest.mark.parametrize("language", ["question", "question_zh"])
def test_old_normalized_question_rejected(validator, scenario, language):
    scenario["old_questions"] = [{language: scenario["drafts"][0][language].upper() + " !!"}]
    with pytest.raises(ValueError, match="duplicate/legacy question"):
        validator.validate(**scenario)


def test_nfkc_casefold_and_punctuation_normalization(validator):
    assert validator.normalized(" ＯＩＬ: Export １２？ ") == validator.normalized("oil export12")
    assert validator.normalized("naïve, oil?") == validator.normalized("naïve oil")


@pytest.mark.parametrize("language", ["question", "question_zh"])
def test_empty_normalized_question_rejected(validator, scenario, language):
    scenario["drafts"][0][language] = "   !!!？"
    with pytest.raises(ValueError, match="Empty/duplicate/legacy"):
        validator.validate(**scenario)


@pytest.mark.parametrize("key", sorted({"answerability", "answer_points", "reference_answer", "qrels",
                                       "event_status", "human_label", "human_reviewed", "model_verification"}))
def test_answer_label_injection_rejected(validator, scenario, key):
    scenario["drafts"][0][key] = "synthetic unwanted label"
    with pytest.raises(ValueError, match="contains answer labels"):
        validator.validate(**scenario)


@pytest.mark.parametrize("question", ["What does this evidence state?", "What does THIS PASSAGE state?",
                                    "What does the above text state?", "What does the given passage state?"])
def test_nonstandalone_evidence_template_rejected(validator, scenario, question):
    scenario["drafts"][0]["question"] = question
    with pytest.raises(ValueError, match="not standalone"):
        validator.validate(**scenario)


@pytest.mark.parametrize("cutoff", ["2017-12-31", "2026-10-06", "2024-02-30", "20240630",
                                  "2024-06-30T00:00:00", "not-a-date"])
def test_bad_cutoff_rejected(validator, scenario, cutoff):
    scenario["drafts"][0]["as_of"] = cutoff
    with pytest.raises(ValueError):
        validator.validate(**scenario)


@pytest.mark.parametrize("cutoff", ["2018-01-01", "2026-10-05"])
def test_fixed_window_boundary_cutoffs_supported(validator, scenario, cutoff):
    scenario["drafts"][0]["as_of"] = cutoff
    scenario["passages"][0]["available_by"] = cutoff
    assert len(validator.validate(**scenario)[0]) == 4


@pytest.mark.parametrize("target", ["passage", "document", "source_page"])
def test_unknown_anchor_identity_fails_closed(validator, scenario, target):
    if target == "passage":
        scenario["drafts"][0]["anchors"][0]["passage_id"] = "fixture-missing-passage"
    elif target == "document":
        scenario["passages"][0]["document_id"] = "fixture-missing-document"
    else:
        scenario["passages"][0]["source_page_id"] = "fixture-missing-page"
    with pytest.raises((KeyError, ValueError)):
        validator.validate(**scenario)


@pytest.mark.parametrize("quote", ["", "   ", None, 42, "Crude exports were stable.",
                                 "crude exports were unstable."])
def test_quote_requires_exact_nonempty_passage_span(validator, scenario, quote):
    scenario["drafts"][0]["anchors"][0]["quote"] = quote
    with pytest.raises(ValueError, match="not an exact accepted-passage span"):
        validator.validate(**scenario)


@pytest.mark.parametrize("mutation", ["changed_page", "start_shift", "end_shift"])
def test_passage_exact_source_offsets_rejected_when_changed(validator, scenario, mutation):
    if mutation == "changed_page":
        scenario["pages"][0]["text"] = scenario["pages"][0]["text"].replace("crude exports", "total liquids")
    elif mutation == "start_shift":
        scenario["passages"][0]["char_start"] += 1
    else:
        scenario["passages"][0]["char_end"] -= 1
    with pytest.raises(ValueError, match="does not match exact source-page offsets"):
        validator.validate(**scenario)


@pytest.mark.parametrize("alias", ["document_id", "url", "raw_sha256"])
def test_known_legacy_source_alias_rejected(validator, scenario, alias):
    doc = scenario["documents"][0]
    keys = {"document_id": "blocked_document_ids", "url": "blocked_urls", "raw_sha256": "blocked_raw_sha256"}
    value = doc[alias]
    if alias == "url":
        doc["url"] = "HTTPS://FIXTURE.INVALID/oil/1#different-fragment"
    scenario["inventory"][keys[alias]] = [value]
    with pytest.raises(ValueError, match="Legacy source reused"):
        validator.validate(**scenario)


def test_legacy_alias_in_second_anchor_is_not_ignored(validator, scenario):
    scenario["drafts"][0]["anchors"].append(deepcopy(scenario["drafts"][1]["anchors"][0]))
    scenario["inventory"]["blocked_document_ids"] = [scenario["documents"][1]["document_id"]]
    with pytest.raises(ValueError, match="Legacy source reused"):
        validator.validate(**scenario)


@pytest.mark.parametrize("available", [None, "", "not-a-date", "20240101", "2024-02-30"])
def test_unknown_or_noncanonical_source_availability_rejected(validator, scenario, available):
    scenario["passages"][0]["available_by"] = available
    with pytest.raises(ValueError):
        validator.validate(**scenario)


def test_future_source_can_be_marked_explicit_leakage_control_without_answer_label(validator, scenario):
    scenario["passages"][0]["available_by"] = "2024-07-01"
    scenario["drafts"][0]["anchors"][0]["role"] = "future_leakage_control"
    questions, authoring, stats = validator.validate(**scenario)
    assert stats["future_leakage_controls"] == 1
    assert authoring[0]["anchors"][0]["role"] == "future_leakage_control"
    assert not validator.LABEL_KEYS.intersection(questions[0])
    assert "anchors" not in questions[0]


@pytest.mark.parametrize("available", ["2024-01-01", "2024-06-30"])
def test_future_control_must_be_strictly_after_cutoff(validator, scenario, available):
    scenario["passages"][0]["available_by"] = available
    scenario["drafts"][0]["anchors"][0]["role"] = "future_leakage_control"
    with pytest.raises(ValueError, match="role/cutoff mismatch"):
        validator.validate(**scenario)


def test_future_source_cannot_be_support_context(validator, scenario):
    scenario["passages"][0]["available_by"] = "2024-07-01"
    with pytest.raises(ValueError, match="role/cutoff mismatch"):
        validator.validate(**scenario)


def test_unknown_anchor_role_rejected(validator, scenario):
    scenario["drafts"][0]["anchors"][0]["role"] = "reference_answer"
    with pytest.raises(ValueError, match="role/cutoff mismatch"):
        validator.validate(**scenario)


def test_shared_document_cannot_be_assigned_different_event_groups(validator, scenario):
    scenario["drafts"][1]["anchors"] = deepcopy(scenario["drafts"][0]["anchors"])
    with pytest.raises(ValueError, match="different event groups"):
        validator.validate(**scenario)


def test_shared_document_can_reserve_same_event_group(validator, scenario):
    scenario["drafts"][1]["anchors"] = deepcopy(scenario["drafts"][0]["anchors"])
    scenario["drafts"][1]["event_group"] = scenario["drafts"][0]["event_group"]
    stats = validator.validate(**scenario)[2]
    assert stats["event_groups"] == 3
    assert stats["authoring_source_documents"] == 3


@pytest.mark.parametrize("url, expected", [
    ("HTTPS://EXAMPLE.ORG/Oil?report=June#chapter", "https://example.org/Oil?report=June"),
    ("https://example.org/oil?report=june", "https://example.org/oil?report=june"),
    ("https://example.org:443/oil#two", "https://example.org:443/oil"),
])
def test_url_normalization_preserves_case_sensitive_path_and_query(validator, url, expected):
    assert validator.normalized_url(url) == expected


def test_legacy_exposure_includes_generation_candidates_inputs_group_and_probes_with_transitive_aliases(validator):
    original = [dict(passage_id=f"passage-{name}", document_id=name)
                for name in ("generated", "relevant", "candidate")]
    records = [dict(event_group="document-group-only", generation_passage_id="passage-generated",
                    evidence_ids=["passage-relevant"], candidate_pool_ids=["passage-candidate"])]
    inputs = [dict(evidence=[dict(document_id="input-only")])]
    probes = [dict(anchors=[dict(document_id="probe-only")])]
    documents = [dict(document_id=name, url=f"https://fixture.invalid/{name}", raw_sha256=name)
                 for name in ("generated", "relevant", "candidate", "group-only", "input-only", "probe-only")]
    documents.extend([
        dict(document_id="url-alias", url="HTTPS://FIXTURE.INVALID/candidate#fragment", raw_sha256="shared-alias-bytes"),
        dict(document_id="raw-alias", url="https://fixture.invalid/raw-alias", raw_sha256="shared-alias-bytes"),
        dict(document_id="transitive-url-alias", url="https://fixture.invalid/raw-alias#anchor", raw_sha256="final-alias-bytes"),
        dict(document_id="unrelated", url="https://fixture.invalid/unrelated", raw_sha256="unrelated-bytes"),
    ])
    assert validator.legacy_document_ids(records, inputs, original, probes, documents) == {
        "generated", "relevant", "candidate", "group-only", "input-only", "probe-only",
        "url-alias", "raw-alias", "transitive-url-alias"}


def test_standalone_legacy_quote_passage_is_full_evidence_exposure(validator):
    records = [dict(quotes=[dict(passage_id="quoted-passage", text="synthetic quote")])]
    original = [dict(passage_id="quoted-passage", document_id="quoted-document")]
    documents = [dict(document_id="quoted-document", url="https://fixture.invalid/quoted", raw_sha256="quoted-bytes")]
    assert validator.legacy_document_ids(records, [], original, [], documents) == {"quoted-document"}


def test_unknown_legacy_passage_fails_closed(validator):
    with pytest.raises((KeyError, ValueError)):
        validator.legacy_document_ids([dict(candidate_pool_ids=["unknown-passage"])], [], [], [], [])


def test_relative_file_accepts_existing_descendant_only(validator, tmp_path):
    folder = tmp_path / "fixtures"
    folder.mkdir()
    target = folder / "input.json"
    target.write_text("{}", encoding="utf-8")
    assert validator.relative_file(tmp_path, "fixtures/input.json") == target.resolve()


@pytest.mark.parametrize("name", [None, 42, "", ".", "..", "../outside.json", "nested/../input.json",
                               "./input.json", "nested//input.json", "nested/", "/absolute.json",
                               r"nested\input.json", "C:/input.json", "input.json:stream"])
def test_unsafe_relative_path_refused(validator, tmp_path, name):
    with pytest.raises(ValueError):
        validator.relative_file(tmp_path, name)


def test_missing_file_and_directory_refused(validator, tmp_path):
    with pytest.raises(ValueError):
        validator.relative_file(tmp_path, "missing.json")
    (tmp_path / "folder").mkdir()
    with pytest.raises(ValueError):
        validator.relative_file(tmp_path, "folder")


def test_resolved_symlink_escape_refused(validator, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    link = base / "escaped.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Host cannot create fixture symlink: {error}")
    with pytest.raises(ValueError, match="escapes its directory"):
        validator.relative_file(base, "escaped.json")
