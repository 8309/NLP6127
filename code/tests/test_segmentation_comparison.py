"""Isolated source-native comparison checks: no live data, model or network."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

import oil_evidence.retrieval as retrieval
from oil_evidence.store import digest, read_jsonl, words, write_json, write_jsonl


def load_script():
    path = Path(__file__).parents[1] / "scripts/compare_segmentation.py"
    spec = importlib.util.spec_from_file_location("isolated_segmentation_comparison", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SETTINGS = {"retrieval": {"embedding_model": "test-model", "top_k": 5, "candidate_k": 50, "rrf_k": 60}}
QUOTE = "Crude oil production increased by two million barrels per day during the reporting period."
OTHER_QUOTE = "Emergency crude oil deliveries were rerouted through alternative export terminals after the reported disruption."


def passage(pid, document, text, available="2023-01-01", page=1):
    return dict(passage_id=pid, document_id=document, source="EIA", title=document,
                url="https://example.test/" + document, page=page,
                published_date=available, available_by=available,
                date_precision="day" if available else "unknown", date_source="fixture",
                regions=[], text=text, word_count=len(words(text)))


def make_index(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "passages.jsonl", rows)
    (root / "index").mkdir(exist_ok=True)
    vectors = np.array([[1, 0] if "production" in row["text"].lower() else [0, 1] for row in rows], dtype=np.float32)
    np.save(root / "index/vectors.npy", vectors, allow_pickle=False)
    write_json(root / "index/manifest.json", {
        "model": "test-model", "corpus_sha256": digest((root / "passages.jsonl").read_bytes()),
        "passage_ids": [row["passage_id"] for row in rows], "dimension": 2,
        "vectors_sha256": digest((root / "index/vectors.npy").read_bytes()),
        "onnx_sha256": {"fixture.onnx": "test"}})


class FakeEncoder:
    def __init__(self, callback=None):
        self.calls, self.callback = [], callback

    def query_embed(self, questions):
        self.calls.extend(questions)
        if self.callback:
            self.callback()
        return iter(np.array([1, 0], dtype=np.float32) for _ in questions)


@pytest.fixture
def case(tmp_path, monkeypatch):
    module = load_script()
    monkeypatch.setattr(module, "config", lambda: SETTINGS)
    monkeypatch.setattr(retrieval, "config", lambda: SETTINGS)
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    first, second = QUOTE.split("million", 1)
    b = [passage("b-a1", "a", first.strip()), passage("b-a2", "a", "million" + second),
         passage("b-b", "b", "Petroleum shipments reached oil export terminals during the reporting period."),
         passage("b-future", "future", "Future crude oil production increased.", "2026-01-01"),
         passage("b-unknown", "unknown", "Undated crude oil production increased.", None),
         passage("b-extra", "old-only", "Extra crude oil production page.")]
    c = [passage("c-a", "a", QUOTE),
         passage("c-b1", "b", "Petroleum shipments reached oil export terminals"),
         passage("c-b2", "b", "during the reporting period."),
         passage("c-future", "future", "Future crude oil production increased.", "2026-01-01"),
         passage("c-unknown", "unknown", "Undated crude oil production increased.", None),
         passage("c-extra", "new-only", OTHER_QUOTE)]
    make_index(baseline, b)
    make_index(candidate, c)
    seeds = tmp_path / "questions.json"
    write_json(seeds, [
        {"question": "How did crude oil production change?", "as_of": "2024-12-31", "type": "fact"},
        {"question": "What crude oil supply risks were known before collection?", "as_of": "2017-12-31", "type": "insufficient"}])
    pages = []
    for document, text, available in [
        ("a", QUOTE, "2023-01-01"),
        ("b", "Petroleum shipments reached oil export terminals during the reporting period. " + OTHER_QUOTE, "2023-01-01"),
        ("future", "Future crude oil production increased.", "2026-01-01"),
        ("unknown", "Undated crude oil production increased.", None),
        ("new-only", OTHER_QUOTE, "2023-01-01")]:
        page = passage("ignored", document, text, available)
        page.pop("passage_id")
        page["source_page_id"] = "source-" + document
        pages.append(page)
    source = tmp_path / "source_pages.jsonl"
    write_jsonl(source, pages)
    probes = tmp_path / "probes.json"
    write_json(probes, {"source_pages_sha256": digest(source.read_bytes()),
                       "selection_method": "source_only_before_retrieval", "human_gold": False,
                       "provenance": "codex_source_grounded", "probes": [
        {"probe_id": "p-spanning", "question": "What quantity was reported for the production increase?", "as_of": "2024-12-31",
         "anchors": [{"document_id": "a", "page": 1, "quote": QUOTE}]},
        {"probe_id": "p-missing", "question": "How were emergency petroleum deliveries rerouted?", "as_of": "2024-12-31",
         "anchors": [{"document_id": "b", "page": 1, "quote": OTHER_QUOTE}]},
        {"probe_id": "p-outside-scope", "question": "Which route change was described outside the shared source set?", "as_of": "2024-12-31",
         "anchors": [{"document_id": "new-only", "page": 1, "quote": OTHER_QUOTE}]}]})
    return module, baseline, candidate, seeds, source, probes


def run_case(case, tmp_path, **kwargs):
    module, baseline, candidate, seeds, source, probes = case
    return module.compare(baseline, candidate, seeds, tmp_path / "comparison.json",
                          source_pages_path=source, probes_path=probes, encoder=kwargs.pop("encoder", FakeEncoder()), **kwargs)


def test_three_modes_fixed_scope_missing_probes_retained_inputs_unchanged(case, tmp_path):
    module, baseline, candidate, seeds, source, probes = case
    before = {str(p): p.read_bytes() for root in (baseline, candidate) for p in root.rglob("*") if p.is_file()}
    data_reference = retrieval.DATA
    encoder = FakeEncoder()
    report = run_case(case, tmp_path, encoder=encoder)
    assert retrieval.DATA == data_reference
    assert report["inputs_unchanged"] is True
    assert report["queries"] == 5 and report["seed_queries"] == 2 and report["probe_queries"] == 3
    assert report["source_scope"]["shared_comparable_pages"] == 4
    assert report["source_scope"]["baseline_common_passages"] == 5
    assert report["source_scope"]["candidate_common_passages"] == 5
    assert {row["document_id"] for row in report["source_scope"]["baseline_only_pages"]} == {"old-only"}
    assert {row["document_id"] for row in report["source_scope"]["candidate_only_pages"]} == {"new-only"}
    assert len(encoder.calls) == 5  # one vector per unique question, shared across versions/modes
    assert report["answer_accuracy"] is None and report["exhaustive_retrieval_recall"] is None
    assert report["human_gold"] is False and report["external_model_calls"] == 0
    for version in ("baseline", "candidate"):
        assert len(report["results"][version]) == 15
        for row in report["results"][version]:
            assert row["metrics"]["temporal_errors"] == 0
            assert {hit["document_id"] for hit in row["evidence"]} <= {"a", "b"}
            if row["as_of"] == "2017-12-31":
                assert row["metrics"]["returned_passages"] == 0
                assert row["metrics"]["context_words"] == 0
            if row["query_id"] != "seed-02":
                assert row["metrics"]["returned_passages"] == 3
                assert row["metrics"]["unique_pages"] == 2
                assert row["metrics"]["duplicate_page_slots"] == 1
            if row["query_id"] in ("p-missing", "p-outside-scope"):
                assert row["metrics"]["assembled_anchor_support_coverage"] == 0
    spanning = next(row for row in report["results"]["baseline"] if row["query_id"] == "p-spanning" and row["mode"] == "dense")
    assert spanning["metrics"]["single_passage_anchor_support_coverage"] == 0
    assert spanning["metrics"]["assembled_anchor_support_coverage"] == 1
    assert spanning["metrics"]["source_anchor_reciprocal_rank"] == 0.5
    assert any(not row["in_comparable_scope"] for row in report["source_anchor_catalog"])
    assert {str(p): p.read_bytes() for root in (baseline, candidate) for p in root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("problem", ["corpus_hash", "vector_hash", "model", "order", "dimension", "nan", "zero", "unnormalized", "duplicate_ids", "words", "date"])
def test_index_and_metadata_guards(case, problem):
    module, baseline, *_ = case
    manifest_path = baseline / "index/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if problem == "corpus_hash":
        manifest["corpus_sha256"] = "bad"
    elif problem == "vector_hash":
        manifest["vectors_sha256"] = "bad"
    elif problem == "model":
        manifest["model"] = "different"
    elif problem == "order":
        manifest["passage_ids"].reverse()
    elif problem == "dimension":
        manifest["dimension"] = 3
    elif problem in ("nan", "zero", "unnormalized"):
        vector_path = baseline / "index/vectors.npy"
        vectors = np.load(vector_path)
        vectors[0] = {"nan": [np.nan, 1], "zero": [0, 0], "unnormalized": [2, 0]}[problem]
        np.save(vector_path, vectors)
        manifest["vectors_sha256"] = digest(vector_path.read_bytes())
    else:
        rows = read_jsonl(baseline / "passages.jsonl")
        if problem == "duplicate_ids":
            rows[1]["passage_id"] = rows[0]["passage_id"]
        elif problem == "words":
            rows[0]["word_count"] += 1
        elif problem == "date":
            rows[0]["available_by"] = "2023-02-30"
        write_jsonl(baseline / "passages.jsonl", rows)
        manifest["corpus_sha256"] = digest((baseline / "passages.jsonl").read_bytes())
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        module.load_corpus(baseline, "test-model")


def test_changed_source_metadata_excluded_and_not_repaired_in_place(case):
    module, baseline, candidate, *_ = case
    b = module.load_corpus(baseline, "test-model")
    c = module.load_corpus(candidate, "test-model")
    c.metadata[("a", 1)] = ("OTHER",) + c.metadata[("a", 1)][1:]
    scope, report = module.shared_scope(b, c)
    assert ("a", 1) not in scope
    assert report["metadata_mismatch_pages_excluded"] == [{"document_id": "a", "page": 1}]


def test_source_metadata_must_match_comparable_active_page(case, tmp_path):
    module, _, _, _, source, _ = case
    rows = read_jsonl(source)
    rows[0]["available_by"] = "2024-01-01"
    write_jsonl(source, rows)
    # No probe hash involved: validates provenance independently of probes.
    _, baseline, candidate, seeds, *_ = case
    with pytest.raises(ValueError, match="Frozen source metadata"):
        module.compare(baseline, candidate, seeds, tmp_path / "comparison.json", source_pages_path=source, encoder=FakeEncoder())


@pytest.mark.parametrize("problem", ["hash", "selection", "human", "provenance", "short_quote", "invented_quote", "missing_page", "duplicate_id", "future", "answer_in_question", "duplicate_anchor"])
def test_probe_guards(case, problem):
    module, _, _, _, source, probes_path = case
    payload = json.loads(probes_path.read_text())
    probe = payload["probes"][0]
    if problem == "hash":
        payload["source_pages_sha256"] = "bad"
    elif problem == "selection":
        payload["selection_method"] = "from_successful_retrieval_hits"
    elif problem == "human":
        payload["human_gold"] = True
    elif problem == "provenance":
        payload["provenance"] = "independent_human"
    elif problem == "short_quote":
        probe["anchors"][0]["quote"] = "Crude oil production"
    elif problem == "invented_quote":
        probe["anchors"][0]["quote"] = "Crude oil production reached seven million barrels per day according to the supplied source."
    elif problem == "missing_page":
        probe["anchors"][0]["page"] = 9
    elif problem == "duplicate_id":
        payload["probes"][1]["probe_id"] = probe["probe_id"]
    elif problem == "future":
        probe["as_of"] = "2022-12-31"
    elif problem == "answer_in_question":
        probe["question"] = "Please confirm: " + QUOTE
    elif problem == "duplicate_anchor":
        probe["anchors"].append(dict(probe["anchors"][0]))
    write_json(probes_path, payload)
    with pytest.raises(ValueError):
        module.load_probes(probes_path, module.load_source_pages(source), digest(source.read_bytes()))


def test_input_change_during_embedding_is_detected_no_report_written(case, tmp_path):
    _, _, candidate, *_ = case
    def mutate():
        path = candidate / "passages.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="input changed during retrieval"):
        run_case(case, tmp_path, encoder=FakeEncoder(mutate))
    assert not (tmp_path / "comparison.json").exists()


def test_query_change_during_embedding_is_detected(case, tmp_path):
    _, _, _, seeds, *_ = case
    def mutate():
        seeds.write_bytes(seeds.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="query/source input changed"):
        run_case(case, tmp_path, encoder=FakeEncoder(mutate))


def test_invalid_query_encoder_has_no_result(case, tmp_path):
    class BadEncoder:
        def query_embed(self, questions):
            return iter([[np.nan, 1]])
    with pytest.raises(ValueError, match="invalid vectors"):
        run_case(case, tmp_path, encoder=BadEncoder())


def test_existing_report_is_not_overwritten(case, tmp_path):
    output = tmp_path / "comparison.json"
    output.write_bytes(b"original")
    with pytest.raises(ValueError, match="not overwritten"):
        run_case(case, tmp_path)
    assert output.read_bytes() == b"original"


@pytest.mark.parametrize("path", ["index/new.json", "annotations/new.json"])
def test_output_cannot_enter_index_or_annotations(case, path):
    module, baseline, candidate, seeds, source, probes = case
    with pytest.raises(ValueError, match="protected"):
        module.compare(baseline, candidate, seeds, candidate / path, encoder=FakeEncoder())
    assert not (candidate / path).exists()


def test_unknown_and_future_dates_fail_closed_in_metric_check(case):
    module, *_ = case
    query = {"question": "What production?", "as_of": "2024-01-01"}
    for available in (None, "2026-01-01"):
        hit = dict(passage("p", "a", QUOTE, available), rank=1)
        with pytest.raises(ValueError, match="Temporal retrieval leak"):
            module.query_metrics(query, [hit], {"p": 0})


def test_literal_quote_matching_does_not_erase_numbers_or_units(case):
    module, *_ = case
    hit = dict(passage("p", "a", QUOTE.replace("two million", "three thousand")), rank=1)
    support = module.quote_support({"document_id": "a", "page": 1, "quote": QUOTE}, [hit], {"p": 0})
    assert not support["assembled_page_supported"]
    assert not support["single_passage_supported"]


def test_all_anchors_rank_requires_each_anchor_not_any(case):
    module, *_ = case
    query = {"question": "What changed?", "as_of": "2024-01-01", "anchors": [
        {"document_id": "a", "page": 1, "quote": QUOTE},
        {"document_id": "b", "page": 1, "quote": OTHER_QUOTE}]}
    first = dict(passage("p1", "a", QUOTE), rank=1)
    second = dict(passage("p2", "b", OTHER_QUOTE), rank=2)
    order = {"p1": 0, "p2": 1}
    missing = module.query_metrics(query, [first], order)
    assert missing["assembled_anchor_support_coverage"] == 0.5
    assert missing["source_anchor_reciprocal_rank"] == 0
    complete = module.query_metrics(query, [first, second], order)
    assert complete["all_anchors_supported"] is True
    assert complete["source_anchor_reciprocal_rank"] == 0.5


def test_no_probes_seeds_are_only_diagnostics(case, tmp_path):
    module, baseline, candidate, seeds, source, *_ = case
    report = module.compare(baseline, candidate, seeds, tmp_path / "comparison.json", encoder=FakeEncoder())
    assert report["probe_queries"] == 0 and report["source_anchor_catalog"] == []
    assert len(report["results"]["baseline"]) == 6
    assert all(not row["kind"] == "source_anchor_probe" for row in report["results"]["candidate"])


def test_local_model_files_must_match_both_indexes(case, tmp_path, monkeypatch):
    module, baseline, candidate, *_ = case
    monkeypatch.setattr(module, "ROOT", tmp_path)
    model_path = tmp_path / "models/fixture.onnx"
    model_path.parent.mkdir()
    model_path.write_bytes(b"fixture onnx")
    b = module.load_corpus(baseline, "test-model")
    c = module.load_corpus(candidate, "test-model")
    b.manifest["onnx_sha256"] = c.manifest["onnx_sha256"] = {"fixture.onnx": digest(model_path.read_bytes())}
    assert module.verify_model_files(b, c) == {str(model_path.resolve()): digest(model_path.read_bytes())}
    c.manifest["onnx_sha256"] = {"fixture.onnx": "other"}
    with pytest.raises(ValueError, match="identical"):
        module.verify_model_files(b, c)
    c.manifest["onnx_sha256"] = b.manifest["onnx_sha256"]
    model_path.write_bytes(b"changed onnx")
    with pytest.raises(ValueError, match="differs"):
        module.verify_model_files(b, c)


def test_source_page_duplicate_and_invalid_page_are_rejected(case):
    module, _, _, _, source, _ = case
    rows = read_jsonl(source)
    write_jsonl(source, rows + [rows[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        module.load_source_pages(source)
    with pytest.raises(ValueError, match="positive integer"):
        module.page_key({"document_id": "a", "page": True})
