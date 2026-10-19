"""Synthetic guards for blinded development retrieval, not relevance scoring."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setitem(sys.modules, "unify_corpus", SimpleNamespace(PROJECT=Path("original"), DATA=Path("original/data"), RUN=Path("original/run")))
    monkeypatch.setitem(sys.modules, "build_staged_index", SimpleNamespace(assert_unchanged=lambda value: None, file_hashes=lambda value: {}))
    monkeypatch.setitem(sys.modules, "check_staged_index", SimpleNamespace(run_file=lambda base, name: base / name))
    spec = importlib.util.spec_from_file_location("isolated_development_retrieval", Path(__file__).parents[1] / "scripts/run_development_retrieval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def query():
    return dict(question_id="fixture-001", question="What happened to crude exports?",
                question_zh="Synthetic translated question", as_of="2024-06-30")


@pytest.fixture
def corpus():
    passages, pages = {}, {}
    for number in range(1, 4):
        text = f"Synthetic crude exports source {number}."
        pid, page_id = f"fixture-passage-{number}", f"fixture-page-{number}"
        row = dict(passage_id=pid, source_page_id=page_id, document_id=f"fixture-doc-{number}",
                   text=text, char_start=7, char_end=7 + len(text), available_by="2024-01-01",
                   source="SYNTHETIC", page=1, regions=["Middle East"],
                   url=f"https://fixture.invalid/{number}", segmentation="synthetic")
        passages[pid] = row
        pages[page_id] = dict(text="Header " + text + " Footer")
    return passages, pages


def hit(row, rank=1):
    return dict(row, rank=rank, score=1.0 / rank, bm25_score=2.0, dense_score=0.5)


def test_four_field_queries_are_valid(runner, query):
    assert runner.validate_queries([query], expected_count=1) == [query]


@pytest.mark.parametrize("key", ["type", "anchors", "answerability", "event_group", "design_intent", "qrels", "topic", "difficulty_reason_zh"])
def test_query_hints_and_labels_rejected(runner, query, key):
    query[key] = "unwanted hint"
    with pytest.raises(ValueError, match="labels or authoring hints"):
        runner.validate_queries([query], expected_count=1)


def test_query_count_and_duplicate_ids_rejected(runner, query):
    with pytest.raises(ValueError, match="count"):
        runner.validate_queries([query])
    with pytest.raises(ValueError, match="Duplicate"):
        runner.validate_queries([query, deepcopy(query)], expected_count=2)


@pytest.mark.parametrize("cutoff", ["2017-12-31", "2026-10-06", "2024-6-3", "not-a-date"])
def test_invalid_cutoff_rejected(runner, query, cutoff):
    query["as_of"] = cutoff
    with pytest.raises(ValueError):
        runner.validate_queries([query], expected_count=1)


def test_rank_union_is_deduplicated_and_blinded(runner, query, corpus):
    passages, _ = corpus
    rows = list(passages.values())
    rankings = {"bm25": [hit(rows[0]), hit(rows[1], 2)], "dense": [hit(rows[2]), hit(rows[0], 2)], "hybrid": [hit(rows[1])]}
    pool = runner.blinded_pool(query, rankings, passages)
    assert set(pool) == runner.QUERY_FIELDS | {"evidence"}
    assert len(pool["evidence"]) == 3
    assert {e["passage_id"] for e in pool["evidence"]} == set(passages)
    for evidence in pool["evidence"]:
        assert set(evidence).issubset(runner.EVIDENCE_FIELDS)
        assert not {"rank", "score", "bm25_score", "dense_score", "mode", "segmentation"}.intersection(evidence)
    reversed_rankings = {mode: list(reversed(values)) for mode, values in rankings.items()}
    assert runner.blinded_pool(query, reversed_rankings, passages) == pool


def test_empty_query_results_not_removed(runner, query):
    pool = runner.blinded_pool(query, {mode: [] for mode in runner.MODES}, {})
    assert pool == dict(query, evidence=[])


def test_all_modes_required_for_pool(runner, query):
    with pytest.raises(ValueError, match="all three modes"):
        runner.blinded_pool(query, {"bm25": []}, {})


def test_original_offsets_ranks_and_filters_validate(runner, query, corpus):
    passages, pages = corpus
    results = [hit(list(passages.values())[0])]
    runner.validate_hits(results, query, passages, pages, region="Middle East")
    runner.validate_hits([], query, passages, pages)


@pytest.mark.parametrize("mutation", ["duplicate", "rank", "score", "text", "future", "offset", "region", "unknown"])
def test_hit_corruption_rejected(runner, query, corpus, mutation):
    passages, pages = corpus
    results = [hit(list(passages.values())[0])]
    if mutation == "duplicate":
        results *= 2
    elif mutation == "rank":
        results[0]["rank"] = 2
    elif mutation == "score":
        results[0]["score"] = float("nan")
    elif mutation == "text":
        results[0]["text"] = "Changed text"
    elif mutation == "future":
        query["as_of"] = "2023-12-31"
    elif mutation == "offset":
        pages[results[0]["source_page_id"]]["text"] = "Changed source page"
    elif mutation == "unknown":
        results[0]["passage_id"] = "unknown"
    region = "Europe" if mutation == "region" else None
    with pytest.raises(ValueError):
        runner.validate_hits(results, query, passages, pages, region=region)


def test_latency_summary_uses_only_mode_records(runner):
    records = [dict(mode=mode, elapsed_seconds=seconds) for mode in runner.MODES for seconds in (1.0, 3.0)]
    summary = runner.latency_summary(records)
    for value in summary.values():
        assert value["queries"] == 2
        assert value["mean_seconds"] == value["median_seconds"] == 2
        assert value["total_seconds"] == 4
        assert value["sequential_queries_per_second"] == 0.5


@pytest.mark.parametrize("value", ["../escape", "CON", "nul", "bad/id", "ab", "a" * 82])
def test_unsafe_run_ids_rejected(runner, value):
    with pytest.raises(ValueError, match="Invalid run"):
        runner.safe_id(value)


def test_context_restored_after_error(runner):
    old = runner.pipeline.PROJECT, runner.pipeline.DATA, runner.pipeline.RUN
    with pytest.raises(RuntimeError):
        with runner.canonical_context(Path("fixture"), "fixture-corpus") as run:
            assert run == Path("fixture/data/unified_runs/fixture-corpus")
            raise RuntimeError("fixture failure")
    assert (runner.pipeline.PROJECT, runner.pipeline.DATA, runner.pipeline.RUN) == old
