"""Synthetic parity and blinding tests; do not contribute real labels/counts."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from oil_evidence.retrieval import Retriever


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("candidate_retrieval_test", scripts / "retrieve_machine_questions.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class SyntheticEncoder:
    def query_embed(self, texts):
        return iter(np.array([0.8, 0.2], dtype=np.float32) for _ in texts)


@pytest.fixture
def corpus():
    rows, pages = [], {}
    texts = ["Libya crude oil exports and production rose in the east.",
             "A pipeline disruption stopped crude oil exports from Libya.",
             "Oil refinery throughput changed and crude imports fell.",
             "Nigeria crude production was reported as a forecast.",
             "Future Libya crude oil exports were not known in March.",
             "Unknown date crude export report requires careful treatment.",
             "Shipping constraints affected crude deliveries in Asia."]
    for index, text in enumerate(texts):
        pid, page_id = f"synthetic-passage-{index}", f"synthetic-page-{index}"
        rows.append(dict(passage_id=pid, document_id=f"synthetic-document-{index}", text=text,
                         source_page_id=page_id, char_start=3, char_end=3 + len(text),
                         available_by="2024-05-01" if index == 4 else None if index == 5 else "2024-02-01",
                         title="Synthetic fixture", source="SYNTHETIC", url="https://example.invalid", page=1))
        pages[page_id] = dict(text="ABC" + text)
    vectors = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [1., 0.], [0.7, 0.3], [0.2, 0.8]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return rows, pages, vectors


def query(cutoff="2024-03-01"):
    return dict(record_id="synthetic-query", question="How did Libya's crude oil exports change?",
                as_of=cutoff, event_group="synthetic-family", split="test")


@pytest.mark.parametrize("cutoff", ["2024-03-01", "2024-06-01", "2018-01-01"])
def test_score_reuse_is_exact_core_ranking_parity(module, corpus, cutoff):
    rows, pages, vectors = corpus
    retriever = Retriever(rows, vectors, SyntheticEncoder())
    target = query(cutoff)
    rankings = module.rank_all(retriever, target, np.array([0.8, 0.2], dtype=np.float32))
    originals = {row["passage_id"]: row for row in rows}
    module.validate_rankings(rankings, target, originals, pages)
    for mode in module.MODES:
        for temporal in (True, False):
            expected = retriever.search(target["question"], mode=mode, top_k=5, as_of=cutoff, temporal=temporal)
            assert rankings[f"{mode}:{int(temporal)}"] == expected


def test_pool_hides_ranks_modes_groups_split_and_authoring_hints(module, corpus):
    rows, _, vectors = corpus
    target = query()
    rankings = module.rank_all(Retriever(rows, vectors, SyntheticEncoder()), target, [0.8, 0.2])
    originals = {row["passage_id"]: row for row in rows}
    pool = module.blinded_pool(target, rankings, originals)
    assert set(pool) == {"record_id", "question", "as_of", "evidence"}
    assert all(not {"rank", "score", "mode", "bm25_score", "dense_score", "anchor_spans", "qrels"} & set(row)
               for row in pool["evidence"])
    reversed_ranks = {key: list(reversed(hits)) for key, hits in rankings.items()}
    assert module.blinded_pool(target, reversed_ranks, originals) == pool
    assert any(row["available_by"] is None or row["available_by"] > target["as_of"] for row in pool["evidence"])


@pytest.mark.parametrize("top_k", [1, 10, True, 5.0])
def test_new_pool_depth_locked_without_touching_old_protocol(module, corpus, top_k):
    rows, _, vectors = corpus
    with pytest.raises(ValueError, match="depth"):
        module.rank_all(Retriever(rows, vectors, SyntheticEncoder()), query(), [0.8, 0.2], top_k=top_k)


def test_future_hit_in_temporal_on_or_bad_source_offsets_fails(module, corpus):
    rows, pages, vectors = corpus
    originals = {row["passage_id"]: row for row in rows}
    target = query()
    rankings = module.rank_all(Retriever(rows, vectors, SyntheticEncoder()), target, [0.8, 0.2])
    broken = deepcopy(rankings)
    broken["bm25:1"][0]["text"] = "Altered source"
    with pytest.raises(ValueError, match="provenance"):
        module.validate_rankings(broken, target, originals, pages)
    bad_pages = deepcopy(pages)
    bad_pages[rankings["bm25:1"][0]["source_page_id"]]["text"] = "Missing exact quote"
    with pytest.raises(ValueError, match="offsets"):
        module.validate_rankings(rankings, target, originals, bad_pages)


def test_invalid_embedding_fails(module, corpus):
    rows, _, vectors = corpus
    with pytest.raises(ValueError, match="embedding"):
        module.rank_all(Retriever(rows, vectors, SyntheticEncoder()), query(), [float("nan"), 0.2])
