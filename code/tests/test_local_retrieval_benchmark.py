"""Synthetic diagnostics for bounded local retrieval, never performance claims."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import benchmark_local_retrieval as module


def fixture_runtime():
    text = "Synthetic crude evidence."
    row = dict(passage_id="fixture-p", source_page_id="fixture-page", text=text,
        available_by="2024-01-01", regions=[], char_start=0, char_end=len(text))
    queries, lock = [], threading.Lock()
    class Encoder:
        def query_embed(self, texts):
            with lock:
                queries.extend(texts)
            yield [1.0]
    retriever = SimpleNamespace(encoder=Encoder())
    def search(question, **kwargs):
        list(retriever.encoder.query_embed([question]))
        return [dict(row, rank=1, score=1.0)]
    retriever.search = search
    rt = SimpleNamespace(retriever=retriever, passages={row["passage_id"]:row}, pages={row["source_page_id"]:dict(text=text)})
    cases = [dict(record_id=f"fixture-q-{i}", question=f"Synthetic query {i}?", as_of="2024-01-01", split="development") for i in range(4)]
    return rt, cases, queries


def test_every_query_embeds_independently_and_shared_encoder_is_restored():
    runtime, cases, queries = fixture_runtime()
    original = runtime.retriever.encoder
    result = module.benchmark(runtime, cases)
    assert [row["local_workers"] for row in result] == [1,2,4]
    assert len(queries) == 3*(len(cases)+1)  # Three real warmups and 12 timed local encodes.
    assert runtime.retriever.encoder is original
    for condition in result:
        assert condition["queries"] == 4 and condition["measured_cohort_wall_seconds"] > 0
        assert condition["worker_retrieval_p95_seconds"] >= condition["worker_retrieval_median_seconds"]
        assert all(row["fixed_burst_response_seconds"] >= row["worker_retrieval_seconds"] for row in condition["per_query"])


def test_test_questions_and_unbounded_client_counts_rejected():
    runtime, cases, _ = fixture_runtime()
    with pytest.raises(ValueError, match="DEVELOPMENT"):
        module.benchmark(runtime, [dict(cases[0], split="test")])
    with pytest.raises(ValueError, match="worker counts"):
        module.benchmark(runtime, cases, concurrency=(8,))


def test_changed_rankings_or_bad_provenance_rejected_encoder_restored():
    runtime, cases, _ = fixture_runtime()
    original = runtime.retriever.encoder
    def broken(*args, **kwargs):
        return [dict(next(iter(runtime.passages.values())), text="Changed source.")]
    runtime.retriever.search = broken
    with pytest.raises(ValueError, match="differs"):
        module.benchmark(runtime, cases)
    assert runtime.retriever.encoder is original
