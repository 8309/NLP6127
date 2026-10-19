"""Retrieve a frozen candidate question snapshot using the unchanged local index.

New protocol: top five in each BM25/dense/hybrid mode, temporal both on and off.
The annotation pool is their deterministic unranked union, never augmented from
question-authoring anchors. This creates evidence candidates, not relevance or
answer labels. Local query embeddings are batched once, with timing separated
from ranking and whole-run wall time; no end-to-end per-query latency is claimed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

import aggregate_machine_questions as aggregation
import run_codex_annotation as native
import run_development_retrieval as verified
from build_staged_index import assert_unchanged, file_hashes
from oil_evidence import machine_evaluation as protocol
from oil_evidence.retrieval import Retriever, permitted
from oil_evidence.store import config, words

TOP_K = 5
PROTOCOL = "machine-candidate-top5-three-mode-two-temporal-pool-v1"
MODES = ("bm25", "dense", "hybrid")
RANKING_KEYS = {f"{mode}:{int(temporal)}" for mode in MODES for temporal in (True, False)}


def rank_all(retriever, query, embedding, *, top_k=TOP_K, settings=None):
    """Reuse exact core score/rank rules, computing each score array only once."""
    if top_k != TOP_K or type(top_k) is not int:
        raise ValueError("The new candidate protocol fixes retrieval depth at five")
    aggregation.check_query(query)
    settings = config()["retrieval"] if settings is None else settings
    vector = np.asarray(embedding)
    if vector.ndim != 1 or vector.shape[0] != retriever.vectors.shape[1] or not np.all(np.isfinite(vector)):
        raise ValueError("Invalid local query embedding")
    vector = vector / max(np.linalg.norm(vector), 1e-12)
    sparse = retriever.bm25.get_scores(words(query["question"]))
    dense = retriever.vectors @ vector
    rows, result = retriever.passages, {}
    for temporal in (True, False):
        eligible = [index for index, row in enumerate(rows) if permitted(row, query["as_of"], None, temporal)]
        candidates = {
            "bm25": sorted(eligible, key=lambda index: (-sparse[index], rows[index]["passage_id"]))[:max(top_k, settings["candidate_k"])],
            "dense": sorted(eligible, key=lambda index: (-dense[index], rows[index]["passage_id"]))[:max(top_k, settings["candidate_k"])],
        }
        for mode in MODES:
            if mode == "hybrid":
                scores = {}
                for ranked in candidates.values():
                    for rank, index in enumerate(ranked, 1):
                        scores[index] = scores.get(index, 0) + 1 / (settings["rrf_k"] + rank)
            else:
                values = sparse if mode == "bm25" else dense
                scores = {index: float(values[index]) for index in candidates[mode]}
            selected = sorted(scores, key=lambda index: (-scores[index], rows[index]["passage_id"]))[:top_k]
            result[f"{mode}:{int(temporal)}"] = [dict(rows[index], rank=rank, score=scores[index],
                bm25_score=float(sparse[index]), dense_score=None if mode == "bm25" else float(dense[index]))
                for rank, index in enumerate(selected, 1)]
    return result


def validate_rankings(rankings, query, originals, pages):
    if set(rankings) != RANKING_KEYS:
        raise ValueError("Missing ranking mode or temporal condition")
    for key, hits in rankings.items():
        temporal = key.endswith(":1")
        if len(hits) > TOP_K or len({hit["passage_id"] for hit in hits}) != len(hits):
            raise ValueError("Retrieval depth or uniqueness differs")
        for rank, hit in enumerate(hits, 1):
            original = originals.get(hit["passage_id"])
            if (original is None or any(hit.get(key) != value for key, value in original.items())
                    or hit.get("rank") != rank or not np.isfinite(hit.get("score", np.nan))
                    or (temporal and not permitted(original, query["as_of"], temporal=True))):
                raise ValueError("Retrieval source provenance, score or date differs")
            page = pages.get(hit["source_page_id"])
            start, end = hit["char_start"], hit["char_end"]
            if (page is None or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(page["text"]) or page["text"][start:end] != hit["text"]):
                raise ValueError("Retrieved evidence does not match exact original page offsets")


def blinded_pool(query, rankings, originals):
    aggregation.check_query(query)
    if set(rankings) != RANKING_KEYS:
        raise ValueError("Pool requires all six ranking conditions")
    ids = {hit["passage_id"] for hits in rankings.values() for hit in hits}
    ordered = sorted(ids, key=lambda identifier: (protocol.sha256(
        (PROTOCOL + "\0" + query["record_id"] + "\0" + identifier).encode("utf-8")), identifier))
    row = dict(record_id=query["record_id"], question=query["question"], as_of=query["as_of"],
               evidence=[originals[identifier] for identifier in ordered])
    return protocol.clean_packet(row)["payload"]


def load_cohort(root, cohort_path):
    root, cohort_path = Path(root).resolve(), Path(cohort_path).resolve()
    allowed = (root / "data/machine_runs/question-cohorts").resolve()
    if not cohort_path.is_relative_to(allowed) or cohort_path == allowed:
        raise ValueError("Use a unique immutable aggregated candidate-question snapshot")
    receipt = native.read_json(cohort_path / "receipt.json")
    if (receipt.get("status") != "actual_question_candidates_aggregated_not_gold_or_frozen_evaluation"
            or receipt.get("original_inputs_unchanged") is not True or receipt.get("human_gold") is not False
            or receipt.get("qa_annotations_created") != 0 or receipt.get("source_selection_precedes_retrieval") is not True):
        raise ValueError("Expected an audited unlabelled candidate question cohort")
    protected = {}
    for name, expected in receipt["artifact_sha256"].items():
        path = native.contained_file(cohort_path, name)
        if protocol.sha256(path.read_bytes()) != expected:
            raise ValueError("Candidate cohort artifact changed")
        protected[str(path)] = expected
    protected[str(cohort_path / "receipt.json")] = protocol.sha256((cohort_path / "receipt.json").read_bytes())
    questions = protocol.parse_jsonl((cohort_path / "questions.jsonl").read_bytes(), "cohort questions")
    protocol._validate_questions(questions)
    if dict(Counter(row["split"] for row in questions)) != receipt["actual_counts"]:
        raise ValueError("Cohort actual counts differ from receipt")
    for row in questions:
        aggregation.check_query(row)
    return questions, protected, receipt


def run(root, cohort_path):
    started = perf_counter()
    root, cohort_path = Path(root).resolve(), Path(cohort_path).resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    queries, cohort_protected, cohort_receipt = load_cohort(root, cohort_path)
    _, _, passages, pages, vectors, manifest, protected, bindings = verified.load_verified_inputs(
        root, "oil-dev50-v1", "semantic-page-v2-buffer")
    protected.update(cohort_protected)
    implementation = file_hashes([Path(__file__), Path(aggregation.__file__), Path(verified.__file__),
                                   Path(protocol.__file__), Path(native.__file__)])
    protected.update(implementation)
    assert_unchanged(protected)
    parent = root / "data/machine_runs/candidate-retrieval"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / (native.utc_now().replace(":", "-") + "-" + uuid4().hex[:10])
    output.mkdir(exist_ok=False)
    native.write_new(output / "inputs.json", protocol.canonical_bytes(dict(protocol=PROTOCOL,
        top_k=TOP_K, retrieval_modes=list(MODES), temporal_conditions=[True, False],
        cohort=str(cohort_path), actual_counts=cohort_receipt["actual_counts"],
        bindings=bindings, protected_sha256=protected, original_receipts_not_rebound=True,
        current_implementation_sha256=implementation, author_anchor_injection=False)))
    native.write_new(output / "retrieval_script.py", Path(__file__).read_bytes())
    try:
        from fastembed import TextEmbedding
        tick = perf_counter()
        encoder = TextEmbedding(model_name=manifest["model"], cache_dir=str(root / "models"),
                                threads=4, local_files_only=True)
        retriever = Retriever(passages=passages, vectors=vectors, encoder=encoder)
        initialization_seconds = perf_counter() - tick
        tick = perf_counter()
        embeddings = list(encoder.query_embed([query["question"] for query in queries], batch_size=32))
        embedding_seconds = perf_counter() - tick
        if len(embeddings) != len(queries):
            raise ValueError("Local embedding count differs from actual query count")
        original = {row["passage_id"]: row for row in passages}
        pool_counts, future_counts, rank_times = [], [], []
        with (output / "rankings.jsonl").open("xb") as rank_file, (output / "annotation_pool.jsonl").open("xb") as pool_file:
            for index, (query, embedding) in enumerate(zip(queries, embeddings, strict=True), 1):
                tick = perf_counter()
                rankings = rank_all(retriever, query, embedding)
                rank_times.append(perf_counter() - tick)
                validate_rankings(rankings, query, original, pages)
                pool = blinded_pool(query, rankings, original)
                pool_counts.append(len(pool["evidence"]))
                future_counts.append(sum(not permitted(row, query["as_of"], temporal=True) for row in pool["evidence"]))
                rank_file.write(protocol.canonical_bytes(dict(record_id=query["record_id"],
                    ranking_seconds_excluding_embedding=rank_times[-1], conditions=rankings)) + b"\n")
                pool_file.write(protocol.canonical_bytes(pool) + b"\n")
                if index % 25 == 0 or index == len(queries):
                    print(f"Candidate retrieval {index}/{len(queries)}: three modes, two temporal conditions, top5", flush=True)
        assert_unchanged(protected)
        summary = dict(status="candidate_evidence_pools_created_not_annotations_or_accuracy_results", output=str(output),
            protocol=PROTOCOL, top_k=TOP_K, actual_queries=len(queries), actual_counts=cohort_receipt["actual_counts"],
            retrieval_conditions_per_question=6, ranking_records=len(queries) * 6,
            original_inputs_unchanged=True, local_encoder=manifest["model"], encoder_downloads=0,
            initialization_seconds=initialization_seconds, batch_query_embedding_seconds=embedding_seconds,
            ranking_seconds_excluding_embedding=sum(rank_times), wall_seconds=perf_counter() - started,
            mean_union_evidence_count=float(np.mean(pool_counts)), maximum_union_evidence_count=max(pool_counts),
            pooled_future_or_unknown_date_evidence=sum(future_counts),
            author_anchor_injection=False, evidence_order_encodes_rank=False, machine_annotations_created=0,
            human_gold=False, hosted_model_calls=0,
            artifact_sha256={path.name: protocol.sha256(path.read_bytes()) for path in output.iterdir() if path.is_file()},
            limitations=["Finite depth5 pooled judgments cannot establish exhaustive corpus recall.",
                         "Future/unknown-date candidates are intentionally retained for temporal validity annotation.",
                         "Score reuse and batched embeddings do not measure interactive per-query end-to-end latency.",
                         "Questions and pools are not reference answers or relevance gold."])
        native.write_new(output / "receipt.json", protocol.canonical_bytes(summary))
        print(json.dumps({key: value for key, value in summary.items() if key != "artifact_sha256"}, indent=2), flush=True)
        return output, summary
    except Exception as error:
        native.write_new(output / "failure.json", protocol.canonical_bytes(dict(status="failed_not_complete",
                         error_type=type(error).__name__, error=str(error))))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    args = parser.parse_args()
    run(args.canonical_root, args.cohort)
