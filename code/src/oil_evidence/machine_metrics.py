"""Ranked evaluation against explicitly machine-adjudicated, pooled judgments.

This module never invokes a model or reads project data. These metrics estimate
retrieval against a judged pool, not exhaustive corpus relevance or human gold.
"""
from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import mean, median


MODES = ("bm25", "dense", "hybrid")
METRICS = ("precision_at_k", "pooled_recall_at_k", "mrr_at_k", "ndcg_at_k", "judged_fraction")


def ranked_metrics(ranked_ids, qrels, k=10):
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if (not isinstance(ranked_ids, list) or any(not isinstance(pid, str) or not pid for pid in ranked_ids)
            or len(set(ranked_ids)) != len(ranked_ids)):
        raise ValueError("Ranking contains duplicate or invalid IDs")
    if (not isinstance(qrels, dict) or any(not isinstance(pid, str) or not pid
            or type(grade) is not int or grade not in (0, 1, 2) for pid, grade in qrels.items())):
        raise ValueError("Invalid pooled relevance judgments")
    ranked = ranked_ids[:k]
    positive = {pid for pid, grade in qrels.items() if grade > 0}
    hits = sum(pid in positive for pid in ranked)
    dcg = sum((2 ** qrels.get(pid, 0) - 1) / math.log2(rank + 2) for rank, pid in enumerate(ranked))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return dict(precision_at_k=hits / k, pooled_recall_at_k=hits / len(positive) if positive else None,
                mrr_at_k=next((1 / (rank + 1) for rank, pid in enumerate(ranked) if pid in positive), 0.0),
                ndcg_at_k=dcg / idcg if idcg else None,
                judged_fraction=sum(pid in qrels for pid in ranked) / len(ranked) if ranked else 0.0,
                retrieved_count=len(ranked), positive_pool_count=len(positive))


def paired_bootstrap(left, right, samples=2000, seed=6127):
    """Resample paired queries, excluding pairs with an undefined metric."""
    if len(left) != len(right) or type(samples) is not int or samples < 100:
        raise ValueError("Paired arrays and at least 100 bootstrap samples are required")
    differences = [b - a for a, b in zip(left, right, strict=True) if a is not None and b is not None]
    if not differences:
        return dict(paired_queries=0, mean_delta=None, bootstrap_95_percent_interval=None)
    generator = random.Random(seed)
    draws = sorted(mean(generator.choices(differences, k=len(differences))) for _ in range(samples))
    return dict(paired_queries=len(differences), mean_delta=mean(differences),
                bootstrap_95_percent_interval=[draws[int(.025 * (samples - 1))], draws[int(.975 * (samples - 1))]],
                bootstrap_samples=samples, seed=seed,
                note="Query-resampling interval; synthetic question and machine-label biases are not captured.")


def evaluate_rankings(rankings, decisions, expected_ids, k=10, bootstrap_samples=2000):
    """Require complete, unique mode/query coverage; never silently subset."""
    expected = set(expected_ids)
    if not expected or len(expected) != len(expected_ids):
        raise ValueError("Expected question IDs must be nonempty and unique")
    labels = {}
    for decision in decisions:
        record_id = decision.get("record_id")
        if record_id in labels or record_id not in expected:
            raise ValueError("Adjudication contains duplicate or unexpected records")
        if (decision.get("status") != "accepted" or decision.get("provenance") != "machine_adjudicated"
                or decision.get("human_gold") is not False):
            raise ValueError("Only accepted, explicitly machine-adjudicated records may be evaluated")
        labels[record_id] = decision["label"]
    if set(labels) != expected:
        raise ValueError("Incomplete accepted adjudication coverage")
    grouped, seen, scored = defaultdict(list), set(), []
    for row in rankings:
        record_id = row.get("record_id", row.get("question_id"))
        mode = row.get("mode")
        if mode not in MODES or record_id not in expected or (record_id, mode) in seen:
            raise ValueError("Duplicate or unexpected ranked query/mode")
        seen.add((record_id, mode))
        duration = row.get("elapsed_seconds")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Missing or invalid measured retrieval latency")
        evidence = row.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError("Ranking evidence must be a list")
        if any(not isinstance(p, dict) or type(p.get("rank")) is not int or p["rank"] != rank
               for rank, p in enumerate(evidence, 1)):
            raise ValueError("Ranking positions are not sequential")
        cutoff = row.get("as_of")
        future = sum(bool(cutoff and p.get("available_by") and p["available_by"] > cutoff) for p in evidence[:k])
        metrics = ranked_metrics([p["passage_id"] for p in evidence], labels[record_id]["qrels"], k)
        result = dict(record_id=record_id, mode=mode, elapsed_seconds=duration,
                      future_evidence_count=future, metrics=metrics)
        grouped[mode].append(result)
        scored.append(result)
    if seen != {(record_id, mode) for record_id in expected for mode in MODES}:
        raise ValueError("Incomplete three-mode ranked coverage")
    summary = {}
    for mode in MODES:
        rows = grouped[mode]
        durations = sorted(row["elapsed_seconds"] for row in rows)
        summary[mode] = dict(queries=len(rows), median_seconds=median(durations),
                             p95_seconds=durations[math.ceil(.95 * len(durations)) - 1],
                             sequential_queries_per_second=len(rows) / sum(durations) if sum(durations) else None,
                             future_evidence_count=sum(row["future_evidence_count"] for row in rows))
        for metric in METRICS:
            values = [row["metrics"][metric] for row in rows if row["metrics"][metric] is not None]
            summary[mode][metric] = mean(values) if values else None
            summary[mode][metric + "_defined_queries"] = len(values)
    by_key = {(row["record_id"], row["mode"]): row for row in scored}
    comparisons = {}
    ordered = sorted(expected)
    for left, right in (("bm25", "hybrid"), ("dense", "hybrid")):
        comparisons[f"{right}_minus_{left}"] = {
            metric: paired_bootstrap([by_key[(qid, left)]["metrics"][metric] for qid in ordered],
                                     [by_key[(qid, right)]["metrics"][metric] for qid in ordered],
                                     samples=bootstrap_samples) for metric in METRICS[:4]}
    return dict(provenance="machine_adjudicated_pooled_retrieval_evaluation", human_gold=False,
                questions=len(expected), ranked_runs=len(scored), k=k, summary=summary,
                paired_comparisons=comparisons, per_query=scored,
                limitations=["Machine adjudication is not human gold or a measured correctness guarantee.",
                             "Recall denominator includes only positive judgments in the fixed evidence pool.",
                             "Unjudged hits contribute zero gain; judged_fraction is reported separately.",
                             "Precision divides by k even when fewer results are returned.",
                             "A zero-positive pool has undefined recall/nDCG; those query counts are explicit.",
                             "Sequential retrieval throughput is not concurrent or generative throughput."])
