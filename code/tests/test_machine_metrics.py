"""Synthetic ranking checks; no live model calls or accuracy claims."""
from copy import deepcopy
import math

import pytest

from oil_evidence.machine_metrics import evaluate_rankings, paired_bootstrap, ranked_metrics


def fixture_rows():
    decisions = [dict(record_id="q1", status="accepted", provenance="machine_adjudicated", human_gold=False,
                      label=dict(qrels={"p1": 2, "p2": 1, "p3": 0}))]
    rankings = [dict(question_id="q1", mode=mode, as_of="2024-01-01", elapsed_seconds=.1,
                     evidence=[dict(passage_id=pid, rank=rank, available_by="2023-01-01")
                               for rank, pid in enumerate(pids, 1)])
                for mode, pids in (("bm25", ["p3", "p1"]), ("dense", ["p2", "p3"]), ("hybrid", ["p1", "p2"]))]
    return rankings, decisions


def test_exact_small_ranking():
    result = ranked_metrics(["p2", "p1", "unknown"], {"p1": 2, "p2": 1, "p3": 0}, k=3)
    assert result["precision_at_k"] == 2 / 3
    assert result["pooled_recall_at_k"] == 1
    assert result["mrr_at_k"] == 1
    assert result["judged_fraction"] == 2 / 3
    assert result["ndcg_at_k"] == pytest.approx((1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))


def test_short_empty_and_zero_positive_pools():
    assert ranked_metrics(["p"], {"p": 2}, k=10)["precision_at_k"] == .1
    result = ranked_metrics([], {"p": 0})
    assert result["pooled_recall_at_k"] is None and result["ndcg_at_k"] is None
    assert result["mrr_at_k"] == 0 and result["judged_fraction"] == 0


@pytest.mark.parametrize("ids,qrels,k", [(["p", "p"], {"p": 2}, 1), ([""], {}, 1),
                                         (["p"], {"p": True}, 1), (["p"], {"p": 3}, 1),
                                         (["p"], {"p": 2}, 0), (["p"], {"p": 2}, True)])
def test_invalid_ranking_or_judgments(ids, qrels, k):
    with pytest.raises(ValueError):
        ranked_metrics(ids, qrels, k)


def test_complete_paired_summary():
    rows, decisions = fixture_rows()
    report = evaluate_rankings(rows, decisions, ["q1"], k=2, bootstrap_samples=100)
    assert report["human_gold"] is False and report["ranked_runs"] == 3
    assert report["summary"]["hybrid"]["precision_at_k"] == 1
    delta = report["paired_comparisons"]["hybrid_minus_bm25"]["precision_at_k"]
    assert delta["mean_delta"] == .5 and delta["bootstrap_95_percent_interval"] == [.5, .5]


@pytest.mark.parametrize("case", ["missing_mode", "duplicate_mode", "bad_mode", "missing_label",
                                  "duplicate_label", "human_gold", "unresolved", "bad_rank", "bad_latency"])
def test_coverage_and_provenance_fail_closed(case):
    rows, decisions = deepcopy(fixture_rows())
    if case == "missing_mode":
        rows.pop()
    elif case == "duplicate_mode":
        rows.append(deepcopy(rows[0]))
    elif case == "bad_mode":
        rows[0]["mode"] = "other"
    elif case == "missing_label":
        decisions.clear()
    elif case == "duplicate_label":
        decisions.append(deepcopy(decisions[0]))
    elif case == "human_gold":
        decisions[0]["human_gold"] = True
    elif case == "unresolved":
        decisions[0]["status"] = "unresolved"
    elif case == "bad_rank":
        rows[0]["evidence"][0]["rank"] = 2
    else:
        rows[0]["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError):
        evaluate_rankings(rows, decisions, ["q1"], bootstrap_samples=100)


def test_bootstrap_reproducible_and_undefined_pairs_excluded():
    first = paired_bootstrap([0, None, .2], [.4, .9, .3], samples=100)
    assert first == paired_bootstrap([0, None, .2], [.4, .9, .3], samples=100)
    assert first["paired_queries"] == 2 and first["mean_delta"] == pytest.approx(.25)
    assert paired_bootstrap([None], [1], samples=100)["mean_delta"] is None
    with pytest.raises(ValueError):
        paired_bootstrap([1], [], samples=100)
