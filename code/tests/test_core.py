import json

import numpy as np
import pytest

from oil_evidence.annotations import agreement, validate_record
from oil_evidence.corpus import clean_page, date_from_text, month_from_filename, split_passages
from oil_evidence.evaluation import answer_f1, retrieval_metrics
from oil_evidence.llm import ModelClient, ModelUnavailable
from oil_evidence.query import query, validate_claims
from oil_evidence.retrieval import Retriever, permitted
from oil_evidence.store import latest_reviews, save_review


def p(pid, text, available="2025-01-01"):
    return dict(passage_id=pid, text=text, available_by=available, regions=["Middle East"], title=pid, url="https://example.org", page=1)


def test_future_and_unknown_dates_excluded_but_event_date_not_used():
    assert permitted(dict(p("old", "oil"), event_date="2030-01-01"), "2025-02-01")
    assert not permitted(p("future", "oil", "2026-01-01"), "2025-02-01")
    assert not permitted(p("unknown", "oil", None), "2025-02-01")
    assert permitted(p("unknown", "oil", None), "2025-02-01", temporal=False)
    with pytest.raises(ValueError):
        permitted(p("x", "oil"), "not-a-date")


def test_month_precision_is_conservative():
    assert month_from_filename("https://eia.gov/archives/feb24.pdf") == "2024-02-29"
    assert month_from_filename("https://opec.org/momr-march-2025.pdf") == "2025-03-31"
    assert date_from_text("March 12, 2025") == "2025-03-12"


def test_cleaning_does_not_drop_units_and_announcement():
    text = clean_page("The producer announced a cut of 1 million barrels per day.\n2025 123 124 125 126\nActual production has not yet changed.")
    assert "announced" in text and "1 million barrels per day" in text and "not yet" in text
    assert "123 124" not in text


def test_filter_applied_before_topk():
    passages = [p("new", "oil oil production announced cut", "2026-01-01"),
                p("old", "oil production was unchanged", "2024-01-01"), p("other", "gas prices falling")]
    class Encoder:
        def query_embed(self, queries):
            return [np.array([1., 0.])]
    r = Retriever(passages=passages, vectors=np.array([[1., 0.], [.8, .2], [.1, .9]]), encoder=Encoder())
    for mode in ["bm25", "dense", "hybrid"]:
        hits = r.search("oil production", mode=mode, top_k=1, as_of="2025-01-01")
        assert hits[0]["passage_id"] == "old"


def test_no_silent_dense_fallback():
    r = Retriever(passages=[p("a", "oil supply"), p("b", "gas demand")])
    with pytest.raises(ValueError, match="Dense index"):
        r.search("oil", mode="hybrid")


def test_invalid_citations_removed():
    good = dict(text="Production was unchanged.", citations=["a"])
    bad = dict(text="Production collapsed.", citations=["invented"])
    valid, removed = validate_claims([good, bad], [p("a", "Production was unchanged.")])
    assert valid == [good] and removed == [bad]


def test_extractive_does_not_claim_answerability_or_generation():
    r = Retriever(passages=[p("a", "Oil production was forecast to increase."), p("b", "Petroleum stocks were stable.")])
    result = query("oil production", mode="bm25", generation="extractive", verify=True, retriever=r, log=False)
    assert result["answerable"] is None and result["verification_method"] == "exact_quote"
    assert not result["model_calls"]
    assert query("oil", mode="bm25", as_of="2000-01-01", retriever=r, log=False)["answerable"] is False


def test_record_rejects_hallucinated_quote_and_future_evidence():
    record = dict(question="Was a cut implemented?", event_group="cut", answerability="sufficient", as_of="2024-01-01",
                  evidence_ids=["a"], quotes=[dict(passage_id="a", text="Cut implemented")], answer_points=["Yes"], qrels={"a": 2})
    errors = validate_record(record, {"a": p("a", "A cut was announced, but not implemented.")})
    assert any("cutoff" in e for e in errors) and any("exact span" in e for e in errors)


def test_agreement_excludes_model_assistance_and_requires_two_people():
    first = dict(record_id="q", annotator="a", mode="human_independent", payload=dict(answerability="partial", event_status="announced", qrels={"p": 2}))
    assisted = dict(first, annotator="b", mode="human_reviewed")
    assert agreement([first, assisted])["independently_double_labelled_records"] == 0
    second = dict(first, annotator="b", payload=dict(answerability="sufficient", event_status="announced", qrels={"p": 0}))
    report = agreement([first, second])
    assert report["fields"]["answerability"]["raw_agreement"] == 0
    assert not report["threshold_met"]


def test_review_history_preserved(tmp_path):
    path = tmp_path / "a.sqlite3"
    save_review("q", "person", "human_independent", {"answerability": "partial"}, path)
    save_review("q", "person", "human_reviewed", {"answerability": "sufficient"}, path)
    import sqlite3
    with sqlite3.connect(path) as db:
        assert db.execute("select count(*) from reviews").fetchone()[0] == 2
    assert latest_reviews(path)[0]["mode"] == "human_reviewed"


def test_suggestion_exposure_survives_restart_and_cannot_be_downgraded(tmp_path):
    from oil_evidence.store import mark_suggestion_view, has_suggestion_view
    path = tmp_path / "a.sqlite3"
    mark_suggestion_view("q", "person", path)
    assert has_suggestion_view("q", "person", path)
    save_review("q", "person", "human_independent", {}, path)
    assert latest_reviews(path)[0]["mode"] == "human_reviewed"


def test_freeze_refuses_insufficient_unlabelled_records(tmp_path):
    from oil_evidence.annotations import freeze
    path = tmp_path / "tiny.jsonl"
    path.write_text(json.dumps({"question":"x"}) + "\n")
    with pytest.raises(ValueError, match="at least 1200"):
        freeze(path)


def test_metrics_known_rankings_and_unjudged():
    m = retrieval_metrics(["x", "a", "b"], {"a": 2, "b": 1, "c": 1}, 3)
    assert m["precision_at_k"] == pytest.approx(2/3)
    assert m["pooled_recall_at_k"] == pytest.approx(2/3)
    assert m["mrr_at_k"] == .5
    assert m["judged_fraction"] == pytest.approx(2/3)
    assert answer_f1("oil production rose", "oil production rose") == 1
    assert answer_f1("oil", "gas") == 0


def test_disabled_model_fails_without_network():
    with pytest.raises(ModelUnavailable):
        ModelClient(dict(enabled=False, model="", verifier_model="", base_url="")).complete("system", "prompt")


def test_chunking_rejects_non_oil_text():
    assert not split_passages("Bananas and apples are delicious. " * 100)
    result = split_passages("Crude oil production was stable during the reporting period. " * 50)
    assert result and all(len(x.split()) <= 265 for x in result)


def test_pdf_chart_pages_are_not_prose_even_with_oil_labels():
    from oil_evidence.corpus import is_prose_page
    assert not is_prose_page("Oil stocks million barrels 0.8 2.2 1.8 1.5 0.0 1.0 2.0 3.0 4.0 5.0 " * 8)
    assert not is_prose_page("Crude oil production million barrels daily average monthly forecast " * 20)
    assert is_prose_page("Crude oil production increased in the region during the last quarter of the year. Inventories also increased because refinery maintenance reduced demand for crude oil over the same reporting period.")
