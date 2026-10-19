import importlib.util
import json
from pathlib import Path
import sys

import pytest
from oil_evidence.store import digest, read_jsonl, write_json, write_jsonl


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    spec = importlib.util.spec_from_file_location("review_module", Path(__file__).parents[1] / "scripts/passage_review.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.DATA = tmp_path
    p = dict(passage_id="p", document_id="d", title="Oil forecast", source="EIA", page=1,
             published_date="2025-01-01", available_by="2025-01-01", date_precision="day",
             url="https://example.org", text="EIA forecasts production of 10 million barrels per day in 2026.")
    label = dict(passage_id="p", topics=["production_supply"], oil_relevance="direct", entities=["EIA"],
                 event_status="forecast", events=[], quantities=[], evidence_spans=["EIA forecasts production"],
                 summary_zh="Production is forecast", needs_review=True, review_reasons=["Forecast needs review"],
                 provenance="codex_model_passage_annotation", human_reviewed=False)
    write_jsonl(tmp_path / "passages.jsonl", [p])
    write_jsonl(tmp_path / "raw/d.html", [])
    write_jsonl(tmp_path / "documents.jsonl", [dict(document_id="d", raw_path="raw/d.html", raw_sha256=digest(b""))])
    write_jsonl(tmp_path / "annotations/passage_labels.jsonl", [label])
    h = digest((tmp_path / "passages.jsonl").read_bytes())
    snapshot = tmp_path / f"annotations/passage_label_runs/{h[:16]}/published/base/labels.jsonl"
    write_jsonl(snapshot, [label])
    write_json(tmp_path / "reports/passage_annotation.json", dict(version="base", corpus_sha256=h,
               labels_sha256=digest((tmp_path / "annotations/passage_labels.jsonl").read_bytes())))
    root = Path(m.prepare()["root"])
    return m, root, p, label


def blind_complete(m, root, label):
    write_json(root / "outputs/batch-001.json", dict(labels=[{k: label[k] for k in m.CORE}]))
    write_json(root / "audit/batch-001.json", dict(actor="blind model", output_sha256=digest((root / "outputs/batch-001.json").read_bytes())))


def judgment():
    return dict(n=1, outcome="confirmed", choice="old", note="The source explicitly forecasts production; no ambiguity flag is needed.",
                remaining_reasons=[], exclude=False, support=[["focus", "EIA forecasts production"]])


def test_blinded_inputs_contain_no_old_labels(prepared):
    m, root, _, _ = prepared
    p = read_jsonl(root / "inputs/batch-001.jsonl")[0]
    assert not {"old_label", "needs_review", "review_reasons", "summary_zh"} & p.keys()
    assert m.progress(root)["reviewed"] == 0
    with pytest.raises(ValueError, match="All blind"):
        m.compare(root)
    with pytest.raises(ValueError, match="incomplete"):
        m.publish(root)


def test_fabricated_context_quote_rejected(prepared):
    m, root, _, label = prepared
    blind_complete(m, root, label)
    m.compare(root)
    r = judgment()
    r["support"] = [["focus", "Russia produced 20"]]
    write_json(root / "judge.json", [r])
    with pytest.raises(ValueError, match="exact context"):
        m.import_judgments(root, 1, root / "judge.json", "model")


def test_unresolved_requires_reason(prepared):
    m, root, _, label = prepared
    blind_complete(m, root, label)
    m.compare(root)
    r = judgment()
    r["outcome"] = "unresolved"
    write_json(root / "judge.json", [r])
    with pytest.raises(ValueError, match="Unresolved"):
        m.import_judgments(root, 1, root / "judge.json", "model")


def test_tampered_comparison_rejected(prepared):
    m, root, _, label = prepared
    blind_complete(m, root, label)
    m.compare(root)
    p = root / "comparisons/batch-001.jsonl"
    rows = read_jsonl(p)
    rows[0]["old_label"]["summary_zh"] = "Tampered"
    write_jsonl(p, rows)
    write_json(root / "judge.json", [judgment()])
    with pytest.raises(ValueError, match="Comparison"):
        m.import_judgments(root, 1, root / "judge.json", "model")


def test_review_publish_preserves_base_and_is_machine_only(prepared):
    m, root, _, label = prepared
    corpus = (m.DATA / "passages.jsonl").read_bytes()
    base = (m.DATA / "annotations/passage_labels.jsonl").read_bytes()
    blind_complete(m, root, label)
    m.compare(root)
    write_json(root / "judge.json", [judgment()])
    m.import_judgments(root, 1, root / "judge.json", "model")
    r = m.publish(root)
    assert r["reviewed"] == 1 and r["needs_review"] == 0
    assert (m.DATA / "passages.jsonl").read_bytes() == corpus
    snapshot = json.loads((root / "manifest.json").read_text())["base_snapshot"]
    assert (m.DATA / snapshot).read_bytes() == base
    assert not read_jsonl(m.DATA / "annotations/passage_review.jsonl")[0]["human_reviewed"]
    with pytest.raises(ValueError, match="never overwrite"):
        m.publish(root)


def test_source_and_input_hash_guards(prepared):
    m, root, _, _ = prepared
    path = root / "inputs/batch-001.jsonl"
    rows = read_jsonl(path)
    rows[0]["text"] += " changed"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="Blinded input"):
        m.progress(root)


def test_selected_labels_hash_guard(prepared):
    m, root, _, _ = prepared
    path = root / "original_labels.jsonl"
    rows = read_jsonl(path)
    rows[0]["event_status"] = "observed"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="Selected original"):
        m.progress(root)


def test_comparison_resume_preserves_content(prepared):
    m, root, _, label = prepared
    blind_complete(m, root, label)
    m.compare(root)
    path = root / "comparisons/batch-001.jsonl"
    before = path.read_bytes()
    m.compare(root)
    assert path.read_bytes() == before


def test_invalid_amendment_cannot_add_context_only_quote(prepared):
    m, root, _, label = prepared
    blind_complete(m, root, label)
    m.compare(root)
    r = judgment()
    r.update(outcome="corrected", choice="amended", amended_label={k: label[k] for k in m.CORE})
    r["amended_label"]["evidence_spans"] = ["An invented source statement"]
    write_json(root / "judge.json", [r])
    with pytest.raises(ValueError, match="valid focus"):
        m.import_judgments(root, 1, root / "judge.json", "model")


def test_append_only_judgment_revision_preserves_worker_audit(prepared):
    m, root, _, label = prepared
    blind_complete(m, root, label)
    m.compare(root)
    write_json(root / "judge.json", [judgment()])
    m.import_judgments(root, 1, root / "judge.json", "worker")
    accepted = (root / "judgments/batch-001.json").read_bytes()
    audit = (root / "judgment_audit/batch-001.json").read_bytes()
    revised = judgment()
    revised["note"] = "Reviewed under the established protocol; retain the original label."
    write_json(root / "revision.json", [revised])
    m.revise_judgments(root, 1, root / "revision.json", "parent", "Apply the established protocol consistently")
    assert (root / "judgments/batch-001.json").read_bytes() == accepted
    assert (root / "judgment_audit/batch-001.json").read_bytes() == audit
    rows, actors = m.effective_judgments(root, 1)
    assert rows == [revised] and actors == {1: "parent"}
    assert m.progress(root)["reviewed"] == 1
    with pytest.raises(ValueError, match="preserve"):
        m.revise_judgments(root, 1, root / "revision.json", "parent", "Duplicate")
    m.publish(root)
    with pytest.raises(ValueError, match="new run"):
        m.revise_judgments(root, 1, root / "revision.json", "parent", "Already published")


def test_tampered_revision_rejected(prepared):
    m, root, _, label = prepared
    blind_complete(m, root, label)
    m.compare(root)
    write_json(root / "judge.json", [judgment()])
    m.import_judgments(root, 1, root / "judge.json", "worker")
    write_json(root / "revision.json", [judgment()])
    result = m.revise_judgments(root, 1, root / "revision.json", "parent", "Review")
    write_json(root / f"judgment_revisions/batch-001/{result['revision']}/judgments.json", {})
    with pytest.raises(ValueError, match="Revision audit"):
        m.progress(root)
