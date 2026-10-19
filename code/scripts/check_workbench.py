"""Unpaid current-corpus checks; never writes human labels or calls a remote LLM."""
import json
import sqlite3
import sys

from passage_labels import schema, validate_batch

from streamlit.testing.v1 import AppTest

from oil_evidence.annotations import validate_record
from oil_evidence.corpus_versions import annotation_passages, active_label_coverage, active_corpus_version, original_parent
from oil_evidence.query import query
from oil_evidence.retrieval import Retriever
from oil_evidence.store import DATA, digest, now, read_jsonl, write_json

sys.stdout.reconfigure(encoding="utf-8")
with sqlite3.connect(DATA / "annotations.sqlite3") as db:
    human_before = db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
with sqlite3.connect(DATA / "model_usage.sqlite3") as db:
    paid_before = db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
passages = read_jsonl(DATA / "passages.jsonl")
by_id = {p["passage_id"]: p for p in passages}
annotation_rows = annotation_passages()
annotation_by_id = {p["passage_id"]: p for p in annotation_rows}
version_report = active_corpus_version()
assert len(by_id) == len(passages)
candidates = read_jsonl(DATA / "annotations/candidates.jsonl")
for record in candidates:
    errors = validate_record(record, annotation_by_id)
    assert not errors, f"{record['record_id']}: {errors}"
paragraph_labels = read_jsonl(DATA / "annotations/passage_labels.jsonl")
if paragraph_labels:
    fields = schema()["properties"]["labels"]["items"]["properties"]
    assert len(paragraph_labels) <= len(annotation_rows)
    assert len({r['passage_id'] for r in paragraph_labels}) == len(paragraph_labels)
    assert all(r['passage_id'] in annotation_by_id for r in paragraph_labels)
    assert all(r["provenance"] == "codex_model_passage_annotation" and not r["human_reviewed"] for r in paragraph_labels)
    errors = validate_batch([{key: r[key] for key in fields} for r in paragraph_labels], [annotation_by_id[r['passage_id']] for r in paragraph_labels])
    assert not errors, str(errors[:10])
    published = json.loads((DATA / "reports/passage_annotation.json").read_text(encoding="utf-8"))
    assert published["labels_sha256"] == digest((DATA / "annotations/passage_labels.jsonl").read_bytes())
    if published['corpus_sha256'] != digest((DATA/'passages.jsonl').read_bytes()) and version_report is None:
        expansion = json.loads((DATA/'reports/corpus_expansion.json').read_text(encoding='utf-8'))
        assert expansion['corpus_sha256'] == digest((DATA/'passages.jsonl').read_bytes())
        annotation_path=__import__('pathlib').Path(expansion.get('annotation_baseline_path') or (__import__('pathlib').Path(expansion['snapshot_root'])/'baseline/passages.jsonl'))
        assert digest(annotation_path.read_bytes())==published['corpus_sha256']
        assert expansion.get('annotation_baseline_sha256',expansion['baseline_sha256']) == published['corpus_sha256']
        baseline = read_jsonl(annotation_path)
        assert all(by_id[p['passage_id']]==p for p in baseline)
        run_baseline_path=__import__('pathlib').Path(expansion['snapshot_root'])/'baseline/passages.jsonl'
        assert digest(run_baseline_path.read_bytes())==expansion['baseline_sha256']
        assert all(by_id[p['passage_id']]==p for p in read_jsonl(run_baseline_path))
coverage = active_label_coverage(paragraph_labels)
if version_report is not None:
    assert version_report["corpus_sha256"] == digest((DATA / "passages.jsonl").read_bytes())
    for field, expected in [("passages", len(passages)), ("labels_coverage", coverage),
                            ("archived_machine_labels", len(paragraph_labels)),
                            ("unlabelled_corpus_passages", len(passages) - coverage)]:
        if field in version_report:
            assert version_report[field] == expected, field
codex = read_jsonl(DATA / "annotations/codex.jsonl")
if codex:
    assert len(codex) == 100 and len({r["record_id"] for r in codex}) == 100
    assert all(r["provenance"] == "codex_model_annotated" and r["status"] == "machine_completed" for r in codex)
    for record in codex:
        assert not validate_record(record, annotation_by_id), record["record_id"]
        assert len(record["qrels"]) == len(record["candidate_pool_ids"])
    manifest = json.loads((DATA / "reports/codex_annotation.json").read_text(encoding="utf-8"))
    assert manifest["labels_sha256"] == digest((DATA / "annotations/codex.jsonl").read_bytes())
    originals = {name: read_jsonl(DATA / f"annotations/{name}.jsonl") for name in ["pilot", "candidates"]}
    for row in read_jsonl(DATA / "annotations/codex_input.jsonl"):
        assert row["original_sha256"] == digest(json.dumps(originals[row["dataset"]][row["number"]-1], sort_keys=True))

retriever = Retriever()
index_manifest = json.loads((DATA / "index/manifest.json").read_text(encoding="utf-8"))
assert index_manifest["corpus_sha256"] == digest((DATA / "passages.jsonl").read_bytes())
assert index_manifest["passage_ids"] == [p["passage_id"] for p in passages]
assert retriever.vectors is not None and retriever.vectors.shape == (len(passages), index_manifest["dimension"])
cutoff = "2020-12-31"
result = query("What were oil supply forecasts in 2020?", as_of=cutoff,
               generation="extractive", retriever=retriever, log=False)
assert result["evidence"] and all(p["available_by"] <= cutoff for p in result["evidence"])
assert not result["model_calls"]
validated_contexts = 0
for passage in result["evidence"]:
    context_key = "source_page_id" if passage.get("source_page_id") else "parent_passage_id"
    if passage.get(context_key):
        parent = original_parent(passage)
        parent_key = "source_page_id" if context_key == "source_page_id" else "passage_id"
        assert parent[parent_key] == passage[context_key]
        assert all(passage[context_key] not in c["citations"] for c in result["claims"])
        validated_contexts += 1

app = AppTest.from_file(str(DATA.parent / "app.py"), default_timeout=60).run()
assert not app.exception, str(app.exception)
assert len(app.tabs) == 4
assert any(tab.label == "Passage labels" for tab in app.tabs)
passage_progress = DATA / "reports/passage_annotation_progress.json"
if passage_progress.exists():
    assert any("source passage annotations" in s.value for s in app.subheader)
if paragraph_labels:
    assert any(f"Current retrieval passage label coverage: {coverage:,} / {len(passages):,}" in s.value for s in app.caption)
    if annotation_rows != passages:
        assert any("Archived annotations" in s.value for s in app.info)
    assert len(next(s for s in app.selectbox if s.label == "Source passage").options) == len(paragraph_labels)
    next(c for c in app.checkbox if c.label == "Show only ambiguous passages / Items needing review").check().run()
    assert not app.exception
    assert len(next(s for s in app.selectbox if s.label == "Source passage").options) == sum(r["needs_review"] for r in paragraph_labels)
    next(c for c in app.checkbox if c.label == "Show only ambiguous passages / Items needing review").uncheck().run()
    next(t for t in app.text_input if t.label.startswith("Passage label search")).set_value("Brent").run()
    expected = sum("brent" in (annotation_by_id[r["passage_id"]]["title"] + " " + r["summary_zh"] + " " + " ".join(r["entities"])).casefold() for r in paragraph_labels)
    assert not app.exception and expected > 0
    assert len(next(s for s in app.selectbox if s.label == "Source passage").options) == expected
    next(t for t in app.text_input if t.label.startswith("Passage label search")).set_value("").run()
    next(s for s in app.selectbox if s.label == "Passage topic filter").select("production_supply").run()
    assert not app.exception
    assert len(next(s for s in app.selectbox if s.label == "Source passage").options) == sum("production_supply" in r["topics"] for r in paragraph_labels)
    next(s for s in app.selectbox if s.label == "Passage topic filter").select("all").run()
    second_reviews = read_jsonl(DATA / "annotations/passage_review.jsonl")
    if second_reviews:
        assert len(second_reviews) == 728 and len({r["passage_id"] for r in second_reviews}) == 728
        assert all(r["provenance"] == "codex_model_second_pass" and not r["human_reviewed"] for r in second_reviews)
        second_report = json.loads((DATA / "reports/passage_review.json").read_text(encoding="utf-8"))
        assert second_report["reviews_sha256"] == digest((DATA / "annotations/passage_review.jsonl").read_bytes())
        for outcome in ["confirmed", "corrected", "unresolved"]:
            next(s for s in app.selectbox if s.label == "Second-pass machine review result").select(outcome).run()
            expected = sum(r["outcome"] == outcome for r in second_reviews)
            assert not app.exception
            paragraph_boxes = [s for s in app.selectbox if s.label == "Source passage"]
            assert (len(paragraph_boxes[0].options) if paragraph_boxes else 0) == expected
        next(s for s in app.selectbox if s.label == "Second-pass machine review result").select("all").run()
if codex:
    assert next(s for s in app.selectbox if s.label == "Review dataset").value == "codex"
    assert next(c for c in app.checkbox if c.label.startswith("Show model")).value
    assert any("Codex review result" in s.value for s in app.subheader)
    assert any(codex[0]["codex_review"]["note_zh"] in s.value for s in app.markdown)
    next(s for s in app.selectbox if s.label == "Review result filter").select("Partially answerable / Insufficient evidence").run()
    assert not app.exception and len(next(s for s in app.selectbox if s.label == "Record").options) == 22
    next(s for s in app.selectbox if s.label == "Review result filter").select("Corrected").run()
    assert not app.exception and len(next(s for s in app.selectbox if s.label == "Record").options) == 16
    next(s for s in app.selectbox if s.label == "Review result filter").select("All").run()
next(b for b in app.button if b.label == "Search").click().run()
assert not app.exception and not app.error, str(app.exception) + str(app.error)
assert app.session_state["query_result"]["evidence"]
next(s for s in app.selectbox if s.label == "Review dataset").select("candidates").run()
assert not app.exception and not app.error, str(app.exception) + str(app.error)
assert any(b.label == "Save human judgment" for b in app.button)
with sqlite3.connect(DATA / "model_usage.sqlite3") as db:
    assert db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == paid_before
    calls = [dict(zip(["model", "status", "count", "estimated_or_reserved_usd"], row)) for row in
             db.execute("SELECT model,status,COUNT(*),SUM(COALESCE(cost,reserved)) FROM calls GROUP BY model,status")]
with sqlite3.connect(DATA / "annotations.sqlite3") as db:
    human_after = db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
assert human_before == human_after
report = dict(at=now(), corpus_sha256=digest((DATA / "passages.jsonl").read_bytes()),
              passages=len(passages), validated_model_candidates=len(candidates),
              temporal_check_passed=True, streamlit_tabs=len(app.tabs),
              streamlit_hybrid_query_passed=True, paid_calls_in_this_check=0,
              streamlit_candidate_review_loaded=True,
              validated_codex_labels=len(codex), streamlit_codex_filters_passed=bool(codex),
              validated_passage_labels=len(paragraph_labels),
              active_passage_label_coverage=coverage,
              archived_machine_labels=len(paragraph_labels),
              annotation_version_passages=len(annotation_rows),
              annotation_version_is_archive=annotation_rows != passages,
              active_segmentation_version=(version_report or {}).get("version"),
              validated_display_only_source_contexts=validated_contexts,
              unlabelled_corpus_passages=len(passages)-coverage,
              streamlit_passage_labels_loaded=passage_progress.exists(),
              streamlit_passage_filters_passed=bool(paragraph_labels),
              validated_second_pass_reviews=len(read_jsonl(DATA / "annotations/passage_review.jsonl")),
              streamlit_second_pass_filters_passed=bool(read_jsonl(DATA / "annotations/passage_review.jsonl")),
              human_reviews_before=human_before, human_reviews_after=human_after,
              cumulative_model_ledger=calls,
              raw_bytes=sum(p.stat().st_size for p in (DATA / "raw").iterdir() if p.is_file()))
write_json(DATA / "reports/workbench_checks.json", report)
print(json.dumps(report, ensure_ascii=False, indent=2))
