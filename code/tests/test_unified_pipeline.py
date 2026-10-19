"""Independent source-page pipeline guards, using only temporary fixtures.

No source downloads, live writes, model calls or human review records occur.
The segmentation stub isolates source/publishing guards; sentence-policy tests
belong to test_segmentation.py and are not substituted by these fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from oil_evidence import unified_corpus, store
from oil_evidence.corpus import extraction_suffix
from oil_evidence.store import digest, read_jsonl, words, write_json, write_jsonl


def load_script():
    path = Path(__file__).parents[1] / "scripts/unify_corpus.py"
    spec = importlib.util.spec_from_file_location("isolated_uniform_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def narrative(number):
    return (f"Crude oil production in field {number} remained stable across the surveyed region "
            + "during the reporting period according to the official supply report " * 7
            + "while petroleum exports continued through the established terminal.")


@dataclass
class PolicyStub:
    target_words: int
    minimum_words: int
    maximum_words: int
    maximum_context_words: int


def segment_stub(text, policy):
    return dict(accepted=[dict(start=0, end=len(text), text=text, word_count=len(words(text)), exceptions=[])],
                quarantined=[], policy=dict(version="semantic-page-v1", target_words=policy.target_words,
                  minimum_words=policy.minimum_words, maximum_words=policy.maximum_words,
                  maximum_context_words=policy.maximum_context_words, overlap_words=0))


@pytest.fixture
def uniform(tmp_path, monkeypatch):
    module = load_script()
    project = tmp_path / "project"
    data = project / "data"
    data.mkdir(parents=True)
    monkeypatch.setattr(module, "DATA", data)
    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(module, "RUN", data / "unified_runs/isolated-v1")
    monkeypatch.setattr(module, "CODE", ["guard_implementation.py"])
    monkeypatch.setattr(unified_corpus, "DATA", data)
    settings = dict(corpus=dict(start_date="2018-01-01", end_date="2026-10-05", chunk_words=160,
                               min_words=65, max_words=240, max_context_words=480),
                    retrieval=dict(embedding_model="fixture/local-embedding", top_k=5, candidate_k=50, rrf_k=60))
    monkeypatch.setattr(unified_corpus, "config", lambda: settings)
    monkeypatch.setattr(store, "config", lambda: settings)
    segmentation = types.ModuleType("oil_evidence.segmentation")
    segmentation.SegmentationPolicy = PolicyStub
    segmentation.segment_page = segment_stub
    monkeypatch.setitem(sys.modules, "oil_evidence.segmentation", segmentation)
    for folder in ["src", "scripts", "tests", "seeds"]:
        (project / folder).mkdir()
    for name in ["config.toml", "uv.lock", "pyproject.toml", "guard_implementation.py"]:
        (project / name).write_text("# isolated fixture only\n", encoding="utf-8")
    docs, sources, old_rows = [], [], []
    for number in [1, 2]:
        text = narrative(number)
        source = dict(source="EIA-TIE", title=f"Oil supply report {number}",
                      url=f"https://example.test/oil-{number}", published_date="2023-01-01",
                      available_by="2023-01-01", date_precision="day", date_source="official archive",
                      event_date=None, event_status="observed", redistribution="eia_attribution")
        raw = ("<main><article>" + text + "</article></main>").encode("utf-8")
        raw_path = data / f"raw/doc-{number}.html"
        raw_path.parent.mkdir(exist_ok=True)
        raw_path.write_bytes(raw)
        doc = dict(source, document_id=f"doc-{number}", raw_path=raw_path.relative_to(data).as_posix(),
                   raw_sha256=digest(raw), bytes=len(raw))
        cache = data / "extracted" / (doc["raw_sha256"] + extraction_suffix(doc["source"]))
        write_json(cache, [text])
        docs.append(doc)
        sources.append(source)
        old_rows.append(dict(source, passage_id=f"old-{number}", document_id=doc["document_id"], page=1,
                             text=text, word_count=len(words(text)), regions=[], region_method="keyword_heuristic"))
    write_jsonl(data / "documents.jsonl", docs)
    write_jsonl(data / "sources.jsonl", sources)
    write_jsonl(data / "passages.jsonl", old_rows)
    archive = data / "expansion_runs/first/baseline/passages.jsonl"
    write_jsonl(archive, old_rows[:1])
    write_json(data / "reports/corpus.json", dict(passages=2, words=sum(row["word_count"] for row in old_rows)))
    write_json(data / "reports/corpus_resegmentation.json", dict(
        corpus_sha256=digest((data / "passages.jsonl").read_bytes()),
        annotation_baseline_path=archive.relative_to(data).as_posix(), annotation_baseline_sha256=digest(archive.read_bytes())))
    write_json(data / "reports/passage_annotation.json", dict(corpus_sha256=digest(archive.read_bytes())))
    for name in ["codex", "passage_labels", "passage_review"]:
        write_jsonl(data / f"annotations/{name}.jsonl", [dict(passage_id="old-1", provenance="model_only", fixture=name)])
    (data / "index").mkdir()
    np.save(data / "index/vectors.npy", np.full((2, 2), 2 ** -0.5, dtype="float32"), allow_pickle=False)
    write_json(data / "index/manifest.json", dict(corpus_sha256=digest((data / "passages.jsonl").read_bytes()),
                  vectors_sha256=digest((data / "index/vectors.npy").read_bytes()), passage_ids=[p["passage_id"] for p in old_rows],
                  dimension=2, model=settings["retrieval"]["embedding_model"]))
    (data / "index/embedding_cache.sqlite3").write_bytes(b"fixture-cache-not-opened")
    return module, data, docs, old_rows, monkeypatch


def stage_fixture(uniform):
    module, _, _, _, _ = uniform
    module.snapshot()
    module.stage()
    return module.RUN / "stage"


def fake_index(uniform):
    module, _, _, _, _ = uniform
    target = module.RUN / "stage"
    rows = read_jsonl(target / "passages.jsonl")
    (target / "index").mkdir(exist_ok=True)
    np.save(target / "index/vectors.npy", np.full((len(rows), 2), 2 ** -0.5, dtype="float32"), allow_pickle=False)
    write_json(target / "index/manifest.json", dict(corpus_sha256=digest((target / "passages.jsonl").read_bytes()),
       vectors_sha256=digest((target / "index/vectors.npy").read_bytes()), passage_ids=[p["passage_id"] for p in rows],
       dimension=2, model="fixture/local-embedding"))
    (target / "index/embedding_cache.sqlite3").write_bytes(b"fixture-cache-not-opened")


def release_reports(uniform):
    module, _, _, _, _ = uniform
    target = module.RUN / "stage"
    report = module.load(target / "reports/corpus.json")
    rows = read_jsonl(target / "passages.jsonl")
    write_json(target / "reports/semantic_review.json", dict(corpus_sha256=report["corpus_sha256"],
                         source_pages_sha256=report["source_pages_sha256"], human_gold=False,
                         provenance="codex_source_grounded_review", human_labels_created=0,
                         critical_active_ids=[], reviewed_ids=module.required_review_ids(rows)))
    compare = module._comparison_module()
    seeds_path = module.PROJECT / "seeds/questions.json"
    write_json(seeds_path, [dict(question="What factors affected crude oil supply?", as_of="2024-12-31")])
    source_path = target / "source_pages.jsonl"
    pages = compare.load_source_pages(source_path)
    probes_path = target / "reports/source_probes.json"
    probe_records = []
    for number in range(12):
        key = sorted(pages)[number % len(pages)]
        page = pages[key]
        quote = " ".join(page["text"].split()[:20])
        probe_records.append(dict(probe_id=f"probe-{number:02d}",
             question=f"Which supply mechanism was reported in source probe number {number}?",
             as_of="2024-12-31", anchors=[dict(document_id=key[0], page=key[1], quote=quote)]))
    write_json(probes_path, dict(source_pages_sha256=report["source_pages_sha256"],
         selection_method="source_only_before_retrieval", human_gold=False,
         provenance="codex_source_grounded", probes=probe_records))
    queries = compare.load_seeds(seeds_path) + compare.load_probes(probes_path, pages, report["source_pages_sha256"])
    corpora = {name: compare.load_corpus(root, "fixture/local-embedding") for name, root in
               [("baseline", module.RUN / "baseline"), ("candidate", target)]}
    scope, scope_report = compare.shared_scope(corpora["baseline"], corpora["candidate"])
    results = {}
    for name, corpus in corpora.items():
        results[name] = []
        order = {p["passage_id"]: index for index, p in enumerate(corpus.rows)}
        for query in queries:
            hits = [dict(p, rank=index, score=0.5, bm25_score=None, dense_score=None)
                    for index, p in enumerate((p for p in corpus.rows if compare.page_key(p) in scope), 1)]
            for mode in compare.MODES:
                metrics = dict(compare.query_metrics(query, hits, order), ranking_seconds=0.001)
                results[name].append(dict(query_id=query["query_id"], kind=query["kind"], question=query["question"],
                    as_of=query["as_of"], mode=mode, metrics=metrics, evidence=[compare.evidence_record(hit) for hit in hits]))
    comparison = dict(diagnostic_only=True, human_gold=False, inputs_unchanged=True,
                     external_model_calls=0, human_labels_created=0, generated_answers=0,
                     settings=dict(embedding_model="fixture/local-embedding", top_k=5, candidate_k=50, rrf_k=60,
                                   temporal=True, modes=list(compare.MODES)),
                     script_sha256=digest(Path(compare.__file__).read_bytes()),
                     input_hashes={name: dict(root=str(c.root.resolve()), **c.hashes) for name, c in corpora.items()},
                     auxiliary_input_hashes={str(p.resolve()): digest(p.read_bytes()) for p in [seeds_path, source_path, probes_path]},
                     probe_queries=12, seed_queries=1, queries=13, source_scope=scope_report,
                     summaries={name: compare.aggregate(records) for name, records in results.items()},
                     results=results, local_encoder_model_hashes={})
    write_json(target / "reports/comparison.json", comparison)


def live_bytes(data):
    return {p.relative_to(data).as_posix(): p.read_bytes() for p in data.rglob("*")
            if p.is_file() and not p.is_relative_to(data / "unified_runs")}


@pytest.mark.parametrize("unsafe", ["../escape", "folder/run", "run\\bad", "C:\\escape", "", "ab", "Upper", "con", "com1", "lpt9"])
def test_uniform_run_ids_cannot_escape_or_alias_devices(uniform, unsafe):
    module, _, _, _, _ = uniform
    before = module.RUN
    with pytest.raises(ValueError):
        module.select_run(unsafe)
    assert module.RUN == before


def test_uniform_stage_isolated_and_all_records_have_exact_page_offsets(uniform):
    module, data, _, old_rows, _ = uniform
    before = live_bytes(data)
    target = stage_fixture(uniform)
    result = module.audit()
    pages = {p["source_page_id"]: p for p in read_jsonl(target / "source_pages.jsonl")}
    rows = read_jsonl(target / "passages.jsonl")
    assert result["uniform_policy"] and result["words_accounted"]
    assert len(rows) == 2 and not {r["passage_id"] for r in rows} & {r["passage_id"] for r in old_rows}
    assert all(pages[r["source_page_id"]]["text"][r["char_start"]:r["char_end"]] == r["text"] for r in rows)
    assert live_bytes(data) == before


@pytest.mark.parametrize("name", ["passages.jsonl", "documents.jsonl", "sources.jsonl", "annotations/passage_labels.jsonl", "index/manifest.json"])
def test_uniform_lost_update_live_inputs_fail_closed(uniform, name):
    module, data, _, _, _ = uniform
    module.snapshot()
    (data / name).write_bytes((data / name).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Live input changed"):
        module.stage()


@pytest.mark.parametrize("group", ["raw_hashes", "extraction_hashes"])
def test_uniform_raw_and_extraction_frozen_hashes_verified(uniform, group):
    module, data, _, _, _ = uniform
    snapshot = module.snapshot()
    name = next(iter(snapshot[group]))
    (data / name).write_bytes((data / name).read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Source or extraction cache changed"):
        module.verify_inputs()


def test_uniform_annotation_archive_is_immutable(uniform):
    module, data, _, _, _ = uniform
    snapshot = module.snapshot()
    (data / snapshot["annotation_baseline_path"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="annotation archive changed"):
        module.verify_inputs()


@pytest.mark.parametrize("mutate", ["metadata", "offset", "word_count", "policy", "old_parent", "missing_tail", "duplicate"])
def test_uniform_audit_recomputes_actual_evidence_guards(uniform, mutate):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    rows = read_jsonl(target / "passages.jsonl")
    if mutate == "metadata":
        rows[0]["available_by"] = "2024-01-01"
    elif mutate == "offset":
        rows[0]["char_end"] -= 1
    elif mutate == "word_count":
        rows[0]["word_count"] += 1
    elif mutate == "policy":
        rows[0]["segmentation"]["target_words"] += 1
    elif mutate == "old_parent":
        rows[0]["parent_passage_id"] = "old-1"
    elif mutate == "missing_tail":
        rows[0]["text"] = rows[0]["text"][:-1]
        rows[0]["char_end"] -= 1
    elif mutate == "duplicate":
        rows.append(dict(rows[0]))
    write_jsonl(target / "passages.jsonl", rows)
    report = module.load(target / "reports/corpus.json")
    report["corpus_sha256"] = digest((target / "passages.jsonl").read_bytes())
    write_json(target / "reports/corpus.json", report)
    with pytest.raises(ValueError):
        module.audit()


def test_uniform_cleaned_pages_must_reproduce_from_verified_extraction(uniform):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    pages = read_jsonl(target / "source_pages.jsonl")
    pages[0]["text"] += " Oil exports were interrupted."
    write_jsonl(target / "source_pages.jsonl", pages)
    with pytest.raises(ValueError, match="does not reproduce|hash is stale"):
        module.audit()


def test_uniform_document_and_physical_page_exclusions_applied(uniform):
    module, data, docs, _, _ = uniform
    docs[0]["ingest_exclusion_reason"] = "Unverified later attachment"
    docs[1]["excluded_physical_pages"] = {"1": "Flattened table"}
    write_jsonl(data / "documents.jsonl", docs)
    target = stage_fixture(uniform)
    assert read_jsonl(target / "passages.jsonl") == []
    excluded = read_jsonl(target / "source_exclusions.jsonl")
    assert {r["scope"] for r in excluded} == {"document", "page"}
    assert len(excluded) == 2
    assert module.audit()["active_passages"] == 0


def test_uniform_missing_pdf_cache_is_not_silently_reextracted(uniform):
    _, data, docs, _, _ = uniform
    doc = docs[0]
    pdf = b"%PDF-fixture-unverified-source"
    path = data / "raw/missing.pdf"
    path.write_bytes(pdf)
    changed = dict(doc, raw_path="raw/missing.pdf", raw_sha256=digest(pdf))
    with pytest.raises(ValueError, match="Missing verified PDF extraction cache"):
        unified_corpus.extract_existing(changed, data)


@pytest.mark.parametrize("mutation", ["order", "hash", "shape", "model", "nonfinite", "zero"])
def test_uniform_index_alignment_hash_shape_and_norm_fail_closed(uniform, mutation):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    fake_index(uniform)
    manifest = module.load(target / "index/manifest.json")
    if mutation == "order":
        manifest["passage_ids"].reverse()
    elif mutation == "hash":
        manifest["corpus_sha256"] = "0" * 64
    elif mutation == "model":
        manifest["model"] = "wrong-model"
    else:
        matrix = np.load(target / "index/vectors.npy", allow_pickle=False)
        if mutation == "shape":
            matrix = matrix[:1]
        elif mutation == "nonfinite":
            matrix[0, 0] = np.nan
        elif mutation == "zero":
            matrix[0] = 0
        np.save(target / "index/vectors.npy", matrix, allow_pickle=False)
        manifest["vectors_sha256"] = digest((target / "index/vectors.npy").read_bytes())
    write_json(target / "index/manifest.json", manifest)
    with pytest.raises(ValueError, match="Index"):
        module.audit(require_index=True)


def test_uniform_publication_rolls_back_on_replacement_failure(uniform):
    module, data, _, _, monkeypatch = uniform
    stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    before = live_bytes(data)
    original_replace = Path.replace
    counter = 0
    def fail_third(path, target):
        nonlocal counter
        if "publication_transaction/new" in path.as_posix():
            counter += 1
            if counter == 3:
                raise OSError("Injected replacement failure")
        return original_replace(path, target)
    monkeypatch.setattr(Path, "replace", fail_third)
    with pytest.raises(OSError, match="Injected replacement"):
        module.publish()
    assert live_bytes(data) == before
    journal = module.load(module.RUN / "publication_transaction/journal.json")
    assert journal["state"] == "rolled_back"


@pytest.mark.parametrize("kind", ["stale_review", "critical", "empty_review", "stale_comparison", "failed_comparison"])
def test_uniform_basic_release_reports_must_match_current_corpus(uniform, kind):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    name = "semantic_review" if kind in {"stale_review", "critical", "empty_review"} else "comparison"
    report = module.load(target / f"reports/{name}.json")
    if kind == "stale_review":
        report["corpus_sha256"] = "0" * 64
    elif kind == "critical":
        report["critical_active_ids"] = ["one-critical"]
    elif kind == "empty_review":
        report["reviewed_ids"] = []
    elif kind == "stale_comparison":
        report["candidate_corpus_sha256"] = "0" * 64
    elif kind == "failed_comparison":
        report["summaries"]["candidate"]["source_anchor_probe/hybrid"]["mean_assembled_anchor_support_coverage"] = 0
    write_json(target / f"reports/{name}.json", report)
    with pytest.raises(ValueError):
        module.publish()


def test_uniform_legacy_excluded_passage_ids_cannot_be_reintroduced_as_new_ids(uniform):
    from oil_evidence.corpus import clean_page, split_passages
    module, data, docs, _, _ = uniform
    text = narrative(1)
    old_units = split_passages(clean_page(text), target=200, minimum=65)
    assert len(old_units) == 1
    old_id = docs[0]["document_id"] + "-" + digest(" ".join(words(old_units[0])))[:16]
    docs[0]["excluded_passage_ids"] = {old_id: "Previously verified input-catalogue exclusion"}
    write_jsonl(data / "documents.jsonl", docs)
    target = stage_fixture(uniform)
    assert not any(p["document_id"] == docs[0]["document_id"] for p in read_jsonl(target / "passages.jsonl"))
    assert any(r["document_id"] == docs[0]["document_id"] for r in
               read_jsonl(target / "quarantine.jsonl") + read_jsonl(target / "source_exclusions.jsonl"))


@pytest.mark.parametrize("field", ["raw_sha256", "raw_path", "extraction_sha256", "extraction_path"])
def test_uniform_source_page_extraction_metadata_is_source_bound(uniform, field):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    pages = read_jsonl(target / "source_pages.jsonl")
    pages[0][field] = "forged-extraction-provenance"
    write_jsonl(target / "source_pages.jsonl", pages)
    report = module.load(target / "reports/corpus.json")
    report["source_pages_sha256"] = digest((target / "source_pages.jsonl").read_bytes())
    write_json(target / "reports/corpus.json", report)
    with pytest.raises(ValueError):
        module.audit()


@pytest.mark.parametrize("field", ["quarantine_sha256", "source_exclusions_sha256"])
def test_uniform_stage_ledger_hashes_cannot_be_stale(uniform, field):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    report = module.load(target / "reports/corpus.json")
    report[field] = "0" * 64
    write_json(target / "reports/corpus.json", report)
    with pytest.raises(ValueError):
        module.audit()


def test_uniform_page_word_counts_verified_individually_not_only_in_total(uniform):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    pages = read_jsonl(target / "source_pages.jsonl")
    pages[0]["word_count"] += 1
    pages[1]["word_count"] -= 1
    write_jsonl(target / "source_pages.jsonl", pages)
    report = module.load(target / "reports/corpus.json")
    report["source_pages_sha256"] = digest((target / "source_pages.jsonl").read_bytes())
    write_json(target / "reports/corpus.json", report)
    with pytest.raises(ValueError):
        module.audit()


def test_uniform_policy_cannot_be_forged_by_modifying_all_rows_and_report(uniform):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    rows = read_jsonl(target / "passages.jsonl")
    for row in rows:
        row["segmentation"]["version"] = "legacy-hardcut-v0"
    write_jsonl(target / "passages.jsonl", rows)
    report = module.load(target / "reports/corpus.json")
    report["chunk_policy"] = rows[0]["segmentation"]
    report["corpus_sha256"] = digest((target / "passages.jsonl").read_bytes())
    write_json(target / "reports/corpus.json", report)
    with pytest.raises(ValueError):
        module.audit()


def test_uniform_generated_ids_cannot_reuse_frozen_old_ids(uniform):
    module, _, _, old_rows, _ = uniform
    target = stage_fixture(uniform)
    rows = read_jsonl(target / "passages.jsonl")
    rows[0]["passage_id"] = old_rows[0]["passage_id"]
    write_jsonl(target / "passages.jsonl", rows)
    report = module.load(target / "reports/corpus.json")
    report["corpus_sha256"] = digest((target / "passages.jsonl").read_bytes())
    write_json(target / "reports/corpus.json", report)
    with pytest.raises(ValueError):
        module.audit()


def test_uniform_audit_rechecks_recorded_page_exclusions(uniform):
    module, data, docs, _, monkeypatch = uniform
    docs[0]["excluded_physical_pages"] = {"1": "Previously excluded flattened table"}
    write_jsonl(data / "documents.jsonl", docs)
    original_gate = unified_corpus.page_exclusion
    monkeypatch.setattr(unified_corpus, "page_exclusion", lambda *args: None)
    stage_fixture(uniform)
    monkeypatch.setattr(unified_corpus, "page_exclusion", original_gate)
    with pytest.raises(ValueError):
        module.audit()


@pytest.mark.parametrize("reviewed", [["not-an-active-passage"], [""], ["duplicate", "duplicate"]])
def test_uniform_review_ids_must_refer_uniquely_to_actual_active_evidence(uniform, reviewed):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    report = module.load(target / "reports/semantic_review.json")
    report["reviewed_ids"] = reviewed
    write_json(target / "reports/semantic_review.json", report)
    with pytest.raises(ValueError):
        module.publish()


def test_uniform_comparison_cannot_claim_the_wrong_baseline_hash(uniform):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    report = module.load(target / "reports/comparison.json")
    report["baseline_corpus_sha256"] = "0" * 64
    write_json(target / "reports/comparison.json", report)
    with pytest.raises(ValueError):
        module.publish()


@pytest.mark.parametrize("field,value", [("document_id", "doc-2"), ("page", 7)])
def test_uniform_quarantine_metadata_remains_bound_to_its_source_page(uniform, field, value):
    module, _, _, _, monkeypatch = uniform
    def reject_first(text, policy):
        result = segment_stub(text, policy)
        if "field 1 " in text:
            unit = result["accepted"][0]
            result["accepted"] = []
            result["quarantined"] = [dict(unit, reason="incomplete_unit")]
        return result
    monkeypatch.setattr(sys.modules["oil_evidence.segmentation"], "segment_page", reject_first)
    target = stage_fixture(uniform)
    rejected = read_jsonl(target / "quarantine.jsonl")
    assert len(rejected) == 1
    rejected[0][field] = value
    write_jsonl(target / "quarantine.jsonl", rejected)
    report = module.load(target / "reports/corpus.json")
    report["quarantine_sha256"] = digest((target / "quarantine.jsonl").read_bytes())
    write_json(target / "reports/corpus.json", report)
    with pytest.raises(ValueError):
        module.audit()


def test_uniform_duplicate_quarantine_must_name_matching_active_text(uniform):
    module, data, docs, _, _ = uniform
    first_raw = (data / docs[0]["raw_path"]).read_bytes()
    (data / docs[1]["raw_path"]).write_bytes(first_raw)
    docs[1].update(raw_sha256=docs[0]["raw_sha256"], bytes=len(first_raw))
    write_jsonl(data / "documents.jsonl", docs)
    target = stage_fixture(uniform)
    rejected = read_jsonl(target / "quarantine.jsonl")
    assert len(rejected) == 1 and rejected[0]["reason"] == "exact_normalized_duplicate"
    rejected[0]["duplicate_of"] = "not-an-active-passage"
    write_jsonl(target / "quarantine.jsonl", rejected)
    report = module.load(target / "reports/corpus.json")
    report["quarantine_sha256"] = digest((target / "quarantine.jsonl").read_bytes())
    write_json(target / "reports/corpus.json", report)
    with pytest.raises(ValueError):
        module.audit()


def test_uniform_lost_update_during_prepare_leaves_live_bytes_unpublished(uniform):
    module, data, _, _, monkeypatch = uniform
    stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    original_copy = module.shutil.copy2
    altered = False
    def change_live_during_prepare(source, target, *args, **kwargs):
        nonlocal altered
        result = original_copy(source, target, *args, **kwargs)
        if not altered and "publication_transaction/new" in Path(target).as_posix():
            altered = True
            (data / "sources.jsonl").write_bytes((data / "sources.jsonl").read_bytes() + b"\n")
        return result
    before_passages = (data / "passages.jsonl").read_bytes()
    monkeypatch.setattr(module.shutil, "copy2", change_live_during_prepare)
    with pytest.raises(ValueError, match="Live input changed"):
        module.publish()
    assert (data / "passages.jsonl").read_bytes() == before_passages


def test_uniform_changed_implementation_during_freeze_cannot_publish(uniform):
    module, data, _, _, monkeypatch = uniform
    stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    before = live_bytes(data)
    original_copy = module.shutil.copy2
    altered = False
    def change_code_during_freeze(source, target, *args, **kwargs):
        nonlocal altered
        if not altered and "implementation" in Path(target).parts:
            altered = True
            Path(source).write_bytes(Path(source).read_bytes() + b"# changed during freeze\n")
        return original_copy(source, target, *args, **kwargs)
    monkeypatch.setattr(module.shutil, "copy2", change_code_during_freeze)
    with pytest.raises(ValueError):
        module.publish()
    assert live_bytes(data) == before


def appended_source(uniform):
    module, data, _, _, _ = uniform
    text = narrative(3)
    source = dict(source="EIA-TIE", title="Appended official oil article", url="https://example.test/appended",
                  published_date="2023-02-01", available_by="2023-02-01", date_precision="day",
                  date_source="official archive", event_date=None, event_status="observed", redistribution="eia_attribution")
    raw = ("<article>" + text + "</article>").encode()
    path = data / "raw/appended.html"
    path.write_bytes(raw)
    doc = dict(source, document_id="appended", raw_path="raw/appended.html", raw_sha256=digest(raw), bytes=len(raw))
    write_json(data / "extracted" / (doc["raw_sha256"] + extraction_suffix(doc["source"])), [text])
    write_jsonl(module.RUN / "supplement/documents.jsonl", [doc])
    write_jsonl(module.RUN / "supplement/sources.jsonl", [source])
    return doc, source


def test_uniform_supplement_stages_registry_but_does_not_register_live_until_publish(uniform):
    module, data, _, _, _ = uniform
    module.snapshot()
    old_documents = (data / "documents.jsonl").read_bytes()
    old_sources = (data / "sources.jsonl").read_bytes()
    doc, source = appended_source(uniform)
    target = stage_fixture(uniform)
    assert (data / "documents.jsonl").read_bytes() == old_documents
    assert (data / "sources.jsonl").read_bytes() == old_sources
    assert read_jsonl(target / "documents.jsonl")[-1] == doc
    assert read_jsonl(target / "sources.jsonl")[-1] == source
    assert module.audit()["active_passages"] == 3
    fake_index(uniform)
    release_reports(uniform)
    published = module.publish()
    assert read_jsonl(data / "documents.jsonl")[-1] == doc
    assert read_jsonl(data / "sources.jsonl")[-1] == source
    assert (module.RUN / "baseline/documents.jsonl").read_bytes() == old_documents
    assert published["supplemental_documents"] == 1


@pytest.mark.parametrize("mutation", ["missing_sources", "reused_id", "reused_url", "reused_raw_path", "metadata", "raw_sha", "escape"])
def test_uniform_supplement_inputs_fail_closed_before_registration(uniform, mutation):
    module, data, docs, _, _ = uniform
    module.snapshot()
    doc, source = appended_source(uniform)
    if mutation == "missing_sources":
        (module.RUN / "supplement/sources.jsonl").unlink()
    else:
        if mutation == "reused_id":
            doc["document_id"] = docs[0]["document_id"]
        elif mutation == "reused_url":
            doc["url"] = source["url"] = docs[0]["url"]
        elif mutation == "reused_raw_path":
            doc["raw_path"] = docs[0]["raw_path"]
        elif mutation == "metadata":
            source["available_by"] = "2024-01-01"
        elif mutation == "raw_sha":
            doc["raw_sha256"] = "0" * 64
        elif mutation == "escape":
            doc["raw_path"] = "../outside.html"
        write_jsonl(module.RUN / "supplement/documents.jsonl", [doc])
        write_jsonl(module.RUN / "supplement/sources.jsonl", [source])
    before = (data / "documents.jsonl").read_bytes()
    with pytest.raises(ValueError):
        module.stage()
    assert (data / "documents.jsonl").read_bytes() == before


@pytest.mark.parametrize("mutation", ["manifest", "raw", "cache"])
def test_uniform_supplement_fingerprints_stay_bound_to_staged_text(uniform, mutation):
    module, data, _, _, _ = uniform
    module.snapshot()
    doc, _ = appended_source(uniform)
    stage_fixture(uniform)
    if mutation == "manifest":
        path = module.RUN / "supplement/documents.jsonl"
    elif mutation == "raw":
        path = data / doc["raw_path"]
    else:
        path = data / "extracted" / (doc["raw_sha256"] + extraction_suffix(doc["source"]))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        module.audit()


def test_uniform_source_review_must_cover_fixed_length_and_source_samples(uniform):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    review = module.load(target / "reports/semantic_review.json")
    review["reviewed_ids"] = review["reviewed_ids"][:1]
    write_json(target / "reports/semantic_review.json", review)
    with pytest.raises(ValueError, match="fixed source/length"):
        module.check_release_gates(module.load(target / "reports/corpus.json"))


@pytest.mark.parametrize("mutation", ["aggregate", "per_query_support", "future_cutoff", "unknown_evidence", "text_sha",
                                     "missing_mode", "duplicate_query", "probe_count", "input_hash", "source_hash", "gold_claim"])
def test_uniform_release_metrics_and_evidence_cannot_be_forged_with_a_true_gate(uniform, mutation):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    comparison = module.load(target / "reports/comparison.json")
    comparison["release_gate_passed"] = True
    if mutation == "aggregate":
        comparison["summaries"]["candidate"]["source_anchor_probe/hybrid"]["mean_assembled_anchor_support_coverage"] = 0.9
    elif mutation == "per_query_support":
        row = next(r for r in comparison["results"]["candidate"] if r["kind"] == "source_anchor_probe")
        row["metrics"]["assembled_anchor_support_coverage"] = 0.9
    elif mutation == "future_cutoff":
        comparison["results"]["candidate"][0]["as_of"] = "2017-12-31"
    elif mutation == "unknown_evidence":
        comparison["results"]["candidate"][0]["evidence"][0]["passage_id"] = "not-active"
    elif mutation == "text_sha":
        comparison["results"]["candidate"][0]["evidence"][0]["text_sha256"] = "0" * 64
    elif mutation == "missing_mode":
        comparison["results"]["candidate"].pop()
    elif mutation == "duplicate_query":
        comparison["results"]["candidate"].append(comparison["results"]["candidate"][0])
    elif mutation == "probe_count":
        comparison["probe_queries"] = 11
    elif mutation == "input_hash":
        comparison["input_hashes"]["baseline"]["passages.jsonl"] = "0" * 64
    elif mutation == "source_hash":
        comparison["auxiliary_input_hashes"][str((target / "source_pages.jsonl").resolve())] = "0" * 64
    elif mutation == "gold_claim":
        comparison["human_gold"] = True
    write_json(target / "reports/comparison.json", comparison)
    with pytest.raises(ValueError):
        module.check_release_gates(module.load(target / "reports/corpus.json"))


def test_uniform_real_probe_support_regression_rejected_even_if_report_gate_true(uniform):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    compare = module._comparison_module()
    comparison = module.load(target / "reports/comparison.json")
    pages = compare.load_source_pages(target / "source_pages.jsonl")
    corpus = module.load(target / "reports/corpus.json")
    probes = compare.load_probes(target / "reports/source_probes.json", pages, corpus["source_pages_sha256"])
    query_by_id = {p["query_id"]: p for p in probes}
    for row in comparison["results"]["candidate"]:
        if row["kind"] == "source_anchor_probe" and row["mode"] == "hybrid":
            row["evidence"] = []
            row["metrics"] = dict(compare.query_metrics(query_by_id[row["query_id"]], [], {}), ranking_seconds=0.001)
    comparison["summaries"]["candidate"] = compare.aggregate(comparison["results"]["candidate"])
    comparison["release_gate_passed"] = True
    write_json(target / "reports/comparison.json", comparison)
    with pytest.raises(ValueError, match="support regressed"):
        module.check_release_gates(corpus)


def test_uniform_release_boolean_does_not_replace_computed_gate(uniform):
    module, _, _, _, _ = uniform
    target = stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    comparison = module.load(target / "reports/comparison.json")
    comparison["release_gate_passed"] = False
    write_json(target / "reports/comparison.json", comparison)
    decision = module.check_release_gates(module.load(target / "reports/corpus.json"))
    assert decision["release_gate_passed"] and decision["diagnostic_only"] and not decision["human_gold"]


@pytest.mark.parametrize("moment", ["freeze", "prepare"])
def test_uniform_stage_mutation_during_publication_cannot_reach_live(uniform, moment):
    module, data, _, _, monkeypatch = uniform
    target = stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    before = live_bytes(data)
    original = module.shutil.copy2
    altered = False
    def change_stage(source, dest, *args, **kwargs):
        nonlocal altered
        destination = Path(dest)
        trigger = "implementation" in destination.parts if moment == "freeze" else "publication_transaction/new" in destination.as_posix()
        result = original(source, dest, *args, **kwargs)
        if trigger and not altered:
            altered = True
            (target / "reports/semantic_review.json").write_bytes((target / "reports/semantic_review.json").read_bytes() + b"\n")
        return result
    monkeypatch.setattr(module.shutil, "copy2", change_stage)
    with pytest.raises(ValueError, match="Stage/supplement changed"):
        module.publish()
    assert live_bytes(data) == before
