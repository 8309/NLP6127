"""Isolated supplement routing/parser/append-only tests; no network or live files."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from oil_evidence.store import digest, read_jsonl, write_json, write_jsonl


TITLE = "DOE Announces Crude Oil Release from Strategic Petroleum Reserve"
BODY = ("Crude oil from the Strategic Petroleum Reserve will be released after an emergency supply interruption. "
        "Official notices specify contract award dates separately from delivery dates, and the government does not treat an announcement as a completed shipment. "
        "Crude oil deliveries support the affected refinery after the pipeline disruption and these barrels must later be returned under the published emergency exchange contract. "
        "The original statement preserves the announced quantity and delivery period for the oil barrels. "
        "The agency reported that the reserve supplied crude oil to the regional refinery and maintained separate records for exchange repayments.")


def doe_html(title=TITLE, display_date="April 21, 2022", template="field--name-field-text", text=BODY):
    return (f"<main><h1>{title}</h1><article typeof='schema:Article' about='/articles/example'>"
            f"<div class='beneath-title'><span class='display-date'>{display_date}</span></div>"
            f"<span property='schema:name' content='{title}'></span>"
            f"<div class='{template}'><p>{text}</p></div></article>"
            "<time datetime='2030-01-01'>January 1, 2030</time></main>").encode()


def source(kind="DOE-SPR-NEWS"):
    return dict(source=kind, url="https://www.energy.gov/hgeo/opr/articles/doe-crude-release",
                title=TITLE, published_date="2022-04-21", available_by="2022-04-21",
                date_precision="day", date_source="https://www.energy.gov/collection/filter/819896",
                event_date=None, event_status="unlabelled", redistribution="review_required")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("isolated_supplement", scripts / "collect_unified_supplement.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    # Mirror the real CLI launcher, whose script directory remains on sys.path
    # for lazy source adapters. Keep all data/requests isolated below.
    monkeypatch.syspath_prepend(str(scripts))
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(module, "DATA", data)
    for name in ("sources.jsonl", "documents.jsonl", "passages.jsonl", "annotations/codex.jsonl",
                 "annotations/passage_labels.jsonl", "annotations/passage_review.jsonl"):
        write_jsonl(data / name, [])
    write_json(data / "index/manifest.json", {"fixture": True})
    (data / "index/vectors.npy").write_bytes(b"untouched fixture vector bytes")
    class FakeFetcher:
        def __init__(self):
            self.settings = {"request_delay_seconds": 1, "max_document_mb": 25, "download_budget_mb": 300}
            self.bytes = 0
        def get(self, url):
            raw, final = response["raw"], response.get("final", url)
            self.bytes += len(raw)
            return raw, final, "text/html"
        def close(self):
            pass
    response = {"raw": doe_html()}
    monkeypatch.setattr(module, "Fetcher", FakeFetcher)
    monkeypatch.setattr(module, "SupplementFetcher", FakeFetcher)
    candidate = tmp_path / "candidates.jsonl"
    write_jsonl(candidate, [source()])
    return module, data, candidate, response


@pytest.mark.parametrize("template", ["field--name-body", "field--name-field-text"])
def test_both_doe_templates_only_article_local_date_and_title(isolated, template):
    module, *_ = isolated
    parsed = module.parse_source(source(), doe_html(template=template))
    assert parsed["published_date"] == "2022-04-21"
    assert parsed["body_title_verified"] is True
    assert "Crude oil" in parsed["text"]
    assert "January 1, 2030" not in parsed["text"]


def test_candidate_title_normalization_keeps_numbers_and_words(isolated):
    module, *_ = isolated
    row = source()
    row["title"] = "  doe announces\nCRUDE OIL release from strategic petroleum reserve  "
    assert module.parse_source(row, doe_html())["body_title_verified"] is True
    row["title"] = TITLE.replace("Crude Oil Release", "Crude Oil Purchase")
    assert module.parse_source(row, doe_html())["body_title_verified"] is False


@pytest.mark.parametrize("url", ["https://www.energy.gov.evil.test/articles/doe-crude-release",
                                "https://user@www.energy.gov/articles/doe-crude-release",
                                "http://www.energy.gov/articles/doe-crude-release",
                                "https://stage.energy.gov/articles/doe-crude-release",
                                "https://www.energy.gov/topics/strategic-petroleum-reserve-spr",
                                "https://www.energy.gov:444/articles/doe-crude-release"])
def test_strict_official_route(isolated, url):
    module, *_ = isolated
    with pytest.raises(ValueError):
        module.admitted_route(source(), url)


def test_existing_treasury_parser_and_route_still_work(isolated):
    module, *_ = isolated
    row = source("TREASURY-OIL")
    row["url"] = "https://home.treasury.gov/news/press-releases/sb0215/"
    raw = ("<article class='entity--type-node'><time datetime='2022-04-21T12:00:00Z'></time>"
           f"<p>{BODY}</p><p>{BODY}</p></article>").encode()
    assert module.admitted_route(row) == "TREASURY-OIL"
    parsed = module.parse_source(row, raw)
    assert parsed["published_date"] == "2022-04-21"
    assert parsed["body_title_verified"] is None


def test_collection_separate_manifest_append_only_and_live_unchanged(isolated):
    module, data, candidates, response = isolated
    protected = {str(p): p.read_bytes() for p in data.rglob("*") if p.is_file()}
    report = module.collect("test-v1", candidates, 100)
    assert report["attempts"] == report["added_raw_documents"] == 1
    assert report["accepted_by_source"] == {"DOE-SPR-NEWS": 1}
    assert report["candidate_file_sha256"] == digest(candidates.read_bytes())
    assert report["live_protected_files_unchanged"] is True
    docs = read_jsonl(data / "unified_runs/test-v1/supplement/documents.jsonl")
    assert docs[0]["body_date_verified"] and docs[0]["body_title_verified"] and docs[0]["body_oil_verified"]
    for path, original in protected.items():
        assert Path(path).read_bytes() == original
    before_raw = {str(p): p.read_bytes() for p in (data / "raw").rglob("*") if p.is_file()}
    before_cache = {str(p): p.read_bytes() for p in (data / "extracted").rglob("*") if p.is_file()}
    repeated = module.collect("test-v1", candidates, 100)
    assert repeated["attempts"] == repeated["added_raw_documents"] == 0
    assert {str(p): p.read_bytes() for p in (data / "raw").rglob("*") if p.is_file()} == before_raw
    assert {str(p): p.read_bytes() for p in (data / "extracted").rglob("*") if p.is_file()} == before_cache


@pytest.mark.parametrize("problem,exclusion", [("date", "article_date_conflicts_with_official_index"),
                                               ("title", "article_title_conflicts_with_official_index")])
def test_conflicting_date_or_title_retained_for_audit_but_excluded(isolated, problem, exclusion):
    module, data, candidates, response = isolated
    response["raw"] = doe_html(display_date="April 22, 2022") if problem == "date" else doe_html(title=TITLE + " 2")
    report = module.collect("test-v1", candidates, 100)
    assert report["dated_scope_accepted"] == 0 and report["excluded_downloads"] == 1
    doc = read_jsonl(data / "unified_runs/test-v1/supplement/documents.jsonl")[0]
    assert doc["ingest_exclusion_reason"] == exclusion
    assert (data / doc["raw_path"]).read_bytes() == response["raw"]


def test_redirect_outside_allowlist_records_failure_not_ingestion(isolated):
    module, data, candidates, response = isolated
    response["final"] = "https://evil.test/articles/redirected"
    report = module.collect("test-v1", candidates, 100)
    assert report["added_raw_documents"] == 0 and report["documents_sha256"] is None
    log = read_jsonl(data / "unified_runs/test-v1/supplement/collection.jsonl")
    assert log[0]["status"] == "failed"
    assert not (data / "raw").exists()


def test_existing_raw_bytes_not_overwritten(isolated):
    module, data, candidates, response = isolated
    path = data / "raw" / (digest(source()["url"])[:20] + ".html")
    path.parent.mkdir()
    path.write_bytes(b"original immutable raw")
    report = module.collect("test-v1", candidates, 100)
    assert report["added_raw_documents"] == 0
    assert path.read_bytes() == b"original immutable raw"
    assert "cannot be overwritten" in read_jsonl(data / "unified_runs/test-v1/supplement/collection.jsonl")[0]["error"]


def test_missing_article_date_does_not_use_unrelated_time(isolated):
    module, data, candidates, response = isolated
    response["raw"] = doe_html().replace(b"<span class='display-date'>April 21, 2022</span>", b"")
    report = module.collect("test-v1", candidates, 100)
    assert report["added_raw_documents"] == 1
    assert report['dated_scope_accepted'] == 0
    record = read_jsonl(data / "unified_runs/test-v1/supplement/documents.jsonl")[0]
    assert record['ingest_exclusion_reason'] == 'unverified_article_body_or_publication_date'
    assert 'article-local publication date missing' in record['body_parser_error']
    assert (data / record['raw_path']).read_bytes() == response['raw']


def test_generic_sanctions_and_reserve_name_not_oil_scope(isolated):
    module, data, candidates, response = isolated
    text = "Strategic Petroleum Reserve staff discussed administrative sanctions and general banking procedures. " * 12
    response["raw"] = doe_html(text=text)
    report = module.collect("test-v1", candidates, 100)
    assert report["dated_scope_accepted"] == 0
    assert read_jsonl(data / "unified_runs/test-v1/supplement/documents.jsonl")[0]["ingest_exclusion_reason"] == "no_strong_mineral_oil_body_evidence"


def test_existing_shared_lock_is_preserved_without_collection(isolated):
    module, data, candidates, response = isolated
    lock = data / ".corpus_writer.lock"
    lock.write_bytes(b"another writer owns this lock")
    with pytest.raises(FileExistsError):
        module.collect("test-v1", candidates, 100)
    assert lock.read_bytes() == b"another writer owns this lock"
    assert not (data / "unified_runs").exists()


def test_own_lock_is_present_during_work_and_removed_after(isolated):
    module, data, *_ = isolated
    lock = data / ".corpus_writer.lock"
    with module.writer_lock("test-v1"):
        assert lock.exists()
        assert "collect_unified_supplement" in lock.read_text()
        with pytest.raises(FileExistsError):
            with module.writer_lock("second-v1"):
                pytest.fail("A second writer entered the locked region")
    assert not lock.exists()


def test_lock_cleanup_never_deletes_a_replacement_owner(isolated):
    module, data, *_ = isolated
    lock = data / ".corpus_writer.lock"
    with pytest.raises(RuntimeError, match="ownership changed"):
        with module.writer_lock("test-v1"):
            lock.unlink()
            lock.write_bytes(b"replacement writer")
    assert lock.read_bytes() == b"replacement writer"


def test_lock_is_released_on_work_failure(isolated):
    module, data, *_ = isolated
    with pytest.raises(ValueError, match="fixture failure"):
        with module.writer_lock("test-v1"):
            raise ValueError("fixture failure")
    assert not (data / ".corpus_writer.lock").exists()


def test_prior_treasury_document_record_is_preserved_exactly(isolated):
    module, data, candidates, response = isolated
    prior = dict(source("TREASURY-OIL"), url="https://home.treasury.gov/news/press-releases/sb0215/",
                 title="Prior frozen Treasury record", document_id="old-doc", untouched_provenance={"generation": 1})
    target = data / "unified_runs/test-v1/supplement"
    write_jsonl(target / "documents.jsonl", [prior])
    write_jsonl(target / "sources.jsonl", [prior])
    module.collect("test-v1", candidates, 100)
    docs = read_jsonl(target / "documents.jsonl")
    assert docs[0] == prior and len(docs) == 2
    assert read_jsonl(target / 'sources.jsonl')[0] == prior


def test_manifest_pair_mismatch_rejected_before_fetch(isolated):
    module, data, candidates, response = isolated
    target = data / 'unified_runs/test-v1/supplement'
    write_jsonl(target / 'documents.jsonl', [source()])
    with pytest.raises(ValueError, match='complete manifest pair'):
        module.collect('test-v1', candidates, 100)


def test_duplicate_seed_urls_rejected_not_collapsed(isolated):
    module, data, candidates, response = isolated
    target = data / 'unified_runs/test-v1/supplement'
    write_jsonl(target / 'documents.jsonl', [source(), source()])
    write_jsonl(target / 'sources.jsonl', [source(), source()])
    with pytest.raises(ValueError, match='Duplicate URL'):
        module.collect('test-v1', candidates, 100)


def test_eia_adapter_exact_title_day_and_source_native_body(isolated):
    module, data, candidates, response = isolated
    row = source('EIA-OIL-NEWS')
    row['url'] = 'https://www.eia.gov/todayinenergy/detail.php?id=12345'
    write_jsonl(candidates, [row])
    response['raw'] = (f"<div class='tie-article'><div class='date'>April 21, 2022</div>"
                       f"<h1>{TITLE}</h1><p>{BODY}</p>"
                       "<table><tr><td>Future oil output 9999</td></tr></table></div>").encode()
    report = module.collect('test-v1', candidates, 100)
    assert report['dated_scope_accepted'] == 1
    doc = read_jsonl(data / 'unified_runs/test-v1/supplement/documents.jsonl')[0]
    assert doc['body_date_verified'] and doc['body_title_verified']
    assert len(doc['body_parser_sha256']) == 64
    cache = data / 'extracted' / (doc['raw_sha256'] + module.extraction_suffix(doc['source']))
    assert '9999' not in cache.read_text()


def test_eia_oil_title_is_not_body_oil_evidence(isolated):
    module, data, candidates, response = isolated
    row = source('EIA-OIL-NEWS')
    row['url'] = 'https://www.eia.gov/todayinenergy/detail.php?id=12345'
    write_jsonl(candidates, [row])
    response['raw'] = (f"<div class='tie-article'><div class='date'>April 21, 2022</div><h1>{TITLE}</h1>"
                       + ''.join(f'<p>Electric cars consumed renewable electricity and demand for power rose across city district {i}. </p>' for i in range(12))
                       + '</div>').encode()
    report = module.collect('test-v1', candidates, 100)
    assert report['dated_scope_accepted'] == 0
    assert read_jsonl(data / 'unified_runs/test-v1/supplement/documents.jsonl')[0]['ingest_exclusion_reason'] == 'no_strong_mineral_oil_body_evidence'


@pytest.mark.parametrize('modified,excluded', [('2022-05-01', False), ('2026-10-06', True)])
def test_cer_current_version_availability_is_not_backdated(isolated, modified, excluded):
    module, data, candidates, response = isolated
    row = source('CER-OIL-SNAPSHOT')
    row['url'] = 'https://www.cer-rec.gc.ca/en/data-analysis/energy-markets/market-snapshots/2022/market-oil.html'
    write_jsonl(candidates, [row])
    response['raw'] = ("<main property='mainContentOfPage'>"
        f"<h1 id='wb-cont' property='name'>Market Snapshot: {TITLE}</h1><p>Release date: 2022-04-21</p>"
        f"<p>{BODY}</p><div id='wb-dtmd'><time property='dateModified' datetime='{modified}'>{modified}</time></div></main>").encode()
    report = module.collect('test-v1', candidates, 100)
    doc = read_jsonl(data / 'unified_runs/test-v1/supplement/documents.jsonl')[0]
    assert doc['published_date'] == '2022-04-21' and doc['available_by'] == modified
    assert doc['body_title_verified'] is True
    assert report['dated_scope_accepted'] == int(not excluded)
    if excluded:
        assert doc['ingest_exclusion_reason'] == 'current_article_version_outside_corpus_window_or_date_unknown'
