"""Offline collector I/O guards in tmp DATA; fixture PDFs are not real data."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import httpx
import pytest

from oil_evidence.store import digest, read_jsonl, write_json, write_jsonl


RUN = "test-boem-buffer"
PDF = b"%PDF-1.7\nsynthetic offline guard fixture"
PAGES = [
    "September 2023\nOil and gas leasing scenarios are proposals, not realized production.",
    "Crude oil production was discussed in the environmental assessment.\n"
    "The alternatives preserve original units and assumptions rather than predicting actual output.",
]


def source(number=1):
    url = f"https://www.boem.gov/sites/default/files/draft-peis-vol1-fixture-{number}.pdf"
    return dict(url=url, source="BOEM-EIS", title=f"BOEM Oil and Gas Draft EIS {number}",
                published_date="2023-09-30", available_by="2023-09-30", date_precision="day",
                date_source="https://www.boem.gov/oil-gas-energy/environment",
                event_date=None, event_status="unlabelled", redistribution="review_required",
                requires_pdf=True, discovery_method="actually_linked_fixture_index")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("isolated_boem_collection_guards",
                                                scripts / "collect_boem_supplement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    helper = importlib.import_module("collect_unified_supplement")
    parser = importlib.import_module("discover_boem_buffer")
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(module, "DATA", data)
    monkeypatch.setattr(helper, "DATA", data)
    # Hash-change tests touch only these temporary __file__ stand-ins, never a
    # running collector/parser or another agent's actual implementation.
    implementation = tmp_path / "collector-hash-stand-in.py"
    parser_file = tmp_path / "parser-hash-stand-in.py"
    implementation.write_bytes(b"immutable collector implementation fixture")
    parser_file.write_bytes(b"immutable parser implementation fixture")
    monkeypatch.setattr(module, "__file__", str(implementation))
    monkeypatch.setattr(parser, "__file__", str(parser_file))
    for name in ["sources.jsonl", "documents.jsonl", "passages.jsonl",
                 "annotations/codex.jsonl", "annotations/passage_labels.jsonl",
                 "annotations/passage_review.jsonl"]:
        write_jsonl(data / name, [])
    for name in ["reports/corpus.json", "reports/corpus_resegmentation.json",
                 "reports/passage_annotation.json", "index/manifest.json"]:
        write_json(data / name, {"fixture": name})
    for name in ["index/vectors.npy", "model_usage.sqlite3", "annotations.sqlite3"]:
        (data / name).write_bytes(b"unchanged synthetic ledger/index bytes")
    (data / "raw").mkdir()
    (data / "raw/prior.pdf").write_bytes(b"%PDF previous frozen bytes")
    write_json(data / "extracted/prior.json", ["Prior extraction stays immutable."])
    write_json(data / "unified_runs/semantic-page-v1/reports/release_hold.json",
               {"fixture": "old held run is protected"})
    candidate = tmp_path / "candidates.jsonl"
    write_jsonl(candidate, [source()])
    state = dict(raw=PDF, final=None, calls=[], closed=0, mutation=None,
                 settings=dict(request_delay_seconds=1, max_document_mb=25, download_budget_mb=300))

    class FakeFetcher:
        def __init__(self):
            self.settings = dict(state["settings"])
            self.bytes = 0

        def get(self, url):
            state["calls"].append(url)
            if state["mutation"]:
                state["mutation"]()
            if state.get("fetch_error"):
                raise state["fetch_error"]
            raw = state.get("raw_by_url", {}).get(url, state["raw"])
            self.bytes += len(raw)
            return raw, state["final"] or url, "application/pdf"

        def close(self):
            state["closed"] += 1

    def parse_pdf(raw, row):
        if state.get("parser_error"):
            raise ValueError(state["parser_error"])
        return dict(pages=list(state.get("pages", PAGES)), published_date=row["published_date"],
                    available_by=row["available_by"], date_precision=row["date_precision"],
                    date_confirmed=True, cover_month="2023-09", pdf_creation_date="2023-09-20",
                    pdf_modified_date="2023-09-20", parser="synthetic_offline_parser_v1")

    monkeypatch.setattr(module, "BoemFetcher", FakeFetcher)
    monkeypatch.setattr(module, "parse_boem_pdf", parse_pdf)
    return module, data, candidate, state, implementation, parser_file


def manifests(data):
    target = data / "unified_runs" / RUN / "supplement"
    return target, read_jsonl(target / "documents.jsonl"), read_jsonl(target / "sources.jsonl")


def file_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_success_preserves_live_raw_cache_ledger_and_prior_run(isolated):
    module, data, candidates, state, *_ = isolated
    original = file_bytes(data)
    report = module.collect(RUN, candidates)
    _, docs, sources = manifests(data)
    assert report["added_raw_documents"] == report["added_version_checked_pdf_documents"] == 1
    assert report["external_model_calls"] == report["human_labels_created"] == 0
    assert report["accepted_passages"] == "not_counted_until_unchanged_unified_build"
    assert len(docs) == len(sources) == 1 and docs[0]["body_date_verified"]
    for name, value in original.items():
        assert (data / name).read_bytes() == value
    assert not (data / ".corpus_writer.lock").exists()


def test_second_invocation_skips_identical_url_without_refetch_or_overwrite(isolated):
    module, data, candidates, state, *_ = isolated
    module.collect(RUN, candidates)
    target, docs, sources = manifests(data)
    original_manifests = [(target / name).read_bytes() for name in ["documents.jsonl", "sources.jsonl"]]
    original_raw, original_cache = file_bytes(data / "raw"), file_bytes(data / "extracted")
    original_receipts = {p.name: p.read_bytes() for p in target.glob("pdf_collection_report-*.json")}
    state["calls"].clear()
    report = module.collect(RUN, candidates)
    assert report["attempts"] == report["added_raw_documents"] == 0 and state["calls"] == []
    assert manifests(data)[1:] == (docs, sources)
    assert [(target / name).read_bytes() for name in ["documents.jsonl", "sources.jsonl"]] == original_manifests
    assert file_bytes(data / "raw") == original_raw and file_bytes(data / "extracted") == original_cache
    assert len(list(target.glob("pdf_collection_report-*.json"))) == 2
    for name, value in original_receipts.items():
        assert (target / name).read_bytes() == value


def test_resume_appends_new_object_without_normalizing_prior_objects(isolated):
    module, data, candidates, state, *_ = isolated
    module.collect(RUN, candidates)
    target, docs, sources = manifests(data)
    # Arbitrary provenance fields must survive resumed collection exactly.
    docs[0]["custom_provenance"] = {"generation": 1, "notes": ["not a model label"]}
    sources[0]["custom_source_provenance"] = {"generation": 1}
    write_jsonl(target / "documents.jsonl", docs)
    write_jsonl(target / "sources.jsonl", sources)
    old_raw, old_cache = file_bytes(data / "raw"), file_bytes(data / "extracted")
    write_jsonl(candidates, [source(), source(2)])
    state["raw_by_url"] = {source(2)["url"]: PDF + b"\nsecond fixture"}
    report = module.collect(RUN, candidates)
    _, final_docs, final_sources = manifests(data)
    assert report["added_raw_documents"] == 1 and len(final_docs) == 2
    assert final_docs[0] == docs[0] and final_sources[0] == sources[0]
    for name, value in old_raw.items():
        assert (data / "raw" / name).read_bytes() == value
    for name, value in old_cache.items():
        assert (data / "extracted" / name).read_bytes() == value


def test_transient_failure_is_logged_and_a_later_invocation_can_resume(isolated):
    module, data, candidates, state, *_ = isolated
    state["fetch_error"] = httpx.ReadTimeout("fixture timeout; no real request")
    first = module.collect(RUN, candidates)
    target, docs, _ = manifests(data)
    assert first["added_raw_documents"] == 0 and docs == []
    assert read_jsonl(target / "pdf_collection.jsonl")[0]["status"] == "failed_no_bypass"
    state.pop("fetch_error")
    resumed = module.collect(RUN, candidates)
    assert resumed["added_raw_documents"] == 1
    assert [r["status"] for r in read_jsonl(target / "pdf_collection.jsonl")] == ["failed_no_bypass", "downloaded_version_checked_pdf"]


def test_registered_live_url_is_skipped_without_mutating_live_manifest(isolated):
    module, data, candidates, state, *_ = isolated
    write_jsonl(data / "documents.jsonl", [source()])
    original = (data / "documents.jsonl").read_bytes()
    report = module.collect(RUN, candidates)
    assert report["attempts"] == 0 and state["calls"] == []
    assert (data / "documents.jsonl").read_bytes() == original


def test_raw_conflict_is_not_overwritten_or_registered(isolated):
    module, data, candidates, *_ = isolated
    path = data / "raw" / (digest(source()["url"])[:20] + ".pdf")
    path.write_bytes(b"%PDF immutable conflicting source")
    report = module.collect(RUN, candidates)
    target, docs, _ = manifests(data)
    assert report["added_raw_documents"] == 0 and docs == []
    assert path.read_bytes() == b"%PDF immutable conflicting source"
    assert "cannot be overwritten" in read_jsonl(target / "pdf_collection.jsonl")[0]["error"]


def test_existing_identical_raw_is_reused_without_byte_changes(isolated):
    module, data, candidates, *_ = isolated
    path = data / "raw" / (digest(source()["url"])[:20] + ".pdf")
    path.write_bytes(PDF)
    module.collect(RUN, candidates)
    assert path.read_bytes() == PDF and len(manifests(data)[1]) == 1


def test_extraction_cache_conflict_stays_unchanged_and_excludes_document(isolated):
    module, data, candidates, *_ = isolated
    cache = data / "extracted" / (digest(PDF) + module.extraction_suffix("BOEM-EIS"))
    write_json(cache, ["Prior immutable extraction"])
    original = cache.read_bytes()
    report = module.collect(RUN, candidates)
    doc = manifests(data)[1][0]
    assert cache.read_bytes() == original
    assert report["added_raw_documents"] == 1 and report["added_version_checked_pdf_documents"] == 0
    assert doc["ingest_exclusion_reason"] == "unverified_pdf_publication_or_extraction"
    assert "refuse overwrite" in doc["body_parser_error"]


def test_extraction_audit_conflict_is_not_overwritten(isolated):
    module, data, candidates, *_ = isolated
    target = data / "unified_runs" / RUN / "supplement"
    audit = target / "pdf_extraction_audits" / (digest(source()["url"])[:20] + ".json")
    write_json(audit, {"prior": "immutable extraction audit"})
    original = audit.read_bytes()
    report = module.collect(RUN, candidates)
    assert audit.read_bytes() == original
    assert report["added_version_checked_pdf_documents"] == 0
    assert "audit differs" in manifests(data)[1][0]["body_parser_error"]


def test_parser_date_failure_preserves_raw_and_explicit_excluded_record(isolated):
    module, data, candidates, state, *_ = isolated
    state["parser_error"] = "PDF current version date cannot be verified"
    report = module.collect(RUN, candidates)
    _, docs, sources = manifests(data)
    assert report["added_raw_documents"] == 1 and report["added_version_checked_pdf_documents"] == 0
    doc = docs[0]
    assert (data / doc["raw_path"]).read_bytes() == PDF
    assert doc["body_date_verified"] is False
    assert doc["ingest_exclusion_reason"] == "unverified_pdf_publication_or_extraction"
    assert state["parser_error"] == doc["body_parser_error"]
    assert sources[0]["ingest_exclusion_reason"] == doc["ingest_exclusion_reason"]
    assert not (data / "extracted" / (digest(PDF) + module.extraction_suffix("BOEM-EIS"))).exists()


@pytest.mark.parametrize("problem", ["missing_pair", "duplicate", "url_set", "metadata"])
def test_invalid_manifest_pair_rejected_before_any_fetch(isolated, problem):
    module, data, candidates, state, *_ = isolated
    target = data / "unified_runs" / RUN / "supplement"
    if problem == "missing_pair":
        write_jsonl(target / "documents.jsonl", [source()])
    elif problem == "duplicate":
        write_jsonl(target / "documents.jsonl", [source(), source()])
        write_jsonl(target / "sources.jsonl", [source(), source()])
    elif problem == "url_set":
        write_jsonl(target / "documents.jsonl", [source()])
        write_jsonl(target / "sources.jsonl", [source(2)])
    else:
        write_jsonl(target / "documents.jsonl", [source()])
        write_jsonl(target / "sources.jsonl", [dict(source(), title="Conflicting source title")])
    before = file_bytes(target)
    with pytest.raises(ValueError):
        module.collect(RUN, candidates)
    assert state["calls"] == [] and file_bytes(target) == before
    assert not (data / ".corpus_writer.lock").exists()


@pytest.mark.parametrize("final", ["https://www.boem.gov.evil.test/file.pdf",
                                  "http://www.boem.gov/file.pdf", "https://user@www.boem.gov/file.pdf",
                                  "https://www.boem.gov:444/file.pdf", "https://evil.test/file.pdf"])
def test_unofficial_final_redirect_never_registers_or_saves_pdf(isolated, final):
    module, data, candidates, state, *_ = isolated
    state["final"] = final
    before = file_bytes(data / "raw")
    report = module.collect(RUN, candidates)
    assert report["added_raw_documents"] == 0 and manifests(data)[1] == []
    assert file_bytes(data / "raw") == before
    assert "Official attachment" in read_jsonl(manifests(data)[0] / "pdf_collection.jsonl")[0]["error"]


@pytest.mark.parametrize("key,value", [("request_delay_seconds", 0.99),
                                       ("max_document_mb", 25.01), ("download_budget_mb", 300.01)])
def test_relaxed_rate_size_or_budget_parameters_fail_closed_before_fetch(isolated, key, value):
    module, data, candidates, state, *_ = isolated
    state["settings"][key] = value
    with pytest.raises(ValueError, match="rate/size/budget"):
        module.collect(RUN, candidates)
    assert state["calls"] == [] and state["closed"] == 1
    assert not (data / ".corpus_writer.lock").exists()


@pytest.mark.parametrize("which", ["collector", "parser", "candidate", "prior_raw", "ledger"])
def test_mid_invocation_hash_change_rejected_using_temporary_files_only(isolated, which):
    module, data, candidates, state, implementation, parser_file = isolated
    paths = dict(collector=implementation, parser=parser_file, candidate=candidates,
                 prior_raw=data / "raw/prior.pdf", ledger=data / "model_usage.sqlite3")
    state["mutation"] = lambda: paths[which].write_bytes(b"changed fixture bytes")
    with pytest.raises(ValueError, match="changed during PDF collection"):
        module.collect(RUN, candidates)
    assert not (data / ".corpus_writer.lock").exists()


def test_published_run_rejected_before_fetch(isolated):
    module, data, candidates, state, *_ = isolated
    (data / "unified_runs" / RUN / "published").mkdir(parents=True)
    with pytest.raises(ValueError, match="immutable"):
        module.collect(RUN, candidates)
    assert state["calls"] == []


def test_shared_writer_lock_is_not_removed_by_other_collector(isolated):
    module, data, candidates, state, *_ = isolated
    lock = data / ".corpus_writer.lock"
    lock.write_bytes(b"another fixture writer owns this lock")
    with pytest.raises(FileExistsError):
        module.collect(RUN, candidates)
    assert lock.read_bytes() == b"another fixture writer owns this lock" and state["calls"] == []


def test_verified_pilot_reuse_is_local_and_hash_bound_without_fetch(isolated):
    module, data, candidates, state, *_ = isolated
    pilot = data / "unified_runs" / RUN / "reports/pilot.pdf"
    pilot.parent.mkdir(parents=True)
    pilot.write_bytes(PDF)
    row = dict(source(), pilot_pdf_path=str(pilot), pilot_raw_sha256=digest(PDF),
               pilot_final_url=source()["url"])
    write_jsonl(candidates, [row])
    report = module.collect(RUN, candidates)
    assert report["added_version_checked_pdf_documents"] == 1 and state["calls"] == []
    assert manifests(data)[1][0]["raw_acquisition"] == "verified_publisher_pdf_pilot_reuse"


@pytest.mark.parametrize("problem", ["outside_run", "wrong_hash"])
def test_unverified_pilot_reuse_is_rejected_without_fetch(isolated, problem):
    module, data, candidates, state, *_ = isolated
    pilot = data / ("unrelated/pilot.pdf" if problem == "outside_run" else f"unified_runs/{RUN}/reports/pilot.pdf")
    pilot.parent.mkdir(parents=True)
    pilot.write_bytes(PDF)
    row = dict(source(), pilot_pdf_path=str(pilot), pilot_raw_sha256="0" * 64 if problem == "wrong_hash" else digest(PDF),
               pilot_final_url=source()["url"])
    write_jsonl(candidates, [row])
    report = module.collect(RUN, candidates)
    assert report["added_raw_documents"] == 0 and state["calls"] == []


def test_attachment_page_follows_only_its_observed_official_pdf(isolated):
    module, data, candidates, state, *_ = isolated
    entry = dict(source(), url="https://www.boem.gov/oil-gas-energy/fixture-eis")
    attachment = "https://www.boem.gov/sites/default/files/actually-linked-fixture.pdf"
    write_jsonl(candidates, [entry])
    state["raw_by_url"] = {entry["url"]: f"<main><a href='{attachment}'>PDF</a></main>".encode(), attachment: PDF}
    report = module.collect(RUN, candidates)
    assert state["calls"] == [entry["url"], attachment]
    assert report["added_version_checked_pdf_documents"] == 1


def test_offsite_only_attachment_link_is_not_fetched(isolated):
    module, data, candidates, state, *_ = isolated
    state["raw"] = b"<main><a href='https://evil.test/file.pdf'>PDF</a></main>"
    report = module.collect(RUN, candidates)
    assert state["calls"] == [source()["url"]] and report["added_raw_documents"] == 0
