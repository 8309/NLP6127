"""Source version and page-provenance checks with isolated fixtures only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from oil_evidence.store import digest, read_jsonl, write_jsonl


@pytest.fixture
def versions(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("isolated_new_source_versions", scripts / "validate_new_sources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value,expected", [
    ("D:20240229123000+08'00'", "2024-02-29"),
    ("20230102000000Z", "2023-01-02"),
    ("D:20250120", "2025-01-20"),
    ("D:20230229", None), ("D:20241301", None), ("D:20240100", None),
    ("D:202401", None), ("not a date", None), (None, None), ("", None),
])
def test_pdf_date_accepts_valid_days_and_unknowns_fail_without_invention(versions, value, expected):
    assert versions.pdf_date(value) == expected


def _isolated_validation(versions, tmp_path, monkeypatch, source, raw, metadata=None, available="2024-04-01"):
    data = tmp_path / "data"
    root = data / "expansion_runs/test-v2"
    (data / "raw").mkdir(parents=True)
    path = data / "raw/new-source.bin"
    path.write_bytes(raw)
    document = dict(document_id="new-document", source=source, url="https://example.test/new",
                    available_by=available, published_date=available, date_precision="day",
                    raw_path="raw/new-source.bin", raw_sha256=digest(raw), requires_pdf=raw.startswith(b"%PDF"))
    write_jsonl(data / "documents.jsonl", [document])
    write_jsonl(data / "sources.jsonl", [document])
    write_jsonl(root / "baseline/documents.jsonl", [])
    monkeypatch.setattr(versions, "DATA", data)
    monkeypatch.setattr(versions, "select_run", lambda run: root)
    monkeypatch.setattr(versions, "snapshot", lambda: {})
    monkeypatch.setattr(versions, "verify_baseline_sources", lambda: None)
    if raw.startswith(b"%PDF"):
        page = SimpleNamespace(extract_text=lambda: "Official report cover")
        monkeypatch.setattr(versions, "PdfReader", lambda path: SimpleNamespace(
            metadata=metadata or {}, pages=[page]))
    return data, root


@pytest.mark.parametrize("metadata,excluded", [
    ({"/CreationDate": "D:20240301", "/ModDate": "D:20240402"}, True),
    ({"/CreationDate": "D:20240402", "/ModDate": "D:20240301"}, True),
    ({"/CreationDate": "D:20240401", "/ModDate": "D:20240401"}, False),
    ({"/CreationDate": "D:20230301"}, False),
])
def test_uk_attachment_later_file_date_is_quarantined(versions, tmp_path, monkeypatch, metadata, excluded):
    data, _ = _isolated_validation(versions, tmp_path, monkeypatch, "UK-DUKES", b"%PDF fixture", metadata)
    report = versions.validate("test-v2")
    row = read_jsonl(data / "documents.jsonl")[0]
    assert bool(row.get("ingest_exclusion_reason")) == excluded
    assert report["excluded"] == int(excluded)
    assert row["available_by"] == "2024-04-01"  # Never invent a later historical release.
    source = read_jsonl(data / "sources.jsonl")[0]
    assert source.get("ingest_exclusion_reason") == row.get("ingest_exclusion_reason")


@pytest.mark.parametrize("term,available,excluded", [
    ("Gulf of America", "2020-06-01", True),
    ("Gulf of Mexico", "2020-06-01", False),
    ("Gulf of America", "2025-06-01", False),
])
def test_renamed_bsee_body_not_used_as_unchanged_old_snapshot(versions, tmp_path, monkeypatch, term, available, excluded):
    raw = (f'<main><article><div class="field--name-body">Crude oil production in the {term}.</div></article></main>').encode()
    data, _ = _isolated_validation(versions, tmp_path, monkeypatch, "BSEE-HURRICANE", raw, available=available)
    report = versions.validate("test-v2")
    row = read_jsonl(data / "documents.jsonl")[0]
    assert bool(row.get("ingest_exclusion_reason")) == excluded
    assert report["excluded"] == int(excluded)
    assert row["available_by"] == available


@pytest.mark.parametrize("body", ["<article>Gulf of America oil production.</article>",
                                 '<article><div class="field--name-field-body">Gulf of America oil production.</div></article>'])
def test_unknown_bsee_body_shape_fails_closed(versions, tmp_path, monkeypatch, body):
    data, _ = _isolated_validation(versions, tmp_path, monkeypatch, "BSEE-HURRICANE",
                                   ("<main>" + body + "</main>").encode(), available="2020-06-01")
    report = versions.validate("test-v2")
    assert report["excluded"] == 1
    assert read_jsonl(data / "documents.jsonl")[0].get("ingest_exclusion_reason")


@pytest.mark.parametrize("source", ["AU-REQ", "UK-DUKES", "UK-ENERGY-TRENDS", "DOE-SPR",
                                    "BOEM-POLICY", "EIA-METHODS", "EIA-FINANCE", "EIA-IEO", "BSEE-HURRICANE"])
def test_new_source_extraction_cannot_reuse_legacy_body_cache(source):
    from oil_evidence.corpus import extraction_suffix

    assert extraction_suffix(source).endswith(".json")
    assert extraction_suffix(source) not in {extraction_suffix("EIA"), extraction_suffix("WB-CMO")}


def test_original_cache_versions_remain_unchanged():
    from oil_evidence.corpus import extraction_suffix

    assert extraction_suffix("EIA") == "-body-v2.json"
    assert extraction_suffix("EIA-CAB") == "-body-v2.json"
    assert extraction_suffix("WB-CMO") == "-body-v3-fonttools.json"


def test_scope_dependent_series_have_different_cache_suffixes():
    from oil_evidence.corpus import NEW_PDF_SERIES, extraction_suffix

    assert len({extraction_suffix(source) for source in NEW_PDF_SERIES}) == len(NEW_PDF_SERIES)


def test_running_furniture_removal_preserves_physical_page_slots():
    from oil_evidence.corpus import remove_running_headers

    header = "Resources and Energy Quarterly | March 2024"
    pages = [header + "\nDistinct paragraph one.",
             header + "\nDistinct paragraph two.", "",
             header + "\nDistinct paragraph three."]
    original = pages.copy()
    result = remove_running_headers(pages)
    assert len(result) == len(pages) and result[2] == ""
    assert all(header not in page for page in result)
    assert result == ["Distinct paragraph one.", "Distinct paragraph two.", "", "Distinct paragraph three."]
    assert pages == original


def test_repeated_body_units_are_not_page_furniture():
    from oil_evidence.corpus import remove_running_headers

    units = "All volumes are in million barrels per day."
    pages = [f"Distinct heading {index}.\nDistinct introductory sentence {index}.\n{units}\nDistinct conclusion {index}.\nDistinct footer {index}."
             for index in range(3)]
    assert all(units in page for page in remove_running_headers(pages))


def test_new_annual_oil_scope_keeps_actual_pdf_page_number(tmp_path, monkeypatch):
    import oil_evidence.corpus as corpus

    monkeypatch.setattr(corpus, "DATA", tmp_path)
    monkeypatch.setattr(corpus, "config", lambda: {"corpus": dict(
        start_date="2018-01-01", end_date="2026-10-05", chunk_words=200, min_words=65)})
    prose = (
        "In this country crude oil production depended on offshore infrastructure and access to export pipelines throughout the reporting year. "
        "Supply disruptions at the terminal reduced deliveries to refineries and increased reliance on imported petroleum products during seasonal demand peaks. "
        "Government policy addressed strategic inventories and transport constraints because those factors materially affected the availability of crude oil for domestic refining operations. "
        "The report also distinguished crude oil production from total liquids output when explaining the supply balance."
    )
    pages = ["7. Gas\nNon-oil chapter mentions oil costs. " + prose,
             "8. Oil\n" + prose,
             "9. Gold\nNon-oil chapter mentions oil costs. " + prose]
    raw = b"%PDF isolated page fixture"
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/book.pdf").write_bytes(raw)
    write_jsonl(tmp_path / "documents.jsonl", [dict(document_id="mixed-book", source="AU-REQ",
        title="Annual commodity report", url="https://example.test/book.pdf", raw_path="raw/book.pdf",
        raw_sha256=digest(raw), available_by="2024-06-30", published_date=None,
        date_precision="month_upper_bound")])
    monkeypatch.setattr(corpus, "PdfReader", lambda raw: SimpleNamespace(
        pages=[SimpleNamespace(extract_text=lambda text=text: text) for text in pages]))
    report = corpus.build_corpus(output_dir=tmp_path / "stage")
    rows = read_jsonl(tmp_path / "stage/passages.jsonl")
    assert not report["extraction_errors"] and rows
    assert {p["page"] for p in rows} == {2}
    assert all("Non-oil chapter" not in p["text"] for p in rows)
    cached = __import__("json").loads((tmp_path / "extracted" /
        (digest(raw) + corpus.extraction_suffix("AU-REQ"))).read_text(encoding="utf-8"))
    assert len(cached) == 3 and cached[0] == cached[2] == "" and cached[1]


def test_ingest_exclusion_skips_even_existing_extract_cache(tmp_path, monkeypatch):
    import oil_evidence.corpus as corpus

    monkeypatch.setattr(corpus, "DATA", tmp_path)
    monkeypatch.setattr(corpus, "config", lambda: {"corpus": dict(chunk_words=200, min_words=65)})
    write_jsonl(tmp_path / "documents.jsonl", [dict(document_id="excluded", source="UK-DUKES",
        raw_path="not-even-read.pdf", raw_sha256="unused",
        ingest_exclusion_reason="Current attachment postdates historic release")])
    report = corpus.build_corpus(output_dir=tmp_path / "stage")
    assert report["passages"] == 0 and not report["extraction_errors"]
