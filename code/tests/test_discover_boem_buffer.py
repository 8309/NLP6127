import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("discover_boem_buffer", Path(__file__).parents[1] / "scripts/discover_boem_buffer.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.mark.parametrize("url", ["http://www.boem.gov/a", "https://www.boem.gov.evil.test/a", "https://user@www.boem.gov/a", "https://www.boem.gov:444/a"])
def test_official_host_is_exact(url):
    assert not mod.official(url)


def test_official_hosts():
    assert mod.official("https://www.boem.gov/a") and mod.official("https://boem.gov/a")


def test_archive_dates_not_program_years():
    assert mod.archive_date("Oil and Gas Program 2024–2029") == (None, None, None)
    assert mod.archive_date("Final Supplemental EIS January 2023") == ("2023-01-31", "month_upper_bound", None)
    assert mod.archive_date("July 8, 2022 Draft Programmatic EIS 2023–2028") == ("2022-07-08", "day", "2022-07-08")


def test_index_volume_one_and_appendices_are_separate():
    raw = b"<main><table><tr><td>Oil and Gas Proposed Program July 8, 2022 Draft Programmatic EIS <a href='/oil-gas-energy/eis-v1'>Volume I</a><a href='/oil-gas-energy/eis-v2'>Volume II</a></td></tr></table></main>"
    rows, excluded, observed = mod.parse_index(raw, mod.SEEDS[0])
    assert len(rows) == 1 and rows[0]["url"].endswith("eis-v1")
    assert rows[0]["available_by"] == "2022-07-08" and rows[0]["requires_pdf"]
    assert excluded["non_oil_or_appendix_document"] == 1


def test_index_old_wind_and_unknown_dates_not_admitted():
    raw = b"<main><ul><li>Oil and Gas EIS July 2016 <a href='/oil-gas-energy/old'>Final EIS</a></li><li>Gulf Renewable Energy Leasing May 2023 <a href='/renewable/wind'>Wind Final Environmental Impact Statement</a></li><li>National OCS Proposed Program 2024-2029 <a href='/oil-gas-energy/new'>Programmatic EIS</a></li></ul></main>"
    rows, rejected, observed = mod.parse_index(raw, mod.SEEDS[0])
    assert not rows
    assert rejected["outside_window"] == 1 and rejected["unknown_or_ambiguous_archive_publication"] == 1


def test_source_native_oil_program_heading_scopes_its_dated_proposal_row():
    raw = b"<main><h1>National OCS Oil and Gas Leasing Program</h1><table><tr><td>Draft Proposed Program November 24, 2025 <a href='/oil-gas-energy/first'>1st Analysis and Proposal PDF</a></td></tr></table></main>"
    rows, _, _ = mod.parse_index(raw, mod.SEEDS[-1])
    assert len(rows) == 1 and rows[0]["available_by"] == "2025-11-24"


def test_attachment_pdf_must_be_observed_and_unique():
    assert mod.linked_pdf(b"<main><a href='/sites/default/files/a.pdf'>PDF</a></main>", mod.SEEDS[0]).endswith("/a.pdf")
    with pytest.raises(ValueError):
        mod.linked_pdf(b"<main><a href='https://evil.test/a.pdf'>PDF</a></main>", mod.SEEDS[0])


def test_cover_month_is_standalone_not_plan_year():
    assert mod.cover_month(["2024–2029 National Program\nSeptember 2023\nBOEM 2023-042"]) == "2023-09"
    with pytest.raises(ValueError):
        mod.cover_month(["National 2024–2029 Program\nThis follows July 2022 meetings."])


def test_cover_native_publisher_verso_date_is_not_lost_in_two_columns():
    assert mod.cover_month(["Oil and Gas EIS", "Author\nBureau of Ocean Energy Management\n"
        "Published by\nU.S. Department of the Interior New Orleans\n"
        "Bureau of Ocean Energy Management August 2025\nGulf of America OCS Region"]) == "2025-08"


@pytest.mark.parametrize("page", [
    "Author\nBureau of Ocean Energy Management August 2025",
    "Published by\nA meeting of Bureau of Ocean Energy Management August 2025",
    "Published by\nBureau of Ocean Energy Management met in August 2025",
])
def test_inline_month_outside_exact_native_publisher_line_is_not_a_cover_date(page):
    with pytest.raises(ValueError, match="lack a standalone"):
        mod.cover_month([page])


@pytest.mark.parametrize("value,expected", [("D:20230922141200-05'00'", "2023-09-22"), ("D:20230230000000", None), (None, None), ("Unknown", None)])
def test_pdf_timestamp_validation(value, expected):
    assert mod.pdf_timestamp(value) == expected


def test_pdf_date_order_and_physical_pages(monkeypatch):
    class FakePage:
        def __init__(self, text): self.text = text
        def extract_text(self): return self.text
    class FakeReader:
        pages = [FakePage("Oil and Gas\nSeptember 2023"), FakePage("")]
        metadata = {"/CreationDate": "D:20230905000000", "/ModDate": "D:20230928000000", "/Title": "Oil and Gas EIS"}
    monkeypatch.setattr(mod, "PdfReader", lambda stream: FakeReader())
    source = dict(available_by="2023-10-02", published_date="2023-10-02", date_precision="day")
    parsed = mod.parse_boem_pdf(b"%PDF", source)
    assert parsed["pages"] == ["Oil and Gas\nSeptember 2023", ""]
    assert parsed["physical_pages"] == 2 and parsed["date_confirmed"]
    FakeReader.metadata["/ModDate"] = "D:20240301000000"
    revised = mod.parse_boem_pdf(b"%PDF", source)
    assert revised["available_by"] == "2024-03-01" and revised["published_date"] == "2023-10-02"
    assert revised["indexed_available_by"] == "2023-10-02" and revised["current_version_date_adjusted"]
    assert "not backdated" in revised["version_caution"]
    FakeReader.metadata["/ModDate"] = "D:20261006000000"
    with pytest.raises(ValueError, match="fixed corpus cutoff"):
        mod.parse_boem_pdf(b"%PDF", source)


def test_pdf_old_cover_cannot_be_rescued_by_later_archive(monkeypatch):
    class FakePage:
        def extract_text(self): return "National Program 2018\nDecember 2017"
    class FakeReader:
        pages = [FakePage()]
        metadata = {"/CreationDate": "D:20171205000000"}
    monkeypatch.setattr(mod, "PdfReader", lambda stream: FakeReader())
    with pytest.raises(ValueError, match="cover is outside"):
        mod.parse_boem_pdf(b"%PDF", dict(available_by="2018-01-31", published_date=None, date_precision="month_upper_bound"))


def test_redirect_exhaustion_is_not_outer_retryable_timeout(monkeypatch):
    monkeypatch.setattr(mod.Fetcher, "get", lambda *args, **kwargs: (_ for _ in ()).throw(mod.httpx.ReadTimeout("bounded target failed")))
    fetch = mod.BoemFetcher()
    try:
        with pytest.raises(RuntimeError, match="exhausted bounded"):
            fetch.get("https://www.boem.gov/file.pdf", 1)
        assert fetch.requested_urls == [dict(url="https://www.boem.gov/file.pdf", redirect_depth=1)]
        with pytest.raises(ValueError, match="official"):
            fetch.get("https://evil.test/file.pdf", 1)
    finally:
        fetch.close()


def test_graphic_cover_sidecar_is_exact_raw_and_render_bound(monkeypatch, tmp_path):
    raw = b"%PDF graphic-cover fixture"
    png = b"fixture-root-reviewed-rendering"
    path = tmp_path / "unified_runs/fixture/reports/cover.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(png)
    expected = dict(machine_cover_month="2022-07", physical_page=1, rendered_png_sha256=mod.digest(png))
    monkeypatch.setattr(mod, "DATA", tmp_path)
    monkeypatch.setattr(mod, "APPROVED_VISUAL_COVERS", {mod.digest(raw): expected})
    supplied = dict(expected, raw_sha256=mod.digest(raw), rendered_png_path=str(path))
    assert mod.validated_visual_cover_month(raw, supplied) == "2022-07"
    for key, bad in [("raw_sha256", "0" * 64), ("machine_cover_month", "2021-07"),
                     ("physical_page", 2), ("physical_page", True), ("rendered_png_sha256", "0" * 64)]:
        with pytest.raises(ValueError):
            mod.validated_visual_cover_month(raw, dict(supplied, **{key: bad}))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        mod.validated_visual_cover_month(raw, supplied)


def test_unknown_graphic_cover_cannot_use_other_review(monkeypatch):
    monkeypatch.setattr(mod, "APPROVED_VISUAL_COVERS", {})
    with pytest.raises(ValueError, match="not separately root reviewed"):
        mod.validated_visual_cover_month(b"%PDF unknown", dict(raw_sha256=mod.digest(b"%PDF unknown")))
