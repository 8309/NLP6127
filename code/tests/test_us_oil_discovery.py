import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def adapter():
    spec = importlib.util.spec_from_file_location(
        "us_oil_discovery", Path(__file__).parents[1] / "scripts/discover_us_oil.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bsee_archive_day_and_abbreviations(adapter):
    raw = '''<ul>
    <li><a href="/newsroom/ida">BSEE Hurricane Ida Activity Final Report: September 23, 2021</a></li>
    <li><a href="/newsroom/michael">Tropical Storm Michael Activity Final Report: Oct. 16, 2018</a></li>
    <li>September 5, 2018 - <a href="/newsroom/gordon">Tropical Storm Gordon Activity Statistics</a></li>
    </ul>'''
    rows = adapter.bsee_sources(raw)
    assert [row["available_by"] for row in rows] == ["2021-09-23", "2018-10-16", "2018-09-05"]
    assert all(row["published_date"] == row["available_by"] for row in rows)


def test_bsee_no_future_old_navigation_or_reused_url(adapter):
    raw = '''<ul>
    <li><a href="/newsroom/future">October 6, 2026 - MMA Monitors Oil and Gas Activities</a></li>
    <li><a href="/newsroom/old">Hurricane Activity Final Report: September 1, 2017</a></li>
    <li><a href="https://example.org/newsroom/foo">September 1, 2021 - Activity Report</a></li>
    <li><a href="/newsroom/reused">September 1, 2021 - Oil Activity</a></li>
    <li><a href="/newsroom/reused">September 2, 2021 - Oil Activity</a></li>
    </ul>'''
    assert adapter.bsee_sources(raw) == []


def test_doe_calendar_year_is_not_publication_date(adapter):
    raw = '<a href="/documents/spr-2023-annual-report-congress">Calendar Year 2023</a>'
    row = adapter.doe_spr_sources(raw)[0]
    assert row["available_by"] == "2026-07-31"
    assert row["published_date"] is None
    assert row["date_precision"] == "month_upper_bound"
    assert row["event_date"] is None and row["reporting_year"] == 2023
    assert row["requires_pdf"] is True


def test_doe_later_upload_and_changed_unknown_links_fail_closed(adapter):
    url = adapter.DOE_REPORTS[2022]["url"]
    rows = adapter.doe_spr_sources(f'<a href="{url}">Calendar Year 2022</a>')
    assert rows[0]["available_by"] == "2023-12-31"
    assert rows[0]["date_evidence"]["cover_month"] == "2023-10"
    raw = '<a href="https://example.org/report.pdf">Calendar Year 2021</a><a href="/new">Calendar Year 2024</a>'
    assert adapter.doe_spr_sources(raw) == []


def test_methodology_only_observed_smaller_links(adapter):
    url = next(iter(adapter.BOEM_REPORTS))
    raw = f'<a href="{url}">Final Methodology</a><a href="/full-final-program">Full Program</a>'
    rows = adapter.boem_small_sources(raw)
    assert len(rows) == 1 and rows[0]["available_by"] == "2023-09-30"
    assert rows[0]["requires_pdf"] is True
    report = list(adapter.BOEM_REPORTS.values())[1]
    draft = adapter.boem_small_sources(f'<a href="{report["archive_link"]}">Draft Methodology</a>')
    assert len(draft) == 1 and draft[0]["available_by"] == "2022-07-31"


def test_discover_uses_supplied_fetcher_without_writes(adapter):
    class FakeFetcher:
        def __init__(self):
            self.calls = []

        def get(self, url):
            self.calls.append(url)
            return b"<html></html>", url, "text/html"

    fetch = FakeFetcher()
    result = adapter.discover(fetch, include_boem=False)
    assert fetch.calls == [adapter.BSEE_ARCHIVE, adapter.DOE_ARCHIVE]
    assert result == []


def test_invalid_dates_fail_closed(adapter):
    assert adapter.day_from_archive("September 99, 2024") is None
    with pytest.raises(ValueError):
        adapter.month_end("2024-13")
