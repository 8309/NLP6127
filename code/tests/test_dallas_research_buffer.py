"""Offline Dallas research adapters/network constraints; not real-source QA."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest

from oil_evidence.store import read_jsonl, write_jsonl


spec = importlib.util.spec_from_file_location("dallas_research_guard_tests",
                                            Path(__file__).parents[1] / "scripts/discover_dallas_research_buffer.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


BODY = " ".join([
    "Crude oil production increased in 2024 and petroleum exports supplied refineries across international markets.",
    "The reported stock of crude oil is distinct from daily production measured in barrels per day.",
    "Refinery outages reduce gasoline and diesel supply while changing crude feedstock demand and the resulting crack spread.",
    "Researchers use model scenarios to compare oil supply disruptions and do not treat projections as realized output.",
    "The source preserves the statistical year and distinguishes observed petroleum flows from future policy assumptions.",
    "Crude oil imports and product exports have different commodity definitions, and the analysis retains those original units.",
    "Oil inventories and tanker movements are reported separately from capacity and completed deliveries in this analysis.",
])


def article(title="Crude oil exports increased", day="June 23, 2026", text=BODY, extra=""):
    return (f"<nav>January 1, 2030</nav><div id='content'><div class='dal-main-content'>"
            f"<h1 class='dal-headline'>{title}</h1><div class='dal-author'>Research Author</div>"
            f"<div class='dal-inline-list'><p class='dal-content-date'>{day}</p></div>"
            f"<p>{text}</p>{extra}<div class='dal-info__container'><p>BADPROFILE</p>"
            "<div class='dal-disclaimer'>Authors' views are not Federal Reserve policy.</div>"
            "</div></div></div>").encode()


@pytest.mark.parametrize("url", [
    "http://www.dallasfed.org/research/economics/2026/0623",
    "https://www.dallasfed.org.evil.test/research/economics/2026/0623",
    "https://user@www.dallasfed.org/research/economics/2026/0623",
    "https://www.dallasfed.org:444/research/economics/2026/0623",
    "https://www.dallasfed.org/research/economics/2026/0623?version=2",
    "https://www.dallasfed.org/research/economics/2026/0623#footnote",
    "https://www.dallasfed.org/events/2026/oil",
    "https://www.dallasfed.org/research/surveys/des/2026/2601",
    "https://www.dallasfed.org/research/papers/2026/wp2625",
])
def test_unadmitted_or_ambiguous_article_routes(url):
    with pytest.raises(ValueError):
        mod.admitted_dallas_research_route(url)


@pytest.mark.parametrize("url", [
    "https://www.dallasfed.org/research/economics/2026/0623",
    "https://www.dallasfed.org/research/swe/2026/swe2614",
    "https://www.dallasfed.org/research/energy/indicators/2026/en2603",
])
def test_observed_native_families(url):
    assert mod.admitted_dallas_research_route(url) == url


def test_linked_entry_and_pagination_are_literal_not_constructed():
    entry = b"<div id='content'><a href='/topics/energy.aspx'>Energy articles and analysis</a></div>"
    assert mod.linked_topic(entry, mod.ENTRY).endswith("energy.aspx")
    raw = b"<div id='content'><div class='dal-main-content'><ul><li class='dal-results-list__item'><p class='dal-headline'><a href='/research/economics/2024/0101'>Oil supply changes</a></p><p class='dal-content-date'>January 1, 2024</p></li></ul><nav class='dal-pagination__container'><a class='next' aria-disabled='false' href='/topics/energy?page=7'>Next</a></nav></div></div>"
    rows, next_url, rejected, inspected = mod.parse_archive(raw, "https://www.dallasfed.org/topics/energy")
    assert len(rows) == inspected == 1 and not rejected
    assert next_url.endswith("?page=7")
    assert rows[0]["body_date_verified"] is False


def test_archive_excludes_known_future_event_and_abstract_without_body_claims():
    items = [
        ("/research/economics/2024/0101", "Oil supply", "January 1, 2024"),
        ("/research/economics/2026/1008", "Oil supply", "October 8, 2026"),
        ("/events/2024/oil", "Oil event", "January 1, 2024"),
        ("/research/papers/2024/wp2401", "Oil paper", "January 1, 2024"),
    ]
    raw = ("<div id='content'><div class='dal-main-content'>" + "".join(
        f"<li class='dal-results-list__item'><p class='dal-headline'><a href='{u}'>{t}</a></p><p class='dal-content-date'>{d}</p></li>" for u,t,d in items) + "</div></div>").encode()
    rows, _, rejected, inspected = mod.parse_archive(raw, "https://www.dallasfed.org/topics/energy",
                                                    {"https://www.dallasfed.org/research/economics/2024/0101"})
    assert rows == [] and inspected == 4
    assert rejected == {"already_registered": 1, "missing_day_or_outside_window": 1,
                        "unadmitted_family_event_survey_abstract_or_nonofficial": 2}


def test_native_exact_header_and_authors_disclaimer_preserved_as_metadata():
    parsed = mod.parse_dallas_research_body(article())
    assert parsed["published_date"] == parsed["available_by"] == "2026-06-23"
    assert parsed["title"] == "Crude oil exports increased"
    assert parsed["authors"] == "Research Author" and parsed["author_disclaimers"]
    assert "BADPROFILE" not in parsed["text"] and "2030" not in parsed["text"]
    assert "not_official_fed_policy" in parsed["evidence_type"]


def test_observed_legacy_indicator_direct_date_is_header_metadata_only():
    raw = article(title="Energy Indicators").replace(
        b"<div class='dal-inline-list'><p class='dal-content-date'>June 23, 2026</p></div>",
        b"<div class='dal-inline-list'></div><p class='date'>June 23, 2026</p>")
    parsed = mod.parse_dallas_research_body(raw)
    assert parsed["published_date"] == "2026-06-23"
    assert "legacy Energy Indicators" in parsed["date_selector"]
    assert "June 23, 2026" not in parsed["scope_text"]


def test_legacy_date_not_used_for_unobserved_article_template_or_after_body():
    raw = article().replace(b"dal-content-date", b"unknown")
    raw = raw.replace(b"<p>Crude", b"<p class='date'>June 23, 2026</p><p>Crude")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        mod.parse_dallas_research_body(raw)
    raw = article(title="Energy Indicators").replace(b"dal-content-date", b"unknown")
    raw = raw.replace(b"<div class='dal-info__container'>", b"<p class='date'>June 23, 2026</p><div class='dal-info__container'>")
    with pytest.raises(ValueError, match="after body prose"):
        mod.parse_dallas_research_body(raw)


def test_observed_legacy_date_metadata_whitespace_is_not_a_missing_date():
    raw = article(title="Energy Indicators").replace(
        b"<div class='dal-inline-list'><p class='dal-content-date'>June 23, 2026</p></div>",
        b"<div class='dal-inline-list'></div><p class='date'>May  25, 2023</p>")
    assert mod.parse_dallas_research_body(raw)["published_date"] == "2023-05-25"


def test_title_and_subheading_do_not_make_non_oil_body_eligible():
    prose = "Solar panels and electric power markets expanded across several cities and institutions. " * 12
    parsed = mod.parse_dallas_research_body(article(text=prose, extra="<h2>Crude oil production</h2>"))
    assert parsed["body_oil_verified"] is False
    assert "oil" not in parsed["scope_text"].lower()


def test_actual_tables_figures_captions_footnotes_and_footer_are_not_flattened():
    extra = ("<table><tr><td><p>BADTABLE 99 million barrels</p></td></tr></table>"
             "<figure><p>BADFIGURE</p><figcaption>BADCAPTION</figcaption></figure>"
             "<ol class='dal-notes__list'><li>BADREFERENCE</li></ol><footer><p>BADFOOTER</p></footer>")
    parsed = mod.parse_dallas_research_body(article(extra=extra))
    assert "BAD" not in parsed["text"]


@pytest.mark.parametrize("notice", ["Correction: The source article was revised.",
                                    "Updated June 12 to correct the article."])
def test_undated_visible_revision_fails_closed(notice):
    with pytest.raises(ValueError, match="verified year/day"):
        mod.parse_dallas_research_body(article(extra=f"<p>{notice}</p>"))


def test_verified_iso_revision_raises_current_version_availability():
    parsed = mod.parse_dallas_research_body(article(extra="<p>Updated: 2026-07-01 to correct this article.</p>"))
    assert parsed["published_date"] == "2026-06-23" and parsed["available_by"] == "2026-07-01"
    assert parsed["modified_date"] == "2026-07-01"


def test_regular_estimate_revision_prose_is_not_article_revision_notice():
    parsed = mod.parse_dallas_research_body(article(extra="<p>Researchers revised oil production estimates in July 2026.</p>"))
    assert parsed["available_by"] == "2026-06-23"


@pytest.mark.parametrize("replace", [
    lambda raw: raw.replace(b"dal-main-content", b"unknown"),
    lambda raw: raw.replace(b"dal-content-date", b"unknown"),
    lambda raw: raw.replace(b"June 23, 2026", b"Unknown date"),
])
def test_missing_native_header_has_no_footer_or_url_fallback(replace):
    with pytest.raises(ValueError):
        mod.parse_dallas_research_body(replace(article()))


def test_six_fixed_date_spread_pilots_do_not_depend_on_length_or_answers():
    rows = [dict(url=f"u{i}", published_date=f"2024-01-{i+1:02d}") for i in range(15)]
    chosen = mod.select_pilots(rows)
    assert len(chosen) == 6 and chosen[0]["url"] == "u0" and chosen[-1]["url"] == "u14"
    assert chosen == mod.select_pilots(list(reversed(rows)))


@pytest.fixture
def transport_fetch(monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    clients = []

    def create(handler):
        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        fetch = mod.DallasResearchFetcher(client)
        clients.append(fetch)
        return fetch

    yield create
    for fetch in clients:
        fetch.close()


def test_robots_disallow_prevents_article_request(transport_fetch):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text="User-agent: *\nDisallow: /research/")
    fetch = transport_fetch(handler)
    with pytest.raises(ValueError, match="disallows"):
        fetch.get("https://www.dallasfed.org/research/economics/2024/0101")
    assert calls == ["https://www.dallasfed.org/robots.txt"]


def test_403_is_not_retried_or_bypassed(transport_fetch):
    calls=[]
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(403)
    with pytest.raises(httpx.HTTPStatusError):
        transport_fetch(handler).get(mod.ENTRY)
    assert calls == ["https://www.dallasfed.org/robots.txt"]


def test_unofficial_redirect_destination_never_requested(transport_fetch):
    calls=[]
    def handler(request):
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200,text="User-agent: *\nDisallow:")
        return httpx.Response(302,headers={"location":"https://evil.test/oil"})
    with pytest.raises(ValueError,match="official-host"):
        transport_fetch(handler).get(mod.ENTRY)
    assert calls == ["https://www.dallasfed.org/robots.txt",mod.ENTRY]


def test_redirect_timeout_three_target_attempts_not_multiplied(transport_fetch):
    calls=[]
    target="https://www.dallasfed.org/research/economics/2024/0101"
    def handler(request):
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200,text="User-agent: *\nDisallow:")
        if str(request.url)==mod.ENTRY:
            return httpx.Response(302,headers={"location":target})
        raise httpx.ReadTimeout("fixture transient failure",request=request)
    with pytest.raises(httpx.ReadTimeout):
        transport_fetch(handler).get(mod.ENTRY)
    assert calls.count(target)==3 and calls.count(mod.ENTRY)==1


def test_byte_accounting_includes_robots(transport_fetch):
    robots=b"User-agent: *\nDisallow:"
    def handler(request):
        return httpx.Response(200,content=robots if request.url.path=="/robots.txt" else b"article")
    fetch=transport_fetch(handler)
    assert fetch.get(mod.ENTRY)[0]==b"article"
    assert fetch.bytes==len(robots)+len(b"article")


def test_response_budget_limit_fails_closed(transport_fetch):
    def handler(request):
        return httpx.Response(200,content=b"User-agent: *\nDisallow:" if request.url.path=="/robots.txt" else b"x"*200)
    fetch=transport_fetch(handler)
    fetch.settings=dict(fetch.settings,download_budget_mb=0.0001)
    with pytest.raises(ValueError,match="budget"):
        fetch.get(mod.ENTRY)


def test_output_directory_immutable_and_candidate_cap(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,"DATA",tmp_path)
    target=tmp_path/"unified_runs/test/reports/discovery"
    target.mkdir(parents=True)
    with pytest.raises(ValueError,match="immutable"):
        mod.discover(target)
    with pytest.raises(ValueError,match="6..100"):
        mod.discover(target,101)
