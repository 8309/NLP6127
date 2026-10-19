import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/discover_official_buffer.py"
spec = importlib.util.spec_from_file_location("discover_official_buffer", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

URL = "https://www.cer-rec.gc.ca/en/data-analysis/energy-markets/market-snapshots/2024/market-snapshot-oil-production.html"
TITLE = "Crude oil production increased in 2023"
BODY = " ".join([
    "Canadian crude oil production increased in 2023 as operators expanded capacity and petroleum exports rose.",
    "The original source reports crude oil volumes separately from natural gas liquids and other petroleum products.",
    "Oil was shipped through pipelines and rail terminals to domestic refineries and international markets in 2023.",
    "The report measures production in barrels per day, preserving both the statistical period and its defined unit.",
    "The pipeline capacity estimate describes a scenario rather than a guarantee of future realized transport volumes.",
    "Changes in refinery runs, maintenance schedules and international demand affected the reported crude oil market.",
])


def article(extra="", released="2024-03-06", modified="2024-03-08", title=TITLE):
    return (f"<main property='mainContentOfPage'><h1 id='wb-cont' property='name'>Market Snapshot: {title}</h1>"
            f"<aside><p>BADCONTACT</p></aside><p>Release date: {released}</p><p>{BODY}</p>{extra}"
            f"<dl id='wb-dtmd'><dt>Date modified:</dt><dd><time property='dateModified' datetime='{modified}'>{modified}</time></dd></dl></main>").encode()


def archive_row(day="2024-03-06", title=TITLE, commodity="oil", url=URL):
    return f"<tr><td>{day}</td><td><a href='{url}'>{title}</a></td><td>{commodity}</td><td>crude oil</td></tr>"


def archive(rows):
    return f"<main><table id='snapshots'><tbody>{rows}</tbody></table></main>".encode()


@pytest.mark.parametrize("url", [URL.replace("https:", "http:"), URL.replace(mod.HOST, mod.HOST + ".evil.test"),
                                URL.replace(mod.HOST, "user@" + mod.HOST), URL.replace(mod.HOST, mod.HOST + ":444"),
                                URL + "?alternate=1", URL + "#fragment", URL.replace("/2024/", "/fr/"), mod.ARCHIVE])
def test_exact_route_rejects_untrusted_and_ambiguous(url):
    with pytest.raises(ValueError):
        mod.admitted_cer_route(url)


def test_source_routes():
    assert mod.admitted_cer_route(URL) == URL
    assert mod.admitted_cer_route(mod.ARCHIVE, archive=True) == mod.ARCHIVE


def test_literal_archive_links_with_exact_date_and_oil_tag():
    raw = archive(archive_row() + archive_row("2017-12-31", url=URL.replace("oil-production", "old"))
                  + archive_row(commodity="electricity", url=URL.replace("oil-production", "electric"))
                  + archive_row("unknown", url=URL.replace("oil-production", "unknown"))
                  + archive_row(url=URL.replace(mod.HOST, "evil.test")))
    rows, excluded, count = mod.parse_archive(raw)
    assert count == 5 and len(rows) == 1
    assert rows[0]["published_date"] == "2024-03-06" and rows[0]["body_date_verified"] is False
    assert excluded == {"outside_window": 1, "not_official_oil_commodity_tag": 1,
                        "missing_official_release_day": 1, "non_official_or_ambiguous_article_route": 1}


def test_known_registered_url_not_rediscovered():
    rows, excluded, _ = mod.parse_archive(archive(archive_row()), known={URL})
    assert rows == [] and excluded == {"already_registered": 1}


def test_conflicting_duplicate_archive_metadata_fails_closed():
    with pytest.raises(ValueError, match="conflicting"):
        mod.parse_archive(archive(archive_row() + archive_row("2024-03-07")))


def test_title_only_allows_fixed_series_prefix_normalization():
    assert mod.normalized_cer_title(" Market Snapshot:  " + TITLE.upper()) == mod.normalized_cer_title(TITLE)
    assert mod.normalized_cer_title(TITLE) != mod.normalized_cer_title(TITLE.replace("2023", "2024"))


def test_release_and_modified_dates_are_not_conflated():
    parsed = mod.parse_cer_body(article())
    assert parsed["published_date"] == "2024-03-06"
    assert parsed["available_by"] == parsed["visible_revision_date"] == "2024-03-08"
    assert parsed["body_oil_verified"] is True
    assert "BADCONTACT" not in parsed["text"] and "Release date:" not in parsed["text"]


@pytest.mark.parametrize("change", [lambda raw: raw.replace(b"property='mainContentOfPage'", b"class='unknown'"),
                                    lambda raw: raw.replace(b"id='wb-cont'", b"id='unknown'"),
                                    lambda raw: raw.replace(b"Release date:", b"Statistical period:"),
                                    lambda raw: raw.replace(b"property='dateModified'", b"property='unrelated'"),
                                    lambda raw: raw.replace(b"2024-03-08", b"unknown-date")])
def test_missing_source_native_metadata_fails_closed(change):
    with pytest.raises(ValueError):
        mod.parse_cer_body(change(article()))


def test_modified_before_release_fails_closed():
    with pytest.raises(ValueError, match="precedes"):
        mod.parse_cer_body(article(modified="2024-03-01"))


def test_neighbor_publication_day_not_used():
    parsed = mod.parse_cer_body(article("<aside><p>Release date: 2030-01-01</p></aside>"))
    # Aside publication-like metadata may not be mistaken for this article.
    assert parsed["published_date"] == "2024-03-06"


def test_figure_tables_text_alternatives_and_footnotes_not_flattened():
    extra = ("<table><tr><td>BADTABLE</td></tr></table><figure><p>BADFIGURE</p></figure>"
             "<details><summary>Source and Text Alternative</summary><p>BADCHART</p></details>"
             "<section class='wb-fnote'><p>BADFOOTNOTE</p></section>"
             "<p>The offshore oil area covers 1.8 million km<sup>2</sup> in Canada."
             "<sup><a href='#fn1'>Footnote 1</a></sup></p>")
    parsed = mod.parse_cer_body(article(extra))
    assert "BAD" not in parsed["text"] and "Footnote 1" not in parsed["text"]
    assert "km 2" in parsed["text"]


def test_list_paragraphs_are_not_duplicated():
    parsed = mod.parse_cer_body(article("<ul><li><p>Crude oil exports increased in 2023.</p></li></ul>"))
    assert parsed["text"].count("Crude oil exports increased in 2023.") == 1


def test_six_date_spread_pilots_deterministic():
    rows = [dict(url=f"u{i}", published_date=f"2024-01-{i + 1:02d}") for i in range(20)]
    picks = mod.select_pilots(rows, 6)
    assert picks == mod.select_pilots(list(reversed(rows)), 6)
    assert len(picks) == 6 and picks[0]["url"] == "u0" and picks[-1]["url"] == "u19"


def test_six_pilot_discovery_isolated_immutable_and_not_a_passage_count(tmp_path, monkeypatch):
    from oil_evidence.store import read_jsonl, write_jsonl
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(mod, "DATA", data)
    write_jsonl(data / "sources.jsonl", [])
    write_jsonl(data / "documents.jsonl", [])
    write_jsonl(data / "passages.jsonl", [{"passage_id": "untouched"}])
    records = "".join(archive_row(day=f"2024-03-{i + 1:02d}", url=URL.replace("production", f"production-{i}")) for i in range(8))
    class FakeFetcher:
        def __init__(self):
            self.bytes = 0
            self.settings = {"request_delay_seconds": 1, "max_document_mb": 25, "download_budget_mb": 300}
        def get(self, url):
            if url == mod.ARCHIVE:
                raw = archive(records)
            else:
                index = int(url.rsplit("-", 1)[1].split(".")[0])
                day = f"2024-03-{index + 1:02d}"
                raw = article(released=day, modified=day)
            self.bytes += len(raw)
            return raw, url, "text/html"
        def close(self):
            pass
    monkeypatch.setattr(mod, "Fetcher", FakeFetcher)
    out = data / "unified_runs/fixture-v1/reports/official-discovery"
    before = (data / "passages.jsonl").read_bytes()
    report = mod.discover(out, 6)
    assert report["new_unregistered_title_candidates"] == 8
    assert report["accessible_dated_oil_pilots"] == 6
    assert report["accepted_new_passages"] is None and report["raw_documents_written"] == 0
    assert report["external_model_calls"] == 0 and report["live_inputs_unchanged"] is True
    assert len(read_jsonl(out / "source_candidates.jsonl")) == 8
    assert (data / "passages.jsonl").read_bytes() == before
    with pytest.raises(ValueError, match="immutable"):
        mod.discover(out, 6)


def test_discovery_honors_robots_denial_without_candidate_recovery(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(mod, "DATA", data)
    class DeniedFetcher:
        def __init__(self):
            self.bytes = 0
            self.settings = {"request_delay_seconds": 1, "max_document_mb": 25, "download_budget_mb": 300}
        def get(self, url):
            raise RuntimeError("Cannot verify robots.txt: 403 https://www.cer-rec.gc.ca")
        def close(self):
            pass
    monkeypatch.setattr(mod, "Fetcher", DeniedFetcher)
    report = mod.discover(data / "unified_runs/fixture-v1/reports/denied", 6)
    assert report["new_unregistered_title_candidates"] == report["accessible_dated_oil_pilots"] == 0
    assert report["status"] == "blocked_or_archive_failure_no_bypass"
    assert len(report["errors"]) == 1 and "403" in report["errors"][0]["error"]


def test_known_urls_include_old_and_new_supplement_registries(tmp_path, monkeypatch):
    from oil_evidence.store import write_jsonl
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(mod, "DATA", data)
    for run in ("old-v1", "new-v2"):
        write_jsonl(data / f"unified_runs/{run}/supplement/sources.jsonl", [{"url": URL.replace("production", run)}])
    assert len(mod.known_urls()) == 2
