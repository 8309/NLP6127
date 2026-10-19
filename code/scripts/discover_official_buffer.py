"""Immutable CER oil-snapshot discovery and revision-aware HTML parsing.

This script never adds a document or passage to a corpus.  Follow only literal
links in CER's full snapshot table, then verify bounded pilot article bodies.
Publication and the current page's visible modified date are deliberately
different fields.  A later modification is a conservative availability bound,
not proof of what the page contained on its original release date.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from oil_evidence.corpus import Fetcher
from oil_evidence.segmentation import oil_scope_supported
from oil_evidence.store import DATA, digest, now, read_jsonl, words, write_json, write_jsonl


ARCHIVE = "https://www.cer-rec.gc.ca/en/data-analysis/energy-markets/market-snapshots/index.html"
START, END = "2018-01-01", "2026-10-05"
HOST = "www.cer-rec.gc.ca"
ARTICLE = re.compile(r"/en/data-analysis/energy-markets/market-snapshots/20\d{2}/[a-z0-9-]+\.html")
ISO_DAY = re.compile(r"20\d{2}-\d{2}-\d{2}")


def canonical(url):
    p = urlparse(url)
    return p._replace(fragment="").geturl().rstrip("/")


def admitted_cer_route(url, *, archive=False):
    p = urlparse(url)
    if (p.scheme != "https" or p.hostname != HOST or p.username or p.password
            or p.port not in {None, 443} or p.query or p.fragment):
        raise ValueError("CER source must have an exact, uncredentialed official HTTPS route")
    expected = urlparse(ARCHIVE).path if archive else ARTICLE
    if (p.path != expected if archive else not ARTICLE.fullmatch(p.path)):
        raise ValueError("CER URL is not the observed oil-snapshot archive/article route")
    return url


def exact_day(text):
    if not isinstance(text, str) or not ISO_DAY.fullmatch(text.strip()):
        return None
    try:
        return date.fromisoformat(text.strip()).isoformat()
    except ValueError:
        return None


def normalized_cer_title(title):
    if not isinstance(title, str) or not title.strip():
        raise ValueError("CER source title missing")
    # The index omits this fixed series prefix on many otherwise identical
    # titles.  Do not normalize arbitrary words, quantities or dates away.
    title = re.sub(r"^\s*Market\s+Snapshot\s*:\s*", "", title, flags=re.I)
    return unicodedata.normalize("NFKC", " ".join(title.split())).casefold()


def parse_archive(raw, origin=ARCHIVE, known=()):
    admitted_cer_route(origin, archive=True)
    soup = BeautifulSoup(raw, "html.parser")
    table = soup.select_one("main table#snapshots tbody")
    if table is None:
        raise ValueError("CER's source-native snapshots table missing")
    rows, rejected, count = {}, Counter(), 0
    for tr in table.find_all("tr", recursive=False):
        count += 1
        cells = tr.find_all("td", recursive=False)
        if len(cells) != 4:
            rejected["unexpected_table_schema"] += 1
            continue
        released = exact_day(cells[0].get_text(" ", strip=True))
        if not released:
            rejected["missing_official_release_day"] += 1
            continue
        if not START <= released <= END:
            rejected["outside_window"] += 1
            continue
        commodities = {value.strip().casefold() for value in cells[2].get_text(" ", strip=True).split(",")}
        if "oil" not in commodities:
            rejected["not_official_oil_commodity_tag"] += 1
            continue
        anchor = cells[1].find("a", href=True)
        if anchor is None:
            rejected["missing_article_link"] += 1
            continue
        url = urljoin(origin, anchor["href"])
        try:
            admitted_cer_route(url)
        except ValueError:
            rejected["non_official_or_ambiguous_article_route"] += 1
            continue
        title = anchor.get_text(" ", strip=True)
        if not title:
            rejected["missing_article_title"] += 1
            continue
        if canonical(url) in known:
            rejected["already_registered"] += 1
            continue
        if url in rows:
            if rows[url]["published_date"] != released or normalized_cer_title(rows[url]["title"]) != normalized_cer_title(title):
                raise ValueError("CER archive has conflicting records for the same literal URL")
            rejected["duplicate_archive_url"] += 1
            continue
        rows[url] = dict(url=url, source="CER-OIL-SNAPSHOT", title=title,
            published_date=released, available_by=released, date_precision="day", date_source=origin,
            event_date=None, event_status="unlabelled", redistribution="review_required", requires_pdf=False,
            content_scope="official_canadian_mineral_oil_market_narratives", archive_url=origin,
            discovery_method="actual_official_snapshot_table_article_link_and_oil_commodity_tag",
            archive_publication_date=released, official_commodity_tags=sorted(commodities),
            date_evidence="CER table release day; article Release date must agree; visible Date modified is a conservative current-version availability bound",
            pilot_status="not_fetched", body_date_verified=False, body_title_verified=False, body_oil_verified=False)
    return list(rows.values()), dict(rejected), count


def parse_cer_body(raw):
    """Return current-version prose, publication and conservative availability.

    Tables, figures, expandable chart descriptions, contact panels and footnote
    backlinks are not flattened into prose.  Superscripts not containing a
    footnote backlink (units such as km2) remain untouched.  Underlying raw HTML
    must be retained by the caller; linked third-party material is not fetched.
    """
    soup = BeautifulSoup(raw, "html.parser")
    main = soup.select_one("main[property='mainContentOfPage']")
    heading = main.select_one("h1#wb-cont[property='name']") if main else None
    if main is None or heading is None:
        raise ValueError("CER source-native main and article title missing")
    title = heading.get_text(" ", strip=True)
    if not re.match(r"^Market\s+Snapshot\s*:", title, re.I):
        raise ValueError("CER source is not a Market Snapshot article")
    release_nodes = [node for node in main.select("p")
                     if not node.find_parent(["aside", "nav", "footer", "table", "figure", "details"])
                     and re.fullmatch(r"Release\s+date\s*:\s*20\d{2}-\d{2}-\d{2}", node.get_text(" ", strip=True), re.I)]
    releases = {exact_day(re.search(ISO_DAY, node.get_text(" ", strip=True)).group()) for node in release_nodes}
    if len(releases) != 1 or None in releases:
        raise ValueError("CER article-local explicit Release date missing or ambiguous")
    released = next(iter(releases))
    modified_nodes = main.select("#wb-dtmd time[property='dateModified']")
    modified = {exact_day(node.get("datetime", "") or node.get_text(" ", strip=True)) for node in modified_nodes}
    if len(modified) != 1 or None in modified:
        raise ValueError("CER explicit current-page Date modified missing or ambiguous")
    revised = next(iter(modified))
    if revised < released:
        raise ValueError("CER modified date precedes the article's release date")
    available = max(released, revised)
    # Preserve source-native units and exact paragraph boundaries.  Chart text
    # alternatives are excluded, not repackaged as independent natural prose.
    for node in main.select("aside,nav,footer,table,figure,details,script,style,#wb-dtmd,.wb-fnote,.panel,.module-related"):
        node.decompose()
    for node in main.select("sup a[href^='#fn'],sup a[href^='#fnb'],sup a[href^='#footnote']"):
        if node.parent and node.parent.name == "sup":
            node.parent.decompose()
    blocks = [title]
    for node in main.select("p,li,h2,h3,h4"):
        if node.name == "li" and node.find("p"):
            continue
        if node.find_parent(["nav", "footer", "aside", "table", "figure", "details"]):
            continue
        text = node.get_text(" ", strip=True)
        if not text or re.match(r"^(?:Release\s+date\s*:|Source\s*:|Text\s+Alternative\s*:|Figure\s+\d+\s*:|Footnotes?\s*$)", text, re.I):
            continue
        if not blocks or text != blocks[-1]:
            blocks.append(text)
    body = "\n\n".join(blocks)
    if len(words(body)) < 80:
        raise ValueError("CER article has insufficient source-native prose")
    return dict(published_date=released, available_by=available, modified_date=revised, visible_revision_date=revised,
                title=title, text=body, blocks=blocks, body_oil_verified=oil_scope_supported(body),
                date_selector="main p explicit Release date; main #wb-dtmd time[property=dateModified]",
                body_selector="main[property=mainContentOfPage] p/li/h2/h3/h4 excluding charts, tables, contact and footnote regions",
                body_parser="official_cer_snapshot_revision_aware_v1",
                version_caution="Current HTML version only; available_by=max(release, visible modification). Original release-version content not independently verified.")


def select_pilots(rows, limit=6):
    ordered = sorted(rows, key=lambda row: (row["published_date"], row["url"]))
    if limit <= 0 or not ordered:
        return []
    if len(ordered) <= limit:
        return ordered
    return [ordered[round(i * (len(ordered) - 1) / (limit - 1))] for i in range(limit)] if limit > 1 else [ordered[0]]


def protected_hashes():
    names = ("sources.jsonl", "documents.jsonl", "passages.jsonl", "index/manifest.json", "index/vectors.npy",
             "annotations/codex.jsonl", "annotations/passage_labels.jsonl", "annotations/passage_review.jsonl")
    return {name: digest((DATA / name).read_bytes()) for name in names if (DATA / name).exists()}


def known_urls():
    paths = [DATA / "sources.jsonl", DATA / "documents.jsonl"]
    # All prior supplement registries, not labels or staged passage rows.
    paths += sorted((DATA / "unified_runs").glob("*/supplement/sources.jsonl"))
    paths += sorted((DATA / "unified_runs").glob("*/supplement/documents.jsonl"))
    return {canonical(row["url"]) for path in paths for row in read_jsonl(path)}


def discover(output_dir, pilot_limit=6):
    if not 0 <= pilot_limit <= 8:
        raise ValueError("CER discovery permits at most eight bounded article pilots")
    output_dir = Path(output_dir).resolve()
    if (output_dir.parent.name != "reports" or output_dir.parent.parent.parent != (DATA / "unified_runs").resolve()):
        raise ValueError("Discovery outputs must be a new unified-run reports subdirectory")
    if output_dir.exists():
        raise ValueError("Discovery artifacts immutable; choose a new reports directory")
    before, known, rows, pilots, errors, archives = protected_hashes(), known_urls(), [], [], [], []
    fetch = Fetcher()
    if (fetch.settings["request_delay_seconds"] < 1 or fetch.settings["max_document_mb"] > 25
            or fetch.settings["download_budget_mb"] > 300):
        fetch.close()
        raise ValueError("Fetcher must retain project robots/rate/size/budget constraints")
    inspected, rejected, status = 0, {}, "not_attempted"
    try:
        try:
            raw, final, mime = fetch.get(ARCHIVE)
            admitted_cer_route(final, archive=True)
            rows, rejected, inspected = parse_archive(raw, final, known)
            archives.append(dict(url=final, sha256=digest(raw), bytes=len(raw), mime=mime,
                                 items=inspected, candidate_count=len(rows)))
            status = "archive_scanned"
            print(f"CER archive: {len(rows)} unregistered oil-tagged literal article URLs", flush=True)
        except Exception as exc:
            status = "blocked_or_archive_failure_no_bypass"
            errors.append(dict(stage="archive", url=ARCHIVE, error=str(exc)))
        for row in select_pilots(rows, pilot_limit):
            pilot = dict(url=row["url"], archive_publication_date=row["published_date"], archive_title=row["title"])
            try:
                raw, final, mime = fetch.get(row["url"])
                admitted_cer_route(final)
                parsed = parse_cer_body(raw)
                same_date = parsed["published_date"] == row["published_date"]
                same_title = normalized_cer_title(parsed["title"]) == normalized_cer_title(row["title"])
                inside = START <= parsed["available_by"] <= END
                good = same_date and same_title and inside and parsed["body_oil_verified"]
                state = "accessible_dated_oil_prose" if good else "metadata_or_scope_review_required"
                pilot.update(status=state, final_url=final, raw_sha256=digest(raw), bytes=len(raw), mime=mime,
                             article_publication_date=parsed["published_date"], visible_revision_date=parsed["visible_revision_date"],
                             current_version_available_by=parsed["available_by"], date_matches_archive=same_date,
                             title_matches_archive=same_title, body_oil_verified=parsed["body_oil_verified"],
                             body_words=len(words(parsed["text"])), blocks=len(parsed["blocks"]),
                             body_sha256=digest(parsed["text"]), body_preview=parsed["text"][:700],
                             date_selector=parsed["date_selector"], body_selector=parsed["body_selector"])
                row.update(pilot_status=state, body_date_verified=same_date, body_title_verified=same_title,
                           body_oil_verified=parsed["body_oil_verified"], article_title=parsed["title"],
                           visible_revision_date=parsed["visible_revision_date"], available_by=parsed["available_by"],
                           pilot_raw_sha256=digest(raw), pilot_body_sha256=digest(parsed["text"]),
                           pilot_body_words=len(words(parsed["text"])), historical_article_version="not_independently_verified")
            except Exception as exc:
                pilot.update(status="failed_no_bypass", error=str(exc))
                row.update(pilot_status="failed_no_bypass", pilot_error=str(exc))
                # An access denial is a terminal host condition, not an excuse
                # to try alternative domains, query identities or browsers.
                if "403" in str(exc) or "robots.txt disallows" in str(exc):
                    pilots.append(pilot)
                    errors.append(dict(stage="pilot_access_denial", url=row["url"], error=str(exc)))
                    break
            pilots.append(pilot)
            print(f"CER pilot: {pilot['status']} {row['published_date']} {row['url']}", flush=True)
    finally:
        downloaded = fetch.bytes
        fetch.close()
    if protected_hashes() != before:
        raise ValueError("Protected live corpus/index/annotations changed during discovery")
    ordered = sorted(rows, key=lambda row: (row["published_date"], row["url"]))
    good = [row for row in pilots if row["status"] == "accessible_dated_oil_prose"]
    report = dict(at=now(), discovery_only=True, status=status, window=[START, END],
                  source_family="CER-OIL-SNAPSHOT", archive=archives, archive_items_inspected=inspected,
                  new_unregistered_title_candidates=len(ordered), known_url_count=len(known), excluded=rejected,
                  candidate_by_publication_year=dict(sorted(Counter(row["published_date"][:4] for row in ordered).items())),
                  pilots=pilots, accessible_dated_oil_pilots=len(good), errors=errors,
                  pilot_body_words=[row["body_words"] for row in good],
                  rough_word_budget_paragraph_equivalents=round(sum(row["body_words"] for row in good) / len(good) / 160 * len(ordered)) if good else None,
                  rough_estimate_is_not_accepted_passage_count=True, downloaded_response_bytes=downloaded,
                  raw_documents_written=0, live_registry_writes=0, external_model_calls=0,
                  accepted_new_passages=None, protected_live_hashes_before_and_after=before,
                  live_inputs_unchanged=True, script_sha256=digest(Path(__file__).read_bytes()),
                  rights="Canadian official public HTML. Redistribution review_required; keep attribution. Do not treat third-party figures, maps, images or linked sources as blanket government-licensed content.",
                  limitations=["Oil tags and six bounded pilots are discovery evidence, not corpus qualification.",
                               "Release and modified dates differ. Current pages are conservatively unavailable before visible modification; original historical versions are not reconstructed.",
                               "Main narrative only; figures, text alternatives, tables and footnote regions require separate audited extraction. Their exclusion can remove useful qualifiers and must be reviewed in semantic QA.",
                               "The 160-word equivalence estimate is a rough capacity hint before semantic checks, exact deduplication and source screening, not an accepted or promised yield."])
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "source_candidates.jsonl", ordered)
    report["candidate_file_sha256"] = digest((output_dir / "source_candidates.jsonl").read_bytes())
    write_json(output_dir / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA / "unified_runs/semantic-page-v2-buffer/reports/official-discovery")
    parser.add_argument("--pilot-limit", type=int, default=6)
    args = parser.parse_args()
    result = discover(args.output_dir, args.pilot_limit)
    print(json.dumps({key: result[key] for key in ("status", "new_unregistered_title_candidates", "accessible_dated_oil_pilots",
                                                  "pilot_body_words", "rough_word_budget_paragraph_equivalents", "errors", "candidate_file_sha256")}, indent=2))
