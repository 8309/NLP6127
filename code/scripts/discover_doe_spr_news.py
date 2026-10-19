"""Bounded, independent DOE SPR operations-news discovery; no live mutation.

Follow the actual Petroleum Reserves collection link on DOE's SPR topic page,
then actual next-page links.  Archive dates and article-local .display-date are
used; unrelated previous/next-news time tags are never publication evidence.
Only archive indexes and 1-3 pilot bodies are fetched, not bulk raw documents.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
import json

from bs4 import BeautifulSoup

from oil_evidence.corpus import Fetcher, date_from_text
from oil_evidence.store import DATA, digest, now, read_jsonl, words, write_json, write_jsonl


TOPIC = "https://www.energy.gov/topics/strategic-petroleum-reserve-spr"
START, END = "2018-01-01", "2026-10-05"
RESERVE = re.compile(r"\b(?:strategic petroleum reserve|SPR)\b", re.I)
OPERATIONS = re.compile(r"\b(?:release\w*|replenish\w*|refill\w*|sale\w*|purchase\w*|acquisition\w*|exchange\w*|solicitation\w*|RFP|emergency|supply disruption\w*|deliver\w*|contracts?)\b", re.I)
NONSUPPLY = re.compile(r"\b(?:management and operating|management.*operating|management.*contract|visit\w*|tour\w*|commemorat\w*|anniversary|ceremony|congratulat\w*)\b", re.I)
MINERAL_OIL = re.compile(r"\b(?:crude|petroleum|barrels?)\b", re.I)
MAX_ARCHIVE_PAGES = 20


def official(url):
    p = urlparse(url)
    return p.scheme == "https" and p.hostname == "www.energy.gov" and not p.username and not p.password


def canonical(url):
    return url.rstrip("/")


def exact_day(text):
    value = date_from_text(text)
    if value and date.fromisoformat(value).isoformat() == value:
        return value
    return None


def linked_collection(raw, origin):
    soup = BeautifulSoup(raw, "html.parser")
    for a in soup.select("main a[href]"):
        if a.get_text(" ", strip=True) != "Petroleum Reserves":
            continue
        href = urljoin(origin, a["href"])
        if official(href) and urlparse(href).path.startswith("/collection/filter/"):
            return href
    raise ValueError("SPR topic page lacks the actual linked Petroleum Reserves collection")


def parse_archive(raw, origin, known=()):
    soup = BeautifulSoup(raw, "html.parser")
    main = soup.select_one("main")
    if main is None:
        raise ValueError("DOE archive lacks its main region")
    wrappers = main.select(".collection-item-wrapper")
    if not wrappers:
        raise ValueError("DOE collection item selector has no records")
    rows, observed_dates, excluded = [], [], Counter()
    for wrapper in wrappers:
        a = wrapper.select_one(".collection-item__link[href]")
        date_node = wrapper.select_one(".collection-item__date")
        if a is None or date_node is None:
            excluded["missing_link_or_date"] += 1
            continue
        title, exact = a.get_text(" ", strip=True), exact_day(date_node.get_text(" ", strip=True))
        if not exact:
            excluded["unknown_date"] += 1
            continue
        observed_dates.append(exact)
        if not START <= exact <= END:
            excluded["outside_window"] += 1
            continue
        href = urljoin(origin, a["href"])
        if not official(href) or "/articles/" not in urlparse(href).path:
            excluded["non_official_or_non_article"] += 1
            continue
        if not RESERVE.search(title) or not OPERATIONS.search(title) or NONSUPPLY.search(title):
            excluded["not_spr_supply_operation"] += 1
            continue
        if canonical(href) in known:
            excluded["already_registered"] += 1
            continue
        summary = wrapper.select_one(".collection-item__summary")
        rows.append(dict(url=href, source="DOE-SPR-NEWS", title=title, published_date=exact,
            available_by=exact, date_precision="day", date_source=origin, event_date=None,
            event_status="unlabelled", redistribution="review_required", requires_pdf=False,
            content_scope="spr_crude_release_refill_emergency_exchange", discovery_method="actual_official_collection_item_link",
            date_evidence="DOE collection-item publication date; must agree with article-local display-date before acceptance",
            archive_summary=summary.get_text(" ", strip=True) if summary else None,
            pilot_status="not_fetched", body_date_verified=False, body_oil_verified=False))
    next_a = main.select_one("nav.usa-pagination a[aria-label='Next page'][href]")
    next_url = urljoin(origin, next_a["href"]) if next_a else None
    if next_url and (not official(next_url) or not urlparse(next_url).path.startswith("/collection/filter/")):
        raise ValueError("DOE next-page link leaves its official collection")
    return rows, next_url, observed_dates, dict(excluded), len(wrappers)


def parse_doe_body(raw):
    """Contract: {published_date,title,text,blocks,date_selector,body_selector}."""
    soup = BeautifulSoup(raw, "html.parser")
    article = soup.select_one("main article[typeof='schema:Article'][about]")
    if article is None:
        raise ValueError("DOE source-native schema:Article selector missing")
    # Do NOT use main time: those belong to different previous/next news items.
    date_node = article.select_one(".beneath-title .display-date")
    exact = exact_day(date_node.get_text(" ", strip=True)) if date_node else None
    if not exact:
        raise ValueError("DOE article-local publication date missing")
    heading = soup.select_one("main h1")
    title = heading.get_text(" ", strip=True) if heading else None
    schema_name = article.select_one("[property='schema:name'][content]")
    if not title or schema_name is None or schema_name["content"] != title:
        raise ValueError("DOE article title does not match its source-native schema:name")
    blocks = []
    for container in article.select(".field--name-body,.field--name-field-text"):
        for node in container.select("p,li,h2,h3,h4"):
            if node.name == "li" and node.find("p"):
                continue
            text = node.get_text(" ", strip=True)
            if not text or text == "###" or node.find_parent(["nav", "footer"]):
                continue
            # Adjacent duplicate subtitle only; preserve repeated substantive
            # facts otherwise, pending deterministic corpus-level deduplication.
            if not blocks or text != blocks[-1]:
                blocks.append(text)
    text = "\n\n".join(blocks)
    if len(words(text)) < 80:
        raise ValueError("DOE article has no substantial factual prose")
    if not RESERVE.search(text) or not MINERAL_OIL.search(text):
        raise ValueError("DOE body lacks explicit SPR/mineral-oil evidence")
    return dict(published_date=exact, title=title, text=text, blocks=blocks,
                date_selector="main article[typeof='schema:Article'][about] .beneath-title .display-date",
                body_selector="main article[typeof='schema:Article'][about] (.field--name-body OR .field--name-field-text) p/li/h2/h3/h4")


def select_pilots(rows, limit):
    ordered = sorted(rows, key=lambda row: (row["available_by"], row["url"]))
    if not ordered or limit == 0:
        return []
    indices = [0, len(ordered) // 2, len(ordered) - 1]
    return list({ordered[i]["url"]: ordered[i] for i in indices}.values())[:limit]


def live_hashes():
    return {name: digest((DATA / name).read_bytes()) for name in
            ("sources.jsonl", "documents.jsonl", "passages.jsonl", "index/manifest.json", "index/vectors.npy",
             "annotations/codex.jsonl", "annotations/passage_labels.jsonl", "annotations/passage_review.jsonl")
            if (DATA / name).exists()}


def discover(output_dir, pilot_limit=3):
    if not 0 <= pilot_limit <= 3:
        raise ValueError("DOE discovery has at most three pilot article fetches")
    output_dir = Path(output_dir).resolve()
    targets = [output_dir / "source_candidates.jsonl", output_dir / "report.json"]
    if any(path.exists() for path in targets):
        raise ValueError("DOE discovery artifacts already exist; choose a new directory")
    if output_dir == DATA.resolve() or any(output_dir.is_relative_to(DATA / name) for name in ("index", "annotations")):
        raise ValueError("DOE discovery cannot replace a live registry/index/annotation directory")
    before = live_hashes()
    known = {canonical(row["url"]) for row in read_jsonl(DATA / "sources.jsonl") + read_jsonl(DATA / "documents.jsonl")}
    fetch = Fetcher()
    if fetch.settings["request_delay_seconds"] < 1 or fetch.settings["max_document_mb"] > 25 or fetch.settings["download_budget_mb"] > 300:
        fetch.close()
        raise ValueError("Fetcher must preserve 1 second/host, 25 MB/document, 300 MB/invocation limits")
    archives, rows, errors, pilots = [], {}, [], []
    status, visited, excluded, inspected = "not_attempted", set(), Counter(), 0
    try:
        try:
            raw, final, mime = fetch.get(TOPIC)
            archives.append({"url": final, "sha256": digest(raw), "bytes": len(raw), "mime": mime, "role": "official_spr_topic_seed"})
            url = linked_collection(raw, final)
            while url and len(visited) < MAX_ARCHIVE_PAGES:
                if url in visited:
                    raise ValueError("Repeated DOE archive pagination URL")
                visited.add(url)
                raw, final, mime = fetch.get(url)
                found, next_url, dates, rejected, count = parse_archive(raw, final, known)
                rows.update((row["url"], row) for row in found)
                inspected += count
                excluded.update(rejected)
                archives.append({"url": final, "sha256": digest(raw), "bytes": len(raw), "mime": mime,
                                 "role": "actual_linked_collection_page", "items": count, "candidates": len(found),
                                 "first_date": max(dates) if dates else None, "last_date": min(dates) if dates else None})
                print(f"Official DOE collection: {len(found)} SPR operation candidates from {final}", flush=True)
                if dates and max(dates) < START:
                    break
                url = next_url
            status = "archive_scanned" if not url else "bounded_archive_scan"
            if len(visited) >= MAX_ARCHIVE_PAGES and url:
                errors.append({"stage": "archive_bound", "url": url, "error": "Configured archive page cap reached; remaining links not fetched"})
        except Exception as exc:
            status = "blocked_or_archive_failure_no_bypass"
            errors.append({"stage": "official_archive", "error": str(exc)})
        for row in select_pilots(list(rows.values()), pilot_limit):
            pilot = {"url": row["url"], "title": row["title"], "archive_date": row["available_by"]}
            try:
                raw, final, mime = fetch.get(row["url"])
                parsed = parse_doe_body(raw)
                same_date = parsed["published_date"] == row["available_by"]
                same_title = parsed["title"] == row["title"]
                pilot_status = "accessible_dated_spr_oil_prose" if same_date and same_title else "date_or_title_review_required"
                pilot.update(status=pilot_status, final_url=final, article_date=parsed["published_date"],
                             article_title=parsed["title"], date_matches_archive=same_date, title_matches_archive=same_title,
                             bytes=len(raw), mime=mime, raw_sha256=digest(raw), body_sha256=digest(parsed["text"]),
                             body_words=len(words(parsed["text"])), factual_blocks=len(parsed["blocks"]),
                             date_selector=parsed["date_selector"], body_selector=parsed["body_selector"], body_preview=parsed["text"][:600])
                row.update(pilot_status=pilot_status, body_date_verified=same_date, body_oil_verified=True,
                           pilot_body_words=len(words(parsed["text"])), pilot_raw_sha256=digest(raw),
                           pilot_body_sha256=digest(parsed["text"]), article_date=parsed["published_date"])
            except Exception as exc:
                pilot.update(status="failed_no_bypass", error=str(exc))
                row.update(pilot_status="failed_no_bypass", pilot_error=str(exc))
            pilots.append(pilot)
    finally:
        downloaded = fetch.bytes
        fetch.close()
    if live_hashes() != before:
        raise ValueError("Live corpus/registry/index/labels changed during DOE discovery")
    ordered = sorted(rows.values(), key=lambda row: (row["available_by"], row["url"]))
    good = [pilot for pilot in pilots if pilot["status"] == "accessible_dated_spr_oil_prose"]
    report = {"at": now(), "discovery_only": True, "status": status, "window": [START, END],
              "new_title_candidates": len(ordered), "by_year": dict(sorted(Counter(row["available_by"][:4] for row in ordered).items())),
              "archive_pages": len(visited), "archive_items_inspected": inspected, "excluded": dict(excluded),
              "archives": archives, "pilots": pilots, "errors": errors,
              "accessible_dated_spr_oil_pilots": len(good), "pilot_body_words": [p["body_words"] for p in good],
              "downloaded_response_bytes": downloaded, "raw_documents_written": 0, "live_registry_writes": 0,
              "external_model_calls": 0, "live_inputs_unchanged": True, "live_hashes_before_and_after": before,
              "accepted_new_passages": None, "estimated_bulk_yield": None,
              "candidate_file_sha256": None, "script_sha256": digest(Path(__file__).read_bytes()),
              "rights": "Official US DOE government press releases; attribution/source URLs retained. Redistribution is review_required, not blanket approval. Exclude third-party photos, embeds and linked copyrighted materials.",
              "limitations": ["Candidate URLs were observed on actual official collection pages; title matching and three pilots are not corpus qualification.",
                               "Archive/body publication date and title must agree; .display-date is article-local. main time tags can describe unrelated previous/next news and are not date evidence.",
                               "Sale notices, solicitations, awards, deliveries and realized inventory changes have different statuses; announcement is not completion.",
                               "Emergency exchange loans/repaid barrels differ from permanent sales or purchases; preserve source units, quantities and dates.",
                               "Repeated policy boilerplate and overlapping announcement series need content deduplication and the same semantic policy, not artificial subdivision.",
                               "Only bounded archive indexes and three pilot bodies were fetched. No candidate or pilot word count is an accepted new passage count or proof of closing a corpus gap."]}
    write_jsonl(targets[0], ordered)
    report["candidate_file_sha256"] = digest(targets[0].read_bytes())
    write_json(targets[1], report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA / "unified_runs/semantic-page-v1/reports/doe-discovery")
    parser.add_argument("--pilot-limit", type=int, default=3)
    args = parser.parse_args()
    report = discover(args.output_dir, args.pilot_limit)
    print(json.dumps({key: report[key] for key in ("status", "new_title_candidates", "by_year", "archive_pages",
                                                 "accessible_dated_spr_oil_pilots", "pilot_body_words", "errors", "live_inputs_unchanged")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
