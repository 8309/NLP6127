"""Official oil-source discovery into a separate unified-run candidate archive.

Use actual linked Treasury public manifest/shard URLs, not guessed article URLs.
Download only archive indexes and a few pilot article bodies in memory.  Do not
write raw documents, live registries, corpora, indexes, or labels.  IEA robots
failure is recorded and never bypassed, and paid OMR content is never requested.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
import statistics
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from oil_evidence.corpus import Fetcher
from oil_evidence.store import DATA, digest, now, read_jsonl, words, write_json, write_jsonl


TREASURY = "https://home.treasury.gov/news/press-releases"
# Actual official URL located in web search. It is an initial discovery page,
# not an inferred series endpoint; if robots fail, no article fetch takes place.
IEA = "https://www.iea.org/reports/oil-market-report-september-2026"
START, END = "2018-01-01", "2026-10-05"
DIRECT_OIL = re.compile(r"\b(oil|crude|petroleum|tanker\w*|refiner\w*|shadow fleet|price cap|energy export\w*)\b", re.I)
COUNTRY = re.compile(r"\b(iran\w*|russia\w*|venezuela\w*)\b", re.I)
TRADE = re.compile(r"\b(network\w*|trade|export\w*|ship\w*|fleet\w*|energy|production)\b", re.I)
SANCTIONS = re.compile(r"\b(sanction\w*|target\w*|designat\w*|pressure|network\w*|restrict\w*|price cap)\b", re.I)
BODY_OIL = re.compile(r"\b(crude|petroleum|refiner\w*|tanker\w*|barrels?)\b|\boil\b[^.!?]{0,100}\b(export\w*|sale\w*|trade|transport\w*|ship\w*|supply|production|sector|revenue\w*)\b", re.I)


def canonical(url):
    return url.rstrip("/")


def official(url, hostname):
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname == hostname and not parsed.username and not parsed.password


def parse_manifest_link(raw, origin):
    soup = BeautifulSoup(raw, "html.parser")
    node = soup.select_one("[data-news-manifest]")
    if not node:
        raise ValueError("Official Treasury page has no linked public news manifest")
    href = urljoin(origin, node["data-news-manifest"])
    if not official(href, "home.treasury.gov"):
        raise ValueError("Treasury manifest left its official host")
    return href


def parse_shard_links(raw, origin):
    manifest = json.loads(raw)
    links = []
    for shard in manifest.get("searchShards", []):
        low, high = shard.get("startYear"), shard.get("endYear")
        if not isinstance(low, int) or not isinstance(high, int) or high < 2018 or low > 2026:
            continue
        href = urljoin(origin, shard["path"])
        if not official(href, "home.treasury.gov"):
            raise ValueError("Treasury search shard left its official host")
        links.append(href)
    if not links:
        raise ValueError("Official manifest has no matching year shards")
    return list(dict.fromkeys(links))


def candidates_from_shard(raw, origin, known):
    payload = json.loads(raw)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Treasury official search shard has no item list")
    rows = []
    all_window_items = 0
    for item in items:
        exact = item.get("date")
        try:
            if not isinstance(exact, str) or date.fromisoformat(exact).isoformat() != exact:
                continue
        except ValueError:
            continue
        if not START <= exact <= END:
            continue
        all_window_items += 1
        title = item.get("title", "")
        if not isinstance(title, str) or not title:
            continue
        direct = bool(DIRECT_OIL.search(title))
        related = bool(COUNTRY.search(title) and TRADE.search(title) and SANCTIONS.search(title))
        if not (direct or related):
            continue
        href = urljoin(origin, item.get("url", ""))
        if not official(href, "home.treasury.gov") or not re.fullmatch(r"/news/press-releases/[a-z0-9-]+/?", urlparse(href).path):
            continue
        if canonical(href) in known:
            continue
        rows.append(dict(url=href, source="TREASURY-OIL", title=title, published_date=exact,
            available_by=exact, date_precision="day", date_source=origin, event_date=None,
            event_status="unlabelled", redistribution="review_required", requires_pdf=False,
            content_scope="oil_sanctions_shipping_trade" if direct else "country_trade_network_requires_body_oil_check",
            discovery_method="actually_linked_official_news_manifest_search_shard",
            date_evidence="Official Treasury indexed publication day; article time must agree before collection acceptance",
            pilot_status="not_fetched", body_date_verified=False, body_oil_verified=False,
            title_match="direct_oil" if direct else "country_trade_network"))
    return rows, all_window_items, len(items)


def parse_treasury_body(raw):
    soup = BeautifulSoup(raw, "html.parser")
    article = soup.select_one("article.entity--type-node")
    if not article:
        raise ValueError("Treasury factual article body selector missing")
    time_tag = article.select_one("time[datetime]") or soup.select_one("time[datetime]")
    exact = time_tag.get("datetime", "")[:10] if time_tag else None
    if not exact or date.fromisoformat(exact).isoformat() != exact:
        raise ValueError("Treasury article lacks a valid publication day")
    # Keep headings and paragraphs in source order, ignore share/footer menus.
    blocks = []
    for node in article.select("p,li,h2,h3,h4"):
        if node.find_parent(["nav", "footer"]):
            continue
        if node.name == "li" and node.find("p"):
            continue
        text = node.get_text(" ", strip=True)
        if text and text not in {"Twitter", "Facebook", "###"}:
            blocks.append(text)
    text = "\n\n".join(blocks)
    if len(words(text)) < 100:
        raise ValueError("Treasury article has no substantial factual prose")
    return exact, text, blocks


def select_pilots(rows, maximum=6):
    # Fixed source/title/date stratification, not selecting retrieval successes
    # or longest bodies. Include older policy regimes and country coverage.
    ordered = sorted([row for row in rows if row["title_match"] == "direct_oil"],
                     key=lambda row: (row["available_by"], row["url"]))
    picks = []
    for country in ("iran", "russia", "venezuela"):
        group = [row for row in ordered if country in row["title"].casefold()]
        if group:
            picks.append(group[0])
            picks.append(group[-1])
    if ordered:
        picks.extend([ordered[0], ordered[len(ordered) // 2], ordered[-1]])
    chosen = list({row["url"]: row for row in picks}.values())[:maximum]
    return chosen


def frozen_live_hashes():
    paths = ["sources.jsonl", "documents.jsonl", "passages.jsonl", "index/manifest.json", "index/vectors.npy",
             "annotations/codex.jsonl", "annotations/passage_labels.jsonl", "annotations/passage_review.jsonl"]
    return {name: digest((DATA / name).read_bytes()) for name in paths if (DATA / name).exists()}


def discover(output_dir, pilot_limit=6):
    if not 0 <= pilot_limit <= 8:
        raise ValueError("Discovery pilot limit must be between zero and eight")
    output_dir = Path(output_dir).resolve()
    targets = [output_dir / "source_candidates.jsonl", output_dir / "discovery_report.json"]
    if any(path.exists() for path in targets):
        raise ValueError("Existing discovery artifacts are frozen; use a new output directory")
    if output_dir == DATA.resolve() or output_dir.is_relative_to(DATA / "index") or output_dir.is_relative_to(DATA / "annotations"):
        raise ValueError("Discovery output cannot replace a live registry/index/annotation directory")
    before = frozen_live_hashes()
    known = {canonical(row["url"]) for row in read_jsonl(DATA / "sources.jsonl")}
    known |= {canonical(row["url"]) for row in read_jsonl(DATA / "documents.jsonl")}
    fetch = Fetcher()
    if fetch.settings["request_delay_seconds"] < 1 or fetch.settings["max_document_mb"] > 25 or fetch.settings["download_budget_mb"] > 300:
        fetch.close()
        raise ValueError("Fetcher limits must satisfy 1 second/host, 25 MB/document, 300 MB/invocation")
    archives, errors, rows, pilots = [], [], {}, []
    all_window_items, index_items = 0, 0
    iea = {"source": "IEA-OMR-PUBLIC", "url": IEA,
           "scope": "Public overview only; no subscriber PDF, credentials, or access-control workaround",
           "redistribution": "review_required", "status": "not_attempted", "candidate_yield": None}
    try:
        try:
            raw, final, mime = fetch.get(IEA)
            iea.update(status="accessible", final_url=final, bytes=len(raw), mime=mime, sha256=digest(raw))
            # Initial discovery only. A successful public page does not authorize
            # subscription downloads or establish republication rights.
            soup = BeautifulSoup(raw, "html.parser")
            iea["actually_linked_previous_editions"] = sorted({urljoin(final, a["href"]) for a in soup.select("a[href]")
                if "/reports/oil-market-report-" in a["href"]})
        except Exception as exc:
            iea.update(status="blocked_no_bypass", error=str(exc))
        raw, final, mime = fetch.get(TREASURY)
        archives.append({"url": final, "bytes": len(raw), "mime": mime, "sha256": digest(raw), "role": "official_archive_html"})
        manifest_url = parse_manifest_link(raw, final)
        raw, final, mime = fetch.get(manifest_url)
        archives.append({"url": final, "bytes": len(raw), "mime": mime, "sha256": digest(raw), "role": "actually_linked_manifest"})
        shards = parse_shard_links(raw, final)
        for url in shards:
            try:
                raw, final, mime = fetch.get(url)
                found, window_count, item_count = candidates_from_shard(raw, final, known)
                all_window_items += window_count
                index_items += item_count
                archives.append({"url": final, "bytes": len(raw), "mime": mime, "sha256": digest(raw),
                                 "role": "actually_linked_year_search_shard", "index_items": item_count,
                                 "window_items": window_count, "title_candidates": len(found)})
                rows.update((row["url"], row) for row in found)
                print(f"Official linked Treasury index: {len(found)} new title candidates from {final}", flush=True)
            except Exception as exc:
                errors.append({"url": url, "stage": "archive_index", "error": str(exc)})
        for row in select_pilots(list(rows.values()), pilot_limit):
            pilot = {"url": row["url"], "title": row["title"], "archive_date": row["available_by"]}
            try:
                raw, final, mime = fetch.get(row["url"])
                exact, text, blocks = parse_treasury_body(raw)
                # The generic corpus topic helper also matches "sanctions".
                # For a sanctions-source pilot that is not a mineral-oil test.
                topical = bool(BODY_OIL.search(text))
                same_date = exact == row["available_by"]
                status = "accessible_dated_oil_prose" if topical and same_date else "body_scope_or_date_review_required"
                pilot.update(status=status, final_url=final, article_date=exact, date_matches_archive=same_date,
                             bytes=len(raw), mime=mime, raw_sha256=digest(raw), body_sha256=digest(text),
                             body_words=len(words(text)), factual_blocks=len(blocks), explicit_oil_operation_anchor=topical,
                             body_preview=text[:600])
                row.update(pilot_status=status, body_date_verified=same_date, body_oil_verified=topical,
                           pilot_body_words=len(words(text)), pilot_body_sha256=digest(text),
                           pilot_raw_sha256=digest(raw), article_date=exact)
            except Exception as exc:
                pilot.update(status="failed_no_bypass", error=str(exc))
                row.update(pilot_status="failed_no_bypass", pilot_error=str(exc))
            pilots.append(pilot)
    finally:
        downloaded = fetch.bytes
        fetch.close()
    if frozen_live_hashes() != before:
        raise ValueError("Live data changed during read-only discovery; artifact not accepted")
    ordered = sorted(rows.values(), key=lambda row: (row["available_by"], row["url"]))
    good = [p for p in pilots if p["status"] == "accessible_dated_oil_prose"]
    counts = [p["body_words"] for p in good]
    direct_count = sum(row["title_match"] == "direct_oil" for row in ordered)
    estimation = {
        "status": "unvalidated_scaling_proxy_not_collected_yield", "title_candidates": len(ordered),
        "direct_oil_title_candidates": direct_count,
        "country_trade_network_candidates_not_scaled": len(ordered) - direct_count,
        "pilot_articles": len(pilots), "accessible_dated_oil_pilots": len(good),
        "pilot_word_min": min(counts) if counts else None, "pilot_word_median": statistics.median(counts) if counts else None,
        "pilot_word_max": max(counts) if counts else None,
        "median_scaled_direct_title_body_words": int(statistics.median(counts) * direct_count) if counts else None,
        "limitations": "Small deterministic direct-oil-title/date/country-stratified pilot, not random representative sampling. Body word totals include legal boilerplate and non-oil sentences; they are not oil evidence counts. Country/trade/network-only titles are NOT scaled from direct-oil pilots. Full body collection, original-source date consistency, deduplication, identical semantic policy, per-block explicit-oil/quality gates and rights review may remove substantial content. No estimate is an accepted passage count, corpus expansion, or proof of filling the gap."}
    report = {"at": now(), "discovery_only": True, "window": [START, END], "external_model_calls": 0,
              "raw_documents_written": 0, "live_registry_writes": 0, "live_inputs_unchanged": True,
              "live_hashes_before_and_after": before, "downloaded_response_bytes": downloaded,
              "limits": {"request_delay_seconds": 1, "max_document_mb": 25, "download_budget_mb": 300, "attempts": 3},
              "official_index_items_inspected": index_items, "window_index_items": all_window_items,
              "new_title_candidates": len(ordered), "by_source": dict(Counter(row["source"] for row in ordered)),
              "by_year": dict(sorted(Counter(row["available_by"][:4] for row in ordered).items())),
              "candidate_file_sha256": None, "archives": archives, "pilots": pilots, "errors": errors,
              "iea": iea, "yield_estimate": estimation,
              "rights": {"TREASURY-OIL": "Official US-government factual press releases, with attribution URLs retained. Redistribution remains review_required; exclude photos, third-party embeds, and linked copyrighted materials. Public accessibility is not itself a blanket license finding.",
                         "IEA-OMR-PUBLIC": "Only public overview considered; OMR subscription/download access and republication terms unresolved, and robots failure blocks this pipeline."},
              "date_version_risks": ["Archive publication days and article time tags can conflict: fail closed until checked.",
                                     "Treasury claims about sanctions and targeted oil trades are attributed official assertions, not independently proven production or export outcomes.",
                                     "Announcement date does not automatically equal legal effective date, event date, vessel behavior, or realized supply impact.",
                                     "Dynamic archive/body versions are bound by hashes for this discovery only; full collection must store raw response and verify actual article dates.",
                                     "A high title-match candidate count is not a count of accepted sources or semantic passages."],
              "script_sha256": digest(Path(__file__).read_bytes())}
    write_jsonl(targets[0], ordered)
    report["candidate_file_sha256"] = digest(targets[0].read_bytes())
    write_json(targets[1], report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA / "unified_runs/semantic-page-v1/reports")
    parser.add_argument("--pilot-limit", type=int, default=6)
    args = parser.parse_args()
    report = discover(args.output_dir, args.pilot_limit)
    print(json.dumps({key: report[key] for key in ("new_title_candidates", "by_year", "iea", "yield_estimate", "live_inputs_unchanged")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
