"""Discover new, actually linked EIA oil/transport prose; never write live data.

Start from the official Today in Energy archive and follow its linked years.
The broader title screen finds fuels and transportation stories missed by the
original mineral-oil title screen. It is discovery, not proof of oil relevance:
every collected body still needs exact day/title and strong mineral-oil checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from oil_evidence.corpus import Fetcher, date_from_text
from oil_evidence.segmentation import oil_scope_supported
from oil_evidence.store import DATA, digest, now, read_jsonl, words, write_json, write_jsonl


ARCHIVE = "https://www.eia.gov/todayinenergy/archive.php"
START, END = "2018-01-01", "2026-10-05"
# Transportation/fuel titles can contain substantive petroleum facts without
# saying "oil". Non-oil power/natural-gas articles are not automatically included.
BROAD_TITLE = re.compile(
    r"\b(?:oil|crude|petroleum|opec|refin\w*|tanker\w*|barrels?|gasoline|diesel|"
    r"distillate\w*|jet fuel|fuel oil|liquid fuels?|hydrocarbon\w*|biofuel\w*|"
    r"ethanol|biodiesel|propane|butane|naphtha|SPR|transport\w*|airline\w*|"
    r"aviation|vehicles?|cars?|trucks?|fuel economy|motor fuels?|pipelines?)\b", re.I)
DATE_FURNITURE = re.compile(r"^(?:Tags\s*:|Principal contributors?\s*:)", re.I)
REVISION_NOTICE = re.compile(
    r"^\s*(?:(?:last\s+)?(?:updated?|correction|corrected|revised?|revision)"
    r"(?=\s*(?:[:\-–—]|20\d{2}-|\d{1,2}[/ ]|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|$))|"
    r"editor(?:['’]s|s['’]?)?\s+note\s*:|"
    r"(?:note\s*:\s*)?this\s+(?:article|story|post|text)\s+"
    r"(?:was|is|has\s+been)\s+(?:updated|corrected|revised)\b)", re.I)
NOTICE_DAYS = re.compile(
    r"\b20\d{2}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/20\d{2}\b|"
    r"\b\d{1,2} [A-Za-z]+ 20\d{2}\b|\b[A-Za-z]+ \d{1,2}, 20\d{2}\b")


def revision_guard(article, published):
    """A visible article revision cannot inherit the original release day.

    Match notice prefixes, not ordinary prose about revised estimates. Only
    article-native blocks are considered; source captions/navigation are not
    revision evidence. Undated or invalid-day notices fail closed, and ISO
    days are checked alongside the source's usual English date format.
    """
    for node in article.select("p,li,div"):
        if node.find_parent(["table", "figure", "nav", "footer", "script", "style"]):
            continue
        if node.find_parent(class_="source") or "source" in node.get("class", []):
            continue
        # Inspect actual blocks instead of flattened containers, whose leading
        # publication date can otherwise be mistaken for a revision's day.
        if node.name == "div" and node.find(["p", "li", "div", "h1", "h2", "h3", "h4"]):
            continue
        value = node.get_text(" ", strip=True)
        if not REVISION_NOTICE.match(value):
            continue
        days = []
        for token in NOTICE_DAYS.findall(value):
            try:
                day = date.fromisoformat(token).isoformat() if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", token) else date_from_text(token)
            except ValueError:
                day = None
            if not day:
                raise ValueError("EIA current article revision notice has an invalid day; historical body availability unverified")
            days.append(day)
        if not days:
            raise ValueError("EIA current article has an undated revision notice; historical body availability unverified")
        if max(days) > published:
            raise ValueError("EIA current article reports a later revision; historical body availability unverified")


def official(url):
    p = urlparse(url)
    return (p.scheme == "https" and p.hostname == "www.eia.gov"
            and not p.username and not p.password and p.port in {None, 443})


def canonical(url):
    """Drop fragments; article identity is its one official numeric id."""
    p = urlparse(url)
    if p.path == "/todayinenergy/detail.php":
        query = parse_qs(p.query)
        ids = query.get("id", [])
        if len(ids) == 1 and re.fullmatch(r"[0-9]+", ids[0]):
            return urlunparse((p.scheme, p.netloc, p.path, "", urlencode({"id": ids[0]}), ""))
    return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", p.query, ""))


def admitted_eia_route(url):
    p = urlparse(url)
    query = parse_qs(p.query, keep_blank_values=True)
    if (not official(url) or p.path != "/todayinenergy/detail.php"
            or set(query) != {"id"} or len(query["id"]) != 1
            or not re.fullmatch(r"[0-9]+", query["id"][0]) or p.fragment):
        raise ValueError("EIA supplement requires an exact official Today in Energy numeric article route")
    return canonical(url)


def linked_year_archives(raw, origin):
    soup = BeautifulSoup(raw, "html.parser")
    result = {}
    for a in soup.select("a[href]"):
        href = urljoin(origin, a["href"])
        p = urlparse(href)
        q = parse_qs(p.query)
        years = q.get("my", [])
        if (not official(href) or p.path != "/todayinenergy/archive.php"
                or set(q) != {"my"} or len(years) != 1 or not years[0].isdigit()):
            continue
        year = int(years[0])
        if 2018 <= year <= 2026:
            result[year] = href
    if not result:
        raise ValueError("EIA archive contains no actually linked year archives")
    return [result[year] for year in sorted(result)]


def parse_archive(raw, origin, known=()):
    soup = BeautifulSoup(raw, "html.parser")
    rows, exclusions, inspected = {}, Counter(), 0
    for a in soup.select("a[href]"):
        href = urljoin(origin, a["href"])
        if urlparse(href).path != "/todayinenergy/detail.php":
            continue
        inspected += 1
        try:
            href = admitted_eia_route(href)
        except ValueError:
            exclusions["unadmitted_article_route"] += 1
            continue
        title = a.get_text(" ", strip=True)
        item = a.find_parent("li")
        exact = date_from_text(item.get_text(" ", strip=True)) if item else None
        if not exact:
            exclusions["missing_article_local_archive_day"] += 1
            continue
        if not START <= exact <= END:
            exclusions["outside_window"] += 1
            continue
        if href in known:
            exclusions["already_registered"] += 1
            continue
        if not BROAD_TITLE.search(title):
            exclusions["not_oil_fuel_transport_title"] += 1
            continue
        rows[href] = dict(url=href, source="EIA-OIL-NEWS", title=title,
            published_date=exact, available_by=exact, date_precision="day",
            date_source=origin, event_date=None, event_status="unlabelled",
            redistribution="eia_attribution", requires_pdf=False,
            content_scope="fuel_transport_title_requires_strong_mineral_oil_body",
            discovery_method="actually_linked_official_today_in_energy_archive",
            date_evidence="EIA per-item archive publication day; article-local .tie-article .date must agree",
            title_match="broader_fuel_transport", pilot_status="not_fetched",
            body_date_verified=False, body_title_verified=False, body_oil_verified=False)
    if not inspected:
        raise ValueError("EIA archive has no actual article links")
    return list(rows.values()), dict(exclusions), inspected


def parse_eia_body(raw):
    """Collector contract: {published_date,title,text,blocks,... selectors}.

    Do not use a site-wide date, image captions or footer links as article text.
    Explicit subsequent corrections are fail-closed: the current article body
    cannot silently stand for the historical publication version.
    """
    soup = BeautifulSoup(raw, "html.parser")
    articles = soup.select(".tie-article")
    if len(articles) != 1:
        raise ValueError("EIA article-native .tie-article must be unique")
    article = articles[0]
    dates = article.select(".date")
    if len(dates) != 1:
        raise ValueError("EIA article-local publication day must be unique")
    exact = date_from_text(dates[0].get_text(" ", strip=True))
    if not exact or date.fromisoformat(exact).isoformat() != exact:
        raise ValueError("EIA article-local publication day is invalid")
    headings = article.select("h1")
    if len(headings) != 1:
        raise ValueError("EIA article-local title must be unique")
    title = headings[0].get_text(" ", strip=True)
    if not title:
        raise ValueError("EIA article title is missing")
    revision_guard(article, exact)
    for node in article.select("script,style,nav,footer,table,figure,.source,.article-type"):
        node.decompose()
    blocks = []
    for node in article.select("h1,h2,h3,h4,p,li"):
        # Some source-native in-brief markup leaves its image <p> unclosed.
        # Do not flatten the entire nested article into that image wrapper;
        # retain actual paragraphs in order without duplicate descendant text.
        if node.name == "p" and node.find(["p", "li", "h1", "h2", "h3", "h4"]):
            continue
        if node.find_parent("li"):
            continue
        if node.name == "li" and node.find("p"):
            # Keep the paragraph once rather than duplicate it via its wrapper.
            pieces = [p.get_text(" ", strip=True) for p in node.select("p")]
            value = "\n\n".join(p for p in pieces if p)
        else:
            value = node.get_text(" ", strip=True)
        if value and not DATE_FURNITURE.search(value) and (not blocks or blocks[-1] != value):
            blocks.append(value)
    text = "\n\n".join(blocks)
    if len(words(text)) < 100:
        raise ValueError("EIA article has fewer than 100 factual body words")
    return dict(published_date=exact, title=title, text=text, blocks=blocks,
                date_selector=".tie-article .date", body_selector=".tie-article h1,h2,h3,h4,p,li; tables/figures/captions excluded",
                body_oil_verified=oil_scope_supported(text))


def live_hashes():
    names = ["sources.jsonl", "documents.jsonl", "passages.jsonl", "index/manifest.json",
             "index/vectors.npy", "annotations/codex.jsonl", "annotations/passage_labels.jsonl",
             "annotations/passage_review.jsonl"]
    return {name: digest((DATA / name).read_bytes()) for name in names if (DATA / name).exists()}


def known_urls():
    paths = [DATA / "sources.jsonl", DATA / "documents.jsonl"]
    paths.extend((DATA / "unified_runs").glob("*/supplement/sources.jsonl"))
    paths.extend((DATA / "unified_runs").glob("*/supplement/documents.jsonl"))
    return {canonical(row["url"]) for path in paths for row in read_jsonl(path)}


def select_pilots(rows, limit=8):
    """Fixed date-stratified selection, not based on length or retrieval ranks."""
    ordered = sorted(rows, key=lambda r: (r["available_by"], r["url"]))
    if not ordered:
        return []
    positions = [round(i * (len(ordered) - 1) / max(1, limit - 1)) for i in range(limit)]
    return list({ordered[i]["url"]: ordered[i] for i in positions}.values())


def discover(output_dir, pilot_limit=8):
    if not 6 <= pilot_limit <= 12:
        raise ValueError("EIA buffer discovery needs a bounded six-to-twelve article pilot")
    output_dir = Path(output_dir).resolve()
    targets = [output_dir / "source_candidates.jsonl", output_dir / "discovery_report.json"]
    if any(path.exists() for path in targets):
        raise ValueError("Existing EIA discovery artifacts are immutable; use a new directory")
    allowed = (DATA / "unified_runs").resolve()
    if not output_dir.is_relative_to(allowed) or output_dir == allowed:
        raise ValueError("EIA discovery outputs must be inside a separate unified run")
    before, known = live_hashes(), known_urls()
    fetch = Fetcher()
    if (fetch.settings["request_delay_seconds"] < 1 or fetch.settings["max_document_mb"] > 25
            or fetch.settings["download_budget_mb"] > 300):
        fetch.close()
        raise ValueError("Fetcher must preserve 1 second/host, 25 MB/document and 300 MB/invocation")
    archives, errors, rows, pilots, excluded = [], [], {}, [], Counter()
    inspected, status = 0, "not_attempted"
    try:
        try:
            raw, final, mime = fetch.get(ARCHIVE)
            year_links = linked_year_archives(raw, final)
            page_cache = {canonical(final): (raw, final, mime)}
            for url in year_links:
                try:
                    if canonical(url) in page_cache:
                        raw, final, mime = page_cache[canonical(url)]
                    else:
                        raw, final, mime = fetch.get(url)
                    found, rejected, count = parse_archive(raw, final, known)
                    rows.update((row["url"], row) for row in found)
                    excluded.update(rejected)
                    inspected += count
                    archives.append(dict(url=final, bytes=len(raw), mime=mime, sha256=digest(raw),
                                         items=count, new_candidates=len(found), role="actually_linked_year_archive"))
                    print(f"EIA archive: {len(found)} new fuel/transport title candidates from {final}", flush=True)
                except Exception as exc:
                    errors.append(dict(url=url, stage="year_archive", error=str(exc)))
            status = "archive_scanned" if not errors else "partly_scanned_failures_preserved"
        except Exception as exc:
            status = "archive_blocked_no_bypass"
            errors.append(dict(url=ARCHIVE, stage="seed_archive", error=str(exc)))
        for row in select_pilots(list(rows.values()), pilot_limit):
            pilot = dict(url=row["url"], title=row["title"], archive_date=row["available_by"])
            try:
                raw, final, mime = fetch.get(row["url"])
                admitted_eia_route(final)
                parsed = parse_eia_body(raw)
                same_date = parsed["published_date"] == row["available_by"]
                same_title = " ".join(parsed["title"].split()).casefold() == " ".join(row["title"].split()).casefold()
                oil = parsed["body_oil_verified"]
                pilot_status = "accessible_dated_oil_prose" if same_date and same_title and oil else "date_title_or_body_scope_review_required"
                pilot.update(status=pilot_status, final_url=final, article_date=parsed["published_date"],
                             article_title=parsed["title"], date_matches_archive=same_date, title_matches_archive=same_title,
                             explicit_mineral_oil_body=oil, raw_sha256=digest(raw), body_sha256=digest(parsed["text"]),
                             bytes=len(raw), mime=mime, body_words=len(words(parsed["text"])), factual_blocks=len(parsed["blocks"]),
                             body_preview=parsed["text"][:600], date_selector=parsed["date_selector"], body_selector=parsed["body_selector"])
                row.update(pilot_status=pilot_status, body_date_verified=same_date, body_title_verified=same_title,
                           body_oil_verified=oil, article_date=parsed["published_date"], pilot_body_words=pilot["body_words"],
                           pilot_raw_sha256=pilot["raw_sha256"], pilot_body_sha256=pilot["body_sha256"])
            except Exception as exc:
                pilot.update(status="failed_no_bypass", error=str(exc))
                row.update(pilot_status="failed_no_bypass", pilot_error=str(exc))
            pilots.append(pilot)
            print(f"EIA pilot: {pilot['status']} {row['url']}", flush=True)
    finally:
        downloaded = fetch.bytes
        fetch.close()
    if live_hashes() != before:
        raise ValueError("EIA discovery found protected live input changes; artifact not accepted")
    ordered = sorted(rows.values(), key=lambda r: (r["available_by"], r["url"]))
    good = [p for p in pilots if p["status"] == "accessible_dated_oil_prose"]
    report = dict(at=now(), discovery_only=True, status=status, window=[START, END],
                  new_title_candidates=len(ordered), by_year=dict(sorted(Counter(r["available_by"][:4] for r in ordered).items())),
                  archive_items_inspected=inspected, excluded=dict(excluded), archives=archives, pilots=pilots, errors=errors,
                  accessible_dated_oil_pilots=len(good), pilot_body_words=[p["body_words"] for p in good],
                  pilot_body_word_total=sum(p["body_words"] for p in good), downloaded_response_bytes=downloaded,
                  accepted_new_passages=None, estimated_bulk_yield=None, raw_documents_written=0,
                  live_registry_writes=0, external_model_calls=0, live_inputs_unchanged=True,
                  live_hashes_before_and_after=before, candidate_file_sha256=None,
                  script_sha256=digest(Path(__file__).read_bytes()),
                  rights="EIA factual government prose with source attribution; exclude third-party images, captions and linked copyrighted articles. Attribution is not approval to redistribute third-party material.",
                  limitations=["Fuel/transport/pipeline titles are candidates, not oil-evidence or accepted-passage counts.",
                      "Every body must pass exact article-local publication day, title and strong mineral-oil scope checks; correction dates later than publication fail closed.",
                      "Current source bodies may include undated corrections; response hashes bind collection, not proof of an unchanged historical version.",
                      "Publication, referenced statistical periods, forecasts and realized events are distinct; preserve units and event status.",
                      "Eight deterministic date-stratified pilots are not representative evidence of bulk yield; collection, deduplication and unchanged semantic policy determine accepted passage counts.",
                      "Original sources, old live corpus, index and labels are never rewritten by discovery."])
    write_jsonl(targets[0], ordered)
    report["candidate_file_sha256"] = digest(targets[0].read_bytes())
    write_json(targets[1], report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=DATA / "unified_runs/semantic-page-v2-buffer/reports/eia-discovery")
    parser.add_argument("--pilot-limit", type=int, default=8)
    args = parser.parse_args()
    report = discover(args.output_dir, args.pilot_limit)
    print(json.dumps({key: report[key] for key in ("status", "new_title_candidates", "by_year",
                     "accessible_dated_oil_pilots", "pilot_body_words", "candidate_file_sha256", "live_inputs_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
