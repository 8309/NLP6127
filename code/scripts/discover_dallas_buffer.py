"""Actually linked Dallas Fed Energy Survey discovery; no live/raw writes.

Archive labels contain quarters, not publication days or article headlines.
Every admitted candidate therefore gets its exact day and title from the
source-native report header. Date-stratified pilots precede further discovery.
Survey diffusion indexes, forecasts and respondent comments are not verified
realized production statistics. No model, paid API or guessed URL is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
import httpx

from oil_evidence.corpus import Fetcher, date_from_text
from oil_evidence.segmentation import oil_scope_supported
from oil_evidence.store import DATA, digest, now, read_jsonl, words, write_json, write_jsonl


ARCHIVE = "https://www.dallasfed.org/research/surveys/des"
ENERGY = "https://www.dallasfed.org/research/energy"
START, END = "2018-01-01", "2026-10-05"
MAX_REPORTS = 40
SURVEY_ROUTE = re.compile(r"/research/surveys/des/(20\d{2})/(\d{2}0[1-4])(?:/(\d{2}0[1-4])update)?(?:\.aspx)?/?")
SECTION_IDS = ("des-report", "des-questions", "des-comments")


def canonical(url):
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", p.query, ""))


def admitted_dallas_route(url):
    """Exact official survey HTML route; discovery must still observe the link."""
    p = urlparse(url)
    m = SURVEY_ROUTE.fullmatch(p.path)
    if (p.scheme != "https" or p.hostname != "www.dallasfed.org" or p.username or p.password
            or p.port not in {None, 443} or p.query or p.fragment or m is None):
        raise ValueError("Dallas supplement requires an exact official quarterly survey HTML route")
    year, edition, update = m.groups()
    if not 2018 <= int(year) <= 2026 or edition[:2] != year[2:] or (update and update != edition):
        raise ValueError("Dallas survey route has conflicting edition/year or is outside the window")
    return canonical(url)


def linked_reports(raw, origin=ARCHIVE, known=()):
    """Never construct missing quarters; 2018 PDF-only reports are not HTML."""
    soup = BeautifulSoup(raw, "html.parser")
    body = soup.select_one("#content .dal-main-content")
    if body is None:
        raise ValueError("Dallas survey archive source-native main-content selector missing")
    rows, excluded = {}, Counter()
    for a in body.select("a[href]"):
        target = urljoin(origin, a["href"])
        p = urlparse(target)
        if "/surveys/des/" not in p.path:
            continue
        try:
            url = admitted_dallas_route(target)
        except ValueError:
            excluded["not_admitted_quarterly_html_route"] += 1
            continue
        if url in known:
            excluded["already_registered"] += 1
            continue
        m = SURVEY_ROUTE.fullmatch(p.path)
        rows[url] = dict(url=url, archive_label=a.get_text(" ", strip=True),
                         edition_year=m.group(1), edition_code=m.group(2),
                         archive_url=origin, discovery_method="actual_official_archive_link")
    if len(rows) > MAX_REPORTS:
        raise ValueError("Dallas survey archive exceeds the bounded report-discovery limit")
    return sorted(rows.values(), key=lambda r: (r["edition_year"], r["edition_code"], r["url"])), dict(excluded)


def _exact_day(value):
    parsed = date_from_text(value)
    if not parsed or date.fromisoformat(parsed).isoformat() != parsed:
        raise ValueError("Dallas report has no exact article-local publication day")
    return parsed


def _revision_guard(body, original):
    # Ignore global copyright/navigation dates. An explicit correction/update
    # within this report is different: do not silently use its current text at
    # the older original release date. Undated corrections also fail closed.
    for node in body.select("p,h5,time"):
        value = node.get_text(" ", strip=True)
        if re.match(r"^(?:last\s+)?(?:updated?|revised?|corrected?|correction)\b", value, re.I):
            revised = date_from_text(value)
            if not revised or revised > original:
                raise ValueError("Dallas current report has a later or undated revision; do not backdate")


def parse_dallas_body(raw):
    """Contract for root collector: exact published_date/title/text/blocks.

    Observed 2019/2021 templates: #des-report h5.survey-date OR
    div.survey-date--des, and #des-report h2.
    Extract each of the actual prose/question/comment sections once; do not
    flatten numeric tables, forecast charts or the print-only duplicate copy.
    """
    soup = BeautifulSoup(raw, "html.parser")
    main = soup.select_one("#content .dal-main-content")
    report = main.select_one("#des-report") if main else None
    if report is None:
        raise ValueError("Dallas source-native #content .dal-main-content #des-report missing")
    native_day = report.select_one("h5.survey-date,div.survey-date--des")
    headline = report.select_one("h2")
    if native_day is None or headline is None:
        raise ValueError("Dallas report-local publication header or headline missing")
    date_label = native_day.get_text(" ", strip=True)
    published = _exact_day(date_label)
    if not START <= published <= END:
        raise ValueError("Dallas report publication day outside the fixed corpus window")
    title = headline.get_text(" ", strip=True)
    if not title:
        raise ValueError("Dallas article headline empty")
    og = soup.select_one("meta[property='og:title'][content]")
    if og and " ".join(og["content"].split()).casefold() != " ".join(title.split()).casefold():
        raise ValueError("Dallas source-native report headline conflicts with og:title")
    _revision_guard(main, published)
    sections = [main.select_one("#" + section) for section in SECTION_IDS]
    # Work on detached copies: never let removing a table alter another section
    # or the date/title evidence captured above.
    blocks, structured, prose_scope = [], [], []
    removed_tables = 0
    for original in (section for section in sections if section is not None):
        section = BeautifulSoup(str(original), "html.parser").select_one("div")
        removed_tables += len(section.select("table"))
        for node in section.select("table,figure,script,style,nav,footer,.visible-print,.chart-full,.text-center,.dal-social__container--share"):
            if node.parent is not None:
                node.decompose()
        section_id = original.get("id")
        pending_headings, question = [], None
        for node in section.select("p,li,h2,h3,h4,h5,div.survey-date--des,div.report-date--des"):
            if node.name == "li" and node.find("p"):
                continue
            text = node.get_text(" ", strip=True)
            if not text or re.match(r"^(?:Next release:|Report in PDF|Additional Comments\s*[»>]?|Downloadable chart\b|Chart data\b)", text, re.I):
                continue
            if node.name.startswith("h") or node.name == "div":
                pending_headings.append(text)
                if node.name == "h4" and "?" in text:
                    question = text
                elif node.name == "h2":
                    question = None
                continue
            # Bind source-native headings/questions to the first following
            # response, without repeating or inventing content for each row.
            combined = "\n".join(pending_headings + [text])
            pending_headings = []
            prose_scope.append(text)
            blocks.append(combined)
            structured.append(dict(section_id=section_id, text=combined,
                                   survey_release_header=date_label, question=question,
                                   evidence_type="survey_respondent_comment" if node.name == "li" or node.find_parent("li") else "survey_analysis_or_expectation"))
    text = "\n\n".join(blocks)
    scope_text = "\n\n".join(prose_scope)
    if len(words(scope_text)) < 100 or not oil_scope_supported(scope_text):
        raise ValueError("Dallas extracted survey has no substantial explicit mineral-oil prose")
    return dict(published_date=published, available_by=published, modified_date=None,
                title=title, text=text, blocks=blocks, structured_blocks=structured, scope_text=scope_text,
                survey_period=date_label.split("|")[0].strip(), survey_release_header=date_label,
                evidence_type="official_survey_report_with_respondent_expectations_and_comments",
                removed_actual_table_subtrees=removed_tables,
                date_selector="#content .dal-main-content #des-report (h5.survey-date OR div.survey-date--des)",
                title_selector="#content .dal-main-content #des-report h2",
                body_selector="#content .dal-main-content #des-report/#des-questions/#des-comments p/li/h2/h3/h4/h5; excluding actual tables/charts/print-only copies")


def select_pilots(rows, limit=8):
    ordered = sorted(rows, key=lambda r: (r["edition_year"], r["edition_code"], r["url"]))
    if not ordered:
        return []
    positions = [round(i * (len(ordered) - 1) / max(1, limit - 1)) for i in range(limit)]
    return list({ordered[i]["url"]: ordered[i] for i in positions}.values())


def live_hashes():
    return {name: digest((DATA / name).read_bytes()) for name in
            ("sources.jsonl", "documents.jsonl", "passages.jsonl", "index/manifest.json", "index/vectors.npy",
             "annotations/codex.jsonl", "annotations/passage_labels.jsonl", "annotations/passage_review.jsonl")
            if (DATA / name).exists()}


def known_urls():
    paths = [DATA / "sources.jsonl", DATA / "documents.jsonl"]
    paths.extend((DATA / "unified_runs").glob("*/supplement/documents.jsonl"))
    return {canonical(row["url"]) for path in paths for row in read_jsonl(path)}


def discover(output_dir, pilot_limit=8):
    if not 6 <= pilot_limit <= 12:
        raise ValueError("Dallas discovery needs six-to-twelve edition-stratified pilots")
    output_dir = Path(output_dir).resolve()
    targets = [output_dir / "source_candidates.jsonl", output_dir / "discovery_report.json"]
    if any(path.exists() for path in targets):
        raise ValueError("Existing Dallas discovery artifacts are immutable; use a new directory")
    allowed = (DATA / "unified_runs").resolve()
    if not output_dir.is_relative_to(allowed) or output_dir == allowed:
        raise ValueError("Dallas outputs must be inside a separate unified run")
    before, known = live_hashes(), known_urls()
    script_hash_before = digest(Path(__file__).read_bytes())
    fetch = Fetcher()
    fetch.client.timeout = httpx.Timeout(20, connect=10)
    if (fetch.settings["request_delay_seconds"] < 1 or fetch.settings["max_document_mb"] > 25
            or fetch.settings["download_budget_mb"] > 300):
        fetch.close()
        raise ValueError("Fetcher must retain 1 second/host, 25 MB/document and 300 MB/invocation")
    rows, pilots, errors, archives, excluded = [], [], [], [], {}
    status, accepted, success = "not_attempted", {}, 0
    linked = []
    try:
        try:
            raw, final, mime = fetch.get(ARCHIVE)
            if canonical(final) != ARCHIVE:
                raise ValueError("Dallas archive redirected away from the admitted survey entry point")
            linked, excluded = linked_reports(raw, final, known)
            archives.append(dict(url=final, bytes=len(raw), mime=mime, sha256=digest(raw), actual_linked_reports=len(linked)))
            selected = select_pilots(linked, pilot_limit)
            order = selected + [r for r in linked if r["url"] not in {p["url"] for p in selected}]
            for number, row in enumerate(order):
                if number >= len(selected) and success < 6:
                    status = "pilot_gate_not_passed_no_bulk_discovery"
                    break
                result = dict(url=row["url"], archive_label=row["archive_label"], is_fixed_pilot=number < len(selected))
                try:
                    raw, final, mime = fetch.get(row["url"])
                    if admitted_dallas_route(final) != row["url"]:
                        raise ValueError("Dallas article redirected to a different survey edition")
                    parsed = parse_dallas_body(raw)
                    exact = parsed["published_date"]
                    if exact[:4] != row["edition_year"]:
                        raise ValueError("Dallas publication day conflicts with its observed edition year")
                    result.update(status="accessible_dated_oil_survey", article_title=parsed["title"],
                                  published_date=exact, final_url=final, raw_sha256=digest(raw),
                                  body_sha256=digest(parsed["text"]), bytes=len(raw), mime=mime,
                                  body_words=len(words(parsed["text"])), factual_blocks=len(parsed["blocks"]),
                                  removed_actual_table_subtrees=parsed["removed_actual_table_subtrees"],
                                  date_selector=parsed["date_selector"], body_selector=parsed["body_selector"],
                                  body_preview=parsed["text"][:600])
                    candidate = dict(row, source="DALLAS-OIL", title=parsed["title"], published_date=exact,
                        available_by=exact, date_precision="day", date_source=row["url"], event_date=None,
                        event_status="unlabelled", redistribution="review_required", requires_pdf=False,
                        content_scope="mineral_oil_survey_expectations_and_attributed_respondent_comments",
                        evidence_type=parsed["evidence_type"], survey_period=parsed["survey_period"],
                        survey_release_header=parsed["survey_release_header"],
                        date_evidence="source-native article-local #des-report h5.survey-date OR div.survey-date--des, not archive quarter/URL/date of a referenced event",
                        body_date_verified=True, body_title_verified=True, body_oil_verified=True,
                        pilot_status=result["status"] if result["is_fixed_pilot"] else "body_header_verified_during_discovery",
                        pilot_raw_sha256=result["raw_sha256"], pilot_body_sha256=result["body_sha256"])
                    accepted[row["url"]] = candidate
                    if result["is_fixed_pilot"]:
                        success += 1
                except Exception as exc:
                    result.update(status="failed_no_bypass", error=str(exc))
                    errors.append(dict(url=row["url"], stage="pilot" if result["is_fixed_pilot"] else "source_header_discovery", error=str(exc)))
                    # Any 403 ends this host's discovery, not just this URL.
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 403:
                        status = "host_403_stopped_no_bypass"
                        if result["is_fixed_pilot"]:
                            pilots.append(result)
                        break
                if result["is_fixed_pilot"]:
                    pilots.append(result)
                print(f"Dallas source header {number + 1}/{len(order)}: {result['status']} {row['url']}", flush=True)
            else:
                status = "linked_survey_headers_verified" if not errors else "partly_verified_failures_preserved"
        except Exception as exc:
            status = "archive_or_robots_blocked_no_bypass"
            errors.append(dict(stage="official_archive", url=ARCHIVE, error=str(exc)))
    finally:
        downloaded = fetch.bytes
        fetch.close()
    if live_hashes() != before:
        raise ValueError("Protected live corpus/index/labels changed during Dallas discovery")
    if script_hash_before != digest(Path(__file__).read_bytes()):
        raise ValueError("Dallas discovery script changed during execution; do not freeze mismatched artifacts")
    rows = sorted(accepted.values(), key=lambda r: (r["available_by"], r["url"]))
    report = dict(at=now(), discovery_only=True, status=status, window=[START, END],
        new_title_candidates=len(rows), actually_linked_reports=len(linked),
        by_year=dict(sorted(Counter(r["available_by"][:4] for r in rows).items())),
        archives=archives, pilots=pilots, errors=errors, excluded=excluded,
        accessible_dated_oil_pilots=success, pilot_gate_passed=success >= 6,
        pilot_body_words=[p["body_words"] for p in pilots if p["status"] == "accessible_dated_oil_survey"],
        downloaded_response_bytes=downloaded, raw_documents_written=0, live_registry_writes=0,
        external_model_calls=0, live_inputs_unchanged=True, live_hashes_before_and_after=before,
        accepted_new_passages=None, estimated_bulk_yield=None, candidate_file_sha256=None,
        script_sha256=script_hash_before,
        admitted_host="www.dallasfed.org", admitted_paths="/research/surveys/des/YYYY/YY0[1-4] plus actually linked matching update child; HTML only",
        rights="Federal Reserve Bank of Dallas text with retained attribution/source links; redistribution review_required. Third-party images, charts, tables, quotations and linked materials are not granted blanket rights.",
        limitations=["Candidates have verified report-local publication dates/headlines, but discovery body words are not accepted passage counts.",
            "First/Fourth quarter archive labels and URL year are edition selection evidence only, never exact publication dates.",
            "Respondent anecdotes, expected WTI prices, diffusion indexes and realized physical production are different evidence types; preserve survey attribution.",
            "Only source-native narrative/question/comment sections are extracted; actual tables/charts and print-only copies are excluded rather than flattened.",
            "Explicit later or undated corrections fail closed; collection response hashes do not prove an unchanged historical version when corrections are undisclosed.",
            "2018 survey links are PDFs, not admitted HTML reports; they need separate layout-aware extraction/rights review.",
            "Question/response headings are retained once in source order; structured_blocks carries exact survey-release/question context without artificially repeating it in each paragraph.",
            "The broader energy research archive is not admitted by this survey-only parser without its own native-date/title pilot validation."])
    write_jsonl(targets[0], rows)
    report["candidate_file_sha256"] = digest(targets[0].read_bytes())
    write_json(targets[1], report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA / "unified_runs/semantic-page-v2-buffer/reports/dallas-discovery")
    parser.add_argument("--pilot-limit", type=int, default=8)
    args = parser.parse_args()
    report = discover(args.output_dir, args.pilot_limit)
    print(json.dumps({key: report[key] for key in ("status", "new_title_candidates", "actually_linked_reports", "by_year", "accessible_dated_oil_pilots", "pilot_gate_passed", "candidate_file_sha256", "live_inputs_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
