"""Bounded Dallas energy-research discovery and six fixed native-HTML pilots.

Follow literal archive links only. No live/raw registry writes, guessed URLs,
PDF/chart downloads, model calls, or claims that candidates are accepted data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
import httpx

from oil_evidence.corpus import Fetcher, USER_AGENT, date_from_text
from oil_evidence.segmentation import oil_scope_supported
from oil_evidence.store import DATA, digest, now, read_jsonl, words, write_json, write_jsonl


ENTRY = "https://www.dallasfed.org/research/energy"
START, END = "2018-01-01", "2026-10-05"
HOST = "www.dallasfed.org"
ARTICLE = re.compile(r"/research/(?:economics/20\d{2}/\d{4}|swe/20\d{2}/swe\d{4}[a-z]?|energy/indicators/20\d{2}/en\d{4})(?:\.aspx)?/?")
TITLE_SCOPE = re.compile(r"\b(?:oil|crude|petroleum|OPEC|refin\w*|gasoline|diesel|distillate\w*|jet\s+fuel|Hormuz|tanker\w*|barrels?|fuel[- ]price)\b", re.I)
REVISION = re.compile(r"^\s*(?:(?:last\s+)?(?:updated?|correction|corrected|revised?|revision)(?=\s*(?:[:\-–—]|20\d{2}-|\d|(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|$))|editor(?:['’]s|s['’]?)?\s+note\s*:|(?:note\s*:\s*)?this\s+(?:article|story|post|text)\s+(?:was|has\s+been|is)\s+(?:updated|corrected|revised)\b)", re.I)
NOTICE_DAY = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/20\d{2}\b|\b\d{1,2} [A-Za-z]+ 20\d{2}\b|\b[A-Za-z]+ \d{1,2}, 20\d{2}\b")


def official(url):
    p = urlparse(url)
    return p.scheme == "https" and p.hostname == HOST and not p.username and not p.password and p.port in {None, 443}


def canonical(url):
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, re.sub(r"\.aspx$", "", p.path.rstrip("/")), "", p.query, ""))


def admitted_dallas_research_route(url):
    p = urlparse(url)
    if not official(url) or p.query or p.fragment or not ARTICLE.fullmatch(p.path):
        raise ValueError("Dallas research requires a literal official economics/Southwest Economy/energy-indicator HTML article route")
    return canonical(url)


def admitted_archive(url):
    p = urlparse(url)
    q = parse_qs(p.query, keep_blank_values=True)
    if (not official(url) or p.fragment or canonical(url).split("?", 1)[0] != "https://www.dallasfed.org/topics/energy"
            or (q and (set(q) != {"page"} or len(q["page"]) != 1 or not q["page"][0].isdigit() or not 1 <= int(q["page"][0]) <= 29))):
        raise ValueError("Not an actually linked Dallas energy-topic archive route")
    return canonical(url)


class DallasResearchFetcher(Fetcher):
    """All robots/article redirects, bytes, host delays and attempts are bounded."""
    def __init__(self, client=None):
        super().__init__()
        self.client.timeout = httpx.Timeout(25, connect=10)
        if client is not None:
            self.client.close()
            self.client = client
        self.requests = []

    def _request(self, url, role, allow_missing=False, delay=None):
        if not official(url):
            raise ValueError("Every Dallas request/redirect must retain the exact official HTTPS host")
        delay = max(self.settings["request_delay_seconds"], delay or 0)
        for attempt in range(1, 4):
            time.sleep(max(0, delay - (time.monotonic() - self.last.get(HOST, 0))))
            self.last[HOST] = time.monotonic()
            record = dict(url=url, role=role, attempt=attempt)
            try:
                with self.client.stream("GET", url) as response:
                    record["status"] = response.status_code
                    if response.is_redirect:
                        record["location"] = response.headers.get("location")
                        self.requests.append(record)
                        return b"", dict(response.headers), response.status_code
                    if not (allow_missing and response.status_code in {404, 410}):
                        response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        self.bytes += len(chunk)
                        if size > self.settings["max_document_mb"] * 1024**2:
                            raise ValueError("Dallas response exceeds document size limit")
                        if self.bytes > self.settings["download_budget_mb"] * 1024**2:
                            raise ValueError("Dallas discovery download budget reached")
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    record.update(bytes=size, sha256=digest(raw))
                    self.requests.append(record)
                    return raw, dict(response.headers), response.status_code
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                record["error"] = str(exc)
                self.requests.append(record)
                if attempt == 3:
                    raise
            except Exception as exc:
                record["error"] = str(exc)
                self.requests.append(record)
                raise

    def _resource(self, url, role, allow_missing=False, delay=None):
        for redirects in range(6):
            raw, headers, status = self._request(url, role, allow_missing, delay)
            if status not in {301, 302, 303, 307, 308}:
                return raw, url, headers.get("content-type", ""), status
            if redirects == 5 or not headers.get("location"):
                raise ValueError("Dallas source exceeded five redirects or has no redirect target")
            url = urljoin(url, headers["location"])
        raise AssertionError("Unreachable redirect loop")

    def get(self, url, role="source"):
        if not official(url):
            raise ValueError("Dallas source must retain the exact official HTTPS host")
        if HOST not in self.robots:
            raw, final, mime, status = self._resource("https://www.dallasfed.org/robots.txt", "robots", True)
            parser = RobotFileParser()
            if status in {404, 410}:
                parser.parse([])
            elif status == 200 and b"<html" not in raw[:300].lower():
                parser.parse(raw.decode("utf-8", errors="replace").splitlines())
            else:
                raise ValueError("Dallas robots.txt cannot be verified")
            self.robots[HOST] = parser
        parser = self.robots[HOST]
        # Check each observed destination before it is requested; no recursive
        # outer retry can restart another redirect destination's three attempts.
        for redirects in range(6):
            if not official(url) or not parser.can_fetch(USER_AGENT, url):
                raise ValueError("Dallas robots/official-host validation disallows this destination")
            raw, headers, status = self._request(url, role, delay=parser.crawl_delay(USER_AGENT))
            if status not in {301, 302, 303, 307, 308}:
                return raw, url, headers.get("content-type", "")
            if redirects == 5 or not headers.get("location"):
                raise ValueError("Dallas source exceeded five redirects or has no redirect target")
            url = urljoin(url, headers["location"])
        raise AssertionError("Unreachable article redirect loop")


def linked_topic(raw, origin):
    soup = BeautifulSoup(raw, "html.parser")
    matches = {urljoin(origin, a["href"]) for a in soup.select("#content a[href]")
               if a.get_text(" ", strip=True) == "Energy articles and analysis"}
    if len(matches) != 1:
        raise ValueError("Research entry must expose its one actual Energy articles and analysis link")
    url = matches.pop()
    admitted_archive(url)
    return url


def parse_archive(raw, origin, known=()):
    admitted_archive(origin)
    soup = BeautifulSoup(raw, "html.parser")
    main = soup.select_one("#content .dal-main-content")
    items = main.select("li.dal-results-list__item") if main else []
    if not items:
        raise ValueError("Dallas source-native energy-topic list missing")
    rows, excluded = {}, Counter()
    for item in items:
        heading, day_node = item.select_one("p.dal-headline a[href]"), item.select_one("p.dal-content-date")
        if heading is None or day_node is None:
            excluded["missing_native_item_title_or_day"] += 1
            continue
        url = urljoin(origin, heading["href"])
        exact = date_from_text(day_node.get_text(" ", strip=True))
        if not exact or not START <= exact <= END:
            excluded["missing_day_or_outside_window"] += 1
            continue
        try:
            url = admitted_dallas_research_route(url)
        except ValueError:
            excluded["unadmitted_family_event_survey_abstract_or_nonofficial"] += 1
            continue
        if url in known:
            excluded["already_registered"] += 1
            continue
        title = heading.get_text(" ", strip=True)
        teaser = item.select_one("p.dal-teaser")
        teaser = teaser.get_text(" ", strip=True) if teaser else ""
        if not TITLE_SCOPE.search(title + " " + teaser):
            excluded["no_oil_title_or_native_teaser_anchor"] += 1
            continue
        record = dict(url=url, source="DALLAS-RESEARCH-OIL", title=title,
                      published_date=exact, available_by=exact, date_precision="day", date_source=origin,
                      event_date=None, event_status="unlabelled", redistribution="review_required",
                      requires_pdf=False, archive_teaser=teaser, archive_url=origin,
                      discovery_method="literal_energy_topic_item_title_link_and_publication_day",
                      content_scope="authorial_oil_research_or_energy_statistics_not_official_policy",
                      body_date_verified=False, body_title_verified=False, body_oil_verified=False,
                      pilot_status="not_fetched")
        if url in rows and rows[url] != record:
            raise ValueError("Conflicting archive rows for the same Dallas research URL")
        rows[url] = record
    next_link = main.select_one("nav.dal-pagination__container a.next[href][aria-disabled='false']")
    next_url = urljoin(origin, next_link["href"]) if next_link else None
    if next_url:
        next_url = admitted_archive(next_url)
    return list(rows.values()), next_url, dict(excluded), len(items)


def normalized_title(value):
    return unicodedata.normalize("NFKC", " ".join(value.split())).casefold()


def parse_dallas_research_body(raw):
    soup = BeautifulSoup(raw, "html.parser")
    mains = soup.select("#content .dal-main-content")
    if len(mains) != 1:
        raise ValueError("Dallas research source-native main must be unique")
    main = mains[0]
    headings = main.select(":scope > h1.dal-headline")
    dates = main.select(":scope > .dal-inline-list > .dal-content-date")
    # Observed 2021/2023 Energy Indicators use a direct p.date after the
    # banner, whereas the 2024/2026 template uses the inline-list field. This
    # is a source-native series template, not fallback to arbitrary body dates.
    date_selector = "#content .dal-main-content > .dal-inline-list > .dal-content-date"
    if not dates and len(headings) == 1 and normalized_title(headings[0].get_text(" ", strip=True)) == "energy indicators":
        dates = main.select(":scope > p.date")
        if dates:
            first_prose = next((p for p in main.find_all("p", recursive=False) if "date" not in p.get("class", [])), None)
            children = list(main.children)
            if first_prose is not None and children.index(dates[0]) > children.index(first_prose):
                raise ValueError("Dallas legacy indicator date occurs after body prose, not in its publication header")
            date_selector = "#content .dal-main-content > p.date (observed legacy Energy Indicators header only)"
    if len(headings) != 1 or len(dates) != 1:
        raise ValueError("Dallas research direct article-native title/publication day missing or ambiguous")
    title = headings[0].get_text(" ", strip=True)
    # Native legacy headers can contain doubled HTML text-layer spaces (e.g.
    # "May  25, 2023"). Normalize only this metadata label, never the prose.
    exact = date_from_text(" ".join(dates[0].get_text(" ", strip=True).split()))
    if not title or not exact or date.fromisoformat(exact).isoformat() != exact:
        raise ValueError("Dallas research exact header day/title invalid")
    author_node = main.select_one(":scope > .dal-author")
    authors = author_node.get_text(" ", strip=True) if author_node else None
    disclaimers = [node.get_text(" ", strip=True) for node in main.select(".dal-disclaimer")]
    revision_notes, revisions = [], []
    for node in main.select("p,li"):
        if node.find_parent(["table", "figure", "nav", "footer", "aside"]):
            continue
        text = node.get_text(" ", strip=True)
        if not REVISION.match(text):
            continue
        tokens = NOTICE_DAY.findall(text)
        if not tokens:
            raise ValueError("Dallas research visible revision notice has no verified year/day")
        parsed_days = []
        for token in tokens:
            try:
                day = date.fromisoformat(token).isoformat() if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", token) else date_from_text(token)
            except ValueError:
                day = None
            if not day:
                raise ValueError("Dallas research revision notice day invalid")
            parsed_days.append(day)
        revision_notes.append(text)
        revisions.extend(parsed_days)
    available = max([exact, *revisions])
    for node in main.select("table,figure,script,style,nav,footer,aside,.dal-info__container,.dal-author,.dal-inline-list,.dal-social__container--share,.dal-notes__list,.dal-notes__headline,.dal-content-topics__container,.dal-disclaimer,:scope > p.date,.chart-full"):
        if node.parent is not None:
            node.decompose()
    for node in main.select("sup a[href^='#n'],sup a[href^='#fn']"):
        if node.parent is not None and node.parent.name == "sup":
            node.parent.decompose()
    blocks, scope_blocks = [], []
    for node in main.select("h2,h3,h4,p,li"):
        if node.name == "li" and node.find("p"):
            continue
        if node.name == "p" and node.find(["p", "li", "h2", "h3", "h4"]):
            continue
        if node.find_parent(["table", "figure", "nav", "footer", "aside"]):
            continue
        text = node.get_text(" ", strip=True)
        if not text or re.match(r"^(?:Chart data|Download|Subscribe|Share this)\b", text, re.I):
            continue
        if not blocks or blocks[-1] != text:
            blocks.append(text)
            if node.name in {"p", "li"}:
                scope_blocks.append(text)
    body, scope = "\n\n".join(blocks), "\n\n".join(scope_blocks)
    if len(words(scope)) < 100:
        raise ValueError("Dallas research has fewer than 100 heading-free factual body words")
    return dict(published_date=exact, available_by=available, modified_date=max(revisions) if revisions else None,
                title=title, authors=authors, author_disclaimers=disclaimers, text=body, blocks=blocks,
                scope_text=scope, body_oil_verified=oil_scope_supported(scope), visible_revision_notices=revision_notes,
                evidence_type="authorial_research_analysis_statistics_or_model_projection_not_official_fed_policy",
                date_selector=date_selector,
                body_selector="#content .dal-main-content source-native p/li/h2/h3/h4 excluding tables/figures/author-footer/reference panels",
                version_caution="Current HTML response is hash-bound; unknown historical/undisclosed revisions are not independently verified. Known revision dates conservatively raise available_by.")


def known_urls():
    paths = [DATA / "documents.jsonl", DATA / "sources.jsonl"]
    paths += sorted((DATA / "unified_runs").glob("*/supplement/documents.jsonl"))
    paths += sorted((DATA / "unified_runs").glob("*/supplement/sources.jsonl"))
    return {canonical(row["url"]) for path in paths for row in read_jsonl(path)}


def live_hashes():
    names = ["documents.jsonl", "sources.jsonl", "passages.jsonl", "index/manifest.json", "index/vectors.npy",
             "annotations/codex.jsonl", "annotations/passage_labels.jsonl", "annotations/passage_review.jsonl",
             "reports/corpus.json", "reports/corpus_resegmentation.json", "reports/passage_annotation.json",
             "model_usage.sqlite3", "annotations.sqlite3"]
    return {name: digest((DATA / name).read_bytes()) for name in names if (DATA / name).exists()}


def select_pilots(rows):
    ordered = sorted(rows, key=lambda r: (r["published_date"], r["url"]))
    if len(ordered) < 6:
        return ordered
    return [ordered[round(i * (len(ordered) - 1) / 5)] for i in range(6)]


def discover(output_dir, candidate_limit=100):
    if not 6 <= candidate_limit <= 100:
        raise ValueError("Dallas research candidate cap must be 6..100")
    output_dir = Path(output_dir).resolve()
    allowed = (DATA / "unified_runs").resolve()
    if not output_dir.is_relative_to(allowed) or output_dir == allowed or output_dir.exists():
        raise ValueError("Use a new immutable Dallas discovery directory inside a unified run")
    before, known = live_hashes(), known_urls()
    script_sha = digest(Path(__file__).read_bytes())
    fetch = DallasResearchFetcher()
    if (fetch.settings["request_delay_seconds"] < 1 or fetch.settings["max_document_mb"] > 25
            or fetch.settings["download_budget_mb"] > 300):
        fetch.close()
        raise ValueError("Preserve 1 second/host, 25 MB/document and 300 MB/invocation caps")
    output_dir.mkdir(parents=True)
    records, excluded, archives, errors, pilots, visited = {}, Counter(), [], [], [], set()
    inspected, state = 0, "not_attempted"
    try:
        try:
            raw, final, mime = fetch.get(ENTRY, "entry")
            if canonical(final) != ENTRY:
                raise ValueError("Dallas research entry redirected to another resource")
            (output_dir / "entry.html").write_bytes(raw)
            url = linked_topic(raw, final)
            while url and len(visited) < 29 and len(records) < candidate_limit:
                identity = admitted_archive(url)
                if identity in visited:
                    raise ValueError("Dallas actual pagination loop")
                visited.add(identity)
                raw, final, mime = fetch.get(url, "archive")
                if admitted_archive(final) != identity:
                    raise ValueError("Dallas archive redirected to a different page identity")
                found, next_url, rejected, count = parse_archive(raw, final, known)
                path = output_dir / "archive_html" / (digest(final)[:20] + ".html")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                inspected += count
                excluded.update(rejected)
                for row in found:
                    prior = records.get(row["url"])
                    if prior and any(prior[k] != row[k] for k in ["title", "published_date"]):
                        raise ValueError("Conflicting Dallas metadata across archive pages")
                    records.setdefault(row["url"], row)
                archives.append(dict(url=final, bytes=len(raw), sha256=digest(raw), items=count, new_candidates=len(found), next_actual_link=next_url))
                print(f"Dallas research archive {len(visited)}: {len(records)} actual oil-title/teaser candidates", flush=True)
                url = next_url
            state = "archive_scanned_candidate_cap" if len(records) >= candidate_limit else "archive_scanned"
        except Exception as exc:
            state = "archive_failure_preserved_no_bypass"
            errors.append(dict(stage="entry_or_archive", error=str(exc)))
        ordered = sorted(records.values(), key=lambda r: (r["published_date"], r["url"]))[:candidate_limit]
        fixed = select_pilots(ordered)
        # Archive access denial/budget exhaustion is terminal for this host.
        terminal = any("403" in e["error"] or "robots" in e["error"].lower() or "budget" in e["error"].lower() for e in errors)
        for number, row in enumerate(fixed, 1):
            pilot = dict(url=row["url"], title=row["title"], archive_day=row["published_date"], fixed_selection_position=number)
            if terminal:
                pilot.update(status="not_fetched_after_terminal_host_condition")
                pilots.append(pilot)
                continue
            try:
                raw, final, mime = fetch.get(row["url"], "fixed_pilot")
                if admitted_dallas_research_route(final) != row["url"]:
                    raise ValueError("Dallas pilot redirected to a different article identity")
                pilot_path = output_dir / "pilot_html" / (digest(row["url"])[:20] + ".html")
                pilot_path.parent.mkdir(parents=True, exist_ok=True)
                pilot_path.write_bytes(raw)
                parsed = parse_dallas_research_body(raw)
                same_day = parsed["published_date"] == row["published_date"]
                same_title = normalized_title(parsed["title"]) == normalized_title(row["title"])
                inside = START <= parsed["available_by"] <= END
                qualified = same_day and same_title and inside and parsed["body_oil_verified"]
                state_pilot = "accessible_dated_heading_free_oil_prose" if qualified else "date_title_version_or_body_scope_review_required"
                body_path = output_dir / "pilot_bodies" / (digest(row["url"])[:20] + ".json")
                write_json(body_path, parsed)
                pilot.update(status=state_pilot, final_url=final, bytes=len(raw), mime=mime, raw_sha256=digest(raw),
                             raw_path=str(pilot_path), body_json_path=str(body_path), body_sha256=digest(parsed["text"]),
                             body_words=len(words(parsed["text"])), heading_free_words=len(words(parsed["scope_text"])),
                             title_matches_archive=same_title, day_matches_archive=same_day, available_by=parsed["available_by"],
                             modified_date=parsed["modified_date"], body_oil_verified=parsed["body_oil_verified"],
                             authors=parsed["authors"], author_disclaimers=parsed["author_disclaimers"],
                             version_caution=parsed["version_caution"], factual_blocks=len(parsed["blocks"]))
                row.update(pilot_status=state_pilot, body_date_verified=same_day, body_title_verified=same_title,
                           body_oil_verified=parsed["body_oil_verified"], available_by=parsed["available_by"],
                           pilot_raw_sha256=digest(raw), pilot_body_words=pilot["body_words"],
                           pilot_body_sha256=pilot["body_sha256"], authors=parsed["authors"],
                           author_disclaimers=parsed["author_disclaimers"], evidence_type=parsed["evidence_type"],
                           historical_version_status=parsed["version_caution"])
            except Exception as exc:
                pilot.update(status="failed_no_bypass", error=str(exc))
                row.update(pilot_status="failed_no_bypass", pilot_error=str(exc))
                if "403" in str(exc) or "robots" in str(exc).lower() or "budget" in str(exc).lower():
                    terminal = True
            pilots.append(pilot)
            print(f"Dallas research fixed pilot {number}/{len(fixed)}: {pilot['status']} {row['url']}", flush=True)
    finally:
        fetch.close()
    if before != live_hashes() or script_sha != digest(Path(__file__).read_bytes()):
        raise ValueError("Protected live inputs or discovery implementation changed")
    ordered = sorted(records.values(), key=lambda r: (r["published_date"], r["url"]))[:candidate_limit]
    target = output_dir / "source_candidates.jsonl"
    write_jsonl(target, ordered)
    report = dict(at=now(), discovery_only=True, status=state, window=[START, END],
                  admitted_source="DALLAS-RESEARCH-OIL", native_html_families=["economics", "Southwest Economy", "energy indicators"],
                  archive_pages_scanned=len(archives), actual_topic_items_inspected=inspected,
                  new_actual_linked_candidates=len(ordered), candidate_cap=candidate_limit, excluded=dict(excluded),
                  candidate_by_year=dict(sorted(Counter(r["published_date"][:4] for r in ordered).items())),
                  fixed_pilot_urls=[p["url"] for p in pilots], fixed_pilots=pilots, errors=errors, archives=archives,
                  accessible_dated_heading_free_oil_pilots=sum(p["status"] == "accessible_dated_heading_free_oil_prose" for p in pilots),
                  requests=fetch.requests, downloaded_response_bytes_including_robots=fetch.bytes,
                  candidate_file_sha256=digest(target.read_bytes()), script_sha256=script_sha,
                  protected_live_hashes_before_and_after=before, live_inputs_unchanged=True,
                  live_raw_registry_writes=0, corpus_passages_written=0, bulk_collection_run=False,
                  accepted_new_passages=None, estimated_bulk_yield=None, external_model_calls=0, human_labels_created=0,
                  rights="Redistribution review_required. Retain source attribution and authors' non-Fed-policy disclaimer; charts, tables, photographs, linked papers and third-party material are excluded rather than assumed reusable.",
                  limitations=["Actual archive links/title/teaser dates are candidates, not validated documents or accepted passages.",
                               "Only six fixed date-stratified pilots are fetched; failures are not replaced with easier articles.",
                               "Pilot source paragraphs require manual source-quality review before bulk collection/adaptor activation.",
                               "Research authors' models, scenarios, observations and attributed quotations are not interchangeable with official Federal Reserve policy or realized output.",
                               "Current response hashes do not prove unchanged historical text; visible revisions raise availability and unknown-dated notices fail closed.",
                               "Publication versus event/statistical dates, crude versus products, and stocks versus flows require downstream semantic QA."])
    write_json(output_dir / "discovery_report.json", report)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=DATA / "unified_runs/semantic-page-v2-buffer/reports/dallas-research-discovery")
    cli.add_argument("--candidate-limit", type=int, default=100)
    args = cli.parse_args()
    result = discover(args.output_dir, args.candidate_limit)
    print(json.dumps({k: result[k] for k in ["status", "new_actual_linked_candidates", "archive_pages_scanned", "accessible_dated_heading_free_oil_pilots", "candidate_file_sha256", "live_inputs_unchanged"]}, indent=2))
