"""Pure official-source discovery adapters; no network or manifest writes.

Archive functions return actual observed landing-page URLs. Publication parsers
return only linked, first-party PDF candidates. The caller must use Fetcher for
robots/rate/size checks and must restrict extracted prose to the oil sector.
Licences exclude third-party figures, logos and other separately credited work.
"""

import calendar
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


START_DATE = "2018-01-01"
END_DATE = "2026-10-05"
REQ_ARCHIVE = "https://www.industry.gov.au/publications/resources-and-energy-quarterly"
UK_ET_ARCHIVE = "https://www.gov.uk/government/collections/energy-trends"
UK_DUKES_ARCHIVE = "https://www.gov.uk/government/collections/digest-of-uk-energy-statistics-dukes"
WB_ARCHIVE = "https://www.worldbank.org/en/research/commodity-markets/report-archive"
MONTHS = {name.lower(): n for n in range(1, 13)
          for name in (calendar.month_name[n], calendar.month_abbr[n])}
EDITION = re.compile(r"\b([A-Za-z]+)\s+(20\d{2})\b")


def _edition_date(text):
    for month, year in EDITION.findall(text):
        number = MONTHS.get(month.lower())
        if number:
            return f"{year}-{number:02d}-{calendar.monthrange(int(year), number)[1]:02d}"
    return None


def _day(text):
    match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(20\d{2})\b", text)
    if not match or match[2].lower() not in MONTHS:
        return None
    try:
        return datetime(int(match[3]), MONTHS[match[2].lower()], int(match[1])).date().isoformat()
    except ValueError:
        return None


def _in_window(date, start=START_DATE, end=END_DATE):
    return bool(date and start <= date <= end)


def _same_host(url, hosts):
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in hosts and not parsed.username and not parsed.password


def _metadata(raw):
    soup = BeautifulSoup(raw, "html.parser")
    main = soup.find("main") or soup
    title = main.find("h1")
    title = title.get_text(" ", strip=True) if title else ""
    return soup, main, title


def _candidate(url, title, kind, available, origin, precision, licence, **extra):
    if not _in_window(available):
        return None
    return dict(url=url, title=title, source=kind,
                published_date=available if precision == "day" else None,
                available_by=available, date_precision=precision, date_source=origin,
                event_date=None, event_status="unlabelled", redistribution=licence,
                content_scope="oil_narratives_only", requires_pdf=extra.get("candidate_type") != "landing", **extra)


def req_archive_pages(raw, origin=REQ_ARCHIVE):
    """Observed public edition landing pages; dates refined on each landing."""
    _, main, _ = _metadata(raw)
    rows = {}
    for anchor in main.select("a[href]"):
        title = anchor.get_text(" ", strip=True)
        url = urljoin(origin, anchor["href"])
        path = urlparse(url).path
        if not title.lower().startswith("resources and energy quarterly") or not _edition_date(title):
            continue
        if not _same_host(url, {"www.industry.gov.au"}) or not (
                path.startswith("/publications/resources-and-energy-quarterly-") or re.fullmatch(r"/node/\d+", path)):
            continue
        row = _candidate(url, title, "AU-REQ", _edition_date(title), origin,
                         "month_upper_bound", "cc_by_4_attribution_with_exclusions", candidate_type="landing")
        if row:
            rows[url] = row
    return list(rows.values())


def req_publication_sources(raw, origin):
    """Only full-report PDF hyperlinks on a verified REQ edition page."""
    _, main, title = _metadata(raw)
    if not title.lower().startswith("resources and energy quarterly"):
        return []
    date_field = main.select_one(".field--name-field-date .field__item")
    exact = _day(date_field.get_text(" ", strip=True)) if date_field else None
    available = exact or _edition_date(title)
    rows = {}
    for anchor in main.select("a[href]"):
        url = urljoin(origin, anchor["href"])
        filename = urlparse(url).path.rsplit("/", 1)[-1].lower()
        if not _same_host(url, {"www.industry.gov.au"}) or not re.fullmatch(
                r"resources-and-energy-quarterly-(?:march|june|september|december)-20\d{2}\.pdf", filename):
            continue
        size_node = anchor.select_one(".file-size")
        size_text = size_node.get_text(" ", strip=True) if size_node else anchor.get_text(" ", strip=True)
        size = re.search(r"([\d,.]+)\s*(MB|KB)\b", size_text, re.I)
        if size and float(size[1].replace(",", "")) * (1024 * 1024 if size[2].upper() == "MB" else 1024) > 25 * 1024 * 1024:
            continue
        # The edition in the linked filename must match the landing title.
        if _edition_date(filename.replace("-", " ")) != _edition_date(title):
            continue
        row = _candidate(url, title, "AU-REQ", available, origin,
                         "day" if exact else "month_upper_bound", "cc_by_4_attribution_with_exclusions",
                         publication_date_quote=date_field.get_text(" ", strip=True) if date_field else None)
        if row:
            rows[url] = row
    return list(rows.values())


def uk_energy_trends_pages(raw, origin=UK_ET_ARCHIVE):
    """Observed edition links, excluding press releases and dataset tables."""
    _, main, _ = _metadata(raw)
    rows = {}
    for anchor in main.select("a[href]"):
        title = anchor.get_text(" ", strip=True)
        url = urljoin(origin, anchor["href"])
        if not re.match(r"^Energy\s+Trends\s*:\s*[A-Za-z]+\s+20\d{2}$", title, re.I):
            continue
        if not _same_host(url, {"www.gov.uk"}) or not urlparse(url).path.startswith("/government/statistics/energy-trends-"):
            continue
        item = anchor.find_parent("li")
        time = item.find("time", attrs={"datetime": True}) if item else None
        exact = time["datetime"][:10] if time and re.match(r"20\d{2}-\d{2}-\d{2}", time["datetime"]) else None
        if exact:
            try:
                datetime.strptime(exact, "%Y-%m-%d")
            except ValueError:
                exact = None
        row = _candidate(url, title, "UK-ENERGY-TRENDS", exact or _edition_date(title), origin,
                         "day" if exact else "month_upper_bound", "OGL_with_exclusions", candidate_type="landing")
        if row:
            rows[url] = row
    return list(rows.values())


def uk_energy_trends_sources(raw, origin):
    """Actual official assets PDFs; preserve visible revision-date provenance."""
    return _uk_publication_sources(raw, origin, "UK-ENERGY-TRENDS")


def uk_dukes_pages(raw, origin=UK_DUKES_ARCHIVE):
    """Observed annual edition pages, not frequently replaced current chapters."""
    _, main, _ = _metadata(raw)
    rows = {}
    for anchor in main.select("a[href]"):
        title = anchor.get_text(" ", strip=True)
        match = re.fullmatch(r"Digest of UK Energy Statistics \(DUKES\) (20\d{2})", title, re.I)
        url = urljoin(origin, anchor["href"])
        if not match or not _same_host(url, {"www.gov.uk"}):
            continue
        if not re.fullmatch(r"/government/statistics/digest-of-uk-energy-statistics-dukes-20\d{2}", urlparse(url).path):
            continue
        # Do not invent the release month/day from the statistical reporting year.
        item = anchor.find_parent("li")
        time = item.find("time", attrs={"datetime": True}) if item else None
        exact = time["datetime"][:10] if time and re.fullmatch(r"20\d{2}-\d{2}-\d{2}.*", time["datetime"]) else None
        row = _candidate(url, title, "UK-DUKES", exact or match[1] + "-12-31", origin,
                         "day" if exact else "year_upper_bound", "OGL_with_exclusions", candidate_type="landing")
        if row:
            rows[url] = row
    return list(rows.values())


def uk_dukes_sources(raw, origin):
    return _uk_publication_sources(raw, origin, "UK-DUKES")


def _uk_publication_sources(raw, origin, kind):
    soup, main, title = _metadata(raw)
    pattern = (r"^Digest of UK Energy Statistics \(DUKES\) 20\d{2}$" if kind == "UK-DUKES"
               else r"^Energy\s+Trends\s*:\s*[A-Za-z]+\s+20\d{2}$")
    if not re.match(pattern, title, re.I):
        return []
    published = None
    updated = None
    for term in main.select(".gem-c-metadata__term"):
        value = term.find_next_sibling("dd")
        text = value.get_text(" ", strip=True) if value else ""
        if "Published" in term.get_text():
            published = _day(text)
        elif "updated" in term.get_text().lower():
            updated = _day(text)
    # GOV.UK also exposes updates in the published-dates footer, outside main.
    footer = soup.select_one(".gem-c-published-dates")
    if footer:
        match = re.search(r"Last updated\s+(\d{1,2}\s+[A-Za-z]+\s+20\d{2})", footer.get_text(" ", strip=True), re.I)
        if match:
            updated = _day(match[1])
    available = max(date for date in (published, updated) if date) if published or updated else _edition_date(title)
    rows = {}
    for section in main.select(".gem-c-attachment"):
        anchor = section.select_one(".gem-c-attachment__title a[href]")
        if not anchor:
            continue
        if kind == "UK-DUKES" and re.search(r"\b(annex(?:es)?|appendic(?:es|e))\b", anchor.get_text(" ", strip=True), re.I):
            continue
        url = urljoin(origin, anchor["href"])
        filename = urlparse(url).path.rsplit("/", 1)[-1].lower()
        if not _same_host(url, {"assets.publishing.service.gov.uk", "www.gov.uk"}) or not filename.endswith(".pdf"):
            continue
        recognised = ("dukes" in filename or "oil" in filename or "petroleum" in filename) if kind == "UK-DUKES" else ("energy_trends" in filename or "energy-trends" in filename or "oil" in filename)
        if not recognised:
            continue
        size = re.search(r"([\d,.]+)\s*(MB|KB)\b", section.get_text(" ", strip=True), re.I)
        if size and float(size[1].replace(",", "")) * (1024 * 1024 if size[2].upper() == "MB" else 1024) > 25 * 1024 * 1024:
            continue
        row = _candidate(url, title, kind, available, origin,
                         "day" if published or updated else "month_upper_bound", "OGL_with_exclusions",
                         original_publication_date=published, visible_revision_date=updated,
                         historical_attachment_version="not_independently_verified")
        if row:
            rows[url] = row
    return list(rows.values())


def discover(fetch, families=None):
    """Use a caller-owned Fetcher/RecordingFetcher, strictly serially.

    Only return PDF source rows; caller owns snapshot/manifest/corpus writes.
    A failed page is printed and skipped rather than retried with another URL
    or collection identity. Both publication and archive HTML provenance will
    be captured by RecordingFetcher when provided by the caller.
    """
    selected = set(families or ("AU-REQ", "UK-ENERGY-TRENDS", "UK-DUKES", "WB-CMO"))
    series = (("AU-REQ", REQ_ARCHIVE, req_archive_pages, req_publication_sources),
              ("UK-ENERGY-TRENDS", UK_ET_ARCHIVE, uk_energy_trends_pages, uk_energy_trends_sources),
              ("UK-DUKES", UK_DUKES_ARCHIVE, uk_dukes_pages, uk_dukes_sources))
    rows = {}
    for kind, archive, page_parser, source_parser in series:
        if kind not in selected:
            continue
        try:
            raw, final, _ = fetch.get(archive)
            pages = page_parser(raw, final)
        except Exception as error:
            print(f"Skipped official archive {archive}: {error}", flush=True)
            continue
        print(f"Observed {len(pages)} {kind} edition pages", flush=True)
        for page in pages:
            try:
                raw, final, _ = fetch.get(page["url"])
                found = source_parser(raw, final)
                for row in found:
                    row["archive_url"] = archive
                    rows.setdefault(row["url"], row)
                print(f"Discovered {len(found)} PDF candidates from {final}", flush=True)
            except Exception as error:
                print(f"Skipped edition {page['url']}: {error}", flush=True)
    if "WB-CMO" in selected:
        try:
            raw, final, _ = fetch.get(WB_ARCHIVE)
            for row in worldbank_special_oil_sources(raw, final):
                rows.setdefault(row["url"], row)
        except Exception as error:
            print(f"Skipped official archive {WB_ARCHIVE}: {error}", flush=True)
    return list(rows.values())


def worldbank_special_oil_sources(raw, origin=WB_ARCHIVE):
    """Direct, explicitly oil-focused thematic PDFs in CMO archive rows."""
    soup = BeautifulSoup(raw, "html.parser")
    rows = {}
    for tr in soup.select("tr"):
        match = re.match(r"(20\d{2})\s*-\s*([A-Za-z]+)", tr.get_text(" ", strip=True))
        if not match or match[2].lower() not in MONTHS:
            continue
        available = _edition_date(f"{match[2]} {match[1]}")
        for anchor in tr.select("a[href]"):
            title = anchor.get_text(" ", strip=True)
            url = urljoin(origin, anchor["href"])
            if not re.search(r"\boil\b", title, re.I) or not _same_host(url, {"thedocs.worldbank.org", "documents1.worldbank.org"}):
                continue
            if not urlparse(url).path.lower().endswith(".pdf"):
                continue
            row = _candidate(url, f"World Bank CMO {match[1]}-{MONTHS[match[2].lower()]:02d}: {title}",
                             "WB-CMO", available, origin, "month_upper_bound", "review_required")
            if row:
                rows[url] = row
    return list(rows.values())
