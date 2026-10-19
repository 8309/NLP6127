"""Read-only discovery adapters for dated, official US oil evidence.

This module never writes sources, documents, corpus, labels, or model outputs.
The caller owns serial collection and raw/archive audit persistence.  PDF dates
are conservative month-end bounds observed on the linked report covers, not
calendar years in titles, signature days, or unverified HTTP modification dates.
"""
from __future__ import annotations

import calendar
import re
import sys
from datetime import date
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup

from oil_evidence.corpus import Fetcher
from oil_evidence.store import config

BSEE_ARCHIVE = "https://www.bsee.gov/resources-tools/planning-preparedness/hurricane/hurricane-history"
DOE_ARCHIVE = "https://www.energy.gov/hgeo/opr/spr-reports"
BOEM_ARCHIVE = "https://www.boem.gov/oil-gas-energy/energy-economics/oil-and-gas-economic-modeling"

RIGHTS = {
    "DOE-SPR": "https://www.energy.gov/web-policies",
    "BOEM-POLICY": "https://www.boem.gov/about-boem/copyright-restrictions-and-permissions",
    # Do not automatically treat all third-party images or contractor text as
    # government-authored; the pipeline's report-specific review is retained.
    "BSEE-HURRICANE": "https://www.bsee.gov/",
}

# These are actual links observed in DOE's SPR Reports archive or its linked
# attachment page. The text after the month is an evidence label, not a day.
DOE_REPORTS = {
    2016: {
        "url": "https://www.energy.gov/sites/prod/files/2019/02/f59/EXEC-2017-005106%20-%202016%20SRP%20Report.pdf",
        "cover_month": "2018-10", "cover_quote": "October 2018",
    },
    2017: {
        "url": "https://www.energy.gov/sites/prod/files/2019/02/f59/EXEC-2018-001277%20-%202017%20SPR%20Report.pdf",
        "cover_month": "2018-12", "cover_quote": "December 2018",
    },
    2018: {
        "url": "https://www.energy.gov/ceser/articles/exec-2019-001076-2018-spr-report",
        "cover_month": "2020-01", "cover_quote": "January 2020",
    },
    2019: {
        "url": "https://www.energy.gov/sites/default/files/2021-12/EXEC-2020-001011%20-%20S1%20Granholm%20signed%20%202019%20SPR%20Annual%20Report%20-.pdf",
        "archive_link": "https://www.energy.gov/ceser/exec-2020-001011-2019-spr-report",
        "cover_month": "2021-05", "cover_quote": "May 2021",
    },
    2020: {
        "url": "https://www.energy.gov/sites/default/files/2022-11/EXEC-2021-004325%20-%202020%20SPR%20Annual%20Report%20to%20Congress%20-%20August%202022.pdf",
        "archive_link": "https://www.energy.gov/ceser/exec-2021-004325-2020-annual-report",
        # The filename says August; the physical cover says September.
        "cover_month": "2022-09", "cover_quote": "September 2022",
    },
    2021: {
        "url": "https://www.energy.gov/ceser/articles/spr-2021-annual-report-congress",
        "cover_month": "2023-11", "cover_quote": "November 2023",
    },
    2022: {
        "url": "https://www.energy.gov/sites/default/files/2023-12/SPR%202022%20Annual%20Report%20to%20Congress.pdf",
        "cover_month": "2023-10", "cover_quote": "October 2023",
    },
    2023: {
        "url": "https://www.energy.gov/documents/spr-2023-annual-report-congress",
        "cover_month": "2026-07", "cover_quote": "July 2026",
    },
}

# Only the smaller methodology reports, not the 36--49 MB full programs.
BOEM_REPORTS = {
    "https://www.boem.gov/oil-gas-energy/leasing/final-economic-analysis-methodology-2024-2029-program": {
        "cover_month": "2023-09", "cover_quote": "September 2023",
        "title": "BOEM Final Economic Analysis Methodology for the 2024–2029 National OCS Oil and Gas Leasing Program",
    },
    "https://www.boem.gov/sites/default/files/documents/oil-gas-energy/national-program/Draft_Economic_Analysis_Methdology_2023-2028ProposeProgram_July2022.pdf": {
        "archive_link": "https://www.boem.gov/oil-gas-energy/national-program/draft-economic-analysis-methodology-2023-2028-propose-program",
        "cover_month": "2022-07", "cover_quote": "July 2022",
        "title": "BOEM Draft Economic Analysis Methodology for the 2023–2028 National OCS Oil and Gas Leasing Program",
    },
}


def month_end(month: str) -> str:
    """Reject invalid months instead of fabricating a date."""
    year, number = (int(part) for part in month.split("-"))
    return date(year, number, calendar.monthrange(year, number)[1]).isoformat()


def upload_month(url: str) -> str | None:
    path = unquote(urlparse(url).path)
    match = re.search(r"/files/(20\d{2})[-/](0[1-9]|1[0-2])(?:/|$)", path)
    return f"{match[1]}-{match[2]}" if match else None


def day_from_archive(text: str) -> str | None:
    months = {name.lower(): number for number in range(1, 13)
              for name in (calendar.month_name[number], calendar.month_abbr[number])}
    months["sept"] = 9
    for match in re.finditer(r"\b([A-Za-z]+)\.?\s+(\d{1,2}),\s*(20\d{2})\b", text):
        month = months.get(match[1].lower())
        if month:
            try:
                return date(int(match[3]), month, int(match[2])).isoformat()
            except ValueError:
                continue
    return None


def _source(url, title, source, available, origin, precision, evidence):
    settings = config()["corpus"]
    if not settings["start_date"] <= available <= settings["end_date"]:
        return None
    return dict(url=url, title=title, source=source,
                published_date=available if precision == "day" else None,
                available_by=available, date_precision=precision,
                date_source=origin, date_evidence=evidence,
                event_date=None, event_status="unlabelled",
                redistribution="review_required", rights_source=RIGHTS[source])


def bsee_sources(raw, origin=BSEE_ARCHIVE):
    """Keep dated storm-operation updates only; ignore navigation and warnings."""
    soup = BeautifulSoup(raw, "html.parser")
    rows, conflicts = {}, set()
    for anchor in soup.select("a[href]"):
        title = anchor.get_text(" ", strip=True)
        href = urljoin(origin, anchor["href"])
        parsed = urlparse(href)
        if parsed.hostname != "www.bsee.gov":
            continue
        if not (parsed.path.startswith("/newsroom/") or
                parsed.path.lower().endswith(".pdf")):
            continue
        if not re.search(r"activity|monitors|statistics|final\s+(?:report|update)", title, re.I):
            continue
        available = day_from_archive(title)
        date_quote = title
        if not available:
            # Some entries put the date before the anchor in the list item.
            parent = anchor.find_parent("li")
            date_quote = parent.get_text(" ", strip=True) if parent else title
            available = day_from_archive(date_quote)
        if not available:
            continue
        row = _source(href, title, "BSEE-HURRICANE", available, origin, "day",
                      {"kind": "official_archive_day", "quote": date_quote})
        if row:
            if href in rows and rows[href]["available_by"] != available:
                conflicts.add(href)  # A reused URL cannot be a frozen dated version.
            rows[href] = row
    return [row for href, row in rows.items() if href not in conflicts]


def _dated_pdf(url, title, source, origin, report):
    cover = report["cover_month"]
    upload = upload_month(url)
    conservative = max(cover, upload) if upload else cover
    evidence = {"kind": "observed_cover_month_conservative",
                "cover_quote": report["cover_quote"], "cover_month": cover,
                "upload_directory_month": upload,
                "policy": "Month end of cover publication month or later upload directory; no signature-day inference."}
    row = _source(url, title, source, month_end(conservative), origin,
                  "month_upper_bound", evidence)
    if row:
        row["requires_pdf"] = True
    return row


def doe_spr_sources(raw, origin=DOE_ARCHIVE):
    """Require the current official archive to contain each verified report link.

    Attachment-page routes for 2019/2020 are resolved only to previously observed
    attachments; unknown editions fail closed pending a cover-date check.
    """
    soup = BeautifulSoup(raw, "html.parser")
    rows = {}
    for anchor in soup.select("a[href]"):
        title = anchor.get_text(" ", strip=True)
        match = re.fullmatch(r"Calendar\s+Year\s+(20\d{2})", title, re.I)
        if not match:
            continue
        year = int(match[1])
        report = DOE_REPORTS.get(year)
        if not report:
            continue
        actual = urljoin(origin, anchor["href"])
        if unquote(actual) != unquote(report.get("archive_link", report["url"])):
            continue
        row = _dated_pdf(report["url"],
                         f"DOE Strategic Petroleum Reserve Annual Report for Calendar Year {year}",
                         "DOE-SPR", origin, report)
        if row:
            row["reporting_year"] = year
            row["observed_archive_link"] = actual
            rows[row["url"]] = row
    return list(rows.values())


def boem_small_sources(raw, origin=BOEM_ARCHIVE):
    """Two observed methodology attachments; downloads still obey the 25 MB cap."""
    soup = BeautifulSoup(raw, "html.parser")
    observed = {unquote(urljoin(origin, anchor["href"]))
                for anchor in soup.select("a[href]")}
    rows = []
    for url, report in BOEM_REPORTS.items():
        if unquote(report.get("archive_link", url)) not in observed:
            continue
        row = _dated_pdf(url, report["title"], "BOEM-POLICY", origin, report)
        if row:
            row["observed_archive_link"] = report.get("archive_link", url)
            rows.append(row)
    return rows


def archive_pages(include_boem=True):
    pages = [(BSEE_ARCHIVE, bsee_sources), (DOE_ARCHIVE, doe_spr_sources)]
    if include_boem:
        pages.append((BOEM_ARCHIVE, boem_small_sources))
    return pages


def discover(fetch: Fetcher, include_boem=True):
    """Fetch archive HTML only, using the common robots/cap/rate-limited client.

    The caller supplies its shared RecordingFetcher for archive audit persistence.
    No private client, report downloads, corpus writes, labels, or model calls.
    """
    rows = {}
    for url, parser in archive_pages(include_boem):
        try:
            raw, final, _ = fetch.get(url)
            rows.update((row["url"], row) for row in parser(raw, final))
        except Exception as error:
            # The supplied RecordingFetcher owns structured failure persistence.
            # Do not retry 403 through an alternate hostname or unobserved route.
            print(f"US oil archive skipped: {url}: {error}", file=sys.stderr, flush=True)
    return list(rows.values())
