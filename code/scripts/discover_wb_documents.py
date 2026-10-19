"""Two independently observed public World Bank document-landing adapters.

These URLs were returned by official-site research, NOT fabricated from PDF
paths and NOT substitutes for denied or blocked Open Knowledge requests.
Keep the original collection failure records. Only MAIN DOCUMENT hyperlinks
from a successful public HTML page become new PDF candidates.
"""

import calendar
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


ARCHIVE = "https://www.worldbank.org/en/research/commodity-markets/report-archive"
OBSERVED_LANDINGS = (
    ("https://thedocs.worldbank.org/en/doc/1b388949805c9a0ae3736bdacb32ea94-0050012025/commodity-markets-outlook-april-2025", 2025, 4),
    ("https://thedocs.worldbank.org/en/doc/55ed8b04df8228fea1607a97b5561f73-0050012025/", 2025, 10),
)
_PDF_HOSTS = {"thedocs.worldbank.org", "documents1.worldbank.org"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def _first_party(url):
    parsed = urlparse(url)
    return (parsed.scheme == "https" and parsed.hostname in _PDF_HOSTS
            and not parsed.username and not parsed.password)


def document_sources(raw, origin, year, month):
    """Parse actual MAIN DOCUMENT anchors; ignore RELATED files and webpage dates."""
    if not _first_party(origin) or not 1 <= month <= 12:
        return []
    available = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
    if not "2018-01-01" <= available <= "2026-10-05":
        return []
    soup = BeautifulSoup(raw, "html.parser")
    marker = soup.find(string=re.compile(r"^\s*MAIN DOCUMENT\s*$", re.I))
    if marker is None:
        return []
    heading = marker.find_parent(list(_HEADINGS)) or marker.parent
    if heading is None:
        return []
    expected_filename = f"CMO-{calendar.month_name[month]}-{year}.pdf".lower()
    rows = {}
    for node in heading.next_elements:
        name = getattr(node, "name", None)
        if name in _HEADINGS:
            break
        if name != "a" or not node.get("href"):
            continue
        url = urljoin(origin, node["href"])
        filename = urlparse(url).path.rsplit("/", 1)[-1].lower()
        label = node.get_text(" ", strip=True)
        identity_matches = filename == expected_filename or (
            urlparse(url).hostname == "documents1.worldbank.org" and label.lower() == expected_filename)
        if not _first_party(url) or not filename.endswith(".pdf") or not identity_matches:
            continue
        title = f"World Bank Commodity Markets Outlook {year}-{month:02d}"
        rows[url] = dict(
            url=url, title=title, source="WB-CMO", published_date=None,
            available_by=available, date_precision="month_upper_bound", date_source=ARCHIVE,
            event_date=None, event_status="unlabelled", redistribution="review_required",
            requires_pdf=True, content_scope="oil_narratives_only", archive_url=ARCHIVE,
            document_landing_url=origin, observed_download_label=label,
            discovery_provenance="Independently observed official document landing; literal MAIN DOCUMENT hyperlink",
            date_caution="Archive issue month only; landing metadata not used as an exact publication day",
        )
    return list(rows.values())


def discover(fetch):
    """Serially read only the two observed theDocs pages using caller's Fetcher.

    No HTML contributes passages and no PDF is downloaded here. Save raw HTML
    and checksums in this run's discovery directory, never source manifests.
    A caller RecordingFetcher may additionally keep its general request log.
    """
    from oil_evidence.store import DATA, append_jsonl, digest, now

    root = DATA / "expansion_runs/official-2018-2026-v2/discovery/wb-documents"
    root.mkdir(parents=True, exist_ok=True)
    rows = {}
    for url, year, month in OBSERVED_LANDINGS:
        try:
            raw, final, content_type = fetch.get(url)
            if not _first_party(final) or raw.startswith(b"%PDF") or "html" not in content_type.lower():
                raise ValueError("Expected a public first-party HTML document landing")
            path = root / (digest(raw) + ".html")
            if not path.exists():
                path.write_bytes(raw)
            found = document_sources(raw, final, year, month)
            append_jsonl(root / "discovery.jsonl", dict(at=now(), url=url, final_url=final,
                raw_path=str(path), raw_sha256=digest(raw), status="ok", candidates=len(found)))
            for row in found:
                rows.setdefault(row["url"], row)
            print(f"Discovered {len(found)} PDF candidates from observed World Bank landing {final}", flush=True)
        except Exception as error:
            append_jsonl(root / "discovery.jsonl", dict(at=now(), url=url, status="failed", error=str(error)))
            print(f"Skipped observed World Bank landing {url}: {error}", flush=True)
    return list(rows.values())
