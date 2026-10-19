"""Bounded BOEM oil-leasing EIS discovery and PDF pilots, no corpus writes.

Follow actual official index/document links, retain full physical page order,
and fail closed on old report covers or later PDF timestamps. A linked EIS,
its page count and preliminary oil-page screening are never accepted passages.
"""
from __future__ import annotations

import argparse
import calendar
from collections import Counter
from datetime import date, datetime, timezone
import io
import json
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup
import httpx
from pypdf import PdfReader

from oil_evidence.corpus import Fetcher, date_from_text, has_table_heading, is_prose_page, is_reference_page
from oil_evidence.segmentation import oil_scope_supported
from oil_evidence.store import DATA, digest, now, read_jsonl, words, write_json, write_jsonl


START, END = "2018-01-01", "2026-10-05"
SEEDS = [
    "https://www.boem.gov/oil-gas-energy/national-program/2024-2029-ocs-oil-and-gas-leasing-program",
    "https://www.boem.gov/environment/environmental-assessment/nepa-activities-gulf-america",
    "https://www.boem.gov/environment/environmental-assessment/gulf-america-regional-ocs-oil-and-gas-programmatic",
    "https://www.boem.gov/oil-gas-energy/national-program/national-ocs-oil-and-gas-leasing-program",
]
OIL_TITLE = re.compile(r"\b(?:oil\s*(?:and|&)\s*gas|oil\s*[-–]\s*gas|oil\s+and\s+gas|lease\s+sale|national\s+ocs|gom|gulf\s+of\s+(?:mexico|america))\b", re.I)
DOCUMENT = re.compile(r"environmental impact|programmatic\s+eis|supplemental\s+eis|\b(?:PEIS|SEIS|EIS)\b|record of decision|analysis\s*(?:and|&)\s*proposal|proposed\s+(?:final\s+)?program|\b(?:PFP|DPP)\b", re.I)
EXCLUDE = re.compile(r"renewable energy leasing|geological and geophysical|appendic(?:es|es)|appendix|references|wind|keyword index|volume\s+(?:ii|iii|2|3)\b", re.I)
MONTH_NAMES = "|".join(calendar.month_name[1:])
MONTH_YEAR = re.compile(rf"\b({MONTH_NAMES})\s+(20\d{{2}})\b", re.I)
# This is not a generic OCR/date bypass. Root inspected this exact publisher
# PDF's physical cover and rendering; new graphic covers need separate review.
APPROVED_VISUAL_COVERS = {
    "fb769e848a17ed9472f083465bb7568e16d17a4ecdf063b23cb8a30f3739229a": {
        "machine_cover_month": "2022-07", "physical_page": 1,
        "rendered_png_sha256": "de245359679c0577dca6887478df7c5d11547c66f222353c8bcdf2efd5eca76b",
    }
}


class BoemFetcher(Fetcher):
    """Keep inherited robots/caps/rates but never multiply redirect retries."""
    def __init__(self):
        super().__init__()
        self.requested_urls = []

    def get(self, url, _redirects=0):
        if not official(url):
            raise ValueError("Every BOEM redirect target must retain its official HTTPS host")
        self.requested_urls.append(dict(url=url, redirect_depth=_redirects))
        try:
            return super().get(url, _redirects)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if _redirects:
                # Super's target request exhausted its at-most-three attempts.
                # Its caller must not retry the alias and restart that budget.
                raise RuntimeError(f"BOEM redirect target exhausted bounded transient attempts: {url}: {exc}") from exc
            raise


def official(url):
    p = urlparse(url)
    return p.scheme == "https" and p.hostname in {"www.boem.gov", "boem.gov"} and not p.username and not p.password and p.port in {None, 443}


def canonical(url):
    return url.split("#", 1)[0].rstrip("/")


def archive_date(context):
    exact = date_from_text(context)
    if exact:
        return exact, "day", exact
    matches = list(MONTH_YEAR.finditer(context))
    values = {f"{m[2]}-{list(calendar.month_name).index(m[1].capitalize()):02d}" for m in matches}
    if len(values) == 1:
        month = values.pop()
        year, number = map(int, month.split("-"))
        return date(year, number, calendar.monthrange(year, number)[1]).isoformat(), "month_upper_bound", None
    return None, None, None


def parse_index(raw, origin, known=()):
    soup = BeautifulSoup(raw, "html.parser")
    main = soup.select_one("main")
    if main is None:
        raise ValueError("BOEM official index has no main region")
    heading = main.select_one("h1")
    page_title = heading.get_text(" ", strip=True) if heading else ""
    rows, excluded, observed = {}, Counter(), []
    for a in main.select("a[href]"):
        href = urljoin(origin, a["href"])
        title = a.get_text(" ", strip=True)
        if not official(href) or href in SEEDS:
            continue
        context_node = a.find_parent("tr") or a.find_parent("li") or a.find_parent("p") or a.parent
        context = context_node.get_text(" ", strip=True)
        if len(context) > 1200:
            excluded["ambiguous_wide_context"] += 1
            continue
        if not (DOCUMENT.search(context) and (OIL_TITLE.search(context) or OIL_TITLE.search(page_title))):
            continue
        if EXCLUDE.search(title) or re.search(r"\b(?:volume\s+(?:ii|iii|2|3)|appendices|appendix|renewable|wind)\b", context, re.I):
            # A table row mentioning Volume II also contains a valid Volume I;
            # exclude only an actually non-oil row or this anchor's appendices.
            if EXCLUDE.search(title) or re.search(r"renewable energy leasing|wind|geological and geophysical", context, re.I):
                excluded["non_oil_or_appendix_document"] += 1
                continue
        if re.search(r"notice|federal register|methodology|call for|request for|comment|presentation|agenda|summary", title, re.I):
            excluded["notice_or_already_covered_methodology"] += 1
            continue
        available, precision, exact = archive_date(context)
        descriptive = (page_title + ": " + context) if page_title and not OIL_TITLE.search(context) else context
        item = dict(url=canonical(href), source="BOEM-EIS", title=descriptive if len(descriptive) <= 350 else title,
                    linked_title=title, published_date=exact, available_by=available,
                    date_precision=precision, date_source=origin, event_date=None, event_status="unlabelled",
                    redistribution="review_required", rights_source="https://www.boem.gov/about-boem/copyright-restrictions-and-permissions",
                    requires_pdf=True, content_scope="oil_leasing_policy_and_activity_narratives_only",
                    discovery_method="actually_linked_official_oil_leasing_or_nepa_index",
                    date_evidence=context, pilot_status="not_fetched", body_date_verified=False)
        observed.append(item)
        if not available:
            excluded["unknown_or_ambiguous_archive_publication"] += 1
            continue
        if not START <= available <= END:
            excluded["outside_window"] += 1
            continue
        if canonical(href) in known:
            excluded["already_registered"] += 1
            continue
        old = rows.get(item["url"])
        if old is None or (old["date_precision"] != "day" and precision == "day"):
            rows[item["url"]] = item
    return list(rows.values()), dict(excluded), observed


def linked_pdf(raw, origin):
    soup = BeautifulSoup(raw, "html.parser")
    main = soup.select_one("main")
    if main is None:
        raise ValueError("BOEM attachment page has no main region")
    links = {urljoin(origin, a["href"]) for a in main.select("a[href]")
             if official(urljoin(origin, a["href"])) and urlparse(a["href"]).path.lower().endswith(".pdf")}
    if len(links) != 1:
        raise ValueError("BOEM attachment page must expose exactly one actual PDF link")
    return links.pop()


def pdf_timestamp(value):
    if not isinstance(value, str):
        return None
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", value)
    if not match:
        return None
    try:
        return date(*map(int, match.groups())).isoformat()
    except ValueError:
        return None


def cover_month(pages):
    for page in pages[:2]:
        values = set()
        # Cover dates occupy a complete line. Do not infer dates from program
        # years, BOEM report numbers, contents, signatures or historical prose.
        for line in page.splitlines():
            match = re.fullmatch(rf"\s*({MONTH_NAMES})\s+(20\d{{2}})\s*", line, re.I)
            if match:
                values.add(f"{match[2]}-{list(calendar.month_name).index(match[1].capitalize()):02d}")
        # The observed 2025 title verso places its date in a right-hand column
        # on the publisher's line. Preserve that native evidence, not a date
        # found in report prose, an author block, program years or a signature.
        publisher = re.search(r"(?im)^\s*Published\s+by\s*$", page)
        if publisher:
            for line in page[publisher.end():].splitlines():
                match = re.fullmatch(
                    rf"\s*Bureau\s+of\s+Ocean\s+Energy\s+Management\s+({MONTH_NAMES})\s+(20\d{{2}})\s*",
                    line, re.I)
                if match:
                    values.add(f"{match[2]}-{list(calendar.month_name).index(match[1].capitalize()):02d}")
        if len(values) == 1:
            return values.pop()
        if len(values) > 1:
            raise ValueError("BOEM cover has conflicting publication months")
    raise ValueError("BOEM first two physical pages lack a standalone publication month")


def validated_visual_cover_month(raw, supplied):
    if not isinstance(supplied, dict):
        raise ValueError("BOEM visual cover validation must be a SHA-bound mapping")
    raw_sha = digest(raw)
    expected = APPROVED_VISUAL_COVERS.get(raw_sha)
    if expected is None or supplied.get("raw_sha256") != raw_sha:
        raise ValueError("BOEM graphic cover raw SHA is not separately root reviewed")
    if (type(supplied.get("physical_page")) is not int or supplied["physical_page"] != 1
            or supplied.get("machine_cover_month") != expected["machine_cover_month"]
            or supplied.get("rendered_png_sha256") != expected["rendered_png_sha256"]):
        raise ValueError("BOEM graphic cover month/page/render binding differs from the root-reviewed source")
    rendering = supplied.get("rendered_png_path")
    if not isinstance(rendering, str):
        raise ValueError("BOEM graphic cover must retain its reviewed rendering path")
    path = Path(rendering).resolve()
    if not path.is_relative_to((DATA / "unified_runs").resolve()) or not path.is_file():
        raise ValueError("BOEM graphic cover rendering must be a preserved unified-run audit artifact")
    if digest(path.read_bytes()) != expected["rendered_png_sha256"]:
        raise ValueError("BOEM graphic cover rendering bytes changed after review")
    return expected["machine_cover_month"]


def parse_boem_pdf(raw, source):
    """Return pages in original physical order; no synthetic facts/page masks."""
    if not raw.startswith(b"%PDF"):
        raise ValueError("BOEM PDF source returned non-PDF bytes")
    if len(raw) > 25 * 1024**2:
        raise ValueError("BOEM PDF exceeds the 25 MB document limit")
    reader = PdfReader(io.BytesIO(raw))
    pages = [page.extract_text() or "" for page in reader.pages]
    cover_method = "standalone_cover_text_layer"
    try:
        month = cover_month(pages)
    except ValueError as exc:
        if ("lack a standalone publication month" not in str(exc)
                or source.get("visual_cover_validation") is None):
            raise
        month = validated_visual_cover_month(raw, source["visual_cover_validation"])
        cover_method = "exact_raw_sha_root_confirmed_codex_visual_cover_sidecar_not_human"
    year, number = map(int, month.split("-"))
    upper = date(year, number, calendar.monthrange(year, number)[1]).isoformat()
    available = source.get("indexed_available_by") or source.get("available_by")
    if not available or date.fromisoformat(available).isoformat() != available or not START <= available <= END:
        raise ValueError("BOEM requires an official indexed availability date within the fixed corpus window")
    if not START[:7] <= month <= END[:7]:
        raise ValueError("BOEM report cover is outside the corpus window, even if a later archive title mentions 2018+")
    if month > available[:7]:
        raise ValueError("BOEM cover is later than its alleged indexed publication")
    metadata = reader.metadata or {}
    created, modified = pdf_timestamp(metadata.get("/CreationDate")), pdf_timestamp(metadata.get("/ModDate"))
    timestamps = [d for d in (created, modified) if d]
    if not timestamps:
        raise ValueError("BOEM PDF timestamps missing; current attachment's historical version cannot be checked")
    indexed_available = available
    available = max([indexed_available] + timestamps)
    if available > END:
        raise ValueError("BOEM current PDF version timestamp is later than the fixed corpus cutoff")
    adjusted = available > indexed_available
    candidates = []
    for physical, text in enumerate(pages, 1):
        if (oil_scope_supported(text) and len(words(text)) >= 100 and is_prose_page(text)
                and not has_table_heading(text) and not is_reference_page(text)
                and not re.search(r"(?im)^\s*(?:figure|chart)\s+(?:[A-Z]{1,3}[.-])?\d", text)):
            candidates.append(physical)
    return dict(pages=pages, cover_month=month, cover_month_upper_bound=upper,
                published_date=source.get("published_date"), available_by=available,
                indexed_available_by=indexed_available,
                date_precision=source["date_precision"], date_confirmed=True,
                pdf_creation_date=created, pdf_modified_date=modified,
                metadata_title=str(metadata.get("/Title", "")), physical_pages=len(pages),
                cover_date_method=cover_method,
                current_version_date_adjusted=adjusted,
                version_date_basis="max_official_index_bound_valid_pdf_creation_modification_days",
                version_caution=(f"Current PDF version is not backdated to its original indexed release; do not use it before {available}. PDF timestamps cannot prove an unchanged historical file."
                                 if adjusted else "PDF timestamps predate indexed availability; response hash binds the observed file, not proof of an unchanged historical version."),
                provisional_oil_prose_physical_pages=candidates,
                extracted_words=sum(len(words(p)) for p in pages),
                parser="boem_source_pdf_physical_order_cover_current_version_v2")


def live_hashes():
    names = ["sources.jsonl", "documents.jsonl", "passages.jsonl", "index/manifest.json", "index/vectors.npy",
             "annotations/codex.jsonl", "annotations/passage_labels.jsonl", "annotations/passage_review.jsonl"]
    return {name: digest((DATA / name).read_bytes()) for name in names if (DATA / name).exists()}


def known_urls():
    paths = [DATA / "sources.jsonl", DATA / "documents.jsonl"]
    paths.extend((DATA / "unified_runs").glob("*/supplement/documents.jsonl"))
    return {canonical(row["url"]) for p in paths for row in read_jsonl(p)}


def select_pilots(rows, limit=6):
    ordered = sorted(rows, key=lambda r: (
        not bool(re.search(r"peis|seis|programmatic-eis", r["url"], re.I)),
        not bool(re.search(r"vol1", r["url"], re.I)),
        r["available_by"], r["url"]))
    return ordered[:limit]


def discover(output_dir, pilot_limit=6):
    if not 2 <= pilot_limit <= 6:
        raise ValueError("BOEM discovery pilot must be two-to-six documents")
    output_dir = Path(output_dir).resolve()
    allowed = (DATA / "unified_runs").resolve()
    targets = [output_dir / "source_candidates.jsonl", output_dir / "discovery_report.json"]
    if not output_dir.is_relative_to(allowed) or any(p.exists() for p in targets):
        raise ValueError("Use a new immutable BOEM discovery directory inside a separate unified run")
    before, known = live_hashes(), known_urls()
    fetch = BoemFetcher()
    if (fetch.settings["request_delay_seconds"] < 1 or fetch.settings["max_document_mb"] > 25
            or fetch.settings["download_budget_mb"] > 300):
        fetch.close()
        raise ValueError("Fetcher must preserve 1 second/host, 25 MB/document, 300 MB/invocation limits")
    archives, errors, rows, pilots, observed, excluded = [], [], {}, [], [], Counter()
    blocked = False
    try:
        for url in SEEDS:
            print(f"BOEM official index: {url}", flush=True)
            try:
                raw, final, mime = fetch.get(url)
                if not official(final):
                    raise ValueError("BOEM index left its official host")
                found, rejected, links = parse_index(raw, final, known)
                rows.update((row["url"], row) for row in found)
                observed.extend(links)
                excluded.update(rejected)
                archives.append(dict(url=final, mime=mime, bytes=len(raw), sha256=digest(raw), new_candidates=len(found)))
                print(f"BOEM index: {len(found)} dated new linked document candidates", flush=True)
            except Exception as exc:
                errors.append(dict(url=url, stage="index", error=str(exc)))
                if "403" in str(exc):
                    blocked = True
                    break
        if not blocked:
            for source in select_pilots(list(rows.values()), pilot_limit):
                print(f"BOEM pilot: {source['url']}", flush=True)
                pilot = dict(url=source["url"], title=source["title"], indexed_available_by=source["available_by"])
                request_start = len(fetch.requested_urls)
                try:
                    raw, final, mime = fetch.get(source["url"])
                    if not official(final):
                        raise ValueError("BOEM document left its official host")
                    if not raw.startswith(b"%PDF"):
                        actual = linked_pdf(raw, final)
                        pilot["actually_linked_pdf_url"] = actual
                        raw, final, mime = fetch.get(actual)
                        if not official(final):
                            raise ValueError("BOEM linked PDF left its official host")
                    if not raw.startswith(b"%PDF"):
                        raise ValueError("BOEM attachment returned non-PDF bytes")
                    path = output_dir / "pilots" / (digest(raw) + ".pdf")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists() and digest(path.read_bytes()) != digest(raw):
                        raise ValueError("BOEM pilot raw PDF archive differs; refuse overwrite")
                    if not path.exists():
                        path.write_bytes(raw)
                    pilot.update(final_url=final, bytes=len(raw), mime=mime, raw_sha256=digest(raw), pilot_pdf_path=str(path))
                    # Preserve an under-limit failed-version source for original
                    # cover/layout review as well, but never admit it by default.
                    head_reader = PdfReader(io.BytesIO(raw))
                    head_meta = head_reader.metadata or {}
                    pilot.update(physical_pages=len(head_reader.pages),
                                 cover_physical_page_text=head_reader.pages[0].extract_text() or "",
                                 pdf_creation_date=pdf_timestamp(head_meta.get("/CreationDate")),
                                 pdf_modified_date=pdf_timestamp(head_meta.get("/ModDate")))
                    parsed = parse_boem_pdf(raw, source)
                    sample_pages = parsed.pop("pages")
                    candidates = parsed["provisional_oil_prose_physical_pages"]
                    selected = list(dict.fromkeys(candidates[:2] + candidates[len(candidates)//2:len(candidates)//2+1] + candidates[-1:]))
                    pilot.update(status="accessible_date_version_checked_oil_pdf" if candidates else "date_checked_no_provisional_oil_prose",
                                 final_url=final, bytes=len(raw), mime=mime, raw_sha256=digest(raw), pilot_pdf_path=str(path),
                                 **parsed, provisional_oil_prose_page_count=len(candidates),
                                 sample_physical_pages=[dict(page=p, text=sample_pages[p-1], word_count=len(words(sample_pages[p-1]))) for p in selected])
                    source.update(pilot_status=pilot["status"], body_date_verified=True,
                                  pilot_raw_sha256=digest(raw), pilot_pdf_path=str(path), pilot_final_url=final,
                                  pilot_physical_pages=parsed["physical_pages"], pilot_cover_month=parsed["cover_month"],
                                  pilot_pdf_creation_date=parsed["pdf_creation_date"], pilot_pdf_modified_date=parsed["pdf_modified_date"])
                except Exception as exc:
                    pilot.update(status="failed_or_version_unverified_no_bypass", error=str(exc))
                    source.update(pilot_status=pilot["status"], pilot_error=str(exc))
                    if "403" in str(exc):
                        blocked = True
                    print(f"BOEM pilot error: {exc}", flush=True)
                pilot["actual_requested_urls"] = fetch.requested_urls[request_start:]
                source["actually_observed_fetch_or_redirect_urls"] = pilot["actual_requested_urls"]
                pilots.append(pilot)
                print(f"BOEM pilot result: {pilot['status']}", flush=True)
                if blocked:
                    break
    finally:
        downloaded = fetch.bytes
        fetch.close()
    if live_hashes() != before:
        raise ValueError("BOEM discovery found changed protected live inputs")
    ordered = sorted(rows.values(), key=lambda r: (r["available_by"], r["url"]))
    report = dict(at=now(), discovery_only=True, window=[START, END],
                  status="403_stopped_no_bypass" if blocked else "bounded_discovery_complete",
                  new_linked_date_candidates=len(ordered), archives=archives, excluded=dict(excluded),
                  observed_links_including_undated_and_old=observed, pilots=pilots, errors=errors,
                  date_version_checked_oil_pilots=sum(p["status"] == "accessible_date_version_checked_oil_pdf" for p in pilots),
                  download_response_bytes=downloaded, raw_registry_writes=0, live_registry_writes=0,
                  external_model_calls=0, accepted_new_passages=None, estimated_bulk_yield=None,
                  live_inputs_unchanged=True, live_hashes_before_and_after=before,
                  candidate_file_sha256=None, script_sha256=digest(Path(__file__).read_bytes()),
                  fetch_requested_urls=fetch.requested_urls,
                  limitations=["No PDF URL is guessed; actual index and attachment/redirect links are required.",
                      "Cover publication month and PDF creation/modification timestamps are checked against the official indexed availability; later timestamps fail closed.",
                      "Timestamp checks can detect later edits but cannot prove the file never changed; raw hashes bind this discovery response only.",
                      "EIS alternatives, modeled production, proposed lease sales, final decisions and realized output are distinct. Event status remains unlabelled.",
                      "Provisional oil-prose physical pages are a diagnostic, not accepted evidence. Tables, figures, references, footnotes, non-oil sections and incomplete contexts still require unchanged unified policy and original-page review.",
                      "A 25 MB document cap may exclude full-program PDFs. Do not bypass it, invent chapter URLs or split downloads to circumvent it.",
                      "Bounded pilot records and raw page counts do not demonstrate >=12000 qualified unique passages.",
                      "Redistribution is review_required, including contractor-written reports and third-party figures; public access does not establish a blanket license."])
    write_jsonl(targets[0], ordered)
    report["candidate_file_sha256"] = digest(targets[0].read_bytes())
    write_json(targets[1], report)
    return report


def refine_catalog(prior_dir, output_dir):
    """Offline, immutable successor: two root-reviewed PDF dates; no network."""
    prior_dir, output_dir = Path(prior_dir).resolve(), Path(output_dir).resolve()
    allowed = (DATA / "unified_runs").resolve()
    targets = [output_dir / "source_candidates.jsonl", output_dir / "catalog_refinement_report.json",
               output_dir / "risk_ledger.jsonl"]
    if (not prior_dir.is_relative_to(allowed) or not output_dir.is_relative_to(allowed)
            or any(p.exists() for p in targets)):
        raise ValueError("Use preserved input and a new immutable unified-run successor directory")
    before = live_hashes()
    candidate_path, report_path = prior_dir / "source_candidates.jsonl", prior_dir / "discovery_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    input_hash = digest(candidate_path.read_bytes())
    if input_hash != report["candidate_file_sha256"]:
        raise ValueError("Original BOEM candidate manifest changed after discovery")
    pilot_by_url = {p["url"]: p for p in report["pilots"]}
    approved = {
        "fb769e848a17ed9472f083465bb7568e16d17a4ecdf063b23cb8a30f3739229a":
            dict(month="2022-07", available="2022-07-08", pages=298),
        "5bc2dca7dae40879495fc21a10f0434d4eb9322c2f13cb0edac6a7845bb2d665":
            dict(month="2023-09", available="2023-10-03", pages=302),
    }
    rows, risks = [], []
    for original in read_jsonl(candidate_path):
        source = dict(original)
        path = urlparse(source["url"]).path
        if path.startswith("/newsroom/"):
            risks.append(dict(url=source["url"], action="excluded_successor", reason="newsroom_press_release_is_not_a_pdf_source",
                              original_candidate=source, accepted_passages=None))
            continue
        if path == "/oil-gas-energy/decision-memo-national-program-signed":
            risks.append(dict(url=source["url"], action="excluded_successor", reason="paragraph_pfp_publication_day_misattributed_to_later_decision_memo",
                              original_candidate=source, accepted_passages=None,
                              note="No document PDF/date check was performed; do not silently infer a corrected date from a different publication event."))
            continue
        source["original_discovery_candidate_sha256"] = input_hash
        source["indexed_available_by"] = original["available_by"]
        pilot = pilot_by_url.get(source["url"])
        raw_sha = pilot.get("raw_sha256") if pilot else None
        if raw_sha in approved:
            evidence = approved[raw_sha]
            raw_path = Path(pilot["pilot_pdf_path"]).resolve()
            if (not raw_path.is_relative_to(prior_dir) or not raw_path.is_file()
                    or digest(raw_path.read_bytes()) != raw_sha
                    or raw_path.stat().st_size != pilot["bytes"]):
                raise ValueError("Reviewed BOEM publisher PDF archive changed")
            source.update(pilot_status="root_reviewed_source_date_successor_not_original_strict_pass",
                          pilot_pdf_path=str(raw_path), pilot_raw_sha256=raw_sha,
                          pilot_final_url=pilot["final_url"], pilot_physical_pages=evidence["pages"],
                          pilot_cover_month=evidence["month"], body_date_verified=True,
                          pilot_fetched_at=datetime.fromtimestamp(raw_path.stat().st_mtime, timezone.utc).isoformat(),
                          pilot_fetched_at_source="publisher_pdf_archive_write_time_not_http_publication",
                          pilot_discovery_finished_at=report["at"],
                          available_by=evidence["available"],
                          publication_date_precision=original["date_precision"],
                          current_version_date_adjusted=evidence["available"] != original["available_by"],
                          version_date_basis="max_official_index_bound_valid_pdf_creation_modification_days",
                          version_caution=("This is the observed current PDF version, not a historical-file guarantee. "
                                           f"Do not use it before {evidence['available']}; original indexed availability is {original['available_by']}."))
            if raw_sha in APPROVED_VISUAL_COVERS:
                render = (prior_dir / "pilots/draft-cover.png").resolve()
                validation = dict(APPROVED_VISUAL_COVERS[raw_sha], raw_sha256=raw_sha,
                                  rendered_png_path=str(render), review_origin="codex_source_visual_review_root_confirmed_not_human")
                validated_visual_cover_month(raw_path.read_bytes(), validation)
                source["visual_cover_validation"] = validation
                source["source_date_review_sidecar"] = str(prior_dir / "draft-cover-visual-review.json")
            source["pilot_error_preserved_in_prior_report"] = source.pop("pilot_error", None)
            risks.append(dict(url=source["url"], action="source_date_successor_root_reviewed",
                              raw_sha256=raw_sha, original_strict_status=pilot["status"], original_strict_error=pilot.get("error"),
                              indexed_available_by=original["available_by"], current_available_by=source["available_by"],
                              cover_month=evidence["month"], physical_pages=evidence["pages"],
                              accepted_passages=None, review_origin="Codex root source visual/metadata review; not human annotation"))
        else:
            source["pilot_status"] = "not_fetched_requires_pdf_cover_metadata_scope_validation"
            source["body_date_verified"] = False
        rows.append(source)
    write_jsonl(targets[0], rows)
    write_jsonl(targets[2], risks)
    out = dict(at=now(), offline_refinement=True, external_model_calls=0, network_requests=0,
               original_candidates=len(read_jsonl(candidate_path)), successor_candidates=len(rows),
               excluded_candidates=sum(r["action"] == "excluded_successor" for r in risks),
               root_reviewed_source_date_pilots=sum(r["action"] == "source_date_successor_root_reviewed" for r in risks),
               accepted_new_passages=None, original_candidate_sha256=input_hash,
               original_discovery_report_sha256=digest(report_path.read_bytes()),
               successor_candidate_sha256=digest(targets[0].read_bytes()), risk_ledger_sha256=digest(targets[2].read_bytes()),
               script_sha256=digest(Path(__file__).read_bytes()),
               live_inputs_unchanged=live_hashes() == before, live_hashes_before_and_after=before,
               notes=["The original 24-candidate manifest, strict refusals and strict implementation snapshot remain unchanged.",
                      "Only two source-specific publication/version checks were root-reviewed; neither is human gold or passage acceptance.",
                      "Unknown, old, conflicting covers and missing metadata still fail closed; all unpiloted documents need full collection validation.",
                      "Reuse requires publisher raw SHA, indexed publication context, alias/final URL provenance and current-version date bound; never reconstruct tables or inflate passage counts."])
    if not out["live_inputs_unchanged"] or digest(candidate_path.read_bytes()) != input_hash:
        raise ValueError("Original manifest or protected live inputs changed during offline successor creation")
    write_json(targets[1], out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=DATA / "unified_runs/semantic-page-v2-buffer/reports/boem-discovery")
    parser.add_argument("--pilot-limit", type=int, default=6)
    parser.add_argument("--refine-from", type=Path)
    args = parser.parse_args()
    if args.refine_from:
        print(json.dumps(refine_catalog(args.refine_from, args.output_dir), indent=2))
        return
    report = discover(args.output_dir, args.pilot_limit)
    print(json.dumps({key: report[key] for key in ("status", "new_linked_date_candidates",
                     "date_version_checked_oil_pilots", "download_response_bytes", "candidate_file_sha256", "live_inputs_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
