from __future__ import annotations

import calendar
import io
import re
import time
from collections import Counter
from datetime import datetime
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .store import DATA, append_jsonl, config, digest, now, read_jsonl, words, write_json, write_jsonl

USER_AGENT = "OilEvidenceAcademic/0.1 (+local university research; rate limited)"
ARCHIVES = {
    "EIA": "https://www.eia.gov/outlooks/steo/outlook.php",
    "OPEC": "https://www.opec.org/monthly-oil-market-report.html",
}
OIL = re.compile(r"\b(oil|crude|petroleum|opec|refiner\w*|tanker\w*|barrels?|sanctions?|hormuz|suez|red sea)\b", re.I)
REGIONS = {"Middle East": r"saudi|iraq|iran|kuwait|uae|oman|hormuz|middle east|red sea|yemen",
           "Europe": r"europe|russia|ukraine|north sea|norway|brent",
           "Asia": r"china|india|japan|korea|asia|singapore",
           "Americas": r"united states|u\.s\.|canada|brazil|venezuela|mexico|gulf of mexico",
           "Africa": r"libya|nigeria|angola|algeria|africa"}


class Fetcher:
    def __init__(self):
        self.client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=45, follow_redirects=False)
        self.robots = {}
        self.last = {}
        self.bytes = 0
        self.settings = config()["corpus"]

    def close(self):
        self.client.close()

    def get(self, url, _redirects=0):
        if _redirects > 5:
            raise ValueError("Too many source redirects")
        p = urlparse(url)
        if p.scheme not in {"http", "https"}:
            raise ValueError("HTTP(S) sources only")
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self.robots:
            r = self.client.get(origin + "/robots.txt", follow_redirects=True)
            self.last[origin] = time.monotonic()
            rp = RobotFileParser()
            if r.status_code in {404, 410}:
                rp.parse([])
            elif r.status_code == 200 and "<html" not in r.text[:200].lower():
                rp.parse(r.text.splitlines())
            else:
                raise RuntimeError(f"Cannot verify robots.txt: {r.status_code} {origin}")
            self.robots[origin] = rp
        rp = self.robots[origin]
        if not rp.can_fetch(USER_AGENT, url):
            raise RuntimeError(f"robots.txt disallows {url}")
        delay = max(self.settings["request_delay_seconds"], rp.crawl_delay(USER_AGENT) or 0)
        for attempt in range(3):
            time.sleep(max(0, delay - (time.monotonic() - self.last.get(origin, 0))))
            self.last[origin] = time.monotonic()
            try:
                with self.client.stream("GET", url) as r:
                    if r.is_redirect:
                        # Re-check robots and rate limits at every redirect destination.
                        return self.get(urljoin(url, r.headers['location']), _redirects + 1)
                    r.raise_for_status()
                    chunks = []
                    size = 0
                    for chunk in r.iter_bytes():
                        size += len(chunk)
                        self.bytes += len(chunk)
                        if size > self.settings["max_document_mb"] * 1024**2:
                            raise ValueError("Document exceeds configured size limit")
                        if self.bytes > self.settings["download_budget_mb"] * 1024**2:
                            raise ValueError("Download budget reached")
                        chunks.append(chunk)
                    return b"".join(chunks), str(r.url), r.headers.get("content-type", "")
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)


def date_from_text(text):
    for pattern, fmt in [(r"\b\d{1,2}/\d{1,2}/20\d{2}\b", "%m/%d/%Y"),
                         (r"\b\d{1,2} [A-Za-z]+ 20\d{2}\b", "%d %B %Y"),
                         (r"\b[A-Za-z]+ \d{1,2}, 20\d{2}\b", "%B %d, %Y")]:
        for m in re.finditer(pattern, text):
            try:
                return datetime.strptime(m.group(), fmt).date().isoformat()
            except ValueError:
                pass
    return None


def month_from_filename(url):
    name = urlparse(url).path.split("/")[-1].lower()
    for month in range(1, 13):
        for token in {calendar.month_name[month].lower(), calendar.month_abbr[month].lower()}:
            m = re.search(rf"{token}[-_ ]?(20\d{{2}}|\d{{2}})(?!\d)", name)
            if m:
                year = int(m.group(1))
                year = year + 2000 if year < 100 else year
                return f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
    return None


def discover():
    settings = config()["corpus"]
    existing = {d["url"]: d for d in read_jsonl(DATA / "sources.jsonl")}
    fetch = Fetcher()
    for source, url in ARCHIVES.items():
        try:
            raw, final, _ = fetch.get(url)
            soup = BeautifulSoup(raw, "html.parser")
            count = 0
            for a in soup.select("a[href]"):
                href = urljoin(final, a["href"])
                if not urlparse(href).path.lower().endswith(".pdf"):
                    continue
                if source == "EIA" and "/archives/" not in href:
                    continue
                if source == "OPEC" and "momr" not in href.lower():
                    continue
                context = a.find_parent("tr") or a.parent
                exact = date_from_text(context.get_text(" ", strip=True))
                upper = exact or month_from_filename(href)
                if not upper or not settings["start_date"] <= upper <= settings["end_date"]:
                    continue
                title = f"{source} {'STEO' if source == 'EIA' else 'MOMR'} {upper[:7]}"
                existing.setdefault(href, dict(url=href, source=source, title=title,
                    published_date=exact, available_by=upper, date_precision="day" if exact else "month_upper_bound",
                    date_source=url, event_date=None, event_status="unlabelled",
                    redistribution="eia_attribution" if source == "EIA" else "review_required"))
                count += 1
            append_jsonl(DATA / "logs/discovery.jsonl", dict(at=now(), source=source, candidates=count, status="ok"))
        except Exception as e:
            append_jsonl(DATA / "logs/discovery.jsonl", dict(at=now(), source=source, status="failed", error=str(e)))
    # Public archive HTML supplies real dates; do not invent URLs for reports.
    extra_archives = [("EIA-TIE", f"https://www.eia.gov/todayinenergy/archive.php?my={year}")
                      for year in range(int(settings["start_date"][:4]), int(settings["end_date"][:4]) + 1)]
    extra_archives += [("OFAC", f"https://ofac.treasury.gov/recent-actions?page={page}") for page in range(4)]
    extra_archives += [("EIA-TWIP", "https://www.eia.gov/petroleum/weekly/")]
    for source, url in extra_archives:
        try:
            raw, final, _ = fetch.get(url)
            soup = BeautifulSoup(raw, "html.parser")
            count = 0
            for a in soup.select("a[href]"):
                href = urljoin(final, a["href"])
                title = a.get_text(" ", strip=True)
                if source == "EIA-TIE":
                    if "detail.php?id=" not in href or not OIL.search(title):
                        continue
                    parent = a.find_parent("li")
                    exact = date_from_text(parent.get_text(" ", strip=True)) if parent else None
                elif source == "EIA-TWIP":
                    match = re.search(r"/archive/(20\d{2})/(\d{6})/includes/analysis_print.php", href)
                    if not match or not OIL.search(title):
                        continue
                    exact = datetime.strptime(match.group(2), "%y%m%d").date().isoformat()
                else:
                    match = re.search(r"/recent-actions/(20\d{6})(?:_|$)", href)
                    if not match or not re.search(r"iran|russia|venezuela|oil|petroleum", title, re.I):
                        continue
                    exact = datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
                if not exact or not settings["start_date"] <= exact <= settings["end_date"]:
                    continue
                existing.setdefault(href, dict(url=href, source=source, title=title,
                    published_date=exact, available_by=exact, date_precision="day", date_source=url,
                    event_date=None, event_status="unlabelled",
                    redistribution="eia_attribution" if source.startswith("EIA") else "review_required"))
                count += 1
            append_jsonl(DATA / "logs/discovery.jsonl", dict(at=now(), source=source, url=url, candidates=count, status="ok"))
        except Exception as e:
            append_jsonl(DATA / "logs/discovery.jsonl", dict(at=now(), source=source, url=url, status="failed", error=str(e)))
    fetch.close()
    # Deterministic round-robin across sources, newest first, so a small pilot includes both.
    queues = [[d for d in sorted(existing.values(), key=lambda x: x["available_by"], reverse=True) if d["source"] == s]
              for s in sorted({d["source"] for d in existing.values()})]
    rows = []
    while any(queues):
        for queue in queues:
            if queue:
                rows.append(queue.pop(0))
    write_jsonl(DATA / "sources.jsonl", rows)
    return dict(sources=len(rows), by_source=dict(Counter(d["source"] for d in rows)))


def collect(limit=6, source_filter=None, url_prefix=None):
    fetch = Fetcher()
    try:
        completed = {d["url"]: d for d in read_jsonl(DATA / "documents.jsonl")}
        failures = []
        added = 0
        # Round-robin source/year strata instead of filling the pilot with recent reports only.
        from collections import defaultdict
        buckets = defaultdict(list)
        for source in read_jsonl(DATA / "sources.jsonl"):
            if (not source_filter or source["source"] == source_filter) and (not url_prefix or source['url'].startswith(url_prefix)):
                buckets[(source["source"], source["available_by"][:4])].append(source)
        ordered = []
        while any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key]:
                    ordered.append(buckets[key].pop(0))
        for source in ordered:
            if added >= limit:
                break
            if source["url"] in completed:
                old = completed[source["url"]]
                path = DATA / old["raw_path"]
                if path.exists() and digest(path.read_bytes()) == old["raw_sha256"]:
                    continue
                raise ValueError('Existing raw source missing or changed; refuse refetch/overwrite: '+old['document_id'])
            try:
                raw, final, content_type = fetch.get(source["url"])
                is_pdf = raw.startswith(b"%PDF")
                if (source.get('requires_pdf') or source["url"].lower().endswith(".pdf") or source["source"] in {"EIA-CAB", "WB-CMO"}) and not is_pdf:
                    raise ValueError("PDF URL returned non-PDF content")
                doc_id = digest(source["url"])[:20]
                path = DATA / "raw" / (doc_id + (".pdf" if is_pdf else ".html"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                doc = dict(source, document_id=doc_id, final_url=final, content_type=content_type,
                           raw_path=path.relative_to(DATA).as_posix(), raw_sha256=digest(raw), bytes=len(raw), fetched_at=now())
                completed[source["url"]] = doc
                write_jsonl(DATA / "documents.jsonl", completed.values())
                added += 1
                print(f"Collected {added}: {source['title']} ({len(raw):,} bytes)", flush=True)
            except Exception as e:
                failure = dict(at=now(), url=source["url"], error=str(e))
                failures.append(failure)
                append_jsonl(DATA / "logs/collection_errors.jsonl", failure)
                print(f"Skipped {source['url']}: {e}", flush=True)
                if "budget" in str(e).lower():
                    break
        return dict(added=added, total=len(completed), failures=len(failures), downloaded_bytes=fetch.bytes)
    finally:
        fetch.close()


def clean_page(text):
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if re.fullmatch(r"[-+\d.,()%/]+", line):
            # A wrapped year after 'in/since/from...' is sentence evidence, not
            # a chart tick. Retain the literal year; never infer a missing value.
            wrapped_year=bool(re.fullmatch(r'(?:19|20)\d{2}[.,;]?',line) and lines and
                re.search(r'\b(?:in|since|from|until|during|after|before|to)\s*$',lines[-1],re.I))
            if not wrapped_year:
                continue
        if re.search(r"^(OPEC Monthly Oil Market Report|U\.S\. Energy Information Administration|Short-Term Energy Outlook)\s*[|–—-]", line, re.I):
            continue
        # Drop predominantly numeric chart/table rows; do not invent table semantics.
        toks = line.split()
        numeric = sum(bool(re.fullmatch(r"[-+\d.,()%/]+", t)) for t in toks)
        if len(toks) >= 4 and numeric / len(toks) > .45:
            continue
        lines.append(line)
    text = " ".join(lines)
    return re.sub(r"(?<=[a-z])-\s+(?=[a-z])", "", text)


def is_prose_page(text):
    """Reject numeric chart/table pages before line cleaning can hide their structure."""
    tokens = text.split()
    if not tokens:
        return False
    numeric = sum(bool(re.fullmatch(r"[-+\d.,()%/]+", t)) for t in tokens)
    if numeric / len(tokens) > .18:
        return False
    sentences = re.split(r"(?<=[.!?])\s+", text)
    prose = [s for s in sentences if len(re.findall(r"\b[A-Za-z]{2,}\b", s)) >= 12]
    return len(prose) >= 2


def is_reference_page(text):
    """Conservative bibliography detection, not evidence reconstruction."""
    if re.search(r'(?im)^\s*(endnotes|references)\s*(?:\d+[-–]\d+(?:\s+[A-Za-z]+\s+20\d{2})?)?\s*$', text):
        return True
    numbered = re.findall(r'(?m)^\s*\d{1,3}\s+[A-Za-z“\"]', text)
    citations = len(re.findall(r'https?://|\baccessed\b|\bReuters\b|\bMiddle East Economic Survey\b',text,re.I))
    return len(numbered) >= 3 and citations >= 3


def oil_topic_supported(text):
    """Company names and vegetable oils alone do not establish mineral-oil scope."""
    text = re.sub(r'\b(?:China\s+)?National\s+(?:Offshore\s+)?(?:Petroleum|Oil)\s+(?:Corporation|Company)\b', '', text, flags=re.I)
    text = re.sub(r'\bJapan Oil, Gas and Metals National Corporation\b', '', text, flags=re.I)
    text = re.sub(r'\b(?:soybean|vegetable|palm|cooking)\s+oil\b', '', text, flags=re.I)
    return bool(OIL.search(text))


def has_table_heading(text):
    return bool(re.search(r'(?im)^\s*Table\s+(?:[A-Z][.-])?\d+(?:[-.]\d+)*[.:\s]', text))


NEW_PDF_SERIES = {'AU-REQ','UK-ENERGY-TRENDS','UK-DUKES','DOE-SPR','BOEM-POLICY','EIA-METHODS','EIA-FINANCE','EIA-IEO'}


def extraction_suffix(source):
    if source == 'WB-CMO':
        return '-body-v3-fonttools.json'
    if source == 'AU-REQ':
        return '-body-v5-au-req-oil-scope.json'
    if source in NEW_PDF_SERIES or source=='BSEE-HURRICANE':
        return '-body-v4-'+source.lower()+'-oil-scope.json'
    return '-body-v2.json'


def remove_running_headers(pages):
    """Remove exact repeated short page furniture in new PDF series only."""
    furniture=[]
    for page in pages:
        lines=[line.strip() for line in page.splitlines() if line.strip()]
        # Only an exact report-title/running header or institutional attribution
        # at the margin qualifies. Repeated body sentences and units stay intact.
        margin=lines[:2]+lines[-2:]
        furniture.append({line for line in margin if len(line)<160 and re.search(
            r'Energy Information Administration|Resources and Energy Quarterly|Department of Energy|^Strategic Petroleum Reserve$',line,re.I)})
    counts=Counter(line for page in furniture for line in page)
    threshold=max(3,sum(bool(p.strip()) for p in pages)//3)
    repeated={line for line,count in counts.items() if count>=threshold}
    return ['\n'.join(line for line in page.splitlines() if line.strip() not in repeated
        and not re.search(r'^\s*(?:Strategic Petroleum Reserve Annual Report[^\n]*\|\s*Page\s*\d+|USDOI Economic Analysis Methodology[^\n]*BOEM)\s*$',line,re.I)) for page in pages]


def split_passages(text, target=200, minimum=65):
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    groups, buffer = [], []
    for sentence in sentences:
        # Bound pathological unpunctuated PDF blocks without overlapping duplicates.
        tokens = sentence.split()
        for start in range(0, len(tokens), target):
            part = tokens[start:start + target]
            if buffer and len(buffer) + len(part) > target:
                groups.append(" ".join(buffer))
                buffer = []
            buffer.extend(part)
    if buffer:
        if groups and len(buffer) < minimum and len(groups[-1].split()) + len(buffer) <= target + minimum:
            groups[-1] += " " + " ".join(buffer)
        else:
            groups.append(" ".join(buffer))
    return [g for g in groups if len(words(g)) >= minimum and oil_topic_supported(g)]


def build_corpus(output_dir=None, baseline=None, baseline_document_ids=None):
    settings = config()["corpus"]
    if settings.get('segmentation_version') == 'semantic-page-v1':
        if baseline or baseline_document_ids:
            raise ValueError('Uniform corpus cannot retain legacy chunks; rebuild verified source pages in a new staged run')
        if output_dir is None:
            raise ValueError('Use oil build / scripts/unify_corpus.py stage; direct live overwrite is disabled')
        from .unified_corpus import build_unified_corpus
        return build_unified_corpus(output_dir)
    output = DATA if output_dir is None else output_dir
    passages = list(baseline or [])
    errors, seen, all_words = [], {digest(" ".join(words(p["text"]))) for p in passages}, Counter()
    retained_documents = {p["document_id"] for p in passages}
    retained_documents.update(baseline_document_ids or [])
    for p in passages:
        all_words.update(words(p["text"]))
    excluded_pages = 0
    mixed_table_pages = 0
    documents = read_jsonl(DATA / "documents.jsonl")
    for doc_number, doc in enumerate(documents, 1):
        if doc["document_id"] in retained_documents:
            continue
        if doc.get('ingest_exclusion_reason'):
            continue
        path = DATA / doc["raw_path"]
        try:
            if digest(path.read_bytes()) != doc["raw_sha256"]:
                raise ValueError("Raw file hash mismatch")
            extraction_version = extraction_suffix(doc['source'])
            cache = DATA / "extracted" / (doc["raw_sha256"] + extraction_version)
            if cache.exists():
                import json
                pages = json.loads(cache.read_text(encoding="utf-8"))
            elif path.suffix == ".pdf":
                reader = PdfReader(io.BytesIO(path.read_bytes()))
                pages = [p.extract_text() or "" for p in reader.pages]
                if doc['source'] in NEW_PDF_SERIES:
                    # Keep non-oil chapters blank at the same physical page indexes.
                    from oil_evidence.pdf_scope import oil_page_numbers
                    scope=oil_page_numbers(pages,doc['source'])
                    if scope is not None:
                        pages=[p if i in scope else '' for i,p in enumerate(pages,1)]
                    pages=remove_running_headers(pages)
                write_json(cache, pages)
            else:
                soup = BeautifulSoup(path.read_bytes(), "html.parser")
                for el in soup.select("script,style,nav,header,footer"):
                    el.decompose()
                if doc['source']=='BSEE-HURRICANE':
                    main=soup.select_one("main article .field--name-body")
                else:
                    main = soup.select_one(".tie-article") or soup.select_one(".l-col.article") or soup.select_one(".pagecontent.print .main_col") or soup.select_one("article .field--name-field-body") or soup.select_one("article") or soup.select_one("main")
                if main is None:
                    raise ValueError("No recognized article body; refuse navigation-heavy whole-page extraction")
                if doc['source']=='BSEE-HURRICANE':
                    for el in main.select('.usa-alert,figure'):
                        el.decompose()
                body = main.get_text("\n", strip=True)
                if doc["source"] == "OFAC":
                    # Names/addresses in sanctions-list updates are not an oil-policy narrative corpus.
                    body = re.split(r"Specially Designated Nationals List Updates", body, flags=re.I)[0]
                pages = [body]
                write_json(cache, pages)
            for page, raw in enumerate(pages, 1):
                if str(page) in doc.get('excluded_physical_pages',{}):
                    excluded_pages += 1
                    continue
                if doc['source']=='UK-ENERGY-TRENDS' and re.search(r'About\s+this\s+release',raw,re.I) and re.search(r'In\s+this\s+release',raw,re.I):
                    # Navigation/contents covers are not supply evidence, even if
                    # they name the oil section or include a short headline.
                    excluded_pages += 1
                    continue
                if doc['source']=='UK-DUKES' and re.search(r'\b(?:19|20)\d{3}\b',raw):
                    # A verified year+superscript artifact (1998¹ -> 19981) is
                    # ambiguous in flat text. Exclude the page, do not repair by guess.
                    excluded_pages += 1
                    continue
                if doc['source']=='EIA-TIE':
                    # Contributor/footer tags are metadata, never relevance evidence.
                    raw=re.split(r'\bPrincipal contributors?\s*:|\bTags\s*:',raw,flags=re.I)[0]
                if doc['source'] in NEW_PDF_SERIES and re.search(r'(?im)^\s*(?:abbreviations\s*(?:and|&)\s*acronyms|variable\s+definitions|data\s+dictionary)\b',raw):
                    excluded_pages += 1
                    continue
                if doc['source']=='EIA-METHODS' and re.search(r'Variable\s+Description\s+Unit\s+Source|LFMM\s+variable\s+Definition\s+API\s+keys',raw,re.I):
                    # These are flattened input-variable/API tables, not supply
                    # narratives; prose punctuation is not sufficient to admit them.
                    excluded_pages += 1
                    continue
                if doc['source'] in {'EIA-CAB', 'WB-CMO'} | NEW_PDF_SERIES and is_reference_page(raw):
                    excluded_pages += 1
                    continue
                if doc['source'] in {'EIA-CAB', 'WB-CMO'} | NEW_PDF_SERIES and has_table_heading(raw):
                    # New-series PDFs mix tables with prose. Fail closed rather than
                    # treat flattened rows as reliable evidence; preserve original PDF.
                    excluded_pages += 1
                    mixed_table_pages += 1
                    continue
                if path.suffix == ".pdf" and not is_prose_page(raw):
                    excluded_pages += 1
                    continue
                for text in split_passages(clean_page(raw), settings["chunk_words"], settings["min_words"]):
                    fingerprint = digest(" ".join(words(text)))
                    if doc['document_id']+'-'+fingerprint[:16] in doc.get('excluded_passage_ids',{}):
                        continue
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    passage = {k: doc.get(k) for k in ["document_id", "source", "title", "url", "published_date", "available_by", "date_precision", "date_source", "event_date", "event_status", "redistribution"]}
                    passage.update(passage_id=doc["document_id"] + "-" + fingerprint[:16], text=text, page=page,
                                   regions=[name for name, pattern in REGIONS.items() if re.search(pattern, text, re.I)],
                                   region_method="keyword_heuristic", word_count=len(words(text)))
                    passages.append(passage)
                    all_words.update(words(text))
        except Exception as e:
            errors.append(dict(document_id=doc["document_id"], error=str(e)))
        if doc_number % 10 == 0:
            print(f"Extracted {doc_number}/{len(documents)} documents; {len(passages)} unique oil passages", flush=True)
    write_jsonl(output / "passages.jsonl", passages)
    report = dict(passages=len(passages), documents=len({p["document_id"] for p in passages}),
                  words=sum(all_words.values()), unique_words=len(all_words),
                  by_source=dict(Counter(p["source"] for p in passages)),
                  by_year=dict(Counter((p.get("available_by") or "unknown")[:4] for p in passages)),
                  date_precision=dict(Counter(p["date_precision"] for p in passages)),
                  min_words=settings["min_words"], target_words=settings["chunk_words"], overlap_words=0,
                  exact_deduplication=True, near_duplicate_audit="pending", table_handling="numeric-heavy, insufficient-prose, bibliography and new-series table-heading pages excluded; AU/UK annual non-oil chapters excluded; repeated short PDF furniture removed; manual QA pending",
                  excluded_pdf_pages=excluded_pages,
                  excluded_new_mixed_table_pages=mixed_table_pages,
                  meets_size_requirement=len(passages) >= 10000 and sum(all_words.values()) >= 100000,
                  retained_baseline_passages=len(baseline or []),
                  extraction_errors=errors, corpus_sha256=digest((output / "passages.jsonl").read_bytes()))
    write_json(output / "reports/corpus.json", report)
    return report
