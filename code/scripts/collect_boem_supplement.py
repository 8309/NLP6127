"""Collect actually linked BOEM PDFs into an append-only staged supplement.

Original bytes and physical page order are retained. Only recognized repeated
margin furniture is removed from the cache, with an exact line-level audit.
The unchanged unified builder decides page/paragraph eligibility and counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re

from collect_unified_supplement import writer_lock, validate_manifest_pair, prior_protections
from bounded_boem_fetcher import BoundedBoemFetcher as BoemFetcher
from discover_boem_buffer import official, linked_pdf, parse_boem_pdf
from oil_evidence.corpus import extraction_suffix
from oil_evidence.store import DATA, digest, now, read_jsonl, words, write_json, write_jsonl, append_jsonl


HEADER = re.compile(r'^(?:(?:20\d{2}\s+)?National\s+OCS\b|\d{4}\s*[-–]\s*\d{4}\s+National\s+OCS\b|'
    r'(?:Draft|Final|Supplemental|Programmatic)\s+(?:Environmental\s+Impact\s+Statement|EIS|PEIS|SEIS)\b|'
    r'(?:Gulf\s+of\s+(?:Mexico|America)|GOM|GOA)\s+(?:Regional|OCS|Oil|Lease)\b|'
    r'USDOI\s*[|:]?\s*BOEM\b|U\.S\.\s+Department\s+of\s+the\s+Interior\b|'
    r'Bureau\s+of\s+Ocean\s+Energy\s+Management\b|'
    r'Chapter\s+\d+\s*:.*USDOI\s*\|\s*BOEM\s*$)', re.I)
FINITE = re.compile(r'\b(?:is|are|was|were|will|would|could|may|has|have|provides|proposes|results)\b', re.I)
PAGE_FOOTER = re.compile(r'^Chapter\s+\d+\s*:\s*[^\n]{1,135}?\s+\d+(?:[-–]\d+)?\s+USDOI\s*\|\s*BOEM\s*$', re.I)


def remove_margin_furniture(pages):
    candidates = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        possible = set(lines[:3] + lines[-3:])
        candidates.update(line for line in possible
            if len(line) <= 180 and HEADER.search(line) and not FINITE.search(line))
    repeated = {line for line, count in candidates.items() if count >= max(3, len(pages) // 10)}
    cleaned, audit = [], []
    for physical, page in enumerate(pages, 1):
        lines = page.splitlines()
        nonblank = [i for i, line in enumerate(lines) if line.strip()]
        margin = set(nonblank[:3] + nonblank[-3:])
        kept, removed = [], []
        for i, line in enumerate(lines):
            if i in margin and PAGE_FOOTER.fullmatch(line.strip()) and not FINITE.search(line):
                removed.append(dict(line=i + 1, text=line, reason='visually_verified_page_numbered_BOEM_margin_footer'))
            elif i in margin and line.strip() in repeated:
                removed.append(dict(line=i + 1, text=line, reason='exact_repeated_recognized_pdf_margin_furniture'))
            elif (i in margin and re.fullmatch(r'\s*20\d{2}\s*', line) and i + 1 < len(lines)
                  and lines[i + 1].strip() in repeated and re.search(r'National\s+OCS', lines[i + 1], re.I)):
                removed.append(dict(line=i + 1, text=line, reason='split_year_of_verified_repeated_report_header'))
            else:
                kept.append(line)
        cleaned.append('\n'.join(kept))
        if removed:
            audit.append(dict(physical_page=physical, original_page_sha256=digest(page),
                              cleaned_page_sha256=digest(cleaned[-1]), removed_lines=removed))
    return cleaned, audit


def collect(run, candidates, limit=20):
    if not 1 <= limit <= 25 or not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,80}', run):
        raise ValueError('Require safe run name and 1..25 PDF document limit')
    root = DATA / 'unified_runs' / run
    if (root / 'published').exists():
        raise ValueError('Published run is immutable')
    with writer_lock(run):
        target = root / 'supplement'
        old_docs, old_sources = validate_manifest_pair(target)
        docs, sources = ({row['url']: row for row in rows} for rows in [old_docs, old_sources])
        known = {row['url'].rstrip('/') for row in read_jsonl(DATA / 'documents.jsonl')}
        candidate_path = Path(candidates).resolve()
        candidate_sha = digest(candidate_path.read_bytes())
        rows = read_jsonl(candidate_path)
        prior_size_rejections = {row['url'] for row in read_jsonl(target / 'pdf_collection.jsonl')
            if row.get('status') == 'failed_no_bypass' and row.get('error') in {
                'Document exceeds configured size limit',
                'BOEM declared response size exceeds configured size limit',
                'BOEM robots/document exceeds configured size limit',
            }}
        protected = prior_protections()
        for name in ['sources.jsonl', 'documents.jsonl', 'passages.jsonl', 'index/manifest.json', 'index/vectors.npy',
                     'annotations/codex.jsonl', 'annotations/passage_labels.jsonl', 'annotations/passage_review.jsonl',
                     'reports/corpus.json', 'reports/corpus_resegmentation.json', 'reports/passage_annotation.json',
                     'model_usage.sqlite3', 'annotations.sqlite3']:
            path = DATA / name
            if path.exists():
                protected[path] = digest(path.read_bytes())
        import discover_boem_buffer
        import bounded_boem_fetcher
        parser_sha = digest(Path(discover_boem_buffer.__file__).read_bytes())
        fetcher_sha = digest(Path(bounded_boem_fetcher.__file__).read_bytes())
        self_sha = digest(Path(__file__).read_bytes())
        attempts, added, qualified = 0, 0, 0
        fetch = BoemFetcher()
        if (fetch.settings['request_delay_seconds'] < 1 or fetch.settings['max_document_mb'] > 25
                or fetch.settings['download_budget_mb'] > 300):
            fetch.close()
            raise ValueError('Fetcher must retain project rate/size/budget rules')
        try:
            for source in sorted(rows, key=lambda r: (
                not bool(re.search(r'draft-peis-vol1|final-peis-vol1|draft-seis|seisfinal|programmatic-eis|/final-goa-peispdf', r['url'], re.I)),
                r['available_by'], r['url'])):
                if attempts >= limit:
                    break
                if source['url'] in docs or source['url'].rstrip('/') in known:
                    continue
                if source['url'] in prior_size_rejections:
                    continue  # Do not repeatedly download a source already rejected by this run's cap.
                if source.get('source') != 'BOEM-EIS' or not source.get('requires_pdf') or not official(source['url']):
                    raise ValueError('Only actually discovered official BOEM PDF candidates are admitted')
                if not source.get('available_by') or not '2018-01-01' <= source['available_by'] <= '2026-10-05':
                    raise ValueError('BOEM indexed date outside fixed window')
                date.fromisoformat(source['available_by'])
                attempts += 1
                log = dict(at=now(), url=source['url'], candidate_file_sha256=candidate_sha)
                request_start = len(getattr(fetch, 'requested_urls', []))
                try:
                    reused_pilot = False
                    if source.get('pilot_pdf_path') and source.get('pilot_raw_sha256') and source.get('pilot_final_url'):
                        pilot = Path(source['pilot_pdf_path']).resolve()
                        if not pilot.is_relative_to((root / 'reports').resolve()) or not pilot.is_file():
                            raise ValueError('Pilot reuse requires an existing PDF inside this run reports')
                        raw, final, mime = pilot.read_bytes(), source['pilot_final_url'], 'application/pdf'
                        if digest(raw) != source['pilot_raw_sha256'] or len(raw) > 25 * 1024**2:
                            raise ValueError('Pilot raw hash/size does not match frozen discovery')
                        reused_pilot = True
                    else:
                        raw, final, mime = fetch.get(source['url'])
                        if not raw.startswith(b'%PDF'):
                            attachment = linked_pdf(raw, final)
                            raw, final, mime = fetch.get(attachment)
                    if not official(final) or not raw.startswith(b'%PDF'):
                        raise ValueError('Official attachment did not return a BOEM PDF')
                    docid = digest(source['url'])[:20]
                    path = DATA / 'raw' / (docid + '.pdf')
                    if path.exists() and digest(path.read_bytes()) != digest(raw):
                        raise ValueError('Existing raw PDF cannot be overwritten')
                    if not path.exists():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(raw)
                    doc = dict(source, document_id=docid, final_url=final, content_type=mime,
                               raw_path=path.relative_to(DATA).as_posix(), raw_sha256=digest(raw), bytes=len(raw),
                               fetched_at=now(), body_parser_sha256=parser_sha,
                               raw_acquisition='verified_publisher_pdf_pilot_reuse' if reused_pilot else 'official_http_download',
                               evidence_type='environmental_assessment_scenarios_and_leasing_policy_not_realized_output')
                    try:
                        parsed = parse_boem_pdf(raw, source)
                        pages, margin_audit = remove_margin_furniture(parsed['pages'])
                        cache = DATA / 'extracted' / (doc['raw_sha256'] + extraction_suffix(doc['source']))
                        if cache.exists() and json.loads(cache.read_text(encoding='utf-8')) != pages:
                            raise ValueError('Existing PDF extraction cache differs; refuse overwrite')
                        if not cache.exists():
                            write_json(cache, pages)
                        audit_path = target / 'pdf_extraction_audits' / (docid + '.json')
                        audit_record = dict(raw_sha256=doc['raw_sha256'], parser_sha256=parser_sha,
                            source_pdf_pages_sha256=digest(json.dumps(parsed['pages'], ensure_ascii=False)),
                            cache_sha256=digest(cache.read_bytes()), physical_pages=len(pages),
                            pages_with_removed_margin_lines=margin_audit,
                            removal_policy='recognized_exact_repeated_margin_only; no body rewrites or synthetic facts')
                        if audit_path.exists() and json.loads(audit_path.read_text(encoding='utf-8')) != audit_record:
                            raise ValueError('Existing extraction audit differs; refuse overwrite')
                        if not audit_path.exists():
                            write_json(audit_path, audit_record)
                        doc.update(published_date=parsed['published_date'], available_by=parsed['available_by'],
                                   date_precision=parsed['date_precision'], body_date_verified=parsed['date_confirmed'],
                                   cover_month=parsed['cover_month'], pdf_creation_date=parsed['pdf_creation_date'],
                                   pdf_modified_date=parsed['pdf_modified_date'], physical_pages=len(pages),
                                   extraction_audit_path=audit_path.relative_to(DATA).as_posix(),
                                   extraction_audit_sha256=digest(audit_path.read_bytes()),
                                   body_parser=parsed['parser'], body_word_count=sum(len(words(p)) for p in pages),
                                   publication_version_caution=parsed.get('version_caution',
                                       'Scenario/assessment prose is not observed production; publication and event dates are separate.'))
                        qualified += 1
                        state = 'downloaded_version_checked_pdf'
                    except Exception as exc:
                        doc.update(ingest_exclusion_reason='unverified_pdf_publication_or_extraction',
                                   body_date_verified=False, body_parser_error=str(exc))
                        state = 'downloaded_excluded'
                    docs[source['url']] = doc
                    sources[source['url']] = {k: v for k, v in doc.items() if k not in
                        {'document_id', 'final_url', 'content_type', 'raw_path', 'raw_sha256', 'bytes', 'fetched_at'}}
                    write_jsonl(target / 'documents.jsonl', docs.values())
                    write_jsonl(target / 'sources.jsonl', sources.values())
                    added += 1
                    log.update(status=state, document_id=docid, raw_sha256=doc['raw_sha256'],
                               error=doc.get('body_parser_error'), physical_pages=doc.get('physical_pages'),
                               raw_acquisition=doc['raw_acquisition'])
                    print(f'BOEM PDF {attempts}/{limit}: {state}; {doc.get("physical_pages", 0)} physical pages', flush=True)
                except Exception as exc:
                    log.update(status='failed_no_bypass', error=str(exc))
                    print(f'BOEM PDF {attempts}/{limit}: {exc}', flush=True)
                log['actual_requested_urls'] = getattr(fetch, 'requested_urls', [])[request_start:]
                append_jsonl(target / 'pdf_collection.jsonl', log)
                if fetch.bytes >= 300 * 1024**2:
                    break
        finally:
            downloaded = fetch.bytes
            fetch.close()
        if (candidate_sha != digest(candidate_path.read_bytes()) or self_sha != digest(Path(__file__).read_bytes())
                or parser_sha != digest(Path(discover_boem_buffer.__file__).read_bytes())
                or fetcher_sha != digest(Path(bounded_boem_fetcher.__file__).read_bytes())
                or any(not path.exists() or digest(path.read_bytes()) != sha for path, sha in protected.items())):
            raise ValueError('Protected source/run/ledger/candidate/implementation changed during PDF collection')
        final_docs, final_sources = validate_manifest_pair(target)
        if final_docs[:len(old_docs)] != old_docs or final_sources[:len(old_sources)] != old_sources:
            raise ValueError('Prior supplemental records were not preserved')
        report = dict(at=now(), attempts=attempts, added_raw_documents=added,
            added_version_checked_pdf_documents=qualified, total_supplement_documents=len(final_docs),
            response_bytes=downloaded, candidate_file_sha256=candidate_sha, script_sha256=self_sha,
            parser_sha256=parser_sha, fetcher_sha256=fetcher_sha,
            prior_size_rejections_not_retried=sorted(prior_size_rejections),
            prior_protected_files_preserved=len(protected),
            prior_manifest_objects_preserved=True, external_model_calls=0, human_labels_created=0,
            accepted_passages='not_counted_until_unchanged_unified_build',
            limitations=['Version/date checks and PDF page count are not semantic approval or accepted passage count.',
                         'Tables, figures, references and unsupported paragraphs are judged by the unchanged unified builder.',
                         'Fetch deadlines cover normal transport activity, but native operating-system resolver latency can exceed Python timeout.'])
        # Repeating a failed batch need not increase the manifest size. Keep
        # every invocation receipt instead of overwriting that earlier failure.
        receipt = target / f'pdf_collection_report-{len(final_docs)}-{digest(json.dumps(report, sort_keys=True))[:16]}.json'
        if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8')) != report:
            raise ValueError('Existing PDF invocation receipt differs; refuse overwrite')
        if not receipt.exists():
            write_json(receipt, report)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(collect(args.run, args.candidates, args.limit), indent=2))
