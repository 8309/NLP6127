"""Append-only raw collection into a separate run manifest, not the live corpus."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import date
import json
import importlib
import os
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse
import uuid

import httpx

from discover_doe_spr_news import parse_doe_body
from discover_unified_sources import parse_treasury_body
from oil_evidence.corpus import Fetcher, extraction_suffix
from oil_evidence.segmentation import oil_scope_supported
from oil_evidence.store import DATA, append_jsonl, digest, now, read_jsonl, words, write_json, write_jsonl


EXTRA_ADAPTERS = {
    'EIA-OIL-NEWS': ('discover_eia_buffer', 'admitted_eia_route', 'parse_eia_body'),
    'CER-OIL-SNAPSHOT': ('discover_official_buffer', 'admitted_cer_route', 'parse_cer_body'),
    'DALLAS-OIL': ('discover_dallas_buffer', 'admitted_dallas_route', 'parse_dallas_body'),
}


def adapter(source):
    entry = EXTRA_ADAPTERS.get(source.get('source'))
    return (importlib.import_module(entry[0]), entry) if entry else (None, None)


def validate_manifest_pair(target):
    paths = [target / name for name in ['documents.jsonl', 'sources.jsonl']]
    if paths[0].exists() != paths[1].exists():
        raise ValueError('Supplement requires a complete manifest pair; refuse crash recovery by guessing')
    rows, sources = (read_jsonl(path) for path in paths)
    if len({r['url'] for r in rows}) != len(rows) or len({r['url'] for r in sources}) != len(sources):
        raise ValueError('Duplicate URL in supplement manifest')
    if len(rows) != len(sources) or {r['url'] for r in rows} != {r['url'] for r in sources}:
        raise ValueError('Supplement source/document manifest mismatch')
    by_url = {r['url']: r for r in sources}
    keys = ['source', 'title', 'published_date', 'available_by', 'date_precision',
            'date_source', 'event_date', 'event_status', 'redistribution']
    for row in rows:
        if any(row.get(k) != by_url[row['url']].get(k) for k in keys):
            raise ValueError('Supplement source/document provenance mismatch')
    return rows, sources


def prior_protections():
    """Bind every prior raw/cache and completed v1 artifact, without secrets."""
    protected = {}
    roots = [DATA / 'raw', DATA / 'extracted', DATA / 'unified_runs/semantic-page-v1']
    for root in roots:
        if root.exists():
            for path in root.rglob('*'):
                if path.is_file():
                    protected[path] = digest(path.read_bytes())
    return protected


def normalized_title(title):
    if not isinstance(title, str) or not title.strip():
        raise ValueError('A source title is required')
    return unicodedata.normalize('NFKC', ' '.join(title.split())).casefold()


def admitted_route(source, url=None):
    """Exact official host and article route, never string-prefix host trust."""
    target = urlparse(url or source['url'])
    if target.scheme != 'https' or target.username or target.password or target.port not in {None, 443}:
        raise ValueError('Supplement requires an uncredentialed HTTPS official source')
    module, entry = adapter(source)
    if module:
        getattr(module, entry[1])(url or source['url'])
        return source['source']
    if source.get('source') == 'TREASURY-OIL':
        allowed = target.hostname == 'home.treasury.gov' and re.fullmatch(
            r'/news/press-releases/[a-z0-9-]+/?', target.path)
    elif source.get('source') == 'DOE-SPR-NEWS':
        allowed = target.hostname == 'www.energy.gov' and re.fullmatch(
            r'/(?:[a-z0-9-]+/)*articles/[a-z0-9-]+/?', target.path)
    else:
        allowed = False
    if not allowed:
        raise ValueError('Supplement requires an explicitly admitted official source-specific article route')
    return source['source']


def parse_source(source, raw):
    module, entry = adapter(source)
    if module:
        parsed = getattr(module, entry[2])(raw)
        normalize = module.normalized_cer_title if source['source'] == 'CER-OIL-SNAPSHOT' else normalized_title
        parsed['body_title_verified'] = normalize(parsed['title']) == normalize(source['title'])
        # Titles are not enough to admit a source as oil prose.
        if 'scope_text' not in parsed:
            parsed['scope_text'] = '\n\n'.join(block for block in parsed['blocks']
                if normalize(block) != normalize(parsed['title']))
        parsed['parser'] = entry[0] + '_source_native_v1'
        parsed['parser_sha256'] = digest(Path(module.__file__).read_bytes())
        return parsed
    if source['source'] == 'TREASURY-OIL':
        exact, body, blocks = parse_treasury_body(raw)
        return dict(published_date=exact, text=body, blocks=blocks, title=None,
                    body_title_verified=None, parser='official_treasury_article_v1')
    if source['source'] == 'DOE-SPR-NEWS':
        parsed = parse_doe_body(raw)
        parsed['body_title_verified'] = normalized_title(parsed['title']) == normalized_title(source['title'])
        parsed['parser'] = 'official_doe_article_local_date_v1'
        return parsed
    raise ValueError('No parser for unadmitted supplement source')


class SupplementFetcher(Fetcher):
    """Validate every article redirect and prevent nested retry amplification."""
    def __init__(self):
        super().__init__()
        self.article_source = None
        self.client.timeout = httpx.Timeout(45, connect=12)

    def get(self, url, _redirects=0):
        if self.article_source is not None:
            admitted_route(self.article_source, url)
        try:
            return super().get(url, _redirects)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if _redirects:
                raise RuntimeError('Article redirect target exhausted bounded transient attempts: ' + url) from exc
            raise


@contextmanager
def writer_lock(run):
    """Coordinate with unify_corpus using its exact shared writer lock path."""
    path = DATA / '.corpus_writer.lock'
    payload = json.dumps(dict(pid=os.getpid(), run=run, at=now(),
                              purpose='collect_unified_supplement', owner_token=uuid.uuid4().hex))
    # Existing (including apparently stale) locks are never removed or reused.
    with path.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    identity = path.stat().st_ino
    try:
        yield
    finally:
        if not path.exists():
            raise RuntimeError('Shared corpus writer lock disappeared during collection')
        if path.stat().st_ino != identity or path.read_text(encoding='utf-8') != payload:
            raise RuntimeError('Shared corpus writer lock ownership changed; replacement lock is preserved')
        path.unlink()


def collect(run, candidates, limit):
    with writer_lock(run):
        return _collect_unlocked(run, candidates, limit)


def _collect_unlocked(run, candidates, limit):
    if not 1 <= limit <= 200:
        raise ValueError('Collection limit must be 1..200')
    root = (DATA / 'unified_runs' / run).resolve()
    if root.parent != (DATA / 'unified_runs').resolve():
        raise ValueError('Invalid run path')
    if (root / 'published').exists():
        raise ValueError('Cannot add sources to a published immutable run')
    protected = ['sources.jsonl', 'documents.jsonl', 'passages.jsonl', 'index/manifest.json', 'index/vectors.npy',
                 'annotations/codex.jsonl', 'annotations/passage_labels.jsonl', 'annotations/passage_review.jsonl',
                 'reports/corpus.json', 'reports/corpus_resegmentation.json', 'reports/passage_annotation.json',
                 'model_usage.sqlite3', 'annotations.sqlite3']
    before = {name: digest((DATA / name).read_bytes()) for name in protected if (DATA / name).exists()}
    prior_files = prior_protections()
    script_hash = digest(Path(__file__).read_bytes())
    candidates = Path(candidates).resolve()
    candidate_hash = digest(candidates.read_bytes())
    target = root / 'supplement'
    old_docs, old_sources = validate_manifest_pair(target)
    docs = {d['url']: d for d in old_docs}
    sources = {s['url']: s for s in old_sources}
    known = {d['url'].rstrip('/') for d in read_jsonl(DATA / 'documents.jsonl')}
    rows = read_jsonl(candidates)
    fetch = SupplementFetcher()
    if fetch.settings['request_delay_seconds'] < 1 or fetch.settings['max_document_mb'] > 25 or fetch.settings['download_budget_mb'] > 300:
        fetch.close()
        raise ValueError('Fetcher must retain 1 second/host, 25 MB/document, 300 MB/invocation limits')
    attempts = added = 0
    prior_count = len(docs)
    attempted_sources = Counter()
    try:
        # Direct oil title first, then country/network candidates requiring body
        # screening. Stable order; no query/retrieval outcomes inform selection.
        ordered = sorted(rows, key=lambda r: (r.get('title_match') != 'direct_oil', r['available_by'], r['url']))
        for source in ordered:
            if attempts >= limit:
                break
            if source['url'] in docs or source['url'].rstrip('/') in known:
                continue
            admitted_route(source)
            exact_index_date = source.get('published_date')
            available_index = source.get('available_by')
            if (not isinstance(exact_index_date, str) or date.fromisoformat(exact_index_date).isoformat() != exact_index_date
                    or not '2018-01-01' <= exact_index_date <= '2026-10-05'
                    or not isinstance(available_index, str) or date.fromisoformat(available_index).isoformat() != available_index
                    or available_index < exact_index_date or source.get('date_precision') != 'day'):
                raise ValueError('Supplement requires a verified official publication day inside the fixed corpus window')
            attempts += 1
            attempted_sources[source['source']] += 1
            record = dict(at=now(), url=source['url'], candidate_file_sha256=candidate_hash)
            try:
                fetch.article_source = source
                raw, final, mime = fetch.get(source['url'])
                admitted_route(source, final)
                docid = digest(source['url'])[:20]
                path = DATA / 'raw' / (docid + '.html')
                if path.exists() and digest(path.read_bytes()) != digest(raw):
                    raise ValueError('Existing raw source cannot be overwritten')
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                parser_error = None
                try:
                    parsed = parse_source(source, raw)
                except Exception as exc:
                    parser_error = str(exc)
                    parsed = dict(published_date=None, text='', blocks=[], title=None,
                                  body_title_verified=False, parser='source_parser_failed_no_inference')
                exact, body = parsed['published_date'], parsed['text']
                article_available = parsed.get('available_by', exact)
                scope_text = parsed.get('scope_text', body)
                exclusion = None
                if parser_error:
                    exclusion = 'unverified_article_body_or_publication_date'
                elif exact != source['published_date']:
                    exclusion = 'article_date_conflicts_with_official_index'
                elif parsed['body_title_verified'] is False:
                    exclusion = 'article_title_conflicts_with_official_index'
                elif (not isinstance(article_available, str)
                      or date.fromisoformat(article_available).isoformat() != article_available
                      or article_available < exact or not '2018-01-01' <= article_available <= '2026-10-05'):
                    exclusion = 'current_article_version_outside_corpus_window_or_date_unknown'
                elif not oil_scope_supported(scope_text):
                    exclusion = 'no_strong_mineral_oil_body_evidence'
                doc = dict(source, document_id=docid, final_url=final, content_type=mime,
                           raw_path=path.relative_to(DATA).as_posix(), raw_sha256=digest(raw), bytes=len(raw), fetched_at=now(),
                           body_date_verified=exact == source['published_date'], body_oil_verified=oil_scope_supported(scope_text),
                           body_title_verified=parsed['body_title_verified'], article_title=parsed['title'],
                           body_parser=parsed['parser'], article_date=exact, body_word_count=len(words(body)))
                if parsed.get('available_by'):
                    doc.update(available_by=article_available, article_modified_date=parsed.get('modified_date'),
                               availability_date_source=final,
                               availability_method=('max_verified_release_and_visible_revision_day' if parsed.get('modified_date')
                                                    else 'source_native_release_no_later_revision_observed'),
                               article_version_caution=parsed.get('version_caution'))
                for key in ['evidence_type', 'survey_period', 'survey_release_header']:
                    if parsed.get(key):
                        doc[key] = parsed[key]
                if parsed.get('version_caution'):
                    doc['publication_version_caution'] = parsed['version_caution']
                if parsed.get('structured_blocks'):
                    doc['source_block_attribution'] = parsed['structured_blocks']
                if parsed.get('parser_sha256'):
                    doc['body_parser_sha256'] = parsed['parser_sha256']
                if parser_error:
                    doc['body_parser_error'] = parser_error
                if exclusion:
                    doc['ingest_exclusion_reason'] = exclusion
                cache = DATA / 'extracted' / (doc['raw_sha256'] + extraction_suffix(source['source']))
                if cache.exists():
                    if json.loads(cache.read_text(encoding='utf-8')) != [body]:
                        raise ValueError('Existing extraction cache differs; refuse overwrite')
                else:
                    write_json(cache, [body])
                docs[source['url']] = doc
                write_jsonl(target / 'documents.jsonl', docs.values())
                sources[source['url']] = {k: v for k, v in doc.items() if k not in
                    {'document_id', 'final_url', 'content_type', 'raw_path', 'raw_sha256', 'bytes', 'fetched_at'}}
                write_jsonl(target / 'sources.jsonl', sources.values())
                added += 1
                record.update(status='downloaded_excluded' if exclusion else 'downloaded_dated_oil_body',
                              document_id=docid, raw_sha256=doc['raw_sha256'], exclusion=exclusion,
                              parser_error=parser_error, body_words=doc['body_word_count'])
                print(f'Supplement {attempts}/{limit}: {doc["body_word_count"]} body words; {exclusion or "date/scope checks passed"}', flush=True)
            except Exception as exc:
                record.update(status='failed', error=str(exc))
                print(f'Supplement attempt {attempts}: {exc}', flush=True)
            append_jsonl(target / 'collection.jsonl', record)
            if fetch.bytes >= fetch.settings['download_budget_mb'] * 1024**2:
                break
    finally:
        downloaded = fetch.bytes
        fetch.close()
    if (candidate_hash != digest(candidates.read_bytes())
            or script_hash != digest(Path(__file__).read_bytes())
            or any(digest((DATA / name).read_bytes()) != sha for name, sha in before.items())
            or any(not path.exists() or digest(path.read_bytes()) != sha for path, sha in prior_files.items())):
        raise ValueError('Live protected files or candidate manifest changed during collection')
    final_docs, final_sources = validate_manifest_pair(target)
    if final_docs[:len(old_docs)] != old_docs or final_sources[:len(old_sources)] != old_sources:
        raise ValueError('Prior supplemental records were changed rather than appended')
    report = dict(at=now(), attempts=attempts, added_raw_documents=added, total_supplement_documents=len(docs),
                  prior_supplement_documents=prior_count, attempted_by_source=dict(attempted_sources),
                  dated_scope_accepted=sum(not d.get('ingest_exclusion_reason') for d in docs.values()),
                  excluded_downloads=sum(bool(d.get('ingest_exclusion_reason')) for d in docs.values()),
                  by_source=dict(Counter(d['source'] for d in docs.values())),
                  accepted_by_source=dict(Counter(d['source'] for d in docs.values() if not d.get('ingest_exclusion_reason'))),
                  candidate_file_sha256=candidate_hash, candidate_file=str(candidates),
                  documents_sha256=digest((target / 'documents.jsonl').read_bytes()) if (target / 'documents.jsonl').exists() else None,
                  sources_sha256=digest((target / 'sources.jsonl').read_bytes()) if (target / 'sources.jsonl').exists() else None,
                  live_hashes_before_and_after=before, prior_raw_cache_run_files_preserved=len(prior_files),
                  prior_manifest_objects_preserved=True, script_sha256=script_hash,
                  downloaded_bytes=downloaded, live_protected_files_unchanged=True, external_model_calls=0)
    append_jsonl(target / 'collection_reports.jsonl', report)
    write_json(target / f'collection_report-{len(docs)}.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='semantic-page-v1')
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(collect(args.run, args.candidates, args.limit), indent=2))
