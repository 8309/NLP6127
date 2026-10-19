"""Resumable official-source expansion. No model calls or automatic labels."""
import argparse
import calendar
from collections import Counter
from datetime import datetime
import io
import json
from pathlib import Path
import re
import shutil
from contextlib import contextmanager
from urllib.parse import quote, urljoin, urlparse

from bs4 import BeautifulSoup
from pypdf import PdfReader

from oil_evidence.corpus import Fetcher, build_corpus, collect, date_from_text
from oil_evidence.store import DATA, append_jsonl, config, digest, now, read_jsonl, write_json, write_jsonl

ROOT = DATA / "expansion_runs" / "official-2018-2026-v1"


def select_run(run_id):
    if (not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,80}',run_id)
            or re.fullmatch(r'con|prn|aux|nul|com[1-9]|lpt[1-9]',run_id)):
        raise ValueError('Invalid expansion run ID')
    global ROOT
    ROOT=DATA/'expansion_runs'/run_id
    return ROOT


@contextmanager
def pipeline_lock():
    """Exclusive local manifest writer; fail closed on a foreign or stale lock."""
    path=DATA/'.corpus_writer.lock'
    try:
        with path.open('x',encoding='utf-8') as f:
            import os
            json.dump(dict(pid=os.getpid(),at=now(),run=str(ROOT)),f)
    except FileExistsError as e:
        raise RuntimeError('Another pipeline writer or interrupted lock exists: '+str(path)) from e
    try:
        yield
    finally:
        path.unlink()


def verify_baseline_sources():
    """Later collection must not rewrite any source or raw file from this run's baseline."""
    for name,key in [('documents.jsonl','document_id'),('sources.jsonl','url')]:
        current={r[key]:r for r in read_jsonl(DATA/name)}
        for old in read_jsonl(ROOT/'baseline'/name):
            if current.get(old[key])!=old:
                raise ValueError('Baseline metadata changed: '+old[key])
            if name=='documents.jsonl' and digest((DATA/old['raw_path']).read_bytes())!=old['raw_sha256']:
                raise ValueError('Baseline raw file changed: '+old['document_id'])


def implementation_hashes():
    base=DATA.parent
    names=['config.toml','uv.lock','src/oil_evidence/corpus.py','scripts/expand_corpus.py','scripts/audit_expansion.py',
           'scripts/discover_us_oil.py','scripts/discover_international_oil.py','scripts/discover_eia_series.py',
           'src/oil_evidence/pdf_scope.py','scripts/validate_new_sources.py','scripts/discover_wb_documents.py',
           'scripts/apply_source_page_exclusions.py']
    return {n:digest((base/n).read_bytes()) for n in names if (base/n).exists()}
COUNTRIES = [
    'https://www.eia.gov/international/content/analysis/countries_long/Iran/',
    'https://www.eia.gov/international/content/analysis/countries_long/Canada/',
    'https://www.eia.gov/international/content/analysis/countries_long/Brazil/',
    'https://www.eia.gov/international/content/analysis/countries_long/Venezuela/',
    'https://www.eia.gov/international/content/analysis/countries_long/Nigeria/',
    'https://www.eia.gov/international/content/analysis/countries_long/russia/',
    'https://www.eia.gov/international/content/analysis/countries_long/Iraq/background.htm',
]


def snapshot():
    target = ROOT / 'baseline'
    if target.exists():
        manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['hashes'].items():
            if digest((target / name).read_bytes()) != expected:
                raise ValueError('Baseline backup changed: ' + name)
        if manifest.get('annotation_baseline_path') and digest(Path(manifest['annotation_baseline_path']).read_bytes())!=manifest['annotation_baseline_sha256']:
            raise ValueError('Original annotation baseline changed')
        return manifest
    import tempfile
    ROOT.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.baseline-',dir=ROOT))
    try:
        paths = ['passages.jsonl', 'documents.jsonl', 'sources.jsonl', 'reports/corpus.json',
                 'annotations/codex.jsonl', 'annotations/passage_labels.jsonl',
                 'annotations/passage_review.jsonl', 'reports/passage_annotation.json',
                 'index/manifest.json', 'index/vectors.npy', 'reports/corpus_expansion.json']
        hashes = {}
        for name in paths:
            if (DATA / name).exists():
                (temporary / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(DATA / name, temporary / name)
                hashes[name] = digest((temporary / name).read_bytes())
        label_report=json.loads((DATA/'reports/passage_annotation.json').read_text(encoding='utf-8'))
        if hashes['passages.jsonl']==label_report['corpus_sha256']:
            annotation_path=target/'passages.jsonl'
            annotation_file=temporary/'passages.jsonl'
        else:
            parent=json.loads((DATA/'reports/corpus_expansion.json').read_text(encoding='utf-8'))
            annotation_path=Path(parent.get('annotation_baseline_path') or (Path(parent['snapshot_root'])/'baseline/passages.jsonl'))
            annotation_file=annotation_path
        if digest(annotation_file.read_bytes())!=label_report['corpus_sha256']:
            raise ValueError('Cannot locate immutable original annotation corpus')
        for name in ['config.toml','uv.lock','pyproject.toml']:
            (temporary/'project').mkdir(exist_ok=True)
            shutil.copy2(DATA.parent/name,temporary/'project'/name)
            hashes['project/'+name]=digest((temporary/'project'/name).read_bytes())
        result = dict(at=now(), hashes=hashes, passages=len(read_jsonl(temporary / 'passages.jsonl')),
            annotation_baseline_path=str(annotation_path),annotation_baseline_sha256=label_report['corpus_sha256'])
        write_json(temporary / 'manifest.json', result)
        temporary.rename(target)
        return result
    finally:
        if temporary.exists():
            # Only remove the exact scratch directory created above, never the baseline.
            if temporary.resolve().parent!=ROOT.resolve() or not temporary.name.startswith('.baseline-'):
                raise RuntimeError('Refuse unsafe snapshot scratch cleanup')
            shutil.rmtree(temporary)


def source(url, title, kind, available, origin, precision='day', published=None):
    settings = config()['corpus']
    if not available or not settings['start_date'] <= available <= settings['end_date']:
        return None
    return dict(url=url, title=title, source=kind, published_date=published or (available if precision=='day' else None),
                available_by=available, date_precision=precision, date_source=origin,
                event_date=None, event_status='unlabelled',
                redistribution='eia_attribution' if kind.startswith('EIA') else 'review_required')


def twip_sources(raw, origin):
    soup = BeautifulSoup(raw, 'html.parser')
    rows = {}
    for a in soup.select('a[href]'):
        href = urljoin(origin, a['href'])
        m = re.search(r'/archive/(20\d{2})/(\d{6})/includes/analysis_print.php$', href)
        if not m:
            continue
        from datetime import datetime
        date = datetime.strptime(m[2], '%y%m%d').date().isoformat()
        row = source(href, a.get_text(' ', strip=True), 'EIA-TWIP', date, origin)
        if row:
            rows[href] = row
    return list(rows.values())


def country_sources(raw, origin):
    soup = BeautifulSoup(raw, 'html.parser')
    date = date_from_text(soup.get_text(' ', strip=True)[:220])
    rows = []
    for a in soup.select('a[href]'):
        href = a['href']
        if not href.lower().endswith('.pdf') or 'countries_long/' not in href:
            continue
        # The embedded article is rendered at the /international/ SPA root.
        href = urljoin('https://www.eia.gov/international/', href) if href.startswith('content/') else urljoin(origin, href)
        if a.get_text(' ', strip=True).upper() == 'PDF':
            available, precision = date, 'day'
        else:
            m = re.search(r'\b(20\d{2})\b', a.get_text())
            if not m:
                continue
            available, precision = m[1] + '-12-31', 'year_upper_bound'
        country = origin.split('countries_long/')[1].split('/')[0]
        row = source(href, f'EIA Country Analysis: {country} {available}', 'EIA-CAB', available, origin, precision)
        if row:
            rows.append(row)
    return rows


def worldbank_sources(raw, origin):
    soup = BeautifulSoup(raw, 'html.parser')
    rows = []
    for tr in soup.select('tr'):
        m = re.match(r'(20\d{2})\s*-\s*([A-Za-z]+)', tr.get_text(' ', strip=True))
        if not m:
            continue
        months = {name.lower(): n for n in range(1,13) for name in [calendar.month_name[n],calendar.month_abbr[n]]}
        month = months.get(m[2].lower())
        if not month:
            continue
        available = f'{m[1]}-{month:02d}-{calendar.monthrange(int(m[1]),month)[1]:02d}'
        for a in tr.select('a[href]'):
            if a.get_text(' ', strip=True) != 'PDF':
                continue
            href = urljoin(origin, a['href'])
            if urlparse(href).hostname not in {'openknowledge.worldbank.org','thedocs.worldbank.org','documents1.worldbank.org','documents.worldbank.org'}:
                continue  # Do not ingest opaque short-link redirects.
            row = source(href, f'World Bank Commodity Markets Outlook {m[1]}-{month:02d}', 'WB-CMO', available, origin,'month_upper_bound')
            if row:
                rows.append(row)
    return rows


def tie_sources(raw, origin):
    soup = BeautifulSoup(raw, 'html.parser')
    rows = []
    relevant = re.compile(r'\b(oil|crude|petroleum|opec|refiner\w*|tanker\w*|barrels?|sanctions?|hormuz|suez|red sea|gasoline|diesel|distillate|jet fuel|propane|ethane|hydrocarbon|pipeline|biofuel\w*|liquid fuels|drilled|drilling|wells)\b',re.I)
    for a in soup.select('a[href]'):
        href = urljoin(origin, a['href'])
        title = a.get_text(' ',strip=True)
        if 'detail.php?id=' not in href or not relevant.search(title):
            continue
        parent = a.find_parent('li')
        date = date_from_text(parent.get_text(' ',strip=True)) if parent else None
        row = source(href,title,'EIA-TIE',date,origin)
        if row:
            rows.append(row)
    return rows


def country_pages(registry):
    """Use the official site's public country registry, not guessed report URLs."""
    return [('https://www.eia.gov/international/content/analysis/countries_long/'
             + quote(row['country'], safe='_') + '/', country_sources)
            for row in registry if row.get('longAnalysis') is True
            and isinstance(row.get('country'), str)
            and re.fullmatch(r'[A-Za-z_-]+', row['country'])]


class RecordingFetcher:
    def __init__(self):
        self.fetch=Fetcher()

    def get(self,url):
        try:
            raw,final,content_type=self.fetch.get(url)
            ext='.pdf' if raw.startswith(b'%PDF') else '.json' if 'json' in content_type else '.html'
            path=ROOT/'discovery'/(digest(raw)+ext)
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                path.write_bytes(raw)
            append_jsonl(ROOT/'discovery_log.jsonl',dict(at=now(),url=url,final_url=final,
                raw_sha256=digest(raw),raw_path=str(path),status='ok'))
            return raw,final,content_type
        except Exception as e:
            append_jsonl(ROOT/'discovery_log.jsonl',dict(at=now(),url=url,error=str(e),status='failed'))
            raise

    def close(self):
        self.fetch.close()


def discover_extra(eia_only=False):
    """Adapters discover actual first-party links; one serial, auditable fetcher."""
    snapshot()
    rows={r['url']:r for r in read_jsonl(DATA/'sources.jsonl')}
    fetch=RecordingFetcher()
    if eia_only:
        import discover_eia_series
        adapters=[discover_eia_series]
    else:
        import discover_us_oil,discover_international_oil,discover_wb_documents
        adapters=[discover_us_oil,discover_international_oil,discover_wb_documents]
    added=0
    try:
        for adapter in adapters:
            for row in adapter.discover(fetch):
                available=row.get('available_by')
                if not available or not config()['corpus']['start_date']<=available<=config()['corpus']['end_date']:
                    continue
                if row['url'] not in rows:
                    rows[row['url']]=row
                    added+=1
            write_jsonl(DATA/'sources.jsonl',rows.values())
    finally:
        fetch.close()
    result=dict(at=now(),new_sources=added,sources=len(rows),by_source=dict(Counter(r['source'] for r in rows.values())),external_model_calls=0)
    append_jsonl(ROOT/'source_discovery_runs.jsonl',result)
    return result


def discover(worldbank_only=False, tie_only=False, countries_only=False):
    snapshot()
    rows = {r['url']: r for r in read_jsonl(DATA / 'sources.jsonl')}
    fetch = Fetcher()
    pages = [('https://www.eia.gov/petroleum/weekly/', twip_sources)] + [(u, country_sources) for u in COUNTRIES]
    pages += [('https://www.worldbank.org/en/research/commodity-markets/report-archive', worldbank_sources)]
    if worldbank_only:
        pages = pages[-1:]
    if tie_only:
        pages = [(f'https://www.eia.gov/todayinenergy/archive.php?my={year}',tie_sources) for year in range(2018,2027)]
    if countries_only:
        registry_url = 'https://www.eia.gov/international/content/api/analysis/index.php?content=countries'
        raw, final, _ = fetch.get(registry_url)
        path = ROOT / 'discovery' / (digest(raw) + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        registry = json.loads(raw)
        pages = country_pages(registry)
        append_jsonl(ROOT / 'discovery_log.jsonl', dict(at=now(), url=registry_url,
            final_url=final, raw_sha256=digest(raw), raw_path=str(path),
            countries=len(registry), long_analysis_pages=len(pages), status='ok'))
    for url, parser in pages:
        try:
            raw, final, _ = fetch.get(url)
            path = ROOT / 'discovery' / (digest(raw) + '.html')
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(raw)
            found = parser(raw, final)
            for row in found:
                rows.setdefault(row['url'], row)
            append_jsonl(ROOT / 'discovery_log.jsonl', dict(at=now(), url=url, final_url=final,
                raw_path=str(path), raw_sha256=digest(raw), candidates=len(found), status='ok'))
            write_jsonl(DATA / 'sources.jsonl', rows.values())
            print(f'Discovered {len(found)} candidates from {url}', flush=True)
        except Exception as e:
            append_jsonl(ROOT / 'discovery_log.jsonl', dict(at=now(), url=url, error=str(e), status='failed'))
            print(f'Skipped {url}: {e}', flush=True)
    fetch.close()
    result = dict(at=now(), sources=len(rows), by_source=dict(Counter(r['source'] for r in rows.values())),
        opec='Not bulk-collected: terms permit occasional/infrequent educational use; chapter returned 403.',
        new_paragraph_labels_generated=0, external_model_calls=0)
    write_json(ROOT / 'discovery_report.json', result)
    return result


def stage():
    snapshot()
    verify_baseline_sources()
    old_ids={d['document_id'] for d in read_jsonl(ROOT/'baseline/documents.jsonl')}
    new_docs=[d for d in read_jsonl(DATA/'documents.jsonl') if d['document_id'] not in old_ids]
    if any(d.get('requires_pdf') or d['source']=='BSEE-HURRICANE' for d in new_docs):
        version_path=ROOT/'new_source_version_audit.json'
        if not version_path.exists():
            raise ValueError('New-source version audit missing; validate before staging')
        version=json.loads(version_path.read_text(encoding='utf-8'))
        if (version['documents_sha256']!=digest((DATA/'documents.jsonl').read_bytes()) or
            version['sources_sha256']!=digest((DATA/'sources.jsonl').read_bytes()) or
            version['validator_sha256']!=digest((DATA.parent/'scripts/validate_new_sources.py').read_bytes()) or
            {r['document_id'] for r in version['records']}!={d['document_id'] for d in new_docs}):
            raise ValueError('New-source version audit missing or stale; validate before staging')
    baseline = read_jsonl(ROOT / 'baseline/passages.jsonl')
    report = build_corpus(output_dir=ROOT / 'stage', baseline=baseline,
        baseline_document_ids={d['document_id'] for d in read_jsonl(ROOT/'baseline/documents.jsonl')})
    current = read_jsonl(ROOT / 'stage/passages.jsonl')
    assert current[:len(baseline)] == baseline
    assert len({r['passage_id'] for r in current}) == len(current)
    fatal = [e for e in report['extraction_errors'] if not e['error'].startswith('No recognized article body')]
    assert not fatal, fatal
    report['quarantined_documents'] = report['extraction_errors']
    report.update(new_passages=len(current)-len(baseline), baseline_sha256=digest((ROOT/'baseline/passages.jsonl').read_bytes()),
                  labels_coverage=len(read_jsonl(DATA/'annotations/passage_labels.jsonl')),
                  external_model_calls=0, human_labels_created=0)
    report['quarantined_new_version_documents']=[dict(document_id=d['document_id'],reason=d['ingest_exclusion_reason']) for d in new_docs if d.get('ingest_exclusion_reason')]
    write_json(ROOT / 'stage/reports/corpus.json', report)
    return report


def refine_dates():
    """Use only explicit release/update dates from newly collected PDF covers."""
    baseline_ids = {d['document_id'] for d in read_jsonl(ROOT/'baseline/documents.jsonl')}
    docs = read_jsonl(DATA/'documents.jsonl')
    sources = {s['url']:s for s in read_jsonl(DATA/'sources.jsonl')}
    changes=[]
    for d in docs:
        if d['document_id'] in baseline_ids or d['source'] != 'EIA-CAB':
            continue
        path=DATA/d['raw_path']
        if digest(path.read_bytes()) != d['raw_sha256']:
            raise ValueError('Raw source changed')
        text=PdfReader(io.BytesIO(path.read_bytes())).pages[0].extract_text() or ''
        # Some PDFs split month names ("Decem ber") in their text layer.
        for month in calendar.month_name[1:]:
            text=re.sub(r'\b'+r'\s*'.join(month)+r'\b',month,text,flags=re.I)
        m=re.search(r'(?:Last\s+Updated|Update)\s*:?\s*([A-Za-z]+\s+(?:\d{1,2},?\s+)?20\d{2})',text,re.I)
        exact=date_from_text(m[0]) if m else None
        precision='day'
        if not exact and m:
            try:
                month=datetime.strptime(' '.join(m[1].split()),'%B %Y')
                exact=f'{month.year}-{month.month:02d}-{calendar.monthrange(month.year,month.month)[1]:02d}'
                precision='month_upper_bound'
            except ValueError:
                pass
        if not exact:
            continue
        if not config()['corpus']['start_date']<=exact<=config()['corpus']['end_date']:
            raise ValueError('New PDF release outside allowed window')
        if d['available_by'] != exact or d['date_precision']!=precision:
            original={k:d.get(k) for k in ['available_by','published_date','date_precision','date_source']}
            replacement=dict(available_by=exact,published_date=exact if precision=='day' else None,date_precision=precision,date_source=d['url']+'#page=1')
            d.setdefault('original_date_metadata',original)
            d.update(replacement)
            sources[d['url']].setdefault('original_date_metadata',original)
            sources[d['url']].update(replacement)
            changes.append(dict(document_id=d['document_id'],original=original,replacement=replacement,
                raw_sha256=d['raw_sha256'],quote=m[0],human_reviewed=False))
    write_jsonl(DATA/'documents.jsonl',docs)
    write_jsonl(DATA/'sources.jsonl',sources.values())
    append_jsonl(ROOT/'date_refinement_runs.jsonl',dict(at=now(),changes=changes,method='Explicit PDF cover Last Updated date; event dates never substituted'))
    return dict(refined=len(changes),unchanged_baseline_documents=len(baseline_ids))


def index():
    from oil_evidence import retrieval
    target = ROOT / 'stage/index'
    target.mkdir(parents=True, exist_ok=True)
    for name in ['embedding_cache.sqlite3', 'manifest.json', 'vectors.npy']:
        if not (target/name).exists() and (DATA/'index'/name).exists():
            shutil.copy2(DATA/'index'/name, target/name)
    retrieval.DATA = ROOT / 'stage'
    return retrieval.build_index()


def publish():
    manifest = snapshot()
    verify_baseline_sources()
    staged = ROOT / 'stage'
    report = json.loads((staged/'reports/corpus.json').read_text(encoding='utf-8'))
    quality = json.loads((staged/'reports/expansion_quality.json').read_text(encoding='utf-8'))
    if quality['corpus_sha256'] != report['corpus_sha256'] or not quality['new_passages_page_provenance_passed']:
        raise ValueError('Source-integrity audit missing or stale')
    if quality.get('documents_sha256')!=digest((DATA/'documents.jsonl').read_bytes()) or quality.get('sources_sha256')!=digest((DATA/'sources.jsonl').read_bytes()) or quality.get('implementation_hashes')!=implementation_hashes():
        raise ValueError('Source manifests or processing implementation changed after audit')
    if digest((DATA/'passages.jsonl').read_bytes())!=manifest['hashes']['passages.jsonl']:
        raise ValueError('Live corpus changed after run snapshot; refuse lost-update publication')
    meta = json.loads((staged/'index/manifest.json').read_text(encoding='utf-8'))
    if digest((staged/'passages.jsonl').read_bytes()) != report['corpus_sha256'] or meta['corpus_sha256'] != report['corpus_sha256']:
        raise ValueError('Staged corpus/index mismatch')
    if digest((staged/'index/vectors.npy').read_bytes()) != meta['vectors_sha256']:
        raise ValueError('Staged vector hash mismatch')
    rows = read_jsonl(staged/'passages.jsonl')
    import numpy as np
    vectors=np.load(staged/'index/vectors.npy',allow_pickle=False)
    if meta['passage_ids'] != [p['passage_id'] for p in rows] or vectors.shape!=(len(rows),meta['dimension']) or not np.isfinite(vectors).all():
        raise ValueError('Index order, shape, or numerical integrity mismatch')
    old = read_jsonl(ROOT/'baseline/passages.jsonl')
    if rows[:len(old)] != old or len(rows) <= len(old):
        raise ValueError('Expansion must preserve every baseline passage and add new sources')
    for name in ['annotations/codex.jsonl','annotations/passage_labels.jsonl','annotations/passage_review.jsonl']:
        if digest((DATA/name).read_bytes()) != manifest['hashes'][name]:
            raise ValueError('Annotation changed during expansion: '+name)
    # A later staging run must never erase the evidence for a published corpus.
    frozen = ROOT / 'published' / report['corpus_sha256']
    for name in ['passages.jsonl', 'reports/corpus.json', 'index/manifest.json',
                 'reports/expansion_quality.json','reports/near_duplicate_candidates.jsonl','reports/source_qa_samples.jsonl']:
        dest = frozen/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and digest(dest.read_bytes()) != digest((staged/name).read_bytes()):
            raise ValueError('Published snapshot already exists with different contents: '+name)
        if not dest.exists():
            shutil.copy2(staged/name, dest)
    for name in ['documents.jsonl','sources.jsonl']:
        dest = frozen/name
        if dest.exists() and digest(dest.read_bytes()) != digest((DATA/name).read_bytes()):
            raise ValueError('Published source manifest changed: '+name)
        if not dest.exists():
            shutil.copy2(DATA/name,dest)
    run_manifest=ROOT/'baseline/manifest.json'
    frozen_manifest=frozen/'run_baseline_manifest.json'
    if frozen_manifest.exists() and digest(frozen_manifest.read_bytes())!=digest(run_manifest.read_bytes()):
        raise ValueError('Published run baseline manifest changed')
    if not frozen_manifest.exists():
        shutil.copy2(run_manifest,frozen_manifest)
    if (ROOT/'new_source_version_audit.json').exists():
        src=ROOT/'new_source_version_audit.json'
        dst=frozen/src.name
        if dst.exists() and digest(dst.read_bytes())!=digest(src.read_bytes()):
            raise ValueError('Published source-version audit changed')
        if not dst.exists():
            shutil.copy2(src,dst)
    live_files=['passages.jsonl','reports/corpus.json','index/vectors.npy','index/manifest.json','index/embedding_cache.sqlite3',
                 'reports/expansion_quality.json','reports/near_duplicate_candidates.jsonl','reports/source_qa_samples.jsonl']
    # Prepare every file before replacing any live file, so a copy/permission or
    # space failure cannot leave the live corpus paired with its old index.
    for name in live_files:
        dest = DATA/name
        tmp = dest.with_suffix(dest.suffix+'.expansion.tmp')
        shutil.copy2(staged/name, tmp)
    for name in live_files:
        dest = DATA/name
        tmp = dest.with_suffix(dest.suffix+'.expansion.tmp')
        tmp.replace(dest)
    write_json(DATA/'reports/corpus_expansion.json', dict(at=now(), **{k:report[k] for k in ['passages','new_passages','corpus_sha256','baseline_sha256','labels_coverage','meets_size_requirement']},
        unlabelled_new_passages=report['new_passages'], snapshot_root=str(ROOT),
        published_snapshot=str(frozen), retained_machine_labels=report['labels_coverage'],
        unlabelled_corpus_passages=report['passages']-report['labels_coverage'],
        annotation_baseline_path=manifest.get('annotation_baseline_path',str(ROOT/'baseline/passages.jsonl')),
        annotation_baseline_sha256=manifest.get('annotation_baseline_sha256',manifest['hashes']['passages.jsonl']),
        human_labels_created=0, external_model_calls=0, stage='published_with_rebuilt_index'))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['snapshot','discover','discover-worldbank','discover-tie','discover-countries','discover-extra','discover-eia','collect','refine_dates','stage','index','publish'])
    parser.add_argument('--run',default='official-2018-2026-v1')
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--source')
    parser.add_argument('--url-prefix')
    args = parser.parse_args()
    select_run(args.run)
    with pipeline_lock():
        snapshot()
        verify_baseline_sources()
        result = collect(args.limit, args.source, args.url_prefix) if args.action=='collect' else discover(worldbank_only=True) if args.action=='discover-worldbank' else discover(tie_only=True) if args.action=='discover-tie' else discover(countries_only=True) if args.action=='discover-countries' else discover_extra(eia_only=True) if args.action=='discover-eia' else discover_extra() if args.action=='discover-extra' else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
