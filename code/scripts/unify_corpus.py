"""Staged full-source rebuild, checks, local index and guarded publication.

Unlike the historical resegmentation, count is a report, not a release target.
Never reuse an already published run for later collection: choose a new run ID.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np

from oil_evidence.store import DATA, ROOT as PROJECT, digest, now, read_jsonl, words, write_json, write_jsonl
from oil_evidence.unified_corpus import (PROVENANCE, OPTIONAL_PROVENANCE,
                                       optional_provenance, source_native_context,
                                       passage_source_context, build_unified_corpus)

VERSION = 'semantic-page-v1'
RUN = DATA / 'unified_runs' / VERSION
INPUTS = ['passages.jsonl', 'documents.jsonl', 'sources.jsonl', 'reports/corpus.json',
          'reports/corpus_resegmentation.json', 'reports/corpus_unified.json',
          'reports/passage_annotation.json', 'annotations/codex.jsonl',
          'annotations/passage_labels.jsonl', 'annotations/passage_review.jsonl',
          'index/manifest.json', 'index/vectors.npy']
CODE = ['config.toml', 'uv.lock', 'pyproject.toml', 'src/oil_evidence/store.py',
        'src/oil_evidence/corpus.py', 'src/oil_evidence/unified_corpus.py',
        'src/oil_evidence/segmentation.py', 'src/oil_evidence/retrieval.py',
        'src/oil_evidence/cli.py', 'src/oil_evidence/corpus_versions.py', 'app.py',
        'scripts/unify_corpus.py', 'scripts/compare_segmentation.py',
        'scripts/reproduce_unified.py', 'scripts/check_workbench.py',
        'scripts/reprocess_boem_supplement.py', 'scripts/discover_boem_buffer.py',
        'scripts/collect_boem_supplement.py', 'scripts/bounded_boem_fetcher.py']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def select_run(value):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,80}', value) or re.fullmatch(r'con|prn|aux|nul|com[1-9]|lpt[1-9]', value):
        raise ValueError('Invalid run ID')
    global RUN
    RUN = DATA / 'unified_runs' / value
    return RUN


def contained(value):
    path = (DATA / value).resolve()
    if not path.is_relative_to(DATA.resolve()):
        raise ValueError('Path escapes data directory')
    return path


@contextmanager
def lock():
    path = DATA / '.corpus_writer.lock'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), run=str(RUN), at=now()), stream)
    try:
        yield
    finally:
        path.unlink()


def hashes():
    return {name: digest((PROJECT / name).read_bytes()) for name in CODE if (PROJECT / name).exists()}


def snapshot():
    baseline = RUN / 'baseline'
    if baseline.exists():
        manifest = load(baseline / 'manifest.json')
        for name, expected in manifest['hashes'].items():
            if digest((baseline / name).read_bytes()) != expected:
                raise ValueError('Frozen baseline modified: ' + name)
        return manifest
    RUN.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='.baseline-', dir=RUN))
    copied = {}
    for name in INPUTS:
        if (DATA / name).exists():
            dest = scratch / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(DATA / name, dest)
            copied[name] = digest(dest.read_bytes())
    old_report = load(DATA / ('reports/corpus_unified.json' if (DATA / 'reports/corpus_unified.json').exists() else 'reports/corpus_resegmentation.json'))
    annotation = contained(old_report['annotation_baseline_path'])
    if digest(annotation.read_bytes()) != old_report['annotation_baseline_sha256']:
        raise ValueError('Original annotation archive changed')
    raw, extraction = {}, {}
    from oil_evidence.corpus import extraction_suffix
    for doc in read_jsonl(DATA / 'documents.jsonl'):
        path = contained(doc['raw_path'])
        actual = digest(path.read_bytes())
        if actual != doc['raw_sha256']:
            raise ValueError('Raw source changed: ' + doc['document_id'])
        raw[doc['raw_path']] = actual
        cache = DATA / 'extracted' / (doc['raw_sha256'] + extraction_suffix(doc['source']))
        if cache.exists():
            extraction[cache.relative_to(DATA).as_posix()] = digest(cache.read_bytes())
    # Preserve the legacy implementation as well as immutable data, before
    # changing default entry points. No secrets or environment files copied.
    for folder in ['src', 'scripts', 'tests', 'seeds']:
        shutil.copytree(PROJECT / folder, scratch / 'project' / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['config.toml', 'uv.lock', 'pyproject.toml']:
        shutil.copy2(PROJECT / name, scratch / 'project' / name)
    manifest = dict(at=now(), hashes=copied, raw_hashes=raw, extraction_hashes=extraction,
                    annotation_baseline_path=old_report['annotation_baseline_path'],
                    annotation_baseline_sha256=old_report['annotation_baseline_sha256'])
    write_json(scratch / 'manifest.json', manifest)
    scratch.rename(baseline)
    return manifest


def verify_inputs():
    manifest = snapshot()
    for name in INPUTS:
        path = DATA / name
        expected = manifest['hashes'].get(name)
        if (digest(path.read_bytes()) if path.exists() else None) != expected:
            raise ValueError('Live input changed since snapshot: ' + name)
    for group in ['raw_hashes', 'extraction_hashes']:
        for name, expected in manifest[group].items():
            if digest(contained(name).read_bytes()) != expected:
                raise ValueError('Source or extraction cache changed: ' + name)
    if digest(contained(manifest['annotation_baseline_path']).read_bytes()) != manifest['annotation_baseline_sha256']:
        raise ValueError('Original annotation archive changed')
    return manifest


def _optional_attributes(record):
    # Audit shape as well as values: an injected null/blank optional key is not
    # the legacy absence of that key, even when .get() would treat them alike.
    return {key: record[key] for key in OPTIONAL_PROVENANCE if key in record}


def supplement_inputs():
    """Verify appended sources without registering or replacing live sources."""
    document_path, source_path = (RUN / 'supplement' / name for name in ['documents.jsonl', 'sources.jsonl'])
    admission_path = RUN / 'supplement/admission_exclusions.jsonl'
    reprocessing_path = RUN / 'supplement/reprocessing_admissions.jsonl'
    if document_path.exists() != source_path.exists():
        raise ValueError('Supplement needs both documents and sources manifests')
    if not document_path.exists():
        if admission_path.exists() or reprocessing_path.exists():
            raise ValueError('Admission decisions require existing supplement manifests')
        return [], [], {}
    documents, sources = read_jsonl(document_path), read_jsonl(source_path)
    original = read_jsonl(DATA / 'documents.jsonl')
    ids, urls, raw_paths = ({d[key] for d in original} for key in ['document_id', 'url', 'raw_path'])
    source_by_url = {s['url']: s for s in sources}
    if len(source_by_url) != len(sources) or len(sources) != len(documents):
        raise ValueError('Supplement source/document manifest count or URL uniqueness mismatch')
    bound = {p.relative_to(DATA).as_posix(): digest(p.read_bytes()) for p in [document_path, source_path]}
    if reprocessing_path.exists():
        from reprocess_boem_supplement import apply_reprocessing_admissions
        documents, sources, repaired_hashes = apply_reprocessing_admissions(
            documents, sources, data=DATA, runroot=RUN)
        bound.update(repaired_hashes)
        source_by_url = {source['url']: source for source in sources}
    if admission_path.exists():
        decisions, docs_by_id = {}, {doc['document_id']: doc for doc in documents}
        admission_bytes = admission_path.read_bytes()
        for line in admission_bytes.decode('utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('Admission exclusion must be an object')
            identifier = record.get('document_id')
            if not isinstance(identifier, str) or identifier not in docs_by_id:
                raise ValueError('Admission exclusion references an unknown supplemental document')
            if identifier in decisions:
                raise ValueError('Admission exclusions have duplicate document IDs')
            if record.get('raw_sha256') != docs_by_id[identifier]['raw_sha256']:
                raise ValueError('Admission exclusion raw hash differs from supplemental document')
            if (any(not isinstance(record.get(key), str) or not record[key].strip()
                    for key in ['reason', 'decision_version', 'proof_path', 'review_sha256'])
                    or not re.fullmatch(r'[0-9a-f]{64}', record['review_sha256'])):
                raise ValueError('Admission exclusion requires nonempty audit proof fields and a review SHA256')
            decisions[identifier] = dict(record)
        bound[admission_path.relative_to(DATA).as_posix()] = digest(admission_bytes)
        # Append-only decisions derive stage-only copies. Original registry
        # objects/files stay unchanged; proof_path is metadata, never executed.
        def derived(record, decision):
            return dict(record, ingest_exclusion_reason='audited_post_collection:' + decision['reason'],
                        admission_exclusion_proof=dict(decision))
        decisions_by_url = {docs_by_id[key]['url']: decision for key, decision in decisions.items()}
        documents = [derived(doc, decisions[doc['document_id']]) if doc['document_id'] in decisions else doc
                     for doc in documents]
        sources = [derived(source, decisions_by_url[source['url']]) if source['url'] in decisions_by_url else source
                   for source in sources]
        source_by_url = {source['url']: source for source in sources}
    from oil_evidence.corpus import extraction_suffix
    for doc in documents:
        if doc['document_id'] in ids or doc['url'] in urls or doc['raw_path'] in raw_paths:
            raise ValueError('Supplement reuses a registered document, URL or raw path')
        source = source_by_url.get(doc['url'])
        if (source is None or any(source.get(k) != doc.get(k) for k in PROVENANCE if k != 'document_id')
                or _optional_attributes(source) != _optional_attributes(doc)
                or {k: source[k] for k in ['source_block_attribution'] if k in source}
                   != {k: doc[k] for k in ['source_block_attribution'] if k in doc}):
            raise ValueError('Supplement source metadata differs from document metadata')
        ids.add(doc['document_id'])
        urls.add(doc['url'])
        raw_paths.add(doc['raw_path'])
        raw = contained(doc['raw_path'])
        if not raw.is_relative_to((DATA / 'raw').resolve()) or digest(raw.read_bytes()) != doc['raw_sha256']:
            raise ValueError('Supplement raw path/hash is not a verified appended source')
        bound[doc['raw_path']] = doc['raw_sha256']
        cache = DATA / 'extracted' / (doc['raw_sha256'] + extraction_suffix(doc['source']))
        if cache.exists():
            bound[cache.relative_to(DATA).as_posix()] = digest(cache.read_bytes())
    return documents, sources, bound


def _documents():
    return read_jsonl(DATA / 'documents.jsonl') + supplement_inputs()[0]


def _registries(target, supplemental_documents, supplemental_sources):
    write_jsonl(target / 'documents.jsonl', read_jsonl(DATA / 'documents.jsonl') + supplemental_documents)
    write_jsonl(target / 'sources.jsonl', read_jsonl(DATA / 'sources.jsonl') + supplemental_sources)


def stage():
    if (RUN / 'published').exists():
        raise ValueError('Published run immutable; use a new run ID')
    before = verify_inputs()
    implementation_before = hashes()
    additional_documents, additional_sources, supplemental_hashes = supplement_inputs()
    target = RUN / 'stage'
    if (target / 'passages.jsonl').exists():
        sha = digest((target / 'passages.jsonl').read_bytes())
        frozen = RUN / 'qa_inputs' / sha
        if not frozen.exists():
            shutil.copytree(target, frozen, ignore=shutil.ignore_patterns('index'))
    report = build_unified_corpus(target, additional_documents=additional_documents)
    verify_inputs()
    if implementation_before != hashes() or supplemental_hashes != supplement_inputs()[2]:
        raise ValueError('Implementation changed during build; freeze candidate and rebuild before indexing')
    _registries(target, additional_documents, additional_sources)
    report.update(implementation_hashes=hashes(), baseline_passages_path=(RUN / 'baseline/passages.jsonl').relative_to(DATA).as_posix(),
                  baseline_passages_sha256=before['hashes']['passages.jsonl'],
                  annotation_baseline_path=before['annotation_baseline_path'], annotation_baseline_sha256=before['annotation_baseline_sha256'],
                  source_pages_path=(target / 'source_pages.jsonl').relative_to(DATA).as_posix(),
                  supplemental_input_hashes=supplemental_hashes, supplemental_documents=len(additional_documents),
                  documents_sha256=digest((target / 'documents.jsonl').read_bytes()),
                  sources_sha256=digest((target / 'sources.jsonl').read_bytes()))
    write_json(target / 'reports/corpus.json', report)
    strata = defaultdict(list)
    for row in read_jsonl(target / 'passages.jsonl'):
        strata[row['source']].append(row)
    write_jsonl(target / 'reports/quality_samples.jsonl', [row for key, group in sorted(strata.items())
                for row in sorted(group, key=lambda p: digest(p['passage_id']))[:3]])
    return report


def stage_hashes():
    return {name: digest((RUN / 'stage' / name).read_bytes()) for name in
            ['passages.jsonl', 'source_pages.jsonl', 'quarantine.jsonl', 'source_exclusions.jsonl',
             'documents.jsonl', 'sources.jsonl', 'reports/corpus.json']}


def check_index(target):
    manifest = load(target / 'index/manifest.json')
    rows = read_jsonl(target / 'passages.jsonl')
    vectors = np.load(target / 'index/vectors.npy', allow_pickle=False)
    from oil_evidence.store import config
    if (manifest['model'] != config()['retrieval']['embedding_model'] or
        manifest['corpus_sha256'] != digest((target / 'passages.jsonl').read_bytes()) or
        manifest['vectors_sha256'] != digest((target / 'index/vectors.npy').read_bytes()) or
        manifest['passage_ids'] != [p['passage_id'] for p in rows] or
        vectors.shape != (len(rows), manifest['dimension']) or not np.isfinite(vectors).all() or
        not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError('Index corpus/model/hash/order/shape/norm mismatch')
    return manifest


def audit(require_index=False):
    verify_inputs()
    target = RUN / 'stage'
    report = load(target / 'reports/corpus.json')
    if report['implementation_hashes'] != hashes():
        raise ValueError('Processing implementation changed since build')
    additional_documents, additional_sources, supplemental_hashes = supplement_inputs()
    if report.get('supplemental_input_hashes') != supplemental_hashes or report.get('supplemental_documents') != len(additional_documents):
        raise ValueError('Supplement manifest/raw/cache changed since build')
    for field, name in [('corpus_sha256', 'passages.jsonl'), ('source_pages_sha256', 'source_pages.jsonl'),
                        ('quarantine_sha256', 'quarantine.jsonl'), ('source_exclusions_sha256', 'source_exclusions.jsonl'),
                        ('documents_sha256', 'documents.jsonl'), ('sources_sha256', 'sources.jsonl')]:
        if report[field] != digest((target / name).read_bytes()):
            raise ValueError('Stage report hash is stale: ' + name)
    rows, pages, rejected = (read_jsonl(target / name) for name in ['passages.jsonl', 'source_pages.jsonl', 'quarantine.jsonl'])
    if report['corpus_sha256'] != digest((target / 'passages.jsonl').read_bytes()):
        raise ValueError('Corpus report is stale')
    pages_by_id = {p['source_page_id']: p for p in pages}
    if len(pages_by_id) != len(pages):
        raise ValueError('Duplicate source page IDs')
    if len({p['passage_id'] for p in rows}) != len(rows) or len({digest(' '.join(words(p['text']))) for p in rows}) != len(rows):
        raise ValueError('Duplicate IDs or normalized text')
    documents = {d['document_id']: d for d in _documents()}
    if read_jsonl(target / 'documents.jsonl') != list(documents.values()) or read_jsonl(target / 'sources.jsonl') != read_jsonl(DATA / 'sources.jsonl') + additional_sources:
        raise ValueError('Staged document/source registry differs from verified input manifests')
    from oil_evidence.unified_corpus import clean_source_page, extract_existing
    raw_cache = {}
    for page in pages:
        doc = documents[page['document_id']]
        if doc['document_id'] not in raw_cache:
            raw_cache[doc['document_id']] = extract_existing(doc, DATA)[0]
        raw = raw_cache[doc['document_id']][page['page'] - 1]
        if page['text'] != clean_source_page(raw, doc)[0] or page['raw_page_sha256'] != digest(raw):
            raise ValueError('Frozen clean source page does not reproduce from extraction')
        if (any(page.get(k) != doc.get(k) for k in PROVENANCE)
                or _optional_attributes(page) != optional_provenance(doc)):
            raise ValueError('Source page metadata does not match document')
        expected_native = source_native_context(doc, raw, page['text'])
        if {k: page[k] for k in ['source_native_blocks', 'source_hard_boundaries'] if k in page} != expected_native:
            raise ValueError('Source-native block boundaries or attribution differ from document')
    ranges = defaultdict(list)
    for row in rows + rejected:
        page = pages_by_id[row['source_page_id']]
        a, b = row['char_start'], row['char_end']
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(page['text']) or page['text'][a:b] != row['text']:
            raise ValueError('Exact source page offsets failed')
        if row['word_count'] != len(words(row['text'])):
            raise ValueError('Word count mismatch')
        ranges[row['source_page_id']].append((a, b))
    for row in rows:
        page = pages_by_id[row['source_page_id']]
        if (any(row.get(k) != page.get(k) for k in PROVENANCE) or row['page'] != page['page']
                or _optional_attributes(row) != optional_provenance(page)):
            raise ValueError('Passage provenance changed')
        if ({k: row[k] for k in ['source_block_attribution'] if k in row}
                != passage_source_context(page, row['char_start'], row['char_end'])
                or any(row['char_start'] < boundary < row['char_end'] for boundary in page.get('source_hard_boundaries', []))):
            raise ValueError('Passage crosses independent source boundary or changed source-block attribution')
        if row['segmentation'] != report['chunk_policy'] or row.get('parent_passage_id'):
            raise ValueError('Mixed segmentation policy or old-parent fallback')
    old_ids = {p['passage_id'] for p in read_jsonl(RUN / 'baseline/passages.jsonl')}
    annotation_ids = {p['passage_id'] for p in read_jsonl(contained(snapshot()['annotation_baseline_path']))}
    if any(p['passage_id'] in old_ids or p['passage_id'] in annotation_ids for p in rows):
        raise ValueError('Uniform version reused a historical passage ID')
    for pid, page in pages_by_id.items():
        cursor = 0
        for a, b in sorted(ranges[pid]):
            if a < cursor or page['text'][cursor:a].strip():
                raise ValueError('Overlapping or unaccounted source text')
            cursor = b
        if page['text'][cursor:].strip():
            raise ValueError('Unaccounted source page tail')
    total_words = sum(len(words(p['text'])) for p in rows)
    if (len(rows) != report['passages'] or total_words != report['words'] or
        sum(p['word_count'] for p in pages) != total_words + sum(p['word_count'] for p in rejected) or
        len(pages) != report['source_pages'] or report['source_pages_sha256'] != digest((target / 'source_pages.jsonl').read_bytes())):
        raise ValueError('Corpus/page/quarantine accounting mismatch')
    if require_index:
        check_index(target)
    # Recompute with the actual fixed algorithm and document/page exclusions,
    # not merely a report saying offsets and totals pass. This also verifies
    # per-page extraction metadata and every rejection/deduplication decision.
    with tempfile.TemporaryDirectory(prefix='.audit-replay-', dir=RUN) as temporary:
        scratch = Path(temporary)
        if scratch.resolve().parent != RUN.resolve():
            raise ValueError('Unsafe audit scratch path')
        reconstructed = build_unified_corpus(scratch, additional_documents=additional_documents)
        for name in ['passages.jsonl', 'source_pages.jsonl', 'quarantine.jsonl', 'source_exclusions.jsonl']:
            if digest((scratch / name).read_bytes()) != digest((target / name).read_bytes()):
                raise ValueError('Source-to-policy reconstruction differs: ' + name)
        for key, value in reconstructed.items():
            if report.get(key) != value:
                raise ValueError('Computed corpus statistics differ: ' + key)
    result = dict(at=now(), errors=[], corpus_sha256=report['corpus_sha256'], stage_hashes=stage_hashes(),
                  implementation_hashes=hashes(), active_passages=len(rows), source_pages=len(pages),
                  exact_source_offsets=True, non_whitespace_accounting=True, words_accounted=True,
                  uniform_policy=True, old_labels_preserved=True, index_verified=require_index,
                  external_model_calls=0, human_labels_created=0)
    write_json(target / 'reports/unified_quality.json', result)
    return result


def index():
    audit()
    from oil_evidence import retrieval
    target = RUN / 'stage/index'
    target.mkdir(parents=True, exist_ok=True)
    # Text-hash cache is safe across new IDs. Do not reuse old ID->vector map.
    if not (target / 'embedding_cache.sqlite3').exists():
        shutil.copy2(DATA / 'index/embedding_cache.sqlite3', target / 'embedding_cache.sqlite3')
    previous = retrieval.DATA
    try:
        retrieval.DATA = RUN / 'stage'
        return retrieval.build_index()
    finally:
        retrieval.DATA = previous


def required_review_ids(rows):
    """Fixed source-stratified plus length-extreme sample, before retrieval."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['source']].append(row)
    required = {row['passage_id'] for group in groups.values()
                for row in sorted(group, key=lambda p: digest(p['passage_id']))[:3]}
    ordered = sorted(rows, key=lambda p: (len(words(p['text'])), p['passage_id']))
    required.update(row['passage_id'] for row in ordered[:10] + ordered[-10:])
    return sorted(required)


def _comparison_module():
    path = Path(__file__).with_name('compare_segmentation.py')
    name = 'unified_release_comparison_' + digest(path.read_bytes())[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _project_path(value):
    if not isinstance(value, str) or not value or '..' in value.replace('\\', '/').split('/'):
        raise ValueError('Unsafe comparison auxiliary path')
    path = Path(value).resolve()
    if not path.is_relative_to(PROJECT.resolve()) or not path.is_file():
        raise ValueError('Comparison auxiliary input is outside project or missing')
    return path


def _same(actual, expected):
    """JSON-safe comparison without treating True as 1 or accepting NaNs."""
    if isinstance(expected, float):
        return (type(actual) in [float, int] and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(_same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def check_release_gates(report):
    """Recompute source support; a report's release boolean is not authority.

    This is a small model/source-grounded diagnostic, not gold evaluation. It
    verifies returned source text and all frozen probe anchors, not semantic
    relevance labels or whether a different ranking would have been better.
    """
    target = RUN / 'stage'
    rows = read_jsonl(target / 'passages.jsonl')
    review = load(target / 'reports/semantic_review.json')
    reviewed = review.get('reviewed_ids')
    required = required_review_ids(rows)
    active_ids = {p['passage_id'] for p in rows}
    if (review.get('corpus_sha256') != report['corpus_sha256']
            or review.get('source_pages_sha256') != report['source_pages_sha256']
            or review.get('critical_active_ids') != [] or not isinstance(reviewed, list)
            or not reviewed or any(not isinstance(pid, str) or pid not in active_ids for pid in reviewed)
            or len(reviewed) != len(set(reviewed)) or not set(required) <= set(reviewed)
            or review.get('human_gold') is not False or review.get('human_labels_created') != 0
            or not isinstance(review.get('provenance'), str) or not review['provenance'].strip()):
        raise ValueError('Current machine source review must cover fixed source/length samples without critical failures')
    comparison = load(target / 'reports/comparison.json')
    compare = _comparison_module()
    from oil_evidence.store import config
    settings = config()['retrieval']
    expected_settings = dict(embedding_model=settings['embedding_model'], top_k=settings.get('top_k', 5),
                             candidate_k=settings['candidate_k'], rrf_k=settings['rrf_k'], temporal=True,
                             modes=list(compare.MODES))
    if not _same(comparison.get('settings'), expected_settings):
        raise ValueError('Comparison settings differ from fixed retrieval configuration')
    if (comparison.get('diagnostic_only') is not True or comparison.get('human_gold') is not False
            or comparison.get('inputs_unchanged') is not True
            or any(type(comparison.get(k)) is not int or comparison[k] != 0
                   for k in ['external_model_calls', 'human_labels_created', 'generated_answers'])
            or comparison.get('script_sha256') != digest(Path(compare.__file__).read_bytes())):
        raise ValueError('Comparison provenance/version is missing or falsely presented as gold')
    corpora, input_fingerprints = {}, {}
    for name, root in [('baseline', RUN / 'baseline'), ('candidate', target)]:
        corpus = compare.load_corpus(root, settings['embedding_model'])
        expected = dict(root=str(root.resolve()), **corpus.hashes)
        given = comparison.get('input_hashes', {}).get(name)
        if not isinstance(given, dict) or not _same(given, expected):
            raise ValueError('Comparison baseline/candidate input hashes or roots do not match')
        corpora[name] = corpus
        input_fingerprints[name] = expected
    if corpora['baseline'].hashes['passages.jsonl'] != report['baseline_passages_sha256'] or corpora['candidate'].hashes['passages.jsonl'] != report['corpus_sha256']:
        raise ValueError('Comparison corpus versions differ from frozen build baseline/candidate')
    for field, expected in [('baseline_corpus_sha256', report['baseline_passages_sha256']),
                            ('candidate_corpus_sha256', report['corpus_sha256'])]:
        if field in comparison and comparison[field] != expected:
            raise ValueError('Comparison compatibility corpus hash field is inconsistent')
    auxiliary = comparison.get('auxiliary_input_hashes')
    if not isinstance(auxiliary, dict) or not auxiliary:
        raise ValueError('Comparison has no frozen source/query/probe input hashes')
    resolved = {}
    for value, expected in auxiliary.items():
        path = _project_path(value)
        if not _valid_sha(expected) or digest(path.read_bytes()) != expected or path in resolved:
            raise ValueError('Comparison auxiliary input changed or duplicated')
        resolved[path] = expected
    source_path = (target / 'source_pages.jsonl').resolve()
    if resolved.get(source_path) != report['source_pages_sha256']:
        raise ValueError('Comparison does not bind the current cleaned source pages')
    source_pages = compare.load_source_pages(source_path)
    seed_paths, probe_paths = [], []
    for path in resolved:
        if path == source_path:
            continue
        payload = load(path)
        if isinstance(payload, list):
            seed_paths.append(path)
        elif isinstance(payload, dict) and 'probes' in payload:
            probe_paths.append(path)
        else:
            raise ValueError('Unknown comparison auxiliary input')
    if len(seed_paths) != 1 or len(probe_paths) != 1:
        raise ValueError('Comparison must bind one frozen seed file and one source-only probe file')
    probes = compare.load_probes(probe_paths[0], source_pages, report['source_pages_sha256'])
    seeds = compare.load_seeds(seed_paths[0])
    queries = seeds + probes
    query_by_id = {q['query_id']: q for q in queries}
    if (len(probes) < 12 or len(query_by_id) != len(queries)
            or comparison.get('probe_queries') != len(probes) or comparison.get('seed_queries') != len(seeds)
            or comparison.get('queries') != len(queries)):
        raise ValueError('Comparison requires all frozen seeds and at least 12 unique source-only probes')
    scope, scope_report = compare.shared_scope(corpora['baseline'], corpora['candidate'])
    if not _same(comparison.get('source_scope'), scope_report):
        raise ValueError('Comparison source-page intersection does not match current inputs')
    for key in scope:
        if key not in source_pages or tuple(source_pages[key].get(k) for k in compare.PAGE_METADATA) != corpora['baseline'].metadata[key]:
            raise ValueError('Comparison source metadata differs from shared corpus pages')
    recomputed = {}
    all_pairs = {(q['query_id'], mode) for q in queries for mode in compare.MODES}
    for name, corpus in corpora.items():
        result_rows = comparison.get('results', {}).get(name)
        if not isinstance(result_rows, list):
            raise ValueError('Comparison detailed retrieval results are missing')
        actual_pairs, checked_rows = set(), []
        by_id = {p['passage_id']: p for p in corpus.rows}
        order = {p['passage_id']: i for i, p in enumerate(corpus.rows)}
        for row in result_rows:
            pair = row.get('query_id'), row.get('mode')
            if pair not in all_pairs or pair in actual_pairs:
                raise ValueError('Comparison query/mode rows are incomplete, duplicated or unknown')
            actual_pairs.add(pair)
            query = query_by_id[pair[0]]
            if any(row.get(k) != query[k] for k in ['kind', 'question', 'as_of']):
                raise ValueError('Comparison result changed frozen query metadata')
            evidence = row.get('evidence')
            if not isinstance(evidence, list) or len(evidence) > expected_settings['top_k']:
                raise ValueError('Comparison returned an invalid top-k evidence list')
            hits = []
            for rank, entry in enumerate(evidence, 1):
                original = by_id.get(entry.get('passage_id'))
                if original is None or compare.page_key(original) not in scope or type(entry.get('rank')) is not int or entry['rank'] != rank:
                    raise ValueError('Comparison evidence ID/rank/source page is not in the fixed comparable corpus')
                hit = dict(original, **{k: entry.get(k) for k in ['rank', 'score', 'bm25_score', 'dense_score']})
                if not _same(entry, compare.evidence_record(hit)):
                    raise ValueError('Comparison evidence metadata/text hashes differ from actual corpus')
                hits.append(hit)
            metrics = compare.query_metrics(query, hits, order)
            seconds = row.get('metrics', {}).get('ranking_seconds')
            if type(seconds) not in [int, float] or not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Comparison ranking time is not a finite nonnegative value')
            metrics['ranking_seconds'] = seconds
            if not _same(row.get('metrics'), metrics):
                raise ValueError('Comparison source support or temporal metrics do not recompute from actual evidence')
            checked_rows.append(dict(row, metrics=metrics))
        if actual_pairs != all_pairs:
            raise ValueError('Comparison omitted frozen queries or retrieval modes')
        recomputed[name] = compare.aggregate(checked_rows)
        if not _same(comparison.get('summaries', {}).get(name), recomputed[name]):
            raise ValueError('Comparison aggregate summaries differ from recomputed query results')
        if any(item['temporal_errors'] != 0 for item in recomputed[name].values()):
            raise ValueError('Comparison contains temporal retrieval errors')
    metric = 'mean_assembled_anchor_support_coverage'
    baseline = recomputed['baseline']['source_anchor_probe/hybrid'][metric]
    candidate = recomputed['candidate']['source_anchor_probe/hybrid'][metric]
    if candidate + 1e-12 < baseline:
        raise ValueError(f'Candidate source-anchor support regressed: {candidate} < {baseline}')
    for value, expected in comparison.get('local_encoder_model_hashes', {}).items():
        path = _project_path(value)
        if not _valid_sha(expected) or digest(path.read_bytes()) != expected:
            raise ValueError('Comparison local encoder model hash changed')
        resolved[path] = expected
    decision = dict(corpus_sha256=report['corpus_sha256'], baseline_corpus_sha256=report['baseline_passages_sha256'],
                    source_pages_sha256=report['source_pages_sha256'], reviewed_ids=reviewed,
                    required_review_ids=required, probe_queries=len(probes),
                    baseline_hybrid_assembled_anchor_support=baseline, candidate_hybrid_assembled_anchor_support=candidate,
                    all_modes_temporal_errors=0, release_gate_passed=True, diagnostic_only=True, human_gold=False,
                    external_model_calls=0, human_labels_created=0,
                    auxiliary_input_hashes={str(path): sha for path, sha in resolved.items()},
                    limitations=['Recomputed literal source-anchor support is not semantic correctness or human gold.',
                                 'Evidence/support is revalidated; ranking timings and the reported ordering are not independently benchmarked.'])
    return decision


def release_fingerprints():
    names = list(stage_hashes()) + ['index/vectors.npy', 'index/manifest.json', 'index/embedding_cache.sqlite3',
             'reports/quality_samples.jsonl', 'reports/semantic_review.json', 'reports/comparison.json',
             'reports/unified_quality.json', 'reports/release_decision.json']
    return {name: digest((RUN / 'stage' / name).read_bytes()) for name in names}


def require_fresh_release(expected_stage, expected_code, auxiliary):
    verify_inputs()
    report = load(RUN / 'stage/reports/corpus.json')
    if hashes() != expected_code or report['implementation_hashes'] != expected_code:
        raise ValueError('Implementation changed during publication')
    if supplement_inputs()[2] != report['supplemental_input_hashes'] or release_fingerprints() != expected_stage:
        raise ValueError('Stage/supplement changed during publication')
    for value, expected in auxiliary.items():
        if digest(_project_path(value).read_bytes()) != expected:
            raise ValueError('Comparison auxiliary input changed during publication')
    check_index(RUN / 'stage')


def publish():
    target = RUN / 'stage'
    report = load(target / 'reports/corpus.json')
    audited = audit(True)
    decision = check_release_gates(report)
    write_json(target / 'reports/release_decision.json', decision)
    implementation = hashes()
    fingerprints = release_fingerprints()
    require_fresh_release(fingerprints, implementation, decision['auxiliary_input_hashes'])
    frozen = RUN / 'published' / report['corpus_sha256']
    if frozen.exists() or (RUN / 'publication_transaction').exists():
        raise ValueError('Immutable publication/transaction already exists; use new run')
    shutil.copytree(target, frozen)
    for name in CODE:
        dest = frozen / 'implementation' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, dest)
        if digest(dest.read_bytes()) != implementation[name]:
            raise ValueError('Frozen implementation differs from audited implementation')
    require_fresh_release(fingerprints, implementation, decision['auxiliary_input_hashes'])
    if any(digest((frozen / name).read_bytes()) != expected for name, expected in fingerprints.items()):
        raise ValueError('Frozen release differs from audited stage')
    published = dict(report, source_pages_path=(frozen / 'source_pages.jsonl').relative_to(DATA).as_posix(),
                     published_snapshot=frozen.relative_to(DATA).as_posix(), stage='published_uniform_source_page_version',
                     audit_sha256=digest((target / 'reports/unified_quality.json').read_bytes()))
    write_json(frozen / 'reports/corpus_unified.json', published)
    transaction = RUN / 'publication_transaction'
    files = ['passages.jsonl', 'documents.jsonl', 'sources.jsonl', 'reports/corpus.json', 'index/vectors.npy', 'index/manifest.json',
             'index/embedding_cache.sqlite3', 'reports/unified_quality.json', 'reports/corpus_unified.json', 'reports/release_decision.json']
    journal = dict(at=now(), state='preparing', files=files, replaced=[], old_hashes={}, new_hashes={})
    for name in files:
        dest = transaction / 'new' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frozen / name, dest)
        journal['new_hashes'][name] = digest(dest.read_bytes())
        old = DATA / name
        journal['old_hashes'][name] = digest(old.read_bytes()) if old.exists() else None
        if old.exists():
            backup = transaction / 'old' / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, backup)
    write_json(transaction / 'journal.json', journal)
    require_fresh_release(fingerprints, implementation, decision['auxiliary_input_hashes'])
    if any(digest((transaction / 'new' / name).read_bytes()) != expected for name, expected in journal['new_hashes'].items()):
        raise ValueError('Prepared publication bytes changed')
    journal['state'] = 'committing'
    write_json(transaction / 'journal.json', journal)
    try:
        for name in files:
            (DATA / name).parent.mkdir(parents=True, exist_ok=True)
            (transaction / 'new' / name).replace(DATA / name)
            journal['replaced'].append(name)
            write_json(transaction / 'journal.json', journal)
        journal['state'] = 'committed'
        write_json(transaction / 'journal.json', journal)
    except Exception:
        for name in reversed(journal['replaced']):
            if journal['old_hashes'][name] is None:
                (DATA / name).unlink()
            else:
                shutil.copy2(transaction / 'old' / name, DATA / name)
        journal['state'] = 'rolled_back'
        write_json(transaction / 'journal.json', journal)
        raise
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['snapshot', 'stage', 'audit', 'index', 'publish'])
    parser.add_argument('--run', default=VERSION)
    parser.add_argument('--require-index', action='store_true')
    args = parser.parse_args()
    select_run(args.run)
    with lock():
        result = audit(args.require_index) if args.action == 'audit' else globals()[args.action]()
    if args.action == 'snapshot':
        print(json.dumps(dict(snapshot=str(RUN / 'baseline'), protected_inputs=len(result['hashes']),
                              raw_documents=len(result['raw_hashes']), extraction_caches=len(result['extraction_hashes'])), indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
