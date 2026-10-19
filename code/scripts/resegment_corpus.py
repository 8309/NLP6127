"""Versioned sentence resegmentation, not duplicate inflation or label migration.

Original parents, sources, annotations and quarantine decisions stay auditable.
Only a fully audited >=10k corpus with an aligned index may replace live files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import date
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from analyze_chunk_feasibility import ABBREVIATION, DEPENDENT_START, TERMINAL, normalized, propose_split
from oil_evidence.corpus import oil_topic_supported
from oil_evidence.store import DATA, digest, now, read_jsonl, words, write_json, write_jsonl

ROOT = DATA / 'resegmentation_runs/semantic-80-160-v1'
MIN_PASSAGES, MIN_WORDS = 10000, 100000
MIN_CHILD_WORDS, MAX_CHILD_WORDS = 80, 160
VERSION = 'semantic-80-160-v1'
INCOMPLETE_ABBREVIATION = re.compile(r'\b(?:U\.\s*S|U\.\s*K|E\.\s*U|e\.g|i\.e|Dr|Mr|Mrs|Ms|Prof|Fig|Eq|No|vs|approx|St|Inc|Ltd|Co|Corp)\.$', re.I)
INPUTS = ['passages.jsonl', 'documents.jsonl', 'sources.jsonl', 'reports/corpus.json',
          'reports/corpus_expansion.json', 'reports/passage_annotation.json',
          'annotations/codex.jsonl', 'annotations/passage_labels.jsonl',
          'annotations/passage_review.jsonl', 'index/manifest.json', 'index/vectors.npy']
# Independently read extraction failures, not labels. A source repair can bring
# these back in a later version; no raw document is deleted.
KNOWN_DEFECTS = {
    'c6aa80c0e62debbb9980-3f10573401f90125',
    '87c092ca35bbe2c0fcb6-944fd40f50785815',
    'fbfd5c4fcc1167c6653c-9fd13955b870ce58',
    '382558f39ac0d4259fb6-2dd7a4008072e1a1',
    '0db69c557693df1f33a4-c8c9b639417f2bbf',
    '25b8dab89909c58a9cf3-2f7da58af0555da5',
    '37f3cdf077e4a56c9184-d8f662f88071130c',
    '1956bc5fdbd251e36370-948acd113b3c7a8a',
    'c66494af2ccae33b4d25-140fb23929a5226f',
    '28a69d2f9cdfe0591d7d-8875de216b7c26e4',
    '0db69c557693df1f33a4-5acc363b0ef4ac5e',
    '04a2117b19df936d9b04-dcb1748e02c9774f',
    '58073b274ebbafe5efce-3e1e10f664f44eb8',
    '1b0c0e679121f9e53cce-8f4810659d08fcd2',
    '8911f134d610c2d47690-528a8b956d5d5d33',
    '43bb080c548391113324-abefbb4d65afa339',
    'ac3ad11088b7d1875882-68b73c1a3b89137c',
}
# Fixed real-stage failures stay recorded even after the changed segmentation.
# Keep their full original context; do not promote weak/fragmented children.
DECLINE_SPLITS = {
    '1d9ad68d307b106d2917-6b89700fc7ca9941',  # Syria/HTS, generic sanctions only
    'f38038b87330cebe9003-6de8bcdaade74aba',  # product-list continuation
    '37f3cdf077e4a56c9184-bbfeee5975acb012',  # metal literature, oil comparator
    'fd2a138607589d57e130-109502c2d677005a',  # refineries fragment / St. James
    '3cd72fbe52c48a850f9e-22998d2545ad91b4',  # graph legends, no plotted values
}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def within_data(path):
    path = Path(path)
    path = (DATA / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(DATA.resolve()):
        raise ValueError('Provenance path escapes data directory')
    return path


def select_run(run_id):
    if (not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,80}', run_id)
            or re.fullmatch(r'con|prn|aux|nul|com[1-9]|lpt[1-9]', run_id)):
        raise ValueError('Invalid resegmentation run ID')
    global ROOT
    ROOT = DATA / 'resegmentation_runs' / run_id
    return ROOT


@contextmanager
def pipeline_lock():
    path = DATA / '.corpus_writer.lock'
    try:
        with path.open('x', encoding='utf-8') as stream:
            json.dump({'pid': os.getpid(), 'run': str(ROOT), 'at': now()}, stream)
    except FileExistsError as error:
        raise RuntimeError('Foreign or interrupted corpus writer lock: ' + str(path)) from error
    try:
        yield
    finally:
        path.unlink()


def implementation_hashes():
    names = ['config.toml', 'uv.lock', 'pyproject.toml', 'scripts/resegment_corpus.py',
             'scripts/analyze_chunk_feasibility.py', 'src/oil_evidence/corpus.py',
             'src/oil_evidence/store.py', 'src/oil_evidence/retrieval.py']
    return {name: digest((DATA.parent / name).read_bytes()) for name in names
            if (DATA.parent / name).exists()}


def snapshot():
    baseline = ROOT / 'baseline'
    if baseline.exists():
        manifest = load(baseline / 'manifest.json')
        for name, expected in manifest['hashes'].items():
            if digest((baseline / name).read_bytes()) != expected:
                raise ValueError('Frozen baseline changed: ' + name)
        return manifest
    ROOT.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='.baseline-', dir=ROOT))
    try:
        hashes = {}
        for name in INPUTS:
            src = DATA / name
            if src.exists():
                dst = scratch / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                hashes[name] = digest(dst.read_bytes())
        expansion = load(DATA / 'reports/corpus_expansion.json')
        annotation = within_data(expansion['annotation_baseline_path'])
        expected = load(DATA / 'reports/passage_annotation.json')['corpus_sha256']
        if digest(annotation.read_bytes()) != expected:
            raise ValueError('Original annotation corpus does not match label version')
        near = within_data(expansion['published_snapshot']) / 'reports/near_duplicate_candidates.jsonl'
        name = 'reports/near_duplicate_candidates.jsonl'
        (scratch / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(near, scratch / name)
        hashes[name] = digest((scratch / name).read_bytes())
        raw = {}
        for document in read_jsonl(DATA / 'documents.jsonl'):
            raw_path = within_data(document['raw_path'])
            actual = digest(raw_path.read_bytes())
            if actual != document['raw_sha256']:
                raise ValueError('Raw source hash mismatch: ' + document['document_id'])
            raw[str(raw_path.relative_to(DATA.resolve())).replace('\\', '/')] = actual
        result = dict(at=now(), hashes=hashes, raw_hashes=raw,
                      implementation_hashes=implementation_hashes(),
                      annotation_baseline_path=annotation.relative_to(DATA.resolve()).as_posix(),
                      annotation_baseline_sha256=expected)
        write_json(scratch / 'manifest.json', result)
        scratch.rename(baseline)
        return result
    finally:
        if scratch.exists():
            if scratch.resolve().parent != ROOT.resolve() or not scratch.name.startswith('.baseline-'):
                raise RuntimeError('Unsafe snapshot scratch path')
            shutil.rmtree(scratch)


def verify_baseline():
    manifest = snapshot()
    for name, expected in manifest['hashes'].items():
        if name == 'reports/near_duplicate_candidates.jsonl':
            continue  # A copied immutable diagnostic, not a mutable live input.
        if digest((DATA / name).read_bytes()) != expected:
            raise ValueError('Live input changed since snapshot: ' + name)
    annotation = within_data(manifest['annotation_baseline_path'])
    if digest(annotation.read_bytes()) != manifest['annotation_baseline_sha256']:
        raise ValueError('Original annotation archive changed')
    for name, expected in manifest['raw_hashes'].items():
        if digest(within_data(name).read_bytes()) != expected:
            raise ValueError('Raw source changed: ' + name)
    return manifest


def stage_paths():
    stage = ROOT / 'stage'
    return {'passages': stage / 'passages.jsonl', 'quarantine': stage / 'quarantine.jsonl',
            'report': stage / 'reports/corpus.json', 'audit': stage / 'reports/resegmentation_quality.json',
            'index_manifest': stage / 'index/manifest.json', 'vectors': stage / 'index/vectors.npy'}


def near_ids():
    return {pair[key] for pair in read_jsonl(ROOT / 'baseline/reports/near_duplicate_candidates.jsonl')
            for key in ('left', 'right')}


def quarantine_reason(parent, pending):
    if parent['passage_id'] in pending:
        return 'pending_near_duplicate'
    if parent['source'] == 'EIA-FINANCE':
        return 'catalogue_source_pending_evidence_extraction'
    if parent['passage_id'] in KNOWN_DEFECTS:
        return 'confirmed_extraction_defect'
    if (re.search(r'\b(?:and|or|of|for|to|with|from|in|by)\s*[•]', parent['text'], re.I)
            or re.search(r'(?:\b[A-Z]\s+){10,}', parent['text'])):
        return 'structural_extraction_pending_repair'
    return None


def child_quality(text):
    if not re.match(r'^[A-Z0-9“"(]', text) or not TERMINAL.search(text):
        return 'incomplete_sentence_edges'
    if INCOMPLETE_ABBREVIATION.search(text[-25:]):
        return 'abbreviation_at_incomplete_final_edge'
    if (DEPENDENT_START.search(text) or re.match(r'^(?:Instead|In turn|By contrast|On the other hand|Conversely|Meanwhile|Thus|Hence|Otherwise)\b', text, re.I)
            or re.match(r'^Yet\s+(?:it|its|this|these|that|those|they|their)\b', text, re.I)):
        return 'dependent_opening'
    topical = re.sub(r'\b(?:tonnes?|tons?|barrels?|million tonnes?) of oil equivalent\b', '', text, flags=re.I)
    topical = re.sub(r'\boil[-\s]+equivalent\b|\b(?:seed|virgin|soy|corn|soybean|vegetable|palm|cooking)[-\s]+oil\b', '', topical, flags=re.I)
    if not oil_topic_supported(topical) or not re.search(
            r'\b(?:oil|crude|petroleum|opec|refin\w*|gasoline|diesel|distillates?|jet fuel|propane|ethane|hydrocarbons?|liquid fuels|barrels?|bbls?)\b', topical, re.I):
        return 'no_mineral_oil_evidence'
    lower_clause = any(not ABBREVIATION.search(text[max(0,m.start()-25):m.start()+1])
                       for m in re.finditer(r'[.!?]\s+[a-z]', text))
    if lower_clause or re.search(r'\b(?:oil and|and the|of the)\s*[•]', text):
        return 'interrupted_or_scrambled_clause'
    # Data source credits, an official link in factual prose and procedural
    # "accessed" are not bibliography. Reject reference-only structure instead.
    citation_years = len(re.findall(r'\b(?:19|20)\d{2}[a-z]?(?:[,).]|-)', text))
    quoted_titles = len(re.findall(r'[“\"][^”\"]{8,}[”\"]', text))
    if (re.search(r'\b(?:References|Bibliography)\b', text, re.I)
            or (citation_years >= 3 and quoted_titles >= 2 and
                (text.count('http') >= 2 or re.search(r'\baccessed\b', text, re.I)))):
        return 'embedded_bibliography_or_furniture'
    if (re.search(r'\bU\.S\. energy market indicators\b', text, re.I)
            or (re.search(r'\b(?:Table|FIGURE)\s+[A-Z0-9]', text) and
                sum(bool(re.search(r'\d', token)) for token in text.split()) > len(text.split())*.22)):
        return 'mixed_table_not_prose_evidence'
    if re.search(r'(?:\|\s*Page\b|\bNext release date\s*:|@|\bTel(?:ephone)?\s*:)', text, re.I):
        return 'embedded_report_or_contact_furniture'
    if re.search(r'^(?:[A-Z]\s+){8,}', text):
        return 'spaced_letter_report_furniture'
    if re.search(r'\b\d{1,3}\s+(?:U\.S\. Energy|International Energy Agency|Energy Information Administration|Kerogen)\b', text):
        return 'interrupted_footnote'
    return None


def split_parent(parent):
    text = parent['text']
    if parent['passage_id'] in DECLINE_SPLITS:
        return None, 'fixed_semantic_spotcheck_declined_split'
    if normalized(text) != text:
        return None, 'original_whitespace_not_offset_safe'
    if INCOMPLETE_ABBREVIATION.search(text[-25:]):
        return None, 'parent_incomplete_abbreviation_edge'
    pieces, error = propose_split(text, MIN_CHILD_WORDS, MAX_CHILD_WORDS, screen_dependency=True)
    if pieces is None:
        return None, error
    children, cursor = [], 0
    for piece in pieces:
        issue = child_quality(piece)
        if issue:
            return None, issue
        start = text.find(piece, cursor)
        if start < cursor or text[cursor:start].strip():
            raise ValueError('Child is not the next contiguous original substring')
        end = start + len(piece)
        child = dict(parent, passage_id=parent['document_id'] + '-s-' + digest(
            parent['passage_id'] + ':' + str(start) + ':' + str(end) + ':' + piece)[:20],
            text=piece, word_count=len(words(piece)), parent_passage_id=parent['passage_id'],
            char_start=start, char_end=end, segmentation=dict(version=VERSION,
                minimum_words=MIN_CHILD_WORDS, maximum_words=MAX_CHILD_WORDS, overlap_words=0))
        children.append(child)
        cursor = end
    if text[cursor:].strip():
        raise ValueError('Split loses final parent text')
    return children, None


def stage():
    manifest = verify_baseline()
    parents = read_jsonl(ROOT / 'baseline/passages.jsonl')
    pending = near_ids()
    if not pending <= {p['passage_id'] for p in parents}:
        raise ValueError('Near duplicate diagnostic has foreign parent IDs')
    originals = {digest(' '.join(words(p['text']))) for p in parents}
    active, quarantined, fingerprints = [], [], set()
    rejected = Counter()
    for parent in parents:
        reason = quarantine_reason(parent, pending)
        if reason:
            quarantined.append({'parent': parent, 'reason': reason})
            continue
        children, error = split_parent(parent)
        if children:
            checks = [digest(' '.join(words(p['text']))) for p in children]
            if len(set(checks)) != len(checks) or any(fp in originals or fp in fingerprints for fp in checks):
                children, error = None, 'child_exact_duplicate'
        if not children:
            rejected[error] += 1
            children = [parent]
        for child in children:
            fp = digest(' '.join(words(child['text'])))
            if fp in fingerprints:
                raise ValueError('Exact duplicate active passage')
            fingerprints.add(fp)
            active.append(child)
    paths = stage_paths()
    write_jsonl(paths['passages'], active)
    write_jsonl(paths['quarantine'], quarantined)
    vocabulary = Counter(token for row in active for token in words(row['text']))
    annotation = {p['passage_id']: p for p in read_jsonl(within_data(manifest['annotation_baseline_path']))}
    labelled = {p['passage_id'] for p in read_jsonl(DATA / 'annotations/passage_labels.jsonl')}
    coverage = sum(p['passage_id'] in labelled and annotation.get(p['passage_id']) == p for p in active)
    report = dict(at=now(), passages=len(active), documents=len({p['document_id'] for p in active}),
        words=sum(vocabulary.values()), unique_words=len(vocabulary), by_source=dict(Counter(p['source'] for p in active)),
        by_year=dict(Counter(p['available_by'][:4] for p in active)),
        date_precision=dict(Counter(p['date_precision'] for p in active)),
        corpus_sha256=digest(paths['passages'].read_bytes()), source_corpus_sha256=manifest['hashes']['passages.jsonl'],
        source_passages=len(parents), quarantined_passages=len(quarantined),
        quarantine_reasons=dict(Counter(q['reason'] for q in quarantined)),
        source_words=sum(len(words(p['text'])) for p in parents),
        quarantined_words=sum(len(words(q['parent']['text'])) for q in quarantined),
        replaced_parents=len({p['parent_passage_id'] for p in active if 'parent_passage_id' in p}),
        child_passages=sum('parent_passage_id' in p for p in active),
        split_declined_reasons=dict(rejected), segmentation_version=VERSION,
        chunk_policy={'new_children_min_words': MIN_CHILD_WORDS, 'new_children_max_words': MAX_CHILD_WORDS,
                      'unchanged_parent_fallback': True, 'overlap_words': 0, 'sentence_boundaries_not_paragraph_recovery': True},
        exact_deduplication=True, pending_near_duplicate_parents_excluded=len(pending),
        implementation_hashes=implementation_hashes(),
        labels_coverage=coverage, archived_machine_labels=len(labelled),
        unlabelled_corpus_passages=len(active)-coverage,
        source_passages_path=(ROOT/'baseline/passages.jsonl').relative_to(DATA).as_posix(),
        source_passages_sha256=manifest['hashes']['passages.jsonl'],
        annotation_baseline_path=manifest['annotation_baseline_path'],
        annotation_baseline_sha256=manifest['annotation_baseline_sha256'],
        meets_size_requirement=len(active)>=MIN_PASSAGES and sum(vocabulary.values())>=MIN_WORDS,
        external_model_calls=0, human_labels_created=0,
        limitations=['Structural gates and Codex spot checks are not exhaustive semantic or human review.',
                     'Unsplit accepted baseline passages retain their existing extraction limitations.',
                     'Near duplicate candidates are conservatively held out, not adjudicated true duplicates.',
                     'Old parent labels and QA are archived; split children never inherit them.'])
    write_json(paths['report'], report)
    return report


def verify_index():
    paths = stage_paths()
    rows = read_jsonl(paths['passages'])
    meta = load(paths['index_manifest'])
    if meta['corpus_sha256'] != digest(paths['passages'].read_bytes()):
        raise ValueError('Staged index corpus mismatch')
    if meta['vectors_sha256'] != digest(paths['vectors'].read_bytes()):
        raise ValueError('Staged vector hash mismatch')
    vectors = np.load(paths['vectors'], allow_pickle=False)
    if (meta['passage_ids'] != [p['passage_id'] for p in rows]
            or vectors.shape != (len(rows), meta['dimension']) or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) <= 0)):
        raise ValueError('Index order, shape, or numerical integrity mismatch')
    return True


def stage_hashes(include_index):
    paths = stage_paths()
    keys = ['passages', 'quarantine', 'report'] + (['index_manifest', 'vectors'] if include_index else [])
    result = {key: digest(paths[key].read_bytes()) for key in keys}
    semantic = ROOT/'stage/reports/semantic_spotcheck.json'
    if semantic.exists():
        result['semantic_spotcheck'] = digest(semantic.read_bytes())
    return result


def audit(require_index=False):
    manifest = verify_baseline()
    paths = stage_paths()
    rows, quarantine = read_jsonl(paths['passages']), read_jsonl(paths['quarantine'])
    original = {p['passage_id']: p for p in read_jsonl(ROOT / 'baseline/passages.jsonl')}
    pending = near_ids()
    if len(original) != len(read_jsonl(ROOT / 'baseline/passages.jsonl')):
        raise ValueError('Duplicate original IDs')
    seen, groups, quarantined = set(), {}, {}
    fingerprints = set()
    for q in quarantine:
        parent = q['parent']
        pid = parent['passage_id']
        if (pid not in original or original[pid] != parent or pid in quarantined
                or q['reason'] != quarantine_reason(parent, pending) or not q['reason']):
            raise ValueError('Invalid quarantine identity or reason')
        quarantined[pid] = q
    forbidden = {'labels', 'label', 'answer_points', 'answerability', 'model_verification', 'reference_answer'}
    for row in rows:
        pid = row['passage_id']
        fp = digest(' '.join(words(row['text'])))
        if pid in seen or fp in fingerprints or forbidden.intersection(row):
            raise ValueError('Duplicate ID/text or annotation fields in active corpus')
        seen.add(pid)
        fingerprints.add(fp)
        if row['word_count'] != len(words(row['text'])):
            raise ValueError('Incorrect word count')
        available = date.fromisoformat(row['available_by'])
        if not date(2018,1,1) <= available <= date(2026,10,5):
            raise ValueError('Date outside fixed corpus window')
        parent_id = row.get('parent_passage_id', pid)
        if parent_id not in original or parent_id in quarantined:
            raise ValueError('Unknown or quarantined parent in active corpus')
        parent = original[parent_id]
        if 'parent_passage_id' not in row:
            if row != parent:
                raise ValueError('Unsplit original changed under same ID')
        else:
            extras = {'passage_id', 'text', 'word_count', 'parent_passage_id', 'char_start', 'char_end', 'segmentation'}
            if {k:v for k,v in row.items() if k not in extras} != {k:v for k,v in parent.items() if k not in extras}:
                raise ValueError('Child source metadata changed')
            start, end = row['char_start'], row['char_end']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(parent['text']):
                raise ValueError('Invalid child offsets')
            if parent['text'][start:end] != row['text']:
                raise ValueError('Child is not exact original substring')
            if not MIN_CHILD_WORDS <= row['word_count'] <= MAX_CHILD_WORDS or child_quality(row['text']):
                raise ValueError('Child violates semantic proxy/size gates')
            if row['segmentation'] != dict(version=VERSION, minimum_words=MIN_CHILD_WORDS,
                    maximum_words=MAX_CHILD_WORDS, overlap_words=0):
                raise ValueError('Child segmentation version changed')
            expected_id = parent['document_id']+'-s-'+digest(parent_id+':'+str(start)+':'+str(end)+':'+row['text'])[:20]
            if pid != expected_id:
                raise ValueError('Child ID is not content-and-offset derived')
        groups.setdefault(parent_id, []).append(row)
    if set(groups) & set(quarantined) or set(groups) | set(quarantined) != set(original):
        raise ValueError('Parents are missing or multiply accounted for')
    for pid, children in groups.items():
        if any('parent_passage_id' not in p for p in children):
            if len(children) != 1:
                raise ValueError('Parent and children counted together')
            continue
        if len(children) < 2:
            raise ValueError('Replacement must contain at least two children')
        ordered = sorted(children, key=lambda p:p['char_start'])
        cursor = 0
        for child in ordered:
            if child['char_start'] < cursor or original[pid]['text'][cursor:child['char_start']].strip():
                raise ValueError('Child overlap or lost parent text')
            cursor = child['char_end']
        if original[pid]['text'][cursor:].strip() or words(' '.join(p['text'] for p in ordered)) != words(original[pid]['text']):
            raise ValueError('Parent text/word conservation failed')
    report = load(paths['report'])
    if report.get('implementation_hashes') != implementation_hashes():
        raise ValueError('Stage implementation changed; restage before auditing')
    actual_words = sum(len(words(p['text'])) for p in rows)
    quarantine_words = sum(len(words(q['parent']['text'])) for q in quarantine)
    if (report['corpus_sha256'] != digest(paths['passages'].read_bytes())
            or report['passages'] != len(rows) or report['words'] != actual_words
            or report['quarantined_passages'] != len(quarantine)
            or report['source_words'] != actual_words + quarantine_words
            or report['quarantined_words'] != quarantine_words):
        raise ValueError('Corpus report does not match recomputed facts')
    annotation = {p['passage_id']:p for p in read_jsonl(within_data(manifest['annotation_baseline_path']))}
    label_ids = {p['passage_id'] for p in read_jsonl(DATA/'annotations/passage_labels.jsonl')}
    vocabulary = {token for p in rows for token in words(p['text'])}
    coverage = sum(p['passage_id'] in label_ids and annotation.get(p['passage_id']) == p for p in rows)
    facts = dict(source_passages=len(original), documents=len({p['document_id'] for p in rows}),
        unique_words=len(vocabulary), by_source=dict(Counter(p['source'] for p in rows)),
        by_year=dict(Counter(p['available_by'][:4] for p in rows)),
        date_precision=dict(Counter(p['date_precision'] for p in rows)),
        replaced_parents=sum(any('parent_passage_id' in p for p in values) for values in groups.values()),
        child_passages=sum('parent_passage_id' in p for p in rows),
        quarantine_reasons=dict(Counter(q['reason'] for q in quarantine)),
        labels_coverage=coverage, archived_machine_labels=len(label_ids),
        unlabelled_corpus_passages=len(rows)-coverage,
        source_corpus_sha256=manifest['hashes']['passages.jsonl'],
        source_passages_sha256=manifest['hashes']['passages.jsonl'],
        source_passages_path=(ROOT/'baseline/passages.jsonl').relative_to(DATA).as_posix(),
        annotation_baseline_path=manifest['annotation_baseline_path'],
        annotation_baseline_sha256=manifest['annotation_baseline_sha256'],
        pending_near_duplicate_parents_excluded=len(pending),
        meets_size_requirement=len(rows)>=MIN_PASSAGES and actual_words>=MIN_WORDS)
    if any(report.get(key) != value for key,value in facts.items()):
        raise ValueError('Reported coverage, provenance, policy or statistics are incorrect')
    semantic = ROOT/'stage/reports/semantic_spotcheck.json'
    if semantic.exists():
        review = load(semantic)
        if (review.get('checked_corpus_sha256') != report['corpus_sha256']
                or review.get('critical_active_child_ids') != []):
            raise ValueError('Semantic spotcheck is stale or has unresolved active critical children')
    if require_index:
        verify_index()
    result = dict(at=now(), errors=[], corpus_sha256=report['corpus_sha256'],
        source_corpus_sha256=manifest['hashes']['passages.jsonl'],
        stage_hashes=stage_hashes(require_index), implementation_hashes=implementation_hashes(),
        index_verified=bool(require_index), source_word_conservation_passed=True,
        exact_source_offsets_passed=True, metadata_and_date_inheritance_passed=True,
        parents_not_double_counted=True, frozen_annotations_unchanged=True,
        source_passages=len(original), active_passages=len(rows), quarantined_passages=len(quarantine),
        external_model_calls=0, human_labels_created=0)
    write_json(paths['audit'], result)
    return result


def require_fresh_audit():
    result = load(stage_paths()['audit'])
    if (result.get('errors') or not result.get('index_verified')
            or result.get('stage_hashes') != stage_hashes(True)
            or result.get('implementation_hashes') != implementation_hashes()):
        raise ValueError('Missing, incomplete or stale resegmentation audit')
    verify_baseline()
    verify_index()
    return result


def index():
    verify_baseline()
    from oil_evidence import retrieval
    target = ROOT / 'stage/index'
    target.mkdir(parents=True, exist_ok=True)
    for name in ['embedding_cache.sqlite3', 'manifest.json', 'vectors.npy']:
        if not (target/name).exists() and (DATA/'index'/name).exists():
            shutil.copy2(DATA/'index'/name, target/name)
    previous = retrieval.DATA
    try:
        retrieval.DATA = ROOT / 'stage'
        return retrieval.build_index()
    finally:
        retrieval.DATA = previous


def freeze_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination.read_bytes()) != digest(source.read_bytes()):
            raise ValueError('Frozen publication cannot be overwritten: ' + str(destination))
    else:
        shutil.copy2(source, destination)


def publish():
    require_fresh_audit()
    report = load(stage_paths()['report'])
    if report['passages'] < MIN_PASSAGES or report['words'] < MIN_WORDS:
        raise ValueError('Actual active corpus does not meet 10000 passages / 100000 words')
    # Recompute every guard rather than trusting a file saying success.
    audited = audit(require_index=True)
    frozen = ROOT / 'published' / report['corpus_sha256']
    names = ['passages.jsonl', 'quarantine.jsonl', 'reports/corpus.json',
             'reports/resegmentation_quality.json', 'index/manifest.json', 'index/vectors.npy']
    if (ROOT/'stage/reports/semantic_spotcheck.json').exists():
        names.append('reports/semantic_spotcheck.json')
    for name in names:
        freeze_file(ROOT/'stage'/name, frozen/name)
    for name in ['documents.jsonl', 'sources.jsonl']:
        freeze_file(DATA/name, frozen/name)
    freeze_file(ROOT/'baseline/manifest.json', frozen/'baseline_manifest.json')
    published = dict(report, published_snapshot=frozen.relative_to(DATA).as_posix(),
                     stage='published_with_rebuilt_index', audit_sha256=digest(stage_paths()['audit'].read_bytes()))
    frozen_report = frozen / 'reports/corpus_resegmentation.json'
    if frozen_report.exists():
        if load(frozen_report) != published:
            raise ValueError('Frozen release manifest differs')
    else:
        write_json(frozen_report, published)
    transaction = ROOT / 'publication_transaction'
    if transaction.exists():
        raise ValueError('Existing publication transaction requires explicit recovery review')
    transaction.mkdir()
    files = ['passages.jsonl', 'reports/corpus.json', 'index/vectors.npy', 'index/manifest.json',
             'index/embedding_cache.sqlite3', 'reports/resegmentation_quality.json',
             'reports/corpus_resegmentation.json']
    files = [name for name in files if name != 'index/embedding_cache.sqlite3' or (ROOT/'stage'/name).exists()]
    journal = {'at': now(), 'state': 'preparing', 'replaced': [], 'files': files,
               'old_hashes': {}, 'new_hashes': {}, 'corpus_sha256': report['corpus_sha256']}
    write_json(transaction/'journal.json', journal)
    for name in files:
        src = frozen_report if name == 'reports/corpus_resegmentation.json' else ROOT/'stage'/name
        dest, new, old = DATA/name, transaction/'new'/name, transaction/'old'/name
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, new)
        journal['new_hashes'][name] = digest(new.read_bytes())
        if dest.exists():
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, old)
            journal['old_hashes'][name] = digest(old.read_bytes())
        else:
            journal['old_hashes'][name] = None
        write_json(transaction/'journal.json', journal)
    write_json(transaction/'journal.json', journal)
    verify_baseline()  # Prevent lost update while preparing copies.
    journal['state'] = 'committing'
    write_json(transaction/'journal.json', journal)
    try:
        for name in files:
            (DATA/name).parent.mkdir(parents=True, exist_ok=True)
            (transaction/'new'/name).replace(DATA/name)
            journal['replaced'].append(name)
            write_json(transaction/'journal.json', journal)
        journal['state'] = 'committed'
        write_json(transaction/'journal.json', journal)
    except Exception:
        for name in reversed(journal['replaced']):
            if journal['old_hashes'][name] is None:
                (DATA/name).unlink()
            else:
                shutil.copy2(transaction/'old'/name, DATA/name)
        journal['state'] = 'rolled_back'
        write_json(transaction/'journal.json', journal)
        raise
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['snapshot', 'stage', 'audit', 'index', 'publish'])
    parser.add_argument('--run', default=VERSION)
    parser.add_argument('--require-index', action='store_true')
    args = parser.parse_args()
    select_run(args.run)
    with pipeline_lock():
        result = audit(args.require_index) if args.action == 'audit' else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
