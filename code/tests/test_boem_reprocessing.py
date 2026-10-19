"""Local-only repair, tamper checks, and immutable supplemental provenance."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def repair_fixture(monkeypatch, tmp_path):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('boem_reprocessing_test_subject', scripts / 'reprocess_boem_supplement.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / 'data'
    runroot = data / 'unified_runs/repair-fixture'
    target = runroot / 'supplement'
    target.mkdir(parents=True)
    monkeypatch.setattr(module, 'DATA', data)
    monkeypatch.setattr(module, 'writer_lock', lambda run: nullcontext())
    raw = b'%PDF-local-only fixture'
    url = 'https://www.boem.gov/environment/environmental-assessment/final-goa-peispdf'
    identifier = module.digest(url)[:20]
    raw_path = data / 'raw' / (identifier + '.pdf')
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(raw)
    source = dict(url=url, source='BOEM-EIS', title='Final oil and gas PEIS', published_date='2025-08-29',
        available_by='2025-08-29', date_precision='day', date_source='https://www.boem.gov/index',
        event_date=None, event_status='unlabelled', redistribution='us_government_with_attribution',
        requires_pdf=True, body_date_verified=False, body_parser_sha256='a' * 64,
        ingest_exclusion_reason='unverified_pdf_publication_or_extraction',
        body_parser_error='cryptography>=3.1 is required for AES algorithm')
    doc = dict(source, document_id=identifier, final_url='https://www.boem.gov/assets/report.pdf',
        content_type='application/pdf', raw_path=raw_path.relative_to(data).as_posix(),
        raw_sha256=module.digest(raw), bytes=len(raw), fetched_at='2026-10-12T00:00:00+00:00')
    module.append_jsonl(target / 'documents.jsonl', doc)
    module.append_jsonl(target / 'sources.jsonl', source)
    # A blank page is intentionally retained as a physical page, never omitted.
    pages = ['Final oil and gas PEIS\nPublished by\nBureau of Ocean Energy Management August 2025', '',
             'Oil production scenarios are projections, not observed production.']
    parsed = dict(pages=pages, published_date='2025-08-29', available_by='2025-09-08',
        indexed_available_by='2025-08-29', date_precision='day', date_confirmed=True, cover_month='2025-08',
        pdf_creation_date='2025-09-04', pdf_modified_date='2025-09-08', physical_pages=3,
        parser='boem_fixture_physical_order_parser', version_caution='Not available before 2025-09-08; not observed output.')
    calls = []
    def parse(raw_bytes, original_source):
        assert raw_bytes == raw and original_source == source
        calls.append(True)
        return copy.deepcopy(parsed)
    monkeypatch.setattr(module.parser_module, 'parse_boem_pdf', parse)
    monkeypatch.setattr(module, '_pdf_dates', lambda raw_bytes: ('2025-09-04', '2025-09-08'))
    return dict(module=module, data=data, runroot=runroot, target=target, doc=doc, source=source,
                raw_path=raw_path, raw=raw, parsed=parsed, calls=calls)


def process(fixture):
    module = fixture['module']
    return module.reprocess_existing('repair-fixture', document_ids=[fixture['doc']['document_id']])


def apply(fixture):
    return fixture['module'].apply_reprocessing_admissions([fixture['doc']], [fixture['source']],
        data=fixture['data'], runroot=fixture['runroot'])


def ledger(fixture):
    return fixture['target'] / 'reprocessing_admissions.jsonl'


def alter_row(fixture, edit):
    module = fixture['module']
    rows = module.read_jsonl(ledger(fixture))
    edit(rows[0])
    ledger(fixture).write_text(json.dumps(rows[0], ensure_ascii=False) + '\n', encoding='utf-8')


def test_new_repair_preserves_raw_manifests_and_input_objects(repair_fixture):
    f = repair_fixture
    originals = {path: path.read_bytes() for path in (f['raw_path'], f['target'] / 'documents.jsonl', f['target'] / 'sources.jsonl')}
    original_doc, original_source = copy.deepcopy(f['doc']), copy.deepcopy(f['source'])
    report = process(f)
    docs, sources, bound = apply(f)
    assert f['doc'] == original_doc and f['source'] == original_source
    assert all(path.read_bytes() == blob for path, blob in originals.items())
    assert report['external_model_calls'] == 0 and report['human_labels_created'] == 0
    assert report['semantic_approval'] is False and report['accepted_passages'] is None
    assert docs[0]['available_by'] == sources[0]['available_by'] == '2025-09-08'
    assert docs[0]['published_date'] == sources[0]['published_date'] == '2025-08-29'
    assert 'ingest_exclusion_reason' not in docs[0] and 'body_parser_error' not in docs[0]
    assert docs[0]['prior_error']['body_parser_error'] == original_doc['body_parser_error']
    assert docs[0]['physical_pages'] == 3
    row = f['module'].read_jsonl(ledger(f))[0]
    cache = json.loads((f['data'] / row['cache']['path']).read_text(encoding='utf-8'))
    full_pages = json.loads((f['data'] / row['source_pages']['path']).read_text(encoding='utf-8'))
    assert len(cache) == len(full_pages) == 3 and cache[1] == full_pages[1] == ''
    assert bound[row['cache']['path']] == row['cache']['sha256']
    assert row['original_document_sha256'] == f['module'].canonical_sha(original_doc)
    assert row['original_source_sha256'] == f['module'].canonical_sha(original_source)


def test_no_ledger_returns_independent_copies(repair_fixture):
    f = repair_fixture
    docs, sources, bound = apply(f)
    docs[0]['title'] = 'changed returned copy'
    sources[0]['title'] = 'changed returned source copy'
    assert f['doc']['title'] == f['source']['title'] == 'Final oil and gas PEIS'
    assert bound == {}


def test_rerun_does_not_duplicate_admission_or_reextract(repair_fixture):
    f = repair_fixture
    process(f)
    original = ledger(f).read_bytes()
    report = process(f)
    assert ledger(f).read_bytes() == original
    assert len(f['calls']) == 1
    assert report['results'][0]['status'] == 'already_admitted_not_duplicated'


def test_unknown_id_is_not_guessed_or_downloaded(repair_fixture):
    f = repair_fixture
    with pytest.raises(ValueError, match='known and unique'):
        f['module'].reprocess_existing('repair-fixture', document_ids=['unknown'])
    assert f['calls'] == [] and not ledger(f).exists()


def test_preexisting_cache_refuses_overwrite_even_same_data(repair_fixture):
    f = repair_fixture
    cache = f['data'] / 'extracted' / (f['doc']['raw_sha256'] + '-body-v2.json')
    cache.parent.mkdir()
    cache.write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='cache cannot be overwritten'):
        process(f)
    assert cache.read_text(encoding='utf-8') == '[]' and not ledger(f).exists() and f['calls'] == []


def test_parser_date_failure_stays_excluded_and_not_duplicated(repair_fixture, monkeypatch):
    f = repair_fixture
    def fail(*args):
        raise ValueError('BOEM first two physical pages lack a standalone publication month')
    monkeypatch.setattr(f['module'].parser_module, 'parse_boem_pdf', fail)
    first = process(f)
    second = process(f)
    assert first['results'][0]['status'] == second['results'][0]['status'] == 'still_excluded'
    assert not ledger(f).exists()
    failures = f['module'].read_jsonl(f['target'] / 'reprocessing_failures.jsonl')
    assert len(failures) == 1 and failures[0]['human_gold'] is False


@pytest.mark.parametrize('key,value', [
    ('url', 'https://www.boem.gov/different'), ('title', 'different'), ('source', 'EIA-CAB'),
    ('published_date', '2025-08-30'), ('raw_path', 'raw/other.pdf'), ('raw_sha256', 'b' * 64),
    ('bytes', 99), ('event_date', '2025-08-30'), ('redistribution', 'different'), ('event_status', 'implemented'),
    ('fetched_at', '2025-08-29'), ('label', 'human-approved'),
])
def test_positive_cannot_rewrite_identity_or_add_labels(repair_fixture, key, value):
    f = repair_fixture
    process(f)
    alter_row(f, lambda row: row['derived_document'].__setitem__(key, value))
    with pytest.raises(ValueError, match='immutable metadata|unauthorized fields'):
        apply(f)


@pytest.mark.parametrize('field', ['original_document_sha256', 'original_source_sha256', 'raw_sha256'])
def test_original_object_and_raw_sha_are_bound(repair_fixture, field):
    f = repair_fixture
    process(f)
    alter_row(f, lambda row: row.__setitem__(field, 'f' * 64))
    with pytest.raises(ValueError, match='canonical objects/raw'):
        apply(f)


@pytest.mark.parametrize('value', ['2025-08-28', '2026-10-06', '2025-9-8', None])
def test_positive_cannot_backdate_or_pass_cutoff(repair_fixture, value):
    f = repair_fixture
    process(f)
    alter_row(f, lambda row: row['derived_document'].__setitem__('available_by', value))
    with pytest.raises(ValueError):
        apply(f)


@pytest.mark.parametrize('artifact', ['cache', 'audit', 'source_pages', 'proof'])
def test_every_artifact_has_content_sha_binding(repair_fixture, artifact):
    f = repair_fixture
    process(f)
    row = f['module'].read_jsonl(ledger(f))[0]
    path = f['data'] / row[artifact]['path']
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='artifact bytes differ'):
        apply(f)


@pytest.mark.parametrize('artifact', ['cache', 'audit', 'source_pages', 'proof'])
def test_artifact_cannot_escape_its_allowed_root(repair_fixture, artifact):
    f = repair_fixture
    process(f)
    alter_row(f, lambda row: row[artifact].__setitem__('path', '../../outside.json'))
    with pytest.raises(ValueError, match='escapes its allowed root|missing'):
        apply(f)


@pytest.mark.parametrize('implementation', ['parser', 'margin_cleaner', 'script'])
def test_actual_implementation_paths_and_shas_are_bound(repair_fixture, implementation):
    f = repair_fixture
    process(f)
    alter_row(f, lambda row: row['implementations'][implementation].__setitem__('sha256', 'f' * 64))
    with pytest.raises(ValueError, match='implementation path or SHA drifted'):
        apply(f)


def test_duplicate_positive_admission_is_rejected(repair_fixture):
    f = repair_fixture
    process(f)
    f['module'].append_jsonl(ledger(f), f['module'].read_jsonl(ledger(f))[0])
    with pytest.raises(ValueError, match='duplicate positive'):
        apply(f)


def test_negative_ledger_cannot_be_overridden(repair_fixture):
    f = repair_fixture
    process(f)
    f['module'].append_jsonl(f['target'] / 'admission_exclusions.jsonl',
        dict(document_id=f['doc']['document_id'], raw_sha256=f['doc']['raw_sha256'], reason='semantically_invalid'))
    with pytest.raises(ValueError, match='cannot override.*negative admission'):
        apply(f)


def test_negative_overlay_on_input_cannot_be_cleared(repair_fixture):
    f = repair_fixture
    process(f)
    for row in (f['doc'], f['source']):
        row['ingest_exclusion_reason'] = 'audited_post_collection:semantically_invalid'
    with pytest.raises(ValueError, match='cannot override.*negative admission'):
        apply(f)


@pytest.mark.parametrize('field,value', [('model_only', False), ('human_gold', True), ('semantic_approval', True), ('model_only', 1)])
def test_machine_version_check_never_becomes_human_or_semantic_approval(repair_fixture, field, value):
    f = repair_fixture
    process(f)
    alter_row(f, lambda row: row.__setitem__(field, value))
    with pytest.raises(ValueError, match='machine-only'):
        apply(f)


def test_raw_alteration_is_rejected_without_cache_regeneration(repair_fixture):
    f = repair_fixture
    process(f)
    f['raw_path'].write_bytes(f['raw'] + b'changed')
    with pytest.raises(ValueError, match='raw bytes/path/SHA differ'):
        apply(f)


def test_current_pdf_metadata_rechecked_not_just_ledger(repair_fixture, monkeypatch):
    f = repair_fixture
    process(f)
    monkeypatch.setattr(f['module'], '_pdf_dates', lambda raw: ('2025-09-04', '2026-10-06'))
    with pytest.raises(ValueError, match='outside fixed window'):
        apply(f)


def test_derived_source_and_document_must_be_synchronized(repair_fixture):
    f = repair_fixture
    process(f)
    alter_row(f, lambda row: row['derived_source'].__setitem__('physical_pages', 2))
    with pytest.raises(ValueError, match='source/document metadata differs'):
        apply(f)
