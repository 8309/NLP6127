"""Optional source context and append-only admission decisions; fixtures only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from oil_evidence import unified_corpus
from oil_evidence import segmentation as real_segmentation
from oil_evidence.corpus import extraction_suffix
from oil_evidence.store import digest, read_jsonl, write_json, write_jsonl
from test_unified_pipeline import uniform, appended_source, stage_fixture, live_bytes


EXTRA = {
    'evidence_type': 'official_survey_report_with_respondent_expectations_and_comments',
    'survey_period': 'First Quarter',
    'survey_release_header': 'First Quarter | March 27, 2019',
    'publication_version_caution': 'Current collection is not an independently verified historical version.',
}
FILES = ['passages.jsonl', 'source_pages.jsonl', 'quarantine.jsonl', 'source_exclusions.jsonl']


def extra_supplement(uniform):
    module, _, _, _, _ = uniform
    module.snapshot()
    doc, source = appended_source(uniform)
    doc.update(EXTRA)
    source.update(EXTRA)
    write_jsonl(module.RUN / 'supplement/documents.jsonl', [doc])
    write_jsonl(module.RUN / 'supplement/sources.jsonl', [source])
    return doc, source


def refresh_stage_hash(module, target, filename):
    report = module.load(target / 'reports/corpus.json')
    field = {'passages.jsonl': 'corpus_sha256', 'source_pages.jsonl': 'source_pages_sha256'}[filename]
    report[field] = digest((target / filename).read_bytes())
    write_json(target / 'reports/corpus.json', report)


def admission(uniform, doc, **changes):
    module, *_ = uniform
    record = dict(document_id=doc['document_id'], raw_sha256=doc['raw_sha256'],
                  reason='visible_undated_revision_requires_historical_version',
                  decision_version='source_revision_audit_v1', proof_path='reports/offline_review.json',
                  review_sha256='a' * 64)
    record.update(changes)
    path = module.RUN / 'supplement/admission_exclusions.jsonl'
    write_jsonl(path, [record])
    return path, record


def test_original_base_provenance_contract_unchanged():
    assert unified_corpus.PROVENANCE == ['document_id', 'source', 'title', 'url', 'published_date',
        'available_by', 'date_precision', 'date_source', 'event_date', 'event_status', 'redistribution']
    assert unified_corpus.OPTIONAL_PROVENANCE == list(EXTRA)


@pytest.mark.parametrize('value', [None, '', ' \t\n', 0, False, [], {}])
def test_optional_only_real_nonempty_strings_no_null_fill(value):
    assert unified_corpus.optional_provenance({key: value for key in EXTRA}) == {}
    assert unified_corpus.optional_provenance({}) == {}


def test_optional_exact_source_strings_and_no_annotation_fields():
    exact = dict(EXTRA, survey_period='  First Quarter  ', human_gold=True, machine_label='forecast')
    expected = dict(EXTRA, survey_period='  First Quarter  ')
    assert unified_corpus.optional_provenance(exact) == expected


def test_legacy_serialized_objects_identical_to_no_optional_propagation(uniform, tmp_path):
    _, data, _, _, monkeypatch = uniform
    before = live_bytes(data)
    with monkeypatch.context() as legacy:
        legacy.setattr(unified_corpus, 'optional_provenance', lambda _: {})
        old = unified_corpus.build_unified_corpus(tmp_path / 'legacy', data_dir=data)
    current = unified_corpus.build_unified_corpus(tmp_path / 'current', data_dir=data)
    assert old == current
    for name in FILES:
        assert (tmp_path / 'legacy' / name).read_bytes() == (tmp_path / 'current' / name).read_bytes()
    for name in ['passages.jsonl', 'source_pages.jsonl']:
        assert all(not set(EXTRA) & set(row) for row in read_jsonl(tmp_path / 'current' / name))
    assert live_bytes(data) == before


def test_optional_context_matches_document_source_page_and_passage(uniform):
    module, data, *_ = uniform
    doc, source = extra_supplement(uniform)
    original_doc_bytes = (module.RUN / 'supplement/documents.jsonl').read_bytes()
    original_source_bytes = (module.RUN / 'supplement/sources.jsonl').read_bytes()
    target = stage_fixture(uniform)
    assert module.audit()['active_passages'] == 3
    for name in ['source_pages.jsonl', 'passages.jsonl']:
        rows = read_jsonl(target / name)
        matching = [row for row in rows if row['document_id'] == doc['document_id']]
        assert len(matching) == 1
        assert unified_corpus.optional_provenance(matching[0]) == EXTRA
        assert all(not set(EXTRA) & set(row) for row in rows if row['document_id'] != doc['document_id'])
        assert 'human_gold' not in matching[0] and 'machine_label' not in matching[0]
    assert read_jsonl(target / 'documents.jsonl')[-1] == doc
    assert read_jsonl(target / 'sources.jsonl')[-1] == source
    assert (module.RUN / 'supplement/documents.jsonl').read_bytes() == original_doc_bytes
    assert (module.RUN / 'supplement/sources.jsonl').read_bytes() == original_source_bytes


@pytest.mark.parametrize('key', list(EXTRA))
@pytest.mark.parametrize('mutation', ['different', 'source_missing', 'document_missing', 'source_null'])
def test_optional_supplement_source_document_mismatch_rejected(uniform, key, mutation):
    module, *_ = uniform
    doc, source = extra_supplement(uniform)
    if mutation == 'different':
        source[key] = 'Different source evidence context'
    elif mutation == 'source_missing':
        source.pop(key)
    elif mutation == 'document_missing':
        doc.pop(key)
    else:
        source[key] = None
    write_jsonl(module.RUN / 'supplement/documents.jsonl', [doc])
    write_jsonl(module.RUN / 'supplement/sources.jsonl', [source])
    with pytest.raises(ValueError, match='source metadata'):
        module.supplement_inputs()


@pytest.mark.parametrize('filename', ['source_pages.jsonl', 'passages.jsonl'])
@pytest.mark.parametrize('mutation', ['different', 'missing', 'null', 'blank'])
def test_audit_optional_context_tamper_rejected_even_with_refreshed_report_hash(uniform, filename, mutation):
    module, *_ = uniform
    doc, _ = extra_supplement(uniform)
    target = stage_fixture(uniform)
    rows = read_jsonl(target / filename)
    row = next(row for row in rows if row['document_id'] == doc['document_id'])
    if mutation == 'missing':
        row.pop('evidence_type')
    else:
        row['evidence_type'] = {'different': 'observed_physical_production', 'null': None, 'blank': ''}[mutation]
    write_jsonl(target / filename, rows)
    refresh_stage_hash(module, target, filename)
    with pytest.raises(ValueError, match='metadata|provenance'):
        module.audit()


@pytest.mark.parametrize('filename', ['source_pages.jsonl', 'passages.jsonl'])
@pytest.mark.parametrize('value', [None, '', 'invented_survey_context'])
def test_legacy_artifact_cannot_gain_optional_null_or_invented_keys(uniform, filename, value):
    module, *_ = uniform
    target = stage_fixture(uniform)
    rows = read_jsonl(target / filename)
    rows[0]['survey_period'] = value
    write_jsonl(target / filename, rows)
    refresh_stage_hash(module, target, filename)
    with pytest.raises(ValueError, match='metadata|provenance'):
        module.audit()


def test_optional_page_and_passage_both_changed_still_must_match_document(uniform):
    module, *_ = uniform
    doc, _ = extra_supplement(uniform)
    target = stage_fixture(uniform)
    for filename in ['source_pages.jsonl', 'passages.jsonl']:
        rows = read_jsonl(target / filename)
        row = next(row for row in rows if row['document_id'] == doc['document_id'])
        row['evidence_type'] = 'observed_physical_production'
        write_jsonl(target / filename, rows)
        refresh_stage_hash(module, target, filename)
    with pytest.raises(ValueError, match='metadata'):
        module.audit()


def test_admission_ledger_derives_stage_only_exclusion_and_binds_sha(uniform):
    module, _, *_ = uniform
    doc, source = extra_supplement(uniform)
    originals = {name: (module.RUN / 'supplement' / name).read_bytes()
                 for name in ['documents.jsonl', 'sources.jsonl']}
    path, decision = admission(uniform, doc)
    derived_docs, derived_sources, bound = module.supplement_inputs()
    assert bound[path.relative_to(module.DATA).as_posix()] == digest(path.read_bytes())
    expected_reason = 'audited_post_collection:' + decision['reason']
    assert derived_docs[0]['ingest_exclusion_reason'] == derived_sources[0]['ingest_exclusion_reason'] == expected_reason
    assert derived_docs[0]['admission_exclusion_proof'] == derived_sources[0]['admission_exclusion_proof'] == decision
    assert 'ingest_exclusion_reason' not in doc and 'admission_exclusion_proof' not in source
    target = stage_fixture(uniform)
    assert module.audit()['active_passages'] == 2
    assert module.load(target / 'reports/corpus.json')['supplemental_documents'] == 1
    assert all(row['document_id'] != doc['document_id'] for row in read_jsonl(target / 'passages.jsonl'))
    exclusions = read_jsonl(target / 'source_exclusions.jsonl')
    assert exclusions == [dict(document_id=doc['document_id'], scope='document', reason=expected_reason)]
    assert read_jsonl(target / 'documents.jsonl')[-1] == derived_docs[0]
    assert read_jsonl(target / 'sources.jsonl')[-1] == derived_sources[0]
    for name, value in originals.items():
        assert (module.RUN / 'supplement' / name).read_bytes() == value


@pytest.mark.parametrize('mutation', ['unknown_id', 'raw_hash', 'duplicate', 'reason', 'version', 'proof_path', 'review_sha', 'not_object'])
def test_admission_invalid_or_unbound_decisions_fail_closed(uniform, mutation):
    module, *_ = uniform
    doc, _ = extra_supplement(uniform)
    path, decision = admission(uniform, doc)
    if mutation == 'unknown_id':
        decision['document_id'] = 'unknown'
    elif mutation == 'raw_hash':
        decision['raw_sha256'] = 'b' * 64
    elif mutation == 'reason':
        decision['reason'] = '   '
    elif mutation == 'version':
        decision['decision_version'] = ''
    elif mutation == 'proof_path':
        decision['proof_path'] = None
    elif mutation == 'review_sha':
        decision['review_sha256'] = 'not-a-sha256'
    values = [decision, dict(decision)] if mutation == 'duplicate' else [decision]
    write_jsonl(path, [None] if mutation == 'not_object' else values)
    with pytest.raises(ValueError, match='Admission exclusion'):
        module.supplement_inputs()


def test_admission_file_without_manifests_is_not_silently_ignored(uniform):
    module, *_ = uniform
    write_jsonl(module.RUN / 'supplement/admission_exclusions.jsonl', [])
    with pytest.raises(ValueError, match='existing supplement'):
        module.supplement_inputs()


@pytest.mark.parametrize('mutation', ['append', 'remove'])
def test_admission_ledger_cannot_drift_after_staging(uniform, mutation):
    module, *_ = uniform
    doc, _ = extra_supplement(uniform)
    path, _ = admission(uniform, doc)
    stage_fixture(uniform)
    if mutation == 'append':
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        path.unlink()
    with pytest.raises(ValueError, match='Supplement manifest/raw/cache changed'):
        module.audit()


def test_no_admission_ledger_keeps_original_supplement_objects_exact(uniform):
    module, *_ = uniform
    doc, source = extra_supplement(uniform)
    documents, sources, bound = module.supplement_inputs()
    assert documents == [doc] and sources == [source]
    assert not any(name.endswith('admission_exclusions.jsonl') for name in bound)
    assert all('admission_exclusion_proof' not in row and 'ingest_exclusion_reason' not in row for row in documents + sources)


@pytest.fixture
def native_uniform(uniform):
    module, data, _, _, monkeypatch = uniform
    monkeypatch.setitem(sys.modules, 'oil_evidence.segmentation', real_segmentation)
    doc, source = extra_supplement(uniform)
    blocks = [dict(text=text, section_id='comments', evidence_type='survey_respondent_comment',
                   survey_release_header='First Quarter | March 27, 2019', question=None)
              for text in ['Crude oil prices are too low for our company.',
                           'Our refinery will increase oil processing in 2027.',
                           'The crude oil market remains volatile in 2027.']]
    text = '\n\n'.join(block['text'] for block in blocks)
    raw = ('<article>' + text + '</article>').encode()
    (data / doc['raw_path']).write_bytes(raw)
    doc.update(raw_sha256=digest(raw), bytes=len(raw), source_block_attribution=blocks)
    source['source_block_attribution'] = blocks
    write_json(data / 'extracted' / (doc['raw_sha256'] + extraction_suffix(doc['source'])), [text])
    write_jsonl(module.RUN / 'supplement/documents.jsonl', [doc])
    write_jsonl(module.RUN / 'supplement/sources.jsonl', [source])
    return uniform, doc, source


def test_native_comment_boundaries_and_passage_attribution_reproduce(native_uniform):
    uniform, doc, _ = native_uniform
    module, *_ = uniform
    target = stage_fixture(uniform)
    assert module.audit()['active_passages'] == 5
    page = next(row for row in read_jsonl(target / 'source_pages.jsonl') if row['document_id'] == doc['document_id'])
    rows = [row for row in read_jsonl(target / 'passages.jsonl') if row['document_id'] == doc['document_id']]
    assert len(page['source_native_blocks']) == len(rows) == 3
    assert len(page['source_hard_boundaries']) == 2
    for index, row in enumerate(rows):
        assert row['text'] == doc['source_block_attribution'][index]['text']
        assert len(row['source_block_attribution']) == 1
        assert row['source_block_attribution'][0]['source_block_index'] == index
        assert not any(row['char_start'] < boundary < row['char_end'] for boundary in page['source_hard_boundaries'])


@pytest.mark.parametrize('filename,field', [('source_pages.jsonl', 'source_hard_boundaries'),
                                         ('source_pages.jsonl', 'source_native_blocks'),
                                         ('passages.jsonl', 'source_block_attribution')])
def test_native_metadata_tamper_rejected_even_if_artifact_hash_refreshed(native_uniform, filename, field):
    uniform, doc, _ = native_uniform
    module, *_ = uniform
    target = stage_fixture(uniform)
    rows = read_jsonl(target / filename)
    row = next(row for row in rows if row['document_id'] == doc['document_id'])
    row[field] = []
    write_jsonl(target / filename, rows)
    refresh_stage_hash(module, target, filename)
    with pytest.raises(ValueError, match='Source-native|source-block'):
        module.audit()


def test_native_source_document_attribution_mismatch_rejected(native_uniform):
    uniform, _, source = native_uniform
    module, *_ = uniform
    source['source_block_attribution'][0] = dict(source['source_block_attribution'][0], evidence_type='physical_production')
    write_jsonl(module.RUN / 'supplement/sources.jsonl', [source])
    with pytest.raises(ValueError, match='source metadata'):
        module.supplement_inputs()


def test_native_alignment_cannot_use_ambiguous_substring_search():
    doc = dict(title='Oil report', source_block_attribution=[dict(text='Crude oil output was stable in 2027.')])
    with pytest.raises(ValueError, match='exact extraction'):
        unified_corpus.source_native_context(doc, 'Extra text. Crude oil output was stable in 2027.',
                                            'Extra text. Crude oil output was stable in 2027.')


def test_source_question_dependency_uses_contiguous_actual_question_not_metadata_alone():
    question = "Which of the following is your firm's primary goal in 2027?"
    texts = [question + ' Executives expect to grow crude oil production in 2027.',
             'Small firms produced less than 10,000 barrels per day. Their primary goal is grow production.']
    doc = dict(title='Oil report', source_block_attribution=[dict(text=text, question=question,
               evidence_type='survey_analysis_or_expectation') for text in texts])
    raw = '\n\n'.join(texts)
    clean = unified_corpus.clean_source_page(raw, doc)[0]
    page = dict(text=clean, **unified_corpus.source_native_context(doc, raw, clean))
    required = unified_corpus.source_question_dependencies(page)
    assert required == [dict(start=0, end=len(clean), question=question, question_is_local=True)]
    result = real_segmentation.segment_page(clean, hard_boundaries=page['source_hard_boundaries'], required_contexts=required)
    assert result['accepted'][0]['text'] == clean
    assert question in result['accepted'][0]['text']
    assert not result['quarantined']
    # The metadata is not enough if the actual source question is absent.
    missing = dict(text=texts[1], source_native_blocks=[dict(char_start=0, char_end=len(texts[1]), question=question,
                                                          independent_source_block=False)])
    required = unified_corpus.source_question_dependencies(missing)
    assert required[0]['question_is_local'] is False
    held = real_segmentation.segment_page(missing['text'], required_contexts=required)
    assert not held['accepted']
    assert held['quarantined'][0]['reason'] == 'source_question_temporal_context_missing'


def test_temporal_question_group_stops_at_independent_respondent_and_new_question():
    question = 'What are your oil production goals for 2027?'
    texts = [question + ' Our crude oil production will rise in 2027.',
             'Our crude oil output will increase.',
             'A different anonymous respondent expects higher crude oil output.']
    blocks = [dict(text=text, question=question, evidence_type=('survey_respondent_comment' if i == 2 else 'survey_analysis_or_expectation'))
              for i, text in enumerate(texts)]
    doc = dict(title='Oil report', source_block_attribution=blocks)
    raw = '\n\n'.join(texts)
    page = dict(text=raw, **unified_corpus.source_native_context(doc, raw, raw))
    required = unified_corpus.source_question_dependencies(page)
    assert required[0]['end'] == len('\n\n'.join(texts[:2]))
    result = real_segmentation.segment_page(raw, hard_boundaries=page['source_hard_boundaries'], required_contexts=required)
    assert len(result['accepted']) == 2
    assert result['accepted'][1]['text'] == texts[2]


@pytest.mark.parametrize('footer', ['Summary i USDOI | BOEM', 'Summary xiv USDOI | BOEM',
                                  'Chapter 2: Affected Environment 113 USDOI | BOEM'])
def test_real_agency_page_footer_is_removed_but_actual_summary_heading_and_body_remain(footer):
    raw = footer + '\nSummary\nBACKGROUND\nCrude oil leases are scheduled for 2027.'
    clean, removed = unified_corpus.clean_source_page(raw)
    assert footer not in clean
    assert 'Summary BACKGROUND' in clean
    assert 'Crude oil leases are scheduled for 2027.' in clean
    assert removed == [dict(line=1, text=footer, reason='running_page_footer')]


def test_summary_page_lowercase_continuation_is_held_after_real_footer_removal():
    tail = 'as exploration and development plan review, will provide more detail and incorporate new oil information.'
    body = 'The proposed oil leasing program will schedule five lease sales.'
    raw = 'Summary ii USDOI | BOEM\n' + tail + '\n\n' + body
    clean, _ = unified_corpus.clean_source_page(raw)
    result = real_segmentation.segment_page(clean)
    assert result['quarantined'][0]['text'] == tail
    assert result['quarantined'][0]['reason'] == 'incomplete_initial_edge'
    assert result['accepted'][0]['text'] == body


def test_summary_text_with_ordinary_page_or_agency_mentions_is_not_deleted():
    raw = 'The summary on page 4 describes crude oil leasing decisions by USDOI and BOEM.'
    clean, removed = unified_corpus.clean_source_page(raw)
    assert clean == raw and not removed


def save_source_semantic_change_report(previous, current, output):
    """Offline diagnostic export for exact source-ID changes, never gold labels."""
    previous, current, output = Path(previous), Path(current), Path(output)
    if output.exists():
        raise ValueError('Source semantic change reports are immutable')
    old = {row['passage_id']: row for row in read_jsonl(previous / 'passages.jsonl')}
    new = {row['passage_id']: row for row in read_jsonl(current / 'passages.jsonl')}
    held = read_jsonl(current / 'quarantine.jsonl')
    changes = []
    for identifier in sorted(set(old) - set(new)):
        row = old[identifier]
        def overlaps(candidate):
            return (candidate['source_page_id'] == row['source_page_id']
                    and candidate['char_start'] < row['char_end'] and row['char_start'] < candidate['char_end'])
        changes.append(dict(old_passage_id=identifier, source=row['source'], url=row['url'], old_text=row['text'],
            new_accepted=[dict(passage_id=other['passage_id'], text=other['text']) for other in new.values() if overlaps(other)],
            new_quarantined=[dict(text=other['text'], reason=other['reason']) for other in held if overlaps(other)]))
    report = dict(diagnostic_only=True, human_gold=False, external_model_calls=0, human_labels_created=0,
        old_corpus_sha256=digest((previous / 'passages.jsonl').read_bytes()),
        new_corpus_sha256=digest((current / 'passages.jsonl').read_bytes()),
        old_passages=len(old), new_passages=len(new), removed_ids=len(set(old) - set(new)),
        added_ids=len(set(new) - set(old)), unchanged_ids=len(set(old) & set(new)), changes=changes,
        source_pages_byte_equivalent=(previous / 'source_pages.jsonl').read_bytes() == (current / 'source_pages.jsonl').read_bytes())
    write_json(output, report)
    return report
