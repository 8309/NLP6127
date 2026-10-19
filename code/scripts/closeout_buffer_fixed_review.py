"""Bind unchanged fixed cases and the real corpus delta to a new checkpoint.

No resampling, alignment, download, label, index or publication operation.
"""
from collections import defaultdict
import json
from pathlib import Path
import shutil

from oil_evidence.store import DATA, ROOT, digest, read_jsonl, write_json
from review_buffer_case_inputs import FILES, PROTECTED, hashes


RUN = DATA / 'unified_runs/semantic-page-v2-buffer'
PRIOR = RUN / 'reports/source_qa_successor_inputs/5005bed72bc7'
PRIOR_REPORT = RUN / 'reports/buffer_source_review_final.json'
OUTPUT = RUN / 'reports/buffer_source_review-closeout.json'


def row_content(rows):
    return [(row['source_page_id'], row['text'], row['char_start'], row['char_end'], row.get('reason')) for row in rows]


def run():
    if OUTPUT.exists():
        raise ValueError('Do not overwrite a previous closeout review')
    prior_bytes = PRIOR_REPORT.read_bytes()
    previous = json.loads(prior_bytes)
    stage = RUN / 'stage'
    before = hashes(stage, FILES)
    report = json.loads((stage / 'reports/corpus.json').read_text(encoding='utf-8'))
    impl = report['implementation_hashes']
    if (before['passages.jsonl'] != report['corpus_sha256'] or before['source_pages.jsonl'] != report['source_pages_sha256']
            or any(digest((ROOT / name).read_bytes()) != sha for name, sha in impl.items())
            or hashes(DATA, PROTECTED) != previous['protected_live_hashes']):
        raise ValueError('Stage/CODE/live drift before review')
    snapshot = RUN / 'reports/source_qa_closeout_inputs' / report['corpus_sha256'][:12]
    if snapshot.exists():
        raise ValueError('Preserve the existing checkpoint snapshot')
    for name in FILES:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage / name, dest)
    for name in impl:
        dest = snapshot / 'implementation' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(PRIOR_REPORT, snapshot / 'previous_fixed_review.json')
    shutil.copy2(Path(__file__), snapshot / 'review_implementation.py')
    rows = {row['passage_id']: row for row in read_jsonl(stage / 'passages.jsonl')}
    old_rows = {row['passage_id']: row for row in read_jsonl(PRIOR / 'passages.jsonl')}
    held, prior_held = read_jsonl(stage / 'quarantine.jsonl'), read_jsonl(PRIOR / 'quarantine.jsonl')
    pages = {page['source_page_id']: page for page in read_jsonl(stage / 'source_pages.jsonl')}
    docs = {doc['document_id']: doc for doc in read_jsonl(stage / 'documents.jsonl')}
    accepted_by_page, held_by_page = defaultdict(list), defaultdict(list)
    for row in rows.values(): accepted_by_page[row['source_page_id']].append(row)
    for row in held: held_by_page[row['source_page_id']].append(row)
    records = []
    checked_raw = set()
    for item in previous['records']:
        page = item['current_source_page']
        doc = item['current_document']
        if pages.get(page['source_page_id']) != page or docs[doc['document_id']] != doc:
            raise ValueError('Full fixed source page/document/date changed: fresh review required')
        if doc['document_id'] not in checked_raw:
            if digest((DATA / doc['raw_path']).read_bytes()) != doc['raw_sha256']:
                raise ValueError('Fixed raw source changed')
            checked_raw.add(doc['document_id'])
        intervals = item['equal_character_intervals']
        matches = lambda row: any(row['char_start'] < b and a < row['char_end'] for a, b in intervals)
        accepted = [row for row in accepted_by_page[page['source_page_id']] if matches(row)]
        quarantined = [row for row in held_by_page[page['source_page_id']] if matches(row)]
        if (row_content(accepted) != row_content(item['new_accepted_spans'])
                or row_content(quarantined) != row_content(item['new_quarantined_spans'])):
            raise ValueError('Fixed accepted/quarantined disposition changed: fresh review required')
        records.append(dict(item, new_accepted_spans=accepted, new_quarantined_spans=quarantined,
                            closeout_judgment_reused=True, closeout_fresh_semantic_read=False,
                            closeout_identity_raw_dates_offsets_dispositions_verified=True))
    removed = [row for id_, row in old_rows.items() if id_ not in rows]
    added = [row for id_, row in rows.items() if id_ not in old_rows]
    additions = []
    for row in added:
        proof = [q for q in prior_held if q['source_page_id'] == row['source_page_id']
                 and q['text'] == row['text'] and q['char_start'] == row['char_start'] and q['char_end'] == row['char_end']]
        if len(proof) != 1 or proof[0]['reason'] != 'exact_normalized_duplicate' or proof[0]['duplicate_of'] not in {r['passage_id'] for r in removed}:
            raise ValueError('New accepted delta is not the expected released duplicate')
        doc = docs[row['document_id']]
        if digest((DATA / doc['raw_path']).read_bytes()) != doc['raw_sha256']:
            raise ValueError('Delta raw source changed')
        additions.append(dict(passage=row, prior_quarantine=proof[0], removed_duplicate=old_rows[proof[0]['duplicate_of']],
                              source_page=pages[row['source_page_id']], raw_sha256=doc['raw_sha256'],
                              machine_judgment=dict(severity='critical', status='newly_exposed_bibliographic_entry',
                              judgment='Fresh original PDF/native cache inspection confirms a bibliography, not factual body evidence. Exact deduplication suppressed this copy while Appendix A occupied the normalized-text fingerprint. Removing that first copy releases this other reference entry, which must be source-semantically held, not counted as a safe new fact.'),
                              human_gold=False, fresh_source_read=True))
    exclusions = read_jsonl(stage / 'source_exclusions.jsonl')
    native = [row for row in exclusions if 'native_section' in row]
    if hashes(stage, FILES) != before or hashes(DATA, PROTECTED) != previous['protected_live_hashes'] or PRIOR_REPORT.read_bytes() != prior_bytes:
        raise ValueError('Protected inputs or prior final review changed')
    result = dict(previous, records=records, corpus_sha256=report['corpus_sha256'], source_pages_sha256=report['source_pages_sha256'],
        input_hashes=before, implementation_hashes=impl, status='fixed_cases_unchanged_but_new_delta_bibliographic_failures',
        fixed_sample_remaining_critical_count=0, new_delta_critical_count=len(additions), remaining_critical_count=len(additions),
        remaining_critical=[row['passage']['passage_id'] for row in additions], source_critical_zero_for_fixed_review=True,
        closeout_coverage=dict(fixed_cases_rebound_without_reselection=55, original_extra_diagnostics_rebound=2,
                              all57_full_source_document_date_raw_offset_dispositions_equal=True,
                              closeout_new_semantic_judgments_for_original57=0, original57_judgments_reused=57,
                              added_delta_passages_freshly_read=len(additions)),
        prior_complete_review_sha256=digest(prior_bytes), previous_corpus_sha256=previous['corpus_sha256'],
        delta=dict(direct_removed_count=len(removed), direct_removed_words=sum(row['word_count'] for row in removed),
                   newly_accepted_count=len(added), newly_accepted_words=sum(row['word_count'] for row in added),
                   net_reduction_count=len(removed)-len(added), net_reduction_words=sum(row['word_count'] for row in removed)-sum(row['word_count'] for row in added),
                   source_pages_before=len(read_jsonl(PRIOR / 'source_pages.jsonl')), source_pages_after=len(pages),
                   native_physical_pages_held=len(native), removed_passages=removed, new_accepted_details=additions,
                   caution='Direct removal 475 is not net reduction 473; two released duplicate bibliographic entries are new criticals, not safe new body evidence.'),
        snapshot=str(snapshot), finalizer_sha256=digest(Path(__file__).read_bytes()), release_approved=False,
        whole_corpus_approved=False, human_labels_created=0, external_model_calls=0, index_created=False)
    write_json(OUTPUT, result)
    return result


if __name__ == '__main__':
    result = run()
    print(json.dumps(dict(output=str(OUTPUT), corpus_sha256=result['corpus_sha256'], fixed_critical=result['fixed_sample_remaining_critical_count'],
                          delta_critical=result['new_delta_critical_count'], report_sha256=digest(OUTPUT.read_bytes()),
                          delta={key:value for key,value in result['delta'].items() if key not in ('removed_passages','new_accepted_details')}), indent=2))
