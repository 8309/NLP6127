"""Close the saved, fixed 55-case review without reselecting/rebuilding inputs.

Explicit judgments below were authored after source reading. Hash/offset guards
only authorize reuse of unchanged prior machine judgments, never human gold.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from oil_evidence.store import DATA, ROOT, digest, read_jsonl, write_json
from review_buffer_case_inputs import FILES, PROTECTED, hashes


ROOT_RUN = DATA / 'unified_runs/semantic-page-v2-buffer'
SNAPSHOT = ROOT_RUN / 'reports/source_qa_successor_inputs/5005bed72bc7'
PREVIOUS = ROOT_RUN / 'reports/source_qa_inputs/4881e7d689aa'
OUTPUT = ROOT_RUN / 'reports/buffer_source_review_final.json'
ORIGINAL_FAILURE_SHA = '6b525a192753dcf8762ce9f9efb43953d1cdecae33743b0678943b3109d00500'


def spans(items):
    return [(item['text'], item['char_start'], item['char_end'], item.get('reason')) for item in items]


def finalize():
    if OUTPUT.exists():
        raise ValueError('Preserve existing final review; do not overwrite its judgments')
    original_path = ROOT_RUN / 'reports/buffer_source_review.json'
    receipt_path = SNAPSHOT / 'review_receipts.json'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if digest(original_path.read_bytes()) != ORIGINAL_FAILURE_SHA or receipt['original_failed_review_sha256'] != ORIGINAL_FAILURE_SHA:
        raise ValueError('Original failed review changed')
    if len(receipt['records']) != 57 or receipt['fixed_sample_count'] != 55 or receipt['extra_diagnostic_count'] != 2:
        raise ValueError('Keep every fixed case and both named diagnostics')
    stage = ROOT_RUN / 'stage'
    implementation = receipt['implementation_hashes']
    if (hashes(stage, FILES) != receipt['input_hashes'] or hashes(SNAPSHOT, FILES) != receipt['input_hashes']
            or hashes(DATA, PROTECTED) != receipt['protected_live_hashes']
            or any(digest((ROOT / path).read_bytes()) != sha or digest((SNAPSHOT / 'implementation' / path).read_bytes()) != sha
                   for path, sha in implementation.items())):
        raise ValueError('Stage, source snapshot, CODE or protected live files drifted')
    previous_docs = {doc['document_id']: doc for doc in read_jsonl(PREVIOUS / 'documents.jsonl')}
    prior_pages = {page['source_page_id']: page for page in read_jsonl(PREVIOUS / 'source_pages.jsonl')}
    records = []
    identity = ['document_id', 'source', 'title', 'url', 'published_date', 'available_by', 'date_precision', 'date_source', 'raw_sha256']
    raw_checked = set()
    for item in receipt['records']:
        old = item['original_passage']
        doc = item['current_document']
        previous_doc = previous_docs[old['document_id']]
        if any(doc.get(key) != previous_doc.get(key) for key in identity):
            raise ValueError('Source identity/publication availability changed for a fixed case')
        raw = DATA / doc['raw_path']
        if doc['document_id'] not in raw_checked:
            if digest(raw.read_bytes()) != doc['raw_sha256']:
                raise ValueError('Original raw source changed')
            raw_checked.add(doc['document_id'])
        old_page = prior_pages[old['source_page_id']]
        if old_page['text'][old['char_start']:old['char_end']] != old['text']:
            raise ValueError('Original fixed text/offsets changed')
        current = item['current_source_page']
        if current is None or any(current.get(key) != doc.get(key) for key in identity if key != 'raw_sha256'):
            raise ValueError('Current source page or provenance missing')
        for row in item['new_accepted_spans'] + item['new_quarantined_spans']:
            if current['text'][row['char_start']:row['char_end']] != row['text']:
                raise ValueError('Current accepted/quarantined source offsets changed')
        prior = item['original_record']
        if item['origin'] == 'original_html_fixed_case':
            judgment = dict(prior['machine_resolution'])
        else:
            judgment = dict(prior['machine_judgment'])
        changed = False
        if item.get('case') == 38:
            q = 'Which of the following is your firm’s primary goal in 2022?'
            matches = [row for row in item['new_accepted_spans'] if q in row['text'] and old['text'] in row['text']]
            if len(matches) != 1 or matches[0]['word_count'] != 134 or len(matches[0].get('source_block_attribution', [])) != 2:
                raise ValueError('Case 38 did not bind the actual question and two contiguous source answers')
            judgment = dict(severity='pass', status='resolved_by_actual_contiguous_temporal_question_context',
                judgment='Fresh raw h4 and two native p match the 134-word accepted source interval 6031:6810. The indexed text itself now contains 2022, eight-goal question context, executive response percentages and the small/large-firm classification/goals. No question/year is copied from metadata; no table percentages are reconstructed. These are survey goals, not realized 2022 production.')
            changed = True
        elif old['passage_id'] == '04800dc442f5a2f285fc-u-21dad21eb3cae557e878a807':
            if item['new_accepted_spans'] or not any(row['text'] == old['text'] and row['reason'] == 'clipped_coordinate_subject_at_initial_edge'
                                                    for row in item['new_quarantined_spans']):
                raise ValueError('The p131 coordinate-suffix fragment still survives')
            judgment = dict(severity='pass', status='resolved_by_initial_edge_quarantine',
                judgment='Original physical pages 130-131 and saved renders remain source-bound: Harris County is on p130, while p131 starts with its TX suffix. The exact 20-word p131 fragment is now held as clipped_coordinate_subject_at_initial_edge; no accepted overlap survives and no Harris County words are invented or duplicated. Remaining independent page prose is not substituted as this old answer.')
            changed = True
        elif old['passage_id'] == '04800dc442f5a2f285fc-u-f8916188a6bc24c904f28cf7':
            if 'Summary i USDOI | BOEM' in current['text'] or not current['text'].startswith('Summary BACKGROUND'):
                raise ValueError('Summary p3 real footer was not isolated from body heading')
            judgment = dict(severity='moderate', status='footer_resolved_remaining_citation_formatting_warning',
                judgment='Additional original diagnostic retained, not a replacement sample. Original p3/render confirms the complete Summary i/USDOI|BOEM footer line was removed and recorded; actual Summary/Background headings and program body remain. Superscript footnote markers and legal-citation segmentation remain imperfect. Content is a draft leasing-policy/assessment description, not realized output; this is not a clean-layout or whole-corpus approval.')
            changed = True
        elif old['passage_id'] == '04800dc442f5a2f285fc-u-e68382f9fdd1cd259738ccfb':
            tail = 'as exploration and development plan review, will provide more detail and incorporate any new information.'
            if ('Summary ii USDOI | BOEM' in current['text'] or any(tail in row['text'] for row in item['new_accepted_spans'])
                    or not any(row['text'] == tail and row['reason'] == 'incomplete_initial_edge' for row in item['new_quarantined_spans'])):
                raise ValueError('Summary p4 incomplete continuation or real footer still survives')
            judgment = dict(severity='pass', status='resolved_by_real_footer_removal_and_continuation_quarantine',
                judgment='Additional original diagnostic retained. Fresh p3-p4/source-render comparison confirms the p4 lower-case as... text continues the incomplete NEPA sentence from p3. The exact 15-word continuation is now held as incomplete_initial_edge and the recorded real Summary ii/USDOI|BOEM footer is absent from cleaned body. No accepted fragment contains that tail; true program alternatives remain separately traceable, never relabeled as completed lease sales.')
            changed = True
        if not changed:
            same_disposition = (spans(item['new_accepted_spans']) == spans(prior.get('new_accepted_spans', [prior['passage']] if 'passage' in prior else []))
                                and spans(item['new_quarantined_spans']) == spans(prior.get('new_quarantined_spans', [])))
            if not item['exact_source_page_text_unchanged'] or not same_disposition:
                raise ValueError('Cannot reuse a semantic judgment after source text or disposition drift')
        fresh_successor_read = item.get('case') == 38 or item['origin'] != 'original_html_fixed_case'
        proof = dict(raw_path=doc['raw_path'], raw_sha256=doc['raw_sha256'], raw_reverified=True,
                     physical_page=old['page'], published_date=doc['published_date'], available_by=doc['available_by'],
                     event_date=doc.get('event_date'), evidence_type=doc.get('evidence_type'),
                     publication_version_caution=doc.get('publication_version_caution'), original_source_offsets_verified=True,
                     current_source_offsets_verified=True, source_identity_and_dates_unchanged=True)
        records.append(dict(item, machine_judgment=judgment, human_gold=False,
                            fresh_source_read=fresh_successor_read, prior_fresh_source_read=True,
                            judgment_reused=not changed, source_verification=proof,
                            decision_basis=('fresh_machine_read_of_actual_modified_context' if changed else
                                'prior_fresh_machine_judgment_reused_only_after_identical_entire_source_dates_offsets_and_dispositions'),
                            fresh_read_caution='Fresh physical PDF pages were read for all eleven BOEM samples; unchanged HTML cases reuse their original fresh DOM judgment after exact hash/full-page/date/offset/disposition verification. No independent-human judgment.'))
    fixed = [row for row in records if row['origin'] != 'extra_diagnostic_not_fixed_sample']
    extras = [row for row in records if row['origin'] == 'extra_diagnostic_not_fixed_sample']
    critical = [row['original_passage']['passage_id'] for row in records if row['machine_judgment']['severity'] == 'critical']
    warnings = [dict(origin=row['origin'], case=row.get('case'), original_id=row['original_passage']['passage_id'], judgment=row['machine_judgment'])
                for row in records if row['machine_judgment']['severity'] == 'moderate']
    if (hashes(stage, FILES) != receipt['input_hashes'] or hashes(DATA, PROTECTED) != receipt['protected_live_hashes']
            or any(digest((ROOT / path).read_bytes()) != sha for path, sha in implementation.items())
            or digest(original_path.read_bytes()) != ORIGINAL_FAILURE_SHA):
        raise ValueError('Protected state drifted while finalizing')
    report = dict(receipt, records=records, status='fixed_machine_sample_no_remaining_critical_not_whole_corpus_approval',
        reviewer='Codex machine source review; not a human annotator', remaining_critical=critical, remaining_critical_count=len(critical),
        fixed_sample_severity_counts=dict(Counter(row['machine_judgment']['severity'] for row in fixed)),
        extra_diagnostic_severity_counts=dict(Counter(row['machine_judgment']['severity'] for row in extras)), warnings=warnings,
        coverage=dict(fixed_sample_completed=55, fixed_sample_required=55, original_html_cases=44, original_boem_cases=11,
                      extra_diagnostics_completed=2, changed_judgments_freshly_revised=4, unchanged_prior_judgments_reused=53,
                      successor_source_semantic_reads=14, unchanged_html_semantic_judgments_reused=43,
                      boem_fixed_physical_pages_freshly_read=11, all_raw_date_offset_disposition_cases_verified=57,
                      unique_original_raw_documents_verified=len(raw_checked)),
        source_critical_zero_for_fixed_review=not critical, human_gold=False, human_labels_created=0, external_model_calls=0,
        subscription_credit_usage='Not measured; machine review is not a zero-cost or human-annotation claim.',
        whole_corpus_approved=False, release_approved=False, index_created=False,
        initial_failed_review_preserved=True, original_failed_review_sha256=ORIGINAL_FAILURE_SHA,
        successor_receipts_sha256=digest(receipt_path.read_bytes()), finalizer_sha256=digest(Path(__file__).read_bytes()),
        limitation='Fixed 55 previous cases plus 2 named diagnostics only; moderate attribution/formatting/peripheral-evidence warnings remain. The new 917-page PDF has a separate reviewer/sample. This does not certify all 12,327 passages, migrate labels, build/publish an index, or satisfy formal human annotation.')
    write_json(OUTPUT, report)
    return report


if __name__ == '__main__':
    result = finalize()
    print(json.dumps(dict(output=str(OUTPUT), corpus_sha256=result['corpus_sha256'],
                          source_pages_sha256=result['source_pages_sha256'], remaining_critical_count=result['remaining_critical_count'],
                          coverage=result['coverage'], fixed=result['fixed_sample_severity_counts'],
                          extra=result['extra_diagnostic_severity_counts'], report_sha256=digest(OUTPUT.read_bytes())), indent=2))
