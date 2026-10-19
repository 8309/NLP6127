"""Freeze the exact same 44+11 cases and named diagnostics after repairs.

Character alignment is a disposition receipt only, never semantic approval.
Does not choose replacement samples, rebuild an index, or write live data.
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import shutil

from oil_evidence.store import DATA, ROOT, digest, read_jsonl, write_json
from review_buffer_case_inputs import FILES, PROTECTED, hashes


def freeze(run='semantic-page-v2-buffer'):
    root = DATA / 'unified_runs' / run
    if root.resolve().parent != (DATA / 'unified_runs').resolve():
        raise ValueError('Review run escapes unified_runs')
    original_path = root / 'reports/buffer_source_review.json'
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    previous = root / 'reports/source_qa_inputs' / original['corpus_sha256'][:12]
    stage = root / 'stage'
    before = hashes(stage, FILES)
    report = json.loads((stage / 'reports/corpus.json').read_text(encoding='utf-8'))
    if (report['corpus_sha256'] != before['passages.jsonl'] or report['source_pages_sha256'] != before['source_pages.jsonl']
            or any(digest((ROOT / path).read_bytes()) != sha for path, sha in report['implementation_hashes'].items())):
        raise ValueError('Stage or implementation is not frozen/current')
    if before['passages.jsonl'] == original['corpus_sha256']:
        raise ValueError('Successor review requires a genuinely rebuilt version')
    live = hashes(DATA, PROTECTED)
    if live != original['protected_live_hashes']:
        raise ValueError('Protected live corpus/index/labels changed before source review')
    target = root / 'reports/source_qa_successor_inputs' / before['passages.jsonl'][:12]
    if target.exists():
        raise ValueError('Preserve the existing immutable successor snapshot')
    for path in FILES:
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage / path, destination)
    for path, sha in report['implementation_hashes'].items():
        destination = target / 'implementation' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, destination)
        if digest(destination.read_bytes()) != sha:
            raise ValueError('Frozen code bytes differ')
    shutil.copy2(original_path, target / 'previous_failed_review.json')
    shutil.copy2(Path(__file__), target / 'review_implementation.py')
    old_pages = {row['source_page_id']: row for row in read_jsonl(previous / 'source_pages.jsonl')}
    new_pages = {(row['document_id'], row['page']): row for row in read_jsonl(target / 'source_pages.jsonl')}
    docs = {row['document_id']: row for row in read_jsonl(target / 'documents.jsonl')}
    accepted, held = read_jsonl(target / 'passages.jsonl'), read_jsonl(target / 'quarantine.jsonl')
    samples = []
    for item in original['old_receipts']:
        old = item['original_case']
        row = dict(passage_id=old['id'], document_id=old['doc'], page=old['page'], source_page_id=old['source_page_id'],
                   char_start=old['char_start'], char_end=old['char_end'], text=old['text'])
        samples.append(dict(origin='original_html_fixed_case', case=old['case'], original_record=item, original_passage=row))
    for item in original['boem_fixed_samples']:
        samples.append(dict(origin='boem_fixed_sample', original_record=item, original_passage=item['passage']))
    for item in original['extra_diagnostics']:
        samples.append(dict(origin='extra_diagnostic_not_fixed_sample', original_record=item, original_passage=item['passage']))
    if len(samples) != 57 or len({item['original_passage']['passage_id'] for item in samples}) != 57:
        raise ValueError('Preserve all original 55 fixed cases plus the two named diagnostics')
    for item in samples:
        old = item['original_passage']
        page = old_pages[old['source_page_id']]
        if page['text'][old['char_start']:old['char_end']] != old['text']:
            raise ValueError('Original source offsets or text changed')
        current = new_pages.get((old['document_id'], old['page']))
        doc = docs[old['document_id']]
        if digest((DATA / doc['raw_path']).read_bytes()) != doc['raw_sha256'] or doc['raw_sha256'] != page['raw_sha256']:
            raise ValueError('Physical source bytes changed, not just derived cleaning')
        mapped = []
        if current:
            for block in SequenceMatcher(None, page['text'], current['text'], autojunk=False).get_matching_blocks():
                a, b = max(old['char_start'], block.a), min(old['char_end'], block.a + block.size)
                if a < b:
                    mapped.append((block.b + a - block.a, block.b + b - block.a))
        def matches(row):
            return current and row['source_page_id'] == current['source_page_id'] and any(
                row['char_start'] < b and a < row['char_end'] for a, b in mapped)
        item.update(current_source_page=current, current_document=doc,
                    new_accepted_spans=[row for row in accepted if matches(row)],
                    new_quarantined_spans=[row for row in held if matches(row)],
                    exact_source_page_text_unchanged=bool(current and current['text'] == page['text']),
                    equal_character_intervals=mapped, mapped_character_count=sum(b-a for a, b in mapped),
                    fresh_source_read=False, machine_judgment=None, human_gold=False)
    if hashes(stage, FILES) != before or hashes(DATA, PROTECTED) != live or original_path.read_bytes() != original_bytes:
        raise ValueError('Stage/live/original failure report drifted while freezing')
    receipt = dict(status='pending_same_case_fresh_machine_source_review_not_gold', corpus_sha256=before['passages.jsonl'],
                   source_pages_sha256=before['source_pages.jsonl'], input_hashes=before,
                   implementation_hashes=report['implementation_hashes'], protected_live_hashes=live,
                   original_failed_review_sha256=digest(original_bytes), original_corpus_sha256=original['corpus_sha256'],
                   fixed_sample_count=55, extra_diagnostic_count=2, records=samples, external_model_calls=0,
                   human_gold=False, human_labels_created=0, whole_corpus_approved=False, release_approved=False,
                   reviewer_script_sha256=digest(Path(__file__).read_bytes()),
                   limitation='Alignment is a mechanical source-local receipt, not semantic similarity, retrieval scoring, human review or support verification.')
    write_json(target / 'review_receipts.json', receipt)
    return target, receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='semantic-page-v2-buffer')
    args = parser.parse_args()
    path, receipt = freeze(args.run)
    print(json.dumps(dict(snapshot=str(path), corpus_sha256=receipt['corpus_sha256'], fixed_sample_count=55,
                          extra_diagnostic_count=2, status=receipt['status']), indent=2))
