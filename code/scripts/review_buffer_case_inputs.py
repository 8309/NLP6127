"""Freeze read-only source-review inputs and old-case disposition receipts.

Receipts are mechanical mappings, not answers, annotations or gold judgments.
All old fixed cases and failures remain verbatim; no retrieval/model is called.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import re

from oil_evidence.store import DATA, ROOT, digest, read_jsonl, write_json


FILES = ['passages.jsonl', 'source_pages.jsonl', 'quarantine.jsonl', 'source_exclusions.jsonl',
         'documents.jsonl', 'sources.jsonl', 'reports/corpus.json']
PROTECTED = ['passages.jsonl', 'documents.jsonl', 'sources.jsonl', 'index/manifest.json', 'index/vectors.npy',
             'annotations/passage_labels.jsonl', 'annotations/passage_review.jsonl']


def hashes(root, names):
    return {name: digest((root / name).read_bytes()) for name in names}


def overlapping(record, old):
    return (record['source_page_id'] == old['source_page_id']
            and record['char_start'] < old['char_end'] and old['char_start'] < record['char_end'])


def freeze_inputs(run='semantic-page-v2-buffer'):
    root = DATA / 'unified_runs' / run
    if root.resolve().parent != (DATA / 'unified_runs').resolve():
        raise ValueError('Review run escapes unified_runs')
    stage = root / 'stage'
    old_path = root / 'reports/html_source_review.json'
    old_bytes = old_path.read_bytes()
    old_review = json.loads(old_bytes)
    originals = old_review['records']
    if len(originals) != 44 or len({row['id'] for row in originals}) != 44:
        raise ValueError('Keep all 44 original fixed source-review cases')
    live_before = hashes(DATA, PROTECTED)
    before = hashes(stage, FILES)
    report = json.loads((stage / 'reports/corpus.json').read_text(encoding='utf-8'))
    if (report['corpus_sha256'] != before['passages.jsonl'] or report['source_pages_sha256'] != before['source_pages.jsonl']
            or any(digest((ROOT / name).read_bytes()) != sha for name, sha in report['implementation_hashes'].items())):
        raise ValueError('Stage report or frozen implementation is stale')
    snapshot = root / 'reports/source_qa_inputs' / before['passages.jsonl'][:12]
    if snapshot.exists():
        raise ValueError('Source-review snapshot already exists; do not replace it')
    for name in FILES:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage / name, destination)
    shutil.copy2(old_path, snapshot / 'original_html_source_review.json')
    rows = read_jsonl(snapshot / 'passages.jsonl')
    held = read_jsonl(snapshot / 'quarantine.jsonl')
    pages = {row['source_page_id']: row for row in read_jsonl(snapshot / 'source_pages.jsonl')}
    docs = {doc['document_id']: doc for doc in read_jsonl(snapshot / 'documents.jsonl')}
    receipts = []
    for old in originals:
        page = pages.get(old['source_page_id'])
        accepted = [row for row in rows if overlapping(row, old)]
        rejected = [row for row in held if overlapping(row, old)]
        exact = any(row['passage_id'] == old['id'] and row['text'] == old['text'] for row in accepted)
        disposition = ('source_excluded' if page is None else 'retained_exact' if exact else
                       'rebound_or_repartitioned' if accepted else 'quarantined' if rejected else 'unaccounted')
        if page is not None and page['text'][old['char_start']:old['char_end']] != old['text']:
            raise ValueError('Old fixed case text/offsets changed in the current source page')
        doc = docs[old['doc']]
        raw = (DATA / doc['raw_path']).resolve()
        if not raw.is_relative_to((DATA / 'raw').resolve()) or digest(raw.read_bytes()) != doc['raw_sha256']:
            raise ValueError('Fixed source-review raw bytes changed')
        receipts.append(dict(original_case=old, new_disposition=disposition,
                             new_accepted_spans=accepted, new_quarantined_spans=rejected,
                             new_document_metadata=doc, exact_old_offsets=page is not None,
                             fresh_source_read=False, machine_resolution=None, remaining_issue=None,
                             human_gold=False))
    old_doc_ids = {doc['document_id'] for doc in read_jsonl(DATA / 'unified_runs/semantic-page-v1/stage/documents.jsonl')}
    groups = defaultdict(list)
    for row in rows:
        if row['source'].startswith('BOEM') and row['document_id'] not in old_doc_ids:
            groups[row['source']].append(row)
    samples = {}
    for source, group in sorted(groups.items()):
        selected = (sorted(group, key=lambda row: digest(row['passage_id']))[:5]
                    + sorted(group, key=lambda row: (row['word_count'], digest(row['passage_id'])))[:3]
                    + sorted(group, key=lambda row: (-row['word_count'], digest(row['passage_id'])))[:3])
        for row in selected:
            samples[row['passage_id']] = dict(passage=row, document=docs[row['document_id']],
                source_page=pages[row['source_page_id']], fresh_source_read=False, machine_judgment=None,
                source_pdf_page_render=None, human_gold=False)
    if (hashes(stage, FILES) != before or hashes(DATA, PROTECTED) != live_before
            or old_path.read_bytes() != old_bytes or hashes(snapshot, FILES) != before):
        raise ValueError('Input/live/source snapshot changed while freezing source-review receipts')
    receipt = dict(status='pending_fresh_machine_source_review_not_gold', diagnostic_only=True,
        human_gold=False, external_model_calls=0, human_labels_created=0, corpus_sha256=before['passages.jsonl'],
        source_pages_sha256=before['source_pages.jsonl'], input_hashes=before,
        original_html_review_sha256=digest(old_bytes), implementation_hashes=report['implementation_hashes'],
        protected_live_hashes=live_before, fixed_old_case_count=44, old_receipts=receipts,
        new_boem_denominators={source: len(group) for source, group in groups.items()},
        boem_selection_policy='Per new BOEM source: first five SHA256(passage_id), shortest three, longest three; SHA ties; union before judgments.',
        boem_fixed_samples=list(samples.values()), reviewer_script_sha256=digest(Path(__file__).read_bytes()),
        limitation='Mechanical disposition is not source-semantic approval, human annotation or exhaustive whole-corpus review.')
    write_json(snapshot / 'review_receipts.json', receipt)
    write_json(snapshot / 'snapshot_manifest.json', dict(input_hashes=before, corpus_sha256=before['passages.jsonl'],
                                                       original_html_review_sha256=digest(old_bytes)))
    return snapshot, receipt


def finish_source_review(snapshot, judgments_path, output):
    """Attach explicitly authored machine judgments to freshly read source proofs.

    This does not label the corpus, repair production code, or approve release.
    Keep old failures, frozen inputs and implementation bytes alongside the review.
    """
    from bs4 import BeautifulSoup
    from pypdf import PdfReader

    snapshot, judgments_path, output = Path(snapshot), Path(judgments_path), Path(output)
    receipt = json.loads((snapshot / 'review_receipts.json').read_text(encoding='utf-8'))
    judgments = json.loads(judgments_path.read_text(encoding='utf-8'))
    if output.exists():
        raise ValueError('Review output already exists; preserve it and use a successor version')
    if hashes(snapshot, FILES) != receipt['input_hashes'] or hashes(DATA, PROTECTED) != receipt['protected_live_hashes']:
        raise ValueError('Frozen inputs or protected live files drifted')
    implementation = receipt['implementation_hashes']
    parent_snapshot = DATA / 'unified_runs/semantic-page-v2-buffer/qa_inputs' / receipt['corpus_sha256'] / 'implementation'
    implementation_inputs = {}
    for name, sha in implementation.items():
        source = ROOT / name
        if digest(source.read_bytes()) != sha:
            source = parent_snapshot / name
        if not source.exists() or digest(source.read_bytes()) != sha:
            raise ValueError('Cannot recover exact stage implementation bytes: ' + name)
        implementation_inputs[name] = source
    for name in implementation:
        destination = snapshot / 'implementation_files' / name
        if destination.exists():
            if digest(destination.read_bytes()) != implementation[name]:
                raise ValueError('Existing frozen implementation bytes differ')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(implementation_inputs[name], destination)
    old_decisions = judgments['old_cases']
    boem_decisions = judgments['boem_samples']
    if {int(key) for key in old_decisions} != set(range(1, 45)):
        raise ValueError('Every original fixed case needs an explicit judgment')
    fixed_ids = {item['passage']['passage_id'] for item in receipt['boem_fixed_samples']}
    if set(boem_decisions) != fixed_ids:
        raise ValueError('Keep the fixed eleven BOEM cases, without substitution')
    pages = {row['source_page_id']: row for row in read_jsonl(snapshot / 'source_pages.jsonl')}
    docs = {row['document_id']: row for row in read_jsonl(snapshot / 'documents.jsonl')}
    rows = {row['passage_id']: row for row in read_jsonl(snapshot / 'passages.jsonl')}
    soups, readers = {}, {}

    def normalized(value):
        return re.sub(r'[^a-z0-9]', '', value.lower())

    def html_proof(old, doc):
        raw = (DATA / doc['raw_path']).read_bytes()
        if digest(raw) != doc['raw_sha256']:
            raise ValueError('Raw HTML changed during source review')
        soup = soups.setdefault(doc['document_id'], BeautifulSoup(raw, 'html.parser'))
        nodes = soup.find_all(['p', 'li', 'h2', 'h3', 'h4', 'h5'])
        expected = old.get('actual_raw_dom_matches', [])
        if expected:
            selected = [node for match in expected for node in nodes if str(node) == match['outer_html']]
            if len(selected) != len(expected):
                raise ValueError('Original critical DOM evidence no longer matches the actual raw source')
        else:
            anchors = [normalized(old['text'])[:100], normalized(old['text'])[-100:]]
            selected = [node for node in nodes if any(len(anchor) >= 20 and anchor in normalized(node.get_text(' ', strip=True)) for anchor in anchors)]
            if not selected:
                raise ValueError(f"Cannot locate fresh actual DOM for case {old['case']}")
            selected = sorted(selected, key=lambda node: len(node.get_text(' ', strip=True)))[:3]
        result = []
        for node in selected:
            index = next(i for i, item in enumerate(nodes) if item is node)
            result.append(dict(tag=node.name, native_node_index=index, text=node.get_text(' ', strip=True),
                               outer_html=str(node), outer_html_sha256=digest(str(node)),
                               previous_native_nodes=[item.get_text(' ', strip=True) for item in nodes[max(0, index-2):index]],
                               following_native_nodes=[item.get_text(' ', strip=True) for item in nodes[index+1:index+3]]))
        return dict(raw_path=doc['raw_path'], raw_sha256=doc['raw_sha256'], actual_raw_dom=result)

    reviewed_old = []
    for item in receipt['old_receipts']:
        old = item['original_case']
        page = pages[old['source_page_id']]
        if page['text'][old['char_start']:old['char_end']] != old['text']:
            raise ValueError('An original case changed text or offsets')
        proof = html_proof(old, item['new_document_metadata'])
        reviewed_old.append(dict(item, fresh_source_read=True,
            fresh_source_proof=proof, machine_resolution=old_decisions[str(old['case'])],
            source_page_context=page['text'][max(0, old['char_start']-450):old['char_end']+450]))

    def pdf_proof(row, doc):
        raw_path = DATA / doc['raw_path']
        if digest(raw_path.read_bytes()) != doc['raw_sha256']:
            raise ValueError('Raw PDF changed during source review')
        if doc['document_id'] not in readers:
            readers[doc['document_id']] = PdfReader(raw_path)
        reader = readers[doc['document_id']]
        page_no = row['page']
        selected_pages = sorted(set([page_no, max(1, page_no-1), min(len(reader.pages), page_no+1)]))
        extracted = [dict(physical_page=n, text=reader.pages[n-1].extract_text() or '') for n in selected_pages]
        render = ROOT / 'tmp/pdfs/buffer-source-qa-4881e7' / f"{doc['document_id']}-p{page_no}.png"
        return dict(raw_path=doc['raw_path'], raw_sha256=doc['raw_sha256'], physical_page=page_no,
                    extracted_neighbor_pages=extracted,
                    page_render_path=str(render) if render.exists() else None,
                    page_render_sha256=digest(render.read_bytes()) if render.exists() else None,
                    render_caution='Draft Poppler used Helvetica substitution for MyriadPro; text, paragraph boundaries and margins inspected, not font-perfect proof.')

    reviewed_boem = [dict(item, fresh_source_read=True, source_pdf_proof=pdf_proof(item['passage'], item['document']),
                         machine_judgment=boem_decisions[item['passage']['passage_id']])
                     for item in receipt['boem_fixed_samples']]
    diagnostics = []
    for id_, decision in judgments.get('extra_diagnostics', {}).items():
        row = rows[id_]
        diagnostics.append(dict(passage=row, machine_judgment=decision, fresh_source_read=True,
                                source_pdf_proof=pdf_proof(row, docs[row['document_id']]),
                                fixed_sample=False, human_gold=False))
    remaining = [dict(origin='original_html_fixed_case', case=item['original_case']['case'],
                      passage_ids=[row['passage_id'] for row in item['new_accepted_spans']],
                      judgment=item['machine_resolution']) for item in reviewed_old
                 if item['machine_resolution']['severity'] == 'critical']
    remaining += [dict(origin='new_boem_fixed_sample', passage_ids=[item['passage']['passage_id']],
                       judgment=item['machine_judgment']) for item in reviewed_boem
                  if item['machine_judgment']['severity'] == 'critical']
    remaining += [dict(origin='extra_diagnostic_not_fixed_sample', passage_ids=[item['passage']['passage_id']],
                       judgment=item['machine_judgment']) for item in diagnostics
                  if item['machine_judgment']['severity'] == 'critical']
    if hashes(snapshot, FILES) != receipt['input_hashes'] or hashes(DATA, PROTECTED) != receipt['protected_live_hashes']:
        raise ValueError('Frozen inputs or protected live files drifted during source review')
    report = dict(receipt, status='fresh_machine_source_review_failed_not_release_approval',
        old_receipts=reviewed_old, boem_fixed_samples=reviewed_boem, extra_diagnostics=diagnostics,
        fixed_source_review_count=55, remaining_critical=remaining, remaining_critical_count=len(remaining),
        judgments_path=str(judgments_path), judgments_sha256=digest(judgments_path.read_bytes()),
        reviewer_script_sha256=digest(Path(__file__).read_bytes()), code_snapshot_path=str(snapshot / 'implementation_files'),
        subscription_credit_usage='Not measured; this is Codex subscription machine review, not zero-cost or human annotation.',
        release_approved=False, whole_corpus_approved=False,
        limitation='44 prior fixed cases plus fixed 11 new BOEM samples and separately named diagnostics; no exhaustive semantic coverage, no human gold, no retrieval ranking, no index publication.')
    write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='semantic-page-v2-buffer')
    parser.add_argument('--finish-snapshot')
    parser.add_argument('--judgments')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.finish_snapshot:
        if not args.judgments or not args.output:
            parser.error('--finish-snapshot requires --judgments and --output')
        review = finish_source_review(args.finish_snapshot, args.judgments, args.output)
        print(json.dumps(dict(output=args.output, corpus_sha256=review['corpus_sha256'],
                              fixed_source_review_count=review['fixed_source_review_count'],
                              remaining_critical_count=review['remaining_critical_count'], status=review['status']), indent=2))
        raise SystemExit(0)
    snapshot, receipt = freeze_inputs(args.run)
    print(json.dumps(dict(snapshot=str(snapshot), corpus_sha256=receipt['corpus_sha256'],
                         old_cases=receipt['fixed_old_case_count'], boem_denominators=receipt['new_boem_denominators'],
                         boem_samples=len(receipt['boem_fixed_samples']), status=receipt['status']), indent=2))
