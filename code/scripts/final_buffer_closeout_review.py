"""Final linear rebind of the unchanged 57 cases and complete accepted deltas."""
from collections import defaultdict
import json
from pathlib import Path
import shutil

from oil_evidence.store import DATA, ROOT, digest, read_jsonl, words, write_json
from review_buffer_case_inputs import FILES, PROTECTED, hashes
from closeout_buffer_fixed_review import row_content


RUN = DATA / 'unified_runs/semantic-page-v2-buffer'
PREVIOUS = RUN / 'reports/source_qa_successor_inputs/5005bed72bc7'
INTERMEDIATE = RUN / 'reports/source_qa_closeout_inputs/19b82c5e4392'
PRIOR = RUN / 'reports/buffer_source_review_final.json'
FAILED = RUN / 'reports/buffer_source_review-closeout.json'
OUTPUT = RUN / 'reports/buffer_source_review-closeout-final.json'


def finalize():
    if OUTPUT.exists():
        raise ValueError('Preserve final review; never overwrite its judgments')
    previous_bytes, failed_bytes = PRIOR.read_bytes(), FAILED.read_bytes()
    previous = json.loads(previous_bytes)
    stage = RUN / 'stage'
    before = hashes(stage, FILES)
    report = json.loads((stage / 'reports/corpus.json').read_text(encoding='utf-8'))
    code = report['implementation_hashes']
    if (before['passages.jsonl'] != report['corpus_sha256'] or before['source_pages.jsonl'] != report['source_pages_sha256']
            or any(digest((ROOT / path).read_bytes()) != sha for path, sha in code.items())
            or hashes(DATA, PROTECTED) != previous['protected_live_hashes']):
        raise ValueError('Frozen stage, CODE or live input drifted')
    target = RUN / 'reports/source_qa_final_closeout_inputs' / report['corpus_sha256'][:12]
    if target.exists():
        raise ValueError('Keep immutable final inputs')
    for path in FILES:
        dest = target / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage / path, dest)
    for path in code:
        dest = target / 'implementation' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, dest)
    shutil.copy2(Path(__file__), target / 'review_implementation.py')
    shutil.copy2(PRIOR, target / 'prior_5005_fixed_review.json')
    shutil.copy2(FAILED, target / 'prior_19b_failed_review.json')
    rows = {row['passage_id']: row for row in read_jsonl(stage / 'passages.jsonl')}
    held = read_jsonl(stage / 'quarantine.jsonl')
    pages = {page['source_page_id']: page for page in read_jsonl(stage / 'source_pages.jsonl')}
    docs = {doc['document_id']: doc for doc in read_jsonl(stage / 'documents.jsonl')}
    accepted_by_page, held_by_page = defaultdict(list), defaultdict(list)
    for row in rows.values(): accepted_by_page[row['source_page_id']].append(row)
    for row in held: held_by_page[row['source_page_id']].append(row)
    records, raw_checked = [], set()
    for item in previous['records']:
        page, doc = item['current_source_page'], item['current_document']
        if pages.get(page['source_page_id']) != page or docs[doc['document_id']] != doc:
            raise ValueError('Full fixed source/document/date changed; fresh semantic review needed')
        if doc['document_id'] not in raw_checked:
            if digest((DATA / doc['raw_path']).read_bytes()) != doc['raw_sha256']:
                raise ValueError('Fixed raw source changed')
            raw_checked.add(doc['document_id'])
        intervals = item['equal_character_intervals']
        matches = lambda row: any(row['char_start'] < b and a < row['char_end'] for a, b in intervals)
        accepted = [row for row in accepted_by_page[page['source_page_id']] if matches(row)]
        quarantined = [row for row in held_by_page[page['source_page_id']] if matches(row)]
        if row_content(accepted) != row_content(item['new_accepted_spans']) or row_content(quarantined) != row_content(item['new_quarantined_spans']):
            raise ValueError('A fixed accepted/held disposition changed')
        for row in accepted + quarantined:
            if page['text'][row['char_start']:row['char_end']] != row['text']:
                raise ValueError('Exact fixed offsets changed')
        records.append(dict(item, new_accepted_spans=accepted, new_quarantined_spans=quarantined,
                            final_judgment_reused=True, final_fresh_semantic_read=False,
                            final_full_source_metadata_date_raw_offset_disposition_verified=True))
    delta = {}
    prior_versions = {}
    for name, path in [('intermediate_19b82', INTERMEDIATE), ('original_5005', PREVIOUS)]:
        old = {row['passage_id']: row for row in read_jsonl(path / 'passages.jsonl')}
        removed, added = [row for id_, row in old.items() if id_ not in rows], [row for id_, row in rows.items() if id_ not in old]
        prior_versions[name] = old
        delta[name] = dict(prior_passages=len(old), final_passages=len(rows), direct_removed_count=len(removed),
                          direct_removed_words=sum(row['word_count'] for row in removed), newly_accepted_count=len(added),
                          newly_accepted_words=sum(row['word_count'] for row in added), net_reduction_count=len(removed)-len(added),
                          removed_passages=removed, newly_accepted_passages=added)
        if added:
            raise ValueError('All actual new accepted passages need fresh source review, not automatic reuse')
    # These checks verify content as well as IDs, including the original p422
    # entry that survived inside a different, longer p504 bibliography chunk.
    suffixes = ['c005b43089a832f6256eb715', '94841ec42a2f7b368138338c', '19584450103a9f31c9bb8639', 'c355bfa90ed7a23d695a9de7']
    targets = {row['passage_id']: row for old in prior_versions.values() for row in old.values()
               if any(row['passage_id'].endswith(suffix) for suffix in suffixes)}
    normalized = {id_: ' '.join(words(row['text'])) for id_, row in rows.items()}
    residue = {id_: [new_id for new_id, text in normalized.items() if ' '.join(words(row['text'])) in text]
               for id_, row in targets.items()}
    if any(residue.values()):
        raise ValueError('Bibliographic content remains accepted, even under a different ID')
    exclusions = read_jsonl(stage / 'source_exclusions.jsonl')
    native = [row for row in exclusions if 'native_section' in row]
    sections = {}
    for row in native:
        proof = row['native_section']
        key = str(row['document_id']) + ':' + str(proof['first_physical_page'])
        sections[key] = proof
    if (hashes(stage, FILES) != before or hashes(DATA, PROTECTED) != previous['protected_live_hashes']
            or any(digest((ROOT / path).read_bytes()) != sha for path, sha in code.items())
            or PRIOR.read_bytes() != previous_bytes or FAILED.read_bytes() != failed_bytes):
        raise ValueError('Stage, historical reports, CODE or live input drifted during final rebind')
    result = dict(previous, records=records, corpus_sha256=report['corpus_sha256'], source_pages_sha256=report['source_pages_sha256'],
        input_hashes=before, implementation_hashes=code, status='unchanged_fixed57_rebound_no_new_accepted_ids_bibliographic_residue_removed',
        remaining_critical=[], remaining_critical_count=0, source_critical_zero_for_fixed_review=True,
        final_coverage=dict(original_fixed_cases_rebound=55, original_extra_diagnostics_rebound=2,
                            original57_semantic_judgments_reused=57, final_fresh_semantic_judgments=0,
                            all57_full_source_metadata_date_raw_offset_dispositions_unchanged=True,
                            prior31_moderate_warnings_preserved=True, unique_fixed_raw_documents_reverified=len(raw_checked),
                            new_accepted_ids_since_19b82=0, new_accepted_ids_since_5005=0),
        delta=delta, bibliographic_residue_checks=residue, native_physical_page_holds=len(native), native_sections=sections,
        section_hold_caution='Whole mixed start pages sacrifice some valid body prose. M references run through printed M-90, physical916; physical917 has no M-91 and is conservatively held with failed validation, never claimed a verified reference page.',
        source_page_counts=dict(original_5005=5910, intermediate_19b82=5732, final=len(pages)),
        previous_fixed_review_sha256=digest(previous_bytes), intermediate_failed_review_sha256=digest(failed_bytes),
        snapshot=str(target), finalizer_sha256=digest(Path(__file__).read_bytes()), whole_corpus_approved=False,
        release_approved=False, index_created=False, human_labels_created=0, external_model_calls=0,
        limitation='Original55+2 limited machine judgments reused only after exact full-source equality; no fresh whole-corpus semantic approval or human gold. The independent new917 fixed11 review is a separate report. No new accepted IDs were available for additional semantic review.')
    write_json(OUTPUT, result)
    return result


if __name__ == '__main__':
    result = finalize()
    print(json.dumps(dict(output=str(OUTPUT), corpus_sha256=result['corpus_sha256'], source_pages_sha256=result['source_pages_sha256'],
        critical=result['remaining_critical_count'], native_physical_holds=result['native_physical_page_holds'],
        coverage=result['final_coverage'], report_sha256=digest(OUTPUT.read_bytes()),
        delta={name:{key:value for key,value in d.items() if key not in ('removed_passages','newly_accepted_passages')} for name,d in result['delta'].items()}), indent=2))
