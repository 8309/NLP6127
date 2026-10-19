"""Read-only, deterministic feasibility estimates for shorter evidence passages.

This writes only its analysis report. It never changes the corpus, labels,
configuration, indexes, snapshots, or evaluation records. Counts are structural
candidates, not a claim of semantic approval or assignment acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re

from oil_evidence.corpus import oil_topic_supported
from oil_evidence.store import DATA, ROOT, config, digest, now, read_jsonl, words, write_json


BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9“"(])')
ABBREVIATION = re.compile(
    r'(?:\b(?:U\.S|U\.K|E\.U|e\.g|i\.e|Dr|Mr|Mrs|Ms|Prof|Fig|Eq|No|vs|approx)|\b[A-Z])\.$', re.I)
TERMINAL = re.compile(r'[.!?][”\"\x27’)]*$')
DEPENDENT_START = re.compile(
    r'^(?:This|These|Those|It|Its|They|Their|Such|The latter|The former|'
    r'However|Therefore|Consequently|Nevertheless|Furthermore|Additionally|'
    r'In addition|As a result|For example|For instance|Also|Similarly|In contrast|'
    r'During the same period|Over the same period|Other advantages|Benefits|'
    r'Approaches|As noted earlier|In other words|Since then)\b', re.I)


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


@lru_cache(maxsize=100000)
def word_count(text):
    return len(words(text))


@lru_cache(maxsize=20000)
def safe_boundaries(text):
    """Punctuation heuristics only; preserve exact source text token order."""
    positions = [0]
    for match in BOUNDARY.finditer(text):
        before = text[max(0, match.start()-25):match.start()]
        if not ABBREVIATION.search(before):
            positions.append(match.end())
    positions.append(len(text))
    return positions


def propose_split(text, minimum, maximum=130, screen_dependency=False):
    text = normalized(text)
    if not re.match(r'^[A-Z0-9“"(]', text) or not TERMINAL.search(text):
        return None, 'parent_not_complete_at_edges'
    if word_count(text) < 2 * minimum:
        return None, 'parent_too_short'
    positions = safe_boundaries(text)
    if len(positions) < 3:
        return None, 'no_sentence_boundary'
    preferred = (minimum + maximum) / 2

    @lru_cache(maxsize=None)
    def solve(start):
        if start == len(positions)-1:
            return (0.0, ())
        possibilities = []
        for end in range(start+1, len(positions)):
            child = text[positions[start]:positions[end]].strip()
            size = word_count(child)
            if size > maximum:
                break
            if size < minimum or not TERMINAL.search(child):
                continue
            if not oil_topic_supported(child):
                continue
            if screen_dependency and DEPENDENT_START.search(child):
                continue
            suffix = solve(end)
            if suffix is not None:
                score = (size-preferred)**2 + suffix[0]
                possibilities.append((score, (end,)+suffix[1]))
        return min(possibilities, default=None)

    answer = solve(0)
    if answer is None or len(answer[1]) < 2:
        return None, 'no_partition_with_complete_oil_children'
    starts = (0,)+answer[1][:-1]
    children = [text[positions[start]:positions[end]].strip()
                for start, end in zip(starts, answer[1])]
    assert normalized(' '.join(children)) == text
    assert words(' '.join(children)) == words(text)
    assert all(minimum <= len(words(child)) <= maximum for child in children)
    assert all(TERMINAL.search(child) and oil_topic_supported(child) for child in children)
    return children, None


def estimate(rows, protected_ids, minimum, screen_dependency, holdout_ids=frozenset(), maximum=130):
    originals = {digest(' '.join(words(p['text']))): p['passage_id'] for p in rows}
    assert len(originals) == len(rows), 'Live corpus already has exact normalized duplicates'
    accepted_children = set()
    proposals = []
    rejected = Counter()
    for parent in rows:
        if parent['passage_id'] in protected_ids:
            continue
        if parent['passage_id'] in holdout_ids:
            rejected['pending_near_duplicate_parent_held_unchanged'] += 1
            continue
        children, reason = propose_split(parent['text'], minimum, maximum, screen_dependency=screen_dependency)
        if children is None:
            rejected[reason] += 1
            continue
        fingerprints = [digest(' '.join(words(text))) for text in children]
        if len(set(fingerprints)) != len(fingerprints) or any(
                fp in originals or fp in accepted_children for fp in fingerprints):
            rejected['child_exact_duplicate'] += 1
            continue
        accepted_children.update(fingerprints)
        proposals.append((parent, children))
    extra = sum(len(children)-1 for _, children in proposals)
    return {
        'policy': {'minimum_words': minimum, 'maximum_words': maximum,
                   'preferred_words': (minimum+maximum)/2, 'overlap_words': 0,
                   'complete_punctuation_edges_required': True,
                   'oil_keyword_required_in_every_child': True,
                   'dependency_start_screen': screen_dependency,
                   'pending_near_duplicate_guard': bool(holdout_ids),
                   'natural_paragraph_boundaries_available': False,
                   'exact_duplicate_children_rejected': True},
        'parents_replaced': len(proposals),
        'child_passages': sum(len(children) for _, children in proposals),
        'net_extra_passages': extra,
        'potential_passages_after_replacement': len(rows)+extra,
        'meets_numeric_10000_before_semantic_review': len(rows)+extra >= 10000,
        'rejections': dict(rejected),
        'by_source_net_extra': dict(Counter(
            parent['source'] for parent, children in proposals for _ in range(len(children)-1))),
        'child_word_lengths': dict(sorted(Counter(
            len(words(child)) for _, children in proposals for child in children).items())),
    }, proposals


def self_check():
    assert safe_boundaries('The U.S. Government supports oil. Oil production rose.') == [0, 34, 54]
    assert propose_split('oil words without sentence punctuation', 65)[0] is None
    text = 'Oil production ' + 'has increased ' * 37 + '. ' + 'Crude reserves ' + 'are sufficient ' * 37 + '.'
    children, error = propose_split(text, 65)
    assert error is None and len(children) == 2
    assert words(' '.join(children)) == words(text)
    assert not DEPENDENT_START.search('Oil production increased.')
    assert DEPENDENT_START.search('This increase followed sanctions.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA/'reports/chunk_feasibility.json')
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert args.output.is_relative_to((DATA/'reports').resolve()), 'Analysis output must stay in data/reports'
    archive_path = args.output.with_name('chunk_feasibility_initial.json')
    # Retain the exact originally inspected eight examples and original screen
    # counts. Later stronger screening must not silently swap favourable samples.
    if archive_path.exists():
        initial_report = json.loads(archive_path.read_text(encoding='utf-8'))
    elif args.output.exists():
        initial_report = json.loads(args.output.read_text(encoding='utf-8'))
        write_json(archive_path, initial_report)
    else:
        raise RuntimeError('Initial analysis report is required to preserve the inspected fixed examples')
    self_check()
    live_path = DATA/'passages.jsonl'
    baseline_path = DATA/'expansion_runs/official-2018-2026-v1/baseline/passages.jsonl'
    labels_path = DATA/'annotations/passage_labels.jsonl'
    near_path = DATA/('expansion_runs/official-2018-2026-v2/published/'
        'cbf83daf648f23a0d628f47cda0da9f6908e9721db62f365c658304113cf11de/'
        'reports/near_duplicate_candidates.jsonl')
    input_paths = (live_path, baseline_path, labels_path, near_path)
    input_hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path.read_bytes())
                    for path in input_paths}
    assert initial_report['input_hashes']['data/passages.jsonl'] == input_hashes['data/passages.jsonl']
    rows, baseline, labels = (read_jsonl(path) for path in (live_path, baseline_path, labels_path))
    protected_ids = {p['passage_id'] for p in baseline} | {p['passage_id'] for p in labels}
    assert rows[:len(baseline)] == baseline, 'Original 4,500 rows are not intact'
    assert protected_ids <= {p['passage_id'] for p in rows}
    near_pairs = read_jsonl(near_path)
    near_ids = {pair[key] for pair in near_pairs for key in ('left', 'right')}
    assert near_ids <= {p['passage_id'] for p in rows}
    options = []
    sample_proposals = []
    for minimum, maximum, dependency, holdout in (
            (65, 130, False, False), (65, 130, True, False), (80, 130, False, False),
            (80, 130, True, False), (90, 130, True, False),
            (65, 130, True, True), (80, 130, True, True),
            (80, 160, True, True), (75, 160, True, True)):
        result, proposals = estimate(rows, protected_ids, minimum, dependency,
                                     near_ids if holdout else frozenset(), maximum)
        result['analysis_scope'] = 'replace_only_unlabelled_rows_keep_4500_labelled'
        options.append(result)
        if minimum == 65 and dependency and holdout:
            sample_proposals = proposals
    full_result, _ = estimate(rows, set(), 80, True, near_ids, 160)
    full_result['analysis_scope'] = 'whole_corpus_new_version_proposal_only_no_label_migration'
    full_result['requires_label_and_qa_version_realignment'] = True
    options.append(full_result)
    samples = initial_report['samples']
    for sample in samples:
        flags = []
        for child in sample['children']:
            if DEPENDENT_START.search(child['text']):
                flags.append('Child opening is explicitly dependent under the stronger lexical screen: '+child['text'][:70])
        if re.search(r'^\s*(?:[A-Z]\s+){8,}', sample['parent_text']):
            flags.append('Spaced report furniture remains in parent/child; word count includes extraction noise.')
        if re.search(r'^May 2023 years', sample['parent_text']):
            flags.append('Parent begins in an interrupted methodological clause; punctuation is not proof of a full sentence.')
        if sample['metadata_kept']['source'] == 'UK-ENERGY-TRENDS':
            flags.append('Oil-equivalent units may pass keyword check without being a crude-oil fact.')
        if sample['metadata_kept']['source'] == 'EIA-TIE':
            flags.append('Renewable diesel and a company name containing Oil may not satisfy crude-oil scope.')
        sample['stronger_proxy_warnings'] = flags
        sample['fixed_original_examples_not_reselected'] = True
        sample['pending_near_duplicate_holdout'] = sample['parent_id'] in near_ids
        assert words(' '.join(child['text'] for child in sample['children'])) == words(sample['parent_text'])
    report = {
        'created_at': now(), 'provenance': 'deterministic_program_analysis_not_human_annotation',
        'paid_model_calls': 0, 'human_reviews_created': 0,
        'input_hashes': input_hashes,
        'initial_analysis_archive': {'path': str(archive_path.relative_to(ROOT)).replace('\\', '/'),
                                     'sha256': digest(archive_path.read_bytes())},
        'initial_policy_results': initial_report['options'],
        'analyzer_sha256': digest(Path(__file__).read_bytes()),
        'current_chunk_policy': config()['corpus'],
        'before': {'passages': len(rows), 'protected_original_and_labelled': len(protected_ids),
                   'unlabelled_eligible_for_analysis': len(rows)-len(protected_ids),
                   'pending_near_duplicate_candidate_pairs': len(near_pairs),
                   'pending_near_duplicate_involved_ids': len(near_ids),
                   'pending_near_duplicate_unlabelled_ids': len(near_ids-protected_ids),
                   'words': sum(len(words(p['text'])) for p in rows),
                   'documents': len({p['document_id'] for p in rows}),
                   'newline_preserving_paragraph_rows': sum('\n' in p['text'] for p in rows)},
        'options': options,
        'samples_policy': initial_report['samples_policy'],
        'current_dependency_screen_regex': DEPENDENT_START.pattern,
        'samples': samples,
        'invariants': ['Replace selected unlabelled parent rows, never append children beside parents.',
                       'All words retained once; total words and document count unchanged.',
                       'Original 4,500 labelled rows stay protected in unlabelled-only proposals; whole-corpus diagnostic is a separate new-version proposal only.',
                       'All actual corpus rows, labels, indexes and published/frozen files remain untouched by this analyzer.',
                       'Date/title/URL/page/units copied as metadata, never fabricated in child text.',
                       'Current config remains 200-word target; these are proposed alternative-policy statistics only.'],
        'limitations': ['Sentence boundary detection is conservative regex, not semantic parsing.',
                        'Paragraph boundaries were lost by existing clean_page, so no paragraph-recovery claim.',
                        'Keyword and pronoun screens cannot prove standalone answerability or factual completeness.',
                        'Near-duplicate adjudication remains separate; 381 previous candidate pairs are not resolved here.',
                        'New segmentation changes corpus/index/query results and requires a distinct version plus reproducible comparison.',
                        'Published expansion runs enforce append-only baseline invariants; cannot reuse them to silently replace parents.',
                        'Frozen QA may reference changed unlabelled IDs: audit QA references and retain mapping/context before migration.',
                        'No child can automatically inherit a parent annotation; original labelled rows deliberately stay unchanged.',
                        'Meeting a row-count estimate is not confirmation of course acceptance or semantic data quality.'],
    }
    for path in input_paths:
        assert digest(path.read_bytes()) == input_hashes[str(path.relative_to(ROOT)).replace('\\', '/')]
    write_json(args.output, report)
    print({key: report['before'][key] for key in ('passages', 'unlabelled_eligible_for_analysis', 'words')})
    print([{key: option[key] for key in ('policy', 'parents_replaced', 'net_extra_passages',
                                        'potential_passages_after_replacement')} for option in options])
    print('Analysis only; corpus, labels and indexes unchanged. Report:', args.output)


if __name__ == '__main__':
    main()
