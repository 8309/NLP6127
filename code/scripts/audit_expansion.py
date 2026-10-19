"""Reproducible source-integrity checks and candidate-only near-duplicate audit."""
from collections import Counter, defaultdict
from datetime import date
import json
import argparse

import expand_corpus as expansion
from oil_evidence.corpus import clean_page,extraction_suffix
from oil_evidence.store import DATA, config, digest, now, read_jsonl, words, write_json, write_jsonl


def shingles(text, width=5):
    tokens = words(text)
    return {digest(' '.join(tokens[i:i+width]))[:16] for i in range(len(tokens)-width+1)}


def audit(run_id='official-2018-2026-v1'):
    ROOT=expansion.select_run(run_id)
    original = expansion.snapshot()
    expansion.verify_baseline_sources()
    staged = ROOT/'stage'
    rows = read_jsonl(staged/'passages.jsonl')
    baseline = read_jsonl(ROOT/'baseline/passages.jsonl')
    assert rows[:len(baseline)]==baseline
    assert len({p['passage_id'] for p in rows})==len(rows)
    assert len({digest(' '.join(words(p['text']))) for p in rows})==len(rows)
    documents = {d['document_id']:d for d in read_jsonl(DATA/'documents.jsonl')}
    for d in documents.values():
        assert digest((DATA/d['raw_path']).read_bytes())==d['raw_sha256'], d['document_id']
    bounds = config()['corpus']
    cached = {}
    for p in rows[len(baseline):]:
        date.fromisoformat(p['available_by'])
        assert bounds['start_date']<=p['available_by']<=bounds['end_date']
        assert p['word_count']==len(words(p['text']))>=bounds['min_words']
        d = documents[p['document_id']]
        assert not d.get('ingest_exclusion_reason'),p['passage_id']
        assert str(p['page']) not in d.get('excluded_physical_pages',{}),p['passage_id']
        assert p['passage_id'] not in d.get('excluded_passage_ids',{}),p['passage_id']
        for field in ['source','title','url','published_date','available_by','date_precision','date_source','event_date','event_status','redistribution']:
            assert p.get(field)==d.get(field),(p['passage_id'],field)
        suffix = extraction_suffix(d['source'])
        key = (d['raw_sha256'],suffix)
        if key not in cached:
            cached[key] = json.loads((DATA/'extracted'/(d['raw_sha256']+suffix)).read_text(encoding='utf-8'))
        # Provenance guard: each new passage must be from the specified physical page/body.
        normalized = ' '.join(words(clean_page(cached[key][p['page']-1])))
        assert ' '.join(words(p['text'])) in normalized, p['passage_id']
    for name in ['annotations/codex.jsonl','annotations/passage_labels.jsonl','annotations/passage_review.jsonl']:
        assert digest((DATA/name).read_bytes())==original['hashes'][name], name

    # This is a candidate audit, not exhaustive semantic deduplication. Do not discard
    # legitimate forecasts across releases merely because their wording is similar.
    sets = [shingles(p['text']) for p in rows]
    inverted = defaultdict(list)
    for i, ss in enumerate(sets):
        for s in ss:
            inverted[s].append(i)
    pairs = []
    evaluated = 0
    for i in range(len(baseline),len(rows)):
        counts = Counter()
        for s in sets[i]:
            if len(inverted[s])<=12:
                counts.update(j for j in inverted[s] if j<i)
        for j, count in counts.items():
            if count<20:
                continue
            evaluated += 1
            overlap = len(sets[i]&sets[j])
            jaccard = overlap/len(sets[i]|sets[j])
            containment = overlap/min(len(sets[i]),len(sets[j]))
            if jaccard>=.85 or containment>=.92:
                pairs.append(dict(left=rows[j]['passage_id'],right=rows[i]['passage_id'],
                    jaccard=round(jaccard,4),containment=round(containment,4),
                    left_date=rows[j]['available_by'],right_date=rows[i]['available_by'],
                    same_raw_file=documents[rows[j]['document_id']]['raw_sha256']==documents[rows[i]['document_id']]['raw_sha256'],
                    decision='candidate_only_not_removed'))
    write_jsonl(staged/'reports/near_duplicate_candidates.jsonl',pairs)
    buckets=defaultdict(list)
    for p in rows[len(baseline):]:
        buckets[(p['source'],p['available_by'][:4])].append(p)
    samples=[]
    for key, group in sorted(buckets.items()):
        # Stable selection, never select because a result looks favorable.
        for p in sorted(group,key=lambda r:digest(r['passage_id']))[:2]:
            samples.append(dict(p,raw_path=documents[p['document_id']]['raw_path'],
                review_status='selected_for_model_source_check_not_human_reviewed'))
    write_jsonl(staged/'reports/source_qa_samples.jsonl',samples)
    report=dict(at=now(),corpus_sha256=digest((staged/'passages.jsonl').read_bytes()),
        passages=len(rows),new_passages=len(rows)-len(baseline),raw_hash_checks=len(documents),
        baseline_preserved=True,annotation_hashes_preserved=True,unique_ids=True,
        exact_normalized_duplicates=0,new_passages_page_provenance_passed=True,
        new_passage_date_window_passed=True,source_year_samples=len(samples),
        near_duplicate_method='5-word shingles; candidate shared shingles in <=12 passages and >=20 overlap; Jaccard >=0.85 or containment >=0.92; candidates involving new passages only',
        near_duplicate_pairs_evaluated=evaluated,near_duplicate_candidates=len(pairs),
        near_duplicate_audit_exhaustive=False,near_duplicate_removals=0,
        human_reviewed_samples=0,external_model_calls=0)
    report.update(documents_sha256=digest((DATA/'documents.jsonl').read_bytes()),
        sources_sha256=digest((DATA/'sources.jsonl').read_bytes()),implementation_hashes=expansion.implementation_hashes())
    version_path=ROOT/'new_source_version_audit.json'
    if version_path.exists():
        report['new_source_version_audit_sha256']=digest(version_path.read_bytes())
    write_json(staged/'reports/expansion_quality.json',report)
    corpus_report=json.loads((staged/'reports/corpus.json').read_text(encoding='utf-8'))
    corpus_report.update(near_duplicate_audit='candidate_audit_complete_not_exhaustive; resolution pending',
        quality_report='reports/expansion_quality.json',
        excluded_pdf_pages_scope=f'newly processed documents only; {len(baseline)}-passage run baseline unchanged')
    write_json(staged/'reports/corpus.json',corpus_report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='official-2018-2026-v1')
    args=parser.parse_args()
    expansion.select_run(args.run)
    with expansion.pipeline_lock():
        print(json.dumps(audit(args.run),ensure_ascii=False,indent=2))
