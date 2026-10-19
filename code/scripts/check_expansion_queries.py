"""Post-publication retrieval checks, not answer-quality evaluation."""
import json
from pathlib import Path
from time import perf_counter
from oil_evidence.retrieval import Retriever
from oil_evidence.store import DATA,digest,now,read_jsonl,write_json


QUESTIONS = [
    ('What stopped crude oil flows through the Iraq-to-Turkey pipeline in March 2023?', '2024-03-01'),
    ('What routes allow the UAE to export crude oil while bypassing the Strait of Hormuz?', '2026-10-05'),
    ('What are the upstream investment risks for Guyana offshore crude oil production?', '2024-12-31'),
    ('How did sanctions constrain Iranian crude oil exports?', '2020-12-31'),
    ('Compare crude oil production and petroleum and other liquids production in Brazil.', '2025-12-31'),
    ('What were crude oil supply risks?', '2017-12-31'),
    ('How did US Strategic Petroleum Reserve crude oil drawdowns affect reserve inventory and delivery capacity?', '2026-10-05'),
    ('How did Australian crude and condensate exports and oil production change?', '2026-10-05'),
    ('How did UK crude oil production and refinery throughput change?', '2026-10-05'),
    ('How does the EIA Oil and Gas Supply Model represent crude oil drilling and production?', '2026-10-05'),
]


def check():
    r=Retriever()
    expansion=json.loads((DATA/'reports/corpus_expansion.json').read_text(encoding='utf-8'))
    baseline=Path(expansion['snapshot_root'])/'baseline/passages.jsonl'
    assert digest(baseline.read_bytes())==expansion['baseline_sha256']
    original_ids={p['passage_id'] for p in read_jsonl(baseline)}
    rows=[]
    for question,cutoff in QUESTIONS:
        for mode in ['bm25','dense','hybrid']:
            start=perf_counter()
            evidence=r.search(question,mode=mode,as_of=cutoff)
            assert all(p['available_by']<=cutoff for p in evidence)
            assert len({p['passage_id'] for p in evidence})==len(evidence)
            assert bool(evidence)==(cutoff>='2018-01-01')
            rows.append(dict(question=question,as_of=cutoff,mode=mode,
                seconds=perf_counter()-start,evidence=[{k:p[k] for k in
                    ['passage_id','document_id','source','title','url','page','available_by','rank','score']}
                    |dict(new_in_expansion=p['passage_id'] not in original_ids) for p in evidence]))
    new_hits=sum(sum(p['new_in_expansion'] for p in row['evidence']) for row in rows)
    assert new_hits>0, 'Expanded sources never appear in the smoke checks'
    report=dict(at=now(),corpus_sha256=digest((DATA/'passages.jsonl').read_bytes()),
        smoke_only=True,checks=len(rows),temporal_checks_passed=True,new_source_hits=new_hits,
        external_model_calls=0,human_labels_created=0,answer_accuracy=None,retrieval_recall=None,
        limitations='No answer generation or qrels scoring. Hits are not proof of answer correctness. Existing 100 QA labels cover a fixed old candidate pool, not the expanded corpus.',
        queries=rows)
    write_json(DATA/'reports/expansion_queries.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='queries'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    check()
