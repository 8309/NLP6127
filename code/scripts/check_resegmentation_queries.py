"""Unpaid version/lineage/retrieval smoke checks, not qrels or answer scoring."""
import json
from time import perf_counter

from check_expansion_queries import QUESTIONS
from oil_evidence.corpus_versions import original_parent
from oil_evidence.retrieval import Retriever
from oil_evidence.store import DATA, digest, now, read_jsonl, write_json


def check():
    release = json.loads((DATA/'reports/corpus_resegmentation.json').read_text(encoding='utf-8'))
    corpus_hash = digest((DATA/'passages.jsonl').read_bytes())
    assert corpus_hash == release['corpus_sha256']
    parents_path = DATA/release['source_passages_path']
    assert digest(parents_path.read_bytes()) == release['source_corpus_sha256']
    parents = {p['passage_id']:p for p in read_jsonl(parents_path)}
    retriever = Retriever()
    assert retriever.vectors is not None
    checks = []
    for question, cutoff in QUESTIONS:
        for mode in ['bm25','dense','hybrid']:
            start = perf_counter()
            hits = retriever.search(question, mode=mode, as_of=cutoff)
            assert len({p['passage_id'] for p in hits}) == len(hits)
            assert all(p['available_by'] <= cutoff for p in hits)
            assert bool(hits) == (cutoff >= '2018-01-01')
            for hit in hits:
                parent = parents[hit.get('parent_passage_id', hit['passage_id'])]
                for key in ['document_id','source','title','url','page','available_by','published_date']:
                    assert hit[key] == parent[key]
                if hit.get('parent_passage_id'):
                    assert original_parent(hit) == parent
                else:
                    assert hit['text'] == parent['text']
            checks.append(dict(question=question, as_of=cutoff, mode=mode,
                seconds=perf_counter()-start, evidence=[{k:p[k] for k in
                    ['passage_id','document_id','source','title','url','page','available_by','rank','score']}
                    |dict(resegmented=bool(p.get('parent_passage_id')),
                          parent_passage_id=p.get('parent_passage_id')) for p in hits]))
    child_hits = sum(p['resegmented'] for row in checks for p in row['evidence'])
    assert child_hits > 0
    report = dict(at=now(), corpus_sha256=corpus_hash, smoke_only=True, checks=len(checks),
        temporal_checks_passed=True, parent_lineage_checks_passed=True, child_hits=child_hits,
        external_model_calls=0, human_labels_created=0, answer_accuracy=None, retrieval_recall=None,
        limitations='No generated answers or qrels scores. Parent context is display-only; resegmented hits are not newly collected source documents. Known old QA is not ground truth for the new corpus.',
        queries=checks)
    write_json(DATA/'reports/resegmentation_queries.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='queries'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    check()
