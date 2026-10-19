"""Audited Codex diagnostic page exclusions; never change baseline or labels."""
from expand_corpus import select_run,snapshot,verify_baseline_sources,pipeline_lock
from oil_evidence.store import DATA,read_jsonl,write_jsonl,write_json,digest,now

DECISIONS={
    'fbfd5c4fcc1167c6653c':{'fbfd5c4fcc1167c6653c-58ba86ab534480e1':'Flat input-survey catalogue; retain the other supply mechanism text on the same physical page'},
    'ebf20644c98e5d77511b':{'ebf20644c98e5d77511b-d30cb0887f839628':'Variable definitions and program file-reading procedure; exclude technical implementation context'},
}

root=select_run('official-2018-2026-v2')
with pipeline_lock():
    snapshot()
    verify_baseline_sources()
    old={d['document_id'] for d in read_jsonl(root/'baseline/documents.jsonl')}
    docs=read_jsonl(DATA/'documents.jsonl')
    sources={r['url']:r for r in read_jsonl(DATA/'sources.jsonl')}
    records=[]
    for d in docs:
        if d['document_id'] not in DECISIONS:
            continue
        assert d['document_id'] not in old
        assert digest((DATA/d['raw_path']).read_bytes())==d['raw_sha256']
        d.pop('excluded_physical_pages',None)
        sources[d['url']].pop('excluded_physical_pages',None)
        d['excluded_passage_ids']=DECISIONS[d['document_id']]
        sources[d['url']]['excluded_passage_ids']=d['excluded_passage_ids']
        records.append(dict(document_id=d['document_id'],raw_sha256=d['raw_sha256'],
            excluded_passage_ids=d['excluded_passage_ids'],provenance='codex_model_source_diagnostic',human_reviewed=False))
    assert len(records)==len(DECISIONS)
    write_jsonl(DATA/'documents.jsonl',docs)
    write_jsonl(DATA/'sources.jsonl',sources.values())
    write_json(root/'source_page_exclusions.json',dict(at=now(),records=records,human_reviewed=False))
    print('Recorded',len(records),'document page exclusions')
