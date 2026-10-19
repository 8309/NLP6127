"""Register independently observed ordinary public downloads, preserve old entries."""
from expand_corpus import RecordingFetcher,select_run,pipeline_lock,snapshot,verify_baseline_sources
from discover_wb_documents import discover
from oil_evidence.store import DATA,read_jsonl,write_jsonl,append_jsonl,now

root=select_run('official-2018-2026-v2')
with pipeline_lock():
    snapshot()
    verify_baseline_sources()
    rows={r['url']:r for r in read_jsonl(DATA/'sources.jsonl')}
    f=RecordingFetcher()
    try:
        found=discover(f)
    finally:
        f.close()
    for row in found:
        rows.setdefault(row['url'],row)
    write_jsonl(DATA/'sources.jsonl',rows.values())
    append_jsonl(root/'source_discovery_runs.jsonl',dict(at=now(),independent_wb_pdf_candidates=len(found),sources=len(rows)))
    print(len(found),'public PDF candidates registered')
