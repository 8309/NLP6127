"""Serial bounded source-family collections, checkpointed and resumable."""
import argparse
import json
from expand_corpus import select_run,snapshot,verify_baseline_sources,pipeline_lock,collect
from oil_evidence.store import append_jsonl,now

parser=argparse.ArgumentParser()
parser.add_argument('--run',required=True)
parser.add_argument('families',nargs='+')
args=parser.parse_args()
root=select_run(args.run)
for family in args.families:
    with pipeline_lock():
        snapshot()
        verify_baseline_sources()
        result=collect(limit=200,source_filter=family)
        append_jsonl(root/'collection_runs.jsonl',dict(at=now(),family=family,**result))
    print(json.dumps(dict(family=family,**result)),flush=True)
