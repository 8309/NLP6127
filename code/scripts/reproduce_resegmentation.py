"""Reconstruct released passage bytes from frozen parents, without live writes.

No collection, embeddings, model calls or label migration. This separate replay
entry point is usable after publication, when writer lost-update guards rightly
reject reusing the old run to replace the now-changed live corpus.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import resegment_corpus as pipeline
from oil_evidence.store import DATA, digest, now, read_jsonl, words, write_json, write_jsonl


def replay(run):
    pipeline.select_run(run)
    release = pipeline.load(DATA/'reports/corpus_resegmentation.json')
    if release['source_passages_path'] != (pipeline.ROOT/'baseline/passages.jsonl').relative_to(DATA).as_posix():
        raise ValueError('Requested run is not the active release parent version')
    if pipeline.implementation_hashes() != release['implementation_hashes']:
        raise ValueError('Processing code/configuration no longer matches released version')
    manifest = pipeline.snapshot()  # Only verifies an existing immutable baseline.
    before = {name:digest((DATA/name).read_bytes()) for name in pipeline.INPUTS if (DATA/name).exists()}
    for name, expected in manifest['raw_hashes'].items():
        if digest(pipeline.within_data(name).read_bytes()) != expected:
            raise ValueError('Raw source changed: '+name)
    parents = read_jsonl(pipeline.ROOT/'baseline/passages.jsonl')
    pending = pipeline.near_ids()
    originals = {digest(' '.join(words(p['text']))) for p in parents}
    active, quarantine, seen = [], [], set()
    for parent in parents:
        reason = pipeline.quarantine_reason(parent, pending)
        if reason:
            quarantine.append({'parent':parent, 'reason':reason})
            continue
        children, _ = pipeline.split_parent(parent)
        if children:
            checks = [digest(' '.join(words(p['text']))) for p in children]
            if len(set(checks)) != len(checks) or any(fp in originals or fp in seen for fp in checks):
                children = None
        for child in children or [parent]:
            fp = digest(' '.join(words(child['text'])))
            if fp in seen:
                raise ValueError('Reconstruction has duplicate active text')
            seen.add(fp)
            active.append(child)
    target = pipeline.ROOT/'reproductions'/release['corpus_sha256']
    target.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory owns only its exact task scratch directory.
    with tempfile.TemporaryDirectory(prefix='.replay-', dir=target) as temporary:
        scratch = Path(temporary)
        write_jsonl(scratch/'passages.jsonl', active)
        write_jsonl(scratch/'quarantine.jsonl', quarantine)
        if digest((scratch/'passages.jsonl').read_bytes()) != release['corpus_sha256']:
            raise ValueError('Reconstructed bytes differ from released corpus')
        expected_quarantine = DATA/release['published_snapshot']/'quarantine.jsonl'
        if digest((scratch/'quarantine.jsonl').read_bytes()) != digest(expected_quarantine.read_bytes()):
            raise ValueError('Reconstructed quarantine differs from release')
        for name in ['passages.jsonl','quarantine.jsonl']:
            pipeline.freeze_file(scratch/name, target/name)
    for name, expected in before.items():
        if digest((DATA/name).read_bytes()) != expected:
            raise ValueError('Live data changed during read-only reconstruction: '+name)
    report = dict(at=now(), run=run, corpus_sha256=release['corpus_sha256'],
        byte_identical_reconstruction=True, passages=len(active), words=sum(len(words(p['text'])) for p in active),
        quarantined_passages=len(quarantine), by_source=dict(Counter(p['source'] for p in active)),
        output=target.relative_to(DATA).as_posix(), live_inputs_unchanged=True,
        script_sha256=digest(Path(__file__).read_bytes()), external_model_calls=0, human_labels_created=0,
        limitations='Replays frozen text and quarantine, not webpage downloading or exhaustive semantic validation. Index reproduction is independently checked by the released corpus/vector/order hashes.')
    write_json(DATA/'reports/resegmentation_reproduction.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='semantic-80-160-v1')
    replay(parser.parse_args().run)
