"""Replay a frozen uniform release from verified local raw files and caches.

Only this run's reproduction directory is written. No downloading, embedding,
generation, publication, or label migration occurs. Processing code must match
the release; archived implementations remain available when current code drifts.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

import unify_corpus as pipeline
from oil_evidence.corpus import extraction_suffix
from oil_evidence.store import DATA, digest, now, read_jsonl, write_json
from oil_evidence.unified_corpus import build_unified_corpus

FILES = ['passages.jsonl', 'source_pages.jsonl', 'quarantine.jsonl', 'source_exclusions.jsonl']


def copy_verified(source, target, expected):
    if digest(source.read_bytes()) != expected:
        raise ValueError('Replay input hash mismatch: ' + str(source))
    if target.exists():
        if digest(target.read_bytes()) != expected:
            raise ValueError('Existing replay input differs')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # A same-volume hard link avoids copying large PDFs. The replay only reads
    # inputs; cleanup removes the link, not the original source. Safe fallback.
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)
    if digest(target.read_bytes()) != expected:
        raise ValueError('Replay input copy/link mismatch')


def replay(run='semantic-page-v1'):
    pipeline.select_run(run)
    release = pipeline.load(DATA / 'reports/corpus_unified.json')
    release_sha = release.get('corpus_sha256')
    if not isinstance(release_sha, str) or not re.fullmatch(r'[a-f0-9]{64}', release_sha):
        raise ValueError('Release corpus hash must be a strict SHA-256 digest')
    if digest((DATA / 'passages.jsonl').read_bytes()) != release_sha:
        raise ValueError('Active corpus no longer matches the uniform release')
    frozen = pipeline.contained(release['published_snapshot'])
    if frozen.parent != (pipeline.RUN / 'published').resolve() or frozen.name != release_sha:
        raise ValueError('Requested run is not the active uniform release')
    if pipeline.hashes() != release['implementation_hashes']:
        raise ValueError('Current processing code/configuration differs from frozen release')
    report = pipeline.load(frozen / 'reports/corpus.json')
    if report['corpus_sha256'] != release_sha or report['implementation_hashes'] != release['implementation_hashes']:
        raise ValueError('Frozen report and active release disagree')
    bound_inputs = {frozen / 'reports/corpus.json': digest((frozen / 'reports/corpus.json').read_bytes())}
    implementation = (frozen / 'implementation').resolve()
    for name, expected in report['implementation_hashes'].items():
        path = (implementation / name).resolve()
        if not path.is_relative_to(implementation) or digest(path.read_bytes()) != expected:
            raise ValueError('Frozen processing implementation changed')
        bound_inputs[path] = expected
    for key, name in [('corpus_sha256', 'passages.jsonl'), ('source_pages_sha256', 'source_pages.jsonl'),
                      ('quarantine_sha256', 'quarantine.jsonl'), ('source_exclusions_sha256', 'source_exclusions.jsonl'),
                      ('documents_sha256', 'documents.jsonl'), ('sources_sha256', 'sources.jsonl')]:
        if digest((frozen / name).read_bytes()) != report[key]:
            raise ValueError('Frozen release file changed: ' + name)
        bound_inputs[frozen / name] = report[key]
    before = {name: digest((DATA / name).read_bytes()) for name in pipeline.INPUTS if (DATA / name).exists()}
    manifest = pipeline.load(pipeline.RUN / 'baseline/manifest.json')
    cache_hashes = dict(manifest['extraction_hashes'])
    cache_hashes.update({name: sha for name, sha in report['supplemental_input_hashes'].items() if name.startswith('extracted/')})
    target = pipeline.RUN / 'reproductions' / release['corpus_sha256']
    target.mkdir(parents=True, exist_ok=True)
    # Keep scratch short on Windows: nesting under a 64-character release SHA
    # can push hash-named extraction caches over the host's MAX_PATH limit.
    with tempfile.TemporaryDirectory(prefix='.u-replay-', dir=DATA) as temporary:
        scratch = Path(temporary)
        if scratch.resolve().parent != DATA.resolve():
            raise ValueError('Replay scratch escapes its owned data directory')
        inputs, output = scratch / 'inputs', scratch / 'output'
        inputs.mkdir()
        shutil.copy2(frozen / 'documents.jsonl', inputs / 'documents.jsonl')
        # Only used by the builder for the archived count; no label inheritance.
        annotation = pipeline.RUN / 'baseline/annotations/passage_labels.jsonl'
        copy_verified(annotation, inputs / 'annotations/passage_labels.jsonl',
                      manifest['hashes']['annotations/passage_labels.jsonl'])
        bound_inputs[annotation] = manifest['hashes']['annotations/passage_labels.jsonl']
        for doc in read_jsonl(frozen / 'documents.jsonl'):
            source = pipeline.contained(doc['raw_path'])
            if not source.is_relative_to((DATA / 'raw').resolve()):
                raise ValueError('Replay raw path is outside raw directory')
            dest = (inputs / doc['raw_path']).resolve()
            if not dest.is_relative_to((inputs / 'raw').resolve()):
                raise ValueError('Replay input destination escapes owned raw directory')
            copy_verified(source, dest, doc['raw_sha256'])
            bound_inputs[source] = doc['raw_sha256']
            name = 'extracted/' + doc['raw_sha256'] + extraction_suffix(doc['source'])
            if name in cache_hashes:
                copy_verified(pipeline.contained(name), inputs / name, cache_hashes[name])
                bound_inputs[pipeline.contained(name)] = cache_hashes[name]
        reconstructed = build_unified_corpus(output, data_dir=inputs)
        for name in FILES:
            if digest((output / name).read_bytes()) != digest((frozen / name).read_bytes()):
                raise ValueError('Reconstructed bytes differ: ' + name)
        for key, value in reconstructed.items():
            if report.get(key) != value:
                raise ValueError('Reconstructed statistics differ: ' + key)
        if (pipeline.hashes() != release['implementation_hashes'] or
                any(digest(path.read_bytes()) != sha for path, sha in bound_inputs.items())):
            raise ValueError('Frozen source or implementation changed during reconstruction')
        for name in FILES:
            dest = target / name
            if dest.exists() and digest(dest.read_bytes()) != digest((output / name).read_bytes()):
                raise ValueError('Existing immutable reproduction differs: ' + name)
            if not dest.exists():
                shutil.copy2(output / name, dest)
    if any(digest((DATA / name).read_bytes()) != sha for name, sha in before.items()):
        raise ValueError('Live inputs changed during source replay')
    result = dict(at=now(), run=run, corpus_sha256=release['corpus_sha256'],
                  byte_identical_source_reconstruction=True, reconstructed_files=FILES,
                  passages=report['passages'], words=report['words'], output=target.relative_to(DATA).as_posix(),
                  live_inputs_unchanged=True, external_model_calls=0, human_labels_created=0,
                  limitations='Replays local frozen raw/cache processing, not current remote pages, human semantic approval or fresh vector inference.')
    write_json(target / 'replay.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='semantic-page-v1')
    args = parser.parse_args()
    print(json.dumps(replay(args.run), ensure_ascii=False, indent=2))
