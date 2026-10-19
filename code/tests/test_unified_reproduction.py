"""Source replay and public entry guards; no real publication or model calls."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from oil_evidence import corpus, unified_corpus
from oil_evidence.store import digest, read_jsonl, write_json, write_jsonl
from test_unified_pipeline import uniform, stage_fixture, fake_index, release_reports, live_bytes


def load_replay_script():
    scripts = Path(__file__).parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('isolated_unified_source_replay', scripts / 'reproduce_unified.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


@pytest.fixture
def frozen_replay(uniform):
    pipeline, data, _, _, monkeypatch = uniform
    stage_fixture(uniform)
    fake_index(uniform)
    release_reports(uniform)
    release = pipeline.publish()
    replay = load_replay_script()
    monkeypatch.setattr(replay, 'pipeline', pipeline)
    monkeypatch.setattr(replay, 'DATA', data)
    frozen = data / release['published_snapshot']
    return replay, pipeline, data, frozen, release, monkeypatch


def run_replay(fixture):
    replay, pipeline, _, _, _, _ = fixture
    return replay.replay(pipeline.RUN.name)


def test_source_replay_byte_identical_without_live_writes(frozen_replay):
    replay, _, data, frozen, release, _ = frozen_replay
    before = live_bytes(data)
    result = run_replay(frozen_replay)
    assert result['byte_identical_source_reconstruction'] and result['live_inputs_unchanged']
    assert result['external_model_calls'] == result['human_labels_created'] == 0
    assert result['corpus_sha256'] == release['corpus_sha256']
    assert live_bytes(data) == before
    for name in replay.FILES:
        assert (data / result['output'] / name).read_bytes() == (frozen / name).read_bytes()


def test_source_replay_same_existing_target_is_idempotent(frozen_replay):
    replay, _, data, _, _, _ = frozen_replay
    first = run_replay(frozen_replay)
    before = {name: (data / first['output'] / name).read_bytes() for name in replay.FILES}
    second = run_replay(frozen_replay)
    assert first['output'] == second['output']
    assert before == {name: (data / second['output'] / name).read_bytes() for name in replay.FILES}


def test_source_replay_existing_immutable_reconstruction_conflict_rejected(frozen_replay):
    _, _, data, _, _, _ = frozen_replay
    result = run_replay(frozen_replay)
    target = data / result['output'] / 'passages.jsonl'
    target.write_bytes(target.read_bytes() + b'\n')
    changed = target.read_bytes()
    with pytest.raises(ValueError, match='immutable reproduction differs'):
        run_replay(frozen_replay)
    assert target.read_bytes() == changed


@pytest.mark.parametrize('name', ['passages.jsonl', 'source_pages.jsonl', 'quarantine.jsonl', 'source_exclusions.jsonl',
                                 'documents.jsonl', 'sources.jsonl'])
def test_source_replay_frozen_file_sha_changes_fail_closed(frozen_replay, name):
    _, _, _, frozen, _, _ = frozen_replay
    path = frozen / name
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Frozen release file changed'):
        run_replay(frozen_replay)


def test_source_replay_changed_current_processing_code_rejected(frozen_replay):
    _, pipeline, _, _, _, _ = frozen_replay
    path = pipeline.PROJECT / 'guard_implementation.py'
    path.write_bytes(path.read_bytes() + b'# new processing code\n')
    with pytest.raises(ValueError, match='processing code/configuration differs'):
        run_replay(frozen_replay)


def test_source_replay_changed_archived_processing_code_rejected(frozen_replay):
    _, _, _, frozen, _, _ = frozen_replay
    path = frozen / 'implementation/guard_implementation.py'
    path.write_bytes(path.read_bytes() + b'# altered frozen code\n')
    with pytest.raises(ValueError):
        run_replay(frozen_replay)


@pytest.mark.parametrize('sha', ['0' * 64, '../escaped', '..\\escaped', '', None])
def test_source_replay_release_sha_must_match_frozen_corpus_and_owned_target(frozen_replay, sha):
    _, _, data, _, release, _ = frozen_replay
    changed = dict(release, corpus_sha256=sha)
    write_json(data / 'reports/corpus_unified.json', changed)
    with pytest.raises(ValueError):
        run_replay(frozen_replay)


def test_source_replay_stale_active_release_rejected_before_reconstruction(frozen_replay):
    _, _, data, _, _, _ = frozen_replay
    path = data / 'passages.jsonl'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        run_replay(frozen_replay)


@pytest.mark.parametrize('group', ['raw_hashes', 'extraction_hashes'])
def test_source_replay_changed_raw_or_cache_rejected(frozen_replay, group):
    _, pipeline, data, _, _, _ = frozen_replay
    manifest = pipeline.load(pipeline.RUN / 'baseline/manifest.json')
    path = data / next(iter(manifest[group]))
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='input hash mismatch'):
        run_replay(frozen_replay)


def test_source_replay_changed_archived_label_count_input_rejected(frozen_replay):
    _, pipeline, _, _, _, _ = frozen_replay
    path = pipeline.RUN / 'baseline/annotations/passage_labels.jsonl'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='input hash mismatch'):
        run_replay(frozen_replay)


@pytest.mark.parametrize('raw_path', ['../outside.html', 'raw/../../outside.html', '../data/passages.jsonl', 'annotations/passage_labels.jsonl'])
def test_source_replay_malicious_raw_path_cannot_escape_owned_raw_input(frozen_replay, raw_path):
    _, pipeline, _, frozen, _, _ = frozen_replay
    documents = read_jsonl(frozen / 'documents.jsonl')
    documents[0]['raw_path'] = raw_path
    write_jsonl(frozen / 'documents.jsonl', documents)
    report = pipeline.load(frozen / 'reports/corpus.json')
    report['documents_sha256'] = digest((frozen / 'documents.jsonl').read_bytes())
    write_json(frozen / 'reports/corpus.json', report)
    with pytest.raises(ValueError, match='outside raw|escapes data|destination escapes'):
        run_replay(frozen_replay)


def test_source_replay_builder_byte_mismatch_cannot_publish_reproduction(frozen_replay):
    replay, pipeline, _, _, release, monkeypatch = frozen_replay
    original = replay.build_unified_corpus
    def different_bytes(output, **kwargs):
        report = original(output, **kwargs)
        (output / 'passages.jsonl').write_bytes((output / 'passages.jsonl').read_bytes() + b'\n')
        return report
    monkeypatch.setattr(replay, 'build_unified_corpus', different_bytes)
    with pytest.raises(ValueError, match='Reconstructed bytes differ'):
        run_replay(frozen_replay)
    assert not (pipeline.RUN / 'reproductions' / release['corpus_sha256'] / 'passages.jsonl').exists()


def test_source_replay_builder_statistic_mismatch_rejected(frozen_replay):
    replay, _, _, _, _, monkeypatch = frozen_replay
    original = replay.build_unified_corpus
    def different_statistics(output, **kwargs):
        report = original(output, **kwargs)
        return dict(report, words=report['words'] + 1)
    monkeypatch.setattr(replay, 'build_unified_corpus', different_statistics)
    with pytest.raises(ValueError, match='Reconstructed statistics differ'):
        run_replay(frozen_replay)


def test_source_replay_detects_live_mutation_during_readonly_replay(frozen_replay):
    replay, _, data, _, _, monkeypatch = frozen_replay
    original = replay.build_unified_corpus
    def mutate_live(output, **kwargs):
        report = original(output, **kwargs)
        path = data / 'sources.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
        return report
    monkeypatch.setattr(replay, 'build_unified_corpus', mutate_live)
    with pytest.raises(ValueError, match='Live inputs changed'):
        run_replay(frozen_replay)


@pytest.mark.parametrize('group', ['current_code', 'frozen_code', 'raw', 'cache', 'frozen_source_pages',
                                 'frozen_documents', 'archived_labels'])
def test_source_replay_inputs_and_code_cannot_change_during_reconstruction(frozen_replay, group):
    replay, pipeline, data, frozen, release, monkeypatch = frozen_replay
    manifest = pipeline.load(pipeline.RUN / 'baseline/manifest.json')
    paths = {
        'current_code': pipeline.PROJECT / 'guard_implementation.py',
        'frozen_code': frozen / 'implementation/guard_implementation.py',
        'raw': data / next(iter(manifest['raw_hashes'])),
        'cache': data / next(iter(manifest['extraction_hashes'])),
        'frozen_source_pages': frozen / 'source_pages.jsonl',
        'frozen_documents': frozen / 'documents.jsonl',
        'archived_labels': pipeline.RUN / 'baseline/annotations/passage_labels.jsonl',
    }
    original = replay.build_unified_corpus
    def changed_source(output, **kwargs):
        report = original(output, **kwargs)
        path = paths[group]
        path.write_bytes(path.read_bytes() + b'\n')
        return report
    monkeypatch.setattr(replay, 'build_unified_corpus', changed_source)
    with pytest.raises(ValueError):
        run_replay(frozen_replay)
    assert not (pipeline.RUN / 'reproductions' / release['corpus_sha256'] / 'passages.jsonl').exists()


def test_source_replay_duplicate_cached_content_shared_target_is_safe(frozen_replay):
    replay, _, data, _, _, _ = frozen_replay
    source = data / 'raw/copy-fixture.html'
    source.write_bytes(b'official crude oil source fixture')
    target = data / 'unified_runs/copy-fixture/reproductions/owned/raw/shared.html'
    expected = digest(source.read_bytes())
    replay.copy_verified(source, target, expected)
    replay.copy_verified(source, target, expected)
    assert target.read_bytes() == source.read_bytes()
    target.unlink()
    assert source.read_bytes() == b'official crude oil source fixture'


def test_source_replay_existing_input_copy_conflict_is_not_overwritten(frozen_replay):
    replay, _, data, _, _, _ = frozen_replay
    source = data / 'raw/copy-fixture.html'
    source.write_bytes(b'official crude oil source fixture')
    target = data / 'unified_runs/copy-fixture/reproductions/owned/raw/shared.html'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'unrelated earlier input')
    with pytest.raises(ValueError, match='Existing replay input differs'):
        replay.copy_verified(source, target, digest(source.read_bytes()))
    assert target.read_bytes() == b'unrelated earlier input'


def test_source_replay_safe_copy_fallback_when_hardlink_unavailable(frozen_replay):
    replay, _, data, _, _, monkeypatch = frozen_replay
    source = data / 'raw/copy-fixture.html'
    source.write_bytes(b'official crude oil source fixture')
    target = data / 'unified_runs/copy-fixture/reproductions/owned/raw/shared.html'
    def no_hardlinks(*args, **kwargs):
        raise OSError('Host disallows hard links')
    monkeypatch.setattr(Path, 'hardlink_to', no_hardlinks)
    replay.copy_verified(source, target, digest(source.read_bytes()))
    assert target.read_bytes() == source.read_bytes()


@pytest.mark.parametrize('entry', ['baseline', 'baseline_document_ids', 'direct_live'])
def test_public_uniform_build_rejects_legacy_and_direct_live_entries(tmp_path, monkeypatch, entry):
    monkeypatch.setattr(corpus, 'config', lambda: {'corpus': {'segmentation_version': 'semantic-page-v1'}})
    called = []
    monkeypatch.setattr(unified_corpus, 'build_unified_corpus', lambda *args, **kwargs: called.append((args, kwargs)))
    kwargs = {'output_dir': tmp_path / 'stage'}
    if entry == 'baseline':
        kwargs['baseline'] = [{'passage_id': 'old'}]
    elif entry == 'baseline_document_ids':
        kwargs['baseline_document_ids'] = {'old-document'}
    else:
        kwargs = {}
    with pytest.raises(ValueError, match='retain legacy chunks|direct live overwrite is disabled'):
        corpus.build_corpus(**kwargs)
    assert not called and not (tmp_path / 'stage').exists()


def test_public_uniform_build_routes_only_explicit_stage_to_uniform_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, 'config', lambda: {'corpus': {'segmentation_version': 'semantic-page-v1'}})
    target = tmp_path / 'stage'
    sentinel = {'uniform': True}
    called = []
    def capture(output):
        called.append(output)
        return sentinel
    monkeypatch.setattr(unified_corpus, 'build_unified_corpus', capture)
    assert corpus.build_corpus(output_dir=target) is sentinel
    assert called == [target]
