"""Mocked resume safety tests; no native CLI or authentication calls."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def launcher(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("isolated_codex_resume", scripts / "resume_codex_annotation_stream.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_existing_output_selected_and_completed_records_not_rerun(launcher, tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    launcher.native.write_new(output / "manifest.json", {})
    binary = tmp_path / "codex.exe"
    launcher.native.write_new(binary, b"fixture-only")
    monkeypatch.setattr(launcher, "validated_cli", lambda cli: binary)
    original = launcher.native.STREAMS, launcher.native.load_pool
    protected = {}
    manifest = dict(specification=dict(requested_model=None))
    monkeypatch.setattr(launcher, "prepare", lambda *args: (tmp_path, tmp_path, output, manifest, protected,
                        dict(wrapper_sha256="unchanged-wrapper"), {"q-1": {}, "q-2": {}}))
    seen = []
    def mocked_run(root, pool, **kwargs):
        seen.append((launcher.native.STREAMS, kwargs))
        return output, dict(failures=[])
    monkeypatch.setattr(launcher.native, "run", mocked_run)
    launcher.resume(tmp_path, tmp_path, output, "b", 50)
    assert seen == [(("b",), dict(max_records=50, requested_model=None, cli=str(binary), resume=output))]
    receipt = launcher.native.read_json(next((output / "resume_traces").glob("*.json")))
    assert receipt["existing_records"] == 2 and receipt["skipped_record_ids"] == ["q-1", "q-2"]
    assert receipt["account_or_credit_mutations"] == 0
    assert (launcher.native.STREAMS, launcher.native.load_pool) == original


def test_existing_manifest_is_not_rewritten(launcher, tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    old = b'{"old":"unchanged"}\r\n'
    launcher.native.write_new(output / "manifest.json", old)
    monkeypatch.setattr(launcher, "validated_cli", lambda cli: tmp_path / "codex.exe")
    monkeypatch.setattr(launcher, "prepare", lambda *args: (tmp_path, tmp_path, output,
        dict(specification=dict(requested_model=None)), {}, dict(wrapper_sha256="old"), {}))
    monkeypatch.setattr(launcher.native, "run", lambda *args, **kwargs: (output, dict(failures=[])))
    launcher.resume(tmp_path, tmp_path, output, "a")
    assert (output / "manifest.json").read_bytes() == old


def test_resume_error_restores_single_stream_patches_without_retry(launcher, tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    launcher.native.write_new(output / "manifest.json", {})
    monkeypatch.setattr(launcher, "validated_cli", lambda cli: tmp_path / "codex.exe")
    monkeypatch.setattr(launcher, "prepare", lambda *args: (tmp_path, tmp_path, output,
        dict(specification=dict(requested_model=None)), {}, dict(wrapper_sha256="old"), {}))
    old = launcher.native.STREAMS, launcher.native.load_pool
    calls = []
    def failure(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("Synthetic workspace credit denial")
    monkeypatch.setattr(launcher.native, "run", failure)
    with pytest.raises(RuntimeError, match="credit denial"):
        launcher.resume(tmp_path, tmp_path, output, "b")
    assert len(calls) == 1 and (launcher.native.STREAMS, launcher.native.load_pool) == old


def test_outside_source_run_or_wrong_stream_rejected(launcher, tmp_path):
    with pytest.raises(ValueError, match="one stream"):
        launcher.prepare(tmp_path, tmp_path, tmp_path / "unrelated", "b")
    with pytest.raises(ValueError, match="one stream"):
        launcher.prepare(tmp_path, tmp_path, tmp_path / "codex_annotations/run", "both")
