"""Staged-index orchestration guards on temporary, synthetic fixtures only.

The 2 x 2 vectors are deliberately fake unit vectors, not an embedding model
or evidence of retrieval quality. No live data, network or encoder is used.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from oil_evidence import store
from oil_evidence.store import digest, write_json, write_jsonl


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_script(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "isolated_staged_index_pipeline", scripts / "unify_corpus.py")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    monkeypatch.setitem(sys.modules, "unify_corpus", pipeline)
    spec = importlib.util.spec_from_file_location(
        "isolated_staged_index_builder", scripts / "build_staged_index.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder, pipeline


@pytest.fixture
def staged(tmp_path, monkeypatch):
    builder, pipeline = load_script(monkeypatch)
    data = tmp_path / "fixture-project/data"
    run_id = "fixture-staged-v1"
    run = data / "unified_runs" / run_id
    stage = run / "stage"
    stage.mkdir(parents=True)
    monkeypatch.setattr(pipeline, "DATA", data)
    monkeypatch.setattr(pipeline, "RUN", run)
    monkeypatch.setattr(pipeline, "INPUTS", ["passages.jsonl", "index/manifest.json"])
    monkeypatch.setattr(store, "config", lambda: {
        "retrieval": {"embedding_model": "fixture/not-a-real-encoder"}})
    rows = [
        {"passage_id": "fixture-a", "text": "Oil production remained stable."},
        {"passage_id": "fixture-b", "text": "Petroleum exports were reported."},
    ]
    write_jsonl(stage / "passages.jsonl", rows)
    stage_names = ["passages.jsonl", "source_pages.jsonl", "quarantine.jsonl",
                   "source_exclusions.jsonl", "documents.jsonl", "sources.jsonl",
                   "reports/corpus.json"]
    for name in stage_names[1:]:
        write_json(stage / name, {"fixture": name})
    write_json(stage / "reports/unified_quality.json", {"historical_frozen_audit": True})
    for name in ["passages.jsonl", "index/manifest.json", "annotations.sqlite3",
                 "model_usage.sqlite3", "index/embedding_cache.sqlite3"]:
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("protected fixture, not a real database: " + name).encode())
    for name in ["completion_checkpoint.json", "live_closeout_protection.json",
                 "full_suite-final.json", "buffer_source_review-closeout-final.json",
                 "boem-goa-source-check-final.json"]:
        write_json(run / "reports" / name, {"frozen_source_closeout": name})
    implementation = tmp_path / "fixture-implementation.py"
    implementation.write_text("# private temporary fixture\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "stage_hashes", lambda: {
        name: digest((stage / name).read_bytes()) for name in stage_names})
    monkeypatch.setattr(pipeline, "hashes", lambda: {
        "fixture-implementation.py": digest(implementation.read_bytes())})
    events = []

    def fixture_audit(require_index=False):
        events.append("indexed_audit" if require_index else "source_audit")
        value = {"index_verified": require_index, "errors": []}
        pipeline.write_json(stage / "reports/unified_quality.json", value)
        return value

    def fixture_index():
        events.append("index")
        fixture_audit()
        target = stage / "index"
        target.mkdir()
        # Synthetic vectors test orchestration only. No model is called.
        vectors = np.eye(2, dtype=np.float32)
        np.save(target / "vectors.npy", vectors, allow_pickle=False)
        write_json(target / "manifest.json", {
            "model": "fixture/not-a-real-encoder", "dimension": 2,
            "corpus_sha256": digest((stage / "passages.jsonl").read_bytes()),
            "passage_ids": [row["passage_id"] for row in rows],
            "vectors_sha256": digest((target / "vectors.npy").read_bytes()),
            "onnx_sha256": {"fixture-not-an-actual-model.onnx": "0" * 64},
        })
        return {"passages": len(rows), "dimension": 2,
                "model": "fixture/not-a-real-encoder"}

    monkeypatch.setattr(pipeline, "index", fixture_index)
    monkeypatch.setattr(pipeline, "audit", fixture_audit)
    return SimpleNamespace(builder=builder, pipeline=pipeline, data=data,
                           run=run, run_id=run_id, stage=stage, rows=rows,
                           events=events, implementation=implementation,
                           monkeypatch=monkeypatch)


def attempt_dirs(case):
    folder = case.run / "reports/index_builds"
    return sorted(folder.iterdir()) if folder.exists() else []


def test_audit_destination_redirects_only_frozen_receipt_and_restores_writer(staged):
    original_writer = staged.pipeline.write_json
    source = staged.stage / "reports/unified_quality.json"
    before = source.read_bytes()
    destination = staged.run / "reports/new-source-audit.json"
    unrelated = staged.run / "reports/unrelated.json"
    with staged.builder.audit_destination(staged.pipeline, destination):
        staged.pipeline.write_json(source.parent / "." / source.name, {"new_audit": True})
        staged.pipeline.write_json(unrelated, {"unrelated": True})
    assert staged.pipeline.write_json is original_writer
    assert source.read_bytes() == before
    assert read_json(destination) == {"new_audit": True}
    assert read_json(unrelated) == {"unrelated": True}


def test_audit_destination_restores_writer_after_exception(staged):
    original_writer = staged.pipeline.write_json
    with pytest.raises(RuntimeError, match="fixture failure"):
        with staged.builder.audit_destination(staged.pipeline, staged.run / "reports/new.json"):
            raise RuntimeError("fixture failure")
    assert staged.pipeline.write_json is original_writer


@pytest.mark.parametrize("mutation", ["changed", "removed"])
def test_unchanged_guard_refuses_changed_or_missing_inputs(tmp_path, monkeypatch, mutation):
    builder, _ = load_script(monkeypatch)
    protected = tmp_path / "protected-input.txt"
    protected.write_bytes(b"original fixture")
    expected = builder.file_hashes([protected, tmp_path / "missing-before-snapshot"])
    assert expected == {str(protected.resolve()): digest(b"original fixture")}
    if mutation == "changed":
        protected.write_bytes(b"changed fixture")
    else:
        protected.unlink()
    with pytest.raises(ValueError, match="Protected input changed"):
        builder.assert_unchanged(expected)


@pytest.mark.parametrize("run_id", ["../escape", "../fixture-staged-v1", "bad/name",
                                    "nul", "con", "com1", "ab", "UPPERCASE"])
def test_invalid_run_refused_before_lock_or_index(staged, run_id):
    with pytest.raises(ValueError, match="Invalid run ID"):
        staged.builder.build(run_id)
    assert staged.events == []
    assert attempt_dirs(staged) == []
    assert not (staged.data / ".corpus_writer.lock").exists()


def test_published_run_refused_before_index(staged):
    (staged.run / "published").mkdir()
    with pytest.raises(ValueError, match="Published run is immutable"):
        staged.builder.build(staged.run_id)
    assert staged.events == []
    assert attempt_dirs(staged) == []


def test_missing_stage_refused_before_index(staged):
    (staged.stage / "passages.jsonl").unlink()
    with pytest.raises(ValueError, match="Stage the corpus"):
        staged.builder.build(staged.run_id)
    assert staged.events == []
    assert attempt_dirs(staged) == []


def test_success_pointer_written_only_after_audits_and_protection(staged):
    frozen = staged.stage / "reports/unified_quality.json"
    checkpoint = staged.run / "reports/completion_checkpoint.json"
    frozen_before, checkpoint_before = frozen.read_bytes(), checkpoint.read_bytes()
    pointer_path = staged.run / "reports/index_build.json"
    original_check = staged.builder.assert_unchanged
    original_write = staged.builder.write_json

    def checked(expected):
        assert not pointer_path.exists()
        assert staged.events == ["index", "source_audit", "indexed_audit"]
        original_check(expected)
        staged.events.append("protection_checked")

    def recorded_write(path, value):
        if Path(path) == pointer_path:
            assert staged.events[-1] == "protection_checked"
            result_path = staged.run / value["path"]
            assert result_path.is_file()
            assert digest(result_path.read_bytes()) == value["sha256"]
            staged.events.append("success_pointer")
        original_write(path, value)

    staged.monkeypatch.setattr(staged.builder, "assert_unchanged", checked)
    staged.monkeypatch.setattr(staged.builder, "write_json", recorded_write)
    result = staged.builder.build(staged.run_id)
    assert result["status"] == "staged_index_verified_not_published"
    assert result["vectors_shape"] == [2, 2]
    assert result["vectors_dtype"] == "float32"
    assert result["index_verified"] is True
    assert result["published"] is False
    assert result["hosted_model_calls"] == result["generated_answers"] == 0
    assert result["protected_files_unchanged"] is True
    assert staged.events[-1] == "success_pointer"
    assert frozen.read_bytes() == frozen_before
    assert checkpoint.read_bytes() == checkpoint_before
    attempts = attempt_dirs(staged)
    assert len(attempts) == 1
    assert not (attempts[0] / "failure.json").exists()
    for name, binding in result["audit_reports"].items():
        assert binding["path"] == (attempts[0] / name).relative_to(staged.run).as_posix()
        assert digest((staged.run / binding["path"]).read_bytes()) == binding["sha256"]
    assert read_json(attempts[0] / "source_audit.json")["index_verified"] is False
    assert read_json(attempts[0] / "indexed_audit.json")["index_verified"] is True
    assert not (staged.data / ".corpus_writer.lock").exists()


@pytest.mark.parametrize("failure_phase", ["index", "indexed_audit", "protection",
                                          "stage_drift", "implementation_drift",
                                          "publication", "float64"])
def test_failed_attempt_keeps_previous_success_pointer_and_failure_receipt(staged, failure_phase):
    frozen = staged.stage / "reports/unified_quality.json"
    before = frozen.read_bytes()
    original_writer = staged.pipeline.write_json
    original_index, original_audit = staged.pipeline.index, staged.pipeline.audit
    pointer = staged.run / "reports/index_build.json"
    write_json(pointer, {"path": "reports/historical-success.json", "sha256": "a" * 64})
    prior_pointer = pointer.read_bytes()

    def index_with_failure():
        result = original_index()
        if failure_phase == "index":
            raise RuntimeError("synthetic index failure")
        if failure_phase == "protection":
            (staged.data / "model_usage.sqlite3").write_bytes(b"mutated fixture ledger")
        if failure_phase == "stage_drift":
            write_json(staged.stage / "quarantine.jsonl", {"mutated_fixture": True})
        if failure_phase == "implementation_drift":
            staged.implementation.write_text("# changed private fixture\n", encoding="utf-8")
        if failure_phase == "publication":
            (staged.run / "published").mkdir()
        if failure_phase == "float64":
            vectors = staged.stage / "index/vectors.npy"
            np.save(vectors, np.eye(2, dtype=np.float64), allow_pickle=False)
            manifest_path = staged.stage / "index/manifest.json"
            manifest = read_json(manifest_path)
            manifest["vectors_sha256"] = digest(vectors.read_bytes())
            write_json(manifest_path, manifest)
        return result

    def audit_with_failure(require_index=False):
        result = original_audit(require_index)
        if failure_phase == "indexed_audit" and require_index:
            raise RuntimeError("synthetic post-index audit failure")
        return result

    staged.monkeypatch.setattr(staged.pipeline, "index", index_with_failure)
    staged.monkeypatch.setattr(staged.pipeline, "audit", audit_with_failure)
    with pytest.raises((RuntimeError, ValueError)):
        staged.builder.build(staged.run_id)
    attempts = attempt_dirs(staged)
    assert len(attempts) == 1
    failure = read_json(attempts[0] / "failure.json")
    assert failure["successful"] is False
    assert failure["published"] is False
    assert failure["error_type"] in {"RuntimeError", "ValueError"}
    assert (attempts[0] / "inputs.json").is_file()
    assert (attempts[0] / "source_audit.json").is_file()
    assert not (attempts[0] / "result.json").exists()
    assert pointer.read_bytes() == prior_pointer
    assert frozen.read_bytes() == before
    assert staged.pipeline.write_json is original_writer
    assert not (staged.data / ".corpus_writer.lock").exists()
