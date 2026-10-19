"""Smoke-checker guards on private fixtures, never real model/data evidence.

Synthetic identity vectors and a fixed query vector isolate the operational
checks. They do not establish embedding or retrieval quality.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from oil_evidence import retrieval, store
from oil_evidence.store import digest, write_json, write_jsonl


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def smoke_case(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    pipeline = load_module("isolated_smoke_pipeline", scripts / "unify_corpus.py")
    monkeypatch.setitem(sys.modules, "unify_corpus", pipeline)
    builder = load_module("isolated_smoke_builder", scripts / "build_staged_index.py")
    monkeypatch.setitem(sys.modules, "build_staged_index", builder)
    checker = load_module("isolated_smoke_checker", scripts / "check_staged_index.py")
    root = tmp_path / "fixture-project"
    data = root / "data"
    run_id = "fixture-smoke-v1"
    run = data / "unified_runs" / run_id
    stage = run / "stage"
    attempt = run / "reports/index_builds/fixture-attempt"
    attempt.mkdir(parents=True)
    stage.mkdir()
    monkeypatch.setattr(pipeline, "DATA", data)
    monkeypatch.setattr(pipeline, "RUN", run)
    monkeypatch.setattr(checker, "ROOT", root)
    settings = {"retrieval": {"embedding_model": "fixture/not-a-real-encoder",
                               "top_k": 5, "candidate_k": 50, "rrf_k": 60}}
    monkeypatch.setattr(store, "config", lambda: settings)
    monkeypatch.setattr(retrieval, "config", lambda: settings)
    rows, pages = [], []
    for number, (available, regions, text) in enumerate([
            ("2024-01-01", ["Middle East"], "Oil production and exports remained stable."),
            ("2025-01-01", ["United States"], "Crude oil production rose during the year."),
            ("2026-01-01", ["Middle East"], "Petroleum exports and oil supply increased."),
            (None, ["United States"], "Oil exports in the undated fixture were reported.")]):
        page_id = f"fixture-page-{number}"
        prefix = "Temporary source header. "
        rows.append({"passage_id": f"fixture-{number}", "text": text,
                     "available_by": available, "regions": regions,
                     "source_page_id": page_id, "char_start": len(prefix),
                     "char_end": len(prefix) + len(text), "page": 1,
                     "title": "Private synthetic source fixture"})
        pages.append({"source_page_id": page_id, "text": prefix + text})
    write_jsonl(stage / "passages.jsonl", rows)
    write_jsonl(stage / "source_pages.jsonl", pages)
    write_json(stage / "reports/corpus.json", {"fixture_only": True})
    stage_names = ["passages.jsonl", "source_pages.jsonl", "reports/corpus.json"]
    monkeypatch.setattr(pipeline, "stage_hashes", lambda: {
        name: digest((stage / name).read_bytes()) for name in stage_names})
    implementation = root / "fixture-implementation.py"
    implementation.write_text("# private fixture implementation\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "hashes", lambda: {
        "fixture-implementation.py": digest(implementation.read_bytes())})
    model = root / "models/fixture-not-a-real-model.onnx"
    model.parent.mkdir()
    model.write_bytes(b"Synthetic model identity bytes, not an ONNX model")
    index = stage / "index"
    index.mkdir()
    np.save(index / "vectors.npy", np.eye(len(rows), dtype=np.float32), allow_pickle=False)
    manifest = {"model": settings["retrieval"]["embedding_model"], "dimension": len(rows),
                "corpus_sha256": digest((stage / "passages.jsonl").read_bytes()),
                "vectors_sha256": digest((index / "vectors.npy").read_bytes()),
                "passage_ids": [row["passage_id"] for row in rows],
                "onnx_sha256": {model.name: digest(model.read_bytes())}}
    write_json(index / "manifest.json", manifest)
    for name in ["source_audit.json", "indexed_audit.json"]:
        write_json(attempt / name, {"fixture_audit": name})
    protected = data / "live-protected.txt"
    protected.write_bytes(b"original temporary protected input")
    receipt = {"corpus_sha256": manifest["corpus_sha256"],
               "manifest_sha256": digest((index / "manifest.json").read_bytes()),
               "vectors_sha256": manifest["vectors_sha256"],
               "passages": len(rows), "dimension": manifest["dimension"],
               "model": manifest["model"], "vectors_shape": [len(rows), len(rows)],
               "vectors_dtype": "float32", "published": False, "run_id": run_id,
               "stage_hashes": pipeline.stage_hashes(),
               "implementation_hashes": pipeline.hashes(), "index_verified": True,
               "audit_errors": [], "local_encoder_model_hashes": manifest["onnx_sha256"],
               "protected_sha256": builder.file_hashes([protected]),
               "audit_reports": {name: {
                   "path": (attempt / name).relative_to(run).as_posix(),
                   "sha256": digest((attempt / name).read_bytes())}
                   for name in ["source_audit.json", "indexed_audit.json"]}}
    receipt_path = attempt / "result.json"
    write_json(receipt_path, receipt)
    pointer = run / "reports/index_build.json"
    write_json(pointer, {"path": receipt_path.relative_to(run).as_posix(),
                         "sha256": digest(receipt_path.read_bytes())})
    write_json(root / "seeds/questions.json", [{
        "question": f"Synthetic oil production and exports query {number}?"}
        for number in range(14)])
    calls = []

    class FixtureEncoder:
        def query_embed(self, questions):
            calls.extend(questions)
            return [np.array([1., 0., 0., 0.], dtype=np.float32) for _ in questions]

    monkeypatch.setattr(checker, "embedder", lambda model_name: FixtureEncoder())
    return SimpleNamespace(checker=checker, pipeline=pipeline, builder=builder,
                           root=root, data=data, run=run, stage=stage, run_id=run_id,
                           attempt=attempt, receipt=receipt, receipt_path=receipt_path,
                           pointer=pointer, rows=rows, pages=pages, model=model,
                           protected=protected, calls=calls, monkeypatch=monkeypatch)


def bind_receipt(case):
    write_json(case.receipt_path, case.receipt)
    write_json(case.pointer, {"path": case.receipt_path.relative_to(case.run).as_posix(),
                             "sha256": digest(case.receipt_path.read_bytes())})


def assert_refused(case, message):
    smoke_pointer = case.run / "reports/index_smoke.json"
    write_json(smoke_pointer, {"historical_smoke_pointer": True})
    before = smoke_pointer.read_bytes()
    with pytest.raises(ValueError, match=message):
        case.checker.check(case.run_id)
    assert smoke_pointer.read_bytes() == before
    assert list(case.attempt.glob("retrieval_smoke-*.json")) == []
    assert not (case.data / ".corpus_writer.lock").exists()


def test_receipt_path_accepts_only_existing_files_within_run(smoke_case):
    case = smoke_case
    relative = case.receipt_path.relative_to(case.run).as_posix()
    assert case.checker.run_file(case.run, relative) == case.receipt_path.resolve()
    assert case.checker.run_file(case.run, case.receipt_path) == case.receipt_path.resolve()


@pytest.mark.parametrize("target", ["../outside.json", "reports/missing.json", "absolute_outside"])
def test_receipt_path_refuses_escape_and_missing(smoke_case, target):
    case = smoke_case
    outside = case.run.parent / "outside.json"
    write_json(outside, {"outside_fixture": True})
    if target == "absolute_outside":
        target = outside
    with pytest.raises(ValueError, match="outside run or missing"):
        case.checker.run_file(case.run, target)


@pytest.mark.parametrize("mismatch", [
    "pointer_hash", "receipt_corpus", "receipt_manifest", "receipt_stage",
    "receipt_implementation", "receipt_unverified", "receipt_errors",
    "receipt_vectors", "receipt_passages", "receipt_dimension", "receipt_model",
    "receipt_shape", "receipt_dtype", "receipt_published", "receipt_run_id",
    "receipt_nonboolean_verified", "receipt_nonboolean_published",
    "audit", "manifest_order", "vectors", "encoder", "encoder_escape", "protected"])
def test_changed_receipt_index_audit_encoder_or_input_refused(smoke_case, mismatch):
    case = smoke_case
    message = "Index receipt no longer matches"
    if mismatch == "pointer_hash":
        write_json(case.pointer, {"path": case.receipt_path.relative_to(case.run).as_posix(),
                                  "sha256": "0" * 64})
        message = "Index receipt hash differs"
    elif mismatch.startswith("receipt_"):
        key, value = {
            "receipt_corpus": ("corpus_sha256", "0" * 64),
            "receipt_manifest": ("manifest_sha256", "0" * 64),
            "receipt_stage": ("stage_hashes", {}),
            "receipt_implementation": ("implementation_hashes", {}),
            "receipt_unverified": ("index_verified", False),
            "receipt_errors": ("audit_errors", ["synthetic failed audit"]),
            "receipt_vectors": ("vectors_sha256", "0" * 64),
            "receipt_passages": ("passages", 5),
            "receipt_dimension": ("dimension", 5),
            "receipt_model": ("model", "different-fixture-model"),
            "receipt_shape": ("vectors_shape", [4, 3]),
            "receipt_dtype": ("vectors_dtype", "float64"),
            "receipt_published": ("published", True),
            "receipt_run_id": ("run_id", "different-fixture-v1"),
            "receipt_nonboolean_verified": ("index_verified", 1),
            "receipt_nonboolean_published": ("published", 0),
        }[mismatch]
        case.receipt[key] = value
        bind_receipt(case)
    elif mismatch == "audit":
        write_json(case.attempt / "source_audit.json", {"mutated_fixture": True})
        message = "Index audit receipt changed"
    elif mismatch == "manifest_order":
        path = case.stage / "index/manifest.json"
        manifest = read_json(path)
        manifest["passage_ids"].reverse()
        write_json(path, manifest)
        message = "Index corpus/model/hash/order/shape/norm mismatch"
    elif mismatch == "vectors":
        np.save(case.stage / "index/vectors.npy", np.zeros((4, 4), dtype=np.float32), allow_pickle=False)
        message = "Index corpus/model/hash/order/shape/norm mismatch"
    elif mismatch == "encoder":
        case.model.write_bytes(b"changed synthetic model bytes")
        message = "Local encoder file differs"
    elif mismatch == "encoder_escape":
        case.receipt["local_encoder_model_hashes"] = {"../outside-fixture.onnx": "0" * 64}
        bind_receipt(case)
        message = "Local encoder file differs"
    else:
        case.protected.write_bytes(b"changed temporary protected input")
        message = "Protected input changed"
    assert_refused(case, message)
    assert case.calls == []


@pytest.mark.parametrize("bad_result", ["future", "unknown_date", "region", "rank",
                                         "nonfinite_score", "duplicate", "empty", "early_nonempty"])
def test_invalid_retrieval_results_refused(smoke_case, bad_result):
    case = smoke_case
    original_class = case.checker.Retriever

    class BadFixtureRetriever(original_class):
        def search(self, question, **kwargs):
            hits = super().search(question, **kwargs)
            if bad_result == "region" and not kwargs["region"]:
                return hits
            if bad_result == "early_nonempty" and kwargs["as_of"] != "2017-12-31":
                return hits
            if bad_result in {"future", "unknown_date", "region", "early_nonempty"}:
                selected = {"future": 2, "unknown_date": 3, "region": 1, "early_nonempty": 0}[bad_result]
                return [dict(case.rows[selected], rank=1, score=0.5)]
            if bad_result == "empty":
                return []
            if bad_result == "duplicate":
                return hits + hits[:1]
            hits[0]["rank" if bad_result == "rank" else "score"] = (
                2 if bad_result == "rank" else float("nan"))
            return hits

    case.monkeypatch.setattr(case.checker, "Retriever", BadFixtureRetriever)
    message = ("Unexpected empty/nonempty" if bad_result in {"empty", "early_nonempty"}
               else "Duplicate retrieval evidence" if bad_result == "duplicate"
               else "provenance/rank/filter/source offset mismatch")
    assert_refused(case, message)


def test_exact_source_offsets_checked_separately_from_hit_metadata(smoke_case):
    case = smoke_case
    original_read = case.checker.read_jsonl

    def mismatched_source_page(path):
        records = original_read(path)
        if Path(path) == case.stage / "source_pages.jsonl":
            for record in records:
                record["text"] = "Intentionally incorrect in-memory fixture page."
        return records

    case.monkeypatch.setattr(case.checker, "read_jsonl", mismatched_source_page)
    assert_refused(case, "provenance/rank/filter/source offset mismatch")


def test_input_changed_during_query_refused_before_report(smoke_case):
    case = smoke_case
    original_class = case.checker.Retriever

    class MutatingFixtureRetriever(original_class):
        def search(self, question, **kwargs):
            result = super().search(question, **kwargs)
            case.protected.write_bytes(b"changed during temporary fixture query")
            return result

    case.monkeypatch.setattr(case.checker, "Retriever", MutatingFixtureRetriever)
    assert_refused(case, "Protected input changed")


def test_published_run_refused_without_success_report(smoke_case):
    case = smoke_case
    (case.run / "published").mkdir()
    assert_refused(case, "Unexpected publication")


def test_successful_fixture_smoke_report_keeps_provenance_and_no_live_writes(smoke_case):
    case = smoke_case
    paths = [case.protected, case.receipt_path, case.pointer,
             case.stage / "passages.jsonl", case.stage / "source_pages.jsonl",
             case.stage / "index/manifest.json", case.stage / "index/vectors.npy"]
    before = case.builder.file_hashes(paths)
    result = case.checker.check(case.run_id)
    assert result["status"] == "operational_smoke_passed_not_formal_evaluation"
    assert result["checks"] == 24
    assert result["errors"] == []
    assert result["passages"] == result["bm25_documents"] == 4
    assert result["inputs_unchanged"] is True
    assert result["exact_offsets_verified"] is True
    assert result["date_and_region_filters_verified"] is True
    assert result["generated_answers"] == result["hosted_model_calls"] == 0
    assert result["human_gold"] is result["published"] is False
    assert len(result["results"]) == 24
    assert {record["mode"] for record in result["results"]} == {"bm25", "dense", "hybrid"}
    assert all(not record["evidence"] for record in result["results"] if record["expect_empty"])
    assert all("Middle East" in hit["regions"]
               for record in result["results"] if record["region"]
               for hit in record["evidence"])
    case.builder.assert_unchanged(before)
    pointer = read_json(case.run / "reports/index_smoke.json")
    report_path = case.checker.run_file(case.run, pointer["path"])
    assert report_path.parent == case.attempt.resolve()
    assert digest(report_path.read_bytes()) == pointer["sha256"]
    assert read_json(report_path) == result
    assert not (case.data / ".corpus_writer.lock").exists()
