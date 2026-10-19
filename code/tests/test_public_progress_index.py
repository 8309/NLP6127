"""Public progress guards using temporary synthetic receipts and 2 x 2 vectors.

These fixtures test validation only, never a real encoder or retrieval accuracy.
No live project data, network, model or paid service is used.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oil_evidence.store import digest, write_json, write_jsonl


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hashes(base, names):
    return {name: digest((base / name).read_bytes()) for name in names}


def rebind_index(case, value=None):
    if value is not None:
        write_json(case.result, value)
    write_json(case.run / "reports/index_build.json", {
        "path": case.result.relative_to(case.run).as_posix(),
        "sha256": digest(case.result.read_bytes()),
    })


def rebind_smoke(case, value=None):
    if value is not None:
        write_json(case.smoke, value)
    write_json(case.run / "reports/index_smoke.json", {
        "path": case.smoke.relative_to(case.run).as_posix(),
        "sha256": digest(case.smoke.read_bytes()),
    })


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/export_public_progress.py"
    spec = importlib.util.spec_from_file_location("isolated_public_progress", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    run = tmp_path / "data/unified_runs/semantic-page-v2-buffer"
    stage = run / "stage"
    for name in exporter.IMPLEMENTATION_FILES:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# temporary fixture " + name, encoding="utf-8")
    (tmp_path / "config.toml").write_text(
        '[retrieval]\nembedding_model="fixture/not-a-real-model"\n', encoding="utf-8")
    for name in ("scripts/build_staged_index.py", "scripts/check_staged_index.py"):
        (tmp_path / name).write_text("# synthetic fixture orchestration", encoding="utf-8")
    implementation = hashes(tmp_path, exporter.IMPLEMENTATION_FILES)
    rows = [{"passage_id": "private-fixture-id-a", "text": "Private fixture oil production.",
             "available_by": "2020-01-01", "regions": ["Middle East"],
             "source_page_id": "private-page-a", "char_start": 0, "char_end": 31},
            {"passage_id": "private-fixture-id-b", "text": "Private fixture petroleum exports.",
             "available_by": "2021-01-01", "regions": ["Europe"],
             "source_page_id": "private-page-b", "char_start": 0, "char_end": 34}]
    for row in rows:
        row["char_end"] = len(row["text"])
    write_jsonl(stage / "passages.jsonl", rows)
    write_jsonl(stage / "source_pages.jsonl", [
        {"source_page_id": row["source_page_id"], "text": row["text"]} for row in rows])
    for name in exporter.STAGE_FILES[2:-1]:
        write_jsonl(stage / name, [{"private_fixture": name}])
    corpus = {"passages": 2, "documents": 2, "words": 10, "unique_words": 7, "source_pages": 2,
              "corpus_sha256": digest((stage / "passages.jsonl").read_bytes()),
              "source_pages_sha256": digest((stage / "source_pages.jsonl").read_bytes()),
              "implementation_hashes": implementation}
    write_json(stage / "reports/corpus.json", corpus)
    stage_hashes = hashes(stage, exporter.STAGE_FILES)
    audit = {"corpus_sha256": corpus["corpus_sha256"], "errors": [], "index_verified": False,
             "stage_hashes": stage_hashes, "implementation_hashes": implementation, "active_passages": 2}
    write_json(stage / "reports/unified_quality.json", audit)
    for label in ("legacy", "new"):
        write_json(run / f"reports/{label}-qa.json", {
            "remaining_critical_count": 0, "input_hashes": stage_hashes,
            "implementation_hashes": implementation,
            "private_source_quote": "Not permitted in the public summary."})
    junit = run / "reports/suite.xml"
    junit.write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="1"/></testsuites>',
                     encoding="utf-8")
    suite = {"corpus_sha256": corpus["corpus_sha256"], "passed": 2, "skipped": 1, "failed": 0,
             "elapsed_seconds": 0.1, "skip_reason": "Synthetic fixture skip", "junit_xml": "reports/suite.xml",
             "junit_sha256": digest(junit.read_bytes())}
    write_json(run / "reports/full-suite.json", suite)
    checkpoint = {
        "corpus": {"passages": 2, "words": 10, "unique_words": 7, "contributing_documents": 2,
                   "cleaned_source_pages": 2, "minimum_passages": 10000, "passage_buffer": -9998,
                   "passage_buffer_percent": -99.98, "inherited_labels": 0,
                   "sha256": corpus["corpus_sha256"], "source_pages_sha256": corpus["source_pages_sha256"],
                   "private_extra_field": "private-corpus-text-must-not-export"},
        "segmentation": {"target_words": 160, "soft_minimum_words": 65, "soft_maximum_words": 240,
                         "maximum_context_words": 480, "parameters_unchanged": True},
        "source_holds": {"physical_pages": 1, "full_appendices": 0, "trailing_reference_pages": 0,
                         "unverified_tail_page": 1, "mixed_start_page_body_loss_disclosed": True,
                         "originals_retained": True, "repaired_report_final_passages": 0},
        "tests": {"report": "reports/full-suite.json", "sha256": digest((run / "reports/full-suite.json").read_bytes())},
        "reconstruction_audit": {"report": "stage/reports/unified_quality.json",
                                 "sha256": digest((stage / "reports/unified_quality.json").read_bytes())},
        "source_qa": {"fixed_cases": 2, "additional_legacy_diagnostics": 0, "total_original_cases_preserved": 2,
                      "new_original_critical_cases_excluded_not_passed": 1, "warnings_retained": 1,
                      "remaining_known_sample_criticals": 0, "human_gold": False, "whole_corpus_approved": False,
                      **{f"{label}_final_report": f"reports/{label}-qa.json" for label in ("legacy", "new")},
                      **{f"{label}_sha256": digest((run / f"reports/{label}-qa.json").read_bytes()) for label in ("legacy", "new")}},
        "protection": {"default_corpus_passages": 1, "new_tracked_paid_model_api_calls": 0, "new_human_labels": 0},
        "release": {"new_index_created": False, "published": False, "evaluation_set_frozen": False,
                    "formal_assignment_complete": False},
    }
    checkpoint_path = run / "reports/completion_checkpoint.json"
    write_json(checkpoint_path, checkpoint)
    for name in ("passages.jsonl", "index/manifest.json", "index/vectors.npy", "annotations.sqlite3", "model_usage.sqlite3"):
        file = tmp_path / "data" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(("protected synthetic fixture " + name).encode())
    protected_paths = [stage / name for name in exporter.STAGE_FILES] + [
        stage / "reports/unified_quality.json", checkpoint_path] + [
        tmp_path / "data" / name for name in (
            "passages.jsonl", "index/manifest.json", "index/vectors.npy", "annotations.sqlite3", "model_usage.sqlite3")]
    protected = {str(file.resolve()): digest(file.read_bytes()) for file in protected_paths}
    index = stage / "index"
    index.mkdir()
    np.save(index / "vectors.npy", np.eye(2, dtype=np.float32), allow_pickle=False)
    onnx = tmp_path / "models" / "fixture/not-an-actual-model.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"Fixture is deliberately not a real encoder")
    manifest = {"model": "fixture/not-a-real-model", "dimension": 2, "corpus_sha256": corpus["corpus_sha256"],
                "passage_ids": [row["passage_id"] for row in rows],
                "vectors_sha256": digest((index / "vectors.npy").read_bytes()),
                "onnx_sha256": {"fixture/not-an-actual-model.onnx": digest(onnx.read_bytes())}}
    write_json(index / "manifest.json", manifest)
    attempt = run / "reports/index_builds/fixture-attempt"
    attempt.mkdir(parents=True)
    for name in ("source_audit.json", "indexed_audit.json"):
        write_json(attempt / name, {**audit, "index_verified": name == "indexed_audit.json"})
    indexed = {
        "schema_version": 1, "run_id": run.name, "status": "staged_index_verified_not_published",
        "index_verified": True, "audit_errors": [], "protected_files_unchanged": True, "published": False,
        "formal_retrieval_evaluation_complete": False, "generated_answers": 0, "new_human_labels": 0,
        "hosted_model_calls": 0, "stage_hashes": stage_hashes, "implementation_hashes": implementation,
        "orchestrator_sha256": digest((tmp_path / "scripts/build_staged_index.py").read_bytes()),
        "manifest_sha256": digest((index / "manifest.json").read_bytes()), "vectors_sha256": manifest["vectors_sha256"],
        "passages": 2, "model": manifest["model"], "dimension": 2, "vectors_shape": [2, 2],
        "vectors_dtype": "float32", "elapsed_seconds": 0.1, "corpus_sha256": corpus["corpus_sha256"],
        "local_encoder_model_hashes": manifest["onnx_sha256"], "protected_sha256": protected,
        "audit_reports": {name: {"path": (attempt / name).relative_to(run).as_posix(),
                                  "sha256": digest((attempt / name).read_bytes())}
                          for name in ("source_audit.json", "indexed_audit.json")},
        "private_extra_key": "do-not-export-this-private-key",
    }
    result = attempt / "result.json"
    write_json(result, indexed)
    records = []
    for case in range(8):
        for mode in ("bm25", "dense", "hybrid"):
            records.append({"question": "Private machine-only question that must never be exported?",
                            "as_of": "2017-12-31" if case == 6 else "2026-10-05",
                            "region": "Middle East" if case == 7 else None, "expect_empty": case == 6,
                            "mode": mode, "elapsed_seconds": 0.001,
                            "evidence": [] if case == 6 else [{**rows[0], "rank": 1, "score": 0.2}]})
    smoke = attempt / "retrieval_smoke-fixture.json"
    checked = {"schema_version": 1, "run_id": run.name, "status": "operational_smoke_passed_not_formal_evaluation",
               "errors": [], "checks": 24, "results": records, "index_receipt_sha256": digest(result.read_bytes()),
               "corpus_sha256": corpus["corpus_sha256"], "manifest_sha256": indexed["manifest_sha256"],
               "vectors_sha256": indexed["vectors_sha256"], "passages": 2, "bm25_documents": 2,
               "modes": ["bm25", "dense", "hybrid"], "exact_offsets_verified": True,
               "date_and_region_filters_verified": True, "inputs_unchanged": True, "human_gold": False,
               "published": False, "generated_answers": 0, "hosted_model_calls": 0, "new_human_labels": 0,
               "checker_sha256": digest((tmp_path / "scripts/check_staged_index.py").read_bytes())}
    write_json(smoke, checked)
    return SimpleNamespace(root=tmp_path, run=run, stage=stage, index=index, attempt=attempt,
                           result=result, smoke=smoke, exporter=exporter, checkpoint=checkpoint_path,
                           protected=protected, rows=rows, indexed=indexed, checked=checked)


def export(case):
    return case.exporter.export()[1]


def test_absent_optional_receipts_preserve_historical_no_index_status(fixture):
    result = export(fixture)
    assert result["release"]["new_index_created"] is False
    assert result["release"]["published"] is False
    assert "current_index" not in result and "retrieval_smoke" not in result
    assert result["source_reconstruction"]["index_verified"] is False
    assert "Historical" in result["source_reconstruction"]["scope"]


def test_valid_index_without_smoke_is_reported_separately(fixture):
    rebind_index(fixture)
    result = export(fixture)
    assert result["current_index"]["index_verified"] is True
    assert result["current_index"]["vectors_shape"] == [2, 2]
    assert result["release"]["new_index_created"] is True
    assert result["release"]["published"] is False
    assert result["source_reconstruction"]["index_verified"] is False
    assert "retrieval_smoke" not in result


def rebind_model_mapping(case, mapping):
    manifest_path = case.index / "manifest.json"
    manifest = load(manifest_path)
    manifest["onnx_sha256"] = mapping
    write_json(manifest_path, manifest)
    indexed = copy.deepcopy(case.indexed)
    indexed["local_encoder_model_hashes"] = mapping
    indexed["manifest_sha256"] = digest(manifest_path.read_bytes())
    rebind_index(case, indexed)


def test_windows_relative_model_keys_work_without_mutating_manifest_or_receipt(fixture):
    mapping = fixture.indexed["local_encoder_model_hashes"]
    windows_mapping = {name.replace("/", "\\"): sha for name, sha in mapping.items()}
    rebind_model_mapping(fixture, windows_mapping)
    before = (fixture.index / "manifest.json").read_bytes(), fixture.result.read_bytes()
    result = export(fixture)
    assert result["current_index"]["index_verified"] is True
    assert result["release"]["new_index_created"] is True
    assert result["release"]["published"] is False
    assert before == ((fixture.index / "manifest.json").read_bytes(), fixture.result.read_bytes())
    assert load(fixture.index / "manifest.json")["onnx_sha256"] == windows_mapping


@pytest.mark.parametrize("name", [
    "C:\\outside.onnx", "C:/outside.onnx", "C:outside.onnx", "..\\outside.onnx",
    "fixture\\..\\outside.onnx", "\\\\server\\share\\outside.onnx", "//server/share/outside.onnx",
    "\\outside.onnx", "/outside.onnx", "fixture\\.\\not-an-actual-model.onnx",
])
def test_legacy_model_keys_still_refuse_drive_unc_absolute_and_traversal(fixture, name):
    sha = next(iter(fixture.indexed["local_encoder_model_hashes"].values()))
    rebind_model_mapping(fixture, {name: sha})
    with pytest.raises(ValueError):
        export(fixture)


def test_legacy_model_keys_refuse_duplicate_normalized_paths(fixture):
    name, sha = next(iter(fixture.indexed["local_encoder_model_hashes"].items()))
    rebind_model_mapping(fixture, {name: sha, name.replace("/", "\\"): sha})
    with pytest.raises(ValueError, match="Duplicate normalized"):
        export(fixture)


def test_valid_index_and_smoke_export_only_whitelisted_aggregates(fixture):
    frozen = fixture.checkpoint.read_bytes(), (fixture.stage / "reports/unified_quality.json").read_bytes()
    rebind_index(fixture)
    rebind_smoke(fixture)
    result = export(fixture)
    assert result["retrieval_smoke"]["checks"] == 24
    assert result["retrieval_smoke"]["human_gold"] is False
    assert result["release"]["formal_assignment_complete"] is False
    assert "do not establish retrieval accuracy" in " ".join(result["limitations"])
    serialized = json.dumps(result)
    for private in (str(fixture.root), "private-fixture-id", "Private fixture", "Private machine-only question",
                    "do-not-export-this-private-key", "private-corpus-text-must-not-export", "protected_sha256",
                    "local_encoder_model_hashes", "passage_ids", '"results"'):
        assert private not in serialized
    assert frozen == (fixture.checkpoint.read_bytes(), (fixture.stage / "reports/unified_quality.json").read_bytes())


def test_orphan_smoke_receipt_fails_closed(fixture):
    rebind_smoke(fixture)
    with pytest.raises(ValueError, match="without a verified index"):
        export(fixture)


@pytest.mark.parametrize("name", ["../outside.json", "/outside.json", "C:/outside.json",
                                  "reports/../outside.json", "reports\\outside.json", "reports//outside.json"])
def test_index_pointer_unsafe_paths_fail_closed(fixture, name):
    write_json(fixture.run / "reports/index_build.json", {"path": name, "sha256": "0" * 64})
    with pytest.raises(ValueError):
        export(fixture)


@pytest.mark.parametrize("mutation", ["hash", "malformed_hash", "missing", "invalid_json"])
def test_bad_index_pointer_fails_closed(fixture, mutation):
    rebind_index(fixture)
    pointer = fixture.run / "reports/index_build.json"
    value = load(pointer)
    if mutation == "hash":
        value["sha256"] = "0" * 64
    elif mutation == "malformed_hash":
        value["sha256"] = "not a hash"
    elif mutation == "missing":
        value["path"] = "reports/missing-result.json"
    else:
        fixture.result.write_text("{ invalid json }", encoding="utf-8")
        value["sha256"] = digest(fixture.result.read_bytes())
    write_json(pointer, value)
    with pytest.raises((ValueError, KeyError)):
        export(fixture)
    assert not (fixture.root / "reports/progress.json").exists()


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("index_verified", False), ("audit_errors", ["bad"]),
    ("published", True), ("protected_files_unchanged", False), ("formal_retrieval_evaluation_complete", True),
    ("hosted_model_calls", 1), ("generated_answers", 1), ("new_human_labels", 1), ("run_id", "another-run"),
    ("passages", 3), ("model", "different/model"), ("dimension", 3), ("vectors_shape", [3, 2]),
    ("vectors_dtype", "float64"), ("corpus_sha256", "0" * 64), ("stage_hashes", {}),
    ("implementation_hashes", {}), ("orchestrator_sha256", "0" * 64), ("local_encoder_model_hashes", {}),
    ("protected_sha256", {}), ("audit_reports", {}),
])
def test_invalid_index_receipt_state_fails_closed(fixture, field, value):
    indexed = copy.deepcopy(fixture.indexed)
    indexed[field] = value
    rebind_index(fixture, indexed)
    with pytest.raises(ValueError):
        export(fixture)


@pytest.mark.parametrize("mutation", ["order", "duplicate_id", "model", "corpus", "dimension"])
def test_rebound_but_invalid_manifest_fails_closed(fixture, mutation):
    manifest = load(fixture.index / "manifest.json")
    if mutation == "order":
        manifest["passage_ids"].reverse()
    elif mutation == "duplicate_id":
        manifest["passage_ids"][1] = manifest["passage_ids"][0]
    else:
        manifest[{"model": "model", "corpus": "corpus_sha256", "dimension": "dimension"}[mutation]] = (
            "0" * 64 if mutation == "corpus" else "different/model" if mutation == "model" else 3)
    write_json(fixture.index / "manifest.json", manifest)
    indexed = copy.deepcopy(fixture.indexed)
    indexed["manifest_sha256"] = digest((fixture.index / "manifest.json").read_bytes())
    rebind_index(fixture, indexed)
    with pytest.raises(ValueError, match="Index corpus/model/hash/order/shape/finite/norm"):
        export(fixture)


@pytest.mark.parametrize("mutation", ["float64", "wrong_shape", "nonfinite", "zero_norm", "wrong_norm"])
def test_rebound_but_invalid_real_vectors_fail_closed(fixture, mutation):
    vectors = np.eye(2, dtype=np.float64 if mutation == "float64" else np.float32)
    if mutation == "wrong_shape":
        vectors = vectors[:1]
    elif mutation == "nonfinite":
        vectors[0, 0] = np.nan
    elif mutation == "zero_norm":
        vectors[0] = 0
    elif mutation == "wrong_norm":
        vectors[0] *= 2
    np.save(fixture.index / "vectors.npy", vectors, allow_pickle=False)
    manifest = load(fixture.index / "manifest.json")
    manifest["vectors_sha256"] = digest((fixture.index / "vectors.npy").read_bytes())
    write_json(fixture.index / "manifest.json", manifest)
    indexed = copy.deepcopy(fixture.indexed)
    indexed["vectors_sha256"] = manifest["vectors_sha256"]
    indexed["manifest_sha256"] = digest((fixture.index / "manifest.json").read_bytes())
    rebind_index(fixture, indexed)
    with pytest.raises(ValueError, match="Index corpus/model/hash/order/shape/finite/norm"):
        export(fixture)


@pytest.mark.parametrize("mutation", ["corpus", "onnx", "implementation", "live_ledger", "missing_protected",
                                     "published_marker", "orchestrator"])
def test_postbuild_artifact_or_protection_drift_fails_closed(fixture, mutation):
    rebind_index(fixture)
    paths = {"corpus": fixture.stage / "passages.jsonl",
             "onnx": fixture.root / "models/fixture/not-an-actual-model.onnx",
             "implementation": fixture.root / "app.py", "live_ledger": fixture.root / "data/model_usage.sqlite3",
             "orchestrator": fixture.root / "scripts/build_staged_index.py"}
    if mutation == "missing_protected":
        value = copy.deepcopy(fixture.indexed)
        value["protected_sha256"].pop(str((fixture.root / "data/model_usage.sqlite3").resolve()))
        rebind_index(fixture, value)
    elif mutation == "published_marker":
        (fixture.run / "published").mkdir()
    else:
        paths[mutation].write_bytes(b"Changed after the verified fixture build")
    with pytest.raises(ValueError):
        export(fixture)


@pytest.mark.parametrize("mutation", ["hash", "errors", "index_false", "corpus", "stage", "implementation"])
def test_indexed_audit_must_be_current_success(fixture, mutation):
    indexed = copy.deepcopy(fixture.indexed)
    audit_path = fixture.attempt / "indexed_audit.json"
    audit = load(audit_path)
    if mutation == "hash":
        indexed["audit_reports"]["indexed_audit.json"]["sha256"] = "0" * 64
    else:
        fields = {"errors": ("errors", ["bad"]), "index_false": ("index_verified", False),
                  "corpus": ("corpus_sha256", "0" * 64), "stage": ("stage_hashes", {}),
                  "implementation": ("implementation_hashes", {})}
        key, value = fields[mutation]
        audit[key] = value
        write_json(audit_path, audit)
        indexed["audit_reports"]["indexed_audit.json"]["sha256"] = digest(audit_path.read_bytes())
    rebind_index(fixture, indexed)
    with pytest.raises(ValueError):
        export(fixture)


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("errors", ["bad"]), ("checks", 23), ("results", []), ("run_id", "different-run"),
    ("index_receipt_sha256", "0" * 64), ("corpus_sha256", "0" * 64), ("manifest_sha256", "0" * 64),
    ("vectors_sha256", "0" * 64), ("bm25_documents", 1), ("modes", ["bm25"]),
    ("exact_offsets_verified", False), ("date_and_region_filters_verified", False), ("inputs_unchanged", False),
    ("human_gold", True), ("published", True), ("hosted_model_calls", 1), ("checker_sha256", "0" * 64),
])
def test_smoke_receipt_state_and_binding_fail_closed(fixture, field, value):
    rebind_index(fixture)
    checked = copy.deepcopy(fixture.checked)
    checked[field] = value
    rebind_smoke(fixture, checked)
    with pytest.raises(ValueError):
        export(fixture)


@pytest.mark.parametrize("mutation", ["cutoff", "region", "rank", "score", "text", "duplicate", "unexpected_empty", "mode_coverage"])
def test_smoke_results_are_verified_not_just_success_flags(fixture, mutation):
    rebind_index(fixture)
    checked = copy.deepcopy(fixture.checked)
    record = checked["results"][0]
    if mutation == "cutoff":
        record["as_of"] = "2019-12-31"
    elif mutation == "region":
        record["region"] = "Europe"
    elif mutation == "rank":
        record["evidence"][0]["rank"] = 2
    elif mutation == "score":
        record["evidence"][0]["score"] = float("inf")
    elif mutation == "text":
        record["evidence"][0]["text"] = "Not the original fixture text"
    elif mutation == "duplicate":
        record["evidence"].append(record["evidence"][0])
    elif mutation == "unexpected_empty":
        record["evidence"] = []
    else:
        record["mode"] = "dense"
    rebind_smoke(fixture, checked)
    with pytest.raises(ValueError):
        export(fixture)


def test_public_output_cannot_escape_reports(fixture):
    with pytest.raises(ValueError, match="Public export must stay"):
        fixture.exporter.export(fixture.root / "outside.json")
    assert not (fixture.root / "outside.json").exists()
