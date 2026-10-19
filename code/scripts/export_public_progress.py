"""Export a whitelisted, hash-verified public progress summary, never raw data."""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
import re
import tomllib
import xml.etree.ElementTree as ET

import numpy as np

from oil_evidence.store import ROOT, digest, now, read_jsonl


STAGE_FILES = (
    "passages.jsonl", "source_pages.jsonl", "quarantine.jsonl", "source_exclusions.jsonl",
    "documents.jsonl", "sources.jsonl", "reports/corpus.json",
)
IMPLEMENTATION_FILES = (
    "config.toml", "uv.lock", "pyproject.toml", "src/oil_evidence/store.py",
    "src/oil_evidence/corpus.py", "src/oil_evidence/unified_corpus.py",
    "src/oil_evidence/segmentation.py", "src/oil_evidence/retrieval.py",
    "src/oil_evidence/cli.py", "src/oil_evidence/corpus_versions.py", "app.py",
    "scripts/unify_corpus.py", "scripts/compare_segmentation.py",
    "scripts/reproduce_unified.py", "scripts/check_workbench.py",
    "scripts/reprocess_boem_supplement.py", "scripts/discover_boem_buffer.py",
    "scripts/collect_boem_supplement.py", "scripts/bounded_boem_fetcher.py",
)


def relative_file(base, name):
    """Accept existing, normalized relative paths only, including on Windows."""
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Unsafe progress receipt path")
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError("Progress receipt path escapes its directory or is missing")
    return path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("Malformed progress SHA256 binding")
    if digest(Path(path).read_bytes()) != expected:
        raise ValueError(f"Progress binding changed: {Path(path).name}")


def verify_mapping(base, values, expected=None):
    if not isinstance(values, dict) or not values or (expected is not None and set(values) != set(expected)):
        raise ValueError("Progress hash inventory differs")
    for name, sha in values.items():
        verify(relative_file(base, name), sha)


def verify_model_mapping(values):
    """Read legacy pathlib model keys without relaxing receipt-path safety.

    The unchanged index builder serializes Windows-relative ONNX paths with
    backslashes. Normalize only these local model keys, preserving the original
    manifest/receipt bytes and refusing traversal, drives, UNC or aliases.
    """
    if not isinstance(values, dict) or not values:
        raise ValueError("Missing local encoder model hash inventory")
    normalized = set()
    for name, sha in values.items():
        if not isinstance(name, str):
            raise ValueError("Invalid local encoder model path")
        relative = name.replace("\\", "/")
        if relative in normalized:
            raise ValueError("Duplicate normalized local encoder model path")
        normalized.add(relative)
        verify(relative_file(ROOT / "models", relative), sha)


def receipt(run, pointer_path):
    pointer = load(pointer_path)
    path = relative_file(run, pointer["path"])
    verify(path, pointer["sha256"])
    return path, pointer["sha256"], load(path)


def verified_index(run, corpus):
    """Recheck the real artifacts; receipt assertions alone cannot mark success."""
    pointer = run / "reports/index_build.json"
    smoke_pointer = run / "reports/index_smoke.json"
    if not pointer.exists():
        if smoke_pointer.exists():
            raise ValueError("Smoke receipt exists without a verified index receipt")
        return None, None, {}
    path, receipt_sha, indexed = receipt(run, pointer)
    stage = run / "stage"
    if (path.name != "result.json" or path.parent.parent != (run / "reports/index_builds").resolve()
            or indexed["schema_version"] != 1 or indexed["run_id"] != run.name
            or indexed["status"] != "staged_index_verified_not_published"
            or indexed["index_verified"] is not True or indexed["audit_errors"] != []
            or indexed["protected_files_unchanged"] is not True
            or indexed["published"] is not False or (run / "published").exists()
            or indexed["formal_retrieval_evaluation_complete"] is not False
            or any(indexed[key] != 0 for key in ("generated_answers", "new_human_labels", "hosted_model_calls"))):
        raise ValueError("Unexpected index receipt state")
    stage_hashes = {name: digest(relative_file(stage, name).read_bytes()) for name in STAGE_FILES}
    if indexed["stage_hashes"] != stage_hashes or indexed["implementation_hashes"] != corpus["implementation_hashes"]:
        raise ValueError("Index receipt refers to changed stage or implementation")
    verify(ROOT / "scripts/build_staged_index.py", indexed["orchestrator_sha256"])
    manifest_path = stage / "index/manifest.json"
    vector_path = stage / "index/vectors.npy"
    verify(manifest_path, indexed["manifest_sha256"])
    verify(vector_path, indexed["vectors_sha256"])
    manifest = load(manifest_path)
    with (ROOT / "config.toml").open("rb") as stream:
        expected_model = tomllib.load(stream)["retrieval"]["embedding_model"]
    rows = read_jsonl(stage / "passages.jsonl")
    ids = [row["passage_id"] for row in rows]
    vectors = np.load(vector_path, allow_pickle=False, mmap_mode="r")
    dimension = manifest["dimension"]
    if (type(dimension) is not int or dimension <= 0 or not rows or len(set(ids)) != len(ids)
            or manifest["passage_ids"] != ids or len(rows) != corpus["passages"]
            or indexed["passages"] != len(rows) or indexed["dimension"] != dimension
            or manifest["model"] != expected_model or indexed["model"] != expected_model
            or manifest["corpus_sha256"] != corpus["corpus_sha256"]
            or indexed["corpus_sha256"] != corpus["corpus_sha256"]
            or manifest["vectors_sha256"] != indexed["vectors_sha256"]
            or vectors.shape != (len(rows), dimension) or indexed["vectors_shape"] != list(vectors.shape)
            or vectors.dtype != np.float32 or indexed["vectors_dtype"] != "float32"
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError("Index corpus/model/hash/order/shape/finite/norm mismatch")
    if indexed["local_encoder_model_hashes"] != manifest["onnx_sha256"]:
        raise ValueError("Index local model hash inventory differs")
    verify_model_mapping(manifest["onnx_sha256"])
    if set(indexed["audit_reports"]) != {"source_audit.json", "indexed_audit.json"}:
        raise ValueError("Index audit inventory differs")
    for name, binding in indexed["audit_reports"].items():
        audit_path = relative_file(run, binding["path"])
        if audit_path.parent != path.parent or audit_path.name != name:
            raise ValueError("Index audits must belong to the immutable build attempt")
        verify(audit_path, binding["sha256"])
        audit = load(audit_path)
        if (audit["errors"] != [] or audit["corpus_sha256"] != corpus["corpus_sha256"]
                or audit["stage_hashes"] != stage_hashes
                or audit["implementation_hashes"] != corpus["implementation_hashes"]
                or audit["active_passages"] != len(rows)
                or audit["index_verified"] is not (name == "indexed_audit.json")):
            raise ValueError("Index audit is stale or failed")
    protected = indexed["protected_sha256"]
    if not isinstance(protected, dict) or not protected:
        raise ValueError("Missing protected-input receipt")
    for name, expected in protected.items():
        protected_path = Path(name).resolve()
        if not Path(name).is_absolute() or not protected_path.is_relative_to((ROOT / "data").resolve()):
            raise ValueError("Protected input escapes project data")
        verify(protected_path, expected)
    required_protected = [stage / name for name in STAGE_FILES] + [
        stage / "reports/unified_quality.json", run / "reports/completion_checkpoint.json",
        ROOT / "data/passages.jsonl", ROOT / "data/index/manifest.json", ROOT / "data/index/vectors.npy",
        ROOT / "data/annotations.sqlite3", ROOT / "data/model_usage.sqlite3"]
    if not {str(item.resolve()) for item in required_protected}.issubset(protected):
        raise ValueError("Protected-input inventory omits required live or frozen files")
    aggregate = {key: indexed[key] for key in (
        "status", "passages", "model", "dimension", "vectors_shape", "vectors_dtype", "elapsed_seconds",
        "corpus_sha256", "vectors_sha256", "manifest_sha256", "index_verified", "protected_files_unchanged")}
    aggregate["scope"] = "Unpublished staged index; integrity is not retrieval accuracy or a release gate."
    bindings = {"index_build": receipt_sha, "indexed_source_audit": indexed["audit_reports"]["indexed_audit.json"]["sha256"]}
    smoke = None
    if smoke_pointer.exists():
        smoke_path, smoke_sha, checked = receipt(run, smoke_pointer)
        verify(ROOT / "scripts/check_staged_index.py", checked["checker_sha256"])
        if (smoke_path.parent != path.parent or checked["schema_version"] != 1
                or checked["run_id"] != run.name or checked["status"] != "operational_smoke_passed_not_formal_evaluation"
                or checked["errors"] != [] or checked["checks"] != 24 or len(checked["results"]) != 24
                or checked["index_receipt_sha256"] != receipt_sha
                or checked["corpus_sha256"] != indexed["corpus_sha256"]
                or checked["manifest_sha256"] != indexed["manifest_sha256"]
                or checked["vectors_sha256"] != indexed["vectors_sha256"]
                or checked["passages"] != len(rows) or checked["bm25_documents"] != len(rows)
                or checked["modes"] != ["bm25", "dense", "hybrid"]
                or any(checked[key] is not True for key in (
                    "exact_offsets_verified", "date_and_region_filters_verified", "inputs_unchanged"))
                or checked["human_gold"] is not False or checked["published"] is not False
                or any(checked[key] != 0 for key in ("generated_answers", "hosted_model_calls", "new_human_labels"))):
            raise ValueError("Smoke receipt is stale or failed")
        by_id = {row["passage_id"]: row for row in rows}
        pages = {row["source_page_id"]: row for row in read_jsonl(stage / "source_pages.jsonl")}
        mode_counts = {mode: 0 for mode in checked["modes"]}
        for record in checked["results"]:
            if record["mode"] not in mode_counts or not record["question"].strip():
                raise ValueError("Smoke query or mode is invalid")
            mode_counts[record["mode"]] += 1
            date.fromisoformat(record["as_of"])
            evidence = record["evidence"]
            if (type(record["expect_empty"]) is not bool or bool(evidence) == record["expect_empty"]
                    or len(evidence) > 5 or len({hit["passage_id"] for hit in evidence}) != len(evidence)):
                raise ValueError("Smoke evidence count differs")
            for rank, hit in enumerate(evidence, 1):
                original = by_id[hit["passage_id"]]
                if (any(hit[key] != value for key, value in original.items()) or hit["rank"] != rank
                        or not math.isfinite(hit["score"]) or not hit.get("available_by")
                        or hit["available_by"] > record["as_of"]
                        or (record["region"] and record["region"] not in hit["regions"])
                        or pages[hit["source_page_id"]]["text"][hit["char_start"]:hit["char_end"]] != hit["text"]):
                    raise ValueError("Smoke provenance/rank/filter/source offset differs")
        if set(mode_counts.values()) != {8}:
            raise ValueError("Smoke mode coverage differs")
        smoke = {key: checked[key] for key in (
            "status", "checks", "errors", "passages", "bm25_documents", "modes", "exact_offsets_verified",
            "date_and_region_filters_verified", "inputs_unchanged", "human_gold")}
        smoke["scope"] = "Operational filtering and provenance checks only; no answer accuracy or formal evaluation."
        bindings["index_smoke"] = smoke_sha
    return aggregate, smoke, bindings


def export(output=None):
    run = ROOT / "data/unified_runs/semantic-page-v2-buffer"
    checkpoint_path = run / "reports/completion_checkpoint.json"
    checkpoint = load(checkpoint_path)
    stage = run / "stage"
    corpus = load(stage / "reports/corpus.json")
    verify(stage / "passages.jsonl", checkpoint["corpus"]["sha256"])
    verify(stage / "source_pages.jsonl", checkpoint["corpus"]["source_pages_sha256"])
    verify_mapping(ROOT, corpus["implementation_hashes"], IMPLEMENTATION_FILES)
    receipts = {}
    for field in ("tests", "reconstruction_audit"):
        entry = checkpoint[field]
        path = relative_file(run, entry["report"])
        verify(path, entry["sha256"])
        receipts[field] = load(path)
    for label in ("legacy", "new"):
        entry = checkpoint["source_qa"]
        path = relative_file(run, entry[f"{label}_final_report"])
        verify(path, entry[f"{label}_sha256"])
        qa = load(path)
        if qa["remaining_critical_count"] != 0:
            raise ValueError("Unresolved fixed-sample source critical")
        verify_mapping(stage, qa["input_hashes"], STAGE_FILES)
        if qa["implementation_hashes"] != corpus["implementation_hashes"]:
            raise ValueError("Source QA implementation binding differs")
        verify_mapping(ROOT, qa["implementation_hashes"], IMPLEMENTATION_FILES)
    for report in receipts.values():
        if report["corpus_sha256"] != checkpoint["corpus"]["sha256"]:
            raise ValueError("Receipt refers to another corpus")
    audit = receipts["reconstruction_audit"]
    if (audit["errors"] != [] or audit["index_verified"] is not False
            or checkpoint["release"]["published"] is not False
            or checkpoint["release"]["new_index_created"] is not False):
        raise ValueError("Unexpected audit or release state")
    verify_mapping(stage, audit["stage_hashes"], STAGE_FILES)
    if audit["implementation_hashes"] != corpus["implementation_hashes"]:
        raise ValueError("Historical source audit implementation binding differs")
    suite = receipts["tests"]
    junit_path = relative_file(run, suite["junit_xml"])
    verify(junit_path, suite["junit_sha256"])
    suites = list(ET.parse(junit_path).getroot().iter("testsuite"))
    totals = {key: sum(int(item.get(key, "0")) for item in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    passed = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    if (passed != suite["passed"] or totals["skipped"] != suite["skipped"]
            or totals["failures"] + totals["errors"] != suite["failed"]):
        raise ValueError("Test receipt and JUnit counts differ")
    for key in ("passages", "words", "unique_words", "contributing_documents", "cleaned_source_pages"):
        field = {"contributing_documents": "documents", "cleaned_source_pages": "source_pages"}.get(key, key)
        if checkpoint["corpus"][key] != corpus[field]:
            raise ValueError("Checkpoint and corpus statistics differ")
    current_index, smoke, index_bindings = verified_index(run, corpus)
    summary = {
        "schema_version": 1,
        "exported_utc": now(),
        "status": "source_buffer_closeout_complete_new_corpus_staged_not_published",
        "publication_scope": "Code, documentation and whitelisted aggregate metrics only; original data and labels excluded.",
        "corpus": {key: checkpoint["corpus"][key] for key in (
            "passages", "words", "unique_words", "contributing_documents", "cleaned_source_pages",
            "minimum_passages", "passage_buffer", "passage_buffer_percent", "inherited_labels",
            "sha256", "source_pages_sha256")},
        "segmentation": {key: checkpoint["segmentation"][key] for key in (
            "target_words", "soft_minimum_words", "soft_maximum_words", "maximum_context_words", "parameters_unchanged")},
        "source_holds": {key: checkpoint["source_holds"][key] for key in (
            "physical_pages", "full_appendices", "trailing_reference_pages", "unverified_tail_page",
            "mixed_start_page_body_loss_disclosed", "originals_retained", "repaired_report_final_passages")},
        "tests": {key: suite[key] for key in ("passed", "skipped", "failed", "elapsed_seconds", "skip_reason")},
        "tests_origin": "Recorded local run, not GitHub Actions",
        "source_reconstruction": {"errors": audit["errors"], "byte_rebuilt_artifact_classes": 4,
                                  "implementation_files": len(corpus["implementation_hashes"]),
                                  "index_verified": audit["index_verified"],
                                  "scope": "Historical source-closeout reconstruction audit; current index is reported separately."},
        "machine_source_qa": {key: checkpoint["source_qa"][key] for key in (
            "fixed_cases", "additional_legacy_diagnostics", "total_original_cases_preserved",
            "new_original_critical_cases_excluded_not_passed", "warnings_retained",
            "remaining_known_sample_criticals", "human_gold", "whole_corpus_approved")},
        "default_corpus": {"passages": checkpoint["protection"]["default_corpus_passages"],
                           "archived_original_paragraph_machine_labels": 4500,
                           "existing_machine_qa_records": 100, "existing_machine_reviews": 728},
        "release": {key: checkpoint["release"][key] for key in (
            "new_index_created", "published", "evaluation_set_frozen", "formal_assignment_complete")},
        "new_tracked_paid_api_calls": checkpoint["protection"]["new_tracked_paid_model_api_calls"],
        "new_human_labels": checkpoint["protection"]["new_human_labels"],
        "codex_subscription_credit_usage": "Not measured",
        "local_receipt_sha256": {
            "completion_checkpoint": digest(checkpoint_path.read_bytes()),
            "tests": checkpoint["tests"]["sha256"],
            "reconstruction_audit": checkpoint["reconstruction_audit"]["sha256"],
            "legacy_fixed_review": checkpoint["source_qa"]["legacy_sha256"],
            "new_fixed_review": checkpoint["source_qa"]["new_sha256"], **index_bindings},
        "limitations": ["No raw documents, corpus, annotation records, databases or indexes are shipped.",
                        "Historical reconstruction requires the excluded matching local snapshots.",
                        "Machine source checks are neither human gold labels nor whole-corpus approval.",
                        "Course approval, held-out evaluation, formal retrieval comparison and publication remain pending."]}
    if current_index is not None:
        summary["status"] = "source_buffer_closeout_complete_new_index_verified_staged_not_published"
        summary["current_index"] = current_index
        summary["release"]["new_index_created"] = True
        summary["release"]["published"] = False
        summary["limitations"].append("Index integrity and optional operational smoke do not establish retrieval accuracy or resolve the prior failed comparison.")
        if smoke is not None:
            summary["retrieval_smoke"] = smoke
    destination = Path(output) if output else ROOT / "reports/progress.json"
    destination = destination.resolve()
    if not destination.is_relative_to((ROOT / "reports").resolve()):
        raise ValueError("Public export must stay inside the project reports directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return destination, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    path, result = export(args.output)
    print(json.dumps({"output": path.relative_to(ROOT).as_posix(),
                      "passages": result["corpus"]["passages"],
                      "published": result["release"]["published"]}, ensure_ascii=False))
