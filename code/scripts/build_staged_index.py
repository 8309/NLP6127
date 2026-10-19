"""Build a staged index without overwriting the frozen source-closeout audit.

The unchanged unified pipeline performs both full source reconstruction audits.
Only its audit report destination is redirected to a new attempt directory.
No corpus publication, source collection, annotation or hosted model is invoked.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

import unify_corpus as pipeline
from oil_evidence.store import digest, now, write_json


def file_hashes(paths):
    return {str(path.resolve()): digest(path.read_bytes()) for path in paths if path.is_file()}


def assert_unchanged(expected):
    for name, sha in expected.items():
        path = Path(name)
        if not path.is_file() or digest(path.read_bytes()) != sha:
            raise ValueError("Protected input changed: " + name)


@contextmanager
def audit_destination(module, destination):
    """Redirect just the audit receipt, restoring the writer even on failure."""
    original = module.write_json
    source = (module.RUN / "stage/reports/unified_quality.json").resolve()

    def routed(path, value):
        original(destination if Path(path).resolve() == source else path, value)

    module.write_json = routed
    try:
        yield
    finally:
        module.write_json = original


def build(run_id):
    run = pipeline.select_run(run_id)
    stage = run / "stage"
    if (run / "published").exists():
        raise ValueError("Published run is immutable; use a new run ID")
    if not (stage / "passages.jsonl").is_file():
        raise ValueError("Stage the corpus before building its index")
    started = perf_counter()
    with pipeline.lock():
        # These are snapshots of existing protected files, never new baselines.
        protected_paths = [pipeline.DATA / name for name in pipeline.INPUTS]
        protected_paths += [pipeline.DATA / name for name in (
            "annotations.sqlite3", "model_usage.sqlite3", "index/embedding_cache.sqlite3")]
        protected_paths += [stage / name for name in pipeline.stage_hashes()]
        protected_paths += [stage / "reports/unified_quality.json"]
        protected_paths += [run / "reports" / name for name in (
            "completion_checkpoint.json", "live_closeout_protection.json", "full_suite-final.json",
            "buffer_source_review-closeout-final.json", "boem-goa-source-check-final.json")]
        protected = file_hashes(protected_paths)
        implementation = pipeline.hashes()
        stage_before = pipeline.stage_hashes()
        attempt = run / "reports/index_builds" / (now().replace(":", "-") + "-" + uuid4().hex[:8])
        attempt.mkdir(parents=True, exist_ok=False)
        write_json(attempt / "inputs.json", dict(
            started_utc=now(), run_id=run_id, stage_hashes=stage_before,
            implementation_hashes=implementation, protected_sha256=protected,
            orchestrator_sha256=digest(Path(__file__).read_bytes())))
        try:
            print("Checking source reconstruction before indexing...", flush=True)
            with audit_destination(pipeline, attempt / "source_audit.json"):
                result = pipeline.index()
            print("Index built; checking reconstruction and index integrity...", flush=True)
            with audit_destination(pipeline, attempt / "indexed_audit.json"):
                audited = pipeline.audit(require_index=True)
            manifest = pipeline.check_index(stage)
            vectors = np.load(stage / "index/vectors.npy", allow_pickle=False)
            if vectors.dtype != np.float32:
                raise ValueError("Expected float32 index vectors")
            if pipeline.stage_hashes() != stage_before or pipeline.hashes() != implementation:
                raise ValueError("Stage or implementation changed while indexing")
            assert_unchanged(protected)
            if (run / "published").exists():
                raise ValueError("Unexpected publication while indexing")
            receipt = dict(
                schema_version=1, status="staged_index_verified_not_published", completed_utc=now(),
                run_id=run_id, elapsed_seconds=round(perf_counter() - started, 3),
                corpus_sha256=manifest["corpus_sha256"], stage_hashes=stage_before,
                implementation_hashes=implementation, orchestrator_sha256=digest(Path(__file__).read_bytes()),
                passages=result["passages"], model=manifest["model"], dimension=manifest["dimension"],
                vectors_shape=list(vectors.shape), vectors_dtype=str(vectors.dtype),
                vectors_sha256=manifest["vectors_sha256"],
                manifest_sha256=digest((stage / "index/manifest.json").read_bytes()),
                local_encoder_model_hashes=manifest["onnx_sha256"],
                audit_reports={name: dict(path=(attempt / name).relative_to(run).as_posix(),
                                         sha256=digest((attempt / name).read_bytes()))
                               for name in ("source_audit.json", "indexed_audit.json")},
                index_verified=audited["index_verified"], audit_errors=audited["errors"],
                protected_sha256=protected, protected_files_unchanged=True,
                published=False, generated_answers=0, new_human_labels=0, hosted_model_calls=0,
                codex_subscription_credit_usage="Not measured",
                formal_retrieval_evaluation_complete=False)
            write_json(attempt / "result.json", receipt)
            # Latest success points to an immutable attempt; failed attempts never replace it.
            write_json(run / "reports/index_build.json", dict(
                path=(attempt / "result.json").relative_to(run).as_posix(),
                sha256=digest((attempt / "result.json").read_bytes())))
            return receipt
        except Exception as error:
            write_json(attempt / "failure.json", dict(
                failed_utc=now(), error_type=type(error).__name__, error=str(error),
                successful=False, published=False))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--offline", action="store_true", help="Use cached model files only")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    receipt = build(args.run)
    print(json.dumps({key: receipt[key] for key in (
        "status", "passages", "model", "dimension", "vectors_shape", "elapsed_seconds",
        "index_verified", "protected_files_unchanged", "published")}, ensure_ascii=False, indent=2))
