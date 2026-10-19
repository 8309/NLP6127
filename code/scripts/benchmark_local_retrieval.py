"""Bounded local development-only retrieval concurrency diagnostic.

Each measured query computes its own cached-local query embedding. A lock guards
the shared encoder because this script does not assume wrapper thread safety.
CPU ranking can overlap; the serialization policy is explicit. This is neither
a HTTP/service benchmark nor generative throughput, and invokes no LLM.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import math
import os
from pathlib import Path
import platform
from statistics import mean, median
import threading
from time import perf_counter
from uuid import uuid4

import retrieve_machine_questions as retrieval_inputs
from oil_evidence import machine_evaluation as protocol
from oil_evidence.codex_runtime import _assert_hashes, _validate_hits, load_staged_runtime
from oil_evidence.store import now


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(protocol.canonical_bytes(value) + b"\n")


class GuardedEncoder:
    def __init__(self, encoder):
        self.encoder, self.lock = encoder, threading.Lock()

    def query_embed(self, texts):
        with self.lock:
            result = list(self.encoder.query_embed(texts))
        yield from result


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(fraction * len(values))-1)]


def benchmark(runtime, questions, concurrency=(1,2,4)):
    if (not questions or len(questions) > 50 or len({q["record_id"] for q in questions}) != len(questions)
            or any(q.get("split") != "development" for q in questions)):
        raise ValueError("Use one to 50 unique fixed DEVELOPMENT questions, never held-out TEST")
    if tuple(concurrency) != (1,2,4):
        raise ValueError("This bounded diagnostic fixes local worker counts at one, two and four")
    original_encoder = runtime.retriever.encoder
    if original_encoder is None:
        raise ValueError("Verified cached-local encoder must already be initialized")
    runtime.retriever.encoder = GuardedEncoder(original_encoder)
    results, reference_rankings = [], {}
    try:
        for workers in concurrency:
            warmup = questions[0]
            warm_hits = runtime.retriever.search(warmup["question"], mode="hybrid", top_k=5,
                as_of=warmup["as_of"], region=warmup.get("region"), temporal=True)
            _validate_hits(warm_hits, runtime, warmup["as_of"], True)
            started_utc, started = now(), perf_counter()
            def execute(question, submitted):
                tick = perf_counter()
                evidence = runtime.retriever.search(question["question"], mode="hybrid", top_k=5,
                    as_of=question["as_of"], region=question.get("region"), temporal=True)
                _validate_hits(evidence, runtime, question["as_of"], True)
                if question.get("region") and any(question["region"] not in h.get("regions", []) for h in evidence):
                    raise ValueError("Retrieved hit violates query region")
                completed = perf_counter()
                return dict(record_id=question["record_id"], worker_retrieval_seconds=completed-tick,
                    executor_queue_seconds=tick-submitted, fixed_burst_response_seconds=completed-submitted,
                    retrieved_ids=[h["passage_id"] for h in evidence])
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(execute, question, perf_counter()) for question in questions]
                rows = [future.result() for future in futures]
            elapsed, completed_utc = perf_counter()-started, now()
            actual = {row["record_id"]:row["retrieved_ids"] for row in rows}
            if reference_rankings and actual != reference_rankings:
                raise ValueError("Ranked evidence changed between concurrency conditions")
            reference_rankings = actual
            worker, response = ([row[field] for row in rows] for field in
                ("worker_retrieval_seconds", "fixed_burst_response_seconds"))
            results.append(dict(local_workers=workers, queries=len(rows), started_utc=started_utc,
                completed_utc=completed_utc, measured_cohort_wall_seconds=elapsed,
                measured_completed_queries_per_wall_second=len(rows)/elapsed,
                worker_retrieval_median_seconds=median(worker), worker_retrieval_p95_seconds=percentile(worker,.95),
                fixed_burst_response_median_seconds=median(response), fixed_burst_response_p95_seconds=percentile(response,.95),
                mean_executor_queue_seconds=mean(row["executor_queue_seconds"] for row in rows), per_query=rows))
    finally:
        runtime.retriever.encoder = original_encoder
    return results


def run(runtime, cohort_path, limit=24):
    if type(limit) is not int or not 4 <= limit <= 50:
        raise ValueError("Choose a bounded four-to-50 development query count")
    questions, protected, receipt = retrieval_inputs.load_cohort(runtime.canonical_root, cohort_path)
    selected = [row for row in questions if row["split"] == "development"][:limit]
    if len(selected) != limit:
        raise ValueError("Requested fixed development coverage is unavailable")
    protected.update(runtime.protected_sha256)
    source_bytes = {Path(__file__).name:Path(__file__).read_bytes(),
        Path(retrieval_inputs.__file__).name:Path(retrieval_inputs.__file__).read_bytes()}
    _assert_hashes(protected)
    output = runtime.canonical_root / "data/machine_runs/local-retrieval-benchmarks" / (now().replace(":", "-") + "-" + uuid4().hex[:8])
    write_new(output / "inputs.json", dict(split="development", fixed_record_ids=[q["record_id"] for q in selected],
        cohort=str(Path(cohort_path).resolve()), cohort_receipt_sha256=protocol.sha256((Path(cohort_path) / "receipt.json").read_bytes()),
        corpus_sha256=runtime.corpus_sha256, protected_sha256=protected, mode="hybrid", temporal=True, top_k=5,
        initialization_excluded=True, cached_encoder=True, per_query_embedding_included=True,
        shared_encoder_calls_serialized=True, per_condition_warmup_queries=1, concurrency=[1,2,4]))
    started_utc, started = now(), perf_counter()
    try:
        results = benchmark(runtime, selected)
        _assert_hashes(protected)
        versions = {}
        for package in ("numpy", "fastembed", "onnxruntime", "rank-bm25"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        report = dict(status="actual_local_development_retrieval_concurrency_diagnostic", split="development",
            development_queries=limit, timed_queries=limit*3, native_model_calls=0, hosted_api_calls=0,
            started_utc=started_utc, completed_utc=now(), diagnostic_wall_seconds=perf_counter()-started,
            hardware=dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                logical_cpu_count=os.cpu_count(), configured_encoder_threads=4, python=platform.python_version()),
            library_versions=versions, corpus_sha256=runtime.corpus_sha256, corpus_passages=len(runtime.passages),
            source_sha256={name:protocol.sha256(raw) for name, raw in source_bytes.items()},
            results=results, original_inputs_unchanged=True,
            limitations=["Local shared-process retrieval only; no HTTP request, answer generation or concurrent service throughput is measured.",
                "Each query embeds independently, but shared encoder calls are serialized deliberately; ONNX wrapper thread safety is not assumed.",
                "Twenty-four-or-selected-count fixed requests are submitted as one burst; executor queue and worker retrieval times are separate.",
                "Warm initialized index/model and explicit local CPU settings; not isolated production load or general hardware scaling.",
                "Sequential condition order and background activity can affect this bounded diagnostic; no significance claim."])
        for name, raw in source_bytes.items():
            path = output / "implementation" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as target:
                target.write(raw)
        write_new(output / "result.json", report)
        return output, report
    except Exception as error:
        write_new(output / "failure.json", dict(error_type=type(error).__name__, error=str(error), complete=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--question-cohort", type=Path, required=True)
    parser.add_argument("--corpus-run", default="semantic-page-v2-buffer")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    runtime = load_staged_runtime(args.canonical_root, args.corpus_run)
    output, report = run(runtime, args.question_cohort, args.limit)
    print(protocol.canonical_bytes(dict(output=str(output), summary=[{key:value for key,value in row.items()
        if key != "per_query"} for row in report["results"]], hardware=report["hardware"])).decode())
