"""Run frozen, unlabelled development queries against a read-only staged index.

The deterministic evidence pool is blinded to retrieval mode, rank and question
authoring hints. Rankings and operational diagnostics remain separate. This is
not a retrieval accuracy evaluation and it creates no answer or human labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import date
import json
import os
from pathlib import Path
import re
from statistics import mean, median
from time import perf_counter
from uuid import uuid4

import numpy as np

import unify_corpus as pipeline
from build_staged_index import assert_unchanged, file_hashes
from check_staged_index import run_file
from oil_evidence.retrieval import Retriever
from oil_evidence.store import ROOT, digest, now, read_jsonl, write_json, write_jsonl

MODES = ("bm25", "dense", "hybrid")
QUERY_FIELDS = {"question_id", "question", "question_zh", "as_of"}
EVIDENCE_FIELDS = {
    "passage_id", "document_id", "source_page_id", "title", "source", "url",
    "published_date", "available_by", "date_precision", "date_source",
    "event_date", "event_status", "page", "text", "regions", "word_count",
    "char_start", "char_end", "source_block_attribution",
}
BLINDING_VERSION = "unranked-evidence-pool-v1"


def safe_id(value):
    if (not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,80}", value)
            or re.fullmatch(r"con|prn|aux|nul|com[1-9]|lpt[1-9]", value)):
        raise ValueError("Invalid run ID")
    return value


@contextmanager
def canonical_context(root, corpus_run):
    """Point read-only audit helpers at original receipts, never rebind files."""
    previous = pipeline.PROJECT, pipeline.DATA, pipeline.RUN
    pipeline.PROJECT = root
    pipeline.DATA = root / "data"
    pipeline.RUN = pipeline.DATA / "unified_runs" / safe_id(corpus_run)
    try:
        yield pipeline.RUN
    finally:
        pipeline.PROJECT, pipeline.DATA, pipeline.RUN = previous


def checked_file(base, name, expected):
    path = run_file(base, name)
    if digest(path.read_bytes()) != expected:
        raise ValueError("Input receipt or artifact hash differs")
    return path


def validate_queries(queries, expected_count=50):
    if len(queries) != expected_count:
        raise ValueError("Unexpected frozen development query count")
    seen = set()
    for query in queries:
        if set(query) != QUERY_FIELDS:
            raise ValueError("Annotation query contains labels or authoring hints")
        qid = query["question_id"]
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("Duplicate or missing question ID")
        seen.add(qid)
        for key in ("question", "question_zh"):
            if not isinstance(query[key], str) or not query[key].strip():
                raise ValueError("Empty frozen question")
        cutoff = query["as_of"]
        if (not isinstance(cutoff, str) or date.fromisoformat(cutoff).isoformat() != cutoff
                or not "2018-01-01" <= cutoff <= "2026-10-05"):
            raise ValueError("Invalid frozen query cutoff")
    return queries


def validate_hits(hits, query, passages, pages, top_k=10, region=None):
    if len(hits) > top_k or len({h["passage_id"] for h in hits}) != len(hits):
        raise ValueError("Retrieval result count or duplicate ID mismatch")
    for rank, hit in enumerate(hits, 1):
        original = passages.get(hit["passage_id"])
        if (original is None or any(hit.get(k) != value for k, value in original.items())
                or hit.get("rank") != rank or not np.isfinite(hit.get("score", np.nan))
                or not hit.get("available_by") or hit["available_by"] > query["as_of"]
                or (region and region not in hit.get("regions", []))):
            raise ValueError("Retrieval provenance, ranking or filter mismatch")
        source = pages.get(hit["source_page_id"])
        start, end = hit["char_start"], hit["char_end"]
        if (source is None or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(source["text"])
                or source["text"][start:end] != hit["text"]):
            raise ValueError("Exact source-page offset mismatch")


def blinded_pool(query, rankings, passages):
    """One union pool per query; no relevance inference or anchor injection."""
    if set(query) != QUERY_FIELDS or set(rankings) != set(MODES):
        raise ValueError("Pool must contain only four query fields and all three modes")
    ids = {hit["passage_id"] for hits in rankings.values() for hit in hits}
    ordered = sorted(ids, key=lambda pid: (digest(BLINDING_VERSION + "\0" + query["question_id"] + "\0" + pid), pid))
    evidence = []
    for pid in ordered:
        original = passages[pid]
        evidence.append({key: original[key] for key in sorted(EVIDENCE_FIELDS) if key in original})
    return dict(query, evidence=evidence)


def latency_summary(records):
    summary = {}
    for mode in MODES:
        durations = [r["elapsed_seconds"] for r in records if r["mode"] == mode]
        if not durations:
            raise ValueError("Missing retrieval mode latency records")
        total = sum(durations)
        summary[mode] = dict(queries=len(durations), mean_seconds=mean(durations),
                             median_seconds=median(durations), p95_seconds=float(np.percentile(durations, 95)),
                             total_seconds=total, sequential_queries_per_second=len(durations) / total if total else None)
    return summary


def load_verified_inputs(root, development_run, corpus_run):
    development = root / "data/development_runs" / safe_id(development_run)
    current_path = development / "current.json"
    current = pipeline.load(current_path)
    question_manifest_path = checked_file(development, current["path"], current["sha256"])
    question_manifest = pipeline.load(question_manifest_path)
    frozen = question_manifest_path.parent
    artifacts = {
        str(checked_file(frozen, name, expected)): expected
        for name, expected in question_manifest["artifact_sha256"].items()
    }
    if (question_manifest.get("human_gold") is not False
            or question_manifest.get("question_selection_before_retrieval") is not True
            or question_manifest.get("qa_annotations_created") != 0):
        raise ValueError("Expected frozen unlabelled source-first development queries")
    queries = validate_queries(read_jsonl(frozen / "annotation_queries.jsonl"))
    with canonical_context(root, corpus_run) as run:
        stage = run / "stage"
        pointer_path = run / "reports/index_build.json"
        pointer = pipeline.load(pointer_path)
        receipt_path = checked_file(run, pointer["path"], pointer["sha256"])
        receipt = pipeline.load(receipt_path)
        manifest = pipeline.check_index(stage)
        if (receipt["corpus_sha256"] != manifest["corpus_sha256"]
                or question_manifest["corpus_sha256"] != manifest["corpus_sha256"]
                or receipt["manifest_sha256"] != digest((stage / "index/manifest.json").read_bytes())
                or receipt["vectors_sha256"] != manifest["vectors_sha256"]
                or receipt["passages"] != len(manifest["passage_ids"])
                or receipt["dimension"] != manifest["dimension"]
                or receipt["model"] != manifest["model"]
                or receipt["vectors_shape"] != [receipt["passages"], manifest["dimension"]]
                or receipt["vectors_dtype"] != "float32"
                or receipt["stage_hashes"] != pipeline.stage_hashes()
                or receipt["implementation_hashes"] != pipeline.hashes()
                or receipt["index_verified"] is not True or receipt["audit_errors"]
                or receipt["published"] is not False or receipt["run_id"] != corpus_run):
            raise ValueError("Verified staged-index receipt binding differs")
        audit_hashes = {str(checked_file(run, entry["path"], entry["sha256"])): entry["sha256"]
                        for entry in receipt["audit_reports"].values()}
        # Historical receipts certify the canonical implementation, not translated UI files.
        canonical_code = {str(root / name): expected for name, expected in receipt["implementation_hashes"].items()}
        model_hashes = {}
        for name, expected in receipt["local_encoder_model_hashes"].items():
            model_hashes[str(checked_file(root / "models", name, expected))] = expected
        protected = {**question_manifest["input_sha256"], **receipt["protected_sha256"],
                     **canonical_code, **artifacts, **audit_hashes, **model_hashes,
                     **file_hashes([current_path, question_manifest_path, pointer_path, receipt_path,
                                    stage / "index/manifest.json", stage / "index/vectors.npy"])}
        assert_unchanged(protected)
        # The public retrieval implementation itself must be the original verified core.
        for name in ("src/oil_evidence/retrieval.py", "src/oil_evidence/store.py", "config.toml"):
            if digest((ROOT / name).read_bytes()) != receipt["implementation_hashes"][name]:
                raise ValueError("Public retrieval, store or settings differ from verified core")
        rows = read_jsonl(stage / "passages.jsonl")
        pages = {page["source_page_id"]: page for page in read_jsonl(stage / "source_pages.jsonl")}
        vectors = np.load(stage / "index/vectors.npy", allow_pickle=False)
        if vectors.dtype != np.float32 or len({r["passage_id"] for r in rows}) != len(rows):
            raise ValueError("Index dtype or passage ID mismatch")
    return development, queries, rows, pages, vectors, manifest, protected, dict(
        question_manifest_sha256=current["sha256"], questions_sha256=question_manifest["questions_sha256"],
        annotation_queries_sha256=question_manifest["artifact_sha256"]["annotation_queries.jsonl"],
        index_receipt_sha256=pointer["sha256"], manifest_sha256=receipt["manifest_sha256"],
        corpus_sha256=manifest["corpus_sha256"], vectors_sha256=manifest["vectors_sha256"],
        canonical_protected_input_count=len(question_manifest["input_sha256"]),
        canonical_protected_code_count=len(canonical_code))


def run(root, development_run="oil-dev50-v1", corpus_run="semantic-page-v2-buffer", top_k=10):
    root = Path(root).resolve()
    if not (root / "data").is_dir() or not (root / "models").is_dir():
        raise ValueError("Canonical data and cached model directories are required")
    if type(top_k) is not int or top_k != 10:
        raise ValueError("This locked development protocol requires top_k=10")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = perf_counter()
    (development, queries, rows, pages, vectors, manifest, protected, bindings) = load_verified_inputs(
        root, development_run, corpus_run)
    inputs_verified_seconds = perf_counter() - started
    output_parent = development / "retrieval"
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / (now().replace(":", "-") + "-" + uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    implementation = file_hashes([Path(__file__), ROOT / "src/oil_evidence/retrieval.py", ROOT / "src/oil_evidence/store.py", ROOT / "config.toml"])
    protected.update(implementation)
    write_json(output / "inputs.json", dict(schema_version=1, started_utc=now(),
               protocol="development-top10-three-mode-pool-v1", bindings=bindings,
               protected_sha256=protected, current_execution_implementation_sha256=implementation,
               original_receipts_are_not_rebound=True))
    try:
        from fastembed import TextEmbedding
        init_started = perf_counter()
        encoder = TextEmbedding(model_name=manifest["model"], cache_dir=str(root / "models"),
                                threads=4, local_files_only=True)
        retriever = Retriever(passages=rows, vectors=vectors, encoder=encoder)
        init_seconds = perf_counter() - init_started
        by_id = {row["passage_id"]: row for row in rows}
        warmup, records, pooled = [], [], []
        for mode in MODES:
            tick = perf_counter()
            hits = retriever.search(queries[0]["question"], mode=mode, top_k=top_k,
                                    as_of=queries[0]["as_of"], region=None, temporal=True)
            seconds = perf_counter() - tick
            validate_hits(hits, queries[0], by_id, pages, top_k)
            warmup.append(dict(mode=mode, elapsed_seconds=seconds, hits=len(hits)))
        for query in queries:
            rankings = {}
            for mode in MODES:
                tick = perf_counter()
                hits = retriever.search(query["question"], mode=mode, top_k=top_k,
                                        as_of=query["as_of"], region=None, temporal=True)
                seconds = perf_counter() - tick
                validate_hits(hits, query, by_id, pages, top_k)
                rankings[mode] = hits
                records.append(dict(question_id=query["question_id"], question=query["question"],
                                    as_of=query["as_of"], region=None, temporal=True, mode=mode,
                                    elapsed_seconds=seconds, evidence=hits))
                print(f"Development retrieval {len(records)}/150: {query['question_id']}, {mode}, hits={len(hits)}", flush=True)
            pooled.append(blinded_pool(query, rankings, by_id))
        # Additional known-empty and region controls are operational only, not new development questions.
        controls = []
        for cutoff, region, expect_empty in (("2017-12-31", None, True), ("2026-10-05", "Middle East", False)):
            query = dict(queries[0], as_of=cutoff)
            for mode in MODES:
                hits = retriever.search(query["question"], mode=mode, top_k=top_k,
                                        as_of=cutoff, region=region, temporal=True)
                validate_hits(hits, query, by_id, pages, top_k, region)
                if bool(hits) == expect_empty:
                    raise ValueError("Empty/region control failed")
                controls.append(dict(as_of=cutoff, region=region, mode=mode, hits=len(hits), passed=True))
        assert_unchanged(protected)
        write_jsonl(output / "rankings.jsonl", records)
        write_jsonl(output / "annotation_pool.jsonl", pooled)
        pool_counts = [len(row["evidence"]) for row in pooled]
        result = dict(schema_version=1, status="development_retrieval_operational_verified_not_gold_evaluation",
                      completed_utc=now(), development_run=development_run, corpus_run=corpus_run,
                      **bindings, top_k=top_k, modes=list(MODES), questions=len(queries),
                      retrieval_runs=len(records), passages=len(rows), index_dimension=manifest["dimension"],
                      embedding_model=manifest["model"], cached_encoder_only=True,
                      input_verification_seconds=inputs_verified_seconds, encoder_and_bm25_init_seconds=init_seconds,
                      warmup=warmup, latency=latency_summary(records), control_checks=controls,
                      empty_results_by_mode={mode: sum(not r["evidence"] for r in records if r["mode"] == mode) for mode in MODES},
                      pool=dict(blinding_version=BLINDING_VERSION, records=len(pooled),
                                total_query_evidence_pairs=sum(pool_counts), min_evidence=min(pool_counts),
                                max_evidence=max(pool_counts), mean_evidence=mean(pool_counts),
                                empty_query_ids=[p["question_id"] for p in pooled if not p["evidence"]],
                                rank_or_mode_exposed=False, authoring_hints_exposed=False,
                                union_sources=Counter(e["source"] for row in pooled for e in row["evidence"])),
                      artifacts_sha256={name: digest((output / name).read_bytes()) for name in
                                        ("inputs.json", "rankings.jsonl", "annotation_pool.jsonl")},
                      inputs_unchanged=True, exact_offsets_verified=True, date_and_region_checks_passed=True,
                      generated_answers=0, hosted_model_calls=0, new_human_labels=0, qa_annotations_created=0,
                      human_gold=False, formal_test_set_evaluated=False, corpus_published=False,
                      codex_subscription_credit_usage="Not measured",
                      limitations=["Operational integrity and latency are not precision, recall, MRR or answer accuracy.",
                                   "The pool only covers three modes at depth 10; unpooled relevant passages may exist.",
                                   "Question authoring anchors and scenario intentions are not qrels.",
                                   "Development questions are not an independent held-out test set.",
                                   "The English BGE encoder truncates text at its model context limit."])
        write_json(output / "result.json", result)
        assert_unchanged(protected)
        return output, result
    except Exception as error:
        write_json(output / "failure.json", dict(failed_utc=now(), successful=False,
                   error_type=type(error).__name__, error=str(error), original_inputs_not_overwritten=True))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--development-run", default="oil-dev50-v1")
    parser.add_argument("--corpus-run", default="semantic-page-v2-buffer")
    args = parser.parse_args()
    output, result = run(args.canonical_root, args.development_run, args.corpus_run)
    print(json.dumps(dict(output=str(output), **{key: result[key] for key in (
        "status", "questions", "retrieval_runs", "passages", "inputs_unchanged", "pool", "latency")}), indent=2))
