"""Run real local retrieval smoke checks on a verified, unpublished stage.

These checks establish operational integrity and filtering, not answer accuracy.
No answer generation, annotation, query logs or paid APIs are invoked.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

import unify_corpus as pipeline
from build_staged_index import assert_unchanged, file_hashes
from oil_evidence.retrieval import Retriever, embedder
from oil_evidence.store import ROOT, digest, now, read_jsonl, write_json


def run_file(run, name):
    path = (run / name).resolve()
    if not path.is_relative_to(run.resolve()) or not path.is_file():
        raise ValueError("Receipt path is outside run or missing")
    return path


def check(run_id):
    run = pipeline.select_run(run_id)
    stage = run / "stage"
    with pipeline.lock():
        pointer = pipeline.load(run / "reports/index_build.json")
        receipt_path = run_file(run, pointer["path"])
        if digest(receipt_path.read_bytes()) != pointer["sha256"]:
            raise ValueError("Index receipt hash differs")
        receipt = pipeline.load(receipt_path)
        manifest = pipeline.check_index(stage)
        if (receipt["corpus_sha256"] != manifest["corpus_sha256"]
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
                or receipt["published"] is not False or receipt["run_id"] != run_id):
            raise ValueError("Index receipt no longer matches the verified stage")
        for entry in receipt["audit_reports"].values():
            if digest(run_file(run, entry["path"]).read_bytes()) != entry["sha256"]:
                raise ValueError("Index audit receipt changed")
        for name, expected in receipt["local_encoder_model_hashes"].items():
            path = (ROOT / "models" / name).resolve()
            if (not path.is_relative_to((ROOT / "models").resolve())
                    or digest(path.read_bytes()) != expected):
                raise ValueError("Local encoder file differs from index build")
        assert_unchanged(receipt["protected_sha256"])
        protected = {**receipt["protected_sha256"], **file_hashes([
            stage / "index/manifest.json", stage / "index/vectors.npy",
            ROOT / "seeds/questions.json", receipt_path, run / "reports/index_build.json"])}
        rows = read_jsonl(stage / "passages.jsonl")
        by_id = {row["passage_id"]: row for row in rows}
        pages = {page["source_page_id"]: page for page in read_jsonl(stage / "source_pages.jsonl")}
        seeds = pipeline.load(ROOT / "seeds/questions.json")
        # Existing representative questions, not a new gold or development set.
        cases = [dict(question=seeds[i]["question"], as_of=cutoff, region=None, expect_empty=False)
                 for i in (1, 5, 13) for cutoff in ("2025-06-30", "2026-10-05")]
        cases += [dict(question=seeds[1]["question"], as_of="2017-12-31", region=None, expect_empty=True),
                  dict(question=seeds[5]["question"], as_of="2026-10-05", region="Middle East", expect_empty=False)]
        vectors = np.load(stage / "index/vectors.npy", allow_pickle=False)
        retriever = Retriever(passages=rows, vectors=vectors, encoder=embedder(manifest["model"]))
        records = []
        for case in cases:
            for mode in ("bm25", "dense", "hybrid"):
                started = perf_counter()
                hits = retriever.search(case["question"], mode=mode, top_k=5,
                                        as_of=case["as_of"], region=case["region"], temporal=True)
                seconds = perf_counter() - started
                if bool(hits) == case["expect_empty"]:
                    raise ValueError("Unexpected empty/nonempty retrieval result")
                if len({hit["passage_id"] for hit in hits}) != len(hits):
                    raise ValueError("Duplicate retrieval evidence ID")
                for rank, hit in enumerate(hits, 1):
                    original = by_id[hit["passage_id"]]
                    if (any(hit[key] != value for key, value in original.items())
                            or hit["rank"] != rank or not np.isfinite(hit["score"])
                            or not hit.get("available_by") or hit["available_by"] > case["as_of"]
                            or (case["region"] and case["region"] not in hit["regions"])
                            or pages[hit["source_page_id"]]["text"][hit["char_start"]:hit["char_end"]] != hit["text"]):
                        raise ValueError("Retrieval provenance/rank/filter/source offset mismatch")
                records.append(dict(case, mode=mode, elapsed_seconds=round(seconds, 6), evidence=hits))
                print(f"Checked {len(records)}/24: {mode}, cutoff={case['as_of']}, hits={len(hits)}", flush=True)
        assert_unchanged(protected)
        if (run / "published").exists():
            raise ValueError("Unexpected publication while checking the staged index")
        result = dict(schema_version=1, completed_utc=now(), run_id=run_id,
                      status="operational_smoke_passed_not_formal_evaluation", checks=len(records),
                      errors=[], corpus_sha256=manifest["corpus_sha256"],
                      index_receipt_sha256=pointer["sha256"], manifest_sha256=receipt["manifest_sha256"],
                      vectors_sha256=manifest["vectors_sha256"],
                      checker_sha256=digest(Path(__file__).read_bytes()),
                      passages=len(rows), bm25_documents=len(retriever.bm25.doc_len),
                      modes=["bm25", "dense", "hybrid"], exact_offsets_verified=True,
                      date_and_region_filters_verified=True, inputs_unchanged=True,
                      generated_answers=0, hosted_model_calls=0, new_human_labels=0,
                      human_gold=False, published=False, results=records,
                      limitations=["Operational smoke checks are not retrieval accuracy or a release gate.",
                                   "The English BGE encoder truncates inputs to its model context limit.",
                                   "The prior failed retrieval comparison remains unresolved."])
        attempt = receipt_path.parent
        output = attempt / ("retrieval_smoke-" + uuid4().hex[:8] + ".json")
        write_json(output, result)
        write_json(run / "reports/index_smoke.json", dict(
            path=output.relative_to(run).as_posix(), sha256=digest(output.read_bytes())))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    result = check(args.run)
    print(json.dumps({key: result[key] for key in (
        "status", "checks", "errors", "passages", "bm25_documents", "inputs_unchanged", "published")},
        ensure_ascii=False, indent=2))
