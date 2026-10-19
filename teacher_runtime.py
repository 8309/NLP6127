"""Relocatable adapter for the preserved retrieval implementation and index."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parent


def verify_package(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "TEACHER_MANIFEST.json").read_bytes())
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        path = (root / name).resolve()
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or not path.is_relative_to(root) or not path.is_file()):
            raise ValueError(f"Missing or unsafe package member: {name}")
        raw = path.read_bytes()
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"Package hash mismatch: {name}; extract a fresh copy")
    return manifest


def load_runtime(root: Path = ROOT):
    root = root.resolve()
    manifest = verify_package(root)
    sys.path.insert(0, str(root / "code/src"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from fastembed import TextEmbedding
    import numpy as np
    from oil_evidence.codex_runtime import StagedRuntime
    from oil_evidence.retrieval import Retriever

    corpus = root / "data/corpus"
    rows = [json.loads(line) for line in (corpus / "passages.jsonl").read_bytes().splitlines() if line.strip()]
    meta = json.loads((corpus / "index/manifest.json").read_bytes())
    vectors = np.load(corpus / "index/vectors.npy", allow_pickle=False)
    ids = [row["passage_id"] for row in rows]
    if (ids != meta["passage_ids"] or len(set(ids)) != len(ids)
            or hashlib.sha256((corpus / "passages.jsonl").read_bytes()).hexdigest() != meta["corpus_sha256"]
            or hashlib.sha256((corpus / "index/vectors.npy").read_bytes()).hexdigest() != meta["vectors_sha256"]
            or vectors.shape != (len(rows), meta["dimension"]) or vectors.dtype != np.float32
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError("Corpus and vector index do not match")
    encoder = TextEmbedding(model_name=meta["model"], specific_model_path=str(root / "models/embedding"),
                            cache_dir=str(root / ".model-cache"), local_files_only=True,
                            providers=["CPUExecutionProvider"], threads=4)
    retriever = Retriever(passages=rows, vectors=vectors, encoder=encoder)
    if retriever.model_name != meta["model"]:
        raise ValueError("Source configuration specifies a different embedding model")
    pages = {row["source_page_id"]: row for line in (corpus / "source_pages.jsonl").read_bytes().splitlines()
             if line.strip() for row in [json.loads(line)]}
    for row in rows:
        page = pages.get(row["source_page_id"])
        if page is None or page["text"][row["char_start"]:row["char_end"]] != row["text"]:
            raise ValueError("Passage differs from its source page")
    protected = {str(root / name): entry["sha256"] for name, entry in manifest["files"].items()
                 if name.startswith(("data/corpus/", "models/embedding/", "code/src/", "code/scripts/"))}
    return StagedRuntime(root, corpus, meta["corpus_sha256"], retriever,
                         {row["passage_id"]: row for row in rows}, pages, protected)


def search(runtime, question: str, mode="hybrid", as_of="2026-10-05", region=None):
    from oil_evidence.query import extract_quotes
    started = perf_counter()
    with runtime.lock:
        hits = runtime.retriever.search(question, mode=mode, as_of=as_of, region=region, temporal=True)
    for hit in hits:
        page = runtime.pages[hit["source_page_id"]]
        if page["text"][hit["char_start"]:hit["char_end"]] != hit["text"]:
            raise ValueError("Returned evidence failed its exact source-offset check")
    return {"question": question, "as_of": as_of, "mode": mode,
            "claims": extract_quotes(question, hits) if hits else [], "evidence": hits,
            "answer_type": "source_excerpt_preview_not_synthesized_answer",
            "answerability": None, "model_calls": 0,
            "elapsed_seconds": perf_counter() - started}
