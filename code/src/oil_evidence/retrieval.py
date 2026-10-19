from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .store import DATA, ROOT, config, digest, read_jsonl, words, write_json


def permitted(passage, as_of=None, region=None, temporal=True):
    if temporal and as_of:
        date.fromisoformat(as_of)
        available = passage.get("available_by")
        if not available or available > as_of:
            return False
    return not region or region in passage.get("regions", [])


def embedder(model=None):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=model or config()["retrieval"]["embedding_model"],
                         cache_dir=str(ROOT / "models"), threads=4)


def build_index():
    passages = read_jsonl(DATA / "passages.jsonl")
    if not passages:
        raise ValueError("Build a nonempty corpus before indexing")
    model_name = config()["retrieval"]["embedding_model"]
    encoder = embedder(model_name)
    index = DATA / "index"
    index.mkdir(parents=True, exist_ok=True)
    # Cache by model + full text hash so growth of the corpus does not re-embed old text.
    db = sqlite3.connect(index / "embedding_cache.sqlite3")
    db.execute("CREATE TABLE IF NOT EXISTS embeddings (model TEXT,text_hash TEXT,vector BLOB,PRIMARY KEY(model,text_hash))")
    old_manifest = index / "manifest.json"
    if old_manifest.exists() and (index / "vectors.npy").exists():
        old = json.loads(old_manifest.read_text(encoding="utf-8"))
        if old["model"] == model_name and old["vectors_sha256"] == digest((index / "vectors.npy").read_bytes()):
            by_id = {p["passage_id"]: p for p in passages}
            old_vectors = np.load(index / "vectors.npy", allow_pickle=False)
            for pid, vector in zip(old["passage_ids"], old_vectors, strict=True):
                if pid in by_id:
                    db.execute("INSERT OR IGNORE INTO embeddings VALUES (?,?,?)", (model_name, digest(by_id[pid]["text"]), vector.astype(np.float32).tobytes()))
            db.commit()
    arrays = [None] * len(passages)
    missing = []
    for i, p in enumerate(passages):
        row = db.execute("SELECT vector FROM embeddings WHERE model=? AND text_hash=?", (model_name, digest(p["text"]))).fetchone()
        if row:
            arrays[i] = np.frombuffer(row[0], dtype=np.float32)
        else:
            missing.append(i)
    print(f"Embedding cache: {len(passages)-len(missing)} reused, {len(missing)} new", flush=True)
    for start in range(0, len(missing), 128):
        indices = missing[start:start + 128]
        batch = list(encoder.passage_embed([passages[i]["text"] for i in indices], batch_size=32))
        for i, vector in zip(indices, batch, strict=True):
            vector = np.asarray(vector, dtype=np.float32)
            arrays[i] = vector
            db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?)", (model_name, digest(passages[i]["text"]), vector.tobytes()))
        db.commit()
        print(f"Embedded new {min(start + 128, len(missing))}/{len(missing)}", flush=True)
    db.close()
    vectors = np.stack(arrays)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    temp = index / "vectors.tmp.npy"
    np.save(temp, vectors, allow_pickle=False)
    temp.replace(index / "vectors.npy")
    meta = dict(model=model_name, corpus_sha256=digest((DATA / "passages.jsonl").read_bytes()),
                passage_ids=[p["passage_id"] for p in passages], dimension=int(vectors.shape[1]),
                vectors_sha256=digest((index / "vectors.npy").read_bytes()),
                onnx_sha256={str(p.relative_to(ROOT / "models")): digest(p.read_bytes()) for p in (ROOT / "models").rglob("*.onnx")})
    write_json(index / "manifest.json", meta)
    return dict(passages=len(passages), dimension=meta["dimension"], model=model_name)


class Retriever:
    def __init__(self, passages=None, vectors=None, encoder=None, validate_index=True):
        self.passages = passages if passages is not None else read_jsonl(DATA / "passages.jsonl")
        if not self.passages:
            raise ValueError("No passages. Run discover, collect, build first.")
        self.bm25 = BM25Okapi([words(p["text"]) for p in self.passages])
        self.vectors = vectors
        self.encoder = encoder
        self.model_name = config()["retrieval"]["embedding_model"]
        self.index_error = "Dense index not built"
        path = DATA / "index/manifest.json"
        if passages is None and path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta["corpus_sha256"] != digest((DATA / "passages.jsonl").read_bytes()):
                self.index_error = "Dense index is stale; rebuild it"
            elif meta["model"] != self.model_name:
                self.index_error = "Embedding model changed; rebuild the index"
            elif validate_index and meta["vectors_sha256"] != digest((DATA / "index/vectors.npy").read_bytes()):
                self.index_error = "Dense vectors hash mismatch"
            else:
                self.vectors = np.load(DATA / "index/vectors.npy", allow_pickle=False)
                self.index_error = None

    def search(self, question, mode="hybrid", top_k=None, as_of=None, region=None, temporal=True):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        if not question.strip():
            raise ValueError("Question cannot be empty")
        settings = config()["retrieval"]
        top_k = settings["top_k"] if top_k is None else top_k
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        eligible = [i for i, p in enumerate(self.passages) if permitted(p, as_of, region, temporal)]
        if not eligible:
            return []
        sparse = self.bm25.get_scores(words(question))
        rankings, dense = {}, None
        if mode in {"bm25", "hybrid"}:
            rankings["bm25"] = sorted(eligible, key=lambda i: (-sparse[i], self.passages[i]["passage_id"]))[:max(top_k, settings["candidate_k"])]
        if mode in {"dense", "hybrid"}:
            if self.vectors is None:
                raise ValueError(self.index_error)
            if self.encoder is None:
                self.encoder = embedder(self.model_name)
            q = np.array(list(self.encoder.query_embed([question]))[0])
            q /= max(np.linalg.norm(q), 1e-12)
            dense = self.vectors @ q
            rankings["dense"] = sorted(eligible, key=lambda i: (-dense[i], self.passages[i]["passage_id"]))[:max(top_k, settings["candidate_k"])]
        scores = {}
        if mode == "hybrid":
            for ranked in rankings.values():
                for rank, i in enumerate(ranked, 1):
                    scores[i] = scores.get(i, 0) + 1 / (settings["rrf_k"] + rank)
        else:
            scores = {i: float(sparse[i] if mode == "bm25" else dense[i]) for i in rankings[mode]}
        selected = sorted(scores, key=lambda i: (-scores[i], self.passages[i]["passage_id"]))[:top_k]
        return [dict(self.passages[i], rank=rank, score=scores[i], bm25_score=float(sparse[i]),
                     dense_score=None if dense is None else float(dense[i])) for rank, i in enumerate(selected, 1)]
