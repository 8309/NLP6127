"""Read-only, source-page-matched segmentation diagnostics, never formal gold.

Both corpora must already have valid indexes.  No DATA monkeypatch, index build,
answer generation, paid API, label migration, or selection from retrieval hits.
The optional mini probes are independently frozen, source-native quote anchors;
their literal support metrics are not human accuracy or exhaustive recall.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np

from oil_evidence.retrieval import Retriever, embedder, permitted
from oil_evidence.store import ROOT, config, digest, now, read_jsonl, words, write_json


MODES = ("bm25", "dense", "hybrid")
PAGE_METADATA = ("source", "title", "url", "published_date", "available_by",
                 "date_precision", "date_source")
# Exact lexical coverage only, not semantic relevance.  Keep all numeric tokens
# and uncommon content words rather than silently imposing an oil-only ontology.
STOP_WORDS = frozenset("a an and are as at be between by can do does every for from had has have how in is it latest main of on or solely than that the their these they this to was were what when which while will with without according available evidence exact published reported reports supplied sources enough specific particular expected establish prove known amount same accounting verified contain connects shows would could should about into rather period may much".split())


def normalized(text):
    """Whitespace and case only: numbers, punctuation, and units stay intact."""
    return " ".join(text.split()).casefold()


def page_key(row):
    document, page = row.get("document_id"), row.get("page")
    if not isinstance(document, str) or not document.strip():
        raise ValueError("Missing source document_id")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("Source page must be a positive integer")
    return document, page


def page_record(key):
    return {"document_id": key[0], "page": key[1]}


def valid_date(value, label, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError(f"{label} must be an ISO day date")
    return value


def grouped_pages(rows):
    grouped = defaultdict(list)
    metadata = {}
    for row in rows:
        key = page_key(row)
        signature = tuple(row.get(field) for field in PAGE_METADATA)
        if key in metadata and metadata[key] != signature:
            raise ValueError("Inconsistent metadata within source page: " + str(key))
        metadata[key] = signature
        grouped[key].append(row)
    return dict(grouped), metadata


@dataclass
class Corpus:
    root: Path
    rows: list
    vectors: np.ndarray
    manifest: dict
    hashes: dict
    pages: dict
    metadata: dict


def input_hashes(root):
    root = Path(root).resolve()
    names = ("passages.jsonl", "index/manifest.json", "index/vectors.npy")
    return {name: digest((root / name).read_bytes()) for name in names}


def load_corpus(root, expected_model):
    root = Path(root).resolve()
    hashes = input_hashes(root)
    rows = read_jsonl(root / "passages.jsonl")
    if not rows:
        raise ValueError("A comparison corpus cannot be empty")
    ids = [row.get("passage_id") for row in rows]
    if any(not isinstance(pid, str) or not pid for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate passage IDs")
    for row in rows:
        if not isinstance(row.get("text"), str) or not words(row["text"]):
            raise ValueError("Empty passage text")
        valid_date(row.get("available_by"), "available_by", optional=True)
        if row.get("word_count") is not None and row["word_count"] != len(words(row["text"])):
            raise ValueError("Passage word_count mismatch")
    manifest = json.loads((root / "index/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("corpus_sha256") != hashes["passages.jsonl"]:
        raise ValueError("Corpus/index hash mismatch")
    if manifest.get("vectors_sha256") != hashes["index/vectors.npy"]:
        raise ValueError("Vectors hash mismatch")
    if manifest.get("model") != expected_model:
        raise ValueError("Embedding model differs from fixed comparison model")
    if manifest.get("passage_ids") != ids:
        raise ValueError("Index passage order differs from corpus")
    vectors = np.load(root / "index/vectors.npy", allow_pickle=False)
    dimension = manifest.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Invalid embedding dimension")
    if vectors.ndim != 2 or vectors.shape != (len(rows), dimension):
        raise ValueError("Vector shape differs from corpus/manifest")
    if not np.issubdtype(vectors.dtype, np.floating) or not np.isfinite(vectors).all():
        raise ValueError("Vectors must be finite floating-point values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=1e-3, atol=1e-3):
        raise ValueError("Comparison indexes must contain unit-normalized nonzero vectors")
    pages, metadata = grouped_pages(rows)
    if input_hashes(root) != hashes:
        raise ValueError("Comparison input changed during loading")
    return Corpus(root, rows, vectors, manifest, hashes, pages, metadata)


def shared_scope(baseline, candidate):
    shared = set(baseline.pages) & set(candidate.pages)
    mismatch = {key for key in shared if baseline.metadata[key] != candidate.metadata[key]}
    comparable = shared - mismatch
    if not comparable:
        raise ValueError("No identical-metadata source pages shared by both versions")
    return comparable, {
        "method": "identical document_id/page and publication metadata intersection",
        "baseline_pages": len(baseline.pages), "candidate_pages": len(candidate.pages),
        "shared_comparable_pages": len(comparable),
        "baseline_only_pages": [page_record(k) for k in sorted(set(baseline.pages) - set(candidate.pages))],
        "candidate_only_pages": [page_record(k) for k in sorted(set(candidate.pages) - set(baseline.pages))],
        "metadata_mismatch_pages_excluded": [page_record(k) for k in sorted(mismatch)],
        "baseline_common_passages": sum(page_key(p) in comparable for p in baseline.rows),
        "candidate_common_passages": sum(page_key(p) in comparable for p in candidate.rows),
        "limitations": "Intersection controls source-page coverage, not identical text content. Cleaning, quarantine, and segmentation can still change retained content on the same page; source-anchor catalog availability is reported separately. BM25 is fitted only on each version's corresponding common-page passages, never on unmatched extra pages.",
    }


class CachedEncoder:
    """Shared, eagerly warmed query vectors; ranking timings exclude encoding."""
    def __init__(self, encoder, dimension):
        self.encoder, self.dimension = encoder, dimension
        self.cache, self.seconds = {}, {}

    def warm(self, question):
        if question not in self.cache:
            start = perf_counter()
            encoded = list(self.encoder.query_embed([question]))
            if len(encoded) != 1:
                raise ValueError("Query encoder returned the wrong batch size")
            vector = np.asarray(encoded[0], dtype=np.float32)
            if vector.shape != (self.dimension,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
                raise ValueError("Query encoder returned invalid vectors")
            self.cache[question] = vector
            self.seconds[question] = perf_counter() - start
        return self.cache[question]

    def query_embed(self, questions):
        return iter(self.warm(question) for question in questions)


def make_retriever(corpus, scope, encoder):
    indices = [i for i, row in enumerate(corpus.rows) if page_key(row) in scope]
    # Explicit arguments avoid reading, replacing, or monkeypatching global DATA.
    return Retriever(passages=[corpus.rows[i] for i in indices],
                     vectors=corpus.vectors[indices], encoder=encoder)


def verify_model_files(baseline, candidate):
    expected = baseline.manifest.get("onnx_sha256")
    if not isinstance(expected, dict) or not expected or expected != candidate.manifest.get("onnx_sha256"):
        raise ValueError("Both comparison indexes must bind identical nonempty ONNX model hashes")
    checked = {}
    model_root = (ROOT / "models").resolve()
    for relative, sha in expected.items():
        path = (model_root / relative).resolve()
        if not path.is_relative_to(model_root) or not path.is_file():
            raise ValueError("Missing or unsafe locally frozen ONNX model path")
        actual = digest(path.read_bytes())
        if actual != sha:
            raise ValueError("Local query encoder model differs from indexed model")
        checked[str(path)] = actual
    return checked


def load_seeds(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Seed questions must be a nonempty JSON list")
    seen = set()
    result = []
    for i, row in enumerate(data, 1):
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Seed question is missing")
        signature = (normalized(row["question"]), valid_date(row.get("as_of"), "query as_of"))
        if signature in seen:
            raise ValueError("Duplicate question/cutoff in fixed seeds")
        seen.add(signature)
        result.append(dict(row, query_id=f"seed-{i:02d}", kind="seed_diagnostic"))
    return result


def load_source_pages(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Empty source-page snapshot")
    result, identifiers = {}, set()
    for row in rows:
        key = page_key(row)
        if key in result:
            raise ValueError("Duplicate source-native document/page")
        identifier = row.get("source_page_id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Missing or duplicate source_page_id")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Empty source-native page text")
        valid_date(row.get("available_by"), "source available_by", optional=True)
        result[key] = row
        identifiers.add(identifier)
    return result


def load_probes(path, source_pages, source_pages_hash):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("source_pages_sha256") != source_pages_hash:
        raise ValueError("Frozen probes do not bind the supplied source-page snapshot")
    if payload.get("selection_method") != "source_only_before_retrieval":
        raise ValueError("Probes must declare source-only selection before retrieval")
    if payload.get("human_gold") is not False or payload.get("provenance") != "codex_source_grounded":
        raise ValueError("Mini probes must explicitly disclose model/source-grounded, not human gold")
    probes = payload.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError("Frozen probe list cannot be empty")
    result, seen = [], set()
    for probe in probes:
        identifier, question = probe.get("probe_id"), probe.get("question")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Missing or duplicate probe ID")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Missing probe question")
        cutoff = valid_date(probe.get("as_of"), "probe as_of")
        anchors = probe.get("anchors")
        if not isinstance(anchors, list) or not anchors:
            raise ValueError("Probe has no source-native anchors")
        anchor_keys = set()
        for anchor in anchors:
            key = page_key(anchor)
            if key not in source_pages:
                raise ValueError("Probe references a missing source page")
            quote = anchor.get("quote")
            if not isinstance(quote, str) or len(words(quote)) < 12:
                raise ValueError("Source-native anchor quote must contain at least 12 words")
            if normalized(quote) not in normalized(source_pages[key]["text"]):
                raise ValueError("Probe quote is not a literal source-page substring")
            identity = key, normalized(quote)
            if identity in anchor_keys:
                raise ValueError("Duplicate probe anchor")
            anchor_keys.add(identity)
            if not permitted(source_pages[key], as_of=cutoff):
                raise ValueError("Probe source is not available by its query cutoff")
        # The same literal quote is not itself a useful retrieval query.
        if any(normalized(anchor["quote"]) in normalized(question) for anchor in anchors):
            raise ValueError("Probe question copies its complete answer quote")
        seen.add(identifier)
        result.append(dict(probe, query_id=identifier, kind="source_anchor_probe"))
    return result


def quote_support(anchor, evidence, corpus_order):
    key, quote = page_key(anchor), normalized(anchor["quote"])
    matching = [hit for hit in evidence if page_key(hit) == key]
    single = [hit for hit in matching if quote in normalized(hit["text"])]
    ordered = sorted(matching, key=lambda hit: corpus_order[hit["passage_id"]])
    # Literal assembly handles a quote crossing a genuine segment boundary.
    # Source validation establishes quote existence; no paraphrase or judge is
    # involved. Context, correctness, and exhaustive relevance are not assessed.
    assembled = quote in normalized(" ".join(hit["text"] for hit in ordered))
    return dict(single_passage_supported=bool(single), assembled_page_supported=assembled,
                single_passage_first_rank=min((hit["rank"] for hit in single), default=None),
                single_passage_ids=[hit["passage_id"] for hit in single],
                retrieved_same_page_ids=[hit["passage_id"] for hit in ordered])


def query_metrics(query, evidence, corpus_order):
    cutoff = query["as_of"]
    errors = [hit["passage_id"] for hit in evidence if not permitted(hit, as_of=cutoff)]
    if errors:
        raise ValueError("Temporal retrieval leak: " + ", ".join(errors))
    if len({hit["passage_id"] for hit in evidence}) != len(evidence):
        raise ValueError("Duplicate retrieval hits")
    terms = set(words(query["question"])) - STOP_WORDS
    found = terms & set(words(" ".join(hit["text"] for hit in evidence)))
    pages = Counter(page_key(hit) for hit in evidence)
    metrics = dict(returned_passages=len(evidence), unique_pages=len(pages),
                   unique_documents=len({hit["document_id"] for hit in evidence}),
                   context_words=sum(len(words(hit["text"])) for hit in evidence),
                   max_same_page_hits=max(pages.values(), default=0),
                   max_same_page_share=max(pages.values(), default=0) / len(evidence) if evidence else 0,
                   duplicate_page_slots=len(evidence) - len(pages), temporal_errors=len(errors),
                   lexical_query_term_coverage=len(found) / len(terms) if terms else None,
                   lexical_query_terms=sorted(terms), lexical_terms_found=sorted(found))
    if "anchors" in query:
        supports = [dict(anchor, **quote_support(anchor, evidence, corpus_order)) for anchor in query["anchors"]]
        first_all = next((k for k in range(1, len(evidence) + 1)
                          if all(quote_support(a, evidence[:k], corpus_order)["assembled_page_supported"]
                                 for a in query["anchors"])), None)
        metrics.update(anchors=supports,
                       single_passage_anchor_support_coverage=sum(s["single_passage_supported"] for s in supports) / len(supports),
                       assembled_anchor_support_coverage=sum(s["assembled_page_supported"] for s in supports) / len(supports),
                       all_anchors_supported=all(s["assembled_page_supported"] for s in supports),
                       first_rank_supporting_all_anchors=first_all,
                       source_anchor_reciprocal_rank=1 / first_all if first_all else 0)
    return metrics


def evidence_record(hit):
    return {key: hit.get(key) for key in ("passage_id", "document_id", "page", "source", "title", "url",
                                        "available_by", "rank", "score", "bm25_score", "dense_score")} | {
        "word_count": len(words(hit["text"])), "text_sha256": digest(hit["text"]),
        "text_preview": hit["text"][:250]}


def aggregate(rows):
    fields = ("returned_passages", "unique_pages", "unique_documents", "context_words", "max_same_page_hits",
              "max_same_page_share", "duplicate_page_slots", "lexical_query_term_coverage", "ranking_seconds",
              "single_passage_anchor_support_coverage", "assembled_anchor_support_coverage", "source_anchor_reciprocal_rank")
    result = {}
    for kind in ("seed_diagnostic", "source_anchor_probe"):
        for mode in MODES:
            selected = [row for row in rows if row["kind"] == kind and row["mode"] == mode]
            if not selected:
                continue
            item = {"queries": len(selected), "temporal_errors": sum(row["metrics"]["temporal_errors"] for row in selected)}
            for field in fields:
                values = [row["metrics"][field] for row in selected if row["metrics"].get(field) is not None]
                if values:
                    item["mean_" + field] = statistics.mean(values)
            item["p95_ranking_seconds"] = float(np.percentile([row["metrics"]["ranking_seconds"] for row in selected], 95))
            result[kind + "/" + mode] = item
    return result


def compare(baseline_path, candidate_path, seeds_path, output, *, source_pages_path=None,
            probes_path=None, top_k=5, encoder=None):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    output = Path(output).resolve()
    if output.suffix != ".json" or output.exists():
        raise ValueError("Comparison report must be a new .json file; existing frozen reports are not overwritten")
    settings = config()
    model = settings["retrieval"]["embedding_model"]
    baseline, candidate = (load_corpus(path, model) for path in (baseline_path, candidate_path))
    if baseline.manifest["dimension"] != candidate.manifest["dimension"]:
        raise ValueError("Comparison embedding dimensions differ")
    protected = {corpus.root / name for corpus in (baseline, candidate) for name in corpus.hashes}
    protected.update(Path(path).resolve() for path in (seeds_path, source_pages_path, probes_path) if path)
    if output in protected or any(output.is_relative_to(c.root / "index") or output.is_relative_to(c.root / "annotations")
                                  for c in (baseline, candidate)):
        raise ValueError("Comparison output would overwrite protected input/index/annotations")
    scope, scope_report = shared_scope(baseline, candidate)
    queries = load_seeds(seeds_path)
    auxiliary = {str(Path(seeds_path).resolve()): digest(Path(seeds_path).read_bytes())}
    source_pages = None
    if source_pages_path:
        source_path = Path(source_pages_path).resolve()
        auxiliary[str(source_path)] = digest(source_path.read_bytes())
        source_pages = load_source_pages(source_path)
    if probes_path:
        if source_pages is None:
            raise ValueError("Source-native probes require --source-pages")
        auxiliary[str(Path(probes_path).resolve())] = digest(Path(probes_path).read_bytes())
        probes = load_probes(probes_path, source_pages, auxiliary[str(Path(source_pages_path).resolve())])
        if set(q["query_id"] for q in probes) & set(q["query_id"] for q in queries):
            raise ValueError("Probe and seed IDs collide")
        queries += probes
    if source_pages is not None:
        missing = scope - set(source_pages)
        if missing:
            raise ValueError("Comparable active source page absent from frozen source snapshot")
        # A changed date or source URL invalidates as-of comparability, not merely
        # a warning in a report. Region keywords may legitimately vary with text.
        for key in scope:
            expected = tuple(source_pages[key].get(field) for field in PAGE_METADATA)
            if expected != baseline.metadata[key]:
                raise ValueError("Frozen source metadata differs from comparable active page: " + str(key))
    actual_model_hashes = {}
    if encoder is None:
        actual_model_hashes = verify_model_files(baseline, candidate)
        encoder = embedder(model)
    cached = CachedEncoder(encoder, baseline.manifest["dimension"])
    for query in queries:
        cached.warm(query["question"])
    retrievers = {"baseline": make_retriever(baseline, scope, cached),
                  "candidate": make_retriever(candidate, scope, cached)}
    corpora = {"baseline": baseline, "candidate": candidate}
    order = {name: {p["passage_id"]: i for i, p in enumerate(c.rows)} for name, c in corpora.items()}
    all_rows, pairs = {name: [] for name in corpora}, []
    for query in queries:
        for mode in MODES:
            matched = {}
            for name in ("baseline", "candidate"):
                start = perf_counter()
                hits = retrievers[name].search(query["question"], mode=mode, top_k=top_k, as_of=query["as_of"])
                seconds = perf_counter() - start
                metrics = query_metrics(query, hits, order[name])
                metrics["ranking_seconds"] = seconds
                row = {"query_id": query["query_id"], "kind": query["kind"], "question": query["question"],
                       "as_of": query["as_of"], "mode": mode, "metrics": metrics,
                       "evidence": [evidence_record(hit) for hit in hits]}
                all_rows[name].append(row)
                matched[name] = row
            page_sets = {name: {page_key(hit) for hit in row["evidence"]} for name, row in matched.items()}
            union = page_sets["baseline"] | page_sets["candidate"]
            first_ranks = {}
            for name, row in matched.items():
                ranks = {}
                for hit in row["evidence"]:
                    ranks.setdefault(page_key(hit), hit["rank"])
                first_ranks[name] = ranks
            pairs.append({"query_id": query["query_id"], "kind": query["kind"], "mode": mode,
                          "top_page_jaccard": len(page_sets["baseline"] & page_sets["candidate"]) / len(union) if union else None,
                          "page_first_ranks": [{**page_record(k), "baseline_rank": first_ranks["baseline"].get(k),
                                                "candidate_rank": first_ranks["candidate"].get(k)} for k in sorted(union)],
                          "candidate_minus_baseline": {field: matched["candidate"]["metrics"][field] - matched["baseline"]["metrics"][field]
                              for field in ("unique_pages", "context_words", "duplicate_page_slots", "ranking_seconds")}})
    catalog = []
    for query in queries:
        if "anchors" not in query:
            continue
        for anchor in query["anchors"]:
            row = {"query_id": query["query_id"], **anchor, "in_comparable_scope": page_key(anchor) in scope}
            for name, corpus in corpora.items():
                candidates = [dict(p, rank=i) for i, p in enumerate(corpus.pages.get(page_key(anchor), []), 1)]
                supported = quote_support(anchor, candidates, order[name])
                row[name + "_catalog_single_passage_support"] = supported["single_passage_supported"]
                row[name + "_catalog_assembled_support"] = supported["assembled_page_supported"]
            catalog.append(row)
    for corpus in corpora.values():
        if input_hashes(corpus.root) != corpus.hashes:
            raise ValueError("Comparison input changed during retrieval")
    for path, expected in auxiliary.items():
        if digest(Path(path).read_bytes()) != expected:
            raise ValueError("Frozen query/source input changed during comparison")
    for path, expected in actual_model_hashes.items():
        if digest(Path(path).read_bytes()) != expected:
            raise ValueError("Local query encoder model changed during comparison")
    if config() != settings:
        raise ValueError("Retrieval configuration changed during comparison")
    report = {"at": now(), "diagnostic_only": True, "human_gold": False,
              "external_model_calls": 0, "human_labels_created": 0, "generated_answers": 0,
              "answer_accuracy": None, "exhaustive_retrieval_recall": None,
              "settings": {"embedding_model": model, "top_k": top_k, "candidate_k": settings["retrieval"]["candidate_k"],
                           "rrf_k": settings["retrieval"]["rrf_k"], "temporal": True, "modes": list(MODES)},
              "input_hashes": {name: {"root": str(c.root), **c.hashes} for name, c in corpora.items()},
              "auxiliary_input_hashes": auxiliary, "inputs_unchanged": True, "source_scope": scope_report,
              "queries": len(queries), "seed_queries": sum(q["kind"] == "seed_diagnostic" for q in queries),
              "probe_queries": sum(q["kind"] == "source_anchor_probe" for q in queries),
              "query_embedding_seconds": cached.seconds,
              "local_encoder_model_hashes": actual_model_hashes,
              "encoder_provenance": "verified_local_onnx" if actual_model_hashes else "injected_encoder_for_isolated_tests",
              "summaries": {name: aggregate(rows) for name, rows in all_rows.items()},
              "paired_diagnostics": pairs, "source_anchor_catalog": catalog, "results": all_rows,
              "script_sha256": digest(Path(__file__).read_bytes()),
              "metric_definitions": {
                  "lexical_query_term_coverage": "Fraction of non-stopword query tokens occurring verbatim in top-k text; no semantic relevance or answer label.",
                  "page_coverage": "Distinct source pages in top-k, not relevant-page recall.",
                  "max_same_page_share": "Largest page hit count divided by returned hit count; measures top-k concentration, not relevance.",
                  "anchor_support": "Frozen source quote literal substring in one passage, or same-page retrieved passages joined in original corpus order. A quote may span chunks. Source-native probe anchors are model-selected, not independently human annotated or exhaustive.",
                  "source_anchor_reciprocal_rank": "Reciprocal of earliest top-n prefix whose assembled evidence contains every frozen anchor, zero if none; not general-purpose ground-truth MRR.",
                  "ranking_seconds": "Single local ranking run with shared prewarmed query embeddings; excludes index loading, encoding, generation, and network, with no confidence interval or throughput claim.",
                  "temporal_errors": "Returned metadata missing available_by or later than query cutoff, not a detector of event-date or body-text errors."},
              "limitations": ["Same documents/pages, queries, model, and retrieval parameters; retained text may differ due to cleaning and quarantine as well as chunk boundaries.",
                               "Old 100 QA records and old labels are not loaded or reused as gold.",
                               "No answer correctness, citation semantic support, human agreement, causal inference, or full benchmark acceptance follows from this comparison.",
                               "All frozen probes are retained, including catalog-missing, scope-excluded, and unretrieved anchors; no successful-hit filtering.",
                               "Unequal top-k context lengths are measured, not automatically compensated; evidence-token-budget comparisons are a separate experiment.",
                               "Timings are sequential smoke measurements, not a controlled performance benchmark. Local embedding model use is unpaid, but Codex subscription usage is not measured here."]}
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, default=ROOT / "seeds/questions.json")
    parser.add_argument("--source-pages", type=Path)
    parser.add_argument("--probes", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, args.seeds, args.output,
                     source_pages_path=args.source_pages, probes_path=args.probes, top_k=args.top_k)
    print(json.dumps({key: report[key] for key in ("diagnostic_only", "queries", "seed_queries", "probe_queries",
                                                 "inputs_unchanged", "source_scope", "summaries")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
