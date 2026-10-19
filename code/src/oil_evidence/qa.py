"""Deterministic extraction audit queue and conservative near-duplicate candidates."""
import random
from collections import defaultdict

from .store import DATA, digest, read_jsonl, words, write_json, write_jsonl


def audit_corpus():
    passages = read_jsonl(DATA / "passages.jsonl")
    docs = {d["document_id"]: d for d in read_jsonl(DATA / "documents.jsonl")}
    errors = []
    signatures = defaultdict(list)
    for p in passages:
        if p["document_id"] not in docs:
            errors.append(dict(passage_id=p["passage_id"], error="missing document"))
        if p["word_count"] != len(words(p["text"])):
            errors.append(dict(passage_id=p["passage_id"], error="word count mismatch"))
        # Shared opening/closing 25-word spans flag possible template or near-duplicate content.
        tokens = words(p["text"])
        signatures[digest(" ".join(tokens[:25]))].append(p["passage_id"])
        signatures[digest(" ".join(tokens[-25:]))].append(p["passage_id"])
    candidates = []
    seen = set()
    for group in signatures.values():
        if len(group) > 1:
            key = tuple(sorted(set(group)))
            if len(key) > 1 and key not in seen:
                candidates.append(dict(passage_ids=list(key), status="requires_review", reason="shared 25-word boundary"))
                seen.add(key)
    rng = random.Random(6127)
    strata = defaultdict(list)
    for p in passages:
        strata[(p["source"], p["available_by"][:4])].append(p)
    sample = []
    for group, rows in sorted(strata.items()):
        sample.extend(rng.sample(rows, min(2, len(rows))))
    write_jsonl(DATA / "qa/extraction_samples.jsonl", [dict(p, qa_status="pending_human") for p in sample])
    write_jsonl(DATA / "qa/near_duplicate_candidates.jsonl", candidates)
    result = dict(passages=len(passages), integrity_errors=errors, near_duplicate_candidate_groups=len(candidates),
                  extraction_samples=len(sample), manual_qa_completed=0,
                  method="Deterministic source/year stratified samples; boundary-match near-duplicate triage, not exhaustive semantic deduplication")
    write_json(DATA / "reports/qa.json", result)
    return result
