"""Validate and lock source-first development queries, never answer labels.

Authoring anchors/design intentions are kept apart from blinded query inputs.
This command performs no retrieval, answer generation, annotation or API call.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import json
import os
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import unify_corpus as pipeline
from build_staged_index import assert_unchanged, file_hashes
from oil_evidence.store import DATA, ROOT, digest, now, read_jsonl, write_json, write_jsonl


TYPES = {"fact": 16, "temporal": 14, "comparison": 10, "insufficient": 10}
LABEL_KEYS = {"answerability", "answer_points", "reference_answer", "qrels", "event_status",
              "human_label", "human_reviewed", "model_verification"}


def normalized(text):
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).casefold() if ch.isalnum())


def normalized_url(url):
    value = urlsplit(url)
    return urlunsplit((value.scheme.lower(), value.netloc.lower(), value.path, value.query, ""))


def relative_file(base, name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Unsafe input path")
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError("Input is missing or escapes its directory")
    return path


def legacy_document_ids(records, inputs, original, probes, documents):
    """Derive full old evidence exposure, not only question-generation passages."""
    by_id = {row["passage_id"]: row for row in original}
    blocked = {anchor["document_id"] for row in probes for anchor in row["anchors"]}
    for row in records:
        group = row.get("event_group", "")
        if group.startswith("document-"):
            blocked.add(group.removeprefix("document-"))
        for key in ("evidence_ids", "candidate_pool_ids"):
            for pid in row.get(key, []):
                blocked.add(by_id[pid]["document_id"])
        for quote in row.get("quotes", []):
            blocked.add(by_id[quote["passage_id"]]["document_id"])
        for pid in row.get("qrels", {}):
            blocked.add(by_id[pid]["document_id"])
        if row.get("generation_passage_id"):
            blocked.add(by_id[row["generation_passage_id"]]["document_id"])
    for row in inputs:
        for evidence in row.get("evidence", []):
            blocked.add(evidence["document_id"])
    # Different registry IDs with identical URL/raw bytes are not independent sources.
    while True:
        selected = [row for row in documents if row["document_id"] in blocked]
        urls = {normalized_url(row["url"]) for row in selected}
        raw = {row["raw_sha256"] for row in selected}
        extra = {row["document_id"] for row in documents
                 if normalized_url(row["url"]) in urls or row["raw_sha256"] in raw}
        if extra.issubset(blocked):
            return blocked
        blocked |= extra


def validate(drafts, passages, pages, documents, inventory, old_questions,
             expected_count=50, expected_types=TYPES):
    if len(drafts) != expected_count or Counter(row["type"] for row in drafts) != expected_types:
        raise ValueError("Development count or scenario distribution differs")
    by_id = {row["passage_id"]: row for row in passages}
    source_pages = {row["source_page_id"]: row for row in pages}
    docs = {row["document_id"]: row for row in documents}
    blocked = set(inventory["blocked_document_ids"])
    blocked_urls = set(inventory["blocked_urls"])
    blocked_raw = set(inventory["blocked_raw_sha256"])
    known = {lang: {normalized(row[lang]) for row in old_questions if row.get(lang)}
             for lang in ("question", "question_zh")}
    seen = {lang: set() for lang in known}
    local_ids, evidence_docs, doc_groups = set(), set(), defaultdict(set)
    questions, authoring, future_count = [], [], 0
    for i, row in enumerate(drafts, 1):
        if LABEL_KEYS.intersection(row):
            raise ValueError("Authoring record contains answer labels or model suggestions")
        if row["local_id"] in local_ids or not re.fullmatch(r"[a-z]+[0-9]{2}", row["local_id"]):
            raise ValueError("Duplicate or invalid local question ID")
        local_ids.add(row["local_id"])
        for lang in known:
            signature = normalized(row[lang])
            if not signature or signature in seen[lang] or signature in known[lang]:
                raise ValueError("Empty/duplicate/legacy question text: " + row["local_id"])
            seen[lang].add(signature)
        if re.search(r"this (?:evidence|passage)|above text|given passage", row["question"], re.I):
            raise ValueError("Question is not standalone")
        if (date.fromisoformat(row["as_of"]).isoformat() != row["as_of"]
                or not "2018-01-01" <= row["as_of"] <= "2026-10-05"):
            raise ValueError("Cutoff outside the fixed corpus window")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", row["event_group"]):
            raise ValueError("Missing meaningful event group")
        if not row.get("topic") or not row.get("difficulty_reason_zh") or not row.get("anchors"):
            raise ValueError("Missing question design or authoring context")
        qid = f"dev50-{i:03d}"
        questions.append(dict(question_id=qid, question=row["question"], question_zh=row["question_zh"],
                              as_of=row["as_of"], event_group=row["event_group"], split="development",
                              type=row["type"], topic=row["topic"]))
        anchors = []
        for anchor in row["anchors"]:
            if anchor.get("passage_id") not in by_id:
                raise ValueError("Unknown authoring passage ID")
            passage = by_id[anchor["passage_id"]]
            doc = docs[passage["document_id"]]
            if (doc["document_id"] in blocked or normalized_url(doc["url"]) in blocked_urls
                    or doc["raw_sha256"] in blocked_raw):
                raise ValueError("Legacy source reused: " + row["local_id"])
            quote = anchor["quote"]
            if not isinstance(quote, str) or not quote.strip() or quote not in passage["text"]:
                raise ValueError("Authoring quote is not an exact accepted-passage span")
            source = source_pages[passage["source_page_id"]]
            if source["text"][passage["char_start"]:passage["char_end"]] != passage["text"]:
                raise ValueError("Passage does not match exact source-page offsets")
            available = passage.get("available_by")
            if not available or date.fromisoformat(available).isoformat() != available:
                raise ValueError("Unknown source availability")
            role = anchor["role"]
            if (role == "support_context" and available > row["as_of"]
                    or role == "future_leakage_control" and available <= row["as_of"]
                    or role not in {"support_context", "future_leakage_control"}):
                raise ValueError("Authoring anchor role/cutoff mismatch")
            future_count += role == "future_leakage_control"
            start = passage["text"].index(quote)
            anchors.append(dict(anchor, document_id=doc["document_id"], source=passage["source"],
                                source_page_id=source["source_page_id"], page=passage["page"],
                                available_by=available, url=doc["url"], raw_sha256=doc["raw_sha256"],
                                passage_char_start=start, source_char_start=passage["char_start"] + start,
                                source_char_end=passage["char_start"] + start + len(quote)))
            evidence_docs.add(doc["document_id"])
            doc_groups[doc["document_id"]].add(row["event_group"])
        authoring.append(dict(question_id=qid, local_id=row["local_id"],
                              design_intent=row["type"], difficulty_reason_zh=row["difficulty_reason_zh"],
                              selection_notes=row.get("selection_notes"), anchors=anchors,
                              provenance="codex_source_first_question_authoring_not_qrels", human_gold=False))
    if any(len(groups) > 1 for groups in doc_groups.values()):
        raise ValueError("Shared source document assigned to different event groups")
    return questions, authoring, dict(
        questions=len(questions), scenario_counts=dict(Counter(q["type"] for q in questions)),
        event_groups=len({q["event_group"] for q in questions}), authoring_source_documents=len(evidence_docs),
        source_counts=dict(Counter(a["source"] for row in authoring for a in row["anchors"])),
        authoring_anchors=sum(len(row["anchors"]) for row in authoring),
        future_leakage_controls=future_count, exact_duplicate_questions=0, legacy_document_overlap=0)


def preview(questions):
    names = {"fact": "Fact", "temporal": "Temporal", "comparison": "Comparison", "insufficient": "Evidence-sufficiency challenge"}
    lines = ["# Oil development questions", "", "50 development queries for retrieval debugging, not a formal test set or human gold standard.",
             "Scenario types are design intentions, not answer labels. Authoring evidence and hints are excluded from independent annotation inputs.", ""]
    for row in questions:
        lines += [f"## {row['question_id']} {names[row['type']]}", "", row["question_zh"], "",
                  row["question"], "", f"Query cutoff date: {row['as_of']}.", ""]
    return "\n".join(lines)


def prepare(run_id, corpus_run):
    if (not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,80}", run_id)
            or re.fullmatch(r"con|prn|aux|nul|com[1-9]|lpt[1-9]", run_id)):
        raise ValueError("Invalid development run ID")
    run = DATA / "development_runs" / run_id
    stage = pipeline.select_run(corpus_run) / "stage"
    with pipeline.lock():
        inventory_path = run / "legacy_inventory.json"
        inventory = pipeline.load(inventory_path)
        for name, sha in inventory["input_hashes"].items():
            if digest(relative_file(ROOT, name).read_bytes()) != sha:
                raise ValueError("Legacy inventory binding changed")
        documents = read_jsonl(stage / "documents.jsonl")
        legacy = read_jsonl(DATA / "annotations/codex.jsonl")
        inputs = read_jsonl(DATA / "annotations/codex_input.jsonl")
        original = read_jsonl(DATA / "expansion_runs/official-2018-2026-v1/baseline/passages.jsonl")
        probes = pipeline.load(DATA / "unified_runs/semantic-page-v1/reports/source_probes-cabd757df2e0.json")["probes"]
        if legacy_document_ids(legacy, inputs, original, probes, documents) != set(inventory["blocked_document_ids"]):
            raise ValueError("Legacy document exclusion inventory differs from actual old inputs")
        expected_aliases = [row for row in documents if row["document_id"] in inventory["blocked_document_ids"]]
        if ({normalized_url(row["url"]) for row in expected_aliases} != set(inventory["blocked_urls"])
                or {row["raw_sha256"] for row in expected_aliases} != set(inventory["blocked_raw_sha256"])):
            raise ValueError("Legacy URL/raw-byte exclusion inventory differs")
        index_pointer = pipeline.load(stage.parent / "reports/index_build.json")
        index_receipt = pipeline.load(relative_file(stage.parent, index_pointer["path"]))
        if digest(relative_file(stage.parent, index_pointer["path"]).read_bytes()) != index_pointer["sha256"]:
            raise ValueError("Index receipt changed")
        assert_unchanged(index_receipt["protected_sha256"])
        if pipeline.stage_hashes() != index_receipt["stage_hashes"] or pipeline.hashes() != index_receipt["implementation_hashes"]:
            raise ValueError("Staged corpus or implementation changed")
        paths = [run / "drafts" / name for name in ("eia.json", "policy.json", "other.json")]
        protected = {**index_receipt["protected_sha256"], **file_hashes([
            *paths, inventory_path, ROOT / "scripts/prepare_development_questions.py",
            stage / "index/manifest.json", stage / "index/vectors.npy"])}
        drafts = sum([pipeline.load(path) for path in paths], [])
        old_questions = [*legacy, *pipeline.load(ROOT / "seeds/questions.json"), *probes]
        questions, authoring, stats = validate(drafts, read_jsonl(stage / "passages.jsonl"),
                                             read_jsonl(stage / "source_pages.jsonl"), documents, inventory, old_questions)
        selected_docs = {a["document_id"] for row in authoring for a in row["anchors"]}
        for doc in documents:
            if doc["document_id"] in selected_docs:
                path = relative_file(DATA, doc["raw_path"])
                if digest(path.read_bytes()) != doc["raw_sha256"]:
                    raise ValueError("Authoring raw source changed")
                protected[str(path)] = doc["raw_sha256"]
        # The actual query list is locked before a single retrieval ranking is seen.
        canonical_json = json.dumps(questions, ensure_ascii=False, indent=2)
        # The unchanged store writer uses native text newlines on Windows.
        # Bind the physical file bytes, not only the LF canonical representation.
        qsha = digest(canonical_json.replace("\n", os.linesep))
        output = run / "frozen" / qsha
        if output.exists():
            old = pipeline.load(output / "manifest.json")
            for name, sha in old["artifact_sha256"].items():
                if digest(relative_file(output, name).read_bytes()) != sha:
                    raise ValueError("Locked development artifact changed")
            if old["input_sha256"] != protected:
                raise ValueError("Locked question inputs changed; use a new development run")
        else:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "questions.json", questions)
            write_jsonl(output / "annotation_queries.jsonl", [
                {key: row[key] for key in ("question_id", "question", "question_zh", "as_of")} for row in questions])
            write_jsonl(output / "authoring_anchors.jsonl", authoring)
            write_json(output / "heldout_exclusions.json", dict(
                split="development_only", event_groups=sorted({row["event_group"] for row in questions}),
                document_ids=sorted({a["document_id"] for row in authoring for a in row["anchors"]}),
                urls=sorted({normalized_url(a["url"]) for row in authoring for a in row["anchors"]}),
                raw_sha256=sorted({a["raw_sha256"] for row in authoring for a in row["anchors"]}),
                reserve_same_event_sibling_reports=True, semantic_event_independence_verified=False,
                policy_lineage_review_pending=[row["question_id"] for row in authoring if row["selection_notes"]]))
            (output / "preview.md").write_text(preview(questions), encoding="utf-8", newline="\n")
            artifacts = {p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}
            if artifacts["questions.json"] != qsha:
                raise ValueError("Serialized question bytes differ from their version hash")
            manifest = dict(
                schema_version=1, status="development_questions_locked_not_annotated", run_id=run_id,
                locked_utc=now(), corpus_sha256=index_receipt["corpus_sha256"],
                questions_sha256=qsha, canonical_questions_sha256=digest(canonical_json),
                statistics=stats, artifact_sha256=artifacts, input_sha256=protected,
                provenance="codex_source_first_question_authoring_and_machine_sanity_checks",
                observed_author_model_id=None, author_model_identity_note="Exact active model ID not exposed to this task.",
                human_gold=False, human_labels_created=0, qa_annotations_created=0, hosted_model_calls=0,
                retrieval_runs=0, question_selection_before_retrieval=True, formal_test_set_frozen=False,
                limitations=["Scenario intentions are not answerability labels or qrels.",
                             "Author anchors are not exhaustive relevant evidence or reference answers.",
                             "Exact/source duplicate checks are not exhaustive semantic deduplication.",
                             "Event groups are conservative development reservations, not verified independent events.",
                             "BOEM policy lineage and same-event sibling reports require review before a held-out test.",
                             "Independent pooled-evidence annotations and formal course approval remain pending.",
                             "Codex subscription usage is not measured."])
            assert_unchanged(protected)
            write_json(output / "manifest.json", manifest)
        assert_unchanged(protected)
        write_json(run / "current.json", dict(path=(output / "manifest.json").relative_to(run).as_posix(),
                                             sha256=digest((output / "manifest.json").read_bytes())))
        return output, pipeline.load(output / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="oil-dev50-v1")
    parser.add_argument("--corpus-run", default="semantic-page-v2-buffer")
    args = parser.parse_args()
    output, result = prepare(args.run, args.corpus_run)
    print(json.dumps(dict(output=str(output), status=result["status"], **result["statistics"]),
                     ensure_ascii=False, indent=2))
