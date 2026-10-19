"""Prepare source-grounded reading packs and import current-chat Codex decisions.

No remote calls. Decisions are authored in the Codex conversation, not generated
by a heuristic, and remain machine-only labels outside the human review DB.
"""
import argparse
import json
import sys
from collections import Counter

from oil_evidence.annotations import validate_record
from oil_evidence.retrieval import Retriever
from oil_evidence.store import DATA, digest, now, read_jsonl, write_json, write_jsonl

PACK = DATA / "annotations/codex_input.jsonl"
DECISIONS = DATA / "annotations/codex_decisions"


def prepare():
    if PACK.exists():
        raise ValueError("Reading pack already exists; use its fixed evidence snapshot")
    retriever = Retriever()
    passages = {p["passage_id"]: p for p in retriever.passages}
    rows = []
    for dataset in ["candidates", "pilot"]:
        for index, record in enumerate(read_jsonl(DATA / f"annotations/{dataset}.jsonl"), 1):
            if dataset == "candidates" or record.get("question_type") == "source_reading_template":
                evidence = [passages[p] for p in record["evidence_ids"] if p in passages]
            else:
                evidence = retriever.search(record["question"], mode="hybrid", top_k=3, as_of=record.get("as_of"))
            rows.append(dict(dataset=dataset, number=index, record=record, evidence=evidence,
                             original_sha256=digest(json.dumps(record, sort_keys=True)),
                             corpus_sha256=digest((DATA / "passages.jsonl").read_bytes())))
    write_jsonl(PACK, rows)
    print(json.dumps(dict(records=len(rows), path=str(PACK))))


def show(dataset, start, count):
    for row in read_jsonl(PACK):
        if row["dataset"] == dataset and start <= row["number"] < start + count:
            r = row["record"]
            print(f"\n=== {dataset} {row['number']} | {r['record_id']} | cutoff {r.get('as_of')} ===")
            print("QUESTION:", r["question"])
            print("ORIGINAL:", json.dumps(dict(points=r.get("answer_points"), state=r.get("event_status")), ensure_ascii=False))
            for i, p in enumerate(row["evidence"], 1):
                print(f"E{i} | {p['title']} | p{p['page']} | available {p['available_by']}")
                print(p["text"])


def publish():
    inputs = read_jsonl(PACK)
    lookup = {(r["dataset"], r["number"]): r for r in inputs}
    decisions = {}
    decision_files = sorted(DECISIONS.glob("*.json"))
    for path in decision_files:
        for d in json.loads(path.read_text(encoding="utf-8")):
            key = (d["dataset"], d["number"])
            if key in decisions:
                raise ValueError(f"Duplicate decision {key}")
            decisions[key] = d
    if decisions.keys() != lookup.keys():
        raise ValueError(f"Incomplete annotation: missing={sorted(lookup.keys()-decisions.keys())}")
    by_id = {p["passage_id"]: p for p in read_jsonl(DATA / "passages.jsonl")}
    corpus_hash = digest((DATA / "passages.jsonl").read_bytes())
    results = []
    for key, row in lookup.items():
        d = decisions[key]
        if row["corpus_sha256"] != corpus_hash:
            raise ValueError("Corpus changed during annotation")
        old = row["record"]
        pool = row["evidence"]
        if len(d["grades"]) != len(pool):
            raise ValueError(f"Incomplete evidence grades for {key}")
        if d["result"] not in {"passed", "corrected", "insufficient", "partial"}:
            raise ValueError(f"Invalid decision for {key}")
        quotes = []
        for q in d.get("quotes", []):
            quotes.append(dict(passage_id=pool[q["e"]-1]["passage_id"], text=q["text"]))
        if d.get("keep_original"):
            points = old["answer_points"]
            quotes = old["quotes"]
        else:
            points = d["points"]
        evidence_ids = list(dict.fromkeys(q["passage_id"] for q in quotes))
        if not evidence_ids:
            evidence_ids = [p["passage_id"] for p in pool]
        out = dict(old)
        out.pop("evidence", None)
        out.update(question=d.get("question", old["question"]), question_zh=d.get("question_zh", old.get("question_zh", "")),
                   answer_points=points, quotes=quotes, evidence_ids=evidence_ids,
                   candidate_pool_ids=[p["passage_id"] for p in pool],
                   qrels={p["passage_id"]: grade for p, grade in zip(pool, d["grades"], strict=True)},
                   answerability=d["answerability"], event_status=d["event_status"], as_of=d.get("as_of", old.get("as_of")),
                   recommended_for_oil_evaluation=not bool(set(d.get("issues", [])) & {"out_of_scope", "duplicate_question", "source_template", "source_revision_after_cutoff"}),
                   provenance="codex_model_annotated", status="machine_completed", event_group_review_required=True,
                   codex_review=dict(result=d["result"], note_zh=d["note"], issues=d.get("issues", []),
                                     dataset=row["dataset"], number=row["number"], at=now(),
                                     actor="current Codex conversation", model_family="Codex; exact model identifier not recorded",
                                     execution="direct conversation; no external API calls", scope="major-error review",
                                     original_sha256=row["original_sha256"], corpus_sha256=corpus_hash),
                   original_annotation=dict(answer_points=old.get("answer_points"), answerability=old.get("answerability"),
                                            event_status=old.get("event_status"), as_of=old.get("as_of"), provenance=old.get("provenance")))
        errors = validate_record(out, by_id)
        if errors:
            raise ValueError(f"{key}: {errors}")
        out["validation_errors"] = []
        results.append(out)
    publisher_hash = digest(__import__("pathlib").Path(__file__).read_bytes())
    version = digest("".join(p.read_text(encoding="utf-8") for p in decision_files) + corpus_hash + publisher_hash)[:16]
    output = DATA / "annotations/codex_runs" / version
    if output.exists():
        raise ValueError("Already published; do not overwrite the audit version")
    summary = dict(version=version, records=len(results), by_dataset=dict(Counter(r["codex_review"]["dataset"] for r in results)),
                   outcomes=dict(Counter(r["codex_review"]["result"] for r in results)),
                   answerability=dict(Counter(r["answerability"] for r in results)),
                   corpus_sha256=corpus_hash, at=now(), validation_errors=0, human_labels_created=0,
                   publisher_sha256=publisher_hash,
                   external_api_calls=0, codex_credit_cost="Not available from this annotation process; see Codex usage dashboard",
                   inputs_sha256=digest(PACK.read_bytes()))
    write_jsonl(output / "labels.jsonl", results)
    summary["labels_sha256"] = digest((output / "labels.jsonl").read_bytes())
    write_json(output / "manifest.json", summary)
    write_jsonl(DATA / "annotations/codex.jsonl", results)
    write_json(DATA / "reports/codex_annotation.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "show", "publish"])
    parser.add_argument("--dataset", default="candidates")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "show":
        show(args.dataset, args.start, args.count)
    else:
        publish()
