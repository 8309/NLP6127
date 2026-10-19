"""Local blinded second-pass evidence packages; never infers semantic judgments."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from pypdf import PdfReader
from oil_evidence.store import DATA, digest, now, read_jsonl, write_json, write_jsonl
from passage_labels import import_decisions, schema, validate_batch

CORE = list(schema()["properties"]["labels"]["items"]["properties"])
OUTCOMES = {"confirmed", "corrected", "unresolved"}


def prepare(size=30):
    base = json.loads((DATA / "reports/passage_annotation.json").read_text(encoding="utf-8"))
    labels_path = DATA / "annotations/passage_labels.jsonl"
    if digest(labels_path.read_bytes()) != base["labels_sha256"]:
        raise ValueError("Published label hash changed")
    labels = read_jsonl(labels_path)
    targets = [r for r in labels if r["needs_review"]]
    root = DATA / "annotations/passage_review_runs" / base["version"]
    if root.exists():
        raise ValueError("Review run exists; resume it without overwriting")
    passages = read_jsonl(DATA / "passages.jsonl")
    corpus_hash = digest((DATA / "passages.jsonl").read_bytes())
    if corpus_hash != base["corpus_sha256"]:
        raise ValueError("Corpus changed")
    by_id = {p["passage_id"]: p for p in passages}
    docs = {d["document_id"]: d for d in read_jsonl(DATA / "documents.jsonl")}
    groups = defaultdict(list)
    for p in passages:
        groups[p["document_id"]].append(p)
    readers, packets = {}, []
    for r in targets:
        p = by_id[r["passage_id"]]
        group = groups[p["document_id"]]
        index = next(i for i, x in enumerate(group) if x["passage_id"] == p["passage_id"])
        contexts = [dict(id="focus", text=p["text"], kind="focus", passage_id=p["passage_id"])]
        for side, pos in [("previous", index-1), ("next", index+1)]:
            if 0 <= pos < len(group):
                x = group[pos]
                contexts.append(dict(id=side, text=x["text"], kind="adjacent_passage", passage_id=x["passage_id"], page=x["page"]))
        doc = docs[p["document_id"]]
        raw = DATA / doc["raw_path"]
        if not raw.resolve().is_relative_to((DATA / "raw").resolve()):
            raise ValueError("Raw source outside local raw directory")
        source = dict(path=doc["raw_path"], sha256=doc["raw_sha256"], page=p["page"], layout_verified=False)
        if raw.suffix.lower() == ".pdf":
            if p["document_id"] not in readers:
                if digest(raw.read_bytes()) != doc["raw_sha256"]:
                    raise ValueError("Raw PDF hash changed")
                readers[p["document_id"]] = PdfReader(raw)
            reader = readers[p["document_id"]]
            page = p["page"]-1
            if 0 <= page < len(reader.pages):
                text = reader.pages[page].extract_text() or ""
                contexts.append(dict(id="raw_page", kind="raw_pdf_text_not_layout", page=p["page"], text=text))
        packets.append(dict(**{k: p.get(k) for k in ["passage_id", "document_id", "title", "source", "page", "published_date", "available_by", "date_precision", "text"]},
                            contexts=contexts, raw_source=source, url=p["url"]))
    batches = []
    for offset in range(0, len(packets), size):
        number = len(batches)+1
        path = root / "inputs" / f"batch-{number:03d}.jsonl"
        write_jsonl(path, packets[offset:offset+size])
        batches.append(dict(batch=number, count=len(packets[offset:offset+size]), input=str(path.relative_to(root)), input_sha256=digest(path.read_bytes())))
    write_jsonl(root / "original_labels.jsonl", targets)
    manifest = dict(at=now(), corpus_sha256=corpus_hash, base_version=base["version"], base_labels_sha256=base["labels_sha256"],
                    selected_labels_sha256=digest((root / "original_labels.jsonl").read_bytes()),
                    base_snapshot=str((labels_path.parent / "passage_label_runs" / corpus_hash[:16] / "published" / base["version"] / "labels.jsonl").relative_to(DATA)),
                    targets=len(targets), batches=batches, human_labels_created=0, external_api_calls=0,
                    blinded_input="No old labels, flags, review reasons or summaries; target selection is not blinded")
    write_json(root / "manifest.json", manifest)
    return dict(root=str(root), targets=len(targets), batches=len(batches))


def checked(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if digest((DATA / "passages.jsonl").read_bytes()) != manifest["corpus_sha256"]:
        raise ValueError("Corpus changed")
    if digest((DATA / manifest["base_snapshot"]).read_bytes()) != manifest["base_labels_sha256"]:
        raise ValueError("Original immutable labels changed")
    if digest((root / "original_labels.jsonl").read_bytes()) != manifest["selected_labels_sha256"]:
        raise ValueError("Selected original labels changed")
    for b in manifest["batches"]:
        if digest((root / b["input"]).read_bytes()) != b["input_sha256"]:
            raise ValueError("Blinded input changed")
    return manifest


def blind_labels(root, b):
    output = root / "outputs" / f"batch-{b['batch']:03d}.json"
    audit_path = root / "audit" / output.name
    if not output.exists() or not audit_path.exists():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if digest(output.read_bytes()) != audit["output_sha256"]:
        raise ValueError("Blind output audit mismatch")
    labels = json.loads(output.read_text(encoding="utf-8"))["labels"]
    errors = validate_batch(labels, read_jsonl(root / b["input"]))
    if errors:
        raise ValueError(str(errors))
    return labels


def compare(root):
    root = Path(root)
    manifest = checked(root)
    labels = {r["passage_id"]: r for r in read_jsonl(root / "original_labels.jsonl")}
    verified = [(b, blind_labels(root, b)) for b in manifest["batches"]]
    if any(rows is None for _, rows in verified):
        raise ValueError("All blind batches must finish before old labels are revealed")
    for b, rows in verified:
        packets = read_jsonl(root / b["input"])
        path = root / "comparisons" / f"batch-{b['batch']:03d}.jsonl"
        expected = [dict(n=i+1, evidence=p, old_label=labels[p["passage_id"]], blind_label=rows[i]) for i, p in enumerate(packets)]
        if path.exists():
            if read_jsonl(path) != expected:
                raise ValueError("Comparison changed; do not replace")
        else:
            write_jsonl(path, expected)
    return dict(revealed=True, targets=manifest["targets"])


def validate_judgments(rows, packets, old, blind):
    if sorted(r.get("n", -1) for r in rows) != list(range(1, len(packets)+1)):
        raise ValueError("One adjudication per input required")
    for row in rows:
        required = {"n", "outcome", "choice", "note", "remaining_reasons", "exclude", "support"}
        if row.get("choice") == "amended":
            required.add("amended_label")
        if set(row) != required:
            raise ValueError("Unexpected judgment fields")
        if row["outcome"] not in OUTCOMES or row["choice"] not in {"old", "blind", "amended"}:
            raise ValueError("Invalid judgment outcome/choice")
        if row["choice"] == "amended" and validate_batch([row["amended_label"]], [packets[row["n"]-1]]):
            raise ValueError("Amended label lacks valid focus evidence")
        if not isinstance(row["note"], str) or not row["note"].strip() or type(row["exclude"]) is not bool:
            raise ValueError("Missing explanation or exclusion judgment")
        if not isinstance(row["remaining_reasons"], list) or any(not isinstance(x, str) or not x.strip() for x in row["remaining_reasons"]):
            raise ValueError("Invalid unresolved reasons")
        unresolved = row["outcome"] == "unresolved"
        if unresolved != bool(row["remaining_reasons"]):
            raise ValueError("Unresolved outcome requires explicit remaining reasons; resolved outcomes must have none")
        if row["outcome"] == "confirmed" and row["choice"] != "old":
            raise ValueError("Confirmed preserves original substantive label")
        if row["outcome"] == "corrected":
            chosen = row.get("amended_label") if row["choice"] == "amended" else blind[row["n"]-1]
            if row["choice"] == "old" or all(old[row["n"]-1][k] == chosen[k] for k in CORE if k not in {"needs_review", "review_reasons"}):
                raise ValueError("Corrected requires an explicitly different blind label")
        texts = {c["id"]: c["text"] for c in packets[row["n"]-1]["contexts"]}
        if not row["support"] or any(not isinstance(s, list) or len(s) != 2 or s[0] not in texts or not s[1] or s[1] not in texts[s[0]] for s in row["support"]):
            raise ValueError("Judgment requires exact context quote support")


def import_judgments(root, number, path, actor):
    root = Path(root)
    m = checked(root)
    b = next(b for b in m["batches"] if b["batch"] == number)
    comp = root / "comparisons" / f"batch-{number:03d}.jsonl"
    packets = read_jsonl(comp)
    if not packets:
        raise ValueError("Old labels may only be revealed after blinded phase")
    original = {r["passage_id"]: r for r in read_jsonl(root / "original_labels.jsonl")}
    blind = blind_labels(root, b)
    expected = [dict(n=i+1, evidence=p, old_label=original[p["passage_id"]], blind_label=blind[i])
                for i, p in enumerate(read_jsonl(root / b["input"]))]
    if packets != expected:
        raise ValueError("Comparison evidence/labels changed")
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_judgments(rows, [r["evidence"] for r in packets], [r["old_label"] for r in packets], [r["blind_label"] for r in packets])
    output = root / "judgments" / f"batch-{number:03d}.json"
    if output.exists():
        raise ValueError("Adjudications exist; preserve them")
    write_json(output, dict(judgments=sorted(rows, key=lambda r: r["n"])))
    write_json(root / "judgment_audit" / output.name, dict(at=now(), actor=actor, output_sha256=digest(output.read_bytes()),
               comparison_sha256=digest(comp.read_bytes()), decisions_sha256=digest(Path(path).read_bytes()), human_labels_created=0, external_api_calls=0))
    return dict(batch=number, reviewed=len(rows))


def progress(root):
    root = Path(root)
    m = checked(root)
    blind, adjudicated, outcomes = 0, 0, Counter()
    for b in m["batches"]:
        labels = blind_labels(root, b)
        blind += len(labels or [])
        path = root / "judgments" / f"batch-{b['batch']:03d}.json"
        audit_path = root / "judgment_audit" / path.name
        if path.exists() and audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            comp = root / "comparisons" / f"batch-{b['batch']:03d}.jsonl"
            if digest(path.read_bytes()) != audit["output_sha256"] or digest(comp.read_bytes()) != audit["comparison_sha256"]:
                raise ValueError("Judgment audit mismatch")
            packets = read_jsonl(comp)
            rows = json.loads(path.read_text(encoding="utf-8"))["judgments"]
            validate_judgments(rows, [p["evidence"] for p in packets], [p["old_label"] for p in packets], [p["blind_label"] for p in packets])
            rows, _ = effective_judgments(root, b["batch"])
            adjudicated += len(rows)
            outcomes.update(r["outcome"] for r in rows)
    report = dict(at=now(), total=m["targets"], blind_completed=blind, reviewed=adjudicated, remaining=m["targets"]-adjudicated,
                  outcomes=dict(outcomes), status="complete" if adjudicated == m["targets"] else "incomplete", human_labels_created=0, external_api_calls=0)
    write_json(root / "progress.json", report)
    write_json(DATA / "reports/passage_review_progress.json", report)
    return report


def effective_judgments(root, number):
    """Read an append-only parent revision without changing accepted worker files."""
    root = Path(root)
    name = f"batch-{number:03d}.json"
    original = root / "judgments" / name
    original_audit = root / "judgment_audit" / name
    audit = json.loads(original_audit.read_text(encoding="utf-8"))
    original_actor = audit["actor"]
    if digest(original.read_bytes()) != audit["output_sha256"]:
        raise ValueError("Original judgment audit mismatch")
    path = original
    head = root / "judgment_revision_heads" / name
    if head.exists():
        info = json.loads(head.read_text(encoding="utf-8"))
        folder = (root / info["revision"]).resolve()
        if not folder.is_relative_to((root / "judgment_revisions").resolve()):
            raise ValueError("Revision outside run directory")
        path = folder / "judgments.json"
        audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
        if audit["base_output_sha256"] != digest(original.read_bytes()) or audit["output_sha256"] != digest(path.read_bytes()):
            raise ValueError("Revision audit mismatch")
    comp = root / "comparisons" / name.replace(".json", ".jsonl")
    if audit["comparison_sha256"] != digest(comp.read_bytes()):
        raise ValueError("Revision comparison mismatch")
    packets = read_jsonl(comp)
    rows = json.loads(path.read_text(encoding="utf-8"))["judgments"]
    validate_judgments(rows, [p["evidence"] for p in packets], [p["old_label"] for p in packets], [p["blind_label"] for p in packets])
    actor = {r["n"]: audit["actor"] if r["n"] in audit["changed_records"] else original_actor for r in rows} if head.exists() else original_actor
    return rows, actor


def revise_judgments(root, number, path, actor, reason):
    root = Path(root)
    checked(root)
    if (root / "published").exists():
        raise ValueError("Published review revisions require a new run")
    if not reason.strip():
        raise ValueError("Revision reason required")
    old_rows, _ = effective_judgments(root, number)
    name = f"batch-{number:03d}.json"
    comp = root / "comparisons" / name.replace(".json", ".jsonl")
    packets = read_jsonl(comp)
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_judgments(rows, [p["evidence"] for p in packets], [p["old_label"] for p in packets], [p["blind_label"] for p in packets])
    version = digest(Path(path).read_bytes())[:16]
    folder = root / "judgment_revisions" / name.removesuffix(".json") / version
    head = root / "judgment_revision_heads" / name
    if folder.exists() or head.exists():
        raise ValueError("Revision exists; preserve it")
    write_json(folder / "judgments.json", dict(judgments=sorted(rows, key=lambda r: r["n"])))
    write_json(folder / "audit.json", dict(at=now(), actor=actor, reason=reason,
        changed_records=[r["n"] for r in rows if r != next(o for o in old_rows if o["n"] == r["n"])],
        base_output_sha256=digest((root / "judgments" / name).read_bytes()),
        output_sha256=digest((folder / "judgments.json").read_bytes()),
        comparison_sha256=digest(comp.read_bytes()), decisions_sha256=digest(Path(path).read_bytes()),
        human_labels_created=0, external_api_calls=0))
    write_json(head, dict(revision=str(folder.relative_to(root))))
    return dict(batch=number, revision=version, reviewed=len(rows))


def publish(root):
    root = Path(root)
    report = progress(root)
    if report["status"] != "complete":
        raise ValueError("Cannot publish incomplete review")
    m = checked(root)
    replacements, reviews = {}, []
    for b in m["batches"]:
        comp = read_jsonl(root / "comparisons" / f"batch-{b['batch']:03d}.jsonl")
        rows, actor = effective_judgments(root, b["batch"])
        for row in rows:
            row_actor = actor[row["n"]] if isinstance(actor, dict) else actor
            c = comp[row["n"]-1]
            previous = c["old_label"]
            chosen = row["amended_label"] if row["choice"] == "amended" else c[row["choice"] + "_label"]
            label = dict(previous, **{k: chosen[k] for k in CORE})
            label.update(needs_review=row["outcome"] == "unresolved", review_reasons=row["remaining_reasons"])
            label.update(second_pass_outcome=row["outcome"], second_pass_actor=row_actor, exclude_from_strict_evaluation=row["exclude"])
            label["original_annotation_actor"] = previous.get("annotation_actor")
            if row["choice"] != "old":
                label["annotation_actor"] = row_actor
            replacements[label["passage_id"]] = label
            reviews.append(dict(passage_id=label["passage_id"], **row, actor=row_actor, batch=b["batch"], provenance="codex_model_second_pass", human_reviewed=False,
                                base_version=m["base_version"], changed_fields=[k for k in CORE if previous[k] != label[k]], context_layout_verified=False))
    labels = [replacements.get(r["passage_id"], r) for r in read_jsonl(DATA / m["base_snapshot"])]
    fields = [{k: r[k] for k in CORE} for r in labels]
    errors = validate_batch(fields, read_jsonl(DATA / "passages.jsonl"))
    if errors:
        raise ValueError(str(errors))
    version = digest(json.dumps(reviews, sort_keys=True, ensure_ascii=False))[:16]
    snapshot = root / "published" / version
    if snapshot.exists():
        raise ValueError("Review snapshot exists; never overwrite")
    write_jsonl(snapshot / "labels.jsonl", labels)
    write_jsonl(snapshot / "reviews.jsonl", reviews)
    report.update(version=version, base_version=m["base_version"], corpus_sha256=m["corpus_sha256"], validated=len(labels),
                  labels_sha256=digest((snapshot / "labels.jsonl").read_bytes()), reviews_sha256=digest((snapshot / "reviews.jsonl").read_bytes()),
                  needs_review=sum(r["needs_review"] for r in labels), excluded=sum(r["exclude"] for r in reviews),
                  topics=dict(Counter(t for r in labels for t in r["topics"])), states=dict(Counter(r["event_status"] for r in labels)),
                  relevance=dict(Counter(r["oil_relevance"] for r in labels)), quantities=sum(len(r["quantities"]) for r in labels), events=sum(len(r["events"]) for r in labels),
                  strict_evaluation_exclusion="Advisory label; does not delete sources or automatically filter online retrieval",
                  publisher_sha256=digest(Path(__file__).read_bytes()), codex_credit_cost="Subscription usage not measured")
    write_json(snapshot / "manifest.json", report)
    write_jsonl(DATA / "annotations/passage_labels.jsonl", labels)
    write_jsonl(DATA / "annotations/passage_review.jsonl", reviews)
    write_json(DATA / "reports/passage_review.json", report)
    write_json(DATA / "reports/passage_annotation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "progress", "compare", "import", "judge", "publish"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--actor", default="Codex second pass")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare()
    elif args.action == "import":
        result = import_decisions(args.root, args.batch, args.decisions, actor=args.actor, refresh_progress=False)
    elif args.action == "judge":
        result = import_judgments(args.root, args.batch, args.decisions, args.actor)
    else:
        result = globals()[args.action](args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
