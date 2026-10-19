"""Audit actual source-first questions without creating gold labels or padding.

Completed immutable authoring attempts are hash checked. Historical execution
code is retained, not rebound to today's implementation. Cross-attempt duplicate
and scope decisions are recorded separately; raw model responses stay untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from uuid import uuid4

import author_machine_questions as author
import prepare_machine_cohort as preparation
import run_codex_annotation as native
from oil_evidence import machine_evaluation as protocol

QUERY_FIELDS = {"record_id", "question", "as_of", "event_group", "split"}


def check_query(row):
    if not isinstance(row, dict) or set(row) != QUERY_FIELDS:
        raise ValueError("Cohort query contains missing fields or authoring/label hints")
    clean = protocol.clean_question(row)
    if clean != row:
        raise ValueError("Question differs from its strict clean representation")
    return clean


def aggregate_records(candidates, previous_signatures=(), original=()):
    """Pure admission audit; metadata and author anchors never become answers."""
    signatures = set(previous_signatures)
    templates = {author.nonnumeric_signature(value) for value in signatures}
    ids, admitted, audits, decisions = set(), [], [], []
    packet_admissions = Counter()
    original_ids = set()
    for question, audit in original:
        question = check_query(question)
        if question["split"] != "development" or question["record_id"] in ids:
            raise ValueError("Original development membership or uniqueness differs")
        ids.add(question["record_id"])
        original_ids.add(question["record_id"])
        admitted.append(question)
        audits.append(audit)
        signatures.add(preparation.normalized(question["question"]))
        templates.add(author.nonnumeric_signature(question["question"]))
    for question, audit, execution in candidates:
        question = check_query(question)
        if audit.get("record_id") != question["record_id"] or audit.get("human_gold") is not False:
            raise ValueError("Question and machine authoring audit membership differ")
        exact = preparation.normalized(question["question"])
        template = author.nonnumeric_signature(question["question"])
        reasons = []
        packet_id = audit.get("packet_id")
        if not isinstance(packet_id, str) or not packet_id:
            raise ValueError("New question candidate lacks its exact source-packet lineage")
        if packet_admissions[packet_id] >= 2:
            reasons.append("two_admitted_questions_per_native_source_packet_limit")
        if question["record_id"] in ids:
            reasons.append("duplicate_record_id")
        if exact in signatures:
            reasons.append("normalized_question_duplicate")
        if template in templates:
            reasons.append("nonnumeric_question_template_duplicate")
        reasons.extend(author.scope_flags(question))
        # Even a held or rejected prior candidate remains in the duplicate audit.
        ids.add(question["record_id"])
        signatures.add(exact)
        templates.add(template)
        status = "held_question_candidate_not_label" if reasons else "admitted_question_candidate_not_label"
        decisions.append(dict(record_id=question["record_id"], split=question["split"],
                              status=status, reasons=reasons, authoring_execution=execution,
                              human_gold=False))
        if not reasons:
            admitted.append(question)
            audits.append(audit)
            packet_admissions[packet_id] += 1
    groups, documents = defaultdict(set), defaultdict(set)
    for question, audit in zip(admitted, audits, strict=True):
        groups[question["event_group"]].add(question["split"])
        for anchor in audit.get("anchors", []):
            documents[anchor["document_id"]].add(question["split"])
    if any(len(value) != 1 for value in groups.values()) or any(len(value) != 1 for value in documents.values()):
        raise ValueError("Authoring event family or source document crosses split")
    counts = Counter(question["split"] for question in admitted)
    return dict(questions=admitted, authoring=audits, decisions=decisions,
                actual_counts=dict(counts), original_development_count=len(original_ids),
                new_candidate_count=len(candidates), held_candidate_count=sum(bool(row["reasons"]) for row in decisions),
                targets_met=counts["development"] >= 200 and counts["test"] >= 1000,
                target_shortfall={key: max(0, goal - counts[key]) for key, goal in preparation.TARGETS.items()},
                event_family_cross_split=False, source_document_cross_split=False,
                semantic_independence_established=False, question_quality_established=False,
                qa_annotations_created=0, human_gold=False, formal_evaluation_frozen=False)


def completed_attempt(path, preparation_receipt_sha256):
    receipt = native.read_json(path / "receipt.json")
    specification = native.read_json(path / "specification.json")
    split = specification.get("authoring_split", "development")
    if (split not in {"development", "test"}
            or receipt.get("status") != split + "_question_candidates_authored_not_labels_or_frozen_cohort"
            or receipt.get("original_inputs_unchanged") is not True
            or receipt.get("human_gold") is not False
            or receipt.get("answer_annotations_created") != 0
            or specification.get("preparation_receipt_sha256") != preparation_receipt_sha256):
        raise ValueError("Expected a completed bound source-first authoring execution")
    for name, expected in receipt["artifact_sha256"].items():
        artifact = native.contained_file(path, name)
        if protocol.sha256(artifact.read_bytes()) != expected:
            raise ValueError("Authoring artifact changed: " + name)
    questions = protocol.parse_jsonl((path / "questions.jsonl").read_bytes(), "authored questions")
    anchors = protocol.parse_jsonl((path / "authoring_anchors.jsonl").read_bytes(), "authoring anchors")
    if (len(questions) != receipt["actual_validated_questions"] or len(questions) != len(anchors)
            or any(row["split"] != split for row in questions)):
        raise ValueError("Actual question count or split differs from execution receipt")
    by_id = {row["record_id"]: row for row in anchors}
    if len(by_id) != len(anchors) or set(by_id) != {row["record_id"] for row in questions}:
        raise ValueError("Duplicate/missing authoring audit membership")
    return specification, receipt, [(row, by_id[row["record_id"]], str(path)) for row in questions]


def original_development(root):
    development = root / "data/development_runs/oil-dev50-v1"
    current = native.read_json(development / "current.json")
    manifest_path = native.contained_file(development, current["path"])
    if protocol.sha256(manifest_path.read_bytes()) != current["sha256"]:
        raise ValueError("Original development receipt changed")
    manifest = native.read_json(manifest_path)
    for name, expected in manifest["artifact_sha256"].items():
        artifact = native.contained_file(manifest_path.parent, name)
        if protocol.sha256(artifact.read_bytes()) != expected:
            raise ValueError("Original development artifact changed")
    raw = native.read_json(manifest_path.parent / "questions.json")
    audits = protocol.parse_jsonl((manifest_path.parent / "authoring_anchors.jsonl").read_bytes(), "original authoring")
    by_id = {row["question_id"]: row for row in audits}
    if len(raw) != 50 or len(by_id) != 50 or set(by_id) != {row["question_id"] for row in raw}:
        raise ValueError("Expected exactly 50 original unlabelled development queries")
    return [(protocol.clean_question(row), dict(by_id[row["question_id"]], record_id=row["question_id"])) for row in raw]


def select_question_core(questions, authoring, targets=None):
    """Select exact target counts in existing authoring order, never by labels."""
    targets = preparation.TARGETS if targets is None else targets
    counts = Counter(row["split"] for row in questions)
    if any(counts[split] < target for split, target in targets.items()):
        raise ValueError("Real admitted question targets are not met; no padding or lowered count guard")
    audits = {row["record_id"]: row for row in authoring}
    if len(audits) != len(authoring) or set(audits) != {row["record_id"] for row in questions}:
        raise ValueError("Question selection source-audit membership differs")
    selected, reserves, selected_counts = [], [], Counter()
    for row in questions:
        check_query(row)
        if selected_counts[row["split"]] < targets[row["split"]]:
            selected.append(row)
            selected_counts[row["split"]] += 1
        else:
            reserves.append(row)
    selected_audits = [audits[row["record_id"]] for row in selected]
    preparation.validate_final_cohort(selected, selected_audits, required_counts=targets)
    return selected, selected_audits, reserves


def freeze_question_selection(root, parent_cohort):
    """Lock questions only. Formal machine gold requires future paired labels."""
    import retrieve_machine_questions as retrieval
    root, parent_cohort = Path(root).resolve(), Path(parent_cohort).resolve()
    questions, protected, parent_receipt = retrieval.load_cohort(root, parent_cohort)
    authoring = protocol.parse_jsonl((parent_cohort / "private_authoring_audit.jsonl").read_bytes(), "private source audit")
    selected, audits, reserves = select_question_core(questions, authoring)
    output = root / "data/machine_runs/question-cohorts" / (native.utc_now().replace(":", "-") + "-" + uuid4().hex[:10])
    output.mkdir(exist_ok=False)
    for name, rows in (("questions.jsonl", selected), ("private_authoring_audit.jsonl", audits),
                       ("reserve_questions.jsonl", reserves)):
        native.write_new(output / name, b"".join(protocol.canonical_bytes(row) + b"\n" for row in rows))
    for split in ("development", "test"):
        native.write_new(output / (split + "_questions.jsonl"), b"".join(protocol.canonical_bytes(row) + b"\n"
                         for row in selected if row["split"] == split))
    for name in ("attempts.json", "admission_decisions.jsonl"):
        native.write_new(output / name, (parent_cohort / name).read_bytes())
    native.write_new(output / "aggregation_script.py", Path(__file__).read_bytes())
    inputs = native.read_json(parent_cohort / "inputs.json")
    inputs.update(parent_candidate_cohort=str(parent_cohort), parent_candidate_receipt_sha256=protocol.sha256(
        (parent_cohort / "receipt.json").read_bytes()), selection_rule="first_admitted_in_fixed_authoring_order_per_split_not_labels_or_ranks",
        protected_sha256=protected)
    native.write_new(output / "inputs.json", inputs)
    native.protected_unchanged(protected)
    summary = dict(parent_receipt, output=str(output), actual_counts=dict(preparation.TARGETS),
        parent_candidate_cohort=str(parent_cohort), question_selection_locked=True,
        selected_questions=len(selected), reserve_questions=len(reserves), qa_annotations_created=0,
        human_gold=False, formal_evaluation_frozen=False,
        selection_rule="first_admitted_in_fixed_authoring_order_per_split_not_labels_or_ranks",
        artifact_sha256={path.name: protocol.sha256(path.read_bytes()) for path in output.iterdir() if path.is_file()})
    native.write_new(output / "receipt.json", summary)
    return output, summary


def run(root, prepared, *, include_original=True, additional_preparations=()):
    root, prepared = Path(root).resolve(), Path(prepared).resolve()
    packets, previous, _, protected, preparation_receipt = author.load_preparation(root, prepared)
    prep_hash = protocol.sha256((prepared / "receipt.json").read_bytes())
    preparations = [dict(path=str(prepared), receipt_sha256=prep_hash)]
    preparation_indices = {prep_hash: 0}
    for extra in additional_preparations:
        extra = Path(extra).resolve()
        extra_packets, _, _, extra_protected, extra_receipt = author.load_preparation(root, extra)
        if extra_receipt.get("original_preparation_receipt_sha256") != prep_hash:
            raise ValueError("Additional preparation does not retain the original cohort lineage")
        extra_hash = protocol.sha256((extra / "receipt.json").read_bytes())
        if extra_hash in preparation_indices:
            raise ValueError("Repeated supplemental preparation")
        preparation_indices[extra_hash] = len(preparations)
        preparations.append(dict(path=str(extra), receipt_sha256=extra_hash))
        packets.extend(extra_packets)
        protected.update(extra_protected)
    attempts, candidates, attempt_records = [], [], []
    parent = root / "data/machine_runs/question-authoring"
    for path in parent.iterdir():
        if not path.is_dir() or not (path / "receipt.json").exists() or not (path / "specification.json").exists():
            continue
        specification = native.read_json(path / "specification.json")
        attempt_preparation = specification.get("preparation_receipt_sha256")
        if attempt_preparation not in preparation_indices:
            continue
        spec, receipt, rows = completed_attempt(path, attempt_preparation)
        attempts.append((bool(receipt.get("augmentation_protocol")), preparation_indices[attempt_preparation], spec.get("authoring_split", "development"),
                         spec.get("source_packet_offset", spec.get("development_packet_offset", 0)), path.name, rows, receipt))
    for augmentation_phase, preparation_index, split, offset, name, rows, receipt in sorted(attempts):
        candidates.extend(rows)
        path = parent / name
        attempt_records.append(dict(path=str(path), split=split, source_packet_offset=offset,
                                    augmentation_phase=augmentation_phase,
                                    preparation_receipt_sha256=preparations[preparation_index]["receipt_sha256"],
                                    receipt_sha256=protocol.sha256((path / "receipt.json").read_bytes()),
                                    actual_questions=len(rows), raw_generated_records=receipt["raw_generated_records"],
                                    source_packets_selected=receipt["source_packets_selected"],
                                    subscription_usage=receipt["subscription_usage"],
                                    observed_identity=receipt["observed_identity"]))
    originals = original_development(root) if include_original else []
    original_docs = {anchor["document_id"] for _, audit in originals for anchor in audit["anchors"]}
    source_docs = {passage["document_id"] for packet in packets for passage in packet["evidence"]}
    if original_docs & source_docs:
        raise ValueError("Source packets overlap original development authoring documents")
    result = aggregate_records(candidates, previous, originals)
    output_parent = root / "data/machine_runs/question-cohorts"
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / (native.utc_now().replace(":", "-") + "-" + uuid4().hex[:10])
    output.mkdir(exist_ok=False)
    for split in ("development", "test"):
        native.write_new(output / (split + "_questions.jsonl"), b"".join(protocol.canonical_bytes(row) + b"\n"
                         for row in result["questions"] if row["split"] == split))
    for name, rows in (("questions.jsonl", result["questions"]), ("private_authoring_audit.jsonl", result["authoring"]),
                       ("admission_decisions.jsonl", result["decisions"])):
        native.write_new(output / name, b"".join(protocol.canonical_bytes(row) + b"\n" for row in rows))
    native.write_new(output / "attempts.json", protocol.canonical_bytes(attempt_records))
    native.write_new(output / "aggregation_script.py", Path(__file__).read_bytes())
    native.write_new(output / "inputs.json", protocol.canonical_bytes(dict(preparation=str(prepared),
                     preparations=preparations,
                     preparation_receipt_sha256=prep_hash, protected_sha256=protected,
                     original_source_documents_excluded=len(original_docs), source_aliases_reserved_by_preparation=True)))
    preparation.assert_unchanged(protected, root)
    summary = {key: value for key, value in result.items() if key not in {"questions", "authoring", "decisions"}}
    summary.update(status="actual_question_candidates_aggregated_not_gold_or_frozen_evaluation",
                   output=str(output), completed_authoring_calls=len(attempt_records), original_inputs_unchanged=True,
                   source_selection_precedes_retrieval=True, preparation_receipt_sha256=prep_hash,
                   preparations=preparations,
                   artifact_sha256={path.name: protocol.sha256(path.read_bytes()) for path in output.iterdir() if path.is_file()},
                   limitations=["Counts are actual admitted questions, not plans or source-packet counts.",
                                "Exact and nonnumeric duplicate checks do not establish exhaustive semantic uniqueness.",
                                "Conservative family/source exclusions are not proof of semantic event independence.",
                                "No relevance, answerability, answer gold or human labels are created."])
    native.write_new(output / "receipt.json", protocol.canonical_bytes(summary))
    print(json.dumps({key: value for key, value in summary.items() if key != "artifact_sha256"}, indent=2), flush=True)
    return output, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--exclude-original-development", action="store_true")
    parser.add_argument("--additional-preparation", type=Path, action="append", default=[])
    args = parser.parse_args()
    run(args.canonical_root, args.preparation, include_original=not args.exclude_original_development,
        additional_preparations=args.additional_preparation)
