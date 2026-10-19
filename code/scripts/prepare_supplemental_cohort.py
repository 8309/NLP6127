"""Seal additional original paragraphs as a distinct source-only preparation.

Use exactly the earlier authoring prompt, unchanged family partitions and prior
exposure reservations. Completed question strings extend generator deduplication,
but are never converted to answers, relevance judgments or authoring hints.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from uuid import uuid4

import aggregate_machine_questions as aggregation
import author_machine_questions as author
import inspect_supplemental_sources as inspection
import prepare_machine_cohort as preparation
import run_codex_annotation as native
from oil_evidence import machine_evaluation as protocol


def validate_supplement(packets, original, families):
    seen = {row["passage_id"] for packet in original for row in packet["evidence"]}
    split_by_family = {packet["event_family_id"]: packet["candidate_split"] for packet in original}
    counts = Counter(row["document_id"] for packet in original for row in packet["evidence"])
    packet_ids = set()
    for packet in packets:
        family_id = packet["event_family_id"]
        family = families.get(family_id)
        if (not family or family["reserved_due_to_prior_exposure"] is not False
                or split_by_family.get(family_id) != packet["candidate_split"]
                or packet["provenance"] != "accepted_source_packet_not_question_or_label"
                or packet["human_gold"] is not False or packet["questions_created"] != 0
                or packet["packet_id"] in packet_ids or len(packet["evidence"]) != 1):
            raise ValueError("Supplement changes a family partition/reservation or packet provenance")
        packet_ids.add(packet["packet_id"])
        for row in packet["evidence"]:
            if row["passage_id"] in seen or row["document_id"] not in family["document_ids"]:
                raise ValueError("Supplement repeats an original paragraph or changes source-family membership")
            seen.add(row["passage_id"])
            counts[row["document_id"]] += 1
            if counts[row["document_id"]] > 6:
                raise ValueError("Supplement exceeds six original paragraphs per document")
    return dict(source_packets=len(packets), candidate_packets_by_split=dict(Counter(packet["candidate_split"] for packet in packets)),
                actual_qa_questions=0, actual_qa_labels=0, desired_qa_counts=preparation.TARGETS,
                total_original_paragraph_cap=6, source_corpus_resplit=False, event_partitions_unchanged=True,
                prior_exposure_reservations_unchanged=True, semantic_task_distinctness_established=False,
                source_packet_capacity_is_not_qa_yield=True)


def run(root, original_preparation, feasibility):
    root, original_preparation, feasibility = Path(root).resolve(), Path(original_preparation).resolve(), Path(feasibility).resolve()
    original, previous, _, protected, original_receipt = author.load_preparation(root, original_preparation)
    allowed = (root / "data/machine_runs/cohort-supplement-feasibility").resolve()
    if not feasibility.is_relative_to(allowed) or feasibility == allowed:
        raise ValueError("Use a unique audited supplemental-source feasibility attempt")
    receipt = native.read_json(feasibility / "receipt.json")
    if (receipt.get("status") != "supplemental_native_source_feasibility_not_qa_or_authoring_run"
            or receipt.get("original_inputs_unchanged") is not True or receipt.get("actual_new_questions") != 0
            or receipt.get("family_reservations_weakened") is not False):
        raise ValueError("Expected unchanged source-only supplemental feasibility")
    for name, expected in receipt["artifact_sha256"].items():
        path = native.contained_file(feasibility, name)
        if protocol.sha256(path.read_bytes()) != expected:
            raise ValueError("Supplemental feasibility artifact changed")
        protected[str(path)] = expected
    inputs = native.read_json(feasibility / "inputs.json")
    original_hash = protocol.sha256((original_preparation / "receipt.json").read_bytes())
    if inputs["preparation_receipt_sha256"] != original_hash:
        raise ValueError("Supplemental source preparation lineage differs")
    packets = protocol.parse_jsonl((feasibility / "supplemental_source_packets.jsonl").read_bytes(), "supplemental source packets")
    families = native.read_json(original_preparation / "event_family_reservations.json")
    statistics = validate_supplement(packets, original, families)
    completed = []
    for path in (root / "data/machine_runs/question-authoring").iterdir():
        if not (path / "receipt.json").is_file() or not (path / "specification.json").is_file():
            continue
        specification = native.read_json(path / "specification.json")
        if specification.get("preparation_receipt_sha256") != original_hash:
            continue
        _, _, rows = aggregation.completed_attempt(path, original_hash)
        previous.extend(preparation.normalized(question["question"]) for question, _, _ in rows)
        receipt_path = path / "receipt.json"
        protected[str(receipt_path)] = protocol.sha256(receipt_path.read_bytes())
        question_path = path / "questions.jsonl"
        protected[str(question_path)] = protocol.sha256(question_path.read_bytes())
        completed.append(str(path))
    parent = root / "data/machine_runs/cohort-preparation"
    output = parent / (native.utc_now().replace(":", "-") + "-" + uuid4().hex[:10])
    output.mkdir(exist_ok=False)
    native.write_new(output / "source_packets.jsonl", b"".join(protocol.canonical_bytes(packet) + b"\n" for packet in packets))
    native.write_new(output / "event_family_reservations.json", (original_preparation / "event_family_reservations.json").read_bytes())
    native.write_new(output / "question_generator_prompt.txt", (original_preparation / "question_generator_prompt.txt").read_bytes())
    native.write_new(output / "excluded_question_signatures.json", protocol.canonical_bytes(dict(english_question_signatures=sorted(set(previous)))))
    native.write_new(output / "inputs.json", protocol.canonical_bytes(dict(protected_sha256=protected, bindings=original_receipt["bindings"],
        original_preparation=str(original_preparation), original_preparation_receipt_sha256=original_hash,
        supplemental_feasibility=str(feasibility), completed_question_executions_reserved=completed)))
    native.write_new(output / "preparation_script.py", Path(__file__).read_bytes())
    preparation.assert_unchanged(protected, root)
    summary = dict(schema_version="machine-source-cohort-preparation-supplement-v1",
        status="source_packets_prepared_not_questions_or_labels", output=str(output), bindings=original_receipt["bindings"],
        statistics=statistics, questions_created=0, qa_annotations_created=0, human_gold=False,
        protected_files_unchanged=True, native_model_calls=0, original_preparation_receipt_sha256=original_hash,
        unchanged_authoring_prompt_sha256=protocol.sha256((output / "question_generator_prompt.txt").read_bytes()),
        artifact_sha256={path.name: protocol.sha256(path.read_bytes()) for path in output.iterdir() if path.is_file()})
    native.write_new(output / "receipt.json", protocol.canonical_bytes(summary))
    print(json.dumps(summary, indent=2), flush=True)
    return output, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--original-preparation", type=Path, required=True)
    parser.add_argument("--feasibility", type=Path, required=True)
    args = parser.parse_args()
    run(args.canonical_root, args.original_preparation, args.feasibility)
