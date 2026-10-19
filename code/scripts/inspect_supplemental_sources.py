"""Inspect unselected original paragraphs under unchanged exposure reservations.

Raise only the local per-document selection cap from three to six. This neither
resplits the accepted corpus nor weakens conservative event-family holds. Source
packets are not questions, labels, established independent events, or QA yield.
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


def supplemental_packets(passages, pages, existing_packets, families, *, max_per_document=6):
    if type(max_per_document) is not int or max_per_document != 6:
        raise ValueError("Supplemental feasibility fixes the total native-paragraph cap at six per document")
    assignment, splits = {}, {}
    for family_id, family in families.items():
        for identifier in family["document_ids"]:
            if identifier in assignment:
                raise ValueError("Document belongs to multiple source families")
            assignment[identifier] = family_id
    existing_ids = {row["passage_id"] for packet in existing_packets for row in packet["evidence"]}
    texts = {preparation.normalized(row["text"]) for packet in existing_packets for row in packet["evidence"]}
    counts = Counter(row["document_id"] for packet in existing_packets for row in packet["evidence"])
    for packet in existing_packets:
        family_id, split = packet["event_family_id"], packet["candidate_split"]
        if family_id in splits and splits[family_id] != split:
            raise ValueError("Existing source family crosses partition")
        splits[family_id] = split
    selected, rejected, candidates = [], Counter(), defaultdict(list)
    for passage in passages:
        if passage["passage_id"] in existing_ids:
            rejected["already_selected_original_paragraph"] += 1
            continue
        family_id = assignment.get(passage["document_id"])
        if family_id is None:
            raise ValueError("Accepted paragraph has no frozen source-family reservation")
        family = families[family_id]
        if family["reserved_due_to_prior_exposure"]:
            rejected["unchanged_prior_exposure_family_reservation"] += 1
            continue
        available = passage.get("available_by")
        if not preparation.valid_day(available) or not preparation.WINDOW[0] <= available <= preparation.WINDOW[1]:
            rejected["unknown_or_out_of_window_availability"] += 1
            continue
        domain = preparation.passage_domain(passage)
        if domain is None:
            rejected["insufficient_standalone_oil_task_context"] += 1
            continue
        preparation.validate_source_span(passage, pages)
        signature = preparation.normalized(passage["text"])
        if signature in texts:
            rejected["repeated_native_paragraph_text"] += 1
            continue
        candidates[passage["document_id"]].append((passage, domain, signature))
    eligible = sum(len(rows) for rows in candidates.values())
    for document_id in sorted(candidates):
        for passage, domain, signature in sorted(candidates[document_id], key=lambda row: (-row[1][1], row[0]["passage_id"])):
            if counts[document_id] >= max_per_document:
                rejected["six_native_paragraph_cap"] += 1
                continue
            if signature in texts:
                rejected["repeated_native_paragraph_text"] += 1
                continue
            family_id = assignment[document_id]
            if family_id not in splits:
                raise ValueError("Supplement must retain a previously selected family partition")
            evidence = [{key: passage[key] for key in preparation.SOURCE_FIELDS if key in passage}]
            selected.append(dict(packet_id="source-packet-" + protocol.sha256(protocol.canonical_bytes(
                dict(family=family_id, ids=[passage["passage_id"]])))[:20],
                event_family_id=family_id, candidate_split=splits[family_id], source=passage["source"],
                publication_year=passage["available_by"][:4], primary_domain=domain[0], task_domains=domain[2],
                evidence=evidence, provenance="accepted_source_packet_not_question_or_label", human_gold=False,
                questions_created=0, uncertainty_flags=families[family_id]["uncertainty_flags"] +
                ["same_document_additional_native_paragraph_semantic_task_distinctness_unverified"]))
            counts[document_id] += 1
            texts.add(signature)
    selected = preparation.balanced_packets(selected)
    return selected, dict(accepted_corpus_passages=len(passages), original_source_packets=len(existing_packets),
                          unselected_eligible_native_paragraphs=eligible, supplemental_packets=len(selected),
                          supplemental_by_split=dict(Counter(packet["candidate_split"] for packet in selected)),
                          supplemental_documents=len({row["document_id"] for packet in selected for row in packet["evidence"]}),
                          total_native_paragraph_cap=6, rejection_counts=dict(rejected),
                          source_corpus_resplit=False, family_reservations_weakened=False,
                          actual_new_questions=0, qa_annotations_created=0, human_gold=False,
                          semantic_task_distinctness_established=False, source_packet_counts_are_not_qa_yield=True)


def run(root, prepared):
    root, prepared = Path(root).resolve(), Path(prepared).resolve()
    existing, _, _, protected, _ = author.load_preparation(root, prepared)
    passages, _, pages, _, canonical, bindings, _ = preparation.verified_inputs(root)
    protected.update(canonical)
    families = native.read_json(prepared / "event_family_reservations.json")
    packets, summary = supplemental_packets(passages, pages, existing, families)
    parent = root / "data/machine_runs/cohort-supplement-feasibility"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / (native.utc_now().replace(":", "-") + "-" + uuid4().hex[:10])
    output.mkdir(exist_ok=False)
    native.write_new(output / "supplemental_source_packets.jsonl", b"".join(protocol.canonical_bytes(packet) + b"\n" for packet in packets))
    native.write_new(output / "inputs.json", protocol.canonical_bytes(dict(preparation=str(prepared),
                     preparation_receipt_sha256=protocol.sha256((prepared / "receipt.json").read_bytes()),
                     bindings=bindings, protected_sha256=protected)))
    native.write_new(output / "inspection_script.py", Path(__file__).read_bytes())
    preparation.assert_unchanged(protected, root)
    summary.update(status="supplemental_native_source_feasibility_not_qa_or_authoring_run", output=str(output),
                   original_inputs_unchanged=True, model_calls=0,
                   artifact_sha256={path.name: protocol.sha256(path.read_bytes()) for path in output.iterdir() if path.is_file()})
    native.write_new(output / "receipt.json", protocol.canonical_bytes(summary))
    print(json.dumps(summary, indent=2), flush=True)
    return output, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args()
    run(args.canonical_root, args.preparation)
