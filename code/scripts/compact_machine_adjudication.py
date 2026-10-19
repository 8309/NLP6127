"""Lossless compact source-adjudication selections, never automatic consensus.

The reviewer still receives each full source packet and both original labels.
Only response encoding changes: a validated original can be selected whole,
or a new complete label supplied. Actual compact raw bytes remain authoritative.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import adjudicate_machine_annotations as original
import run_codex_annotation as native
from oil_evidence import machine_evaluation as protocol

VERSION = "compact-source-label-selection-v1"
FORMAT_INSTRUCTIONS = """
Compact response encoding: compact-source-label-selection-v1.
These instructions replace ONLY the response-format sentences above. Every
source-review, evidence, cutoff, units, completeness and uncertainty rule remains
unchanged. Inspect every record's full evidence and BOTH complete original labels.
Do not accept a label merely because the two originals agree. Select a complete
original only if its relevance judgments, answerability, event status, answers
and quotes are defensible against the actual source and question.
Return exactly one object keyed by the supplied record IDs. Every decision has:
status: accepted or unresolved;
label_basis: a, b, new, or none;
new_label: null, except a NEW complete six-field label when label_basis is new;
reason: a concise source-based explanation;
unresolved_reasons: an array, empty only for an accepted defensible decision.
For accepted decisions, choose a or b to retain that EXACT whole original label,
or choose new and provide every required label field and every pooled judgment.
Never output a patched fragment or rewrite citations for an a/b selection.
For unresolved decisions, use none, new_label null and explicit reasons.
Use no tools or external knowledge. Return only the compact JSON object.
"""
SYSTEM = original.SYSTEM + FORMAT_INSTRUCTIONS


def prepare_batch(packets, left, right, record_ids):
    payload, _ = original.prepare_batch(packets, left, right, record_ids)
    properties = {}
    for key in record_ids:
        properties[key] = dict(type="object", additionalProperties=False,
            required=["status", "label_basis", "new_label", "reason", "unresolved_reasons"], properties=dict(
                status=dict(type="string", enum=["accepted", "unresolved"]),
                label_basis=dict(type="string", enum=["a", "b", "new", "none"]),
                new_label=dict(anyOf=[native.output_schema(packets[key]), dict(type="null")]),
                reason=dict(type="string"), unresolved_reasons=dict(type="array", items=dict(type="string"))))
    return payload, dict(type="object", additionalProperties=False, required=record_ids, properties=properties)


def _retained_raw(receipt, raw, record_ids):
    """Bind selections to actual whole native raw bytes, even for subset rebuilds."""
    if (receipt.get("accepted") is not True or not isinstance(receipt.get("execution_id"), str)
            or not receipt["execution_id"].strip()):
        raise ValueError("Compact selections require an accepted identifiable native receipt")
    directory = Path(receipt["artifact_directory"])
    receipt_path, raw_path = directory / "receipt.json", directory / "raw_final.json"
    actual_receipt = json.loads(receipt_path.read_bytes(), object_pairs_hook=protocol._unique_object,
                                parse_constant=protocol._invalid_constant)
    if actual_receipt != receipt:
        raise ValueError("Compact selection receipt differs from retained execution metadata")
    raw_bytes = raw_path.read_bytes()
    if protocol.sha256(raw_bytes) != receipt["artifacts_sha256"].get("raw_final.json"):
        raise ValueError("Compact full raw response hash differs from its native receipt")
    parsed = json.loads(raw_bytes, object_pairs_hook=protocol._unique_object, parse_constant=protocol._invalid_constant)
    if not isinstance(parsed, dict) or any(key not in parsed or parsed[key] != raw[key] for key in record_ids):
        raise ValueError("Compact decision does not match its exact retained raw JSON path")
    return raw_path, receipt_path


def retain_compact_decisions(raw, packets, left, right, record_ids, receipt):
    """Decode declared a/b/new selections losslessly; invalid labels stay unresolved."""
    if (not isinstance(raw, dict) or not record_ids or len(set(record_ids)) != len(record_ids)
            or set(raw) != set(record_ids)):
        raise ValueError("Compact adjudication response coverage differs from its batch")
    for key in record_ids:
        if key not in packets or key not in left or key not in right:
            raise ValueError("Compact adjudication lacks an original question or paired envelope")
        original.validate_pair(packets[key], left[key], right[key])
        if receipt["execution_id"] in {left[key]["identity"]["execution_id"], right[key]["identity"]["execution_id"]}:
            raise ValueError("Compact source review requires a fresh context distinct from both originals")
    raw_path, receipt_path = _retained_raw(receipt, raw, record_ids)
    decisions = []
    for key in record_ids:
        value = raw[key]
        required = {"status", "label_basis", "new_label", "reason", "unresolved_reasons"}
        if (not isinstance(value, dict) or set(value) != required
                or value["status"] not in {"accepted", "unresolved"}
                or value["label_basis"] not in {"a", "b", "new", "none"}
                or not isinstance(value["reason"], str) or not value["reason"].strip()
                or not isinstance(value["unresolved_reasons"], list)
                or any(not isinstance(reason, str) or not reason.strip() for reason in value["unresolved_reasons"])):
            raise ValueError("Malformed compact raw decision")
        basis, status = value["label_basis"], value["status"]
        errors, label = [], None
        reasons = list(value["unresolved_reasons"])
        if status == "unresolved":
            if basis != "none" or value["new_label"] is not None:
                errors.append("Unresolved compact decision requires basis none and new_label null")
            if not reasons:
                reasons.append(value["reason"])
        else:
            if reasons:
                errors.append("Accepted compact decision also declares unresolved reasons")
            if basis in {"a", "b"}:
                if value["new_label"] is not None:
                    errors.append("A/B selection requires new_label null; patched fragments are prohibited")
                label = deepcopy((left if basis == "a" else right)[key]["label"])
            elif basis == "new":
                label = deepcopy(value["new_label"])
            else:
                errors.append("Accepted compact decision cannot use basis none")
            if label is None:
                errors.append("Accepted compact decision lacks a complete selected label")
            else:
                errors.extend(protocol.validate_label(label, packets[key]))
        reasons.extend(errors)
        normalized_status = "unresolved" if errors else status
        normalized_label = label if normalized_status == "accepted" else None
        review = dict(reason=value["reason"], both_originals_seen=True,
            original_output_sha256=[protocol.sha256(protocol.canonical_bytes(stream[key])) for stream in (left, right)],
            protocol_sha256=protocol.sha256(SYSTEM.encode("utf-8")), execution_id=receipt["execution_id"],
            raw_response_path=str(raw_path), raw_decision_path=[key],
            raw_response_sha256=receipt["artifacts_sha256"]["raw_final.json"], execution_receipt_path=str(receipt_path),
            observed_model=receipt["observed_model"], raw_encoding=VERSION, label_basis=basis,
            resolved_label_sha256=protocol.sha256(protocol.canonical_bytes(normalized_label)) if normalized_label is not None else None,
            note="Source-based machine adjudication with lossless compact selection; not a blinded third label or human gold.")
        decision = dict(record_id=key, status=normalized_status, provenance="machine_adjudicated", human_gold=False,
            label=normalized_label, adjudication=review, raw_model_status=status, mechanical_validation_errors=errors)
        if normalized_status == "unresolved":
            decision["unresolved_reasons"] = reasons
        decisions.append(decision)
    return decisions
