"""Lossless structural-ID transport helpers, never an annotation runner.

Canonical questions/evidence retain their identities. A native wire protocol
must use a separately frozen integration that preserves unmodified raw model
bytes and explicitly records this codec; historical unencoded outputs stay intact.
Only structural record/passage identifiers may change. Source text, dates,
titles, URLs, grades, status, reasons and reference content are never invented.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from oil_evidence import machine_evaluation as protocol

VERSION = "oil-local-structural-id-wire-v1"
WIRE_INSTRUCTIONS = """
Transport encoding: oil-local-structural-id-wire-v1.
Record identifiers r0/r1 and evidence identifiers r0p0/r1p0 are reversible opaque
transport aliases, not relevance ranks or labels. Evidence aliases are scoped
ONLY to their own record. Never borrow evidence or facts from another record.
Return exactly the supplied record aliases and the same six-field label schema.
Use only each record's own evidence aliases for qrels, citations and quotes.
The questions, full evidence text and metadata have not been shortened.
"""


def _canonical_rows(rows):
    result = []
    seen = set()
    for row in rows:
        packet = protocol.clean_packet(row)
        if packet["payload"] != row:
            raise ValueError("Codec accepts only clean full canonical payloads, without hints")
        identifier = row["record_id"]
        if identifier in seen:
            raise ValueError("Duplicate canonical record ID")
        seen.add(identifier)
        result.append(deepcopy(row))
    if not result:
        raise ValueError("Empty codec batch")
    return result


def encode_payloads(rows):
    """Map IDs only; bind canonical order, exact source values and wire bytes."""
    rows = _canonical_rows(rows)
    wire, records = {}, {}
    for index, canonical in enumerate(rows):
        record_alias = f"r{index}"
        value = deepcopy(canonical)
        value["record_id"] = record_alias
        evidence_map = {}
        for evidence_index, evidence in enumerate(value["evidence"]):
            alias = f"r{index}p{evidence_index}"
            evidence_map[alias] = evidence["passage_id"]
            evidence["passage_id"] = alias
        wire[record_alias] = value
        records[record_alias] = dict(record_id=canonical["record_id"],
            canonical_payload_sha256=protocol.sha256(protocol.canonical_bytes(canonical)),
            ordered_canonical_passage_ids=[evidence["passage_id"] for evidence in canonical["evidence"]],
            passage_ids=evidence_map)
    codec = dict(version=VERSION, ordered_canonical_record_ids=[row["record_id"] for row in rows],
        canonical_payloads_sha256=protocol.sha256(protocol.canonical_bytes(rows)),
        wire_payload_sha256=protocol.sha256(protocol.canonical_bytes(wire)), records=records,
        transformations=["structural_record_ids", "structural_passage_ids"],
        source_content_unchanged=True, semantic_labels_created=0, encoding_instructions_sha256=protocol.sha256(WIRE_INSTRUCTIONS.encode()))
    codec["mapping_sha256"] = protocol.sha256(protocol.canonical_bytes(codec))
    return wire, codec


def verify_codec(codec, rows, wire=None):
    """Regenerate the only known codec; never trust a supplied arbitrary mapping."""
    expected_wire, expected_codec = encode_payloads(rows)
    if codec != expected_codec or (wire is not None and wire != expected_wire):
        raise ValueError("Codec mapping, canonical order/source content or wire payload changed")
    return expected_wire


def prepare_batch(rows, base_prompt, output_schema):
    """Pure preparation only; actual schema bytes and extra instructions are explicit."""
    wire, codec = encode_payloads(rows)
    prompt = base_prompt + WIRE_INSTRUCTIONS
    packets = {alias: protocol.clean_packet(value, prompt=prompt) for alias, value in wire.items()}
    schema = dict(type="object", additionalProperties=False, required=list(wire), properties={
        alias: dict(type="object", additionalProperties=False, required=["record_id", "label"],
            properties=dict(record_id=dict(type="string", enum=[alias]), label=output_schema(packet)))
        for alias, packet in packets.items()})
    stdin = prompt.encode() + b"\n\nUNTRUSTED_RECORD_PACKETS_JSON\n" + protocol.canonical_bytes(wire) + b"\n"
    canonical_packets = {row["record_id"]: protocol.clean_packet(row, prompt=base_prompt) for row in rows}
    return dict(wire_payload=wire, codec=codec, prompt=prompt, schema=schema, stdin=stdin,
        canonical_semantic_packets=canonical_packets,
        canonical_semantic_payloads_sha256=codec["canonical_payloads_sha256"],
        actual_wire_stdin_sha256=protocol.sha256(stdin), actual_wire_schema_sha256=protocol.sha256(protocol.canonical_bytes(schema)),
        raw_codec_provenance_required=True, integration_status="unused_prototype_not_valid_in_historical_wrappers")


def _map_label(label, passage_ids):
    """Change identifier fields only; no recursive/string/quote-content rewriting."""
    if not isinstance(label, dict) or set(label) != protocol.LABEL_KEYS:
        raise ValueError("Wire label must have the unchanged six fields")
    result = deepcopy(label)

    def restore(identifier):
        if not isinstance(identifier, str) or identifier not in passage_ids:
            raise ValueError("Unknown or cross-record wire passage alias")
        return passage_ids[identifier]

    if not isinstance(label["qrels"], dict) or set(label["qrels"]) != set(passage_ids):
        raise ValueError("Wire qrels must cover every own-pool alias exactly")
    result["qrels"] = {restore(key): value for key, value in label["qrels"].items()}
    if not isinstance(result["answer_points"], list) or not isinstance(result["quotes"], list):
        raise ValueError("Wire answer points and quotes must be arrays")
    for point in result["answer_points"]:
        if not isinstance(point, dict) or not isinstance(point.get("citations"), list):
            raise ValueError("Invalid wire answer-point citation structure")
        point["citations"] = [restore(identifier) for identifier in point["citations"]]
    for quote in result["quotes"]:
        if not isinstance(quote, dict):
            raise ValueError("Invalid wire quote structure")
        quote["passage_id"] = restore(quote.get("passage_id"))
    return result


def decode_batch(raw, codec, canonical_rows):
    """Restore known IDs losslessly, retaining the EXACT whole model response."""
    verify_codec(codec, canonical_rows)
    raw_text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    if not isinstance(raw_text, str):
        raise ValueError("Retain actual raw model UTF-8 bytes/text, not a parsed replacement")
    parsed = json.loads(raw_text, object_pairs_hook=protocol._unique_object, parse_constant=protocol._invalid_constant)
    if not isinstance(parsed, dict) or set(parsed) != set(codec["records"]):
        raise ValueError("Wire response record coverage differs")
    canonical = {row["record_id"]: row for row in canonical_rows}
    result = {}
    for alias, binding in codec["records"].items():
        value = parsed[alias]
        if not isinstance(value, dict) or set(value) != {"record_id", "label"} or value["record_id"] != alias:
            raise ValueError("Wire response record ID differs from its own key")
        label = _map_label(value["label"], binding["passage_ids"])
        errors = protocol.validate_label(label, protocol.clean_packet(canonical[binding["record_id"]]))
        if errors:
            raise ValueError("Decoded canonical label invalid: " + "; ".join(errors))
        result[binding["record_id"]] = dict(record_id=binding["record_id"], label=label)
    return dict(decoded_records=result, raw_output=raw_text, raw_output_sha256=protocol.sha256(raw_text.encode()),
        raw_codec=deepcopy(codec), normalization="Lossless structural identifiers only; actual model raw bytes are unchanged",
        canonical_output_is_raw_model_output=False, creates_semantic_labels=False)


def decode_record(raw, codec, canonical_row):
    """Validate one retained alias label against its unchanged canonical source.

    Full-batch input/schema and mapping verification belongs to the execution
    runner via verify_codec. This pure per-record validator additionally checks
    the complete mapping checksum/shape, exact own-source binding and raw batch
    coverage; it never asserts that a metadata claim proves an actual call.
    """
    keys = {"version", "ordered_canonical_record_ids", "canonical_payloads_sha256",
            "wire_payload_sha256", "records", "transformations", "source_content_unchanged",
            "semantic_labels_created", "encoding_instructions_sha256", "mapping_sha256"}
    if not isinstance(codec, dict) or set(codec) != keys or codec.get("version") != VERSION:
        raise ValueError("Unknown or incomplete structural-ID codec")
    unsigned = {key: value for key, value in codec.items() if key != "mapping_sha256"}
    if codec["mapping_sha256"] != protocol.sha256(protocol.canonical_bytes(unsigned)):
        raise ValueError("Structural-ID mapping checksum changed")
    if (codec["transformations"] != ["structural_record_ids", "structural_passage_ids"]
            or codec["source_content_unchanged"] is not True
            or type(codec["semantic_labels_created"]) is not int or codec["semantic_labels_created"] != 0
            or codec["encoding_instructions_sha256"] != protocol.sha256(WIRE_INSTRUCTIONS.encode())):
        raise ValueError("Structural-ID normalization protocol changed")
    ids = codec["ordered_canonical_record_ids"]
    if (not isinstance(ids, list) or not ids or any(not isinstance(value, str) or not value for value in ids)
            or len(ids) != len(set(ids)) or not isinstance(codec["records"], dict)
            or set(codec["records"]) != {f"r{index}" for index in range(len(ids))}):
        raise ValueError("Structural-ID record coverage changed")
    for key in ("canonical_payloads_sha256", "wire_payload_sha256"):
        if not isinstance(codec[key], str) or not re.fullmatch(r"[a-f0-9]{64}", codec[key]):
            raise ValueError("Structural-ID source/wire hash invalid")
    for index, identifier in enumerate(ids):
        binding = codec["records"][f"r{index}"]
        if (not isinstance(binding, dict) or set(binding) != {"record_id", "canonical_payload_sha256",
                "ordered_canonical_passage_ids", "passage_ids"} or binding["record_id"] != identifier):
            raise ValueError("Structural-ID record binding changed")
        passages = binding["ordered_canonical_passage_ids"]
        if (not isinstance(passages, list) or any(not isinstance(value, str) or not value for value in passages)
                or len(passages) != len(set(passages))
                or binding["passage_ids"] != {f"r{index}p{other}": value for other, value in enumerate(passages)}
                or not isinstance(binding["canonical_payload_sha256"], str)
                or not re.fullmatch(r"[a-f0-9]{64}", binding["canonical_payload_sha256"])):
            raise ValueError("Structural-ID evidence binding changed")
    canonical_row = _canonical_rows([canonical_row])[0]
    identifier = canonical_row["record_id"]
    if identifier not in ids:
        raise ValueError("Canonical record is absent from codec")
    alias = f"r{ids.index(identifier)}"
    binding = codec["records"][alias]
    if (binding["canonical_payload_sha256"] != protocol.sha256(protocol.canonical_bytes(canonical_row))
            or binding["ordered_canonical_passage_ids"] != [value["passage_id"] for value in canonical_row["evidence"]]):
        raise ValueError("Canonical source content/order differs from codec")
    parsed = json.loads(raw, object_pairs_hook=protocol._unique_object, parse_constant=protocol._invalid_constant)
    if not isinstance(parsed, dict) or set(parsed) != set(codec["records"]):
        raise ValueError("Wire response record coverage differs")
    for record_alias, value in parsed.items():
        if (not isinstance(value, dict) or set(value) != {"record_id", "label"}
                or value["record_id"] != record_alias):
            raise ValueError("Wire response record ID differs from its own key")
    label = _map_label(parsed[alias]["label"], binding["passage_ids"])
    errors = protocol.validate_label(label, protocol.clean_packet(canonical_row))
    if errors:
        raise ValueError("Decoded canonical label invalid: " + "; ".join(errors))
    return dict(record_alias=alias, record_id=identifier, label=label)


def simulate_encoding_for_measurement(canonical_output, codec, canonical_rows):
    """Representation-only estimate; NEVER present this as a model execution."""
    verify_codec(codec, canonical_rows)
    if set(canonical_output) != set(codec["ordered_canonical_record_ids"]):
        raise ValueError("Canonical output coverage differs")
    result = {}
    for alias, binding in codec["records"].items():
        value = canonical_output[binding["record_id"]]
        if set(value) != {"record_id", "label"} or value["record_id"] != binding["record_id"]:
            raise ValueError("Canonical output record mismatch")
        reverse = {canonical: wire for wire, canonical in binding["passage_ids"].items()}
        result[alias] = dict(record_id=alias, label=_map_label(value["label"], reverse))
    return result
