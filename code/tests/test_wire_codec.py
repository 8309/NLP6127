"""Pure synthetic wire representation tests, not model outputs or benchmark QA."""
from copy import deepcopy
import json

import pytest

from oil_evidence import machine_evaluation as protocol
from oil_evidence import wire_codec as codec


@pytest.fixture
def rows():
    return [dict(record_id=f"synthetic-canonical-record-{index}", question=f"What did synthetic region {index} report?",
        as_of="2024-02-01", evidence=[dict(passage_id=f"synthetic-original-passage-{index}-{other}",
            document_id=f"synthetic-document-{index}", title="Synthetic full title", url="https://example.invalid/full-source",
            text=f"Synthetic region {index} reported crude exports of {other + 1} million barrels per day.",
            available_by="2024-01-01") for other in range(2)]) for index in range(2)]


def canonical_outputs(rows):
    result = {}
    for row in rows:
        evidence = row["evidence"][0]
        result[row["record_id"]] = dict(record_id=row["record_id"], label=dict(answerability="sufficient", event_status="observed",
            qrels={item["passage_id"]: 2 if index == 0 else 1 for index, item in enumerate(row["evidence"])},
            answer_points=[dict(text=evidence["text"], citations=[evidence["passage_id"]])],
            quotes=[dict(passage_id=evidence["passage_id"], text=evidence["text"])], reasons=["Synthetic own-pool quote."]))
    return result


def test_only_structural_ids_change_all_source_content_preserved(rows):
    wire, metadata = codec.encode_payloads(rows)
    assert metadata["semantic_labels_created"] == 0
    for index, row in enumerate(rows):
        value = deepcopy(wire[f"r{index}"])
        value["record_id"] = row["record_id"]
        for other, evidence in enumerate(value["evidence"]):
            evidence["passage_id"] = row["evidence"][other]["passage_id"]
        assert value == row
    assert codec.verify_codec(metadata, rows, wire) == wire


def test_round_trip_preserves_labels_quotes_reasons_and_exact_raw(rows):
    _, metadata = codec.encode_payloads(rows)
    canonical = canonical_outputs(rows)
    aliases = codec.simulate_encoding_for_measurement(canonical, metadata, rows)
    raw = (json.dumps(aliases, indent=4) + "\r\n").encode()
    decoded = codec.decode_batch(raw, metadata, rows)
    assert decoded["decoded_records"] == canonical
    assert decoded["raw_output"].encode() == raw
    assert decoded["raw_output_sha256"] == protocol.sha256(raw)
    assert decoded["canonical_output_is_raw_model_output"] is False
    assert decoded["creates_semantic_labels"] is False


@pytest.mark.parametrize("mutation", ["order", "source", "mapping", "wire"])
def test_any_mapping_order_source_or_wire_mutation_rejected(rows, mutation):
    wire, metadata = codec.encode_payloads(rows)
    changed_rows = deepcopy(rows)
    if mutation == "order":
        changed_rows.reverse()
    elif mutation == "source":
        changed_rows[0]["evidence"][0]["text"] += " Changed source."
    elif mutation == "mapping":
        metadata["records"]["r0"]["passage_ids"]["e0"] = rows[1]["evidence"][0]["passage_id"]
    else:
        wire["r0"]["question"] = "Different question"
    with pytest.raises(ValueError, match="changed"):
        codec.verify_codec(metadata, changed_rows, wire)


@pytest.mark.parametrize("target", ["qrels", "citations", "quotes"])
def test_unknown_or_cross_record_alias_rejected(rows, target):
    _, metadata = codec.encode_payloads(rows)
    wire = codec.simulate_encoding_for_measurement(canonical_outputs(rows), metadata, rows)
    if target == "qrels":
        wire["r0"]["label"]["qrels"]["unknown"] = 2
    elif target == "citations":
        wire["r0"]["label"]["answer_points"][0]["citations"] = ["r1p0"]
    else:
        wire["r0"]["label"]["quotes"][0]["passage_id"] = "unknown"
    with pytest.raises(ValueError, match="alias"):
        codec.decode_batch(json.dumps(wire), metadata, rows)


def test_boolean_grade_not_synthetically_corrected(rows):
    _, metadata = codec.encode_payloads(rows)
    wire = codec.simulate_encoding_for_measurement(canonical_outputs(rows), metadata, rows)
    wire["r0"]["label"]["qrels"]["r0p0"] = True
    with pytest.raises(ValueError, match="invalid"):
        codec.decode_batch(json.dumps(wire), metadata, rows)


def test_future_positive_grade_retains_original_date_guard(rows):
    rows[0]["evidence"][0]["available_by"] = "2025-01-01"
    _, metadata = codec.encode_payloads(rows)
    wire = codec.simulate_encoding_for_measurement(canonical_outputs(rows), metadata, rows)
    with pytest.raises(ValueError, match="invalid"):
        codec.decode_batch(json.dumps(wire), metadata, rows)


def test_modified_quote_not_synthetically_corrected(rows):
    _, metadata = codec.encode_payloads(rows)
    wire = codec.simulate_encoding_for_measurement(canonical_outputs(rows), metadata, rows)
    wire["r0"]["label"]["quotes"][0]["text"] = "Invented quotation"
    with pytest.raises(ValueError, match="invalid"):
        codec.decode_batch(json.dumps(wire), metadata, rows)


def test_quote_and_reason_literals_are_not_globally_replaced(rows):
    marker = rows[0]["evidence"][0]["passage_id"]
    rows[0]["evidence"][0]["text"] += " Literal identifier " + marker
    _, metadata = codec.encode_payloads(rows)
    canonical = canonical_outputs(rows)
    canonical[rows[0]["record_id"]]["label"]["reasons"] = [marker]
    wire = codec.simulate_encoding_for_measurement(canonical, metadata, rows)
    assert marker in wire["r0"]["label"]["quotes"][0]["text"]
    assert wire["r0"]["label"]["reasons"] == [marker]
    assert codec.decode_batch(json.dumps(wire), metadata, rows)["decoded_records"] == canonical


def test_prepared_input_is_reproducible_and_explicitly_new_wire_protocol(rows):
    def schema(packet):
        return dict(type="object", properties=dict(qrels=dict(enum=[item["passage_id"] for item in packet["payload"]["evidence"]])))
    first = codec.prepare_batch(rows, "SYNTHETIC BASE PROMPT", schema)
    second = codec.prepare_batch(rows, "SYNTHETIC BASE PROMPT", schema)
    assert first == second
    assert codec.WIRE_INSTRUCTIONS.encode() in first["stdin"]
    assert first["actual_wire_stdin_sha256"] == protocol.sha256(first["stdin"])
    assert first["raw_codec_provenance_required"] is True
    assert first["integration_status"].startswith("unused")


def test_duplicate_json_keys_or_missing_records_rejected(rows):
    _, metadata = codec.encode_payloads(rows)
    wire = codec.simulate_encoding_for_measurement(canonical_outputs(rows), metadata, rows)
    with pytest.raises(ValueError):
        codec.decode_batch('{"r0":{},"r0":{}}', metadata, rows)
    del wire["r1"]
    with pytest.raises(ValueError, match="coverage"):
        codec.decode_batch(json.dumps(wire), metadata, rows)


def test_missing_or_unknown_label_fields_rejected(rows):
    _, metadata = codec.encode_payloads(rows)
    wire = codec.simulate_encoding_for_measurement(canonical_outputs(rows), metadata, rows)
    wire["r0"]["label"]["synthetic_gold"] = 2
    with pytest.raises(ValueError, match="six fields"):
        codec.decode_batch(json.dumps(wire), metadata, rows)


def test_batch_global_passage_aliases_are_unique_even_for_shared_canonical_source(rows):
    rows[1]["evidence"][0] = deepcopy(rows[0]["evidence"][0])
    wire, metadata = codec.encode_payloads(rows)
    aliases = [item["passage_id"] for row in wire.values() for item in row["evidence"]]
    assert len(aliases) == len(set(aliases))
    assert metadata["records"]["r0"]["passage_ids"]["r0p0"] == metadata["records"]["r1"]["passage_ids"]["r1p0"]


def wrapped_wire_output(rows):
    _, mapping = codec.encode_payloads(rows)
    wire = codec.simulate_encoding_for_measurement(canonical_outputs(rows), mapping, rows)
    raw = json.dumps(wire, indent=2) + "\r\n"
    packet = protocol.clean_packet(rows[1])
    identity = dict(service="synthetic", requested_model="synthetic-request", observed_model=None,
        execution_id="synthetic-execution", observation_source="unavailable", identity_note="Synthetic fixture, not a model call.")
    output = protocol.wrap_independent_output(packet, raw, identity, "synthetic-context", {},
        execution_input_sha256=protocol.sha256(b"synthetic actual full wire stdin"),
        raw_label_path=["r1", "label"], raw_record_id_path=["r1", "record_id"], raw_codec=mapping)
    return packet, output, raw


def test_registered_wire_envelope_keeps_actual_raw_and_canonical_semantic_binding(rows):
    packet, output, raw = wrapped_wire_output(rows)
    assert output["validation_errors"] == []
    assert output["raw_output"] == raw
    assert output["label"] == canonical_outputs(rows)[rows[1]["record_id"]]["label"]
    assert output["input_sha256"] == packet["input_sha256"]
    assert output["execution_input_sha256"] != packet["input_sha256"]
    assert output["raw_encoding"] == codec.VERSION
    assert output["canonical_output_is_raw_model_output"] is False
    assert protocol.validate_independent_output(output, packet) == []


@pytest.mark.parametrize("field", ["raw_label_path", "raw_record_id_path", "raw_encoding",
    "canonical_output_is_raw_model_output", "label", "raw_output_sha256"])
def test_envelope_wire_provenance_and_content_tampering_fails(rows, field):
    packet, output, _ = wrapped_wire_output(rows)
    if field == "label":
        output[field]["answer_points"][0]["text"] = "Invented synthetic statement."
    else:
        output[field] = "changed"
    assert protocol.validate_independent_output(output, packet)


@pytest.mark.parametrize("change", ["unknown_version", "missing_key", "bad_checksum", "changed_source",
    "changed_evidence_order", "foreign_record", "nonunique_ids", "unknown_alias"])
def test_per_record_decoder_rejects_unregistered_or_mismatched_mapping(rows, change):
    wire, mapping = codec.encode_payloads(rows)
    raw = json.dumps(codec.simulate_encoding_for_measurement(canonical_outputs(rows), mapping, rows))
    row = deepcopy(rows[1])
    if change == "unknown_version":
        mapping["version"] = "unknown"
    elif change == "missing_key":
        del mapping["wire_payload_sha256"]
    elif change == "bad_checksum":
        mapping["records"]["r1"]["canonical_payload_sha256"] = "0" * 64
    elif change == "changed_source":
        row["evidence"][0]["text"] += " Changed original source."
    elif change == "changed_evidence_order":
        row["evidence"].reverse()
    elif change == "foreign_record":
        row["record_id"] = "foreign"
    elif change == "nonunique_ids":
        mapping["ordered_canonical_record_ids"][0] = row["record_id"]
        unsigned = {key: value for key, value in mapping.items() if key != "mapping_sha256"}
        mapping["mapping_sha256"] = protocol.sha256(protocol.canonical_bytes(unsigned))
    else:
        parsed = json.loads(raw)
        parsed["r1"]["label"]["answer_points"][0]["citations"] = ["r0p0"]
        raw = json.dumps(parsed)
    with pytest.raises(ValueError):
        codec.decode_record(raw, mapping, row)


def test_raw_encoding_without_codec_is_not_silently_treated_as_unencoded(rows):
    packet, output, _ = wrapped_wire_output(rows)
    del output["raw_codec"]
    assert protocol.validate_independent_output(output, packet)
