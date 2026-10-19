"""Synthetic compact-codec tests; no model invocation or human-label claim."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def codec(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("compact_source_codec_tests", scripts / "compact_machine_adjudication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def paired(codec):
    row = dict(record_id="synthetic-q", question="What did the synthetic crude report observe?", as_of="2024-03-01",
        evidence=[dict(passage_id="p-eligible", text="Synthetic crude exports were 2 million barrels per day.", available_by="2024-01-01"),
                  dict(passage_id="p-future", text="A future report observed 4 million barrels per day.", available_by="2025-01-01")])
    packet = codec.protocol.clean_packet(row, prompt="Synthetic fixed protocol")
    label = dict(answerability="sufficient", event_status="observed", qrels={"p-eligible": 2, "p-future": 0},
        answer_points=[dict(text=row["evidence"][0]["text"], citations=["p-eligible"])],
        quotes=[dict(passage_id="p-eligible", text=row["evidence"][0]["text"])], reasons=["Synthetic exact eligible source."])
    streams = []
    for side in ("a", "b"):
        value = deepcopy(label)
        if side == "b":
            value["reasons"] = ["Different synthetic source rationale."]
        raw = json.dumps(value, indent=2) + "\r\n"
        identity = dict(service="synthetic", requested_model="unverified-alias", observed_model=None,
                        execution_id="synthetic-original-" + side, observation_source="unavailable", identity_note="Synthetic metadata unavailable.")
        output = codec.protocol.wrap_independent_output(packet, raw, identity, identity["execution_id"], {}, usage=None,
                                                       execution_input_sha256="1" * 64)
        assert output["validation_errors"] == []
        streams.append({row["record_id"]: output})
    return {row["record_id"]: packet}, streams[0], streams[1]


def raw_decision(basis="a", status="accepted", new_label=None, reasons=None):
    return {"synthetic-q": dict(status=status, label_basis=basis, new_label=new_label,
        reason="Synthetic reviewer inspected full evidence and both originals.", unresolved_reasons=reasons or [])}


def retained_receipt(codec, tmp_path, raw, execution_id="synthetic-fresh-review"):
    directory = tmp_path / "execution"
    directory.mkdir(exist_ok=True)
    raw_bytes = (json.dumps(raw, indent=2) + "\r\n").encode()
    (directory / "raw_final.json").write_bytes(raw_bytes)
    receipt = dict(artifact_directory=str(directory), execution_id=execution_id, accepted=True,
                   observed_model=None, artifacts_sha256={"raw_final.json": codec.protocol.sha256(raw_bytes)})
    (directory / "receipt.json").write_bytes(codec.protocol.canonical_bytes(receipt))
    return receipt


def test_full_source_both_originals_same_payload_no_consensus_shortcut(codec, paired):
    packets, left, right = paired
    compact_payload, schema = codec.prepare_batch(packets, left, right, ["synthetic-q"])
    original_payload, _ = codec.original.prepare_batch(packets, left, right, ["synthetic-q"])
    assert compact_payload == original_payload
    assert "Do not accept a label merely because the two originals agree" in codec.SYSTEM
    assert codec.SYSTEM.startswith(codec.original.SYSTEM)
    fields = schema["properties"]["synthetic-q"]["properties"]
    assert fields["label_basis"]["enum"] == ["a", "b", "new", "none"]
    assert fields["new_label"]["anyOf"][0]["properties"]["qrels"]["required"] == ["p-eligible", "p-future"]


@pytest.mark.parametrize("basis", ["a", "b"])
def test_whole_original_selection_exact_label_hash_and_raw_bytes(codec, paired, tmp_path, basis):
    packets, left, right = paired
    raw = raw_decision(basis)
    receipt = retained_receipt(codec, tmp_path, raw)
    before = (Path(receipt["artifact_directory"]) / "raw_final.json").read_bytes()
    result = codec.retain_compact_decisions(raw, packets, left, right, ["synthetic-q"], receipt)[0]
    source = (left if basis == "a" else right)["synthetic-q"]["label"]
    assert result["status"] == "accepted" and result["label"] == source and result["label"] is not source
    assert result["adjudication"]["resolved_label_sha256"] == codec.protocol.sha256(codec.protocol.canonical_bytes(source))
    assert result["adjudication"]["raw_encoding"] == codec.VERSION and result["adjudication"]["label_basis"] == basis
    assert result["adjudication"]["protocol_sha256"] == codec.protocol.sha256(codec.SYSTEM.encode())
    assert result["adjudication"]["original_output_sha256"] == [codec.protocol.sha256(codec.protocol.canonical_bytes(stream["synthetic-q"])) for stream in (left, right)]
    assert result["human_gold"] is False and result["provenance"] == "machine_adjudicated"
    assert before == (Path(receipt["artifact_directory"]) / "raw_final.json").read_bytes()
    assert b'"new_label": null' in before and b'"qrels"' not in before


def test_new_complete_label_and_old_full_decoder_same_selected_label(codec, paired, tmp_path):
    packets, left, right = paired
    label = deepcopy(left["synthetic-q"]["label"])
    label["reasons"] = ["Fresh synthetic complete source review."]
    raw = raw_decision("new", new_label=label)
    receipt = retained_receipt(codec, tmp_path, raw)
    compact = codec.retain_compact_decisions(raw, packets, left, right, ["synthetic-q"], receipt)[0]
    original_raw = {"synthetic-q": dict(status="accepted", reason=raw["synthetic-q"]["reason"], label=label, unresolved_reasons=[])}
    original = codec.original.retain_decisions(original_raw, packets, left, right, ["synthetic-q"], receipt)[0]
    assert compact["label"] == original["label"] == label
    assert compact["adjudication"]["protocol_sha256"] != original["adjudication"]["protocol_sha256"]


def test_none_unresolved_retained_without_inventing_label(codec, paired, tmp_path):
    packets, left, right = paired
    raw = raw_decision("none", "unresolved", reasons=["Synthetic source conflict cannot be resolved."])
    result = codec.retain_compact_decisions(raw, packets, left, right, ["synthetic-q"], retained_receipt(codec, tmp_path, raw))[0]
    assert result["status"] == "unresolved" and result["label"] is None
    assert result["unresolved_reasons"] == raw["synthetic-q"]["unresolved_reasons"]
    assert result["adjudication"]["resolved_label_sha256"] is None


@pytest.mark.parametrize("basis,status,new_label,reasons", [
    ("a", "accepted", {}, []), ("b", "accepted", {}, []), ("new", "accepted", None, []),
    ("none", "accepted", None, []), ("a", "unresolved", None, []),
    ("new", "unresolved", {}, []), ("a", "accepted", None, ["Synthetic unresolved conflict."])])
def test_invalid_combinations_become_explicit_unresolved(codec, paired, tmp_path, basis, status, new_label, reasons):
    packets, left, right = paired
    raw = raw_decision(basis, status, new_label, reasons)
    result = codec.retain_compact_decisions(raw, packets, left, right, ["synthetic-q"], retained_receipt(codec, tmp_path, raw))[0]
    assert result["status"] == "unresolved" and result["label"] is None
    assert result["raw_model_status"] == status and result["mechanical_validation_errors"]
    assert result["unresolved_reasons"]


@pytest.mark.parametrize("mutation", ["future_positive", "inexact_quote", "missing_qrel"])
def test_new_label_keeps_all_original_mechanical_guards(codec, paired, tmp_path, mutation):
    packets, left, right = paired
    label = deepcopy(left["synthetic-q"]["label"])
    if mutation == "future_positive":
        label["qrels"]["p-future"] = 2
    elif mutation == "inexact_quote":
        label["quotes"][0]["text"] = "Not in the synthetic source."
    else:
        label["qrels"].pop("p-future")
    raw = raw_decision("new", new_label=label)
    result = codec.retain_compact_decisions(raw, packets, left, right, ["synthetic-q"], retained_receipt(codec, tmp_path, raw))[0]
    assert result["status"] == "unresolved" and result["mechanical_validation_errors"]


@pytest.mark.parametrize("mutation", ["raw_tamper", "selection_tamper", "receipt_tamper", "original_tamper", "reused_context"])
def test_raw_receipt_original_and_fresh_context_bindings(codec, paired, tmp_path, mutation):
    packets, left, right = paired
    raw = raw_decision()
    receipt = retained_receipt(codec, tmp_path, raw)
    if mutation == "raw_tamper":
        (Path(receipt["artifact_directory"]) / "raw_final.json").write_bytes(b"{}")
    elif mutation == "selection_tamper":
        raw["synthetic-q"]["label_basis"] = "b"
    elif mutation == "receipt_tamper":
        receipt["execution_id"] = "synthetic-different-context"
    elif mutation == "original_tamper":
        left["synthetic-q"]["label"]["qrels"]["p-future"] = 1
    else:
        receipt = retained_receipt(codec, tmp_path, raw, "synthetic-original-a")
    with pytest.raises(ValueError):
        codec.retain_compact_decisions(raw, packets, left, right, ["synthetic-q"], receipt)


def test_subset_rebuild_reads_actual_compact_batch_and_never_replaces_raw(codec, paired, tmp_path):
    packets, left, right = paired
    raw = raw_decision()
    entire = dict(raw, unrelated_synthetic_record=dict(raw["synthetic-q"]))
    receipt = retained_receipt(codec, tmp_path, entire)
    result = codec.retain_compact_decisions(raw, packets, left, right, ["synthetic-q"], receipt)[0]
    assert result["adjudication"]["raw_decision_path"] == ["synthetic-q"]
    assert len(json.loads((Path(receipt["artifact_directory"]) / "raw_final.json").read_bytes())) == 2


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_basis", "empty_reason"])
def test_malformed_compact_wire_never_silently_decoded(codec, paired, tmp_path, mutation):
    packets, left, right = paired
    raw = raw_decision()
    if mutation == "missing":
        raw["synthetic-q"].pop("label_basis")
    elif mutation == "extra":
        raw["synthetic-q"]["label"] = left["synthetic-q"]["label"]
    elif mutation == "wrong_basis":
        raw["synthetic-q"]["label_basis"] = "automatic-consensus"
    else:
        raw["synthetic-q"]["reason"] = ""
    with pytest.raises(ValueError, match="Malformed"):
        codec.retain_compact_decisions(raw, packets, left, right, ["synthetic-q"], retained_receipt(codec, tmp_path, raw))
