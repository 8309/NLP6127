"""Mocked source adjudication protocol tests; no subscription calls."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import adjudicate_machine_annotations as module
from test_freeze_machine_dataset import make_inputs
from oil_evidence import machine_evaluation as protocol


def scenario(tmp_path):
    source = make_inputs(tmp_path)
    artifacts = {key: (source / name).read_bytes() for key, name in
                 dict(corpus="corpus.jsonl", questions="questions.jsonl", evidence_pool="evidence_pool.jsonl").items()}
    packets = protocol.prepare_inputs(*(protocol.parse_jsonl(artifacts[key]) for key in
                                      ("questions", "evidence_pool", "corpus")))
    left, right = (module.read_stream(source / name) for name in ("independent_a.jsonl", "independent_b.jsonl"))
    return source, packets, left, right


def receipt_for(tmp_path, parsed):
    directory = tmp_path / "native"
    directory.mkdir(parents=True, exist_ok=True)
    raw = protocol.canonical_bytes(parsed)
    (directory / "raw_final.json").write_bytes(raw)
    receipt = dict(artifact_directory=str(directory), execution_id="synthetic-adjudication", observed_model=None,
                   accepted=True, artifacts_sha256={"raw_final.json": protocol.sha256(raw)})
    (directory / "receipt.json").write_bytes(protocol.canonical_bytes(receipt))
    return receipt


def raw_decision(left):
    return dict(q1=dict(status="accepted", reason="Synthetic source review agrees with direct quantities.",
                        label=deepcopy(left["q1"]["label"]), unresolved_reasons=[]))


def test_batch_sees_both_labels_but_not_author_metadata(tmp_path):
    _, packets, left, right = scenario(tmp_path)
    payload, schema = module.prepare_batch(packets, left, right, ["q1"])
    assert payload["q1"]["independent_a"] == left["q1"]["label"]
    assert payload["q1"]["independent_b"] == right["q1"]["label"]
    assert "event_group" not in payload["q1"] and "split" not in payload["q1"]
    assert schema["properties"]["q1"]["additionalProperties"] is False


@pytest.mark.parametrize("change", ["same_context", "same_execution", "unequal_input"])
def test_conflicting_originals_cannot_be_adjudicated(tmp_path, change):
    _, packets, left, right = scenario(tmp_path)
    if change == "same_context":
        right["q1"]["context_id"] = left["q1"]["context_id"]
    elif change == "same_execution":
        right["q1"]["identity"]["execution_id"] = left["q1"]["identity"]["execution_id"]
    else:
        right["q1"]["execution_input_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="conflicting"):
        module.prepare_batch(packets, left, right, ["q1"])


def test_raw_decision_and_originals_bound_without_human_claim(tmp_path):
    _, packets, left, right = scenario(tmp_path)
    raw = raw_decision(left)
    receipt = receipt_for(tmp_path, raw)
    decision = module.retain_decisions(raw, packets, left, right, ["q1"], receipt)[0]
    assert decision["status"] == "accepted" and decision["human_gold"] is False
    assert decision["adjudication"]["both_originals_seen"] is True
    assert decision["adjudication"]["original_output_sha256"] == [
        protocol.sha256(protocol.canonical_bytes(stream["q1"])) for stream in (left, right)]
    assert decision["adjudication"]["observed_model"] is None


def test_mechanically_invalid_accepted_decision_is_retained_as_unresolved(tmp_path):
    _, packets, left, right = scenario(tmp_path)
    raw = raw_decision(left)
    raw["q1"]["label"]["quotes"][0]["text"] = "Fabricated synthetic quote"
    receipt = receipt_for(tmp_path, raw)
    decision = module.retain_decisions(raw, packets, left, right, ["q1"], receipt)[0]
    assert decision["raw_model_status"] == "accepted" and decision["status"] == "unresolved"
    assert decision["label"] is None and decision["unresolved_reasons"]
    assert json.loads((Path(receipt["artifact_directory"]) / "raw_final.json").read_bytes()) == raw


def test_unresolved_no_label_preserved(tmp_path):
    _, packets, left, right = scenario(tmp_path)
    raw = dict(q1=dict(status="unresolved", reason="Synthetic ambiguous source period.", label=None,
                      unresolved_reasons=["The period cannot be resolved."]))
    decision = module.retain_decisions(raw, packets, left, right, ["q1"], receipt_for(tmp_path, raw))[0]
    assert decision["status"] == "unresolved" and decision["label"] is None


def test_run_resume_reuses_raw_execution_and_detects_tampered_decision(tmp_path):
    source, packets, left, right = scenario(tmp_path)
    packets_path = source / "packets.json"
    packets_path.write_bytes(protocol.canonical_bytes(packets))
    calls = []
    def caller(system, payload, schema, output_root, **kwargs):
        calls.append(payload)
        parsed = raw_decision(left)
        return parsed, receipt_for(output_root, parsed)
    args = (packets_path, source / "independent_a.jsonl", source / "independent_b.jsonl", tmp_path / "adjudications")
    result = module.run(*args, caller=caller)
    assert result["accepted"] == 1 and len(calls) == 1
    repeated = module.run(*args, resume=result["output"], caller=caller)
    assert repeated["new_native_calls"] == 0 and len(calls) == 1
    path = Path(result["output"]) / "decisions/q1.json"
    value = json.loads(path.read_bytes())
    value["label"]["answer_points"][0]["text"] = "Changed synthetic answer"
    path.write_bytes(protocol.canonical_bytes(value))
    with pytest.raises(ValueError, match="exact raw"):
        module.run(*args, resume=result["output"], caller=caller)
