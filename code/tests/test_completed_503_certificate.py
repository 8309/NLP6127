"""Pure event/certificate fixtures; no original files or native calls are altered."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def certifier(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("completed_503_test", scripts / "certify_completed_503_result.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def warning(index):
    return f"Reconnecting... {index}/5 (unexpected status 503 Service Unavailable: An error occurred while processing your request. You can retry your request, or contact us through our help center at help.openai.com if the error persists.)"


@pytest.fixture
def event_fixture(certifier):
    raw = b'{"synthetic":true}'
    receipt = dict(accepted=False, process_exit_code=0, execution_id="synthetic-original-thread",
        usage=dict(input_tokens=123, output_tokens=45), errors=["CLI emitted a failed/error event"] * 2)
    events = [dict(type="thread.started", thread_id=receipt["execution_id"]), dict(type="turn.started"),
        dict(type="error", message=warning(2)), dict(type="error", message=warning(3)),
        dict(type="item.completed", item=dict(type="agent_message", text=raw.decode())),
        dict(type="turn.completed", usage=receipt["usage"])]
    return raw, receipt, events


def encode(module, events):
    return b"".join(module.protocol.canonical_bytes(event) + b"\n" for event in events)


def test_generic_event_proof_keeps_operational_false_and_original_usage(certifier, event_fixture):
    raw, receipt, events = event_fixture
    original = certifier.protocol.canonical_bytes(receipt)
    proof = certifier.event_proof(receipt, raw, encode(certifier, events))
    assert proof["native_usage"] == receipt["usage"] and proof["reconnect_attempts"] == [2, 3]
    assert proof["exact_final_raw_bytes"] and receipt["accepted"] is False
    assert certifier.protocol.canonical_bytes(receipt) == original


@pytest.mark.parametrize("bad", ["unknown", "quota", "turn_failed", "tool", "after_final", "nonmonotonic", "missing_warning"])
def test_unknown_or_nonterminal_warning_sequences_are_rejected(certifier, event_fixture, bad):
    raw, receipt, events = event_fixture
    if bad == "unknown":
        events[2]["message"] = "Unexpected HTTP500 error"
    elif bad == "quota":
        events[2]["message"] = "Quota exceeded"
    elif bad == "turn_failed":
        events.insert(-1, dict(type="turn.failed"))
    elif bad == "tool":
        events.insert(-2, dict(type="item.completed", item=dict(type="command_execution")))
    elif bad == "after_final":
        events[-3], events[-2] = events[-2], events[-3]
    elif bad == "nonmonotonic":
        events[2]["message"], events[3]["message"] = warning(3), warning(2)
    else:
        events = events[:2] + events[-2:]
        receipt["errors"] = []
    with pytest.raises(ValueError):
        certifier.event_proof(receipt, raw, encode(certifier, events))


@pytest.mark.parametrize("field, value", [("accepted", True), ("process_exit_code", 1), ("usage", None),
    ("errors", ["Synthetic semantic error"]), ("execution_id", "synthetic-other-thread")])
def test_failed_or_mixed_receipt_premises_cannot_be_reclassified(certifier, event_fixture, field, value):
    raw, receipt, events = event_fixture
    receipt[field] = value
    with pytest.raises(ValueError):
        certifier.event_proof(receipt, raw, encode(certifier, events))


def test_altered_raw_even_equivalent_json_whitespace_is_rejected(certifier, event_fixture):
    raw, receipt, events = event_fixture
    with pytest.raises(ValueError, match="exactly equal"):
        certifier.event_proof(receipt, raw + b"\n", encode(certifier, events))


def test_altered_measured_usage_is_rejected(certifier, event_fixture):
    raw, receipt, events = event_fixture
    events[-1]["usage"] = dict(input_tokens=1, output_tokens=45)
    with pytest.raises(ValueError, match="usage"):
        certifier.event_proof(receipt, raw, encode(certifier, events))


def test_actual_admission_is_not_enabled_by_generic_valid_events(certifier, tmp_path, event_fixture):
    raw, receipt, events = event_fixture
    path = tmp_path / "receipt.json"
    certifier.native.write_new(path, receipt)
    with pytest.raises(ValueError, match="registered"):
        certifier.common_proof("annotation_b38", path)
    with pytest.raises(ValueError, match="registered"):
        certifier.common_proof("source_group2", path)


@pytest.mark.parametrize("artifact", ["stdin.txt", "schema.json", "raw_final.json", "command.json", "native.exe"])
def test_rehashed_receipt_cannot_override_frozen_input_schema_raw_or_binary_pins(certifier, tmp_path, event_fixture, monkeypatch, artifact):
    raw, receipt, events = event_fixture
    binary = tmp_path / "native.exe"
    binary.write_bytes(b"synthetic native executable")
    values = {"stdin.txt": b"synthetic full source input", "schema.json": dict(type="object"),
        "raw_final.json": raw, "command.json": [str(binary), "exec", "-"], "events.jsonl": encode(certifier, events)}
    for name, value in values.items():
        certifier.native.write_new(tmp_path / name, value)
    receipt["artifacts_sha256"] = {name: certifier.protocol.sha256((tmp_path / name).read_bytes()) for name in values}
    path = tmp_path / "receipt.json"
    certifier.native.write_new(path, receipt)
    pin = dict(receipt_sha256=certifier.protocol.sha256(path.read_bytes()),
        stdin_sha256=receipt["artifacts_sha256"]["stdin.txt"], schema_sha256=receipt["artifacts_sha256"]["schema.json"],
        raw_sha256=receipt["artifacts_sha256"]["raw_final.json"], command_sha256=receipt["artifacts_sha256"]["command.json"],
        native_binary_sha256=certifier.protocol.sha256(binary.read_bytes()))
    monkeypatch.setitem(certifier.KNOWN, "annotation_b38", pin)
    assert certifier.common_proof("annotation_b38", path)[-1]["exact_final_raw_bytes"]
    (tmp_path / artifact).write_bytes(b"synthetic changed frozen bytes")
    if artifact != "native.exe":
        receipt["artifacts_sha256"][artifact] = certifier.protocol.sha256((tmp_path / artifact).read_bytes())
        path.write_bytes(certifier.protocol.canonical_bytes(receipt))
        pin["receipt_sha256"] = certifier.protocol.sha256(path.read_bytes())
    with pytest.raises(ValueError, match="Pinned|changed"):
        certifier.common_proof("annotation_b38", path)


def test_isolated_snapshot_supports_frozen_relative_imports_and_dataclasses(certifier, tmp_path):
    path = tmp_path / "synthetic_runtime.py"
    certifier.native.write_new(path, b"from dataclasses import dataclass\nfrom .store import ROOT\n@dataclass\nclass SyntheticState:\n    value: int = 1\n")
    module = certifier.isolated_file(path)
    assert module.__package__ == "oil_evidence" and module.__file__ == str(path)
    assert module.SyntheticState().value == 1 and module.ROOT is not None
