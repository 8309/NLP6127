"""Metadata-only tests use a synthetic backend, never an actual model service."""
import importlib.util
import json
from pathlib import Path

import pytest

from test_correct_codex_wire_batch import adapter, rows, prepared, backend


@pytest.fixture
def verifier(monkeypatch, adapter):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("revalidate_pair_test", scripts / "revalidate_codex_corrective_pair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "correction", adapter)
    monkeypatch.setattr(module, "native", adapter.native)
    return module


@pytest.fixture
def false_alarm_pair(verifier, adapter, prepared, monkeypatch):
    directory, authorization, setup, source = prepared
    pretty = directory / "root_authorization.json"
    pretty.write_bytes((json.dumps(adapter.native.read_json(authorization), indent=2) + "\n").encode())
    calls = []
    receipts, certificate = adapter.execute_pair_once(directory, pretty, native_setup=setup, executor=backend(adapter, calls))
    assert len(calls) == 2
    assert all(value["errors"] == [verifier.FALSE_ALARM] for value in receipts.values())
    monkeypatch.setattr(verifier, "EXPECTED_MANIFEST", adapter.protocol.sha256((directory / "manifest.json").read_bytes()))
    monkeypatch.setattr(verifier, "EXPECTED_AUTHORIZATION", adapter.protocol.sha256(pretty.read_bytes()))
    monkeypatch.setattr(verifier, "EXPECTED_ADAPTER", adapter.protocol.sha256(Path(adapter.__file__).read_bytes()))
    monkeypatch.setattr(verifier, "EXPECTED_FAILED_CERTIFICATE", adapter.protocol.sha256((directory / "certificate.json").read_bytes()))
    monkeypatch.setattr(verifier, "EXPECTED_RECEIPTS", {s: adapter.protocol.sha256((directory / s / "execution/receipt.json").read_bytes()) for s in adapter.native.STREAMS})
    for index, entry in enumerate(source["rejected_receipts"]):
        fake = directory / ("synthetic-old-" + str(index))
        fake.mkdir()
        adapter.native.write_new(fake / "command.json", adapter.native.cli_command(setup[0], fake / "context",
            fake / "context/schema.json", fake / "context/final.json", None, []))
        entry["path"] = str(fake / "receipt.json")
    adapter.native.write_new(directory / "root_metadata_authorization.json", dict(root_authorized=True,
        route=verifier.VERSION, new_native_calls_allowed=0,
        original_authorization_sha256=verifier.EXPECTED_AUTHORIZATION,
        reviewed_receipt_sha256=verifier.EXPECTED_RECEIPTS,
        metadata_verifier_sha256=adapter.protocol.sha256(Path(verifier.__file__).read_bytes())))
    return directory, calls


def test_pretty_authorization_canonical_copy_resolved_without_calls_or_label_changes(verifier, false_alarm_pair):
    directory, calls = false_alarm_pair
    old_receipts = {s: (directory / s / "execution/receipt.json").read_bytes() for s in ("a", "b")}
    old_envelopes = {s: (directory / s / "execution/envelopes.json").read_bytes() for s in ("a", "b")}
    result = verifier.seal(directory)
    report = result["verification"]
    assert report["validation_status"] == "source_valid_pending_adjudication"
    assert report["new_native_calls"] == 0 and report["native_receipts_remain_accepted_false"]
    assert report["corrective_pair_agreement"]["paired_records"] == 10
    for stream in ("a", "b"):
        assert len(verifier.completed_correction_records(directory, stream)) == 10
        assert (directory / stream / "execution/receipt.json").read_bytes() == old_receipts[stream]
        assert (directory / stream / "execution/envelopes.json").read_bytes() == old_envelopes[stream]
    assert len(calls) == 2


def test_changed_original_authorization_fails(verifier, false_alarm_pair):
    directory, _ = false_alarm_pair
    path = directory / "root_authorization.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="authorization differs"):
        verifier.verify(directory)


def test_changed_copied_authorization_values_fail(verifier, false_alarm_pair, monkeypatch):
    directory, _ = false_alarm_pair
    path = directory / "a/execution/authorization.json"
    value = verifier.strict_json(path)
    value["no_label_feedback"] = False
    path.write_bytes(verifier.protocol.canonical_bytes(value))
    receipt_path = directory / "a/execution/receipt.json"
    receipt = verifier.strict_json(receipt_path)
    receipt["artifacts_sha256"]["authorization.json"] = verifier.protocol.sha256(path.read_bytes())
    receipt_path.write_bytes(verifier.protocol.canonical_bytes(receipt))
    verifier.EXPECTED_RECEIPTS["a"] = verifier.protocol.sha256(receipt_path.read_bytes())
    auth_path = directory / "root_metadata_authorization.json"
    auth = verifier.strict_json(auth_path)
    auth["reviewed_receipt_sha256"] = verifier.EXPECTED_RECEIPTS
    auth_path.write_bytes(verifier.protocol.canonical_bytes(auth))
    with pytest.raises(ValueError, match="JSON values differ"):
        verifier.verify(directory)


def test_mixed_quote_error_is_not_a_software_false_alarm(verifier, false_alarm_pair):
    directory, _ = false_alarm_pair
    path = directory / "a/execution/receipt.json"
    value = verifier.strict_json(path)
    value["errors"].append("Every answer point needs citations with exact eligible quotes")
    path.write_bytes(verifier.protocol.canonical_bytes(value))
    verifier.EXPECTED_RECEIPTS["a"] = verifier.protocol.sha256(path.read_bytes())
    auth_path = directory / "root_metadata_authorization.json"
    auth = verifier.strict_json(auth_path)
    auth["reviewed_receipt_sha256"] = verifier.EXPECTED_RECEIPTS
    auth_path.write_bytes(verifier.protocol.canonical_bytes(auth))
    with pytest.raises(ValueError, match="Mixed/model/native/quota"):
        verifier.verify(directory)


def test_source_input_tamper_rejected_even_with_self_reported_new_artifact_hash(verifier, false_alarm_pair):
    directory, _ = false_alarm_pair
    path = directory / "a/execution/stdin.txt"
    path.write_bytes(path.read_bytes() + b"tampered")
    receipt_path = directory / "a/execution/receipt.json"
    receipt = verifier.strict_json(receipt_path)
    receipt["artifacts_sha256"]["stdin.txt"] = verifier.protocol.sha256(path.read_bytes())
    receipt_path.write_bytes(verifier.protocol.canonical_bytes(receipt))
    verifier.EXPECTED_RECEIPTS["a"] = verifier.protocol.sha256(receipt_path.read_bytes())
    auth_path = directory / "root_metadata_authorization.json"
    auth = verifier.strict_json(auth_path)
    auth["reviewed_receipt_sha256"] = verifier.EXPECTED_RECEIPTS
    auth_path.write_bytes(verifier.protocol.canonical_bytes(auth))
    with pytest.raises(ValueError, match="physical input changed"):
        verifier.verify(directory)


def test_same_json_but_noncanonical_copy_is_not_allowed(verifier, tmp_path):
    original, copied = tmp_path / "original.json", tmp_path / "copy.json"
    original.write_bytes(b'{\n  "allowed": true\n}\n')
    copied.write_bytes(b'{ "allowed" : true }')
    with pytest.raises(ValueError, match="canonical serialization"):
        verifier.authorization_equivalence(original, copied, verifier.protocol.sha256(original.read_bytes()))


def test_duplicate_json_key_rejected(verifier, tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_bytes(b'{"allowed":true,"allowed":false}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        verifier.strict_json(path)


def test_usage_ledger_duplicate_or_missing_execution_fails(verifier, false_alarm_pair):
    directory, _ = false_alarm_pair
    path = directory / "execution_usage.jsonl"
    first = path.read_bytes().splitlines(keepends=True)[0]
    path.write_bytes(first + first)
    with pytest.raises(ValueError, match="usage ledger"):
        verifier.verify(directory)


def test_fresh_context_destinations_strict(verifier):
    command = ["cli", "--cd", "synthetic-context", "--output-schema", "synthetic-context/schema.json",
        "--output-last-message", "synthetic-context/final.json"]
    assert verifier.native_destinations(command)["--cd"].name == "synthetic-context"
    with pytest.raises(ValueError, match="One exact"):
        verifier.native_destinations(command + ["--cd", "another"])
    with pytest.raises(ValueError, match="destinations"):
        verifier.native_destinations(["cli", "--cd", "synthetic-context", "--output-schema", "different/schema.json",
            "--output-last-message", "synthetic-context/final.json"])
