"""Synthetic admission fixtures; never certify files or invoke a native model."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import py_compile
from types import SimpleNamespace
import sys

import pytest


@pytest.fixture
def admission(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("completed_503_admission_tests", scripts / "completed_503_admission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["annotation_b38", "source_group2"])
def case(admission, tmp_path, monkeypatch, request):
    mode = request.param
    root, directory = tmp_path / "synthetic-code", tmp_path / "synthetic-certificate"
    wire, receipt, raw = root / "wire_codec.py", tmp_path / "original/receipt.json", tmp_path / "original/raw_final.json"
    admission.native.write_new(wire, b"synthetic unchanged decoder")
    admission.native.write_new(receipt, dict(accepted=False, execution_id="synthetic-original-native"))
    admission.native.write_new(raw, b"synthetic original raw")
    authorization = tmp_path / "synthetic-authorization.json"
    admission.native.write_new(authorization, dict(root_authorized=True, new_native_calls_allowed=0))
    records = ({"synthetic-body": dict(label="synthetic unchanged label")} if mode == "annotation_b38" else
        dict(decisions=[dict(record_id=f"synthetic-{i}") for i in range(9)], actual_argv_recorded=False,
            actual_command_bytes_unavailable=True, reconstructed_isolation_template=["synthetic-template"],
            source_decisions_pending_certificate_aware_assembly=True))
    verified = dict(mode=mode, original_receipt=str(receipt.resolve()), original_raw=str(raw.resolve()),
        original_receipt_sha256=admission.protocol.sha256(receipt.read_bytes()),
        original_raw_sha256=admission.protocol.sha256(raw.read_bytes()), operational_receipt_accepted=False,
        status="completed_output_mechanically_revalidated_with_transient_503_events", new_native_calls=0,
        original_invocation_usage_counted_once=True, semantic_accuracy_claimed=False, human_gold=False,
        proof=dict(native_usage=dict(input_tokens=123, output_tokens=45)), derived_records=records,
        actual_argv_recorded=mode == "annotation_b38", actual_command_bytes_unavailable=mode == "source_group2",
        derived_record_count=1 if mode == "annotation_b38" else 9)
    calls = []
    def verify(actual_mode, actual_path):
        assert actual_mode == mode and actual_path == receipt.resolve()
        assert sys.dont_write_bytecode is True
        calls.append((actual_mode, actual_path))
        return deepcopy(verified)
    certifier = SimpleNamespace(VERSION="completed-503-zero-call-mechanical-certificate-v1", verify=verify)
    def source_guards(actual_mode, actual_path, actual_receipt, actual_certifier, guards):
        assert actual_receipt["accepted"] is False
        admission.guard(receipt, verified["original_receipt_sha256"], guards)
        admission.guard(raw, verified["original_raw_sha256"], guards)
    monkeypatch.setattr(admission, "ROOT", root)
    monkeypatch.setattr(admission, "DEPENDENCIES", {"wire_codec.py": admission.protocol.sha256(wire.read_bytes())})
    monkeypatch.setattr(admission, "_certifier", lambda: certifier)
    monkeypatch.setattr(admission, "_source_guards", source_guards)
    admission.native.write_new(directory / "derived_records.json", dict(certificate_path=str(directory / "certificate.json"),
        original_receipt=str(receipt.resolve()), operational_receipt_accepted=False, records=records))
    derived_sha = admission.protocol.sha256((directory / "derived_records.json").read_bytes())
    authorization_sha = admission.protocol.sha256(authorization.read_bytes())
    certificate = {key: value for key, value in verified.items() if key != "derived_records"}
    certificate.update(version=certifier.VERSION, certifier_sha256=admission.CERTIFIER_SHA256,
        authorization_path=str(authorization.resolve()), authorization_sha256=authorization_sha,
        derived_records_sha256=derived_sha)
    admission.native.write_new(directory / "certificate.json", certificate)
    monkeypatch.setitem(admission.PINNED, mode, (admission.protocol.sha256((directory / "certificate.json").read_bytes()),
        derived_sha, authorization_sha))
    return SimpleNamespace(mode=mode, directory=directory, certificate=certificate, records=records, receipt=receipt,
        raw=raw, wire=wire, authorization=authorization, calls=calls, certifier=certifier, verified=verified)


def test_read_only_admission_returns_exact_record_shape_and_through_call_guards(admission, case):
    before = {path: path.read_bytes() for path in case.receipt.parent.iterdir()}
    certificate, records, guards = admission.load(case.mode, case.directory)
    assert certificate == case.certificate and records == case.records
    assert {path: path.read_bytes() for path in case.receipt.parent.iterdir()} == before
    assert admission.native.read_json(case.receipt)["accepted"] is False and len(case.calls) == 1
    assert {str(case.receipt), str(case.raw), str(case.wire), str(case.authorization)} <= set(guards)
    if case.mode == "source_group2":
        assert len(records["decisions"]) == 9 and records["actual_argv_recorded"] is False


@pytest.mark.parametrize("name", ["certificate.json", "derived_records.json", "authorization", "wire"])
def test_tampered_sealed_certificate_or_dependency_stops_before_proof(admission, case, name):
    path = case.directory / name if name.endswith(".json") else getattr(case, name)
    path.write_bytes(b"synthetic changed bytes")
    with pytest.raises(ValueError, match="changed"):
        admission.load(case.mode, case.directory)
    assert case.calls == []


@pytest.mark.parametrize("field,value", [("operational_receipt_accepted", True), ("human_gold", True),
    ("semantic_accuracy_claimed", True), ("new_native_calls", 1), ("new_native_calls", False)])
def test_even_resealed_fake_acceptance_or_new_call_claim_is_rejected(admission, case, monkeypatch, field, value):
    path = case.directory / "certificate.json"
    modified = dict(case.certificate, **{field: value})
    path.write_bytes(admission.protocol.canonical_bytes(modified))
    pin = admission.PINNED[case.mode]
    monkeypatch.setitem(admission.PINNED, case.mode, (admission.protocol.sha256(path.read_bytes()), *pin[1:]))
    with pytest.raises(ValueError, match="promote"):
        admission.load(case.mode, case.directory)
    assert case.calls == []


def test_original_receipt_cannot_be_changed_to_accepted(admission, case):
    case.receipt.write_bytes(admission.protocol.canonical_bytes(dict(accepted=True)))
    with pytest.raises(ValueError, match="remain operationally false"):
        admission.load(case.mode, case.directory)


def test_original_mutation_during_full_proof_and_bytecode_flag_are_guarded(admission, case):
    previous = sys.dont_write_bytecode
    def mutate(mode, path):
        case.raw.write_bytes(b"synthetic concurrent mutation")
        return deepcopy(case.verified)
    case.certifier.verify = mutate
    with pytest.raises(ValueError, match="Protected"):
        admission.load(case.mode, case.directory)
    assert sys.dont_write_bytecode is previous


def test_resealed_derived_label_change_still_fails_full_proof_equality(admission, case, monkeypatch):
    path = case.directory / "derived_records.json"
    changed = admission.native.read_json(path)
    changed["records"] = dict(synthetic_rewritten_label=True)
    path.write_bytes(admission.protocol.canonical_bytes(changed))
    derived_sha = admission.protocol.sha256(path.read_bytes())
    certificate = dict(case.certificate, derived_records_sha256=derived_sha)
    cert_path = case.directory / "certificate.json"
    cert_path.write_bytes(admission.protocol.canonical_bytes(certificate))
    pin = admission.PINNED[case.mode]
    monkeypatch.setitem(admission.PINNED, case.mode, (admission.protocol.sha256(cert_path.read_bytes()), derived_sha, pin[2]))
    with pytest.raises(ValueError, match="full-mode proof"):
        admission.load(case.mode, case.directory)


def test_unregistered_mode_cannot_enter_even_with_a_real_directory(admission, tmp_path):
    with pytest.raises(ValueError, match="registered"):
        admission.load("all_other_503_receipts", tmp_path)


def test_verified_byte_loader_ignores_stale_archived_bytecache_without_changing_it(admission, tmp_path):
    source = tmp_path / "archived_snapshot.py"
    source.write_bytes(b"VALUE = 'old'\n")
    py_compile.compile(str(source), doraise=True)
    timestamp = source.stat()
    source.write_bytes(b"VALUE = 'new'\n")
    os.utime(source, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    module = admission._compiled(source, admission.protocol.sha256(source.read_bytes()))
    assert module.VALUE == "new" and module.__file__ == str(source.resolve())
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_verified_byte_loader_preserves_relative_imports_and_dataclass_metadata(admission, tmp_path):
    source = tmp_path / "synthetic_runtime.py"
    source.write_bytes(b"from . import machine_evaluation\nfrom dataclasses import dataclass\n"
        b"@dataclass\nclass Probe:\n    value: int = 7\n")
    module = admission._compiled(source, admission.protocol.sha256(source.read_bytes()))
    assert module.Probe().value == 7 and module.__package__ == "oil_evidence"
    assert not (tmp_path / "__pycache__").exists()
    with pytest.raises(ValueError, match="immutable pin"):
        admission._compiled(source, "0" * 64)
