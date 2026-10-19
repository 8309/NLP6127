"""Pure narrow assembly-bridge tests; no real assembly or native invocation."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from test_compact_machine_adjudication import codec, paired, raw_decision


@pytest.fixture
def bridge(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("certified_assembly_test", scripts / "assemble_certified_machine_freeze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def context(bridge, codec, paired, tmp_path):
    directory = tmp_path / "original-owner/executions/synthetic-review"
    directory.mkdir(parents=True)
    raw = raw_decision()
    raw_bytes = codec.protocol.canonical_bytes(raw)
    (directory / "raw_final.json").write_bytes(raw_bytes)
    receipt = dict(accepted=False, execution_id="synthetic-source-review-context", observed_model=None,
        artifact_directory=str(directory), artifacts_sha256={"raw_final.json": codec.protocol.sha256(raw_bytes)})
    (directory / "receipt.json").write_bytes(codec.protocol.canonical_bytes(receipt))
    certificate_dir = tmp_path / "external-certificate"
    certificate_dir.mkdir()
    (certificate_dir / "certificate.json").write_bytes(b'{"synthetic":true}')
    value = dict(decisions_path=directory.parent.parent / "decisions.jsonl", receipt_path=directory / "receipt.json",
        receipt=receipt, raw_path=directory / "raw_final.json", raw=raw_bytes, rows={"synthetic-q": {}},
        guards={}, certificate_directory=certificate_dir, completion_path=tmp_path / "completion.json")
    assembler = bridge.private_assembler(value)
    packets, left, right = paired
    decisions = assembler.compact.retain_compact_decisions(raw, packets, left, right, ["synthetic-q"], receipt)
    value["rows"] = {row["record_id"]: row for row in decisions}
    value["decisions_path"].write_bytes(b"".join(codec.protocol.canonical_bytes(row) + b"\n" for row in decisions))
    return value, assembler, packets, left, right


def test_only_private_compact_callback_admits_exact_original_false_source(bridge, codec, context):
    value, assembler, packets, left, right = context
    before = value["receipt_path"].read_bytes()
    raw = raw_decision()
    retained = bridge.certified_retained(value, value["receipt"], raw, ["synthetic-q"])
    assert retained == (value["raw_path"], value["receipt_path"])
    assert assembler.compact is not codec
    with pytest.raises(ValueError, match="accepted"):
        codec._retained_raw(value["receipt"], raw, ["synthetic-q"])
    assert value["receipt_path"].read_bytes() == before and value["receipt"]["accepted"] is False


def test_narrow_predicate_rebuilds_complete_selected_source_label(bridge, context):
    value, assembler, packets, left, right = context
    decision = value["rows"]["synthetic-q"]
    assert assembler._native_admitted(value["receipt"], value["raw"], decision["adjudication"], "synthetic-q",
        value["decisions_path"], value["receipt_path"], packets, left, right)
    assert decision["label"] == left["synthetic-q"]["label"] and decision["human_gold"] is False


@pytest.mark.parametrize("mutation", ["unknown_receipt", "relocated_owner", "wrong_decision_file", "unknown_record", "changed_review"])
def test_unknown_false_receipts_or_owners_never_use_certificate(bridge, context, mutation):
    value, assembler, packets, left, right = context
    receipt, review = deepcopy(value["receipt"]), deepcopy(value["rows"]["synthetic-q"]["adjudication"])
    path, receipt_path, rid = value["decisions_path"], value["receipt_path"], "synthetic-q"
    if mutation == "unknown_receipt":
        receipt["execution_id"] = "synthetic-unregistered-review"
    elif mutation == "relocated_owner":
        receipt_path = receipt_path.with_name("relocated-receipt.json")
    elif mutation == "wrong_decision_file":
        path = path.with_name("another-decision-file.jsonl")
    elif mutation == "unknown_record":
        rid = "synthetic-unregistered-record"
    else:
        review["reason"] = "Synthetic altered review"
    with pytest.raises(ValueError, match="unknown false"):
        assembler._native_admitted(receipt, value["raw"], review, rid, path, receipt_path, packets, left, right)


def test_normal_accepted_receipts_still_use_original_strict_compact_retention(bridge, context):
    value, assembler, packets, left, right = context
    receipt = deepcopy(value["receipt"])
    receipt["accepted"] = True
    assert assembler._native_admitted(receipt, value["raw"], {}, "synthetic-unrelated", Path("different"), Path("different"), {}, {}, {})
    # The original retained file is false: the unchanged normal callback rejects a synthetic promotion.
    with pytest.raises(ValueError, match="differs from retained"):
        assembler.compact._retained_raw(receipt, raw_decision(), ["synthetic-q"])


@pytest.mark.parametrize("mutation", ["raw", "receipt", "pair", "selected_label"])
def test_unchanged_raw_pair_and_selected_label_guards_are_preserved(bridge, context, mutation):
    value, assembler, packets, left, right = context
    if mutation == "raw":
        value["raw_path"].write_bytes(b"{}")
    elif mutation == "receipt":
        value["receipt_path"].write_bytes(b"{}")
    elif mutation == "pair":
        left["synthetic-q"]["label"]["qrels"]["p-future"] = 1
    else:
        value["rows"]["synthetic-q"]["label"]["reasons"] = ["Synthetic changed selected label"]
    with pytest.raises(ValueError):
        assembler._native_admitted(value["receipt"], value["raw"], value["rows"]["synthetic-q"]["adjudication"],
            "synthetic-q", value["decisions_path"], value["receipt_path"], packets, left, right)


def test_fresh_assembly_receipt_has_explicit_certificate_scope_and_no_new_calls(bridge, context, tmp_path):
    value, assembler, packets, left, right = context
    target = tmp_path / "fresh-output/assembly_receipt.json"
    assembler.write_new(target, bridge.protocol.canonical_bytes(dict(status="synthetic-not-real-assembly", source_sha256={})))
    receipt = bridge.native.read_json(target)
    assert receipt["assembly_bridge_version"] == bridge.VERSION and receipt["certified_source_rows"] == 9
    assert receipt["original_false_source_receipt_retained"] is True and receipt["new_native_calls"] == 0
    assert receipt["actual_source_argv_recorded"] is False and receipt["actual_source_command_bytes_unavailable"] is True
    assert value["receipt"]["accepted"] is False


def test_dynamic_fresh_digest_cannot_overwrite_an_existing_source_seal(bridge, tmp_path):
    path = tmp_path / "synthetic-source.txt"
    path.write_bytes(b"original sealed source")
    original = bridge.protocol.sha256(path.read_bytes())
    guards = {str(path.resolve()): original}
    path.write_bytes(b"changed source after its original read")
    fresh = bridge.protocol.sha256(path.read_bytes())
    with pytest.raises(ValueError, match="sealed.*cannot be overwritten"):
        bridge.guard(path, fresh, guards)
    assert guards[str(path.resolve())] == original
