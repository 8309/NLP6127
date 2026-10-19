"""Pure guards for the approved sealed-input-only source retry controller."""
import importlib.util
from pathlib import Path
import shutil

import pytest

from test_transport_review_recovery import recovery, retry_case, runner, source, accepted_caller


@pytest.fixture
def controller(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("sealed_source_retry_tests", scripts / "run_sealed_source_retry.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def sealed_case(controller, retry_case):
    case = retry_case
    review = case["source"]["root"] / "sealed-review"
    chunk = review / "chunks/chunk-0002"
    shutil.copytree(case["chunk"], chunk)
    ids = list(case["source"]["packets"])[:20]
    manifest = dict(version=controller.recovery.VERSION, implementation_sha256={}, source_bindings={},
        chunks=[ids, ids], original_source_review_stop=case["stopped"])
    controller.recovery.native.write_new(review / "manifest.json", manifest)
    manifest_sha = controller.recovery.protocol.sha256((review / "manifest.json").read_bytes())
    gate = dict(manifest_sha256=manifest_sha, new_native_calls=0, reused_reviews=20,
        historical_cache_records=590, exact_retry_stdin_equal=True, exact_retry_schema_equal=True)
    controller.recovery.native.write_new(review / "export_gate_readback.json", gate)
    controller.recovery.native.write_new(review / "source_retry_preflight_stop.json", dict(
        native_retry_started=False, new_source_native_calls=0, new_source_native_receipts=0,
        one_call_authorization_consumed=False))
    root_auth = controller.recovery.native.read_json(case["authorization"])
    wrapper_sha = controller.recovery.protocol.sha256(Path(controller.recovery.__file__).read_bytes())
    gate_sha = controller.recovery.protocol.sha256((review / "export_gate_readback.json").read_bytes())
    output = review / "adjudications/chunk-0002"
    original = case["attempt"]
    stdin_sha = controller.recovery.protocol.sha256((original / "stdin.txt").read_bytes())
    schema_sha = controller.recovery.protocol.sha256((original / "schema.json").read_bytes())
    root_auth.update(recovery_output_parent=str(output), review_manifest_sha256=manifest_sha,
        wrapper_sha256=wrapper_sha, approved_export_gate_sha256=gate_sha, stdin_sha256=stdin_sha, schema_sha256=schema_sha)
    case["authorization"].write_bytes(controller.recovery.protocol.canonical_bytes(root_auth))
    anchors = dict(manifest=manifest_sha, wrapper=wrapper_sha, export_gate=gate_sha,
        preflight_stop=controller.recovery.protocol.sha256((review / "source_retry_preflight_stop.json").read_bytes()),
        chunk_export=controller.recovery.protocol.sha256((chunk / "export_receipt.json").read_bytes()),
        authorization=controller.recovery.protocol.sha256(case["authorization"].read_bytes()),
        failure=case["stopped"]["failed_pipeline_sha256"], failed_receipt=case["stopped"]["failed_receipt_sha256"],
        checkpoint=controller.recovery.protocol.sha256((case["previous"] / "routing_stop_checkpoint.json").read_bytes()),
        stdin=stdin_sha, schema=schema_sha)
    return {**case, "review": review, "sealed_chunk": chunk, "anchors": anchors, "output": output}


def validated(controller, case):
    return controller.validate_snapshot(case["review"], case["previous"], case["authorization"],
        anchors=case["anchors"], failed_chunk=2, expected_group=10)


def test_sealed_checks_never_prepare_or_parse_later_failed_producer_progress(controller, sealed_case, monkeypatch):
    monkeypatch.setattr(controller.recovery, "prepare", lambda *args, **kwargs: pytest.fail("No producer preparation"))
    monkeypatch.setattr(controller.recovery.frozen, "resolved_progress_snapshot", lambda *args, **kwargs: pytest.fail("No global progress projection"))
    export = controller.recovery.native.read_json(sealed_case["sealed_chunk"] / "export_receipt.json")
    path = Path(export["source_annotation_run"]) / "progress.jsonl"
    path.write_bytes(path.read_bytes() + b'{"type":"failed","batch":"later-source-failure","accepted":false}\n')
    base, stopped, chunk, output, guards, proof = validated(controller, sealed_case)
    assert proof["native_calls"] == 0 and proof["global_progress_snapshot_called"] is False
    assert proof["producer_prepare_called"] is False and proof["later_failed_producer_labels_read"] is False
    assert proof["record_ids"] == sealed_case["group"] and chunk == sealed_case["sealed_chunk"]
    assert output == sealed_case["output"]


@pytest.mark.parametrize("artifact", ["manifest.json", "export_gate_readback.json", "source_retry_preflight_stop.json",
    "chunks/chunk-0002/export_receipt.json", "chunks/chunk-0002/independent_a.jsonl"])
def test_sealed_manifest_export_preflight_or_raw_envelope_tampering_rejected(controller, sealed_case, artifact):
    path = sealed_case["review"] / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="immutable hash"):
        validated(controller, sealed_case)


def test_root_authorization_destination_cannot_be_rebound(controller, sealed_case):
    path = sealed_case["authorization"]
    auth = controller.recovery.native.read_json(path)
    auth["recovery_output_parent"] = str(sealed_case["review"] / "wrong-destination")
    path.write_bytes(controller.recovery.protocol.canonical_bytes(auth))
    sealed_case["anchors"]["authorization"] = controller.recovery.protocol.sha256(path.read_bytes())
    with pytest.raises(ValueError, match="exact once-only"):
        validated(controller, sealed_case)


def test_sealed_controller_calls_unchanged_one_retry_once_then_stops(controller, sealed_case):
    base, stopped, chunk, output, guards, proof = validated(controller, sealed_case)
    calls = []
    result = controller.recovery.one_retry(base, chunk, output, stopped, sealed_case["authorization"],
        caller=accepted_caller(controller.recovery, sealed_case, calls))
    assert calls == [sealed_case["group"]] and result["new_native_calls"] == 1
    assert result["complete_chunk"] is False and result["accepted"] == 10
    controller.recovery.verify_retry_gate(base, chunk, output, stopped)
    with pytest.raises(ValueError, match="already been planned"):
        validated(controller, sealed_case)
