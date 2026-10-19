"""One approved source retry from an immutable pre-failure export only.

This controller never prepares a producer, projects its progress, reads later
failed labels, or starts full source continuation. It calls the unchanged
one_retry API once after verifying the already approved physical input bytes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import pipeline_transport_recovery_adjudication as recovery

VERSION = "sealed-source-export-one-shot-transport-retry-v1"
ANCHORS = dict(
    manifest="35504bded78c74846645563abe48d35884c46e32ee19cfbc27373aa3d8c42559",
    wrapper="0d3ed9f3a767ff79d0a23a97ee3f4a94e9e62b91f259e6b3430d74c5c99d3119",
    export_gate="ff32a98ecc0b8fd738a8ab307c34cf7bc98c926bf053ab9b37316bd97634a294",
    preflight_stop="ef27e078bfc71d47fda3e38bbb961643ff99df3f0666fafab3c5ef78d32c69be",
    chunk_export="14f571975888d78e0e0a05960b63b12e73e7ccb48673ff315cc54abc8542e3d0",
    authorization="c5e7034d7c8de819a83156d95abed4b919e5d60671a2451de2eeb507d484a27e",
    failure=recovery.FAILED_PIPELINE_SHA256,
    failed_receipt=recovery.FAILED_RECEIPT_SHA256,
    checkpoint=recovery.STOP_CHECKPOINT_SHA256,
    stdin="6c346916ee3a92e6df8c1c0579dabcbaa881830f2a604b00a4a37217aeb4769b",
    schema="93bf081ce8519be84798a4ce9589403b1e991cf8fcbd5aee0bcb09b01f89f653")


def validate_snapshot(review_run, previous_run, authorization, *, anchors=None, failed_chunk=29, expected_group=9):
    """Zero-call physical checks; later producer events are not interpreted."""
    expected = ANCHORS if anchors is None else anchors
    review, previous, authorization = map(lambda value: Path(value).resolve(), (review_run, previous_run, authorization))
    guards = {}
    files = dict(manifest=review / "manifest.json", wrapper=Path(recovery.__file__),
        export_gate=review / "export_gate_readback.json", preflight_stop=review / "source_retry_preflight_stop.json",
        chunk_export=review / "chunks" / f"chunk-{failed_chunk:04d}" / "export_receipt.json",
        authorization=authorization)
    for key, path in files.items():
        recovery.guard_file(path, guards, expected[key])
    manifest = recovery.native.read_json(files["manifest"])
    if manifest.get("version") != recovery.VERSION:
        raise ValueError("Sealed source retry does not match the approved unchanged recovery wrapper")
    recovery.merge_guards(guards, manifest["implementation_sha256"])
    recovery.merge_guards(guards, manifest["source_bindings"])
    stopped = recovery.verify_stopped_review(previous, receipt_sha256=expected["failed_receipt"],
        failure_sha256=expected["failure"], checkpoint_sha256=expected["checkpoint"], expected_last_chunk=failed_chunk - 1)
    recovery.merge_guards(guards, stopped["protected_sha256"])
    if manifest["original_source_review_stop"] != stopped:
        raise ValueError("Sealed source manifest no longer binds the exact rejected original invocation")
    gate = recovery.native.read_json(files["export_gate"])
    if (gate.get("manifest_sha256") != expected["manifest"] or gate.get("new_native_calls") != 0
            or gate.get("reused_reviews") != 20 * (failed_chunk - 1)
            or gate.get("historical_cache_records") != 590
            or gate.get("exact_retry_stdin_equal") is not True or gate.get("exact_retry_schema_equal") is not True):
        raise ValueError("Approved zero-call export gate differs from its sealed scope and input proof")
    preflight = recovery.native.read_json(files["preflight_stop"])
    if (preflight.get("native_retry_started") is not False or preflight.get("new_source_native_calls") != 0
            or preflight.get("new_source_native_receipts") != 0
            or preflight.get("one_call_authorization_consumed") is not False):
        raise ValueError("Original preflight must prove the one authorized invocation was not consumed")
    chunk = files["chunk_export"].parent
    export = recovery.native.read_json(files["chunk_export"])
    if export["record_ids"] != manifest["chunks"][failed_chunk - 1] or export["records"] != 20:
        raise ValueError("Sealed source chunk is not the exact original fixed twenty-question export")
    for name, digest in export["artifacts_sha256"].items():
        recovery.guard_file(recovery.native.contained_file(chunk, name), guards, digest)
    recovery.merge_guards(guards, export["source_bindings"])
    progress = Path(export["source_annotation_run"]) / "progress.jsonl"
    with progress.open("rb") as stream:
        prefix = stream.read(export["source_progress_prefix_bytes"])
    if recovery.protocol.sha256(prefix) != export["source_progress_prefix_sha256"]:
        raise ValueError("Sealed source export durable prefix changed")
    # Read only the already manifested prefix, never the later failed events.
    base = recovery.legacy.isolated_base()
    recovery.legacy.configure(base, recovery.legacy.isolated_base(), dict(output=None, records=0, chunks={}, sha256={}))
    base.VERSION = recovery.VERSION
    spec, packets, left, right = recovery.bounded_spec(base, chunk)
    group = spec["groups"][0]
    if len(group) != expected_group or group != stopped["failed_group"]:
        raise ValueError("One-shot sealed retry group differs from the originally rejected invocation")
    payload, schema = base.adjudicator.prepare_batch(packets, left, right, group)
    stdin, schema_bytes = base.actual_adjudication_bytes(payload, schema)
    original_dir = Path(stopped["failed_receipt"]).parent
    if (recovery.protocol.sha256(stdin) != expected["stdin"]
            or recovery.protocol.sha256(schema_bytes) != expected["schema"]
            or stdin != (original_dir / "stdin.txt").read_bytes()
            or schema_bytes != (original_dir / "schema.json").read_bytes()):
        raise ValueError("Sealed retry full source input and schema are not byte-identical to the original")
    output_parent = review / "adjudications" / f"chunk-{failed_chunk:04d}"
    auth = recovery.native.read_json(authorization)
    if (Path(auth.get("recovery_output_parent", "")).resolve() != output_parent
            or auth.get("review_manifest_sha256") != expected["manifest"]
            or auth.get("wrapper_sha256") != expected["wrapper"]
            or auth.get("approved_export_gate_sha256") != expected["export_gate"]
            or auth.get("max_retry_native_calls") != 1
            or auth.get("stdin_sha256") != expected["stdin"] or auth.get("schema_sha256") != expected["schema"]):
        raise ValueError("Root authorization does not bind this exact once-only sealed input destination")
    if (output_parent / "bounded-run/retry_request.json").exists():
        raise ValueError("The one approved sealed source invocation has already been planned")
    recovery.guard_file(Path(__file__), guards)
    recovery.native.protected_unchanged(guards)
    proof = dict(version=VERSION, controller_sha256=guards[str(Path(__file__).resolve())],
        approved_manifest_sha256=expected["manifest"], approved_wrapper_sha256=expected["wrapper"],
        sealed_export_sha256=expected["chunk_export"], approved_authorization_sha256=expected["authorization"],
        preserved_preflight_stop_sha256=expected["preflight_stop"], record_ids=group,
        stdin_sha256=expected["stdin"], schema_sha256=expected["schema"], native_calls=0,
        producer_prepare_called=False, global_progress_snapshot_called=False,
        later_failed_producer_labels_read=False, failed_producer_labels_accepted=False,
        full_source_continuation_authorized=False, original_failed_usage=None,
        original_failed_usage_status="unknown_not_zero", human_gold=False)
    return base, stopped, chunk, output_parent, guards, proof


def run(review_run, previous_run, authorization, *, check_only=False, caller=None):
    base, stopped, chunk, output_parent, guards, proof = validate_snapshot(review_run, previous_run, authorization)
    review = Path(review_run).resolve()
    proof_path = review / "sealed_snapshot_check.json"
    if proof_path.exists():
        if recovery.native.read_json(proof_path) != proof:
            raise ValueError("Existing sealed source preflight proof changed")
    else:
        recovery.native.write_new(proof_path, proof)
    if check_only:
        return proof
    recovery.native.protected_unchanged(guards)
    result = recovery.one_retry(base, chunk, output_parent, stopped, authorization, caller=caller)
    recovery.native.protected_unchanged(guards)
    recovery.verify_retry_gate(base, chunk, output_parent, stopped)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("review-run", "previous-run", "authorization"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = run(args.review_run, args.previous_run, args.authorization, check_only=args.check_only)
    print(json.dumps(result, indent=2))
