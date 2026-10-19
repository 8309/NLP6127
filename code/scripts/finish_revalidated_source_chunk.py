"""Finish one sealed 9+9+2 source chunk without replaying completed outputs.

The original accepted gate, false HTTP503 receipt and raw bytes stay immutable.
An external, allowlisted zero-call certificate admits only the second group.
Only the final two cases may receive one newly authorized native invocation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

import pipeline_transport_recovery_adjudication as recovery

native, protocol, legacy = recovery.native, recovery.protocol, recovery.legacy
VERSION = "sealed-revalidated-source-chunk-completion-v1"
AUTH_VERSION = "source-chunk-completion-one-call-authorization-v1"
ANCHORS = dict(
    manifest="35504bded78c74846645563abe48d35884c46e32ee19cfbc27373aa3d8c42559",
    export="14f571975888d78e0e0a05960b63b12e73e7ccb48673ff315cc54abc8542e3d0",
    inputs="77539763483364992830b0479298e3387c427b280cd6fbc427fe0fa95b4ed15a",
    gate="6158e95d60e0a09d566977bd5f55b01b9ce41ba6b01adc0884c057c9dbbcce55",
    request="cc994a1041fbf93ac1bb96fcaa643b654ba355899f4f4de9c6f4d065917d283b",
    binding="1941a778e577fc969da65b90d10c276508caaef4b71aeb7a69d9bba50abae62d",
    first_receipt="7540edff01d9ce16525909522b6a1a9a36286c5df9a155301ab5abeb1472a99c",
    second_receipt="f76bad2521713df93e9de8ccc8fc32cf501d4ecb6160d8940efd34de14dcfbda",
    old_authorization="c5e7034d7c8de819a83156d95abed4b919e5d60671a2451de2eeb507d484a27e",
    certificate="3d7abfdec7961bb325449de37452dae1c991f32935490770940f2e6d3f0ee7cd",
    derived="1c019c508d549c5a8e32e806976887f98119afa7e4e6d960e2238ddc893e13f6",
    third_stdin="b084fa939117e9ba8f79f42b522f782b59f58a77afbdcacc16b4a9128feab25f",
    third_schema="2d1c44d9415ee92037282b235c978dca9d221315e50768c8c7819a50c63d5329")


def load_admission(directory):
    """The shared helper admits the exact registered false receipt, not a class."""
    import completed_503_admission
    return completed_503_admission.load("source_group2", directory)


def physical(context, group):
    payload, schema = context["base"].adjudicator.prepare_batch(
        context["packets"], context["left"], context["right"], group)
    stdin, schema_bytes = context["base"].actual_adjudication_bytes(payload, schema)
    if len(stdin) + len(schema_bytes) > 262144:
        raise ValueError("A complete sealed source group exceeds the hard byte bound")
    return payload, schema, stdin, schema_bytes


def accepted_group(context, path, group, guards):
    """Rebuild a normal accepted native group against original full source bytes."""
    path = Path(path).resolve()
    if not path.is_relative_to(context["owner"] / "executions"):
        raise ValueError("Native source receipt escapes its original execution owner")
    receipt = recovery.receipt_artifacts(path, guards)
    if (receipt.get("accepted") is not True or receipt.get("process_exit_code") != 0
            or receipt.get("codex_cli_invocations") != 1 or receipt.get("external_paid_api_calls") != 0):
        raise ValueError("A new or normal source native receipt was not accepted")
    _, _, stdin, schema_bytes = physical(context, group)
    if ((path.parent / "stdin.txt").read_bytes() != stdin
            or (path.parent / "schema.json").read_bytes() != schema_bytes):
        raise ValueError("Native full source input or schema differs from its sealed group")
    events = native.parse_execution_events((path.parent / "events.jsonl").read_bytes())
    if (events["errors"] or events["thread_id"] != receipt.get("execution_id")
            or events["usage"] != receipt.get("usage")
            or events["observed_model"] != receipt.get("observed_model")
            or sum(row.get("type") == "thread.started" for row in events["events"]) != 1
            or sum(row.get("type") == "turn.completed" for row in events["events"]) != 1):
        raise ValueError("Normal source completion identity, events or measured usage changed")
    originals = {stream[key]["identity"]["execution_id"]
                 for stream in (context["left"], context["right"]) for key in context["ids"]}
    if not receipt.get("execution_id") or receipt["execution_id"] in originals:
        raise ValueError("Source review must have a fresh context distinct from both originals")
    raw = native.read_json(path.parent / "raw_final.json")
    decisions = context["base"].adjudicator.retain_decisions(raw, context["packets"],
        context["left"], context["right"], group, receipt)
    return receipt, raw, decisions


def check_prefix(context):
    export = context["export"]
    with (Path(export["source_annotation_run"]) / "progress.jsonl").open("rb") as stream:
        prefix = stream.read(export["source_progress_prefix_bytes"])
    if protocol.sha256(prefix) != export["source_progress_prefix_sha256"]:
        raise ValueError("Original sealed source durable prefix changed")


def validate_snapshot(review_run, certificate_directory, *, anchors=None, admission_loader=None):
    """Read-only preflight: no producer preparation, global progress or writes."""
    anchors = ANCHORS if anchors is None else anchors
    review, certificate_directory = Path(review_run).resolve(), Path(certificate_directory).resolve()
    chunk, owner = review / "chunks/chunk-0029", review / "adjudications/chunk-0029/bounded-run"
    guards = {}
    paths = dict(manifest=review / "manifest.json", export=chunk / "export_receipt.json",
        inputs=owner / "inputs.json", gate=owner / "retry_gate.json", request=owner / "retry_request.json",
        binding=owner / "retained_batches/batch-0001.json", certificate=certificate_directory / "certificate.json",
        derived=certificate_directory / "derived_records.json")
    for key, path in paths.items():
        recovery.guard_file(path, guards, anchors[key])
    manifest, export, spec, gate, request, binding = (native.read_json(paths[key]) for key in
        ("manifest", "export", "inputs", "gate", "request", "binding"))
    if manifest.get("version") != recovery.VERSION or spec.get("version") != recovery.VERSION:
        raise ValueError("Sealed source completion protocol differs from its original owner")
    recovery.merge_guards(guards, manifest["implementation_sha256"])
    recovery.merge_guards(guards, manifest["original_source_review_stop"]["protected_sha256"])
    recovery.merge_guards(guards, export["source_bindings"])
    for name, digest in export["artifacts_sha256"].items():
        recovery.guard_file(native.contained_file(chunk, name), guards, digest)
    base = legacy.isolated_base()
    legacy.configure(base, legacy.isolated_base(), dict(output=None, records=0, chunks={}, sha256={}))
    base.VERSION = recovery.VERSION
    rebuilt, packets, left, right = recovery.bounded_spec(base, chunk)
    ids, groups = list(packets), spec["groups"]
    if (rebuilt != spec or [len(group) for group in groups] != [9, 9, 2]
            or [key for group in groups for key in group] != ids or len(set(ids)) != 20
            or export["record_ids"] != ids or export["records"] != 20
            or manifest["chunks"][28] != ids):
        raise ValueError("Original fixed twenty cases or sealed 9+9+2 groups changed")
    recovery.merge_guards(guards, spec["input_sha256"])
    context = dict(review=review, chunk=chunk, owner=owner, base=base, packets=packets,
        left=left, right=right, ids=ids, groups=groups, export=export, guards=guards,
        anchors=anchors, certificate_directory=certificate_directory)
    first_path = native.contained_file(owner, binding["receipt_path"])
    recovery.guard_file(first_path, guards, anchors["first_receipt"])
    first, _, first_decisions = accepted_group(context, first_path, groups[0], guards)
    authorization = Path(request["authorization"]).resolve()
    recovery.guard_file(authorization, guards, anchors["old_authorization"])
    old_auth = native.read_json(authorization)
    stopped = manifest["original_source_review_stop"]
    original_path = Path(stopped["failed_receipt"]).resolve()
    recovery.guard_file(original_path, guards, stopped["failed_receipt_sha256"])
    original = recovery.receipt_artifacts(original_path, guards)
    recovery.check_retry_identity(first, original)
    _, _, first_stdin, first_schema = physical(context, groups[0])
    if (first_stdin != (original_path.parent / "stdin.txt").read_bytes()
            or first_schema != (original_path.parent / "schema.json").read_bytes()
            or request.get("stdin_sha256") != protocol.sha256(first_stdin)
            or request.get("schema_sha256") != protocol.sha256(first_schema)
            or request.get("original_failed_receipt") != stopped["failed_receipt"]
            or request.get("original_failed_receipt_sha256") != stopped["failed_receipt_sha256"]
            or request.get("original_failure_sha256") != stopped["failed_pipeline_sha256"]
            or request.get("record_ids") != groups[0] or request.get("max_new_native_calls") != 1
            or request.get("authorization_sha256") != guards[str(authorization)]
            or old_auth.get("version") != "source-review-transport-retry-authorization-v1"
            or old_auth.get("max_retry_native_calls") != 1
            or old_auth.get("failed_receipt_sha256") != stopped["failed_receipt_sha256"]
            or old_auth.get("original_failure_sha256") != stopped["failed_pipeline_sha256"]
            or old_auth.get("same_physical_input") is not True or old_auth.get("same_model_request") is not True
            or old_auth.get("new_failures_stop") is not True
            or Path(old_auth.get("recovery_output_parent", "")).resolve() != owner.parent
            or gate.get("version") != recovery.VERSION or gate.get("new_native_calls") != 1
            or gate.get("validation_status") != "one_exact_input_retry_retained_pending_full_chunk"
            or gate.get("complete_chunk") is not False or gate.get("record_ids") != groups[0]
            or gate.get("retry_request_sha256") != anchors["request"]
            or gate.get("first_batch_binding_sha256") != anchors["binding"]
            or gate.get("receipt_path") != str(first_path) or gate.get("receipt_sha256") != anchors["first_receipt"]
            or binding.get("receipt_sha256") != anchors["first_receipt"] or binding.get("record_ids") != groups[0]
            or binding.get("decisions") != first_decisions or gate.get("execution_id") != first["execution_id"]
            or gate.get("usage") != first.get("usage") or gate.get("records") != 9
            or gate.get("accepted") != sum(row["status"] == "accepted" for row in first_decisions)
            or gate.get("unresolved") != sum(row["status"] == "unresolved" for row in first_decisions)):
        raise ValueError("Original first-nine gate, authorization or exact raw selection changed")
    certificate, derived, admission_guards = (admission_loader or load_admission)(certificate_directory)
    recovery.merge_guards(guards, admission_guards)
    second_path = Path(certificate["original_receipt"]).resolve()
    recovery.guard_file(second_path, guards, anchors["second_receipt"])
    second = recovery.receipt_artifacts(second_path, guards)
    recovery.check_retry_identity(second, first)
    second_decisions = derived["decisions"]
    if (not second_path.is_relative_to(owner / "executions") or second.get("accepted") is not False
            or second["execution_id"] == first["execution_id"]
            or [row["record_id"] for row in second_decisions] != groups[1]
            or derived.get("actual_argv_recorded") is not False
            or derived.get("actual_command_bytes_unavailable") is not True
            or derived.get("source_decisions_pending_certificate_aware_assembly") is not True):
        raise ValueError("Only the exact certified false second-nine output may be reused")
    _, _, second_stdin, second_schema = physical(context, groups[1])
    if ((second_path.parent / "stdin.txt").read_bytes() != second_stdin
            or (second_path.parent / "schema.json").read_bytes() != second_schema):
        raise ValueError("Certified second group differs from original full source bytes")
    for row in second_decisions:
        review_data = row["adjudication"]
        if (Path(review_data["execution_receipt_path"]).resolve() != second_path
                or Path(review_data["raw_response_path"]).resolve() != second_path.parent / "raw_final.json"
                or review_data["original_output_sha256"] != [protocol.sha256(protocol.canonical_bytes(stream[row["record_id"]]))
                                                           for stream in (left, right)]
                or review_data["execution_id"] != second["execution_id"]):
            raise ValueError("Certified decisions lost exact paired-source or native ownership")
        if row["status"] == "accepted" and protocol.validate_label(row["label"], packets[row["record_id"]]):
            raise ValueError("Certified selected label fails unchanged source/date/quote guards")
    _, _, third_stdin, third_schema = physical(context, groups[2])
    if protocol.sha256(third_stdin) != anchors["third_stdin"] or protocol.sha256(third_schema) != anchors["third_schema"]:
        raise ValueError("Planned final-two full stdin or schema changed")
    context.update(first_path=first_path, first_receipt=first, first_decisions=first_decisions,
        second_path=second_path, second_receipt=second, second_decisions=second_decisions,
        certificate=certificate, derived=derived, third_stdin=third_stdin, third_schema=third_schema)
    recovery.guard_file(Path(__file__), guards)
    check_prefix(context)
    native.protected_unchanged(guards)
    return context


def inventory(context, intent_directory):
    """No orphan, unknown, interrupted or newly failed execution is replayed."""
    intent_directory = Path(intent_directory).resolve()
    binding_path = intent_directory / "third_batch_binding.json"
    expected = {context["first_path"].parent, context["second_path"].parent}
    if (intent_directory / "failure.json").exists():
        raise ValueError("A new final-two failure stops without another invocation")
    binding = native.read_json(binding_path) if binding_path.exists() else None
    if binding is not None:
        third_path = Path(binding["receipt_path"]).resolve()
        recovery.guard_file(third_path, context["guards"], binding["receipt_sha256"])
        receipt, _, decisions = accepted_group(context, third_path, context["groups"][2], context["guards"])
        recovery.check_retry_identity(receipt, context["first_receipt"])
        if (receipt["execution_id"] in {context["first_receipt"]["execution_id"], context["second_receipt"]["execution_id"]}
                or binding.get("record_ids") != context["groups"][2] or binding.get("decisions") != decisions):
            raise ValueError("Bound final-two output changed or reused an earlier context")
        expected.add(third_path.parent)
    actual = set((context["owner"] / "executions").iterdir())
    if actual != expected or any(not directory.is_dir() or not (directory / "receipt.json").is_file() for directory in actual):
        raise ValueError("Unknown, interrupted, unbound or failed native attempt requires owner review")
    return binding


def validate_authorization(context, authorization, intent_directory):
    authorization, destination = Path(authorization).resolve(), Path(intent_directory).resolve()
    if destination.is_relative_to(context["owner"]) or destination == context["review"]:
        raise ValueError("Final-two intent must be external to the original source owner")
    recovery.guard_file(authorization, context["guards"])
    auth = native.read_json(authorization)
    required = dict(version=AUTH_VERSION, root_authorized=True, max_new_native_calls=1,
        controller_sha256=protocol.sha256(Path(__file__).read_bytes()),
        bounded_owner=str(context["owner"]), intent_directory=str(destination),
        certificate_sha256=context["anchors"]["certificate"], third_record_ids=context["groups"][2],
        third_stdin_sha256=protocol.sha256(context["third_stdin"]),
        third_schema_sha256=protocol.sha256(context["third_schema"]),
        same_model_request=True, new_failures_stop=True)
    if any(auth.get(key) != value or type(auth.get(key)) is not type(value) for key, value in required.items()):
        raise ValueError("One final-two invocation requires exact bound root authorization")
    return authorization, required


def run(review_run, certificate_directory, authorization, intent_directory, *, check_only=False,
        caller=None, anchors=None, admission_loader=None):
    context = validate_snapshot(review_run, certificate_directory, anchors=anchors, admission_loader=admission_loader)
    actual_calls = 0
    destination = Path(intent_directory).resolve()
    authorization, auth = validate_authorization(context, authorization, destination)
    original_guards = dict(context["guards"])
    binding = inventory(context, destination)
    proof = dict(version=VERSION, bounded_owner=str(context["owner"]), record_ids=context["ids"],
        groups=context["groups"], certificate_sha256=context["anchors"]["certificate"],
        third_stdin_sha256=auth["third_stdin_sha256"], third_schema_sha256=auth["third_schema_sha256"],
        third_full_input_and_schema_bytes=len(context["third_stdin"]) + len(context["third_schema"]),
        root_authorization=str(authorization), root_authorization_sha256=protocol.sha256(authorization.read_bytes()),
        original_second_receipt_accepted=False, new_native_calls=0, global_producer_progress_read=False, human_gold=False)
    if check_only:
        return proof
    intent_path = destination / "intent.json"
    if intent_path.exists():
        intent = native.read_json(intent_path)
        if intent.get("proof") != proof or intent.get("protected_sha256") != original_guards:
            raise ValueError("Existing final-two invocation intent changed")
        if binding is None:
            raise ValueError("Final-two invocation was already planned; never automatically replay it")
    elif destination.exists():
        raise ValueError("Final-two intent destination is not fresh")
    else:
        originals = {str(path): protocol.sha256(path.read_bytes()) for path in context["owner"].rglob("*") if path.is_file()}
        native.write_new(intent_path, dict(proof=proof, protected_sha256=original_guards, original_owner_sha256=originals))
    intent = native.read_json(intent_path)
    if binding is None:
        try:
            native.protected_unchanged(intent["protected_sha256"])
            native.protected_unchanged(intent["original_owner_sha256"])
            check_prefix(context)
            payload, schema, _, _ = physical(context, context["groups"][2])
            actual_calls += 1
            raw, receipt = (caller or context["base"].runtime.call_structured_json)(
                context["base"].adjudicator.SYSTEM, payload, schema, context["owner"] / "executions", role="adjudicator")
            receipt_path = Path(receipt["artifact_directory"]).resolve() / "receipt.json"
            accepted, actual_raw, decisions = accepted_group(context, receipt_path, context["groups"][2], context["guards"])
            if receipt != accepted or protocol.canonical_bytes(raw) != protocol.canonical_bytes(actual_raw):
                raise ValueError("Final-two caller output differs from its actual retained raw execution")
            recovery.check_retry_identity(receipt, context["first_receipt"])
            if receipt["execution_id"] in {context["first_receipt"]["execution_id"], context["second_receipt"]["execution_id"]}:
                raise ValueError("Final-two source invocation reused an original review context")
            native.protected_unchanged(intent["protected_sha256"])
            native.protected_unchanged(intent["original_owner_sha256"])
            check_prefix(context)
            binding = dict(record_ids=context["groups"][2], receipt_path=str(receipt_path),
                receipt_sha256=protocol.sha256(receipt_path.read_bytes()), decisions=decisions, new_native_calls=1)
            native.write_new(destination / "third_batch_binding.json", binding)
            inventory(context, destination)
        except Exception as error:
            native.write_new(destination / "failure.json", dict(version=VERSION, error=str(error),
                new_native_calls_at_most=1, automatic_retry=False, all_attempts_preserved=True, human_gold=False))
            raise
    completion_path = destination / "completion.json"
    decisions = context["first_decisions"] + context["second_decisions"] + binding["decisions"]
    if completion_path.exists():
        result = native.read_json(completion_path)
        for path, digest in result["completion_artifact_sha256"].items():
            recovery.guard_file(path, context["guards"], digest)
        decisions_path = Path(result["adjudications_path"]).resolve()
        if (decisions_path.parent != context["owner"]
                or decisions_path.read_bytes() != b"".join(protocol.canonical_bytes(row) + b"\n" for row in decisions)
                or result.get("records") != 20 or result.get("complete_chunk") is not True
                or result.get("original_second_receipt_accepted") is not False
                or result.get("adjudications_sha256") != protocol.sha256(decisions_path.read_bytes())):
            raise ValueError("Completed whole source chunk differs from its exact original raw decisions")
        sidecar = native.read_json(Path(result["certificate_admission_path"]))
        if (sidecar.get("certificate_sha256") != context["anchors"]["certificate"]
                or sidecar.get("derived_records_sha256") != context["anchors"]["derived"]
                or sidecar.get("certified_record_ids") != context["groups"][1]
                or sidecar.get("operational_receipt_accepted") is not False):
            raise ValueError("Completed source certificate admission changed")
        native.protected_unchanged(context["guards"])
        native.protected_unchanged(intent["original_owner_sha256"])
        check_prefix(context)
        return dict(result, new_native_calls=0)
    if [row["record_id"] for row in decisions] != context["ids"]:
        raise ValueError("Completed source chunk lost fixed complete coverage")
    token = uuid4().hex[:10]
    decisions_path = context["owner"] / ("adjudications-revalidated-" + token + ".jsonl")
    native.write_new(decisions_path, b"".join(protocol.canonical_bytes(row) + b"\n" for row in decisions))
    sidecar_path = context["owner"] / ("revalidated-source-admission-" + token + ".json")
    native.write_new(sidecar_path, dict(version=VERSION, adjudications_path=str(decisions_path),
        adjudications_sha256=protocol.sha256(decisions_path.read_bytes()), certificate_directory=str(context["certificate_directory"]),
        certificate_sha256=context["anchors"]["certificate"], derived_records_sha256=context["anchors"]["derived"],
        certified_record_ids=context["groups"][1], original_native_receipt=str(context["second_path"]),
        operational_receipt_accepted=False, actual_argv_recorded=False, actual_command_bytes_unavailable=True,
        source_decisions_pending_certificate_aware_assembly=True, new_certificate_native_calls=0, human_gold=False))
    result = dict(version=VERSION, records=20, accepted=sum(row["status"] == "accepted" for row in decisions),
        unresolved=sum(row["status"] == "unresolved" for row in decisions), complete_chunk=True,
        adjudications_path=str(decisions_path), adjudications_sha256=protocol.sha256(decisions_path.read_bytes()),
        certificate_admission_path=str(sidecar_path), new_native_calls=actual_calls, retained_native_calls=3,
        lifetime_final_two_native_calls=1,
        operationally_accepted_native_calls=2, certified_original_false_native_calls=1,
        original_second_receipt_accepted=False, original_failure_usage_status="unknown_not_zero",
        original_unknown_failed_native_calls=1, original_unknown_failed_usage=None,
        retained_usage_by_execution_id={value["execution_id"]: value.get("usage") for value in
            (context["first_receipt"], context["second_receipt"], native.read_json(binding["receipt_path"]))},
        human_gold=False,
        completion_artifact_sha256={str(path): protocol.sha256(path.read_bytes()) for path in
            (decisions_path, sidecar_path, destination / "third_batch_binding.json")})
    native.protected_unchanged(intent["protected_sha256"])
    native.protected_unchanged(intent["original_owner_sha256"])
    check_prefix(context)
    native.write_new(completion_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("review-run", "certificate-directory", "authorization", "intent-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(protocol.canonical_bytes(run(args.review_run, args.certificate_directory, args.authorization,
        args.intent_directory, check_only=args.check_only)).decode())
