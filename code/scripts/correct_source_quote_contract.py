"""One allowlisted source review with unchanged system/schema and neutral data.

The prior unresolved record and whole chunk remain intact. Only a generic
quote-validation contract is added to the new physical input. A valid accepted
correction creates a new original-owned twenty-record file; nineteen raw JSONL
lines are copied byte-for-byte. No prior judge outcome is shown to the model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import compact_machine_adjudication as compact
import pipeline_transport_recovery_adjudication as recovery
from oil_evidence import codex_runtime as runtime

native, protocol = recovery.native, recovery.protocol
VERSION = "one-source-body-quote-contract-correction-v1"
AUTH_VERSION = "one-source-quote-contract-authorization-v1"
RECORD_ID = "test-f4a76745ee219167836f"
CONTRACT = dict(quoteable_source_field="evidence[].text",
    non_quoteable_metadata_fields=["title", "source", "url"],
    quoted_text_must_be_exact_nonempty_substring=True,
    forecasts_are_not_realized_observations=True, independent_labels_are_not_gold=True)
PINS = dict(result="c050982378c32c2df50bb9e66ddc9aae75c55d75d1f111c5d66a857153e1ebe6",
    decisions="af76d33293b9c5fbc1359dcf634d5449ca09107dfada035e0c523208c714b64d",
    packets="193437087aa223e96815262e4b63c2afcc42817da646517229a611ed6003e195",
    a="87703b1359b39e966b7607fbade7e05af604daf80dc5d9b0337f327e4e4d70e9",
    b="d93a43679cda092f901fcf68cf63210ce1108a6f8bc01bf2633c9ab134743764",
    export="346f8cc10c935ae7a1e7e22937735967ef7df547162b79c270512fea2d892406",
    old_receipt="8a9716771dc92a269b2e96e77a7185f3996017a90ec02679ee4aee97b572f18f")


def corrected_input(packets, left, right, identifier):
    original, schema = compact.prepare_batch(packets, left, right, [identifier])
    payload = deepcopy(original)
    if "mechanical_quote_validation_contract" in payload[identifier]:
        raise ValueError("Original source packet already contains an ambiguous contract field")
    payload[identifier]["mechanical_quote_validation_contract"] = deepcopy(CONTRACT)
    stdin = (runtime.ISOLATION_PROMPT + "\n" + compact.SYSTEM + "\nUNTRUSTED_PAYLOAD_JSON\n").encode()
    stdin += protocol.canonical_bytes(payload)
    schema_bytes = protocol.canonical_bytes(schema)
    if len(stdin) + len(schema_bytes) > 262144:
        raise ValueError("The complete one-case corrective input exceeds the fixed hard byte bound")
    return original, payload, schema, stdin, schema_bytes


def validate_snapshot(review_run, *, pins=None, identifier=RECORD_ID):
    pins = PINS if pins is None else pins
    review = Path(review_run).resolve()
    chunk = review / "chunks/chunk-0031"
    result_path = review / "retained/chunk-0031.json"
    guards = {}
    recovery.guard_file(result_path, guards, pins["result"])
    result = native.read_json(result_path)
    owner = review / "adjudications/chunk-0031/bounded-run"
    decisions_path = Path(result["adjudications_path"]).resolve()
    if decisions_path.parent != owner or result.get("records") != 20 or result.get("accepted") != 19 or result.get("unresolved") != 1:
        raise ValueError("Only the registered original-owned nineteen/one chunk may be corrected")
    recovery.guard_file(decisions_path, guards, pins["decisions"])
    files = dict(packets=chunk / "packets.json", a=chunk / "independent_a.jsonl",
        b=chunk / "independent_b.jsonl", export=chunk / "export_receipt.json")
    for key, path in files.items():
        recovery.guard_file(path, guards, pins[key])
    export = native.read_json(files["export"])
    recovery.merge_guards(guards, export["source_bindings"])
    for name, digest in export["artifacts_sha256"].items():
        recovery.guard_file(native.contained_file(chunk, name), guards, digest)
    packets = native.read_json(files["packets"])
    left, right = (compact.original.read_stream(files[side]) for side in ("a", "b"))
    lines = decisions_path.read_bytes().splitlines(keepends=True)
    rows = [protocol.parse_jsonl(line, "original source decision line")[0] for line in lines]
    ids = list(packets)
    target = next((row for row in rows if row["record_id"] == identifier), None)
    if ([row["record_id"] for row in rows] != ids or len(ids) != 20 or len(set(ids)) != 20
            or set(left) != set(ids) or set(right) != set(ids) or export["record_ids"] != ids
            or target is None or target.get("status") != "unresolved"
            or target.get("label") is not None or target.get("raw_model_status") != "accepted"
            or target.get("mechanical_validation_errors") != ["Quote is not an exact frozen evidence span"]
            or any(row["status"] != "accepted" for row in rows if row is not target)):
        raise ValueError("Only the exact one-case body-span mechanical failure may be corrected")
    contexts = set()
    for row in rows:
        metadata = row["adjudication"]
        receipt_path = Path(metadata["execution_receipt_path"]).resolve()
        raw_path = Path(metadata["raw_response_path"]).resolve()
        if not receipt_path.is_relative_to(owner / "executions") or raw_path.parent != receipt_path.parent:
            raise ValueError("Original source review raw ownership changed")
        receipt = recovery.receipt_artifacts(receipt_path, guards)
        if receipt.get("accepted") is not True or metadata.get("protocol_sha256") != protocol.sha256(compact.SYSTEM.encode()):
            raise ValueError("The original operationally accepted compact protocol changed")
        raw = native.read_json(raw_path)
        key = row["record_id"]
        rebuilt = compact.retain_compact_decisions({key: raw[key]}, packets, left, right, [key], receipt)[0]
        if rebuilt != row:
            raise ValueError("Original normalized source decision differs from its exact full native raw output")
        contexts.add(receipt["execution_id"])
    contexts.update(stream[key]["identity"]["execution_id"] for stream in (left, right) for key in ids)
    old_receipt_path = Path(target["adjudication"]["execution_receipt_path"])
    recovery.guard_file(old_receipt_path, guards, pins["old_receipt"])
    original, payload, schema, stdin, schema_bytes = corrected_input(packets, left, right, identifier)
    for path in (Path(__file__), Path(compact.__file__), Path(runtime.__file__), Path(protocol.__file__), Path(native.__file__)):
        recovery.guard_file(path, guards)
    native.protected_unchanged(guards)
    return dict(review=review, chunk=chunk, owner=owner, original_result_path=result_path, original_result=result,
        original_decisions_path=decisions_path, lines=lines, rows=rows, packets=packets, left=left, right=right,
        identifier=identifier, target=target, old_receipt=native.read_json(old_receipt_path),
        old_contexts=contexts, original_payload=original, payload=payload, schema=schema,
        stdin=stdin, schema_bytes=schema_bytes, guards=guards)


def proof(context):
    return dict(version=VERSION, record_id=context["identifier"], owner=str(context["owner"]),
        controller_sha256=protocol.sha256(Path(__file__).read_bytes()),
        original_result_sha256=protocol.sha256(context["original_result_path"].read_bytes()),
        original_decisions_sha256=protocol.sha256(context["original_decisions_path"].read_bytes()),
        contract_sha256=protocol.sha256(protocol.canonical_bytes(CONTRACT)),
        original_system_sha256=protocol.sha256(compact.SYSTEM.encode()),
        full_stdin_sha256=protocol.sha256(context["stdin"]), schema_sha256=protocol.sha256(context["schema_bytes"]),
        full_input_and_schema_bytes=len(context["stdin"]) + len(context["schema_bytes"]),
        old_outcome_or_reason_supplied_to_model=False, original_fields_and_output_schema_unchanged=True,
        max_new_native_calls=1, human_gold=False)


def run(review_run, intent_directory, authorization=None, *, check_only=False, caller=None, pins=None, identifier=RECORD_ID):
    context = validate_snapshot(review_run, pins=pins, identifier=identifier)
    plan = proof(context)
    if check_only:
        return plan
    destination = Path(intent_directory).resolve()
    if destination.exists() or destination.is_relative_to(context["owner"]):
        raise ValueError("One-case corrective intent must be fresh and external; no automatic retry")
    if authorization is None:
        raise ValueError("The one native corrective context requires bound root authorization")
    authorization = Path(authorization).resolve()
    auth = native.read_json(authorization)
    required = dict(plan, version=AUTH_VERSION, root_authorized=True, intent_directory=str(destination),
        same_requested_model=True, new_failures_stop=True)
    if any(auth.get(key) != value or type(auth.get(key)) is not type(value) for key, value in required.items()):
        raise ValueError("Corrective authorization differs from the exact allowlisted case and physical input")
    recovery.guard_file(authorization, context["guards"])
    native.write_new(destination / "intent.json", dict(plan=plan, authorization=str(authorization),
        authorization_sha256=protocol.sha256(authorization.read_bytes()), original_sha256=context["guards"]))
    try:
        native.protected_unchanged(context["guards"])
        raw, receipt = (caller or runtime.call_structured_json)(compact.SYSTEM, context["payload"], context["schema"],
            context["owner"] / "executions", role="adjudicator")
        directory = Path(receipt["artifact_directory"]).resolve()
        if not directory.is_relative_to(context["owner"] / "executions"):
            raise ValueError("Corrective native execution escaped the original chunk owner")
        actual = recovery.receipt_artifacts(directory / "receipt.json", context["guards"])
        if (actual != receipt or receipt.get("accepted") is not True or receipt.get("process_exit_code") != 0
                or receipt.get("codex_cli_invocations") != 1 or receipt.get("external_paid_api_calls") != 0):
            raise ValueError("The single corrective native attempt was not operationally accepted")
        recovery.check_retry_identity(receipt, context["old_receipt"])
        if receipt["execution_id"] in context["old_contexts"]:
            raise ValueError("Corrective source context must differ from both annotators and every original judge context")
        if (directory / "stdin.txt").read_bytes() != context["stdin"] or (directory / "schema.json").read_bytes() != context["schema_bytes"]:
            raise ValueError("Corrective full physical input/schema differs from its explicit contract plan")
        events = native.parse_execution_events((directory / "events.jsonl").read_bytes())
        if (events["errors"] or events["thread_id"] != receipt["execution_id"] or events["usage"] != receipt.get("usage")
                or events["observed_model"] != receipt.get("observed_model")
                or sum(row.get("type") == "turn.completed" for row in events["events"]) != 1):
            raise ValueError("Corrective execution must be one clean fresh measured native turn")
        if protocol.canonical_bytes(raw) != protocol.canonical_bytes(native.read_json(directory / "raw_final.json")):
            raise ValueError("Caller decision differs from its original retained full raw output")
        corrected = compact.retain_compact_decisions(raw, context["packets"], context["left"], context["right"],
            [identifier], receipt)[0]
        native.write_new(destination / "new_record.json", corrected)
        if corrected["status"] != "accepted":
            raise ValueError("New source review remains unresolved; preserve it without replacing any original row")
        native.protected_unchanged(context["guards"])
        output_lines = [protocol.canonical_bytes(corrected) + b"\n" if row["record_id"] == identifier else line
                        for row, line in zip(context["rows"], context["lines"])]
        token = uuid4().hex[:10]
        decisions_path = context["owner"] / ("adjudications-quote-contract-" + token + ".jsonl")
        native.write_new(decisions_path, b"".join(output_lines))
        sidecar_path = context["owner"] / ("quote-contract-replacement-" + token + ".json")
        sidecar = dict(plan, original_result_path=str(context["original_result_path"]),
            original_decisions_path=str(context["original_decisions_path"]),
            replacement_adjudications_path=str(decisions_path), replacement_sha256=protocol.sha256(decisions_path.read_bytes()),
            additional_contract=CONTRACT, original_system_unchanged=True, nineteen_other_lines_byte_identical=True,
            new_receipt_path=str(directory / "receipt.json"), new_receipt_sha256=protocol.sha256((directory / "receipt.json").read_bytes()),
            execution_id=receipt["execution_id"], usage=receipt.get("usage"), usage_counted_once=True,
            original_source_summary_not_rewritten=True, new_native_calls=1, subscription_dollar_cost=None)
        native.write_new(sidecar_path, sidecar)
        native.protected_unchanged(context["guards"])
        result = dict(version=VERSION, records=20, accepted=20, unresolved=0, record_id=identifier,
            adjudications_path=str(decisions_path), adjudications_sha256=sidecar["replacement_sha256"],
            replacement_sidecar_path=str(sidecar_path), new_native_calls=1, execution_id=receipt["execution_id"],
            usage=receipt.get("usage"), human_gold=False, original_summary_unchanged=True)
        native.write_new(destination / "result.json", result)
        return result
    except Exception as error:
        native.write_new(destination / "failure.json", dict(version=VERSION, error=str(error),
            max_new_native_calls=1, automatic_retry=False, originals_and_all_new_attempts_retained=True, human_gold=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-run", type=Path, required=True)
    parser.add_argument("--intent-directory", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(protocol.canonical_bytes(run(args.review_run, args.intent_directory, args.authorization,
        check_only=args.check_only)).decode())
