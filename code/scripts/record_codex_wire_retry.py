"""Record one owner-authorized exact-input retry without rewriting failed history."""
from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

import run_codex_annotation as native
import run_codex_cohort_wire as wire
from oil_evidence import machine_evaluation as protocol


INPUT_ARTIFACTS = (
    "stdin.txt", "schema.json", "codec.json", "wire_payload.json",
    "canonical_semantic_packets.json",
)
AUTHORIZATION = (
    "Root authorized one fresh-context, whole-batch exact-input retry of rejected "
    "B batch-0019. Keep the failed raw response, receipt, progress and usage; pass "
    "no validation feedback or other labels. Stop if this one retry fails."
)


def artifact_hashes(attempt):
    return {name: protocol.sha256((Path(attempt) / name).read_bytes()) for name in INPUT_ARTIFACTS}


def verify_raw_artifacts(path, receipt):
    for name, expected in receipt["artifacts_sha256"].items():
        if protocol.sha256(native.contained_file(path.parent, name).read_bytes()) != expected:
            raise ValueError("Original/retry raw artifact changed")


def requested_model_from_command(path):
    command = native.read_json(path)
    matches = [command[index + 1] for index, value in enumerate(command[:-1]) if value in ("--model", "-m")]
    matches += [value.split("=", 1)[1] for value in command if value.startswith("--model=")]
    if len(matches) > 1:
        raise ValueError("Ambiguous actual requested model")
    return matches[0] if matches else None


def accepted_receipts(output, events):
    values = {}
    for event in events:
        if event.get("type") == "completed" and event.get("accepted") is True:
            key = event["stream"] + ":" + event["batch"]
            if key in values:
                raise ValueError("An accepted batch was repeated")
            path = native.contained_file(output, event["receipt"])
            if protocol.sha256(path.read_bytes()) != event["receipt_sha256"]:
                raise ValueError("Accepted receipt differs from durable history")
            values[key] = {"receipt": str(path), "receipt_sha256": event["receipt_sha256"]}
    return values


def prepare(output, failure):
    output, failure = Path(output).resolve(), Path(failure).resolve()
    if not failure.is_relative_to(output) or failure.name != "receipt.json":
        raise ValueError("Exact failed receipt must belong to this annotation run")
    receipt = native.read_json(failure)
    verify_raw_artifacts(failure, receipt)
    if receipt.get("accepted") is not False or receipt.get("stream") != "b" or receipt.get("batch") != "batch-0019":
        raise ValueError("This authorization applies only to rejected whole B0019")
    if receipt.get("process_exit_code") != 0 or "Every answer point needs citations with exact eligible quotes" not in " ".join(receipt.get("errors", [])):
        raise ValueError("Unexpected failure; this is not a quota or general retry authority")
    manifest = native.read_json(output / "manifest.json")
    native.protected_unchanged(native.read_json(output / "protected_inputs.json"))
    for name, expected in manifest["immutable_sha256"].items():
        if protocol.sha256(native.contained_file(output, name).read_bytes()) != expected:
            raise ValueError("Sealed annotation run changed")
    progress = (output / "progress.jsonl").read_bytes()
    events = protocol.parse_jsonl(progress, "pre-retry progress")
    accepted = accepted_receipts(output, events)
    if "b:batch-0019" in accepted:
        raise ValueError("Accepted outputs must never be rerun")
    original_event = [e for e in events if e.get("receipt") == failure.relative_to(output).as_posix()]
    if len(original_event) != 1 or original_event[0].get("accepted") is not False or original_event[0]["receipt_sha256"] != protocol.sha256(failure.read_bytes()):
        raise ValueError("Original failure is not uniquely bound to durable progress")
    counterpart = Path(accepted["a:batch-0019"]["receipt"])
    counterpart_receipt = native.read_json(counterpart)
    verify_raw_artifacts(counterpart, counterpart_receipt)
    original_hashes = artifact_hashes(failure.parent)
    if artifact_hashes(counterpart.parent) != original_hashes or counterpart_receipt["record_ids"] != receipt["record_ids"]:
        raise ValueError("Original A/B physical inputs differ")
    requested_model = manifest["specification"]["requested_model"]
    if requested_model_from_command(failure.parent / "command.json") != requested_model or requested_model_from_command(counterpart.parent / "command.json") != requested_model:
        raise ValueError("Actual original A/B model request differs from frozen configuration")
    groups = native.read_json(output / "groups.json")
    selected_group = [g for g in groups if g["batch"] == receipt["batch"]]
    if len(selected_group) != 1 or selected_group[0]["record_ids"] != receipt["record_ids"]:
        raise ValueError("Rejected group differs from frozen schedule")
    endpoint = len(native.read_json(output / "history.json")["record_ids"])
    planned = []
    for group in groups:
        endpoint += len(group["record_ids"])
        for stream in native.STREAMS:
            if stream + ":" + group["batch"] not in accepted:
                planned.append(stream + ":" + group["batch"])
        if group["batch"] == receipt["batch"]:
            break
    if planned != ["b:batch-0019"]:
        raise ValueError("The bounded resume would invoke more than the one authorized retry")
    # Keep receipt paths below the Windows path-length boundary of this deep run.
    directory = output / "retry" / uuid4().hex[:10]
    directory.mkdir(parents=True, exist_ok=False)
    native.write_new(directory / "original_progress_prefix.jsonl", progress)
    usage = (output / "execution_usage.jsonl").read_bytes()
    native.write_new(directory / "original_usage_prefix.jsonl", usage)
    authorization = {"authorizing_agent": "/root", "root_authorized": True,
                     "reason": "controlled source-validation exact-input retry", "instruction": AUTHORIZATION,
                     "retries_authorized": 1, "no_label_feedback": True, "native_calls_before_request": 0}
    native.write_new(directory / "authorization.json", authorization)
    request = {
        "created_utc": native.utc_now(), "annotation_run": str(output), "stream": "b", "batch": "batch-0019",
        "record_ids": receipt["record_ids"], "max_records_endpoint": endpoint,
        "original_failure_receipt": str(failure), "original_failure_receipt_sha256": protocol.sha256(failure.read_bytes()),
        "original_failure_execution_id": receipt["execution_id"],
        "original_progress_prefix": str(directory / "original_progress_prefix.jsonl"),
        "original_progress_prefix_sha256": protocol.sha256(progress), "original_progress_prefix_bytes": len(progress),
        "original_usage_prefix_sha256": protocol.sha256(usage), "original_usage_prefix_bytes": len(usage),
        "counterpart_a_receipt": str(counterpart), "counterpart_a_receipt_sha256": protocol.sha256(counterpart.read_bytes()),
        "physical_input_artifacts_sha256": original_hashes,
        "requested_model": requested_model,
        "manifest_sha256": protocol.sha256((output / "manifest.json").read_bytes()),
        "accepted_receipts_before": accepted, "retries_authorized": 1, "no_label_feedback": True,
        "original_failure_retained": True, "whole_batch_atomic": True,
        "authorization_sha256": protocol.sha256((directory / "authorization.json").read_bytes()),
    }
    native.write_new(directory / "request.json", request)
    return directory, request


def seal(directory):
    directory = Path(directory).resolve()
    request = native.read_json(directory / "request.json")
    if protocol.sha256((directory / "authorization.json").read_bytes()) != request["authorization_sha256"]:
        raise ValueError("Owner retry authorization changed")
    output = Path(request["annotation_run"])
    if protocol.sha256((output / "manifest.json").read_bytes()) != request["manifest_sha256"]:
        raise ValueError("Frozen run changed during recovery")
    prefix = (directory / "original_progress_prefix.jsonl").read_bytes()
    progress = (output / "progress.jsonl").read_bytes()
    if protocol.sha256(prefix) != request["original_progress_prefix_sha256"] or not progress.startswith(prefix):
        raise ValueError("Original failed progress was changed or deleted")
    appended = protocol.parse_jsonl(progress[len(prefix):], "single retry progress")
    if len(appended) != 1 or appended[0].get("type") != "completed" or appended[0].get("stream") != "b" or appended[0].get("batch") != "batch-0019":
        raise ValueError("Only one authorized retry event may be appended before resolution")
    event = appended[0]
    retry = native.contained_file(output, event["receipt"])
    receipt = native.read_json(retry)
    verify_raw_artifacts(retry, receipt)
    if protocol.sha256(retry.read_bytes()) != event["receipt_sha256"] or receipt["record_ids"] != request["record_ids"]:
        raise ValueError("Retry receipt differs from its original group")
    failure = Path(request["original_failure_receipt"])
    if protocol.sha256(failure.read_bytes()) != request["original_failure_receipt_sha256"]:
        raise ValueError("Original failed receipt was rewritten")
    verify_raw_artifacts(failure, native.read_json(failure))
    if artifact_hashes(failure.parent) != request["physical_input_artifacts_sha256"] or artifact_hashes(retry.parent) != request["physical_input_artifacts_sha256"]:
        raise ValueError("Retry physical input/schema/codec differs from rejected execution")
    if requested_model_from_command(retry.parent / "command.json") != request["requested_model"]:
        raise ValueError("Retry actual model request differs")
    for value in request["accepted_receipts_before"].values():
        if protocol.sha256(Path(value["receipt"]).read_bytes()) != value["receipt_sha256"]:
            raise ValueError("An existing accepted receipt was changed")
    usage_prefix = (directory / "original_usage_prefix.jsonl").read_bytes()
    usage = (output / "execution_usage.jsonl").read_bytes()
    if protocol.sha256(usage_prefix) != request["original_usage_prefix_sha256"] or not usage.startswith(usage_prefix):
        raise ValueError("Original execution usage changed")
    new_usage = protocol.parse_jsonl(usage[len(usage_prefix):], "single retry usage")
    if len(new_usage) != 1 or new_usage[0]["execution_id"] != receipt["execution_id"] or receipt["execution_id"] == request["original_failure_execution_id"]:
        raise ValueError("Retry must be one distinct, counted execution")
    if receipt.get("accepted") is True:
        packets = native.read_json(output / "packets.json")
        wire.verify_attempt(retry.parent, receipt, packets)
    resolution = {
        "created_utc": native.utc_now(), "annotation_run": str(output), "stream": "b", "batch": "batch-0019",
        "record_ids": request["record_ids"], "request_path": str(directory / "request.json"),
        "request_sha256": protocol.sha256((directory / "request.json").read_bytes()),
        "authorization_path": str(directory / "authorization.json"),
        "authorization_sha256": request["authorization_sha256"],
        "original_failure_receipt": request["original_failure_receipt"],
        "original_failure_receipt_sha256": request["original_failure_receipt_sha256"],
        "original_failure_execution_id": request["original_failure_execution_id"],
        "original_progress_prefix": request["original_progress_prefix"],
        "original_progress_prefix_sha256": request["original_progress_prefix_sha256"],
        "original_progress_prefix_bytes": request["original_progress_prefix_bytes"],
        "counterpart_a_receipt": request["counterpart_a_receipt"],
        "counterpart_a_receipt_sha256": request["counterpart_a_receipt_sha256"],
        "retry_receipt": str(retry), "retry_receipt_sha256": event["receipt_sha256"],
        "retry_execution_id": receipt["execution_id"], "retry_accepted": receipt.get("accepted") is True,
        "physical_equal_artifact_hashes": request["physical_input_artifacts_sha256"],
        "same_requested_model": True, "requested_model": request["requested_model"],
        "actual_command_model_values": {"original_failure": requested_model_from_command(failure.parent / "command.json"),
            "unchanged_a": requested_model_from_command(Path(request["counterpart_a_receipt"]).parent / "command.json"),
            "retry": requested_model_from_command(retry.parent / "command.json")},
        "unchanged_A_and_other_accepted_receipts": True, "root_authorized": True,
        "original_failure_retained": True, "whole_batch_atomic": True,
        "retries_authorized": 1, "native_calls_actual": 1, "no_label_feedback": True,
        "failed_attempt_excluded_from_accepted_counts": True,
        "status": "single_exact_input_retry_accepted" if receipt.get("accepted") is True else "single_exact_input_retry_failed_stop",
    }
    native.write_new(directory / "resolution.json", resolution)
    return resolution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("prepare")
    before.add_argument("--annotation-run", type=Path, required=True)
    before.add_argument("--failure-receipt", type=Path, required=True)
    after = commands.add_parser("seal")
    after.add_argument("--recovery-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        directory, value = prepare(args.annotation_run, args.failure_receipt)
        print(directory)
        print(value["max_records_endpoint"])
    else:
        print(seal(args.recovery_directory))
