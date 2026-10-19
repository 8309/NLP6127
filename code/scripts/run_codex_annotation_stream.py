"""Run one independent annotation stream in its own process and output directory.

The existing individual-record protocol and input bytes are unchanged. This
wrapper selects only A or B, adds its source hash to protected inputs, and saves
a caller receipt before generation. It never edits the shared runner or helper.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import run_codex_annotation as native

V1_SNAPSHOT = "520f8973a3b0788923ed4a22a3324c56351abf6b"
V1_CORRECTED_ERROR = "Insufficient label must not invent supported answer points or quotes"


def migration_inputs(source, stream, rows, protected, snapshot=V1_SNAPSHOT):
    """Read bound v1 bytes; changed runtime sources must match the saved Git blob."""
    source = Path(source).resolve()
    manifest = native.read_json(source / "manifest.json")
    bound = {}
    for name, expected in manifest["immutable_sha256"].items():
        path = native.contained_file(source, name)
        if native.protocol.sha256(path.read_bytes()) != expected:
            raise ValueError("Historical immutable annotation input changed")
        bound[str(path)] = expected
    expected_prompt = (native.GUARD + "\n" + native.protocol.ANNOTATION_PROMPT).encode("utf-8")
    if (source / "protocol.txt").read_bytes() != expected_prompt:
        raise ValueError("Lossless migration requires identical annotation protocol bytes")
    saved_sources = {}
    for name, expected in native.read_json(source / "protected_inputs.json").items():
        path = Path(name)
        if path.is_file() and native.protocol.sha256(path.read_bytes()) == expected:
            bound[str(path)] = expected
            continue
        if not path.resolve().is_relative_to(native.ROOT.resolve()):
            raise ValueError("Historical protected data changed outside versioned runtime")
        relative = path.resolve().relative_to(native.ROOT.resolve()).as_posix()
        raw = subprocess.run(["git", "show", f"{snapshot}:{relative}"], cwd=native.ROOT,
                             capture_output=True, check=True).stdout
        if native.protocol.sha256(raw) != expected:
            raise ValueError("Historical runtime differs from its preserved Git snapshot")
        saved_sources[relative] = raw
    old_packets = native.read_json(source / "packets.json")
    current_packets = {row["question_id"]: native.packet_and_prompt(row)[0] for row in rows}
    if old_packets != current_packets:
        raise ValueError("Question, evidence, protocol or packet digest changed during migration")
    progress_path = source / "progress.jsonl"
    progress = native.protocol.parse_jsonl(progress_path.read_bytes(), "historical progress")
    receipts = {event["receipt"]: event.get("receipt_sha256") for event in progress
                if event.get("type") == "completed" and event.get("stream") == stream}
    attempts = []
    seen = set()
    for relative, expected in sorted(receipts.items()):
        receipt_path = native.contained_file(source, relative)
        if native.protocol.sha256(receipt_path.read_bytes()) != expected:
            raise ValueError("Historical receipt differs from retained progress")
        receipt = native.read_json(receipt_path)
        old_errors = receipt.get("errors", [])
        if not receipt.get("accepted") and old_errors != [V1_CORRECTED_ERROR]:
            continue
        attempt = receipt_path.parent
        for name, sha in receipt["artifacts_sha256"].items():
            path = native.contained_file(attempt, name)
            if native.protocol.sha256(path.read_bytes()) != sha:
                raise ValueError("Historical raw execution artifact changed")
            bound[str(path)] = sha
        envelope = native.read_json(attempt / "envelope.json")
        record_id = envelope["record_id"]
        if record_id in seen or record_id not in current_packets:
            raise ValueError("Historical duplicate or unexpected record")
        if native.protocol.validate_independent_output(envelope, current_packets[record_id]):
            raise ValueError("Unchanged raw label fails the new declared guard")
        if (envelope["raw_output"].encode("utf-8") != (attempt / "raw_final.json").read_bytes()
                or envelope["execution_input_sha256"] != native.protocol.sha256((attempt / "stdin.txt").read_bytes())):
            raise ValueError("Historical original response or actual input bytes changed")
        events = native.parse_execution_events((attempt / "events.jsonl").read_bytes())
        if events["errors"] or receipt["process_exit_code"] != 0:
            raise ValueError("A validation-only migration cannot rescue an operational failure")
        seen.add(record_id)
        bound[str(receipt_path)] = expected
        attempts.append(dict(source=attempt, receipt=receipt, envelope=envelope,
                             record_id=record_id, source_receipt_sha256=expected))
    bound[str(source / "manifest.json")] = native.protocol.sha256((source / "manifest.json").read_bytes())
    bound[str(progress_path)] = native.protocol.sha256(progress_path.read_bytes())
    return attempts, {**protected, **bound}, saved_sources


def migrate_outputs(output, stream, attempts, saved_sources, source):
    """Create new v2 receipts; never replace a v1 file, verdict or raw label."""
    for number, (name, raw) in enumerate(sorted(saved_sources.items()), 1):
        native.write_new(output / "v1" / f"code-{number:02d}.py", raw)
    records = []
    for old in attempts:
        record_id = old["record_id"]
        destination = output / stream / record_id / ("attempt-migrated-" + uuid4().hex[:8])
        destination.mkdir(parents=True, exist_ok=False)
        for name in ("stdin.txt", "schema.json", "events.jsonl", "stderr.txt", "raw_final.json"):
            native.write_new(destination / name, (old["source"] / name).read_bytes())
        native.write_new(destination / "v1_envelope.json", (old["source"] / "envelope.json").read_bytes())
        native.write_new(destination / "v1_receipt.json", (old["source"] / "receipt.json").read_bytes())
        envelope = deepcopy(old["envelope"])
        envelope["validation_errors"] = []
        envelope["validation_schema_version"] = native.protocol.SCHEMA_VERSION
        envelope["lossless_migration"] = dict(source_receipt_sha256=old["source_receipt_sha256"],
            original_accepted=old["receipt"]["accepted"], original_errors=old["receipt"]["errors"],
            original_guard_git_snapshot=V1_SNAPSHOT, labels_changed=False, new_model_calls=0)
        native.write_new(destination / "envelope.json", envelope)
        receipt = deepcopy(old["receipt"])
        receipt.update(completed_utc=native.utc_now(), accepted=True, errors=[],
            operation="lossless_guard_revalidation_not_new_annotation", validation_schema_version=native.protocol.SCHEMA_VERSION,
            source_receipt_sha256=old["source_receipt_sha256"], original_guard_verdict=old["receipt"]["errors"],
            original_guard_accepted=old["receipt"]["accepted"], new_model_calls=0,
            artifacts_sha256={path.name: native.protocol.sha256(path.read_bytes()) for path in destination.iterdir() if path.is_file()})
        native.write_new(destination / "receipt.json", receipt)
        native.append_progress(output / "progress.jsonl", dict(type="completed", stream=stream,
            record_id=record_id, accepted=True, errors=[], operation="lossless_v1_to_v2_revalidation",
            receipt=destination.relative_to(output).as_posix() + "/receipt.json",
            receipt_sha256=native.protocol.sha256((destination / "receipt.json").read_bytes())))
        records.append(dict(record_id=record_id, original_accepted=old["receipt"]["accepted"],
            original_errors=old["receipt"]["errors"], source_receipt_sha256=old["source_receipt_sha256"],
            raw_output_sha256=envelope["raw_output_sha256"], actual_stdin_sha256=envelope["execution_input_sha256"],
            original_execution_id=envelope["identity"]["execution_id"]))
    native.write_new(output / "migration_v2.json", dict(validation_schema_version=native.protocol.SCHEMA_VERSION,
        source_manifest_sha256=native.protocol.sha256((Path(source) / "manifest.json").read_bytes()),
        preserved_v1_git_snapshot=V1_SNAPSHOT, stream=stream, migrated_records=len(records), records=records,
        new_model_calls=0, raw_labels_changed=0, original_inputs_and_raw_outputs_unchanged=True,
        preserved_runtime_sources={name: native.protocol.sha256(raw) for name, raw in saved_sources.items()},
        usage_policy="Inherited usage is an existing execution reference, never additional model consumption.",
        interrupted_attempts="Not migrated as successful labels; their raw traces and unknown usage remain preserved."))


def validated_cli(cli):
    executable = native.discover_native_cli(cli)
    if executable.name.lower() != "codex.exe" or not executable.is_file():
        raise ValueError("The stream wrapper requires the native codex.exe launcher")
    return executable


def run_stream(root, pool_dir, stream="b", max_records=50, cli=native.DEFAULT_CLI, migrate_from=None):
    if stream not in {"a", "b"}:
        raise ValueError("Select stream a or b")
    executable = validated_cli(cli)
    wrapper = Path(__file__).resolve()
    wrapper_sha = native.protocol.sha256(wrapper.read_bytes())
    runner_sha = native.protocol.sha256(Path(native.__file__).read_bytes())
    metadata = dict(schema_version=1, stream=stream, wrapper_sha256=wrapper_sha,
                    native_runner_sha256=runner_sha, native_cli_name=executable.name,
                    native_cli_sha256=native.protocol.sha256(executable.read_bytes()),
                    python_executable=str(Path(sys.executable).resolve()),
                    protocol_sha256=native.protocol.sha256((native.GUARD + "\n" + native.protocol.ANNOTATION_PROMPT).encode("utf-8")),
                    context_policy="Unmodified individual record protocol; one fresh context per selected stream record",
                    authentication="Existing ChatGPT login is checked by the unchanged native runner",
                    validation_schema_version=native.protocol.SCHEMA_VERSION,
                    separate_output_directory=True, shared_progress_writers=False,
                    external_paid_api_calls=0, human_gold=False)
    previous_streams, previous_loader, previous_writer = native.STREAMS, native.load_pool, native.write_new
    migration_state = {}

    def guarded_loader(selected_root, selected_pool):
        directory, rows, protected = previous_loader(selected_root, selected_pool)
        if migrate_from is not None:
            source = Path(migrate_from).resolve()
            if not source.is_relative_to((directory / "codex_annotations").resolve()):
                raise ValueError("Migration source is outside its original annotation run directory")
            attempts, protected, sources = migration_inputs(source, stream, rows, protected)
            migration_state.update(attempts=attempts, saved_sources=sources, source=source)
        return directory, rows, {**protected, str(wrapper): wrapper_sha}

    def receipt_writer(path, value):
        previous_writer(path, value)
        if Path(path).name == "manifest.json" and isinstance(value, dict) and "specification" in value:
            previous_writer(Path(path).parent / "stream_wrapper_receipt.json", metadata)
            if migration_state:
                migrate_outputs(Path(path).parent, stream, **migration_state)

    native.STREAMS = (stream,)
    native.load_pool = guarded_loader
    native.write_new = receipt_writer
    try:
        # No resume: the original concurrent run's progress must never be written.
        return native.run(root, pool_dir, max_records=max_records, cli=str(executable))
    finally:
        native.STREAMS, native.load_pool, native.write_new = previous_streams, previous_loader, previous_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--pool-dir", required=True, type=Path)
    parser.add_argument("--stream", choices=["a", "b"], default="b")
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--codex-cli", default=native.DEFAULT_CLI)
    parser.add_argument("--migrate-from", type=Path)
    args = parser.parse_args()
    _, summary = run_stream(args.canonical_root, args.pool_dir, args.stream,
                            args.max_records, args.codex_cli, args.migrate_from)
    print(json.dumps(summary, indent=2))
    raise SystemExit(1 if summary["failures"] else 0)
