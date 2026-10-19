"""Batch closed-book Codex annotation without changing individual-run receipts.

Each fresh execution receives up to five independent question/evidence packets.
Full raw batch responses are retained and each label binds its own JSON key.
Native usage is stored once per execution, not once per extracted label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from uuid import uuid4

import run_codex_annotation as native
from oil_evidence import machine_evaluation as protocol
from oil_evidence.store import ROOT

BATCH_VERSION = "codex-own-evidence-batch-v1"
BATCH_INSTRUCTIONS = """
Batch context protocol: codex-own-evidence-batch-v1.
The input is an object keyed by record ID. For EACH record, use ONLY evidence
inside that record, applying its own query cutoff. Never borrow facts, evidence
IDs, quotes or answers from another record, even when questions appear related.
Return one JSON object with exactly the same record-ID keys. Each value must be
{"record_id":"the matching object key","label":the six-field annotation label}.
Judge every passage in each
record's own evidence pool. Do not include any explanation outside this object.
"""
BATCH_PROMPT = native.GUARD + "\n" + protocol.ANNOTATION_PROMPT + BATCH_INSTRUCTIONS


def prepare_batch(rows):
    packets = {}
    for row in rows:
        packet = protocol.clean_packet(row, prompt=BATCH_PROMPT)
        record_id = packet["payload"]["record_id"]
        if record_id in packets:
            raise ValueError("Duplicate batch record ID")
        packets[record_id] = packet
    if not packets:
        raise ValueError("Empty annotation batch")
    payload = {record_id: packet["payload"] for record_id, packet in packets.items()}
    stdin = BATCH_PROMPT.encode("utf-8") + b"\n\nUNTRUSTED_RECORD_PACKETS_JSON\n" + protocol.canonical_bytes(payload) + b"\n"
    schema = dict(type="object", additionalProperties=False, required=list(packets),
                  properties={record_id: dict(type="object", additionalProperties=False,
                      required=["record_id", "label"], properties=dict(record_id=dict(type="string", enum=[record_id]),
                      label=native.output_schema(packet))) for record_id, packet in packets.items()})
    return packets, stdin, schema


def extract_outputs(raw, packets, identity, usage, stdin, decoding):
    """Keep the ENTIRE raw response and extract by the question's own key."""
    parsed = json.loads(raw, object_pairs_hook=protocol._unique_object, parse_constant=protocol._invalid_constant)
    if not isinstance(parsed, dict) or set(parsed) != set(packets):
        raise ValueError("Batch response record-ID coverage differs")
    envelopes = {}
    for record_id, packet in packets.items():
        envelope = protocol.wrap_independent_output(packet, raw, identity, identity["execution_id"],
                    decoding, usage=None, execution_input_sha256=protocol.sha256(stdin), raw_label_path=[record_id, "label"])
        envelope["batch_context_version"] = BATCH_VERSION
        envelope["usage_reference_execution_id"] = identity["execution_id"]
        envelope["usage_note"] = "Full execution usage retained once in its batch receipt and execution_usage.jsonl."
        envelopes[record_id] = envelope
    return envelopes


def completed_records(output, stream, packets):
    records = {}
    if not (output / stream).exists():
        return records
    progress = protocol.parse_jsonl((output / "progress.jsonl").read_bytes(), "batch progress")
    bound = {event["receipt"]: event.get("receipt_sha256") for event in progress
             if event.get("type") == "completed" and event.get("accepted") is True}
    for path in sorted((output / stream).glob("batch-*/attempt-*/receipt.json")):
        if not path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Batch receipt escapes output")
        receipt = native.read_json(path)
        if receipt.get("accepted") is not True:
            continue
        if bound.get(path.relative_to(output).as_posix()) != protocol.sha256(path.read_bytes()):
            raise ValueError("Batch receipt differs from progress binding")
        attempt = path.parent
        for name, expected in receipt["artifacts_sha256"].items():
            if protocol.sha256(native.contained_file(attempt, name).read_bytes()) != expected:
                raise ValueError("Retained batch artifact changed")
        envelope_rows = native.read_json(attempt / "envelopes.json")
        if set(envelope_rows) != set(receipt["record_ids"]):
            raise ValueError("Batch envelope coverage changed")
        raw = (attempt / "raw_final.json").read_bytes()
        stdin = (attempt / "stdin.txt").read_bytes()
        for record_id, envelope in envelope_rows.items():
            if (record_id not in packets or record_id in records
                    or envelope["raw_output"].encode("utf-8") != raw
                    or envelope["execution_input_sha256"] != protocol.sha256(stdin)
                    or envelope["raw_label_path"] != [record_id, "label"]
                    or protocol.validate_independent_output(envelope, packets[record_id])):
                raise ValueError("Completed batch label no longer binds its raw response and packet")
            records[record_id] = envelope
    return records


def observed_usage(output):
    """Deduplicate emitted usage by actual fresh CLI execution ID."""
    path = output / "execution_usage.jsonl"
    rows = protocol.parse_jsonl(path.read_bytes(), "execution usage") if path.exists() else []
    unique = {}
    for row in rows:
        execution_id = row["execution_id"]
        if execution_id in unique and unique[execution_id] != row:
            raise ValueError("Conflicting usage for the same execution")
        unique[execution_id] = row
    totals = {}
    for row in unique.values():
        for key, value in (row.get("usage") or {}).items():
            if type(value) is int and value >= 0:
                totals[key] = totals.get(key, 0) + value
    return dict(executions=len(unique), known_usage_executions=sum(row.get("usage") is not None for row in unique.values()),
                native_token_totals=totals, counting_unit="unique execution_id, never extracted label count",
                subscription_dollar_cost=None, credit_conversion="Not measured")


def run(root, pool_dir, max_records=5, batch_size=5, requested_model=None,
        cli=native.DEFAULT_CLI, resume=None, timeout_seconds=900, executor=native.invoke_cli):
    if type(batch_size) is not int or not 1 <= batch_size <= 5:
        raise ValueError("Choose a batch size from 1 through 5")
    if type(max_records) is not int or max_records < 1:
        raise ValueError("Choose a positive explicit record limit")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("Choose a positive timeout")
    root = Path(root).resolve()
    pool_dir, rows, protected = native.load_pool(root, pool_dir)
    if max_records > len(rows) or (max_records % batch_size and max_records != len(rows)):
        raise ValueError("Use whole fixed batches or the complete final remainder")
    binary = native.discover_native_cli(cli)
    environment = native.child_environment()
    mcp_names = native.configured_mcp_names(environment)
    version = subprocess.run([str(binary), "--version"], capture_output=True, check=True, env=environment).stdout.decode("utf-8").strip()
    login = subprocess.run([str(binary), "login", "status"], capture_output=True, check=True, env=environment)
    if (login.stdout + login.stderr).decode("utf-8").strip() != "Logged in using ChatGPT":
        raise ValueError("Existing ChatGPT subscription login required; no API fallback")
    all_packets = {row["question_id"]: protocol.clean_packet(row, prompt=BATCH_PROMPT) for row in rows}
    implementation = {str(path.resolve()): protocol.sha256(path.read_bytes()) for path in
                      (Path(__file__), Path(native.__file__), ROOT / "src/oil_evidence/machine_evaluation.py")}
    protected.update(implementation)
    spec = dict(batch_context_version=BATCH_VERSION, batch_size=batch_size,
                pool_sha256=protocol.sha256((pool_dir / "annotation_pool.jsonl").read_bytes()),
                protocol_sha256=protocol.sha256(BATCH_PROMPT.encode("utf-8")),
                packets_sha256=protocol.sha256(protocol.canonical_bytes(all_packets)),
                execution_implementation_sha256=implementation, requested_model=requested_model,
                cli_version=version, native_cli_sha256=protocol.sha256(binary.read_bytes()),
                disabled_configured_mcp_servers=mcp_names, streams=list(native.STREAMS))
    parent = pool_dir / "codex_batch_annotations"
    if resume:
        output = Path(resume).resolve()
        if not output.is_relative_to(parent.resolve()):
            raise ValueError("Batch resume is outside its distinct output directory")
        native.verify_resume(output, spec)
    else:
        output = parent / ("run-" + uuid4().hex[:12])
        output.mkdir(parents=True, exist_ok=False)
        native.write_new(output / "packets.json", all_packets)
        native.write_new(output / "protocol.txt", BATCH_PROMPT.encode("utf-8"))
        native.write_new(output / "protected_inputs.json", protected)
        snapshots = {"code/batch.py": Path(__file__), "code/native.py": Path(native.__file__),
                     "code/protocol.py": ROOT / "src/oil_evidence/machine_evaluation.py"}
        for name, path in snapshots.items():
            native.write_new(output / name, path.read_bytes())
        names = ["packets.json", "protocol.txt", "protected_inputs.json"] + list(snapshots)
        native.write_new(output / "manifest.json", dict(specification=spec,
            immutable_sha256={name: protocol.sha256((output / name).read_bytes()) for name in names},
            created_utc=native.utc_now(), provenance="machine_only", human_gold=False,
            context_policy="fresh execution per batch; identical A/B full inputs; no prior outputs",
            usage_counting="one native usage row per execution, not per extracted record",
            limitations=["Questions in one batch share context, so their label errors may be correlated.",
                         "Own-pool citation checks are mechanical; they do not certify semantic correctness.",
                         "Unknown model identities remain null; repeat executions do not establish model diversity.",
                         "This protocol is distinct from the preserved individual-record run and cannot be silently mixed with it."] ))
    completed = {stream: completed_records(output, stream, all_packets) for stream in native.STREAMS}
    failures, new_calls = [], 0
    for stream in native.STREAMS:
        for offset in range(0, max_records, batch_size):
            batch_rows = rows[offset:min(offset + batch_size, max_records)]
            packets, stdin, schema = prepare_batch(batch_rows)
            record_ids = list(packets)
            done = set(record_ids) & set(completed[stream])
            if done == set(record_ids):
                continue
            if done:
                raise ValueError("Partially accepted batch cannot be resumed as a new context")
            native.protected_unchanged(protected)
            batch_name = f"batch-{offset // batch_size + 1:03d}"
            attempt = output / stream / batch_name / ("attempt-" + uuid4().hex[:12])
            attempt.mkdir(parents=True, exist_ok=False)
            context = Path(tempfile.mkdtemp(prefix="oil-codex-batch-"))
            native.write_new(attempt / "stdin.txt", stdin)
            native.write_new(attempt / "schema.json", schema)
            native.write_new(context / "packets.json", {qid: packet["payload"] for qid, packet in packets.items()})
            native.write_new(context / "protocol.txt", BATCH_PROMPT.encode("utf-8"))
            native.write_new(context / "schema.json", schema)
            final_path = context / "final.json"
            command = native.cli_command(binary, context, context / "schema.json", final_path, requested_model, mcp_names)
            native.append_progress(output / "progress.jsonl", dict(type="started", stream=stream, batch=batch_name,
                                   record_ids=record_ids, attempt=attempt.relative_to(output).as_posix()))
            tick = perf_counter()
            try:
                new_calls += 1
                exit_code = executor(command, stdin, context, attempt, environment, timeout_seconds)
                elapsed = perf_counter() - tick
                if final_path.exists():
                    native.write_new(attempt / "raw_final.json", final_path.read_bytes())
                raw = (attempt / "raw_final.json").read_bytes() if (attempt / "raw_final.json").exists() else b""
                events = native.parse_execution_events((attempt / "events.jsonl").read_bytes())
                execution_id = events["thread_id"] or "unavailable-" + uuid4().hex
                native.append_progress(output / "execution_usage.jsonl", dict(execution_id=execution_id,
                    stream=stream, record_ids=record_ids, usage=events["usage"],
                    native_events_sha256=protocol.sha256((attempt / "events.jsonl").read_bytes())))
                identity = dict(service="codex_cli_chatgpt_subscription", requested_model=requested_model or "cli_default_unresolved",
                    observed_model=events["observed_model"], execution_id=execution_id,
                    observation_source="execution_metadata" if events["observed_model"] else "unavailable",
                    identity_note="Execution metadata only; assistant self-report is never used.")
                errors = list(events["errors"])
                try:
                    envelopes = extract_outputs(raw.decode("utf-8"), packets, identity, events["usage"], stdin,
                                                dict(batch_context_version=BATCH_VERSION, requested_model=requested_model,
                                                     other_settings="Inherited CLI defaults; not independently observed"))
                    errors.extend(f"{qid}: {error}" for qid, envelope in envelopes.items() for error in envelope["validation_errors"])
                except (ValueError, TypeError) as error:
                    envelopes = {}
                    errors.append(str(error))
                if exit_code != 0:
                    errors.append("Native Codex process returned nonzero status")
                native.write_new(attempt / "envelopes.json", envelopes)
                native.protected_unchanged(protected)
                receipt = dict(completed_utc=native.utc_now(), accepted=not errors, errors=errors, stream=stream,
                    batch=batch_name, record_ids=record_ids, execution_id=execution_id, usage=events["usage"],
                    elapsed_seconds=elapsed, process_exit_code=exit_code, actual_full_stdin_sha256=protocol.sha256(stdin),
                    artifacts_sha256={path.name: protocol.sha256(path.read_bytes()) for path in attempt.iterdir() if path.is_file()},
                    inputs_unchanged=True, external_paid_api_calls=0, human_gold=False)
                native.write_new(attempt / "receipt.json", receipt)
                native.append_progress(output / "progress.jsonl", dict(type="completed", accepted=not errors, stream=stream,
                    batch=batch_name, record_ids=record_ids, errors=errors,
                    receipt=attempt.relative_to(output).as_posix() + "/receipt.json",
                    receipt_sha256=protocol.sha256((attempt / "receipt.json").read_bytes())))
                if errors:
                    failures.append(dict(stream=stream, batch=batch_name, errors=errors))
                    break
                completed[stream].update(envelopes)
                print(f"Codex batch {stream.upper()} {batch_name}: {len(record_ids)} records accepted, {elapsed:.2f} seconds", flush=True)
            except Exception as error:
                failure = dict(failed_utc=native.utc_now(), stream=stream, batch=batch_name,
                               error_type=type(error).__name__, error=str(error), accepted=False)
                native.write_new(attempt / "failure.json", failure)
                native.append_progress(output / "progress.jsonl", dict(type="failed", **failure))
                failures.append(failure)
                break
        if failures:
            break
    native.protected_unchanged(protected)
    summary = dict(completed_utc=native.utc_now(), output=str(output), batch_context_version=BATCH_VERSION,
                   records_requested_per_stream=max_records, batch_size=batch_size,
                   accepted_counts={stream: len(completed[stream]) for stream in native.STREAMS},
                   new_codex_cli_invocations=new_calls, failures=failures, usage=observed_usage(output),
                   original_inputs_unchanged=True, external_paid_api_calls=0, new_human_labels=0, human_gold=False,
                   status="selected_batch_complete" if not failures else "stopped_on_retained_failure")
    native.write_new(output / ("summary-" + uuid4().hex[:12] + ".json"), summary)
    return output, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--pool-dir", required=True, type=Path)
    parser.add_argument("--max-records", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--requested-model")
    parser.add_argument("--codex-cli", default=native.DEFAULT_CLI)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    _, summary = run(args.canonical_root, args.pool_dir, args.max_records, args.batch_size,
                     args.requested_model, args.codex_cli, args.resume, args.timeout_seconds)
    print(json.dumps(summary, indent=2))
    raise SystemExit(1 if summary["failures"] else 0)
