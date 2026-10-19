"""Paired closed-book annotations for a distinct top5 candidate-cohort pool.

Never relax the historical dev50 loader or mix its top10 labels into these inputs.
Fresh A/B contexts receive the same exact batch bytes, without author anchors,
rankings or split/event metadata. Native usage is counted once per execution;
labels remain machine-only, provisional, and distinct from a formal test freeze.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from uuid import uuid4

import retrieve_machine_questions as retrieval
import run_codex_annotation as native
import run_codex_batch_annotations as batch
from oil_evidence import machine_evaluation as protocol

VERSION = "codex-top5-candidate-cohort-paired-v2"
PARTITIONS = {"new_development", "all_new", "all_current"}


def validate_pool_rows(rows, questions, corpus):
    originals = {row["passage_id"]: row for row in corpus}
    queries = {row["record_id"]: row for row in questions}
    if len(queries) != len(questions) or len(rows) != len(questions):
        raise ValueError("Candidate question/pool count or uniqueness differs")
    seen = set()
    for row in rows:
        if set(row) != {"record_id", "question", "as_of", "evidence"}:
            raise ValueError("Candidate pool contains rankings, authoring or split hints")
        identifier = row["record_id"]
        query = queries.get(identifier)
        if (query is None or identifier in seen or any(row[key] != query[key] for key in ("question", "as_of"))):
            raise ValueError("Candidate pool and question membership/text/cutoff differ")
        seen.add(identifier)
        if protocol.clean_packet(row)["payload"] != row:
            raise ValueError("Evidence contains unknown fields or retrieval hints")
        for evidence in row["evidence"]:
            original = originals.get(evidence["passage_id"])
            if original is None or any(original.get(key) != value for key, value in evidence.items()):
                raise ValueError("Pooled evidence no longer matches the frozen original corpus")
    return rows


def load_pool(root, pool_dir):
    root, pool_dir = Path(root).resolve(), Path(pool_dir).resolve()
    allowed = (root / "data/machine_runs/candidate-retrieval").resolve()
    if not pool_dir.is_relative_to(allowed) or pool_dir == allowed:
        raise ValueError("Use a unique top5 candidate-retrieval attempt, not the old dev50 pool")
    receipt = native.read_json(pool_dir / "receipt.json")
    inputs = native.read_json(pool_dir / "inputs.json")
    if (receipt.get("status") != "candidate_evidence_pools_created_not_annotations_or_accuracy_results"
            or receipt.get("protocol") != retrieval.PROTOCOL or receipt.get("top_k") != 5
            or receipt.get("original_inputs_unchanged") is not True or receipt.get("author_anchor_injection") is not False
            or receipt.get("human_gold") is not False or receipt.get("machine_annotations_created") != 0
            or inputs.get("top_k") != 5 or inputs.get("temporal_conditions") != [True, False]
            or inputs.get("retrieval_modes") != list(retrieval.MODES)):
        raise ValueError("Expected the separately sealed anchor-free top5 six-condition candidate pool")
    artifact_bindings = {}
    for name, expected in receipt["artifact_sha256"].items():
        path = native.contained_file(pool_dir, name)
        if protocol.sha256(path.read_bytes()) != expected:
            raise ValueError("Candidate retrieval artifact changed")
        artifact_bindings[str(path)] = expected
    artifact_bindings[str(pool_dir / "receipt.json")] = protocol.sha256((pool_dir / "receipt.json").read_bytes())
    queries, cohort_protected, cohort_receipt = retrieval.load_cohort(root, Path(inputs["cohort"]))
    _, _, corpus, _, _, manifest, canonical_protected, canonical_bindings = retrieval.verified.load_verified_inputs(
        root, "oil-dev50-v1", "semantic-page-v2-buffer")
    if (inputs["bindings"] != canonical_bindings or receipt["actual_counts"] != cohort_receipt["actual_counts"]
            or manifest["corpus_sha256"] != canonical_bindings["corpus_sha256"]):
        raise ValueError("Candidate retrieval corpus/model/source receipt lineage differs")
    rows = protocol.parse_jsonl((pool_dir / "annotation_pool.jsonl").read_bytes(), "candidate annotation pool")
    validate_pool_rows(rows, queries, corpus)
    if len(rows) != receipt["actual_queries"]:
        raise ValueError("Candidate pool actual count differs")
    # A historical execution's public implementation may legitimately evolve.
    # Keep its original hash map in bindings; never certify it against new code.
    # All immutable raw/artifact bytes and original canonical sources are checked.
    protected = {**canonical_protected, **cohort_protected, **artifact_bindings}
    native.protected_unchanged(protected)
    bindings = dict(candidate_pool_receipt_sha256=artifact_bindings[str(pool_dir / "receipt.json")],
                    candidate_pool_sha256=protocol.sha256((pool_dir / "annotation_pool.jsonl").read_bytes()),
                    cohort_receipt_sha256=cohort_protected[str(Path(inputs["cohort"]) / "receipt.json")],
                    canonical_bindings=canonical_bindings,
                    historical_retrieval_implementation_sha256=inputs["current_implementation_sha256"],
                    historical_implementation_not_rebound=True)
    return pool_dir, rows, queries, protected, bindings


def select_rows(rows, questions, partition, max_records):
    if partition not in PARTITIONS or type(max_records) is not int or max_records <= 0:
        raise ValueError("Choose an explicit annotation partition and positive record limit")
    queries = {row["record_id"]: row for row in questions}
    eligible = [row for row in rows if (partition == "all_current" or not row["record_id"].startswith("dev50-"))
                and (partition != "new_development" or queries[row["record_id"]]["split"] == "development")]
    if max_records > len(eligible) or (max_records % 5 and max_records != len(eligible)):
        raise ValueError("Record limit exceeds eligible queries or cuts a stable five-record batch")
    return eligible, eligible[:max_records]


def execute_attempt(output, stream, batch_number, rows, binary, environment, mcp_names,
                    protected, requested_model, timeout_seconds, executor):
    native.protected_unchanged(protected)
    packets, stdin, schema = batch.prepare_batch(rows)
    batch_name = f"batch-{batch_number:04d}"
    attempt = output / stream / batch_name / ("attempt-" + uuid4().hex[:12])
    attempt.mkdir(parents=True, exist_ok=False)
    context = Path(tempfile.mkdtemp(prefix="oil-codex-cohort-annotation-"))
    native.write_new(attempt / "stdin.txt", stdin)
    native.write_new(attempt / "schema.json", schema)
    native.write_new(context / "packets.json", {key: value["payload"] for key, value in packets.items()})
    native.write_new(context / "protocol.txt", batch.BATCH_PROMPT.encode("utf-8"))
    native.write_new(context / "schema.json", schema)
    final_path = context / "final.json"
    command = native.cli_command(binary, context, context / "schema.json", final_path, requested_model, mcp_names)
    native.write_new(attempt / "command.json", command)
    tick = perf_counter()
    try:
        exit_code = executor(command, stdin, context, attempt, environment, timeout_seconds)
        if final_path.exists():
            native.write_new(attempt / "raw_final.json", final_path.read_bytes())
        raw = (attempt / "raw_final.json").read_bytes() if (attempt / "raw_final.json").exists() else b""
        events = native.parse_execution_events((attempt / "events.jsonl").read_bytes())
        execution_id = events["thread_id"] or "unavailable-" + uuid4().hex
        identity = dict(service="codex_cli_chatgpt_subscription", requested_model=requested_model or "cli_default_unresolved",
                        observed_model=events["observed_model"], execution_id=execution_id,
                        observation_source="execution_metadata" if events["observed_model"] else "unavailable",
                        identity_note="Native execution metadata only, never assistant self-report.")
        errors = list(events["errors"])
        if exit_code != 0:
            errors.append("Native Codex process returned nonzero status")
        try:
            envelopes = batch.extract_outputs(raw.decode("utf-8"), packets, identity, events["usage"], stdin,
                dict(batch_context_version=batch.BATCH_VERSION, cohort_protocol=VERSION,
                     requested_model=requested_model, other_settings="Inherited CLI configuration, not independently observed"))
            errors.extend(f"{key}: {error}" for key, value in envelopes.items() for error in value["validation_errors"])
        except (ValueError, TypeError, UnicodeError) as error:
            envelopes = {}
            errors.append(str(error))
        native.write_new(attempt / "envelopes.json", envelopes)
        native.protected_unchanged(protected)
        receipt = dict(accepted=not errors, stream=stream, batch=batch_name, record_ids=list(packets),
            errors=errors, process_exit_code=exit_code, elapsed_seconds=perf_counter() - tick,
            execution_id=execution_id, usage=events["usage"], full_stdin_sha256=protocol.sha256(stdin),
            artifacts_sha256={path.name: protocol.sha256(path.read_bytes()) for path in attempt.iterdir() if path.is_file()},
            original_inputs_unchanged=True, human_gold=False, subscription_dollar_cost=None, external_paid_api_calls=0)
        native.write_new(attempt / "receipt.json", receipt)
        return dict(attempt=attempt, receipt=receipt, envelopes=envelopes,
                    usage=dict(execution_id=execution_id, stream=stream, record_ids=list(packets), usage=events["usage"],
                               native_events_sha256=protocol.sha256((attempt / "events.jsonl").read_bytes())))
    except Exception as error:
        failure = dict(accepted=False, stream=stream, batch=batch_name, record_ids=list(packets),
                       error_type=type(error).__name__, errors=[str(error)])
        native.write_new(attempt / "failure.json", failure)
        return dict(attempt=attempt, receipt=failure, envelopes={}, usage=None)


def validate_paired_inputs(output):
    """Actual whole stdin bytes must match A/B for each accepted paired batch."""
    progress_path = output / "progress.jsonl"
    progress = protocol.parse_jsonl(progress_path.read_bytes(), "paired progress") if progress_path.exists() else []
    accepted = {}
    for event in progress:
        if event.get("type") != "completed" or event.get("accepted") is not True:
            continue
        path = native.contained_file(output, event["receipt"])
        receipt = native.read_json(path)
        if protocol.sha256(path.read_bytes()) != event["receipt_sha256"]:
            raise ValueError("Paired receipt changed")
        accepted[(receipt["batch"], receipt["stream"])] = (path.parent / "stdin.txt").read_bytes()
    paired = 0
    for batch_name, stream in accepted:
        if stream == "a" and (batch_name, "b") in accepted:
            if accepted[(batch_name, "a")] != accepted[(batch_name, "b")]:
                raise ValueError("A/B actual batch stdin bytes differ")
            paired += 1
    return paired


def run(root, pool_dir, max_records=20, *, partition="new_development", requested_model=None,
        cli=native.DEFAULT_CLI, resume=None, timeout_seconds=900, executor=native.invoke_cli):
    root = Path(root).resolve()
    pool_dir, rows, queries, protected, bindings = load_pool(root, pool_dir)
    eligible, selected = select_rows(rows, queries, partition, max_records)
    binary = native.discover_native_cli(cli)
    environment = native.child_environment()
    mcp_names = native.configured_mcp_names(environment)
    version = subprocess.run([str(binary), "--version"], capture_output=True, env=environment, check=True).stdout.decode("utf-8").strip()
    login = subprocess.run([str(binary), "login", "status"], capture_output=True, env=environment, check=True)
    if (login.stdout + login.stderr).decode("utf-8").strip() != "Logged in using ChatGPT":
        raise ValueError("Existing ChatGPT subscription authentication required; no API fallback")
    packets = {row["record_id"]: protocol.clean_packet(row, prompt=batch.BATCH_PROMPT) for row in eligible}
    implementation = {str(path.resolve()): protocol.sha256(path.read_bytes()) for path in
        (Path(__file__), Path(batch.__file__), Path(native.__file__), Path(protocol.__file__))}
    protected.update(implementation)
    specification = dict(version=VERSION, bindings=bindings, partition=partition, batch_size=5,
        eligible_record_ids=list(packets), eligible_packets_sha256=protocol.sha256(protocol.canonical_bytes(packets)),
        batch_prompt_sha256=protocol.sha256(batch.BATCH_PROMPT.encode("utf-8")), requested_model=requested_model,
        native_cli_sha256=protocol.sha256(binary.read_bytes()), cli_version=version,
        disabled_mcp_names=mcp_names, implementation_sha256=implementation,
        human_gold=False, independent_model_diversity_claimed=False, formal_test_frozen=False)
    specification["old_protocol_outputs_reused"] = False
    specification["changed_top5_old50_inputs_require_fresh_annotations"] = partition == "all_current"
    if resume:
        output = Path(resume).resolve()
        if not output.is_relative_to((pool_dir / "cohort_annotations").resolve()):
            raise ValueError("Resume escapes its candidate pool annotation directory")
        native.verify_resume(output, specification)
    else:
        output = pool_dir / "cohort_annotations" / (native.utc_now().replace(":", "-") + "-" + uuid4().hex[:10])
        output.mkdir(parents=True, exist_ok=False)
        for name, value in (("packets.json", packets), ("protocol.txt", batch.BATCH_PROMPT.encode("utf-8")),
                            ("protected_inputs.json", protected), ("runner_script.py", Path(__file__).read_bytes())):
            native.write_new(output / name, value)
        immutable = {name: protocol.sha256((output / name).read_bytes()) for name in
                     ("packets.json", "protocol.txt", "protected_inputs.json", "runner_script.py")}
        native.write_new(output / "manifest.json", dict(specification=specification, immutable_sha256=immutable,
            provenance="machine_only", human_gold=False, actual_labels_created=0,
            context_policy="fresh_native_context_per_stream_and_batch_no_prior_labels_or_author_anchors"))
    completed = {stream: batch.completed_records(output, stream, packets) for stream in native.STREAMS}
    jobs = []
    for offset in range(0, len(selected), 5):
        part = selected[offset:offset + 5]
        ids = {row["record_id"] for row in part}
        for stream in native.STREAMS:
            done = ids & set(completed[stream])
            if done == ids:
                continue
            if done:
                raise ValueError("Partially accepted stable batch cannot be regrouped")
            jobs.append((stream, offset // 5 + 1, part))
    failures, invocations = [], 0
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending, next_job = {}, 0
        while pending or next_job < len(jobs):
            stopped = bool(failures) or (output / "STOP_REQUESTED").exists()
            while not stopped and next_job < len(jobs) and len(pending) < 2:
                stream, number, part = jobs[next_job]
                future = workers.submit(execute_attempt, output, stream, number, part, binary, environment,
                    mcp_names, protected, requested_model, timeout_seconds, executor)
                pending[future] = (stream, number)
                next_job += 1
                invocations += 1
            if not pending:
                break
            ready, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in ready:
                stream, number = pending.pop(future)
                result = future.result()
                receipt, attempt = result["receipt"], result["attempt"]
                if result["usage"] is not None:
                    native.append_progress(output / "execution_usage.jsonl", result["usage"])
                if (attempt / "receipt.json").exists():
                    native.append_progress(output / "progress.jsonl", dict(type="completed", accepted=receipt["accepted"],
                        stream=stream, batch=receipt["batch"], record_ids=receipt["record_ids"],
                        receipt=attempt.relative_to(output).as_posix() + "/receipt.json",
                        receipt_sha256=protocol.sha256((attempt / "receipt.json").read_bytes())))
                else:
                    native.append_progress(output / "progress.jsonl", dict(type="failed", **receipt))
                if not receipt["accepted"]:
                    failures.append(receipt)
                else:
                    completed[stream].update(result["envelopes"])
                print(f"Cohort annotation {stream.upper()} batch{number}: accepted={receipt['accepted']}, records={len(receipt['record_ids'])}", flush=True)
    native.protected_unchanged(protected)
    paired_batches = validate_paired_inputs(output)
    for stream in native.STREAMS:
        native.write_new(output / (stream + "_outputs-" + uuid4().hex[:10] + ".jsonl"),
            b"".join(protocol.canonical_bytes(value) + b"\n" for value in completed[stream].values()))
    summary = dict(output=str(output), status="selected_paired_machine_annotations_complete" if not failures and not (output / "STOP_REQUESTED").exists()
        else "stopped_without_automatic_retry", partition=partition, records_requested_per_stream=max_records,
        accepted_counts={key: len(value) for key, value in completed.items()}, paired_batch_count=paired_batches,
        new_native_invocations=invocations, failures=failures, subscription_usage=batch.observed_usage(output),
        original_inputs_unchanged=True, external_paid_api_calls=0, human_labels_created=0, human_gold=False,
        formal_test_frozen=False, raw_agreement_not_adjudication=True,
        limitations=["Same subscription model in independent contexts is not independent model diversity.",
                     "Candidate annotation counts are not a 1000-question accepted formal test freeze.",
                     "Raw preadjudication A/B outputs must be preserved before any reconciliation."])
    native.write_new(output / ("summary-" + uuid4().hex[:10] + ".json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return output, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=20)
    parser.add_argument("--partition", choices=sorted(PARTITIONS), default="new_development")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--requested-model")
    args = parser.parse_args()
    _, summary = run(args.canonical_root, args.pool_dir, args.max_records, partition=args.partition,
                     requested_model=args.requested_model, resume=args.resume)
    raise SystemExit(1 if summary["failures"] else 0)
