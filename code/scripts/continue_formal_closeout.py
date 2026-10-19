"""One-shot, sealed prerequisite watcher; never retry failures or change models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

VERSION = "sealed-machine-formal-closeout-v1"
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_RUN = "2026-10-13T03-43-43-formal-v4"
RECEIPT_SHA = "90879d0215db9f2d7b224a31eae3e31ab33dfaf1501838d7b0d68ef0b8fda86e"
SAMPLE_SHA = "079d484d1100f4a62dd9c687f1a0d575bd92bcc87694c58970ab3de84d1f4baf"
RUNTIME_KEYS = {"generation_prompt", "verification_prompt", "judge_prompt", "implementation", "index_manifest", "retrieval_config"}
SOURCES = ["scripts/continue_formal_closeout.py", "scripts/run_codex_annotation.py",
    "scripts/run_machine_experiments.py", "scripts/assemble_machine_freeze.py",
    "scripts/freeze_machine_dataset.py", "scripts/pipeline_wire_cohort_adjudication.py",
    "scripts/pipeline_cohort_adjudication.py", "scripts/pipeline_compact_cohort_adjudication.py",
    "scripts/compact_machine_adjudication.py", "scripts/adjudicate_machine_annotations.py",
    "scripts/run_codex_clarified_cohort.py", "scripts/run_codex_corrected_cohort.py",
    "src/oil_evidence/codex_runtime.py", "src/oil_evidence/machine_evaluation.py",
    "src/oil_evidence/wire_codec.py", "config.toml"]


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else
                     (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())


def inside(path, parent):
    value = str(path)
    if os.name == "nt" and value.startswith("\\\\?\\"):
        value = value[4:]
    resolved = Path(value).resolve()
    if resolved == Path(parent).resolve() or not resolved.is_relative_to(Path(parent).resolve()):
        raise ValueError("Artifact escapes its dedicated run directory")
    return resolved


def guards(plan):
    for name, expected in plan["protected_sha256"].items():
        if digest(name) != expected:
            raise ValueError("Sealed closeout input changed: " + name)


def prepare(canonical_root, producer_run, review_run, *, max_formal_contexts):
    if type(max_formal_contexts) is not int or not 1 <= max_formal_contexts <= 1200:
        raise ValueError("Seal an explicit formal-attempt cap before execution")
    root = Path(canonical_root).resolve()
    producer = inside(producer_run, root / "data/machine_runs/candidate-retrieval")
    review = inside(review_run, root / "data/machine_runs/cohort_pipeline")
    source, watcher = read(producer / "manifest.json"), read(review / "manifest.json")
    spec = source["specification"]
    if (spec["version"] != "codex-format-clarified-first-attempt-cohort-v1"
            or watcher["version"] != "wire-source-compact-adjudication-clarified-cache-v4"
            or Path(watcher["source_annotation_run"]).resolve() != producer
            or len(spec["eligible_record_ids"]) != 1200
            or watcher["source_order_record_ids"] != spec["eligible_record_ids"]
            or watcher["chunks"] != [spec["eligible_record_ids"][i:i + 20] for i in range(0, 1200, 20)]
            or watcher["chunk_size"] != 20
            or spec["requested_model"] is not None):
        raise ValueError("Expected the unchanged complete clarified1200 source and watcher")
    runtime = root / "data/machine_runs/experiment-runtime" / RUNTIME_RUN
    receipt, mapping = read(runtime / "receipt.json"), read(runtime / "runtime_artifacts.json")
    if digest(runtime / "receipt.json") != RECEIPT_SHA or set(mapping) != RUNTIME_KEYS:
        raise ValueError("Use the exact sealed formal-v4 runtime receipt and six artifacts")
    cfg = receipt["config"]
    if (receipt["contract"] != "frozen-machine-1000-retrieval-100-answer-v4"
            or cfg["requested_model"] is not None or cfg["max_cases"] != 8
            or cfg["max_request_bytes"] != 160000):
        raise ValueError("Frozen formal configuration changed")
    protected = {str(ROOT / name): digest(ROOT / name) for name in SOURCES}
    for name in (producer / "manifest.json", review / "manifest.json", runtime / "receipt.json", runtime / "runtime_artifacts.json"):
        protected[str(name)] = digest(name)
    for key, name in mapping.items():
        path = inside(name, runtime / "runtime")
        if digest(path) != receipt["runtime_sha256"][key]:
            raise ValueError("Actual formal runtime artifact changed")
        protected[str(path)] = digest(path)
    for name, item in read(mapping["implementation"]).items():
        matches = [ROOT / "scripts" / name, ROOT / "src/oil_evidence" / name]
        paths = [path for path in matches if path.is_file()]
        if len(paths) != 1 or digest(paths[0]) != item["sha256"]:
            raise ValueError("Archived formal implementation differs")
        protected[str(paths[0])] = item["sha256"]
    sample = inside(cfg["answer_sample_binding"]["directory"], root / "data/machine_runs/answer-samples")
    if digest(sample / "sample_plan.json") != SAMPLE_SHA:
        raise ValueError("Preselected100 answer sample changed")
    for name in ("sample_plan.json", "receipt.json"):
        protected[str(sample / name)] = digest(sample / name)
    python = root / ".venv/Scripts/python.exe"
    if not python.is_file():
        raise ValueError("Canonical project Python launcher is unavailable")
    output = root / "data/machine_runs/formal-closeout" / ("run-" + uuid4().hex[:10])
    plan = dict(version=VERSION, canonical_root=str(root), project_root=str(ROOT), python=str(python),
        producer_run=str(producer), review_run=str(review), runtime_map=str(runtime / "runtime_artifacts.json"),
        retrieval_run=str(producer.parents[1]), protected_sha256=protected, max_formal_contexts=max_formal_contexts,
        instructor_approval="user_reported100_answer_experiment_approved", human_gold=False,
        subscription_dollar_cost=None, publication=False, cloud_submission=False, prepare_native_calls=0)
    guards(plan)
    write(output / "manifest.json", plan)
    write(output / "prepare_receipt.json", dict(manifest_sha256=digest(output / "manifest.json"), native_calls=0))
    return output, plan


def prerequisite(plan):
    producer, review = Path(plan["producer_run"]), Path(plan["review_run"])
    for run in (producer, review):
        if (run / "failure.json").exists() or (run / "STOP_REQUESTED").exists():
            raise ValueError("Source failure or explicit stop retained; no retry")
        progress = run / "progress.jsonl"
        if progress.exists():
            raw = progress.read_bytes()
            for line in raw[:raw.rfind(b"\n") + 1].splitlines():
                event = json.loads(line, object_pairs_hook=unique)
                if event.get("accepted") is False or event.get("type") == "failed":
                    raise ValueError("Rejected source progress; no retry")
    receipts = list(producer.glob("*/batch-*/attempt-*/receipt.json"))
    receipts += list(review.glob("adjudications/chunk-*/bounded-run/executions/*/receipt.json"))
    for path in receipts:
        value = read(path)
        if value.get("accepted") is not True or value.get("errors") or value.get("process_exit_code") != 0:
            raise ValueError("Current source/native/quota failure retained; no retry")
    producer_ready = False
    for path in producer.glob("summary-*.json"):
        value = read(path)
        if value.get("failures") or value.get("status") == "stopped_without_automatic_retry":
            raise ValueError("Source producer stopped; no retry")
        producer_ready |= (value.get("status") == "selected_clarified_calls_complete"
            and value.get("accepted_counts") == {"a": 1200, "b": 1200}
            and value.get("human_gold") is False and value.get("external_paid_api_calls") == 0)
    for path in sorted(review.glob("summary-*.json")):
        value = read(path)
        if value.get("complete") is not True:
            continue
        if (value.get("expected_records") != 1200
                or value.get("retained_records") != 1200 or value.get("accepted") != 1200
                or value.get("unresolved") != 0 or value.get("retained_chunks") != 60
                or value.get("exported_chunks") != 60 or value.get("human_gold") is not False
                or Path(value["output"]).resolve() != review):
            raise ValueError("Complete source review does not meet the full1200 acceptance gate")
        inside(value["complete_source_inputs"], review)
        if Path(value["complete_source_inputs"]).resolve() != review / "complete_source_inputs":
            raise ValueError("Source export is not the original complete-source directory")
        files = value["adjudication_files"]
        if len(files) != 60 or len(set(files)) != 60:
            raise ValueError("Use exactly the60 original-owned decision files from the final summary")
        for name in files:
            if not inside(name, Path(plan["canonical_root"]) / "data/machine_runs/cohort_pipeline").is_file():
                raise ValueError("Original adjudication file is missing")
        return (path, value) if producer_ready else None
    return None


def trailing_object(stdout):
    text = stdout.decode("utf-8")
    decoder = json.JSONDecoder(object_pairs_hook=unique)
    for offset in reversed([i for i, char in enumerate(text) if char == "{" and (i == 0 or text[i - 1] == "\n")]):
        try:
            value, end = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not text[end:].strip():
            return value
    raise ValueError("Stage stdout lacks an unambiguous final JSON object")


def subprocess_executor(command, cwd, environment):
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, check=False)
    return result.returncode, result.stdout, result.stderr


def run(output, *, max_formal_contexts, max_wait_seconds=21600, poll_seconds=5,
        executor=subprocess_executor, sleeper=time.sleep, clock=time.monotonic):
    output = Path(output).resolve()
    plan = read(output / "manifest.json")
    if digest(output / "manifest.json") != read(output / "prepare_receipt.json")["manifest_sha256"]:
        raise ValueError("Prepared closeout manifest changed")
    if (plan["version"] != VERSION or type(max_formal_contexts) is not int
            or max_formal_contexts != plan["max_formal_contexts"] or not 0 <= max_wait_seconds <= 21600
            or not 1 <= poll_seconds <= 30):
        raise ValueError("Explicit formal-attempt cap and bounded waiting policy required")
    inside(output, Path(plan["canonical_root"]) / "data/machine_runs/formal-closeout")
    guards(plan)
    write(output / "run_started.json", dict(max_formal_contexts=max_formal_contexts,
        max_wait_seconds=max_wait_seconds, poll_seconds=poll_seconds, automatic_retry=False))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")
    def stage(name, script, arguments):
        guards(plan)
        directory = output / "stages" / name
        command = [plan["python"], "-X", "utf8", str(ROOT / "scripts" / script), *map(str, arguments)]
        write(directory / "attempt.json", dict(command=command, cwd=str(ROOT), automatic_retry=False))
        try:
            code, stdout, stderr = executor(command, ROOT, environment)
        except Exception as error:
            code, stdout, stderr = None, b"", (type(error).__name__ + ": " + str(error)).encode()
        write(directory / "stdout.txt", stdout)
        write(directory / "stderr.txt", stderr)
        write(directory / "receipt.json", dict(exit_code=code, stdout_sha256=digest(directory / "stdout.txt"),
            stderr_sha256=digest(directory / "stderr.txt"), subscription_dollar_cost=None))
        if code != 0:
            raise ValueError("Failed closeout stage retained: " + name)
        result = trailing_object(stdout)
        guards(plan)
        write(directory / "result.json", result)
        return result
    try:
        started = clock()
        while True:
            guards(plan)
            ready = prerequisite(plan)
            if ready:
                break
            if clock() - started >= max_wait_seconds:
                raise TimeoutError("Prerequisites not complete; no downstream call started")
            sleeper(min(poll_seconds, max_wait_seconds - (clock() - started)))
        summary_path, summary = ready
        write(output / "source_ready.json", dict(summary_path=str(summary_path), summary_sha256=digest(summary_path), summary=summary))
        root = Path(plan["canonical_root"])
        arguments = ["--source", summary["complete_source_inputs"], "--runtime-artifacts", plan["runtime_map"],
                     "--output-parent", root / "data/machine_runs/freeze_inputs"]
        for name in summary["adjudication_files"]:
            arguments.extend(["--adjudications", name])
        assembled = stage("assemble", "assemble_machine_freeze.py", arguments)
        inside(assembled["output"], root / "data/machine_runs/freeze_inputs")
        if assembled.get("records") != 1200:
            raise ValueError("Assembly did not preserve all1200 questions")
        frozen = stage("freeze", "freeze_machine_dataset.py", ["--source", assembled["output"],
            "--output-parent", root / "data/machine_runs/frozen_evaluation"])
        inside(frozen["output"], root / "data/machine_runs/frozen_evaluation")
        if frozen.get("formal_test_ready") is not True or frozen.get("human_gold") is not False:
            raise ValueError("Frozen TEST acceptance gate failed")
        common = ["--canonical-root", root, "--corpus-run", "semantic-page-v2-buffer", "run",
                  "--frozen-dir", frozen["output"], "--retrieval-run", plan["retrieval_run"]]
        gate = stage("retrieval_gate", "run_machine_experiments.py", common + ["--max-new-calls", 0])
        formal = inside(gate["output"], root / "data/machine_runs/formal-experiments")
        if (gate.get("status") != "partial_new_attempt_limit_reached"
                or gate.get("completed_retrieval_questions") != 1000 or gate.get("new_native_attempts_started") != 0):
            raise ValueError("Zero-call1000 retrieval gate failed")
        result = stage("answers", "run_machine_experiments.py", common + ["--resume", formal, "--max-new-calls", max_formal_contexts])
        if inside(result["output"], root / "data/machine_runs/formal-experiments") != formal:
            raise ValueError("Formal answers did not resume the same immutable run")
        if result.get("status") not in {"completed_formal_machine_only_experiment", "partial_new_attempt_limit_reached"}:
            raise ValueError("Formal experiment failed; no automatic retry")
        if result["status"] == "completed_formal_machine_only_experiment" and (
                result.get("completed_answer_questions") != 100 or result.get("completed_variant_answers") != 600):
            raise ValueError("Completed answer coverage differs from100 questions and600 variants")
        write(output / "result.json", result)
        return result
    except Exception as error:
        write(output / "failure.json", dict(error_type=type(error).__name__, error=str(error), automatic_retry=False,
            human_gold=False, subscription_dollar_cost=None))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("prepare")
    for name in ("canonical-root", "producer-run", "review-run"):
        before.add_argument("--" + name, type=Path, required=True)
    before.add_argument("--max-formal-contexts", type=int, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--max-formal-contexts", type=int, required=True)
    execute.add_argument("--max-wait-seconds", type=int, default=21600)
    execute.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args()
    if args.command == "prepare":
        directory, plan = prepare(args.canonical_root, args.producer_run, args.review_run, max_formal_contexts=args.max_formal_contexts)
        print(json.dumps(dict(output=str(directory), version=VERSION, native_calls=0, protected_files=len(plan["protected_sha256"]))))
    else:
        print(json.dumps(run(args.output, max_formal_contexts=args.max_formal_contexts,
            max_wait_seconds=args.max_wait_seconds, poll_seconds=args.poll_seconds), indent=2))
