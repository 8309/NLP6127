"""Bounded parallel scheduling of unchanged frozen formal question groups.

Only scheduling changes: extract the exact pinned original group loop, retain
its batching, requests, stage processing and immutable jobs, then aggregate once
with the original serial experiment. No blanket failed-job retry or API fallback.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import subprocess
import textwrap
import threading
import time
from uuid import uuid4

import resume_formal_completed_503 as transport

o = transport.original
VERSION = "bounded-parallel-original-formal-groups-v1"
INTERRUPTED_JOB = "judge-350567f069be594f27a8621f"
INTERRUPTED_DESCRIPTOR_SHA = "345fd5b53a11e0f6d7d8334508372b1d3c35d61bdfa33a927d46af3cbadab49a"
INTERRUPTED_REQUEST = "attempts/2026-10-14T07-59-39.922143+00-00-76c1749e/request.json"
INTERRUPTED_REQUEST_SHA = "6cb4bbc850cd9cff45319ded89175e4470fcbb80bf0debf2effc91ee4416f887"
REVIEWED_RETRIES_VERSION = "explicit-reviewed-parallel-retries-v1"


class SchedulingStopped(RuntimeError):
    pass


class LockedBudget:
    def __init__(self, maximum, stop):
        self.inner = o.ExecutionBudget(maximum)
        self.stop, self.lock = stop, threading.Lock()

    @property
    def started(self):
        with self.lock:
            return self.inner.started

    def reserve(self):
        with self.lock:
            if self.stop.is_set():
                raise SchedulingStopped("A prior failure stopped new native attempts")
            self.inner.reserve()


class StartGate:
    """At most two workers; stagger native starts without hiding retries."""
    def __init__(self, interval, stop):
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval < 0:
            raise ValueError("Start interval must be nonnegative")
        self.interval, self.stop = interval, stop
        self.lock, self.last = threading.Lock(), None

    def enter(self):
        with self.lock:
            if self.stop.is_set():
                raise SchedulingStopped("A prior failure stopped new native starts")
            delay = 0 if self.last is None else max(0, self.interval - (time.monotonic() - self.last))
            if self.stop.wait(delay):
                raise SchedulingStopped("A prior failure stopped a waiting native start")
            self.last = time.monotonic()


@contextmanager
def exclusive_run(run):
    """OS lock releases on interruption; prevents two owners of this scheduler."""
    path = run / "parallel_scheduling" / "runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def group_source():
    source = Path(o.__file__).read_text(encoding="utf-8")
    if o.file_hash(o.__file__) != transport.DRIVER_SHA:
        raise ValueError("Original formal driver changed")
    begin = "            group_ids = ids[offset:offset+cohort.cfg[\"max_cases\"]]\n"
    end = "        status = \"completed_formal_machine_only_experiment\"\n"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError("Pinned original group boundaries are not unique")
    body = source.split(begin, 1)[1].split(end, 1)[0]
    body = textwrap.dedent(begin + body)
    return ("def original_group(offset, ids, cohort, rankings, timing, run, budget, caller, retry_failed=False):\n"
            "    all_rows = []\n" + textwrap.indent(body, "    ") + "    return all_rows\n")


def prepare_source():
    """Use the exact original preflight, including its complete plan equality."""
    source = Path(o.__file__).read_text(encoding="utf-8")
    if o.file_hash(o.__file__) != transport.DRIVER_SHA:
        raise ValueError("Original formal driver changed")
    experiment = source.split("def experiment(", 1)[1]
    begin = "    started = perf_counter()\n"
    end = "    budget = ExecutionBudget(max_new_calls)\n"
    if experiment.count(begin) != 1 or experiment.count(end) != 1:
        raise ValueError("Pinned original preparation boundaries are not unique")
    body = begin + experiment.split(begin, 1)[1].split(end, 1)[0]
    return ("def original_prepare(runtime, frozen, retrieval_run, output_parent, resume):\n" + body
            + "    return run, cohort, ids, rankings, timing\n")


def interrupted_retry_binding(run, authorized):
    if not authorized:
        return {}
    directory = run / "jobs" / INTERRUPTED_JOB
    expected = {"job.json": INTERRUPTED_DESCRIPTOR_SHA, INTERRUPTED_REQUEST: INTERRUPTED_REQUEST_SHA}
    actual = {p.relative_to(directory).as_posix(): o.file_hash(p) for p in directory.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Exact interrupted job inventory changed; no automatic retry")
    return expected


class ReviewedRetries:
    """One user-reviewed retry per hash-bound job; never a blanket retry."""
    def __init__(self, run, path, expected_sha256, native_sha256, rollover_policy_sha256):
        self.run, self.path = o.long_path(run), o.long_path(path)
        parent = o.plain_path(self.run / "parallel_scheduling")
        if not o.plain_path(self.path).is_relative_to(parent) or o.plain_path(self.path) == parent:
            raise ValueError("Reviewed retry manifest must belong to this run's parallel_scheduling directory")
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("Reviewed retry manifest requires an exact SHA-256")
        self.sha256 = expected_sha256
        self.verify()
        self.manifest = o.read_json(self.path)
        authorization = self.manifest.get("user_authorization")
        if (self.manifest.get("schema_version") != REVIEWED_RETRIES_VERSION
                or self.manifest.get("native_cli_sha256") != native_sha256
                or not rollover_policy_sha256
                or self.manifest.get("cli_rollover_policy_sha256") != rollover_policy_sha256
                or type(self.manifest.get("max_retries_per_job")) is not int
                or self.manifest["max_retries_per_job"] != 1
                or not isinstance(authorization, dict)
                or authorization.get("directly_confirmed") is not True
                or not isinstance(authorization.get("scope"), str) or not authorization["scope"].strip()):
            raise ValueError("Reviewed retry manifest metadata differs from the exact user-approved continuation")
        self.allowed_jobs = self.manifest.get("allowed_jobs")
        if not isinstance(self.allowed_jobs, dict) or not self.allowed_jobs:
            raise ValueError("Reviewed retry manifest must name explicit jobs")
        for job_id, entry in self.allowed_jobs.items():
            if (not isinstance(job_id, str) or not re.fullmatch(r"[a-z][a-z_]*-[0-9a-f]{24}", job_id)
                    or not isinstance(entry, dict) or set(entry) != {"original_artifacts_sha256"}):
                raise ValueError("Reviewed retry manifest contains an invalid immutable job binding")
            hashes = entry["original_artifacts_sha256"]
            if (not isinstance(hashes, dict) or "job.json" not in hashes
                    or not any(name.startswith("attempts/") and name.endswith("/request.json") for name in hashes)):
                raise ValueError("Reviewed retry must preserve its descriptor and original attempt request")
            directory = self.run / "jobs" / job_id
            for name, digest in hashes.items():
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Invalid reviewed retry artifact SHA-256")
                if o.file_hash(o.contained(directory, name)) != digest:
                    raise ValueError("Reviewed retry original artifact changed")
            descriptor = o.read_json(directory / "job.json")
            expected_id = descriptor["stage"] + "-" + o.protocol.sha256(o.protocol.canonical_bytes(descriptor))[:24]
            if expected_id != job_id:
                raise ValueError("Reviewed retry owning descriptor differs from its immutable job ID")
            if not (directory / "completed.json").exists():
                self.validate_job(job_id, descriptor)

    def verify(self):
        if o.file_hash(self.path) != self.sha256:
            raise ValueError("Reviewed retry manifest changed")

    def validate_job(self, job_id, key):
        self.verify()
        entry = self.allowed_jobs[job_id]
        directory = self.run / "jobs" / job_id
        if o.read_json(directory / "job.json") != key:
            raise ValueError("Reviewed retry owning job descriptor changed")
        expected = entry["original_artifacts_sha256"]
        actual = {path.relative_to(directory).as_posix():o.file_hash(path)
                  for path in directory.rglob("*") if path.is_file()}
        if actual != expected:
            raise ValueError("Reviewed retry entire original artifact inventory changed")


class Dispatcher:
    def __init__(self, driver, admission, budget, stop, interval, retry_binding, native_binding=None, reviewed_retries=None):
        self.driver, self.admission, self.budget, self.stop = driver, admission, budget, stop
        self.gate = StartGate(interval, stop)
        self.retry_binding = retry_binding
        self.native_binding = native_binding
        self.reviewed_retries = reviewed_retries
        self.used_reviewed_retries = set()
        self.lock, self.active, self.retry_used = threading.Lock(), set(), False

    def caller(self, *args, **kwargs):
        self.gate.enter()
        print("Native start: " + str(args[3]), flush=True)
        try:
            if self.reviewed_retries is not None:
                self.reviewed_retries.verify()
            if self.native_binding is not None:
                helper, _ = o._load_isolation_snapshot()
                if o.file_hash(helper.discover_native_cli(helper.DEFAULT_CLI)) != self.native_binding:
                    raise ValueError("Current continuation native binary changed; stop before invocation")
            result = self.admission.caller(*args, **kwargs)
            if self.native_binding is not None and result[1].get("native_cli_sha256") != self.native_binding:
                raise ValueError("Native receipt differs from this continuation's pinned binary")
            print("Native completed: " + result[1]["execution_id"], flush=True)
            return result
        except Exception:
            self.stop.set()
            raise

    def job(self, run, stage, ids, request, no_call_result, process, cohort, budget, caller, retry_failed=False):
        if retry_failed:
            raise ValueError("Blanket retry is forbidden")
        key = dict(stage=stage, item_ids=ids, request=request, dataset_version=cohort.manifest["version"],
                   runtime_sha256=cohort.manifest["runtime_sha256"])
        job_id = stage + "-" + o.protocol.sha256(o.protocol.canonical_bytes(key))[:24]
        retry = False
        with self.lock:
            if job_id in self.active:
                raise ValueError("Concurrent owners of one immutable job")
            if self.reviewed_retries is not None:
                try:
                    self.reviewed_retries.verify()
                    if job_id in self.reviewed_retries.allowed_jobs and not (run / "jobs" / job_id / "completed.json").exists():
                        if job_id in self.used_reviewed_retries:
                            raise ValueError("Reviewed immutable job retry already consumed")
                        self.reviewed_retries.validate_job(job_id, key)
                        self.used_reviewed_retries.add(job_id)
                        retry = True
                except Exception:
                    self.stop.set()
                    raise
            if job_id == INTERRUPTED_JOB and self.retry_binding and not (run / "jobs" / job_id / "completed.json").exists():
                if self.retry_used:
                    raise ValueError("Reviewed interrupted job retry already consumed")
                try:
                    interrupted_retry_binding(run, True)
                except Exception:
                    self.stop.set()
                    raise
                self.retry_used, retry = True, True
            self.active.add(job_id)
        try:
            return self.driver.job(run, stage, ids, request, no_call_result, process, cohort,
                                   self.budget, self.caller, retry_failed=retry)
        except Exception:
            self.stop.set()
            raise
        finally:
            with self.lock:
                self.active.remove(job_id)


def execute(args):
    if args.workers != 2:
        raise ValueError("This reviewed continuation is bounded to exactly two workers")
    runtime = o.load_staged_runtime(args.canonical_root, "semantic-page-v2-buffer")
    preparation = prepare_source()
    preparation_namespace = dict(o.__dict__)
    exec(compile(preparation, "<pinned-original-formal-preflight>", "exec"), preparation_namespace)
    run, cohort, ids, rankings, timing = preparation_namespace["original_prepare"](
        runtime, args.frozen_dir, args.retrieval_run,
        runtime.canonical_root / "data/machine_runs/formal-experiments", args.resume)
    helper, _ = o._load_isolation_snapshot()
    native = helper.discover_native_cli(helper.DEFAULT_CLI)
    if o.file_hash(native) != args.native_cli_sha256:
        raise ValueError("Native CLI differs from the explicitly pinned continuation binary")
    native_version = subprocess.run([str(native), "--version"], capture_output=True, check=True).stdout.decode("utf-8").strip()
    if native_version != transport.NATIVE_CLI_VERSION:
        raise ValueError("Native CLI version changed; original execution interface not established")
    admission = transport.Admission(run, cohort)
    driver = transport.driver(admission)
    rollover = None
    if getattr(args, "initialize_approved_cli_rollover", False) or getattr(args, "cli_rollover_policy_sha256", None):
        from formal_cli_rollover import ReviewedRollover
        rollover = ReviewedRollover(run, admission,
            initialize_approved=getattr(args, "initialize_approved_cli_rollover", False),
            expected_policy_sha256=getattr(args, "cli_rollover_policy_sha256", None))
        driver = rollover.driver()
    # Fail before any new invocation if the saved run's execution identity is
    # incompatible. The explicit amendment is the only permitted exception.
    driver.usage_audit(run)
    stop = threading.Event()
    budget = LockedBudget(args.max_new_calls, stop)
    binding = interrupted_retry_binding(run, args.retry_exact_interrupted_job)
    reviewed_path = getattr(args, "reviewed_retry_manifest", None)
    reviewed_sha256 = getattr(args, "reviewed_retry_manifest_sha256", None)
    if bool(reviewed_path) != bool(reviewed_sha256):
        raise ValueError("Reviewed retry manifest and SHA-256 must be supplied together")
    reviewed = ReviewedRetries(run, reviewed_path, reviewed_sha256, args.native_cli_sha256,
        o.file_hash(rollover.policy_path) if rollover else None) if reviewed_path else None
    dispatcher = Dispatcher(driver, admission, budget, stop, args.start_interval_seconds, binding,
        args.native_cli_sha256, reviewed)
    generated = group_source()
    namespace = dict(driver.__dict__, job=dispatcher.job)
    exec(compile(generated, "<pinned-original-group-scheduling>", "exec"), namespace)
    group = namespace["original_group"]
    record = dict(version=VERSION, workers=2, global_native_start_interval_seconds=args.start_interval_seconds,
        scheduler_sha256=o.file_hash(__file__), original_driver_sha256=transport.DRIVER_SHA,
        extracted_group_sha256=o.protocol.sha256(generated.encode()),
        extracted_preflight_sha256=o.protocol.sha256(preparation.encode()),
        transport_policy_sha256=o.file_hash(admission.policy_path),
        max_new_calls=args.max_new_calls, frozen_inputs_unchanged=True, human_gold=False,
        retry_exact_interrupted_job=INTERRUPTED_JOB if binding else None, interrupted_original_sha256=binding,
        reviewed_retry_manifest=str(reviewed.path) if reviewed else None,
        reviewed_retry_manifest_sha256=reviewed.sha256 if reviewed else None,
        reviewed_retry_allowed_job_ids=sorted(reviewed.allowed_jobs) if reviewed else [],
        reviewed_retry_maximum_per_job=1 if reviewed else None,
        prior_interrupted_execution_usage="unknown_not_zero", failed_job_automatic_retry=False,
        user_authorization=reviewed.manifest["user_authorization"]["scope"] if reviewed else
            "Continue in parallel; resume the exact interrupted request once",
        native_cli_sha256=args.native_cli_sha256, native_cli_version=native_version,
        historical_native_cli_sha256=transport.NATIVE_CLI_SHA,
        updated_binary_is_distinct_execution_lineage=True,
        completed503_admission_still_requires_historical_binary=True,
        cli_rollover_policy_sha256=o.file_hash(rollover.policy_path) if rollover else None,
        cli_rollover_explicitly_approved=rollover is not None,
        requested_model=cohort.cfg["requested_model"], changed_prompt_schema_sample_or_scores=False)
    checkpoint = run / "parallel_scheduling" / (o.now().replace(":", "-") + "-" + uuid4().hex[:8])
    o.write_new(checkpoint / "policy.json", record)
    failures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(group, offset, ids, cohort, rankings, timing, run, budget, dispatcher.caller): offset
                   for offset in range(0, len(ids), cohort.cfg["max_cases"])}
        for future in as_completed(futures):
            offset = futures[future]
            try:
                rows = future.result()
                print(f"Parallel group saved: offset={offset} questions={len(rows)//6}", flush=True)
            except Exception as error:
                stop.set()
                failures.append(dict(offset=offset, error_type=type(error).__name__, error=str(error)))
                for pending in futures:
                    pending.cancel()
    o._assert_hashes(admission.guards)
    o.write_new(checkpoint / "completion.json", dict(policy_sha256=o.file_hash(checkpoint / "policy.json"),
        started_native_attempts=budget.started, interrupted_retry_used=dispatcher.retry_used,
        used_reviewed_retry_job_ids=sorted(dispatcher.used_reviewed_retries),
        reviewed_retry_jobs_consumed=len(dispatcher.used_reviewed_retries),
        failures=failures, completed_groups_sha256={p.name:o.file_hash(p) for p in sorted((run / "groups").glob("*.json"))},
        stopped_on_error=bool(failures), pending_inflight_drained=True, automatic_retry=False))
    if rollover:
        o.write_new(checkpoint / "cli_rollover_usage.json", rollover.lineage_report())
    if failures:
        print(json.dumps(dict(status="parallel_stopped_not_complete", checkpoint=str(checkpoint), failures=failures), indent=2))
        return 1
    _, result = driver.experiment(runtime, args.frozen_dir, args.retrieval_run,
        runtime.canonical_root / "data/machine_runs/formal-experiments", resume=run,
        max_new_calls=0, retry_failed=False, caller=lambda *a, **k: (_ for _ in ()).throw(ValueError("Aggregation forbids new native calls")))
    print(json.dumps(dict(output=str(run), **result), indent=2))
    return 0 if result["status"] == "completed_formal_machine_only_experiment" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-root", "frozen-dir", "retrieval-run", "resume"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start-interval-seconds", type=float, default=20)
    parser.add_argument("--max-new-calls", type=int, default=1200)
    parser.add_argument("--native-cli-sha256", required=True)
    parser.add_argument("--initialize-approved-cli-rollover", action="store_true")
    parser.add_argument("--cli-rollover-policy-sha256")
    parser.add_argument("--retry-exact-interrupted-job", action="store_true")
    parser.add_argument("--reviewed-retry-manifest", type=Path)
    parser.add_argument("--reviewed-retry-manifest-sha256")
    args = parser.parse_args()
    with exclusive_run(o.long_path(args.resume)):
        return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
