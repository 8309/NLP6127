"""Synthetic scheduling fixtures only; no native call or actual data access."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest


@pytest.fixture
def scheduler(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("parallel_formal_fixture", scripts / "run_parallel_formal_jobs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.o, "call_structured_json", lambda *a, **k: pytest.fail("Native calls forbidden"))
    return module


def test_group_body_exactly_matches_pinned_original_loop(scheduler):
    source = Path(scheduler.o.__file__).read_text(encoding="utf-8")
    original = ast.parse(source)
    experiment = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "experiment")
    protected = next(node for node in experiment.body if isinstance(node, ast.Try))
    loop = next(node for node in protected.body if isinstance(node, ast.For))
    extracted = ast.parse(scheduler.group_source()).body[0]
    assert ast.dump(ast.Module(body=loop.body, type_ignores=[]), include_attributes=False) == ast.dump(
        ast.Module(body=extracted.body[1:-1], type_ignores=[]), include_attributes=False)
    assert scheduler.o.file_hash(scheduler.o.__file__) == scheduler.transport.DRIVER_SHA


def test_group_source_rejects_changed_original_pin(scheduler, monkeypatch):
    monkeypatch.setattr(scheduler.transport, "DRIVER_SHA", "0" * 64)
    with pytest.raises(ValueError, match="driver changed"):
        scheduler.group_source()


def test_prepare_body_exactly_matches_original_complete_preflight(scheduler):
    experiment = next(node for node in ast.parse(Path(scheduler.o.__file__).read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "experiment")
    budget_index = next(index for index, node in enumerate(experiment.body)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "budget" for target in node.targets))
    extracted = ast.parse(scheduler.prepare_source()).body[0]
    assert ast.dump(ast.Module(body=experiment.body[:budget_index], type_ignores=[]), include_attributes=False) == ast.dump(
        ast.Module(body=extracted.body[:-1], type_ignores=[]), include_attributes=False)


@pytest.mark.parametrize("maximum", [-1, 1.5, True, None])
def test_budget_rejects_invalid_maximum(scheduler, maximum):
    with pytest.raises(ValueError):
        scheduler.LockedBudget(maximum, threading.Event())


def test_locked_budget_parallel_reservations_do_not_exceed_limit(scheduler):
    budget = scheduler.LockedBudget(7, threading.Event())
    start = threading.Barrier(32)

    def reserve():
        start.wait(timeout=5)
        try:
            budget.reserve()
            return True
        except scheduler.o.SegmentLimit:
            return False

    with ThreadPoolExecutor(max_workers=32) as executor:
        completed = list(executor.map(lambda _: reserve(), range(32)))
    assert sum(completed) == 7
    assert budget.started == 7


def test_locked_budget_stops_before_reservation(scheduler):
    stopped = threading.Event()
    stopped.set()
    budget = scheduler.LockedBudget(1, stopped)
    with pytest.raises(scheduler.SchedulingStopped, match="stopped"):
        budget.reserve()
    assert budget.started == 0


@pytest.mark.parametrize("interval", [-1, float("nan"), float("inf"), -float("inf"), None, "20", True])
def test_start_gate_rejects_invalid_intervals(scheduler, interval):
    with pytest.raises(ValueError):
        scheduler.StartGate(interval, threading.Event())


def test_start_gate_preserves_global_spacing_without_real_wait(scheduler, monkeypatch):
    clock = [100.0]
    delays = []

    class Stop:
        def is_set(self):
            return False

        def wait(self, seconds):
            assert math.isfinite(seconds) and seconds >= 0
            delays.append(seconds)
            clock[0] += seconds
            return False

    monkeypatch.setattr(scheduler.time, "monotonic", lambda: clock[0])
    gate = scheduler.StartGate(20, Stop())
    gate.enter()
    clock[0] += 3
    gate.enter()
    clock[0] += 25
    gate.enter()
    assert delays == [0, 17, 0]
    assert gate.last == 145


def test_start_gate_stop_before_first_start(scheduler):
    stop = threading.Event()
    stop.set()
    with pytest.raises(scheduler.SchedulingStopped, match="stopped"):
        scheduler.StartGate(0, stop).enter()


def test_start_gate_stop_interrupts_waiting_start(scheduler):
    stop = threading.Event()
    gate = scheduler.StartGate(10, stop)
    gate.enter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(gate.enter)
        stop.set()
        with pytest.raises(scheduler.SchedulingStopped, match="stopped"):
            future.result(timeout=2)


def fake_job_args(scheduler, tmp_path):
    run = scheduler.o.long_path(tmp_path / "synthetic-run")
    cohort = SimpleNamespace(manifest=dict(version="synthetic", runtime_sha256={"fixture": "f" * 64}),
        cfg=dict(requested_model=None), protected={})
    request = dict(prompt="Synthetic", payload={"case": "fixture"}, role="judge",
        schema=dict(type="object", properties={}, required=[], additionalProperties=False))
    args = (run, "judge", ["synthetic-case"], request, None, lambda _: {}, cohort, None, None)
    key = dict(stage="judge", item_ids=["synthetic-case"], request=request,
        dataset_version=cohort.manifest["version"], runtime_sha256=cohort.manifest["runtime_sha256"])
    job_id = "judge-" + scheduler.o.protocol.sha256(scheduler.o.protocol.canonical_bytes(key))[:24]
    return args, key, job_id


def dispatcher(scheduler, driver, binding=None):
    stop = threading.Event()
    return scheduler.Dispatcher(driver, SimpleNamespace(caller=lambda *a, **k: pytest.fail("Native forbidden")),
        scheduler.LockedBudget(5, stop), stop, 0, binding or {})


def test_duplicate_immutable_job_owner_is_rejected(scheduler, tmp_path):
    args, _, job_id = fake_job_args(scheduler, tmp_path)
    entered, release = threading.Event(), threading.Event()

    def owned_job(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return {"synthetic": True}

    dispatch = dispatcher(scheduler, SimpleNamespace(job=owned_job))
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(dispatch.job, *args)
        assert entered.wait(2)
        with pytest.raises(ValueError, match="Concurrent owners"):
            dispatch.job(*args)
        assert job_id in dispatch.active
        release.set()
        assert first.result(timeout=2) == {"synthetic": True}
    assert dispatch.active == set()


def interrupted_fixture(scheduler, tmp_path, monkeypatch):
    args, key, job_id = fake_job_args(scheduler, tmp_path)
    run = args[0]
    directory = run / "jobs" / job_id
    request_path = "attempts/synthetic-interrupted/request.json"
    scheduler.o.write_new(directory / "job.json", key)
    scheduler.o.write_new(directory / request_path, args[3])
    monkeypatch.setattr(scheduler, "INTERRUPTED_JOB", job_id)
    monkeypatch.setattr(scheduler, "INTERRUPTED_REQUEST", request_path)
    monkeypatch.setattr(scheduler, "INTERRUPTED_DESCRIPTOR_SHA", scheduler.o.file_hash(directory / "job.json"))
    monkeypatch.setattr(scheduler, "INTERRUPTED_REQUEST_SHA", scheduler.o.file_hash(directory / request_path))
    return args, directory


def test_interrupted_retry_requires_exact_inventory_and_authorization(scheduler, tmp_path, monkeypatch):
    args, directory = interrupted_fixture(scheduler, tmp_path, monkeypatch)
    assert scheduler.interrupted_retry_binding(args[0], False) == {}
    assert len(scheduler.interrupted_retry_binding(args[0], True)) == 2
    scheduler.o.write_new(directory / "unexpected.txt", b"retained fixture")
    with pytest.raises(ValueError, match="inventory changed"):
        scheduler.interrupted_retry_binding(args[0], True)


def test_interrupted_retry_rejects_changed_pins(scheduler, tmp_path, monkeypatch):
    args, _ = interrupted_fixture(scheduler, tmp_path, monkeypatch)
    monkeypatch.setattr(scheduler, "INTERRUPTED_REQUEST_SHA", "0" * 64)
    with pytest.raises(ValueError, match="inventory changed"):
        scheduler.interrupted_retry_binding(args[0], True)


def test_exact_interrupted_retry_consumed_once_only(scheduler, tmp_path, monkeypatch):
    args, _ = interrupted_fixture(scheduler, tmp_path, monkeypatch)
    binding = scheduler.interrupted_retry_binding(args[0], True)
    seen = []
    dispatch = dispatcher(scheduler, SimpleNamespace(job=lambda *a, **k: seen.append(k["retry_failed"]) or {}), binding)
    dispatch.job(*args)
    assert seen == [True]
    assert dispatch.retry_used is True
    with pytest.raises(ValueError, match="already consumed"):
        dispatch.job(*args)
    assert seen == [True]
    assert dispatch.active == set()


def test_changed_interrupted_binding_does_not_leak_owner(scheduler, tmp_path, monkeypatch):
    args, directory = interrupted_fixture(scheduler, tmp_path, monkeypatch)
    binding = scheduler.interrupted_retry_binding(args[0], True)
    dispatch = dispatcher(scheduler, SimpleNamespace(job=lambda *a, **k: pytest.fail("No dispatch allowed")), binding)
    scheduler.o.write_new(directory / "new-native-artifact.txt", b"fixture")
    with pytest.raises(ValueError, match="inventory changed"):
        dispatch.job(*args)
    assert dispatch.active == set()
    assert dispatch.retry_used is False


def test_blanket_retry_is_forbidden(scheduler, tmp_path):
    args, _, _ = fake_job_args(scheduler, tmp_path)
    dispatch = dispatcher(scheduler, SimpleNamespace(job=lambda *a, **k: pytest.fail("No dispatch allowed")))
    with pytest.raises(ValueError, match="Blanket retry"):
        dispatch.job(*args, retry_failed=True)
    assert dispatch.active == set()


def test_pinned_native_binary_change_stops_before_caller(scheduler, tmp_path, monkeypatch):
    binary = tmp_path / "synthetic-cli.bin"
    binary.write_bytes(b"synthetic-only-not-executable")
    helper = SimpleNamespace(DEFAULT_CLI="synthetic", discover_native_cli=lambda _: binary)
    monkeypatch.setattr(scheduler.o, "_load_isolation_snapshot", lambda: (helper, {}))
    dispatch = dispatcher(scheduler, SimpleNamespace())
    dispatch.native_binding = "0" * 64
    with pytest.raises(ValueError, match="stop before invocation"):
        dispatch.caller("Synthetic", {}, {}, tmp_path, role="judge", requested_model=None)
    assert dispatch.stop.is_set()


def test_pinned_native_receipt_change_stops_after_retained_output(scheduler, tmp_path, monkeypatch):
    binary = tmp_path / "synthetic-cli.bin"
    binary.write_bytes(b"synthetic-only-not-executable")
    helper = SimpleNamespace(DEFAULT_CLI="synthetic", discover_native_cli=lambda _: binary)
    monkeypatch.setattr(scheduler.o, "_load_isolation_snapshot", lambda: (helper, {}))
    calls = []
    dispatch = dispatcher(scheduler, SimpleNamespace())
    dispatch.native_binding = scheduler.o.file_hash(binary)
    dispatch.admission = SimpleNamespace(caller=lambda *a, **k: calls.append(True) or ({}, {"native_cli_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="receipt differs"):
        dispatch.caller("Synthetic", {}, {}, tmp_path, role="judge", requested_model=None)
    assert calls == [True]
    assert dispatch.stop.is_set()


def test_pinned_native_binary_and_receipt_match(scheduler, tmp_path, monkeypatch):
    binary = tmp_path / "synthetic-cli.bin"
    binary.write_bytes(b"synthetic-only-not-executable")
    helper = SimpleNamespace(DEFAULT_CLI="synthetic", discover_native_cli=lambda _: binary)
    monkeypatch.setattr(scheduler.o, "_load_isolation_snapshot", lambda: (helper, {}))
    dispatch = dispatcher(scheduler, SimpleNamespace())
    dispatch.native_binding = scheduler.o.file_hash(binary)
    expected = ({"fixture": True}, {"native_cli_sha256": dispatch.native_binding, "execution_id": "synthetic-no-native"})
    dispatch.admission = SimpleNamespace(caller=lambda *a, **k: expected)
    assert dispatch.caller("Synthetic", {}, {}, tmp_path, role="judge", requested_model=None) == expected
    assert not dispatch.stop.is_set()


def test_unrelated_job_never_receives_retry_permission(scheduler, tmp_path):
    args, _, _ = fake_job_args(scheduler, tmp_path)
    seen = []
    dispatch = dispatcher(scheduler, SimpleNamespace(job=lambda *a, **k: seen.append(k["retry_failed"]) or {}), {"dummy": "pin"})
    dispatch.job(*args)
    assert seen == [False]
    assert dispatch.retry_used is False


def test_ordinary_failed_job_original_guard_retained(scheduler, tmp_path):
    args, key, job_id = fake_job_args(scheduler, tmp_path)
    scheduler.o.write_new(args[0] / "jobs" / job_id / "job.json", key)
    scheduler.o.write_new(args[0] / "jobs" / job_id / "attempts" / "prior" / "request.json", args[3])
    dispatch = dispatcher(scheduler, scheduler.o)
    with pytest.raises(ValueError, match="retry requires explicit"):
        dispatch.job(*args)
    assert dispatch.stop.is_set()
    assert dispatch.active == set()
    assert dispatch.budget.started == 0


def test_exclusive_run_rejects_second_owner_and_releases(scheduler, tmp_path):
    run = scheduler.o.long_path(tmp_path / "synthetic-run")
    with scheduler.exclusive_run(run):
        with pytest.raises(OSError):
            with scheduler.exclusive_run(run):
                pytest.fail("Second owner acquired exclusive lock")
    with scheduler.exclusive_run(run):
        assert (run / "parallel_scheduling" / "runner.lock").is_file()


def test_two_independent_groups_preserve_requests_batches_and_results(scheduler, tmp_path):
    o = scheduler.o
    ids = [f"synthetic-{index}" for index in range(4)]
    cases = {qid: dict(question="Synthetic oil amount?", as_of="2024-07-01") for qid in ids}
    evidence = {qid: dict(passage_id="passage-" + qid, document_id="document-" + qid,
        text="Synthetic production was 1 million barrels per day.", available_by="2024-06-01", rank=1, score=1.0)
        for qid in ids}
    refs = {qid: dict(label=dict(answerability="sufficient", qrels={evidence[qid]["passage_id"]: 2}),
        reference_evidence=[evidence[qid]]) for qid in ids}
    ranks = {qid: dict(conditions={condition: [evidence[qid]] for condition in o.CONDITIONS}) for qid in ids}
    cohort = SimpleNamespace(questions=cases, references=refs,
        cfg=dict(max_cases=2, max_request_bytes=160000, requested_model=None))
    timing = dict(run="synthetic-fixed-rankings")

    def execute_groups(target, parallel):
        calls, job_records = [], []
        record_lock = threading.Lock()

        def fake_caller(prompt, payload, schema, output, *, role):
            request = dict(prompt=prompt, payload=payload, schema=schema, role=role)
            with record_lock:
                calls.append(deepcopy(request))
            if role == "generator":
                answer = {qid: dict(answerability="sufficient", claims=[dict(text="Production was 1 million barrels per day.",
                    citations=[entry["evidence"][0]["passage_id"]])], quotes=[dict(passage_id=entry["evidence"][0]["passage_id"],
                    text=entry["evidence"][0]["text"])], limitations=[]) for qid, entry in payload.items()}
            elif role == "verifier":
                answer = {qid: dict(supported_indices=[0], reasons=["Synthetic supported"] ) for qid in payload}
            elif role == "judge":
                assert set(payload) == {"shared_cases", "answers"}
                answer = {key: dict(correctness=2, relevance=2, claim_supported=[True], citation_supported=[True],
                    context_relevant=[True], insufficiency_handling=True, reasons=["Synthetic fixture"])
                    for key in payload["answers"]}
            else:
                pytest.fail("Unexpected role")
            o.Draft202012Validator(schema).validate(answer)
            return answer, dict(execution_id="synthetic-no-native")

        budget = scheduler.LockedBudget(100, threading.Event())

        def fake_job(run, stage, selected, request, skipped, process, cohort, budget, caller, retry_failed=False):
            assert not retry_failed
            with record_lock:
                job_records.append(deepcopy(dict(stage=stage, ids=selected, request=request)))
            if request is None:
                return skipped
            budget.reserve()

            def replay(prompt, payload, schema, directory, *, role):
                assert dict(prompt=prompt, payload=payload, schema=schema, role=role) == request
                return caller(prompt, payload, schema, directory, role=role)

            return process(replay)

        namespace = dict(o.__dict__, job=fake_job)
        exec(compile(scheduler.group_source(), "<synthetic-group>", "exec"), namespace)
        group = namespace["original_group"]
        run = o.long_path(target)
        if parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(group, offset, ids, cohort, ranks, timing, run, budget, fake_caller)
                    for offset in (0, 2)]
                rows = [row for future in futures for row in future.result(timeout=5)]
        else:
            rows = [row for offset in (0, 2) for row in group(offset, ids, cohort, ranks, timing, run, budget, fake_caller)]
        order = lambda value: o.protocol.canonical_bytes(value)
        return rows, sorted(calls, key=order), sorted(job_records, key=order), budget.started, run

    sequential = execute_groups(tmp_path / "serial", False)
    parallel = execute_groups(tmp_path / "parallel", True)
    assert parallel[:4] == sequential[:4]
    assert len(parallel[0]) == 24
    assert {(row["record_id"], row["variant"]) for row in parallel[0]} == {
        (qid, variant) for qid in ids for variant in o.pilot.VARIANTS}
    assert parallel[3] == 24
    for filename in ("000000.json", "000002.json"):
        assert o.read_json(sequential[4] / "groups" / filename) == o.read_json(parallel[4] / "groups" / filename)
    assert all(len(job["ids"]) <= 2 for job in parallel[2])
