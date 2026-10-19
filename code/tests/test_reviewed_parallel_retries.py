"""Synthetic explicit-retry fixtures; no native calls or actual run access."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("reviewed_parallel_retry_fixture", scripts / "run_parallel_formal_jobs.py")
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    monkeypatch.setattr(scheduler.o, "call_structured_json", lambda *a, **k: pytest.fail("Native calls forbidden"))
    return scheduler


@pytest.fixture
def case(module, tmp_path):
    o = module.o
    run = o.long_path(tmp_path / "synthetic-run")
    native, policy = "e" * 64, "b" * 64
    cohort = SimpleNamespace(manifest=dict(version="synthetic", runtime_sha256={"fixture": "f" * 64}),
        cfg=dict(requested_model=None), protected={})
    args, bindings = {}, {}
    for qid in ("synthetic-timeout", "synthetic-routing"):
        request = dict(prompt="Synthetic judge only", payload={qid: "fixture"}, role="judge",
            schema=dict(type="object", properties={}, required=[], additionalProperties=False))
        key = dict(stage="judge", item_ids=[qid], request=request, dataset_version=cohort.manifest["version"],
            runtime_sha256=cohort.manifest["runtime_sha256"])
        job_id = "judge-" + o.protocol.sha256(o.protocol.canonical_bytes(key))[:24]
        directory = run / "jobs" / job_id
        o.write_new(directory / "job.json", key)
        o.write_new(directory / "attempts/prior/request.json", request)
        o.write_new(directory / "attempts/prior/runner_failure.json", dict(automatic_retry=False, error_type="SyntheticFailure"))
        if qid.endswith("routing"):
            o.write_new(directory / "attempts/prior/calls/call/receipt.json", dict(accepted=False, usage=None,
                execution_id="synthetic-routing", native_cli_sha256=native))
        else:
            o.write_new(directory / "attempts/prior/calls/call/events.jsonl", b'{"type":"synthetic-timeout"}\n')
        bindings[job_id] = dict(original_artifacts_sha256={p.relative_to(directory).as_posix():o.file_hash(p)
            for p in directory.rglob("*") if p.is_file()})
        args[job_id] = (run, "judge", [qid], request, None, lambda _: {}, cohort, None, None)
    manifest = dict(schema_version=module.REVIEWED_RETRIES_VERSION, allowed_jobs=bindings,
        native_cli_sha256=native, cli_rollover_policy_sha256=policy, max_retries_per_job=1,
        user_authorization=dict(directly_confirmed=True, scope="Retry each of these two exact synthetic requests once"))
    path = run / "parallel_scheduling/reviewed-retries.json"
    o.write_new(path, manifest)
    return SimpleNamespace(run=run, path=path, manifest=manifest, native=native, policy=policy, args=args,
        sha=o.file_hash(path))


def binding(module, case):
    return module.ReviewedRetries(case.run, case.path, case.sha, case.native, case.policy)


def make_dispatcher(module, reviewed, fake_job):
    stop = threading.Event()
    return module.Dispatcher(SimpleNamespace(job=fake_job),
        SimpleNamespace(caller=lambda *a, **k: pytest.fail("Native calls forbidden")),
        module.LockedBudget(10, stop), stop, 0, {}, reviewed_retries=reviewed)


def test_exact_manifest_binds_both_whole_failed_inventories_read_only(module, case):
    before = {str(p):p.read_bytes() for p in case.run.rglob("*") if p.is_file()}
    reviewed = binding(module, case)
    assert set(reviewed.allowed_jobs) == set(case.args)
    assert reviewed.sha256 == case.sha
    assert before == {str(p):p.read_bytes() for p in case.run.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field,value", [("schema_version", "different"), ("native_cli_sha256", "0" * 64),
    ("cli_rollover_policy_sha256", "0" * 64), ("max_retries_per_job", 2), ("max_retries_per_job", True),
    ("user_authorization", {"directly_confirmed": False, "scope": "fixture"})])
def test_manifest_metadata_mismatch_rejected(module, case, field, value):
    manifest = deepcopy(case.manifest)
    manifest[field] = value
    case.path.write_bytes(module.o.protocol.canonical_bytes(manifest))
    case.sha = module.o.file_hash(case.path)
    with pytest.raises(ValueError, match="metadata differs"):
        binding(module, case)


def test_manifest_hash_mutation_rejected(module, case):
    case.path.write_bytes(case.path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest changed"):
        binding(module, case)


def test_manifest_outside_exact_run_rejected(module, case, tmp_path):
    outside = tmp_path / "outside.json"
    module.o.write_new(outside, case.manifest)
    with pytest.raises(ValueError, match="parallel_scheduling directory"):
        module.ReviewedRetries(case.run, outside, module.o.file_hash(outside), case.native, case.policy)


@pytest.mark.parametrize("kind", ["added", "changed", "missing"])
def test_whole_original_inventory_changes_rejected(module, case, kind):
    job_id = next(iter(case.args))
    directory = case.run / "jobs" / job_id
    path = directory / "attempts/prior/request.json"
    if kind == "added":
        module.o.write_new(directory / "unexpected.txt", b"synthetic")
    elif kind == "changed":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path.unlink()
    with pytest.raises(ValueError):
        binding(module, case)


def test_each_explicit_job_receives_one_retry_with_two_workers(module, case):
    reviewed = binding(module, case)
    seen, lock = [], threading.Lock()

    def fake_job(*args, **kwargs):
        with lock:
            seen.append((args[2][0], kwargs["retry_failed"]))
        return {"fixture": True}

    dispatch = make_dispatcher(module, reviewed, fake_job)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(dispatch.job, *args) for args in case.args.values()]
        assert all(future.result(timeout=2) == {"fixture": True} for future in futures)
    assert len(seen) == 2 and all(retry for _, retry in seen)
    assert dispatch.used_reviewed_retries == set(case.args)
    with pytest.raises(ValueError, match="already consumed"):
        dispatch.job(*next(iter(case.args.values())))
    assert len(seen) == 2 and dispatch.active == set()


def test_completed_allowed_job_reused_without_retry_permission(module, case):
    job_id = next(iter(case.args))
    module.o.write_new(case.run / "jobs" / job_id / "completed.json", dict(synthetic_completed=True))
    reviewed = binding(module, case)
    seen = []
    dispatch = make_dispatcher(module, reviewed, lambda *a, **k: seen.append(k["retry_failed"]) or {})
    dispatch.job(*case.args[job_id])
    assert seen == [False] and dispatch.used_reviewed_retries == set()


def test_unlisted_failed_job_keeps_original_no_retry_guard(module, case):
    reviewed = binding(module, case)
    args = list(next(iter(case.args.values())))
    args[2] = ["unlisted-synthetic"]
    key = dict(stage="judge", item_ids=args[2], request=args[3], dataset_version=args[6].manifest["version"],
        runtime_sha256=args[6].manifest["runtime_sha256"])
    job_id = "judge-" + module.o.protocol.sha256(module.o.protocol.canonical_bytes(key))[:24]
    module.o.write_new(case.run / "jobs" / job_id / "job.json", key)
    module.o.write_new(case.run / "jobs" / job_id / "attempts/prior/request.json", args[3])
    dispatch = make_dispatcher(module, reviewed, module.o.job)
    with pytest.raises(ValueError, match="retry requires explicit"):
        dispatch.job(*args)
    assert dispatch.used_reviewed_retries == set() and dispatch.stop.is_set()


def test_manifest_mutation_after_loading_stops_before_job_dispatch(module, case):
    dispatch = make_dispatcher(module, binding(module, case), lambda *a, **k: pytest.fail("Dispatch forbidden"))
    case.path.write_bytes(case.path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest changed"):
        dispatch.job(*next(iter(case.args.values())))
    assert dispatch.stop.is_set() and dispatch.active == set() and not dispatch.used_reviewed_retries


def test_artifact_mutation_after_loading_stops_before_once_retry(module, case):
    dispatch = make_dispatcher(module, binding(module, case), lambda *a, **k: pytest.fail("Dispatch forbidden"))
    job_id = next(iter(case.args))
    module.o.write_new(case.run / "jobs" / job_id / "unexpected.txt", b"synthetic")
    with pytest.raises(ValueError, match="inventory changed"):
        dispatch.job(*case.args[job_id])
    assert dispatch.stop.is_set() and dispatch.active == set() and not dispatch.used_reviewed_retries


def test_changed_descriptor_request_never_receives_retry(module, case):
    dispatch = make_dispatcher(module, binding(module, case), lambda *a, **k: pytest.fail("Dispatch forbidden"))
    args = list(next(iter(case.args.values())))
    args[3] = deepcopy(args[3])
    args[3]["prompt"] = "Changed synthetic request"
    seen = []
    dispatch.driver = SimpleNamespace(job=lambda *a, **k: seen.append(k["retry_failed"]) or {})
    dispatch.job(*args)
    assert seen == [False] and not dispatch.used_reviewed_retries
