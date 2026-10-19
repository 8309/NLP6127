"""Synthetic fail-closed offline-auditor fixtures; no actual model calls."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import audit_formal_results as module
from test_machine_experiments import cohort_fixture, synthetic_caller


@pytest.fixture(scope="module")
def completed_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("formal-audit-synthetic")
    runtime, frozen, retrieval = cohort_fixture(root, max_cases=32)
    caller, calls = synthetic_caller(fail_first=True)
    parent = runtime.canonical_root / "data/machine_runs/formal-experiments"
    run, failed = module.driver.experiment(runtime, frozen, retrieval, parent, max_new_calls=1, caller=caller)
    assert failed["status"] == "failed_not_complete"
    run, result = module.driver.experiment(runtime, frozen, retrieval, parent, resume=run,
                                            max_new_calls=2000, retry_failed=True, caller=caller)
    assert result["status"] == "completed_formal_machine_only_experiment"
    return runtime, frozen, retrieval, run, result, calls


def test_complete_synthetic_cohort_replays_without_model_or_file_writes(completed_fixture, monkeypatch):
    runtime, frozen, retrieval, run, result, calls = completed_fixture
    before = {str(path): module.driver.file_hash(path) for path in run.rglob("*") if path.is_file()}
    calls_before = len(calls)
    monkeypatch.setattr(module.driver, "call_structured_json", lambda *a, **k: pytest.fail("Offline audit cannot call models"))
    audited = module.audit(runtime, frozen, retrieval, run)
    after = {str(path): module.driver.file_hash(path) for path in run.rglob("*") if path.is_file()}
    assert before == after and len(calls) == calls_before
    assert audited["status"] == "completed_formal_result_offline_audit_passed"
    assert audited["variant_answers"] == 600 and audited["ranked_query_conditions"] == 6000
    assert audited["human_gold"] is False and audited["native_calls"] == 0
    assert audited["usage"] == result["usage"]
    assert len(audited["usage"]["interrupted_or_precall_attempts_with_unknown_execution"]) == 1
    assert audited["retained_native_attempts"] == audited["usage"]["known_native_execution_count"] + 1


@pytest.mark.parametrize("field,value", [("status", "partial_new_attempt_limit_reached"),
    ("formal_heldout_test", False), ("human_gold", True), ("completed_variant_answers", 599),
    ("completed_answer_questions", 99), ("completed_retrieval_questions", 999),
    ("answer_sample_plan_sha256", "changed"), ("original_frozen_inputs_unchanged", False)])
def test_partial_or_inflated_result_flags_rejected(completed_fixture, field, value):
    *_, run, original, _calls = completed_fixture
    result = deepcopy(original)
    result[field] = value
    with pytest.raises(ValueError):
        module.validate_result_flags(result, module.driver.read_json(run / "plan.json"))


def group_fixture(completed_fixture, tmp_path):
    *_, run, original, _calls = completed_fixture
    target = tmp_path / "run"
    shutil.copytree(run / "groups", target / "groups")
    return target, module.driver.read_json(run / "plan.json"), deepcopy(original)


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate_row", "wrong_variant", "changed_hash", "wrong_group_order"])
def test_group_coverage_and_hash_bindings_fail_closed(completed_fixture, tmp_path, kind):
    run, plan, result = group_fixture(completed_fixture, tmp_path)
    name = sorted(result["group_artifacts_sha256"])[0]
    path = run / name
    if kind == "missing":
        path.unlink()
    elif kind == "extra":
        (run / "groups/extra.json").write_bytes(path.read_bytes())
    elif kind == "changed_hash":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        group = module.driver.read_json(path)
        if kind == "duplicate_row":
            group["rows"][1] = deepcopy(group["rows"][0])
        elif kind == "wrong_variant":
            group["rows"][0]["variant"] = "not-a-variant"
        else:
            group["record_ids"].reverse()
        path.write_bytes(module.driver.protocol.canonical_bytes(group))
        result["group_artifacts_sha256"][name] = module.driver.file_hash(path)
    with pytest.raises(ValueError):
        module.audit_groups(run, plan, result)


def test_summary_and_bootstrap_recomputed_exactly(completed_fixture):
    *_, run, result, _calls = completed_fixture
    plan = module.driver.read_json(run / "plan.json")
    rows = module.audit_groups(run, plan, result)
    summary, paired = module.recompute_quality(rows, plan["answer_record_ids"])
    assert summary == result["summary"] and paired == result["paired_automatic_correctness"]
    rows[0]["automatic_judge"]["correctness"] = 0
    changed, changed_paired = module.recompute_quality(rows, plan["answer_record_ids"])
    assert changed != summary and changed_paired != paired


def test_no_completed_result_cannot_be_promoted(tmp_path):
    from types import SimpleNamespace
    runtime = SimpleNamespace(canonical_root=tmp_path)
    run = tmp_path / "data/machine_runs/formal-experiments/partial"
    run.mkdir(parents=True)
    with pytest.raises(ValueError, match="No completed result"):
        module.audit(runtime, tmp_path / "frozen", tmp_path / "retrieval", run)


def test_replay_invalid_stage_cannot_write_rejection(tmp_path):
    path = tmp_path / "must-not-exist.json"
    original = module.driver.pilot.write_new
    with pytest.raises(ValueError, match="writes are forbidden"):
        with module.read_only_stage_replay():
            module.driver.pilot.write_new(path, {"invalid": True})
    assert not path.exists() and module.driver.pilot.write_new is original


def test_false_operational_receipt_not_admitted_by_default(completed_fixture, tmp_path):
    from types import SimpleNamespace
    *_, source_run, _result, _calls = completed_fixture
    source = next(path for path in (source_run / "jobs").glob("*") if (path / "completed.json").exists())
    run = tmp_path / "run"
    directory = run / "jobs" / source.name
    shutil.copytree(source, directory)
    descriptor = module.driver.read_json(directory / "job.json")
    completed = module.driver.read_json(directory / "completed.json")
    receipt_path = next((directory / "attempts").glob("*/calls/*/receipt.json"))
    receipt = module.driver.read_json(receipt_path)
    receipt.update(accepted=False, artifact_directory=str(receipt_path.parent))
    receipt_path.write_bytes(module.driver.protocol.canonical_bytes(receipt))
    failure = receipt_path.parents[2] / "runner_failure.json"
    failure.write_bytes(module.driver.protocol.canonical_bytes(dict(automatic_retry=False, error="Synthetic operational failure")))
    completed["artifacts_sha256"][receipt_path.relative_to(directory).as_posix()] = module.driver.file_hash(receipt_path)
    (directory / "completed.json").write_bytes(module.driver.protocol.canonical_bytes(completed))
    cohort = SimpleNamespace(manifest=dict(version=descriptor["dataset_version"], runtime_sha256=descriptor["runtime_sha256"]))
    jobs = module.OfflineJobs(run, cohort)
    with pytest.raises(ValueError, match="one accepted or explicitly certified"):
        jobs.replay(descriptor["stage"], descriptor["item_ids"], descriptor["request"], None,
                    lambda caller: pytest.fail("A false operational receipt must not reach stage replay"))


def test_transport_opt_in_requires_both_pins_and_existing_policy(tmp_path):
    with pytest.raises(ValueError, match="both explicit"):
        module.load_transport_admission(tmp_path, None, policy_sha256="a" * 64)
    with pytest.raises(ValueError, match="exact preexisting policy"):
        module.load_transport_admission(tmp_path, None, policy_sha256="a" * 64, adapter_sha256="b" * 64)
    assert not (tmp_path / "transport_admissions").exists()


@pytest.mark.parametrize("value", ["missing-separator", "x=" + "z" * 64, "x=" + "A" * 64, "=" + "a" * 64])
def test_retained_receipt_pin_format_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        module.parse_retained_pin(value)


def test_new_audit_output_rejects_experiment_target_and_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(module.driver, "ROOT", tmp_path / "public")
    canonical = tmp_path / "canonical"
    with pytest.raises(ValueError, match="new ignored"):
        module.write_optional_new_output(canonical / "data/machine_runs/formal-experiments/x/result.json", {}, canonical)
    new = canonical / "data/machine_runs/engineering-checkpoints/offline-audit/audit.json"
    module.write_optional_new_output(new, {"synthetic_fixture": True}, canonical)
    assert module.driver.read_json(new) == {"synthetic_fixture": True}
    with pytest.raises(ValueError, match="Never overwrite"):
        module.write_optional_new_output(new, {}, canonical)
