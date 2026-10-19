"""Synthetic amended-auditor replay; no actual experiment or native invocation."""
from copy import deepcopy
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import audit_formal_cli_rollover as module
import formal_cli_rollover as rollover_module
import resume_formal_completed_503 as transport
from test_machine_experiments import cohort_fixture, synthetic_caller


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("rollover-audit-fixture")
    runtime, frozen, rankings = cohort_fixture(root, max_cases=32)
    fake, calls = synthetic_caller()
    def caller(*args, **kwargs):
        raw, receipt = fake(*args, **kwargs)
        receipt["native_cli_sha256"] = rollover_module.OLD_BINARY
        path = Path(receipt["artifact_directory"]) / "receipt.json"
        path.write_bytes(module.original.protocol.canonical_bytes(receipt))
        return raw, receipt
    parent = runtime.canonical_root / "data/machine_runs/formal-experiments"
    run, result = module.original.experiment(runtime, frozen, rankings, parent, max_new_calls=2000, caller=caller)
    assert result["status"] == "completed_formal_machine_only_experiment"
    cohort = module.original.load_cohort(frozen, runtime)
    admission = transport.Admission(run, cohort)
    rollover = rollover_module.ReviewedRollover(run, admission, initialize_approved=True)
    job = next((run / "jobs").glob("*"))
    request = next((job / "attempts").glob("*/request.json")).read_bytes()
    retained = {}
    for index in range(4):
        attempt = job / "attempts" / f"retained-fixture-{index}"
        module.original.write_new(attempt / "request.json", request)
        path = attempt / "calls" / str(uuid4()) / "receipt.json"
        value = dict(accepted=False, artifact_directory=str(path.parent), execution_id=str(uuid4()),
            errors=["Synthetic retained429"], process_exit_code=1, observed_model=None, usage=None,
            native_cli_sha256=rollover_module.OLD_BINARY if index < 2 else rollover_module.CURRENT_BINARY,
            artifacts_sha256={})
        module.original.write_new(path, value)
        module.original.write_new(attempt / "runner_failure.json", dict(automatic_retry=False, error="Synthetic retained failure"))
        retained[path.relative_to(run).as_posix()] = module.original.file_hash(path)
    result["usage"] = rollover.driver().usage_audit(run)
    (run / "result.json").write_bytes(module.original.protocol.canonical_bytes(result))
    options = dict(cli_rollover_policy_sha256=module.original.file_hash(rollover.policy_path),
        cli_rollover_adapter_sha256=module.original.file_hash(rollover_module.__file__),
        transport_policy_sha256=module.original.file_hash(admission.policy_path), transport_adapter_sha256=module.TRANSPORT_SHA)
    return SimpleNamespace(runtime=runtime, frozen=frozen, rankings=rankings, run=run,
        result=result, calls=calls, options=options, retained=retained, rollover=rollover)


@pytest.fixture
def case(completed, monkeypatch):
    monkeypatch.setattr(module, "REQUIRED_RETAINED", completed.retained)
    return completed


def execute(case, **extra):
    return module.audit(case.runtime, case.frozen, case.rankings, case.run, **case.options, **extra)


def test_complete_two_cli_fixture_replays_with_no_calls_writes_or_global_mutation(case, monkeypatch):
    before = {str(p):module.original.file_hash(p) for p in case.run.rglob("*") if p.is_file()}
    pilot_write, usage = module.original.pilot.write_new, module.original.usage_audit
    old_auditor_driver = module.original_auditor.driver
    count = len(case.calls)
    monkeypatch.setattr(module.original, "call_structured_json", lambda *a, **k:pytest.fail("Native calls forbidden"))
    result = execute(case)
    assert result["status"] == "completed_formal_result_offline_audit_passed"
    assert result["variant_answers"] == 600 and result["ranked_query_conditions"] == 6000
    assert result["native_calls"] == 0 and result["human_gold"] is False
    assert result["reviewed_runtime_amendment"] and not result["original_single_cli_runtime_contract_unmodified"]
    assert result["summary"] == case.result["summary"] and result["usage"] == case.result["usage"]
    assert result["cli_runtime_lineage"]["aggregate_usage"] == result["usage"]
    assert set(result["cli_runtime_lineage"]["by_native_cli_sha256"]) == set(rollover_module.ALLOWED_BINARIES)
    assert all(row["explicit_completed_transport_certificate"] is False for row in result["failed_receipts"])
    assert len(result["failed_receipts"]) == 4 and len(case.calls) == count
    assert module.original.pilot.write_new is pilot_write and module.original.usage_audit is usage
    assert module.original_auditor.driver is old_auditor_driver
    assert before == {str(p):module.original.file_hash(p) for p in case.run.rglob("*") if p.is_file()}
    assert str(Path(module.__file__).resolve()) in result["protected_sha256"]


def test_original_single_binary_audit_still_rejects_mixed_fixture(case):
    with pytest.raises(ValueError, match="executable/model changed"):
        module.original.usage_audit(case.run)


def test_private_driver_replaces_only_usage_and_does_not_patch_globals(case):
    before = module.original.pilot.write_new
    private, rollover, pins = module.load_private_auditor(case.run, case.runtime, case.frozen,
        case.options["cli_rollover_policy_sha256"], case.options["cli_rollover_adapter_sha256"],
        case.options["transport_policy_sha256"], case.options["transport_adapter_sha256"])
    assert private.driver is not module.original and private.driver.pilot is not module.original.pilot
    original_ast = {node.name:node for node in ast.parse(Path(module.original.__file__).read_text()).body if isinstance(node, ast.FunctionDef)}
    private_ast = {node.name:node for node in ast.parse(private.driver._compiled_source).body if isinstance(node, ast.FunctionDef)}
    for name in ("validate_receipt", "stage_result", "quality_summary", "load_cohort", "load_rankings"):
        assert ast.dump(private_ast[name]) == ast.dump(original_ast[name])
    assert private.driver.usage_audit(case.run) == case.result["usage"]
    with pytest.raises(ValueError, match="forbids"):
        private.driver.pilot.write_new(case.run / "forbidden.json", {})
    with pytest.raises(ValueError, match="forbids"):
        private.driver.call_structured_json()
    assert not (case.run / "forbidden.json").exists() and module.original.pilot.write_new is before


@pytest.mark.parametrize("key", ["cli_rollover_policy_sha256", "cli_rollover_adapter_sha256",
    "transport_policy_sha256", "transport_adapter_sha256"])
def test_missing_or_wrong_explicit_pin_fails_closed(case, key):
    for invalid in (None, "0" * 64):
        options = dict(case.options, **{key:invalid})
        with pytest.raises((ValueError, FileNotFoundError)):
            module.audit(case.runtime, case.frozen, case.rankings, case.run, **options)


def test_missing_policy_cannot_be_initialized_by_audit(case):
    policy = case.rollover.policy_path
    kept = policy.with_suffix(".retained-fixture")
    policy.rename(kept)
    try:
        with pytest.raises((ValueError, FileNotFoundError)):
            execute(case)
        assert not policy.exists()
    finally:
        kept.rename(policy)


def test_isolated_transport_cannot_recreate_policy_after_load(case):
    private, rollover, pins = module.load_private_auditor(case.run, case.runtime, case.frozen,
        case.options["cli_rollover_policy_sha256"], case.options["cli_rollover_adapter_sha256"],
        case.options["transport_policy_sha256"], case.options["transport_adapter_sha256"])
    policy = case.run / "transport_admissions/policy.json"
    kept = policy.with_suffix(".retained-fixture")
    policy.rename(kept)
    try:
        with pytest.raises(ValueError, match="forbids"):
            private._transport.Admission(case.run, private.driver.load_cohort(case.frozen, case.runtime))
        assert not policy.exists()
    finally:
        kept.rename(policy)


def test_isolated_transport_cannot_create_even_a_fixture_certificate(case, monkeypatch):
    private, rollover, pins = module.load_private_auditor(case.run, case.runtime, case.frozen,
        case.options["cli_rollover_policy_sha256"], case.options["cli_rollover_adapter_sha256"],
        case.options["transport_policy_sha256"], case.options["transport_adapter_sha256"])
    admission = private._transport.Admission(case.run, private.driver.load_cohort(case.frozen, case.runtime))
    receipt = dict(execution_id=str(uuid4()))
    certificate = admission.path(receipt)
    monkeypatch.setattr(admission, "proof", lambda *args:({"synthetic":True}, {}))
    with pytest.raises(ValueError, match="forbids"):
        admission.certify(receipt, {})
    assert not certificate.exists()


def test_no_complete_result_is_never_promoted(tmp_path):
    runtime = SimpleNamespace(canonical_root=tmp_path)
    run = tmp_path / "data/machine_runs/formal-experiments/partial"
    run.mkdir(parents=True)
    with pytest.raises(ValueError, match="No completed result"):
        module.audit(runtime, tmp_path / "frozen", tmp_path / "rankings", run)
    assert not list(run.iterdir())


@pytest.mark.parametrize("key,value", [("human_gold",True), ("completed_variant_answers",599),
    ("completed_answer_questions",99), ("formal_heldout_test",False), ("original_frozen_inputs_unchanged",False)])
def test_incomplete_or_incompatible_result_flags_stop_before_loading(case, tmp_path, key, value):
    run = tmp_path / "partial"
    run.mkdir()
    changed = deepcopy(case.result)
    changed[key] = value
    (run / "result.json").write_bytes(module.original.protocol.canonical_bytes(changed))
    (run / "plan.json").write_bytes((case.run / "plan.json").read_bytes())
    with pytest.raises(ValueError):
        module.audit(case.runtime, case.frozen, case.rankings, run)
    assert not (run / "cli_rollover").exists()


def test_required_historical_pins_cannot_be_removed_or_changed(case):
    assert module.retained_pins() == case.retained
    name = next(iter(case.retained))
    with pytest.raises(ValueError, match="cannot change"):
        module.retained_pins({name:"0" * 64})


def test_additional_receipt_pin_is_preserved_and_missing_one_rejected(case):
    path = next((case.run / "jobs").glob("*/attempts/*/calls/*/receipt.json"))
    name = path.relative_to(case.run).as_posix()
    pins = {name:module.original.file_hash(path)}
    assert module.retained_pins(pins)[name] == pins[name]
    with pytest.raises(ValueError, match="historical native receipt|Reuse artifact"):
        execute(case, retained_receipts={"jobs/missing/receipt.json":"0" * 64})


def test_default_cohort_requires_exact_four_original_failures():
    assert len(module.REQUIRED_RETAINED) == 4
    assert "681aa061189def9e57145b63e63ae7d1ca9f28666e75b4fd26081c14aed28350" in module.REQUIRED_RETAINED.values()
    assert all(len(v) == 64 for v in module.REQUIRED_RETAINED.values())


@pytest.mark.parametrize("kind", ["third_binary", "missing_binary", "multiple_models"])
def test_original_model_guard_and_reviewed_binary_allowlist_still_fail_closed(case, kind):
    private, rollover, pins = module.load_private_auditor(case.run, case.runtime, case.frozen,
        case.options["cli_rollover_policy_sha256"], case.options["cli_rollover_adapter_sha256"],
        case.options["transport_policy_sha256"], case.options["transport_adapter_sha256"])
    paths = [case.run / name for name in list(case.retained)[:2]]
    original_bytes = {p:p.read_bytes() for p in paths}
    try:
        for index, path in enumerate(paths):
            receipt = module.original.read_json(path)
            if kind == "multiple_models":
                receipt["observed_model"] = "synthetic-model-" + str(index)
            else:
                receipt["native_cli_sha256"] = "f" * 64 if kind == "third_binary" else None
            path.write_bytes(module.original.protocol.canonical_bytes(receipt))
        with pytest.raises(ValueError, match="executable/model changed"):
            private.driver.usage_audit(case.run)
    finally:
        for path, raw in original_bytes.items():
            path.write_bytes(raw)
