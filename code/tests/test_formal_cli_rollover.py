"""Synthetic two-binary lineage fixtures; no native calls or actual run data."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("formal_cli_rollover_fixture", scripts / "formal_cli_rollover.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    monkeypatch.setattr(wrapper.o, "call_structured_json", lambda *a, **k: pytest.fail("Native calls forbidden"))
    return wrapper


@pytest.fixture
def case(module, tmp_path):
    o = module.o
    run = o.long_path(tmp_path / "synthetic-run")
    protected = run / "synthetic-source.txt"
    o.write_new(protected, b"unchanged fixture")
    plan = dict(protected_sha256={str(protected): o.file_hash(protected)}, cfg=dict(requested_model=None),
        frozen_manifest_sha256="f" * 64, answer_sample_plan_sha256="a" * 64)
    o.write_new(run / "plan.json", plan)
    transport_path = run / "transport_admissions" / "policy.json"
    o.write_new(transport_path, dict(synthetic=True, original_operational_receipts_unchanged=True))
    admission = SimpleNamespace(run=run, guards=plan["protected_sha256"], policy_path=transport_path,
        admitted=lambda *a: False)
    return SimpleNamespace(run=run, admission=admission, plan=plan, protected=protected)


def add_receipt(module, case, identity, binary, usage=None, accepted=True, job="synthetic", attempt="attempt"):
    path = case.run / "jobs" / job / "attempts" / attempt / "calls" / "call" / "receipt.json"
    value = dict(execution_id=identity, native_cli_sha256=binary, observed_model=None, usage=usage,
        accepted=accepted, artifacts_sha256={})
    module.o.write_new(path, value)
    return path, value


def approved(module, case):
    return module.ReviewedRollover(case.run, case.admission, initialize_approved=True)


def test_exact_five_predicate_extension_and_pinned_original_sources(module):
    original = Path(module.o.__file__).read_text(encoding="utf-8")
    source = module.derived_source()
    assert source == module.transport.patched_source(original).replace(module.PREDICATE, module.REPLACEMENT, 1)
    assert source.count("_transport_admitted(") == 4
    assert source.count("_reviewed_binary_lineage(") == 1
    assert "or len(known_models) > 1:" in source
    assert module.o.file_hash(module.o.__file__) == module.transport.DRIVER_SHA
    assert module.o.file_hash(module.transport.__file__) == module.TRANSPORT_SHA
    assert module.o.file_hash(Path(module.__file__).parent / "audit_formal_results.py") == module.AUDITOR_SHA
    assert module.transport.NATIVE_CLI_SHA == module.OLD_BINARY


def test_replacement_requires_unique_original_predicate(module, monkeypatch):
    monkeypatch.setattr(module, "PREDICATE", "missing predicate")
    with pytest.raises(ValueError, match="exactly once"):
        module.derived_source()


def test_initialization_requires_direct_approval(module, case):
    with pytest.raises(ValueError, match="reviewed initialization"):
        module.ReviewedRollover(case.run, case.admission)
    assert not (case.run / "cli_rollover/policy.json").exists()


def test_initialization_policy_is_immutable_and_replay_requires_exact_sha(module, case):
    rollover = approved(module, case)
    before = {str(p): p.read_bytes() for p in case.run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="exact policy SHA"):
        module.ReviewedRollover(case.run, case.admission, initialize_approved=True)
    digest = module.o.file_hash(rollover.policy_path)
    replay = module.ReviewedRollover(case.run, case.admission, expected_policy_sha256=digest)
    assert replay.policy == rollover.policy
    assert before == {str(p): p.read_bytes() for p in case.run.rglob("*") if p.is_file()}
    assert rollover.policy["new_binary_completed503_admitted"] is False
    assert rollover.policy["additional_usage_predicate_changes"] == 1


@pytest.mark.parametrize("kind", ["policy", "source", "plan", "transport"])
def test_policy_or_bound_source_mutation_refused(module, case, monkeypatch, kind):
    rollover = approved(module, case)
    digest = module.o.file_hash(rollover.policy_path)
    if kind == "policy":
        rollover.policy_path.write_bytes(rollover.policy_path.read_bytes() + b"\n")
    elif kind == "source":
        monkeypatch.setattr(module, "AUDITOR_SHA", "0" * 64)
    elif kind == "plan":
        (case.run / "plan.json").write_bytes((case.run / "plan.json").read_bytes() + b"\n")
    else:
        case.admission.policy_path.write_bytes(case.admission.policy_path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        module.ReviewedRollover(case.run, case.admission, expected_policy_sha256=digest)


def test_active_policy_guard_refuses_post_initialization_mutation(module, case):
    rollover = approved(module, case)
    case.protected.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="changed"):
        rollover.driver()


def test_requested_model_change_rejected(module, case):
    plan = deepcopy(case.plan)
    plan["cfg"]["requested_model"] = "different-model"
    (case.run / "plan.json").write_bytes(module.o.protocol.canonical_bytes(plan))
    with pytest.raises(ValueError, match="requested model"):
        approved(module, case)


def test_other_run_cannot_reuse_admission(module, case, tmp_path):
    with pytest.raises(ValueError, match="exact transport admission run"):
        module.ReviewedRollover(tmp_path / "other", case.admission, initialize_approved=True)


def test_both_binaries_accepted_without_changing_usage_schema(module, case):
    add_receipt(module, case, "old", module.OLD_BINARY, dict(input_tokens=11, output_tokens=2), job="old")
    add_receipt(module, case, "new", module.CURRENT_BINARY, dict(input_tokens=7, output_tokens=3), job="new")
    with pytest.raises(ValueError, match="executable/model changed"):
        module.o.usage_audit(case.run)
    rollover = approved(module, case)
    usage = rollover.driver().usage_audit(case.run)
    assert usage["known_native_execution_count"] == 2
    assert usage["known_native_token_totals"] == dict(input_tokens=18, output_tokens=5)
    assert usage["observed_model"] is None
    assert usage["observed_model_identity_available_for_all"] is False
    single = SimpleNamespace(run=module.o.long_path(case.run.parent / "single"))
    add_receipt(module, single, "single", module.OLD_BINARY, dict(input_tokens=1))
    assert set(usage) == set(module.o.usage_audit(single.run))


@pytest.mark.parametrize("binary", [None, "", "0" * 64, "third-binary"])
def test_unknown_or_third_native_binary_rejected(module, case, binary):
    add_receipt(module, case, "fixture", binary, dict(input_tokens=1))
    rollover = approved(module, case)
    with pytest.raises(ValueError, match="executable/model changed"):
        rollover.driver().usage_audit(case.run)


def test_unknown_execution_identity_still_requires_allowed_binary(module, case):
    add_receipt(module, case, None, "0" * 64)
    with pytest.raises(ValueError, match="executable/model changed"):
        approved(module, case).driver().usage_audit(case.run)


def test_known_model_change_still_rejected(module, case):
    first, receipt = add_receipt(module, case, "old", module.OLD_BINARY, dict(input_tokens=1), job="old")
    receipt["observed_model"] = "observed-A"
    first.write_bytes(module.o.protocol.canonical_bytes(receipt))
    second, receipt = add_receipt(module, case, "new", module.CURRENT_BINARY, dict(input_tokens=1), job="new")
    receipt["observed_model"] = "observed-B"
    second.write_bytes(module.o.protocol.canonical_bytes(receipt))
    with pytest.raises(ValueError, match="executable/model changed"):
        approved(module, case).driver().usage_audit(case.run)


def test_per_binary_counts_once_known_unknown_and_original_false_retained(module, case):
    first, false_receipt = add_receipt(module, case, "old-complete503", module.OLD_BINARY,
        dict(input_tokens=11, output_tokens=2, reasoning_output_tokens=1), accepted=False, job="old")
    duplicate = case.run / "jobs/duplicate/attempts/attempt/calls/call/receipt.json"
    module.o.write_new(duplicate, first.read_bytes())
    add_receipt(module, case, "old-unknown", module.OLD_BINARY, None, accepted=False, job="old-unknown")
    add_receipt(module, case, "new", module.CURRENT_BINARY, dict(input_tokens=7, output_tokens=3), job="new")
    add_receipt(module, case, None, module.CURRENT_BINARY, dict(input_tokens=999), accepted=False, job="unidentified")
    module.o.write_new(case.run / "jobs/old/completed.json", dict(execution_id="old-complete503"))
    module.o.write_new(case.run / "jobs/precall/attempts/attempt/request.json", dict(synthetic=True))
    rollover = approved(module, case)
    before = {str(p): p.read_bytes() for p in case.run.rglob("*") if p.is_file()}
    report = rollover.lineage_report()
    assert before == {str(p): p.read_bytes() for p in case.run.rglob("*") if p.is_file()}
    assert module.o.read_json(first) == false_receipt and false_receipt["accepted"] is False
    old, new = (report["by_native_cli_sha256"][binary] for binary in (module.OLD_BINARY, module.CURRENT_BINARY))
    assert old["known_native_execution_count"] == 2 and old["rejected_native_executions"] == 2
    assert old["accepted_experiment_output_executions"] == 1 and old["executions_with_unknown_usage"] == 1
    assert old["known_native_token_totals"] == dict(input_tokens=11, output_tokens=2, reasoning_output_tokens=1)
    assert new["known_native_execution_count"] == 1 and new["known_native_token_totals"] == dict(input_tokens=7, output_tokens=3)
    assert len(new["unknown_execution_identity_receipts"]) == 1
    assert report["aggregate_usage"]["known_native_token_totals"] == dict(input_tokens=18, output_tokens=5, reasoning_output_tokens=1)
    assert report["aggregate_usage"]["known_native_execution_count"] == 3
    assert len(report["interrupted_or_precall_attempts_with_unknown_execution"]) == 1
    assert report["physical_model_diversity_claimed"] is False
    assert report["native_calls"] == 0 and report["quality_certified"] is False


def test_conflicting_execution_identity_retains_original_guard(module, case):
    add_receipt(module, case, "same", module.OLD_BINARY, dict(input_tokens=1), job="old")
    add_receipt(module, case, "same", module.CURRENT_BINARY, dict(input_tokens=2), job="new")
    with pytest.raises(ValueError, match="Conflicting receipts"):
        approved(module, case).lineage_report()


def test_native_artifact_mutation_retains_original_guard(module, case):
    path, receipt = add_receipt(module, case, "old", module.OLD_BINARY, dict(input_tokens=1))
    artifact = path.parent / "raw.txt"
    module.o.write_new(artifact, b"initial")
    receipt["artifacts_sha256"] = {"raw.txt": module.o.file_hash(artifact)}
    path.write_bytes(module.o.protocol.canonical_bytes(receipt))
    rollover = approved(module, case)
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="audit artifact changed"):
        rollover.lineage_report()
