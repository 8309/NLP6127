"""Synthetic terminal-503 fixtures only. Never invoke native Codex or actual data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from jsonschema.exceptions import ValidationError


@pytest.fixture
def adapter(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("formal_completed503_tests", scripts / "resume_formal_completed_503.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.original, "call_structured_json", lambda *a, **k: pytest.fail("Native call forbidden in fixtures"))
    return module


@pytest.fixture
def case(adapter, tmp_path):
    o = adapter.original
    run = o.long_path(tmp_path / "synthetic-run")
    request = dict(prompt="Synthetic verifier task", payload={"synthetic-case": {"text": "fixture only"}},
        schema=dict(type="object", properties={"supported": dict(type="boolean")},
            required=["supported"], additionalProperties=False), role="verifier")
    manifest = dict(version="synthetic-frozen-version", runtime_sha256={"synthetic-runtime": "fixture"})
    source_bytes = b"synthetic frozen implementation\n"
    native_snapshots = {name: o.protocol.sha256(source_bytes) for name in adapter.REQUIRED_SNAPSHOT_NAMES}
    source_hashes = dict(native_snapshots, **{"synthetic_outer_driver.py": "d" * 64})
    protected_input = run / "synthetic-protected-input.txt"
    o.write_new(protected_input, b"unchanged fixture input")
    protected = {str(protected_input): o.file_hash(protected_input)}
    cohort = SimpleNamespace(manifest=manifest, cfg=dict(requested_model=None), source_hashes=source_hashes,
        protected=protected)
    o.write_new(run / "plan.json", dict(protected_sha256=protected, frozen_manifest_sha256="f" * 64,
        answer_sample_plan_sha256="a" * 64))
    key = dict(stage="hybrid_AB", item_ids=["synthetic-case"], request=request,
        dataset_version=manifest["version"], runtime_sha256=manifest["runtime_sha256"])
    job = run / "jobs" / ("hybrid_AB-" + o.protocol.sha256(o.protocol.canonical_bytes(key))[:24])
    o.write_new(job / "job.json", key)
    attempt = job / "attempts" / "synthetic-attempt"
    o.write_new(attempt / "request.json", request)
    directory = attempt / "calls" / "synthetic-call"
    raw = o.protocol.canonical_bytes(dict(supported=True))
    usage = dict(input_tokens=123, output_tokens=5, cached_input_tokens=0)
    identity = "00000000-0000-4000-8000-000000000001"
    warning = ("Reconnecting... 2/5 (unexpected status 503 Service Unavailable: "
        "An error occurred while processing your request. You can retry your request, "
        "or contact us through our help center at help.openai.com if the error persists.)")
    events = [dict(type="thread.started", thread_id=identity), dict(type="turn.started"),
        dict(type="error", message=warning),
        dict(type="item.completed", item=dict(type="agent_message", text=raw.decode("utf-8"))),
        dict(type="turn.completed", usage=usage)]
    for name, value in {"stdin.txt": o.request_bytes(request), "schema.json": o.protocol.canonical_bytes(request["schema"]),
        "raw_final.json": raw, "events.jsonl": b"\n".join(o.protocol.canonical_bytes(event) for event in events) + b"\n",
        **{"implementation/" + name: source_bytes for name in native_snapshots}}.items():
        o.write_new(directory / name, value)
    receipt = dict(accepted=False, process_exit_code=0, usage=usage, execution_id=identity,
        errors=["CLI emitted a failed/error event"], observed_model=None, role="verifier",
        codex_cli_invocations=1, external_paid_api_calls=0, model_tools_disabled=True,
        execpolicy_rules_not_disabled=True, authentication="existing_ChatGPT_subscription",
        requested_model="cli_default_unresolved", source_snapshots_sha256=native_snapshots,
        native_cli_sha256=adapter.NATIVE_CLI_SHA, cli_version=adapter.NATIVE_CLI_VERSION,
        prompt_sha256=o.protocol.sha256(o.request_bytes(request)),
        schema_sha256=o.protocol.sha256(o.protocol.canonical_bytes(request["schema"])),
        artifacts_sha256={p.relative_to(directory).as_posix(): o.file_hash(p)
            for p in directory.rglob("*") if p.is_file()}, artifact_directory=str(directory))
    o.write_new(directory / "receipt.json", receipt)
    admission = adapter.Admission(run, cohort)

    def save_receipt(value=None):
        if value is not None:
            receipt.clear()
            receipt.update(value)
        (directory / "receipt.json").write_bytes(o.protocol.canonical_bytes(receipt))

    def replace_artifact(name, value):
        (directory / name).write_bytes(value)
        receipt["artifacts_sha256"][name] = o.file_hash(directory / name)
        save_receipt()

    def replace_events(value):
        replace_artifact("events.jsonl", b"\n".join(o.protocol.canonical_bytes(e) for e in value) + b"\n")

    return SimpleNamespace(run=run, job=job, attempt=attempt, directory=directory, raw=raw, events=events,
        receipt=receipt, request=request, cohort=cohort, protected_input=protected_input, admission=admission,
        save_receipt=save_receipt, replace_artifact=replace_artifact, replace_events=replace_events)


def test_driver_source_pin_and_exactly_four_private_predicate_changes(adapter, case):
    source = Path(adapter.original.__file__).read_text(encoding="utf-8")
    assert adapter.original.file_hash(adapter.original.__file__) == adapter.DRIVER_SHA
    assert adapter.original.file_hash(adapter.certifier.__file__) == adapter.CERTIFIER_SHA
    patched = adapter.patched_source(source)
    assert patched.count("_transport_admitted(") == 4
    assert 'receipt.get("process_exit_code") != 0' in patched
    assert 'if prior_attempts and not retry_failed:' in patched
    assert 'mechanical_output_rejection_not_answer_quality_selection=True' in patched
    assert source == Path(adapter.original.__file__).read_text(encoding="utf-8")
    for before in ['if (events["errors"] or events["thread_id"]', 'if read_json(path).get("accepted") is True',
        'if receipt.get("accepted") and file_hash(receipt_path)']:
        with pytest.raises(ValueError, match="exactly once"):
            adapter.patched_source(source.replace(before, before + "\n" + before))


def test_changed_driver_or_event_proof_pin_rejects_before_staging(adapter, case, monkeypatch):
    for name in ["DRIVER_SHA", "CERTIFIER_SHA"]:
        with monkeypatch.context() as patch:
            patch.setattr(adapter, name, "0" * 64)
            with pytest.raises(ValueError, match="validator changed"):
                adapter.driver(case.admission)


def test_certificate_is_zero_call_idempotent_and_preserves_original_false_receipt(adapter, case):
    before = {p: p.read_bytes() for p in case.directory.rglob("*") if p.is_file()}
    certificate = case.admission.certify(case.receipt, case.request)
    assert certificate["new_native_calls"] == 0
    assert certificate["operational_receipt_accepted"] is False
    assert certificate["human_gold"] is False
    assert certificate["terminal_proof"]["reconnect_attempts"] == [2]
    assert case.admission.certify(case.receipt, case.request) == certificate
    assert case.admission.admitted(case.receipt, case.request, case.cohort, case.run)
    assert {p: p.read_bytes() for p in case.directory.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("edit", [
    lambda c: c.events.pop(),
    lambda c: c.events.insert(2, dict(type="error", message="Your workspace is out of credits. Add credits to continue.")),
    lambda c: c.events.insert(2, dict(type="turn.failed", error=dict(message="failure"))),
    lambda c: c.events.insert(2, dict(type="item.completed", item=dict(type="tool_call"))),
    lambda c: c.events.insert(2, deepcopy(c.events[2])),
    lambda c: c.events[-2]["item"].update(text='{"supported":false}'),
    lambda c: c.events[-1].update(usage=dict(input_tokens=999, output_tokens=5)),
    lambda c: c.events[0].update(thread_id="another-native-execution"),
])
def test_event_proof_rejects_incomplete_quota_mixed_errors_tools_duplicates_and_mismatch(adapter, case, edit):
    edit(case)
    case.replace_events(case.events)
    with pytest.raises(ValueError):
        case.admission.certify(case.receipt, case.request)
    assert not case.admission.path(case.receipt).exists()


@pytest.mark.parametrize("field,value", [
    ("accepted", True), ("process_exit_code", 1), ("usage", None),
    ("errors", ["CLI emitted a failed/error event", "additional error"]),
    ("codex_cli_invocations", 2), ("external_paid_api_calls", 1), ("model_tools_disabled", False),
    ("requested_model", "different-model"), ("observed_model", "invented-model"),
])
def test_operational_or_usage_contract_mismatch_never_certifies(adapter, case, field, value):
    case.receipt[field] = value
    case.save_receipt()
    with pytest.raises(ValueError):
        case.admission.certify(case.receipt, case.request)


@pytest.mark.parametrize("name", ["stdin.txt", "schema.json", "raw_final.json", "events.jsonl", "implementation/codex_runtime.py"])
def test_changed_artifact_hash_or_missing_inventory_rejects(adapter, case, name):
    (case.directory / name).write_bytes(b"synthetic changed bytes")
    with pytest.raises(ValueError, match="bytes changed"):
        case.admission.certify(case.receipt, case.request)


@pytest.mark.parametrize("name", ["stdin.txt", "schema.json", "raw_final.json", "events.jsonl", "implementation/codex_runtime.py"])
def test_missing_required_inventory_rejects_before_certificate(adapter, case, name):
    del case.receipt["artifacts_sha256"][name]
    case.save_receipt()
    with pytest.raises(ValueError, match="inventory"):
        case.admission.certify(case.receipt, case.request)


def test_raw_duplicate_key_nonfinite_or_schema_failure_rejects_even_with_terminal_message(adapter, case):
    for raw in [b'{"supported":true,"supported":false}', b'{"supported":NaN}', b'{"supported":"yes"}']:
        case.events[-2]["item"]["text"] = raw.decode()
        case.replace_artifact("raw_final.json", raw)
        case.replace_events(case.events)
        with pytest.raises((ValueError, ValidationError)):
            case.admission.certify(case.receipt, case.request)


def test_changed_owning_request_runtime_and_source_snapshots_reject(adapter, case):
    original = adapter.original.read_json(case.job / "job.json")
    for field, value in [("request", dict(case.request, payload={"changed": True})),
        ("runtime_sha256", {"changed": "runtime"}), ("dataset_version", "another-frozen-version")]:
        changed = dict(original, **{field: value})
        (case.job / "job.json").write_bytes(adapter.original.protocol.canonical_bytes(changed))
        with pytest.raises(ValueError):
            case.admission.certify(case.receipt, case.request)
    (case.job / "job.json").write_bytes(adapter.original.protocol.canonical_bytes(original))
    case.receipt["source_snapshots_sha256"] = {"codex_runtime.py": "0" * 64}
    case.save_receipt()
    with pytest.raises(ValueError, match="contract"):
        case.admission.certify(case.receipt, case.request)


def test_changed_protected_input_certificate_policy_or_receipt_reject(adapter, case):
    case.admission.certify(case.receipt, case.request)
    certificate_path = case.admission.path(case.receipt)
    original_certificate = certificate_path.read_bytes()
    changed = adapter.original.read_json(certificate_path)
    changed["new_native_calls"] = 1
    certificate_path.write_bytes(adapter.original.protocol.canonical_bytes(changed))
    with pytest.raises(ValueError, match="certificate"):
        case.admission.admitted(case.receipt, case.request, case.cohort, case.run)
    certificate_path.write_bytes(original_certificate)
    case.receipt["extra_synthetic_metadata"] = True
    case.save_receipt()
    with pytest.raises(ValueError, match="certificate"):
        case.admission.admitted(case.receipt, case.request, case.cohort, case.run)
    del case.receipt["extra_synthetic_metadata"]
    case.save_receipt()
    policy = case.admission.policy_path.read_bytes()
    case.admission.policy_path.write_bytes(b"synthetic changed policy")
    with pytest.raises(ValueError, match="input changed"):
        case.admission.proof(case.receipt, case.request, case.cohort, case.run)
    case.admission.policy_path.write_bytes(policy)
    case.protected_input.write_bytes(b"synthetic changed protected input")
    with pytest.raises(ValueError, match="input changed"):
        case.admission.proof(case.receipt, case.request, case.cohort, case.run)


def test_original_stage_validation_and_completed_job_resume_reuse_without_second_call(adapter, case):
    case.admission.certify(case.receipt, case.request)
    staged = adapter.driver(case.admission)
    calls = []
    def process(replay):
        calls.append("original-stage-validator")
        raw, receipt = replay(case.request["prompt"], case.request["payload"], case.request["schema"], None, role="verifier")
        assert raw == {"supported": True} and receipt["accepted"] is False
        return dict(answers={"synthetic-case": raw}, removed_claims={"synthetic-case": []})
    budget = staged.ExecutionBudget(0)
    first = staged.job(case.run, "hybrid_AB", ["synthetic-case"], case.request, None, process,
        case.cohort, budget, lambda *a, **k: pytest.fail("No second native call"), retry_failed=False)
    second = staged.job(case.run, "hybrid_AB", ["synthetic-case"], case.request, None, process,
        case.cohort, budget, lambda *a, **k: pytest.fail("No second native call"), retry_failed=False)
    assert first == second and calls == ["original-stage-validator"] and budget.started == 0
    assert adapter.original.read_json(case.directory / "receipt.json")["accepted"] is False
    usage = staged.usage_audit(case.run)
    assert usage["accepted_native_executions"] == 0 and usage["accepted_experiment_output_executions"] == 1
    assert usage["known_native_execution_count"] == 1
    assert usage["known_native_token_totals"] == case.receipt["usage"]


def test_uncertified_failed_attempt_stops_and_stage_rejection_still_stops(adapter, case):
    staged = adapter.driver(case.admission)
    with pytest.raises(ValueError, match="retry requires explicit"):
        staged.job(case.run, "hybrid_AB", ["synthetic-case"], case.request, None, lambda _: {}, case.cohort,
            staged.ExecutionBudget(0), lambda *a, **k: pytest.fail("Native forbidden"))
    case.admission.certify(case.receipt, case.request)
    def reject_stage(_):
        raise ValueError("synthetic original stage rejects unsupported output")
    with pytest.raises(ValueError, match="original stage rejects"):
        staged.job(case.run, "hybrid_AB", ["synthetic-case"], case.request, None, reject_stage, case.cohort,
            staged.ExecutionBudget(0), lambda *a, **k: pytest.fail("Native forbidden"))
    assert not (case.job / "completed.json").exists()
    assert len(list(case.job.glob("validation_rejection-*.json"))) == 1


def test_caller_makes_at_most_one_original_invocation_then_mechanical_admission(adapter, case, monkeypatch):
    calls = []
    def retained_failure(*args, **kwargs):
        calls.append("one synthetic original invocation")
        raise adapter.CodexUnavailable("synthetic operational 503 failure")
    monkeypatch.setattr(adapter.original, "call_structured_json", retained_failure)
    raw, receipt = case.admission.caller(case.request["prompt"], case.request["payload"], case.request["schema"],
        case.directory.parent, role="verifier", requested_model=None)
    assert raw == {"supported": True} and receipt["accepted"] is False and len(calls) == 1
    assert case.admission.path(receipt).exists()


def test_main_rejects_unregistered_receipt_hash_before_policy_or_call(adapter, case, monkeypatch):
    monkeypatch.setattr(adapter.original, "load_staged_runtime", lambda *_: SimpleNamespace(canonical_root=case.run))
    monkeypatch.setattr(adapter.original, "load_cohort", lambda *_: case.cohort)
    monkeypatch.setattr(sys, "argv", ["fixture", "--canonical-root", str(case.run), "--frozen-dir", str(case.run),
        "--retrieval-run", str(case.run), "--resume", str(case.run),
        "--registered-receipt", str(case.directory / "receipt.json")])
    with pytest.raises(ValueError, match="exact reviewed preexisting"):
        adapter.main()


def test_main_rejects_registered_hash_with_changed_native_identity(adapter, case, monkeypatch):
    monkeypatch.setattr(adapter.original, "load_staged_runtime", lambda *_: SimpleNamespace(canonical_root=case.run))
    monkeypatch.setattr(adapter.original, "load_cohort", lambda *_: case.cohort)
    monkeypatch.setattr(adapter, "REGISTERED_RECEIPT_SHA", adapter.original.file_hash(case.directory / "receipt.json"))
    monkeypatch.setattr(sys, "argv", ["fixture", "--canonical-root", str(case.run), "--frozen-dir", str(case.run),
        "--retrieval-run", str(case.run), "--resume", str(case.run),
        "--registered-receipt", str(case.directory / "receipt.json")])
    with pytest.raises(ValueError, match="native identity changed"):
        adapter.main()


@pytest.mark.parametrize("missing", ["codex_runtime.py", "machine_evaluation.py", "run_codex_annotation.py"])
def test_missing_mandatory_native_snapshot_or_extra_snapshot_cannot_be_certified(adapter, case, missing):
    del case.cohort.source_hashes[missing]
    with pytest.raises(ValueError, match="mandatory native implementation"):
        case.admission.certify(case.receipt, case.request)


def test_extra_native_snapshot_metadata_rejected_but_outer_driver_cohort_files_expected(adapter, case):
    assert "synthetic_outer_driver.py" in case.cohort.source_hashes
    case.receipt["source_snapshots_sha256"]["synthetic_outer_driver.py"] = "d" * 64
    case.save_receipt()
    with pytest.raises(ValueError, match="execution contract"):
        case.admission.certify(case.receipt, case.request)


@pytest.mark.parametrize("field,value", [("native_cli_sha256", "0" * 64), ("cli_version", "another-cli")])
def test_changed_native_binary_or_cli_version_stops(adapter, case, field, value):
    case.receipt[field] = value
    case.save_receipt()
    with pytest.raises(ValueError, match="execution contract"):
        case.admission.certify(case.receipt, case.request)
