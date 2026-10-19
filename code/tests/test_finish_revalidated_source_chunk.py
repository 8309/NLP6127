"""Synthetic one-chunk completion guards; no subscription or API calls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil

import pytest

from test_cohort_adjudication_pipeline import runner, source, paired
from test_compact_cohort_pipeline import compact_backend


@pytest.fixture
def controller(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("finish_revalidated_test", scripts / "finish_revalidated_source_chunk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity():
    return dict(requested_model="cli_default_unresolved", authentication="existing_ChatGPT_subscription",
        native_cli_sha256="synthetic-binary", cli_version="synthetic-cli", disabled_configured_mcp_servers=[],
        model_tools_disabled=True, execpolicy_rules_not_disabled=True, role="adjudicator",
        process_exit_code=0, codex_cli_invocations=1, external_paid_api_calls=0, usage={})


def backend(controller, calls, mutate=None):
    caller = compact_backend(controller.legacy, calls)
    def execute(*args, **kwargs):
        raw, receipt = caller(*args, **kwargs)
        receipt.update(identity())
        if mutate:
            mutate(receipt)
        (Path(receipt["artifact_directory"]) / "receipt.json").write_bytes(controller.protocol.canonical_bytes(receipt))
        return raw, receipt
    return execute


@pytest.fixture
def sealed(controller, runner, source, monkeypatch):
    paired(source, 4)
    ctx = runner.prepare(source["root"], source["pool"], source["run"])
    prefix, entries = runner.progress_snapshot(source["run"])
    exported = runner.export_chunk(ctx, source["root"] / "exports", 1, prefix, entries)
    review = source["root"] / "sealed-review"
    chunk = review / "chunks/chunk-0029"
    shutil.copytree(exported, chunk)
    owner = review / "adjudications/chunk-0029/bounded-run"
    base = controller.legacy.isolated_base()
    controller.legacy.configure(base, controller.legacy.isolated_base(), dict(output=None, records=0, chunks={}, sha256={}))
    base.VERSION = controller.recovery.VERSION
    normal_bounded_spec = controller.recovery.bounded_spec
    def bounded_spec(selected, path):
        spec, packets, left, right = normal_bounded_spec(selected, path)
        ids = list(packets)
        spec["groups"] = [ids[:9], ids[9:18], ids[18:]]
        return spec, packets, left, right
    monkeypatch.setattr(controller.recovery, "bounded_spec", bounded_spec)
    spec, packets, left, right = bounded_spec(base, chunk)
    groups, ids = spec["groups"], list(packets)
    controller.native.write_new(owner / "inputs.json", spec)
    context = dict(base=base, packets=packets, left=left, right=right)
    payload, schema, stdin, schema_bytes = controller.physical(context, groups[0])
    failed = source["root"] / "old-rejected/receipt.json"
    controller.native.write_new(failed.parent / "stdin.txt", stdin)
    controller.native.write_new(failed.parent / "schema.json", schema_bytes)
    original = dict(identity(), accepted=False, artifact_directory=str(failed.parent), execution_id="original-routing-failure",
        artifacts_sha256={name: controller.protocol.sha256((failed.parent / name).read_bytes())
                          for name in ("stdin.txt", "schema.json")})
    controller.native.write_new(failed, original)
    stopped = dict(failed_receipt=str(failed), failed_receipt_sha256=controller.protocol.sha256(failed.read_bytes()),
        failed_pipeline_sha256="synthetic-original-failure", protected_sha256={})
    controller.native.write_new(review / "manifest.json", dict(version=controller.recovery.VERSION,
        implementation_sha256={}, original_source_review_stop=stopped, chunks=[ids] * 29))
    old_auth = source["root"] / "old-one-shot-authorization.json"
    controller.native.write_new(old_auth, dict(version="source-review-transport-retry-authorization-v1",
        failed_receipt_sha256=stopped["failed_receipt_sha256"], original_failure_sha256=stopped["failed_pipeline_sha256"],
        max_retry_native_calls=1, same_physical_input=True, same_model_request=True, new_failures_stop=True,
        recovery_output_parent=str(owner.parent)))
    seed_calls = []
    first_raw, first = backend(controller, seed_calls)(base.adjudicator.SYSTEM, payload, schema, owner / "executions", role="adjudicator")
    first_path = Path(first["artifact_directory"]) / "receipt.json"
    first_decisions = base.adjudicator.retain_decisions(first_raw, packets, left, right, groups[0], first)
    binding = dict(record_ids=groups[0], decisions=first_decisions, receipt_path=first_path.relative_to(owner).as_posix(),
        receipt_sha256=controller.protocol.sha256(first_path.read_bytes()))
    controller.native.write_new(owner / "retained_batches/batch-0001.json", binding)
    request = dict(authorization=str(old_auth), authorization_sha256=controller.protocol.sha256(old_auth.read_bytes()),
        stdin_sha256=controller.protocol.sha256(stdin), schema_sha256=controller.protocol.sha256(schema_bytes),
        original_failed_receipt=str(failed), original_failed_receipt_sha256=stopped["failed_receipt_sha256"],
        original_failure_sha256=stopped["failed_pipeline_sha256"], record_ids=groups[0], max_new_native_calls=1)
    controller.native.write_new(owner / "retry_request.json", request)
    gate = dict(version=controller.recovery.VERSION, new_native_calls=1,
        validation_status="one_exact_input_retry_retained_pending_full_chunk", complete_chunk=False,
        record_ids=groups[0], retry_request_sha256=controller.protocol.sha256((owner / "retry_request.json").read_bytes()),
        first_batch_binding_sha256=controller.protocol.sha256((owner / "retained_batches/batch-0001.json").read_bytes()),
        receipt_path=str(first_path), receipt_sha256=binding["receipt_sha256"], execution_id=first["execution_id"],
        usage=first["usage"], records=9, accepted=9, unresolved=0)
    controller.native.write_new(owner / "retry_gate.json", gate)
    second_payload, second_schema, _, _ = controller.physical(context, groups[1])
    second_raw, second = backend(controller, seed_calls)(base.adjudicator.SYSTEM, second_payload, second_schema,
        owner / "executions", role="adjudicator")
    second_decisions = base.adjudicator.retain_decisions(second_raw, packets, left, right, groups[1], second)
    second["accepted"] = False
    second["errors"] = ["CLI emitted a failed/error event"] * 4
    second_path = Path(second["artifact_directory"]) / "receipt.json"
    second_path.write_bytes(controller.protocol.canonical_bytes(second))
    cert_dir = source["root"] / "external-certificate"
    certificate = dict(original_receipt=str(second_path), operational_receipt_accepted=False, new_native_calls=0)
    records = dict(decisions=second_decisions, actual_argv_recorded=False, actual_command_bytes_unavailable=True,
        source_decisions_pending_certificate_aware_assembly=True)
    controller.native.write_new(cert_dir / "certificate.json", certificate)
    controller.native.write_new(cert_dir / "derived_records.json", dict(records=records))
    def admission(directory):
        assert Path(directory) == cert_dir
        guards = {str(path.resolve()): controller.protocol.sha256(path.read_bytes()) for path in
                  (cert_dir / "certificate.json", cert_dir / "derived_records.json")}
        return certificate, deepcopy(records), guards
    paths = dict(manifest=review / "manifest.json", export=chunk / "export_receipt.json", inputs=owner / "inputs.json",
        gate=owner / "retry_gate.json", request=owner / "retry_request.json", binding=owner / "retained_batches/batch-0001.json",
        first_receipt=first_path, second_receipt=second_path, old_authorization=old_auth,
        certificate=cert_dir / "certificate.json", derived=cert_dir / "derived_records.json")
    anchors = {name: controller.protocol.sha256(path.read_bytes()) for name, path in paths.items()}
    _, _, third_stdin, third_schema = controller.physical(context, groups[2])
    anchors.update(third_stdin=controller.protocol.sha256(third_stdin), third_schema=controller.protocol.sha256(third_schema))
    intent = source["root"] / "external-final-two-intent"
    auth_path = source["root"] / "completion-root-authorization.json"
    controller.native.write_new(auth_path, dict(version=controller.AUTH_VERSION, root_authorized=True, max_new_native_calls=1,
        controller_sha256=controller.protocol.sha256(Path(controller.__file__).read_bytes()), bounded_owner=str(owner),
        intent_directory=str(intent), certificate_sha256=anchors["certificate"], third_record_ids=groups[2],
        third_stdin_sha256=anchors["third_stdin"], third_schema_sha256=anchors["third_schema"],
        same_model_request=True, new_failures_stop=True))
    return dict(review=review, owner=owner, chunk=chunk, cert_dir=cert_dir, auth=auth_path, intent=intent,
        anchors=anchors, admission=admission, groups=groups, first_path=first_path, second_path=second_path,
        original_bytes={str(path): path.read_bytes() for path in owner.rglob("*") if path.is_file()})


def run(controller, case, **kwargs):
    return controller.run(case["review"], case["cert_dir"], case["auth"], case["intent"],
        anchors=case["anchors"], admission_loader=case["admission"], **kwargs)


def test_read_only_preflight_does_not_prepare_project_failed_future_or_write(controller, sealed, monkeypatch):
    monkeypatch.setattr(controller.recovery, "prepare", lambda *args, **kwargs: pytest.fail("No global prepare"))
    monkeypatch.setattr(controller.recovery.frozen, "resolved_progress_snapshot", lambda *args, **kwargs: pytest.fail("No global progress"))
    proof = run(controller, sealed, check_only=True, caller=lambda *args, **kwargs: pytest.fail("No model calls"))
    assert proof["new_native_calls"] == 0 and proof["groups"] == sealed["groups"]
    assert not sealed["intent"].exists() and proof["original_second_receipt_accepted"] is False


def test_only_final_two_called_original_false_and_native_owners_unchanged_resume_no_calls(controller, sealed):
    calls = []
    result = run(controller, sealed, caller=backend(controller, calls))
    assert calls == [sealed["groups"][2]] and result["records"] == 20 and result["new_native_calls"] == 1
    assert result["retained_native_calls"] == 3 and result["certified_original_false_native_calls"] == 1
    assert result["accepted"] == 20 and result["unresolved"] == 0
    assert Path(result["adjudications_path"]).parent == sealed["owner"]
    assert controller.native.read_json(sealed["second_path"])["accepted"] is False
    assert all(Path(path).read_bytes() == raw for path, raw in sealed["original_bytes"].items())
    decisions = controller.protocol.parse_jsonl(Path(result["adjudications_path"]).read_bytes())
    assert [row["record_id"] for row in decisions] == [key for group in sealed["groups"] for key in group]
    assert all(Path(row["adjudication"]["raw_response_path"]).is_relative_to(sealed["owner"] / "executions") for row in decisions)
    resumed = run(controller, sealed, caller=lambda *args, **kwargs: pytest.fail("No accepted replay"))
    assert resumed["new_native_calls"] == 0 and resumed["adjudications_path"] == result["adjudications_path"]


def test_durably_bound_final_two_without_completion_reuses_raw_with_zero_new_calls(controller, sealed):
    result = run(controller, sealed, caller=backend(controller, []))
    # Model a crash after binding, before final assembly; fixtures are disposable.
    (sealed["intent"] / "completion.json").unlink()
    Path(result["adjudications_path"]).unlink()
    Path(result["certificate_admission_path"]).unlink()
    resumed = run(controller, sealed, caller=lambda *args, **kwargs: pytest.fail("No bound-call replay"))
    assert resumed["new_native_calls"] == 0 and resumed["lifetime_final_two_native_calls"] == 1
    assert resumed["retained_native_calls"] == 3 and resumed["records"] == 20


def test_unresolved_final_two_are_retained_without_replacement_or_dropping(controller, sealed):
    accepted = backend(controller, [])
    def unresolved(*args, **kwargs):
        raw, receipt = accepted(*args, **kwargs)
        for value in raw.values():
            value.update(status="unresolved", label_basis="none", new_label=None,
                reason="Synthetic unresolved source review.", unresolved_reasons=["Synthetic source ambiguity."])
        directory = Path(receipt["artifact_directory"])
        (directory / "raw_final.json").write_bytes(controller.protocol.canonical_bytes(raw))
        receipt["artifacts_sha256"]["raw_final.json"] = controller.protocol.sha256((directory / "raw_final.json").read_bytes())
        (directory / "receipt.json").write_bytes(controller.protocol.canonical_bytes(receipt))
        return raw, receipt
    result = run(controller, sealed, caller=unresolved)
    assert result["records"] == 20 and result["accepted"] == 18 and result["unresolved"] == 2
    assert len(controller.protocol.parse_jsonl(Path(result["adjudications_path"]).read_bytes())) == 20


@pytest.mark.parametrize("key", ["gate", "request", "binding", "inputs", "certificate", "derived", "first_receipt", "second_receipt"])
def test_any_pinned_gate_source_certificate_or_receipt_tampering_stops(controller, sealed, key):
    paths = dict(gate=sealed["owner"] / "retry_gate.json", request=sealed["owner"] / "retry_request.json",
        binding=sealed["owner"] / "retained_batches/batch-0001.json", inputs=sealed["owner"] / "inputs.json",
        certificate=sealed["cert_dir"] / "certificate.json", derived=sealed["cert_dir"] / "derived_records.json",
        first_receipt=sealed["first_path"], second_receipt=sealed["second_path"])
    paths[key].write_bytes(paths[key].read_bytes() + b" ")
    with pytest.raises(ValueError, match="immutable hash"):
        run(controller, sealed, check_only=True)


@pytest.mark.parametrize("kind", ["orphan", "interrupted", "failed"])
def test_unknown_unbound_or_failed_execution_stops_before_call(controller, sealed, kind):
    directory = sealed["owner"] / "executions" / ("unknown-" + kind)
    directory.mkdir()
    if kind != "interrupted":
        controller.native.write_new(directory / "receipt.json", dict(accepted=kind != "failed"))
    with pytest.raises(ValueError, match="Unknown, interrupted, unbound"):
        run(controller, sealed, caller=lambda *args, **kwargs: pytest.fail("No orphan replay"))


def test_new_failure_keeps_intent_and_cannot_be_retried(controller, sealed):
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("Synthetic native failure")
    with pytest.raises(RuntimeError, match="Synthetic native failure"):
        run(controller, sealed, caller=failed)
    assert len(calls) == 1 and (sealed["intent"] / "intent.json").exists()
    assert (sealed["intent"] / "failure.json").exists()
    with pytest.raises(ValueError, match="stops without"):
        run(controller, sealed, caller=failed)
    assert len(calls) == 1


def test_preexisting_unfinished_intent_never_replays(controller, sealed):
    proof = run(controller, sealed, check_only=True)
    context = controller.validate_snapshot(sealed["review"], sealed["cert_dir"],
        anchors=sealed["anchors"], admission_loader=sealed["admission"])
    controller.validate_authorization(context, sealed["auth"], sealed["intent"])
    controller.native.write_new(sealed["intent"] / "intent.json", dict(proof=proof, protected_sha256=context["guards"],
        original_owner_sha256={}))
    with pytest.raises(ValueError, match="already planned"):
        run(controller, sealed, caller=lambda *args, **kwargs: pytest.fail("No interrupted replay"))


@pytest.mark.parametrize("field,value", [("root_authorized", False), ("max_new_native_calls", 2),
    ("same_model_request", False), ("third_record_ids", ["unknown"]), ("bounded_owner", "elsewhere")])
def test_authorization_cannot_expand_or_rebind_two_case_scope(controller, sealed, field, value):
    auth = controller.native.read_json(sealed["auth"])
    auth[field] = value
    sealed["auth"].write_bytes(controller.protocol.canonical_bytes(auth))
    with pytest.raises(ValueError, match="exact bound root"):
        run(controller, sealed, check_only=True)


def test_changed_future_prefix_fails_but_later_failed_events_are_not_interpreted(controller, sealed):
    export = controller.native.read_json(sealed["chunk"] / "export_receipt.json")
    progress = Path(export["source_annotation_run"]) / "progress.jsonl"
    progress.write_bytes(progress.read_bytes() + b'{"type":"failed","accepted":false,"batch":"future"}\n')
    assert run(controller, sealed, check_only=True)["new_native_calls"] == 0
    progress.write_bytes(b"X" + progress.read_bytes()[1:])
    with pytest.raises(ValueError, match="durable prefix"):
        run(controller, sealed, check_only=True)
