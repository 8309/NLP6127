"""No-call tests for lossless source-review transport recovery and one-shot gates."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil

import pytest

from test_cohort_adjudication_pipeline import runner, source, paired
from test_compact_cohort_pipeline import compact_backend


@pytest.fixture
def recovery(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("transport_review_tests", scripts / "pipeline_transport_recovery_adjudication.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def retry_case(recovery, runner, source):
    paired(source, 4)
    context = runner.prepare(source["root"], source["pool"], source["run"])
    prefix, entries = runner.progress_snapshot(source["run"])
    chunk = runner.export_chunk(context, source["root"] / "old-exports", 1, prefix, entries)
    base = recovery.legacy.isolated_base()
    recovery.legacy.configure(base, recovery.legacy.isolated_base(), dict(output=None, records=0, chunks={}, sha256={}))
    base.VERSION = recovery.VERSION
    spec, packets, left, right = recovery.bounded_spec(base, chunk)
    group = spec["groups"][0]
    payload, schema = base.adjudicator.prepare_batch(packets, left, right, group)
    stdin, schema_bytes = base.actual_adjudication_bytes(payload, schema)
    previous = source["root"] / "previous"
    parent = previous / "adjudications/chunk-0002/bounded-run"
    attempt = parent / "executions/failed-original"
    values = {"stdin.txt": stdin, "schema.json": schema_bytes,
              "events.jsonl": b'{"type":"thread.started","thread_id":"failed-context"}\n'
                    b'{"type":"error","message":"workspace routing discovery failed"}\n'
                    b'{"type":"turn.failed","error":{"message":"workspace routing discovery failed"}}\n'}
    for name, value in values.items():
        recovery.native.write_new(attempt / name, value)
    identity = dict(requested_model="cli_default_unresolved", authentication="existing_ChatGPT_subscription",
        native_cli_sha256="synthetic-native-binary", cli_version="synthetic-cli", disabled_configured_mcp_servers=[],
        model_tools_disabled=True, execpolicy_rules_not_disabled=True, role="adjudicator")
    receipt = dict(identity, artifact_directory=str(attempt), execution_id="failed-context", accepted=False,
        process_exit_code=1, codex_cli_invocations=1, usage=None, external_paid_api_calls=0,
        artifacts_sha256={name: recovery.protocol.sha256(value) for name, value in values.items()})
    recovery.native.write_new(attempt / "receipt.json", receipt)
    recovery.native.write_new(parent / "inputs.json", spec)
    recovery.native.write_new(previous / "failure.json", dict(type="CodexUnavailable", automatic_retry=False))
    failure_sha = recovery.protocol.sha256((previous / "failure.json").read_bytes())
    receipt_sha = recovery.protocol.sha256((attempt / "receipt.json").read_bytes())
    checkpoint = dict(failure_message="workspace routing discovery failed", quota_error_observed=False,
        model_output_from_failed_call=None, failed_usage_status="unknown_not_zero", automatic_retry=False,
        original_outputs_and_failure_bytes_preserved=True, failed_receipt_sha256=receipt_sha,
        original_failure_sha256=failure_sha, failed_receipt=(attempt / "receipt.json").relative_to(previous).as_posix(),
        failed_execution_id="failed-context", failed_actual_stdin_sha256=recovery.protocol.sha256(stdin),
        failed_schema_sha256=recovery.protocol.sha256(schema_bytes), first_failed_chunk=2,
        last_complete_chunk=1, reviewed_records=20, successful_fresh_native_contexts=2, historical_native_contexts_retained=0)
    recovery.native.write_new(previous / "routing_stop_checkpoint.json", checkpoint)
    stopped = recovery.verify_stopped_review(previous, receipt_sha256=receipt_sha, failure_sha256=failure_sha,
        checkpoint_sha256=recovery.protocol.sha256((previous / "routing_stop_checkpoint.json").read_bytes()), expected_last_chunk=1)
    authorization = source["root"] / "root_authorization.json"
    output = source["root"] / "new-adjudications/chunk-0002"
    recovery.native.write_new(authorization, dict(version="source-review-transport-retry-authorization-v1",
        failed_receipt_sha256=receipt_sha, original_failure_sha256=failure_sha, max_retry_native_calls=1,
        same_physical_input=True, same_model_request=True, new_failures_stop=True, recovery_output_parent=str(output)))
    return dict(source=source, base=base, chunk=chunk, previous=previous, stopped=stopped,
                authorization=authorization, identity=identity, attempt=attempt, group=group, output=output)


def accepted_caller(recovery, case, calls, mutate=None):
    backend = compact_backend(recovery.legacy, calls)
    def call(*args, **kwargs):
        raw, receipt = backend(*args, **kwargs)
        receipt.update(case["identity"])
        if mutate:
            mutate(receipt)
        path = Path(receipt["artifact_directory"]) / "receipt.json"
        path.write_bytes(recovery.protocol.canonical_bytes(receipt))
        return raw, receipt
    return call


def test_unanswered_transport_failure_is_explicitly_bound_and_usage_unknown(recovery, retry_case):
    stopped = retry_case["stopped"]
    assert stopped["failed_usage"] is None and stopped["failed_usage_status"] == "unknown_not_zero"
    assert stopped["failed_group"] == retry_case["group"]
    assert stopped["automatic_retry"] is False


@pytest.mark.parametrize("field,value", [("quota_error_observed", True), ("model_output_from_failed_call", {}),
    ("failure_message", "different failure"), ("failed_usage_status", "zero"), ("automatic_retry", True)])
def test_unknown_quota_answered_or_auto_retry_failures_rejected(recovery, retry_case, field, value):
    path = retry_case["previous"] / "routing_stop_checkpoint.json"
    checkpoint = recovery.native.read_json(path)
    checkpoint[field] = value
    path.write_bytes(recovery.protocol.canonical_bytes(checkpoint))
    with pytest.raises(ValueError, match="Unknown, quota"):
        recovery.verify_stopped_review(retry_case["previous"], receipt_sha256=retry_case["stopped"]["failed_receipt_sha256"],
            failure_sha256=retry_case["stopped"]["failed_pipeline_sha256"],
            checkpoint_sha256=recovery.protocol.sha256(path.read_bytes()), expected_last_chunk=1)


def test_one_exact_input_gate_retains_raw_and_resume_contract_without_second_call(recovery, retry_case):
    calls = []
    originals = {str(path): path.read_bytes() for path in retry_case["previous"].rglob("*") if path.is_file()}
    output = retry_case["output"]
    result = recovery.one_retry(retry_case["base"], retry_case["chunk"], output, retry_case["stopped"],
        retry_case["authorization"], caller=accepted_caller(recovery, retry_case, calls))
    assert calls == [retry_case["group"]]
    assert result["records"] == 10 and result["accepted"] == 10 and result["unresolved"] == 0
    assert result["complete_chunk"] is False and result["new_native_calls"] == 1
    assert result["exact_full_stdin_equal"] and result["exact_schema_equal"]
    bounded = output / "bounded-run"
    assert len(recovery.native.read_json(bounded / "inputs.json")["groups"]) == 2
    assert recovery.native.read_json(bounded / "retained_batches/batch-0001.json")["record_ids"] == retry_case["group"]
    recovery.verify_retry_gate(retry_case["base"], retry_case["chunk"], output, retry_case["stopped"])
    assert all(Path(path).read_bytes() == raw for path, raw in originals.items())
    with pytest.raises(ValueError, match="already planned"):
        recovery.one_retry(retry_case["base"], retry_case["chunk"], output, retry_case["stopped"],
            retry_case["authorization"], caller=lambda *args, **kwargs: pytest.fail("No repeat call"))
    completed = retry_case["base"].adjudicate_chunk(retry_case["chunk"], output,
        caller=accepted_caller(recovery, retry_case, calls))
    assert completed["records"] == 20 and completed["new_native_calls"] == 1
    assert len(calls) == 2 and calls[0] == retry_case["group"]


@pytest.mark.parametrize("artifact", ["stdin.txt", "schema.json"])
def test_actual_original_input_tampering_rejected_before_retry(recovery, retry_case, artifact):
    path = retry_case["attempt"] / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="physical source input"):
        recovery.one_retry(retry_case["base"], retry_case["chunk"], retry_case["output"],
            retry_case["stopped"], retry_case["authorization"],
            caller=lambda *args, **kwargs: pytest.fail("No altered input call"))


@pytest.mark.parametrize("field,value", [("requested_model", "changed-model"), ("native_cli_sha256", "changed-binary"),
    ("model_tools_disabled", False), ("execpolicy_rules_not_disabled", False), ("execution_id", "failed-context")])
def test_model_binary_isolation_or_context_change_retained_as_failure(recovery, retry_case, field, value):
    output = retry_case["output"]
    calls = []
    with pytest.raises(ValueError, match="changed native|fresh independent"):
        recovery.one_retry(retry_case["base"], retry_case["chunk"], output,
            retry_case["stopped"], retry_case["authorization"],
            caller=accepted_caller(recovery, retry_case, calls, lambda receipt: receipt.update({field: value})))
    assert len(calls) == 1
    assert (output / "bounded-run/retry_failure.json").exists()
    assert not (output / "bounded-run/retry_gate.json").exists()


def test_a_new_failure_stops_and_cannot_be_automatically_retried(recovery, retry_case):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("Synthetic quota or transport error")
    output = retry_case["output"]
    with pytest.raises(RuntimeError, match="Synthetic quota"):
        recovery.one_retry(retry_case["base"], retry_case["chunk"], output,
            retry_case["stopped"], retry_case["authorization"], caller=fail)
    with pytest.raises(ValueError, match="already planned"):
        recovery.one_retry(retry_case["base"], retry_case["chunk"], output,
            retry_case["stopped"], retry_case["authorization"], caller=fail)
    assert len(calls) == 1


def test_transport_retry_cannot_masquerade_as_original_first_attempt(recovery, retry_case):
    chunk = retry_case["chunk"]
    packets = recovery.native.read_json(chunk / "packets.json")
    context = dict(source_spec=dict(groups=[dict(record_ids=list(packets))]), history=dict(record_ids=[]))
    for side in ("a", "b"):
        path = chunk / f"independent_{side}.jsonl"
        rows = recovery.protocol.parse_jsonl(path.read_bytes())
        for row in rows:
            row.update(annotation_role=recovery.RETRY_ROLE, original_first_attempt=False)
        path.write_bytes(b"".join(recovery.protocol.canonical_bytes(row) + b"\n" for row in rows))
    population = recovery.validate_export_roles(context, chunk)
    assert population["retry_record_ids"] == list(packets)
    assert population["future_original_first_attempt_record_ids"] == []
    rows[0]["original_first_attempt"] = True
    (chunk / "independent_b.jsonl").write_bytes(b"".join(recovery.protocol.canonical_bytes(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="role changed"):
        recovery.validate_export_roles(context, chunk)


def test_unrecognized_producer_version_is_not_dynamically_allowed(recovery, monkeypatch):
    import run_codex_transport_recovery as producer
    monkeypatch.setattr(producer, "VERSION", "unknown-source-protocol")
    with pytest.raises(ValueError, match="Unregistered"):
        recovery.producer_module()


@pytest.fixture
def retained_case(recovery, retry_case):
    case = retry_case
    previous = case["previous"]
    chunk = previous / "chunks/chunk-0001"
    shutil.copytree(case["chunk"], chunk)
    calls = []
    result = case["base"].adjudicate_chunk(chunk, previous / "adjudications/chunk-0001",
        caller=accepted_caller(recovery, case, calls))
    path = previous / "retained/chunk-0001.json"
    recovery.native.write_new(path, result)
    ids = list(recovery.native.read_json(chunk / "packets.json"))
    recovery.native.append_progress(previous / "progress.jsonl", dict(type="chunk_retained", number=1,
        record_ids=ids, result_path=str(path), result_sha256=recovery.protocol.sha256(path.read_bytes())))
    recovery.native.write_new(previous / "manifest.json", dict(version=recovery.frozen.VERSION,
        implementation_sha256={}, source_order_record_ids=list(case["source"]["packets"]),
        original_reviews_cross_source_reuse=dict(chunks={}, sha256={})))
    recovery.native.write_new(previous / "source_inputs_sha256.json", {})
    return {**case, "retained_chunk": chunk, "retained_result": result, "original_calls": calls,
            "manifest_sha256": recovery.protocol.sha256((previous / "manifest.json").read_bytes())}


def test_all_accepted_prefix_rows_keep_original_native_ownership_without_calls(recovery, retained_case, monkeypatch):
    calls = []
    original = recovery.legacy.compact.retain_compact_decisions
    def counted(*args, **kwargs):
        calls.append(args[4])
        return original(*args, **kwargs)
    monkeypatch.setattr(recovery.legacy.compact, "retain_compact_decisions", counted)
    reused = recovery.verify_retained_prefix(retained_case["previous"], retained_case["stopped"],
        manifest_sha256=retained_case["manifest_sha256"])
    assert reused["records"] == 20 and reused["successful_fresh_native_contexts"] == 2
    assert reused["failed_fresh_native_contexts"] == 1 and reused["failed_usage"] is None
    assert len(calls) == 2  # Whole native batches, not twenty repeated raw decodes.
    assert reused["chunks"]["1"]["result"] == retained_case["retained_result"]
    assert reused["native_calls_repeated"] == 0
    for row in recovery.protocol.parse_jsonl(Path(retained_case["retained_result"]["adjudications_path"]).read_bytes()):
        assert Path(row["adjudication"]["raw_response_path"]).is_relative_to(
            Path(retained_case["retained_result"]["adjudications_path"]).parent / "executions")


@pytest.mark.parametrize("kind", ["raw", "label", "progress", "ownership"])
def test_accepted_raw_label_progress_or_ownership_tampering_rejected(recovery, retained_case, kind):
    result_path = retained_case["previous"] / "retained/chunk-0001.json"
    result = recovery.native.read_json(result_path)
    path = Path(result["adjudications_path"])
    rows = recovery.protocol.parse_jsonl(path.read_bytes())
    if kind == "raw":
        raw = Path(rows[0]["adjudication"]["raw_response_path"])
        raw.write_bytes(raw.read_bytes() + b" ")
    elif kind == "progress":
        progress = retained_case["previous"] / "progress.jsonl"
        events = recovery.protocol.parse_jsonl(progress.read_bytes())
        events[0]["record_ids"] = list(reversed(events[0]["record_ids"]))
        progress.write_bytes(b"".join(recovery.protocol.canonical_bytes(row) + b"\n" for row in events))
    else:
        if kind == "label":
            rows[0]["label"]["answer_points"][0]["text"] = "An invented unsupported answer."
        else:
            rows[0]["adjudication"]["execution_receipt_path"] = str(retained_case["attempt"] / "receipt.json")
        path.write_bytes(b"".join(recovery.protocol.canonical_bytes(row) + b"\n" for row in rows))
        result["adjudications_sha256"] = recovery.protocol.sha256(path.read_bytes())
        result_path.write_bytes(recovery.protocol.canonical_bytes(result))
        events = recovery.protocol.parse_jsonl((retained_case["previous"] / "progress.jsonl").read_bytes())
        events[0]["result_sha256"] = recovery.protocol.sha256(result_path.read_bytes())
        (retained_case["previous"] / "progress.jsonl").write_bytes(
            b"".join(recovery.protocol.canonical_bytes(row) + b"\n" for row in events))
    with pytest.raises(ValueError):
        recovery.verify_retained_prefix(retained_case["previous"], retained_case["stopped"],
            manifest_sha256=retained_case["manifest_sha256"])


@pytest.mark.parametrize("target", ["request", "authorization", "binding", "raw", "new_failure"])
def test_continuation_rebuilds_all_gate_proofs_and_rejects_tampering(recovery, retry_case, target):
    output = retry_case["output"]
    recovery.one_retry(retry_case["base"], retry_case["chunk"], output, retry_case["stopped"],
        retry_case["authorization"], caller=accepted_caller(recovery, retry_case, []))
    bounded = output / "bounded-run"
    paths = dict(request=bounded / "retry_request.json", authorization=retry_case["authorization"],
                 binding=bounded / "retained_batches/batch-0001.json")
    if target in paths:
        paths[target].write_bytes(paths[target].read_bytes() + b" ")
    elif target == "raw":
        gate = recovery.native.read_json(bounded / "retry_gate.json")
        path = Path(gate["receipt_path"]).parent / "raw_final.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        recovery.native.write_new(bounded / "retry_failure.json", dict(error="New failure retained"))
    with pytest.raises(ValueError):
        recovery.verify_retry_gate(retry_case["base"], retry_case["chunk"], output, retry_case["stopped"])


def test_once_only_authorization_cannot_be_reused_for_a_fresh_output(recovery, retry_case):
    with pytest.raises(ValueError, match="different recovery output"):
        recovery.one_retry(retry_case["base"], retry_case["chunk"], retry_case["output"].with_name("other-output"),
            retry_case["stopped"], retry_case["authorization"], caller=lambda *args, **kwargs: pytest.fail("No second destination call"))


def test_independent_checkpoint_anchor_cannot_be_rewritten(recovery, retry_case):
    path = retry_case["previous"] / "routing_stop_checkpoint.json"
    expected = recovery.protocol.sha256(path.read_bytes())
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="immutable hash"):
        recovery.verify_stopped_review(retry_case["previous"], receipt_sha256=retry_case["stopped"]["failed_receipt_sha256"],
            failure_sha256=retry_case["stopped"]["failed_pipeline_sha256"], checkpoint_sha256=expected, expected_last_chunk=1)


def test_guard_merge_rejects_conflicts(recovery):
    with pytest.raises(ValueError, match="conflicting"):
        recovery.merge_guards({"original-raw": "old-hash"}, {"original-raw": "changed-hash"})
