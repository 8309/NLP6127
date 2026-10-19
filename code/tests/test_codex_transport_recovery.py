"""Synthetic transport-recovery fixtures; no native service or real corpus reads."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from test_correct_codex_wire_batch import backend


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "transport_recovery_test", scripts / "run_codex_transport_recovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rows():
    return [dict(record_id=f"synthetic-transport-question-{index}",
        question=f"What crude flow did synthetic region {index} report?", as_of="2024-02-01",
        evidence=[dict(passage_id=f"synthetic-transport-source-{index}",
            document_id=f"synthetic-transport-doc-{index}",
            text=f"Synthetic region {index} reported 5 barrels of crude flow.",
            available_by="2024-01-01")]) for index in range(20)]


def write_value(module, path, value):
    path.write_bytes(value if isinstance(value, bytes) else module.protocol.canonical_bytes(value))


def make_group(runner, rows, *, retry=True):
    prepared = runner.clarification.prepare_source_batch(rows)
    return dict(batch="batch-0024" if retry else "batch-0025", split="test",
        record_ids=[row["record_id"] for row in rows],
        full_stdin_sha256=prepared["actual_wire_stdin_sha256"],
        schema_sha256=prepared["actual_wire_schema_sha256"],
        mapping_sha256=prepared["codec"]["mapping_sha256"],
        full_payload_bytes=len(prepared["stdin"]) + len(runner.protocol.canonical_bytes(prepared["schema"])),
        format_instruction_sha256=runner.INSTRUCTION_SHA256,
        annotation_role="transport_retry_without_prior_label" if retry else "format_clarified_first_attempt",
        original_first_attempt=not retry)


def write_execution_manifest(runner, directory, groups, binary):
    context = directory / "synthetic-original-context"
    command = runner.native.cli_command(binary, context, context / "schema.json", context / "final.json", None, [])
    runner.native.write_new(directory / "manifest.json", dict(specification=dict(
        version=runner.VERSION, requested_model=None,
        native_command_identity=runner.clarification.normalized_command(command), groups=groups)))


@pytest.fixture
def execution_fixture(runner, rows, tmp_path):
    binary = tmp_path / "synthetic-native.exe"
    binary.write_bytes(b"synthetic isolated native binary")
    group = make_group(runner, rows[:10])
    packets = {row["record_id"]: runner.protocol.clean_packet(row, prompt=runner.wire.batch.BATCH_PROMPT)
        for row in rows}
    write_execution_manifest(runner, tmp_path, [group], binary)
    return tmp_path, group, packets, binary


def test_recovery_pair_keeps_actual_input_schema_and_fresh_contexts(runner, execution_fixture):
    output, group, packets, binary = execution_fixture
    calls, results = [], {}
    expected = runner.clarification.prepare_source_batch([packets[rid]["payload"] for rid in group["record_ids"]])
    for stream in ("a", "b"):
        results[stream] = runner.execute_attempt(output, stream, group, packets, binary, {}, [], {}, None, 90,
            backend(runner.clarification, calls))
        assert results[stream]["receipt"]["accepted"]
        assert runner.verify_attempt(results[stream]["attempt"], results[stream]["receipt"], packets) == results[stream]["envelopes"]
        assert (results[stream]["attempt"] / "schema.json").read_bytes() == runner.protocol.canonical_bytes(expected["schema"])
        for envelope in results[stream]["envelopes"].values():
            assert envelope["annotation_role"] == "transport_retry_without_prior_label"
            assert envelope["original_first_attempt"] is False
    assert len(calls) == 2 and calls[0][1] == calls[1][1] == expected["stdin"]
    assert calls[0][2] != calls[1][2]
    assert results["a"]["receipt"]["execution_id"] != results["b"]["receipt"]["execution_id"]


def test_transport_recovery_retains_exact_quote_rejection(runner, execution_fixture):
    output, group, packets, binary = execution_fixture
    result = runner.execute_attempt(output, "a", group, packets, binary, {}, [], {}, None, 90,
        backend(runner.clarification, [], bad_quote=True))
    assert not result["receipt"]["accepted"]
    assert any("exact eligible quotes" in error for error in result["receipt"]["errors"])


@pytest.mark.parametrize("name", ["stdin.txt", "schema.json", "command.json"])
def test_recovery_readback_rejects_changed_actual_artifact(runner, execution_fixture, name):
    output, group, packets, binary = execution_fixture
    result = runner.execute_attempt(output, "a", group, packets, binary, {}, [], {}, None, 90,
        backend(runner.clarification, []))
    (result["attempt"] / name).write_bytes(b"synthetic changed physical artifact")
    with pytest.raises(ValueError):
        runner.verify_attempt(result["attempt"], result["receipt"], packets)


def test_native_isolation_settings_cannot_change(runner, execution_fixture):
    output, group, packets, binary = execution_fixture
    manifest = runner.native.read_json(output / "manifest.json")
    command = manifest["specification"]["native_command_identity"]
    command[command.index("--sandbox") + 1] = "danger-full-access"
    write_value(runner, output / "manifest.json", manifest)
    calls = []
    with pytest.raises(ValueError, match="command|settings"):
        runner.execute_attempt(output, "a", group, packets, binary, {}, [], {}, None, 90,
            backend(runner.clarification, calls))
    assert calls == []


def test_receipt_role_hash_and_group_hash_are_checked_independently(runner, execution_fixture):
    output, group, packets, binary = execution_fixture
    result = runner.execute_attempt(output, "a", group, packets, binary, {}, [], {}, None, 90,
        backend(runner.clarification, []))
    for field, changed in (("annotation_role", "format_clarified_first_attempt"),
            ("original_first_attempt", True), ("mapping_sha256", "0" * 64)):
        receipt = deepcopy(result["receipt"])
        receipt[field] = changed
        with pytest.raises(ValueError):
            runner.verify_attempt(result["attempt"], receipt, packets)
    manifest = runner.native.read_json(output / "manifest.json")
    manifest["specification"]["groups"][0]["full_stdin_sha256"] = "0" * 64
    write_value(runner, output / "manifest.json", manifest)
    receipt = deepcopy(result["receipt"])
    receipt["annotation_manifest_sha256"] = runner.protocol.sha256((output / "manifest.json").read_bytes())
    with pytest.raises(ValueError, match="group|bound"):
        runner.verify_attempt(result["attempt"], receipt, packets)


@pytest.fixture
def source_fixture(runner, rows, tmp_path, monkeypatch):
    import run_codex_clarified_cohort as old

    pool = tmp_path / "synthetic-pool"
    source = pool / "cohort_annotations_clarified/run-synthetic-source"
    source.mkdir(parents=True)
    binary = tmp_path / "synthetic-native.exe"
    binary.write_bytes(b"synthetic isolated native binary")
    historical_rows = [dict(rows[0], record_id=f"synthetic-history-{index}") for index in range(590)]
    all_rows = historical_rows + rows
    packets = {row["record_id"]: runner.protocol.clean_packet(row, prompt=runner.wire.batch.BATCH_PROMPT)
        for row in all_rows}
    groups = [make_group(runner, rows[:10]), make_group(runner, rows[10:], retry=False)]
    for group in groups:
        group.pop("annotation_role")
        group.pop("original_first_attempt")
    context = source / "synthetic-command-context"
    command = old.native.cli_command(binary, context, context / "schema.json", context / "final.json", None, [])
    values = {"packets.json": packets, "history.json": dict(record_ids=[row["record_id"] for row in historical_rows],
        protected_sha256={}), "groups.json": groups, "protected_inputs.json": {}}
    for name, value in values.items():
        old.native.write_new(source / name, value)
    original_spec = dict(version=old.VERSION, requested_model=None,
        native_command_identity=old.clarification.normalized_command(command), groups=groups,
        eligible_record_ids=[row["record_id"] for row in all_rows], bindings={},
        native_cli_sha256=runner.protocol.sha256(binary.read_bytes()), disabled_mcp_names=[],
        implementation_sha256={}, format_instruction_sha256=runner.INSTRUCTION_SHA256)
    old.native.write_new(source / "manifest.json", dict(specification=original_spec,
        immutable_sha256={name: runner.protocol.sha256((source / name).read_bytes()) for name in values}))
    failed_paths = {}

    def transport_executor(command, stdin, context, attempt, environment, timeout):
        events = [dict(type="thread.started", thread_id="synthetic-dns-" + attempt.parent.parent.name),
            dict(type="error", message="client call route error (Connect): dns error: No such host is known. (os error 11001)"),
            dict(type="turn.failed", error=dict(message="DNS connection failed with os error 11001"))]
        old.native.write_new(attempt / "events.jsonl", b"".join(old.protocol.canonical_bytes(event) + b"\n" for event in events))
        old.native.write_new(attempt / "stderr.txt", b"synthetic DNS transport error 11001\n")
        return None

    for stream in ("a", "b"):
        result = old.execute_attempt(source, stream, groups[0], packets, binary, {}, [], {}, None, 90, transport_executor)
        path = result["attempt"] / "receipt.json"
        failed_paths[stream] = path
        old.native.append_progress(source / "progress.jsonl", dict(type="completed", accepted=False,
            stream=stream, batch=groups[0]["batch"], record_ids=groups[0]["record_ids"],
            execution_protocol=old.VERSION, receipt=path.relative_to(source).as_posix(),
            receipt_sha256=old.protocol.sha256(path.read_bytes())))
        old.native.append_progress(source / "execution_usage.jsonl", result["usage"])
    historical = {stream: {row["record_id"]: dict(label="synthetic-history-" + stream)
        for row in historical_rows} for stream in ("a", "b")}
    sources = {stream: {row["record_id"]: dict(source_run=str(source),
        annotation_role="synthetic-unmodified-history", record_id=row["record_id"])
        for row in historical_rows} for stream in ("a", "b")}
    monkeypatch.setattr(old, "completed_records", lambda output, stream, packets: historical[stream])
    monkeypatch.setattr(old, "record_sources", lambda output, stream: sources[stream])
    monkeypatch.setattr(old.wire.previous, "load_pool", lambda root, pool_dir:
        (pool, all_rows, [dict(record_id=row["record_id"], split="test") for row in all_rows], {}, {}))
    old.native.write_new(source / "summary-synthetic-stop.json", dict(status="stopped_without_automatic_retry",
        accepted_counts={"a": 590, "b": 590}))
    old.native.write_new(source / "future_first_attempt_agreement-synthetic.json", dict(paired_records=0))
    monkeypatch.setattr(runner, "STOP_SUMMARY_SHA256", runner.protocol.sha256((source / "summary-synthetic-stop.json").read_bytes()))
    monkeypatch.setattr(runner, "STOP_PROGRESS_SHA256", runner.protocol.sha256((source / "progress.jsonl").read_bytes()))
    monkeypatch.setattr(runner, "STOP_MANIFEST_SHA256", runner.protocol.sha256((source / "manifest.json").read_bytes()))
    monkeypatch.setattr(runner, "EXPECTED_TOTAL", len(all_rows))
    monkeypatch.setattr(runner, "historical_usage", lambda directory:
        runner.protocol.parse_jsonl((source / "execution_usage.jsonl").read_bytes()))
    return dict(pool=pool, source=source, binary=binary, packets=packets, rows=all_rows,
        historical=historical, sources=sources, failed_paths=failed_paths, old=old)


def refresh_failed_artifact(runner, fixture, stream, name, value):
    path = fixture["failed_paths"][stream]
    write_value(runner, path.parent / name, value)
    receipt = runner.native.read_json(path)
    receipt["artifacts_sha256"][name] = runner.protocol.sha256((path.parent / name).read_bytes())
    write_value(runner, path, receipt)
    events = runner.protocol.parse_jsonl((fixture["source"] / "progress.jsonl").read_bytes())
    for event in events:
        if event["stream"] == stream:
            event["receipt_sha256"] = runner.protocol.sha256(path.read_bytes())
    write_value(runner, fixture["source"] / "progress.jsonl",
        b"".join(runner.protocol.canonical_bytes(event) + b"\n" for event in events))
    runner.STOP_PROGRESS_SHA256 = runner.protocol.sha256((fixture["source"] / "progress.jsonl").read_bytes())


def test_prepare_seals_590_history_and_unmodified_remaining_groups(runner, source_fixture, tmp_path):
    fixture = source_fixture
    original = {stream: path.read_bytes() for stream, path in fixture["failed_paths"].items()}
    output, records = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)
    assert records == fixture["historical"] and all(len(stream) == 590 for stream in records.values())
    groups = runner.native.read_json(output / "groups.json")
    old_groups = runner.native.read_json(fixture["source"] / "groups.json")
    assert [group["record_ids"] for group in groups] == [group["record_ids"] for group in old_groups]
    for new, old in zip(groups, old_groups):
        assert new["full_stdin_sha256"] == old["full_stdin_sha256"]
        assert new["schema_sha256"] == old["schema_sha256"]
    assert groups[0]["annotation_role"] == "transport_retry_without_prior_label"
    assert groups[0]["original_first_attempt"] is False
    assert groups[1]["annotation_role"] == "format_clarified_first_attempt"
    assert groups[1]["original_first_attempt"] is True
    assert all(path.read_bytes() == original[stream] for stream, path in fixture["failed_paths"].items())


def test_prepare_requires_explicit_transport_retry_authorization(runner, source_fixture, tmp_path):
    fixture = source_fixture
    with pytest.raises(ValueError, match="authoriz"):
        runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=False)


@pytest.mark.parametrize("addition", [
    dict(type="error", message="Quota exceeded after DNS os error 11001"),
    dict(type="item.completed", item=dict(type="agent_message", text="A prior model answer exists.")),
    dict(type="turn.completed", usage=dict(input_tokens=5, output_tokens=1)),
])
def test_retry_rejects_quota_model_output_or_completed_turn(runner, source_fixture, tmp_path, addition):
    fixture = source_fixture
    events = runner.protocol.parse_jsonl((fixture["failed_paths"]["a"].parent / "events.jsonl").read_bytes())
    refresh_failed_artifact(runner, fixture, "a", "events.jsonl",
        b"".join(runner.protocol.canonical_bytes(event) + b"\n" for event in [*events, addition]))
    with pytest.raises(ValueError):
        runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)


def test_retry_rejects_any_retained_prior_raw_answer(runner, source_fixture, tmp_path):
    fixture = source_fixture
    refresh_failed_artifact(runner, fixture, "a", "raw_final.json", b"{\"prior_answer\": true}\n")
    with pytest.raises(ValueError):
        runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)


def test_source_input_tamper_rejected_before_preparation(runner, source_fixture, tmp_path):
    fixture = source_fixture
    (fixture["failed_paths"]["a"].parent / "stdin.txt").write_bytes(b"changed source bytes")
    with pytest.raises(ValueError):
        runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)


def test_even_rehashed_mismatched_pair_input_is_not_a_transport_retry(runner, source_fixture, tmp_path):
    fixture = source_fixture
    refresh_failed_artifact(runner, fixture, "b", "stdin.txt", b"different physical B prompt")
    with pytest.raises(ValueError):
        runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)


def test_undeclared_additional_failed_batch_is_not_authorized(runner, source_fixture, tmp_path):
    fixture = source_fixture
    events = runner.protocol.parse_jsonl((fixture["source"] / "progress.jsonl").read_bytes())
    events[0]["batch"] = "batch-0025"
    write_value(runner, fixture["source"] / "progress.jsonl",
        b"".join(runner.protocol.canonical_bytes(event) + b"\n" for event in events))
    runner.STOP_PROGRESS_SHA256 = runner.protocol.sha256((fixture["source"] / "progress.jsonl").read_bytes())
    with pytest.raises(ValueError):
        runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)


def test_repeated_completed_reads_use_sealed_history_not_old_decoders(runner, source_fixture, tmp_path, monkeypatch):
    fixture = source_fixture
    output, records = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("Historical native labels must not be re-decoded on every cache read")

    monkeypatch.setattr(fixture["old"], "completed_records", unexpected_decode)
    monkeypatch.setattr(fixture["old"], "record_sources", unexpected_decode)
    for _ in range(2):
        for stream in ("a", "b"):
            assert runner.completed_records(output, stream, fixture["packets"]) == records[stream]
            assert runner.record_sources(output, stream) == fixture["sources"][stream]


def test_changed_sealed_history_fails_even_before_any_new_calls(runner, source_fixture, tmp_path):
    fixture = source_fixture
    output, _ = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)
    (output / "history.json").write_bytes(b"{}")
    with pytest.raises((ValueError, KeyError)):
        runner.completed_records(output, "a", fixture["packets"])


def configure_synthetic_runtime(runner, fixture, monkeypatch):
    monkeypatch.setattr(runner.native, "discover_native_cli", lambda *args: fixture["binary"])
    monkeypatch.setattr(runner.native, "child_environment", lambda: {})
    monkeypatch.setattr(runner.native, "configured_mcp_names", lambda environment: [])
    monkeypatch.setattr(runner, "merged_usage", lambda output: dict(executions=2, subscription_dollar_cost=None))


def test_explicit_two_call_bound_executes_only_retry_pair(runner, source_fixture, tmp_path, monkeypatch):
    fixture = source_fixture
    output, _ = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)
    configure_synthetic_runtime(runner, fixture, monkeypatch)
    calls = []
    summary = runner.run(output, max_native_contexts=2, executor=backend(runner.clarification, calls))
    assert len(calls) == 2 and summary["new_native_invocations"] == 2
    assert summary["accepted_counts"] == {"a": 600, "b": 600}
    assert not summary["failures"]
    progress = runner.protocol.parse_jsonl((output / "progress.jsonl").read_bytes())
    assert {event["batch"] for event in progress} == {"batch-0024"}


def test_semantic_failure_stops_without_automatic_second_attempt(runner, source_fixture, tmp_path, monkeypatch):
    fixture = source_fixture
    output, _ = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)
    configure_synthetic_runtime(runner, fixture, monkeypatch)
    calls = []
    summary = runner.run(output, max_native_contexts=2,
        executor=backend(runner.clarification, calls, bad_quote=True))
    assert summary["failures"] and len(calls) == 2
    with pytest.raises(ValueError, match="retry|failure|stop"):
        runner.run(output, max_native_contexts=2, executor=backend(runner.clarification, calls))
    assert len(calls) == 2


def test_orphan_success_receipt_without_progress_cannot_repeat(runner, source_fixture, tmp_path, monkeypatch):
    fixture = source_fixture
    output, _ = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)
    configure_synthetic_runtime(runner, fixture, monkeypatch)
    group = runner.native.read_json(output / "groups.json")[0]
    result = runner.execute_attempt(output, "a", group, fixture["packets"], fixture["binary"], {}, [], {}, None, 90,
        backend(runner.clarification, []))
    assert result["receipt"]["accepted"] and (result["attempt"] / "raw_final.json").exists()
    calls = []
    with pytest.raises(ValueError, match="Undurable|interrupted"):
        runner.run(output, max_native_contexts=2, executor=backend(runner.clarification, calls))
    assert calls == []


def test_interrupted_directory_without_receipt_cannot_repeat(runner, source_fixture, tmp_path, monkeypatch):
    fixture = source_fixture
    output, _ = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)
    configure_synthetic_runtime(runner, fixture, monkeypatch)
    (output / "a/batch-0024/attempt-synthetic-interrupted").mkdir(parents=True)
    calls = []
    with pytest.raises(ValueError, match="Undurable|interrupted"):
        runner.run(output, max_native_contexts=2, executor=backend(runner.clarification, calls))
    assert calls == []


def test_accepted_a_only_recovery_runs_b_before_any_future_group(runner, source_fixture, tmp_path, monkeypatch):
    fixture = source_fixture
    output, _ = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)
    configure_synthetic_runtime(runner, fixture, monkeypatch)
    group = runner.native.read_json(output / "groups.json")[0]
    result = runner.execute_attempt(output, "a", group, fixture["packets"], fixture["binary"], {}, [], {}, None, 90,
        backend(runner.clarification, []))
    receipt_path = result["attempt"] / "receipt.json"
    runner.native.append_progress(output / "progress.jsonl", dict(type="completed", accepted=True, stream="a",
        batch=group["batch"], record_ids=group["record_ids"], execution_protocol=runner.VERSION,
        receipt=receipt_path.relative_to(output).as_posix(), receipt_sha256=runner.protocol.sha256(receipt_path.read_bytes())))
    runner.native.append_progress(output / "execution_usage.jsonl", result["usage"])
    calls = []
    summary = runner.run(output, max_native_contexts=2, executor=backend(runner.clarification, calls))
    assert len(calls) == summary["new_native_invocations"] == 1
    assert calls[0][0].parent.name == "batch-0024" and calls[0][0].parent.parent.name == "b"
    assert summary["accepted_counts"] == {"a": 600, "b": 600}


def test_fresh_four_call_request_is_barred_after_exact_retry_pair(runner, source_fixture, tmp_path, monkeypatch):
    fixture = source_fixture
    output, _ = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)
    configure_synthetic_runtime(runner, fixture, monkeypatch)
    calls = []
    summary = runner.run(output, max_native_contexts=4, executor=backend(runner.clarification, calls))
    assert len(calls) == summary["new_native_invocations"] == 2
    assert {call[0].parent.name for call in calls} == {"batch-0024"}
    assert summary["accepted_counts"] == {"a": 600, "b": 600}


def test_consistent_new_model_and_command_tamper_cannot_rebind_old_source(runner, source_fixture, tmp_path, monkeypatch):
    fixture = source_fixture
    output, _ = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)
    configure_synthetic_runtime(runner, fixture, monkeypatch)
    manifest = runner.native.read_json(output / "manifest.json")
    spec = manifest["specification"]
    spec["requested_model"] = "synthetic-unapproved-model"
    command = spec["native_command_identity"]
    command[command.index("exec") + 1:command.index("exec") + 1] = ["--model", spec["requested_model"]]
    write_value(runner, output / "manifest.json", manifest)
    calls = []
    with pytest.raises(ValueError, match="model|identity"):
        runner.run(output, max_native_contexts=2, executor=backend(runner.clarification, calls))
    assert calls == []


def test_rehashed_new_group_metadata_cannot_rebind_original_group(runner, source_fixture, tmp_path, monkeypatch):
    fixture = source_fixture
    output, _ = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)
    configure_synthetic_runtime(runner, fixture, monkeypatch)
    manifest = runner.native.read_json(output / "manifest.json")
    groups = runner.native.read_json(output / "groups.json")
    groups[0]["full_payload_bytes"] += 1
    write_value(runner, output / "groups.json", groups)
    manifest["specification"]["groups"] = groups
    manifest["immutable_sha256"]["groups.json"] = runner.protocol.sha256((output / "groups.json").read_bytes())
    write_value(runner, output / "manifest.json", manifest)
    calls = []
    with pytest.raises(ValueError, match="groups|roles"):
        runner.run(output, max_native_contexts=2, executor=backend(runner.clarification, calls))
    assert calls == []


@pytest.mark.parametrize("field, changed", [("failed_record_ids", []),
    ("failed_record_ids", ["synthetic-wrong-retry-question"]), ("controlled_retry_contexts", 3)])
def test_recovery_boundary_metadata_cannot_change(runner, source_fixture, tmp_path, monkeypatch, field, changed):
    fixture = source_fixture
    output, _ = runner.prepare(tmp_path, fixture["pool"], fixture["source"], authorize_retry=True)
    configure_synthetic_runtime(runner, fixture, monkeypatch)
    manifest = runner.native.read_json(output / "manifest.json")
    manifest["specification"][field] = changed
    write_value(runner, output / "manifest.json", manifest)
    calls = []
    with pytest.raises(ValueError, match="(?i)boundary|retry|source"):
        runner.run(output, max_native_contexts=2, executor=backend(runner.clarification, calls))
    assert calls == []
