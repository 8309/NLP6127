"""Synthetic-only bounded migration tests; fixtures are not actual QA data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("throughput_test", scripts / "run_codex_cohort_throughput.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    rows, questions = [], []
    for index in range(30):
        identifier = f"synthetic-query-{index:03d}"
        evidence = dict(passage_id=f"synthetic-passage-{index}", document_id=f"synthetic-document-{index}",
                        text=f"Synthetic region {index} reported stable crude oil exports.", available_by="2024-01-01")
        rows.append(dict(record_id=identifier, question=f"What did synthetic region {index} report about crude exports?",
                         as_of="2024-03-01", evidence=[evidence]))
        questions.append(dict(record_id=identifier, question=rows[-1]["question"], as_of=rows[-1]["as_of"],
                              event_group=f"synthetic-family-{index}", split="development" if index < 20 else "test"))
    return rows, questions


def backend(runner, failure=False):
    def execute(command, stdin, context, attempt, environment, timeout):
        payload = json.loads((context / "packets.json").read_bytes())
        values = {}
        for identifier, row in payload.items():
            evidence = row["evidence"][0]
            values[identifier] = dict(record_id=identifier, label=dict(answerability="sufficient", event_status="observed",
                qrels={evidence["passage_id"]: 2}, answer_points=[dict(text=evidence["text"], citations=[evidence["passage_id"]])],
                quotes=[dict(passage_id=evidence["passage_id"], text=evidence["text"])], reasons=["Synthetic exact source quote."]))
        events = [dict(type="thread.started", thread_id="synthetic-execution-" + uuid4().hex)]
        if failure:
            events.extend([dict(type="error", message="Synthetic quota failure"), dict(type="turn.failed")])
        else:
            events.append(dict(type="turn.completed", usage=dict(input_tokens=100, output_tokens=25)))
            (context / "final.json").write_bytes((json.dumps(values, indent=2) + "\r\n").encode())
        (attempt / "events.jsonl").write_bytes(b"".join(runner.protocol.canonical_bytes(event) + b"\n" for event in events))
        (attempt / "stderr.txt").write_bytes(b"")
        return 1 if failure else 0
    return execute


@pytest.fixture
def migration(runner, inputs, monkeypatch, tmp_path):
    rows, questions = inputs
    pool = tmp_path / "pool"
    pool.mkdir()
    binary = tmp_path / "native.exe"
    binary.write_bytes(b"synthetic-native")
    monkeypatch.setattr(runner.previous, "load_pool", lambda root, path: (pool, rows, questions, {}, {"synthetic": True}))
    monkeypatch.setattr(runner.native, "discover_native_cli", lambda cli: binary)
    monkeypatch.setattr(runner.native, "configured_mcp_names", lambda environment: [])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: SimpleNamespace(
        stdout=b"Logged in using ChatGPT" if "login" in command else b"synthetic-cli", stderr=b""))
    source, summary = runner.previous.run(tmp_path, pool, 10, partition="all_current", executor=backend(runner))
    runner.native.write_new(source / "STOP_REQUESTED", b"Synthetic clean stop")
    summary["status"] = "stopped_without_automatic_retry"
    runner.native.write_new(source / "summary-stopped.json", summary)
    return pool, source


def test_groups_preserve_source_order_and_split_boundaries(runner, inputs):
    rows, questions = inputs
    groups = runner.build_groups(rows, questions, [row["record_id"] for row in rows[:5]], 10, 262144)
    assert [len(group["record_ids"]) for group in groups] == [10, 5, 10]
    assert [group["split"] for group in groups] == ["development", "development", "test"]
    assert [identifier for group in groups for identifier in group["record_ids"]] == [row["record_id"] for row in rows[5:]]
    assert all(group["full_payload_bytes"] == group["actual_stdin_bytes"] + group["actual_schema_bytes"] for group in groups)


def test_hard_byte_bound_greedily_keeps_complete_cases(runner, inputs):
    rows, questions = inputs
    _, stdin, schema = runner.batch.prepare_batch(rows[:3])
    limit = len(stdin) + len(runner.protocol.canonical_bytes(schema))
    groups = runner.build_groups(rows, questions, [], 10, limit)
    assert all(group["full_payload_bytes"] <= limit for group in groups)
    assert all(1 <= len(group["record_ids"]) <= 3 for group in groups)
    assert sum(len(group["record_ids"]) for group in groups) == 30


def test_interleaved_locked_splits_are_not_reordered_or_mixed(runner, inputs):
    rows, questions = inputs
    questions[15]["split"] = "test"
    groups = runner.build_groups(rows, questions, [], 10, 262144)
    assert [identifier for group in groups for identifier in group["record_ids"]] == [row["record_id"] for row in rows]
    split_by_id = {row["record_id"]: row["split"] for row in questions}
    assert all({split_by_id[key] for key in group["record_ids"]} == {group["split"]} for group in groups)


def test_single_oversized_case_aborts_never_truncates_or_drops(runner, inputs):
    rows, questions = inputs
    with pytest.raises(ValueError, match="Single complete"):
        runner.build_groups(rows, questions, [], 10, 1)


@pytest.mark.parametrize("limit", [0, True, 11, 20])
def test_explicit_case_limit_never_exceeds_ten(runner, inputs, limit):
    with pytest.raises(ValueError, match="limit"):
        runner.build_groups(*inputs, [], limit, 262144)


def test_test_contexts_cannot_be_migrated_from_old_group_policy(runner, inputs):
    rows, questions = inputs
    with pytest.raises(ValueError, match="pre-test"):
        runner.build_groups(rows, questions, [row["record_id"] for row in rows[:25]], 10, 262144)


def test_source_prefix_cannot_be_reordered(runner, inputs):
    rows, questions = inputs
    with pytest.raises(ValueError, match="prefix"):
        runner.build_groups(rows, questions, [row["record_id"] for row in reversed(rows[:5])], 10, 262144)


def test_partial_new_group_cannot_be_selected(runner, inputs):
    rows, questions = inputs
    groups = runner.build_groups(rows, questions, [row["record_id"] for row in rows[:10]], 10, 262144)
    with pytest.raises(ValueError, match="cuts"):
        runner.selected_groups(groups, 10, 15, 30)
    assert len(runner.selected_groups(groups, 10, 20, 30)) == 1


def test_migration_raw_envelopes_and_usage_are_not_rebound_or_rerun(runner, inputs, migration, tmp_path):
    pool, source = migration
    rows, _ = inputs
    packets = {row["record_id"]: runner.protocol.clean_packet(row, prompt=runner.batch.BATCH_PROMPT) for row in rows}
    original = {stream: runner.batch.completed_records(source, stream, packets) for stream in runner.native.STREAMS}
    output, summary = runner.run(tmp_path, pool, source, 20, case_limit=10, payload_limit=262144, executor=backend(runner))
    assert summary["accepted_counts"] == {"a": 20, "b": 20}
    assert summary["migrated_unchanged_records_per_stream"] == 10
    assert summary["new_native_invocations"] == 2
    assert summary["subscription_usage"]["executions"] == 6
    assert summary["subscription_usage"]["native_token_totals"] == {"input_tokens": 600, "output_tokens": 150}
    for stream in runner.native.STREAMS:
        completed = runner.completed_records(output, stream, packets)
        for identifier, envelope in original[stream].items():
            assert completed[identifier] == envelope
            assert runner.record_sources(output, stream)[identifier]["source_run"] == str(source)
    assert summary["human_labels_created"] == 0
    assert runner.previous.validate_paired_inputs(output) == 1
    _, resumed = runner.run(tmp_path, pool, source, 30, case_limit=10, payload_limit=262144,
                            resume=output, executor=backend(runner))
    assert resumed["new_native_invocations"] == 2
    assert resumed["accepted_counts"] == {"a": 30, "b": 30}
    assert resumed["subscription_usage"]["executions"] == 8


def test_mutated_migration_raw_artifact_rejected(runner, inputs, migration, tmp_path):
    pool, source = migration
    output, _ = runner.run(tmp_path, pool, source, 20, case_limit=10, payload_limit=262144, executor=backend(runner))
    path = next((source / "a").glob("batch-*/attempt-*/raw_final.json"))
    path.write_bytes(path.read_bytes() + b" ")
    packets = runner.native.read_json(output / "packets.json")
    with pytest.raises(ValueError, match="changed"):
        runner.completed_records(output, "a", packets)


def test_new_failure_retained_without_automatic_retry(runner, migration, tmp_path):
    pool, source = migration
    output, summary = runner.run(tmp_path, pool, source, 30, case_limit=10, payload_limit=262144,
                                 executor=backend(runner, failure=True))
    assert summary["status"] == "stopped_without_automatic_retry"
    assert summary["accepted_counts"] == {"a": 10, "b": 10}
    assert summary["new_native_invocations"] <= 2
    assert summary["failures"]
    assert list(output.glob("*/batch-*/attempt-*/events.jsonl"))


def test_actual_schema_bytes_are_part_of_the_guard(runner, inputs, tmp_path):
    rows, questions = inputs
    group = runner.build_groups(rows, questions, [], 10, 262144)[0]
    poisoned = deepcopy(group)
    poisoned["schema_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes changed"):
        runner.execute_attempt(tmp_path, "a", poisoned, rows[:10], tmp_path / "native.exe", {}, [], {}, None, 90, backend(runner))
