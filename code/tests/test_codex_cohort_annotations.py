"""Mocked native execution tests; no real model, subscription or QA count use."""
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
    spec = importlib.util.spec_from_file_location("cohort_annotation_test", scripts / "run_codex_cohort_annotations.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def inputs():
    rows, queries, corpus = [], [], []
    for index in range(10):
        record_id = f"development-fixture-{index:02d}"
        evidence = dict(passage_id=f"synthetic-passage-{index}", document_id=f"synthetic-document-{index}",
                        text=f"Synthetic region {index} reported stable crude oil exports.", available_by="2024-01-01")
        rows.append(dict(record_id=record_id, question=f"What did synthetic region {index} report about crude exports?",
                         as_of="2024-03-01", evidence=[evidence]))
        queries.append(dict(record_id=record_id, question=rows[-1]["question"], as_of=rows[-1]["as_of"],
                            event_group=f"synthetic-family-{index}", split="development" if index < 5 else "test"))
        corpus.append(evidence)
    return rows, queries, corpus


def mock_backend(runner, fail=False):
    def execute(command, stdin, context, attempt, environment, timeout):
        packets = json.loads((context / "packets.json").read_bytes())
        values = {}
        for identifier, row in packets.items():
            evidence = row["evidence"][0]
            values[identifier] = dict(record_id=identifier, label=dict(answerability="sufficient", event_status="observed",
                qrels={evidence["passage_id"]: 2}, answer_points=[dict(text=evidence["text"], citations=[evidence["passage_id"]])],
                quotes=[dict(passage_id=evidence["passage_id"], text=evidence["text"])], reasons=["Synthetic exact eligible quote."]))
        events = [dict(type="thread.started", thread_id="synthetic-execution-" + uuid4().hex), dict(type="turn.started")]
        if fail:
            events.extend([dict(type="error", message="Synthetic quota failure"), dict(type="turn.failed")])
        else:
            events.append(dict(type="turn.completed", usage=dict(input_tokens=100, output_tokens=25)))
            (context / "final.json").write_bytes((json.dumps(values, indent=2) + "\r\n").encode())
        (attempt / "events.jsonl").write_bytes(b"".join(runner.protocol.canonical_bytes(event) + b"\n" for event in events))
        (attempt / "stderr.txt").write_bytes(b"")
        return 1 if fail else 0
    return execute


def test_pool_membership_original_content_and_blinding(runner, inputs):
    rows, questions, corpus = inputs
    assert runner.validate_pool_rows(rows, questions, corpus) == rows
    poisoned = deepcopy(rows)
    poisoned[0]["rank"] = 1
    with pytest.raises(ValueError, match="hints"):
        runner.validate_pool_rows(poisoned, questions, corpus)
    poisoned = deepcopy(rows)
    poisoned[0]["evidence"][0]["text"] = "Altered source"
    with pytest.raises(ValueError, match="corpus"):
        runner.validate_pool_rows(poisoned, questions, corpus)


@pytest.mark.parametrize("hint", ["rank", "score", "anchor", "answerability", "qrels"])
def test_evidence_hints_not_silently_projected_out(runner, inputs, hint):
    rows, questions, corpus = inputs
    rows[0]["evidence"][0][hint] = 1
    with pytest.raises(ValueError, match="hints"):
        runner.validate_pool_rows(rows, questions, corpus)


def test_explicit_new_development_excludes_old50_and_test(runner, inputs):
    rows, questions, _ = inputs
    old = dict(rows[0], record_id="dev50-001")
    old_query = dict(questions[0], record_id="dev50-001")
    eligible, selected = runner.select_rows([old] + rows, [old_query] + questions, "new_development", 5)
    assert len(eligible) == len(selected) == 5
    assert all(not row["record_id"].startswith("dev50-") for row in selected)


def test_explicit_all_current_fresh_labels_old50_under_changed_top5_inputs(runner, inputs):
    rows, questions, _ = inputs
    old = dict(rows[0], record_id="dev50-001")
    old_query = dict(questions[0], record_id="dev50-001")
    eligible, selected = runner.select_rows([old] + rows, [old_query] + questions, "all_current", 11)
    assert len(eligible) == len(selected) == 11
    assert selected[0]["record_id"] == "dev50-001"


def test_synthetic_full_bank_partition_sizes_not_real_question_counts(runner):
    # Bookkeeping-only synthetic IDs: these are NOT authored questions or labels.
    rows, questions = [], []
    for index in range(1200):
        identifier = f"dev50-{index + 1:03d}" if index < 50 else f"synthetic-q-{index:04d}"
        rows.append(dict(record_id=identifier))
        questions.append(dict(record_id=identifier, split="development" if index < 200 else "test"))
    assert len(runner.select_rows(rows, questions, "all_current", 1200)[1]) == 1200
    assert len(runner.select_rows(rows, questions, "all_new", 1150)[1]) == 1150
    assert len(runner.select_rows(rows, questions, "new_development", 150)[1]) == 150


@pytest.mark.parametrize("count", [0, True, 3, 11])
def test_stable_five_record_limit_never_regroups_on_resume(runner, inputs, count):
    with pytest.raises(ValueError, match="limit|partition"):
        runner.select_rows(*inputs[:2], "all_new", count)


def test_mocked_execution_retains_whole_raw_and_once_per_call_usage(runner, inputs, tmp_path):
    rows, _, _ = inputs
    output = tmp_path / "output"
    output.mkdir()
    binary = tmp_path / "native.exe"
    binary.write_bytes(b"synthetic-native")
    result = runner.execute_attempt(output, "a", 1, rows[:5], binary, {}, [], {}, None, 90, mock_backend(runner))
    assert result["receipt"]["accepted"]
    assert result["receipt"]["usage"] == {"input_tokens": 100, "output_tokens": 25}
    for envelope in result["envelopes"].values():
        assert envelope["usage"] is None
        assert envelope["identity"]["observed_model"] is None
        assert envelope["raw_output"].encode() == (result["attempt"] / "raw_final.json").read_bytes()
        assert envelope["execution_input_sha256"] == runner.protocol.sha256((result["attempt"] / "stdin.txt").read_bytes())


def test_mocked_native_failure_retained_not_accepted(runner, inputs, tmp_path):
    rows, _, _ = inputs
    result = runner.execute_attempt(tmp_path, "a", 1, rows[:5], tmp_path / "native.exe", {}, [], {}, None, 90,
                                    mock_backend(runner, fail=True))
    assert result["receipt"]["accepted"] is False
    assert result["usage"]["usage"] is None
    assert (result["attempt"] / "events.jsonl").exists()


def test_paired_run_resume_identical_inputs_fresh_contexts_and_usage(runner, inputs, monkeypatch, tmp_path):
    rows, questions, _ = inputs
    pool = tmp_path / "pool"
    pool.mkdir()
    binary = tmp_path / "native.exe"
    binary.write_bytes(b"synthetic-native")
    monkeypatch.setattr(runner, "load_pool", lambda root, path: (pool, rows, questions, {}, {"synthetic": True}))
    monkeypatch.setattr(runner.native, "discover_native_cli", lambda cli: binary)
    monkeypatch.setattr(runner.native, "configured_mcp_names", lambda environment: [])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: SimpleNamespace(
        stdout=b"Logged in using ChatGPT" if "login" in command else b"synthetic-cli", stderr=b""))
    output, summary = runner.run(tmp_path, pool, 5, partition="all_new", executor=mock_backend(runner))
    assert summary["accepted_counts"] == {"a": 5, "b": 5}
    assert summary["paired_batch_count"] == 1
    assert summary["subscription_usage"]["executions"] == 2
    assert summary["subscription_usage"]["native_token_totals"]["input_tokens"] == 200
    assert summary["formal_test_frozen"] is False
    _, resumed = runner.run(tmp_path, pool, 10, partition="all_new", resume=output, executor=mock_backend(runner))
    assert resumed["accepted_counts"] == {"a": 10, "b": 10}
    assert resumed["new_native_invocations"] == 2
    assert resumed["paired_batch_count"] == 2
    assert resumed["subscription_usage"]["executions"] == 4


def test_run_stops_without_retry_after_failure(runner, inputs, monkeypatch, tmp_path):
    rows, questions, _ = inputs
    pool = tmp_path / "pool"
    pool.mkdir()
    binary = tmp_path / "native.exe"
    binary.write_bytes(b"synthetic-native")
    monkeypatch.setattr(runner, "load_pool", lambda root, path: (pool, rows, questions, {}, {}))
    monkeypatch.setattr(runner.native, "discover_native_cli", lambda cli: binary)
    monkeypatch.setattr(runner.native, "configured_mcp_names", lambda environment: [])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: SimpleNamespace(
        stdout=b"Logged in using ChatGPT" if "login" in command else b"synthetic-cli", stderr=b""))
    _, summary = runner.run(tmp_path, pool, 10, partition="all_new", executor=mock_backend(runner, fail=True))
    assert summary["status"] == "stopped_without_automatic_retry"
    assert summary["new_native_invocations"] <= 2
    assert summary["accepted_counts"] == {"a": 0, "b": 0}
    assert summary["human_gold"] is False
