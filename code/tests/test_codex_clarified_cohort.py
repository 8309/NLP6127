"""Future-format clarification fixtures; no actual native/model calls."""
import importlib.util
from pathlib import Path

import pytest

from test_correct_codex_wire_batch import backend


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("clarified_cohort_test", scripts / "run_codex_clarified_cohort.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    rows = [dict(record_id=f"synthetic-original-{index}", question=f"What crude flow did synthetic region {index} report?",
        as_of="2024-02-01", evidence=[dict(passage_id=f"synthetic-source-{index}", document_id=f"synthetic-doc-{index}",
            text=f"Synthetic region {index} reported 5 barrels of crude flow.", available_by="2024-01-01")]) for index in range(25)]
    questions = [dict(record_id=row["record_id"], split="test" if index < 15 else "development") for index, row in enumerate(rows)]
    return rows, questions


def mock_manifest(runner, directory, group):
    context = directory / "synthetic-old-context"
    command = runner.native.cli_command(directory / "synthetic-native.exe", context, context / "schema.json", context / "final.json", None, [])
    runner.native.write_new(directory / "manifest.json", dict(specification=dict(version=runner.VERSION, requested_model=None,
        native_command_identity=runner.clarification.normalized_command(command), groups=[group])))


def test_grouping_preserves_oldprefix_future_order_and_split_boundaries(runner, inputs):
    rows, questions = inputs
    groups = runner.build_groups(rows, questions, [r["record_id"] for r in rows[:5]])
    assert [g["split"] for g in groups] == ["test", "development"]
    assert [identifier for g in groups for identifier in g["record_ids"]] == [r["record_id"] for r in rows[5:]]
    assert all(len(g["record_ids"]) <= 10 and g["full_payload_bytes"] <= 262144 for g in groups)
    assert all(g["format_instruction_sha256"] == runner.INSTRUCTION_SHA256 for g in groups)


def test_neutral_bytebound_reduces_cases_without_truncation(runner, inputs):
    rows, questions = inputs
    for row in rows:
        row["evidence"][0]["text"] += "x" * 40000
    groups = runner.build_groups(rows, questions, [])
    assert any(len(g["record_ids"]) < 10 for g in groups)
    assert [identifier for g in groups for identifier in g["record_ids"]] == [r["record_id"] for r in rows]
    assert all(g["full_payload_bytes"] <= 262144 for g in groups)


def test_oversize_single_source_aborts_not_drops(runner, inputs):
    rows, questions = inputs
    rows[0]["evidence"][0]["text"] += "x" * 262144
    with pytest.raises(ValueError, match="byte bound"):
        runner.build_groups(rows, questions, [])


def test_history_prefix_cannot_be_reordered_or_selected_by_labels(runner, inputs):
    rows, questions = inputs
    with pytest.raises(ValueError, match="unchanged source-order prefix"):
        runner.build_groups(rows, questions, [rows[2]["record_id"]])


def test_actual_future_pair_fresh_same_input_and_originalfirst_role(runner, inputs, tmp_path):
    rows, questions = inputs
    group = runner.build_groups(rows, questions, [])[0]
    packets = {row["record_id"]: runner.protocol.clean_packet(row, prompt=runner.wire.batch.BATCH_PROMPT) for row in rows}
    calls, outputs = [], {}
    mock_manifest(runner, tmp_path, group)
    for stream in ("a", "b"):
        value = runner.execute_attempt(tmp_path, stream, group, packets, tmp_path / "synthetic-native.exe", {}, [], {}, None, 90,
            backend(runner.clarification, calls))
        assert value["receipt"]["accepted"]
        assert runner.verify_attempt(value["attempt"], value["receipt"], packets) == value["envelopes"]
        outputs[stream] = value["envelopes"]
    assert len(calls) == 2 and calls[0][1] == calls[1][1] and calls[0][2] != calls[1][2]
    for stream in outputs.values():
        for value in stream.values():
            assert value["annotation_role"] == "format_clarified_first_attempt"
            assert value["original_first_attempt"] is True
            assert value["prompt_sha256"] == packets[value["record_id"]]["prompt_sha256"]
    agreement = runner.protocol.raw_agreement({rid: packets[rid] for rid in group["record_ids"]}, list(outputs["a"].values()), list(outputs["b"].values()))
    assert agreement["paired_records"] == 10


def test_future_does_not_weaken_missing_quote_guard(runner, inputs, tmp_path):
    rows, questions = inputs
    group = runner.build_groups(rows, questions, [])[0]
    packets = {row["record_id"]: runner.protocol.clean_packet(row, prompt=runner.wire.batch.BATCH_PROMPT) for row in rows}
    mock_manifest(runner, tmp_path, group)
    value = runner.execute_attempt(tmp_path, "a", group, packets, tmp_path / "synthetic-native.exe", {}, [], {}, None, 90,
        backend(runner.clarification, [], bad_quote=True))
    assert not value["receipt"]["accepted"]
    assert value["envelopes"] == {}
    assert any("exact eligible quotes" in error for error in value["receipt"]["errors"])


def test_changed_actual_future_input_fails_readback(runner, inputs, tmp_path):
    rows, questions = inputs
    group = runner.build_groups(rows, questions, [])[0]
    packets = {row["record_id"]: runner.protocol.clean_packet(row, prompt=runner.wire.batch.BATCH_PROMPT) for row in rows}
    mock_manifest(runner, tmp_path, group)
    value = runner.execute_attempt(tmp_path, "a", group, packets, tmp_path / "synthetic-native.exe", {}, [], {}, None, 90,
        backend(runner.clarification, []))
    (value["attempt"] / "stdin.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.verify_attempt(value["attempt"], value["receipt"], packets)
