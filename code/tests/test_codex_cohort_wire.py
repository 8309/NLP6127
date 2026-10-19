"""Wire driver tests use synthetic sources; no model/API/subscription calls."""
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("wire_driver_test", scripts / "run_codex_cohort_wire.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    rows, questions = [], []
    for index in range(20):
        identifier = f"synthetic-canonical-q-{index}"
        evidence = dict(passage_id=f"synthetic-long-canonical-passage-{index}", document_id=f"synthetic-document-{index}",
                        text=f"Synthetic region {index} reported stable crude exports.", available_by="2024-01-01")
        rows.append(dict(record_id=identifier, question=f"What did synthetic region {index} report?", as_of="2024-02-01", evidence=[evidence]))
        questions.append(dict(record_id=identifier, split="development" if index < 10 else "test"))
    return rows, questions


def backend(runner, fail=False):
    def execute(command, stdin, context, attempt, environment, timeout):
        payload = runner.native.read_json(context / "packets.json")
        values = {}
        for alias, row in payload.items():
            evidence = row["evidence"][0]
            values[alias] = dict(record_id=alias, label=dict(answerability="sufficient", event_status="observed",
                qrels={evidence["passage_id"]: 2}, answer_points=[dict(text=evidence["text"], citations=[evidence["passage_id"]])],
                quotes=[dict(passage_id=evidence["passage_id"], text=evidence["text"])], reasons=["Synthetic own-source quote."]))
        events = [dict(type="thread.started", thread_id="synthetic-execution-" + uuid4().hex)]
        if fail:
            events.extend([dict(type="error", message="Synthetic quota failure"), dict(type="turn.failed")])
        else:
            events.append(dict(type="turn.completed", usage=dict(input_tokens=100, output_tokens=25)))
            (context / "final.json").write_bytes((json.dumps(values, indent=2) + "\r\n").encode())
        (attempt / "events.jsonl").write_bytes(b"".join(runner.protocol.canonical_bytes(value) + b"\n" for value in events))
        (attempt / "stderr.txt").write_bytes(b"")
        return 1 if fail else 0
    return execute


def test_groups_keep_full_payload_source_order_and_dev_test_boundary(runner, inputs):
    rows, questions = inputs
    groups = runner.build_groups(rows, questions, [], 10, 262144)
    assert [value["split"] for value in groups] == ["development", "test"]
    assert [identifier for value in groups for identifier in value["record_ids"]] == [row["record_id"] for row in rows]
    assert all(value["full_payload_bytes"] <= 262144 for value in groups)


def test_oversized_complete_case_aborts_without_truncation(runner, inputs):
    with pytest.raises(ValueError, match="Single full"):
        runner.build_groups(*inputs, [], 10, 1)


def test_test_records_never_migrate_old_grouping(runner, inputs):
    rows, questions = inputs
    with pytest.raises(ValueError, match="pre-test"):
        runner.build_groups(rows, questions, [row["record_id"] for row in rows[:15]])


def test_native_raw_alias_response_kept_exact_and_canonical_hash_separate(runner, inputs, tmp_path):
    rows, questions = inputs
    group = runner.build_groups(rows, questions, [], 10, 262144)[0]
    result = runner.execute_attempt(tmp_path, "a", group, rows[:10], tmp_path / "native.exe", {}, [], {}, None, 90, backend(runner))
    assert result["receipt"]["accepted"]
    assert result["receipt"]["usage"] == {"input_tokens": 100, "output_tokens": 25}
    raw = (result["attempt"] / "raw_final.json").read_bytes()
    assert raw.endswith(b"\r\n")
    packets = {row["record_id"]: runner.protocol.clean_packet(row, prompt=runner.batch.BATCH_PROMPT) for row in rows[:10]}
    assert runner.verify_attempt(result["attempt"], result["receipt"], packets) == result["envelopes"]
    for identifier, value in result["envelopes"].items():
        assert value["raw_output"].encode() == raw
        assert value["input_sha256"] == packets[identifier]["input_sha256"]
        assert value["execution_input_sha256"] != value["input_sha256"]
        assert value["usage"] is None
        assert value["identity"]["observed_model"] is None
        assert value["canonical_output_is_raw_model_output"] is False
        assert value["raw_label_path"][0].startswith("r")


@pytest.mark.parametrize("artifact", ["codec.json", "wire_payload.json", "stdin.txt", "schema.json"])
def test_map_input_or_schema_tampering_fails_full_reconstruction(runner, inputs, tmp_path, artifact):
    rows, questions = inputs
    group = runner.build_groups(rows, questions, [], 10, 262144)[0]
    result = runner.execute_attempt(tmp_path, "a", group, rows[:10], tmp_path / "native.exe", {}, [], {}, None, 90, backend(runner))
    path = result["attempt"] / artifact
    if artifact == "codec.json":
        value = runner.native.read_json(path)
        value["records"]["r0"]["passage_ids"]["r0p0"] = "wrong-source"
        path.write_bytes(runner.protocol.canonical_bytes(value))
    elif artifact == "wire_payload.json":
        value = runner.native.read_json(path)
        value["r0"]["evidence"][0]["text"] += " Altered source"
        path.write_bytes(runner.protocol.canonical_bytes(value))
    else:
        path.write_bytes(path.read_bytes() + b" ")
    packets = {row["record_id"]: runner.protocol.clean_packet(row, prompt=runner.batch.BATCH_PROMPT) for row in rows[:10]}
    with pytest.raises(ValueError, match="changed|differs"):
        runner.verify_attempt(result["attempt"], result["receipt"], packets)


def test_quota_failure_is_retained_without_semantic_label_fabrication(runner, inputs, tmp_path):
    rows, questions = inputs
    group = runner.build_groups(rows, questions, [], 10, 262144)[0]
    result = runner.execute_attempt(tmp_path, "a", group, rows[:10], tmp_path / "native.exe", {}, [], {}, None, 90, backend(runner, True))
    assert result["receipt"]["accepted"] is False
    assert result["envelopes"] == {}
    assert result["usage"]["usage"] is None
    assert (result["attempt"] / "events.jsonl").exists()
