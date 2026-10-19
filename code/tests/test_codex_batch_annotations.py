"""Mocked batch annotation tests; no subscription or external API calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("isolated_codex_batch", scripts / "run_codex_batch_annotations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rows():
    return [dict(question_id=f"q-{i}", question=f"What did synthetic oil report {i} observe?",
                 question_zh="Translated fixture", as_of="2024-06-30", type="private-author-hint",
                 evidence=[dict(passage_id=f"p-{i}", document_id=f"d-{i}", text=f"Crude exports were stable in region {i}.",
                                available_by="2024-01-01", source="SYNTHETIC", rank=1)]) for i in range(1, 6)]


def label(i):
    return dict(answerability="sufficient", event_status="observed", qrels={f"p-{i}": 2},
                answer_points=[dict(text=f"Crude exports were stable in region {i}.", citations=[f"p-{i}"])],
                quotes=[dict(passage_id=f"p-{i}", text=f"Crude exports were stable in region {i}.")], reasons=["Exact eligible source."])


def identity(execution_id="fresh-fixture"):
    return dict(service="fixture", requested_model="default-unresolved", observed_model=None,
                execution_id=execution_id, observation_source="unavailable", identity_note="No execution model metadata.")


def raw_labels(values):
    return {qid: dict(record_id=qid, label=value) for qid, value in values.items()}


def test_batch_schema_requires_each_id_and_only_own_evidence(runner, rows):
    packets, stdin, schema = runner.prepare_batch(rows)
    assert set(schema["required"]) == {f"q-{i}" for i in range(1, 6)}
    assert schema["additionalProperties"] is False
    assert b"private-author-hint" not in stdin and b'"rank"' not in stdin
    for i in range(1, 6):
        assert schema["properties"][f"q-{i}"]["properties"]["label"]["properties"]["qrels"]["required"] == [f"p-{i}"]
        assert set(packets[f"q-{i}"]["payload"]) == {"record_id", "question", "as_of", "evidence"}


def test_batch_streams_use_identical_full_input_and_new_protocol(runner, rows):
    a = runner.prepare_batch(deepcopy(rows))
    b = runner.prepare_batch(deepcopy(rows))
    assert a == b
    assert runner.BATCH_VERSION in runner.BATCH_PROMPT
    assert a[0]["q-1"]["prompt_sha256"] != runner.native.packet_and_prompt(rows[0])[0]["prompt_sha256"]


def test_full_raw_bytes_and_own_key_extraction_retained(runner, rows):
    packets, stdin, _ = runner.prepare_batch(rows)
    raw = json.dumps(raw_labels({f"q-{i}": label(i) for i in range(1, 6)}), indent=2) + "\r\n"
    outputs = runner.extract_outputs(raw, packets, identity(), dict(input_tokens=100), stdin, {})
    for i in range(1, 6):
        output = outputs[f"q-{i}"]
        assert output["raw_output"] == raw
        assert output["raw_label_path"] == [f"q-{i}", "label"]
        assert output["raw_record_id_path"] == [f"q-{i}", "record_id"]
        assert output["label"] == label(i)
        assert output["usage"] is None
        assert output["usage_reference_execution_id"] == "fresh-fixture"
        assert output["execution_input_sha256"] == runner.protocol.sha256(stdin)
        assert output["validation_errors"] == []


def test_other_record_evidence_cannot_be_borrowed(runner, rows):
    packets, stdin, _ = runner.prepare_batch(rows)
    values = {f"q-{i}": label(i) for i in range(1, 6)}
    values["q-1"] = label(2)
    outputs = runner.extract_outputs(json.dumps(raw_labels(values)), packets, identity(), None, stdin, {})
    assert outputs["q-1"]["validation_errors"]
    assert outputs["q-2"]["validation_errors"] == []


@pytest.mark.parametrize("mutation", ["future", "wrong_quote", "missing_qrel"])
def test_own_cutoff_exact_quote_and_qrel_coverage_checked(runner, rows, mutation):
    values = {f"q-{i}": label(i) for i in range(1, 6)}
    if mutation == "future":
        rows[0]["evidence"][0]["available_by"] = "2025-01-01"
    elif mutation == "wrong_quote":
        values["q-1"]["quotes"][0]["text"] = "Not in the source"
    else:
        values["q-1"]["qrels"] = {}
    packets, stdin, _ = runner.prepare_batch(rows)
    assert runner.extract_outputs(json.dumps(raw_labels(values)), packets, identity(), None, stdin, {})["q-1"]["validation_errors"]


def test_missing_extra_and_duplicate_batch_ids_rejected(runner, rows):
    packets, stdin, _ = runner.prepare_batch(rows)
    with pytest.raises(ValueError, match="coverage"):
        runner.extract_outputs(json.dumps({"q-1": label(1)}), packets, identity(), None, stdin, {})
    with pytest.raises(ValueError, match="Duplicate JSON"):
        runner.extract_outputs('{"q-1":{},"q-1":{}}', packets, identity(), None, stdin, {})
    with pytest.raises(ValueError, match="Duplicate batch"):
        runner.prepare_batch([rows[0], deepcopy(rows[0])])


def test_usage_counted_once_per_execution_not_five_labels(runner, tmp_path):
    event = dict(execution_id="fresh-a", usage=dict(input_tokens=100, output_tokens=20), record_ids=["q-1", "q-2", "q-3", "q-4", "q-5"])
    runner.native.append_progress(tmp_path / "execution_usage.jsonl", event)
    summary = runner.observed_usage(tmp_path)
    assert summary["executions"] == 1
    assert summary["native_token_totals"] == {"input_tokens": 100, "output_tokens": 20}


def test_mocked_five_record_pair_and_no_call_resume(runner, rows, tmp_path, monkeypatch):
    root = tmp_path / "c"
    pool = root / "p"
    pool.mkdir(parents=True)
    runner.native.write_new(pool / "annotation_pool.jsonl", b"fixture-input")
    protected = pool / "protected.json"
    runner.native.write_new(protected, b"original")
    binary = tmp_path / "codex.exe"
    runner.native.write_new(binary, b"fixture-only")
    monkeypatch.setattr(runner.native, "load_pool", lambda selected_root, selected_pool: (
        pool, rows, {str(protected): runner.protocol.sha256(protected.read_bytes())}))
    monkeypatch.setattr(runner.native, "discover_native_cli", lambda requested: binary)
    monkeypatch.setattr(runner.native, "configured_mcp_names", lambda environment: [])
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: SimpleNamespace(
        stdout=b"fixture-version\n" if command[-1] == "--version" else b"Logged in using ChatGPT\n", stderr=b""))
    contexts, inputs = [], []
    def mock_execution(command, stdin, context, attempt, environment, timeout):
        contexts.append(context)
        inputs.append(stdin)
        assert set(path.name for path in context.iterdir()) == {"packets.json", "protocol.txt", "schema.json"}
        raw = json.dumps(raw_labels({f"q-{i}": label(i) for i in range(1, 6)}), indent=2).encode("utf-8") + b"\r\n"
        runner.native.write_new(Path(command[command.index("--output-last-message") + 1]), raw)
        events = [dict(type="thread.started", thread_id=f"context-{len(contexts)}"),
                  dict(type="turn.completed", usage=dict(input_tokens=100, output_tokens=20))]
        runner.native.write_new(attempt / "events.jsonl", b"".join(runner.protocol.canonical_bytes(event) + b"\n" for event in events))
        runner.native.write_new(attempt / "stderr.txt", b"")
        return 0
    output, summary = runner.run(root, pool, executor=mock_execution)
    assert summary["accepted_counts"] == {"a": 5, "b": 5}
    assert summary["new_codex_cli_invocations"] == 2
    assert summary["usage"]["native_token_totals"] == {"input_tokens": 200, "output_tokens": 40}
    assert contexts[0] != contexts[1] and inputs[0] == inputs[1]
    before = (output / "progress.jsonl").read_bytes()
    _, resumed = runner.run(root, pool, resume=output, executor=mock_execution)
    assert resumed["new_codex_cli_invocations"] == 0
    assert (output / "progress.jsonl").read_bytes() == before
    assert len(contexts) == 2
