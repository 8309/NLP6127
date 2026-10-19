"""Mocked native execution and exact-evidence guards; no subscription calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oil_evidence import codex_runtime as runtime


@pytest.fixture
def fake_native(tmp_path, monkeypatch):
    helper, sources = runtime._load_isolation_snapshot()
    native = tmp_path / "fixture.exe"
    native.write_bytes(b"Synthetic native executable, never run")
    helper.discover_native_cli = lambda name: native
    helper.configured_mcp_names = lambda environment: []
    monkeypatch.setattr(runtime, "_load_isolation_snapshot", lambda: (helper, sources))
    def status(command, **options):
        assert options["check"] and "OPENAI_API_KEY" not in options["env"]
        return SimpleNamespace(stdout=b"fixture-cli 1" if "--version" in command else b"Logged in using ChatGPT", stderr=b"")
    monkeypatch.setattr(runtime.subprocess, "run", status)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-a-secret")
    return helper, sources


def executor(output, tool=False):
    def execute(command, prompt, context, directory, environment, timeout):
        assert "read-only" in command and "--ephemeral" in command
        assert "--ignore-rules" not in command
        assert "OPENAI_API_KEY" not in environment
        final = Path(command[command.index("--output-last-message") + 1])
        if isinstance(output, bytes):
            final.write_bytes(output)
        else:
            final.write_text(json.dumps(output), encoding="utf-8")
        events = [dict(type="thread.started", thread_id="fixture-execution"), dict(type="turn.started")]
        if tool:
            events.append(dict(type="item.completed", item=dict(type="command_execution")))
        events.append(dict(type="turn.completed", usage=dict(input_tokens=12, output_tokens=3)))
        (directory / "events.jsonl").write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
        (directory / "stderr.txt").write_bytes(b"")
        return 0
    return execute


SCHEMA = dict(type="object", additionalProperties=False, required=["ok"],
              properties=dict(ok=dict(type="boolean")))


def test_structured_call_has_one_usage_and_source_byte_snapshots(tmp_path, fake_native):
    value, receipt = runtime.call_structured_json("Synthetic task", {"data":"fixture"}, SCHEMA,
        tmp_path / "calls", role="adjudicator", executor=executor({"ok":True}))
    assert value == {"ok":True}
    assert receipt["accepted"] and receipt["codex_cli_invocations"] == 1
    assert receipt["observed_model"] is None
    assert receipt["usage"] == dict(input_tokens=12, output_tokens=3)
    assert receipt["usage_counted_once"] and receipt["subscription_dollar_cost"] is None
    directory = Path(receipt["artifact_directory"])
    for name, sha in receipt["artifacts_sha256"].items():
        assert runtime.digest((directory / name).read_bytes()) == sha
    for name, raw in fake_native[1].items():
        assert (directory / "implementation" / name).read_bytes() == raw


@pytest.mark.parametrize("tool,output", [(True, {"ok":True}), (False, {"ok":"invalid"})])
def test_tool_and_schema_failures_keep_original_attempt(tmp_path, fake_native, tool, output):
    with pytest.raises(runtime.CodexUnavailable, match="retained receipt"):
        runtime.call_structured_json("Synthetic task", {}, SCHEMA, tmp_path / "calls", executor=executor(output, tool))
    attempts = list((tmp_path / "calls").iterdir())
    assert len(attempts) == 1
    attempt = attempts[0]
    assert json.loads((attempt / "raw_final.json").read_bytes()) == output
    assert not json.loads((attempt / "receipt.json").read_bytes())["accepted"]
    assert (attempt / "failure.json").is_file()


def test_external_schema_reference_fails_before_native_call(tmp_path):
    with pytest.raises(ValueError, match="External"):
        runtime.call_structured_json("Task", {}, {"type":"object", "$ref":"https://fixture.invalid/schema"}, tmp_path)


def test_unsupported_unique_items_fails_before_a_native_attempt(tmp_path):
    schema = dict(type="object", properties=dict(values=dict(type="array", uniqueItems=True,
        items=dict(type="integer"))), required=["values"], additionalProperties=False)
    with pytest.raises(ValueError, match="uniqueItems is unsupported"):
        runtime.call_structured_json("Task", {}, schema, tmp_path / "calls")
    assert not (tmp_path / "calls").exists()


def test_object_property_named_unique_items_is_not_a_schema_keyword():
    runtime._check_schema(dict(type="object", properties=dict(uniqueItems=dict(type="boolean")),
                              required=["uniqueItems"], additionalProperties=False))


@pytest.mark.parametrize("raw", [b'{"ok":false,"ok":true}', b'{"ok":NaN}'])
def test_duplicate_and_nonfinite_native_json_rejected(tmp_path, fake_native, raw):
    with pytest.raises(runtime.CodexUnavailable, match="retained receipt"):
        runtime.call_structured_json("Task", {}, SCHEMA, tmp_path / "calls", executor=executor(raw))
    attempt = next((tmp_path / "calls").iterdir())
    assert (attempt / "raw_final.json").read_bytes() == raw


def test_nonfinite_payload_fails_before_execution(tmp_path):
    with pytest.raises(ValueError, match="Out of range"):
        runtime.call_structured_json("Task", {"value":float("nan")}, SCHEMA, tmp_path / "calls")


@pytest.fixture
def evidence():
    return [dict(passage_id="fixture-passage", source_page_id="fixture-page", text="Oil output is a synthetic 10 barrels per day.",
                 available_by="2024-01-01", char_start=0, char_end=43, regions=[])]


@pytest.fixture
def answer():
    return dict(answerability="sufficient", claims=[dict(text="Synthetic oil output is 10 barrels per day.", citations=["fixture-passage"])],
                quotes=[dict(passage_id="fixture-passage", text="10 barrels per day")], limitations=[])


def test_exact_quotes_do_not_mean_semantic_judgment(evidence, answer):
    assert runtime.validate_generation(answer, evidence, "2024-06-30") == []
    answer["claims"][0]["text"] = "An unsupported claim can still have a valid citation ID."
    assert runtime.validate_generation(answer, evidence, "2024-06-30") == []


def test_insufficiency_has_no_answer_claims_but_keeps_exact_background_quote(evidence, answer):
    answer["answerability"] = "insufficient"
    answer["limitations"] = ["The requested export-loss quantity is not measured."]
    assert runtime.validate_generation(answer, evidence, "2024-06-30")
    answer["claims"] = []
    assert not runtime.validate_generation(answer, evidence, "2024-06-30")


@pytest.mark.parametrize("mutation", ["quote", "future", "unknown_id", "empty", "no_quote"])
def test_generation_structural_evidence_errors(evidence, answer, mutation):
    if mutation == "quote":
        answer["quotes"][0]["text"] = "Invented quotation"
    elif mutation == "future":
        evidence[0]["available_by"] = "2025-01-01"
    elif mutation == "unknown_id":
        answer["claims"][0]["citations"] = ["other-case"]
    elif mutation == "empty":
        answer["claims"][0]["text"] = ""
    else:
        answer["quotes"] = []
    assert runtime.validate_generation(answer, evidence, "2024-06-30")


def test_no_evidence_insufficiency_skips_native_generation(tmp_path):
    staged = runtime.StagedRuntime(tmp_path, tmp_path / "stage", "fixture-hash",
        SimpleNamespace(search=lambda *args, **kwargs: []), {}, {}, {})
    result = runtime.query_codex("Synthetic question", staged, as_of="2017-12-31",
        structured_call=lambda *args, **kwargs: pytest.fail("No eligible evidence must not invoke a model"))
    assert result["answerability"] == "insufficient" and result["answerable"] is False
    assert result["model_calls"] == [] and result["generation_seconds"] == 0
    assert result["total_seconds"] >= result["retrieval_seconds"]


def test_per_query_timing_and_single_generator_receipt(tmp_path, evidence, answer):
    original = evidence[0]
    original["char_end"] = len(original["text"])
    hit = dict(original, rank=1, score=1.0)
    staged = runtime.StagedRuntime(tmp_path, tmp_path / "stage", "fixture-hash",
        SimpleNamespace(search=lambda *args, **kwargs: [hit]), {original["passage_id"]:original},
        {original["source_page_id"]:dict(text=original["text"])}, {})
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs["role"])
        return answer, dict(generation_seconds=0.0, observed_model=None, usage=dict(input_tokens=4, output_tokens=2))
    result = runtime.query_codex("Synthetic question", staged, as_of="2024-06-30", structured_call=generate)
    assert calls == ["generator"]
    assert len(result["model_calls"]) == 1 and result["observed_models"] == [None]
    assert result["verification_method"] == "citation_ids_exact_quotes_dates_not_semantic_support"
    assert result["total_seconds"] >= result["retrieval_seconds"]
    assert (Path(result["output_directory"]) / "result.json").is_file()
