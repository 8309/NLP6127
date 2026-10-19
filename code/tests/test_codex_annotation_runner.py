"""Mocked native-CLI annotation tests; never consume subscription or API usage."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def adapter():
    spec = importlib.util.spec_from_file_location("isolated_codex_annotation", Path(__file__).parents[1] / "scripts/run_codex_annotation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def row():
    return dict(question_id="q-1", question="What did the synthetic oil report observe?", question_zh="Translated fixture",
                as_of="2024-06-30", evidence=[dict(passage_id="p-1", document_id="d-1", text="Crude exports were stable.",
                    available_by="2024-01-01", source="SYNTHETIC", rank=7, mode="hybrid", score=1.0)])


def event_bytes(values):
    return b"".join(json.dumps(value).encode("utf-8") + b"\n" for value in values)


def test_packet_removes_design_and_rank_fields(adapter, row):
    row.update(type="insufficient", event_group="revealing-author-event", split="development", anchors=["hint"])
    packet, prompt = adapter.packet_and_prompt(row)
    assert set(packet["payload"]) == {"record_id", "question", "as_of", "evidence"}
    assert not {"rank", "mode", "score"}.intersection(packet["payload"]["evidence"][0])
    assert b"revealing-author-event" not in prompt
    assert b"hint" not in prompt
    assert packet["prompt_sha256"] == adapter.protocol.sha256((adapter.GUARD + "\n" + adapter.protocol.ANNOTATION_PROMPT).encode("utf-8"))


def test_a_b_identical_packets_and_actual_stdin_hash(adapter, row):
    left, left_prompt = adapter.packet_and_prompt(deepcopy(row))
    right, right_prompt = adapter.packet_and_prompt(deepcopy(row))
    assert left == right and left_prompt == right_prompt
    assert adapter.protocol.sha256(left_prompt) != left["input_sha256"]
    parsed = json.loads(left_prompt.split(b"UNTRUSTED_EVIDENCE_PACKET_JSON\n")[1])
    assert parsed == left["payload"]


def test_schema_requires_every_pooled_qrel(adapter, row):
    packet, _ = adapter.packet_and_prompt(row)
    schema = adapter.output_schema(packet)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["qrels"]["required"] == ["p-1"]
    assert schema["properties"]["qrels"]["additionalProperties"] is False
    assert schema["properties"]["answer_points"]["items"]["properties"]["citations"]["items"]["enum"] == ["p-1"]


def test_read_only_command_preserves_rules_and_default_model(adapter, tmp_path):
    command = adapter.cli_command(Path("codex.exe"), tmp_path, tmp_path / "schema.json", tmp_path / "final.json")
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--model" not in command
    for forbidden in ("--ignore-rules", "--ignore-user-config", "--dangerously-bypass-approvals-and-sandbox", "--full-auto", "resume", "--add-dir"):
        assert forbidden not in command
    assert "features.shell_tool=false" in command
    assert "features.unified_exec=false" in command
    assert "features.apps=false" in command
    assert "features.multi_agent=false" in command
    assert "--strict-config" in command
    assert "tools.view_image=false" not in command
    assert command[-1] == "-"


def test_explicit_model_is_not_substituted(adapter, tmp_path):
    command = adapter.cli_command("codex.exe", tmp_path, tmp_path / "schema.json", tmp_path / "final.json", "requested-fixture")
    assert command[command.index("--model") + 1] == "requested-fixture"


def test_inherited_mcp_servers_disabled_without_disabling_rules(adapter, tmp_path):
    settings = tmp_path / "config.toml"
    settings.write_text('[mcp_servers.fixture]\ncommand="fixture-only"\n', encoding="utf-8")
    before = settings.read_bytes()
    names = adapter.configured_mcp_names({"CODEX_HOME": str(tmp_path)})
    assert names == ["fixture"]
    command = adapter.cli_command("codex.exe", tmp_path, tmp_path / "schema.json", tmp_path / "final.json", mcp_names=names)
    assert "mcp_servers.fixture.enabled=false" in command
    assert "--ignore-rules" not in command
    assert settings.read_bytes() == before
    assert not (tmp_path / "auth.json").exists()


def test_unsafe_mcp_override_key_fails_closed(adapter, tmp_path):
    with pytest.raises(ValueError, match="MCP server name"):
        adapter.cli_command("codex.exe", tmp_path, tmp_path / "schema.json", tmp_path / "final.json", mcp_names=["name.with.dot"])


def test_child_environment_omits_external_api_keys(adapter):
    source = dict(OIL_LLM_API_KEY="fixture-only", OPENAI_API_KEY="fixture-only", CODEX_API_KEY="fixture-only", PATH="fixture-path")
    assert adapter.child_environment(source) == {"PATH": "fixture-path"}
    assert len(source) == 4


def test_native_events_usage_does_not_accept_model_self_report(adapter):
    events = [dict(type="thread.started", thread_id="fresh-context"),
              dict(type="item.completed", item=dict(type="agent_message", text="My model is fictitious-self-report")),
              dict(type="turn.completed", usage=dict(input_tokens=123, output_tokens=42))]
    parsed = adapter.parse_execution_events(event_bytes(events))
    assert parsed["errors"] == []
    assert parsed["observed_model"] is None
    assert parsed["thread_id"] == "fresh-context"
    assert parsed["usage"] == {"input_tokens": 123, "output_tokens": 42}


def test_observed_model_only_from_execution_event_metadata(adapter):
    events = [dict(type="thread.started", thread_id="fresh-context", model="actual-fixture-metadata"), dict(type="turn.completed")]
    parsed = adapter.parse_execution_events(event_bytes(events))
    assert parsed["errors"] == []
    assert parsed["observed_model"] == "actual-fixture-metadata"
    assert parsed["usage"] is None


@pytest.mark.parametrize("item_type", ["command_execution", "mcp_tool_call", "web_search", "file_change", "unknown-tool"])
def test_any_model_tool_event_invalidates_attempt(adapter, item_type):
    events = [dict(type="thread.started", thread_id="fresh"), dict(type="item.started", item=dict(type=item_type)), dict(type="turn.completed")]
    assert "CLI used a prohibited tool or external context" in adapter.parse_execution_events(event_bytes(events))["errors"]


@pytest.mark.parametrize("events", [[], [dict(type="thread.started", thread_id="fresh")], [dict(type="turn.completed")], [dict(type="error"), dict(type="turn.completed")]])
def test_incomplete_or_failed_stream_not_accepted(adapter, events):
    assert adapter.parse_execution_events(event_bytes(events))["errors"]


def test_non_json_native_output_not_hidden(adapter):
    assert adapter.parse_execution_events(b"not-json\n")["errors"]


def test_item_error_is_not_falsely_reported_as_model_tool_use(adapter):
    events = [dict(type="thread.started", thread_id="fresh"),
              dict(type="item.completed", item=dict(type="error", message="Optional integration startup failed")),
              dict(type="turn.completed")]
    errors = adapter.parse_execution_events(event_bytes(events))["errors"]
    assert errors == ["CLI emitted an item error; this is not evidence of model tool execution"]


def test_exclusive_outputs_and_append_only_progress(adapter, tmp_path):
    path = tmp_path / "raw.json"
    raw = b'{"original":"bytes"}\r\n'
    adapter.write_new(path, raw)
    assert path.read_bytes() == raw
    with pytest.raises(FileExistsError):
        adapter.write_new(path, b"overwrite")
    progress = tmp_path / "progress.jsonl"
    adapter.append_progress(progress, dict(type="started"))
    before = progress.read_bytes()
    adapter.append_progress(progress, dict(type="completed"))
    assert progress.read_bytes().startswith(before)


def test_resume_hash_guards_detect_changed_input(adapter, tmp_path):
    spec = {"fixed": "binding"}
    adapter.write_new(tmp_path / "packets.json", b"original")
    adapter.write_new(tmp_path / "manifest.json", dict(specification=spec, immutable_sha256={"packets.json": adapter.protocol.sha256(b"original")}))
    adapter.verify_resume(tmp_path, spec)
    with pytest.raises(ValueError, match="bindings"):
        adapter.verify_resume(tmp_path, {"fixed": "changed"})
    (tmp_path / "packets.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        adapter.verify_resume(tmp_path, spec)


def test_protected_files_checked_before_resume(adapter, tmp_path):
    path = tmp_path / "protected.json"
    adapter.write_new(path, b"original")
    expected = {str(path): adapter.protocol.sha256(b"original")}
    adapter.protected_unchanged(expected)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Protected input changed"):
        adapter.protected_unchanged(expected)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/private", "sub\\file", "", "sub/./file"])
def test_unsafe_artifact_paths_rejected(adapter, tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        adapter.contained_file(tmp_path, name)


def test_mocked_two_fresh_streams_resume_without_new_calls(adapter, row, tmp_path, monkeypatch):
    root = tmp_path / "canonical"
    pool = root / "data/development_runs/fixture/retrieval/run-one"
    pool.mkdir(parents=True)
    adapter.write_new(pool / "annotation_pool.jsonl", adapter.protocol.canonical_bytes(row) + b"\n")
    protected = pool / "protected-fixture.json"
    adapter.write_new(protected, b"original")
    native = tmp_path / "codex.exe"
    adapter.write_new(native, b"not-an-executable-fixture")
    monkeypatch.setattr(adapter, "load_pool", lambda selected_root, selected_pool: (
        pool, [row], {str(protected): adapter.protocol.sha256(protected.read_bytes())}))
    monkeypatch.setattr(adapter, "discover_native_cli", lambda requested: native)
    monkeypatch.setattr(adapter, "configured_mcp_names", lambda environment: ["fixture"])
    def mocked_subprocess(command, **kwargs):
        return SimpleNamespace(stdout=b"codex-fixture-version\n" if command[-1] == "--version" else b"Logged in using ChatGPT\n", stderr=b"")
    monkeypatch.setattr(adapter.subprocess, "run", mocked_subprocess)
    contexts, prompts = [], []
    label = dict(answerability="sufficient", event_status="observed", qrels={"p-1": 2},
                 answer_points=[dict(text="Crude exports were stable.", citations=["p-1"])],
                 quotes=[dict(passage_id="p-1", text="Crude exports were stable.")], reasons=["Exact eligible source statement."])
    def mocked_execution(command, prompt, context, raw_dir, environment, timeout_seconds):
        contexts.append(context)
        prompts.append(prompt)
        assert set(path.name for path in context.iterdir()) == {"packet.json", "protocol.txt", "schema.json"}
        assert not adapter.EXTERNAL_KEY_NAMES.intersection(environment)
        final = Path(command[command.index("--output-last-message") + 1])
        adapter.write_new(final, adapter.protocol.canonical_bytes(label) + b"\r\n")
        adapter.write_new(raw_dir / "events.jsonl", event_bytes([
            dict(type="thread.started", thread_id=f"fresh-fixture-{len(contexts)}"),
            dict(type="item.completed", item=dict(type="agent_message", text="ignored self-report")),
            dict(type="turn.completed", usage=dict(input_tokens=10, output_tokens=5))]))
        adapter.write_new(raw_dir / "stderr.txt", b"")
        return 0
    output, summary = adapter.run(root, pool, executor=mocked_execution)
    assert summary["accepted_counts"] == {"a": 1, "b": 1}
    assert summary["new_codex_cli_invocations"] == 2
    assert contexts[0] != contexts[1] and prompts[0] == prompts[1]
    packets = adapter.read_json(output / "packets.json")
    outputs = [adapter.completed_outputs(output, side, packets)["q-1"] for side in adapter.STREAMS]
    assert outputs[0]["context_id"] != outputs[1]["context_id"]
    for envelope in outputs:
        assert envelope["identity"]["observed_model"] is None
        assert envelope["raw_output"].endswith("\r\n")
        assert envelope["human_gold"] is False
    before_progress = (output / "progress.jsonl").read_bytes()
    _, resumed = adapter.run(root, pool, resume=output, executor=mocked_execution)
    assert resumed["new_codex_cli_invocations"] == 0
    assert len(contexts) == 2
    assert (output / "progress.jsonl").read_bytes() == before_progress
