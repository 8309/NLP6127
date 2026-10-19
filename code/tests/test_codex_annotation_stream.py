"""Five mocked single-stream wrapper tests; no model or authentication calls."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def wrapper(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("isolated_codex_stream", scripts / "run_codex_annotation_stream.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli(wrapper, tmp_path, monkeypatch):
    binary = tmp_path / "codex.exe"
    wrapper.native.write_new(binary, b"fixture-only-not-executed")
    monkeypatch.setattr(wrapper.native, "discover_native_cli", lambda requested: binary)
    return binary


def test_only_selected_stream_runs_and_globals_restore(wrapper, cli, tmp_path, monkeypatch):
    before = wrapper.native.STREAMS, wrapper.native.load_pool, wrapper.native.write_new
    seen = []
    def mocked_run(root, pool, **kwargs):
        seen.append((wrapper.native.STREAMS, kwargs))
        return tmp_path, dict(failures=[])
    monkeypatch.setattr(wrapper.native, "run", mocked_run)
    wrapper.run_stream(tmp_path, tmp_path, "b", 50)
    assert seen == [(("b",), dict(max_records=50, cli=str(cli)))]
    assert (wrapper.native.STREAMS, wrapper.native.load_pool, wrapper.native.write_new) == before


def test_parent_guards_preserved_and_wrapper_added(wrapper, cli, tmp_path, monkeypatch):
    expected = {"synthetic-original": "original-hash"}
    monkeypatch.setattr(wrapper.native, "load_pool", lambda root, pool: (tmp_path, [], expected))
    def mocked_run(root, pool, **kwargs):
        _, _, protected = wrapper.native.load_pool(root, pool)
        assert protected["synthetic-original"] == "original-hash"
        assert protected[str(Path(wrapper.__file__).resolve())] == wrapper.native.protocol.sha256(Path(wrapper.__file__).read_bytes())
        assert expected == {"synthetic-original": "original-hash"}
        return tmp_path, dict(failures=[])
    monkeypatch.setattr(wrapper.native, "run", mocked_run)
    wrapper.run_stream(tmp_path, tmp_path)


def test_receipt_saved_before_generation_and_prompt_bytes_unchanged(wrapper, cli, tmp_path, monkeypatch):
    row = dict(question_id="q-1", question="What did the synthetic oil report observe?", question_zh="Translated fixture", as_of="2024-06-30",
               evidence=[dict(passage_id="p-1", text="Crude exports were stable.", available_by="2024-01-01")])
    original_packet, original_stdin = wrapper.native.packet_and_prompt(row)
    def mocked_run(root, pool, **kwargs):
        output = tmp_path / "separate-run"
        wrapper.native.write_new(output / "manifest.json", dict(specification=dict(streams=list(wrapper.native.STREAMS))))
        receipt = wrapper.native.read_json(output / "stream_wrapper_receipt.json")
        assert receipt["stream"] == "b"
        assert receipt["wrapper_sha256"] == wrapper.native.protocol.sha256(Path(wrapper.__file__).read_bytes())
        assert receipt["native_cli_name"] == "codex.exe"
        assert receipt["shared_progress_writers"] is False
        assert receipt["external_paid_api_calls"] == 0
        assert wrapper.native.packet_and_prompt(row) == (original_packet, original_stdin)
        assert receipt["protocol_sha256"] == original_packet["prompt_sha256"]
        return output, dict(failures=[])
    monkeypatch.setattr(wrapper.native, "run", mocked_run)
    wrapper.run_stream(tmp_path, tmp_path)


def test_failure_restores_patches_without_fallback_or_second_stream(wrapper, cli, tmp_path, monkeypatch):
    before = wrapper.native.STREAMS, wrapper.native.load_pool, wrapper.native.write_new
    calls = []
    def mocked_run(root, pool, **kwargs):
        calls.append(wrapper.native.STREAMS)
        raise RuntimeError("Synthetic quota/error stop")
    monkeypatch.setattr(wrapper.native, "run", mocked_run)
    with pytest.raises(RuntimeError, match="quota/error"):
        wrapper.run_stream(tmp_path, tmp_path)
    assert calls == [("b",)]
    assert (wrapper.native.STREAMS, wrapper.native.load_pool, wrapper.native.write_new) == before


def test_wrong_executable_or_stream_rejected_before_model_call(wrapper, tmp_path, monkeypatch):
    wrong = tmp_path / "other.exe"
    wrapper.native.write_new(wrong, b"fixture-only")
    monkeypatch.setattr(wrapper.native, "discover_native_cli", lambda requested: wrong)
    monkeypatch.setattr(wrapper.native, "run", lambda *args, **kwargs: pytest.fail("No model run allowed"))
    with pytest.raises(ValueError, match="native codex.exe"):
        wrapper.run_stream(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="stream"):
        wrapper.run_stream(tmp_path, tmp_path, "both")


@pytest.fixture
def historical_run(wrapper, tmp_path):
    source = tmp_path / "v1"
    row = dict(question_id="q-1", question="What operational oil throughput did the notice prove?",
        as_of="2024-06-30", evidence=[dict(passage_id="p-1", document_id="d-1",
            text="The notice announced restrictions on fourteen tankers.", available_by="2024-01-01")])
    packet, stdin = wrapper.native.packet_and_prompt(row)
    label = dict(answerability="insufficient", event_status="announced", qrels={"p-1": 1}, answer_points=[],
        quotes=[dict(passage_id="p-1", text="announced restrictions on fourteen tankers")],
        reasons=["A designation notice provides no observed operational throughput."])
    raw = wrapper.native.protocol.canonical_bytes(label) + b"\r\n"
    identity = dict(service="fixture", requested_model="fixture-default", observed_model=None, execution_id="old-context",
                    observation_source="unavailable", identity_note="No execution identity exposed.")
    envelope = wrapper.native.protocol.wrap_independent_output(packet, raw.decode("utf-8"), identity, "old-context", {},
        usage=dict(input_tokens=10, output_tokens=5), execution_input_sha256=wrapper.native.protocol.sha256(stdin))
    envelope["validation_errors"] = [wrapper.V1_CORRECTED_ERROR]
    attempt = source / "a/q-1/attempt-one"
    files = {"stdin.txt": stdin, "schema.json": wrapper.native.protocol.canonical_bytes(wrapper.native.output_schema(packet)),
        "raw_final.json": raw, "stderr.txt": b"", "envelope.json": wrapper.native.protocol.canonical_bytes(envelope),
        "events.jsonl": b"".join(wrapper.native.protocol.canonical_bytes(event) + b"\n" for event in
            [dict(type="thread.started", thread_id="old-context"), dict(type="turn.completed", usage=dict(input_tokens=10, output_tokens=5))])}
    for name, content in files.items():
        wrapper.native.write_new(attempt / name, content)
    receipt = dict(accepted=False, errors=[wrapper.V1_CORRECTED_ERROR], process_exit_code=0, stream="a", record_id="q-1",
        subscription_usage=dict(input_tokens=10, output_tokens=5), artifacts_sha256={name: wrapper.native.protocol.sha256(raw) for name, raw in files.items()})
    wrapper.native.write_new(attempt / "receipt.json", receipt)
    wrapper.native.append_progress(source / "progress.jsonl", dict(type="completed", stream="a", accepted=False,
        receipt="a/q-1/attempt-one/receipt.json", receipt_sha256=wrapper.native.protocol.sha256((attempt / "receipt.json").read_bytes())))
    wrapper.native.write_new(source / "packets.json", {"q-1": packet})
    wrapper.native.write_new(source / "protocol.txt", (wrapper.native.GUARD + "\n" + wrapper.native.protocol.ANNOTATION_PROMPT).encode("utf-8"))
    wrapper.native.write_new(source / "protected_inputs.json", {})
    wrapper.native.write_new(source / "manifest.json", dict(immutable_sha256={name: wrapper.native.protocol.sha256((source / name).read_bytes())
        for name in ("packets.json", "protocol.txt", "protected_inputs.json")}))
    return source, row, raw


def test_lossless_revalidation_rescues_only_declared_v1_guard_without_model_call(wrapper, historical_run, tmp_path):
    source, row, raw = historical_run
    old_receipt = (source / "a/q-1/attempt-one/receipt.json").read_bytes()
    attempts, protected, snapshots = wrapper.migration_inputs(source, "a", [row], {})
    assert len(attempts) == 1 and snapshots == {}
    output = tmp_path / "v2"
    wrapper.migrate_outputs(output, "a", attempts, snapshots, source)
    receipt = next((output / "a").glob("*/attempt-*/receipt.json"))
    restored = wrapper.native.read_json(receipt)
    assert restored["accepted"] is True and restored["new_model_calls"] == 0
    assert restored["original_guard_verdict"] == [wrapper.V1_CORRECTED_ERROR]
    envelope = wrapper.native.read_json(receipt.parent / "envelope.json")
    assert envelope["identity"]["execution_id"] == "old-context"
    assert envelope["raw_output"].encode("utf-8") == raw
    assert (receipt.parent / "raw_final.json").read_bytes() == raw
    assert (source / "a/q-1/attempt-one/receipt.json").read_bytes() == old_receipt
    wrapper.native.protected_unchanged(protected)


def test_changed_historical_raw_bytes_rejected_before_new_model_call(wrapper, historical_run):
    source, row, _ = historical_run
    (source / "a/q-1/attempt-one/raw_final.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        wrapper.migration_inputs(source, "a", [row], {})


def test_changed_question_or_protocol_does_not_silently_migrate(wrapper, historical_run):
    source, row, _ = historical_run
    row["question"] = "A different question"
    with pytest.raises(ValueError, match="packet digest changed"):
        wrapper.migration_inputs(source, "a", [row], {})
