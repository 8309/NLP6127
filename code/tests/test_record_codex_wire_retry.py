"""Pure fixture tests: no subscription execution or real label modifications."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def recorder(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("record_wire_retry_test", scripts / "record_codex_wire_retry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run(recorder, tmp_path):
    native, protocol = recorder.native, recorder.protocol
    ids = ["synthetic-own-question"]
    events = []
    paths = {}
    for stream, accepted in (("a", True), ("b", False)):
        attempt = tmp_path / stream / "batch-0019" / "original"
        attempt.mkdir(parents=True)
        for name in recorder.INPUT_ARTIFACTS:
            native.write_new(attempt / name, b"synthetic-identical-input")
        native.write_new(attempt / "command.json", ["synthetic-cli", "exec", "-"])
        native.write_new(attempt / "raw_final.json", b"synthetic-unmodified-output")
        receipt = dict(accepted=accepted, stream=stream, batch="batch-0019", record_ids=ids,
            process_exit_code=0, errors=[] if accepted else ["Every answer point needs citations with exact eligible quotes"],
            execution_id="synthetic-original-" + stream,
            artifacts_sha256={path.name: protocol.sha256(path.read_bytes()) for path in attempt.iterdir()})
        native.write_new(attempt / "receipt.json", receipt)
        paths[stream] = attempt / "receipt.json"
        events.append(dict(type="completed", accepted=accepted, stream=stream, batch="batch-0019",
            receipt=paths[stream].relative_to(tmp_path).as_posix(), receipt_sha256=protocol.sha256(paths[stream].read_bytes())))
    native.write_new(tmp_path / "groups.json", [dict(batch="batch-0019", record_ids=ids)])
    native.write_new(tmp_path / "history.json", dict(record_ids=[]))
    native.write_new(tmp_path / "packets.json", {})
    native.write_new(tmp_path / "protected_inputs.json", {})
    native.write_new(tmp_path / "manifest.json", dict(specification=dict(requested_model=None), immutable_sha256={}))
    native.write_new(tmp_path / "progress.jsonl", b"".join(protocol.canonical_bytes(v) + b"\n" for v in events))
    native.write_new(tmp_path / "execution_usage.jsonl", b"")
    return tmp_path, paths


def add_retry(recorder, output, original, *, accepted=True, altered=False):
    native, protocol = recorder.native, recorder.protocol
    attempt = output / "b/batch-0019/retry"
    attempt.mkdir(parents=True)
    for name in (*recorder.INPUT_ARTIFACTS, "command.json", "raw_final.json"):
        native.write_new(attempt / name, (original.parent / name).read_bytes() + (b"altered" if altered and name == "stdin.txt" else b""))
    receipt = dict(accepted=accepted, stream="b", batch="batch-0019", record_ids=["synthetic-own-question"],
        execution_id="synthetic-fresh-retry", artifacts_sha256={p.name: protocol.sha256(p.read_bytes()) for p in attempt.iterdir()})
    native.write_new(attempt / "receipt.json", receipt)
    native.append_progress(output / "progress.jsonl", dict(type="completed", accepted=accepted, stream="b", batch="batch-0019",
        receipt=(attempt / "receipt.json").relative_to(output).as_posix(), receipt_sha256=protocol.sha256((attempt / "receipt.json").read_bytes())))
    native.append_progress(output / "execution_usage.jsonl", dict(execution_id=receipt["execution_id"], usage=dict(input_tokens=1)))
    return attempt


def test_single_exact_retry_keeps_failure_and_old_accepted(recorder, run, monkeypatch):
    output, paths = run
    original = paths["b"].read_bytes()
    directory, request = recorder.prepare(output, paths["b"])
    assert request["max_records_endpoint"] == 1
    add_retry(recorder, output, paths["b"])
    monkeypatch.setattr(recorder.wire, "verify_attempt", lambda *args: {})
    result = recorder.seal(directory)
    assert result["retry_accepted"] is True
    assert result["native_calls_actual"] == 1
    assert paths["b"].read_bytes() == original
    assert result["authorization_sha256"] == request["authorization_sha256"]


def test_physical_input_change_fails_closed(recorder, run, monkeypatch):
    output, paths = run
    directory, _ = recorder.prepare(output, paths["b"])
    add_retry(recorder, output, paths["b"], altered=True)
    with pytest.raises(ValueError, match="physical input"):
        recorder.seal(directory)


def test_second_failed_retry_is_recorded_not_accepted(recorder, run):
    output, paths = run
    directory, _ = recorder.prepare(output, paths["b"])
    add_retry(recorder, output, paths["b"], accepted=False)
    result = recorder.seal(directory)
    assert not result["retry_accepted"]
    assert result["status"] == "single_exact_input_retry_failed_stop"


def test_original_raw_tampering_fails_closed(recorder, run):
    output, paths = run
    (paths["b"].parent / "raw_final.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="raw artifact"):
        recorder.prepare(output, paths["b"])


def test_removed_failed_progress_is_never_resolved(recorder, run):
    output, paths = run
    directory, _ = recorder.prepare(output, paths["b"])
    add_retry(recorder, output, paths["b"])
    (output / "progress.jsonl").write_bytes(b"history removed")
    with pytest.raises(ValueError, match="changed or deleted"):
        recorder.seal(directory)


def test_more_than_one_retry_event_is_not_authorized(recorder, run):
    output, paths = run
    directory, _ = recorder.prepare(output, paths["b"])
    add_retry(recorder, output, paths["b"])
    recorder.native.append_progress(output / "progress.jsonl", dict(type="completed", accepted=False,
        stream="b", batch="batch-0019"))
    with pytest.raises(ValueError, match="Only one"):
        recorder.seal(directory)


@pytest.mark.parametrize("command, expected", [(["exec", "-"], None), (["exec", "--model", "synthetic-model", "-"], "synthetic-model")])
def test_model_request_is_actual_command_not_self_report(recorder, tmp_path, command, expected):
    path = tmp_path / "command.json"
    recorder.native.write_new(path, command)
    assert recorder.requested_model_from_command(path) == expected
