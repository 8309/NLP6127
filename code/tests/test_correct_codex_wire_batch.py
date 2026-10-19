"""Synthetic corrective-pair fixtures; no actual model/subscription execution."""
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def adapter(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("correct_wire_pair_test", scripts / "correct_codex_wire_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rows():
    return [dict(record_id=f"synthetic-question-{index}", question=f"What crude flow did synthetic region {index} report?",
        as_of="2024-02-01", evidence=[dict(passage_id=f"synthetic-source-{index}", document_id=f"synthetic-doc-{index}",
            text=f"Synthetic region {index} reported 5 barrels of crude flow.", available_by="2024-01-01")]) for index in range(10)]


@pytest.fixture
def prepared(adapter, rows, tmp_path, monkeypatch):
    binary = tmp_path / "synthetic-native.exe"
    binary.write_bytes(b"synthetic native binary")
    original = tmp_path / "original/a/batch-0019/attempt"
    original.mkdir(parents=True)
    context = tmp_path / "original-native-context"
    adapter.native.write_new(original / "command.json", adapter.native.cli_command(binary, context,
        context / "schema.json", context / "final.json", None, []))
    source = dict(annotation_run=str(tmp_path / "original"), record_ids=[r["record_id"] for r in rows],
        rejected_receipts=[dict(path="synthetic-first-failure", sha256="1" * 64, execution_id="synthetic-old-b1"),
            dict(path="synthetic-second-failure", sha256="2" * 64, execution_id="synthetic-old-b2")],
        counterpart_a=dict(path=str(original / "receipt.json"), sha256="3" * 64, execution_id="synthetic-old-a"),
        accepted_receipts_before={}, requested_model=None,
        native_specification=dict(native_cli_sha256=adapter.protocol.sha256(binary.read_bytes()), disabled_mcp_names=[]),
        protected_sha256={str(Path(adapter.__file__)): adapter.protocol.sha256(Path(adapter.__file__).read_bytes())},
        rows=rows, progress_prefix=b"synthetic preserved failed history\n", usage_prefix=b"synthetic retained usage\n")
    monkeypatch.setattr(adapter, "source_context", lambda *args: source)
    directory, _ = adapter.prepare(tmp_path / "original", ["synthetic-first-failure", "synthetic-second-failure"])
    authorization = tmp_path / "authorization.json"
    adapter.native.write_new(authorization, dict(root_authorized=True, native_calls_allowed=2, no_label_feedback=True,
        manifest_sha256=adapter.protocol.sha256((directory / "manifest.json").read_bytes()), route=adapter.VERSION,
        corrective_instruction_sha256=adapter.protocol.sha256(adapter.CORRECTION_INSTRUCTION.encode())))
    return directory, authorization, (binary, {}, []), source


def backend(adapter, calls, *, fail_stream=None, bad_quote=False):
    def execute(command, stdin, context, attempt, environment, timeout):
        calls.append((attempt, stdin, context))
        payload = adapter.native.read_json(context / "packets.json")
        values = {}
        for alias, row in payload.items():
            evidence = row["evidence"][0]
            values[alias] = dict(record_id=alias, label=dict(answerability="sufficient", event_status="observed",
                qrels={evidence["passage_id"]: 2}, answer_points=[dict(text=evidence["text"], citations=[evidence["passage_id"]])],
                quotes=[] if bad_quote else [dict(passage_id=evidence["passage_id"], text=evidence["text"])], reasons=["Synthetic source supports the claim."]))
        events = [dict(type="thread.started", thread_id="synthetic-corrective-" + uuid4().hex)]
        failed = attempt.parent.name == fail_stream
        if failed:
            events += [dict(type="error", message="Synthetic quota failure"), dict(type="turn.failed")]
        else:
            events += [dict(type="turn.completed", usage=dict(input_tokens=123, output_tokens=45))]
            (context / "final.json").write_bytes((json.dumps(values, indent=2) + "\r\n").encode())
        adapter.native.write_new(attempt / "events.jsonl", b"".join(adapter.protocol.canonical_bytes(e) + b"\n" for e in events))
        adapter.native.write_new(attempt / "stderr.txt", b"")
        return 1 if failed else 0
    return execute


def test_prompt_only_adds_generic_requirement_sources_schema_unchanged(adapter, rows):
    original = adapter.codec.prepare_batch(rows, adapter.batch.BATCH_PROMPT, adapter.native.output_schema)
    corrective = adapter.prepare_source_batch(rows)
    assert corrective["wire_payload"] == original["wire_payload"]
    assert corrective["schema"] == original["schema"]
    assert corrective["codec"] == original["codec"]
    assert corrective["canonical_semantic_packets"] == original["canonical_semantic_packets"]
    assert corrective["stdin"] != original["stdin"]
    assert corrective["stdin"].split(b"UNTRUSTED_RECORD_PACKETS_JSON\n")[1] == original["stdin"].split(b"UNTRUSTED_RECORD_PACKETS_JSON\n")[1]
    assert "test-defe" not in adapter.CORRECTION_INSTRUCTION and "r2p6" not in adapter.CORRECTION_INSTRUCTION


def test_full_payload_over_byte_bound_aborts_no_truncation(adapter, rows):
    rows[0]["evidence"][0]["text"] += "x" * 262144
    with pytest.raises(ValueError, match="byte bound"):
        adapter.prepare_source_batch(rows)


def test_pair_native_fixture_complete_and_new_raw_kept(adapter, prepared):
    directory, authorization, setup, source = prepared
    calls = []
    receipts, certificate = adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter, calls))
    assert len(calls) == 2 and calls[0][1] == calls[1][1]
    assert calls[0][2] != calls[1][2]
    assert all(r["accepted"] for r in receipts.values())
    assert certificate["correction_accepted"] and certificate["native_calls_actual"] == 2
    assert certificate["original_rejected_event_count"] == 2
    assert certificate["original_agreement_substitution_allowed"] is False
    assert not (Path(source["annotation_run"]) / "progress.jsonl").exists()
    for stream in adapter.native.STREAMS:
        output = adapter.completed_correction_records(directory, stream)
        assert set(output) == set(source["record_ids"])
        raw = (directory / stream / "execution/raw_final.json").read_bytes()
        assert raw.endswith(b"\r\n")
        assert all(value["raw_output"].encode() == raw and value["annotation_role"] == "source_only_corrective_pair" for value in output.values())
    agreement = adapter.native.read_json(directory / "corrective_pair_agreement.json")
    assert agreement["paired_records"] == 10 and agreement["original_first_attempt_agreement"] is False


def test_no_execution_without_separate_review_authorization(adapter, prepared):
    directory, authorization, setup, _ = prepared
    value = adapter.native.read_json(authorization)
    value["root_authorized"] = False
    authorization.write_bytes(adapter.protocol.canonical_bytes(value))
    calls = []
    with pytest.raises(ValueError, match="authorization"):
        adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter, calls))
    assert calls == []


def test_once_only_never_auto_retry_accepted_pair(adapter, prepared):
    directory, authorization, setup, _ = prepared
    calls = []
    adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter, calls))
    with pytest.raises(ValueError, match="once-only"):
        adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter, calls))
    assert len(calls) == 2


def test_rejected_stream_retained_atomic_and_no_eligible_pair(adapter, prepared):
    directory, authorization, setup, _ = prepared
    calls = []
    receipts, certificate = adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter, calls, fail_stream="b"))
    assert len(calls) == 2 and receipts["a"]["accepted"] and not receipts["b"]["accepted"]
    assert not certificate["correction_accepted"]
    with pytest.raises(ValueError, match="Both whole"):
        adapter.completed_correction_records(directory, "a")


def test_missing_quote_guard_is_not_weakened(adapter, prepared):
    directory, authorization, setup, _ = prepared
    receipts, certificate = adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter, [], bad_quote=True))
    assert not certificate["correction_accepted"]
    assert all(any("exact eligible quotes" in error for error in r["errors"]) for r in receipts.values())


@pytest.mark.parametrize("field", ["schema_sha256", "mapping_sha256", "corrective_prompt_sha256", "corrective_instruction_sha256"])
def test_altered_corrective_receipt_metadata_fails(adapter, prepared, field):
    directory, authorization, setup, _ = prepared
    adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter, []))
    path = directory / "a/execution/receipt.json"
    value = adapter.native.read_json(path)
    value[field] = "0" * 64
    path.write_bytes(adapter.protocol.canonical_bytes(value))
    with pytest.raises(ValueError, match="accepted declared"):
        adapter.verify_corrective_attempt(directory, "a")


def test_saved_protected_input_map_is_enforced(adapter, prepared):
    directory, _, _, _ = prepared
    bad = directory / "synthetic-protected"
    bad.write_bytes(b"before")
    path = directory / "protected_inputs.json"
    mapping = adapter.native.read_json(path)
    mapping[str(bad)] = adapter.protocol.sha256(bad.read_bytes())
    path.write_bytes(adapter.protocol.canonical_bytes(mapping))
    manifest = adapter.native.read_json(directory / "manifest.json")
    manifest["immutable_sha256"]["protected_inputs.json"] = adapter.protocol.sha256(path.read_bytes())
    (directory / "manifest.json").write_bytes(adapter.protocol.canonical_bytes(manifest))
    bad.write_bytes(b"after")
    with pytest.raises(ValueError, match="Protected input changed"):
        adapter.verify_prepared(directory)


def test_receipt_usage_must_match_actual_native_events(adapter, prepared):
    directory, authorization, setup, _ = prepared
    adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter, []))
    path = directory / "a/execution/receipt.json"
    value = adapter.native.read_json(path)
    value["usage"]["output_tokens"] += 1
    path.write_bytes(adapter.protocol.canonical_bytes(value))
    with pytest.raises(ValueError, match="native events"):
        adapter.verify_corrective_attempt(directory, "a")


def test_pair_certificate_cannot_reclassify_original_failures(adapter, prepared):
    directory, authorization, setup, _ = prepared
    adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter, []))
    path = directory / "certificate.json"
    value = adapter.native.read_json(path)
    value["original_agreement_substitution_allowed"] = True
    path.write_bytes(adapter.protocol.canonical_bytes(value))
    with pytest.raises(ValueError, match="Both whole"):
        adapter.completed_correction_records(directory, "a")


def test_command_non_ephemeral_settings_cannot_change(adapter):
    base = ["cli", "exec", "--sandbox", "read-only", "--cd", "one", "--output-schema", "a", "--output-last-message", "b", "-"]
    other = ["cli", "exec", "--sandbox", "read-only", "--cd", "two", "--output-schema", "c", "--output-last-message", "d", "-"]
    assert adapter.normalized_command(base) == adapter.normalized_command(other)
    other[3] = "danger-full-access"
    assert adapter.normalized_command(base) != adapter.normalized_command(other)
