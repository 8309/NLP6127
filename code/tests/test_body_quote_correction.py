"""Synthetic source-only body correction; no real corpus or native service calls."""
import importlib.util
import json
from pathlib import Path

import pytest

from test_correct_codex_wire_batch import backend


@pytest.fixture
def adapter(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("body_quote_test", scripts / "correct_body_quote_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rows():
    return [dict(record_id=f"synthetic-body-question-{index}", question="What crude production did the source report?",
        as_of="2024-02-01", evidence=[dict(passage_id=f"synthetic-body-source-{index}",
        document_id=f"synthetic-body-doc-{index}", title="Synthetic title not present in the body",
        text=f"Synthetic region {index} reported 5 barrels of crude production.", available_by="2024-01-01")])
        for index in range(10)]


@pytest.fixture
def prepared(adapter, rows, tmp_path, monkeypatch):
    binary = tmp_path / "synthetic-native.exe"
    binary.write_bytes(b"synthetic native binary")
    source = tmp_path / "synthetic-pool/cohort_annotations_transport_recovery/run-old"
    source.mkdir(parents=True)
    guard = source / "synthetic-original-raw.txt"
    guard.write_bytes(b"original rejected output remains unchanged")
    command_context = source / "synthetic-command-context"
    command = adapter.native.cli_command(binary, command_context, command_context / "schema.json",
        command_context / "final.json", None, [])
    context = dict(source_run=str(source.resolve()), rows=rows, record_ids=[row["record_id"] for row in rows],
        source_specification=dict(requested_model=None, native_command_identity=adapter.correction.normalized_command(command),
            native_cli_sha256=adapter.protocol.sha256(binary.read_bytes()), disabled_mcp_names=[]),
        rejected_receipts={stream: dict(path=f"synthetic-old-{stream}", sha256=("1" if stream == "a" else "2") * 64,
            execution_id="synthetic-rejected-" + stream) for stream in ("a", "b")},
        protected_sha256={str(guard): adapter.protocol.sha256(guard.read_bytes())})
    monkeypatch.setattr(adapter, "source_context", lambda *args: context)
    directory, manifest = adapter.prepare(source)
    authorization = tmp_path / "root-authorization.json"
    adapter.native.write_new(authorization, dict(root_authorized=True, native_calls_allowed=2, no_label_feedback=True,
        manifest_sha256=adapter.protocol.sha256((directory / "manifest.json").read_bytes()),
        corrective_instruction_sha256=adapter.INSTRUCTION_SHA256, route=adapter.VERSION))
    return directory, authorization, (binary, {}, []), context


def test_isolated_primitive_metadata_does_not_mutate_frozen_imports(adapter):
    import correct_codex_wire_batch as old_correction
    import run_codex_transport_recovery as old_transport
    assert old_correction.VERSION == "source-only-every-citation-corrective-pair-v1"
    assert old_correction.CORRECTION_INSTRUCTION == adapter.ORIGINAL_INSTRUCTION
    assert old_transport.VERSION == "codex-unchanged-input-transport-recovery-v1"
    assert old_transport.clarification is old_correction
    assert adapter.engine.clarification is adapter.correction and adapter.engine is not old_transport


def test_body_instruction_is_generic_and_retains_full_sources_schema(adapter, rows):
    old = adapter.codec.prepare_batch(rows, adapter.correction.batch.BATCH_PROMPT, adapter.native.output_schema)
    new = adapter.correction.prepare_source_batch(rows)
    assert new["schema"] == old["schema"] and new["codec"] == old["codec"]
    assert new["wire_payload"] == old["wire_payload"] and new["canonical_semantic_packets"] == old["canonical_semantic_packets"]
    assert "text field" in adapter.BODY_INSTRUCTION and "date metadata" in adapter.BODY_INSTRUCTION
    for prohibited in (adapter.BAD_RECORD_ID, adapter.BAD_PASSAGE_ID, "r7", "2%", "2024", "gold", "reference answer"):
        assert prohibited not in adapter.BODY_INSTRUCTION
    assert new["stdin"].endswith(adapter.protocol.canonical_bytes(new["wire_payload"]) + b"\n")


def test_pair_has_exact_equal_new_input_fresh_contexts_and_corrective_roles(adapter, prepared):
    directory, authorization, setup, context = prepared
    original = Path(next(iter(context["protected_sha256"]))).read_bytes()
    calls = []
    receipts, certificate = adapter.execute_pair_once(directory, authorization, native_setup=setup,
        executor=backend(adapter.correction, calls))
    assert certificate["correction_accepted"] and certificate["native_calls_actual"] == 2
    assert certificate["original_agreement_substitution_allowed"] is False
    assert len(calls) == 2 and calls[0][1] == calls[1][1] == (directory / "stdin.txt").read_bytes()
    assert calls[0][2] != calls[1][2] and receipts["a"]["execution_id"] != receipts["b"]["execution_id"]
    for stream in ("a", "b"):
        records = adapter.completed_correction_records(directory, stream)
        assert len(records) == 10 and all(value["annotation_role"] == adapter.ROLE
            and value["original_first_attempt"] is False for value in records.values())
    assert Path(next(iter(context["protected_sha256"]))).read_bytes() == original


def test_semantic_quote_failure_never_becomes_accepted(adapter, prepared):
    directory, authorization, setup, _ = prepared
    receipts, certificate = adapter.execute_pair_once(directory, authorization, native_setup=setup,
        executor=backend(adapter.correction, [], bad_quote=True))
    assert not certificate["correction_accepted"] and all(not value["accepted"] for value in receipts.values())
    with pytest.raises(ValueError, match="Both whole"):
        adapter.completed_correction_records(directory, "a")


def test_missing_authorization_stops_before_any_fake_native_call(adapter, prepared):
    directory, authorization, setup, _ = prepared
    value = adapter.native.read_json(authorization)
    value["root_authorized"] = False
    authorization.write_bytes(adapter.protocol.canonical_bytes(value))
    calls = []
    with pytest.raises(ValueError, match="authorization"):
        adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter.correction, calls))
    assert calls == []


def test_failed_or_accepted_pair_cannot_automatically_retry(adapter, prepared):
    directory, authorization, setup, _ = prepared
    calls = []
    adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter.correction, calls))
    with pytest.raises(ValueError, match="once-only"):
        adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=backend(adapter.correction, calls))
    assert len(calls) == 2


def test_manifest_role_or_model_rebinding_is_rejected(adapter, prepared):
    directory, _, _, _ = prepared
    manifest = adapter.native.read_json(directory / "manifest.json")
    manifest["specification"]["groups"][0]["original_first_attempt"] = True
    (directory / "manifest.json").write_bytes(adapter.protocol.canonical_bytes(manifest))
    with pytest.raises(ValueError, match="protocol"):
        adapter.verify_prepared(directory)


def test_full_body_input_over_byte_cap_aborts_without_truncation(adapter, rows):
    rows[0]["evidence"][0]["text"] += "x" * 262144
    with pytest.raises(ValueError, match="byte bound"):
        adapter.correction.prepare_source_batch(rows)


@pytest.fixture
def source_fixture(adapter, rows, tmp_path, monkeypatch):
    import run_codex_transport_recovery as original
    source = tmp_path / "synthetic-pool/cohort_annotations_transport_recovery/run-old"
    source.mkdir(parents=True)
    binary = tmp_path / "synthetic-native.exe"
    binary.write_bytes(b"synthetic native binary")
    old_prepared = original.clarification.prepare_source_batch(rows)
    group = dict(batch=adapter.FAILED_BATCH, record_ids=[row["record_id"] for row in rows], split="test",
        full_stdin_sha256=old_prepared["actual_wire_stdin_sha256"], schema_sha256=old_prepared["actual_wire_schema_sha256"],
        mapping_sha256=old_prepared["codec"]["mapping_sha256"],
        full_payload_bytes=len(old_prepared["stdin"]) + len(adapter.protocol.canonical_bytes(old_prepared["schema"])),
        annotation_role="format_clarified_first_attempt", original_first_attempt=True)
    command_context = source / "synthetic-command-context"
    command = adapter.native.cli_command(binary, command_context, command_context / "schema.json", command_context / "final.json", None, [])
    packets = {row["record_id"]: adapter.protocol.clean_packet(row, prompt=original.wire.batch.BATCH_PROMPT) for row in rows}
    adapter.native.write_new(source / "packets.json", packets)
    adapter.native.write_new(source / "protected_inputs.json", {})
    adapter.native.write_new(source / "manifest.json", dict(specification=dict(version=original.VERSION, groups=[group],
        requested_model=None, native_command_identity=adapter.correction.normalized_command(command),
        native_cli_sha256=adapter.protocol.sha256(binary.read_bytes()), disabled_mcp_names=[]),
        immutable_sha256={"packets.json": adapter.protocol.sha256((source / "packets.json").read_bytes())}))
    paths = {}

    def title_backend(command, stdin, context, attempt, environment, timeout):
        result = backend(adapter.correction, [])(command, stdin, context, attempt, environment, timeout)
        parsed = adapter.native.read_json(context / "final.json")
        parsed["r7"]["label"]["quotes"].append(dict(passage_id="r7p0", text=rows[7]["evidence"][0]["title"]))
        (context / "final.json").write_bytes(adapter.protocol.canonical_bytes(parsed))
        return result

    for stream in ("a", "b"):
        result = original.execute_attempt(source, stream, group, packets, binary, {}, [], {}, None, 90, title_backend)
        path = result["attempt"] / "receipt.json"
        paths[stream] = path
        assert result["receipt"]["errors"] == [adapter.SOURCE_ERROR]
        adapter.native.append_progress(source / "progress.jsonl", dict(type="completed", accepted=False, stream=stream,
            batch=adapter.FAILED_BATCH, record_ids=group["record_ids"], receipt=path.relative_to(source).as_posix(),
            receipt_sha256=adapter.protocol.sha256(path.read_bytes())))
    monkeypatch.setattr(adapter, "MANIFEST_SHA256", adapter.protocol.sha256((source / "manifest.json").read_bytes()))
    monkeypatch.setattr(adapter, "PROGRESS_SHA256", adapter.protocol.sha256((source / "progress.jsonl").read_bytes()))
    monkeypatch.setattr(adapter, "FAILURE_SHA256", {stream: adapter.protocol.sha256(path.read_bytes()) for stream, path in paths.items()})
    monkeypatch.setattr(adapter, "BAD_RECORD_ID", rows[7]["record_id"])
    monkeypatch.setattr(adapter, "BAD_PASSAGE_ID", rows[7]["evidence"][0]["passage_id"])
    return source, paths


def test_original_source_diagnosis_requires_nine_valid_and_one_title_only(adapter, source_fixture):
    source, paths = source_fixture
    old = {stream: path.read_bytes() for stream, path in paths.items()}
    context = adapter.source_context(source)
    assert len(context["record_ids"]) == 10
    assert all(value["diagnostic"] == dict(valid_records=9, invalid_records=1, defect="title_not_body_quote")
        for value in context["rejected_receipts"].values())
    assert all(path.read_bytes() == old[stream] for stream, path in paths.items())


def test_original_source_tamper_rejects_even_before_preparation(adapter, source_fixture):
    source, paths = source_fixture
    (paths["a"].parent / "raw_final.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="artifact"):
        adapter.source_context(source)


def test_nonsemantic_native_failure_is_never_corrected(adapter, source_fixture):
    source, paths = source_fixture
    path = paths["a"]
    receipt = adapter.native.read_json(path)
    receipt["errors"] = ["Synthetic quota failure"]
    path.write_bytes(adapter.protocol.canonical_bytes(receipt))
    progress = adapter.protocol.parse_jsonl((source / "progress.jsonl").read_bytes())
    progress[0]["receipt_sha256"] = adapter.protocol.sha256(path.read_bytes())
    (source / "progress.jsonl").write_bytes(b"".join(adapter.protocol.canonical_bytes(event) + b"\n" for event in progress))
    adapter.FAILURE_SHA256["a"] = adapter.protocol.sha256(path.read_bytes())
    adapter.PROGRESS_SHA256 = adapter.protocol.sha256((source / "progress.jsonl").read_bytes())
    with pytest.raises(ValueError, match="Native|quota|mixed"):
        adapter.source_context(source)
