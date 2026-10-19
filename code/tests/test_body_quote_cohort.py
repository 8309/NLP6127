"""Body-only future cohort fixtures; no real source preparation or native calls."""
import importlib.util
from pathlib import Path

import pytest

from test_correct_codex_wire_batch import backend


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("body_cohort_test", scripts / "run_body_quote_cohort.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(runner, tmp_path, monkeypatch):
    rows = [dict(record_id=f"synthetic-body-cohort-{index}", question="What crude production was reported?",
        as_of="2024-02-01", evidence=[dict(passage_id=f"synthetic-body-evidence-{index}",
            document_id=f"synthetic-body-document-{index}", text=f"Synthetic region {index} reported 5 barrels of crude production.",
            available_by="2024-01-01")]) for index in range(30)]
    packets = {row["record_id"]: runner.protocol.clean_packet(row, prompt=runner.old.wire.batch.BATCH_PROMPT) for row in rows}
    binary = tmp_path / "synthetic-native.exe"
    binary.write_bytes(b"synthetic native binary")
    pool = tmp_path / "synthetic-pool"
    source = pool / "cohort_annotations_transport_recovery/run-source"
    source.mkdir(parents=True)
    context = source / "synthetic-context"
    command = runner.native.cli_command(binary, context, context / "schema.json", context / "final.json", None, [])
    groups = []
    for batch, part in (("batch-0026", rows[10:20]), ("batch-0027", rows[20:30])):
        value = runner.old.clarification.prepare_source_batch(part)
        groups.append(dict(batch=batch, record_ids=[row["record_id"] for row in part], split="test",
            full_stdin_sha256=value["actual_wire_stdin_sha256"], schema_sha256=value["actual_wire_schema_sha256"],
            mapping_sha256=value["codec"]["mapping_sha256"],
            full_payload_bytes=len(value["stdin"]) + len(runner.protocol.canonical_bytes(value["schema"])),
            annotation_role="format_clarified_first_attempt", original_first_attempt=True))
    spec = dict(version=runner.old.VERSION, groups=groups, eligible_record_ids=[row["record_id"] for row in rows], bindings={},
        requested_model=None, native_cli_sha256=runner.protocol.sha256(binary.read_bytes()), disabled_mcp_names=[],
        native_command_identity=runner.body.correction.normalized_command(command))
    runner.native.write_new(source / "manifest.json", dict(specification=spec, immutable_sha256={}))
    runner.native.write_new(source / "packets.json", packets)
    records = {stream: {row["record_id"]: dict(label="synthetic-history-" + stream) for row in rows[:20]} for stream in ("a", "b")}
    sources = {stream: {rid: dict(owner="synthetic-original-owner", record_id=rid) for rid in records[stream]} for stream in ("a", "b")}
    history = dict(source_run=str(source.resolve()), correction_dir=str(tmp_path / "synthetic-correction"),
        record_ids=[row["record_id"] for row in rows[:20]], records=records, sources=sources, original_specification=spec,
        protected_sha256={str(source / "manifest.json"): runner.protocol.sha256((source / "manifest.json").read_bytes())},
        execution_usage_records=[dict(execution_id="synthetic-old-unknown-usage", usage=None)])
    monkeypatch.setattr(runner, "HISTORY_COUNT", 20)
    monkeypatch.setattr(runner, "TOTAL_COUNT", 30)
    monkeypatch.setattr(runner, "REMAINING_GROUPS", 1)
    monkeypatch.setattr(runner.body, "MANIFEST_SHA256", runner.protocol.sha256((source / "manifest.json").read_bytes()))
    monkeypatch.setattr(runner, "load_history", lambda *args: (history, records))
    monkeypatch.setattr(runner.old.wire.previous, "load_pool", lambda root, pool_dir:
        (pool, rows, [dict(record_id=row["record_id"], split="test") for row in rows], {}, {}))
    directory, copied = runner.prepare(tmp_path, pool, source, tmp_path / "synthetic-correction")
    monkeypatch.setattr(runner.native, "discover_native_cli", lambda *args: binary)
    monkeypatch.setattr(runner.native, "child_environment", lambda: {})
    monkeypatch.setattr(runner.native, "configured_mcp_names", lambda environment: [])
    return directory, packets, copied, binary, history


def test_new_groups_preserve_full_source_order_schema_and_are_first_attempts(runner, prepared):
    directory, packets, records, binary, history = prepared
    groups = runner.native.read_json(directory / "groups.json")
    assert len(groups) == 1 and groups[0]["batch"] == "batch-0027"
    assert groups[0]["record_ids"] == history["original_specification"]["eligible_record_ids"][20:]
    assert groups[0]["annotation_role"] == "body_only_first_attempt" and groups[0]["original_first_attempt"] is True
    old = history["original_specification"]["groups"][-1]
    assert groups[0]["schema_sha256"] == old["schema_sha256"] and groups[0]["mapping_sha256"] == old["mapping_sha256"]
    assert groups[0]["full_stdin_sha256"] != old["full_stdin_sha256"]


def test_cached_history_reads_never_rerun_cold_original_proofs(runner, prepared, monkeypatch):
    directory, packets, records, binary, history = prepared
    def unexpected(*args):
        raise AssertionError("Cold source history must only be proved during preparation")
    monkeypatch.setattr(runner, "load_history", unexpected)
    monkeypatch.setattr(runner.body, "source_context", unexpected)
    for stream in ("a", "b"):
        assert runner.completed_records(directory, stream, packets) == records[stream]
        assert runner.record_sources(directory, stream) == history["sources"][stream]


def test_fresh_pair_has_equal_full_input_and_future_only_agreement(runner, prepared):
    directory, packets, records, binary, history = prepared
    calls = []
    summary = runner.run(directory, max_native_contexts=2, executor=backend(runner.body.correction, calls))
    assert len(calls) == 2 and calls[0][1] == calls[1][1] and calls[0][2] != calls[1][2]
    assert summary["accepted_counts"] == {"a": 30, "b": 30}
    agreement = runner.native.read_json(next(directory.glob("body_first_attempt_agreement-*.json")))
    assert agreement["paired_records"] == 10 and agreement["planned_records"] == 10
    assert agreement["excludes_original_and_corrective_history"] is True
    assert summary["subscription_usage"]["executions"] == 3
    assert summary["subscription_usage"]["known_usage_executions"] == 2
    assert summary["subscription_dollar_cost"] is None


def test_rejected_semantic_output_stops_without_auto_retry(runner, prepared):
    directory, packets, records, binary, history = prepared
    calls = []
    summary = runner.run(directory, max_native_contexts=2, executor=backend(runner.body.correction, calls, bad_quote=True))
    assert len(calls) == 2 and summary["failures"]
    with pytest.raises(ValueError, match="automatic retry"):
        runner.run(directory, max_native_contexts=2, executor=backend(runner.body.correction, calls))
    assert len(calls) == 2


def test_orphan_interruption_stops_before_any_native_request(runner, prepared):
    directory, packets, records, binary, history = prepared
    (directory / "a/batch-0027/attempt-interrupted").mkdir(parents=True)
    calls = []
    with pytest.raises(ValueError, match="Undurable|interrupted"):
        runner.run(directory, max_native_contexts=2, executor=backend(runner.body.correction, calls))
    assert calls == []


def test_mutated_cached_label_bytes_are_rejected(runner, prepared):
    directory, packets, records, binary, history = prepared
    (directory / "history.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="Protected|changed"):
        runner.completed_records(directory, "a", packets)


def test_model_identity_cannot_be_rebound_by_new_manifest(runner, prepared):
    directory, packets, records, binary, history = prepared
    manifest = runner.native.read_json(directory / "manifest.json")
    manifest["specification"]["requested_model"] = "synthetic-unapproved-model"
    (directory / "manifest.json").write_bytes(runner.protocol.canonical_bytes(manifest))
    calls = []
    with pytest.raises(ValueError, match="identity|history"):
        runner.run(directory, max_native_contexts=2, executor=backend(runner.body.correction, calls))
    assert calls == []


def test_duplicate_execution_usage_is_rejected_without_zero_cost_claim(runner, prepared):
    directory, packets, records, binary, history = prepared
    runner.native.write_new(directory / "execution_usage.jsonl", runner.protocol.canonical_bytes(history["execution_usage_records"][0]) + b"\n")
    with pytest.raises(ValueError, match="duplicated|receipt execution-ID"):
        runner.merged_usage(directory)


@pytest.mark.parametrize("change", ["missing", "extra", "usage", "record_ids"])
def test_corrective_ledger_must_match_both_complete_native_receipts(runner, tmp_path, change):
    ids = [f"synthetic-corrective-{index}" for index in range(10)]
    certificate = dict(record_ids=ids, receipts={})
    values = []
    for stream in ("a", "b"):
        receipt = dict(execution_id="synthetic-pair-" + stream, stream=stream, record_ids=ids,
            usage=dict(input_tokens=123, output_tokens=45))
        path = tmp_path / stream / "receipt.json"
        runner.native.write_new(path, receipt)
        certificate["receipts"][stream] = dict(path=str(path), sha256=runner.protocol.sha256(path.read_bytes()))
        values.append(dict(receipt))
    if change == "missing":
        values.pop()
    elif change == "extra":
        values.append(dict(values[0], execution_id="synthetic-unregistered-native"))
    elif change == "usage":
        values[0]["usage"] = dict(input_tokens=1, output_tokens=45)
    else:
        values[0]["record_ids"] = ids[:-1]
    runner.native.write_new(tmp_path / "execution_usage.jsonl", b"".join(runner.protocol.canonical_bytes(value) + b"\n" for value in values))
    with pytest.raises(ValueError, match="usage|receipt|record"):
        runner.pair_usage(tmp_path, certificate)


def test_missing_new_native_usage_stops_before_continuation(runner, prepared):
    directory, packets, records, binary, history = prepared
    calls = []
    runner.run(directory, max_native_contexts=2, executor=backend(runner.body.correction, calls))
    usage = runner.protocol.parse_jsonl((directory / "execution_usage.jsonl").read_bytes())
    (directory / "execution_usage.jsonl").write_bytes(runner.protocol.canonical_bytes(usage[0]) + b"\n")
    with pytest.raises(ValueError, match="execution-ID"):
        runner.run(directory, max_native_contexts=2, executor=backend(runner.body.correction, calls))
    assert len(calls) == 2
