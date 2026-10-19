"""Synthetic asymmetric zero-call-overlay continuation; no real preparation."""
import importlib.util
from pathlib import Path

import pytest

from test_correct_codex_wire_batch import backend


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("revalidated_body_test", scripts / "run_revalidated_body_cohort.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(runner, tmp_path, monkeypatch):
    rows = [dict(record_id=f"synthetic-revalidated-{index}", question="What crude production was reported?",
        as_of="2024-02-01", evidence=[dict(passage_id=f"synthetic-evidence-{index}", document_id=f"synthetic-document-{index}",
            text=f"Synthetic region {index} reported 5 barrels of crude production.", available_by="2024-01-01")]) for index in range(28)]
    packets = {row["record_id"]: runner.protocol.clean_packet(row, prompt=runner.previous.old.wire.batch.BATCH_PROMPT) for row in rows}
    pool = tmp_path / "synthetic-pool"
    source = pool / "cohort_annotations_body_only/run-old"
    source.mkdir(parents=True)
    binary = tmp_path / "synthetic-native.exe"
    binary.write_bytes(b"synthetic native binary")
    context = source / "synthetic-old-context"
    command = runner.native.cli_command(binary, context, context / "schema.json", context / "final.json", None, [])
    groups = []
    for batch, part in (("batch-0039", rows[8:10]), ("batch-0040", rows[10:20]), ("batch-0041", rows[20:28])):
        value = runner.previous.body.correction.prepare_source_batch(part)
        groups.append(dict(batch=batch, record_ids=[row["record_id"] for row in part], split="test",
            full_stdin_sha256=value["actual_wire_stdin_sha256"], schema_sha256=value["actual_wire_schema_sha256"],
            mapping_sha256=value["codec"]["mapping_sha256"],
            full_payload_bytes=len(value["stdin"]) + len(runner.protocol.canonical_bytes(value["schema"])),
            annotation_role="body_only_first_attempt", original_first_attempt=True))
    spec = dict(version=runner.previous.VERSION, groups=groups, eligible_record_ids=[row["record_id"] for row in rows], bindings={},
        requested_model=None, native_cli_sha256=runner.protocol.sha256(binary.read_bytes()), disabled_mcp_names=[],
        native_command_identity=runner.previous.body.correction.normalized_command(command))
    runner.native.write_new(source / "manifest.json", dict(specification=spec, immutable_sha256={}))
    runner.native.write_new(source / "packets.json", packets)
    counts = {"a": 10, "b": 8}
    records = {stream: {row["record_id"]: dict(label="synthetic-history-" + stream) for row in rows[:count]} for stream, count in counts.items()}
    sources = {stream: {rid: dict(owner="original-source-owner", record_id=rid) for rid in records[stream]} for stream in counts}
    history = dict(source_run=str(source.resolve()), certificate_dir=str(tmp_path / "synthetic-certificate"),
        original_specification=spec, records=records, sources=sources, historical_counts=counts,
        record_ids={stream: [row["record_id"] for row in rows[:count]] for stream, count in counts.items()},
        history_common_paired_ids=[row["record_id"] for row in rows[:8]], one_sided_record_ids={"a": [row["record_id"] for row in rows[8:10]], "b": []},
        protected_sha256={str(source / "manifest.json"): runner.protocol.sha256((source / "manifest.json").read_bytes())},
        execution_usage_records=[dict(execution_id="synthetic-old-unknown-usage", usage=None)])
    monkeypatch.setattr(runner, "COUNTS", counts)
    monkeypatch.setattr(runner, "GROUP_COUNT", 3)
    monkeypatch.setattr(runner, "PENDING_CONTEXTS", 5)
    monkeypatch.setattr(runner, "TOTAL_COUNT", 28)
    monkeypatch.setattr(runner, "MANIFEST_SHA256", runner.protocol.sha256((source / "manifest.json").read_bytes()))
    monkeypatch.setattr(runner, "load_history", lambda *args: (history, records))
    monkeypatch.setattr(runner.previous.old.wire.previous, "load_pool", lambda root, pool_dir:
        (pool, rows, [dict(record_id=row["record_id"], split="test") for row in rows], {}, {}))
    directory, copied = runner.prepare(tmp_path, pool, source, tmp_path / "synthetic-certificate")
    monkeypatch.setattr(runner.native, "discover_native_cli", lambda *args: binary)
    monkeypatch.setattr(runner.native, "child_environment", lambda: {})
    monkeypatch.setattr(runner.native, "configured_mcp_names", lambda environment: [])
    return directory, packets, copied, binary, history


def test_asymmetric_cache_schedules_only_b39_then_untouched_complete_pairs(runner, prepared):
    directory, packets, records, binary, history = prepared
    manifest, rebuilt, guards = runner.verify_prepared(directory)
    jobs = runner.previous.old.previous.pending_jobs(manifest["specification"]["groups"], records, 100)
    assert [(stream, group["batch"]) for stream, group in jobs] == [
        ("b", "batch-0039"), ("a", "batch-0040"), ("b", "batch-0040"), ("a", "batch-0041"), ("b", "batch-0041")]
    assert rebuilt["history_common_paired_ids"] == history["history_common_paired_ids"]
    assert all(group["original_first_attempt"] is True for stream, group in jobs)


def test_history_reads_reuse_sealed_asymmetric_cache_without_cold_certificate_proofs(runner, prepared, monkeypatch):
    directory, packets, records, binary, history = prepared
    def unexpected(*args):
        raise AssertionError("Cold zero-call admission must run only during preparation")
    monkeypatch.setattr(runner, "load_history", unexpected)
    for stream in ("a", "b"):
        assert runner.completed_records(directory, stream, packets) == records[stream]
        assert runner.record_sources(directory, stream) == history["sources"][stream]


def test_one_explicit_call_runs_only_unstarted_b39_and_reuses_a(runner, prepared, monkeypatch):
    directory, packets, records, binary, history = prepared
    monkeypatch.setattr(runner.protocol, "raw_agreement", lambda *args: dict(paired_records=0))
    calls = []
    summary = runner.run(directory, max_native_contexts=1, executor=backend(runner.previous.body.correction, calls))
    assert len(calls) == 1 and calls[0][0].parent.name == "batch-0039" and calls[0][0].parent.parent.name == "b"
    assert summary["accepted_counts"] == {"a": 10, "b": 10}
    assert summary["historical_records_reused_per_stream"] == {"a": 10, "b": 8}
    assert summary["subscription_usage"]["executions"] == 2 and summary["subscription_dollar_cost"] is None
    new = runner.completed_records(directory, "b", packets)
    assert all(new[rid]["annotation_role"] == "body_only_first_attempt" and new[rid]["original_first_attempt"] is True
        for rid in history["one_sided_record_ids"]["a"])


def test_new_semantic_failure_has_no_generic_503_or_retry_bypass(runner, prepared, monkeypatch):
    directory, packets, records, binary, history = prepared
    monkeypatch.setattr(runner.protocol, "raw_agreement", lambda *args: dict(paired_records=0))
    calls = []
    summary = runner.run(directory, max_native_contexts=1, executor=backend(runner.previous.body.correction, calls, bad_quote=True))
    assert summary["failures"] and len(calls) == 1
    with pytest.raises(ValueError, match="retry"):
        runner.run(directory, max_native_contexts=1, executor=backend(runner.previous.body.correction, calls))
    assert len(calls) == 1


def test_new_interrupted_attempt_is_not_implicitly_rerun(runner, prepared):
    directory, packets, records, binary, history = prepared
    (directory / "b/batch-0039/attempt-interrupted").mkdir(parents=True)
    calls = []
    with pytest.raises(ValueError, match="Undurable|interrupted"):
        runner.run(directory, max_native_contexts=1, executor=backend(runner.previous.body.correction, calls))
    assert calls == []


def test_consistent_new_model_rebinding_cannot_change_original_request(runner, prepared):
    directory, packets, records, binary, history = prepared
    manifest = runner.native.read_json(directory / "manifest.json")
    manifest["specification"]["requested_model"] = "synthetic-unapproved-model"
    (directory / "manifest.json").write_bytes(runner.protocol.canonical_bytes(manifest))
    with pytest.raises(ValueError, match="model|source"):
        runner.verify_prepared(directory)


def test_missing_new_usage_row_blocks_continuation_before_native_call(runner, prepared, monkeypatch):
    directory, packets, records, binary, history = prepared
    monkeypatch.setattr(runner.protocol, "raw_agreement", lambda *args: dict(paired_records=0))
    calls = []
    runner.run(directory, max_native_contexts=1, executor=backend(runner.previous.body.correction, calls))
    (directory / "execution_usage.jsonl").write_bytes(b"\n")
    with pytest.raises(ValueError):
        runner.run(directory, max_native_contexts=1, executor=backend(runner.previous.body.correction, calls))
    assert len(calls) == 1
