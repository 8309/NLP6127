"""Composite projection/continuation fixtures do not invoke any model service."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def composite(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("corrected_cohort_test", scripts / "run_codex_corrected_cohort.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_groups_skipped_including_exact_corrective_ten(composite):
    groups = [dict(batch="batch-0019", split="test", record_ids=[f"q{i}" for i in range(10)]),
        dict(batch="batch-0020", split="test", record_ids=[f"q{i}" for i in range(10, 20)]),
        dict(batch="batch-0021", split="development", record_ids=[f"q{i}" for i in range(20, 30)])]
    records = {stream: {f"q{i}": {} for i in range(20)} for stream in ("a", "b")}
    jobs = composite.pending_jobs(groups, records, 100)
    assert [(stream, group["batch"]) for stream, group in jobs] == [("a", "batch-0021"), ("b", "batch-0021")]
    assert groups[0]["split"] == "test" and groups[2]["split"] == "development"


def test_partial_group_never_regrouped_or_rerun(composite):
    groups = [dict(batch="synthetic-batch", record_ids=["a", "b"])]
    with pytest.raises(ValueError, match="Partially accepted"):
        composite.pending_jobs(groups, dict(a={"a": {}}, b={}), 2)


def test_native_context_bound_is_explicit(composite):
    groups = [dict(batch="synthetic-one", record_ids=["one"]), dict(batch="synthetic-two", record_ids=["two"])]
    assert len(composite.pending_jobs(groups, dict(a={}, b={}), 1)) == 1
    with pytest.raises(ValueError, match="positive explicit"):
        composite.pending_jobs(groups, dict(a={}, b={}), 0)


@pytest.fixture
def history_fixture(composite, monkeypatch, tmp_path):
    source, pair = tmp_path / "source", tmp_path / "source/correct/declared"
    pair.mkdir(parents=True)
    ids = [f"synthetic-corrective-{i}" for i in range(10)]
    packets = {identifier: {"synthetic": identifier} for identifier in ids}
    composite.native.write_new(source / "packets.json", packets)
    composite.native.write_new(source / "manifest.json", dict(specification=dict(version=composite.wire.VERSION,
        eligible_record_ids=ids, groups=[dict(batch="batch-0019", record_ids=ids)]), immutable_sha256={}))
    composite.native.write_new(source / "protected_inputs.json", {})
    composite.native.write_new(source / "progress.jsonl", b"synthetic immutable failed progress\n")
    composite.native.write_new(source / "history.json", dict(protected_sha256={}))
    composite.native.write_new(pair / "metadata_certificate.json", dict(verification=dict(record_ids=ids)))
    old = dict(a={identifier: {"label": "old-a"} for identifier in ids}, b={})
    new = {stream: {identifier: {"label": "new-" + stream} for identifier in ids} for stream in ("a", "b")}
    monkeypatch.setattr(composite.wire, "completed_records", lambda source, stream, packets: old[stream])
    monkeypatch.setattr(composite.wire, "record_sources", lambda source, stream: {identifier: {"lineage": "original-" + stream} for identifier in old[stream]})
    monkeypatch.setattr(composite.metadata, "completed_correction_records", lambda pair, stream: new[stream])
    monkeypatch.setattr(composite.metadata, "correction_record_sources", lambda pair, stream: {identifier: {"lineage": "corrective-" + stream} for identifier in ids})
    monkeypatch.setattr(composite.protocol, "validate_independent_output", lambda *args: [])
    monkeypatch.setattr(composite.protocol, "raw_agreement", lambda *args: dict(paired_records=0, missing_b=ids))
    return source, pair, packets, old, new


def test_explicit_ten_overlay_preserves_raw_history_and_baseline(composite, history_fixture):
    source, pair, packets, old, new = history_fixture
    raw_progress = (source / "progress.jsonl").read_bytes()
    history, records = composite.load_history(source, pair, packets)
    assert records == new
    assert old["a"][history["corrective_record_ids"][0]]["label"] == "old-a"
    assert history["original_accepted_counts"] == {"a": 10, "b": 0}
    assert (source / "progress.jsonl").read_bytes() == raw_progress
    baseline = history["first_attempt_agreement_artifact"]
    assert composite.protocol.sha256(Path(baseline["path"]).read_bytes()) == baseline["sha256"]
    assert composite.load_history(source, pair, packets)[0] == history


def test_overlay_cannot_replace_an_already_accepted_original_B(composite, history_fixture, monkeypatch):
    source, pair, packets, old, _ = history_fixture
    old["b"] = dict(old["a"])
    with pytest.raises(ValueError, match="Only the exact rejected"):
        composite.load_history(source, pair, packets)


def test_overlay_requires_same_canonical_source_packets(composite, history_fixture):
    source, pair, packets, _, _ = history_fixture
    altered = dict(packets)
    altered[next(iter(altered))] = {"synthetic": "altered source"}
    with pytest.raises(ValueError, match="canonical packets differ"):
        composite.load_history(source, pair, altered)


def test_usage_includes_failed_and_old_A_once_not_per_label(composite, tmp_path):
    output, source, pair = tmp_path / "new", tmp_path / "original", tmp_path / "correction"
    older, old = tmp_path / "older", tmp_path / "old"
    composite.native.write_new(output / "history.json", dict(source_run=str(source), corrective_pair=str(pair)))
    composite.native.write_new(source / "history.json", dict(older_source_run=str(older), source_run=str(old)))
    for index, directory in enumerate((older, old, source, pair, output)):
        composite.native.write_new(directory / "execution_usage.jsonl", composite.protocol.canonical_bytes(dict(
            execution_id=f"synthetic-execution-{index}", usage=dict(input_tokens=index + 1, output_tokens=2))) + b"\n")
    value = composite.merged_usage(output)
    assert value["executions"] == 5 and value["native_token_totals"] == {"input_tokens": 15, "output_tokens": 10}
    assert value["subscription_dollar_cost"] is None


def test_duplicate_actual_execution_usage_fails(composite, tmp_path):
    output, source, pair = tmp_path / "new", tmp_path / "original", tmp_path / "correction"
    older, old = tmp_path / "older", tmp_path / "old"
    composite.native.write_new(output / "history.json", dict(source_run=str(source), corrective_pair=str(pair)))
    composite.native.write_new(source / "history.json", dict(older_source_run=str(older), source_run=str(old)))
    for directory in (source, pair):
        composite.native.write_new(directory / "execution_usage.jsonl", composite.protocol.canonical_bytes(dict(
            execution_id="synthetic-duplicated-execution", usage=None)) + b"\n")
    with pytest.raises(ValueError, match="duplicated"):
        composite.merged_usage(output)
