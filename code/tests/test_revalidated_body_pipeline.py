"""Pure private-bridge/cache/dispatch tests; no real preparation or model calls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_wire_cohort_pipeline import pipeline, wire_source, runner, source
from test_compact_cohort_pipeline import compact_backend


@pytest.fixture
def wrapper(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("revalidated_body_pipeline_test", scripts / "pipeline_revalidated_body_adjudication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def asymmetric(wrapper, pipeline, wire_source):
    source = wire_source
    first = source["add"](1, "a")
    ids = list(source["packets"])
    records = {"a": {key: first["envelopes"][key] for key in ids[:6]}, "b": {}}
    all_sources = pipeline.wire.record_sources(source["run"], "a")
    sources = {"a": {key: all_sources[key] for key in ids[:6]}, "b": {}}
    context = dict(root=source["root"], pool=source["pool"], run=source["run"], ids=ids,
        rows=source["rows"], questions=source["questions"], packets=source["packets"], chunks=[ids[:6]],
        protected={}, history=deepcopy(source["history"]), source_spec=dict(groups=source["groups"]),
        source_module=pipeline.wire, corrected=False, clarified=False, bindings={"synthetic_source": "fixture-no-model"},
        prompt=pipeline.wire.batch.BATCH_PROMPT.encode(),
        corpus=source["root"] / "data/unified_runs/semantic-page-v2-buffer/stage/passages.jsonl")
    wrapper.seal_cache(context, records, sources, counts={"a": 6, "b": 0})
    return source, context, records, sources


def test_private_bridge_does_not_mutate_original_functions_and_uses_cached_a_coverage(wrapper, pipeline, asymmetric):
    source, context, records, sources = asymmetric
    original = pipeline.snapshot_chunk
    source["add"](1, "b")
    prefix, entries = pipeline.legacy.isolated_base().progress_snapshot(source["run"])
    # The captured A context is historical; only missing B enters new progress.
    entries = [(line, event) for line, event in entries if event["stream"] == "b"]
    bridge = wrapper.private_wire_bridge()
    chunk = bridge.snapshot_chunk(context, source["root"] / "new-source-owner", 1, prefix, entries)
    assert chunk is not None and pipeline.snapshot_chunk is original
    left = wrapper.legacy.full_adjudicator.read_stream(chunk / "independent_a.jsonl")
    assert left == records["a"] and list(left) == context["ids"][:6]
    assert wrapper.native.read_json(chunk / "raw_machine_agreement.json")["paired_records"] == 6


@pytest.mark.parametrize("counts", [{"a": 7, "b": 0}, {"a": 6, "b": 1}, {"a": 6, "b": 0, "c": 0}])
def test_unknown_asymmetry_is_not_admitted(wrapper, asymmetric, counts):
    _, context, records, sources = asymmetric
    with pytest.raises(ValueError, match="six-record"):
        wrapper.seal_cache(context, records, sources, counts=counts)


@pytest.mark.parametrize("kind", ["label", "path", "raw", "receipt"])
def test_cached_a_only_label_receipt_path_and_whole_raw_tampering_stop(wrapper, asymmetric, kind):
    _, context, records, sources = asymmetric
    key = context["ids"][0]
    if kind == "label":
        context["history_cache"]["records"]["a"][key]["label"]["event_status"] = "uncertain"
    elif kind == "path":
        context["history_cache"]["sources"]["a"][key]["receipt"] = "unknown-owner/receipt.json"
    else:
        receipt = Path(sources["a"][key]["receipt"])
        target = receipt if kind == "receipt" else receipt.parent / "raw_final.json"
        target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        wrapper.verify_cache(context)


def test_source_dispatch_requires_registered_future_protocol_and_once_only_usage(wrapper, tmp_path):
    ledger = []
    calls = []
    producer = SimpleNamespace(__file__="synthetic-producer.py", previous=SimpleNamespace(entries=lambda path: ledger),
        verify_attempt=lambda *args: calls.append(1) or {"synthetic": "unchanged source envelope"})
    dispatch = wrapper.source_dispatch(producer)
    receipt = dict(execution_protocol=wrapper.PRODUCER_VERSION, annotation_run=str(tmp_path), execution_id="new-native", usage={})
    with pytest.raises(ValueError, match="once-only"):
        dispatch.verify_attempt(tmp_path, receipt, {})
    ledger.append(dict(execution_id="new-native", usage={}))
    assert dispatch.verify_attempt(tmp_path, receipt, {}) == {"synthetic": "unchanged source envelope"}
    ledger.append(dict(execution_id="new-native", usage={}))
    with pytest.raises(ValueError, match="once-only"):
        dispatch.verify_attempt(tmp_path, receipt, {})
    receipt["execution_protocol"] = "unknown-other-protocol"
    with pytest.raises(ValueError, match="registered"):
        dispatch.verify_attempt(tmp_path, receipt, {})
    assert len(calls) == 3


def test_certificate_cannot_apply_to_other_false_receipt_or_stream(wrapper, monkeypatch, tmp_path):
    import completed_503_admission as admission
    records = {f"synthetic-{index}": dict(label="exact-original") for index in range(10)}
    original = tmp_path / "exact-original/receipt.json"
    certificate = dict(original_receipt=str(original))
    sources = {"a": {}, "b": {key: dict(receipt=str(original),
        receipt_sha256="3d3eb1746772754f9098c17ed8698e511a6d3494745b6ddbd497258a63df915c",
        mechanical_certificate_sha256="71fdd5e5a5e0ab36b29771c1f19208887b89c672d2f7016f3f7cde3ba0fdc56c",
        operational_receipt_accepted=False) for key in records}}
    history = dict(certificate_dir=str(tmp_path), records={"a": {}, "b": records}, sources=sources)
    monkeypatch.setattr(admission, "load", lambda mode, directory: (certificate, records, {}))
    assert wrapper.admitted_history(history) == {}
    history["sources"]["b"]["synthetic-0"]["receipt"] = str(tmp_path / "another-failed-nine/receipt.json")
    with pytest.raises(ValueError, match="another false"):
        wrapper.admitted_history(history)


def test_wrong_frozen_source_bytes_cannot_enable_a_different_namespace_bridge(wrapper, monkeypatch):
    monkeypatch.setattr(wrapper, "WIRE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="dependency changed"):
        wrapper.private_wire_bridge()


def test_configured_future_review_populates_private_context_and_keeps_compact_protocol(wrapper, pipeline, asymmetric, monkeypatch):
    source, context, records, sources = asymmetric
    for number in (2, 3):
        source["add"](number, "a")
        source["add"](number, "b")
    prefix, entries = pipeline.legacy.isolated_base().progress_snapshot(source["run"])
    context["chunks"] = [context["ids"][:6], context["ids"][18:]]
    bridge = wrapper.private_wire_bridge()
    chunk = bridge.snapshot_chunk(context, source["root"] / "future-source-export", 2, prefix, entries)
    monkeypatch.setattr(wrapper, "prepare", lambda *args, **kwargs: context)
    reused = dict(output=None, records=0, chunks={}, sha256={}, source_certificate_admission_path="synthetic-source-sidecar")
    base, holder = wrapper.configure(reused, dict(protected_sha256={}))
    base.prepare(source["root"], source["pool"], source["run"])
    calls = []
    result = base.adjudicate_chunk(chunk, source["root"] / "future-native-owner",
        caller=compact_backend(wrapper.legacy, calls))
    assert holder["context"] is context and result["records"] == 7 and result["accepted"] == 7
    assert calls == [context["ids"][18:]]
    rows = wrapper.protocol.parse_jsonl(Path(result["adjudications_path"]).read_bytes())
    assert all(row["adjudication"]["raw_encoding"] == wrapper.legacy.compact.VERSION for row in rows)


def test_fully_configured_export_only_creation_writes_reuse_receipt_once(wrapper, asymmetric, monkeypatch):
    source, context, records, sources = asymmetric
    # An incomplete future chunk still executes the entire zero-call create path.
    context["chunks"] = [context["ids"][18:]]
    monkeypatch.setattr(wrapper, "prepare", lambda *args, **kwargs: context)
    reused = dict(output=None, records=0, chunks={}, sha256={}, source_certificate_admission_path="synthetic-source-sidecar")
    base, _ = wrapper.configure(reused, dict(protected_sha256={}))
    output, result = base.run(source["root"], source["pool"], source["run"], adjudicate=False,
        caller=lambda *args, **kwargs: pytest.fail("No native call in export-only creation"))
    assert result["new_model_calls"] == 0 and result["exported_chunks"] == 0
    assert (output / "previous_full_response_reuse.json").is_file()
    assert (output / "source_inputs/asymmetric_history_cache.json").is_file()
    assert len(list(output.glob("previous_full_response_reuse.json"))) == 1
    assert base.adjudicator.SYSTEM == wrapper.legacy.compact.SYSTEM


@pytest.mark.parametrize("marker", ["failure.json", "STOP_REQUESTED"])
def test_configured_current_producer_stop_markers_are_not_ignored(wrapper, asymmetric, monkeypatch, marker):
    source, context, records, sources = asymmetric
    monkeypatch.setattr(wrapper, "prepare", lambda *args, **kwargs: context)
    reused = dict(output=None, records=0, chunks={}, sha256={}, source_certificate_admission_path="synthetic-source-sidecar")
    base, _ = wrapper.configure(reused, dict(protected_sha256={}))
    base.prepare(source["root"], source["pool"], source["run"])
    wrapper.native.write_new(source["run"] / marker, b"Synthetic current producer stop")
    with pytest.raises(ValueError, match="producer stop"):
        base.progress_snapshot(source["run"])
    with pytest.raises(ValueError, match="producer stop"):
        base.adjudicate_chunk(source["root"] / "unused-chunk", source["root"] / "unused-owner",
            caller=lambda *args, **kwargs: pytest.fail("No source call after a current producer stop"))
