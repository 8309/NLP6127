"""Synthetic ownership, callback and population tests; no native model calls."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from test_transport_review_recovery import recovery, retry_case, runner, source, accepted_caller


@pytest.fixture
def body_pipeline(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("body_quote_pipeline_tests", scripts / "pipeline_body_quote_adjudication.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def configured_case(body_pipeline, retry_case, runner, monkeypatch):
    case = retry_case
    source = case["source"]
    source["add"](5, "a")
    source["add"](5, "b")
    context = runner.prepare(source["root"], source["pool"], source["run"])
    prefix, entries = runner.progress_snapshot(source["run"])
    fresh_chunk = runner.export_chunk(context, source["root"] / "fresh-export", 2, prefix, entries)
    records = {side: runner.batch.completed_records(source["run"], side, source["packets"]) for side in ("a", "b")}
    context.update(history=dict(record_ids=list(source["packets"]), protected_sha256={}, sources={"a": {}, "b": {}}),
        chunks=[[] for _ in range(28)] + [list(source["packets"])[:20], list(source["packets"])[20:]])
    context["history_cache"] = dict(records=records, history=context["history"], record_ids=context["history"]["record_ids"], protected_sha256={})
    context["history_cache_sha256"] = body_pipeline.protocol.sha256(body_pipeline.protocol.canonical_bytes(context["history_cache"]))
    calls = []
    gate = body_pipeline.previous.one_retry(case["base"], case["chunk"], case["output"], case["stopped"],
        case["authorization"], caller=accepted_caller(body_pipeline.previous, case, calls))
    pending = dict(source_chunk=str(case["chunk"]), output_parent=str(case["output"]),
        bounded_output=str(case["output"] / "bounded-run"), protected_sha256={}, gate_records=gate["records"])
    reused = dict(output=None, records=0, chunks={}, sha256={})
    monkeypatch.setattr(body_pipeline, "prepare", lambda *args, **kwargs: context)
    base = body_pipeline.configure(body_pipeline.legacy.isolated_base(), reused, case["stopped"], pending, case["base"])
    base.prepare(source["root"], source["pool"], source["run"])
    return {**case, "configured": base, "context": context, "fresh_chunk": fresh_chunk,
            "pending": pending, "gate_calls": calls}


def test_nonpending_fresh_review_context_holder_is_populated(body_pipeline, configured_case):
    case = configured_case
    calls = []
    result = case["configured"].adjudicate_chunk(case["fresh_chunk"], case["source"]["root"] / "body-new-owner",
        caller=accepted_caller(body_pipeline.previous, case, calls))
    assert result["records"] == 5 and result["accepted"] == 5 and result["new_native_calls"] == 1
    assert len(calls) == 1


def test_pending_gate_is_not_recalled_and_whole_result_keeps_original_owner(body_pipeline, configured_case):
    case = configured_case
    calls = []
    result = case["configured"].adjudicate_chunk(case["chunk"], case["source"]["root"] / "not-the-owner",
        caller=accepted_caller(body_pipeline.previous, case, calls))
    assert result["records"] == 20 and result["new_native_calls"] == 1 and result["retained_native_calls"] == 2
    assert calls == [list(case["source"]["packets"])[10:20]]
    path = Path(result["adjudications_path"])
    assert path.parent == Path(case["pending"]["bounded_output"])
    result_path = case["source"]["root"] / "pending-result.json"
    body_pipeline.native.write_new(result_path, result)
    assert body_pipeline.verify_pending_result(case["pending"], case["chunk"], list(case["source"]["packets"])[:20], result_path) == result
    originals = body_pipeline.protocol.parse_jsonl(path.read_bytes())
    assert all(Path(row["adjudication"]["raw_response_path"]).is_relative_to(path.parent / "executions") for row in originals)


def test_source_mutation_during_review_fails_post_execution_guard(body_pipeline, configured_case):
    case = configured_case
    path = case["source"]["root"] / "source-guard.txt"
    body_pipeline.native.write_new(path, b"original source")
    case["context"]["protected"][str(path)] = body_pipeline.protocol.sha256(path.read_bytes())
    backend = accepted_caller(body_pipeline.previous, case, [])
    def mutate(*args, **kwargs):
        result = backend(*args, **kwargs)
        path.write_bytes(b"changed source")
        return result
    with pytest.raises(ValueError):
        case["configured"].adjudicate_chunk(case["fresh_chunk"], case["source"]["root"] / "body-new-owner", caller=mutate)


def test_corrective_body_and_transport_pairs_are_excluded_from_first_attempt_population(body_pipeline, configured_case):
    case = configured_case
    chunk = case["fresh_chunk"]
    context = deepcopy(case["context"])
    ids = list(body_pipeline.native.read_json(chunk / "packets.json"))
    for side in ("a", "b"):
        path = chunk / f"independent_{side}.jsonl"
        rows = body_pipeline.protocol.parse_jsonl(path.read_bytes())
        rows[0].update(annotation_role=body_pipeline.CORRECTIVE_ROLE, original_first_attempt=False)
        rows[1].update(annotation_role=body_pipeline.previous.RETRY_ROLE, original_first_attempt=False)
        for row in rows[2:]:
            row.update(annotation_role=body_pipeline.FIRST_ROLE, original_first_attempt=True)
        path.write_bytes(b"".join(body_pipeline.protocol.canonical_bytes(row) + b"\n" for row in rows))
        context["history_cache"]["records"][side].update({row["record_id"]: row for row in rows})
    population = body_pipeline.validate_roles(context, chunk)
    assert population["body_corrective_record_ids"] == ids[:1]
    assert population["transport_retry_record_ids"] == ids[1:2]
    assert population["body_only_first_attempt_record_ids"] == ids[2:]
    context["history_cache"]["records"]["b"][ids[0]]["original_first_attempt"] = True
    rows[0]["original_first_attempt"] = True
    (chunk / "independent_b.jsonl").write_bytes(b"".join(body_pipeline.protocol.canonical_bytes(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="cannot enter"):
        body_pipeline.validate_roles(context, chunk)


def test_unknown_body_source_protocol_is_not_dynamically_dispatched(body_pipeline, monkeypatch):
    import run_body_quote_cohort as producer
    monkeypatch.setattr(producer, "VERSION", "unknown-body-source")
    with pytest.raises(ValueError, match="Unregistered"):
        body_pipeline.producer_module()


@pytest.mark.parametrize("marker", ["STOP_REQUESTED", "failure.json"])
def test_current_producer_stop_markers_block_without_reading_old_failures(body_pipeline, configured_case, marker):
    case = configured_case
    body_pipeline.native.write_new(case["context"]["run"] / marker, b"current producer stopped")
    with pytest.raises(ValueError, match="Current body producer"):
        case["configured"].progress_snapshot(case["context"]["run"])
    with pytest.raises(ValueError, match="Current body producer"):
        case["configured"].adjudicate_chunk(case["fresh_chunk"], case["source"]["root"] / "unused-owner",
            caller=lambda *args, **kwargs: pytest.fail("No new call after current producer stop"))
