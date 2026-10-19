"""Synthetic durable wire exports and original review ownership; no model calls."""
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest

from test_cohort_adjudication_pipeline import runner, source, paired
from test_compact_cohort_pipeline import compact_pipeline, compact_backend
from test_correct_codex_wire_batch import adapter, prepared, rows, backend as corrective_backend
from types import SimpleNamespace


@pytest.fixture
def pipeline(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("wire_pipeline_tests", scripts / "pipeline_wire_cohort_adjudication.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def wire_source(pipeline, source, monkeypatch):
    for row in source["questions"]:
        row["split"] = "development"
    run = source["pool"] / "cohort_annotations_wire/synthetic-wire"
    history = dict(record_ids=[], source_run=str(source["pool"] / "empty"), older_source_run=str(source["pool"] / "empty-older"),
        protected_sha256={}, sources={"a": {}, "b": {}})
    groups = pipeline.wire.build_groups(source["rows"], source["questions"], [], 9, 262144)
    packets = source["packets"]
    values = dict(packets=packets)
    for name, value in (("packets.json", packets), ("protocol.txt", pipeline.wire.batch.BATCH_PROMPT.encode()),
        ("wire_protocol.txt", (pipeline.wire.batch.BATCH_PROMPT + pipeline.wire_codec.WIRE_INSTRUCTIONS).encode()),
        ("groups.json", groups), ("history.json", history), ("protected_inputs.json", {})):
        pipeline.native.write_new(run / name, value)
    spec = dict(version=pipeline.wire.VERSION, partition="all_current", bindings={"synthetic_source": "fixture-no-model"},
        eligible_record_ids=list(packets), case_limit=9, full_payload_byte_limit=262144,
        historical_labels_unchanged=True, canonical_semantic_hash_is_actual_wire_hash=False,
        batch_prompt_sha256=pipeline.protocol.sha256(pipeline.wire.batch.BATCH_PROMPT.encode()),
        eligible_packets_sha256=pipeline.protocol.sha256(pipeline.protocol.canonical_bytes(packets)),
        wire_encoding=pipeline.wire_codec.VERSION,
        wire_instructions_sha256=pipeline.protocol.sha256(pipeline.wire_codec.WIRE_INSTRUCTIONS.encode()),
        source_run=history["source_run"], snapshot_dir="synthetic-snapshot",
        history_sha256=pipeline.protocol.sha256(pipeline.protocol.canonical_bytes(history)), groups=groups)
    manifest = dict(specification=spec, immutable_sha256={path.name: pipeline.protocol.sha256(path.read_bytes()) for path in run.iterdir()})
    pipeline.native.write_new(run / "manifest.json", manifest)
    monkeypatch.setattr(pipeline.wire, "load_history", lambda *args: (history, {}))

    def executor(command, stdin, context, attempt, environment, timeout):
        payload = pipeline.native.read_json(context / "packets.json")
        result = {}
        for alias, row in payload.items():
            evidence = row["evidence"][0]
            result[alias] = dict(record_id=alias, label=dict(answerability="sufficient", event_status="observed",
                qrels={value["passage_id"]: 2 if index == 0 else 0 for index, value in enumerate(row["evidence"])},
                answer_points=[dict(text=evidence["text"], citations=[evidence["passage_id"]])],
                quotes=[dict(passage_id=evidence["passage_id"], text=evidence["text"])], reasons=["Synthetic full own-source evidence."]))
        pipeline.native.write_new(context / "final.json", (json.dumps(result, indent=2) + "\r\n").encode())
        events = [dict(type="thread.started", thread_id="synthetic-wire-" + uuid4().hex), dict(type="turn.completed", usage={})]
        pipeline.native.write_new(attempt / "events.jsonl", b"".join(pipeline.protocol.canonical_bytes(row) + b"\n" for row in events))
        pipeline.native.write_new(attempt / "stderr.txt", b"")
        return 0

    def add(number, side, durable=True):
        group = groups[number - 1]
        rows = [packets[key]["payload"] for key in group["record_ids"]]
        result = pipeline.wire.execute_attempt(run, side, group, rows, source["root"] / "native.exe", {}, [], {}, None, 60, executor)
        assert result["receipt"]["accepted"], result["receipt"]
        if durable:
            pipeline.native.append_progress(run / "progress.jsonl", dict(type="completed", accepted=True, stream=side,
                batch=group["batch"], record_ids=group["record_ids"], receipt=(result["attempt"] / "receipt.json").relative_to(run).as_posix(),
                receipt_sha256=pipeline.protocol.sha256((result["attempt"] / "receipt.json").read_bytes())))
        return result
    return {**source, "run": run, "groups": groups, "history": history, "add": add, "executor": executor}


def empty_reuse():
    return dict(output=None, records=0, chunks={}, sha256={})


def seal_all(value):
    for number in range(1, len(value["groups"]) + 1):
        value["add"](number, "a")
        value["add"](number, "b")


def test_fixed_order_and_exact_prompt_roundtrip(pipeline, wire_source):
    context = pipeline.prepare(wire_source["root"], wire_source["pool"], wire_source["run"])
    assert context["ids"] != sorted(context["ids"])
    assert context["chunks"] == [context["ids"][:20], context["ids"][20:]]
    assert "event_group" not in json.dumps(context["packets"])
    assert context["prompt"] == pipeline.wire.batch.BATCH_PROMPT.encode()


def test_receipt_before_progress_and_one_sided_pair_cannot_export(pipeline, wire_source):
    wire_source["add"](1, "a", durable=False)
    for number in range(1, 4):
        wire_source["add"](number, "b")
    base = pipeline.configure(pipeline.legacy.isolated_base(), empty_reuse())
    output, result = base.run(wire_source["root"], wire_source["pool"], wire_source["run"], max_idle_seconds=0,
        caller=lambda *args, **kwargs: pytest.fail("No incomplete chunk review"))
    assert result["exported_chunks"] == result["new_model_calls"] == 0
    assert not (output / "chunks").exists()


def test_fixed_chunk_spanning_native_group_retains_whole_raw_codec(pipeline, wire_source):
    seal_all(wire_source)
    base = pipeline.configure(pipeline.legacy.isolated_base(), empty_reuse())
    output, result = base.run(wire_source["root"], wire_source["pool"], wire_source["run"], adjudicate=False)
    assert result["exported_chunks"] == 2
    chunk = output / "chunks/chunk-0001"
    records = pipeline.legacy.full_adjudicator.read_stream(chunk / "independent_a.jsonl")
    last = records[wire_source["rows"][19]["record_id"]]
    assert len(json.loads(last["raw_output"])) == 7
    assert last["raw_codec"]["ordered_canonical_record_ids"] == [row["record_id"] for row in wire_source["rows"][18:]]
    assert last["label"]["qrels"]["future-19"] == 0
    assert len(records) == 20
    receipt = pipeline.native.read_json(chunk / "export_receipt.json")
    assert receipt["whole_native_groups_preserved"] and receipt["raw_agreement_computed_before_adjudication"]
    assert len(list((chunk / "durable_snapshot/a").glob("*/attempt-*/codec.json"))) == 3
    source_map = pipeline.native.read_json(chunk / "original_record_sources.json")
    assert source_map["a"][last["record_id"]]["receipt"].startswith(str(wire_source["run"]))


@pytest.mark.parametrize("name", ["codec.json", "stdin.txt", "schema.json", "raw_final.json"])
def test_tampered_whole_native_artifact_fails_before_review(pipeline, wire_source, name):
    seal_all(wire_source)
    prefix, entries = pipeline.legacy.isolated_base().progress_snapshot(wire_source["run"])
    event = entries[0][1]
    path = wire_source["run"] / Path(event["receipt"]).parent / name
    path.write_bytes(path.read_bytes() + b" ")
    context = pipeline.prepare(wire_source["root"], wire_source["pool"], wire_source["run"])
    with pytest.raises(ValueError, match="changed"):
        pipeline.snapshot_chunk(context, wire_source["root"] / "private-export", 1, prefix, entries)


def test_compact_full_source_review_alias_inputs_and_no_repeat_resume(pipeline, wire_source):
    seal_all(wire_source)
    base = pipeline.configure(pipeline.legacy.isolated_base(), empty_reuse())
    calls = []
    output, result = base.run(wire_source["root"], wire_source["pool"], wire_source["run"], caller=compact_backend(pipeline.legacy, calls))
    assert result["complete"] and result["accepted"] == 25 and result["new_model_calls"] == 3
    assert [key for group in calls for key in group] == list(wire_source["packets"])
    for path in result["adjudication_files"]:
        for row in pipeline.protocol.parse_jsonl(Path(path).read_bytes()):
            assert Path(row["adjudication"]["raw_response_path"]).is_relative_to(Path(path).parent / "executions")
    _, resumed = base.run(wire_source["root"], wire_source["pool"], wire_source["run"], resume=output,
        caller=lambda *args, **kwargs: pytest.fail("Retained source contexts must not replay"))
    assert resumed["new_model_calls"] == 0 and resumed["retained_model_calls"] == 3


def test_failed_source_progress_stops_without_retry(pipeline, wire_source):
    pipeline.native.write_new(wire_source["run"] / "progress.jsonl", b'{"type":"failed"}\n')
    base = pipeline.configure(pipeline.legacy.isolated_base(), empty_reuse())
    with pytest.raises(ValueError, match="stops without retry"):
        base.run(wire_source["root"], wire_source["pool"], wire_source["run"],
            caller=lambda *args, **kwargs: pytest.fail("No automatic quota retry"))


def test_cross_source_reuse_rejects_different_original_context(pipeline, wire_source, source, compact_pipeline):
    paired(source, 5)
    previous, result = compact_pipeline.run(source["root"], source["pool"], source["run"], caller=compact_backend(compact_pipeline, []))
    seal_all(wire_source)
    path = previous / "retained/chunk-0001.json"
    reused = dict(output=str(previous), records=20, sha256={}, chunks={"1": dict(result_path=str(path),
        source_chunk=str(previous / "chunks/chunk-0001"), result=compact_pipeline.native.read_json(path), record_ids=list(source["packets"])[:20])})
    base = pipeline.configure(pipeline.legacy.isolated_base(), reused)
    with pytest.raises(ValueError, match="whole original A/B envelopes"):
        base.run(wire_source["root"], wire_source["pool"], wire_source["run"],
            caller=lambda *args, **kwargs: pytest.fail("Different native originals cannot be reused"))


def test_exact_original_reviews_keep_native_file_ownership_without_calls(pipeline, wire_source):
    seal_all(wire_source)
    original = pipeline.configure(pipeline.legacy.isolated_base(), empty_reuse())
    previous, first = original.run(wire_source["root"], wire_source["pool"], wire_source["run"],
        caller=compact_backend(pipeline.legacy, []))
    unchanged = {str(path): path.read_bytes() for path in previous.rglob("*") if path.is_file()}
    chunks = {}
    for number, path in enumerate(sorted((previous / "retained").glob("*.json")), 1):
        chunks[str(number)] = dict(result_path=str(path), source_chunk=str(previous / "chunks" / f"chunk-{number:04d}"),
            result=pipeline.native.read_json(path), record_ids=list(wire_source["packets"])[(number - 1) * 20:number * 20])
    reused = dict(output=str(previous), records=25, chunks=chunks,
        sha256={name: pipeline.protocol.sha256(value) for name, value in unchanged.items()})
    base = pipeline.configure(pipeline.legacy.isolated_base(), reused)
    output, second = base.run(wire_source["root"], wire_source["pool"], wire_source["run"],
        caller=lambda *args, **kwargs: pytest.fail("No repeated accepted context"))
    assert second["new_model_calls"] == 0 and second["adjudication_files"] == first["adjudication_files"]
    assert all(Path(path).read_bytes() == raw for path, raw in unchanged.items())
    assert not (output / "adjudications").exists()
    _, resumed = base.run(wire_source["root"], wire_source["pool"], wire_source["run"], resume=output,
        caller=lambda *args, **kwargs: pytest.fail("No native calls on reuse resume"))
    assert resumed["new_model_calls"] == 0


def test_source_packet_or_protocol_tampering_rejected(pipeline, wire_source):
    path = wire_source["run"] / "wire_protocol.txt"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Immutable"):
        pipeline.prepare(wire_source["root"], wire_source["pool"], wire_source["run"])


def test_historical_only_chunk_has_absent_not_empty_wire_progress(pipeline, wire_source, source):
    paired(source, 5)
    context = pipeline.prepare(wire_source["root"], wire_source["pool"], wire_source["run"])
    records = {side: pipeline.wire.batch.completed_records(source["run"], side, source["packets"]) for side in ("a", "b")}
    sources = {side: {} for side in ("a", "b")}
    prefix, events = pipeline.legacy.isolated_base().progress_snapshot(source["run"])
    for _, event in events:
        receipt_path = source["run"] / event["receipt"]
        receipt = pipeline.native.read_json(receipt_path)
        for key in event["record_ids"]:
            sources[event["stream"]][key] = dict(receipt=str(receipt_path), receipt_sha256=event["receipt_sha256"],
                execution_id=receipt["execution_id"], stdin_sha256=receipt["full_stdin_sha256"],
                source_run=str(source["run"]), migrated_unchanged=True)
    history = dict(record_ids=list(source["packets"]), source_run=str(source["pool"] / "empty"),
        older_source_run=str(source["run"]), protected_sha256={}, sources=sources)
    history_run = source["pool"] / "history-only-fixture"
    pipeline.native.write_new(history_run / "history.json", history)
    context.update(run=history_run, history=history, source_spec={"groups": []})
    pipeline.seal_history_cache(context, records, sources)
    chunk = pipeline.snapshot_chunk(context, source["root"] / "history-export", 1, b"", [])
    assert not (chunk / "durable_snapshot/progress.jsonl").exists()
    assert pipeline.legacy.full_adjudicator.read_stream(chunk / "independent_a.jsonl") == {key: records["a"][key] for key in list(source["packets"])[:20]}


@pytest.fixture
def resolution_case(pipeline, wire_source, monkeypatch):
    """Explicit root-authorization fixture, not an actual production resolution."""
    context = pipeline.prepare(wire_source["root"], wire_source["pool"], wire_source["run"])
    group = {**wire_source["groups"][0], "batch": "batch-0019"}
    context["source_spec"]["groups"] = [group]
    run = wire_source["run"]
    rows = [context["packets"][key]["payload"] for key in group["record_ids"]]
    def execute(side, invalid=False):
        def backend(*args):
            status = wire_source["executor"](*args)
            if invalid:
                context_path = args[2]
                final = context_path / "final.json"
                value = pipeline.native.read_json(final)
                value["r0"]["label"]["quotes"] = []
                final.write_bytes((json.dumps(value, indent=2) + "\r\n").encode())
            return status
        return pipeline.wire.execute_attempt(run, side, group, rows, wire_source["root"] / "native.exe",
            {}, [], {}, None, 60, backend)
    counterpart, old = execute("a"), execute("b", True)
    assert old["receipt"]["accepted"] is False and old["receipt"]["errors"] == [pipeline.AUTHORIZED_SOURCE_ERROR]
    def event(result):
        receipt_path = result["attempt"] / "receipt.json"
        return dict(type="completed", accepted=result["receipt"]["accepted"], stream=result["receipt"]["stream"],
            batch="batch-0019", record_ids=group["record_ids"], receipt=receipt_path.relative_to(run).as_posix(),
            receipt_sha256=pipeline.protocol.sha256(receipt_path.read_bytes()))
    for result in (counterpart, old):
        pipeline.native.append_progress(run / "progress.jsonl", event(result))
    prefix = (run / "progress.jsonl").read_bytes()
    retry = execute("b")
    pipeline.native.append_progress(run / "progress.jsonl", event(retry))
    directory = run / "retry/synthetic-resolution"
    pipeline.native.write_new(directory / "authorization.json", dict(root_authorized=True, retries_authorized=1,
        reason="Synthetic controlled exact-input source-validation retry."))
    pipeline.native.write_new(directory / "request.json", dict(synthetic=True, no_label_feedback=True))
    pipeline.native.write_new(directory / "original_progress.jsonl", prefix)
    original_hash = event(old)["receipt_sha256"]
    monkeypatch.setattr(pipeline, "AUTHORIZED_FAILURE_SHA256", original_hash)
    value = dict(annotation_run=str(run), stream="b", batch="batch-0019", record_ids=group["record_ids"],
        original_failure_receipt=str(old["attempt"] / "receipt.json"), original_failure_receipt_sha256=original_hash,
        counterpart_a_receipt=str(counterpart["attempt"] / "receipt.json"), counterpart_a_receipt_sha256=event(counterpart)["receipt_sha256"],
        retry_receipt=str(retry["attempt"] / "receipt.json"), retry_receipt_sha256=event(retry)["receipt_sha256"],
        original_execution_id=old["receipt"]["execution_id"], retry_execution_id=retry["receipt"]["execution_id"],
        original_failure_retained=True, retry_accepted=True, retries_authorized=1, native_calls_actual=1, no_label_feedback=True,
        authorization_sha256=pipeline.protocol.sha256((directory / "authorization.json").read_bytes()),
        request_sha256=pipeline.protocol.sha256((directory / "request.json").read_bytes()),
        original_progress_prefix=dict(path="original_progress.jsonl", byte_count=len(prefix), sha256=pipeline.protocol.sha256(prefix)),
        physical_equal_artifact_hashes={name: old["receipt"]["artifacts_sha256"][name] for name in
            ("stdin.txt", "schema.json", "codec.json", "wire_payload.json", "canonical_semantic_packets.json")},
        same_requested_model=True, unchanged_A_and_other_accepted_receipts={str(counterpart["attempt"] / "receipt.json"):event(counterpart)["receipt_sha256"]})
    path = directory / "resolution.json"
    pipeline.native.write_new(path, value)
    return dict(context=context, path=path, value=value, old=old, retry=retry, counterpart=counterpart, run=run)


def test_explicit_resolution_keeps_failure_raw_but_selects_only_accepted_retry(pipeline, resolution_case):
    original = {str(path): path.read_bytes() for path in resolution_case["old"]["attempt"].iterdir()}
    proof = pipeline.validate_resolution(resolution_case["context"], resolution_case["path"])
    prefix, projected = pipeline.resolved_progress_snapshot(resolution_case["run"], proof)
    assert prefix == (resolution_case["run"] / "progress.jsonl").read_bytes()
    assert len(projected) == 2 and all(row["accepted"] for _, row in projected)
    assert proof["failure_reclassified_as_accepted"] is False
    assert all(Path(path).read_bytes() == value for path, value in original.items())


@pytest.mark.parametrize("mutation", ["unauthorized", "unknown_failure", "different_input", "same_context", "prior_A_hash"])
def test_resolution_invalid_authority_identity_or_inputs_fail_closed(pipeline, resolution_case, mutation):
    value = resolution_case["value"]
    if mutation == "unauthorized":
        value["retries_authorized"] = 2
    elif mutation == "unknown_failure":
        value["original_failure_receipt_sha256"] = "0" * 64
    elif mutation == "different_input":
        value["physical_equal_artifact_hashes"]["stdin.txt"] = "0" * 64
    elif mutation == "same_context":
        value["retry_execution_id"] = value["original_execution_id"]
    else:
        value["unchanged_A_and_other_accepted_receipts"] = {}
    resolution_case["path"].write_bytes(pipeline.protocol.canonical_bytes(value))
    with pytest.raises(ValueError):
        pipeline.validate_resolution(resolution_case["context"], resolution_case["path"])


@pytest.mark.parametrize("kind", ["new_failure", "quota", "rerun_accepted", "rewritten_prefix"])
def test_new_unresolved_quota_or_duplicate_events_still_stop(pipeline, resolution_case, kind):
    proof = pipeline.validate_resolution(resolution_case["context"], resolution_case["path"])
    path = resolution_case["run"] / "progress.jsonl"
    if kind == "rewritten_prefix":
        path.write_bytes(b" " + path.read_bytes())
    else:
        row = dict(type="failed", stream="b", batch="batch-0020", errors=["quota" if kind == "quota" else "missing quote"])
        if kind == "rerun_accepted":
            row = pipeline.protocol.parse_jsonl(path.read_bytes())[0]
            row.pop("at", None)
        pipeline.native.append_progress(path, row)
    with pytest.raises(ValueError, match="failure|rerun|rewritten"):
        pipeline.resolved_progress_snapshot(resolution_case["run"], proof)


def test_without_explicit_resolution_original_failure_still_stops(pipeline, resolution_case):
    with pytest.raises(ValueError, match="stops without retry"):
        pipeline.resolved_progress_snapshot(resolution_case["run"])


def test_failed_retry_cannot_resolve_original_rejected_batch(pipeline, resolution_case):
    value = resolution_case["value"]
    value["retry_accepted"] = False
    resolution_case["path"].write_bytes(pipeline.protocol.canonical_bytes(value))
    with pytest.raises(ValueError, match="authorization"):
        pipeline.validate_resolution(resolution_case["context"], resolution_case["path"])


@pytest.fixture
def corrective_context(pipeline, adapter, prepared):
    """A genuine mock-executed pair; source-history authority remains synthetic."""
    directory, authorization, setup, source = prepared
    adapter.execute_pair_once(directory, authorization, native_setup=setup, executor=corrective_backend(adapter, []))
    packets = adapter.native.read_json(directory / "canonical_semantic_packets.json")
    first_attempt = directory / "synthetic_first_attempt_agreement.json"
    adapter.native.write_new(first_attempt, dict(expected_records=10, paired_records=0,
        agreement_population="synthetic_original_atomic_rejections"))
    protected = {str(path): pipeline.protocol.sha256(path.read_bytes()) for path in directory.rglob("*") if path.is_file()}
    history = dict(record_ids=list(packets), corrective_record_ids=list(packets), corrective_pair=str(directory),
        protected_sha256=protected, first_attempt_agreement_artifact=dict(path=str(first_attempt),
            sha256=pipeline.protocol.sha256(first_attempt.read_bytes()), agreement_population="originalacceptedfirstattemptpairs_only"))
    view = directory / "synthetic_composite_view"
    adapter.native.write_new(view / "history.json", history)
    module = SimpleNamespace(VERSION=pipeline.COMPOSITE_VERSION, __file__=adapter.__file__,
        completed_records=lambda output, side, checked_packets: adapter.completed_correction_records(directory, side),
        record_sources=lambda output, side: adapter.correction_record_sources(directory, side),
        verify_attempt=lambda attempt, receipt, packets: adapter.verify_corrective_attempt(directory, receipt["stream"])[1])
    return dict(root=directory, run=view, chunks=[list(packets)], ids=list(packets), packets=packets,
        rows=[packet["payload"] for packet in packets.values()], questions=[dict(record_id=key, split="test") for key in packets],
        history=history, source_spec={"groups": []}, source_module=module, corrected=True)


def test_corrective_pair_export_retains_new_inputs_roles_original_baseline_separate(pipeline, corrective_context):
    context = corrective_context
    chunk = pipeline.snapshot_chunk(context, context["root"] / "private_export", 1, b"", [])
    left, right = (pipeline.legacy.full_adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b"))
    for key in context["ids"]:
        assert left[key]["execution_input_sha256"] == right[key]["execution_input_sha256"]
        assert left[key]["input_sha256"] == context["packets"][key]["input_sha256"]
        assert left[key]["context_id"] != right[key]["context_id"]
        assert left[key]["excluded_from_original_agreement"] and right[key]["excluded_from_original_agreement"]
    note = pipeline.native.read_json(chunk / "agreement_population.json")
    assert note["original_first_attempt_agreement"] is False
    assert note["first_attempt_agreement_artifact"] == context["history"]["first_attempt_agreement_artifact"]
    for side in ("a", "b"):
        assert (chunk / "corrective_raw" / side / "instruction.txt").is_file()
        assert (chunk / "corrective_raw" / side / "stdin.txt").read_bytes() == (Path(context["history"]["corrective_pair"]) / "stdin.txt").read_bytes()
    assert (chunk / "corrective_proof/certificate.json").is_file()
    assert pipeline.native.read_json(chunk / "raw_machine_agreement.json")["paired_records"] == 10


def test_corrective_pair_uses_existing_compact_source_reviewer_without_guard_changes(pipeline, corrective_context):
    context = corrective_context
    chunk = pipeline.snapshot_chunk(context, context["root"] / "private_export", 1, b"", [])
    base = pipeline.configure(pipeline.legacy.isolated_base(), empty_reuse())
    calls = []
    result = base.adjudicate_chunk(chunk, context["root"] / "private_reviews", caller=compact_backend(pipeline.legacy, calls))
    assert result["accepted"] == 10 and result["unresolved"] == 0
    path = Path(result["adjudications_path"])
    for row in pipeline.protocol.parse_jsonl(path.read_bytes()):
        assert Path(row["adjudication"]["raw_response_path"]).is_relative_to(path.parent / "executions")
    assert [key for group in calls for key in group] == context["ids"]


def test_corrective_declared_role_cannot_masquerade_as_original(pipeline, corrective_context):
    context = corrective_context
    actual = context["source_module"].completed_records
    def dishonest(output, side, packets):
        values = actual(output, side, packets)
        values[context["ids"][0]]["excluded_from_original_agreement"] = False
        return values
    context["source_module"].completed_records = dishonest
    with pytest.raises(ValueError, match="explicit role"):
        pipeline.snapshot_chunk(context, context["root"] / "private_export", 1, b"", [])


@pytest.fixture
def metadata_corrective_context(pipeline, corrective_context):
    """Isolated dispatch fixture; the real metadata verifier has its own tests."""
    context = corrective_context
    module = context["source_module"]
    originals = {side: module.completed_records(context["run"], side, context["packets"]) for side in ("a", "b")}
    sources = {side: module.record_sources(context["run"], side) for side in ("a", "b")}
    pair = Path(context["history"]["corrective_pair"])
    certificate_path = pair / "metadata_certificate.json"
    certificate = dict(verification=dict(version=pipeline.METADATA_VALIDATION_VERSION,
        validation_status="source_valid_pending_adjudication", software_false_alarm=True,
        new_native_calls=0, native_receipts_remain_accepted_false=True))
    pipeline.native.write_new(certificate_path, certificate)
    for side in ("a", "b"):
        path = pair / side / "execution/receipt.json"
        receipt = pipeline.native.read_json(path)
        receipt.update(accepted=False, errors=[pipeline.AUTHORIZATION_COPY_ERROR])
        path.write_bytes(pipeline.protocol.canonical_bytes(receipt))
        digest = pipeline.protocol.sha256(path.read_bytes())
        for row in sources[side].values():
            row.update(receipt_sha256=digest, validation_protocol=pipeline.METADATA_VALIDATION_VERSION,
                source_validation_status="source_valid_pending_adjudication", native_receipt_accepted=False,
                software_false_alarm=True, validation_certificate=str(certificate_path),
                validation_certificate_sha256=pipeline.protocol.sha256(certificate_path.read_bytes()))
    context["history"]["protected_sha256"] = {}
    module.completed_records = lambda output, side, packets: originals[side]
    module.record_sources = lambda output, side: sources[side]
    module.verify_attempt = lambda attempt, receipt, packets: originals[receipt["stream"]]
    return context


def test_zero_call_metadata_dispatch_keeps_native_false_receipts_and_raw_unchanged(pipeline, metadata_corrective_context):
    context = metadata_corrective_context
    pair = Path(context["history"]["corrective_pair"])
    before = {str(path): path.read_bytes() for path in pair.glob("*/execution/*")}
    chunk = pipeline.snapshot_chunk(context, context["root"] / "private_export", 1, b"", [])
    for side in ("a", "b"):
        receipt = pipeline.native.read_json(chunk / "corrective_raw" / side / "receipt.json")
        assert receipt["accepted"] is False and receipt["errors"] == [pipeline.AUTHORIZATION_COPY_ERROR]
    assert (chunk / "metadata_revalidation/metadata_certificate.json").read_bytes() == (pair / "metadata_certificate.json").read_bytes()
    assert all(Path(path).read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize("mutation", ["no_certificate", "extra_error", "hidden_calls"])
def test_metadata_route_rejects_unknown_additional_errors_or_hidden_calls(pipeline, metadata_corrective_context, mutation):
    context = metadata_corrective_context
    module = context["source_module"]
    if mutation == "no_certificate":
        actual = module.record_sources
        def missing(output, side):
            values = actual(output, side)
            for value in values.values():
                value.pop("validation_protocol")
            return values
        module.record_sources = missing
    elif mutation == "extra_error":
        path = Path(context["history"]["corrective_pair"]) / "a/execution/receipt.json"
        receipt = pipeline.native.read_json(path)
        receipt["errors"].append("Synthetic additional source error")
        path.write_bytes(pipeline.protocol.canonical_bytes(receipt))
    else:
        path = Path(context["history"]["corrective_pair"]) / "metadata_certificate.json"
        value = pipeline.native.read_json(path)
        value["verification"]["new_native_calls"] = 1
        path.write_bytes(pipeline.protocol.canonical_bytes(value))
        actual = module.record_sources
        def updated_hash(output, side):
            values = actual(output, side)
            for row in values.values():
                row["validation_certificate_sha256"] = pipeline.protocol.sha256(path.read_bytes())
            return values
        module.record_sources = updated_hash
    with pytest.raises(ValueError, match="certificate|acceptance|hidden calls|changed"):
        pipeline.snapshot_chunk(context, context["root"] / "private_export", 1, b"", [])


def test_historical_cache_uses_source_verifier_once_and_retains_raw(pipeline, corrective_context):
    context = corrective_context
    module = context["source_module"]
    counters = dict(records=0, sources=0, revalidation=0)
    actual_records, actual_sources = module.completed_records, module.record_sources
    def records(*args):
        counters["records"] += 1
        return actual_records(*args)
    def sources(*args):
        counters["sources"] += 1
        return actual_sources(*args)
    def repeated(*args):
        counters["revalidation"] += 1
        pytest.fail("Historical source contexts must not be decoded per chunk")
    module.completed_records, module.record_sources, module.verify_attempt = records, sources, repeated
    output = context["root"] / "cached-export"
    chunk = pipeline.snapshot_chunk(context, output, 1, b"", [])
    assert pipeline.snapshot_chunk(context, output, 1, b"", []) == chunk
    assert counters == dict(records=2, sources=2, revalidation=0)
    actual = pipeline.legacy.full_adjudicator.read_stream(chunk / "independent_a.jsonl")
    assert actual == context["history_cache"]["records"]["a"]
    assert all(row["raw_output"].encode() == (Path(context["history_cache"]["sources"]["a"][key]["receipt"]).parent / "raw_final.json").read_bytes()
               for key, row in actual.items())


@pytest.mark.parametrize("mutation", ["cached_label", "cached_raw", "history_file", "original_raw", "cache_artifact"])
def test_frozen_history_cache_tampering_fails_closed(pipeline, corrective_context, mutation):
    context = corrective_context
    pipeline.snapshot_chunk(context, context["root"] / "cached-export", 1, b"", [])
    cache = context["history_cache"]
    key = context["ids"][0]
    if mutation == "cached_label":
        cache["records"]["a"][key]["label"]["answerability"] = "insufficient"
    elif mutation == "cached_raw":
        cache["records"]["a"][key]["raw_output"] += " "
    elif mutation == "history_file":
        path = context["run"] / "history.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "original_raw":
        path = Path(cache["sources"]["a"][key]["receipt"]).parent / "raw_final.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = context["root"] / "sealed-cache.json"
        pipeline.native.write_new(path, pipeline.protocol.canonical_bytes(cache) + b" ")
        context["history_cache_artifact"] = path
    with pytest.raises(ValueError, match="changed|Protected"):
        pipeline.verify_history_cache(context)


def test_spanning_whole_native_group_verified_once_without_history_redecode(pipeline, wire_source, monkeypatch):
    seal_all(wire_source)
    actual = pipeline.wire.verify_attempt
    calls = []
    def checked(attempt, receipt, packets):
        calls.append((receipt["stream"], receipt["batch"]))
        return actual(attempt, receipt, packets)
    monkeypatch.setattr(pipeline.wire, "verify_attempt", checked)
    monkeypatch.setattr(pipeline.wire, "completed_records", lambda *args: pytest.fail("No repeated full history loader"))
    base = pipeline.configure(pipeline.legacy.isolated_base(), empty_reuse())
    output, result = base.run(wire_source["root"], wire_source["pool"], wire_source["run"], adjudicate=False)
    assert result["exported_chunks"] == 2
    assert len(calls) == 6 and len(set(calls)) == 6
    assert (output / "source_inputs/historical_source_cache.json").is_file()
    cache_receipt = pipeline.native.read_json(output / "historical_source_cache_receipt.json")
    assert cache_receipt["native_calls"] == 0 and cache_receipt["original_source_validated_once_before_cache"]


def test_immutable_cache_artifact_is_checked_on_resume(pipeline, wire_source):
    seal_all(wire_source)
    base = pipeline.configure(pipeline.legacy.isolated_base(), empty_reuse())
    output, _ = base.run(wire_source["root"], wire_source["pool"], wire_source["run"], adjudicate=False)
    path = output / "source_inputs/historical_source_cache.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="cache artifact changed"):
        base.run(wire_source["root"], wire_source["pool"], wire_source["run"], resume=output,
                 caller=lambda *args, **kwargs: pytest.fail("No call after cache tampering"))


@pytest.fixture
def clarified_context(pipeline, wire_source):
    """Real clarified codec/mock-native bytes; synthetic history authority only."""
    import run_codex_clarified_cohort as clarified
    context = pipeline.prepare(wire_source["root"], wire_source["pool"], wire_source["run"])
    run = wire_source["pool"] / "cohort_annotations_clarified/synthetic-future"
    history = dict(record_ids=[], protected_sha256={}, sources={"a": {}, "b": {}})
    pipeline.native.write_new(run / "history.json", history)
    groups = clarified.build_groups([packet["payload"] for packet in context["packets"].values()], context["questions"], [])
    synthetic_command = pipeline.native.cli_command(wire_source["root"] / "native.exe", run / "synthetic-context",
        run / "synthetic-context/schema.json", run / "synthetic-context/final.json", None, [])
    pipeline.native.write_new(run / "manifest.json", dict(specification=dict(version=clarified.VERSION, groups=groups,
        requested_model=None, native_command_identity=clarified.clarification.normalized_command(synthetic_command))))
    context.update(run=run, history=history, source_module=clarified, source_spec={"groups": groups}, clarified=True)
    pipeline.seal_history_cache(context, {"a": {}, "b": {}}, history["sources"])
    for group in groups:
        for side in ("a", "b"):
            result = clarified.execute_attempt(run, side, group, context["packets"], wire_source["root"] / "native.exe",
                {}, [], {}, None, 60, wire_source["executor"])
            assert result["receipt"]["accepted"], result["receipt"]
            path = result["attempt"] / "receipt.json"
            pipeline.native.append_progress(run / "progress.jsonl", dict(type="completed", accepted=True,
                stream=side, batch=group["batch"], record_ids=group["record_ids"], receipt=path.relative_to(run).as_posix(),
                receipt_sha256=pipeline.protocol.sha256(path.read_bytes())))
    return context


def test_clarified_future_full_inputs_roles_and_first_attempt_population(pipeline, clarified_context):
    context = clarified_context
    prefix, entries = pipeline.legacy.isolated_base().progress_snapshot(context["run"])
    chunk = pipeline.snapshot_chunk(context, context["root"] / "future-export", 1, prefix, entries)
    left, right = (pipeline.legacy.full_adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b"))
    for key in list(left):
        assert left[key]["annotation_role"] == right[key]["annotation_role"] == "format_clarified_first_attempt"
        assert left[key]["original_first_attempt"] is True and left[key]["context_id"] != right[key]["context_id"]
        assert left[key]["execution_input_sha256"] == right[key]["execution_input_sha256"]
        assert left[key]["input_sha256"] == context["packets"][key]["input_sha256"]
    separate = pipeline.native.read_json(chunk / "future_first_attempt_raw_machine_agreement.json")
    assert separate["paired_records"] == 20
    assert separate["original_baseline_and_corrective_pairs_excluded"]
    assert separate["agreement_population"] == "predeclared_format_clarified_original_first_attempt_pairs_only"
    assert all(path.read_bytes() == context["source_module"].clarification.CORRECTION_INSTRUCTION.encode()
               for path in (chunk / "durable_snapshot").rglob("format_instruction.txt"))


def test_clarified_future_instruction_tampering_blocks_export(pipeline, clarified_context):
    context = clarified_context
    prefix, entries = pipeline.legacy.isolated_base().progress_snapshot(context["run"])
    path = context["run"] / Path(entries[0][1]["receipt"]).parent / "format_instruction.txt"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        pipeline.snapshot_chunk(context, context["root"] / "future-export", 1, prefix, entries)


def test_final_source_raw_mutation_during_review_is_rejected(pipeline, wire_source):
    seal_all(wire_source)
    base = pipeline.configure(pipeline.legacy.isolated_base(), empty_reuse())
    calls = []
    backend = compact_backend(pipeline.legacy, calls)
    def mutating(*args, **kwargs):
        value = backend(*args, **kwargs)
        if wire_source["rows"][-1]["record_id"] in calls[-1]:
            path = next((wire_source["run"] / "a" / wire_source["groups"][-1]["batch"]).glob("attempt-*/raw_final.json"))
            path.write_bytes(path.read_bytes() + b" ")
        return value
    with pytest.raises(ValueError, match="Protected input changed"):
        base.run(wire_source["root"], wire_source["pool"], wire_source["run"], caller=mutating)
    assert wire_source["rows"][-1]["record_id"] in calls[-1]


def test_existing_export_restores_future_raw_guards_before_review(pipeline, wire_source):
    seal_all(wire_source)
    base = pipeline.configure(pipeline.legacy.isolated_base(), empty_reuse())
    output, _ = base.run(wire_source["root"], wire_source["pool"], wire_source["run"], adjudicate=False)
    calls = []
    backend = compact_backend(pipeline.legacy, calls)
    def mutating(*args, **kwargs):
        value = backend(*args, **kwargs)
        if wire_source["rows"][-1]["record_id"] in calls[-1]:
            path = next((wire_source["run"] / "b" / wire_source["groups"][-1]["batch"]).glob("attempt-*/stdin.txt"))
            path.write_bytes(path.read_bytes() + b" ")
        return value
    with pytest.raises(ValueError, match="Protected input changed"):
        base.run(wire_source["root"], wire_source["pool"], wire_source["run"], resume=output, caller=mutating)
