"""Durable cohort/adjudication tests with synthetic sources and no model calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("cohort_pipeline_tests", scripts / "pipeline_cohort_adjudication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(runner, tmp_path, tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("cp") / "c"
    pool = root / "p"
    run = pool / "cohort_annotations/synthetic-run"
    rows, queries, corpus = [], [], []
    # Deliberately not lexicographic: the manifested source order is authoritative.
    for index in range(25):
        identifier = f"synthetic-{'z' if index % 2 == 0 else 'a'}-{index:02d}"
        evidence = [dict(passage_id=f"p-{index}", document_id=f"d-{index}",
                         text=f"Synthetic region {index} observed stable crude exports.", available_by="2024-01-01"),
                    dict(passage_id=f"future-{index}", document_id=f"future-d-{index}",
                         text="Future synthetic report is ineligible.", available_by="2025-01-01")]
        rows.append(dict(record_id=identifier, question=f"What did synthetic region {index} observe?",
                         as_of="2024-03-01", evidence=evidence))
        queries.append(dict(record_id=identifier, question=rows[-1]["question"], as_of=rows[-1]["as_of"],
                            event_group=f"synthetic-event-{index}", split="development" if index < 20 else "test"))
        corpus.extend(evidence)
    packets = {row["record_id"]: runner.protocol.clean_packet(row, prompt=runner.batch.BATCH_PROMPT) for row in rows}
    runner.native.write_new(root / "data/unified_runs/semantic-page-v2-buffer/stage/passages.jsonl",
        b"".join(runner.protocol.canonical_bytes(row) + b"\n" for row in corpus))
    runner.native.write_new(run / "packets.json", packets)
    runner.native.write_new(run / "protocol.txt", runner.batch.BATCH_PROMPT.encode())
    runner.native.write_new(run / "protected_inputs.json", {})
    bindings = {"synthetic_source": "fixture-no-model"}
    manifest = dict(specification=dict(version=runner.cohort.VERSION, partition="all_current", batch_size=5,
        old_protocol_outputs_reused=False, bindings=bindings, eligible_record_ids=[row["record_id"] for row in rows],
        batch_prompt_sha256=runner.protocol.sha256(runner.batch.BATCH_PROMPT.encode()),
        eligible_packets_sha256=runner.protocol.sha256(runner.protocol.canonical_bytes(packets))),
        immutable_sha256={name: runner.protocol.sha256((run / name).read_bytes())
                          for name in ("packets.json", "protocol.txt", "protected_inputs.json")})
    runner.native.write_new(run / "manifest.json", manifest)
    monkeypatch.setattr(runner.cohort, "load_pool", lambda selected_root, selected_pool: (pool, rows, queries, {}, bindings))

    def executor(command, stdin, context, attempt, environment, timeout):
        payload = json.loads((context / "packets.json").read_bytes())
        values = {}
        for key, row in payload.items():
            passage = row["evidence"][0]
            values[key] = dict(record_id=key, label=dict(answerability="sufficient", event_status="observed",
                qrels={item["passage_id"]: 2 if offset == 0 else 0 for offset, item in enumerate(row["evidence"])},
                answer_points=[dict(text=passage["text"], citations=[passage["passage_id"]])],
                quotes=[dict(passage_id=passage["passage_id"], text=passage["text"])], reasons=["Synthetic exact eligible source."]))
        runner.native.write_new(context / "final.json", (json.dumps(values, indent=2) + "\r\n").encode())
        events = [dict(type="thread.started", thread_id="synthetic-exec-" + uuid4().hex),
                  dict(type="turn.completed", usage=dict(input_tokens=100, output_tokens=25))]
        runner.native.write_new(attempt / "events.jsonl", b"".join(runner.protocol.canonical_bytes(event) + b"\n" for event in events))
        runner.native.write_new(attempt / "stderr.txt", b"")
        return 0

    def add(batch_number, side, durable=True):
        selected = rows[(batch_number - 1) * 5:batch_number * 5]
        result = runner.cohort.execute_attempt(run, side, batch_number, selected, tmp_path / "synthetic-native.exe",
                                              {}, [], {}, None, 60, executor)
        assert result["receipt"]["accepted"]
        if durable:
            runner.native.append_progress(run / "progress.jsonl", dict(type="completed", accepted=True,
                stream=side, batch=f"batch-{batch_number:04d}", record_ids=[row["record_id"] for row in selected],
                receipt=(result["attempt"] / "receipt.json").relative_to(run).as_posix(),
                receipt_sha256=runner.protocol.sha256((result["attempt"] / "receipt.json").read_bytes())))
            runner.native.append_progress(run / "execution_usage.jsonl", result["usage"])
        return result
    return dict(root=root, pool=pool, run=run, rows=rows, packets=packets, add=add, questions=queries, executor=executor)


def paired(source, batches=4):
    for number in range(1, batches + 1):
        source["add"](number, "a")
        source["add"](number, "b")


def mock_adjudicator(runner, calls, unresolved=False):
    def call(system, payload, schema, output_parent, role):
        assert role == "adjudicator"
        assert "event_group" not in json.dumps(payload) and "split" not in json.dumps(payload)
        assert all(set(row) == {"record_id", "question", "as_of", "evidence", "independent_a", "independent_b"}
                   for row in payload.values())
        calls.append(list(payload))
        parsed = {key: dict(status="unresolved" if unresolved else "accepted", reason="Synthetic source check.",
            label=None if unresolved else deepcopy(row["independent_a"]),
            unresolved_reasons=["Synthetic uncertainty retained."] if unresolved else []) for key, row in payload.items()}
        directory = Path(output_parent) / ("s-" + uuid4().hex[:8])
        raw = (json.dumps(parsed, indent=2) + "\r\n").encode()
        runner.native.write_new(directory / "raw_final.json", raw)
        stdin, schema_bytes = runner.actual_adjudication_bytes(payload, schema)
        runner.native.write_new(directory / "stdin.txt", stdin)
        runner.native.write_new(directory / "schema.json", schema_bytes)
        execution_id = "synthetic-adjud-" + uuid4().hex
        runner.native.write_new(directory / "events.jsonl", b"".join(runner.protocol.canonical_bytes(event) + b"\n"
            for event in [dict(type="thread.started", thread_id=execution_id), dict(type="turn.completed", usage={})]))
        receipt = dict(artifact_directory=str(directory), execution_id=execution_id,
                       accepted=True, observed_model=None, artifacts_sha256={
                           path.name: runner.protocol.sha256(path.read_bytes()) for path in directory.iterdir()})
        runner.native.write_new(directory / "receipt.json", receipt)
        return parsed, receipt
    return call


def test_manifested_order_fixed_twenty_and_last_five(runner, source):
    context = runner.prepare(source["root"], source["pool"], source["run"])
    ids = [row["record_id"] for row in source["rows"]]
    assert context["ids"] == ids and ids != sorted(ids)
    assert context["chunks"] == [ids[:20], ids[20:]]
    assert "event_group" not in json.dumps(context["packets"])


@pytest.mark.parametrize("size", [0, 3, True, -5])
def test_invalid_fixed_chunk_sizes(runner, source, size):
    with pytest.raises(ValueError, match="multiple of five"):
        runner.prepare(source["root"], source["pool"], source["run"], size)


def test_partial_progress_line_ignored_and_failure_stops(runner, tmp_path):
    path = tmp_path / "progress.jsonl"
    runner.native.write_new(path, b'{"type":"started"}\n{"type":"completed"')
    prefix, rows = runner.progress_snapshot(tmp_path)
    assert prefix == b'{"type":"started"}\n' and len(rows) == 1
    path.write_bytes(b'{"type":"failed"}\n')
    with pytest.raises(ValueError, match="stops without retry"):
        runner.progress_snapshot(tmp_path)


def test_receipt_without_durable_progress_and_one_sided_chunk_never_export(runner, source):
    source["add"](1, "a", durable=False)
    for number in range(1, 5):
        source["add"](number, "b")
    output, summary = runner.run(source["root"], source["pool"], source["run"], max_idle_seconds=0,
                                 caller=lambda *args, **kwargs: pytest.fail("No incomplete-chunk calls"))
    assert summary["exported_chunks"] == 0 and summary["new_model_calls"] == 0
    assert not (output / "chunks").exists()


def test_export_preserves_whole_raw_hashes_future_zero_and_source_order(runner, source):
    paired(source)
    output, summary = runner.run(source["root"], source["pool"], source["run"], adjudicate=False)
    assert summary["exported_chunks"] == 1 and summary["complete_source_inputs"] is None
    chunk = output / "chunks/chunk-0001"
    packets = runner.native.read_json(chunk / "packets.json")
    assert list(packets) == [row["record_id"] for row in source["rows"][:20]]
    originals = runner.adjudicator.read_stream(chunk / "independent_a.jsonl")
    for key, envelope in originals.items():
        assert envelope["label"]["qrels"][packets[key]["payload"]["evidence"][1]["passage_id"]] == 0
        assert envelope["raw_label_path"] == [key, "label"]
        raw_container = json.loads(envelope["raw_output"])
        assert len(raw_container) == 5 and envelope["label"] == raw_container[key]["label"]
    receipt = runner.native.read_json(chunk / "export_receipt.json")
    assert receipt["raw_agreement_computed_before_adjudication"]
    assert receipt["records"] == 20
    assert runner.native.read_json(chunk / "raw_machine_agreement.json")["paired_records"] == 20
    _, resumed = runner.run(source["root"], source["pool"], source["run"], adjudicate=False, resume=output)
    assert resumed["new_model_calls"] == 0


@pytest.mark.parametrize("mutation", ["stdin", "raw", "schema", "receipt"])
def test_durable_artifact_tamper_fails_before_adjudication(runner, source, mutation):
    paired(source)
    receipt_path = next((source["run"] / "a/batch-0001").glob("attempt-*/receipt.json"))
    path = receipt_path if mutation == "receipt" else receipt_path.parent / {
        "stdin": "stdin.txt", "raw": "raw_final.json", "schema": "schema.json"}[mutation]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        runner.run(source["root"], source["pool"], source["run"], max_idle_seconds=0,
                   caller=lambda *args, **kwargs: pytest.fail("No calls for changed originals"))


def test_unresolved_all_records_preserved_and_zero_call_resume(runner, source):
    paired(source, 5)
    calls = []
    output, result = runner.run(source["root"], source["pool"], source["run"],
                                caller=mock_adjudicator(runner, calls, unresolved=True))
    assert result["complete"] and result["retained_records"] == 25
    assert result["unresolved"] == 25 and result["accepted"] == 0 and result["human_gold"] is False
    assert [key for group in calls for key in group] == [row["record_id"] for row in source["rows"]]
    full_source = Path(result["complete_source_inputs"])
    assert len(runner.protocol.parse_jsonl((full_source / "independent_a.jsonl").read_bytes())) == 25
    _, resumed = runner.run(source["root"], source["pool"], source["run"], resume=output,
                            caller=lambda *args, **kwargs: pytest.fail("Accepted and unresolved retainers never rerun"))
    assert resumed["new_model_calls"] == 0 and resumed["retained_model_calls"] == 3


def test_bounded_groups_max_ten_actual_prefix_schema_and_greedy_bytes(runner, source):
    paired(source)
    output, _ = runner.run(source["root"], source["pool"], source["run"], adjudicate=False)
    chunk = output / "chunks/chunk-0001"
    packets = runner.native.read_json(chunk / "packets.json")
    left, right = (runner.adjudicator.read_stream(chunk / f"independent_{side}.jsonl") for side in ("a", "b"))
    groups = runner.adjudication_groups(packets, left, right, list(packets))
    assert [len(group) for group in groups] == [10, 10]
    assert [key for group in groups for key in group] == list(packets)
    first = list(packets)[:2]
    payload, schema = runner.adjudicator.prepare_batch(packets, left, right, first)
    stdin, schema_bytes = runner.actual_adjudication_bytes(payload, schema)
    assert stdin.startswith((runner.runtime.ISOLATION_PROMPT + "\n" + runner.adjudicator.SYSTEM).encode())
    assert b"UNTRUSTED_PAYLOAD_JSON" in stdin
    tight = len(stdin) + len(schema_bytes)
    groups = runner.adjudication_groups(packets, left, right, list(packets), max_bytes=tight)
    for group in groups:
        payload, schema = runner.adjudicator.prepare_batch(packets, left, right, group)
        stdin, schema_bytes = runner.actual_adjudication_bytes(payload, schema)
        assert len(group) <= 10 and len(stdin) + len(schema_bytes) <= tight
    with pytest.raises(ValueError, match="One complete.*byte bound"):
        runner.adjudication_groups(packets, left, right, list(packets), max_bytes=1)
    splits = {key: "development" if offset < 5 or offset >= 10 else "test"
              for offset, key in enumerate(packets)}
    split_groups = runner.adjudication_groups(packets, left, right, list(packets), splits=splits)
    assert [len(group) for group in split_groups] == [5, 5, 10]
    assert all(len({splits[key] for key in group}) == 1 for group in split_groups)


@pytest.mark.parametrize("cases,cap", [(11, 262144), (10, 262145), (True, 262144)])
def test_adjudication_hard_policy_cannot_be_relaxed(runner, cases, cap):
    with pytest.raises(ValueError, match="max-ten"):
        runner.adjudication_groups({}, {}, {}, [], max_cases=cases, max_bytes=cap)


def test_bounded_resume_reuses_raw_contexts_and_retains_mechanical_errors(runner, source):
    paired(source)
    output, _ = runner.run(source["root"], source["pool"], source["run"], adjudicate=False)
    chunk = output / "chunks/chunk-0001"
    calls = []
    backend = mock_adjudicator(runner, calls)
    def invalid_final(system, payload, schema, output_parent, role):
        parsed, receipt = backend(system, payload, schema, output_parent, role)
        key = next(iter(parsed))
        # A source-invalid accepted label must become retained unresolved,
        # never silently corrected, removed or sent for an automatic retry.
        parsed[key]["label"]["quotes"][0]["text"] = "Synthetic unsupported quote."
        directory = Path(receipt["artifact_directory"])
        raw = (json.dumps(parsed, indent=2) + "\r\n").encode()
        (directory / "raw_final.json").write_bytes(raw)
        receipt["artifacts_sha256"]["raw_final.json"] = runner.protocol.sha256(raw)
        (directory / "receipt.json").write_bytes(runner.protocol.canonical_bytes(receipt) + b"\n")
        return parsed, receipt
    result = runner.adjudicate_chunk(chunk, output / "standalone", caller=invalid_final)
    assert result["records"] == 20 and result["unresolved"] == 2
    assert result["new_native_calls"] == 2 and result["retained_native_calls"] == 2
    decisions = runner.protocol.parse_jsonl(Path(result["adjudications_path"]).read_bytes())
    assert sum(bool(row["mechanical_validation_errors"]) for row in decisions) == 2
    resumed = runner.adjudicate_chunk(chunk, output / "standalone",
        caller=lambda *args, **kwargs: pytest.fail("No repeated accepted native contexts"))
    assert resumed["new_native_calls"] == 0 and resumed["retained_native_calls"] == 2
    assert runner.protocol.parse_jsonl(Path(resumed["adjudications_path"]).read_bytes()) == decisions


def test_repeated_native_context_rejected_without_retry(runner, source):
    paired(source)
    output, _ = runner.run(source["root"], source["pool"], source["run"], adjudicate=False)
    calls, seen = [], []
    backend = mock_adjudicator(runner, calls)
    def repeated(system, payload, schema, output_parent, role):
        parsed, receipt = backend(system, payload, schema, output_parent, role)
        if seen:
            receipt["execution_id"] = seen[0]
            directory = Path(receipt["artifact_directory"])
            events = b"".join(runner.protocol.canonical_bytes(event) + b"\n" for event in [
                dict(type="thread.started", thread_id=seen[0]), dict(type="turn.completed", usage={})])
            (directory / "events.jsonl").write_bytes(events)
            receipt["artifacts_sha256"]["events.jsonl"] = runner.protocol.sha256(events)
            (directory / "receipt.json").write_bytes(runner.protocol.canonical_bytes(receipt))
        seen.append(receipt["execution_id"])
        return parsed, receipt
    with pytest.raises(ValueError, match="fresh execution contexts"):
        runner.adjudicate_chunk(output / "chunks/chunk-0001", output / "repeat-fixture", caller=repeated)
    assert len(calls) == 2


def test_native_failure_retained_without_automatic_retry(runner, source):
    paired(source)
    calls = []
    def failing(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("Synthetic quota failure")
    with pytest.raises(RuntimeError, match="quota"):
        runner.run(source["root"], source["pool"], source["run"], caller=failing)
    output = next((source["root"] / "data/machine_runs/cohort_pipeline").iterdir())
    assert runner.native.read_json(output / "failure.json")["automatic_retry"] is False
    with pytest.raises(ValueError, match="explicit owner recovery"):
        runner.run(source["root"], source["pool"], source["run"], resume=output, caller=failing)
    assert len(calls) == 1


def test_resumed_pipeline_source_copy_and_export_tamper_rejected(runner, source):
    paired(source)
    output, _ = runner.run(source["root"], source["pool"], source["run"], adjudicate=False)
    path = output / "source_inputs/questions.jsonl"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source input changed"):
        runner.run(source["root"], source["pool"], source["run"], resume=output, adjudicate=False)


def test_retained_result_tamper_rejected_not_replayed(runner, source):
    paired(source)
    calls = []
    output, _ = runner.run(source["root"], source["pool"], source["run"], max_idle_seconds=0,
                           caller=mock_adjudicator(runner, calls))
    path = output / "retained/chunk-0001.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="progress binding"):
        runner.run(source["root"], source["pool"], source["run"], resume=output, max_idle_seconds=0,
                   caller=lambda *args, **kwargs: pytest.fail("No replay of retained originals"))


def test_native_group_can_span_fixed_chunks_without_rewriting_receipts(runner, source):
    paired(source, 2)
    runner.native.write_new(source["run"] / "STOP_REQUESTED", b"Synthetic controlled stop")
    runner.native.write_new(source["run"] / "summary-fixture.json", dict(status="stopped_without_automatic_retry",
        failures=[], accepted_counts={"a": 10, "b": 10}))
    bindings = runner.native.read_json(source["run"] / "manifest.json")["specification"]["bindings"]
    migration, old = runner.throughput.read_migration(source["run"], source["packets"], bindings)
    groups = runner.throughput.build_groups(source["rows"], source["questions"], migration["record_ids"], 10, 262144)
    assert [len(group["record_ids"]) for group in groups] == [10, 5]
    new = source["pool"] / "cohort_annotations_throughput/synthetic-v3"
    spec = dict(version=runner.throughput.VERSION, bindings=bindings, partition="all_current", case_limit=10,
        full_payload_byte_limit=262144, eligible_record_ids=[row["record_id"] for row in source["rows"]], groups=groups,
        eligible_packets_sha256=runner.protocol.sha256(runner.protocol.canonical_bytes(source["packets"])),
        batch_prompt_sha256=runner.protocol.sha256(runner.batch.BATCH_PROMPT.encode()), source_run=str(source["run"]),
        migration_sha256=runner.protocol.sha256(runner.protocol.canonical_bytes(migration)), migrated_record_ids=migration["record_ids"],
        accepted_records_rerun=False, old_protocol_outputs_rebound=False, annotation_semantic_protocol_unchanged=True)
    for name, value in (("packets.json", source["packets"]), ("protocol.txt", runner.batch.BATCH_PROMPT.encode()),
                        ("protected_inputs.json", migration["source_protected_sha256"]), ("migration.json", migration), ("groups.json", groups)):
        runner.native.write_new(new / name, value)
    runner.native.write_new(new / "manifest.json", dict(specification=spec, immutable_sha256={
        name: runner.protocol.sha256((new / name).read_bytes()) for name in
        ("packets.json", "protocol.txt", "protected_inputs.json", "migration.json", "groups.json")}))
    by_id = {row["record_id"]: row for row in source["rows"]}
    for group in groups:
        for side in ("a", "b"):
            value = runner.throughput.execute_attempt(new, side, group, [by_id[key] for key in group["record_ids"]],
                new / "synthetic-native.exe", {}, [], {}, None, 60, source["executor"])
            assert value["receipt"]["accepted"]
            runner.native.append_progress(new / "progress.jsonl", dict(type="completed", accepted=True, stream=side,
                batch=group["batch"], record_ids=group["record_ids"], receipt=(value["attempt"] / "receipt.json").relative_to(new).as_posix(),
                receipt_sha256=runner.protocol.sha256((value["attempt"] / "receipt.json").read_bytes())))
    output, result = runner.run(source["root"], source["pool"], new, chunk_size=5, adjudicate=False)
    assert result["exported_chunks"] == 5 and result["complete_source_inputs"]
    third = output / "chunks/chunk-0003"
    fourth = output / "chunks/chunk-0004"
    for chunk in (third, fourth):
        records = runner.adjudicator.read_stream(chunk / "independent_a.jsonl")
        assert len(records) == 5
        assert all(len(json.loads(row["raw_output"])) == 10 for row in records.values())
        sources = runner.native.read_json(chunk / "original_record_sources.json")["a"]
        receipt_path = Path(next(iter(sources.values()))["receipt"])
        copied = next((chunk / "durable_snapshot/a/batch-0001").glob("attempt-*/receipt.json"))
        assert copied.read_bytes() == receipt_path.read_bytes()
        assert len(runner.native.read_json(copied)["record_ids"]) == 10
    migrated = output / "chunks/chunk-0001"
    originals = runner.adjudicator.read_stream(migrated / "independent_a.jsonl")
    assert all(originals[key] == old["a"][key] for key in originals)
    assert runner.native.read_json(migrated / "original_record_sources.json")["a"][next(iter(originals))]["migrated_unchanged"]
    assert next((migrated / "migrated_raw/a").glob("batch-*/attempt-*/raw_final.json")).is_file()
