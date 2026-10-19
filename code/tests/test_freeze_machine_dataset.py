"""Synthetic local frozen-cohort checks, not annotation accuracy evidence."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freeze_machine_dataset as module
import report_machine_retrieval as reporter
from oil_evidence import machine_evaluation as protocol


def make_inputs(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    corpus = [dict(passage_id="p1", document_id="d1", text="Synthetic output was 1 million barrels per day.",
                   available_by="2024-01-01")]
    question = dict(record_id="q1", question="What was synthetic crude output?", as_of="2024-01-01",
                    event_group="event1", split="development")
    pool = dict(record_id="q1", evidence_ids=["p1"])
    packet = protocol.prepare_inputs([question], [pool], corpus)["q1"]
    label = dict(answerability="sufficient", event_status="observed", qrels={"p1": 2},
                 answer_points=[dict(text="Synthetic output was 1 million barrels per day.", citations=["p1"])],
                 quotes=[dict(passage_id="p1", text=corpus[0]["text"])], reasons=["Synthetic direct support."])
    outputs = []
    for side in ("a", "b"):
        identity = dict(service="synthetic", requested_model="synthetic", observed_model=None,
                        execution_id=side, observation_source="unavailable", identity_note="Synthetic fixture.")
        outputs.append(protocol.wrap_independent_output(packet, protocol.canonical_bytes(label).decode(), identity,
                        side, {}, execution_input_sha256=packet["input_sha256"]))
    decision = dict(record_id="q1", status="accepted", provenance="machine_adjudicated", human_gold=False, label=label,
                    adjudication=dict(reason="Synthetic review.", both_originals_seen=True,
                                      original_output_sha256=[protocol.sha256(protocol.canonical_bytes(x)) for x in outputs]))
    values = dict(corpus=[*corpus], questions=[question], evidence_pool=[pool], independent_a=[outputs[0]],
                  independent_b=[outputs[1]], adjudications=[decision])
    for key, name in reporter.NAMES.items():
        raw = protocol.ANNOTATION_PROMPT.encode() if key == "annotation_prompt" else b"".join(
            protocol.canonical_bytes(row) + b"\n" for row in values[key])
        (source / name).write_bytes(raw)
    runtime = {}
    for key in protocol.RUNTIME_KEYS:
        path = source / (key + ".txt")
        path.write_text("Synthetic artifact " + key)
        runtime[key] = str(path)
    (source / "runtime_artifacts.json").write_text(json.dumps(runtime))
    return source


def test_freeze_preserves_exact_inputs_and_snapshots_runtime(tmp_path):
    source = make_inputs(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output, receipt = module.freeze(source, tmp_path / "frozen")
    assert receipt["human_gold"] is False and receipt["formal_test_ready"] is False
    assert receipt["split_status_counts"] == {"development": {"accepted": 1}, "test": {}}
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    for name in reporter.NAMES.values():
        assert (output / name).read_bytes() == before[name]
    paths = json.loads((output / "runtime_artifacts.json").read_bytes())
    assert all(Path(path).is_relative_to(output) for path in paths.values())
    with pytest.raises(ValueError, match="exists"):
        module.freeze(source, tmp_path / "frozen")


def test_development_metrics_explicitly_not_formal_and_runtime_tamper_fails(tmp_path):
    source = make_inputs(tmp_path)
    output, _ = module.freeze(source, tmp_path / "frozen")
    rankings = [dict(record_id="q1", question="What was synthetic crude output?", mode=mode,
                     as_of="2024-01-01", elapsed_seconds=.1,
                     evidence=[dict(passage_id="p1", document_id="d1", rank=1, available_by="2024-01-01",
                                    text="Synthetic output was 1 million barrels per day.")])
                for mode in ("bm25", "dense", "hybrid")]
    path = tmp_path / "rankings.jsonl"
    path.write_bytes(b"".join(protocol.canonical_bytes(row) + b"\n" for row in rankings))
    result = reporter.evaluate(output, path, tmp_path / "development.json", split="development")
    assert result["questions"] == 1 and result["formal_heldout_test"] is False
    with pytest.raises(ValueError, match="coverage"):
        reporter.evaluate(output, path, tmp_path / "formal.json")
    runtime = json.loads((output / "runtime_artifacts.json").read_bytes())
    Path(runtime["implementation"]).write_text("Synthetic tamper")
    with pytest.raises(ValueError, match="changed"):
        reporter.evaluate(output, path, tmp_path / "tampered.json", split="development")


def test_formal_floor_cannot_be_relabelled_as_one_record(tmp_path):
    source = make_inputs(tmp_path)
    output, _ = module.freeze(source, tmp_path / "frozen")
    with pytest.raises(ValueError, match="1000"):
        reporter.evaluate(output, tmp_path / "unused.jsonl", tmp_path / "report.json", required_test_records=1)


def test_runtime_path_does_not_authorize_arbitrary_host_reads(tmp_path):
    source = make_inputs(tmp_path)
    paths = json.loads((source / "runtime_artifacts.json").read_bytes())
    outside = tmp_path / "outside.txt"
    outside.write_text("Synthetic outside file")
    paths["implementation"] = str(outside)
    (source / "runtime_artifacts.json").write_text(json.dumps(paths))
    with pytest.raises(ValueError, match="inside"):
        module.freeze(source, tmp_path / "frozen")


@pytest.mark.parametrize("change", ["question", "cutoff", "text", "unknown_record"])
def test_ranked_bytes_must_bind_frozen_questions_and_corpus(tmp_path, change):
    source = make_inputs(tmp_path)
    output, _ = module.freeze(source, tmp_path / "frozen")
    row = dict(record_id="q1", question="What was synthetic crude output?", mode="bm25", as_of="2024-01-01",
               elapsed_seconds=.1, evidence=[dict(passage_id="p1", document_id="d1", rank=1,
               available_by="2024-01-01", text="Synthetic output was 1 million barrels per day.")])
    if change == "question":
        row["question"] = "Different synthetic query"
    elif change == "cutoff":
        row["as_of"] = "2024-01-02"
    elif change == "text":
        row["evidence"][0]["text"] = "Fabricated synthetic text"
    else:
        row["record_id"] = "q-unknown"
    path = tmp_path / "ranking.jsonl"
    path.write_bytes(protocol.canonical_bytes(row) + b"\n")
    with pytest.raises(ValueError, match="bind|match"):
        reporter.evaluate(output, path, tmp_path / "metrics.json", split="development")
