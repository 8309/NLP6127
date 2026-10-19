import importlib.util
import json
from pathlib import Path

import pytest

from oil_evidence.store import digest, read_jsonl, write_json, write_jsonl


@pytest.fixture
def publisher(tmp_path):
    spec = importlib.util.spec_from_file_location("codex_labels", Path(__file__).parents[1] / "scripts/codex_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA = tmp_path
    module.PACK = tmp_path / "annotations/codex_input.jsonl"
    module.DECISIONS = tmp_path / "annotations/codex_decisions"
    passage = dict(passage_id="p1", text="Production is forecast to rise.", available_by="2026-01-01")
    write_jsonl(tmp_path / "passages.jsonl", [passage])
    original = dict(record_id="pilot-001", question="What is projected?", event_group="event1", as_of="2026-01-02")
    row = dict(dataset="pilot", number=1, record=original, evidence=[passage],
               original_sha256=digest(json.dumps(original, sort_keys=True)),
               corpus_sha256=digest((tmp_path / "passages.jsonl").read_bytes()))
    write_jsonl(module.PACK, [row])
    decision = dict(dataset="pilot", number=1, grades=[2], result="passed", answerability="sufficient",
                    event_status="uncertain", points=["Production is forecast to rise."],
                    quotes=[dict(e=1, text="Production is forecast to rise.")], note="Forecast, not an observed outcome")
    write_json(module.DECISIONS / "one.json", [decision])
    return module, decision


def test_publish_retains_machine_provenance_and_exact_hash(publisher):
    module, _ = publisher
    module.publish()
    labels = read_jsonl(module.DATA / "annotations/codex.jsonl")
    report = json.loads((module.DATA / "reports/codex_annotation.json").read_text())
    assert labels[0]["provenance"] == "codex_model_annotated"
    assert labels[0]["status"] != "accepted"
    assert labels[0]["original_annotation"]["as_of"] == "2026-01-02"
    assert report["human_labels_created"] == report["external_api_calls"] == 0
    assert report["labels_sha256"] == digest((module.DATA / "annotations/codex.jsonl").read_bytes())
    with pytest.raises(ValueError, match="Already published"):
        module.publish()


def test_missing_decisions_cannot_publish(publisher):
    module, _ = publisher
    write_json(module.DECISIONS / "one.json", [])
    with pytest.raises(ValueError, match="Incomplete annotation"):
        module.publish()


def test_fabricated_quote_cannot_publish(publisher):
    module, decision = publisher
    decision["quotes"][0]["text"] = "Production already recovered."
    write_json(module.DECISIONS / "one.json", [decision])
    with pytest.raises(ValueError):
        module.publish()
    assert not (module.DATA / "annotations/codex.jsonl").exists()


def test_future_positive_evidence_cannot_publish(publisher):
    module, decision = publisher
    decision["as_of"] = "2025-12-31"
    write_json(module.DECISIONS / "one.json", [decision])
    with pytest.raises(ValueError):
        module.publish()


def test_changed_corpus_cannot_publish(publisher):
    module, _ = publisher
    write_jsonl(module.DATA / "passages.jsonl", [])
    with pytest.raises(ValueError, match="Corpus changed"):
        module.publish()
