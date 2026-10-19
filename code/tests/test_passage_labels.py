import importlib.util
import json
from pathlib import Path

import pytest

from oil_evidence.store import digest, read_jsonl, write_json, write_jsonl


@pytest.fixture
def module(tmp_path):
    spec = importlib.util.spec_from_file_location("passage_labels", Path(__file__).parents[1] / "scripts/passage_labels.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.DATA = tmp_path
    return m


@pytest.fixture
def passage():
    return dict(passage_id="p", document_id="d", title="Oil forecast", source="EIA", page=1,
                available_by="2025-01-01", text="EIA forecasts production of 10 million barrels per day in 2026.")


@pytest.fixture
def label():
    return dict(passage_id="p", topics=["production_supply"], oil_relevance="direct", entities=["EIA"],
                event_status="forecast", events=[],
                quantities=[dict(value_text="10", unit_text="million barrels per day", reference_period="2026",
                                 measure="crude_oil", quote="10 million barrels per day in 2026")],
                evidence_spans=["EIA forecasts production"], summary_zh="Production forecast", needs_review=False, review_reasons=[])


def test_exact_coverage_and_quotes(module, passage, label):
    assert module.validate_batch([label], [passage]) == []
    assert module.validate_batch([label, label], [passage])
    label["quantities"][0]["unit_text"] = "barrels per month"
    assert module.validate_batch([label], [passage])


def test_entity_cannot_be_invented(module, passage, label):
    label["entities"] = ["Russia"]
    assert module.validate_batch([label], [passage])


def test_unclear_date_not_assigned_iso(module, passage, label):
    label["events"] = [dict(description="forecast", date_text="2026", iso_date="2026-01-01",
                            precision="year", status="forecast", quote=passage["text"])]
    assert module.validate_batch([label], [passage])


def test_prepare_and_incomplete_publish_guard(module, passage):
    write_jsonl(module.DATA / "passages.jsonl", [passage])
    prepared = module.prepare(1)
    root = Path(prepared["path"])
    assert prepared["labelled"] == 0
    assert module.progress(root)["remaining"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        module.publish(root)
    with pytest.raises(ValueError, match="overwrite"):
        module.prepare(1)


def test_complete_publish_is_machine_only_and_preserves_raw(module, passage, label):
    write_jsonl(module.DATA / "passages.jsonl", [passage])
    raw = (module.DATA / "passages.jsonl").read_bytes()
    root = Path(module.prepare(1)["path"])
    write_json(root / "outputs/batch-001.json", dict(labels=[label]))
    write_json(root / "audit/batch-001.json", dict(output_sha256=digest((root / "outputs/batch-001.json").read_bytes())))
    report = module.publish(root)
    assert report["status"] == "complete" and report["total"] == 1
    assert (module.DATA / "passages.jsonl").read_bytes() == raw
    assert not read_jsonl(module.DATA / "annotations/passage_labels.jsonl")[0]["human_reviewed"]
    with pytest.raises(ValueError, match="immutable"):
        module.publish(root)


def test_authored_decisions_expand_without_inference(module, passage, tmp_path):
    write_jsonl(module.DATA / "passages.jsonl", [passage])
    root = Path(module.prepare(1)["path"])
    decision = dict(n=1, topics=["production_supply"], relevance="direct", state="forecast", entities=["EIA"],
                    spans=["EIA forecasts production"], summary="Future production forecast",
                    q=[["10", "million barrels per day", "2026", "crude_oil", "10 million barrels per day in 2026"]])
    write_json(tmp_path / "decisions.json", [decision])
    result = module.import_decisions(root, 1, tmp_path / "decisions.json")
    assert result["validated"] == 1
    output = json.loads((root / "outputs/batch-001.json").read_text(encoding="utf-8"))["labels"][0]
    assert output["event_status"] == decision["state"]
    assert output["events"] == [] and output["needs_review"] is False
    assert not (module.DATA / "annotations/passage_labels.jsonl").exists()


def test_missing_authored_decision_is_rejected(module, passage, tmp_path):
    write_jsonl(module.DATA / "passages.jsonl", [passage])
    root = Path(module.prepare(1)["path"])
    write_json(tmp_path / "decisions.json", [])
    with pytest.raises(ValueError, match="Every input"):
        module.import_decisions(root, 1, tmp_path / "decisions.json")


def test_schema_missing_or_invalid_fields_rejected(module, passage, label):
    label.pop("events")
    assert module.validate_batch([label], [passage])
    label["events"] = []
    label["topics"] = "production_supply"
    assert module.validate_batch([label], [passage])


def test_iso_date_cannot_infer_missing_year(module, passage, label):
    passage["text"] += " The meeting took place on June 18."
    label["events"] = [dict(description="meeting", date_text="June 18", iso_date="2025-06-18",
                            precision="day", status="observed", quote="The meeting took place on June 18.")]
    assert module.validate_batch([label], [passage])


def test_iso_date_must_match_literal_date(module, passage, label):
    passage["text"] += " The meeting took place on June 18, 2025."
    label["events"] = [dict(description="meeting", date_text="June 18, 2025", iso_date="2025-06-18",
                            precision="day", status="observed", quote="The meeting took place on June 18, 2025.")]
    assert not module.validate_batch([label], [passage])
    label["events"][0]["iso_date"] = "2025-06-19"
    assert module.validate_batch([label], [passage])


def test_review_revision_preserves_original_output(module, passage, label, tmp_path):
    write_jsonl(module.DATA / "passages.jsonl", [passage])
    root = Path(module.prepare(1)["path"])
    write_json(root / "outputs/batch-001.json", dict(labels=[label]))
    label["summary_zh"] = "Corrected forecast explanation"
    write_json(tmp_path / "revision.json", dict(labels=[label]))
    report = module.revise_batch(root, 1, tmp_path / "revision.json", "Remove unsupported detail")
    archives = list((root / "revisions/batch-001").glob("*/labels.json"))
    assert report["revised"] and len(archives) == 1
    old = json.loads(archives[0].read_text(encoding="utf-8"))["labels"][0]
    assert old["summary_zh"] == "Production forecast"
    assert module.progress(root)["validated"] == 1


def test_publish_requires_matching_annotation_audit(module, passage, label):
    write_jsonl(module.DATA / "passages.jsonl", [passage])
    root = Path(module.prepare(1)["path"])
    write_json(root / "outputs/batch-001.json", dict(labels=[label]))
    with pytest.raises(ValueError, match="Missing annotation audit"):
        module.publish(root)
    write_json(root / "audit/batch-001.json", dict(output_sha256="wrong"))
    with pytest.raises(ValueError, match="audit hash mismatch"):
        module.publish(root)
    assert not (root / "published").exists()
