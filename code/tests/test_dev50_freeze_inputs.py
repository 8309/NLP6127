"""Local staging checks only; no CLI, credentials or model calls."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def staging(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("development_stager", scripts / "prepare_dev50_freeze_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(staging, tmp_path, monkeypatch, count=50):
    pool = tmp_path / "pool"
    pool.mkdir()
    a = pool / "codex_annotations" / "a"
    a.mkdir(parents=True)
    b = pool / "codex_annotations" / "b"
    b.mkdir()
    corpus = tmp_path / "passages.jsonl"
    questions = tmp_path / "questions.json"
    staging.native.write_new(corpus, b'{"original":"physical-bytes"}\r\n')
    staging.native.write_new(questions, [dict(question_id=f"q-{n}", question=f"Question {n}",
        as_of="2020-01-01", event_group=f"event-{n}", split="development", author_hint="remove")
        for n in range(count)])
    rows = [dict(question_id=f"q-{n}", evidence=[]) for n in range(count)]
    bindings = dict(corpus_sha256=staging.native.protocol.sha256(corpus.read_bytes()),
                    questions_sha256=staging.native.protocol.sha256(questions.read_bytes()))
    staging.native.write_new(pool / "inputs.json", dict(bindings=bindings))
    staging.native.write_new(a / "protocol.txt", b"exact combined protocol\n")
    packets = {f"q-{n}": dict(payload=dict(record_id=f"q-{n}"), prompt_sha256="prompt", input_sha256="input")
               for n in range(count)}
    staging.native.write_new(a / "packets.json", packets)
    monkeypatch.setattr(staging.native, "load_pool", lambda *args: (pool, rows, {}))
    monkeypatch.setattr(staging.native.protocol, "prepare_inputs", lambda *args, **kwargs: packets)
    monkeypatch.setattr(staging.native.protocol, "raw_agreement", lambda *args:
        dict(paired_records=count, pooled_passages_without_paired_judgment=0, human_agreement=None))
    out = pool / "staged"
    def mock_export(*args, **kwargs):
        out.mkdir()
        for name in ("independent_a.jsonl", "independent_b.jsonl"):
            staging.native.write_new(out / name, b"".join(staging.native.protocol.canonical_bytes(dict(
                record_id=f"q-{n}", raw_output="unchanged raw", label=dict(
                    answerability="insufficient", event_status="unclear", qrels={}))) + b"\n"
                for n in range(count)))
        staging.native.write_new(out / "packets.json", packets)
        staging.native.write_new(out / "source_receipts.json", {})
        return out
    monkeypatch.setattr(staging, "export", mock_export)
    return pool, a, b, corpus, questions


def test_full_physical_corpus_and_exact_protocol_preserved(staging, tmp_path, monkeypatch):
    paths = fixture(staging, tmp_path, monkeypatch)
    out, agreement = staging.stage(tmp_path, *paths)
    assert (out / "corpus.jsonl").read_bytes() == paths[3].read_bytes()
    assert (out / "annotation_prompt.txt").read_bytes() == (paths[1] / "protocol.txt").read_bytes()
    receipt = staging.native.read_json(out / "staging_receipt.json")
    assert receipt["new_model_calls"] == 0 and not receipt["dataset_frozen"]
    assert not (out / "runtime_artifacts.json").exists() and not (out / "adjudications.jsonl").exists()
    assert agreement["human_agreement"] is None
    differences = staging.native.read_json(out / "raw_machine_disagreements.json")
    assert differences["differing_records"] == 0 and differences["semantic_answer_comparison"] is None


def test_author_fields_not_staged_as_clean_questions(staging, tmp_path, monkeypatch):
    paths = fixture(staging, tmp_path, monkeypatch)
    out, _ = staging.stage(tmp_path, *paths)
    rows = staging.native.protocol.parse_jsonl((out / "questions.jsonl").read_bytes())
    assert len(rows) == 50 and all("author_hint" not in row for row in rows)


def test_partial_cohort_not_claimed_full(staging, tmp_path, monkeypatch):
    paths = fixture(staging, tmp_path, monkeypatch, count=49)
    with pytest.raises(ValueError, match="all 50"):
        staging.stage(tmp_path, *paths)


def test_changed_original_question_bytes_rejected(staging, tmp_path, monkeypatch):
    paths = fixture(staging, tmp_path, monkeypatch)
    paths[4].write_bytes(b"[]")
    with pytest.raises(ValueError, match="retrieval bindings"):
        staging.stage(tmp_path, *paths)


def test_packet_projection_mismatch_rejected_before_export(staging, tmp_path, monkeypatch):
    paths = fixture(staging, tmp_path, monkeypatch)
    monkeypatch.setattr(staging.native.protocol, "prepare_inputs", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="actual annotation packets"):
        staging.stage(tmp_path, *paths)
