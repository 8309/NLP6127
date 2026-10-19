"""Blinded chunk export guards; no model calls or live progress writes."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def exporter(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("isolated_annotation_export", scripts / "export_codex_annotation_chunk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(exporter, tmp_path, monkeypatch):
    a, b = (tmp_path / "codex_annotations" / side for side in ("a", "b"))
    packets = {f"q-{n:03d}": dict(payload=dict(record_id=f"q-{n:03d}")) for n in range(50)}
    data = {}
    for side, directory in (("a", a), ("b", b)):
        exporter.native.write_new(directory / "packets.json", packets)
        exporter.native.write_new(directory / "protocol.txt", b"same exact original protocol\n")
        output = dict(record_id="q-000", raw_output="actual raw label bytes",
                      raw_output_sha256="raw", execution_input_sha256="same-input",
                      context_id=side, identity=dict(execution_id=side))
        attempt = directory / side / "q-000" / "attempt-original"
        exporter.native.write_new(attempt / "receipt.json", dict(accepted=True))
        exporter.native.write_new(attempt / "envelope.json", output)
        data[side] = {"q-000": output}
    monkeypatch.setattr(exporter.native, "completed_outputs", lambda directory, side, packets: data[side])
    return a, b, data


def test_actual_raw_string_preserved_with_source_hashes(exporter, tmp_path, monkeypatch):
    a, b, data = inputs(exporter, tmp_path, monkeypatch)
    before = (a / "a/q-000/attempt-original/envelope.json").read_bytes()
    out = exporter.export(tmp_path, a, b, count=1)
    left = exporter.native.protocol.parse_jsonl((out / "independent_a.jsonl").read_bytes())
    assert left[0]["raw_output"] == "actual raw label bytes" and left[0] == data["a"]["q-000"]
    assert (a / "a/q-000/attempt-original/envelope.json").read_bytes() == before
    receipt = exporter.native.read_json(out / "source_receipts.json")
    assert receipt["new_model_calls"] == 0 and not receipt["dataset_frozen"]
    assert receipt["expected_full_development_records"] == 50


def test_cross_stream_actual_stdin_mismatch_rejected(exporter, tmp_path, monkeypatch):
    a, b, data = inputs(exporter, tmp_path, monkeypatch)
    data["b"]["q-000"]["execution_input_sha256"] = "different"
    with pytest.raises(ValueError, match="actual model inputs"):
        exporter.export(tmp_path, a, b, count=1)


def test_same_context_rejected(exporter, tmp_path, monkeypatch):
    a, b, data = inputs(exporter, tmp_path, monkeypatch)
    data["b"]["q-000"]["context_id"] = "a"
    with pytest.raises(ValueError, match="contexts"):
        exporter.export(tmp_path, a, b, count=1)


def test_changed_protocol_rejected(exporter, tmp_path, monkeypatch):
    a, b, _ = inputs(exporter, tmp_path, monkeypatch)
    (b / "protocol.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="protocols differ"):
        exporter.export(tmp_path, a, b, count=1)


def test_negative_or_outside_scope_rejected(exporter, tmp_path, monkeypatch):
    a, b, _ = inputs(exporter, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="positive explicit"):
        exporter.export(tmp_path, a, b, start=0)
    with pytest.raises(ValueError, match="inside their annotation pool"):
        exporter.export(tmp_path, a, tmp_path / "unrelated", count=1)


def test_missing_independent_record_is_not_filled_with_negative_label(exporter, tmp_path, monkeypatch):
    a, b, data = inputs(exporter, tmp_path, monkeypatch)
    data["b"] = {}
    with pytest.raises(KeyError):
        exporter.export(tmp_path, a, b, count=1)
