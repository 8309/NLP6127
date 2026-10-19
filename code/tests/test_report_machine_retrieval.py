"""Explicit ranking depth changes no frozen input or formal test requirement."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_freeze_machine_dataset import make_inputs, module, protocol, reporter


def frozen_fixture(tmp_path):
    source = make_inputs(tmp_path)
    frozen, _ = module.freeze(source, tmp_path / "frozen")
    rows = [dict(record_id="q1", question="What was synthetic crude output?", mode=mode,
                 as_of="2024-01-01", elapsed_seconds=.1, evidence=[dict(passage_id="p1", document_id="d1",
                 rank=1, available_by="2024-01-01", text="Synthetic output was 1 million barrels per day.")])
            for mode in ("bm25", "dense", "hybrid")]
    rankings = tmp_path / "rankings.jsonl"
    rankings.write_bytes(b"".join(protocol.canonical_bytes(row) + b"\n" for row in rows))
    return frozen, rankings


def test_default_ten_preserved_and_five_explicit(tmp_path):
    frozen, rankings = frozen_fixture(tmp_path)
    ten = reporter.evaluate(frozen, rankings, tmp_path / "at10.json", split="development")
    five = reporter.evaluate(frozen, rankings, tmp_path / "at5.json", split="development", k=5)
    assert ten["k"] == 10 and five["k"] == 5
    assert ten["summary"]["hybrid"]["precision_at_k"] == .1
    assert five["summary"]["hybrid"]["precision_at_k"] == .2
    assert five["frozen_manifest_sha256"] == ten["frozen_manifest_sha256"]
    assert five["rankings_sha256"] == ten["rankings_sha256"]
    assert five["verified_input_sha256"] == ten["verified_input_sha256"]
    assert five["original_inputs_unchanged"] is True


def test_all_frozen_bytes_and_old_report_preserved(tmp_path):
    frozen, rankings = frozen_fixture(tmp_path)
    before = {str(path): path.read_bytes() for path in frozen.rglob("*") if path.is_file()}
    old = tmp_path / "old-at10.json"
    reporter.evaluate(frozen, rankings, old, split="development")
    old_bytes, ranked_bytes = old.read_bytes(), rankings.read_bytes()
    reporter.evaluate(frozen, rankings, tmp_path / "new-at5.json", split="development", k=5)
    assert before == {str(path): path.read_bytes() for path in frozen.rglob("*") if path.is_file()}
    assert old.read_bytes() == old_bytes and rankings.read_bytes() == ranked_bytes
    with pytest.raises(ValueError, match="Preserve earlier"):
        reporter.evaluate(frozen, rankings, old, split="development", k=5)


@pytest.mark.parametrize("k", [0, -1, True, 5.5, None])
def test_invalid_depth_fails_before_any_input_read(tmp_path, k):
    with pytest.raises(ValueError, match="positive integer"):
        reporter.evaluate(tmp_path / "missing", tmp_path / "missing.jsonl", tmp_path / "result.json", k=k)


def test_five_does_not_lower_formal_thousand_record_floor(tmp_path):
    frozen, rankings = frozen_fixture(tmp_path)
    with pytest.raises(ValueError, match="coverage"):
        reporter.evaluate(frozen, rankings, tmp_path / "not-formal.json", k=5)
    with pytest.raises(ValueError, match="1000-record"):
        reporter.evaluate(frozen, rankings, tmp_path / "lowered.json", required_test_records=1, k=5)


def test_cli_exposes_explicit_depth():
    completed = subprocess.run([sys.executable, reporter.__file__, "--help"], capture_output=True, text=True, check=True)
    assert "--k K" in completed.stdout and "default: 10" in completed.stdout


def test_mid_calculation_rank_input_tamper_fails_before_output(tmp_path, monkeypatch):
    frozen, rankings = frozen_fixture(tmp_path)
    original = reporter.evaluate_rankings
    def tamper(*args, **kwargs):
        result = original(*args, **kwargs)
        rankings.write_bytes(rankings.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(reporter, "evaluate_rankings", tamper)
    output = tmp_path / "should-not-exist.json"
    with pytest.raises(ValueError, match="changed during"):
        reporter.evaluate(frozen, rankings, output, split="development", k=5)
    assert not output.exists()
