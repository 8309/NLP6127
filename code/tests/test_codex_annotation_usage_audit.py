"""Observed usage is neither duplicated nor replaced by invented zeros."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def auditor(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("annotation_usage_audit", scripts / "audit_codex_annotation_usage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(auditor, tmp_path, name, usage=None, execution_id="execution-1"):
    events = [dict(type="thread.started", thread_id=execution_id)]
    if usage is not None:
        events.append(dict(type="turn.completed", usage=usage))
    auditor.native.write_new(tmp_path / "codex_annotations" / name / "events.jsonl",
                             b"".join(auditor.native.protocol.canonical_bytes(row) + b"\n" for row in events))


def test_migrated_artifact_copies_count_once(auditor, tmp_path):
    save(auditor, tmp_path, "original", dict(input_tokens=100, output_tokens=20))
    save(auditor, tmp_path, "migrated", dict(input_tokens=100, output_tokens=20))
    result = auditor.audit(tmp_path)
    assert result["unique_native_attempts"] == 1
    assert result["observed_tokens"] == dict(input_tokens=100, output_tokens=20)
    assert len(result["executions"][0]["event_paths"]) == 2


def test_interrupted_unknown_usage_not_zero(auditor, tmp_path):
    save(auditor, tmp_path, "interrupted")
    result = auditor.audit(tmp_path)
    assert result["attempts_without_observed_usage"] == 1 and result["observed_tokens"] == {}
    assert result["executions"][0]["usage"] is None and result["subscription_dollar_cost"] is None


def test_conflicting_duplicate_execution_refused(auditor, tmp_path):
    save(auditor, tmp_path, "one", dict(input_tokens=100))
    save(auditor, tmp_path, "two", dict(input_tokens=200))
    with pytest.raises(ValueError, match="Conflicting usage"):
        auditor.audit(tmp_path)
