"""Synthetic Streamlit integration checks; no native or hosted generation."""
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from oil_evidence import codex_runtime, store


@pytest.fixture
def app(tmp_path, monkeypatch):
    root = tmp_path / "canonical"
    pointer = root / "data/unified_runs/semantic-page-v2-buffer/reports/index_build.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OIL_CANONICAL_ROOT", str(root))
    monkeypatch.setenv("OIL_GENERATION_PROVIDER", "codex")
    monkeypatch.setattr(store, "DATA", tmp_path / "public-data")
    monkeypatch.setattr(codex_runtime, "load_staged_runtime", lambda *args: SimpleNamespace(retriever=None, pages={}))
    result = dict(retrieval_mode="hybrid", total_seconds=0.25, retrieval_seconds=0.05,
        generation_seconds=0.1, generation="codex", answerability="insufficient", limitations=["Synthetic evidence gap"],
        claims=[], quotes=[], evidence=[], model_calls=[dict(execution_id="fixture-thread", observed_model=None,
        usage=dict(input_tokens=10, output_tokens=3), generation_seconds=0.1, artifact_directory="synthetic")])
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs)
        return result
    monkeypatch.setattr(codex_runtime, "query_codex", generate)
    return AppTest.from_file(str(Path(__file__).parents[1] / "app.py")), calls


def test_staged_codex_default_has_no_api_key_and_shows_real_timing(app):
    view, calls = app
    view.run()
    assert not view.exception
    assert next(item for item in view.selectbox if item.label == "Generation provider").value == "codex"
    assert all(item.label != "DeepSeek API key" for item in view.text_input)
    next(item for item in view.button if item.label == "Search").click()
    view.run()
    assert not view.exception and not view.error
    assert len(calls) == 1 and calls[0]["generation"] == "codex"
    assert any("Actual single-query end to end" in item.value for item in view.caption)
    assert any("Synthetic evidence gap" in item.value for item in view.info)


def test_staged_optional_deepseek_does_not_open_a_new_ledger(app):
    view, calls = app
    view.run()
    next(item for item in view.selectbox if item.label == "Generation provider").set_value("deepseek")
    view.run()
    next(item for item in view.selectbox if item.label == "Answer mode").set_value("llm")
    next(item for item in view.button if item.label == "Search").click()
    view.run()
    assert not view.exception and not calls
    assert any("retained cost ledger" in item.value for item in view.error)
