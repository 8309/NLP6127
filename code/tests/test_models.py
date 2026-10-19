import importlib
import json
import sqlite3

import httpx
import pytest

from oil_evidence.llm import ModelClient, ModelUnavailable
from oil_evidence.retrieval import Retriever


def settings(**overrides):
    return dict(dict(enabled=True, base_url="https://unit.test/v1", model="generator", verifier_model="checker",
                     api_key_env="TEST_OIL_KEY", budget_usd=1, input_usd_per_million=1,
                     output_usd_per_million=1, max_output_tokens=100), **overrides)


def test_budget_reserved_and_actual_usage_recorded(tmp_path, monkeypatch):
    import oil_evidence.llm as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setenv("TEST_OIL_KEY", "test-not-real")
    def respond(self, url, **kwargs):
        assert kwargs["json"]["model"] == "generator"
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5}})
    monkeypatch.setattr(httpx.Client, "post", respond)
    response, usage = ModelClient(settings()).complete("system", "question")
    assert response == {"ok": True} and usage["cost_usd"] == pytest.approx(.000025)
    with sqlite3.connect(tmp_path / "model_usage.sqlite3") as db:
        assert db.execute("select status from calls").fetchone()[0] == "complete"


def test_budget_refuses_before_network(tmp_path, monkeypatch):
    import oil_evidence.llm as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setenv("TEST_OIL_KEY", "test-not-real")
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **k: pytest.fail("network must not be called"))
    with pytest.raises(ModelUnavailable, match="budget"):
        ModelClient(settings(budget_usd=.000001)).complete("system", "question")


def test_ambiguous_failure_retains_budget_reservation(tmp_path, monkeypatch):
    import oil_evidence.llm as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setenv("TEST_OIL_KEY", "test-not-real")
    def failure(*a, **k):
        raise httpx.ReadTimeout("simulated")
    monkeypatch.setattr(httpx.Client, "post", failure)
    with pytest.raises(httpx.ReadTimeout):
        ModelClient(settings()).complete("system", "question")
    with sqlite3.connect(tmp_path / "model_usage.sqlite3") as db:
        status, reserved, cost = db.execute("select status,reserved,cost from calls").fetchone()
    assert status == "failed_or_unknown" and reserved > 0 and cost is None


def test_same_verifier_model_allowed_but_still_requires_key(monkeypatch):
    monkeypatch.delenv("TEST_OIL_KEY", raising=False)
    with pytest.raises(ModelUnavailable, match="Set TEST_OIL_KEY"):
        ModelClient(settings(verifier_model="generator")).complete("s", "p", role="verifier")


def test_generation_and_semantic_check_contract(monkeypatch):
    module = importlib.import_module("oil_evidence.query")
    seen = []
    def complete(self, system, prompt, role="generator"):
        seen.append((role, json.loads(prompt)))
        if role == "generator":
            return {"answerable":True, "claims":[
                {"text":"A cut was announced.", "citations":["a"]},
                {"text":"Production actually fell.", "citations":["a"]},
                {"text":"Invented.", "citations":["fake"]}],"limitations":[]}, {"usage":{}}
        return {"supported_indices":[0],"reasons":{"1":"Only an announcement"}}, {"usage":{}}
    monkeypatch.setattr(ModelClient, "complete", complete)
    retriever = Retriever(passages=[dict(passage_id="a",text="An oil production cut was announced, not yet implemented.",available_by="2025-01-01",regions=[])])
    result = module.query("oil production", mode="bm25", generation="llm", verify=True, retriever=retriever, log=False)
    assert len(result["claims"]) == 1 and len(result["removed_claims"]) == 2
    assert result["verification_method"] == "model_support_check_not_human"
    assert seen[1][0] == "verifier" and all(b["evidence"][0]["passage_id"] == "a" for b in seen[1][1]["claims"])
