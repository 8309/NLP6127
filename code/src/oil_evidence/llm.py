"""Explicitly configured compatible chat endpoint; conservative durable budget reservations."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from urllib.parse import urlparse

import httpx

from .store import DATA, config, digest, now


class ModelUnavailable(RuntimeError):
    pass


class ModelClient:
    def __init__(self, settings=None):
        self.settings = settings or config()["llm"]

    def complete(self, system, prompt, role="generator"):
        s = self.settings
        model = s["verifier_model"] if role == "verifier" else s["model"]
        if not s["enabled"] or not model or not s["base_url"]:
            raise ModelUnavailable("Model endpoint is not configured; no model call was made")
        parsed = urlparse(s["base_url"])
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not (local and parsed.scheme == "http"):
            raise ValueError("Remote model endpoints require HTTPS")
        key = os.environ.get(s["api_key_env"], "")
        if not local and not key:
            raise ModelUnavailable(f"Set {s['api_key_env']} locally; do not put secrets in config")
        if not local and (s["budget_usd"] <= 0 or s["input_usd_per_million"] <= 0 or s["output_usd_per_million"] <= 0):
            raise ModelUnavailable("Configure a positive budget and provider prices before remote calls")
        max_out = s["max_output_tokens"]
        # UTF-8 bytes plus framing is a conservative token upper bound for ordinary BPE endpoints.
        reserve = ((len((system + prompt).encode()) + 1024) * s["input_usd_per_million"] + max_out * s["output_usd_per_million"]) / 1e6
        DATA.mkdir(exist_ok=True)
        with sqlite3.connect(DATA / "model_usage.sqlite3", timeout=30) as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY,at TEXT,model TEXT,role TEXT,prompt_hash TEXT,status TEXT,reserved REAL,cost REAL,usage TEXT)")
            db.execute("BEGIN IMMEDIATE")
            spent = db.execute("SELECT COALESCE(SUM(COALESCE(cost,reserved)),0) FROM calls").fetchone()[0]
            if not local and spent + reserve > s["budget_usd"]:
                raise ModelUnavailable("Configured model budget exhausted")
            call_id = uuid.uuid4().hex
            db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?)", (call_id, now(), model, role, digest(system + prompt), "reserved", reserve, None, None))
            db.commit()
        body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": max_out, "response_format": {"type": "json_object"}}
        if parsed.hostname == "api.deepseek.com":
            # Explicitly disable default reasoning for bounded, short JSON extraction.
            body["thinking"] = {"type": "disabled"}
        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(s["base_url"].rstrip("/") + "/chat/completions", json=body,
                                       headers={"Authorization": "Bearer " + (key or "local")})
                response.raise_for_status()
                result = response.json()
            usage = result.get("usage", {})
            cost = None
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                cost = (usage["prompt_tokens"] * s["input_usd_per_million"] + usage["completion_tokens"] * s["output_usd_per_million"]) / 1e6
            with sqlite3.connect(DATA / "model_usage.sqlite3") as db:
                db.execute("UPDATE calls SET status=?,cost=?,usage=? WHERE id=?", ("complete", cost, json.dumps(usage), call_id))
            content = json.loads(result["choices"][0]["message"]["content"])
            return content, dict(call_id=call_id, model=model, resolved_model=result.get("model"), usage=usage,
                                 cost_usd=cost, cost_basis="configured conservative price estimate, not provider invoice", reserved_usd=reserve)
        except BaseException:
            with sqlite3.connect(DATA / "model_usage.sqlite3") as db:
                # Keep reservations for ambiguous failures: the provider may have billed them.
                db.execute("UPDATE calls SET status=? WHERE id=?", ("failed_or_unknown", call_id))
            raise
