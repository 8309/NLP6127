"""Prompt without echo; key exists only in this process environment, never a file."""
import getpass
import json
import os
import sys

from oil_evidence.annotations import generate_candidates
from oil_evidence.llm import ModelClient
from oil_evidence.store import DATA, write_json, read_jsonl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if not os.environ.get("OIL_LLM_API_KEY"):
    os.environ["OIL_LLM_API_KEY"] = getpass.getpass("DeepSeek API key (hidden; memory only): ")

try:
    response, usage = ModelClient().complete("Return a JSON object with ok=true. No other content.", "Connectivity test; JSON only.")
    if response.get("ok") is not True:
        raise ValueError("Unexpected connectivity result")
    print("Model connection verified:", usage["model"], flush=True)
    write_json(DATA / "reports/model_connection.json", dict(status="verified", model=usage["model"], usage=usage))
    remaining = max(0, 50 - len(read_jsonl(DATA / "annotations/candidates.jsonl")))
    print(json.dumps(generate_candidates(remaining), ensure_ascii=False, indent=2), flush=True)
finally:
    os.environ.pop("OIL_LLM_API_KEY", None)
