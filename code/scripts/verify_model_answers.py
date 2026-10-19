"""Small real-model end-to-end verification; uses the same durable USD budget ledger."""
import getpass
import os
import sys

from oil_evidence.query import query
from oil_evidence.retrieval import Retriever
from oil_evidence.store import DATA, write_jsonl

sys.stdout.reconfigure(encoding="utf-8")
if not os.environ.get("OIL_LLM_API_KEY"):
    os.environ["OIL_LLM_API_KEY"] = getpass.getpass("DeepSeek API key (hidden; memory only): ")
try:
    retriever = Retriever()
    examples = [
        ("What does EIA report about crude oil supply risks and inventories?", "2026-10-05"),
        ("As of December 2020, what did EIA report about the pandemic's impact on petroleum consumption?", "2020-12-31"),
        ("Do the sources prove the exact Brent oil price for next month?", "2026-10-05"),
        ("Does an announced OPEC production target by itself establish actual production?", "2026-10-05"),
        ("What evidence describes the Strait of Hormuz as an oil transit chokepoint?", "2026-10-05"),
    ]
    results = []
    for question, cutoff in examples:
        result = query(question, as_of=cutoff, generation="llm", verify=True, retriever=retriever)
        results.append(result)
        print(f"Completed {len(results)}/5; claims={len(result['claims'])}; answerable={result['answerable']}; seconds={result['total_seconds']:.2f}", flush=True)
        write_jsonl(DATA / "reports/model_answer_examples.jsonl", results)
finally:
    os.environ.pop("OIL_LLM_API_KEY", None)
