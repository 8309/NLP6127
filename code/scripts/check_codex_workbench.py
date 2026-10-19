"""One real Codex query through Streamlit AppTest; no hosted API or verifier."""
import argparse
import json
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

from oil_evidence.codex_runtime import _write_new, load_staged_runtime
from oil_evidence.store import ROOT, digest, now


def check(canonical_root):
    root = Path(canonical_root).resolve()
    os.environ["OIL_CANONICAL_ROOT"] = str(root)
    os.environ["OIL_STAGED_RUN"] = "semantic-page-v2-buffer"
    os.environ["OIL_GENERATION_PROVIDER"] = "codex"
    runtime = load_staged_runtime(root)
    app_sha = digest((ROOT / "app.py").read_bytes())
    before = {str(root / "data" / name): digest((root / "data" / name).read_bytes()) for name in (
        "passages.jsonl", "index/manifest.json", "index/vectors.npy", "model_usage.sqlite3", "annotations.sqlite3")}
    view = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    view.run()
    assert not view.exception and not view.error, "Initial staged interface failed"
    assert len(view.tabs) == 4
    provider = next(item for item in view.selectbox if item.label == "Generation provider")
    assert provider.value == "codex"
    assert all(item.label != "DeepSeek API key" for item in view.text_input)
    question = next(item for item in view.text_area if item.label == "Research question")
    question.set_value("What does EIA project for US crude oil production, and in what units?")
    next(item for item in view.button if item.label == "Search").click()
    view.run(timeout=180)
    assert not view.exception and not view.error, "Actual Codex UI query failed"
    result = view.session_state["query_result"]
    assert result["generation"] == "codex" and len(result["model_calls"]) == 1
    assert result["external_paid_api_calls"] == 0 and result["original_inputs_unchanged"]
    assert result["corpus_sha256"] == runtime.corpus_sha256
    assert result["total_seconds"] >= result["retrieval_seconds"] + result["generation_seconds"]
    assert all(digest(Path(name).read_bytes()) == expected for name, expected in before.items())
    assert digest((ROOT / "app.py").read_bytes()) == app_sha
    receipt = dict(status="real_codex_streamlit_query_verified_not_gold_evaluation", checked_utc=now(),
        tabs=4, generation="codex", no_api_key_input=True, corpus_sha256=result["corpus_sha256"],
        evidence=len(result["evidence"]), answerability=result["answerability"], claims=len(result["claims"]),
        quotes=len(result["quotes"]), retrieval_seconds=result["retrieval_seconds"],
        generation_seconds=result["generation_seconds"], total_seconds=result["total_seconds"],
        native_invocations=1, usage=result["model_calls"][0]["usage"],
        execution_id=result["model_calls"][0]["execution_id"], observed_model=result["model_calls"][0]["observed_model"],
        external_paid_api_calls=0, subscription_dollar_cost=None, human_gold=False,
        semantic_faithfulness_verified=False, default_inputs_unchanged=True,
        app_sha256=app_sha, checker_sha256=digest(Path(__file__).read_bytes()),
        query_result_sha256=digest((Path(result["output_directory"]) / "result.json").read_bytes()))
    _write_new(Path(result["output_directory"]) / "ui-check.json", receipt)
    return Path(result["output_directory"]), receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    args = parser.parse_args()
    output, receipt = check(args.canonical_root)
    print(json.dumps(dict(output=str(output), **receipt), indent=2))
