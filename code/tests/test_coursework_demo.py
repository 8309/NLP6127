"""Synthetic format, trace and isolation checks; no model calls or gold metrics."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def demo(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("isolated_coursework_demo", scripts / "run_coursework_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cases():
    return [dict(case_id=f"q2-demo-{number:02d}", question="Synthetic research question", as_of="2024-06-30",
                 evidence=[dict(passage_id=f"fixture-{number}", available_by="2024-01-01",
                    text="Synthetic crude export evidence.", title="Synthetic source", url="https://fixture.invalid/doc")],
                 retrieval_seconds=0.1) for number in range(1, 6)]


@pytest.fixture
def answers(cases):
    return dict(cases={case["case_id"]: dict(answerability="sufficient", limitations=[], claims=[dict(
        text="Synthetic evidence statement.", citations=[case["evidence"][0]["passage_id"]])]) for case in cases})


def test_fixed_public_seeds_and_case_count(demo):
    selected = demo.selected_cases(Path(__file__).parents[1] / "seeds/questions.json")
    assert len(selected) == 5
    assert selected[-1]["question"].startswith("What exact Brent price")
    assert all(case["top_k"] == 5 and case["retrieval_mode"] == "hybrid" for case in selected)


def test_missing_or_duplicate_seed_fails(demo, tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or duplicated"):
        demo.selected_cases(path)


def test_schema_constrains_each_cases_own_citations(demo, cases):
    schema = demo.batch_schema(cases)
    properties = schema["properties"]["cases"]["properties"]
    for case in cases:
        citations = properties[case["case_id"]]["properties"]["claims"]["items"]["properties"]["citations"]
        assert citations["items"]["enum"] == [case["evidence"][0]["passage_id"]]


def test_valid_structure_does_not_claim_semantic_support(demo, cases, answers):
    answers["cases"]["q2-demo-01"]["claims"][0]["text"] = "This invented statement has a valid ID but no semantic support."
    assert demo.validate_answers(answers, cases) == []


@pytest.mark.parametrize("mutation", ["missing_case", "extra_case", "cross_case", "empty_claim", "unknown_date", "future_date", "bad_answerability", "no_claims", "bad_limitations"])
def test_invalid_batch_rejected(demo, cases, answers, mutation):
    case = cases[0]
    answer = answers["cases"][case["case_id"]]
    if mutation == "missing_case":
        answers["cases"].pop(case["case_id"])
    elif mutation == "extra_case":
        answers["cases"]["extra"] = deepcopy(answer)
    elif mutation == "cross_case":
        answer["claims"][0]["citations"] = ["fixture-2"]
    elif mutation == "empty_claim":
        answer["claims"][0]["text"] = " "
    elif mutation == "unknown_date":
        case["evidence"][0]["available_by"] = None
    elif mutation == "future_date":
        case["evidence"][0]["available_by"] = "2025-01-01"
    elif mutation == "bad_answerability":
        answer["answerability"] = "verified_by_human"
    elif mutation == "no_claims":
        answer["claims"] = []
    else:
        answer["limitations"] = [None]
    assert demo.validate_answers(answers, cases)


def test_insufficient_answer_can_have_no_claims(demo, cases, answers):
    answers["cases"][cases[0]["case_id"]] = dict(answerability="insufficient", claims=[],
                                                limitations=["Synthetic evidence is insufficient."])
    assert not demo.validate_answers(answers, cases)


def test_html_escapes_untrusted_text_and_counts_usage_once(demo, cases, answers):
    traces = [dict(case, answer=answers["cases"][case["case_id"]]) for case in cases]
    traces[0]["evidence"][0]["text"] = "<script>untrusted()</script>"
    traces[0]["evidence"][0]["url"] = "javascript:untrusted()"
    batch = dict(usage={"output_tokens":123}, corpus_sha256="fixture-hash", elapsed_seconds=12.3)
    rendered = demo.render_html(traces, batch)
    assert "<script>" not in rendered
    assert "javascript:" not in rendered
    assert "&lt;script&gt;" in rendered
    assert rendered.count("output_tokens") == 1
    assert "Per-query generation and end-to-end latency are not measured" in rendered
    assert "No verifier or quality judge ran" in rendered


def test_shared_cli_isolation_removes_external_keys(demo):
    environment = demo.isolation.child_environment({"OPENAI_API_KEY":"synthetic", "OIL_LLM_API_KEY":"synthetic", "SAFE":"ok"})
    assert environment == {"SAFE":"ok"}
    command = demo.isolation.cli_command("fixture.exe", Path("fixture"), Path("schema.json"), Path("output.json"))
    assert "read-only" in command and "--ephemeral" in command
    assert "features.shell_tool=false" in command
    assert "--ignore-rules" not in command and "--dangerously-bypass-approvals-and-sandbox" not in command


def test_shared_native_event_parser_rejects_tool_use(demo):
    events = [dict(type="thread.started", thread_id="fixture-thread"),
              dict(type="item.completed", item=dict(type="command_execution")),
              dict(type="turn.completed", usage={"input_tokens":10})]
    parsed = demo.isolation.parse_execution_events("\n".join(json.dumps(event) for event in events).encode())
    assert parsed["errors"] and parsed["usage"] == {"input_tokens":10}
