"""Synthetic paired-ablation checks; no native calls or real quality scores."""
from copy import deepcopy
import importlib.util
from itertools import count
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def ablations(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("isolated_codex_ablations", scripts / "run_codex_ablations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def answer(pid="fixture-current"):
    return dict(answerability="sufficient", claims=[dict(text="Synthetic reported production.", citations=[pid])],
                quotes=[dict(passage_id=pid, text="Synthetic reported production.")], limitations=[])


def case():
    return dict(question="What is the synthetic reported production?", as_of="2024-06-30")


def evidence(pid="fixture-current", availability="2024-01-01"):
    return dict(passage_id=pid, source_page_id="page-" + pid, text="Synthetic reported production.",
                available_by=availability, char_start=0, char_end=len("Synthetic reported production."), regions=[])


def test_generation_rejects_case_coverage_mismatch(ablations, tmp_path):
    with pytest.raises(ValueError, match="coverage"):
        ablations.generate_batch({"fixture-case":case()}, {"fixture-case":[evidence()]}, True, tmp_path,
            lambda *args, **kwargs: ({"another-case":answer()}, dict(execution_id="synthetic")))


def test_b_reuses_exact_generation_and_preserves_background_quotes(ablations, tmp_path):
    original = {"fixture-case":answer()}
    saved = deepcopy(original)
    checked, removed, receipt = ablations.verify_batch({"fixture-case":case()}, original,
        {"fixture-case":[evidence()]}, tmp_path, lambda *args, **kwargs: (
            {"fixture-case":dict(supported_indices=[], reasons=["Synthetic unsupported claim."])}, dict(execution_id="fixture")))
    assert original == saved
    assert checked["fixture-case"]["answerability"] == "insufficient"
    assert checked["fixture-case"]["claims"] == []
    assert checked["fixture-case"]["quotes"] == saved["fixture-case"]["quotes"]
    assert removed["fixture-case"] == saved["fixture-case"]["claims"]


def test_insufficient_case_skips_unnecessary_verifier_call(ablations, tmp_path):
    raw = dict(answerability="insufficient", claims=[], quotes=[], limitations=["Synthetic gap"])
    checked, removed, receipt = ablations.verify_batch({"fixture-case":case()}, {"fixture-case":raw},
        {"fixture-case":[]}, tmp_path, lambda *args, **kwargs: pytest.fail("No claims need no verifier"))
    assert checked["fixture-case"] == raw and removed["fixture-case"] == [] and receipt is None


def test_duplicate_indices_are_rejected_not_silently_collapsed(ablations, tmp_path):
    def caller(prompt, payload, schema, output, role):
        assert "uniqueItems" not in json.dumps(schema)
        return {"fixture-case":dict(supported_indices=[0, 0], reasons=[])}, {}
    with pytest.raises(ValueError, match="duplicate claim indices"):
        ablations.verify_batch({"fixture-case":case()}, {"fixture-case":answer()},
                               {"fixture-case":[evidence()]}, tmp_path, caller)


def test_missing_verifier_case_is_rejected_not_treated_as_no_support(ablations, tmp_path):
    with pytest.raises(ValueError, match="coverage"):
        ablations.verify_batch({"fixture-case":case()}, {"fixture-case":answer()},
                               {"fixture-case":[evidence()]}, tmp_path, lambda *args, **kwargs: ({}, {}))


@pytest.mark.parametrize("index", [True, -1, 1])
def test_invalid_verifier_indices_rejected(ablations, tmp_path, index):
    with pytest.raises(ValueError, match="invalid claim index"):
        ablations.verify_batch({"fixture-case":case()}, {"fixture-case":answer()},
            {"fixture-case":[evidence()]}, tmp_path, lambda *args, **kwargs: (
                {"fixture-case":dict(supported_indices=[index], reasons=[])}, {}))


def test_six_runtime_snapshots_are_bound_to_saved_bytes(ablations, tmp_path):
    stage = tmp_path / "stage"
    (stage / "index").mkdir(parents=True)
    (stage / "index/manifest.json").write_text('{"synthetic":true}', encoding="utf-8")
    hashes = ablations.snapshot_runtime(tmp_path / "run", SimpleNamespace(stage=stage))
    paths = json.loads((tmp_path / "run/runtime_artifacts.json").read_bytes())
    assert set(hashes) == set(paths) == {"index_manifest", "retrieval_config", "generation_prompt", "verification_prompt", "judge_prompt", "implementation"}
    for name, path in paths.items():
        assert ablations.protocol.sha256(Path(path).read_bytes()) == hashes[name]
    for role in ("generation", "verification", "judge"):
        assert Path(paths[role + "_prompt"]).read_bytes().endswith(b"\nUNTRUSTED_PAYLOAD_JSON\n")
    implementation = json.loads(Path(paths["implementation"]).read_bytes())
    for source in implementation.values():
        assert ablations.protocol.sha256(source["source_utf8"].encode("utf-8")) == source["sha256"]


def test_changed_runtime_fails_before_any_model_call(ablations, tmp_path):
    protected = tmp_path / "protected.json"
    protected.write_bytes(b"changed")
    runtime = SimpleNamespace(protected_sha256={str(protected):"0"*64})
    with pytest.raises(ValueError, match="input changed"):
        ablations.run(runtime, {"fixture-case":dict(payload=case())}, {}, tmp_path / "runs", ["fixture-case"],
            caller=lambda *args, **kwargs: pytest.fail("Changed inputs must fail before a model call"))


def test_explicit_region_checked_before_generation(ablations, tmp_path, monkeypatch):
    row = evidence()
    searches = []
    def search(question, **kwargs):
        searches.append(kwargs)
        return [dict(row, rank=1, score=1.0)]
    runtime = SimpleNamespace(protected_sha256={}, corpus_sha256="fixture", passages={row["passage_id"]:row},
        pages={row["source_page_id"]:dict(text=row["text"])}, retriever=SimpleNamespace(search=search))
    monkeypatch.setattr(ablations, "snapshot_runtime", lambda *args: {})
    monkeypatch.setattr(ablations.protocol, "validate_label", lambda *args: [])
    query = dict(case(), region="Middle East")
    reference = dict(status="accepted", human_gold=False, provenance="machine_adjudicated", label={})
    with pytest.raises(ValueError, match="explicit query region"):
        ablations.run(runtime, {"fixture-case":dict(payload=query)}, {"fixture-case":reference},
            tmp_path / "runs", ["fixture-case"], caller=lambda *args, **kwargs: pytest.fail("Invalid region evidence"))
    assert searches[-1]["region"] == "Middle East"


@pytest.fixture
def reuse_fixture(ablations, tmp_path):
    source, output = tmp_path / "runs/old", tmp_path / "runs/new"
    source.mkdir(parents=True)
    output.mkdir()
    cases = {"fixture-case":case()}
    definitions = (("bm25", "bm25", False), ("dense", "dense", False), ("hybrid_base", "hybrid", False), ("hybrid_A", "hybrid", True))
    hits = {variant:{"fixture-case":[evidence()]} for variant, _, _ in definitions}
    runtime_values = {key:b"Synthetic runtime" for key in ablations.protocol.RUNTIME_KEYS}
    runtime_values["implementation"] = ablations.protocol.canonical_bytes({"fixture.py":dict(
        source_utf8="Synthetic source", sha256=ablations.protocol.sha256(b"Synthetic source"))})
    hashes = {}
    for key, raw in runtime_values.items():
        ablations.write_new(source / "runtime" / (key + ".artifact"), raw)
        hashes[key] = ablations.protocol.sha256(raw)
    inputs = dict(selected_development_ids=list(cases), packets_sha256="packet-hash", machine_reference_sha256="reference-hash",
                  corpus_sha256="corpus-hash", human_gold=False, formal_heldout_test=False, runtime_sha256=hashes)
    ablations.write_new(source / "inputs.json", inputs)
    for variant, _, _ in definitions:
        attempt = source / variant / "calls/attempt"
        payload = {qid:dict(question=value["question"], as_of=value["as_of"], evidence=hits[variant][qid]) for qid, value in cases.items()}
        schema = ablations.object_schema({qid:ablations.answer_schema(hits[variant][qid]) for qid in cases})
        prompt = (ablations.ISOLATION_PROMPT + "\n" + ablations.GENERATION_PROMPT + "\nUNTRUSTED_PAYLOAD_JSON\n").encode() + ablations.protocol.canonical_bytes(payload)
        raw = ablations.protocol.canonical_bytes({"fixture-case":answer()})
        usage = dict(input_tokens=10, output_tokens=2)
        events = [dict(type="thread.started", thread_id="fixture-" + variant), dict(type="turn.started"),
                  dict(type="item.completed", item=dict(type="agent_message", text=raw.decode())),
                  dict(type="turn.completed", usage=usage)]
        for name, content in {"stdin.txt":prompt, "schema.json":ablations.protocol.canonical_bytes(schema), "raw_final.json":raw,
                "events.jsonl":b"\n".join(ablations.protocol.canonical_bytes(event) for event in events), "stderr.txt":b""}.items():
            ablations.write_new(attempt / name, content)
        artifacts = {path.name:ablations.protocol.sha256(path.read_bytes()) for path in attempt.iterdir()}
        receipt = dict(role="generator", accepted=True, process_exit_code=0, errors=[], model_tools_disabled=True,
            execpolicy_rules_not_disabled=True, codex_cli_invocations=1, artifact_directory=str(attempt),
            artifacts_sha256=artifacts, prompt_sha256=ablations.protocol.sha256(prompt),
            schema_sha256=ablations.protocol.sha256(ablations.protocol.canonical_bytes(schema)), execution_id="fixture-" + variant,
            usage=usage, observed_model=None)
        ablations.write_new(attempt / "receipt.json", receipt)
    return source, output, cases, hits, definitions, inputs


def test_recovery_copies_original_bytes_without_generator_calls(ablations, reuse_fixture):
    source, output, cases, hits, definitions, inputs = reuse_fixture
    before = ablations.tree_hashes(source)
    answers, receipts, protected = ablations.reuse_generations(source, output, cases, hits, definitions, inputs)
    assert len(answers) == len(receipts) == 4 and protected == before
    assert ablations.tree_hashes(source) == before
    for variant, _, _ in definitions:
        assert (output / "reused_generation" / variant / "raw_final.json").read_bytes() == (source / variant / "calls/attempt/raw_final.json").read_bytes()
        assert (output / "reused_generation" / variant / "receipt.json").read_bytes() == (source / variant / "calls/attempt/receipt.json").read_bytes()


@pytest.mark.parametrize("changed", ["selected_development_ids", "packets_sha256", "machine_reference_sha256", "corpus_sha256"])
def test_recovery_binding_drift_fails_before_any_call(ablations, reuse_fixture, changed):
    source, output, cases, hits, definitions, inputs = reuse_fixture
    expected = dict(inputs, **{changed:["different"] if changed == "selected_development_ids" else "different"})
    with pytest.raises(ValueError, match="Reuse inputs differ"):
        ablations.reuse_generations(source, output, cases, hits, definitions, expected)


def test_recovery_rejects_changed_current_generation_payload(ablations, reuse_fixture):
    source, output, cases, hits, definitions, inputs = reuse_fixture
    cases["fixture-case"]["question"] = "Changed current question"
    with pytest.raises(ValueError, match="prompt, payload or schema differs"):
        ablations.reuse_generations(source, output, cases, hits, definitions, inputs)


def test_recovery_rejects_original_raw_artifact_drift(ablations, reuse_fixture):
    source, output, cases, hits, definitions, inputs = reuse_fixture
    (source / "bm25/calls/attempt/raw_final.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        ablations.reuse_generations(source, output, cases, hits, definitions, inputs)


def test_mocked_full_cohort_counts_native_usage_once_and_has_no_fabricated_query_latency(ablations, tmp_path, monkeypatch):
    current, future = evidence(), evidence("fixture-future", "2025-01-01")
    passages = {row["passage_id"]: row for row in (current, future)}
    pages = {row["source_page_id"]:dict(text=row["text"]) for row in passages.values()}
    def search(question, temporal=True, **kwargs):
        row = current if temporal else future
        return [dict(row, rank=1, score=1.0)]
    runtime = SimpleNamespace(retriever=SimpleNamespace(search=search), passages=passages, pages=pages,
                              protected_sha256={}, corpus_sha256="fixture-corpus")
    monkeypatch.setattr(ablations, "snapshot_runtime", lambda *args: {"fixture":"hash"})
    monkeypatch.setattr(ablations.protocol, "validate_label", lambda *args: [])
    packets = {"fixture-case":dict(payload=case())}
    refs = {"fixture-case":dict(status="accepted", human_gold=False, provenance="machine_adjudicated",
                               label=dict(qrels={"fixture-current":2, "fixture-future":0}))}
    executions, roles = count(1), []
    def caller(prompt, payload, schema, output, role):
        roles.append(role)
        receipt = dict(execution_id="fixture-" + str(next(executions)), usage=dict(input_tokens=10, output_tokens=2))
        if role == "generator":
            return {qid:answer(value["evidence"][0]["passage_id"]) for qid, value in payload.items()}, receipt
        if role == "verifier":
            return {qid:dict(supported_indices=[] if value["claims"][0]["evidence"][0]["available_by"] > value["as_of"] else [0], reasons=[])
                    for qid, value in payload.items()}, receipt
        return {key:dict(correctness=2, relevance=2, claim_supported=[True]*len(value["answer"]["claims"]),
            citation_supported=[True]*sum(len(claim["citations"]) for claim in value["answer"]["claims"]),
            context_relevant=[True]*len(value["evidence"]), insufficiency_handling=True, reasons=[])
            for key, value in payload.items()}, receipt
    output, result = ablations.run(runtime, packets, refs, tmp_path / "runs", ["fixture-case"], caller=caller)
    assert result["status"] == "completed_development_pilot_not_formal_heldout_experiment"
    assert result["generated_base_answers"] == 4 and result["evaluated_variant_answers"] == 6
    assert roles.count("generator") == 4 and roles.count("verifier") == 2 and roles.count("judge") == 1
    assert result["native_executions"] == 7 and result["native_token_totals"] == dict(input_tokens=70, output_tokens=14)
    rows = {row["variant"]:row for row in result["results"]}
    assert rows["hybrid_B"]["answer"]["claims"] == []
    assert rows["hybrid_base"]["answer"]["claims"]
    assert rows["hybrid_B"]["evidence"] == rows["hybrid_base"]["evidence"]
    assert rows["hybrid_AB"]["answer"]["claims"] == rows["hybrid_A"]["answer"]["claims"]
    assert all(row["per_query_generation_seconds"] is None and row["per_query_total_seconds"] is None for row in rows.values())
    assert result["human_gold"] is False and result["subscription_dollar_cost"] is None
    assert (output / "result.json").is_file()
