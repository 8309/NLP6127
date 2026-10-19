"""Synthetic formal-runner contract tests; no real model calls or quality claims."""
from copy import deepcopy
from itertools import count
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import freeze_machine_dataset as freezer
import run_machine_experiments as module
from test_answer_sample import sampling_fixture
from oil_evidence import machine_evaluation as protocol


def hit():
    text = "Synthetic crude output was one million barrels per day."
    return dict(passage_id="fixture-p", document_id="fixture-d", source_page_id="fixture-page",
        text=text, char_start=0, char_end=len(text), available_by="2024-01-01", regions=[])


def runtime(tmp_path):
    row = hit()
    stage = tmp_path / "canonical/stage"
    (stage / "index").mkdir(parents=True)
    (stage / "index/manifest.json").write_text('{"synthetic_fixture":true}', encoding="utf-8")
    raw = protocol.canonical_bytes(row) + b"\n"
    return SimpleNamespace(stage=stage, canonical_root=tmp_path / "canonical",
        corpus_sha256=protocol.sha256(raw), passages={row["passage_id"]:row},
        pages={row["source_page_id"]:dict(text=row["text"])}, protected_sha256={})


def cohort_fixture(tmp_path, records=1000, unresolved=False, max_cases=32):
    rt = runtime(tmp_path)
    sample_root, sample_cohort, sample_status, sample_progress = sampling_fixture(tmp_path)
    original_questions = module.protocol.parse_jsonl((sample_cohort / "questions.jsonl").read_bytes())[:200]
    original_tests = [dict(record_id=f"fixture-q-{index:04}", question=f"What was synthetic crude output for case {index}?",
        as_of="2024-01-01", split="test", event_group=f"fixture-event-{index}") for index in range(1000)]
    for name, rows in (("questions.jsonl",original_questions+original_tests),("test_questions.jsonl",original_tests)):
        (sample_cohort / name).write_bytes(b"".join(protocol.canonical_bytes(row)+b"\n" for row in rows))
    receipt = module.read_json(sample_cohort / "receipt.json")
    receipt["artifact_sha256"] = {name:module.file_hash(sample_cohort / name) for name in receipt["artifact_sha256"]}
    (sample_cohort / "receipt.json").write_bytes(protocol.canonical_bytes(receipt))
    status = module.read_json(sample_status)
    status.update(questions_sha256=receipt["artifact_sha256"]["questions.jsonl"],
                  test_questions_sha256=receipt["artifact_sha256"]["test_questions.jsonl"])
    sample_status.write_bytes(protocol.canonical_bytes(status))
    sample_dir, _ = module.sampling.prepare(sample_root,sample_cohort,sample_status,module.file_hash(sample_status),sample_progress)
    sample_binding, _ = module.load_sample_binding(sample_dir, sample_root)
    source = tmp_path / "freeze-input"
    source.mkdir()
    corpus = [hit()]
    questions = [dict(record_id=f"fixture-q-{index:04}", question=f"What was synthetic crude output for case {index}?",
        as_of="2024-01-01", split="test", event_group=f"fixture-event-{index}") for index in range(records)]
    pools = [dict(record_id=q["record_id"], evidence_ids=["fixture-p"]) for q in questions]
    packets = protocol.prepare_inputs(questions, pools, corpus)
    label = dict(answerability="sufficient", event_status="observed", qrels={"fixture-p":2},
        answer_points=[dict(text=hit()["text"], citations=["fixture-p"])],
        quotes=[dict(passage_id="fixture-p", text=hit()["text"])], reasons=["Synthetic source fixture."])
    originals = [[], []]
    decisions = []
    for question in questions:
        qid = question["record_id"]
        for index, side in enumerate(("a","b")):
            identity = dict(service="synthetic_fixture", requested_model="synthetic_fixture", observed_model=None,
                execution_id=side+qid, observation_source="unavailable", identity_note="Synthetic fixture, no call.")
            originals[index].append(protocol.wrap_independent_output(packets[qid], protocol.canonical_bytes(label).decode(),
                identity, side+qid, {}, execution_input_sha256=packets[qid]["input_sha256"]))
        decisions.append(dict(record_id=qid, status="unresolved" if unresolved else "accepted",
            provenance="machine_adjudicated", human_gold=False, label=label,
            unresolved_reasons=["Synthetic unresolved fixture."], adjudication=dict(reason="Synthetic review.",
            both_originals_seen=True, original_output_sha256=[protocol.sha256(protocol.canonical_bytes(rows[-1])) for rows in originals])))
    values = dict(corpus=corpus, questions=questions, evidence_pool=pools, independent_a=originals[0],
        independent_b=originals[1], adjudications=decisions)
    for key, name in module.NAMES.items():
        raw = protocol.ANNOTATION_PROMPT.encode() if key == "annotation_prompt" else b"".join(
            protocol.canonical_bytes(row) + b"\n" for row in values[key])
        (source / name).write_bytes(raw)
    snapshots = module.snapshot(rt, tmp_path / "runtime-snapshot", module.settings(max_cases=max_cases, sample_binding=sample_binding))
    (source / "runtime_artifacts.json").write_bytes(protocol.canonical_bytes(snapshots))
    # The freezer requires snapshots staged inside its own source directory.
    paths = {}
    for key, name in snapshots.items():
        path = source / "runtime" / (key + ".artifact")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(Path(name).read_bytes())
        paths[key] = str(path)
    (source / "runtime_artifacts.json").write_bytes(protocol.canonical_bytes(paths))
    frozen, _ = freezer.freeze(source, tmp_path / "frozen")
    retrieval = rt.canonical_root / "data/machine_runs/candidate-retrieval/fixture"
    retrieval.mkdir(parents=True, exist_ok=True)
    conditions = {key:[dict(hit(), rank=1, score=1.0)] for key in module.CONDITIONS}
    rankings = [dict(record_id=q["record_id"], ranking_seconds_excluding_embedding=.1, conditions=conditions) for q in questions]
    payloads = [packets[q["record_id"]]["payload"] for q in questions]
    (retrieval / "rankings.jsonl").write_bytes(b"".join(protocol.canonical_bytes(row)+b"\n" for row in rankings))
    (retrieval / "annotation_pool.jsonl").write_bytes(b"".join(protocol.canonical_bytes(row)+b"\n" for row in payloads))
    (retrieval / "inputs.json").write_bytes(protocol.canonical_bytes(dict(
        protocol="machine-candidate-top5-three-mode-two-temporal-pool-v1", protected_sha256={},
        bindings=dict(corpus_sha256=rt.corpus_sha256))))
    receipt = dict(status="candidate_evidence_pools_created_not_annotations_or_accuracy_results",
        original_inputs_unchanged=True, top_k=5, retrieval_conditions_per_question=6, author_anchor_injection=False,
        batch_query_embedding_seconds=1.0, ranking_seconds_excluding_embedding=.1*records, wall_seconds=2.0,
        artifact_sha256={p.name:module.file_hash(p) for p in retrieval.iterdir() if p.is_file()})
    (retrieval / "receipt.json").write_bytes(protocol.canonical_bytes(receipt))
    return rt, frozen, retrieval


def synthetic_caller(*, fail_first=False, unknown_usage=False, malformed_first=False):
    executions = count()
    calls = []
    def call(prompt, payload, schema, output_root, *, role, requested_model):
        identity = "synthetic-native-" + str(next(executions))
        calls.append((role, deepcopy(payload), identity))
        directory = Path(output_root) / identity
        directory.mkdir(parents=True)
        if fail_first and len(calls) == 1:
            (directory / "events.jsonl").write_bytes(b"")
            (directory / "failure.json").write_bytes(protocol.canonical_bytes(dict(reason="Synthetic failure before receipt")))
            raise RuntimeError("Synthetic native failure")
        if role == "generator":
            value = {qid:dict(answerability="sufficient", claims=[dict(text=hit()["text"], citations=["fixture-p"])],
                quotes=[dict(passage_id="fixture-p", text=hit()["text"])], limitations=[]) for qid in payload}
            assert all("reference" not in case and "label" not in case for case in payload.values())
            if malformed_first and len(calls) == 1:
                for answer in value.values():
                    answer["answerability"] = "insufficient"
        elif role == "verifier":
            value = {qid:dict(supported_indices=[], reasons=["Synthetic removal."]) for qid in payload}
        else:
            assert "shared_cases" in payload and "answers" in payload
            value = {key:dict(correctness=1, relevance=2,
                claim_supported=[True]*len(case["answer"]["claims"]),
                citation_supported=[True]*sum(len(c["citations"]) for c in case["answer"]["claims"]),
                context_relevant=[True]*len(case["retrieved_ids"]), insufficiency_handling=False,
                reasons=["Synthetic judge, no accuracy evidence."]) for key, case in payload["answers"].items()}
            assert all(case["reference_evidence"] and "reference" in case for case in payload["shared_cases"].values())
        raw, framed = protocol.canonical_bytes(value), module.request_bytes(dict(prompt=prompt, payload=payload))
        usage = None if unknown_usage else dict(input_tokens=10, output_tokens=2)
        events = [dict(type="thread.started", thread_id=identity), dict(type="turn.started"),
            dict(type="item.completed", item=dict(type="agent_message", text=raw.decode())),
            dict(type="turn.completed", usage=usage)]
        for name, data in {"stdin.txt":framed, "schema.json":protocol.canonical_bytes(schema),
            "raw_final.json":raw, "stderr.txt":b"", "events.jsonl":b"\n".join(protocol.canonical_bytes(e) for e in events)}.items():
            (directory / name).write_bytes(data)
        receipt = dict(accepted=True, role=role, errors=[], process_exit_code=0, codex_cli_invocations=1,
            external_paid_api_calls=0, model_tools_disabled=True, execpolicy_rules_not_disabled=True,
            authentication="existing_ChatGPT_subscription", requested_model=requested_model or "cli_default_unresolved",
            artifact_directory=str(directory), execution_id=identity, observed_model=None, usage=usage,
            prompt_sha256=protocol.sha256(framed), schema_sha256=protocol.sha256(protocol.canonical_bytes(schema)),
            source_snapshots_sha256={p.name:module.file_hash(p) for p in module.SOURCE_PATHS},
            artifacts_sha256={p.name:module.file_hash(p) for p in directory.iterdir()}, native_cli_sha256="fixture-binary")
        (directory / "receipt.json").write_bytes(protocol.canonical_bytes(receipt))
        return value, receipt
    return call, calls


@pytest.mark.parametrize("records,unresolved", [(50,False),(999,False),(1000,True)])
def test_formal_gate_rejects_missing_accepted_test_labels_before_calls(tmp_path, records, unresolved):
    rt, frozen, retrieval = cohort_fixture(tmp_path, records, unresolved)
    with pytest.raises(ValueError, match="coverage"):
        module.experiment(rt, frozen, retrieval, rt.canonical_root / "data/machine_runs/formal-experiments",
            caller=lambda *a, **k:pytest.fail("The formal gate must precede every native call"))


def test_actual_formal_runtime_six_bindings_and_prior_pilot_implementation_rejected(tmp_path):
    rt, frozen, _ = cohort_fixture(tmp_path)
    cohort = module.load_cohort(frozen, rt)
    assert len(cohort.questions) == 1000 and set(cohort.manifest["runtime_sha256"]) == protocol.RUNTIME_KEYS
    source = json.loads(module.runtime_values(rt, cohort.cfg)["implementation"])
    assert "run_machine_experiments.py" in source and "run_codex_ablations.py" in source
    assert "wire_codec.py" in source and "prepare_answer_sample.py" in source
    (rt.stage / "index/manifest.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="differs from freeze"):
        module.load_cohort(frozen, rt)


def test_optional_wire_codec_source_bytes_are_bound_even_when_not_used(tmp_path):
    rt = runtime(tmp_path)
    sources = json.loads(module.runtime_values(rt, module.settings())["implementation"])
    codec = module.ROOT / "src/oil_evidence/wire_codec.py"
    assert sources[codec.name]["sha256"] == module.file_hash(codec)
    assert sources[codec.name]["source_utf8"] == codec.read_bytes().decode("utf-8")
    assert module.CONTRACT.endswith("-v4")


def test_greedy_bound_never_truncates_complete_evidence_and_rejects_oversized_singleton():
    cfg = module.settings(max_cases=2, max_request_bytes=8192)
    def builder(ids):
        return dict(prompt="Synthetic", role="generator", schema={}, payload={i:"x"*1000 for i in ids})
    assert module.bounded_batches(["a","b","c"], builder, cfg) == [["a","b"],["c"]]
    with pytest.raises(ValueError, match="One complete case"):
        module.bounded_batches(["a"], lambda ids:dict(prompt="Synthetic", schema={}, payload={"a":"x"*9000}), cfg)


def test_original_pool_binding_and_full_retrieval_coverage_required(tmp_path):
    rt, frozen, retrieval = cohort_fixture(tmp_path)
    cohort = module.load_cohort(frozen, rt)
    ranks, _, timing = module.load_rankings(retrieval, cohort, rt)
    assert len(ranks) == 1000 and timing["per_query_mode_embedding_or_total_latency"] is None
    receipt = module.read_json(retrieval / "receipt.json")
    path = retrieval / "rankings.jsonl"
    path.write_bytes(b"\n".join(path.read_bytes().splitlines()[1:])+b"\n")
    receipt["artifact_sha256"]["rankings.jsonl"] = module.file_hash(path)
    (retrieval / "receipt.json").write_bytes(protocol.canonical_bytes(receipt))
    with pytest.raises(ValueError, match="exactly the full frozen"):
        module.load_rankings(retrieval, cohort, rt)


@pytest.mark.parametrize("unknown_extra", [False,True])
def test_only_original_certified_extra_development_can_be_outside_frozen_cohort(tmp_path, unknown_extra):
    rt, frozen, retrieval = cohort_fixture(tmp_path)
    cohort = module.load_cohort(frozen, rt)
    source = Path(cohort.cfg["answer_sample_binding"]["plan"]["source_cohort"])
    originals = protocol.parse_jsonl((source / "questions.jsonl").read_bytes())
    extra_dev = [row for row in originals if row["split"] == "development"]
    inputs = module.read_json(retrieval / "inputs.json")
    inputs.update(cohort=str(source), protected_sha256={str(source / "questions.jsonl"):module.file_hash(source / "questions.jsonl")})
    (retrieval / "inputs.json").write_bytes(protocol.canonical_bytes(inputs))
    ranks = protocol.parse_jsonl((retrieval / "rankings.jsonl").read_bytes())
    ranks.extend(dict(record_id=extra["record_id"], conditions=ranks[0]["conditions"], ranking_seconds_excluding_embedding=.1)
                 for extra in extra_dev)
    if unknown_extra:
        ranks[-1]["record_id"] = "unknown-extra-test"
    (retrieval / "rankings.jsonl").write_bytes(b"".join(protocol.canonical_bytes(row)+b"\n" for row in ranks))
    pools = protocol.parse_jsonl((retrieval / "annotation_pool.jsonl").read_bytes())
    pools.extend(dict(record_id=extra["record_id"], question=extra["question"], as_of=extra["as_of"], evidence=pools[0]["evidence"])
                 for extra in extra_dev)
    if unknown_extra:
        pools[-1]["record_id"] = "unknown-extra-test"
    (retrieval / "annotation_pool.jsonl").write_bytes(b"".join(protocol.canonical_bytes(row)+b"\n" for row in pools))
    receipt = module.read_json(retrieval / "receipt.json")
    receipt["artifact_sha256"] = {name:module.file_hash(retrieval / name) for name in receipt["artifact_sha256"]}
    (retrieval / "receipt.json").write_bytes(protocol.canonical_bytes(receipt))
    if not unknown_extra:
        _, _, timing = module.load_rankings(retrieval, cohort, rt)
        assert timing["certified_extra_development_records_not_experiment_test"] == 200
    else:
        with pytest.raises(ValueError, match="extra DEVELOPMENT"):
            module.load_rankings(retrieval, cohort, rt)


def test_bounded_partial_resume_reuses_raw_answers_and_counts_usage_once(tmp_path):
    rt, frozen, retrieval = cohort_fixture(tmp_path)
    caller, calls = synthetic_caller()
    parent = rt.canonical_root / "data/machine_runs/formal-experiments"
    run, first = module.experiment(rt, frozen, retrieval, parent, max_new_calls=2, caller=caller)
    assert first["status"] == "partial_new_attempt_limit_reached" and len(calls) == 2
    assert first["formal_heldout_test"] is False and first["completed_answer_questions"] == 0
    assert first["completed_retrieval_questions"] == 1000
    completed = list((run / "jobs").glob("*/completed.json"))
    # Simulate a crash after native acceptance but before derived completion.
    completed[0].unlink()
    run, second = module.experiment(rt, frozen, retrieval, parent, resume=run, max_new_calls=4, caller=caller)
    assert len(calls) == 6 and second["usage"]["known_native_execution_count"] == 6
    assert second["usage"]["known_native_token_totals"] == dict(input_tokens=60, output_tokens=12)
    assert len([role for role, _, _ in calls if role == "generator"]) == 4
    assert len([role for role, _, _ in calls if role == "verifier"]) == 2
    for path in (run / "jobs").glob("hybrid_B-*/accepted-*.json"):
        checked = module.read_json(path)
        assert all(answer["claims"] == [] and answer["answerability"] == "insufficient" for answer in checked["answers"].values())
    assert not (run / "result.json").exists()
    plan = module.read_json(run / "plan.json")
    assert len(plan["all_test_record_ids"]) == 1000 and len(plan["answer_record_ids"]) == 100
    assert plan["expected_variant_answers"] == 600


def test_failure_retained_no_automatic_retry_and_unknown_execution_not_zero(tmp_path):
    rt, frozen, retrieval = cohort_fixture(tmp_path)
    caller, calls = synthetic_caller(fail_first=True)
    parent = rt.canonical_root / "data/machine_runs/formal-experiments"
    run, failed = module.experiment(rt, frozen, retrieval, parent, max_new_calls=1, caller=caller)
    assert failed["status"] == "failed_not_complete" and len(calls) == 1
    assert failed["usage"]["interrupted_or_precall_attempts_with_unknown_execution"]
    _, again = module.experiment(rt, frozen, retrieval, parent, resume=run, max_new_calls=1, caller=caller)
    assert len(calls) == 1 and "--retry-failed" in again["error"]
    _, retry = module.experiment(rt, frozen, retrieval, parent, resume=run, max_new_calls=1, caller=caller, retry_failed=True)
    assert len(calls) == 2 and retry["status"] == "partial_new_attempt_limit_reached"
    assert list((run / "jobs").glob("*/attempts/*/runner_failure.json"))


def test_unknown_usage_is_retained_as_unknown_and_not_zero(tmp_path):
    rt, frozen, retrieval = cohort_fixture(tmp_path)
    caller, _ = synthetic_caller(unknown_usage=True)
    _, result = module.experiment(rt, frozen, retrieval, rt.canonical_root / "data/machine_runs/formal-experiments",
        max_new_calls=1, caller=caller)
    assert result["usage"]["executions_with_unknown_usage"] == 1
    assert result["usage"]["known_native_token_totals"] == {}
    assert result["usage"]["subscription_dollar_cost"] is None


def test_changed_completed_native_artifact_prevents_silent_rerun(tmp_path):
    rt, frozen, retrieval = cohort_fixture(tmp_path)
    caller, calls = synthetic_caller()
    parent = rt.canonical_root / "data/machine_runs/formal-experiments"
    run, _ = module.experiment(rt, frozen, retrieval, parent, max_new_calls=1, caller=caller)
    next((run / "jobs").glob("*/attempts/*/calls/*/raw_final.json")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="audit artifact changed"):
        module.experiment(rt, frozen, retrieval, parent, resume=run, max_new_calls=1, caller=caller)
    assert len(calls) == 1


def test_mechanical_rejection_explicit_retry_keeps_original_not_quality_cherry_pick(tmp_path):
    rt, frozen, retrieval = cohort_fixture(tmp_path)
    caller, calls = synthetic_caller(malformed_first=True)
    parent = rt.canonical_root / "data/machine_runs/formal-experiments"
    run, failed = module.experiment(rt, frozen, retrieval, parent, max_new_calls=1, caller=caller)
    assert failed["status"] == "failed_not_complete" and len(calls) == 1
    rejection = next((run / "jobs").glob("*/validation_rejection-*.json"))
    assert module.read_json(rejection)["mechanical_output_rejection_not_answer_quality_selection"] is True
    _, retried = module.experiment(rt, frozen, retrieval, parent, resume=run, max_new_calls=1,
                                  caller=caller, retry_failed=True)
    assert len(calls) == 2 and retried["status"] == "partial_new_attempt_limit_reached"
    assert retried["usage"]["known_native_execution_count"] == 2
    assert retried["usage"]["known_native_token_totals"] == dict(input_tokens=20, output_tokens=4)


def test_mocked_1000_retrieval_100_answer_completion_covers_all_fixed_denominators(tmp_path):
    rt, frozen, retrieval = cohort_fixture(tmp_path)
    caller, calls = synthetic_caller()
    parent = rt.canonical_root / "data/machine_runs/formal-experiments"
    run, complete = module.experiment(rt, frozen, retrieval, parent, max_new_calls=1000, caller=caller)
    assert complete["status"] == "completed_formal_machine_only_experiment"
    assert complete["formal_heldout_test"] is True and complete["human_gold"] is False
    assert complete["completed_retrieval_questions"] == 1000
    assert complete["completed_answer_questions"] == 100 and complete["completed_variant_answers"] == 600
    assert complete["usage"]["known_native_token_totals"] == dict(input_tokens=10*len(calls), output_tokens=2*len(calls))
    assert complete["summary"]["hybrid_base"]["automatic_claim_faithfulness_denominator"] == 100
    assert complete["summary"]["hybrid_B"]["automatic_claim_faithfulness"] is None
    assert complete["summary"]["hybrid_B"]["claims_removed"] == 100
    group_rows = [row for p in (run / "groups").glob("*.json") for row in module.read_json(p)["rows"]]
    assert len({(r["record_id"],r["variant"]) for r in group_rows}) == 600
    assert all(r["per_query_generation_seconds"] is None and r["per_query_total_seconds"] is None
               and r["per_query_retrieval_seconds"] is None for r in group_rows)
    assert len([role for role, _, _ in calls if role == "generator"]) == 4*4
    retrieval_report = module.read_json(run / "retrieval_evaluation.json")
    assert retrieval_report["test_questions"] == 1000 and retrieval_report["ranked_query_conditions"] == 6000
    assert all(value["questions"] == 1000 for value in retrieval_report["summary"].values())
    before = {p:module.file_hash(p) for p in run.rglob("*") if p.is_file()}
    _, resumed = module.experiment(rt, frozen, retrieval, parent, resume=run, max_new_calls=0, caller=caller)
    assert resumed["completed_answer_questions"] == 100 and resumed["new_native_attempts_started"] == 0
    assert all(module.file_hash(p) == digest for p, digest in before.items())


def test_quality_abstention_and_support_denominators_are_explicit():
    rows = []
    for variant in module.pilot.VARIANTS:
        rows.append(dict(variant=variant, reference_answerability="insufficient", answer=dict(answerability="insufficient"),
            removed_claims=[], future_evidence_count=0, ineligible_citation_count=0,
            retrieval_metrics={metric:None for metric in module.METRICS},
            automatic_judge=dict(correctness=2, relevance=2, insufficiency_handling=True,
                                 claim_supported=[], citation_supported=[], context_relevant=[])))
    summary = module.quality_summary(rows)
    assert summary["hybrid_AB"]["categorical_correct_abstention_rate"] == 1
    assert summary["hybrid_AB"]["categorical_false_abstention_rate"] is None
    assert summary["hybrid_AB"]["automatic_claim_faithfulness"] is None
    assert summary["hybrid_AB"]["automatic_claim_faithfulness_denominator"] == 0


def test_judge_shared_source_pool_is_complete_and_not_repeated_per_variant():
    cases = {"fixture-q":dict(question="Synthetic output?", as_of="2024-01-01")}
    original = hit()
    reference = dict(label={}, reference_evidence=[original])
    raw_answer = dict(answerability="insufficient", claims=[], quotes=[], limitations=["Synthetic gap."])
    rows = {variant+"__fixture-q":dict(record_id="fixture-q", answer=raw_answer, evidence=[original])
            for variant in module.pilot.VARIANTS}
    request, _ = module.capture_request("judge", cases, rows=rows, references={"fixture-q":reference})
    assert len(request["payload"]["shared_cases"]) == 1
    assert len(request["payload"]["answers"]) == 6
    assert request["payload"]["shared_cases"]["fixture-q"]["reference_evidence"] == [original]
    assert request["prompt"] == module.FORMAL_JUDGE_PROMPT
    assert all(entry["retrieved_ids"] == ["fixture-p"] and "reference" not in entry
               for entry in request["payload"]["answers"].values())
