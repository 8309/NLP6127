"""Synthetic machine-only protocol tests, not annotation accuracy or live runs."""
from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

from oil_evidence import machine_evaluation as module


def jsonl(rows):
    return b"".join(module.canonical_bytes(row) + b"\n" for row in rows)


def envelope(packet, side, label=None):
    label = deepcopy(label or dict(
        answerability="sufficient", event_status="observed",
        qrels={"p-current": 2, "p-background": 1, "p-future": 0, "p-undated": 0},
        answer_points=[dict(text="Synthetic crude exports were stable.", citations=["p-current"])],
        quotes=[dict(passage_id="p-current", text="crude exports were stable")],
        reasons=["An eligible synthetic source states the answer."]))
    raw = module.canonical_bytes(label).decode("utf-8")
    return dict(record_id=packet["payload"]["record_id"], input_sha256=packet["input_sha256"],
                prompt_sha256=packet["prompt_sha256"], execution_input_sha256=packet["input_sha256"],
                provenance="machine_only", human_gold=False,
                independent_context=True, prior_outputs_seen=False,
                identity=dict(service=f"synthetic-service-{side}", requested_model=f"synthetic-request-{side}",
                              observed_model=f"synthetic-observed-{side}", execution_id=f"synthetic-execution-{side}",
                              observation_source="execution_metadata"),
                context_id=f"synthetic-context-{side}-{packet['payload']['record_id']}",
                decoding={"temperature": 0}, usage=None, raw_output=raw,
                raw_output_sha256=module.sha256(raw.encode("utf-8")), label=label)


def relabel(output, label):
    output["label"] = label
    output["raw_output"] = module.canonical_bytes(label).decode("utf-8")
    output["raw_output_sha256"] = module.sha256(output["raw_output"].encode("utf-8"))


@pytest.fixture
def scenario():
    corpus = [dict(passage_id="p-current", document_id="d-current",
                   text="Synthetic evidence: crude exports were stable at 2 million barrels per day.",
                   available_by="2024-01-01", source="SYNTHETIC", page=1),
              dict(passage_id="p-background", document_id="d-background",
                   text="Synthetic background discusses crude oil transport.",
                   available_by="2023-11-30", source="SYNTHETIC", page=2),
              dict(passage_id="p-future", document_id="d-future",
                   text="Synthetic future evidence predicts higher exports.",
                   available_by="2025-01-01", source="SYNTHETIC", page=1),
              dict(passage_id="p-undated", document_id="d-undated",
                   text="Synthetic undated evidence discusses exports.",
                   available_by=None, source="SYNTHETIC", page=1)]
    questions = [dict(record_id=f"q-{i}", question=f"What crude export development does synthetic oil report {i} establish?",
                      as_of="2024-06-30", event_group=f"synthetic-event-{i}",
                      split="development" if i < 3 else "test") for i in range(1, 5)]
    pools = [dict(record_id=q["record_id"], evidence_ids=[p["passage_id"] for p in corpus]) for q in questions]
    prompt = module.ANNOTATION_PROMPT
    inputs = module.prepare_inputs(questions, pools, corpus, prompt)
    left, right = [], []
    for packet in inputs.values():
        left.append(envelope(packet, "a"))
        right.append(envelope(packet, "b"))
    decisions = [dict(record_id=a["record_id"], status="accepted", provenance="machine_adjudicated",
                      human_gold=False, label=deepcopy(a["label"]),
                      adjudication=dict(reason="Synthetic adjudication checked exact source support, units, and cutoff.",
                                        both_originals_seen=True,
                                        original_output_sha256=[module.sha256(module.canonical_bytes(x)) for x in (a, b)]))
                 for a, b in zip(left, right, strict=True)]
    artifacts = dict(corpus=jsonl(corpus), questions=jsonl(questions), evidence_pool=jsonl(pools),
                     annotation_prompt=prompt.encode("utf-8"), independent_a=jsonl(left),
                     independent_b=jsonl(right), adjudications=jsonl(decisions))
    runtime = {name: module.sha256(name.encode("utf-8")) for name in module.RUNTIME_KEYS}
    return dict(corpus=corpus, questions=questions, pools=pools, inputs=inputs,
                left=left, right=right, decisions=decisions, artifacts=artifacts, runtime=runtime)


def test_module_is_pure_and_has_no_global_corpus_or_model_client():
    assert not hasattr(module, "DATA")
    assert not hasattr(module, "ModelClient")
    assert not hasattr(module, "write_json")


def test_clean_question_excludes_authoring_answers_and_design_intentions(scenario):
    question = dict(scenario["questions"][0], question_zh="Synthetic translated question",
                    anchors=["private author anchor"], answer_points=["private answer"],
                    type="insufficient", topic="author-selected hint")
    assert module.clean_question(question) == scenario["questions"][0]
    question["question_id"] = question.pop("record_id")
    assert module.clean_question(question) == scenario["questions"][0]


def test_input_hash_binds_prompt_question_dates_and_evidence_order(scenario):
    original = scenario["inputs"]["q-1"]
    assert set(original["payload"]) == set(scenario["questions"][0]) - {"event_group", "split"} | {"evidence"}
    assert original["metadata"] == {"event_group": "synthetic-event-1", "split": "development"}
    changed_questions = deepcopy(scenario["questions"])
    changed_questions[0]["as_of"] = "2024-06-29"
    changed = module.prepare_inputs(changed_questions, scenario["pools"], scenario["corpus"])["q-1"]
    assert original["input_sha256"] != changed["input_sha256"]
    changed_pools = deepcopy(scenario["pools"])
    changed_pools[0]["evidence_ids"].reverse()
    changed = module.prepare_inputs(scenario["questions"], changed_pools, scenario["corpus"])["q-1"]
    assert original["input_sha256"] != changed["input_sha256"]
    changed = module.prepare_inputs(scenario["questions"], scenario["pools"], scenario["corpus"], "Different protocol")["q-1"]
    assert original["prompt_sha256"] != changed["prompt_sha256"]


def test_clean_packet_matches_freeze_input_without_author_group_split_or_rank_hints(scenario):
    row = dict(scenario["questions"][0], evidence=deepcopy(scenario["corpus"]),
               answer_points=["private hint"], type="insufficient", question_zh="alternate language")
    row["question_id"] = row.pop("record_id")
    row["evidence"][0]["score"] = 100
    row["evidence"][0]["rank"] = 1
    packet = module.clean_packet(row)
    expected = {key: value for key, value in scenario["inputs"]["q-1"].items() if key != "metadata"}
    assert packet == expected
    assert "event_group" not in packet["payload"]
    assert "split" not in packet["payload"]
    assert "score" not in packet["payload"]["evidence"][0]


@pytest.mark.parametrize("key", ["answerability", "answer_points", "qrels", "anchors", "type", "question_zh", "unknown"])
def test_strict_inputs_reject_labels_and_author_hints(scenario, key):
    scenario["questions"][0][key] = "not allowed"
    with pytest.raises(ValueError, match="hints|unknown"):
        module.prepare_inputs(scenario["questions"], scenario["pools"], scenario["corpus"])


@pytest.mark.parametrize("case", ["question_duplicate", "id_duplicate", "split_overlap", "bad_date", "bad_split",
                                  "pool_missing", "pool_duplicate", "pool_unknown", "rank_hint", "bad_prompt"])
def test_input_integrity_fails_closed(scenario, case):
    prompt = module.ANNOTATION_PROMPT
    if case == "question_duplicate":
        scenario["questions"][1]["question"] = scenario["questions"][0]["question"].upper() + " !!!"
    elif case == "id_duplicate":
        scenario["questions"][1]["record_id"] = "q-1"
    elif case == "split_overlap":
        scenario["questions"][2]["event_group"] = scenario["questions"][0]["event_group"]
    elif case == "bad_date":
        scenario["questions"][0]["as_of"] = "2024-6-30"
    elif case == "bad_split":
        scenario["questions"][0]["split"] = "training"
    elif case == "pool_missing":
        scenario["pools"].pop()
    elif case == "pool_duplicate":
        scenario["pools"][0]["evidence_ids"].append("p-current")
    elif case == "pool_unknown":
        scenario["pools"][0]["evidence_ids"].append("missing")
    elif case == "rank_hint":
        scenario["pools"][0]["author_answer"] = "hint"
    else:
        prompt = ""
    with pytest.raises(ValueError):
        module.prepare_inputs(scenario["questions"], scenario["pools"], scenario["corpus"], prompt)


def test_exact_cutoff_supported_label_valid(scenario):
    packet = scenario["inputs"]["q-1"]
    assert module.validate_independent_output(scenario["left"][0], packet) == []


def test_insufficient_may_quote_eligible_background_without_answer_points(scenario):
    label = deepcopy(scenario["left"][0]["label"])
    label.update(answerability="insufficient", answer_points=[], reasons=[
        "The source discusses transport, not the requested measured export loss."])
    label["quotes"] = [dict(passage_id="p-background", text="discusses crude oil transport")]
    assert module.validate_label(label, scenario["inputs"]["q-1"]) == []


@pytest.mark.parametrize("case", ["future", "undated", "fabricated", "irrelevant", "invented_answer"])
def test_insufficiency_background_quotes_keep_all_support_guards(scenario, case):
    label = deepcopy(scenario["left"][0]["label"])
    label.update(answerability="insufficient", answer_points=[], reasons=["Requested physical effect is not established."])
    if case in {"future", "undated"}:
        pid = "p-" + case
        by_id = {row["passage_id"]: row for row in scenario["corpus"]}
        label["quotes"] = [dict(passage_id=pid, text=by_id[pid]["text"])]
        label["qrels"][pid] = 1
    elif case == "fabricated":
        label["quotes"][0]["text"] = "A nonexistent measured loss"
    elif case == "irrelevant":
        label["qrels"]["p-current"] = 0
    else:
        label["answer_points"] = [dict(text="An invented export reduction.", citations=["p-current"])]
    assert module.validate_label(label, scenario["inputs"]["q-1"])


@pytest.mark.parametrize("case", ["future_qrel", "undated_qrel", "boolean_qrel", "missing_qrel", "wrong_quote",
                                  "quote_without_positive_qrel", "unsupported_point", "no_points", "insufficient_with_points",
                                  "invalid_answerability", "invalid_event", "no_reason", "extra_field"])
def test_label_validation_preserves_time_units_and_missing_judgments(scenario, case):
    label = deepcopy(scenario["left"][0]["label"])
    if case == "future_qrel":
        label["qrels"]["p-future"] = 2
    elif case == "undated_qrel":
        label["qrels"]["p-undated"] = 1
    elif case == "boolean_qrel":
        label["qrels"]["p-current"] = True
    elif case == "missing_qrel":
        del label["qrels"]["p-background"]
    elif case == "wrong_quote":
        label["quotes"][0]["text"] = "not an exact source quote"
    elif case == "quote_without_positive_qrel":
        label["qrels"]["p-current"] = 0
    elif case == "unsupported_point":
        label["answer_points"][0]["citations"] = ["p-background"]
    elif case == "no_points":
        label["answer_points"] = []
    elif case == "insufficient_with_points":
        label["answerability"] = "insufficient"
    elif case == "invalid_answerability":
        label["answerability"] = "maybe"
    elif case == "invalid_event":
        label["event_status"] = "unlabelled"
    elif case == "no_reason":
        label["reasons"] = []
    else:
        label["human_reviewed"] = True
    assert module.validate_label(label, scenario["inputs"]["q-1"])


@pytest.mark.parametrize("field,value", [("provenance", "human_independent"), ("human_gold", True),
                                         ("independent_context", False), ("prior_outputs_seen", True),
                                         ("record_id", "other-question"), ("input_sha256", "wrong"),
                                         ("prompt_sha256", "wrong"), ("context_id", ""), ("decoding", None),
                                         ("usage", "unknown"), ("raw_output", ""), ("raw_output_sha256", "wrong")])
def test_independent_envelope_provenance_is_not_rewritten(scenario, field, value):
    output = deepcopy(scenario["left"][0])
    output[field] = value
    assert module.validate_independent_output(output, scenario["inputs"]["q-1"])
    assert output[field] == value


def test_observed_identity_cannot_be_substituted_for_requested_alias(scenario):
    output = deepcopy(scenario["left"][0])
    output["identity"]["observed_model"] = None
    assert any("explicit metadata-unavailable note" in error for error in module.validate_independent_output(output, scenario["inputs"]["q-1"]))


def test_unavailable_observed_identity_is_explicit_null_not_requested_alias(scenario):
    output = deepcopy(scenario["left"][0])
    output["identity"].update(observed_model=None, observation_source="unavailable",
                              identity_note="Synthetic runner metadata exposes no model ID.")
    assert module.validate_independent_output(output, scenario["inputs"]["q-1"]) == []
    assert output["identity"]["observed_model"] is None


def test_model_self_report_is_not_execution_metadata(scenario):
    output = deepcopy(scenario["left"][0])
    output["identity"]["observation_source"] = "assistant_self_report"
    assert any("self-report" in error for error in module.validate_independent_output(output, scenario["inputs"]["q-1"]))


def test_wrapper_retains_unmodified_raw_output_and_mechanical_errors(scenario):
    original = scenario["left"][0]
    wrapped = module.wrap_independent_output(
        scenario["inputs"]["q-1"], original["raw_output"], original["identity"],
        original["context_id"], original["decoding"],
        execution_input_sha256=original["execution_input_sha256"])
    assert wrapped["raw_output"] == original["raw_output"]
    assert wrapped["label"] == original["label"]
    assert wrapped["validation_errors"] == []
    invalid = module.wrap_independent_output(
        scenario["inputs"]["q-1"], "not JSON", original["identity"], original["context_id"], {},
        execution_input_sha256=original["execution_input_sha256"])
    assert invalid["raw_output"] == "not JSON"
    assert invalid["label"] is None
    assert invalid["validation_errors"]


def test_batch_wrapper_preserves_entire_raw_response_and_explicit_extraction_path(scenario):
    original = scenario["left"][0]
    raw = json.dumps({"records": [{"record_id": "q-1", "label": original["label"]},
                                  {"record_id": "q-2", "label": scenario["left"][1]["label"]}]}, indent=3)
    wrapped = module.wrap_independent_output(
        scenario["inputs"]["q-1"], raw, original["identity"], original["context_id"], {},
        execution_input_sha256=module.sha256(b"synthetic full two-question execution input"),
        raw_label_path=["records", 0, "label"])
    assert wrapped["raw_output"] == raw
    assert wrapped["raw_output_sha256"] == module.sha256(raw.encode("utf-8"))
    assert wrapped["label"] == original["label"]
    assert wrapped["validation_errors"] == []


def test_batch_cannot_swap_two_question_labels_even_if_label_text_matches(scenario):
    original = scenario["left"][0]
    raw = json.dumps({"records": [{"record_id": "q-1", "label": original["label"]},
                                  {"record_id": "q-2", "label": scenario["left"][1]["label"]}]})
    wrapped = module.wrap_independent_output(
        scenario["inputs"]["q-1"], raw, original["identity"], original["context_id"], {},
        execution_input_sha256=module.sha256(b"synthetic full two-question execution input"),
        raw_label_path=["records", 1, "label"])
    assert any("record ID" in error for error in wrapped["validation_errors"])


@pytest.mark.parametrize("path", [["records", True], ["records", -1], ["missing"], "not-list"])
def test_bad_raw_extraction_path_is_retained_as_invalid(scenario, path):
    original = scenario["left"][0]
    wrapped = module.wrap_independent_output(
        scenario["inputs"]["q-1"], original["raw_output"], original["identity"], original["context_id"], {},
        execution_input_sha256=original["execution_input_sha256"], raw_label_path=path)
    assert wrapped["validation_errors"]


def test_different_full_execution_inputs_do_not_count_as_identical_blinded_conditions(scenario):
    scenario["right"][0]["execution_input_sha256"] = module.sha256(b"different surrounding batch inputs")
    result = module.raw_agreement(scenario["inputs"], scenario["left"], scenario["right"])
    assert result["paired_records"] == 3
    assert result["conflicting_context_records"] == ["q-1"]


def test_retained_raw_output_and_parsed_label_must_match(scenario):
    output = deepcopy(scenario["left"][0])
    output["label"]["event_status"] = "uncertain"
    assert any("differs" in error for error in module.validate_independent_output(output, scenario["inputs"]["q-1"]))


def test_malformed_unhashable_quote_id_is_reported_as_invalid_not_crashed(scenario):
    output = deepcopy(scenario["left"][0])
    label = deepcopy(output["label"])
    label["quotes"][0]["passage_id"] = ["p-current"]
    relabel(output, label)
    assert module.validate_independent_output(output, scenario["inputs"]["q-1"])


def test_same_model_contexts_report_agreement_without_claiming_model_diversity(scenario):
    for output in scenario["right"]:
        output["identity"]["service"] = scenario["left"][0]["identity"]["service"]
        output["identity"]["observed_model"] = scenario["left"][0]["identity"]["observed_model"]
    result = module.raw_agreement(scenario["inputs"], scenario["left"], scenario["right"])
    assert result["paired_records"] == 4
    assert result["same_observed_model_record_ids"] == ["q-1", "q-2", "q-3", "q-4"]


def test_agreement_is_raw_machine_agreement_not_human_or_accuracy(scenario):
    agreement = module.raw_agreement(scenario["inputs"], scenario["left"], scenario["right"])
    assert agreement["paired_records"] == 4
    assert agreement["fields"]["relevance"]["judgment_pairs"] == 16
    assert agreement["fields"]["relevance"]["raw_agreement"] == 1
    assert agreement["fields"]["relevance"]["quadratic_weighted_kappa"] == 1
    assert agreement["fields"]["answerability"]["cohens_kappa"] is None
    assert agreement["human_agreement"] is None
    assert agreement["free_text_answer_agreement"] is None


def test_missing_invalid_and_unjudged_outputs_are_reported_not_negative(scenario):
    scenario["right"].pop()
    scenario["right"][0]["input_sha256"] = "wrong"
    changed = deepcopy(scenario["right"][1]["label"])
    del changed["qrels"]["p-background"]
    relabel(scenario["right"][1], changed)
    agreement = module.raw_agreement(scenario["inputs"], scenario["left"], scenario["right"])
    assert agreement["missing_b"] == ["q-4"]
    assert set(agreement["invalid_outputs"]["b"]) == {"q-1"}
    assert agreement["paired_records"] == 2
    assert agreement["fields"]["relevance"]["judgment_pairs"] == 7
    assert agreement["pooled_passages_without_paired_judgment"] == 1
    assert agreement["fields"]["relevance"]["raw_agreement"] == 1


def test_agreement_changes_without_adjudication_contamination(scenario):
    changed = deepcopy(scenario["right"][0]["label"])
    changed["qrels"]["p-background"] = 0
    changed["event_status"] = "uncertain"
    relabel(scenario["right"][0], changed)
    agreement = module.raw_agreement(scenario["inputs"], scenario["left"], scenario["right"])
    assert agreement["fields"]["event_status"]["raw_agreement"] == .75
    assert agreement["fields"]["relevance"]["raw_agreement"] == 15 / 16
    assert agreement["fields"]["relevance"]["quadratic_weighted_kappa"] < 1


def test_shared_context_does_not_count_as_independent_pair(scenario):
    scenario["right"][0]["context_id"] = scenario["left"][0]["context_id"]
    agreement = module.raw_agreement(scenario["inputs"], scenario["left"], scenario["right"])
    assert agreement["paired_records"] == 3
    assert agreement["conflicting_context_records"] == ["q-1"]


def test_empty_agreement_is_undefined_not_zero():
    result = module.raw_agreement({}, [], [])
    assert result["fields"]["relevance"]["raw_agreement"] is None
    assert result["fields"]["answerability"]["cohens_kappa"] is None


def test_manifest_binds_all_original_artifacts_and_does_not_mutate_them(scenario):
    originals = deepcopy(scenario["artifacts"])
    manifest = module.build_manifest(scenario["artifacts"], scenario["runtime"])
    assert manifest["provenance"] == "frozen_machine_only"
    assert manifest["human_gold"] is False
    assert manifest["split_status_counts"] == {"development": {"accepted": 2}, "test": {"accepted": 2}}
    assert set(manifest["artifact_sha256"]) == module.ARTIFACT_KEYS
    assert manifest["agreement"]["paired_records"] == 4
    assert scenario["artifacts"] == originals
    assert module.verify_manifest(manifest, originals, scenario["runtime"]) == manifest


@pytest.mark.parametrize("artifact", sorted(module.ARTIFACT_KEYS))
def test_changed_bytes_cannot_rebind_old_frozen_version(scenario, artifact):
    manifest = module.build_manifest(scenario["artifacts"], scenario["runtime"])
    scenario["artifacts"][artifact] += b"\n"
    with pytest.raises(ValueError, match="changed|mismatch"):
        module.verify_manifest(manifest, scenario["artifacts"], scenario["runtime"])


def test_machine_freeze_preserves_missing_stream_guard_and_classifies_same_model_repeat(scenario):
    scenario["artifacts"]["independent_b"] = jsonl(scenario["right"][:-1])
    with pytest.raises(ValueError, match="coverage"):
        module.build_manifest(scenario["artifacts"])
    for output in scenario["right"]:
        output["identity"]["service"] = scenario["left"][0]["identity"]["service"]
        output["identity"]["requested_model"] = scenario["left"][0]["identity"]["requested_model"]
        output["identity"]["observed_model"] = scenario["left"][0]["identity"]["observed_model"]
    scenario["artifacts"]["independent_b"] = jsonl(scenario["right"])
    for index, (a, b) in enumerate(zip(scenario["left"], scenario["right"], strict=True)):
        scenario["decisions"][index]["adjudication"]["original_output_sha256"] = [
            module.sha256(module.canonical_bytes(output)) for output in (a, b)]
    scenario["artifacts"]["adjudications"] = jsonl(scenario["decisions"])
    manifest = module.build_manifest(scenario["artifacts"])
    assert manifest["annotation_protocol"] == "same_model_repeat"
    assert manifest["model_diversity_established"] is False


def test_same_requested_model_with_unavailable_observation_never_forges_observed_identity(scenario):
    for stream in (scenario["left"], scenario["right"]):
        for output in stream:
            output["identity"].update(service="synthetic-codex", requested_model="synthetic-same-request",
                                      observed_model=None, observation_source="unavailable",
                                      identity_note="Synthetic runner does not emit selected model metadata.")
    for index, (a, b) in enumerate(zip(scenario["left"], scenario["right"], strict=True)):
        scenario["decisions"][index]["adjudication"]["original_output_sha256"] = [
            module.sha256(module.canonical_bytes(output)) for output in (a, b)]
    scenario["artifacts"].update(independent_a=jsonl(scenario["left"]), independent_b=jsonl(scenario["right"]),
                                 adjudications=jsonl(scenario["decisions"]))
    manifest = module.build_manifest(scenario["artifacts"])
    assert manifest["annotation_protocol"] == "same_requested_model_unverified"
    assert manifest["observed_identity_available_for_both"] is False
    assert manifest["model_diversity_established"] is False
    assert all(identity["model"] is None for identity in manifest["observed_annotator_identities"])
    assert manifest["agreement"]["observed_identity_unavailable_record_ids"] == ["q-1", "q-2", "q-3", "q-4"]


@pytest.mark.parametrize("case", ["missing_decision", "forged_human", "original_not_bound", "no_reason", "no_status"])
def test_adjudication_cannot_hide_history_or_provenance(scenario, case):
    decision = scenario["decisions"][0]
    if case == "missing_decision":
        scenario["decisions"].pop()
    elif case == "forged_human":
        decision["human_gold"] = True
    elif case == "original_not_bound":
        decision["adjudication"]["original_output_sha256"][0] = "wrong"
    elif case == "no_reason":
        decision["adjudication"]["reason"] = ""
    else:
        decision.pop("status")
    scenario["artifacts"]["adjudications"] = jsonl(scenario["decisions"])
    with pytest.raises(ValueError):
        module.build_manifest(scenario["artifacts"])


def test_unresolved_records_are_frozen_but_not_accepted_test_coverage(scenario):
    decision = scenario["decisions"][-1]
    decision["status"] = "unresolved"
    decision["unresolved_reasons"] = ["Synthetic source revision has unknown historical availability."]
    decision.pop("label")
    scenario["artifacts"]["adjudications"] = jsonl(scenario["decisions"])
    manifest = module.build_manifest(scenario["artifacts"], scenario["runtime"])
    assert manifest["split_status_counts"]["test"] == {"accepted": 1, "unresolved": 1}
    with pytest.raises(ValueError, match="unresolved"):
        module.preflight_experiments(manifest, required_test_records=1)


def test_experiment_preflight_requires_coverage_and_bound_runtime(scenario):
    manifest = module.build_manifest(scenario["artifacts"], scenario["runtime"])
    with pytest.raises(ValueError, match="coverage"):
        module.preflight_experiments(manifest)
    result = module.preflight_experiments(manifest, required_test_records=2)
    assert result["total_query_traces"] == 12
    assert result["calls_by_role"]["generation"] == 12
    unbound = module.build_manifest(scenario["artifacts"])
    with pytest.raises(ValueError, match="Bind index"):
        module.preflight_experiments(unbound, required_test_records=2)


@pytest.mark.parametrize("case", ["human_manifest", "wrong_version", "incomplete_pair"])
def test_experiment_preflight_never_consumes_human_or_unpaired_manifests(scenario, case):
    manifest = module.build_manifest(scenario["artifacts"], scenario["runtime"])
    if case == "human_manifest":
        manifest["provenance"] = "human_independent"
    elif case == "wrong_version":
        manifest["schema_version"] = "unknown"
    else:
        manifest["agreement"]["paired_records"] = 3
    with pytest.raises(ValueError):
        module.preflight_experiments(manifest, required_test_records=2)


def test_preflight_rejects_changed_manifest_counts_without_rebinding(scenario):
    manifest = module.build_manifest(scenario["artifacts"], scenario["runtime"])
    manifest["split_status_counts"]["test"]["accepted"] = 1000
    with pytest.raises(ValueError, match="content changed"):
        module.preflight_experiments(manifest)


def test_thousand_record_call_plan_separates_generation_verification_and_judging():
    plan = module.experiment_plan()
    assert plan["total_query_traces"] == 6000
    assert plan["calls_by_role"] == dict(generation=6000, support_check_max=2000, automatic_judge=6000,
                                         independent_a=1000, independent_b=1000, adjudication=1000)
    assert plan["maximum_model_calls"] == 17000
    assert [(v["temporal"], v["verify"]) for v in plan["variants"][2:]] == [
        (False, False), (True, False), (False, True), (True, True)]
    assert module.experiment_plan(include_annotation=False)["maximum_model_calls"] == 14000
    assert module.experiment_plan(automatic_judge=False)["calls_by_role"]["automatic_judge"] == 0


@pytest.mark.parametrize("count", [0, -1, True, 1.5, None])
def test_invalid_plan_size_rejected(count):
    with pytest.raises(ValueError):
        module.experiment_plan(count)


def profile(**overrides):
    return dict(role="synthetic-generation", calls=10, input_bytes=4096, max_output_tokens=1500,
                input_usd_per_million=.30, output_usd_per_million=1.20) | overrides


def test_budget_keeps_existing_spend_and_uses_byte_ceiling_not_token_guess():
    result = module.budget_preflight([profile()], cumulative_spent_usd=4.9)
    assert result["planned_reservation_usd"] == pytest.approx(.03336)
    assert result["remaining_usd"] == pytest.approx(.1)
    assert result["fits_existing_cap"] is True
    result = module.budget_preflight([profile(calls=1000)], cumulative_spent_usd=4.9)
    assert result["planned_reservation_usd"] == pytest.approx(3.336)
    assert result["fits_existing_cap"] is False
    assert result["subscription_consumption"] == "unmeasured_separate_from_api_budget"


@pytest.mark.parametrize("field,value", [("calls", True), ("calls", 0), ("input_bytes", -1),
                                         ("max_output_tokens", 1.5), ("input_usd_per_million", 0),
                                         ("output_usd_per_million", math.nan), ("role", "")])
def test_budget_rejects_nonpositive_rates_and_invalid_bounds(field, value):
    with pytest.raises(ValueError):
        module.budget_preflight([profile(**{field: value})], 0)


@pytest.mark.parametrize("spent,cap", [(-1, 5), (math.inf, 5), (0, 0), (True, 5), (0, math.nan)])
def test_budget_rejects_invalid_prior_spend_and_cap(spent, cap):
    with pytest.raises(ValueError):
        module.budget_preflight([profile()], spent, cap)


@pytest.mark.parametrize("raw", [b"", b"[]\n", b'{"q":1,"q":2}\n', b'{"q":NaN}\n'])
def test_original_artifacts_reject_empty_nonobjects_duplicate_keys_and_nan(raw):
    with pytest.raises(ValueError):
        module.parse_jsonl(raw)


def test_hashes_require_raw_bytes_not_reserialized_text():
    with pytest.raises(TypeError):
        module.sha256("text")
    with pytest.raises(TypeError):
        module.parse_jsonl("{}")
