"""Immutable, resumable six-variant experiments on a fully frozen machine cohort.

The formal gate always requires >=1000 accepted TEST records and no unresolved
TEST records. Question selection alone is not an annotation freeze. This runner
uses complete precomputed rankings, never author anchors or reference answers in
generation. B/AB check the exact saved base/A answers, not regenerated answers.
No hosted API, credential read, automatic retry, or fabricated query latency.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4

from jsonschema import Draft202012Validator

import run_codex_ablations as pilot
import prepare_answer_sample as sampling
from report_machine_retrieval import NAMES
from oil_evidence import machine_evaluation as protocol
from oil_evidence.codex_runtime import (
    ISOLATION_PROMPT, _assert_hashes, _load_isolation_snapshot, _validate_hits,
    call_structured_json, load_staged_runtime,
)
from oil_evidence.machine_metrics import METRICS, paired_bootstrap, ranked_metrics
from oil_evidence.store import ROOT, now

FORMAL_MINIMUM = 1000
CONTRACT = "frozen-machine-1000-retrieval-100-answer-v4"
FORMAL_JUDGE_PROMPT = pilot.JUDGE_PROMPT + """
The payload contains shared_cases keyed by question ID and answers keyed by
variant/question entry. Each shared case provides the complete fixed unranked
reference_evidence pool, question, cutoff and machine reference. Each answer
entry identifies its own record_id and ordered retrieved_ids within that pool.
Use original source bodies to check both the answer AND reference; exact quotes
alone do not prove semantic correctness. Never use another question's pool.
For context_relevant, follow that answer's retrieved_ids order. For claim and
citation support, follow its claims and citation order. Output the flat entry
keys requested by the schema, not the shared case keys. Do not assume a reference
is correct; explain conflicts with the source bodies in reasons.
"""
DEFINITIONS = (("bm25", "bm25:0", False), ("dense", "dense:0", False),
               ("hybrid_base", "hybrid:0", False), ("hybrid_A", "hybrid:1", True))
CONDITIONS = {f"{mode}:{temporal}" for mode in ("bm25", "dense", "hybrid") for temporal in (0, 1)}
SOURCE_PATHS = [Path(__file__), Path(pilot.__file__), ROOT / "src/oil_evidence/codex_runtime.py",
                ROOT / "src/oil_evidence/machine_evaluation.py", ROOT / "src/oil_evidence/machine_metrics.py",
                ROOT / "src/oil_evidence/retrieval.py", ROOT / "scripts/run_codex_annotation.py",
                ROOT / "scripts/retrieve_machine_questions.py", Path(sampling.__file__),
                ROOT / "src/oil_evidence/wire_codec.py"]


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=protocol._unique_object,
                      parse_constant=protocol._invalid_constant)


def write_new(path, value):
    pilot.write_new(path, value)


def file_hash(path):
    return protocol.sha256(Path(path).read_bytes())


def contained(base, relative):
    return pilot.inside(Path(base).resolve(), relative)


def plain_path(path):
    value = str(path)
    if os.name == "nt" and value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value).resolve()


def long_path(path):
    """Use native extended paths for deeply nested immutable call artifacts."""
    value = str(plain_path(path))
    return Path("\\\\?\\" + value) if os.name == "nt" else Path(value)


def validate_sample_plan(plan):
    if (not isinstance(plan, dict) or plan.get("schema_version") != sampling.VERSION
            or plan.get("original_test_count") != 1000 or plan.get("answer_sample_count") != 100
            or plan.get("sample_seed") != 6127 or plan.get("retrieval_evaluation_count") != 1000
            or plan.get("answer_variant_results") != 600 or plan.get("labels_answers_ranks_scores_used") is not False
            or plan.get("selected_before_test_labels") is not True
            or plan.get("selected_before_test_native_attempts") is not True):
        raise ValueError("Bind the immutable pre-label uniform100-of1000 TEST sample")
    draw, ordered = sampling.sample_ids(plan.get("original_ordered_test_record_ids", []))
    if draw != plan.get("draw_order_record_ids") or ordered != plan.get("answer_record_ids"):
        raise ValueError("Sample IDs/order do not reproduce the fixed uniform seed6127 draw")


def load_sample_binding(directory, root):
    directory, root = Path(directory).resolve(), Path(root).resolve()
    allowed = root / "data/machine_runs/answer-samples"
    if not directory.is_relative_to(allowed) or directory == allowed:
        raise ValueError("Use a sealed canonical answer-sample run")
    receipt = read_json(contained(directory, "receipt.json"))
    if (receipt.get("status") != "immutable_uniform100_TEST_answer_sample_sealed_before_TEST_labels"
            or receipt.get("original_inputs_unchanged") is not True or receipt.get("selected_questions") != 100
            or receipt.get("population_test_questions") != 1000 or receipt.get("seed") != 6127
            or receipt.get("labels_created") != 0 or receipt.get("scores_created") != 0
            or receipt.get("implementation_sha256") != file_hash(Path(sampling.__file__))):
        raise ValueError("Answer sample receipt or its original preparer implementation differs")
    protected = {str(directory / "receipt.json"):file_hash(directory / "receipt.json")}
    for original_name, expected in receipt.get("original_input_sha256", {}).items():
        original = Path(original_name).resolve()
        if not original.is_relative_to(root / "data") or not original.is_file():
            raise ValueError("Original sampling binding escapes the canonical data")
        protected[str(original)] = expected
    _assert_hashes(protected)
    for name, expected in receipt["artifact_sha256"].items():
        path = contained(directory, name)
        if file_hash(path) != expected:
            raise ValueError("Sealed pre-label sampling artifact changed")
        protected[str(path)] = expected
    for name in ("sample_plan.json","sample_questions.jsonl","pre_label_status.json","pre_label_progress.jsonl"):
        if str(directory / name) not in protected:
            raise ValueError("Sample is missing its original questions or pre-label proof")
    plan = read_json(directory / "sample_plan.json")
    validate_sample_plan(plan)
    if receipt["sample_plan_sha256"] != file_hash(directory / "sample_plan.json"):
        raise ValueError("Original sample-plan hash changed")
    proof = read_json(directory / "pre_label_status.json")
    if (proof.get("accepted_test_labels_by_stream") != {"a":0,"b":0}
            or proof.get("test_native_attempts_started") != 0
            or file_hash(directory / "pre_label_status.json") != plan["pre_label_audit_sha256"]
            or file_hash(directory / "pre_label_progress.jsonl") != plan["pre_label_progress_prefix_sha256"]
            or proof["questions_sha256"] != plan["source_questions_sha256"]
            or proof["test_questions_sha256"] != plan["source_test_questions_sha256"]):
        raise ValueError("Original zero-TEST audit does not bind the chosen sample")
    selected = protocol.parse_jsonl((directory / "sample_questions.jsonl").read_bytes())
    if [row["record_id"] for row in selected] != plan["answer_record_ids"] or any(row.get("split") != "test" for row in selected):
        raise ValueError("Sealed100 sampled questions/order differ")
    return dict(directory=str(directory), sample_plan_sha256=file_hash(directory / "sample_plan.json"),
        receipt_sha256=file_hash(directory / "receipt.json"), plan=plan), protected


def settings(requested_model=None, max_cases=8, max_request_bytes=160000, sample_binding=None):
    if requested_model is not None and (not isinstance(requested_model, str) or not requested_model.strip()):
        raise ValueError("Requested model must be a nonempty name or explicit CLI-default null")
    if type(max_cases) is not int or not 1 <= max_cases <= 32:
        raise ValueError("Batch size must be between one and 32 complete cases")
    if type(max_request_bytes) is not int or not 8192 <= max_request_bytes <= 512000:
        raise ValueError("Full prompt plus schema UTF-8 byte ceiling must be 8192..512000")
    if sample_binding is not None:
        validate_sample_plan(sample_binding.get("plan"))
    return dict(contract=CONTRACT, provider="codex_chatgpt", requested_model=requested_model,
        observed_model_identity_required=False, observed_model_identity_may_be_unavailable=True,
        max_cases=max_cases, max_request_bytes=max_request_bytes, top_k=5,
        definitions=[list(row) for row in DEFINITIONS], variants=list(pilot.VARIANTS),
        derived_from={"hybrid_B":"hybrid_base", "hybrid_AB":"hybrid_A"},
        all_accepted_test_records=True, formal_minimum=FORMAL_MINIMUM,
        retrieval_question_count=1000, answer_question_count=100,
        answer_sample_binding=sample_binding,
        generator_reference_access=False, per_query_generation_latency_measured=False)


def runtime_values(runtime, cfg):
    """Six actual artifacts to stage BEFORE the formal annotation freeze."""
    sources = {path.name:path.read_bytes() for path in SOURCE_PATHS}
    config_bytes = (ROOT / "config.toml").read_bytes()
    return dict(index_manifest=(runtime.stage / "index/manifest.json").read_bytes(),
        retrieval_config=protocol.canonical_bytes(dict(experiment=cfg,
            config_sha256=protocol.sha256(config_bytes), config_utf8=config_bytes.decode("utf-8"))),
        generation_prompt=(ISOLATION_PROMPT + "\n" + pilot.GENERATION_PROMPT + "\nUNTRUSTED_PAYLOAD_JSON\n").encode(),
        verification_prompt=(ISOLATION_PROMPT + "\n" + pilot.VERIFICATION_PROMPT + "\nUNTRUSTED_PAYLOAD_JSON\n").encode(),
        judge_prompt=(ISOLATION_PROMPT + "\n" + FORMAL_JUDGE_PROMPT + "\nUNTRUSTED_PAYLOAD_JSON\n").encode(),
        implementation=protocol.canonical_bytes({name:dict(sha256=protocol.sha256(raw), source_utf8=raw.decode("utf-8"))
                                                 for name, raw in sources.items()}))


def snapshot(runtime, output, cfg):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Never overwrite a runtime snapshot")
    if cfg.get("answer_sample_binding") is None:
        raise ValueError("Seal and bind the100-question pre-label sample before formal runtime snapshots")
    _assert_hashes(runtime.protected_sha256)
    values = runtime_values(runtime, cfg)
    paths = {}
    for key, raw in values.items():
        path = output / "runtime" / (key + ".artifact")
        write_new(path, raw)
        paths[key] = str(path)
    write_new(output / "runtime_artifacts.json", paths)
    write_new(output / "receipt.json", dict(status="actual_formal_runtime_snapshots_not_dataset_freeze",
        contract=CONTRACT, runtime_sha256={key:protocol.sha256(raw) for key, raw in values.items()},
        corpus_sha256=runtime.corpus_sha256, model_calls=0, human_gold=False,
        formal_test_ready=False, config=cfg))
    _assert_hashes(runtime.protected_sha256)
    return paths


@dataclass
class Cohort:
    frozen: Path
    manifest: dict
    questions: dict
    references: dict
    packets: dict
    protected: dict
    cfg: dict
    source_hashes: dict


def load_cohort(frozen, runtime):
    frozen = Path(frozen).resolve()
    artifacts = {key:contained(frozen, name).read_bytes() for key, name in NAMES.items()}
    manifest_path, paths_path = contained(frozen, "manifest.json"), contained(frozen, "runtime_artifacts.json")
    manifest, paths = read_json(manifest_path), read_json(paths_path)
    if not isinstance(paths, dict) or set(paths) != protocol.RUNTIME_KEYS:
        raise ValueError("The formal cohort needs all six runtime bindings")
    snapshot_paths = {key:Path(path).resolve() for key, path in paths.items() if isinstance(path, str)}
    if len(snapshot_paths) != 6 or any(not path.is_relative_to(frozen / "runtime") or not path.is_file()
                                      for path in snapshot_paths.values()):
        raise ValueError("Frozen runtime snapshots must be inside this frozen version")
    values = {key:path.read_bytes() for key, path in snapshot_paths.items()}
    hashes = {key:protocol.sha256(raw) for key, raw in values.items()}
    protocol.verify_manifest(manifest, artifacts, hashes)
    # No caller-supplied reduced threshold or development override exists.
    protocol.preflight_experiments(manifest, FORMAL_MINIMUM)
    if protocol.sha256(artifacts["corpus"]) != runtime.corpus_sha256:
        raise ValueError("Frozen corpus differs from the verified local index corpus")
    binding = json.loads(values["retrieval_config"], object_pairs_hook=protocol._unique_object,
                         parse_constant=protocol._invalid_constant)
    cfg = binding.get("experiment", {})
    expected = settings(cfg.get("requested_model"), cfg.get("max_cases"), cfg.get("max_request_bytes"), cfg.get("answer_sample_binding"))
    if cfg != expected or runtime_values(runtime, cfg) != values:
        raise ValueError("Actual formal implementation, prompts, model configuration or index differs from freeze")
    rows = protocol.parse_jsonl(artifacts["questions"])
    pools = protocol.parse_jsonl(artifacts["evidence_pool"])
    packets = protocol.prepare_inputs(rows, pools, protocol.parse_jsonl(artifacts["corpus"]),
                                      artifacts["annotation_prompt"].decode("utf-8"))
    questions = {row["record_id"]:row for row in rows if row["split"] == "test"}
    binding_sample = cfg.get("answer_sample_binding")
    actual_sample, sample_protected = load_sample_binding(binding_sample["directory"], runtime.canonical_root)
    if actual_sample != binding_sample or list(questions) != actual_sample["plan"]["original_ordered_test_record_ids"]:
        raise ValueError("Frozen1000 TEST population or bound answer sample changed")
    sample_rows = protocol.parse_jsonl((Path(binding_sample["directory"]) / "sample_questions.jsonl").read_bytes())
    if any(row != questions.get(row["record_id"]) for row in sample_rows):
        raise ValueError("Sampled question text/cutoff/group differs from the frozen population")
    references = {row["record_id"]:row for row in protocol.parse_jsonl(artifacts["adjudications"])
                  if row["record_id"] in questions}
    if len(questions) != manifest["split_status_counts"]["test"]["accepted"] or any(
            row["status"] != "accepted" for row in references.values()):
        raise ValueError("All frozen TEST records, not a convenient subset, must be accepted")
    references = {qid:dict(row, reference_evidence=packets[qid]["payload"]["evidence"])
                  for qid, row in references.items()}
    protected = {str(frozen / name):protocol.sha256(artifacts[key]) for key, name in NAMES.items()}
    protected.update({str(manifest_path):file_hash(manifest_path), str(paths_path):file_hash(paths_path)})
    protected.update({str(snapshot_paths[key]):hashes[key] for key in paths})
    protected.update(sample_protected)
    source_hashes = {path.name:file_hash(path) for path in SOURCE_PATHS}
    protected.update({str(path):source_hashes[path.name] for path in SOURCE_PATHS})
    protected[str(ROOT / "config.toml")] = file_hash(ROOT / "config.toml")
    return Cohort(frozen, manifest, questions, references, packets, protected, cfg, source_hashes)


def load_rankings(directory, cohort, runtime):
    """Adopt complete, audited pre-freeze rankings without reranking or retiming."""
    _assert_hashes(cohort.protected)
    directory = Path(directory).resolve()
    allowed = runtime.canonical_root.resolve() / "data/machine_runs/candidate-retrieval"
    if not directory.is_relative_to(allowed) or directory == allowed:
        raise ValueError("Use an immutable canonical candidate-retrieval run")
    receipt, inputs = read_json(contained(directory, "receipt.json")), read_json(contained(directory, "inputs.json"))
    if (receipt.get("status") != "candidate_evidence_pools_created_not_annotations_or_accuracy_results"
            or receipt.get("original_inputs_unchanged") is not True or receipt.get("top_k") != 5
            or receipt.get("retrieval_conditions_per_question") != 6 or receipt.get("author_anchor_injection") is not False
            or inputs.get("protocol") != "machine-candidate-top5-three-mode-two-temporal-pool-v1"
            or inputs.get("bindings", {}).get("corpus_sha256") != runtime.corpus_sha256):
        raise ValueError("Expected the unchanged complete six-condition retrieval protocol")
    protected = {str(directory / "receipt.json"):file_hash(directory / "receipt.json")}
    for name, expected in receipt.get("artifact_sha256", {}).items():
        path = contained(directory, name)
        if file_hash(path) != expected:
            raise ValueError("Precomputed retrieval artifact changed")
        protected[str(path)] = expected
    for required in ("inputs.json", "rankings.jsonl", "annotation_pool.jsonl"):
        if str(directory / required) not in protected:
            raise ValueError("Missing original ranking/input/pool artifact binding")
    # Preserve the original retrieval's source/index/question bindings. A path
    # in a receipt is not authorization to read files outside the canonical data.
    public_sources = set(SOURCE_PATHS) | {ROOT / "scripts/aggregate_machine_questions.py",
        ROOT / "scripts/run_development_retrieval.py", ROOT / "scripts/build_staged_index.py"}
    for name, expected in inputs.get("protected_sha256", {}).items():
        path = Path(name).resolve()
        if (not path.is_relative_to(runtime.canonical_root.resolve()) and path not in public_sources) or not path.is_file():
            raise ValueError("Original retrieval binding escapes the canonical project")
        protected[str(path)] = expected
    _assert_hashes(protected)
    ranks = protocol._index(protocol.parse_jsonl((directory / "rankings.jsonl").read_bytes()), "record_id", "rankings")
    pools = protocol._index(protocol.parse_jsonl((directory / "annotation_pool.jsonl").read_bytes()), "record_id", "retrieval pools")
    all_frozen_ids = set(cohort.packets)
    original_questions = None
    if inputs.get("cohort"):
        original_cohort = Path(inputs["cohort"]).resolve()
        allowed_cohorts = runtime.canonical_root.resolve() / "data/machine_runs/question-cohorts"
        if not original_cohort.is_relative_to(allowed_cohorts) or original_cohort == allowed_cohorts:
            raise ValueError("Original question-cohort pointer escapes the certified canonical data")
        question_path = contained(original_cohort, "questions.jsonl")
        if str(question_path) not in protected:
            raise ValueError("Bind the original selected question file before reusing rankings")
        original_questions = protocol._validate_questions(protocol.parse_jsonl(question_path.read_bytes()))
    if set(ranks) != set(pools) or not all_frozen_ids.issubset(ranks):
        raise ValueError("Original retrieval must cover exactly the full frozen development/test cohort")
    extras = set(ranks) - all_frozen_ids
    if extras and (original_questions is None or set(original_questions) != set(ranks)
            or any(original_questions[qid]["split"] != "development" for qid in extras)):
        raise ValueError("Only certified extra DEVELOPMENT rankings may remain outside the annotation freeze")
    if original_questions is not None:
        original_test = {qid for qid, row in original_questions.items() if row["split"] == "test"}
        if original_test != set(cohort.questions):
            raise ValueError("Frozen TEST records must equal every originally selected TEST question")
        for qid in all_frozen_ids:
            packet = cohort.packets[qid]
            clean = {key:value for key, value in packet["payload"].items() if key != "evidence"}
            clean.update(packet["metadata"])
            if clean != original_questions[qid]:
                raise ValueError("Frozen question text/cutoff/split/group differs from original retrieval selection")
    for qid in all_frozen_ids:
        packet = cohort.packets[qid]["payload"]
        pool = pools[qid]
        if protocol.clean_packet(pool)["payload"] != packet:
            raise ValueError("Retrieved annotation pool differs from the exact frozen input")
        conditions = ranks[qid].get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != CONDITIONS:
            raise ValueError("Missing ranking condition")
        for key, hits in conditions.items():
            if (not isinstance(hits, list) or len(hits) > 5 or len({h["passage_id"] for h in hits}) != len(hits)
                    or any(h.get("rank") != rank or type(h.get("score")) not in (int, float)
                           or not math.isfinite(h["score"]) for rank, h in enumerate(hits, 1))):
                raise ValueError("Ranking depth, uniqueness, positions or score differs")
            _validate_hits(hits, runtime, packet["as_of"], key.endswith(":1"))
            if packet.get("region") and any(packet["region"] not in hit.get("regions", []) for hit in hits):
                raise ValueError("Precomputed ranking violates the explicit query region")
        union = {hit["passage_id"] for hits in conditions.values() for hit in hits}
        if union != {hit["passage_id"] for hit in packet["evidence"]}:
            raise ValueError("Judged frozen pool must equal the complete six-condition union")
    return ranks, protected, dict(run=str(directory), receipt_sha256=file_hash(directory / "receipt.json"),
        rankings_sha256=file_hash(directory / "rankings.jsonl"),
        measured_batch_embedding_seconds=receipt.get("batch_query_embedding_seconds"),
        measured_six_condition_ranking_seconds=receipt.get("ranking_seconds_excluding_embedding"),
        measured_retrieval_wall_seconds=receipt.get("wall_seconds"),
        per_query_mode_embedding_or_total_latency=None, timing_reused_not_remeasured=True,
        certified_extra_development_records_not_experiment_test=len(extras))


class RequestReady(Exception):
    def __init__(self, prompt, payload, schema, role):
        self.request = dict(prompt=prompt, payload=payload, schema=schema, role=role)


def stage_result(stage, cases, evidence, answers, rows, references, directory, caller):
    if stage in {variant for variant, _, _ in DEFINITIONS}:
        temporal = stage == "hybrid_A"
        parsed, _ = pilot.generate_batch(cases, evidence, temporal, directory, caller)
        return dict(answers=parsed)
    if stage in {"hybrid_B", "hybrid_AB"}:
        parsed, removed, _ = pilot.verify_batch(cases, answers, evidence, directory, caller)
        return dict(answers=parsed, removed_claims=removed)
    def shared_judge(prompt, payload, schema, output, *, role):
        shared, entries = {}, {}
        for key, entry in payload.items():
            qid = rows[key]["record_id"]
            pool = references[qid].get("reference_evidence")
            if not isinstance(pool, list):
                raise ValueError("Formal judge needs original full pooled source bodies, not labels/quotes alone")
            by_id = {p["passage_id"]:p for p in pool}
            if any(h["passage_id"] not in by_id or any(h.get(field) != by_id[h["passage_id"]].get(field)
                    for field in ("text","document_id","available_by")) for h in entry["evidence"]):
                raise ValueError("Judge evidence must be original members of its own fixed pool")
            shared[qid] = dict(question=entry["question"], as_of=entry["as_of"], reference=entry["reference"],
                               reference_evidence=pool)
            entries[key] = dict(record_id=qid, answer=entry["answer"],
                                retrieved_ids=[p["passage_id"] for p in entry["evidence"]])
        return caller(FORMAL_JUDGE_PROMPT, dict(shared_cases=shared, answers=entries), schema, output, role=role)
    parsed, _ = pilot.judge_batch(cases, rows, references, directory, shared_judge)
    return dict(judgments=parsed)


def capture_request(stage, cases, evidence=None, answers=None, rows=None, references=None):
    def capture(prompt, payload, schema, directory, *, role):
        raise RequestReady(prompt, payload, schema, role)
    try:
        result = stage_result(stage, cases, evidence, answers, rows, references, Path("not-written"), capture)
    except RequestReady as found:
        return found.request, None
    return None, result  # No factual claims: exact B reuse without a native call.


def request_bytes(request):
    return ((ISOLATION_PROMPT + "\n" + request["prompt"] + "\nUNTRUSTED_PAYLOAD_JSON\n").encode()
            + protocol.canonical_bytes(request["payload"]))


def bounded_batches(items, request_builder, cfg):
    """Greedy stable batches of complete items; no evidence/string truncation."""
    batches, pending = [], []
    for item in items:
        candidate = pending + [item]
        request = request_builder(candidate)
        size = len(request_bytes(request)) + len(protocol.canonical_bytes(request["schema"])) if request else 0
        if len(candidate) > cfg["max_cases"] or size > cfg["max_request_bytes"]:
            if not pending:
                raise ValueError("One complete case exceeds the frozen request ceiling; no truncation or call")
            batches.append(pending)
            pending = [item]
            request = request_builder(pending)
            size = len(request_bytes(request)) + len(protocol.canonical_bytes(request["schema"])) if request else 0
            if size > cfg["max_request_bytes"]:
                raise ValueError("One complete case exceeds the frozen request ceiling; no truncation or call")
        else:
            pending = candidate
    if pending:
        batches.append(pending)
    return batches


def validate_receipt(receipt, request, cohort, run):
    if (not isinstance(receipt, dict) or receipt.get("accepted") is not True
            or receipt.get("role") != request["role"] or receipt.get("process_exit_code") != 0
            or receipt.get("errors") or receipt.get("codex_cli_invocations") != 1
            or receipt.get("external_paid_api_calls") != 0 or receipt.get("model_tools_disabled") is not True
            or receipt.get("execpolicy_rules_not_disabled") is not True
            or receipt.get("authentication") != "existing_ChatGPT_subscription"
            or receipt.get("requested_model") != (cohort.cfg["requested_model"] or "cli_default_unresolved")):
        raise ValueError("Native call receipt violates the frozen execution contract")
    directory = Path(receipt["artifact_directory"]).resolve()
    if not directory.is_relative_to(run / "jobs"):
        raise ValueError("Native call artifacts escape the formal run")
    for name, expected in receipt["artifacts_sha256"].items():
        if file_hash(contained(directory, name)) != expected:
            raise ValueError("Native artifact bytes changed")
    if ((directory / "stdin.txt").read_bytes() != request_bytes(request)
            or receipt["prompt_sha256"] != protocol.sha256(request_bytes(request))
            or (directory / "schema.json").read_bytes() != protocol.canonical_bytes(request["schema"])
            or receipt["schema_sha256"] != protocol.sha256(protocol.canonical_bytes(request["schema"]))):
        raise ValueError("Native schema, prompt or full payload differs from the immutable job")
    for name, digest in receipt["source_snapshots_sha256"].items():
        if cohort.source_hashes.get(name) != digest:
            raise ValueError("Call used a different implementation source version")
    helper, _ = _load_isolation_snapshot()
    events = helper.parse_execution_events((directory / "events.jsonl").read_bytes())
    if (events["errors"] or events["thread_id"] != receipt["execution_id"]
            or events["usage"] != receipt["usage"] or events["observed_model"] != receipt["observed_model"]
            or sum(e.get("type") == "turn.completed" for e in events["events"]) != 1):
        raise ValueError("Native execution events differ from the retained receipt")
    raw = read_json(directory / "raw_final.json")
    messages = [e["item"]["text"] for e in events["events"] if e.get("type") == "item.completed"
                and isinstance(e.get("item"), dict) and e["item"].get("type") == "agent_message"]
    if not messages or json.loads(messages[-1], object_pairs_hook=protocol._unique_object,
                                 parse_constant=protocol._invalid_constant) != raw:
        raise ValueError("Native final message differs from saved raw JSON")
    Draft202012Validator(request["schema"]).validate(raw)
    return raw


class SegmentLimit(Exception):
    pass


class ExecutionBudget:
    def __init__(self, maximum_calls):
        if type(maximum_calls) is not int or maximum_calls < 0:
            raise ValueError("Maximum new native attempts must be a nonnegative integer")
        self.maximum, self.started = maximum_calls, 0

    def reserve(self):
        if self.started >= self.maximum:
            raise SegmentLimit("Configured new-attempt ceiling reached; resume with identical frozen inputs")
        self.started += 1


def job(run, stage, ids, request, no_call_result, process, cohort, budget, caller, retry_failed=False):
    key = dict(stage=stage, item_ids=ids, request=request, dataset_version=cohort.manifest["version"],
               runtime_sha256=cohort.manifest["runtime_sha256"])
    job_id = stage + "-" + protocol.sha256(protocol.canonical_bytes(key))[:24]
    directory = run / "jobs" / job_id
    descriptor = directory / "job.json"
    if descriptor.exists():
        if read_json(descriptor) != key:
            raise ValueError("Immutable experiment job binding changed")
    else:
        write_new(descriptor, key)
    completed = directory / "completed.json"
    rejected_hashes = {read_json(path).get("receipt_sha256") for path in directory.glob("validation_rejection-*.json")}
    if completed.exists():
        record = read_json(completed)
        for name, expected in record["artifacts_sha256"].items():
            if file_hash(contained(directory, name)) != expected:
                raise ValueError("Completed job changed; never rerun silently")
        if request is not None:
            receipts = list((directory / "attempts").glob("*/calls/*/receipt.json"))
            accepted = [read_json(path) for path in receipts if read_json(path).get("accepted") is True
                        and file_hash(path) not in rejected_hashes]
            if len(accepted) != 1:
                raise ValueError("Completed job needs exactly one accepted native execution")
            validate_receipt(accepted[0], request, cohort, run)
        return read_json(contained(directory, record["result"]))
    if request is None:
        write_new(directory / "no_call_result.json", no_call_result)
        write_new(completed, dict(result="no_call_result.json", model_calls=0,
            reason="No claims; reuse the exact saved original", artifacts_sha256={
                "job.json":file_hash(descriptor), "no_call_result.json":file_hash(directory / "no_call_result.json")}))
        return no_call_result
    # If the process stopped after receiving an accepted raw output, adopt and
    # revalidate it. Never issue a second call for an already accepted response.
    recovered = []
    for receipt_path in sorted((directory / "attempts").glob("*/calls/*/receipt.json")):
        receipt = read_json(receipt_path)
        if receipt.get("accepted") and file_hash(receipt_path) not in rejected_hashes:
            raw = validate_receipt(receipt, request, cohort, run)
            recovered.append((raw, receipt, receipt_path))
    if len(recovered) > 1:
        raise ValueError("Multiple accepted native responses for one immutable job")
    attempt = None
    if recovered:
        raw, receipt, receipt_path = recovered[0]
    else:
        prior_attempts = list((directory / "attempts").glob("*/request.json"))
        if prior_attempts and not retry_failed:
            raise ValueError("Failed/interrupted attempt retained; retry requires explicit --retry-failed")
        budget.reserve()
        _assert_hashes(cohort.protected)
        attempt = directory / "attempts" / (now().replace(":", "-") + "-" + uuid4().hex[:8])
        write_new(attempt / "request.json", request)
        started = perf_counter()
        try:
            raw, receipt = caller(request["prompt"], request["payload"], request["schema"], attempt / "calls",
                                  role=request["role"], requested_model=cohort.cfg["requested_model"])
            receipt_path = Path(receipt["artifact_directory"]) / "receipt.json"
            raw = validate_receipt(receipt, request, cohort, run)
            _assert_hashes(cohort.protected)
            write_new(attempt / "native_validated.json", dict(elapsed_seconds=perf_counter()-started,
                receipt=str(receipt_path.relative_to(directory)), receipt_sha256=file_hash(receipt_path),
                execution_id=receipt["execution_id"]))
        except Exception as error:
            write_new(attempt / "runner_failure.json", dict(error_type=type(error).__name__, error=str(error),
                elapsed_seconds=perf_counter()-started, automatic_retry=False, native_usage="inspect_retained_receipts_or_unknown"))
            raise
    def replay(prompt, payload, schema, unused, *, role):
        if dict(prompt=prompt, payload=payload, schema=schema, role=role) != request:
            raise ValueError("Stage request differs from the saved immutable request")
        return deepcopy(raw), receipt
    try:
        result = process(replay)
    except Exception as error:
        rejection = directory / ("validation_rejection-" + uuid4().hex + ".json")
        write_new(rejection, dict(error_type=type(error).__name__, error=str(error),
                                  retained_receipt=str(receipt_path), receipt_sha256=file_hash(receipt_path),
                                  mechanical_output_rejection_not_answer_quality_selection=True,
                                  automatic_retry=False))
        raise
    result_path = directory / ("accepted-" + uuid4().hex + ".json")
    write_new(result_path, result)
    hashes = {"job.json":file_hash(descriptor), result_path.name:file_hash(result_path),
              str(receipt_path.relative_to(directory)).replace("\\", "/"):file_hash(receipt_path)}
    for name, digest in receipt["artifacts_sha256"].items():
        hashes[(Path(receipt["artifact_directory"]).relative_to(directory) / name).as_posix()] = digest
    write_new(completed, dict(result=result_path.name, execution_id=receipt["execution_id"],
                             reused_accepted_native_response=attempt is None, artifacts_sha256=hashes))
    return result


def quality_summary(rows):
    summary = {}
    for variant in pilot.VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        flat = lambda field:[v for r in selected for v in r["automatic_judge"][field]]
        refs_insufficient = [r for r in selected if r["reference_answerability"] == "insufficient"]
        refs_answerable = [r for r in selected if r["reference_answerability"] != "insufficient"]
        values = dict(questions=len(selected), automatic_correctness=mean(r["automatic_judge"]["correctness"]/2 for r in selected),
            automatic_relevance=mean(r["automatic_judge"]["relevance"]/2 for r in selected),
            automatic_insufficiency_handling=mean(r["automatic_judge"]["insufficiency_handling"] for r in selected),
            reference_insufficient_questions=len(refs_insufficient), reference_answerable_questions=len(refs_answerable),
            categorical_correct_abstention_rate=mean(r["answer"]["answerability"] == "insufficient" for r in refs_insufficient)
                if refs_insufficient else None,
            categorical_false_abstention_rate=mean(r["answer"]["answerability"] == "insufficient" for r in refs_answerable)
                if refs_answerable else None,
            claims_removed=sum(len(r["removed_claims"]) for r in selected),
            future_evidence_count=sum(r["future_evidence_count"] for r in selected),
            ineligible_citation_count=sum(r["ineligible_citation_count"] for r in selected))
        for field, label in (("claim_supported","automatic_claim_faithfulness"),
                             ("citation_supported","automatic_citation_support_rate"),
                             ("context_relevant","automatic_context_precision")):
            items = flat(field)
            values[label] = mean(items) if items else None
            values[label + "_denominator"] = len(items)
        for metric in METRICS:
            items = [r["retrieval_metrics"][metric] for r in selected if r["retrieval_metrics"][metric] is not None]
            values[metric] = mean(items) if items else None
            values[metric + "_defined_queries"] = len(items)
        summary[variant] = values
    return summary


def full_retrieval_report(cohort, rankings):
    """Score all1000 TEST questions in all six original ranking conditions."""
    if len(cohort.questions) != 1000:
        raise ValueError("The fixed retrieval evaluation needs every original1000 TEST question")
    rows, summary = [], {}
    for condition in sorted(CONDITIONS):
        selected = []
        for qid, question in cohort.questions.items():
            evidence = rankings[qid]["conditions"][condition]
            selected.append(dict(record_id=qid, condition=condition,
                ranked_ids=[hit["passage_id"] for hit in evidence],
                metrics=ranked_metrics([hit["passage_id"] for hit in evidence], cohort.references[qid]["label"]["qrels"], 5),
                future_evidence_count=sum(not hit.get("available_by") or hit["available_by"] > question["as_of"] for hit in evidence),
                per_query_retrieval_seconds=None))
        values = dict(questions=len(selected), future_evidence_count=sum(row["future_evidence_count"] for row in selected))
        for metric in METRICS:
            defined = [row["metrics"][metric] for row in selected if row["metrics"][metric] is not None]
            values[metric] = mean(defined) if defined else None
            values[metric+"_defined_queries"] = len(defined)
        summary[condition] = values
        rows.extend(selected)
    return dict(status="complete_machine_only1000_TEST_retrieval_evaluation", human_gold=False,
        test_questions=1000, ranking_conditions=6, ranked_query_conditions=6000, k=5,
        condition_meaning={key:dict(mode=key.split(":")[0], temporal=key.endswith(":1")) for key in CONDITIONS},
        dataset_version=cohort.manifest["version"], summary=summary, per_query=rows,
        limitations=["All1000 TEST questions are evaluated; the100-answer sample does not restrict retrieval coverage.",
            "Relevance judgments are machine-adjudicated pooled judgments, not human gold or exhaustive corpus recall.",
            "Original embeddings/scores were batched/reused; no fabricated individual retrieval latency is assigned."])


def save_full_retrieval(run, cohort, rankings):
    path = run / "retrieval_evaluation.json"
    binding = protocol.sha256(protocol.canonical_bytes(dict(
        all_test_ids=list(cohort.questions), references={qid:row["label"] for qid,row in cohort.references.items()},
        rankings={qid:rankings[qid] for qid in cohort.questions}, runtime_sha256=cohort.manifest["runtime_sha256"])))
    receipt_path = run / "retrieval_evaluation_receipt.json"
    if path.exists():
        receipt = read_json(contained(run,"retrieval_evaluation_receipt.json"))
        if receipt["input_sha256"] != binding or receipt["result_sha256"] != file_hash(path):
            raise ValueError("Completed full retrieval result/input binding changed")
    else:
        report = full_retrieval_report(cohort, rankings)
        write_new(path, report)
        write_new(receipt_path, dict(input_sha256=binding, result_sha256=file_hash(path),
            completed_test_questions=1000, ranked_query_conditions=6000, human_gold=False, native_calls=0))
    return dict(path=str(path), sha256=file_hash(path), test_questions=1000, ranked_query_conditions=6000)


def usage_audit(run):
    """Count each native execution once, including rejected attempts when known."""
    unique, unknown = {}, []
    for path in sorted((run / "jobs").glob("*/attempts/*/calls/*/receipt.json")):
        receipt = read_json(path)
        for name, expected in receipt.get("artifacts_sha256", {}).items():
            if file_hash(contained(path.parent, name)) != expected:
                raise ValueError("Native audit artifact changed")
        identity = receipt.get("execution_id")
        if not identity:
            unknown.append(dict(receipt=str(path), reason="Native execution identity unavailable", usage=receipt.get("usage")))
            continue
        if identity in unique and unique[identity]["sha256"] != file_hash(path):
            raise ValueError("Conflicting receipts reuse a native execution identity")
        unique[identity] = dict(receipt=receipt, path=str(path), sha256=file_hash(path))
    tokens = {}
    for entry in unique.values():
        for key, value in (entry["receipt"].get("usage") or {}).items():
            if type(value) is int and value >= 0:
                tokens[key] = tokens.get(key, 0) + value
    attempts_without_receipt = [str(path.parent) for path in (run / "jobs").glob("*/attempts/*/request.json")
                               if not list((path.parent / "calls").glob("*/receipt.json"))]
    identities = [e["receipt"].get("observed_model") for e in unique.values()]
    rejected_receipts = {read_json(path).get("receipt_sha256") for path in (run / "jobs").glob("*/validation_rejection-*.json")}
    completed_ids = {read_json(path).get("execution_id") for path in (run / "jobs").glob("*/completed.json")}
    native_binaries = {e["receipt"].get("native_cli_sha256") for e in unique.values() if e["receipt"].get("native_cli_sha256")}
    known_models = {value for value in identities if value is not None}
    if len(native_binaries) > 1 or len(known_models) > 1:
        raise ValueError("Observed native executable/model changed within the fixed experiment")
    return dict(known_native_execution_count=len(unique), accepted_native_executions=sum(e["receipt"].get("accepted") is True for e in unique.values()),
        rejected_native_executions=sum(e["receipt"].get("accepted") is not True for e in unique.values()),
        mechanically_rejected_native_executions=sum(e["sha256"] in rejected_receipts for e in unique.values()),
        accepted_experiment_output_executions=sum(identity in completed_ids for identity in unique),
        executions_with_known_usage=sum(e["receipt"].get("usage") is not None for e in unique.values()),
        executions_with_unknown_usage=sum(e["receipt"].get("usage") is None for e in unique.values()),
        known_native_token_totals=tokens, unknown_execution_receipts=unknown,
        interrupted_or_precall_attempts_with_unknown_execution=attempts_without_receipt,
        usage_counted_once_by_execution_id=True, external_paid_api_calls=0, subscription_dollar_cost=None,
        observed_model=next(iter(known_models)) if known_models and all(v is not None for v in identities) else None,
        observed_model_identity_available_for_all=bool(identities) and all(v is not None for v in identities))


def experiment(runtime, frozen, retrieval_run, output_parent, *, resume=None, max_new_calls=20,
               retry_failed=False, caller=call_structured_json):
    started = perf_counter()
    cohort = load_cohort(frozen, runtime)
    rankings, rank_protected, timing = load_rankings(retrieval_run, cohort, runtime)
    cohort.protected.update(runtime.protected_sha256)
    cohort.protected.update(rank_protected)
    _assert_hashes(cohort.protected)
    test_ids = list(cohort.questions)
    sample = cohort.cfg["answer_sample_binding"]
    ids = sample["plan"]["answer_record_ids"]  # Immutable pre-label random100; no result-based selector.
    parent = Path(output_parent).resolve()
    allowed = runtime.canonical_root.resolve() / "data/machine_runs/formal-experiments"
    if parent != allowed:
        raise ValueError("Formal results belong in the distinct canonical formal-experiments directory")
    run = plain_path(resume) if resume else parent / (now().replace(":", "-") + "-" + uuid4().hex[:8])
    if not run.is_relative_to(parent) or run == parent:
        raise ValueError("Resume must name one formal experiment run")
    run = long_path(run)
    plan = dict(contract=CONTRACT, dataset_version=cohort.manifest["version"], formal_minimum=FORMAL_MINIMUM,
        human_gold=False, all_test_record_ids=test_ids, questions=len(test_ids), answer_record_ids=ids,
        answer_questions=len(ids), answer_sample_plan_sha256=sample["sample_plan_sha256"],
        expected_variant_answers=len(ids)*6,
        planned_base_generation_answers=len(ids)*4,
        unbatched_native_execution_upper_bound_without_retries=len(ids)*12,
        batching_reduces_invocations_not_evidence_or_question_coverage=True,
        runtime_sha256=cohort.manifest["runtime_sha256"], cfg=cohort.cfg, retrieval=timing,
        frozen_manifest_sha256=file_hash(cohort.frozen / "manifest.json"), protected_sha256=cohort.protected)
    if resume:
        if read_json(contained(run, "plan.json")) != plan:
            raise ValueError("Resume inputs, implementation, corpus or fixed model configuration changed")
    else:
        write_new(run / "plan.json", plan)
    retrieval_result = save_full_retrieval(run, cohort, rankings)
    budget = ExecutionBudget(max_new_calls)
    status, error_text = "partial_not_complete", None
    all_rows = []
    segment = run / "segments" / (now().replace(":", "-") + "-" + uuid4().hex[:8] + ".json")
    try:
        for offset in range(0, len(ids), cohort.cfg["max_cases"]):
            group_ids = ids[offset:offset+cohort.cfg["max_cases"]]
            cases = {qid:cohort.questions[qid] for qid in group_ids}
            evidence = {variant:{qid:rankings[qid]["conditions"][condition] for qid in group_ids}
                        for variant, condition, _ in DEFINITIONS}
            answers, removed = {}, {}
            for stage, _, _ in DEFINITIONS:
                builder = lambda subset: capture_request(stage, {q:cases[q] for q in subset}, {q:evidence[stage][q] for q in subset})[0]
                answers[stage] = {}
                for subset in bounded_batches(group_ids, builder, cohort.cfg):
                    selected_cases, hits = {q:cases[q] for q in subset}, {q:evidence[stage][q] for q in subset}
                    request, skipped = capture_request(stage, selected_cases, hits)
                    process = lambda execute:stage_result(stage, selected_cases, hits, None, None, None,
                        run / "checks" / (stage + "-" + uuid4().hex), execute)
                    answers[stage].update(job(run, stage, subset, request, skipped, process, cohort, budget, caller, retry_failed)["answers"])
                removed[stage] = {qid:[] for qid in group_ids}
            for original, stage in (("hybrid_base","hybrid_B"), ("hybrid_A","hybrid_AB")):
                evidence[stage], answers[stage], removed[stage] = evidence[original], {}, {}
                builder = lambda subset:capture_request(stage, {q:cases[q] for q in subset},
                    {q:evidence[original][q] for q in subset}, {q:answers[original][q] for q in subset})[0]
                for subset in bounded_batches(group_ids, builder, cohort.cfg):
                    selected_cases, hits, originals = ({q:cases[q] for q in subset},
                        {q:evidence[original][q] for q in subset}, {q:answers[original][q] for q in subset})
                    request, skipped = capture_request(stage, selected_cases, hits, originals)
                    process = lambda execute:stage_result(stage, selected_cases, hits, originals, None, None,
                        run / "checks" / (stage + "-" + uuid4().hex), execute)
                    checked = job(run, stage, subset, request, skipped, process, cohort, budget, caller, retry_failed)
                    answers[stage].update(checked["answers"])
                    removed[stage].update(checked["removed_claims"])
            rows = {}
            # Adjacent variants of the same query share its source pool within
            # a judge batch, instead of sending the same full pool six times.
            for qid in group_ids:
                for stage in pilot.VARIANTS:
                    hits, answer = evidence[stage][qid], answers[stage][qid]
                    by_id = {hit["passage_id"]:hit for hit in hits}
                    rows[stage + "__" + qid] = dict(record_id=qid, variant=stage, question=cases[qid]["question"],
                        as_of=cases[qid]["as_of"], answer=answer, evidence=hits, removed_claims=removed[stage][qid],
                        generation_reused_from={"hybrid_B":"hybrid_base", "hybrid_AB":"hybrid_A"}.get(stage),
                        retrieval_reused_from=timing["run"], per_query_generation_seconds=None,
                        per_query_total_seconds=None, per_query_retrieval_seconds=None,
                        future_evidence_count=sum(not h.get("available_by") or h["available_by"] > cases[qid]["as_of"] for h in hits),
                        ineligible_citation_count=sum(not by_id[pid].get("available_by") or by_id[pid]["available_by"] > cases[qid]["as_of"]
                            for claim in answer["claims"] for pid in claim["citations"]),
                        reference_answerability=cohort.references[qid]["label"]["answerability"],
                        retrieval_metrics=ranked_metrics([h["passage_id"] for h in hits], cohort.references[qid]["label"]["qrels"], 5))
            builder = lambda subset:capture_request("judge", cases, rows={key:rows[key] for key in subset}, references=cohort.references)[0]
            judgments = {}
            for subset in bounded_batches(list(rows), builder, cohort.cfg):
                selected_rows = {key:rows[key] for key in subset}
                request, skipped = capture_request("judge", cases, rows=selected_rows, references=cohort.references)
                process = lambda execute:stage_result("judge", cases, None, None, selected_rows, cohort.references, run / "unused", execute)
                judgments.update(job(run, "judge", subset, request, skipped, process, cohort, budget, caller, retry_failed)["judgments"])
            for key, row in rows.items():
                row["automatic_judge"] = judgments[key]
            group_path = run / "groups" / (str(offset).zfill(6) + ".json")
            value = dict(record_ids=group_ids, rows=list(rows.values()))
            if group_path.exists():
                if read_json(group_path) != value:
                    raise ValueError("Completed group differs from its immutable inputs and native results")
            else:
                write_new(group_path, value)
            all_rows.extend(rows.values())
            print(f"Formal machine experiment {len(all_rows)//6}/{len(ids)} questions; new attempts {budget.started}", flush=True)
        status = "completed_formal_machine_only_experiment"
    except SegmentLimit as error:
        status, error_text = "partial_new_attempt_limit_reached", str(error)
    except Exception as error:
        status, error_text = "failed_not_complete", type(error).__name__ + ": " + str(error)
    _assert_hashes(cohort.protected)
    usage = usage_audit(run)
    report = dict(status=status, formal_heldout_test=status == "completed_formal_machine_only_experiment",
        human_gold=False, dataset_version=cohort.manifest["version"], required_test_records=1000,
        planned_test_questions=len(test_ids), completed_retrieval_questions=1000,
        planned_answer_questions=len(ids), completed_answer_questions=len(all_rows)//6,
        answer_sample_plan_sha256=sample["sample_plan_sha256"], retrieval_evaluation=retrieval_result,
        completed_variant_answers=len(all_rows), new_native_attempts_started=budget.started,
        current_segment_wall_seconds=perf_counter()-started, segment_wall_not_independent_query_latency=True,
        original_frozen_inputs_unchanged=True, error=error_text, usage=usage,
        limitations=["References and quality judges are machine-produced, not human gold or calibrated correctness.",
            "Observed identity can be unavailable; fixed requested configuration does not establish model diversity or physical model identity.",
            "Evidence is top-five; recall is against the finite judged union, not all corpus relevance.",
            "No batch time divided into per-query latency; subscription consumption has no verified dollar conversion.",
            "Answer quality covers a pre-label uniform100-of1000 TEST sample, not1000 generated answers per variant.",
            "Paired bootstrap intervals resample100 sampled queries; finite-population, event-cluster and machine-label biases are not fully represented.",
            "The max-new-call bound limits attempts per continuation, not token consumption or prepaid dollar cost."])
    if status == "completed_formal_machine_only_experiment":
        expected = {(qid, variant) for qid in ids for variant in pilot.VARIANTS}
        if {(r["record_id"], r["variant"]) for r in all_rows} != expected or len(all_rows) != len(expected):
            raise ValueError("Final answer coverage differs from the fixed100 sampled TEST records")
        report["summary"] = quality_summary(all_rows)
        by_key = {(r["record_id"], r["variant"]):r for r in all_rows}
        report["paired_automatic_correctness"] = {right + "_minus_" + left:paired_bootstrap(
            [by_key[(qid,left)]["automatic_judge"]["correctness"]/2 for qid in ids],
            [by_key[(qid,right)]["automatic_judge"]["correctness"]/2 for qid in ids])
            for left, right in (("bm25","hybrid_base"), ("dense","hybrid_base"),
                               ("hybrid_base","hybrid_A"), ("hybrid_base","hybrid_B"), ("hybrid_A","hybrid_AB"))}
        report["group_artifacts_sha256"] = {p.relative_to(run).as_posix():file_hash(p) for p in sorted((run / "groups").glob("*.json"))}
        if not (run / "result.json").exists():
            write_new(run / "result.json", report)
    write_new(segment, report)
    return run, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--corpus-run", default="semantic-page-v2-buffer")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("snapshot", help="No model call: stage six actual runtime artifacts before freezing")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--requested-model")
    prepare.add_argument("--max-cases", type=int, default=8)
    prepare.add_argument("--max-request-bytes", type=int, default=160000)
    prepare.add_argument("--answer-sample", type=Path, required=True,
        help="Immutable uniform100 sample sealed before TEST labels/attempts")
    execute = commands.add_parser("run", help="Verify >=1000 accepted frozen TEST labels before any native call")
    execute.add_argument("--frozen-dir", required=True, type=Path)
    execute.add_argument("--retrieval-run", required=True, type=Path)
    execute.add_argument("--resume", type=Path)
    execute.add_argument("--max-new-calls", type=int, default=20)
    execute.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    runtime = load_staged_runtime(args.canonical_root, args.corpus_run)
    if args.command == "snapshot":
        sample, _ = load_sample_binding(args.answer_sample, runtime.canonical_root)
        print(json.dumps(snapshot(runtime, args.output, settings(args.requested_model, args.max_cases, args.max_request_bytes, sample)), indent=2))
    else:
        output, result = experiment(runtime, args.frozen_dir, args.retrieval_run,
            runtime.canonical_root / "data/machine_runs/formal-experiments", resume=args.resume,
            max_new_calls=args.max_new_calls, retry_failed=args.retry_failed)
        print(json.dumps(dict(output=str(output), **result), indent=2))
        if result["status"] == "failed_not_complete":
            raise SystemExit(1)
