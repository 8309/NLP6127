"""Read-only, fail-closed audit of a completed frozen machine experiment.

This auditor never invokes a native model, reranks, or writes into an experiment.
It reconstructs the exact saved jobs and groups from the frozen inputs, validates
their retained native artifacts, and independently recomputes the reported
machine metrics. Synthetic fixture tests are not evidence of actual quality.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import json

import run_machine_experiments as driver

COMPARISONS = (("bm25", "hybrid_base"), ("dense", "hybrid_base"),
               ("hybrid_base", "hybrid_A"), ("hybrid_base", "hybrid_B"),
               ("hybrid_A", "hybrid_AB"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_result_flags(result, plan):
    require(result.get("status") == "completed_formal_machine_only_experiment",
            "An actual completed result.json is required; partial segments are not final results")
    require(result.get("formal_heldout_test") is True and result.get("human_gold") is False,
            "Formal held-out status must be explicit and human_gold must remain false")
    expected = dict(required_test_records=1000, planned_test_questions=1000,
                    completed_retrieval_questions=1000, planned_answer_questions=100,
                    completed_answer_questions=100, completed_variant_answers=600,
                    original_frozen_inputs_unchanged=True,
                    segment_wall_not_independent_query_latency=True, error=None,
                    dataset_version=plan["dataset_version"],
                    answer_sample_plan_sha256=plan["answer_sample_plan_sha256"])
    require(all(result.get(key) == value and type(result.get(key)) is type(value) for key, value in expected.items()),
            "Final result coverage, sample, dataset or immutable-input flags differ")
    require(len(plan.get("all_test_record_ids", [])) == 1000
            and len(set(plan["all_test_record_ids"])) == 1000
            and len(plan.get("answer_record_ids", [])) == 100
            and len(set(plan["answer_record_ids"])) == 100
            and set(plan["answer_record_ids"]).issubset(plan["all_test_record_ids"])
            and plan.get("expected_variant_answers") == 600
            and plan.get("human_gold") is False,
            "The original plan must bind unique 1000 TEST IDs and unique 100 sampled IDs")


def audit_groups(run, plan, result):
    """Require exact ordered groups, hashes, unique IDs and all 600 variants."""
    run = Path(run)
    ids, size = plan["answer_record_ids"], plan["cfg"]["max_cases"]
    expected_files = {f"groups/{offset:06}.json" for offset in range(0, 100, size)}
    actual_files = {p.relative_to(run).as_posix() for p in (run / "groups").glob("*.json")}
    require(actual_files == expected_files, "Missing, duplicate or unexpected final group files")
    hashes = result.get("group_artifacts_sha256")
    require(isinstance(hashes, dict) and set(hashes) == expected_files,
            "Final group_artifacts_sha256 must bind every and only the expected groups")
    rows, seen = [], set()
    for offset in range(0, 100, size):
        name = f"groups/{offset:06}.json"
        path = driver.contained(run, name)
        require(driver.file_hash(path) == hashes[name], "Final group bytes changed")
        group = driver.read_json(path)
        group_ids = ids[offset:offset + size]
        require(set(group) == {"record_ids", "rows"} and group["record_ids"] == group_ids,
                "Final group IDs/order differ from the original sampled plan")
        expected_order = [(qid, variant) for qid in group_ids for variant in driver.pilot.VARIANTS]
        current = group.get("rows")
        require(isinstance(current, list) and len(current) == len(expected_order),
                "Final group must contain every planned question/variant row")
        keys = [(row.get("record_id"), row.get("variant")) for row in current]
        require(keys == expected_order and not seen.intersection(keys) and len(set(keys)) == len(keys),
                "Duplicate, omitted, extra or reordered final question/variant rows")
        seen.update(keys)
        rows.extend(current)
    require(len(rows) == 600 and seen == {(qid, v) for qid in ids for v in driver.pilot.VARIANTS},
            "Final rows do not have exact 100 by six coverage")
    return rows


def recompute_quality(rows, ids):
    by_key = {(row["record_id"], row["variant"]): row for row in rows}
    summary = driver.quality_summary(rows)
    paired = {right + "_minus_" + left: driver.paired_bootstrap(
        [by_key[(qid, left)]["automatic_judge"]["correctness"] / 2 for qid in ids],
        [by_key[(qid, right)]["automatic_judge"]["correctness"] / 2 for qid in ids])
        for left, right in COMPARISONS}
    return summary, paired


@contextmanager
def read_only_stage_replay():
    """Invalid replay may request a rejection file; forbid that side effect."""
    original = driver.pilot.write_new
    def reject_write(*args, **kwargs):
        raise ValueError("Offline stage validation rejected an output; auditor writes are forbidden")
    driver.pilot.write_new = reject_write
    try:
        yield
    finally:
        driver.pilot.write_new = original


class OfflineJobs:
    def __init__(self, run, cohort, admission=None, receipt_validator=None):
        self.run, self.cohort = Path(run), cohort
        self.seen, self.protected = set(), {}
        self.failed_receipts, self.attempts = [], 0
        self.admission = admission
        self.receipt_validator = receipt_validator or driver.validate_receipt
        self.admitted_certificates = {}
        self.certified_output_executions = set()

    def replay(self, stage, ids, request, skipped, process):
        key = dict(stage=stage, item_ids=ids, request=request,
                   dataset_version=self.cohort.manifest["version"],
                   runtime_sha256=self.cohort.manifest["runtime_sha256"])
        job_id = stage + "-" + driver.protocol.sha256(driver.protocol.canonical_bytes(key))[:24]
        require(job_id not in self.seen, "A final immutable job is used more than once")
        self.seen.add(job_id)
        directory = (self.run / "jobs" / job_id).resolve()
        require(directory.is_dir() and driver.plain_path(directory).is_relative_to(driver.plain_path(self.run / "jobs")),
                "Expected immutable job directory is missing or outside this run")
        descriptor, completed = directory / "job.json", directory / "completed.json"
        require(descriptor.is_file() and completed.is_file(), "Final run contains an incomplete expected job")
        require(driver.read_json(descriptor) == key, "Job request differs from its frozen reconstructed inputs")
        record = driver.read_json(completed)
        self.protected[str(descriptor)] = driver.file_hash(descriptor)
        self.protected[str(completed)] = driver.file_hash(completed)
        binding = record.get("artifacts_sha256", {})
        require(isinstance(binding, dict) and binding.get("job.json") == driver.file_hash(descriptor)
                and isinstance(record.get("result"), str) and record["result"] in binding,
                "Completed job must hash-bind its descriptor and exact result")
        for name, expected in binding.items():
            path = driver.contained(directory, name)
            require(driver.file_hash(path) == expected, "Completed job artifact changed")
            self.protected[str(path)] = expected
        rejected = set()
        for path in directory.glob("validation_rejection-*.json"):
            rejection = driver.read_json(path)
            retained = Path(rejection["retained_receipt"]).resolve()
            require(retained.is_relative_to(directory.resolve()) and retained.is_file()
                    and driver.file_hash(retained) == rejection.get("receipt_sha256")
                    and rejection.get("automatic_retry") is False
                    and rejection.get("mechanical_output_rejection_not_answer_quality_selection") is True,
                    "Mechanical rejection must retain its exact receipt without quality selection")
            rejected.add(rejection["receipt_sha256"])
            self.protected[str(path)] = driver.file_hash(path)
        receipts = []
        attempt_directories = sorted((directory / "attempts").glob("*"))
        for attempt in attempt_directories:
            require(attempt.is_dir() and (attempt / "request.json").is_file(), "Unbound native attempt")
            require(driver.read_json(attempt / "request.json") == request, "Retained attempt request changed")
            self.attempts += 1
            self.protected[str(attempt / "request.json")] = driver.file_hash(attempt / "request.json")
            failure = attempt / "runner_failure.json"
            if failure.exists():
                require(driver.read_json(failure).get("automatic_retry") is False,
                        "Failure history must not silently enable automatic retry")
                self.protected[str(failure)] = driver.file_hash(failure)
            for path in sorted((attempt / "calls").glob("*/receipt.json")):
                receipt = driver.read_json(path)
                require(Path(receipt.get("artifact_directory", "")).resolve() == path.parent.resolve(),
                        "Retained receipt must identify its own artifact directory")
                for name, expected in receipt.get("artifacts_sha256", {}).items():
                    artifact = driver.contained(path.parent, name)
                    require(driver.file_hash(artifact) == expected, "Retained attempt artifact changed")
                    self.protected[str(artifact)] = expected
                self.protected[str(path)] = driver.file_hash(path)
                admitted = False
                if receipt.get("accepted") is False and self.admission is not None:
                    admitted = self.admission.admitted(receipt, request, self.cohort, self.run)
                    if admitted:
                        certificate_path = self.admission.path(receipt)
                        self.admitted_certificates[str(certificate_path)] = driver.file_hash(certificate_path)
                        self.protected[str(certificate_path)] = driver.file_hash(certificate_path)
                if receipt.get("accepted") is not True:
                    self.failed_receipts.append(dict(path=str(path), sha256=driver.file_hash(path),
                        execution_id=receipt.get("execution_id"), usage=receipt.get("usage"),
                        explicit_completed_transport_certificate=admitted))
                    require(failure.exists() or admitted,
                            "A failed native receipt needs a retained runner failure or exact transport certificate")
                if (receipt.get("accepted") is True or admitted) and driver.file_hash(path) not in rejected:
                    receipts.append((path, receipt))
        if request is None:
            require(not receipts and not attempt_directories and record.get("model_calls") == 0
                    and record.get("result") == "no_call_result.json", "No-claim reuse must not invoke a model")
            expected_result = skipped
        else:
            require(len(receipts) == 1, "Completed native job needs one accepted or explicitly certified, nonrejected receipt")
            path, receipt = receipts[0]
            if receipt.get("accepted") is False:
                self.certified_output_executions.add(receipt["execution_id"])
            receipt_name = path.relative_to(directory).as_posix()
            require(record.get("execution_id") == receipt.get("execution_id")
                    and binding.get(receipt_name) == driver.file_hash(path),
                    "Completed result is not bound to its selected native execution")
            require(set(receipt.get("artifacts_sha256", {})) >=
                    {"stdin.txt", "schema.json", "raw_final.json", "events.jsonl", "stderr.txt"},
                    "Accepted native receipt is missing mandatory raw artifacts")
            require(all(binding.get((path.parent.relative_to(directory) / name).as_posix()) == expected
                        for name, expected in receipt["artifacts_sha256"].items()),
                    "Completed job must hash-bind every selected native raw artifact")
            raw = self.receipt_validator(receipt, request, self.cohort, self.run)
            def caller(prompt, payload, schema, unused, *, role):
                require(dict(prompt=prompt, payload=payload, schema=schema, role=role) == request,
                        "Offline stage request differs from the immutable saved request")
                return deepcopy(raw), receipt
            expected_result = process(caller)
        actual = driver.read_json(driver.contained(directory, record["result"]))
        require(actual == expected_result, "Saved derived job result differs from strict raw-output replay")
        return actual

    def finish(self):
        directories = {p.name for p in (self.run / "jobs").iterdir() if p.is_dir()}
        require(directories == self.seen, "Unexpected, duplicated or incomplete jobs remain outside final replay")
        self.protected.update({str(path): driver.file_hash(path)
                               for path in (self.run / "jobs").rglob("*") if path.is_file()})
        self.protected.update({str(path): driver.file_hash(path)
                               for path in (self.run / "segments").glob("*.json")})
        if self.admission is not None:
            actual = {str(path) for path in (self.run / "transport_admissions").glob("*/certificate.json")}
            require(actual == set(self.admitted_certificates),
                    "Transport certificates must exactly equal the fully replayed admitted output executions")
            self.protected.update(self.admission.guards)
            for path in (self.run / "transport_admissions").glob("segment-*.json"):
                sidecar = driver.read_json(path)
                require(sidecar.get("version") == self.admission.policy["version"]
                        and sidecar.get("policy_sha256") == driver.file_hash(self.admission.policy_path)
                        and sidecar.get("original_operational_receipts_stay_false") is True
                        and sidecar.get("usage_original_once_counted") is True
                        and sidecar.get("native_retries") == 0
                        and sidecar.get("human_gold") is False,
                        "Transport segment sidecar changes native acceptance, policy, usage or provenance")
                for name, expected in sidecar.get("certificates_sha256", {}).items():
                    certificate = driver.contained(self.run, name.replace("\\", "/"))
                    require(driver.file_hash(certificate) == expected,
                            "Retained transport segment certificate binding changed")
                self.protected[str(path)] = driver.file_hash(path)


def reconstruct_rows(run, cohort, rankings, timing, admission=None, receipt_validator=None):
    """Replay the pinned driver pipeline without experiment(), job(), or writes."""
    jobs, all_rows = OfflineJobs(run, cohort, admission, receipt_validator), []
    ids = cohort.cfg["answer_sample_binding"]["plan"]["answer_record_ids"]
    with read_only_stage_replay():
        for offset in range(0, 100, cohort.cfg["max_cases"]):
            group_ids = ids[offset:offset + cohort.cfg["max_cases"]]
            cases = {qid: cohort.questions[qid] for qid in group_ids}
            evidence = {variant: {qid: rankings[qid]["conditions"][condition] for qid in group_ids}
                        for variant, condition, _ in driver.DEFINITIONS}
            answers, removed = {}, {}
            for stage, _, _ in driver.DEFINITIONS:
                builder = lambda subset: driver.capture_request(stage, {q: cases[q] for q in subset},
                                                                 {q: evidence[stage][q] for q in subset})[0]
                answers[stage] = {}
                for subset in driver.bounded_batches(group_ids, builder, cohort.cfg):
                    selected, hits = {q: cases[q] for q in subset}, {q: evidence[stage][q] for q in subset}
                    request, skipped = driver.capture_request(stage, selected, hits)
                    process = lambda call: driver.stage_result(stage, selected, hits, None, None, None,
                                                               Path("offline-no-writes"), call)
                    answers[stage].update(jobs.replay(stage, subset, request, skipped, process)["answers"])
                removed[stage] = {qid: [] for qid in group_ids}
            for original, stage in (("hybrid_base", "hybrid_B"), ("hybrid_A", "hybrid_AB")):
                evidence[stage], answers[stage], removed[stage] = evidence[original], {}, {}
                builder = lambda subset: driver.capture_request(stage, {q: cases[q] for q in subset},
                    {q: evidence[original][q] for q in subset}, {q: answers[original][q] for q in subset})[0]
                for subset in driver.bounded_batches(group_ids, builder, cohort.cfg):
                    selected, hits, originals = ({q: cases[q] for q in subset},
                        {q: evidence[original][q] for q in subset}, {q: answers[original][q] for q in subset})
                    request, skipped = driver.capture_request(stage, selected, hits, originals)
                    process = lambda call: driver.stage_result(stage, selected, hits, originals, None, None,
                                                               Path("offline-no-writes"), call)
                    checked = jobs.replay(stage, subset, request, skipped, process)
                    answers[stage].update(checked["answers"])
                    removed[stage].update(checked["removed_claims"])
            rows = {}
            for qid in group_ids:
                for stage in driver.pilot.VARIANTS:
                    hits, answer = evidence[stage][qid], answers[stage][qid]
                    by_id = {hit["passage_id"]: hit for hit in hits}
                    rows[stage + "__" + qid] = dict(record_id=qid, variant=stage,
                        question=cases[qid]["question"], as_of=cases[qid]["as_of"], answer=answer,
                        evidence=hits, removed_claims=removed[stage][qid],
                        generation_reused_from={"hybrid_B": "hybrid_base", "hybrid_AB": "hybrid_A"}.get(stage),
                        retrieval_reused_from=timing["run"], per_query_generation_seconds=None,
                        per_query_total_seconds=None, per_query_retrieval_seconds=None,
                        future_evidence_count=sum(not h.get("available_by") or h["available_by"] > cases[qid]["as_of"] for h in hits),
                        ineligible_citation_count=sum(not by_id[pid].get("available_by") or by_id[pid]["available_by"] > cases[qid]["as_of"]
                            for claim in answer["claims"] for pid in claim["citations"]),
                        reference_answerability=cohort.references[qid]["label"]["answerability"],
                        retrieval_metrics=driver.ranked_metrics([h["passage_id"] for h in hits],
                                                               cohort.references[qid]["label"]["qrels"], 5))
            builder = lambda subset: driver.capture_request("judge", cases,
                rows={key: rows[key] for key in subset}, references=cohort.references)[0]
            judgments = {}
            for subset in driver.bounded_batches(list(rows), builder, cohort.cfg):
                selected_rows = {key: rows[key] for key in subset}
                request, skipped = driver.capture_request("judge", cases, rows=selected_rows, references=cohort.references)
                process = lambda call: driver.stage_result("judge", cases, None, None, selected_rows,
                                                           cohort.references, Path("offline-no-writes"), call)
                judgments.update(jobs.replay("judge", subset, request, skipped, process)["judgments"])
            for key, row in rows.items():
                row["automatic_judge"] = judgments[key]
            all_rows.extend(rows.values())
    jobs.finish()
    return all_rows, jobs


def audit_retrieval(run, cohort, rankings, final):
    path, receipt_path = Path(run) / "retrieval_evaluation.json", Path(run) / "retrieval_evaluation_receipt.json"
    receipt = driver.read_json(receipt_path)
    binding = driver.protocol.sha256(driver.protocol.canonical_bytes(dict(
        all_test_ids=list(cohort.questions), references={qid: row["label"] for qid, row in cohort.references.items()},
        rankings={qid: rankings[qid] for qid in cohort.questions}, runtime_sha256=cohort.manifest["runtime_sha256"])))
    require(receipt == dict(input_sha256=binding, result_sha256=driver.file_hash(path),
            completed_test_questions=1000, ranked_query_conditions=6000, human_gold=False, native_calls=0),
            "Full retrieval receipt, hash, input binding or coverage changed")
    require(driver.read_json(path) == driver.full_retrieval_report(cohort, rankings),
            "Full 1000 by six retrieval scores differ from offline recomputation")
    require(final.get("retrieval_evaluation") == dict(path=str(path), sha256=driver.file_hash(path),
            test_questions=1000, ranked_query_conditions=6000), "Final result does not bind the full retrieval artifact")
    return {str(path): driver.file_hash(path), str(receipt_path): driver.file_hash(receipt_path)}


def load_transport_admission(run, cohort, policy_sha256=None, adapter_sha256=None):
    """Opt-in only: verify a preexisting sealed policy and prohibit new sidecars."""
    require((policy_sha256 is None) == (adapter_sha256 is None),
            "Transport audit requires both explicit policy and adapter SHA256 pins")
    if policy_sha256 is None:
        return None, driver.validate_receipt
    policy_path = Path(run) / "transport_admissions/policy.json"
    adapter_path = driver.ROOT / "scripts/resume_formal_completed_503.py"
    require(policy_path.is_file() and driver.file_hash(policy_path) == policy_sha256,
            "Transport audit requires the exact preexisting policy; it cannot initialize or certify one")
    require(adapter_path.is_file() and driver.file_hash(adapter_path) == adapter_sha256,
            "Transport adapter source differs from its explicit immutable pin")
    import resume_formal_completed_503 as adapter
    require(driver.file_hash(adapter.__file__) == adapter_sha256,
            "Imported transport adapter does not match the verified source")
    with read_only_stage_replay():
        admission = adapter.Admission(run, cohort)
        adapted = adapter.driver(admission)
    require(driver.file_hash(policy_path) == policy_sha256,
            "Transport policy changed during read-only adapter initialization")
    return admission, adapted.validate_receipt


def audit(runtime, frozen, retrieval_run, run, *, retained_receipts=None,
          transport_policy_sha256=None, transport_adapter_sha256=None):
    """Verify final files only; no model call, resume, or artifact adoption."""
    run = driver.long_path(run)
    require(driver.plain_path(run).is_relative_to(runtime.canonical_root.resolve() / "data/machine_runs/formal-experiments"),
            "Final audit run must remain inside canonical formal-experiments")
    result_path, plan_path = run / "result.json", run / "plan.json"
    require(result_path.is_file(), "No completed result.json exists; audit must not promote partial progress")
    result, plan = driver.read_json(result_path), driver.read_json(plan_path)
    validate_result_flags(result, plan)
    cohort = driver.load_cohort(frozen, runtime)
    rankings, rank_protected, timing = driver.load_rankings(retrieval_run, cohort, runtime)
    cohort.protected.update(runtime.protected_sha256)
    cohort.protected.update(rank_protected)
    require(plan.get("protected_sha256") == cohort.protected, "Original protected file set/hash bindings changed")
    expected_plan = dict(contract=driver.CONTRACT, dataset_version=cohort.manifest["version"],
        formal_minimum=1000, human_gold=False, all_test_record_ids=list(cohort.questions), questions=1000,
        answer_record_ids=cohort.cfg["answer_sample_binding"]["plan"]["answer_record_ids"], answer_questions=100,
        answer_sample_plan_sha256=cohort.cfg["answer_sample_binding"]["sample_plan_sha256"],
        expected_variant_answers=600, planned_base_generation_answers=400,
        unbatched_native_execution_upper_bound_without_retries=1200,
        batching_reduces_invocations_not_evidence_or_question_coverage=True,
        runtime_sha256=cohort.manifest["runtime_sha256"], cfg=cohort.cfg, retrieval=timing,
        frozen_manifest_sha256=driver.file_hash(cohort.frozen / "manifest.json"), protected_sha256=cohort.protected)
    require(plan == expected_plan, "Original plan differs from the exact frozen sample, runtime and rankings")
    protected = dict(cohort.protected)
    protected.update({str(result_path): driver.file_hash(result_path), str(plan_path): driver.file_hash(plan_path)})
    protected.update(audit_retrieval(run, cohort, rankings, result))
    rows = audit_groups(run, plan, result)
    admission, receipt_validator = load_transport_admission(run, cohort,
        transport_policy_sha256, transport_adapter_sha256)
    replayed, jobs = reconstruct_rows(run, cohort, rankings, timing, admission, receipt_validator)
    require(rows == replayed, "Final groups differ from source-bound native raw results and exact reuse")
    summary, paired = recompute_quality(rows, plan["answer_record_ids"])
    require(result.get("summary") == summary and result.get("paired_automatic_correctness") == paired,
            "Final machine summary or paired bootstrap differs from independent offline recomputation")
    usage = driver.usage_audit(run)
    require(result.get("usage") == usage, "Final once-counted native usage ledger differs or excludes retained attempts")
    protected.update(jobs.protected)
    protected.update({str(run / name): digest for name, digest in result["group_artifacts_sha256"].items()})
    for name, expected in (retained_receipts or {}).items():
        path = driver.contained(run, name)
        require(path.name == "receipt.json" and path.is_file() and driver.file_hash(path) == expected,
                "A required historical native receipt is missing or changed")
        protected[str(path)] = expected
    driver._assert_hashes(protected)
    return dict(status="completed_formal_result_offline_audit_passed", human_gold=False,
        formal_heldout_test=True, native_calls=0, ranked_test_questions=1000,
        ranked_query_conditions=6000, answer_questions=100, variant_answers=600,
        unique_completed_jobs=len(jobs.seen), retained_native_attempts=jobs.attempts,
        transport_policy_sha256=transport_policy_sha256,
        certified_transport_output_executions=len(jobs.certified_output_executions),
        verified_transport_certificates=len(jobs.admitted_certificates),
        original_operational_receipts_unchanged=True,
        failed_receipts=jobs.failed_receipts, protected_file_count=len(protected),
        protected_sha256=protected, result_sha256=driver.file_hash(result_path),
        plan_sha256=driver.file_hash(plan_path), summary=summary,
        paired_automatic_correctness=paired, usage=usage,
        limitations=["This is a byte, provenance, coverage and metric audit, not human quality validation.",
            "Native accepted=false receipts remain false; transport results require explicit pinned preexisting certificates.",
            "Known usage is once-counted; unknown usage and subscription dollar conversion remain unknown."])


def parse_retained_pin(value):
    name, separator, digest = value.rpartition("=")
    require(separator and name and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "Required retained receipt must be relative/path/receipt.json=lowercase_sha256")
    return name, digest


def write_optional_new_output(path, value, canonical_root):
    path = Path(path).resolve()
    allowed = (driver.ROOT / "tmp").resolve(), (Path(canonical_root) / "data/machine_runs/engineering-checkpoints").resolve()
    require(any(path.is_relative_to(base) and path != base for base in allowed),
            "Optional audit output must be a new ignored tmp or engineering-checkpoint file")
    require(not path.exists(), "Never overwrite an existing audit or experiment artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(driver.protocol.canonical_bytes(value) + b"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--corpus-run", default="semantic-page-v2-buffer")
    parser.add_argument("--frozen-dir", required=True, type=Path)
    parser.add_argument("--retrieval-run", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--require-retained-receipt", action="append", default=[], metavar="PATH=SHA256")
    parser.add_argument("--transport-policy-sha256")
    parser.add_argument("--transport-adapter-sha256")
    parser.add_argument("--output-new", type=Path)
    args = parser.parse_args()
    pins = {}
    for value in args.require_retained_receipt:
        name, digest = parse_retained_pin(value)
        require(name not in pins, "Duplicate retained receipt pin")
        pins[name] = digest
    runtime = driver.load_staged_runtime(args.canonical_root, args.corpus_run)
    result = audit(runtime, args.frozen_dir, args.retrieval_run, args.run, retained_receipts=pins,
        transport_policy_sha256=args.transport_policy_sha256,
        transport_adapter_sha256=args.transport_adapter_sha256)
    if args.output_new:
        write_optional_new_output(args.output_new, result, args.canonical_root)
    public = {key: value for key, value in result.items() if key != "protected_sha256"}
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()
