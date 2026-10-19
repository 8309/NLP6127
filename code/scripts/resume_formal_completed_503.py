"""Resume one frozen experiment with explicit zero-call completed-503 admission.

The original driver and every operational receipt remain byte-identical. Four
source predicates in a private in-memory driver admit separately certified
terminal results. This is a recorded transport-policy extension, not native
acceptance, semantic accuracy, another generation, or a change to TEST inputs.
Every qualifying result follows the same outcome-blind mechanical policy.
Incomplete, quota, schema, tool, mixed-error and changed-input failures stop.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType
from uuid import uuid4

import run_machine_experiments as original
import certify_completed_503_result as certifier
from oil_evidence.codex_runtime import CodexUnavailable

ROOT = Path(__file__).resolve().parents[1]
DRIVER_SHA = "12fcfa2af29e1e4d0dfc8a003d2a7ae59c48b4bee1539f2433138d1057467d72"
CERTIFIER_SHA = "a7e9eb6d89ae489de6249f7e316e844aa94749d16ad55dbc058a8b480b2cce4e"
VERSION = "formal-terminal-503-outcome-blind-admission-v1"
REGISTERED_RECEIPT_SHA = "b96570b5e23f400db5945008548f9860d99dc382d6ed4d011a75b67f2d7e863a"
REGISTERED_EXECUTION = "01a13926-7397-7223-ba52-eab4932e683e"
REQUIRED_SNAPSHOT_NAMES = frozenset({"codex_runtime.py", "machine_evaluation.py", "run_codex_annotation.py"})
NATIVE_CLI_SHA = "d13914ced6c7af174d8d638db938284231227312104b06a25e9608c632a7daec"
NATIVE_CLI_VERSION = "codex-cli 0.162.0-alpha.17.2"


def patched_source(source):
    """Exactly four acceptance predicates change, all other guards remain."""
    replacements = [
        ('if (not isinstance(receipt, dict) or receipt.get("accepted") is not True\n'
         '            or receipt.get("role") != request["role"] or receipt.get("process_exit_code") != 0\n'
         '            or receipt.get("errors") or receipt.get("codex_cli_invocations") != 1',
         'if (not isinstance(receipt, dict)\n'
         '            or ((receipt.get("accepted") is not True or receipt.get("errors"))\n'
         '                and not _transport_admitted(receipt, request, cohort, run))\n'
         '            or receipt.get("role") != request["role"] or receipt.get("process_exit_code") != 0\n'
         '            or receipt.get("codex_cli_invocations") != 1'),
        ('if (events["errors"] or events["thread_id"]',
         'if ((events["errors"] and not _transport_admitted(receipt, request, cohort, run)) or events["thread_id"]'),
        ('if read_json(path).get("accepted") is True',
         'if (read_json(path).get("accepted") is True or _transport_admitted(read_json(path), request, cohort, run))'),
        ('if receipt.get("accepted") and file_hash(receipt_path)',
         'if (receipt.get("accepted") or _transport_admitted(receipt, request, cohort, run)) and file_hash(receipt_path)'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("Pinned driver predicate no longer occurs exactly once")
        source = source.replace(before, after, 1)
    return source


class Admission:
    def __init__(self, run, cohort):
        self.run, self.cohort = original.long_path(run), cohort
        self.plan = original.read_json(self.run / "plan.json")
        self.guards = dict(self.plan["protected_sha256"])
        self.guards[str(Path(original.__file__).resolve())] = DRIVER_SHA
        self.guards[str(Path(certifier.__file__).resolve())] = CERTIFIER_SHA
        self.guards[str(Path(__file__).resolve())] = original.file_hash(__file__)
        self.guards[str(self.run / "plan.json")] = original.file_hash(self.run / "plan.json")
        original._assert_hashes(self.guards)
        self.policy = dict(version=VERSION, original_driver_sha256=DRIVER_SHA,
            adapter_sha256=original.file_hash(__file__), event_proof_sha256=CERTIFIER_SHA,
            private_driver_sha256=original.protocol.sha256(patched_source(Path(original.__file__).read_text(encoding="utf-8")).encode()),
            frozen_manifest_sha256=self.plan["frozen_manifest_sha256"],
            answer_sample_plan_sha256=self.plan["answer_sample_plan_sha256"],
            registered_preexisting_execution=REGISTERED_EXECUTION,
            registered_preexisting_receipt_sha256=REGISTERED_RECEIPT_SHA,
            original_operational_acceptance_unchanged=True, new_native_calls_for_admission=0,
            automatic_native_retry=False, semantic_accuracy_claimed=False, human_gold=False,
            criteria="Exact pre-final503 warnings, one terminal raw message and completed turn, known matching usage, identical full request/schema/implementation, original stage validation",
            policy_applies_to_all_qualifying_outcomes=True)
        self.policy_path = self.run / "transport_admissions" / "policy.json"
        if self.policy_path.exists():
            if original.read_json(self.policy_path) != self.policy:
                raise ValueError("Sealed transport admission policy changed")
        else:
            original.write_new(self.policy_path, self.policy)
        self.guards[str(self.policy_path)] = original.file_hash(self.policy_path)

    def proof(self, receipt, request, cohort, run):
        original._assert_hashes(self.guards)
        if original.plain_path(run) != original.plain_path(self.run) or cohort.manifest != self.cohort.manifest:
            raise ValueError("Admission escaped its original frozen experiment")
        directory = original.long_path(receipt["artifact_directory"])
        if not original.plain_path(directory).is_relative_to(original.plain_path(self.run / "jobs")):
            raise ValueError("Admission artifact escapes the exact experiment jobs")
        receipt_path = directory / "receipt.json"
        if original.read_json(receipt_path) != receipt:
            raise ValueError("Admission must use the unmodified original receipt")
        if not REQUIRED_SNAPSHOT_NAMES.issubset(cohort.source_hashes):
            raise ValueError("Frozen cohort lacks one of the three mandatory native implementation snapshots")
        expected_snapshots = {name: cohort.source_hashes[name] for name in REQUIRED_SNAPSHOT_NAMES}
        if (receipt.get("accepted") is not False or receipt.get("role") != request["role"]
                or receipt.get("codex_cli_invocations") != 1 or receipt.get("external_paid_api_calls") != 0
                or receipt.get("model_tools_disabled") is not True or receipt.get("execpolicy_rules_not_disabled") is not True
                or receipt.get("authentication") != "existing_ChatGPT_subscription"
                or receipt.get("requested_model") != (cohort.cfg["requested_model"] or "cli_default_unresolved")
                or receipt.get("source_snapshots_sha256") != expected_snapshots
                or receipt.get("native_cli_sha256") != NATIVE_CLI_SHA
                or receipt.get("cli_version") != NATIVE_CLI_VERSION):
            raise ValueError("Failed receipt violates the unchanged native execution contract")
        job_path = directory.parents[3] / "job.json"
        descriptor = original.read_json(job_path)
        if descriptor["request"] != request or descriptor["dataset_version"] != cohort.manifest["version"]:
            raise ValueError("Full request differs from the immutable owning job")
        if descriptor["runtime_sha256"] != cohort.manifest["runtime_sha256"]:
            raise ValueError("Owning job runtime differs from frozen version")
        required_artifacts = {"stdin.txt", "schema.json", "raw_final.json", "events.jsonl"}
        required_artifacts.update("implementation/" + name for name in expected_snapshots)
        if not required_artifacts.issubset(receipt.get("artifacts_sha256", {})):
            raise ValueError("Original artifact inventory must bind full request, raw, events and implementation")
        if any(receipt["artifacts_sha256"]["implementation/" + name] != expected
                for name, expected in expected_snapshots.items()):
            raise ValueError("Original implementation snapshot bytes differ from the frozen source hashes")
        hashes = {}
        for name, expected in receipt["artifacts_sha256"].items():
            artifact = original.contained(directory, name)
            if original.file_hash(artifact) != expected:
                raise ValueError("Original native artifact bytes changed")
            hashes[str(artifact)] = expected
        stdin, schema = original.request_bytes(request), original.protocol.canonical_bytes(request["schema"])
        if ((directory / "stdin.txt").read_bytes() != stdin or (directory / "schema.json").read_bytes() != schema
                or receipt["prompt_sha256"] != original.protocol.sha256(stdin)
                or receipt["schema_sha256"] != original.protocol.sha256(schema)):
            raise ValueError("Original full stdin or schema changed")
        raw_bytes = (directory / "raw_final.json").read_bytes()
        events_bytes = (directory / "events.jsonl").read_bytes()
        terminal = certifier.event_proof(receipt, raw_bytes, events_bytes)
        if terminal["observed_model"] != receipt["observed_model"]:
            raise ValueError("Original observed identity differs from native events")
        raw = certifier.strict_json(raw_bytes)
        original.Draft202012Validator(request["schema"]).validate(raw)
        original._assert_hashes(self.guards)
        return dict(version=VERSION, policy_sha256=original.file_hash(self.policy_path),
            original_receipt=str(receipt_path), original_receipt_sha256=original.file_hash(receipt_path),
            owning_job_sha256=original.file_hash(job_path), original_artifacts_sha256=hashes,
            operational_receipt_accepted=False, human_gold=False, semantic_accuracy_claimed=False,
            new_native_calls=0, terminal_proof=terminal), raw

    def path(self, receipt):
        identity = receipt.get("execution_id")
        try:
            if str(__import__("uuid").UUID(identity)) != identity:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise ValueError("Invalid admission execution identity") from None
        return self.run / "transport_admissions" / identity / "certificate.json"

    def certify(self, receipt, request):
        certificate, _ = self.proof(receipt, request, self.cohort, self.run)
        path = self.path(receipt)
        if path.exists():
            if original.read_json(path) != certificate:
                raise ValueError("Immutable transport certificate changed")
        else:
            original.write_new(path, certificate)
        return certificate

    def admitted(self, receipt, request, cohort, run):
        if receipt.get("accepted") is not False:
            return False
        path = self.path(receipt)
        if not path.exists():
            return False
        expected, _ = self.proof(receipt, request, cohort, run)
        if original.read_json(path) != expected:
            raise ValueError("Registered certificate no longer matches exact zero-call proof")
        return True

    def caller(self, prompt, payload, schema, directory, *, role, requested_model):
        original._assert_hashes(self.guards)
        try:
            return original.call_structured_json(prompt, payload, schema, directory,
                role=role, requested_model=requested_model)
        except CodexUnavailable:
            paths = list(Path(directory).glob("*/receipt.json"))
            if len(paths) != 1:
                raise
            receipt = original.read_json(paths[0])
            request = dict(prompt=prompt, payload=payload, schema=schema, role=role)
            self.certify(receipt, request)  # No additional invocation or output selection.
            _, raw = self.proof(receipt, request, self.cohort, self.run)
            return deepcopy(raw), receipt


def driver(admission):
    source = Path(original.__file__).read_text(encoding="utf-8")
    if original.file_hash(original.__file__) != DRIVER_SHA or original.file_hash(certifier.__file__) != CERTIFIER_SHA:
        raise ValueError("Original driver or terminal-event validator changed")
    module = ModuleType("formal_completed_503_" + uuid4().hex)
    module.__file__ = original.__file__
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(patched_source(source), original.__file__, "exec"), module.__dict__)
    module._transport_admitted = admission.admitted
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--frozen-dir", required=True, type=Path)
    parser.add_argument("--retrieval-run", required=True, type=Path)
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--registered-receipt", required=True, type=Path)
    parser.add_argument("--max-new-calls", type=int, default=0)
    args = parser.parse_args()
    runtime = original.load_staged_runtime(args.canonical_root, "semantic-page-v2-buffer")
    cohort = original.load_cohort(args.frozen_dir, runtime)
    registered = original.long_path(args.registered_receipt)
    if original.file_hash(registered) != REGISTERED_RECEIPT_SHA:
        raise ValueError("Only the exact reviewed preexisting completed503 receipt may enter this recovery")
    receipt = original.read_json(registered)
    if receipt["execution_id"] != REGISTERED_EXECUTION:
        raise ValueError("Registered preexisting native identity changed")
    admission = Admission(args.resume, cohort)
    owning_job = original.read_json(registered.parents[4] / "job.json")
    admission.certify(receipt, owning_job["request"])
    staged = driver(admission)
    run, result = staged.experiment(runtime, args.frozen_dir, args.retrieval_run,
        runtime.canonical_root / "data/machine_runs/formal-experiments", resume=args.resume,
        max_new_calls=args.max_new_calls, retry_failed=False, caller=admission.caller)
    original._assert_hashes(admission.guards)
    certificates = sorted((admission.run / "transport_admissions").glob("*/certificate.json"))
    sidecar = dict(version=VERSION, original_result_status=result["status"],
        policy=str(admission.policy_path), policy_sha256=original.file_hash(admission.policy_path),
        admitted_completed503_executions=len(certificates),
        certificates_sha256={str(p.relative_to(admission.run)):original.file_hash(p) for p in certificates},
        original_operational_receipts_stay_false=True, usage_original_once_counted=True,
        native_retries=0, original_driver_unchanged=True, human_gold=False,
        completed_question_groups=result["completed_answer_questions"], completed_variants=result["completed_variant_answers"])
    original.write_new(admission.run / "transport_admissions" / ("segment-" + uuid4().hex + ".json"), sidecar)
    print(json.dumps(dict(output=str(run), **result, transport_admission=sidecar), indent=2))
    return 1 if result["status"] == "failed_not_complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
