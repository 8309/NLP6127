"""Explicit, read-only audit of the reviewed two-CLI formal-run amendment.

Replays the pinned original auditor in private dependency namespaces. Only its
driver's usage-audit executable-cardinality predicate is amended; all original
coverage, request, source, model, schema, replay, ranking and quality checks stay.
No native calls, policy initialization, certificate creation or input rewriting.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
from types import ModuleType
from uuid import uuid4

import audit_formal_results as original_auditor
import run_machine_experiments as original

ROOT = Path(__file__).resolve().parents[1]
AUDITOR_SHA = "90121267e679aa67ae6d5200591083b57ae641ec3a08352e443c41afc6420e30"
DRIVER_SHA = "12fcfa2af29e1e4d0dfc8a003d2a7ae59c48b4bee1539f2433138d1057467d72"
TRANSPORT_SHA = "d07210ae96944669e2afcbb087b98c641007b7b65eeb085ab2aa6071904797f1"
REQUIRED_RETAINED = {
    "jobs/bm25-d29fa8be1f60166f7480dfe0/attempts/2026-10-14T05-31-32.560484+00-00-6444ebd2/calls/2026-10-14T05-31-32.591420+00-00-2dca0791cb/receipt.json": "38ef6423e2e4e2621c9b8c506a0aedbf751c94968b0fdc7870468a35430ba4ae",
    "jobs/judge-3b8ee06b25084f1b869e55cd/attempts/2026-10-14T07-02-52.525634+00-00-db1a7dd1/calls/2026-10-14T07-02-53.055518+00-00-7df507735a/receipt.json": "370448831ae60f531a8376785aafc830afff6f86847e3306d3603a1e7e17657d",
    "jobs/judge-f3a0cec8e85a00b32c377dbb/attempts/2026-10-14T07-22-48.794596+00-00-4e7ad2d9/calls/2026-10-14T07-22-58.913199+00-00-6c0be6e642/receipt.json": "a239bd65dab05bdb6ccd4c95644eef5c409175fcb096e0335a7e0c980ddac04b",
    "jobs/hybrid_A-a1c19dff0a7ce5491007d589/attempts/2026-10-14T08-56-26.579782+00-00-02890245/calls/2026-10-14T08-56-34.598325+00-00-2d76894d9e/receipt.json": "681aa061189def9e57145b63e63ae7d1ca9f28666e75b4fd26081c14aed28350",
}


def refuse(*args, **kwargs):
    raise ValueError("Offline rollover audit forbids native calls and experiment writes")


def clone(path, injections=None, replacements=()):
    """Isolate import dependencies, not validation predicates or source files."""
    source = Path(path).read_text(encoding="utf-8")
    for before, after in replacements:
        original_auditor.require(source.count(before) == 1, "Pinned dependency import is not unique")
        source = source.replace(before, after, 1)
    module = ModuleType("offline_rollover_" + uuid4().hex)
    module.__file__, module.__package__ = str(path), ""
    module._compiled_source = source
    module.__dict__.update(injections or {})
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    return module


def retained_pins(extra=None):
    pins = dict(REQUIRED_RETAINED)
    for name, digest in (extra or {}).items():
        original_auditor.require(name not in pins or pins[name] == digest, "Required historical receipt pin cannot change")
        pins[name] = digest
    return pins


def load_private_auditor(run, runtime, frozen, policy_sha, adapter_sha, transport_policy_sha, transport_adapter_sha):
    require = original_auditor.require
    for digest in (policy_sha, adapter_sha, transport_policy_sha, transport_adapter_sha):
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "Explicit rollover and transport policy/adapter SHA256 pins are required")
    paths = {ROOT / "scripts/audit_formal_results.py": AUDITOR_SHA,
             ROOT / "scripts/run_machine_experiments.py": DRIVER_SHA,
             ROOT / "scripts/resume_formal_completed_503.py": TRANSPORT_SHA,
             ROOT / "scripts/formal_cli_rollover.py": adapter_sha,
             run / "cli_rollover/policy.json": policy_sha,
             run / "transport_admissions/policy.json": transport_policy_sha}
    paths = {str(path):digest for path, digest in paths.items()}
    require(transport_adapter_sha == TRANSPORT_SHA, "Only the original transport adapter is allowed")
    original._assert_hashes(paths)
    plan = original.read_json(run / "plan.json")
    pilot_path = ROOT / "scripts/run_codex_ablations.py"
    require(plan["protected_sha256"].get(str(pilot_path.resolve())) == original.file_hash(pilot_path), "Pilot source lost original frozen binding")
    pilot = clone(pilot_path)
    pilot.write_new = pilot.call_structured_json = refuse
    private_driver = clone(ROOT / "scripts/run_machine_experiments.py", {"_pilot":pilot},
        [("import run_codex_ablations as pilot", "pilot = _pilot")])
    private_driver.call_structured_json = refuse
    transport = clone(ROOT / "scripts/resume_formal_completed_503.py", {"_original":private_driver},
        [("import run_machine_experiments as original", "original = _original")])
    cohort = private_driver.load_cohort(frozen, runtime)
    admission = transport.Admission(run, cohort)  # Private write hook forbids initialization.
    wrapper = clone(ROOT / "scripts/formal_cli_rollover.py")
    rollover = wrapper.ReviewedRollover(run, admission, initialize_approved=False, expected_policy_sha256=policy_sha)
    amended = rollover.driver()
    private_driver.usage_audit = amended.usage_audit  # The sole validation-policy amendment.
    auditor = clone(ROOT / "scripts/audit_formal_results.py",
        {"_driver":private_driver, "_transport":transport},
        [("import run_machine_experiments as driver", "driver = _driver"),
         ("    import resume_formal_completed_503 as adapter", "    adapter = _transport")])
    original._assert_hashes(paths)
    return auditor, rollover, paths


def audit(runtime, frozen, retrieval_run, run, *, retained_receipts=None,
          cli_rollover_policy_sha256=None, cli_rollover_adapter_sha256=None,
          transport_policy_sha256=None, transport_adapter_sha256=None):
    run = original.long_path(run)
    original_auditor.require((run / "result.json").is_file(), "No completed result.json exists; partial runs remain partial")
    original_auditor.validate_result_flags(original.read_json(run / "result.json"), original.read_json(run / "plan.json"))
    before = {str(p):original.file_hash(p) for p in run.rglob("*") if p.is_file()}
    auditor, rollover, protected = load_private_auditor(run, runtime, frozen,
        cli_rollover_policy_sha256, cli_rollover_adapter_sha256, transport_policy_sha256, transport_adapter_sha256)
    result = auditor.audit(runtime, frozen, retrieval_run, run, retained_receipts=retained_pins(retained_receipts),
        transport_policy_sha256=transport_policy_sha256, transport_adapter_sha256=transport_adapter_sha256)
    lineage = rollover.lineage_report()
    after = {str(p):original.file_hash(p) for p in run.rglob("*") if p.is_file()}
    original_auditor.require(before == after, "Offline audit changed experiment files or created a policy/certificate")
    protected[str(Path(__file__).resolve())] = original.file_hash(__file__)
    result["protected_sha256"].update(protected)
    result["protected_file_count"] = len(result["protected_sha256"])
    result.update(cli_rollover_policy_sha256=cli_rollover_policy_sha256,
        cli_rollover_adapter_sha256=cli_rollover_adapter_sha256, cli_runtime_lineage=lineage,
        original_single_cli_runtime_contract_unmodified=False, reviewed_runtime_amendment=True,
        audit_amendment="Original offline replay with only reviewed CLI-binary usage cardinality amended")
    result["limitations"].append("The runtime has two reviewed CLI binary strata, not an unmodified single-binary experiment or established model diversity.")
    original._assert_hashes(result["protected_sha256"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-root", "frozen-dir", "retrieval-run", "run"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--corpus-run", default="semantic-page-v2-buffer")
    for name in ("cli-rollover-policy-sha256", "cli-rollover-adapter-sha256", "transport-policy-sha256", "transport-adapter-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--require-retained-receipt", action="append", default=[])
    parser.add_argument("--output-new", type=Path)
    args = parser.parse_args()
    pins = {}
    for value in args.require_retained_receipt:
        name, digest = original_auditor.parse_retained_pin(value)
        original_auditor.require(name not in pins, "Duplicate additional receipt pin")
        pins[name] = digest
    runtime = original.load_staged_runtime(args.canonical_root, args.corpus_run)
    result = audit(runtime, args.frozen_dir, args.retrieval_run, args.run, retained_receipts=pins,
        cli_rollover_policy_sha256=args.cli_rollover_policy_sha256, cli_rollover_adapter_sha256=args.cli_rollover_adapter_sha256,
        transport_policy_sha256=args.transport_policy_sha256, transport_adapter_sha256=args.transport_adapter_sha256)
    if args.output_new:
        original_auditor.write_optional_new_output(args.output_new, result, args.canonical_root)
    print(json.dumps({k:v for k,v in result.items() if k != "protected_sha256"}, indent=2))


if __name__ == "__main__":
    main()
