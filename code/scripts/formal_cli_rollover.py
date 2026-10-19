"""Explicit, sealed two-binary accounting for one reviewed formal continuation.

Only the native-binary homogeneity predicate changes beyond the four historical
terminal-503 transport predicates. Model homogeneity, immutable artifacts,
stage validation and the old-binary-only 503 policy remain unchanged. Per-binary
accounting is a separate sidecar, never a physical-model diversity claim.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType
from uuid import uuid4

import resume_formal_completed_503 as transport

o = transport.original
VERSION = "formal-reviewed-two-cli-binary-rollover-v1"
OLD_BINARY = "d13914ced6c7af174d8d638db938284231227312104b06a25e9608c632a7daec"
CURRENT_BINARY = "e98c516f1c8935789aabe9122ee7e05ec4c3efd1ba6acf179242e031c495b99e"
ALLOWED_BINARIES = frozenset({OLD_BINARY, CURRENT_BINARY})
TRANSPORT_SHA = "d07210ae96944669e2afcbb087b98c641007b7b65eeb085ab2aa6071904797f1"
AUDITOR_SHA = "90121267e679aa67ae6d5200591083b57ae641ec3a08352e443c41afc6420e30"
PREDICATE = "    if len(native_binaries) > 1 or len(known_models) > 1:\n"
REPLACEMENT = "    if not _reviewed_binary_lineage(native_binaries, unique, unknown) or len(known_models) > 1:\n"


def derived_source():
    """Retain the entire pinned driver except exactly five declared predicates."""
    source = Path(o.__file__).read_text(encoding="utf-8")
    if o.file_hash(o.__file__) != transport.DRIVER_SHA:
        raise ValueError("Original formal driver changed")
    source = transport.patched_source(source)
    if source.count(PREDICATE) != 1:
        raise ValueError("Pinned native-binary predicate must occur exactly once")
    return source.replace(PREDICATE, REPLACEMENT, 1)


class ReviewedRollover:
    def __init__(self, run, admission, initialize_approved=False, expected_policy_sha256=None):
        self.run, self.admission = o.long_path(run), admission
        if o.plain_path(self.run) != o.plain_path(admission.run):
            raise ValueError("CLI rollover must belong to the exact transport admission run")
        self.plan_path = self.run / "plan.json"
        self.plan = o.read_json(self.plan_path)
        if self.plan.get("cfg", {}).get("requested_model") is not None:
            raise ValueError("Reviewed rollover preserves the original unresolved requested model")
        self.source = derived_source()
        scripts = Path(__file__).resolve().parent
        sources = {
            str(Path(__file__).resolve()): o.file_hash(__file__),
            str(Path(o.__file__).resolve()): transport.DRIVER_SHA,
            str(Path(transport.__file__).resolve()): TRANSPORT_SHA,
            str(Path(transport.certifier.__file__).resolve()): transport.CERTIFIER_SHA,
            str((scripts / "audit_formal_results.py").resolve()): AUDITOR_SHA,
        }
        self.guards = dict(admission.guards)
        self.guards.update(self.plan["protected_sha256"])
        self.guards.update(sources)
        self.guards[str(self.plan_path)] = o.file_hash(self.plan_path)
        self.guards[str(admission.policy_path)] = o.file_hash(admission.policy_path)
        o._assert_hashes(self.guards)
        self.policy = dict(
            version=VERSION,
            authorization="Direct user approval to record both CLI versions and continue while preserving existing results",
            user_authorization_confirmed=True,
            reason="Exogenous Codex app update changed the native executable; retain all outcomes without result-based selection",
            outcome_blind=True,
            source_sha256=sources,
            original_plan_sha256=o.file_hash(self.plan_path),
            frozen_manifest_sha256=self.plan["frozen_manifest_sha256"],
            answer_sample_plan_sha256=self.plan["answer_sample_plan_sha256"],
            transport_policy_sha256=o.file_hash(admission.policy_path),
            allowed_native_cli_sha256=sorted(ALLOWED_BINARIES),
            preserved_native_cli_sha256=OLD_BINARY,
            continuation_native_cli_sha256=CURRENT_BINARY,
            native_cli_version=transport.NATIVE_CLI_VERSION,
            requested_model=None,
            observed_physical_model=None,
            physical_model_diversity_claimed=False,
            quality_certified=False,
            human_gold=False,
            original_driver_file_unchanged=True,
            original_operational_receipts_unchanged=True,
            historical_transport_predicate_changes=4,
            additional_usage_predicate_changes=1,
            private_driver_sha256=o.protocol.sha256(self.source.encode("utf-8")),
            changed_predicate="Native executable homogeneity becomes the explicit two-binary allowlist; known-model homogeneity still enforced",
            unknown_or_third_binary_allowed=False,
            new_binary_completed503_admitted=False,
            known_usage_counted_once_by_execution_id=True,
            unknown_usage_is_zero=False,
            new_native_calls_for_policy=0,
        )
        self.policy_path = self.run / "cli_rollover" / "policy.json"
        if self.policy_path.exists():
            if expected_policy_sha256 is None:
                raise ValueError("Existing CLI rollover replay requires its exact policy SHA-256")
            if o.file_hash(self.policy_path) != expected_policy_sha256:
                raise ValueError("CLI rollover policy SHA-256 changed")
            if o.read_json(self.policy_path) != self.policy:
                raise ValueError("Immutable CLI rollover policy or source pins changed")
        else:
            if initialize_approved is not True or expected_policy_sha256 is not None:
                raise ValueError("Creating CLI rollover policy requires explicit reviewed initialization")
            o.write_new(self.policy_path, self.policy)
        self.guards[str(self.policy_path)] = o.file_hash(self.policy_path)
        self.verify()

    def verify(self):
        o._assert_hashes(self.guards)
        if o.read_json(self.policy_path) != self.policy:
            raise ValueError("Immutable CLI rollover policy changed")

    def _binary_lineage(self, binaries, unique, unknown):
        self.verify()
        if not binaries.issubset(ALLOWED_BINARIES):
            return False
        if any(entry["receipt"].get("native_cli_sha256") not in ALLOWED_BINARIES for entry in unique.values()):
            return False
        return all(o.read_json(Path(entry["receipt"])).get("native_cli_sha256") in ALLOWED_BINARIES for entry in unknown)

    def driver(self):
        self.verify()
        module = ModuleType("formal_reviewed_cli_rollover_" + uuid4().hex)
        module.__file__, module.__package__ = o.__file__, ""
        sys.modules[module.__name__] = module
        exec(compile(self.source, o.__file__, "exec"), module.__dict__)
        module._transport_admitted = self.admission.admitted
        module._reviewed_binary_lineage = self._binary_lineage
        return module

    def lineage_report(self):
        """Read-only sidecar: same aggregate usage plus separate binary lineages."""
        self.verify()
        aggregate = self.driver().usage_audit(self.run)
        unique, unidentified = {}, []
        for path in sorted((self.run / "jobs").glob("*/attempts/*/calls/*/receipt.json")):
            receipt = o.read_json(path)
            identity = receipt.get("execution_id")
            if not identity:
                unidentified.append(dict(receipt=str(path), native_cli_sha256=receipt["native_cli_sha256"],
                    usage=receipt.get("usage"), usage_added_to_execution_totals=False))
                continue
            unique[identity] = dict(receipt=receipt, path=str(path), sha256=o.file_hash(path))
        completed = {o.read_json(path).get("execution_id") for path in (self.run / "jobs").glob("*/completed.json")}
        rejected = {o.read_json(path).get("receipt_sha256") for path in (self.run / "jobs").glob("*/validation_rejection-*.json")}
        by_binary = {}
        for binary in sorted(ALLOWED_BINARIES):
            entries = {identity:entry for identity, entry in unique.items() if entry["receipt"]["native_cli_sha256"] == binary}
            tokens = {}
            for entry in entries.values():
                for key, value in (entry["receipt"].get("usage") or {}).items():
                    if type(value) is int and value >= 0:
                        tokens[key] = tokens.get(key, 0) + value
            by_binary[binary] = dict(
                known_native_execution_count=len(entries),
                accepted_native_executions=sum(e["receipt"].get("accepted") is True for e in entries.values()),
                rejected_native_executions=sum(e["receipt"].get("accepted") is not True for e in entries.values()),
                mechanically_rejected_native_executions=sum(e["sha256"] in rejected for e in entries.values()),
                accepted_experiment_output_executions=sum(identity in completed for identity in entries),
                executions_with_known_usage=sum(e["receipt"].get("usage") is not None for e in entries.values()),
                executions_with_unknown_usage=sum(e["receipt"].get("usage") is None for e in entries.values()),
                known_native_token_totals=tokens,
                unknown_execution_identity_receipts=[row for row in unidentified if row["native_cli_sha256"] == binary],
                observed_model_identifiers=sorted({e["receipt"]["observed_model"] for e in entries.values()
                    if e["receipt"].get("observed_model") is not None}),
                observed_physical_model=None,
                subscription_dollar_cost=None,
            )
        self.verify()
        return dict(version=VERSION, policy_sha256=o.file_hash(self.policy_path),
            private_driver_sha256=self.policy["private_driver_sha256"], aggregate_usage=aggregate,
            by_native_cli_sha256=by_binary, physical_model_diversity_claimed=False,
            human_gold=False, quality_certified=False, native_calls=0,
            unknown_execution_identity_receipts=unidentified,
            interrupted_or_precall_attempts_with_unknown_execution=aggregate["interrupted_or_precall_attempts_with_unknown_execution"],
            operational_receipts_unchanged=True, known_usage_counted_once_by_execution_id=True)
