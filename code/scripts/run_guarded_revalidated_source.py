"""Extra durable-progress checks around unchanged source-review invocations.

Prepared pipeline manifests and all original code/receipts remain immutable.
After a valid return, a concurrent failure is deferred until its original
bounded retainer has saved the output; the next call or chunk boundary stops.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import run_codex_annotation as native
from oil_evidence import machine_evaluation as protocol

VERSION = "per-call-durable-source-failure-guard-v1"
AUTH_VERSION = "guarded-source-continuation-authorization-v1"
PIPELINE_VERSION = "fixed-order-certificate-revalidated-body-source-v1"
PIPELINE_SHA256 = "cfbe817f7e779e41fe6d9c6cb6b26cf35161f58e959775c420ee1edb7adabeed"


class ProgressGuard:
    def __init__(self, refresh, caller, *, journal=None, max_calls=620):
        if type(max_calls) is not int or max_calls <= 0:
            raise ValueError("A positive explicit native-call ceiling is required")
        self.refresh, self.caller, self.journal = refresh, caller, journal
        self.max_calls, self.calls, self.prefix, self.pending = max_calls, 0, b"", None
        if journal is not None and Path(journal).exists():
            rows = protocol.parse_jsonl(Path(journal).read_bytes(), "guarded native invocation journal")
            starts = [row for row in rows if row.get("type") == "native_start"]
            returns = [row for row in rows if row.get("type") == "native_return"]
            if (any(row.get("type") in {"native_raised", "deferred_stop"} for row in rows)
                    or [row["number"] for row in starts] != list(range(1, len(starts) + 1))
                    or [row["number"] for row in returns] != list(range(1, len(starts) + 1))):
                raise ValueError("An interrupted or stopped guarded attempt cannot be replayed")
            self.calls = len(starts)
            if returns:
                self.prefix = bytes.fromhex(returns[-1]["durable_prefix_hex"])
        self.initial_calls = self.calls

    def record(self, value):
        if self.journal is not None:
            Path(self.journal).parent.mkdir(parents=True, exist_ok=True)
            native.append_progress(Path(self.journal), value)

    def check(self):
        if self.pending is not None:
            raise ValueError("A durable producer failure was observed after a completed source call: " + self.pending)
        prefix, entries = self.refresh()
        if not prefix.startswith(self.prefix):
            raise ValueError("Producer durable progress was rewritten between native-call checks")
        self.prefix = prefix
        return prefix

    def __call__(self, *args, **kwargs):
        self.check()
        if self.calls >= self.max_calls:
            raise ValueError("The explicitly authorized source native-call ceiling was reached")
        self.calls += 1
        self.record(dict(type="native_start", number=self.calls,
            durable_prefix_sha256=protocol.sha256(self.prefix), original_inputs_schema_model_unchanged=True))
        try:
            value = self.caller(*args, **kwargs)
        except Exception as error:
            self.record(dict(type="native_raised", number=self.calls, error=str(error),
                automatic_retry=False, failed_usage_status="unknown_unless_original_receipt_measured"))
            raise
        # Do not throw here: the unchanged bounded retainer must first bind this
        # actual successful return. Its next caller or our chunk boundary stops.
        try:
            self.check()
        except ValueError as error:
            self.pending = str(error)
            self.record(dict(type="deferred_stop", number=self.calls, error=self.pending,
                completed_native_output_must_be_bound=True, automatic_retry=False))
        receipt = value[1]
        self.record(dict(type="native_return", number=self.calls, execution_id=receipt.get("execution_id"),
            usage=receipt.get("usage"), durable_prefix_hex=self.prefix.hex(),
            original_native_receipt_unchanged=True, producer_stop_deferred=self.pending is not None))
        return value


def wrap_base(base, holder, *, journal=None, max_calls=620, protected=None):
    original = base.adjudicate_chunk
    def refresh():
        native.protected_unchanged(protected or {})
        return base.progress_snapshot(holder["context"]["run"])
    guard = ProgressGuard(refresh,
        base.runtime.call_structured_json, journal=journal, max_calls=max_calls)
    def adjudicate(chunk, directory, **kwargs):
        guard.check()
        caller = kwargs.pop("caller", None)
        if caller is not None:
            guard.caller = caller
        result = original(chunk, directory, caller=guard, **kwargs)
        guard.check()
        return result
    base.adjudicate_chunk = adjudicate
    return base, guard


def call_counts(guards):
    return dict(new_native_calls_this_guard=sum(guard.calls - guard.initial_calls for guard in guards),
        previous_native_calls_retained=sum(guard.initial_calls for guard in guards),
        total_native_calls_in_guard_journal=sum(guard.calls for guard in guards))


def run(root, pool_dir, annotation_run, *, previous_review_run, source_completion_directory,
        resume, authorization, guard_state_directory, max_idle_seconds=21600):
    pipeline_path = Path(__file__).with_name("pipeline_revalidated_body_adjudication.py")
    raw = pipeline_path.read_bytes()
    if protocol.sha256(raw) != PIPELINE_SHA256:
        raise ValueError("Prepared pipeline implementation changed")
    pipeline = ModuleType("guarded_revalidated_private_" + uuid4().hex)
    pipeline.__file__ = str(pipeline_path)
    exec(compile(raw, str(pipeline_path), "exec"), pipeline.__dict__)
    if pipeline.VERSION != PIPELINE_VERSION:
        raise ValueError("Only the pinned prepared source protocol may be guarded")
    resume, authorization, state, source = map(lambda value: Path(value).resolve(),
        (resume, authorization, guard_state_directory, annotation_run))
    manifest_path = resume / "manifest.json"
    manifest = native.read_json(manifest_path)
    if manifest.get("version") != PIPELINE_VERSION or Path(manifest.get("source_annotation_run", "")).resolve() != source:
        raise ValueError("Only the exact prepared source pipeline and annotation run may continue")
    auth = native.read_json(authorization)
    required = dict(version=AUTH_VERSION, root_authorized=True, guard_helper_sha256=protocol.sha256(Path(__file__).read_bytes()),
        pipeline_sha256=PIPELINE_SHA256, prepared_pipeline_directory=str(resume),
        prepared_pipeline_manifest_sha256=protocol.sha256(manifest_path.read_bytes()), annotation_run=str(source),
        guard_state_directory=str(state), native_inputs_schema_model_unchanged=True, new_failures_stop=True)
    if (any(auth.get(key) != value or type(auth.get(key)) is not type(value) for key, value in required.items())
            or type(auth.get("max_new_native_calls")) is not int or not 1 <= auth["max_new_native_calls"] <= 620
            or state.is_relative_to(resume) or state.is_relative_to(source)):
        raise ValueError("A separately bound external guarded-continuation authorization is required")
    protected = {str(pipeline_path.resolve()): PIPELINE_SHA256, str(Path(__file__).resolve()): required["guard_helper_sha256"],
        str(manifest_path): required["prepared_pipeline_manifest_sha256"], str(authorization): protocol.sha256(authorization.read_bytes())}
    expected = dict(version=VERSION, authorization=str(authorization), authorization_sha256=protected[str(authorization)],
        prepared_pipeline_manifest_sha256=required["prepared_pipeline_manifest_sha256"], original_pipeline_sha256=PIPELINE_SHA256,
        guard_helper_sha256=required["guard_helper_sha256"], native_inputs_schema_model_unchanged=True)
    receipt_path = state / "guard_receipt.json"
    if receipt_path.exists():
        if native.read_json(receipt_path) != expected or (state / "failure.json").exists():
            raise ValueError("An altered or failed external guard cannot automatically resume")
    elif state.exists():
        raise ValueError("External guard state must be fresh or exactly bound")
    else:
        native.write_new(receipt_path, expected)
    protected[str(receipt_path)] = protocol.sha256(receipt_path.read_bytes())
    native.protected_unchanged(protected)
    original_configure, guards = pipeline.configure, []
    def configured(*args, **kwargs):
        base, holder = original_configure(*args, **kwargs)
        base, guard = wrap_base(base, holder, journal=state / "invocations.jsonl", max_calls=auth["max_new_native_calls"],
            protected=protected)
        guards.append(guard)
        return base, holder
    pipeline.configure = configured  # Private module only; prepared source bytes stay intact.
    try:
        output, result = pipeline.run(root, pool_dir, source, previous_review_run=previous_review_run,
            source_completion_directory=source_completion_directory, resume=resume, max_idle_seconds=max_idle_seconds)
        native.protected_unchanged(protected)
        native.write_new(state / ("result-" + uuid4().hex[:8] + ".json"), dict(version=VERSION, output=str(output),
            **call_counts(guards), original_result=result,
            original_pipeline_manifest_unchanged=True, automatic_retry=False))
        return output, result
    except Exception as error:
        native.write_new(state / "failure.json", dict(version=VERSION, error=str(error), automatic_retry=False,
            all_original_native_outputs_and_bindings_retained=True))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-root", "pool-dir", "annotation-run", "previous-review-run", "source-completion-directory",
                 "resume", "authorization", "guard-state-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-idle-seconds", type=int, default=21600)
    args = parser.parse_args()
    _, result = run(args.canonical_root, args.pool_dir, args.annotation_run, previous_review_run=args.previous_review_run,
        source_completion_directory=args.source_completion_directory, resume=args.resume, authorization=args.authorization,
        guard_state_directory=args.guard_state_directory, max_idle_seconds=args.max_idle_seconds)
    print(protocol.canonical_bytes(result).decode())
