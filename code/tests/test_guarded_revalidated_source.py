"""Failure refresh and retention ordering; synthetic files and no native calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("guarded_source_test", scripts / "run_guarded_revalidated_source.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def fixture_base(module, tmp_path, *, fail_during_call=False, rewrite=False, groups=2):
    progress = tmp_path / "producer/progress.jsonl"
    module.native.write_new(progress, b'{"type":"started"}\n')
    from pipeline_cohort_adjudication import progress_snapshot
    calls = []
    def caller(*args, **kwargs):
        calls.append(1)
        if fail_during_call:
            module.native.append_progress(progress, dict(type="completed", accepted=False, batch="synthetic-failed"))
        if rewrite:
            progress.write_bytes(b'{"type":"different"}\n')
        return {"raw": "unchanged actual output"}, dict(execution_id="synthetic-review", usage=dict(input_tokens=1, output_tokens=1))
    def adjudicate(chunk, directory, caller=None):
        for index in range(groups):
            value = caller("same-system", {"same": "payload"}, {"type": "object"}, directory, role="adjudicator")
            module.native.write_new(Path(directory) / f"binding-{index}.json", value)
        module.native.write_new(Path(directory) / "complete.json", dict(complete=True))
        return dict(records=groups)
    base = SimpleNamespace(adjudicate_chunk=adjudicate, runtime=SimpleNamespace(call_structured_json=caller),
        progress_snapshot=lambda run: progress_snapshot(run))
    holder = dict(context=dict(run=progress.parent))
    return base, holder, calls, progress


def test_failure_after_valid_first_return_retains_binding_and_prevents_second_call(module, tmp_path):
    base, holder, calls, progress = fixture_base(module, tmp_path, fail_during_call=True)
    base, guard = module.wrap_base(base, holder, journal=tmp_path / "guard/invocations.jsonl")
    owner = tmp_path / "original-owner"
    with pytest.raises(ValueError, match="durable producer failure"):
        base.adjudicate_chunk(tmp_path / "same-chunk", owner)
    assert calls == [1] and (owner / "binding-0.json").is_file()
    assert not (owner / "binding-1.json").exists()
    assert guard.calls == 1


def test_single_call_chunk_binds_completed_output_before_boundary_stop(module, tmp_path):
    base, holder, calls, progress = fixture_base(module, tmp_path, fail_during_call=True, groups=1)
    base, guard = module.wrap_base(base, holder)
    owner = tmp_path / "original-owner"
    with pytest.raises(ValueError, match="durable producer failure"):
        base.adjudicate_chunk(tmp_path / "same-chunk", owner)
    assert (owner / "binding-0.json").exists() and (owner / "complete.json").exists() and calls == [1]


def test_regular_inflight_attempt_directories_do_not_stop_guard(module, tmp_path):
    base, holder, calls, progress = fixture_base(module, tmp_path)
    (progress.parent / "a/batch-active/attempt-inflight").mkdir(parents=True)
    base, guard = module.wrap_base(base, holder)
    assert base.adjudicate_chunk(tmp_path / "same-chunk", tmp_path / "original-owner") == dict(records=2)
    assert len(calls) == 2 and guard.pending is None


def test_rewritten_progress_cannot_hide_failure_between_checks(module, tmp_path):
    base, holder, calls, progress = fixture_base(module, tmp_path, rewrite=True)
    base, guard = module.wrap_base(base, holder)
    with pytest.raises(ValueError, match="rewritten"):
        base.adjudicate_chunk(tmp_path / "same-chunk", tmp_path / "original-owner")
    assert len(calls) == 1 and (tmp_path / "original-owner/binding-0.json").exists()


def test_existing_durable_failure_stops_before_any_call(module, tmp_path):
    base, holder, calls, progress = fixture_base(module, tmp_path)
    module.native.append_progress(progress, dict(type="completed", accepted=False, batch="already-failed"))
    base, _ = module.wrap_base(base, holder)
    with pytest.raises(ValueError, match="stops without retry"):
        base.adjudicate_chunk(tmp_path / "same-chunk", tmp_path / "original-owner")
    assert calls == []


def test_external_journal_unknown_interrupted_call_is_not_replayed(module, tmp_path):
    journal = tmp_path / "guard/invocations.jsonl"
    journal.parent.mkdir()
    module.native.append_progress(journal, dict(type="native_start", number=1))
    with pytest.raises(ValueError, match="interrupted"):
        module.ProgressGuard(lambda: (b"", []), lambda *args: pytest.fail("No replay"), journal=journal)


def test_authorized_call_ceiling_stops_without_repeating_completed_call(module, tmp_path):
    base, holder, calls, progress = fixture_base(module, tmp_path)
    base, guard = module.wrap_base(base, holder, max_calls=1)
    with pytest.raises(ValueError, match="ceiling"):
        base.adjudicate_chunk(tmp_path / "same-chunk", tmp_path / "original-owner")
    assert len(calls) == 1 and (tmp_path / "original-owner/binding-0.json").exists()


def test_successfully_completed_journal_resume_has_zero_new_calls_and_once_retained_count(module, tmp_path):
    journal, calls = tmp_path / "guard/invocations.jsonl", []
    def caller(*args, **kwargs):
        calls.append(1)
        return {"raw": "unchanged"}, dict(execution_id="known-original-call", usage=dict(input_tokens=4, output_tokens=1))
    guard = module.ProgressGuard(lambda: (b"same durable prefix", []), caller, journal=journal)
    guard("same-system", {}, {}, tmp_path, role="adjudicator")
    resumed = module.ProgressGuard(lambda: (b"same durable prefix plus new records", []),
        lambda *args, **kwargs: pytest.fail("No completed native invocation replay"), journal=journal)
    resumed.check()
    assert calls == [1]
    assert module.call_counts([resumed]) == dict(new_native_calls_this_guard=0,
        previous_native_calls_retained=1, total_native_calls_in_guard_journal=1)


def test_external_guard_proof_changes_after_return_preserve_binding_then_stop(module, tmp_path):
    base, holder, calls, progress = fixture_base(module, tmp_path)
    artifact = tmp_path / "guard-proof.json"
    module.native.write_new(artifact, dict(original=True))
    original = base.runtime.call_structured_json
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        artifact.write_bytes(b"changed external proof")
        return result
    base.runtime.call_structured_json = mutate
    base, _ = module.wrap_base(base, holder, protected={str(artifact): module.protocol.sha256(artifact.read_bytes())})
    with pytest.raises(ValueError):
        base.adjudicate_chunk(tmp_path / "same-chunk", tmp_path / "original-owner")
    assert calls == [1] and (tmp_path / "original-owner/binding-0.json").exists()


@pytest.mark.parametrize("field,value", [("root_authorized", False), ("max_new_native_calls", 0),
    ("prepared_pipeline_manifest_sha256", "wrong-manifest")])
def test_unbound_or_expanded_guard_authorization_stops_before_state_or_model(module, tmp_path, field, value):
    source, pipeline, state = tmp_path / "producer", tmp_path / "prepared-pipeline", tmp_path / "external-guard"
    module.native.write_new(pipeline / "manifest.json", dict(version=module.PIPELINE_VERSION, source_annotation_run=str(source)))
    auth = dict(version=module.AUTH_VERSION, root_authorized=True,
        guard_helper_sha256=module.protocol.sha256(Path(module.__file__).read_bytes()), pipeline_sha256=module.PIPELINE_SHA256,
        prepared_pipeline_directory=str(pipeline), prepared_pipeline_manifest_sha256=module.protocol.sha256((pipeline / "manifest.json").read_bytes()),
        annotation_run=str(source), guard_state_directory=str(state), native_inputs_schema_model_unchanged=True,
        new_failures_stop=True, max_new_native_calls=1)
    auth[field] = value
    authorization = tmp_path / "authorization.json"
    module.native.write_new(authorization, auth)
    with pytest.raises(ValueError, match="separately bound"):
        module.run(tmp_path, tmp_path / "pool", source, previous_review_run=tmp_path / "previous",
            source_completion_directory=tmp_path / "completion", resume=pipeline,
            authorization=authorization, guard_state_directory=state)
    assert not state.exists()
