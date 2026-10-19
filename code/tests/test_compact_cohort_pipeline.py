"""Synthetic versioned pipeline migration without paid/native calls."""
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest

from test_cohort_adjudication_pipeline import runner, source, paired, mock_adjudicator


@pytest.fixture
def compact_pipeline(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("compact_pipeline_tests", scripts / "pipeline_compact_cohort_adjudication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_backend(pipeline, calls):
    def call(system, payload, schema, output_parent, role):
        assert system == pipeline.compact.SYSTEM and role == "adjudicator"
        assert "event_group" not in json.dumps(payload) and "split" not in json.dumps(payload)
        assert all("independent_a" in row and "independent_b" in row and row["evidence"] for row in payload.values())
        calls.append(list(payload))
        raw = {key: dict(status="accepted", label_basis="a", new_label=None,
                        reason="Synthetic source reviewer selected exact full original A.", unresolved_reasons=[]) for key in payload}
        directory = Path(output_parent) / ("s-" + uuid4().hex[:8])
        directory.mkdir(parents=True)
        from oil_evidence.codex_runtime import ISOLATION_PROMPT
        stdin = (ISOLATION_PROMPT + "\n" + system + "\nUNTRUSTED_PAYLOAD_JSON\n").encode() + pipeline.protocol.canonical_bytes(payload)
        values = {"raw_final.json": (json.dumps(raw, indent=2) + "\r\n").encode(), "stdin.txt": stdin,
                  "schema.json": pipeline.protocol.canonical_bytes(schema)}
        execution = "synthetic-compact-" + uuid4().hex
        values["events.jsonl"] = b"".join(pipeline.protocol.canonical_bytes(event) + b"\n" for event in [
            dict(type="thread.started", thread_id=execution), dict(type="turn.completed", usage={})])
        for name, value in values.items():
            pipeline.native.write_new(directory / name, value)
        receipt = dict(artifact_directory=str(directory), execution_id=execution, accepted=True, observed_model=None,
                       artifacts_sha256={name: pipeline.protocol.sha256(value) for name, value in values.items()})
        pipeline.native.write_new(directory / "receipt.json", receipt)
        return raw, receipt
    return call


def test_fresh_compact_pipeline_full_source_coverage_raw_encoding_and_no_call_resume(compact_pipeline, runner, source):
    paired(source, 5)
    calls = []
    output, result = compact_pipeline.run(source["root"], source["pool"], source["run"], caller=compact_backend(compact_pipeline, calls))
    assert result["complete"] and result["accepted"] == 25 and result["new_model_calls"] == 3
    assert [key for group in calls for key in group] == [row["record_id"] for row in source["rows"]]
    manifest = compact_pipeline.native.read_json(output / "manifest.json")
    assert manifest["version"] == compact_pipeline.VERSION and manifest["raw_encoding"] == compact_pipeline.compact.VERSION
    decisions = compact_pipeline.protocol.parse_jsonl(Path(result["adjudication_files"][0]).read_bytes())
    assert all(row["adjudication"]["raw_encoding"] == compact_pipeline.compact.VERSION for row in decisions)
    _, resumed = compact_pipeline.run(source["root"], source["pool"], source["run"], resume=output,
        caller=lambda *args, **kwargs: pytest.fail("Retained compact raw contexts must not be replayed"))
    assert resumed["new_model_calls"] == 0 and resumed["retained_model_calls"] == 3


def old_stopped_run(runner, source):
    paired(source, 4)
    output, result = runner.run(source["root"], source["pool"], source["run"], max_idle_seconds=0,
                                caller=mock_adjudicator(runner, []))
    runner.native.write_new(output / "controlled_stop_receipt.json", dict(
        status="owned_parent_stopped_after_durable_full_response_chunks", all_original_attempts_preserved=True,
        progress_sha256=runner.protocol.sha256((output / "progress.jsonl").read_bytes()), retained_records=result["retained_records"]))
    return output, result


def test_lossless_previous_full_response_reuse_no_original_rerun(compact_pipeline, runner, source):
    previous, previous_result = old_stopped_run(runner, source)
    originals = {str(path): path.read_bytes() for path in previous.rglob("*") if path.is_file()}
    source["add"](5, "a")
    source["add"](5, "b")
    calls = []
    output, result = compact_pipeline.run(source["root"], source["pool"], source["run"], previous_full_response_run=previous,
                                         caller=compact_backend(compact_pipeline, calls))
    assert result["accepted"] == 25 and result["new_model_calls"] == 1 and result["retained_model_calls"] == 3
    assert calls == [[row["record_id"] for row in source["rows"][20:]]]
    assert (output / "retained/chunk-0001.json").read_bytes() == (previous / "retained/chunk-0001.json").read_bytes()
    assert result["adjudication_files"][0] == previous_result["adjudication_files"][0]
    assert all(Path(path).read_bytes() == value for path, value in originals.items())
    _, resumed = compact_pipeline.run(source["root"], source["pool"], source["run"], previous_full_response_run=previous,
                                      resume=output, caller=lambda *args, **kwargs: pytest.fail("No model repetition on resume"))
    assert resumed["new_model_calls"] == 0


def test_previous_prefix_progress_tamper_rejected_before_new_call(compact_pipeline, runner, source):
    previous, _ = old_stopped_run(runner, source)
    path = previous / "progress.jsonl"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="progress changed"):
        compact_pipeline.run(source["root"], source["pool"], source["run"], previous_full_response_run=previous,
            caller=lambda *args, **kwargs: pytest.fail("No native work when old provenance changes"))


def test_configure_does_not_modify_original_source_adjudicator(compact_pipeline):
    before = compact_pipeline.full_adjudicator.SYSTEM
    base, old_base = compact_pipeline.isolated_base(), compact_pipeline.isolated_base()
    compact_pipeline.configure(base, old_base, dict(output=None, records=0, chunks={}, sha256={}))
    assert base.adjudicator.SYSTEM == compact_pipeline.compact.SYSTEM
    assert old_base.adjudicator.SYSTEM == before == compact_pipeline.full_adjudicator.SYSTEM
    assert base.VERSION != old_base.VERSION


def test_one_development_pilot_context_reused_on_full_resume(compact_pipeline, source):
    paired(source, 5)
    calls = []
    output, pilot = compact_pipeline.run_pilot(source["root"], source["pool"], source["run"],
                                               caller=compact_backend(compact_pipeline, calls))
    assert pilot["pilot_records"] == 10 and pilot["native_calls"] == 1 and len(calls) == 1
    assert pilot["accepted"] == 10 and pilot["formal_dataset_frozen"] is False
    before = Path(pilot["pilot_binding_path"]).read_bytes()
    _, completed = compact_pipeline.run(source["root"], source["pool"], source["run"], resume=output,
                                         caller=compact_backend(compact_pipeline, calls))
    assert completed["complete"] and completed["accepted"] == 25 and completed["new_model_calls"] == 2
    assert completed["retained_model_calls"] == 3 and len(calls) == 3
    assert Path(pilot["pilot_binding_path"]).read_bytes() == before
    assert [key for group in calls for key in group] == [row["record_id"] for row in source["rows"]]


def synthetic_snapshot(pipeline, source):
    directory = source["root"] / "data/machine_runs/guard-snapshots/synthetic-archive"
    current = Path(pipeline.protocol.__file__).resolve()
    # A harmless source comment simulates a distinct historical implementation
    # without changing any real project/core file or label policy.
    archived = current.read_bytes() + b"\n# Synthetic historical snapshot variation.\n"
    pipeline.native.write_new(directory / "machine_evaluation.py", archived)
    pipeline.native.write_new(directory / "manifest.json", dict(copy_bytes_verified_against_original_at_capture=True,
        artifacts_sha256={"machine_evaluation.py": pipeline.protocol.sha256(archived)},
        original_source_paths={"machine_evaluation.py": str(current)}))
    return directory, current, pipeline.protocol.sha256(archived)


def test_historical_code_proof_and_current_execution_hashes_are_separate(compact_pipeline, source):
    directory, current, archived_sha = synthetic_snapshot(compact_pipeline, source)
    guards = compact_pipeline.HistoricalGuards(source["root"], directory)
    guards.check_inputs({str(current): archived_sha})
    guards.check_inputs({str(current): compact_pipeline.protocol.sha256(current.read_bytes())})
    assert str(current) in guards.receipt()["changed_code_verified_against_archived_bytes"]
    with pytest.raises(ValueError, match="Protected input changed"):
        guards.check_inputs({str(current): "0" * 64})
    private = source["root"] / "immutable-corpus-fixture"
    compact_pipeline.native.write_new(private, b"immutable source")
    with pytest.raises(ValueError, match="Protected input changed"):
        guards.check_inputs({str(private): archived_sha})
    assert guards.protocol.ANNOTATION_PROMPT == compact_pipeline.protocol.ANNOTATION_PROMPT


def test_historical_snapshot_tamper_and_path_escape_rejected(compact_pipeline, source):
    directory, _, _ = synthetic_snapshot(compact_pipeline, source)
    guards = compact_pipeline.HistoricalGuards(source["root"], directory)
    path = directory / "machine_evaluation.py"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Protected input changed"):
        guards.check_inputs({})
    with pytest.raises(ValueError, match="snapshot escapes"):
        compact_pipeline.HistoricalGuards(source["root"], source["root"] / "outside")


def test_historical_install_uses_isolated_protocol_and_preserves_live_modules(compact_pipeline, source):
    directory, _, _ = synthetic_snapshot(compact_pipeline, source)
    guards = compact_pipeline.HistoricalGuards(source["root"], directory)
    base, old = compact_pipeline.isolated_base(), compact_pipeline.isolated_base()
    original_live_check = compact_pipeline.native.protected_unchanged
    guards.install(base)
    guards.install(old, historical=True)
    assert base.protocol is compact_pipeline.protocol and old.protocol is guards.protocol
    assert old.adjudicator.protocol is guards.protocol
    assert compact_pipeline.native.protected_unchanged is original_live_check
    assert base.native.protected_unchanged == guards.check_inputs


def test_final_pilot_audit_uses_historical_aware_checker(compact_pipeline, source, monkeypatch):
    paired(source, 5)
    directory, current, archived_sha = synthetic_snapshot(compact_pipeline, source)
    guards = compact_pipeline.HistoricalGuards(source["root"], directory)
    old, base = compact_pipeline.isolated_base(), compact_pipeline.isolated_base()
    context = base.prepare(source["root"], source["pool"], source["run"])
    context["protected"][str(current)] = archived_sha
    guards.install(base)
    base.prepare = lambda *args, **kwargs: context
    monkeypatch.setattr(compact_pipeline, "bases", lambda *args: (old, base, guards))
    calls = []
    _, result = compact_pipeline.run_pilot(source["root"], source["pool"], source["run"],
        historical_guard_snapshot=directory, caller=compact_backend(compact_pipeline, calls))
    assert result["accepted"] == 10 and result["native_calls"] == 1


def test_declared_partial_native_context_recovery_without_pilot_rerun(compact_pipeline, source):
    paired(source, 5)
    calls = []
    prior, pilot = compact_pipeline.run_pilot(source["root"], source["pool"], source["run"],
                                              caller=compact_backend(compact_pipeline, calls))
    snapshot = prior / "historical_driver/synthetic_driver.py"
    compact_pipeline.native.write_new(snapshot, Path(compact_pipeline.__file__).read_bytes())
    compact_pipeline.native.write_new(prior / "outer_failure_receipt.json", dict(
        status="native_compact_development_pilot_accepted_outer_final_audit_failed",
        historical_driver_snapshot="historical_driver/synthetic_driver.py",
        historical_driver_sha256=compact_pipeline.protocol.sha256(snapshot.read_bytes()), native_records=pilot["pilot_records"],
        note="Synthetic final-audit fault scenario; no real native/model invocation."))
    originals = {str(path): path.read_bytes() for path in prior.rglob("*") if path.is_file()}
    output, completed = compact_pipeline.run(source["root"], source["pool"], source["run"], partial_compact_run=prior,
                                              caller=compact_backend(compact_pipeline, calls))
    assert completed["accepted"] == 25 and completed["new_model_calls"] == 2 and len(calls) == 3
    assert [key for group in calls for key in group] == [row["record_id"] for row in source["rows"]]
    assert all(Path(path).read_bytes() == value for path, value in originals.items())
    assert Path(completed["adjudication_files"][0]).is_relative_to(prior)
    _, resumed = compact_pipeline.run(source["root"], source["pool"], source["run"], partial_compact_run=prior, resume=output,
        caller=lambda *args, **kwargs: pytest.fail("No accepted context repetition on recovered resume"))
    assert resumed["new_model_calls"] == 0 and resumed["retained_model_calls"] == 3
