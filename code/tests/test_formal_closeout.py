"""Synthetic closeout scheduling tests; no native model or network calls."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import continue_formal_closeout as closeout


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    public, root = tmp_path / "public", tmp_path / "canonical"
    monkeypatch.setattr(closeout, "ROOT", public)
    monkeypatch.setattr(closeout, "SOURCES", ["scripts/continue_formal_closeout.py", "scripts/run_machine_experiments.py"])
    for name in closeout.SOURCES:
        closeout.write(public / name, b"Synthetic immutable implementation.")
    closeout.write(root / ".venv/Scripts/python.exe", b"Synthetic project launcher.")
    pool = root / "data/machine_runs/candidate-retrieval/selected-pool"
    producer = pool / "cohort_annotations_clarified/producer"
    review = root / "data/machine_runs/cohort_pipeline/review"
    ids = ["synthetic-question-" + str(i) for i in range(1200)]
    closeout.write(producer / "manifest.json", dict(specification=dict(
        version="codex-format-clarified-first-attempt-cohort-v1", eligible_record_ids=ids, requested_model=None)))
    closeout.write(review / "manifest.json", dict(version="wire-source-compact-adjudication-clarified-cache-v4",
        source_annotation_run=str(producer), source_order_record_ids=ids, chunk_size=20,
        chunks=[ids[i:i + 20] for i in range(0, 1200, 20)]))
    sample = root / "data/machine_runs/answer-samples/sample"
    closeout.write(sample / "sample_plan.json", dict(synthetic_preselected=100))
    closeout.write(sample / "receipt.json", dict(synthetic=True))
    monkeypatch.setattr(closeout, "SAMPLE_SHA", closeout.digest(sample / "sample_plan.json"))
    runtime = root / "data/machine_runs/experiment-runtime" / closeout.RUNTIME_RUN
    mapping = {}
    for key in closeout.RUNTIME_KEYS:
        path = runtime / "runtime" / (key + ".artifact")
        value = ({"run_machine_experiments.py": dict(sha256=closeout.digest(public / "scripts/run_machine_experiments.py"))}
                 if key == "implementation" else dict(synthetic=key))
        closeout.write(path, value)
        mapping[key] = str(path)
    closeout.write(runtime / "runtime_artifacts.json", mapping)
    closeout.write(runtime / "receipt.json", dict(contract="frozen-machine-1000-retrieval-100-answer-v4",
        runtime_sha256={key: closeout.digest(path) for key, path in mapping.items()}, config=dict(
            requested_model=None, max_cases=8, max_request_bytes=160000,
            answer_sample_binding=dict(directory=str(sample)))))
    monkeypatch.setattr(closeout, "RECEIPT_SHA", closeout.digest(runtime / "receipt.json"))
    output, plan = closeout.prepare(root, producer, review, max_formal_contexts=getattr(request, "param", 1200))
    return dict(root=root, public=public, producer=producer, review=review, output=output, plan=plan)


def ready(fixture):
    producer, review = fixture["producer"], fixture["review"]
    closeout.write(producer / "summary-complete.json", dict(status="selected_clarified_calls_complete",
        accepted_counts={"a": 1200, "b": 1200}, failures=[], human_gold=False, external_paid_api_calls=0))
    source = review / "complete_source_inputs"
    source.mkdir()
    decisions = []
    for number in range(60):
        # Historical decisions deliberately remain in another original run.
        owner = fixture["root"] / "data/machine_runs/cohort_pipeline/original-owner" if number < 8 else review
        path = owner / "adjudications" / f"chunk-{number:04d}" / "bounded-run/decisions.jsonl"
        closeout.write(path, b"Synthetic retained original-owned decision.\n")
        decisions.append(str(path))
    summary = dict(output=str(review), complete=True, expected_records=1200, retained_records=1200,
        accepted=1200, unresolved=0, retained_chunks=60, exported_chunks=60, human_gold=False,
        complete_source_inputs=str(source), adjudication_files=decisions)
    closeout.write(review / "summary-complete.json", summary)
    return summary


def backend(fixture, calls, *, failed_stage=None, gate_overrides=None, answer_overrides=None):
    def execute(command, cwd, environment):
        assert cwd == fixture["public"]
        assert command[:3] == [str(fixture["root"] / ".venv/Scripts/python.exe"), "-X", "utf8"]
        assert environment["PYTHONPATH"] == str(cwd / "src") + closeout.os.pathsep + str(cwd / "scripts")
        assert not any(flag in command for flag in ("--requested-model", "--retry-failed", "--model"))
        script = Path(command[3]).name
        calls.append(command)
        stage = {"assemble_machine_freeze.py": "assemble", "freeze_machine_dataset.py": "freeze"}.get(script)
        stage = stage or ("answers" if "--resume" in command else "retrieval_gate")
        if stage == failed_stage:
            return 1, b"Retained synthetic diagnostics.\n", b"Synthetic quota failure.\n"
        if stage == "assemble":
            result = dict(output=str(fixture["root"] / "data/machine_runs/freeze_inputs/assembled"), records=1200)
        elif stage == "freeze":
            result = dict(output=str(fixture["root"] / "data/machine_runs/frozen_evaluation/frozen"),
                          formal_test_ready=True, human_gold=False)
        else:
            result = dict(output=str(fixture["root"] / "data/machine_runs/formal-experiments/formal"),
                status="partial_new_attempt_limit_reached" if stage == "retrieval_gate" else "completed_formal_machine_only_experiment",
                completed_retrieval_questions=1000, new_native_attempts_started=0 if stage == "retrieval_gate" else 99,
                completed_answer_questions=0 if stage == "retrieval_gate" else 100,
                completed_variant_answers=0 if stage == "retrieval_gate" else 600, human_gold=False)
            result.update((gate_overrides if stage == "retrieval_gate" else answer_overrides) or {})
        return 0, b"Progress is not JSON.\n" + json.dumps(result, indent=2).encode() + b"\n", b""
    return execute


def test_prepare_seals_zero_call_runtime_sources_and_sample(prepared):
    plan = prepared["plan"]
    assert plan["prepare_native_calls"] == 0 and plan["subscription_dollar_cost"] is None
    assert plan["human_gold"] is False and not plan["publication"] and not plan["cloud_submission"]
    assert plan["retrieval_run"] == str(prepared["producer"].parents[1])
    assert plan["instructor_approval"] == "user_reported100_answer_experiment_approved"
    assert len(list(prepared["output"].iterdir())) == 2


def test_ready_runs_exact_original_owned_files_and_resumes_same_run(prepared):
    summary, calls = ready(prepared), []
    result = closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0,
        executor=backend(prepared, calls))
    assert result["completed_variant_answers"] == 600 and len(calls) == 4
    assert [calls[0][i + 1] for i, arg in enumerate(calls[0]) if arg == "--adjudications"] == summary["adjudication_files"]
    assert calls[2][-2:] == ["--max-new-calls", "0"]
    assert calls[3][-2:] == ["--max-new-calls", "1200"]
    assert "formal-experiments/formal" in calls[3][calls[3].index("--resume") + 1].replace("\\", "/")
    for stage in ("assemble", "freeze", "retrieval_gate", "answers"):
        assert closeout.read(prepared["output"] / "stages" / stage / "receipt.json")["exit_code"] == 0


@pytest.mark.parametrize("prepared", [3], indirect=True)
def test_bounded_wait_performs_no_call_until_ready(prepared):
    calls, sleeps, ticks = [], [], [0]
    def sleeper(duration):
        sleeps.append(duration)
        ticks[0] += duration
        ready(prepared)
    closeout.run(prepared["output"], max_formal_contexts=3, max_wait_seconds=30, poll_seconds=5,
        executor=backend(prepared, calls), sleeper=sleeper, clock=lambda: ticks[0])
    assert sleeps == [5] and len(calls) == 4 and calls[-1][-1] == "3"


def test_complete_review_waits_for_late_producer_final_summary(prepared):
    ready(prepared)
    producer_summary = prepared["producer"] / "summary-complete.json"
    retained = producer_summary.read_bytes()
    producer_summary.unlink()
    assert closeout.prerequisite(prepared["plan"]) is None
    calls, sleeps, ticks = [], [], [0]
    def sleeper(duration):
        sleeps.append(duration)
        ticks[0] += duration
        closeout.write(producer_summary, retained)
    closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=30,
        executor=backend(prepared, calls), sleeper=sleeper, clock=lambda: ticks[0])
    assert sleeps == [5] and len(calls) == 4


@pytest.mark.parametrize("change", ["incomplete_count", "duplicate_decisions", "outside_source"])
def test_malformed_complete_review_is_not_treated_as_ready(prepared, change):
    summary = ready(prepared)
    if change == "incomplete_count":
        summary["accepted"] = 1199
    elif change == "duplicate_decisions":
        summary["adjudication_files"][-1] = summary["adjudication_files"][0]
    else:
        summary["complete_source_inputs"] = str(prepared["root"] / "outside")
    path = prepared["review"] / "summary-complete.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0,
            executor=lambda *args: pytest.fail("No call after malformed review"))


@pytest.mark.parametrize("location", ["producer", "review"])
@pytest.mark.parametrize("marker", ["failure.json", "STOP_REQUESTED"])
def test_failure_markers_stop_before_any_downstream_stage(prepared, location, marker):
    ready(prepared)
    closeout.write(prepared[location] / marker, b"Synthetic stop.")
    with pytest.raises(ValueError, match="Source failure"):
        closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0,
            executor=lambda *args: pytest.fail("No downstream call permitted"))
    assert closeout.read(prepared["output"] / "failure.json")["automatic_retry"] is False


@pytest.mark.parametrize("owner", ["producer", "review"])
def test_operational_native_failure_is_not_ignored(prepared, owner):
    ready(prepared)
    suffix = "a/batch-0001/attempt-bad/receipt.json" if owner == "producer" else "adjudications/chunk-0009/bounded-run/executions/bad/receipt.json"
    closeout.write(prepared[owner] / suffix, dict(accepted=False, errors=["Synthetic quota"], process_exit_code=1))
    with pytest.raises(ValueError, match="native/quota"):
        closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0,
            executor=lambda *args: pytest.fail("No call permitted"))


def test_timeout_stops_without_native_call_or_automatic_rerun(prepared):
    with pytest.raises(TimeoutError):
        closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0,
            executor=lambda *args: pytest.fail("No call before prerequisites"))
    with pytest.raises(FileExistsError):
        closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0,
            executor=lambda *args: pytest.fail("Never rerun a started one-shot task"))


@pytest.mark.parametrize("stage,count", [("assemble", 1), ("freeze", 2), ("retrieval_gate", 3), ("answers", 4)])
def test_failed_stage_retains_logs_and_never_advances_or_reruns(prepared, stage, count):
    ready(prepared)
    calls = []
    with pytest.raises(ValueError, match="Failed closeout stage"):
        closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0,
            executor=backend(prepared, calls, failed_stage=stage))
    assert len(calls) == count
    directory = prepared["output"] / "stages" / stage
    assert closeout.read(directory / "receipt.json")["exit_code"] == 1
    assert directory.joinpath("stdout.txt").read_bytes() and directory.joinpath("stderr.txt").read_bytes()
    with pytest.raises(FileExistsError):
        closeout.run(prepared["output"], max_formal_contexts=1200, executor=lambda *args: pytest.fail("No rerun"))


@pytest.mark.parametrize("change", [{"new_native_attempts_started": 1}, {"completed_retrieval_questions": 999}, {"status": "failed_not_complete"}])
def test_zero_call_retrieval_gate_rejects_mismatches(prepared, change):
    ready(prepared)
    calls = []
    with pytest.raises(ValueError, match="Zero-call"):
        closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0,
            executor=backend(prepared, calls, gate_overrides=change))
    assert len(calls) == 3


@pytest.mark.parametrize("change", [{"completed_variant_answers": 599}, {"output": "outside-run"}, {"status": "failed_not_complete"}])
def test_final_coverage_same_run_and_failure_guards(prepared, change):
    ready(prepared)
    with pytest.raises(ValueError):
        closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0,
            executor=backend(prepared, [], answer_overrides=change))
    assert not (prepared["output"] / "result.json").exists()


@pytest.mark.parametrize("prepared", [2], indirect=True)
def test_explicit_formal_cap_can_stop_partially_without_automatic_continuation(prepared):
    ready(prepared)
    calls = []
    result = closeout.run(prepared["output"], max_formal_contexts=2, max_wait_seconds=0,
        executor=backend(prepared, calls, answer_overrides=dict(status="partial_new_attempt_limit_reached",
            completed_answer_questions=0, completed_variant_answers=0)))
    assert result["status"] == "partial_new_attempt_limit_reached" and len(calls) == 4


@pytest.mark.parametrize("cap", [None, True, 0, 1201, 20])
def test_run_cannot_change_the_prepared_explicit_formal_cap(prepared, cap):
    ready(prepared)
    with pytest.raises(ValueError, match="Explicit formal-attempt cap"):
        closeout.run(prepared["output"], max_formal_contexts=cap,
            executor=lambda *args: pytest.fail("No call with an unsealed cap"))
    assert not (prepared["output"] / "run_started.json").exists()


def test_durable_rejected_source_progress_stops_immediately(prepared):
    ready(prepared)
    closeout.write(prepared["producer"] / "progress.jsonl", dict(type="completed", accepted=False))
    with pytest.raises(ValueError, match="Rejected source progress"):
        closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0,
            executor=lambda *args: pytest.fail("No call after source rejection"))


@pytest.mark.parametrize("target", ["manifest", "code", "sample"])
def test_sealed_plan_or_input_tampering_blocks_downstream_calls(prepared, target):
    ready(prepared)
    path = {"manifest": prepared["output"] / "manifest.json", "code": prepared["public"] / "scripts/run_machine_experiments.py",
            "sample": prepared["root"] / "data/machine_runs/answer-samples/sample/sample_plan.json"}[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0,
            executor=lambda *args: pytest.fail("No call after tampering"))


def test_launcher_exception_retains_exit_receipt_and_error_log(prepared):
    ready(prepared)
    def unavailable(*args):
        raise OSError("Synthetic launcher unavailable")
    with pytest.raises(ValueError, match="Failed closeout stage"):
        closeout.run(prepared["output"], max_formal_contexts=1200, max_wait_seconds=0, executor=unavailable)
    directory = prepared["output"] / "stages/assemble"
    assert closeout.read(directory / "receipt.json")["exit_code"] is None
    assert b"OSError" in (directory / "stderr.txt").read_bytes()


def test_log_parser_accepts_progress_but_rejects_ambiguous_trailing_data():
    assert closeout.trailing_object(b'Progress\n{not-json}\n{"result":1}\n') == {"result": 1}
    with pytest.raises(ValueError):
        closeout.trailing_object(b'{"result":1}\nUnexpected trailing log.\n')
    with pytest.raises(ValueError, match="Duplicate"):
        closeout.trailing_object(b'{"result":1,"result":2}\n')
