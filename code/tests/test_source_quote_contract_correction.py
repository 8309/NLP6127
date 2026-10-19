"""One-case corrective source reviews with exact original ownership; no CLI calls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil

import pytest

from test_transport_review_recovery import recovery, retry_case, runner, source, accepted_caller


@pytest.fixture
def controller(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("source_quote_contract_test", scripts / "correct_source_quote_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def original(controller, retry_case):
    case = retry_case
    review = case["source"]["root"] / "closed-source-pipeline"
    chunk = review / "chunks/chunk-0031"
    shutil.copytree(case["chunk"], chunk)
    packets = controller.native.read_json(chunk / "packets.json")
    identifier = next(iter(packets))
    left = controller.compact.original.read_stream(chunk / "independent_a.jsonl")
    backend = accepted_caller(controller.recovery, case, [])
    def original_bad_title(*args, **kwargs):
        raw, receipt = backend(*args, **kwargs)
        if identifier in raw:
            value = deepcopy(left[identifier]["label"])
            value["quotes"].append(dict(passage_id=value["quotes"][0]["passage_id"],
                text="Synthetic title metadata not present in the source body."))
            raw[identifier].update(label_basis="new", new_label=value, reason="OLD JUDGE OUTCOME MUST NOT ENTER CORRECTIVE INPUT")
            directory = Path(receipt["artifact_directory"])
            (directory / "raw_final.json").write_bytes(controller.protocol.canonical_bytes(raw))
            receipt["artifacts_sha256"]["raw_final.json"] = controller.protocol.sha256((directory / "raw_final.json").read_bytes())
            (directory / "receipt.json").write_bytes(controller.protocol.canonical_bytes(receipt))
        return raw, receipt
    result = case["base"].adjudicate_chunk(chunk, review / "adjudications/chunk-0031", caller=original_bad_title)
    assert result["accepted"] == 19 and result["unresolved"] == 1
    result_path = review / "retained/chunk-0031.json"
    controller.native.write_new(result_path, result)
    rows = controller.protocol.parse_jsonl(Path(result["adjudications_path"]).read_bytes())
    target = next(row for row in rows if row["record_id"] == identifier)
    paths = dict(result=result_path, decisions=Path(result["adjudications_path"]),
        packets=chunk / "packets.json", a=chunk / "independent_a.jsonl", b=chunk / "independent_b.jsonl",
        export=chunk / "export_receipt.json", old_receipt=Path(target["adjudication"]["execution_receipt_path"]))
    pins = {key: controller.protocol.sha256(path.read_bytes()) for key, path in paths.items()}
    intent = case["source"]["root"] / "one-case-corrective-intent"
    plan = controller.run(review, intent, check_only=True, pins=pins, identifier=identifier)
    authorization = case["source"]["root"] / "one-case-root-authorization.json"
    controller.native.write_new(authorization, dict(plan, version=controller.AUTH_VERSION,
        root_authorized=True, intent_directory=str(intent), same_requested_model=True, new_failures_stop=True))
    return dict(case=case, review=review, chunk=chunk, result=result, paths=paths, pins=pins,
        identifier=identifier, intent=intent, authorization=authorization, plan=plan,
        original_files={str(path): path.read_bytes() for path in review.rglob("*") if path.is_file()})


def run(controller, case, **kwargs):
    return controller.run(case["review"], case["intent"], case["authorization"],
        pins=case["pins"], identifier=case["identifier"], **kwargs)


def new_caller(controller, case, calls):
    backend = accepted_caller(controller.recovery, case, calls)
    def execute(*args, **kwargs):
        raw, receipt = backend(*args, **kwargs)
        receipt.update(process_exit_code=0, codex_cli_invocations=1, usage={}, external_paid_api_calls=0)
        (Path(receipt["artifact_directory"]) / "receipt.json").write_bytes(controller.protocol.canonical_bytes(receipt))
        return raw, receipt
    return execute


def test_only_neutral_contract_is_added_system_schema_sources_original_labels_unchanged(controller, original):
    context = controller.validate_snapshot(original["review"], pins=original["pins"], identifier=original["identifier"])
    payload = deepcopy(context["payload"])
    assert payload[original["identifier"]].pop("mechanical_quote_validation_contract") == controller.CONTRACT
    assert payload == context["original_payload"]
    _, expected_schema = controller.compact.prepare_batch(context["packets"], context["left"], context["right"], [original["identifier"]])
    assert expected_schema == context["schema"]
    assert b"OLD JUDGE OUTCOME MUST NOT ENTER CORRECTIVE INPUT" not in context["stdin"]
    assert context["payload"][original["identifier"]]["independent_a"] == context["left"][original["identifier"]]["label"]
    assert original["plan"]["original_system_sha256"] == controller.protocol.sha256(controller.compact.SYSTEM.encode())
    assert not original["intent"].exists()


def test_one_fresh_call_creates_original_owned_whole20_preserving_other19_raw_lines(controller, original):
    calls = []
    corrected = run(controller, original, caller=new_caller(controller, original["case"], calls))
    assert calls == [[original["identifier"]]] and corrected["new_native_calls"] == 1
    assert corrected["accepted"] == 20 and corrected["unresolved"] == 0
    old = original["paths"]["decisions"].read_bytes().splitlines(keepends=True)
    new = Path(corrected["adjudications_path"]).read_bytes().splitlines(keepends=True)
    rows = controller.protocol.parse_jsonl(original["paths"]["decisions"].read_bytes())
    for before, after, row in zip(old, new, rows):
        if row["record_id"] != original["identifier"]:
            assert before == after
    assert Path(corrected["adjudications_path"]).parent == Path(original["result"]["output"])
    assert all(Path(path).read_bytes() == raw for path, raw in original["original_files"].items())
    row = next(row for row in controller.protocol.parse_jsonl(Path(corrected["adjudications_path"]).read_bytes()) if row["record_id"] == original["identifier"])
    assert row["adjudication"]["protocol_sha256"] == controller.protocol.sha256(controller.compact.SYSTEM.encode())
    sidecar = controller.native.read_json(Path(corrected["replacement_sidecar_path"]))
    assert sidecar["nineteen_other_lines_byte_identical"] and sidecar["original_source_summary_not_rewritten"]
    with pytest.raises(ValueError, match="no automatic retry"):
        run(controller, original, caller=lambda *args, **kwargs: pytest.fail("Never repeat accepted correction"))


@pytest.mark.parametrize("artifact", ["result", "decisions", "packets", "a", "b", "old_receipt"])
def test_old_result_raw_source_or_label_tampering_rejected_before_call(controller, original, artifact):
    path = original["paths"][artifact]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        run(controller, original, caller=lambda *args, **kwargs: pytest.fail("No changed-input call"))


@pytest.mark.parametrize("field,value", [("root_authorized", False), ("max_new_native_calls", 2),
    ("record_id", "other-record"), ("same_requested_model", False)])
def test_one_case_root_authorization_cannot_expand_or_rebind(controller, original, field, value):
    auth = controller.native.read_json(original["authorization"])
    auth[field] = value
    original["authorization"].write_bytes(controller.protocol.canonical_bytes(auth))
    with pytest.raises(ValueError, match="exact allowlisted"):
        run(controller, original, caller=lambda *args, **kwargs: pytest.fail("No expanded correction"))


def test_new_invalid_title_quote_stays_unresolved_without_replacement_or_retry(controller, original):
    calls = []
    backend = new_caller(controller, original["case"], calls)
    context = controller.validate_snapshot(original["review"], pins=original["pins"], identifier=original["identifier"])
    def invalid(*args, **kwargs):
        raw, receipt = backend(*args, **kwargs)
        key = original["identifier"]
        value = deepcopy(context["left"][key]["label"])
        value["quotes"].append(dict(passage_id=value["quotes"][0]["passage_id"], text="Synthetic invalid title quote again."))
        raw[key].update(label_basis="new", new_label=value)
        directory = Path(receipt["artifact_directory"])
        (directory / "raw_final.json").write_bytes(controller.protocol.canonical_bytes(raw))
        receipt["artifacts_sha256"]["raw_final.json"] = controller.protocol.sha256((directory / "raw_final.json").read_bytes())
        (directory / "receipt.json").write_bytes(controller.protocol.canonical_bytes(receipt))
        return raw, receipt
    with pytest.raises(ValueError, match="remains unresolved"):
        run(controller, original, caller=invalid)
    assert calls == [[original["identifier"]]] and (original["intent"] / "new_record.json").exists()
    assert controller.native.read_json(original["intent"] / "new_record.json")["status"] == "unresolved"
    assert not list(Path(original["result"]["output"]).glob("adjudications-quote-contract-*.jsonl"))
    with pytest.raises(ValueError, match="no automatic retry"):
        run(controller, original, caller=invalid)
    assert len(calls) == 1


def test_consistent_native_receipt_events_cannot_reuse_an_original_annotator_context(controller, original):
    calls = []
    backend = new_caller(controller, original["case"], calls)
    left = controller.compact.original.read_stream(original["chunk"] / "independent_a.jsonl")
    original_execution = left[original["identifier"]]["identity"]["execution_id"]
    def colliding(*args, **kwargs):
        raw, receipt = backend(*args, **kwargs)
        directory = Path(receipt["artifact_directory"])
        events = controller.protocol.parse_jsonl((directory / "events.jsonl").read_bytes())
        for event in events:
            if event.get("type") == "thread.started":
                event["thread_id"] = original_execution
        (directory / "events.jsonl").write_bytes(b"".join(controller.protocol.canonical_bytes(event) + b"\n" for event in events))
        receipt["execution_id"] = original_execution
        receipt["artifacts_sha256"]["events.jsonl"] = controller.protocol.sha256((directory / "events.jsonl").read_bytes())
        (directory / "receipt.json").write_bytes(controller.protocol.canonical_bytes(receipt))
        return raw, receipt
    with pytest.raises(ValueError, match="both annotators"):
        run(controller, original, caller=colliding)
    assert calls == [[original["identifier"]]]
    assert not list(Path(original["result"]["output"]).glob("adjudications-quote-contract-*.jsonl"))
    assert all(Path(path).read_bytes() == raw for path, raw in original["original_files"].items())
    with pytest.raises(ValueError, match="no automatic retry"):
        run(controller, original, caller=colliding)
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [("codex_cli_invocations", 2), ("external_paid_api_calls", 1)])
def test_multiple_or_paid_native_invocations_do_not_admit_a_replacement(controller, original, field, value):
    backend = new_caller(controller, original["case"], [])
    def changed(*args, **kwargs):
        raw, receipt = backend(*args, **kwargs)
        receipt[field] = value
        (Path(receipt["artifact_directory"]) / "receipt.json").write_bytes(controller.protocol.canonical_bytes(receipt))
        return raw, receipt
    with pytest.raises(ValueError, match="operationally accepted"):
        run(controller, original, caller=changed)
    assert not list(Path(original["result"]["output"]).glob("adjudications-quote-contract-*.jsonl"))


def test_new_native_bytes_changed_after_validation_cannot_produce_accepted_replacement(controller, original, monkeypatch):
    calls = []
    unchanged_retainer = controller.compact.retain_compact_decisions
    mutated = []
    def change_after_validated_retention(*args, **kwargs):
        decision = unchanged_retainer(*args, **kwargs)
        receipt = args[5]
        stdin = Path(receipt["artifact_directory"]) / "stdin.txt"
        if b'"mechanical_quote_validation_contract"' in stdin.read_bytes():
            stdin.write_bytes(stdin.read_bytes() + b" ")
            mutated.append(str(stdin))
        return decision
    monkeypatch.setattr(controller.compact, "retain_compact_decisions", change_after_validated_retention)
    with pytest.raises(ValueError):
        run(controller, original, caller=new_caller(controller, original["case"], calls))
    assert calls == [[original["identifier"]]] and len(mutated) == 1
    assert not list(Path(original["result"]["output"]).glob("adjudications-quote-contract-*.jsonl"))
    assert (original["intent"] / "failure.json").exists()
    assert all(Path(path).read_bytes() == raw for path, raw in original["original_files"].items())
    with pytest.raises(ValueError, match="no automatic retry"):
        run(controller, original, caller=lambda *args, **kwargs: pytest.fail("No repeat after native artifact changed"))
