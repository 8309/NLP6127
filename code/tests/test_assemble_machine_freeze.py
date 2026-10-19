"""Synthetic assembly receipts test byte binding, not annotation accuracy."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import assemble_machine_freeze as assembler
import adjudicate_machine_annotations as adjudicator
import compact_machine_adjudication as compact
from test_machine_adjudication import scenario, raw_decision, receipt_for
from oil_evidence import machine_evaluation as protocol


def fixture(tmp_path, unresolved=False):
    source, packets, left, right = scenario(tmp_path)
    run = tmp_path / "retained_adjudication"
    raw = raw_decision(left)
    if unresolved:
        raw["q1"] = dict(status="unresolved", reason="Synthetic ambiguous period.", label=None,
                         unresolved_reasons=["Period unresolved."])
    receipt = receipt_for(run / "executions", raw)
    decisions = adjudicator.retain_decisions(raw, packets, left, right, ["q1"], receipt)
    decision_path = run / "adjudications.jsonl"
    decision_path.write_bytes(protocol.canonical_bytes(decisions[0]) + b"\n")
    runtime_run = tmp_path / "retained_experiment"
    runtime_run.mkdir()
    snapshots = {}
    for key in protocol.RUNTIME_KEYS:
        path = runtime_run / "runtime" / (key + ".artifact")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(("Synthetic runtime " + key).encode())
        snapshots[key] = str(path)
    runtime_map = runtime_run / "runtime_artifacts.json"
    runtime_map.write_bytes(protocol.canonical_bytes(snapshots))
    return source, decision_path, runtime_map


@pytest.mark.parametrize("unresolved", [False, True])
def test_assembly_preserves_inputs_and_explicit_unresolved(tmp_path, unresolved):
    source, decision_path, runtime_map = fixture(tmp_path, unresolved)
    protected = {str(path): path.read_bytes() for path in source.iterdir() if path.is_file()}
    protected.update({str(path): path.read_bytes() for parent in (decision_path.parent, runtime_map.parent)
                      for path in parent.rglob("*") if path.is_file()})
    output, receipt = assembler.assemble(source, [decision_path], runtime_map, tmp_path / "assembled")
    assert receipt["records"] == 1 and receipt["human_gold"] is False
    status = "unresolved" if unresolved else "accepted"
    assert receipt["split_status_counts"] == {"development": {status: 1}, "test": {}}
    assert all(Path(path).read_bytes() == raw for path, raw in protected.items())
    assert protocol.parse_jsonl((output / "adjudications.jsonl").read_bytes())[0]["status"] == status
    snapshots = json.loads((output / "runtime_artifacts.json").read_bytes())
    assert all(Path(path).is_relative_to(output / "runtime") for path in snapshots.values())
    assert all(protocol.sha256((output / name).read_bytes()) == digest
               for name, digest in receipt["artifacts_sha256"].items())


@pytest.mark.parametrize("field", ["reason", "execution_id", "protocol_sha256", "raw_response_sha256",
                                    "original_output_sha256", "both_originals_seen"])
def test_every_adjudication_field_binds_actual_raw_execution(tmp_path, field):
    source, decision_path, runtime_map = fixture(tmp_path)
    decision = protocol.parse_jsonl(decision_path.read_bytes())[0]
    decision["adjudication"][field] = "Synthetic changed field"
    decision_path.write_bytes(protocol.canonical_bytes(decision) + b"\n")
    with pytest.raises(ValueError, match="changed|exact raw"):
        assembler.assemble(source, [decision_path], runtime_map, tmp_path / "assembled")


def test_accepted_semantically_changed_label_cannot_replace_retained_raw(tmp_path):
    source, decision_path, runtime_map = fixture(tmp_path)
    decision = protocol.parse_jsonl(decision_path.read_bytes())[0]
    decision["label"]["answer_points"][0]["text"] = "Changed synthetic claim."
    decision_path.write_bytes(protocol.canonical_bytes(decision) + b"\n")
    with pytest.raises(ValueError, match="exact raw"):
        assembler.assemble(source, [decision_path], runtime_map, tmp_path / "assembled")


@pytest.mark.parametrize("change", ["outside_raw", "outside_receipt", "receipt_rejected", "changed_raw",
                                     "missing_question", "duplicate_question", "outside_runtime"])
def test_unsafe_paths_and_incomplete_or_changed_inputs_fail_closed(tmp_path, change):
    source, decision_path, runtime_map = fixture(tmp_path)
    decision = protocol.parse_jsonl(decision_path.read_bytes())[0]
    if change == "outside_runtime":
        snapshots = json.loads(runtime_map.read_bytes())
        outside = tmp_path / "outside.artifact"
        outside.write_bytes(b"Synthetic unrelated host data")
        snapshots["implementation"] = str(outside)
        runtime_map.write_bytes(protocol.canonical_bytes(snapshots))
    elif change in {"outside_raw", "outside_receipt"}:
        key = "raw_response_path" if change == "outside_raw" else "execution_receipt_path"
        outside = tmp_path / "outside.json"
        outside.write_bytes(Path(decision["adjudication"][key]).read_bytes())
        decision["adjudication"][key] = str(outside)
        decision_path.write_bytes(protocol.canonical_bytes(decision) + b"\n")
    elif change == "receipt_rejected":
        path = Path(decision["adjudication"]["execution_receipt_path"])
        receipt = json.loads(path.read_bytes())
        receipt["accepted"] = False
        path.write_bytes(protocol.canonical_bytes(receipt))
    elif change == "changed_raw":
        Path(decision["adjudication"]["raw_response_path"]).write_bytes(b"{}")
    elif change == "missing_question":
        decision_path.write_bytes(b"")
    else:
        decision_path.write_bytes((protocol.canonical_bytes(decision) + b"\n") * 2)
    with pytest.raises(ValueError):
        assembler.assemble(source, [decision_path], runtime_map, tmp_path / "assembled")


def test_repeated_input_files_are_not_silently_deduplicated(tmp_path):
    source, decision_path, runtime_map = fixture(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        assembler.assemble(source, [decision_path, decision_path], runtime_map, tmp_path / "assembled")


@pytest.mark.parametrize("basis", ["a", "b", "new", "none"])
def test_compact_and_original_formats_have_exact_raw_source_binding(tmp_path, basis):
    source, decision_path, runtime_map = fixture(tmp_path)
    corpus = protocol.parse_jsonl((source / "corpus.jsonl").read_bytes())
    packets = protocol.prepare_inputs(protocol.parse_jsonl((source / "questions.jsonl").read_bytes()),
        protocol.parse_jsonl((source / "evidence_pool.jsonl").read_bytes()), corpus)
    left, right = (adjudicator.read_stream(source / name) for name in ("independent_a.jsonl", "independent_b.jsonl"))
    raw = dict(q1=dict(status="unresolved" if basis == "none" else "accepted", label_basis=basis,
        new_label=left["q1"]["label"] if basis == "new" else None,
        reason="Synthetic compact source review.",
        unresolved_reasons=["Synthetic unresolved period."] if basis == "none" else []))
    directory = decision_path.parent / "executions" / "compact"
    receipt = receipt_for(directory, raw)
    decisions = compact.retain_compact_decisions(raw, packets, left, right, ["q1"], receipt)
    decision_path.write_bytes(protocol.canonical_bytes(decisions[0]) + b"\n")
    output, result = assembler.assemble(source, [decision_path], runtime_map, tmp_path / "assembled")
    rows = protocol.parse_jsonl((output / "adjudications.jsonl").read_bytes())
    assert rows == decisions and result["records"] == 1
    assert rows[0]["adjudication"]["raw_encoding"] == compact.VERSION
    assert json.loads((Path(receipt["artifact_directory"]) / "raw_final.json").read_bytes()) == raw


def test_unknown_adjudication_encoding_is_rejected_not_guessed(tmp_path):
    source, decision_path, runtime_map = fixture(tmp_path)
    value = protocol.parse_jsonl(decision_path.read_bytes())[0]
    value["adjudication"]["raw_encoding"] = "unknown-synthetic-codec"
    decision_path.write_bytes(protocol.canonical_bytes(value) + b"\n")
    with pytest.raises(ValueError, match="Unknown raw adjudication"):
        assembler.assemble(source, [decision_path], runtime_map, tmp_path / "assembled")


def test_receipt_artifact_directory_cannot_redirect_retainer_reads(tmp_path):
    source, decision_path, runtime_map = fixture(tmp_path)
    decision = protocol.parse_jsonl(decision_path.read_bytes())[0]
    receipt_path = Path(decision["adjudication"]["execution_receipt_path"])
    receipt = json.loads(receipt_path.read_bytes())
    receipt["artifact_directory"] = str(tmp_path / "unrelated-host-directory")
    receipt_path.write_bytes(protocol.canonical_bytes(receipt))
    with pytest.raises(ValueError, match="artifact directory"):
        assembler.assemble(source, [decision_path], runtime_map, tmp_path / "assembled")
