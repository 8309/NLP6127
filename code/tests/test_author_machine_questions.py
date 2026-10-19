"""Synthetic authoring-runner tests; no hosted models or real QA counts."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("isolated_question_author", scripts / "author_machine_questions.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def packets():
    result = []
    for index, split in enumerate(("development", "test", "development")):
        text = (f"Synthetic country {index} reported that crude oil exports increased in 2024. "
                "Reported crude exports were three million barrels per day and imports were one million barrels per day. "
                "The observed flow does not establish transport capacity, a later operational disruption, or any forecast.")
        result.append(dict(packet_id=f"synthetic-packet-{index}", event_family_id=f"synthetic-family-{index}",
                           candidate_split=split, source="SYNTHETIC", publication_year="2024", primary_domain="trade_transport",
                           evidence=[dict(passage_id=f"synthetic-passage-{index}", document_id=f"synthetic-document-{index}",
                                          source_page_id=f"synthetic-page-{index}", available_by="2024-03-01", text=text,
                                          char_start=5, char_end=5 + len(text), page=1)],
                           provenance="accepted_source_packet_not_question_or_label", human_gold=False,
                           questions_created=0, uncertainty_flags=["semantic_event_independence_unverified"]))
    return result


def record(packet, question=None, local_id="q001"):
    return dict(local_id=local_id, packet_id=packet["packet_id"],
                question=question or "How did synthetic country zero's crude oil exports change during March 2024?",
                as_of="2024-03-31", scenario_intent="fact", design_reason="Synthetic source-first task only.",
                anchor_spans=[dict(passage_id=packet["evidence"][0]["passage_id"],
                                  text="crude oil exports increased", role="support_context")])


def raw(rows):
    return json.dumps(dict(records=rows), ensure_ascii=False, indent=2).encode("utf-8")


def test_only_development_packets_are_selected_without_touching_test(module, packets):
    selected = module.select_development(packets, max_packets=2)
    assert [packet["packet_id"] for packet in selected] == ["synthetic-packet-0", "synthetic-packet-2"]
    assert module.select_development(packets, max_packets=1, offset=1) == [packets[2]]


@pytest.mark.parametrize("count,offset", [(0, 0), (21, 0), (True, 0), (3, 0), (1, -1), (1, True)])
def test_batch_limit_and_offsets_fail_closed(module, packets, count, offset):
    with pytest.raises(ValueError):
        module.select_development(packets, max_packets=count, offset=offset)


def test_prompt_binds_guard_generator_text_and_full_source_bytes(module, packets):
    selected = module.select_development(packets, max_packets=1)
    prompt = module.stdin_bytes(selected, "Synthetic generator instructions")
    assert prompt.startswith(module.GUARD.encode("utf-8"))
    assert b"Synthetic generator instructions" in prompt
    assert b"UNTRUSTED_ACCEPTED_SOURCE_PACKETS_JSON" in prompt
    assert selected[0]["evidence"][0]["text"].encode("utf-8") in prompt
    changed = deepcopy(selected)
    changed[0]["evidence"][0]["text"] += " Extra source content."
    assert prompt != module.stdin_bytes(changed, "Synthetic generator instructions")


def test_output_schema_has_no_answer_or_relevance_label_fields(module, packets):
    schema = module.output_schema(module.select_development(packets, max_packets=1))
    row = schema["properties"]["records"]["items"]
    assert set(row["properties"]) == module.cohort.GENERATOR_FIELDS
    assert not module.cohort.FORBIDDEN_LABEL_FIELDS.intersection(row["properties"])
    assert schema["properties"]["records"]["maxItems"] == 2
    assert row["properties"]["packet_id"]["enum"] == ["synthetic-packet-0"]


def test_real_rows_are_validated_without_reserializing_or_padding_raw_output(module, packets):
    selected = module.select_development(packets, max_packets=1)
    original = raw([record(selected[0])])
    result = module.validate_candidates(original, selected)
    assert result["raw_generated_records"] == result["actual_validated_questions"] == 1
    assert result["answer_annotations_created"] == result["human_labels_created"] == 0
    assert original == raw([record(selected[0])])
    assert set(result["questions"][0]) == {"record_id", "question", "as_of", "event_group", "split"}
    assert "anchor_spans" not in result["questions"][0]


def test_date_only_question_changes_are_rejected_as_nonnumeric_templates(module, packets):
    selected = module.select_development(packets, max_packets=1)
    first = record(selected[0])
    second = record(selected[0], question=first["question"].replace("2024", "2023"), local_id="q002")
    result = module.validate_candidates(raw([first, second]), selected)
    assert result["actual_validated_questions"] == 1
    assert result["rejected"][0]["raw_record_index"] == 1
    assert "Nonnumeric" in result["rejected"][0]["reasons"][0]


def test_nonnumeric_legacy_template_is_not_a_new_question(module, packets):
    selected = module.select_development(packets, max_packets=1)
    current = record(selected[0])
    previous = module.cohort.normalized(current["question"].replace("2024", "2019"))
    result = module.validate_candidates(raw([current]), selected, previous_signatures=[previous])
    assert result["actual_validated_questions"] == 0
    assert len(result["rejected"]) == 1


def test_rejected_wrong_quotes_are_retained_not_silently_repaired(module, packets):
    selected = module.select_development(packets, max_packets=1)
    bad = record(selected[0])
    bad["anchor_spans"][0]["text"] = "Invented source span"
    result = module.validate_candidates(raw([bad]), selected)
    assert result["actual_validated_questions"] == 0
    assert result["rejected"][0]["raw_record_index"] == 0
    assert "exact packet span" in result["rejected"][0]["reasons"][0]


def test_duplicate_local_id_is_rejected_even_after_prior_bad_candidate(module, packets):
    selected = module.select_development(packets, max_packets=1)
    bad = record(selected[0])
    bad["anchor_spans"][0]["text"] = "Invented span"
    result = module.validate_candidates(raw([bad, record(selected[0])]), selected)
    assert result["actual_validated_questions"] == 0
    assert "Duplicate" in result["rejected"][1]["reasons"][0]


def test_raw_batch_cannot_exceed_two_records_per_source_bound(module, packets):
    selected = module.select_development(packets, max_packets=1)
    with pytest.raises(ValueError, match="authorized question bound"):
        module.validate_candidates(raw([record(selected[0], local_id=f"q{i:03d}") for i in range(3)]), selected)


def test_test_split_cannot_be_authored_by_this_development_runner(module, packets):
    test = packets[1]
    with pytest.raises(ValueError, match="Held-out test"):
        module.validate_candidates(raw([record(test)]), [test])


def test_english_gate_detects_han_without_hiding_it_as_escapes(module, packets):
    selected = module.select_development(packets, max_packets=1)
    bad = record(selected[0], question=chr(0x4E00) + " What crude oil export development did synthetic country zero report?")
    result = module.validate_candidates(raw([bad]), selected)
    assert result["actual_validated_questions"] == 0
    assert "English" in result["rejected"][0]["reasons"][0]


def test_successful_isolated_runner_preserves_raw_bytes_usage_null_identity_and_original_inputs(module, packets, tmp_path, monkeypatch):
    root = tmp_path / "canonical"
    (root / "data").mkdir(parents=True)
    selected = module.select_development(packets, max_packets=1)
    protected_file = root / "original.txt"
    protected_file.write_bytes(b"synthetic original input")
    protected = {str(protected_file): module.protocol.sha256(protected_file.read_bytes())}
    preparation = root / "preparation"
    preparation.mkdir()
    (preparation / "receipt.json").write_bytes(b"{}")
    monkeypatch.setattr(module, "load_preparation", lambda *_: (selected, [], "Synthetic generator rules", protected,
                                                               dict(bindings={"original_input_count": 71, "original_implementation_file_count": 19})))
    executable = tmp_path / "synthetic-codex.exe"
    executable.write_bytes(b"not executable; mocked")
    monkeypatch.setattr(module.native, "discover_native_cli", lambda _: executable)
    monkeypatch.setattr(module.native, "configured_mcp_names", lambda _: ["synthetic_server"])
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: SimpleNamespace(
        stdout=b"codex-cli synthetic\n" if "--version" in args else b"Logged in using ChatGPT\n", stderr=b""))
    response = raw([record(selected[0])])
    captured = {}

    def executor(command, prompt, context, output, environment, timeout):
        captured.update(command=command, prompt=prompt, environment=environment)
        final = Path(command[command.index("--output-last-message") + 1])
        final.write_bytes(response)
        events = [dict(type="thread.started", thread_id="synthetic-fresh-thread"),
                  dict(type="turn.completed", usage={"input_tokens": 123, "output_tokens": 45})]
        (output / "events.jsonl").write_bytes(b"".join(module.protocol.canonical_bytes(event) + b"\n" for event in events))
        (output / "stderr.txt").write_bytes(b"")
        return 0

    output, receipt = module.run(root, preparation, max_packets=1, executor=executor)
    assert (output / "raw_final.json").read_bytes() == response
    assert (output / "stdin.txt").read_bytes() == captured["prompt"]
    assert receipt["actual_validated_questions"] == 1
    assert receipt["observed_identity"]["observed_model"] is None
    assert receipt["subscription_usage"] == {"input_tokens": 123, "output_tokens": 45}
    assert receipt["external_paid_api_calls"] == receipt["answer_annotations_created"] == 0
    assert "--ephemeral" in captured["command"]
    assert "--strict-config" in captured["command"]
    assert "--ignore-rules" not in captured["command"]
    assert "mcp_servers.synthetic_server.enabled=false" in captured["command"]
    assert module.native.EXTERNAL_KEY_NAMES.isdisjoint(captured["environment"])
    assert protected_file.read_bytes() == b"synthetic original input"
    assert json.loads((output / "questions.jsonl").read_text(encoding="utf-8"))["split"] == "development"
