"""Versioned source-only distinct-task augmentation, never answer annotation.

Only original unchanged paragraphs and earlier question STRINGS from the SAME
packet are exposed. Per-packet admission is capped at two over all versions.
Semantic distinctness is the author's explicit self-check, not independent gold.
No retrieval rankings, answer labels, adjudications or reference answers are read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from uuid import uuid4

import aggregate_machine_questions as aggregation
import author_machine_questions as author
import prepare_machine_cohort as preparation
import retrieve_machine_questions as retrieval
import run_codex_annotation as native
from oil_evidence import machine_evaluation as protocol

VERSION = "codex-source-distinct-task-augmentation-v1"
EXTRA_FIELDS = {"semantic_duplicate", "new_fact_focus", "distinctness_reason"}
INSTRUCTIONS = """
This is codex-source-distinct-task-augmentation-v1, a source-only authoring pass.
Each source packet includes ONLY earlier question strings from that same packet.
Produce at most ONE new genuinely distinct standalone English research task per
packet, focusing on a DIFFERENT source fact, mechanism, contrast or scope relation.
Do not paraphrase earlier tasks, or swap a date, number, country or company name.
For each proposed question explicitly compare its meaning to earlier_questions:
return semantic_duplicate=false ONLY if a different substantive fact/relation is
needed, new_fact_focus (brief task focus, not an answer), and distinctness_reason.
If you cannot make this distinction, skip the packet. Never pad a target count.
We already have no-answer controls; this augmentation must not manufacture more
unanswerable questions. Use fact, temporal or comparison design intentions only,
with exact support_context authoring spans available by the query cutoff.
These are authoring self-checks, NOT answerability labels, answers or qrels. Do not
write answers. Source families, earlier task strings and design notes are private
authoring metadata and will be withheld from independent annotators.
Return only a records array with the normal authoring fields plus the three fields
semantic_duplicate, new_fact_focus and distinctness_reason. Unique batch local_id.
"""


def eligible_packets(packets, questions, audits, split):
    by_question = {row["record_id"]: row for row in questions}
    earlier = defaultdict(list)
    for audit in audits:
        packet_id = audit.get("packet_id")
        if packet_id is None:  # Original dev50 source selections have a different audit schema.
            continue
        question = by_question.get(audit["record_id"])
        if question is None:
            raise ValueError("Prior admitted question/audit membership differs")
        earlier[packet_id].append(question["question"])
    if any(len(values) > 2 for values in earlier.values()):
        raise ValueError("Prior cohort already exceeds two admitted questions per packet")
    return [dict(packet, earlier_questions=list(earlier[packet["packet_id"]]),
                 prior_admitted_question_count=len(earlier[packet["packet_id"]]))
            for packet in packets if packet["candidate_split"] == split and len(earlier[packet["packet_id"]]) < 2]


def validate_candidates(raw, packets, previous_signatures=()):
    parsed = json.loads(raw, object_pairs_hook=protocol._unique_object, parse_constant=protocol._invalid_constant)
    if not isinstance(parsed, dict) or set(parsed) != {"records"} or not isinstance(parsed["records"], list):
        raise ValueError("Augmentation output must contain only a records array")
    if len(parsed["records"]) > len(packets):
        raise ValueError("At most one proposed task per source packet")
    question_rows, audits, rejects, used = [], [], [], set()
    prior = list(previous_signatures)
    for index, row in enumerate(parsed["records"]):
        try:
            if not isinstance(row, dict) or set(row) != preparation.GENERATOR_FIELDS | EXTRA_FIELDS:
                raise ValueError("Augmentation fields differ from the explicit versioned protocol")
            if row["packet_id"] in used:
                raise ValueError("Repeated packet in one-task augmentation")
            used.add(row["packet_id"])
            if row["semantic_duplicate"] is not False:
                raise ValueError("Author identified a semantic duplicate; retain but do not admit")
            if any(not isinstance(row[key], str) or len(row[key].strip()) < 15 for key in ("new_fact_focus", "distinctness_reason")):
                raise ValueError("Missing explicit fact-focus/semantic-distinctness author self-check")
            if row["scenario_intent"] == "evidence_sufficiency" or any(anchor["role"] != "support_context" for anchor in row["anchor_spans"]):
                raise ValueError("Unanswerable/future-control augmentation is not used to fill counts")
            base = {key: row[key] for key in preparation.GENERATOR_FIELDS}
            result = author.validate_candidates(protocol.canonical_bytes(dict(records=[base])), packets, prior,
                                                 split=packets[0]["candidate_split"])
            if result["rejected"]:
                raise ValueError(result["rejected"][0]["reasons"][0])
            if len(result["questions"]) != 1:
                raise ValueError("Augmentation did not yield one genuine mechanically valid question")
            question, audit = result["questions"][0], result["authoring"][0]
            audit.update(augmentation_protocol=VERSION, semantic_duplicate_author_selfcheck=False,
                         new_fact_focus=row["new_fact_focus"], distinctness_reason=row["distinctness_reason"],
                         semantic_distinctness_independently_verified=False)
            source_packet = next(packet for packet in packets if packet["packet_id"] == row["packet_id"])
            if source_packet.get("source_preparation_receipt_sha256"):
                audit["source_preparation_receipt_sha256"] = source_packet["source_preparation_receipt_sha256"]
            question_rows.append(question)
            audits.append(audit)
            prior.append(preparation.normalized(question["question"]))
        except (ValueError, TypeError, KeyError, IndexError) as error:
            rejects.append(dict(raw_record_index=index, reasons=[str(error)], status="retained_rejected_augmentation_not_label"))
    return dict(questions=question_rows, authoring=audits, rejected=rejects, raw_generated_records=len(parsed["records"]),
                actual_validated_questions=len(question_rows),
                scenario_intent_counts=dict(Counter(row["scenario_intent"] for row in audits)))


def load_inputs(root, cohort_path, split):
    questions, protected, receipt = retrieval.load_cohort(root, cohort_path)
    audits = protocol.parse_jsonl((cohort_path / "private_authoring_audit.jsonl").read_bytes(), "private source authoring audit")
    packets, previous, preparations = [], [], receipt.get("preparations", [])
    if not preparations:
        preparations = [dict(path=native.read_json(cohort_path / "inputs.json")["preparation"],
                             receipt_sha256=receipt["preparation_receipt_sha256"])]
    for binding in preparations:
        path = Path(binding["path"])
        source, old, _, source_protected, _ = author.load_preparation(root, path)
        if protocol.sha256((path / "receipt.json").read_bytes()) != binding["receipt_sha256"]:
            raise ValueError("Source preparation binding differs")
        packets.extend(dict(packet, source_preparation_receipt_sha256=binding["receipt_sha256"]) for packet in source)
        previous.extend(old)
        protected.update(source_protected)
    attempted = set()
    for entry in native.read_json(cohort_path / "attempts.json"):
        path = Path(entry["path"])
        selected = native.read_json(path / "selected_packets.json")
        attempted.update(packet["packet_id"] for packet in selected)
        _, _, rows = aggregation.completed_attempt(path, entry.get("preparation_receipt_sha256", receipt["preparation_receipt_sha256"]))
        previous.extend(preparation.normalized(question["question"]) for question, _, _ in rows)
    if {packet["packet_id"] for packet in packets} - attempted:
        raise ValueError("Finish the fixed main and supplemental source authoring before augmentation")
    return eligible_packets(packets, questions, audits, split), sorted(set(previous)), protected, preparations


def run_batch(root, cohort_path, split, offset=0, max_packets=20, *, executor=native.invoke_cli):
    root, cohort_path = Path(root).resolve(), Path(cohort_path).resolve()
    if split not in {"development", "test"} or type(max_packets) is not int or not 1 <= max_packets <= 20:
        raise ValueError("Explicit split and augmentation batch bound1..20 required")
    if type(offset) is not int or offset < 0:
        raise ValueError("Augmentation offset must be nonnegative")
    packets, previous, protected, bindings = load_inputs(root, cohort_path, split)
    selected = packets[offset:offset + max_packets]
    if len(selected) != max_packets:
        raise ValueError("Requested augmentation exceeds remaining genuine source packets")
    preparation_path = Path(bindings[0]["path"])
    _, _, original_prompt, _, preparation_receipt = author.load_preparation(root, preparation_path)
    text = author.GUARD + "\n" + original_prompt + "\n" + INSTRUCTIONS
    stdin = text.encode("utf-8") + b"\n\nUNTRUSTED_SOURCE_PACKET_OWN_EARLIER_QUESTIONS_JSON\n" + protocol.canonical_bytes(dict(source_packets=selected)) + b"\n"
    schema = author.output_schema(selected)
    schema["properties"]["records"]["maxItems"] = len(selected)
    record_schema = schema["properties"]["records"]["items"]
    record_schema["required"].extend(sorted(EXTRA_FIELDS))
    record_schema["properties"].update(semantic_duplicate=dict(type="boolean"), new_fact_focus=dict(type="string"), distinctness_reason=dict(type="string"))
    binary = native.discover_native_cli(native.DEFAULT_CLI)
    environment = native.child_environment()
    mcp_names = native.configured_mcp_names(environment)
    cli_version = subprocess.run([str(binary), "--version"], capture_output=True, env=environment, check=True).stdout.decode().strip()
    login = subprocess.run([str(binary), "login", "status"], capture_output=True, env=environment, check=True)
    if (login.stdout + login.stderr).decode().strip() != "Logged in using ChatGPT":
        raise ValueError("Existing subscription login required; no API fallback")
    for path in (Path(__file__), Path(author.__file__), Path(native.__file__), Path(protocol.__file__), Path(preparation.__file__)):
        protected[str(path.resolve())] = protocol.sha256(path.read_bytes())
    native.protected_unchanged(protected)
    output = root / "data/machine_runs/question-authoring" / (native.utc_now().replace(":", "-") + "-" + uuid4().hex[:10])
    output.mkdir(exist_ok=False)
    context = Path(tempfile.mkdtemp(prefix="oil-codex-source-augmentation-"))
    specification = dict(schema_version=VERSION, authoring_split=split, source_packet_offset=offset,
        cli_version=cli_version, native_cli_sha256=protocol.sha256(binary.read_bytes()),
        disabled_configured_mcp_servers=mcp_names,
        source_packets_selected=len(selected), preparation_receipt_sha256=bindings[0]["receipt_sha256"],
        preparation=str(preparation_path), prior_cohort=str(cohort_path), all_preparations=bindings,
        source_packet_preparation_sha256={packet["packet_id"]: packet["source_preparation_receipt_sha256"] for packet in selected},
        original_bindings=preparation_receipt["bindings"], human_gold=False, maximum_question_candidates=len(selected))
    for name, value in (("stdin.txt", stdin), ("protocol.txt", text.encode()), ("schema.json", schema),
                        ("selected_packets.json", selected), ("specification.json", specification),
                        ("protected_inputs.json", protected), ("authoring_script.py", Path(__file__).read_bytes())):
        native.write_new(output / name, value)
    native.write_new(context / "schema.json", schema)
    native.write_new(context / "packet.json", dict(source_packets=selected))
    native.write_new(context / "protocol.txt", text.encode())
    final = context / "final.json"
    command = native.cli_command(binary, context, context / "schema.json", final, None, mcp_names)
    native.write_new(output / "command.json", command)
    print(f"Source-only augmentation {split} offset{offset}: {len(selected)} original paragraphs; at most one distinct task each", flush=True)
    tick = perf_counter()
    try:
        exit_code = executor(command, stdin, context, output, environment, 900)
        if final.exists():
            native.write_new(output / "raw_final.json", final.read_bytes())
        events = native.parse_execution_events((output / "events.jsonl").read_bytes())
        metadata = dict(exit_code=exit_code, errors=events["errors"], elapsed_seconds=perf_counter() - tick,
            usage=events["usage"], observed_model=events["observed_model"], execution_id=events["thread_id"],
            subscription_dollar_cost=None, full_stdin_sha256=protocol.sha256(stdin), external_paid_api_calls=0)
        native.write_new(output / "execution_metadata.json", metadata)
        if exit_code != 0 or events["errors"]:
            raise ValueError("Native source augmentation failed isolation or completion checks")
        result = validate_candidates((output / "raw_final.json").read_bytes(), selected, previous)
        for name, rows in (("questions.jsonl", result["questions"]), ("authoring_anchors.jsonl", result["authoring"])):
            native.write_new(output / name, b"".join(protocol.canonical_bytes(row) + b"\n" for row in rows))
        native.write_new(output / "rejected_candidates.json", result["rejected"])
        native.protected_unchanged(protected)
        receipt = dict(status=split + "_question_candidates_authored_not_labels_or_frozen_cohort", augmentation_protocol=VERSION,
            authoring_split=split, source_packets_selected=len(selected), raw_generated_records=result["raw_generated_records"],
            actual_validated_questions=result["actual_validated_questions"], rejected_records=len(result["rejected"]),
            scenario_intent_counts=result["scenario_intent_counts"], original_inputs_unchanged=True,
            answer_annotations_created=0, human_gold=False, human_labels_created=0, external_paid_api_calls=0,
            subscription_usage=events["usage"], semantic_quality="author_selfcheck_not_independently_established",
            observed_identity=dict(service="codex_cli_chatgpt_subscription", requested_model="cli_default_unresolved",
                observed_model=events["observed_model"], execution_id=events["thread_id"],
                observation_source="execution_metadata" if events["observed_model"] else "unavailable",
                identity_note="Native event metadata only; never assistant self-report."),
            artifact_sha256={path.name: protocol.sha256(path.read_bytes()) for path in output.iterdir() if path.is_file()},
            limitations=["Source-driven augmentation changes the authoring prompt and may introduce synthetic benchmark bias.",
                         "Explicit author self-check is not independent semantic-duplicate verification.",
                         "Only globally admitted questions count; raw proposals and source packets do not."])
        native.write_new(output / "receipt.json", receipt)
        print(json.dumps(dict(output=str(output), actual_validated_questions=result["actual_validated_questions"],
                              raw_generated_records=result["raw_generated_records"], rejected_records=len(result["rejected"])), indent=2), flush=True)
        return output, receipt
    except Exception as error:
        native.write_new(output / "failure.json", dict(error_type=type(error).__name__, error=str(error), accepted=False))
        raise


def run_until_targets(root, cohort_path, *, max_native_calls=50):
    """Finite versioned schedule; stop on actual targets, exhausted packets or error."""
    root, cohort_path = Path(root).resolve(), Path(cohort_path).resolve()
    if type(max_native_calls) is not int or not 1 <= max_native_calls <= 50:
        raise ValueError("Explicit finite augmentation limit1..50 native calls required")
    questions, _, receipt = retrieval.load_cohort(root, cohort_path)
    bindings = receipt["preparations"]
    plan = []
    for split in ("development", "test"):
        packets, _, _, _ = load_inputs(root, cohort_path, split)
        plan.extend(dict(split=split, offset=offset, packets=min(20, len(packets) - offset))
                    for offset in range(0, len(packets), 20))
    parent = root / "data/machine_runs/question-augmentation-schedules"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / (native.utc_now().replace(":", "-") + "-" + uuid4().hex[:10])
    output.mkdir(exist_ok=False)
    native.write_new(output / "plan.json", dict(protocol=VERSION, initial_cohort=str(cohort_path),
        batches=plan, maximum_native_calls=max_native_calls, max_concurrent=2, authoring_prompt_changed=True,
        model_labels_or_rankings_shown=False, new_tasks_per_packet=1, total_admitted_tasks_per_packet=2))
    print(f"Finite source augmentation schedule: {output}", flush=True)
    counts = dict(Counter(row["split"] for row in questions))
    results, failures, calls = [], [], 0
    current_cohort = cohort_path
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending, next_batch = {}, 0
        while pending or next_batch < len(plan):
            stop = bool(failures) or (output / "STOP_REQUESTED").exists() or all(
                counts.get(split, 0) >= target for split, target in preparation.TARGETS.items())
            while not stop and len(pending) < 2 and next_batch < len(plan) and calls < max_native_calls:
                item = plan[next_batch]
                next_batch += 1
                if counts.get(item["split"], 0) >= preparation.TARGETS[item["split"]]:
                    continue
                pending[workers.submit(run_batch, root, cohort_path, item["split"], item["offset"], item["packets"])] = item
                calls += 1
            if not pending:
                break
            ready, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in ready:
                item = pending.pop(future)
                try:
                    attempt, result = future.result()
                    results.append(dict(**item, output=str(attempt), actual_questions=result["actual_validated_questions"],
                        receipt_sha256=protocol.sha256((attempt / "receipt.json").read_bytes())))
                    current_cohort, admission = aggregation.run(root, Path(bindings[0]["path"]),
                        additional_preparations=[Path(binding["path"]) for binding in bindings[1:]])
                    counts = admission["actual_counts"]
                except Exception as error:
                    failures.append(dict(**item, error_type=type(error).__name__, error=str(error)))
    summary = dict(output=str(output), protocol=VERSION, final_candidate_cohort=str(current_cohort), actual_counts=counts,
        source_batches_completed=len(results), new_native_calls=calls, completed=results, failures=failures,
        target_shortfall={split: max(0, target - counts.get(split, 0)) for split, target in preparation.TARGETS.items()},
        qa_annotations_created=0, human_gold=False, semantic_distinctness_independently_verified=False,
        status="question_targets_met_not_machine_gold" if all(counts.get(split, 0) >= target for split, target in preparation.TARGETS.items())
        else "retained_question_shortfall_no_padding", authoring_prompt_changed=True)
    native.write_new(output / "receipt.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return output, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--split", choices=["development", "test"], required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-packets", type=int, default=20)
    args = parser.parse_args()
    run_batch(args.canonical_root, args.cohort, args.split, args.offset, args.max_packets)
