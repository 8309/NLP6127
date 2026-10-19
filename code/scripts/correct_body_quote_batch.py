"""One reviewed source-only body-quote corrective pair; original failures stay intact."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
from uuid import uuid4


def isolated(name, filename, expected):
    path = Path(__file__).with_name(filename)
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Frozen primitive implementation changed")
    spec = importlib.util.spec_from_file_location(name + uuid4().hex, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERSION = "source-only-body-quote-corrective-pair-v1"
ROLE = "body_quote_corrective_pair"
correction = isolated("body_quote_source_", "correct_codex_wire_batch.py", "8f4bb16bf70aa277d2ac14d5e0def7d34da28ba1d46e1cae3c92e85e0d14bdac")
engine = isolated("body_quote_execution_", "run_codex_transport_recovery.py", "465d5ee77a5c9d1ef33a8791d0eb125852e543b0a5ac4aff9142d9e5015ac5c2")
native, protocol, codec = correction.native, correction.protocol, correction.codec
ORIGINAL_INSTRUCTION = correction.CORRECTION_INSTRUCTION
BODY_INSTRUCTION = """

EXPLICIT BODY-ONLY FACT AND QUOTE REQUIREMENT
Derive factual answer-point content and exact quotes only from each passage's
text field. Its title, source name and URL are metadata, not substitutes for
body facts or quotations. Keep date metadata for the unchanged cutoff checks.
Do not turn forecasts, expectations or announced plans into realized results.
If the body does not support a requested fact, preserve the evidence gap under
the unchanged annotation standards; do not force an answer or relevance grade.
"""
correction.VERSION = VERSION
correction.CORRECTION_INSTRUCTION = ORIGINAL_INSTRUCTION + BODY_INSTRUCTION
INSTRUCTION_SHA256 = protocol.sha256(correction.CORRECTION_INSTRUCTION.encode())
engine.VERSION, engine.INSTRUCTION_SHA256, engine.clarification = VERSION, INSTRUCTION_SHA256, correction
MANIFEST_SHA256 = "a9f99fa62ccf28dffec8dd144603ea9fe79676afac4db992849728a8e84a3cc5"
PROGRESS_SHA256 = "9e6ff4e03acbe4dff9e7aadc994411da04715e3ca8176592a3ceb304cfe9695e"
FAILURE_SHA256 = {"a": "0943caee6e1f66552c430ec8a20d6a03f489f968112c60e50541c84832d0697d",
    "b": "7e5ecae36cba345dd4866ff0c565a98b0c755c59819d614688a5024f2f9d370f"}
FAILED_BATCH = "batch-0026"
BAD_RECORD_ID = "test-f4a76745ee219167836f"
BAD_PASSAGE_ID = "7d1099962fd5832d8750-u-b77cbee93a0b7308063f66c1"
SOURCE_ERROR = "Decoded canonical label invalid: Quote is not an exact frozen evidence span"


def source_context(source_run):
    source = Path(source_run).resolve()
    if (protocol.sha256((source / "manifest.json").read_bytes()) != MANIFEST_SHA256
            or protocol.sha256((source / "progress.jsonl").read_bytes()) != PROGRESS_SHA256):
        raise ValueError("Only the pinned reviewed stopped source may be corrected")
    manifest = native.read_json(source / "manifest.json")
    spec = manifest["specification"]
    if spec["version"] != "codex-unchanged-input-transport-recovery-v1":
        raise ValueError("Unknown source protocol")
    protected = dict(native.read_json(source / "protected_inputs.json"))
    protected.update({str(native.contained_file(source, name)): digest for name, digest in manifest["immutable_sha256"].items()})
    native.protected_unchanged(protected)
    packets = native.read_json(source / "packets.json")
    group = next(group for group in spec["groups"] if group["batch"] == FAILED_BATCH)
    ids = group["record_ids"]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("The complete original ten-record group is required")
    rows = [packets[rid]["payload"] for rid in ids]
    baseline = codec.prepare_batch(rows, correction.batch.BATCH_PROMPT, native.output_schema)
    old_prompt = baseline["prompt"] + ORIGINAL_INSTRUCTION
    old_stdin = old_prompt.encode() + b"\n\nUNTRUSTED_RECORD_PACKETS_JSON\n" + protocol.canonical_bytes(baseline["wire_payload"]) + b"\n"
    progress = protocol.parse_jsonl((source / "progress.jsonl").read_bytes())
    failed = [event for event in progress if event.get("accepted") is False or event.get("type") == "failed"]
    if len(failed) != 2 or {event.get("stream") for event in failed} != {"a", "b"}:
        raise ValueError("Exactly the two registered failed streams are required")
    bindings, contexts = {}, set()
    for event in failed:
        stream = event["stream"]
        path = native.contained_file(source, event["receipt"])
        digest = protocol.sha256(path.read_bytes())
        receipt = native.read_json(path)
        if (digest != FAILURE_SHA256[stream] or event["receipt_sha256"] != digest
                or event["batch"] != FAILED_BATCH or event["record_ids"] != ids
                or receipt.get("accepted") is not False or receipt["stream"] != stream or receipt["batch"] != FAILED_BATCH
                or receipt["record_ids"] != ids or receipt.get("process_exit_code") != 0
                or receipt.get("errors") != [SOURCE_ERROR] or receipt["usage"] is None
                or receipt["annotation_manifest_sha256"] != MANIFEST_SHA256 or receipt["execution_protocol"] != spec["version"]):
            raise ValueError("Native, quota, mixed or unregistered failures are not eligible")
        correction.retry.verify_raw_artifacts(path, receipt)
        events = native.parse_execution_events((path.parent / "events.jsonl").read_bytes())
        if (events["errors"] or events["thread_id"] != receipt["execution_id"] or events["usage"] != receipt["usage"]
                or receipt["execution_id"] in contexts):
            raise ValueError("Original completed native contexts and measured usage changed")
        contexts.add(receipt["execution_id"])
        if ((path.parent / "stdin.txt").read_bytes() != old_stdin
                or (path.parent / "schema.json").read_bytes() != protocol.canonical_bytes(baseline["schema"])
                or (path.parent / "codec.json").read_bytes() != protocol.canonical_bytes(baseline["codec"])
                or (path.parent / "wire_payload.json").read_bytes() != protocol.canonical_bytes(baseline["wire_payload"])
                or (path.parent / "canonical_semantic_packets.json").read_bytes() != protocol.canonical_bytes(baseline["canonical_semantic_packets"])
                or correction.normalized_command(native.read_json(path.parent / "command.json")) != spec["native_command_identity"]
                or correction.retry.requested_model_from_command(path.parent / "command.json") != spec["requested_model"]):
            raise ValueError("Original full input/schema/source/model/isolation changed")
        raw = (path.parent / "raw_final.json").read_bytes()
        codec.verify_codec(baseline["codec"], rows, baseline["wire_payload"])
        parsed = json.loads(raw, object_pairs_hook=protocol._unique_object, parse_constant=protocol._invalid_constant)
        if not isinstance(parsed, dict) or set(parsed) != set(baseline["codec"]["records"]):
            raise ValueError("Original complete alias coverage changed")
        bad = []
        for row in rows:
            try:
                codec.decode_record(raw, baseline["codec"], row)
            except ValueError as error:
                if str(error) != SOURCE_ERROR:
                    raise ValueError("Unknown original semantic defect") from error
                bad.append(row["record_id"])
        if bad != [BAD_RECORD_ID]:
            raise ValueError("Only the declared single title-not-body record may be corrected")
        binding = next((alias, value) for alias, value in baseline["codec"]["records"].items() if value["record_id"] == BAD_RECORD_ID)
        label = codec._map_label(parsed[binding[0]]["label"], binding[1]["passage_ids"])
        passages = {value["passage_id"]: value for value in packets[BAD_RECORD_ID]["payload"]["evidence"]}
        bad_quotes = [quote for quote in label["quotes"] if quote["text"] not in passages[quote["passage_id"]]["text"]]
        if (len(bad_quotes) != 1 or bad_quotes[0]["passage_id"] != BAD_PASSAGE_ID
                or bad_quotes[0]["text"] != passages[BAD_PASSAGE_ID].get("title")):
            raise ValueError("Original quote defect is not the reviewed title-only span")
        bindings[stream] = dict(path=str(path), sha256=digest, execution_id=receipt["execution_id"], record_ids=ids,
            diagnostic=dict(valid_records=9, invalid_records=1, defect="title_not_body_quote"))
    protected.update({str(path): protocol.sha256(path.read_bytes()) for path in source.rglob("*") if path.is_file()})
    protected[str(Path(__file__).resolve())] = protocol.sha256(Path(__file__).read_bytes())
    native.protected_unchanged(protected)
    return dict(source_run=str(source), rows=rows, record_ids=ids, source_specification=spec,
        rejected_receipts=bindings, protected_sha256=protected)


def corrective_group(context, prepared):
    return dict(batch=FAILED_BATCH, record_ids=context["record_ids"], annotation_role=ROLE, original_first_attempt=False,
        full_stdin_sha256=prepared["actual_wire_stdin_sha256"], schema_sha256=prepared["actual_wire_schema_sha256"],
        mapping_sha256=prepared["codec"]["mapping_sha256"], full_payload_bytes=len(prepared["stdin"]) + len(protocol.canonical_bytes(prepared["schema"])))


def prepare(source_run):
    context = source_context(source_run)
    prepared = correction.prepare_source_batch(context["rows"])
    group = corrective_group(context, prepared)
    source = Path(source_run).resolve()
    output = source.parent.parent / "cohort_annotations_body_quote_corrected" / ("run-" + uuid4().hex[:10])
    spec = dict(version=VERSION, groups=[group], requested_model=context["source_specification"]["requested_model"],
        native_command_identity=context["source_specification"]["native_command_identity"], record_ids=context["record_ids"],
        source_run=str(source), failed_receipts=context["rejected_receipts"], native_calls_allowed=2,
        native_calls_prepared=0, original_agreement_substitution_allowed=False, human_gold=False,
        instruction_sha256=INSTRUCTION_SHA256)
    values = {"context.json": context, "packets.json": prepared["canonical_semantic_packets"],
        "stdin.txt": prepared["stdin"], "schema.json": prepared["schema"], "codec.json": prepared["codec"],
        "wire_payload.json": prepared["wire_payload"], "instruction.txt": correction.CORRECTION_INSTRUCTION.encode(),
        "runner_script.py": Path(__file__).read_bytes()}
    for name, value in values.items():
        native.write_new(output / name, value)
    native.write_new(output / "manifest.json", dict(specification=spec,
        immutable_sha256={name: protocol.sha256((output / name).read_bytes()) for name in values}))
    return output, native.read_json(output / "manifest.json")


def verify_prepared(directory):
    directory = Path(directory).resolve()
    manifest = native.read_json(directory / "manifest.json")
    for name, digest in manifest["immutable_sha256"].items():
        if protocol.sha256(native.contained_file(directory, name).read_bytes()) != digest:
            raise ValueError("Prepared body correction artifact changed")
    context = native.read_json(directory / "context.json")
    native.protected_unchanged(context["protected_sha256"])
    if source_context(context["source_run"]) != context:
        raise ValueError("Body correction source lineage changed")
    spec = manifest["specification"]
    prepared = correction.prepare_source_batch(context["rows"])
    if (spec["version"] != VERSION or spec["instruction_sha256"] != INSTRUCTION_SHA256
            or spec["groups"] != [corrective_group(context, prepared)] or spec["record_ids"] != context["record_ids"]
            or spec["requested_model"] != context["source_specification"]["requested_model"]
            or spec["native_command_identity"] != context["source_specification"]["native_command_identity"]
            or spec["failed_receipts"] != context["rejected_receipts"] or spec["native_calls_allowed"] != 2
            or spec["original_agreement_substitution_allowed"] is not False
            or (directory / "stdin.txt").read_bytes() != prepared["stdin"]
            or (directory / "schema.json").read_bytes() != protocol.canonical_bytes(prepared["schema"])):
        raise ValueError("Body correction protocol changed")
    return manifest, context


def execute_pair_once(directory, authorization_path, *, executor=native.invoke_cli, native_setup=None, timeout_seconds=900):
    directory = Path(directory).resolve()
    manifest, context = verify_prepared(directory)
    authorization_path, authorization = correction.validate_authorization(directory, authorization_path)
    if any((directory / stream).exists() for stream in native.STREAMS):
        raise ValueError("Body corrective pair is once-only; no automatic retry")
    binary, environment, mcp = native_setup or (native.discover_native_cli(native.DEFAULT_CLI), native.child_environment(), None)
    if mcp is None:
        mcp = native.configured_mcp_names(environment)
    old = context["source_specification"]
    if protocol.sha256(Path(binary).read_bytes()) != old["native_cli_sha256"] or mcp != old["disabled_mcp_names"]:
        raise ValueError("Body correction native identity or tool isolation changed")
    protected = {**context["protected_sha256"], str(authorization_path): protocol.sha256(authorization_path.read_bytes()),
        str(directory / "manifest.json"): protocol.sha256((directory / "manifest.json").read_bytes())}
    protected.update({str(directory / name): digest for name, digest in manifest["immutable_sha256"].items()})
    packets, group = native.read_json(directory / "packets.json"), manifest["specification"]["groups"][0]
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = {stream: workers.submit(engine.execute_attempt, directory, stream, group, packets, binary, environment, mcp,
            protected, old["requested_model"], timeout_seconds, executor) for stream in native.STREAMS}
        results = {stream: future.result() for stream, future in futures.items()}
    native.protected_unchanged(protected)
    accepted = all(result["receipt"]["accepted"] for result in results.values())
    records = {stream: engine.verify_attempt(value["attempt"], value["receipt"], packets) for stream, value in results.items()} if accepted else {}
    old_contexts = {value["execution_id"] for value in context["rejected_receipts"].values()}
    if accepted and (len({value["receipt"]["execution_id"] for value in results.values()}) != 2
            or any(value["receipt"]["execution_id"] in old_contexts for value in results.values())):
        raise ValueError("Body correction contexts are not independent and fresh")
    agreement = protocol.raw_agreement(packets, list(records["a"].values()), list(records["b"].values())) if accepted else None
    if accepted and agreement["paired_records"] != 10:
        raise ValueError("Both complete corrective streams must have equal actual inputs")
    native.write_new(directory / "execution_usage.jsonl", b"".join(protocol.canonical_bytes(value["usage"]) + b"\n" for value in results.values()))
    certificate = dict(version=VERSION, correction_accepted=accepted, native_calls_actual=2, record_ids=context["record_ids"],
        original_agreement_substitution_allowed=False, corrected_labels_still_require_source_adjudication=True, human_gold=False,
        manifest_sha256=protocol.sha256((directory / "manifest.json").read_bytes()), rejected_receipts=context["rejected_receipts"],
        authorization_sha256=protocol.sha256(authorization_path.read_bytes()), corrective_pair_agreement=agreement,
        receipts={stream: dict(path=str(value["attempt"] / "receipt.json"), sha256=protocol.sha256((value["attempt"] / "receipt.json").read_bytes())) for stream, value in results.items()})
    native.write_new(directory / "certificate.json", certificate)
    return {stream: value["receipt"] for stream, value in results.items()}, certificate


def completed_correction_records(directory, stream):
    directory = Path(directory).resolve()
    manifest, context = verify_prepared(directory)
    certificate = native.read_json(directory / "certificate.json")
    if (certificate["correction_accepted"] is not True or certificate["native_calls_actual"] != 2
            or certificate["manifest_sha256"] != protocol.sha256((directory / "manifest.json").read_bytes())
            or certificate["rejected_receipts"] != context["rejected_receipts"]
            or certificate["original_agreement_substitution_allowed"] is not False):
        raise ValueError("Both whole body-corrective streams must pass")
    results = {}
    packets = native.read_json(directory / "packets.json")
    for name in native.STREAMS:
        value = certificate["receipts"][name]
        path = native.contained_file(directory, Path(value["path"]).relative_to(directory).as_posix())
        if protocol.sha256(path.read_bytes()) != value["sha256"]:
            raise ValueError("Body corrective receipt lineage changed")
        results[name] = engine.verify_attempt(path.parent, native.read_json(path), packets)
    agreement = protocol.raw_agreement(packets, list(results["a"].values()), list(results["b"].values()))
    if agreement["paired_records"] != 10 or protocol.canonical_bytes(agreement) != protocol.canonical_bytes(certificate["corrective_pair_agreement"]):
        raise ValueError("Body corrective equal-input pair agreement changed")
    return results[stream]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("prepare")
    before.add_argument("--source-run", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--prepared-directory", type=Path, required=True)
    execute.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        directory, manifest = prepare(args.source_run)
        print(json.dumps(dict(directory=str(directory), native_calls=0,
            manifest_sha256=protocol.sha256((directory / "manifest.json").read_bytes()))))
    else:
        receipts, certificate = execute_pair_once(args.prepared_directory, args.authorization)
        print(json.dumps(certificate, indent=2))
        raise SystemExit(0 if certificate["correction_accepted"] else 1)
