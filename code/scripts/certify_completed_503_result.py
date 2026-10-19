"""Allowlisted zero-call mechanical certificates; original false receipts stay false."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
from uuid import uuid4

import jsonschema
import run_codex_annotation as native
from oil_evidence import machine_evaluation as protocol

VERSION = "completed-503-zero-call-mechanical-certificate-v1"
STATUS = "completed_output_mechanically_revalidated_with_transient_503_events"
WARNING = re.compile(r"Reconnecting\.\.\. ([2-5])/5 \(unexpected status 503 Service Unavailable: An error occurred while processing your request\. You can retry your request, or contact us through our help center at help\.openai\.com if the error persists\.\)")
KNOWN = {"annotation_b38": dict(receipt_sha256="3d3eb1746772754f9098c17ed8698e511a6d3494745b6ddbd497258a63df915c",
    manifest_sha256="81b7a9f16066245a7fb8323aaca82fa493210dd0af1de60ddf5d271b4b85109f",
    stdin_sha256="754445608ff241a32dfe838e7f45c46be7a1f75453e3286552cbbf26f6951589",
    schema_sha256="f6ceb8ad06b9fecbb55001c303a50183b7e3461aa5dcac0ab186e86ef1e40bd5",
    raw_sha256="bf5e422bd25c02c3a5a3c9a709ba6517a08eaf91a0efad5363c98d93598aa1d1",
    command_sha256="e3e51252546c32ec0eda037146ffe9a2c1003f25eb637207b1cc44c9532bea22",
    native_binary_sha256="d13914ced6c7af174d8d638db938284231227312104b06a25e9608c632a7daec"),
    "source_group2": dict(receipt_sha256="f76bad2521713df93e9de8ccc8fc32cf501d4ecb6160d8940efd34de14dcfbda",
        manifest_sha256="77539763483364992830b0479298e3387c427b280cd6fbc427fe0fa95b4ed15a",
        stdin_sha256="7e7cb037fb777f4206eb844aaf72e3751b89c6899905dc9948cc9f2c90f5294f",
        schema_sha256="605775e0106f491ef20fc805d607fa7d1cfb7e0324a8f3551f7be7e30729fcf5",
        raw_sha256="88582b87f28ff816dfb69b43b91fd9d1236210198482a51920c74e053af46bb1",
        native_binary_sha256="d13914ced6c7af174d8d638db938284231227312104b06a25e9608c632a7daec",
        source_sha256={"packets.json": "9d8477bb6beee1bb64fda216ea3db4d219dfe5bd537a78f9d9aa48da8e43d520",
            "independent_a.jsonl": "5dde95300bfdb11aa37ff786af44af50eb3ef07fed9ed930d54eccaead7959bd",
            "independent_b.jsonl": "511cc695d78d30db20497f80a03c93cb54e58ca2dbfd7b6e9630e4dd1453164f",
            "export_receipt.json": "14f571975888d78e0e0a05960b63b12e73e7ccb48673ff315cc54abc8542e3d0",
            "original_record_sources.json": "429b7ce049841bb1dc627b865ffbb0c55b8590b555830aa2fa0fc035d460cfd1",
            "grouping_metadata.json": "f60752e8538d1e17f1e47bc591526b119edc5a31be9dc48c3d9e206289219660"})}
SNAPSHOTS = {"codex_runtime.py": "60b47a86a9714fc6f7f2e67bdacc64af2017cd9bc13c681634be5e9af52fd97f",
    "machine_evaluation.py": "97da16165468343295d77e456c16422a4f873f48ec965dbefffdc25e5f20caa0",
    "run_codex_annotation.py": "6dadf606ae11ad80714914d98a4edc7e82c75d66b62cce498fb968f8cc941e52"}


def isolated_file(path):
    name = "oil_evidence.completed_503_proof_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required by frozen relative imports/dataclass metadata.
    spec.loader.exec_module(module)
    return module


def strict_json(raw):
    return json.loads(raw, object_pairs_hook=protocol._unique_object, parse_constant=protocol._invalid_constant)


def event_proof(receipt, raw, events_raw):
    """Generic event proof, not admission or an operational error filter."""
    events = protocol.parse_jsonl(events_raw, "completed transient-503 events")
    if receipt.get("accepted") is not False or receipt.get("process_exit_code") != 0 or receipt.get("usage") is None:
        raise ValueError("A preserved false receipt with successful measured completion is required")
    if len(events) < 5 or events[0].get("type") != "thread.started" or events[1].get("type") != "turn.started":
        raise ValueError("Exactly one initial thread/turn pair is required")
    if events[-1].get("type") != "turn.completed" or events[-2].get("type") != "item.completed":
        raise ValueError("The sole final agent output and completed turn must be terminal")
    final = events[-2].get("item", {})
    if final.get("type") != "agent_message" or not isinstance(final.get("text"), str) or final["text"].encode("utf-8") != raw:
        raise ValueError("Retained UTF-8 raw must exactly equal the final agent output")
    if events[0].get("thread_id") != receipt.get("execution_id") or not isinstance(receipt.get("execution_id"), str) or not receipt["execution_id"]:
        raise ValueError("The original native thread identity differs")
    if events[-1].get("usage") != receipt["usage"] or any(type(value) is not int or value < 0 for value in receipt["usage"].values()):
        raise ValueError("Measured usage must match the original terminal native event")
    attempts = []
    for event in events[2:-2]:
        match = WARNING.fullmatch(event.get("message", "")) if event.get("type") == "error" else None
        if match is None or event.get("item") is not None:
            raise ValueError("Only the exact pre-final HTTP503 reconnect warnings are allowed")
        attempts.append(int(match.group(1)))
    if not attempts or attempts != sorted(set(attempts)):
        raise ValueError("Reconnect warnings must be nonempty and strictly monotonic")
    if receipt.get("errors") != ["CLI emitted a failed/error event"] * len(attempts):
        raise ValueError("Mixed, semantic, quota or additional operational errors are not covered")
    parsed = native.parse_execution_events(events_raw)
    if parsed["errors"] != receipt["errors"] or parsed["thread_id"] != receipt["execution_id"] or parsed["usage"] != receipt["usage"]:
        raise ValueError("Native parser evidence differs from the preserved receipt")
    return dict(execution_id=receipt["execution_id"], native_usage=receipt["usage"], observed_model=parsed["observed_model"],
        reconnect_attempts=attempts, exact_final_raw_bytes=True, terminal_completed_turn=True)


def common_proof(mode, receipt_path):
    pin = KNOWN.get(mode)
    path = Path(receipt_path).resolve()
    if not pin or protocol.sha256(path.read_bytes()) != pin["receipt_sha256"]:
        raise ValueError("Only one of the two explicitly registered receipts may be certified")
    for module, expected in ((protocol, SNAPSHOTS["machine_evaluation.py"]), (native, SNAPSHOTS["run_codex_annotation.py"])):
        if protocol.sha256(Path(module.__file__).read_bytes()) != expected:
            raise ValueError("Actual current native/protocol validator implementation changed")
    receipt = strict_json(path.read_bytes())
    directory = path.parent
    for name, digest in receipt["artifacts_sha256"].items():
        if protocol.sha256(native.contained_file(directory, name).read_bytes()) != digest:
            raise ValueError("Original native artifact changed")
    pinned = [("stdin.txt", "stdin_sha256"), ("schema.json", "schema_sha256"), ("raw_final.json", "raw_sha256")]
    if mode == "annotation_b38":
        pinned.append(("command.json", "command_sha256"))
    for name, key in pinned:
        if protocol.sha256((directory / name).read_bytes()) != pin[key]:
            raise ValueError("Pinned native input/schema/raw/isolation bytes changed")
    binary = Path(strict_json((directory / "command.json").read_bytes())[0]) if mode == "annotation_b38" else native.discover_native_cli(native.DEFAULT_CLI)
    if protocol.sha256(binary.read_bytes()) != pin["native_binary_sha256"]:
        raise ValueError("Pinned native binary changed")
    raw = (directory / "raw_final.json").read_bytes()
    proof = event_proof(receipt, raw, (directory / "events.jsonl").read_bytes())
    parsed = strict_json(raw)
    jsonschema.Draft202012Validator(strict_json((directory / "schema.json").read_bytes())).validate(parsed)
    return path, receipt, raw, parsed, proof


def annotation_proof(path, receipt, raw, parsed, pin):
    import run_body_quote_cohort as runner
    if protocol.sha256(Path(runner.body.codec.__file__).read_bytes()) != "5b08efcd5b893def460faadf5fd90615098a77c42e91d27bdab072f8a4ca04e6":
        raise ValueError("Actual current annotation wire decoder changed")
    output = Path(receipt["annotation_run"]).resolve()
    if path != output / "b/batch-0038/attempt-2fdacb027d53/receipt.json":
        raise ValueError("The certificate must reference the exact original B38 receipt owner")
    manifest, history, protected = runner.verify_prepared(output)
    if protocol.sha256((output / "manifest.json").read_bytes()) != pin["manifest_sha256"]:
        raise ValueError("Pinned original annotation source manifest changed")
    packets = native.read_json(output / "packets.json")
    rows = [packets[rid]["payload"] for rid in receipt["record_ids"]]
    prepared = runner.body.correction.prepare_source_batch(rows)
    expected = {"stdin.txt": prepared["stdin"], "schema.json": protocol.canonical_bytes(prepared["schema"]),
        "codec.json": protocol.canonical_bytes(prepared["codec"]), "wire_payload.json": protocol.canonical_bytes(prepared["wire_payload"]),
        "canonical_semantic_packets.json": protocol.canonical_bytes(prepared["canonical_semantic_packets"]),
        "prompt.txt": prepared["prompt"].encode(), "format_instruction.txt": runner.body.correction.CORRECTION_INSTRUCTION.encode()}
    if any((path.parent / name).read_bytes() != value for name, value in expected.items()):
        raise ValueError("Full source/schema/codec/prompt no longer reconstructs exactly")
    group = next(g for g in manifest["specification"]["groups"] if g["batch"] == receipt["batch"])
    if (receipt["batch"] != "batch-0038" or receipt["stream"] != "b" or group["record_ids"] != receipt["record_ids"]
            or len(receipt["record_ids"]) != 10 or receipt["full_payload_bytes"] != len(prepared["stdin"]) + len(protocol.canonical_bytes(prepared["schema"]))
            or receipt["full_payload_bytes"] > 262144):
        raise ValueError("Registered complete body batch identity changed")
    metadata = dict(full_stdin_sha256=prepared["actual_wire_stdin_sha256"], schema_sha256=prepared["actual_wire_schema_sha256"],
        mapping_sha256=prepared["codec"]["mapping_sha256"], actual_prompt_sha256=prepared["corrective_prompt_sha256"],
        format_instruction_sha256=runner.body.INSTRUCTION_SHA256, annotation_manifest_sha256=pin["manifest_sha256"],
        execution_protocol=runner.VERSION, annotation_role="body_only_first_attempt", original_first_attempt=True)
    if any(receipt.get(key) != value for key, value in metadata.items()):
        raise ValueError("Original receipt source/schema/protocol metadata changed")
    sent = native.read_json(path.parent / "sent_input.json")
    if (sent.get("stdin_sha256") != prepared["actual_wire_stdin_sha256"] or sent.get("stdin_bytes") != len(prepared["stdin"])
            or sent.get("schema_sha256") != prepared["actual_wire_schema_sha256"]
            or sent.get("context_protocol_sha256") != prepared["corrective_prompt_sha256"]
            or sent.get("context_payload_sha256") != protocol.sha256(protocol.canonical_bytes(prepared["wire_payload"]))):
        raise ValueError("Original sent-input execution evidence changed")
    command = native.read_json(path.parent / "command.json")
    if (runner.body.correction.normalized_command(command) != manifest["specification"]["native_command_identity"]
            or runner.body.correction.retry.requested_model_from_command(path.parent / "command.json") != manifest["specification"]["requested_model"]):
        raise ValueError("Original native model request or tool isolation changed")
    decoded = runner.body.codec.decode_batch(raw, prepared["codec"], rows)["decoded_records"]
    envelopes = native.read_json(path.parent / "envelopes.json")
    if set(envelopes) != set(decoded):
        raise ValueError("Complete retained envelope coverage differs")
    events = native.parse_execution_events((path.parent / "events.jsonl").read_bytes())
    for rid, value in envelopes.items():
        if (value["label"] != decoded[rid]["label"] or value["raw_output"].encode() != raw
                or value["context_id"] != receipt["execution_id"] or value["identity"]["execution_id"] != receipt["execution_id"]
                or value["identity"]["observed_model"] != events["observed_model"]
                or value["identity"]["requested_model"] != (manifest["specification"]["requested_model"] or "cli_default_unresolved")
                or value["execution_input_sha256"] != prepared["actual_wire_stdin_sha256"]
                or value.get("annotation_role") != "body_only_first_attempt" or value.get("original_first_attempt") is not True
                or protocol.validate_independent_output(value, packets[rid])):
            raise ValueError("Original raw envelope/source/quote/date/identity guards failed")
    native.protected_unchanged(protected)
    return envelopes


def source_proof(path, receipt, raw, parsed, pin):
    directory, bounded = path.parent, path.parent.parent.parent
    inputs_path = bounded / "inputs.json"
    source = bounded.parent.parent.parent / "chunks/chunk-0029"
    if protocol.sha256(inputs_path.read_bytes()) != pin["manifest_sha256"]:
        raise ValueError("Registered source-review grouping premise changed")
    if (Path(receipt["artifact_directory"]).resolve() != directory or receipt["source_snapshots_sha256"] != SNAPSHOTS
            or receipt["role"] != "adjudicator" or receipt["requested_model"] != "cli_default_unresolved"
            or receipt["observed_model"] is not None or receipt["native_cli_sha256"] != pin["native_binary_sha256"]
            or receipt["disabled_configured_mcp_servers"] != ["ace_studio", "node_repl", "openaiDeveloperDocs", "tinyfish"]
            or receipt["model_tools_disabled"] is not True or receipt["execpolicy_rules_not_disabled"] is not True
            or receipt["codex_cli_invocations"] != 1 or receipt["external_paid_api_calls"] != 0
            or receipt["cli_version"] != "codex-cli 0.162.0-alpha.17.2" or receipt["authentication"] != "existing_ChatGPT_subscription"):
        raise ValueError("Registered source native identity/tool-isolation metadata changed")
    for name, digest in SNAPSHOTS.items():
        if protocol.sha256((directory / "implementation" / name).read_bytes()) != digest:
            raise ValueError("Frozen native runtime/helper/validator snapshot changed")
    for name, digest in pin["source_sha256"].items():
        if protocol.sha256((source / name).read_bytes()) != digest:
            raise ValueError("Frozen full source packet or independent label bytes changed")
    compact_path = Path(__file__).with_name("compact_machine_adjudication.py")
    if protocol.sha256(compact_path.read_bytes()) != "1fbca15ec78e885c3800fb09e31a6546f2d5286d76aac3bab8a8bf16ae99b91d":
        raise ValueError("Frozen compact source decoder changed")
    compact = isolated_file(compact_path)
    if protocol.sha256(Path(compact.original.__file__).read_bytes()) != "7206d7e5954433f8d19649de4c5311e8c1539915785c9247ea1c964f7931dc9c":
        raise ValueError("Actual current source pair/label validator changed")
    runtime = isolated_file(directory / "implementation/codex_runtime.py")
    helper = isolated_file(directory / "implementation/run_codex_annotation.py")
    packets = native.read_json(source / "packets.json")
    left = compact.original.read_stream(source / "independent_a.jsonl")
    right = compact.original.read_stream(source / "independent_b.jsonl")
    ids = native.read_json(inputs_path)["groups"][1]
    payload, schema = compact.prepare_batch(packets, left, right, ids)
    stdin = (runtime.ISOLATION_PROMPT + "\n" + compact.SYSTEM + "\nUNTRUSTED_PAYLOAD_JSON\n").encode() + runtime._json_bytes(payload)
    if (len(ids) != 9 or set(parsed) != set(ids) or stdin != (directory / "stdin.txt").read_bytes()
            or protocol.canonical_bytes(schema) != (directory / "schema.json").read_bytes()
            or protocol.sha256(compact.SYSTEM.encode()) != "aa8992bf627594e32690e436b6d20a4fd8da3b1e13079a855f63515a83accf59"):
        raise ValueError("The complete source-review input/schema/protocol does not reconstruct")
    def retained(actual_receipt, actual_raw, record_ids):
        if (actual_receipt != receipt or actual_raw != parsed or record_ids != ids
                or strict_json(path.read_bytes()) != receipt or (directory / "raw_final.json").read_bytes() != raw):
            raise ValueError("Proof-aware compact selection must reference the original false receipt/raw")
        return directory / "raw_final.json", path
    compact._retained_raw = retained
    decisions = compact.retain_compact_decisions(parsed, packets, left, right, ids, receipt)
    if len(decisions) != 9 or any(value["mechanical_validation_errors"] for value in decisions):
        raise ValueError("Unchanged source/selection/label/date/quote guards failed")
    binary = helper.discover_native_cli(helper.DEFAULT_CLI)
    template = helper.cli_command(binary, Path("FRESH_CONTEXT"), Path("FRESH_CONTEXT/schema.json"),
        Path("FRESH_CONTEXT/final.json"), None, receipt["disabled_configured_mcp_servers"])
    return dict(decisions=decisions, actual_argv_recorded=False, actual_command_bytes_unavailable=True,
        reconstructed_isolation_template=template, source_decisions_pending_certificate_aware_assembly=True)


def verify(mode, receipt_path):
    path, receipt, raw, parsed, proof = common_proof(mode, receipt_path)
    records = (annotation_proof if mode == "annotation_b38" else source_proof)(path, receipt, raw, parsed, KNOWN[mode])
    return dict(mode=mode, original_receipt=str(path), original_receipt_sha256=protocol.sha256(path.read_bytes()),
        original_raw=str(path.parent / "raw_final.json"), original_raw_sha256=protocol.sha256(raw),
        operational_receipt_accepted=False, status=STATUS, new_native_calls=0, original_invocation_usage_counted_once=True,
        semantic_accuracy_claimed=False, human_gold=False, proof=proof, derived_records=records,
        actual_argv_recorded=mode == "annotation_b38", actual_command_bytes_unavailable=mode == "source_group2",
        derived_record_count=len(records) if mode == "annotation_b38" else len(records["decisions"]))


def certify(mode, receipt_path, output, authorization_path):
    path, output = Path(receipt_path).resolve(), Path(output).resolve()
    authorization_path = Path(authorization_path).resolve()
    authorization = strict_json(authorization_path.read_bytes())
    if (authorization.get("root_authorized") is not True or authorization.get("route") != VERSION
            or type(authorization.get("new_native_calls_allowed")) is not int or authorization["new_native_calls_allowed"] != 0
            or authorization.get("mode") != mode or authorization.get("receipt_sha256") != protocol.sha256(path.read_bytes())
            or authorization.get("certifier_sha256") != protocol.sha256(Path(__file__).read_bytes())):
        raise ValueError("A separately reviewed exact zero-call certification authorization is required")
    original = strict_json(path.read_bytes())
    old_root = Path(original["annotation_run"]).resolve() if mode == "annotation_b38" else path.parent.parent.parent
    if output.exists() or output.is_relative_to(old_root):
        raise ValueError("Certificate output must be fresh and outside original execution artifacts")
    result = verify(mode, path)
    before = {str(value): protocol.sha256(value.read_bytes()) for value in path.parent.rglob("*") if value.is_file()}
    result.update(version=VERSION, certifier_sha256=protocol.sha256(Path(__file__).read_bytes()),
        authorization_path=str(authorization_path), authorization_sha256=protocol.sha256(authorization_path.read_bytes()))
    native.write_new(output / "derived_records.json", dict(certificate_path=str(output / "certificate.json"),
        original_receipt=str(path), operational_receipt_accepted=False, records=result.pop("derived_records")))
    result["derived_records_sha256"] = protocol.sha256((output / "derived_records.json").read_bytes())
    native.protected_unchanged(before)
    native.write_new(output / "certificate.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(KNOWN), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(certify(args.mode, args.receipt, args.output, args.authorization), indent=2))
