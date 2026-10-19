"""Allowlisted, zero-call revalidation of an authorization-format software alarm.

Native receipts remain accepted=false. This distinct certificate validates only
the reviewed pair's unchanged raw/source outputs; it does not alter any model
label, failed event, authorization file, usage or historical adapter byte.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import correct_codex_wire_batch as correction
from oil_evidence import machine_evaluation as protocol

native = correction.native
VERSION = "metadata-revalidated-source-only-corrective-pair-v1"
FALSE_ALARM = "Reviewed corrective authorization changed during execution"
EXPECTED_MANIFEST = "5659400826bc2a60c3cc28fd4953c939e7c0c788d62ce7b6524ed349b669a54f"
EXPECTED_AUTHORIZATION = "fbae3225ffbefe4442c7a5f732b3f209029ae33d6217be2f1e330621655687c2"
EXPECTED_ADAPTER = "8f4bb16bf70aa277d2ac14d5e0def7d34da28ba1d46e1cae3c92e85e0d14bdac"
EXPECTED_FAILED_CERTIFICATE = "7ec8eab367d37b00f70a94d41b1a8dd78d7593ae66d2556d56e9be3dc94dee89"
EXPECTED_RECEIPTS = {
    "a": "ef15b33958ead97796442edb15a52c15a484cf4056d897e4804c8714d443669c",
    "b": "bebd7ff98b1054883cace77593d2fde007ab2305480e2cdb59db5ab77b8cf216",
}
PHYSICAL_INPUTS = ("stdin.txt", "schema.json", "codec.json", "wire_payload.json",
                   "canonical_semantic_packets.json", "prompt.txt", "instruction.txt")


def strict_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=protocol._unique_object,
                      parse_constant=protocol._invalid_constant)


def authorization_equivalence(original, copied, expected_original_sha256):
    original, copied = Path(original), Path(copied)
    if protocol.sha256(original.read_bytes()) != expected_original_sha256:
        raise ValueError("Reviewed original authorization bytes changed")
    if strict_json(original) != strict_json(copied):
        raise ValueError("Original and copied authorization JSON values differ")
    if copied.read_bytes() != protocol.canonical_bytes(strict_json(original)):
        raise ValueError("Copied authorization is not exactly the recorded canonical serialization")
    return dict(original_byte_sha256=expected_original_sha256,
        copied_byte_sha256=protocol.sha256(copied.read_bytes()),
        identical_json_values=True, copied_is_exact_canonical_json=True,
        byte_difference_explained_only_by_json_serialization=original.read_bytes() != copied.read_bytes())


def native_destinations(command):
    values = {}
    for flag in ("--cd", "--output-schema", "--output-last-message"):
        positions = [index for index, item in enumerate(command) if item == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(command):
            raise ValueError("One exact fresh native context/schema/output destination is required")
        values[flag] = Path(command[positions[0] + 1]).resolve()
    context = values["--cd"]
    if values["--output-schema"] != context / "schema.json" or values["--output-last-message"] != context / "final.json":
        raise ValueError("Native schema/output destinations do not belong to their exact fresh context")
    return values


def verify(directory):
    directory = Path(directory).resolve()
    if protocol.sha256((directory / "manifest.json").read_bytes()) != EXPECTED_MANIFEST:
        raise ValueError("Only the reviewed corrective manifest is eligible")
    if protocol.sha256(Path(correction.__file__).read_bytes()) != EXPECTED_ADAPTER:
        raise ValueError("Frozen corrective adapter was changed")
    if protocol.sha256((directory / "certificate.json").read_bytes()) != EXPECTED_FAILED_CERTIFICATE:
        raise ValueError("Original failed-stop corrective certificate was changed")
    manifest, source, prepared = correction.verify_prepared(directory)
    original_certificate = strict_json(directory / "certificate.json")
    authorization = directory / "root_authorization.json"
    correction.validate_authorization(directory, authorization)
    if protocol.sha256(authorization.read_bytes()) != EXPECTED_AUTHORIZATION:
        raise ValueError("Actual reviewed root authorization differs")
    metadata_authorization_path = directory / "root_metadata_authorization.json"
    metadata_authorization = strict_json(metadata_authorization_path)
    if (metadata_authorization.get("root_authorized") is not True or metadata_authorization.get("route") != VERSION
            or metadata_authorization.get("new_native_calls_allowed") != 0
            or type(metadata_authorization.get("new_native_calls_allowed")) is not int
            or metadata_authorization.get("original_authorization_sha256") != EXPECTED_AUTHORIZATION
            or metadata_authorization.get("reviewed_receipt_sha256") != EXPECTED_RECEIPTS
            or metadata_authorization.get("metadata_verifier_sha256") != protocol.sha256(Path(__file__).read_bytes())):
        raise ValueError("Separately reviewed zero-call metadata authorization is required")
    protected = dict(native.read_json(directory / "protected_inputs.json"))
    protected.update({str(path): protocol.sha256(path.read_bytes()) for path in directory.rglob("*")
                      if path.is_file() and path.name != "metadata_certificate.json"})
    protected[str(Path(__file__).resolve())] = protocol.sha256(Path(__file__).read_bytes())
    records, receipts, proofs, contexts = {}, {}, {}, {}
    old_ids = {source["counterpart_a"]["execution_id"], *(v["execution_id"] for v in source["rejected_receipts"])}
    original_contexts = {native_destinations(strict_json(Path(value["path"]).parent / "command.json"))["--cd"]
        for value in [source["counterpart_a"], *source["rejected_receipts"]]}
    for stream in native.STREAMS:
        attempt = directory / stream / "execution"
        path = attempt / "receipt.json"
        if protocol.sha256(path.read_bytes()) != EXPECTED_RECEIPTS[stream]:
            raise ValueError("Only the exact reviewed false-alarm receipts are eligible")
        receipt = strict_json(path)
        if (receipt.get("accepted") is not False or receipt.get("errors") != [FALSE_ALARM]
                or receipt.get("process_exit_code") != 0 or receipt.get("stream") != stream
                or receipt.get("record_ids") != source["record_ids"] or len(receipt["record_ids"]) != 10
                or receipt.get("execution_protocol") != correction.VERSION
                or receipt.get("execution_role") != "source_only_corrective_pair"
                or receipt.get("original_blinded_annotation") is not False
                or receipt.get("manifest_sha256") != EXPECTED_MANIFEST
                or receipt.get("authorization_sha256") != EXPECTED_AUTHORIZATION):
            raise ValueError("Mixed/model/native/quota errors cannot use metadata revalidation")
        correction.retry.verify_raw_artifacts(path, receipt)
        proof = authorization_equivalence(authorization, attempt / "authorization.json", EXPECTED_AUTHORIZATION)
        correction.validate_authorization(directory, attempt / "authorization.json")
        for name in PHYSICAL_INPUTS:
            if (attempt / name).read_bytes() != (directory / name).read_bytes():
                raise ValueError("Actual source-only corrective physical input changed")
        expected_metadata = dict(full_stdin_sha256=prepared["actual_wire_stdin_sha256"],
            schema_sha256=prepared["actual_wire_schema_sha256"], mapping_sha256=prepared["codec"]["mapping_sha256"],
            corrective_prompt_sha256=prepared["corrective_prompt_sha256"],
            corrective_instruction_sha256=prepared["corrective_instruction_sha256"])
        if any(receipt.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("Corrective receipt source/schema/map/prompt metadata differs")
        sent = strict_json(attempt / "sent_input.json")
        if (sent.get("stdin_sha256") != expected_metadata["full_stdin_sha256"]
                or sent.get("stdin_bytes") != len(prepared["stdin"])
                or sent.get("schema_sha256") != expected_metadata["schema_sha256"]
                or sent.get("context_protocol_sha256") != expected_metadata["corrective_prompt_sha256"]
                or sent.get("context_payload_sha256") != protocol.sha256(protocol.canonical_bytes(prepared["wire_payload"]))):
            raise ValueError("Actual native sent-input evidence differs")
        command = strict_json(attempt / "command.json")
        original_command = strict_json(Path(source["counterpart_a"]["path"]).parent / "command.json")
        if correction.normalized_command(command) != correction.normalized_command(original_command):
            raise ValueError("Native isolation command changed")
        contexts[stream] = native_destinations(command)["--cd"]
        if contexts[stream] in original_contexts:
            raise ValueError("Corrective native context reused an original source context")
        if correction.retry.requested_model_from_command(attempt / "command.json") != source["requested_model"]:
            raise ValueError("Actual native requested model changed")
        events = native.parse_execution_events((attempt / "events.jsonl").read_bytes())
        if (events["errors"] or not events["thread_id"] or events["thread_id"] != receipt["execution_id"]
                or events["usage"] != receipt["usage"] or receipt["execution_id"] in old_ids):
            raise ValueError("Native execution/model/usage evidence is invalid")
        raw = (attempt / "raw_final.json").read_bytes()
        decoded = correction.codec.decode_batch(raw, prepared["codec"], source["rows"])["decoded_records"]
        envelopes = strict_json(attempt / "envelopes.json")
        if set(envelopes) != set(source["record_ids"]):
            raise ValueError("Whole corrective ten-record coverage differs")
        for identifier, value in envelopes.items():
            if (value["raw_output"].encode() != raw or value["label"] != decoded[identifier]["label"]
                    or value["context_id"] != receipt["execution_id"] or value["identity"]["execution_id"] != receipt["execution_id"]
                    or value["identity"]["observed_model"] != events["observed_model"]
                    or value["identity"]["requested_model"] != (source["requested_model"] or "cli_default_unresolved")
                    or value["execution_input_sha256"] != prepared["actual_wire_stdin_sha256"]
                    or value.get("annotation_role") != "source_only_corrective_pair"
                    or value.get("corrective_stream") != stream or value.get("original_blinded_annotation") is not False
                    or value.get("excluded_from_original_agreement") is not True
                    or value["decoding"].get("actual_corrective_prompt_sha256") != prepared["corrective_prompt_sha256"]
                    or value["decoding"].get("corrective_instruction_sha256") != prepared["corrective_instruction_sha256"]
                    or protocol.validate_independent_output(value, prepared["canonical_semantic_packets"][identifier])):
                raise ValueError("Raw corrective envelope/source/label/date/quote/model binding differs")
        records[stream], receipts[stream] = envelopes, receipt
        proofs[stream] = dict(receipt=str(path), receipt_sha256=EXPECTED_RECEIPTS[stream],
            native_receipt_accepted=False, execution_id=receipt["execution_id"], authorization_format_proof=proof,
            raw_sha256=protocol.sha256(raw), native_event_usage=events["usage"], observed_model=events["observed_model"])
    if receipts["a"]["execution_id"] == receipts["b"]["execution_id"]:
        raise ValueError("Corrective contexts must be two distinct executions")
    if contexts["a"] == contexts["b"]:
        raise ValueError("Corrective physical context directories must be distinct")
    if any((directory / "a/execution" / name).read_bytes() != (directory / "b/execution" / name).read_bytes() for name in PHYSICAL_INPUTS):
        raise ValueError("Corrective A/B full physical inputs must be identical")
    agreement = protocol.raw_agreement(prepared["canonical_semantic_packets"], list(records["a"].values()), list(records["b"].values()))
    if agreement["paired_records"] != 10:
        raise ValueError("Unchanged core equal-input pair guard did not pass")
    agreement.update(agreement_population="metadata_revalidated_new_corrective_pair_only", original_first_attempt_agreement=False)
    usage_rows = protocol.parse_jsonl((directory / "execution_usage.jsonl").read_bytes(), "exact two corrective execution usages")
    if (len(usage_rows) != 2 or {row.get("stream") for row in usage_rows} != set(native.STREAMS)
            or len({row.get("execution_id") for row in usage_rows}) != 2
            or any(row.get("execution_id") != receipts[row["stream"]]["execution_id"]
                   or row.get("usage") != receipts[row["stream"]]["usage"] for row in usage_rows)):
        raise ValueError("Corrective usage ledger must preserve exactly two actual distinct native executions")
    report = dict(version=VERSION, validation_status="source_valid_pending_adjudication", software_false_alarm=True,
        scope="Only the two exact authorization-serialization false alarms; no native/model failures admitted",
        correction_directory=str(directory), manifest_sha256=EXPECTED_MANIFEST,
        original_failed_certificate_sha256=EXPECTED_FAILED_CERTIFICATE, original_authorization_sha256=EXPECTED_AUTHORIZATION,
        corrective_streams=proofs, record_ids=source["record_ids"], rejected_original_receipts=source["rejected_receipts"],
        accepted_original_a_preserved=source["counterpart_a"], native_receipts_remain_accepted_false=True,
        raw_or_label_changes=0, new_native_calls=0, native_calls_in_existing_corrective_pair=2,
        original_first_attempt_agreement_substitution_allowed=False, original_rejected_event_count=2,
        corrected_labels_still_require_source_adjudication=True, corrective_pair_agreement=agreement,
        full_stdin_sha256=prepared["actual_wire_stdin_sha256"], schema_sha256=prepared["actual_wire_schema_sha256"],
        mapping_sha256=prepared["codec"]["mapping_sha256"], metadata_verifier_sha256=protocol.sha256(Path(__file__).read_bytes()),
        metadata_authorization_path=str(metadata_authorization_path),
        metadata_authorization_sha256=protocol.sha256(metadata_authorization_path.read_bytes()),
        protected_sha256=protected, human_gold=False)
    native.protected_unchanged(protected)
    return report, records


def seal(directory):
    directory = Path(directory).resolve()
    report, _ = verify(directory)
    certificate = dict(created_utc=native.utc_now(), verification=report)
    native.write_new(directory / "metadata_certificate.json", certificate)
    return certificate


def completed_correction_records(directory, stream):
    directory = Path(directory).resolve()
    saved = strict_json(directory / "metadata_certificate.json")
    report, records = verify(directory)
    if protocol.canonical_bytes(saved.get("verification")) != protocol.canonical_bytes(report):
        raise ValueError("Metadata revalidation certificate differs from exact retained execution evidence")
    return records[stream]


def correction_record_sources(directory, stream):
    directory = Path(directory).resolve()
    records = completed_correction_records(directory, stream)
    certificate = directory / "metadata_certificate.json"
    receipt = directory / stream / "execution/receipt.json"
    value = strict_json(receipt)
    return {identifier: dict(receipt=str(receipt), receipt_sha256=EXPECTED_RECEIPTS[stream],
        execution_id=value["execution_id"], stdin_sha256=value["full_stdin_sha256"], source_run=str(directory),
        migrated_unchanged=False, raw_encoding=correction.codec.VERSION, execution_protocol=correction.VERSION,
        validation_protocol=VERSION, validation_status="source_valid_pending_adjudication",
        source_validation_status="source_valid_pending_adjudication",
        validation_certificate=str(certificate), validation_certificate_sha256=protocol.sha256(certificate.read_bytes()),
        native_receipt_accepted=False, software_false_alarm=True) for identifier in records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "seal"))
    parser.add_argument("--correction-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        value = seal(args.correction_directory)
    else:
        value, _ = verify(args.correction_directory)
    report = value.get("verification", value)
    print(json.dumps(dict(validation_status=report["validation_status"], software_false_alarm=True,
        correction_directory=report["correction_directory"], record_count=len(report["record_ids"]),
        new_native_calls=0, native_receipts_remain_accepted_false=True,
        certificate_sha256=protocol.sha256((args.correction_directory / "metadata_certificate.json").read_bytes())
            if args.command == "seal" else None), indent=2))
