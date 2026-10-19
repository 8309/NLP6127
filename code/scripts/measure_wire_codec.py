"""Estimate unused structural-ID codec representation savings, without a model call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import run_codex_annotation as native
import run_codex_batch_annotations as batch
from oil_evidence import machine_evaluation as protocol
from oil_evidence import wire_codec as codec


def measure(annotation_run):
    run = Path(annotation_run).resolve()
    packets = native.read_json(run / "packets.json")
    originals = wires = original_inputs = wire_inputs = outputs = actual_tokens = 0
    receipts = []
    for path in sorted(run.glob("*/batch-*/attempt-*/receipt.json")):
        receipt = native.read_json(path)
        if receipt.get("accepted") is not True:
            continue
        for name, expected in receipt["artifacts_sha256"].items():
            if protocol.sha256(native.contained_file(path.parent, name).read_bytes()) != expected:
                raise ValueError("Original model artifact changed before codec measurement")
        rows = [packets[identifier]["payload"] for identifier in receipt["record_ids"]]
        raw = (path.parent / "raw_final.json").read_bytes()
        original = json.loads(raw, object_pairs_hook=protocol._unique_object, parse_constant=protocol._invalid_constant)
        prepared = codec.prepare_batch(rows, batch.BATCH_PROMPT, native.output_schema)
        simulation = codec.simulate_encoding_for_measurement(original, prepared["codec"], rows)
        wire_raw = protocol.canonical_bytes(simulation)
        if codec.decode_batch(wire_raw, prepared["codec"], rows)["decoded_records"] != original:
            raise ValueError("Representation-only codec did not preserve actual original labels")
        originals += len(protocol.canonical_bytes(original))
        wires += len(wire_raw)
        original_inputs += len((path.parent / "stdin.txt").read_bytes()) + len((path.parent / "schema.json").read_bytes())
        wire_inputs += len(prepared["stdin"]) + len(protocol.canonical_bytes(prepared["schema"]))
        outputs += len(receipt["record_ids"])
        actual_tokens += receipt["usage"]["output_tokens"]
        receipts.append(dict(receipt=str(path), receipt_sha256=protocol.sha256(path.read_bytes())))
    if not receipts:
        raise ValueError("No actual accepted native receipts available to measure")
    result = dict(kind="pure_unused_representation_estimate_not_model_outputs_or_gold", created_utc=native.utc_now(),
        codec_version=codec.VERSION, native_receipts=receipts, actual_accepted_calls=len(receipts),
        actual_record_outputs=outputs, actual_native_output_tokens=actual_tokens,
        normalized_original_output_bytes=originals, simulated_alias_output_bytes=wires,
        normalized_output_byte_reduction_fraction=1 - wires / originals,
        original_full_stdin_schema_bytes=original_inputs, prototype_full_wire_stdin_schema_bytes=wire_inputs,
        full_input_byte_reduction_fraction=1 - wire_inputs / original_inputs,
        roundtrip_label_equality=True, exact_model_token_savings=None, native_codec_calls=0,
        raw_model_files_modified=False, human_labels=0,
        limitations=["Canonical serialization used on both sides to exclude formatting differences.",
                     "No tokenizer installation or exact native token/latency savings claim.",
                     "Aliased representations are simulated measurement data, never raw model outputs or new labels."])
    output = run / ("unused_wire_codec_measurement-" + uuid4().hex[:10] + ".json")
    native.write_new(output, result)
    return output, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-run", type=Path, required=True)
    args = parser.parse_args()
    output, result = measure(args.annotation_run)
    print(json.dumps({key: value for key, value in result.items() if key != "native_receipts"}, indent=2))
    print(str(output))
