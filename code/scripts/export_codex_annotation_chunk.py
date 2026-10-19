"""Export a declared development chunk of paired raw envelopes for adjudication.

No dataset freeze, score, model call or source mutation occurs. Running annotation
streams keep separate progress writers; only completed, hash-bound outputs enter
this immutable chunk. Expected full coverage remains 50 development questions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

import run_codex_annotation as native


def export(pool_dir, a_dir, b_dir, start=1, count=10):
    pool_dir, a_dir, b_dir = map(lambda path: Path(path).resolve(), (pool_dir, a_dir, b_dir))
    if type(start) is not int or type(count) is not int or start < 1 or count < 1:
        raise ValueError("Select a positive explicit development chunk")
    if any(not path.is_relative_to((pool_dir / "codex_annotations").resolve()) for path in (a_dir, b_dir)):
        raise ValueError("Pair source runs must be inside their annotation pool")
    packets = native.read_json(a_dir / "packets.json")
    if packets != native.read_json(b_dir / "packets.json") or len(packets) != 50:
        raise ValueError("Paired development packets differ or full expected coverage is not 50")
    if (a_dir / "protocol.txt").read_bytes() != (b_dir / "protocol.txt").read_bytes():
        raise ValueError("Paired protocols differ")
    ids = sorted(packets)[start - 1:start - 1 + count]
    if len(ids) != count:
        raise ValueError("Development chunk exceeds the fixed cohort")
    streams = {"a": native.completed_outputs(a_dir, "a", packets),
               "b": native.completed_outputs(b_dir, "b", packets)}
    selected = {side: [rows[qid] for qid in ids] for side, rows in streams.items()}
    for left, right in zip(selected["a"], selected["b"], strict=True):
        if (left["execution_input_sha256"] != right["execution_input_sha256"]
                or left["context_id"] == right["context_id"]):
            raise ValueError("Paired actual model inputs differ or contexts are not independent")
    output = pool_dir / "adjudication_inputs" / (f"part-{start:03d}-{start + count - 1:03d}-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    native.write_new(output / "packets.json", {qid: packets[qid] for qid in ids})
    bindings = {}
    for side, directory in (("a", a_dir), ("b", b_dir)):
        native.write_new(output / f"independent_{side}.jsonl", b"".join(native.protocol.canonical_bytes(row) + b"\n" for row in selected[side]))
        bindings[side] = {}
        for qid in ids:
            paths = list((directory / side / qid).glob("attempt-*/receipt.json"))
            accepted = [path for path in paths if native.read_json(path).get("accepted") is True]
            if len(accepted) != 1:
                raise ValueError("Ambiguous or missing source receipt")
            path = accepted[0]
            bindings[side][qid] = dict(source_receipt=str(path), receipt_sha256=native.protocol.sha256(path.read_bytes()),
                envelope_sha256=native.protocol.sha256((path.parent / "envelope.json").read_bytes()),
                raw_output_sha256=streams[side][qid]["raw_output_sha256"],
                actual_stdin_sha256=streams[side][qid]["execution_input_sha256"],
                original_execution_id=streams[side][qid]["identity"]["execution_id"])
    native.write_new(output / "source_receipts.json", dict(schema_version=1, scope="explicit_development_adjudication_chunk",
        expected_full_development_records=50, exported_record_ids=ids, records=count, not_a_formal_subset_score=True,
        formal_test_set=False, dataset_frozen=False, new_model_calls=0, human_gold=False,
        source_bindings=bindings, annotation_protocol_sha256=native.protocol.sha256((a_dir / "protocol.txt").read_bytes()),
        exported_artifacts_sha256={name: native.protocol.sha256((output / name).read_bytes()) for name in
            ("packets.json", "independent_a.jsonl", "independent_b.jsonl")}))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    print(export(args.pool_dir, args.a_dir, args.b_dir, args.start, args.count))
