"""Stage the complete paired development cohort without claiming a final freeze.

The six actual annotation inputs are staged byte-for-byte or projected through
the shared strict protocol. Adjudications and six real runtime snapshots remain
explicit prerequisites; this script never fabricates them or invokes a model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import run_codex_annotation as native
from export_codex_annotation_chunk import export


def stage(root, pool_dir, a_dir, b_dir, corpus_path, questions_path):
    root, pool_dir, a_dir, b_dir, corpus_path, questions_path = map(
        lambda path: Path(path).resolve(), (root, pool_dir, a_dir, b_dir, corpus_path, questions_path))
    _, rows, protected = native.load_pool(root, pool_dir)
    bindings = native.read_json(pool_dir / "inputs.json")["bindings"]
    for path, key in ((corpus_path, "corpus_sha256"), (questions_path, "questions_sha256")):
        if not path.is_relative_to(root) or native.protocol.sha256(path.read_bytes()) != bindings[key]:
            raise ValueError("Original corpus or question bytes do not match the retrieval bindings")
        protected[str(path)] = bindings[key]
    questions = [native.protocol.clean_question(row) for row in native.read_json(questions_path)]
    if len(questions) != 50 or any(row["split"] != "development" for row in questions):
        raise ValueError("Stage all 50 development records, never a convenient formal subset")
    pools = [dict(record_id=row["question_id"], evidence_ids=[p["passage_id"] for p in row["evidence"]])
             for row in rows]
    prompt_bytes = (a_dir / "protocol.txt").read_bytes()
    corpus_bytes = corpus_path.read_bytes()
    rebuilt = native.protocol.prepare_inputs(questions, pools, native.protocol.parse_jsonl(corpus_bytes),
                                              prompt=prompt_bytes.decode("utf-8"))
    packets = native.read_json(a_dir / "packets.json")
    projection = lambda row: {key: row[key] for key in ("payload", "prompt_sha256", "input_sha256")}
    if {key: projection(value) for key, value in rebuilt.items()} != packets:
        raise ValueError("Clean frozen question/corpus metadata does not reproduce actual annotation packets")
    output = export(pool_dir, a_dir, b_dir, start=1, count=50)
    native.write_new(output / "corpus.jsonl", corpus_bytes)
    native.write_new(output / "questions.jsonl", b"".join(native.protocol.canonical_bytes(row) + b"\n"
                                                         for row in questions))
    native.write_new(output / "evidence_pool.jsonl", b"".join(native.protocol.canonical_bytes(row) + b"\n"
                                                             for row in pools))
    native.write_new(output / "annotation_prompt.txt", prompt_bytes)
    left = native.protocol.parse_jsonl((output / "independent_a.jsonl").read_bytes())
    right = native.protocol.parse_jsonl((output / "independent_b.jsonl").read_bytes())
    agreement = native.protocol.raw_agreement(rebuilt, left, right)
    if agreement["paired_records"] != 50 or agreement["pooled_passages_without_paired_judgment"]:
        raise ValueError("Full paired development coverage is incomplete; output is retained uncertified")
    native.write_new(output / "raw_machine_agreement.json", agreement)
    by_right = {row["record_id"]: row for row in right}
    differences = []
    for a in left:
        b = by_right[a["record_id"]]
        fields = {key: dict(a=a["label"][key], b=b["label"][key])
                  for key in ("answerability", "event_status") if a["label"][key] != b["label"][key]}
        qrels = [dict(passage_id=pid, a=a["label"]["qrels"][pid], b=b["label"]["qrels"][pid])
                 for pid in sorted(a["label"]["qrels"]) if a["label"]["qrels"][pid] != b["label"]["qrels"][pid]]
        if fields or qrels:
            differences.append(dict(record_id=a["record_id"], categorical_fields=fields, qrels=qrels))
    native.write_new(output / "raw_machine_disagreements.json", dict(
        scope="raw_machine_categorical_disagreement_before_adjudication", records=differences,
        differing_records=len(differences), human_disagreement=None, semantic_answer_comparison=None,
        note="Different free-text answers are not automatically treated as semantic disagreements."))
    native.protected_unchanged(protected)
    names = ("corpus.jsonl", "questions.jsonl", "evidence_pool.jsonl", "annotation_prompt.txt",
             "independent_a.jsonl", "independent_b.jsonl", "packets.json", "source_receipts.json",
             "raw_machine_agreement.json", "raw_machine_disagreements.json")
    native.write_new(output / "staging_receipt.json", dict(
        status="complete_development_annotations_staged_not_frozen", records=50, split="development",
        corpus_sha256=bindings["corpus_sha256"], original_questions_sha256=bindings["questions_sha256"],
        annotation_protocol_sha256=native.protocol.sha256(prompt_bytes), original_inputs_unchanged=True,
        new_model_calls=0, human_gold=False, formal_test_ready=False, dataset_frozen=False,
        missing_final_prerequisites=["adjudications.jsonl", "six_actual_runtime_snapshots", "runtime_artifacts.json"],
        artifacts_sha256={name: native.protocol.sha256((output / name).read_bytes()) for name in names}))
    return output, agreement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("canonical-root", "pool-dir", "a-dir", "b-dir", "corpus-path", "questions-path"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    output, _ = stage(args.canonical_root, args.pool_dir, args.a_dir, args.b_dir, args.corpus_path, args.questions_path)
    print(output)
