"""Resume one existing v2 stream without editing its original source bindings.

Completed hash-bound outputs are skipped by the unchanged native runner. Old
failed attempts remain; a new append-only resume trace binds this launcher and
the original wrapper hash. No credits, account settings or rules are modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import run_codex_annotation as native
from run_codex_annotation_stream import validated_cli


def prepare(root, pool_dir, output, stream):
    root, pool_dir, output = map(lambda path: Path(path).resolve(), (root, pool_dir, output))
    if stream not in {"a", "b"} or not output.is_relative_to((pool_dir / "codex_annotations").resolve()):
        raise ValueError("Select one stream inside its existing annotation pool")
    manifest = native.read_json(output / "manifest.json")
    if manifest["specification"]["streams"] != [stream]:
        raise ValueError("Existing output is not the selected single-stream run")
    for name, expected in manifest["immutable_sha256"].items():
        if native.protocol.sha256(native.contained_file(output, name).read_bytes()) != expected:
            raise ValueError("Existing immutable resume input changed")
    protected = native.read_json(output / "protected_inputs.json")
    native.protected_unchanged(protected)
    wrapper = Path(__file__).with_name("run_codex_annotation_stream.py").resolve()
    wrapper_receipt = native.read_json(output / "stream_wrapper_receipt.json")
    if (wrapper_receipt["stream"] != stream
            or wrapper_receipt["wrapper_sha256"] != native.protocol.sha256(wrapper.read_bytes())
            or protected.get(str(wrapper)) != wrapper_receipt["wrapper_sha256"]):
        raise ValueError("Original wrapper source binding changed")
    packets = native.read_json(output / "packets.json")
    completed = native.completed_outputs(output, stream, packets)
    return root, pool_dir, output, manifest, protected, wrapper_receipt, completed


def resume(root, pool_dir, output, stream, max_records=50, cli=native.DEFAULT_CLI):
    root, pool_dir, output, manifest, protected, wrapper_receipt, completed = prepare(root, pool_dir, output, stream)
    executable = validated_cli(cli)
    launcher = Path(__file__).resolve()
    launcher_sha = native.protocol.sha256(launcher.read_bytes())
    trace = output / "resume_traces" / ("resume-" + uuid4().hex[:12] + ".json")
    native.write_new(trace, dict(created_utc=native.utc_now(), stream=stream, existing_records=len(completed),
        skipped_record_ids=sorted(completed), requested_total_records=max_records,
        original_manifest_sha256=native.protocol.sha256((output / "manifest.json").read_bytes()),
        original_wrapper_sha256=wrapper_receipt["wrapper_sha256"], resume_launcher_sha256=launcher_sha,
        unchanged_native_runner=True, account_or_credit_mutations=0, original_attempts_preserved=True,
        unknown_usage_not_replaced_by_zero=True, human_gold=False))
    previous_streams, previous_loader = native.STREAMS, native.load_pool

    def guarded_loader(selected_root, selected_pool):
        directory, rows, current = previous_loader(selected_root, selected_pool)
        expected = {**current, **protected, str(launcher): launcher_sha,
                    str(trace): native.protocol.sha256(trace.read_bytes())}
        native.protected_unchanged(expected)
        return directory, rows, expected

    native.STREAMS = (stream,)
    native.load_pool = guarded_loader
    try:
        return native.run(root, pool_dir, max_records=max_records,
                          requested_model=manifest["specification"]["requested_model"],
                          cli=str(executable), resume=output)
    finally:
        native.STREAMS, native.load_pool = previous_streams, previous_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream", choices=["a", "b"], required=True)
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--codex-cli", default=native.DEFAULT_CLI)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        *_, records = prepare(args.canonical_root, args.pool_dir, args.output, args.stream)
        print(json.dumps(dict(stream=args.stream, accepted=len(records), remaining=50 - len(records), model_calls=0), indent=2))
    else:
        _, summary = resume(args.canonical_root, args.pool_dir, args.output, args.stream,
                            args.max_records, args.codex_cli)
        print(json.dumps(summary, indent=2))
        raise SystemExit(1 if summary["failures"] else 0)
