"""Deduplicate observed native usage by execution, retaining unknown attempts.

Scope is annotation and the separate batch pilot inside one retrieval pool;
adjudication, question generation and answer-generation calls are excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_codex_annotation as native


def audit(pool_dir):
    pool_dir = Path(pool_dir).resolve()
    executions = {}
    source_hashes = {}
    for subtree in ("codex_annotations", "codex_batch_annotations"):
        for path in sorted((pool_dir / subtree).rglob("events.jsonl")):
            if not path.resolve().is_relative_to(pool_dir):
                raise ValueError("Native usage event path escapes its pool")
            raw = path.read_bytes()
            source_hashes[str(path)] = native.protocol.sha256(raw)
            events = [json.loads(line) for line in raw.splitlines() if line.strip()]
            ids = {event["thread_id"] for event in events if event.get("type") == "thread.started"}
            if len(ids) > 1:
                raise ValueError("One native event stream contains multiple execution IDs")
            execution_id = next(iter(ids), "unidentified-attempt:" + str(path.parent))
            usages = [event["usage"] for event in events
                      if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict)]
            if any(usage != usages[0] for usage in usages[1:]):
                raise ValueError("Conflicting repeated completion usage")
            usage = usages[0] if usages else None
            existing = executions.get(execution_id)
            if existing and existing["usage"] != usage:
                raise ValueError("Conflicting usage across duplicate execution artifacts")
            item = executions.setdefault(execution_id, dict(execution_id=execution_id, usage=usage, event_paths=[]))
            item["event_paths"].append(str(path))
    native.protected_unchanged(source_hashes)
    known = [row["usage"] for row in executions.values() if row["usage"] is not None]
    keys = sorted(set().union(*(row.keys() for row in known))) if known else []
    if any(type(value) is not int or value < 0 for row in known for value in row.values()):
        raise ValueError("Native usage must contain nonnegative observed integer counts")
    return dict(scope="annotation_and_separate_batch_pilot_inside_one_pool_excluding_other_workflows",
        unique_native_attempts=len(executions), attempts_with_observed_usage=len(known),
        attempts_without_observed_usage=sum(row["usage"] is None for row in executions.values()),
        observed_tokens={key: sum(row.get(key, 0) for row in known) for key in keys},
        unknown_usage_not_replaced_by_zero=True, execution_artifact_copies_not_charged_twice=True,
        subscription_dollar_cost=None, external_paid_api_calls=0, human_gold=False,
        executions=list(executions.values()), source_sha256=source_hashes,
        note="Known token totals are not all-workflow usage or measured subscription-dollar cost.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(args.pool_dir.resolve()):
        raise ValueError("Save audit only as a new artifact inside the declared pool")
    result = audit(args.pool_dir)
    native.write_new(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"executions", "source_sha256"}}, indent=2))
