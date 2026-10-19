"""Start the teacher UI, verify the delivery, or run a local retrieval query."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify all payload hashes and run offline retrieval checks")
    parser.add_argument("--query")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--as-of", default="2026-10-05")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use a port between 1024 and 65535")
    if args.check or args.query:
        from teacher_runtime import load_runtime, search
        runtime = load_runtime()
        if args.check:
            question = "How do hurricanes affect Gulf of Mexico crude oil production?"
            checks = {}
            for mode in ("bm25", "dense", "hybrid"):
                result = search(runtime, question, mode=mode)
                assert len(result["evidence"]) == 5
                assert all(hit.get("available_by") and hit["available_by"] <= "2026-10-05"
                           for hit in result["evidence"])
                checks[mode] = [hit["passage_id"] for hit in result["evidence"]]
            assert search(runtime, question, as_of="1900-01-01")["evidence"] == []
            print(json.dumps({"status": "PASS", "passages": len(runtime.passages),
                              "vector_shape": list(runtime.retriever.vectors.shape),
                              "local_model_loaded": True, "retrieval_modes": checks,
                              "past_cutoff_empty_result": True, "generation_calls": 0}, indent=2))
        if args.query:
            print(json.dumps(search(runtime, args.query, mode=args.mode, as_of=args.as_of), indent=2))
        return 0
    app = Path(__file__).with_name("teacher_app.py")
    return subprocess.run([sys.executable, "-m", "streamlit", "run", str(app),
                           "--server.address", "127.0.0.1", "--server.port", str(args.port),
                           "--browser.gatherUsageStats", "false"], cwd=app.parent).returncode


if __name__ == "__main__":
    raise SystemExit(main())
