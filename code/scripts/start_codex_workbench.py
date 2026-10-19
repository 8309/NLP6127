"""Start the English localhost UI with an explicit read-only staging library."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

# Use this publication checkout with the documented original project interpreter.
# This affects this launcher process only, not global PATH or installed packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_evidence.codex_runtime import load_staged_runtime
from oil_evidence.store import ROOT


def command(canonical_root, corpus_run, port):
    if not 1024 <= port <= 65535:
        raise ValueError("Choose a nonprivileged local port")
    runtime = load_staged_runtime(canonical_root, corpus_run)
    environment = dict(os.environ)
    environment.update(OIL_CANONICAL_ROOT=str(runtime.canonical_root), OIL_STAGED_RUN=corpus_run,
                       OIL_GENERATION_PROVIDER="codex", PYTHONPATH=str(ROOT / "src"),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    args = [sys.executable, "-X", "utf8", "-m", "streamlit", "run", str(ROOT / "app.py"),
            "--server.address", "127.0.0.1", "--server.port", str(port),
            "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    return args, environment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--corpus-run", default="semantic-page-v2-buffer")
    parser.add_argument("--port", type=int, default=8518)
    options = parser.parse_args()
    args, environment = command(options.canonical_root, options.corpus_run, options.port)
    raise SystemExit(subprocess.run(args, cwd=ROOT, env=environment).returncode)
