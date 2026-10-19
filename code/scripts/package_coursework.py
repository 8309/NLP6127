"""Package source and redacted aggregate results, never raw data or credentials.

ZIP entry timestamps and file ordering are deterministic. The code archive
binds the actual working-tree bytes, not just its older Git base commit.
Original documents, evidence text, model weights and identity files are excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
from uuid import uuid4
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oil_evidence.store import ROOT, digest, now

TOP_FOLDERS = {"src", "scripts", "tests", "docs", "seeds", ".github"}
ROOT_FILES = {"app.py", "AGENTS.md", "README.md", "TASKS.md", "config.toml", "pyproject.toml", "uv.lock",
              ".gitignore", ".gitattributes", "start-workbench.cmd"}
PUBLIC_REPORTS = {"reports/progress.json", "reports/english_publication.json"}
FORBIDDEN_SUFFIXES = {".sqlite", ".sqlite3", ".db", ".pem", ".key", ".p12", ".pfx", ".onnx", ".pdf",
                      ".safetensors", ".pt", ".pth", ".bin"}
PRIVATE_FIELDS = {"text", "question", "question_zh", "evidence", "claims", "quotes", "answer_points", "raw", "raw_output",
    "raw_outputs", "stdin", "stderr", "prompt", "system_prompt", "protected_sha256", "input_sha256",
    "artifact_directory", "output_directory", "output", "corpus_stage", "receipt_directory"}
CREDENTIAL_PATTERN = re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b|\bghp_[A-Za-z0-9]{20,}\b|"
    rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")


def allowed_source(name):
    path = Path(name)
    parts = name.replace("\\", "/").split("/")
    if (not name or Path(name).is_absolute() or ".." in parts or any(part.startswith(".env") for part in parts)
            or any(part in {"data", "models", ".venv", "__pycache__", ".git", "tmp", "logs"} for part in parts)
            or path.name.lower() in {"auth.json", "credentials.json", ".netrc"}
            or path.suffix.lower() in FORBIDDEN_SUFFIXES):
        return False
    return name in ROOT_FILES or name in PUBLIC_REPORTS or parts[0] in TOP_FOLDERS


def sanitize_summary(value):
    """Strip source-bearing fields and private paths from copied summaries."""
    if isinstance(value, dict):
        return {key: sanitize_summary(item) for key, item in value.items()
                if key not in PRIVATE_FIELDS and not re.search(r"key|secret|password|credential|auth", key, re.I)
                and not re.match(r"^[A-Za-z]:[\\/]", key)}
    if isinstance(value, list):
        return [sanitize_summary(item) for item in value]
    if isinstance(value, str) and CREDENTIAL_PATTERN.search(value.encode("utf-8")):
        return "[potential credential excluded]"
    if isinstance(value, str) and (re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith(("\\\\", "/"))):
        return "[local path excluded]"
    return value


def deterministic_zip(path, entries):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw, compresslevel=9)


def guard_source_bytes(name, raw):
    """Refuse potential credential literals without printing or changing them."""
    if CREDENTIAL_PATTERN.search(raw):
        raise ValueError("Potential credential literal in source; package refused: " + name)


def validated_exclusions(exclude_source):
    """Accept existing, allowed files by exact checkout-relative path only."""
    requested = set()
    checkout = ROOT.resolve()
    for name in exclude_source:
        if not isinstance(name, str) or not name or any(char in name for char in "*?[]"):
            raise ValueError("Source exclusion must be an exact allowed file path")
        relative = name.replace("\\", "/")
        if (Path(name).is_absolute() or PureWindowsPath(name).drive
                or PureWindowsPath(name).root or any(part in {"", ".", ".."} for part in relative.split("/"))
                or not allowed_source(relative)):
            raise ValueError("Source exclusion must be an exact allowed file path")
        path = (checkout / relative).resolve()
        if not path.is_relative_to(checkout) or not path.is_file():
            raise ValueError("Source exclusion must name an existing allowed file inside the checkout")
        requested.add(relative)
    return sorted(requested)


def package(canonical_root, receipts=(), exclude_source=()):
    canonical = Path(canonical_root).resolve()
    excluded_requested = validated_exclusions(exclude_source)
    excluded_set, actually_excluded = set(excluded_requested), []
    names = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    source = {}
    for name in sorted(set(names)):
        if not allowed_source(name):
            continue
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError("Source file missing or outside checkout")
        if name in excluded_set:
            actually_excluded.append(name)
            continue
        raw = path.read_bytes()
        guard_source_bytes(name, raw)
        source["code/" + name] = raw
    results, receipt_hashes = {}, {}
    for name in sorted(PUBLIC_REPORTS):
        path = ROOT / name
        if path.exists():
            results["results/" + Path(name).name] = json.dumps(sanitize_summary(json.loads(path.read_bytes())),
                ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    for number, input_path in enumerate(receipts, 1):
        path = Path(input_path).resolve()
        if not path.is_relative_to((canonical / "data").resolve()) or not path.is_file() or path.suffix != ".json":
            raise ValueError("Result receipt must be an existing canonical data JSON")
        name = f"results/aggregate-{number:03d}.json"
        raw = path.read_bytes()
        results[name] = json.dumps(sanitize_summary(json.loads(raw)), ensure_ascii=False,
            sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        receipt_hashes[name] = digest(raw)
    for name, raw in results.items():
        guard_source_bytes(name, raw)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout.decode().strip()
    manifest = dict(schema_version=1, git_base_commit=commit,
        excluded_requested=excluded_requested, actually_excluded=actually_excluded,
        actual_source_sha256={name: digest(raw) for name, raw in source.items()},
        aggregate_result_sha256={name: digest(raw) for name, raw in results.items()},
        original_local_receipt_sha256=receipt_hashes, source_file_count=len(source),
        result_file_count=len(results), raw_originals_included=False, evidence_text_included=False,
        model_weights_included=False, authentication_files_included=False,
        formal_acceptance="Not certified by packaging", data_redistribution="Unresolved; raw and corpus text excluded",
        limitations=["Results archive contains aggregates, not source evidence or the frozen evaluation dataset.",
                     "Original data/model snapshots are still needed for historical replay.",
                     "Final cover identifiers, PDF, presentation and submission links are not supplied by this package.",
                     "Packaging does not establish that all required experiments have completed."])
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    source["MANIFEST.json"] = results["MANIFEST.json"] = manifest_bytes
    output = canonical / "data/machine_runs/submission_packages" / (now().replace(":", "-") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    deterministic_zip(output / "source.zip", source)
    deterministic_zip(output / "results.zip", results)
    (output / "manifest.json").write_bytes(manifest_bytes)
    return output, dict(source_files=manifest["source_file_count"], result_files=manifest["result_file_count"],
        source_zip_sha256=digest((output / "source.zip").read_bytes()),
        results_zip_sha256=digest((output / "results.zip").read_bytes()), limitations=manifest["limitations"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--receipt", action="append", default=[], type=Path)
    parser.add_argument("--exclude-source", action="append", default=[],
        help="Exclude one exact allowed checkout-relative file; repeat for additional files")
    args = parser.parse_args()
    directory, summary = package(args.canonical_root, args.receipt, args.exclude_source)
    print(json.dumps(dict(output=str(directory), **summary), indent=2))
