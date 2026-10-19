"""Reject literal Han text in Git-visible publication files, not ignored data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import unicodedata


def is_han(character):
    name = unicodedata.name(character, "")
    return name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))


def visible_files(root):
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root, check=True, capture_output=True)
    return sorted(set(name for name in result.stdout.decode("utf-8").split("\0") if name))


def scan(root, names):
    root = Path(root).resolve()
    findings = []
    for name in names:
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or unsafe public file path")
        if any(is_han(character) for character in name):
            findings.append(dict(path=name, line=0, reason="Han text in file name"))
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Non-UTF8 public asset requires separate language review: " + name) from exc
        for line, content in enumerate(text.splitlines(), 1):
            if any(is_han(character) for character in content):
                findings.append(dict(path=name, line=line, reason="Literal Han text"))
    return dict(files_checked=len(names), findings=findings, passed=not findings,
                scope="Current Git-visible tree only; ignored data and prior commit history are excluded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = scan(args.root, visible_files(args.root))
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
