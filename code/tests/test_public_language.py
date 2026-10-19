"""Publication language guards use generated temporary fixtures only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def checker():
    path = Path(__file__).parents[1] / "scripts/check_public_language.py"
    spec = importlib.util.spec_from_file_location("isolated_public_language", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_han_detection_keeps_english_middle_dot_and_accents():
    module = checker()
    assert module.is_han(chr(0x4E00))
    assert module.is_han(chr(0x20000))
    assert not any(module.is_han(character) for character in "Oil Evidence · naive cafe")


def test_line_and_filename_findings_do_not_disclose_source_content(tmp_path):
    module = checker()
    name = chr(0x4E00) + ".txt"
    (tmp_path / name).write_text("English\n" + chr(0x4E00) + "\n", encoding="utf-8")
    result = module.scan(tmp_path, [name])
    assert not result["passed"]
    assert [item["line"] for item in result["findings"]] == [0, 2]
    assert all(set(item) == {"path", "line", "reason"} for item in result["findings"])


def test_ignored_source_data_is_not_scanned_without_git_visibility(tmp_path):
    module = checker()
    (tmp_path / "README.md").write_text("English project\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/private.txt").write_text(chr(0x4E00), encoding="utf-8")
    result = module.scan(tmp_path, ["README.md"])
    assert result["passed"] and result["files_checked"] == 1


@pytest.mark.parametrize("name", ["../outside.txt", "missing.txt"])
def test_unsafe_or_missing_public_paths_fail_closed(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe public"):
        checker().scan(tmp_path, [name])


def test_binary_public_assets_require_an_explicit_review(tmp_path):
    (tmp_path / "asset.bin").write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="separate language review"):
        checker().scan(tmp_path, ["asset.bin"])
