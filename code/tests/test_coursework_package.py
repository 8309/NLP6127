"""Deterministic packaging and privacy rules, never redistribution clearance."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


@pytest.fixture
def packager():
    spec = importlib.util.spec_from_file_location("coursework_packager", Path(__file__).parents[1] / "scripts/package_coursework.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["data/raw/report.pdf", "models/model.onnx", ".env", "docs/.env.local", "src/auth.pem", "../secret.json", "tmp/report.md", "src/auth.json", "docs/model.safetensors"])
def test_private_or_rights_sensitive_files_excluded(packager, name):
    assert not packager.allowed_source(name)


def test_code_and_public_summaries_included(packager):
    assert packager.allowed_source("app.py")
    assert packager.allowed_source("src/oil_evidence/codex_runtime.py")
    assert packager.allowed_source("reports/progress.json")
    assert not packager.allowed_source("reports/private.json")


def test_summary_redacts_text_secrets_and_local_paths(packager):
    value = dict(status="complete", usage=dict(input_tokens=12), claims=[dict(text="source")],
        quotes=[dict(text="source")], evidence=[dict(text="source")], artifact_directory="D:/private/path",
        OPENAI_API_KEY="synthetic", nested=dict(path="D:/private/file", corpus_sha256="fixture"))
    result = packager.sanitize_summary(value)
    assert result == dict(status="complete", usage=dict(input_tokens=12), nested=dict(path="[local path excluded]", corpus_sha256="fixture"))


def test_zip_bytes_reproduce_without_timestamp_drift(packager, tmp_path):
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    packager.deterministic_zip(first, {"b.txt":b"two", "a.txt":b"one"})
    packager.deterministic_zip(second, {"a.txt":b"one", "b.txt":b"two"})
    assert first.read_bytes() == second.read_bytes()


def test_credential_literal_refuses_package_without_echoing_value(packager):
    token = b"sk-" + b"x" * 32
    with pytest.raises(ValueError, match="package refused") as failure:
        packager.guard_source_bytes("fixture.py", b'API_KEY="' + token + b'"')
    assert token.decode() not in str(failure.value)
    packager.guard_source_bytes("fixture.py", b'API_KEY="synthetic-placeholder"')


def test_credential_in_arbitrary_summary_value_is_not_copied(packager):
    token = "sk-" + "x" * 32
    result = packager.sanitize_summary(dict(message="Synthetic failure " + token,
        nested=[token, dict(status="complete", value=token)], normal="machine-only"))
    assert result == dict(message="[potential credential excluded]",
        nested=["[potential credential excluded]", dict(status="complete", value="[potential credential excluded]")],
        normal="machine-only")
    assert token not in str(result)


@pytest.fixture
def packaging_checkout(packager, tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    files = {
        "app.py": b"print('public fixture')\n",
        "scripts/continue_recovered_formal_closeout.py": b"# Paused source fixture.\n",
        "tests/test_recovered_formal_closeout.py": b"# Paused test fixture.\n",
    }
    for name, raw in files.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    monkeypatch.setattr(packager, "ROOT", checkout)
    monkeypatch.setattr(packager, "PUBLIC_REPORTS", set())

    def git(args, **kwargs):
        assert kwargs["cwd"] == checkout
        assert args[0] == "git"
        if args[1] == "ls-files":
            assert args == ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
            return SimpleNamespace(stdout=("\0".join(files) + "\0").encode())
        assert args == ["git", "rev-parse", "HEAD"]
        return SimpleNamespace(stdout=b"fixture-commit\n")

    monkeypatch.setattr(packager.subprocess, "run", git)
    return checkout, files, tmp_path / "canonical"


def test_default_package_still_includes_allowed_untracked_drafts(packager, packaging_checkout):
    checkout, files, canonical = packaging_checkout
    output, _ = packager.package(canonical)
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["excluded_requested"] == manifest["actually_excluded"] == []
    with zipfile.ZipFile(output / "source.zip") as archive:
        assert set(archive.namelist()) == {"MANIFEST.json", *("code/" + name for name in files)}
    assert all((checkout / name).read_bytes() == raw for name, raw in files.items())


def test_exact_exclusions_preserve_files_and_record_both_lists(packager, packaging_checkout):
    checkout, files, canonical = packaging_checkout
    excluded = [name for name in files if name != "app.py"]
    output, _ = packager.package(canonical, exclude_source=[*reversed(excluded), excluded[0]])
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["excluded_requested"] == manifest["actually_excluded"] == sorted(excluded)
    with zipfile.ZipFile(output / "source.zip") as archive:
        assert set(archive.namelist()) == {"MANIFEST.json", "code/app.py"}
        assert json.loads(archive.read("MANIFEST.json")) == manifest
    assert all((checkout / name).read_bytes() == raw for name, raw in files.items())


def test_requested_allowed_file_not_in_git_inventory_is_not_actually_excluded(packager, packaging_checkout):
    checkout, _, canonical = packaging_checkout
    relative = "scripts/ignored_fixture.py"
    (checkout / relative).write_bytes(b"# Allowed fixture outside mocked Git inventory.\n")
    output, _ = packager.package(canonical, exclude_source=[relative])
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["excluded_requested"] == [relative]
    assert manifest["actually_excluded"] == []


@pytest.mark.parametrize("name", [
    "/absolute.py", "C:/absolute.py", "C:drive-relative.py", "\\\\host\\share\\source.py",
    "../app.py", "scripts/../app.py", "scripts", "scripts/", "scripts/*.py",
    "scripts/?.py", "scripts/[ab].py", "data/corpus.jsonl", "docs/.env", "scripts/missing.py",
])
def test_invalid_exclusion_refuses_before_creating_package(packager, packaging_checkout, name):
    checkout, files, canonical = packaging_checkout
    with pytest.raises(ValueError, match="Source exclusion"):
        packager.package(canonical, exclude_source=[name])
    assert not canonical.exists()
    assert all((checkout / relative).read_bytes() == raw for relative, raw in files.items())
