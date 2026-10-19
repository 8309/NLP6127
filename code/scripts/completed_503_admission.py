"""Read-only admission of two sealed certificates, never native acceptance."""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType
from uuid import uuid4

import run_codex_annotation as native
from oil_evidence import machine_evaluation as protocol

ROOT = Path(__file__).resolve().parents[1]
VERSION = "sealed-completed-503-read-only-admission-v1"
LOADER_POLICY = "verified-source-bytes-no-archived-bytecache-v1"
CERTIFIER_SHA256 = "a7e9eb6d89ae489de6249f7e316e844aa94749d16ad55dbc058a8b480b2cce4e"
PINNED = {
    "annotation_b38": ("71fdd5e5a5e0ab36b29771c1f19208887b89c672d2f7016f3f7cde3ba0fdc56c",
        "f2ee0fdf6513ae8f2e0607296a23ad18afad7a508abec6a5a8771e0861060c67",
        "91f3ca800a9caeabe9860e4c189466f52e20c80bb86eacc4a4993789a3fc3ce3"),
    "source_group2": ("3d7abfdec7961bb325449de37452dae1c991f32935490770940f2e6d3f0ee7cd",
        "1c019c508d549c5a8e32e806976887f98119afa7e4e6d960e2238ddc893e13f6",
        "d8c6ddfbb2fc032a9e3584be46329857037fc0b7bf7a314986b0d58b4ab969b1"),
}
DEPENDENCIES = {
    "scripts/certify_completed_503_result.py": CERTIFIER_SHA256,
    "src/oil_evidence/machine_evaluation.py": "97da16165468343295d77e456c16422a4f873f48ec965dbefffdc25e5f20caa0",
    "scripts/run_codex_annotation.py": "6dadf606ae11ad80714914d98a4edc7e82c75d66b62cce498fb968f8cc941e52",
    "src/oil_evidence/wire_codec.py": "5b08efcd5b893def460faadf5fd90615098a77c42e91d27bdab072f8a4ca04e6",
    "scripts/adjudicate_machine_annotations.py": "7206d7e5954433f8d19649de4c5311e8c1539915785c9247ea1c964f7931dc9c",
    "scripts/compact_machine_adjudication.py": "1fbca15ec78e885c3800fb09e31a6546f2d5286d76aac3bab8a8bf16ae99b91d",
}


def guard(path, expected, guards):
    path = Path(path).resolve()
    if (protocol.sha256(path.read_bytes()) != expected
            or str(path) in guards and guards[str(path)] != expected):
        raise ValueError("Sealed certificate, dependency or original proof changed")
    guards[str(path)] = expected


def _compiled(path, expected):
    path, raw = Path(path).resolve(), Path(path).read_bytes()
    if protocol.sha256(raw) != expected:
        raise ValueError("Executed proof source bytes differ from their immutable pin")
    name = "oil_evidence.completed_503_admitted_" + uuid4().hex
    module = ModuleType(name)
    module.__file__, module.__package__ = str(path), "oil_evidence"
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def _certifier():
    module = _compiled(ROOT / "scripts/certify_completed_503_result.py", CERTIFIER_SHA256)
    snapshots = {"compact_machine_adjudication.py": DEPENDENCIES["scripts/compact_machine_adjudication.py"],
        "run_codex_annotation.py": DEPENDENCIES["scripts/run_codex_annotation.py"],
        "codex_runtime.py": "60b47a86a9714fc6f7f2e67bdacc64af2017cd9bc13c681634be5e9af52fd97f"}
    module.isolated_file = lambda path: _compiled(path, snapshots[Path(path).name])
    return module


def _source_guards(mode, path, receipt, certifier, guards):
    pin = certifier.KNOWN[mode]
    guard(path, pin["receipt_sha256"], guards)
    for name, digest in receipt["artifacts_sha256"].items():
        guard(native.contained_file(path.parent, name), digest, guards)
    if mode == "annotation_b38":
        run = Path(receipt["annotation_run"]).resolve()
        guard(run / "manifest.json", pin["manifest_sha256"], guards)
        manifest = native.read_json(run / "manifest.json")
        for name, digest in manifest["immutable_sha256"].items():
            guard(native.contained_file(run, name), digest, guards)
        for name, digest in native.read_json(run / "protected_inputs.json").items():
            guard(name, digest, guards)
        binary = Path(native.read_json(path.parent / "command.json")[0])
    else:
        bounded = path.parent.parent.parent
        guard(bounded / "inputs.json", pin["manifest_sha256"], guards)
        for name, digest in native.read_json(bounded / "inputs.json")["input_sha256"].items():
            guard(name, digest, guards)
        source = bounded.parent.parent.parent / "chunks/chunk-0029"
        for name, digest in pin["source_sha256"].items():
            guard(source / name, digest, guards)
        export = native.read_json(source / "export_receipt.json")
        for name, digest in export["artifacts_sha256"].items():
            guard(native.contained_file(source, name), digest, guards)
        for name, digest in export["source_bindings"].items():
            guard(name, digest, guards)
        binary = native.discover_native_cli(native.DEFAULT_CLI)
    guard(binary, pin["native_binary_sha256"], guards)


def load(mode, certificate_directory):
    """Return certificate, exact derived-record object and through-call guards."""
    if mode not in PINNED:
        raise ValueError("Only the two explicitly registered certificate modes are admitted")
    directory, guards = Path(certificate_directory).resolve(), {}
    for name, digest in DEPENDENCIES.items():
        guard(ROOT / name, digest, guards)
    certificate_sha, derived_sha, authorization_sha = PINNED[mode]
    guard(directory / "certificate.json", certificate_sha, guards)
    guard(directory / "derived_records.json", derived_sha, guards)
    certificate = native.read_json(directory / "certificate.json")
    derived = native.read_json(directory / "derived_records.json")
    if (certificate.get("mode") != mode or certificate.get("operational_receipt_accepted") is not False
            or certificate.get("human_gold") is not False or certificate.get("semantic_accuracy_claimed") is not False
            or type(certificate.get("new_native_calls")) is not int or certificate["new_native_calls"] != 0
            or derived.get("operational_receipt_accepted") is not False):
        raise ValueError("Certificates cannot promote native acceptance, human gold or new invocations")
    authorization_path = Path(certificate["authorization_path"]).resolve()
    guard(authorization_path, authorization_sha, guards)
    path = Path(certificate["original_receipt"]).resolve()
    receipt = native.read_json(path)
    if receipt.get("accepted") is not False:
        raise ValueError("Original native receipt must remain operationally false")
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        certifier = _certifier()
        _source_guards(mode, path, receipt, certifier, guards)
        verified = certifier.verify(mode, path)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    records = verified.pop("derived_records")
    expected = dict(verified, version=certifier.VERSION, certifier_sha256=CERTIFIER_SHA256,
        authorization_path=str(authorization_path), authorization_sha256=authorization_sha,
        derived_records_sha256=derived_sha)
    if (certificate != expected or derived != dict(certificate_path=str(directory / "certificate.json"),
            original_receipt=str(path), operational_receipt_accepted=False, records=records)):
        raise ValueError("Sealed certificate/derived records differ from exact current full-mode proof")
    native.protected_unchanged(guards)
    return certificate, records, guards
