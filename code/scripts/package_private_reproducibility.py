"""Local-private, completed-run byte snapshot; no upload or redistribution grant.

Requires a separately produced, explicitly hash-pinned successful offline audit.
Original absolute paths remain in frozen files; the manifest only maps them to
archive entries. Extraction elsewhere does not establish portable execution.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

FORBIDDEN = {"auth.json", "credentials.json", ".netrc", ".git", ".venv", "__pycache__", "logs"}
SECRETS = re.compile(rb"sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
VARIANTS = {"bm25", "dense", "hybrid_base", "hybrid_A", "hybrid_B", "hybrid_AB"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def logical_path(value):
    """Containment identity only; never retain Win32 namespace in root checks."""
    text = os.fspath(value)
    if os.name == "nt":
        text = text.replace("/", "\\")
        if text[:8].upper() == "\\\\?\\UNC\\":
            text = "\\\\" + text[8:]
        elif text.startswith("\\\\?\\"):
            require(bool(re.match(r"^[A-Za-z]:\\", text[4:])), "Unsupported device namespace")
            text = text[4:]
        require(not text.startswith("\\\\.\\"), "Device namespace refused")
    return Path(text).absolute()


def io_path(value):
    """Explicit extended-path IO avoids Windows MAX_PATH failures."""
    path = logical_path(value)
    if os.name != "nt":
        return path
    text = str(path)
    return Path("\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text)


def resolved_path(value):
    return logical_path(io_path(value).resolve())


def sha(path):
    value = hashlib.sha256()
    with io_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(262144), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(io_path(path).read_bytes())


def safe_file(path, roots):
    path = logical_path(path)
    resolved = resolved_path(path)
    require(path == resolved and io_path(path).is_file(), "Missing file, symlink or noncanonical path")
    for label, root in roots.items():
        root = resolved_path(root)
        if resolved.is_relative_to(root):
            parts = resolved.relative_to(root).parts
            require(not any(p.lower() in FORBIDDEN or p.lower().startswith(".env") for p in parts)
                    and resolved.suffix.lower() not in {".pem", ".key", ".p12", ".pfx"}, "Credential/environment path refused")
            return label + "/" + "/".join(parts)
    raise ValueError("Input file escapes the exact project roots")


def inventory(canonical_root, formal_run, audit_path, audit_sha256, runtime_dir, publication_root=None,
              benign_url_pattern_review=None):
    canonical = resolved_path(canonical_root)
    publication = resolved_path(publication_root or Path(__file__).parents[1])
    roots = {"canonical": canonical, "publication": publication}
    require(canonical != publication and not canonical.is_relative_to(publication)
            and not publication.is_relative_to(canonical), "Project roots must be separate")
    run, audit_path, runtime = map(resolved_path, (formal_run, audit_path, runtime_dir))
    require(run.is_relative_to(canonical / "data/machine_runs/formal-experiments")
            and runtime.is_relative_to(canonical / "data/machine_runs/experiment-runtime"), "Unexpected run/runtime directory")
    safe_file(audit_path, roots)
    require(sha(audit_path) == audit_sha256, "Offline audit pin mismatch")
    safe_file(run / "plan.json", roots)
    safe_file(run / "result.json", roots)
    audit, plan, result = read(audit_path), read(run / "plan.json"), read(run / "result.json")
    require(audit.get("status") == "completed_formal_result_offline_audit_passed"
            and audit.get("human_gold") is False and audit.get("formal_heldout_test") is True
            and audit.get("native_calls") == 0, "A completed machine-only offline audit is required")
    expected = {"status": "completed_formal_machine_only_experiment", "human_gold": False,
                "formal_heldout_test": True, "completed_retrieval_questions": 1000,
                "completed_answer_questions": 100, "completed_variant_answers": 600,
                "original_frozen_inputs_unchanged": True, "error": None}
    require(all(result.get(k) == v and type(result.get(k)) is type(v) for k, v in expected.items()), "Actual 100/600 completion is required")
    require(audit.get("answer_questions") == 100 and audit.get("variant_answers") == 600
            and audit.get("ranked_test_questions") == 1000 and audit.get("ranked_query_conditions") == 6000
            and audit.get("result_sha256") == sha(run / "result.json")
            and audit.get("plan_sha256") == sha(run / "plan.json")
            and audit.get("summary") == result.get("summary") and set(result.get("summary", {})) == VARIANTS
            and audit.get("paired_automatic_correctness") == result.get("paired_automatic_correctness")
            and audit.get("usage") == result.get("usage"), "Audit/result bytes, metrics or usage differ")
    files = {}

    def add(path, digest):
        safe_file(path, roots)
        key, alias = logical_path(path), str(Path(path).absolute())
        require(key not in files or files[key]["sha256"] == digest, "Conflicting path-alias SHA256 pins")
        record = files.setdefault(key, dict(sha256=digest, aliases=[]))
        if alias not in record["aliases"]:
            record["aliases"].append(alias)

    for path, digest in audit.get("protected_sha256", {}).items():
        add(path, digest)
    require(files and all(files.get(logical_path(p), {}).get("sha256") == h for p, h in plan["protected_sha256"].items()), "Audit lacks the full original plan closure")
    for path, digest in plan["protected_sha256"].items():
        add(path, digest)
    require(len(plan["all_test_record_ids"]) == len(set(plan["all_test_record_ids"])) == 1000
            and len(plan["answer_record_ids"]) == len(set(plan["answer_record_ids"])) == 100
            and set(plan["answer_record_ids"]).issubset(plan["all_test_record_ids"]), "Original sample IDs differ")
    require(result.get("dataset_version") == plan.get("dataset_version")
            and result.get("answer_sample_plan_sha256") == plan["answer_sample_plan_sha256"], "Dataset/sample binding differs")
    pairs = []
    for name, digest in result.get("group_artifacts_sha256", {}).items():
        group = run / name
        safe_file(group, roots)
        require(resolved_path(group).is_relative_to(run / "groups") and sha(group) == digest, "Saved group pin differs")
        pairs.extend((row["record_id"], row["variant"]) for row in read(group)["rows"])
    require(len(pairs) == 600 and set(pairs) == {(qid, variant) for qid in plan["answer_record_ids"] for variant in VARIANTS}, "Saved groups lack exact 600 unique pairs")

    def selected(suffix):
        found = {logical_path(p) for p in plan["protected_sha256"] if logical_path(p).as_posix().endswith(suffix)}
        require(len(found) == 1, "Expected one protected " + suffix)
        return found.pop()

    documents = selected("/stage/documents.jsonl")
    stage = documents.parent
    frozen = [p.parent for p in files if "frozen_evaluation" in p.parts and p.name == "manifest.json"]
    require(len(frozen) == 1, "Expected one frozen evaluation manifest")
    sample = selected("/sample_plan.json").parent
    retrieval = resolved_path(plan["retrieval"]["run"])
    require(retrieval.is_relative_to(canonical / "data/machine_runs/candidate-retrieval"), "Retrieval path escapes its allowlist")
    for folder in (stage, frozen[0], sample, retrieval, runtime, canonical / "models", run):
        require(io_path(folder).is_dir(), "Required snapshot directory is missing")
        for path in io_path(folder).rglob("*"):
            if path.is_file():
                safe_file(path, roots)
                add(path, sha(path))
    add(audit_path, audit_sha256)
    registry, seen = [], set()
    for line in io_path(documents).read_bytes().splitlines():
        doc = json.loads(line)
        require(doc["document_id"] not in seen, "Duplicate raw document ID")
        seen.add(doc["document_id"])
        raw = canonical / "data" / doc["raw_path"]
        safe_file(raw, roots)
        require(resolved_path(raw).is_relative_to(canonical / "data/raw") and io_path(raw).is_file()
                and io_path(raw).stat().st_size == doc["bytes"] and sha(raw) == doc["raw_sha256"], "Original raw record is missing or changed")
        add(raw, doc["raw_sha256"])
        registry.append({k: doc.get(k) for k in ("document_id", "source", "url", "redistribution")})
    require(registry, "Full staged raw registry must not be empty")
    index = read(stage / "index/manifest.json")
    for name, expected_hash in index["onnx_sha256"].items():
        require(files.get(canonical / "models" / name, {}).get("sha256") == expected_hash, "Original ONNX model pin differs")
    entries = {}
    for path, record in files.items():
        expected_hash = record["sha256"]
        name = safe_file(path, roots)
        require(re.fullmatch("[a-f0-9]{64}", expected_hash) and sha(path) == expected_hash, "Input SHA256 differs")
        entries[name] = {"original_absolute_path": record["aliases"][0], "original_path_aliases": sorted(record["aliases"]),
                         "sha256": expected_hash, "bytes": io_path(path).stat().st_size}
    reviewed = None
    if benign_url_pattern_review is not None:
        review_path = resolved_path(benign_url_pattern_review)
        review_name = safe_file(review_path, roots)
        require(review_path.parent == canonical / "data/machine_runs/private_reproducibility_review"
                and review_path.suffix == ".json", "Unexpected benign review certificate path")
        certificate = read(review_path)
        require(isinstance(certificate, dict) and set(certificate) == {"schema_version", "classification", "review_provenance", "findings"}
                and type(certificate["schema_version"]) is int and certificate["schema_version"] == 1
                and certificate["classification"] == "embedded_public_article_url_slug"
                and certificate["review_provenance"] == "codex_syntax_review_not_human"
                and isinstance(certificate["findings"], list) and certificate["findings"], "Invalid benign review certificate")
        rows, spans = [], set()
        for row in certificate["findings"]:
            require(isinstance(row, dict) and set(row) == {"archive_path", "file_sha256", "byte_offset", "match_length", "match_sha256"},
                    "Invalid benign finding fields")
            name = row["archive_path"]
            require(isinstance(name, str) and re.fullmatch(r"canonical/data/raw/[A-Za-z0-9_.-]+\.(?:html?|pdf)", name)
                    and name in entries, "Benign review may only bind an archived original raw HTML/PDF")
            require(all(isinstance(row[k], str) and re.fullmatch("[a-f0-9]{64}", row[k]) for k in ("file_sha256", "match_sha256"))
                    and row["file_sha256"] == entries[name]["sha256"], "Benign review full-file SHA256 differs")
            require(type(row["byte_offset"]) is int and row["byte_offset"] >= 0
                    and type(row["match_length"]) is int and 1 <= row["match_length"] <= 128
                    and row["byte_offset"] + row["match_length"] <= entries[name]["bytes"], "Invalid benign finding span")
            span = (name, row["byte_offset"], row["match_length"])
            require(span not in spans and not any(n == name and row["byte_offset"] < start + length
                    and start < row["byte_offset"] + row["match_length"] for n, start, length in spans), "Duplicate or overlapping benign finding")
            spans.add(span)
            with io_path(entries[name]["original_absolute_path"]).open("rb") as source:
                source.seek(row["byte_offset"])
                matched = source.read(row["match_length"])
            require(SECRETS.fullmatch(matched) and hashlib.sha256(matched).hexdigest() == row["match_sha256"], "Benign finding bytes differ")
            rows.append(dict(row))
        digest = sha(review_path)
        entries[review_name] = {"original_absolute_path": str(review_path), "original_path_aliases": [str(review_path)],
                               "sha256": digest, "bytes": io_path(review_path).stat().st_size}
        reviewed = dict(review_file=review_name, review_file_sha256=digest,
                        classification=certificate["classification"], review_provenance=certificate["review_provenance"],
                        reviewed_benign_findings=sorted(rows, key=lambda row: (row["archive_path"], row["byte_offset"])))
    manifest = dict(schema_version=1, scope="local_private_only_not_redistribution_clearance",
        input_binding=dict(canonical_root=str(canonical), formal_run=str(run), audit_path=str(audit_path),
            audit_sha256=audit_sha256, runtime_dir=str(runtime), publication_root=str(publication),
            benign_url_pattern_review=str(review_path) if reviewed is not None else None),
        completed_machine_answers=600, audit_sha256=audit_sha256, entries=entries, raw_registry=registry,
        credential_pattern_review=reviewed,
        raw_document_count=len(registry), source_license_counts=dict(Counter(str(r["redistribution"]) for r in registry)),
        rights_review_required=True, cloud_sharing_authorized=False, frozen_files_rewritten=False,
        model_license_file_present=any(p.is_file() and p.name.lower().startswith("license") for p in io_path(canonical / "models").rglob("*")),
        limitations=["The map is portable metadata, not a rewritten or proven portable runtime.",
            "Exact-layout replay needs original absolute paths and compatible locked dependencies.",
            "Offline saved-result replay differs from new generation, which needs the model service.",
            "Supplemental snapshots receive fresh byte hashes, not invented historical pins.",
            "Source review_required and missing model LICENSE remain unresolved; no sharing grant."])
    return manifest


def package(manifest, canonical_root, output):
    require(manifest == inventory(**manifest["input_binding"])
            and resolved_path(canonical_root) == logical_path(manifest["input_binding"]["canonical_root"]), "Inventory changed or was altered")
    output = logical_path(output)
    require(output == resolved_path(output) and output.is_relative_to(resolved_path(canonical_root) / "data/machine_runs/private_reproducibility")
            and output.suffix == ".zip" and not io_path(output).exists(), "Use a new local-private ZIP destination")
    partial = output.with_suffix(".zip.incomplete")
    require(not io_path(partial).exists(), "Never overwrite an incomplete attempt")
    io_path(output.parent).mkdir(parents=True, exist_ok=True)
    approved = {}
    for row in (manifest["credential_pattern_review"] or {}).get("reviewed_benign_findings", []):
        key = (row["archive_path"], row["file_sha256"], row["byte_offset"], row["match_length"], row["match_sha256"])
        approved[key] = row
    observed = set()
    with zipfile.ZipFile(io_path(partial), "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, entry in sorted(manifest["entries"].items()):
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe archive entry")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o100644 << 16
            checksum, carry, consumed = hashlib.sha256(), b"", 0
            with io_path(entry["original_absolute_path"]).open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                for block in iter(lambda: source.read(262144), b""):
                    window = carry + block
                    for match in SECRETS.finditer(window):
                        start = consumed - len(carry) + match.start()
                        key = (name, entry["sha256"], start, len(match.group()), hashlib.sha256(match.group()).hexdigest())
                        # A certified match may straddle blocks; only its exact, hash-bound span can be deferred.
                        if key not in approved and match.end() == len(window) and any(
                                row["archive_path"] == name and row["file_sha256"] == entry["sha256"]
                                and row["byte_offset"] == start and len(match.group()) < row["match_length"]
                                for row in approved.values()):
                            continue
                        require(key in approved, "Potential credential bytes refused")
                        observed.add(key)
                    carry = block[-128:]
                    consumed += len(block)
                    checksum.update(block)
                    target.write(block)
            require(checksum.hexdigest() == entry["sha256"], "Input changed during packaging")
        require(observed == set(approved), "Unused or incomplete benign review findings refused")
        raw = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False).encode()
        require(not SECRETS.search(raw), "Potential credential in manifest")
        info = zipfile.ZipInfo("MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o100644 << 16
        archive.writestr(info, raw)
    require(all(sha(e["original_absolute_path"]) == e["sha256"] for e in manifest["entries"].values()), "Input changed after packaging")
    os.link(io_path(partial), io_path(output))  # Exclusive publication: never replace an existing file.
    io_path(partial).unlink()  # This newly created, verified temporary file only.
    return {"zip_sha256": sha(output), "bytes": io_path(output).stat().st_size, "files": len(manifest["entries"]), "cloud_sharing_authorized": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-root", "formal-run", "audit", "runtime-dir", "output-new"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--benign-url-pattern-review", type=Path,
                        help="Exact hash/span-bound, machine syntax review certificate; never a global pattern exemption")
    args = parser.parse_args()
    manifest = inventory(args.canonical_root, args.formal_run, args.audit, args.audit_sha256, args.runtime_dir,
                         benign_url_pattern_review=args.benign_url_pattern_review)
    print(json.dumps(package(manifest, args.canonical_root, args.output_new), indent=2))
