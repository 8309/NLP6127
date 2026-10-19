"""Independent resegmentation guards: isolated files, no live data or network.

These tests exercise the implementation, not its report's success flags.  The
small corpus fixture preserves a frozen parent/annotation version and includes
a pending near-duplicate parent which must not be counted in the active index.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from oil_evidence.store import digest, read_jsonl, words, write_json, write_jsonl


def _passage(identifier: str, document: str, ordinal: int = 0) -> dict:
    # Two independently complete, topical sentences, each safely 80-160 words.
    # Distinct sentence identifiers avoid artificial exact duplicates.
    sentences = []
    for side, opening in enumerate(("Crude oil production", "Petroleum exports")):
        text = opening + f" in field {ordinal} section {side} "
        text += "remained stable across the surveyed region during the reporting period " * 8
        text += "according to the official report."
        assert 80 <= len(words(text)) <= 160
        sentences.append(text)
    text = " ".join(sentences)
    return dict(passage_id=identifier, document_id=document, source="EIA",
                title="Crude oil report " + document,
                url="https://example.test/" + document,
                published_date="2023-01-01", available_by="2023-01-01",
                date_precision="day", date_source="official archive",
                event_date=None, event_status="unlabelled", redistribution="eia_attribution",
                page=1, regions=["United States"], region_method="keyword_heuristic",
                text=text, word_count=len(words(text)))


def _load_script():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "isolated_resegmentation_guards", scripts / "resegment_corpus.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _live_bytes(data: Path) -> dict[str, bytes]:
    return {name: (data / name).read_bytes() for name in (
        "passages.jsonl", "documents.jsonl", "sources.jsonl", "index/manifest.json",
        "index/vectors.npy", "annotations/codex.jsonl", "annotations/passage_labels.jsonl",
        "annotations/passage_review.jsonl")}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def isolated_resegmentation(tmp_path, monkeypatch):
    module = _load_script()
    data = tmp_path / "project/data"
    data.mkdir(parents=True)
    monkeypatch.setattr(module, "DATA", data)
    monkeypatch.setattr(module, "ROOT", data / "resegmentation_runs/test-reseg-v1")
    # The shared writer lock must stay within this fixture, never the real data.
    if hasattr(module, "expansion"):
        monkeypatch.setattr(module.expansion, "DATA", data)
        monkeypatch.setattr(module.expansion, "ROOT", module.ROOT)
    rows = [_passage("parent-a", "document-a", 1),
            _passage("parent-b", "document-b", 2),
            _passage("parent-near", "document-near", 3),
            _passage("parent-near-b", "document-near-b", 4)]
    write_jsonl(data / "passages.jsonl", rows)
    documents, sources = [], []
    for row in rows:
        source = {key: row[key] for key in ("source", "title", "url", "published_date",
            "available_by", "date_precision", "date_source", "event_date", "event_status", "redistribution")}
        raw = ("<article>" + row["text"] + "</article>").encode("utf-8")
        raw_path = data / "raw" / (row["document_id"] + ".html")
        raw_path.parent.mkdir(exist_ok=True)
        raw_path.write_bytes(raw)
        sources.append(source)
        documents.append(dict(source, document_id=row["document_id"],
            raw_path=raw_path.relative_to(data).as_posix(), raw_sha256=digest(raw), bytes=len(raw)))
    write_jsonl(data / "documents.jsonl", documents)
    write_jsonl(data / "sources.jsonl", sources)
    annotation = data / "expansion_runs/original/baseline/passages.jsonl"
    write_jsonl(annotation, rows[:1])
    frozen = data / "expansion_runs/previous/published/fixture"
    write_jsonl(frozen / "reports/near_duplicate_candidates.jsonl", [
        {"left": "parent-near", "right": "parent-near-b", "similarity": 0.93}])
    write_json(data / "reports/corpus_expansion.json", {
        "corpus_sha256": digest((data / "passages.jsonl").read_bytes()),
        "published_snapshot": str(frozen), "annotation_baseline_path": str(annotation),
        "annotation_baseline_sha256": digest(annotation.read_bytes())})
    write_json(data / "reports/corpus.json", {"passages": len(rows),
        "words": sum(row["word_count"] for row in rows)})
    write_json(data / "reports/passage_annotation.json", {
        "corpus_sha256": digest(annotation.read_bytes()), "labels_sha256": "fixture"})
    for name in ("codex", "passage_labels", "passage_review"):
        write_jsonl(data / f"annotations/{name}.jsonl", [{
            "passage_id": "parent-a", "provenance": "model_only", "fixture": name}])
    (data / "index").mkdir()
    np.save(data / "index/vectors.npy", np.ones((len(rows), 2), dtype="float32"), allow_pickle=False)
    write_json(data / "index/manifest.json", {
        "corpus_sha256": digest((data / "passages.jsonl").read_bytes()),
        "vectors_sha256": digest((data / "index/vectors.npy").read_bytes()),
        "passage_ids": [row["passage_id"] for row in rows], "dimension": 2})
    # Implementation hashes have a genuine mutable fixture file.  No real code
    # or configuration is ever changed by tampering tests.
    for name in ("config.toml", "uv.lock", "pyproject.toml"):
        (data.parent / name).write_text("# isolated guard fixture\n", encoding="utf-8")
    implementation = data.parent / "guard_implementation.py"
    implementation.write_text("# isolated implementation fingerprint\n", encoding="utf-8")
    monkeypatch.setattr(module, "implementation_hashes", lambda: {
        "guard_implementation.py": digest(implementation.read_bytes())})
    return module, data, rows, monkeypatch


@pytest.mark.parametrize("unsafe", ["../outside", "run/path", "run\\path", "D:\\outside", "",
    "ab", "UPPERCASE", "a" * 82, ".hidden", "run name", "con", "prn", "aux", "nul",
    "com1", "com9", "lpt1", "lpt9"])
def test_resegmentation_run_path_fails_closed(isolated_resegmentation, unsafe):
    module, _, _, _ = isolated_resegmentation
    before = module.ROOT
    with pytest.raises(ValueError):
        module.select_run(unsafe)
    assert module.ROOT == before


def test_run_path_is_anchored_in_its_own_data_directory(isolated_resegmentation):
    module, data, _, _ = isolated_resegmentation
    assert module.select_run("oil-natural-80-160-v1") == data / "resegmentation_runs/oil-natural-80-160-v1"


def _stage(module):
    module.snapshot()
    module.stage()
    return module.stage_paths()


def _audit_rejects(module):
    try:
        result = module.audit()
    except (ValueError, RuntimeError):
        return
    assert result.get("errors"), "Damaged evidence must not yield a clean audit"


def _lower_only_size_gates(module, monkeypatch):
    # Production thresholds are checked separately, never silently redefined.
    monkeypatch.setattr(module, "MIN_PASSAGES", 4)
    monkeypatch.setattr(module, "MIN_WORDS", 1)


def _fake_index(module):
    paths = module.stage_paths()
    rows = read_jsonl(paths["passages"])
    paths["vectors"].parent.mkdir(parents=True, exist_ok=True)
    matrix = np.arange(len(rows) * 2, dtype="float32").reshape(len(rows), 2)
    np.save(paths["vectors"], matrix, allow_pickle=False)
    write_json(paths["index_manifest"], {
        "corpus_sha256": digest(paths["passages"].read_bytes()),
        "vectors_sha256": digest(paths["vectors"].read_bytes()),
        "passage_ids": [row["passage_id"] for row in rows],
        "dimension": 2, "embedding_backend": "fixture_no_network"})
    return paths


def _ready_for_publish(module, monkeypatch):
    _lower_only_size_gates(module, monkeypatch)
    _stage(module)
    paths = _fake_index(module)
    result = module.audit(require_index=True)
    assert result.get("errors") == []
    assert result["index_verified"]
    return paths


def test_production_size_and_child_length_requirements_are_not_relaxed(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    assert module.MIN_PASSAGES == 10_000
    assert module.MIN_WORDS == 100_000
    assert module.MIN_CHILD_WORDS == 80
    assert module.MAX_CHILD_WORDS == 160


def test_snapshot_is_idempotent_and_frozen(isolated_resegmentation):
    module, data, _, _ = isolated_resegmentation
    live_before = _live_bytes(data)
    first = module.snapshot()
    assert module.snapshot() == first
    assert _live_bytes(data) == live_before
    baseline = module.ROOT / "baseline/passages.jsonl"
    assert baseline.read_bytes() == live_before["passages.jsonl"]
    baseline.write_bytes(baseline.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        module.snapshot()


@pytest.mark.parametrize("name", ["passages.jsonl", "sources.jsonl", "documents.jsonl",
    "annotations/codex.jsonl", "annotations/passage_labels.jsonl", "annotations/passage_review.jsonl"])
def test_baseline_detects_any_live_corpus_source_or_label_drift(isolated_resegmentation, name):
    module, data, _, _ = isolated_resegmentation
    module.snapshot()
    module.verify_baseline()
    path = data / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        module.verify_baseline()


@pytest.mark.parametrize("damage", ["changed", "removed"])
def test_baseline_checks_actual_raw_bytes_not_only_document_hash_field(isolated_resegmentation, damage):
    module, data, _, _ = isolated_resegmentation
    module.snapshot()
    path = data / read_jsonl(data / "documents.jsonl")[0]["raw_path"]
    if damage == "changed":
        path.write_bytes(b"raw corruption")
    else:
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        module.verify_baseline()


def test_stage_preserves_complete_parent_partition_and_word_accounting(isolated_resegmentation):
    module, data, original, _ = isolated_resegmentation
    before = _live_bytes(data)
    paths = _stage(module)
    active = read_jsonl(paths["passages"])
    quarantine = read_jsonl(paths["quarantine"])
    assert len(active) == 4
    assert len(quarantine) == 2
    assert {row["parent"]["passage_id"] for row in quarantine} == {"parent-near", "parent-near-b"}
    assert not {row["passage_id"] for row in original} & {row["passage_id"] for row in active}
    assert sum(len(words(row["text"])) for row in original) == (
        sum(len(words(row["text"])) for row in active) +
        sum(len(words(row["parent"]["text"])) for row in quarantine))
    assert all(80 <= len(words(row["text"])) <= 160 for row in active)
    assert len({digest(" ".join(words(row["text"]))) for row in active}) == len(active)
    assert all(not {"labels", "label", "answer_points", "answerability", "model_verification"} & set(row)
               for row in active)
    assert _live_bytes(data) == before
    result = module.audit()
    assert not result["errors"]


def test_children_are_exact_nonoverlapping_substrings_with_original_metadata(isolated_resegmentation):
    module, _, original, _ = isolated_resegmentation
    paths = _stage(module)
    parents = {row["passage_id"]: row for row in original}
    grouped = {}
    child_fields = {"parent_passage_id", "char_start", "char_end", "segmentation"}
    for child in read_jsonl(paths["passages"]):
        parent = parents[child["parent_passage_id"]]
        assert child["text"] == parent["text"][child["char_start"]:child["char_end"]]
        assert {key: value for key, value in child.items()
                if key not in child_fields | {"passage_id", "text", "word_count"}} == {
                    key: value for key, value in parent.items()
                    if key not in {"passage_id", "text", "word_count"}}
        assert child["segmentation"]["minimum_words"] == 80
        assert child["segmentation"]["maximum_words"] == 160
        assert child["segmentation"]["overlap_words"] == 0
        grouped.setdefault(parent["passage_id"], []).append(child)
    for parent_id, children in grouped.items():
        children.sort(key=lambda child: child["char_start"])
        parent_text = parents[parent_id]["text"]
        assert children[0]["char_start"] == 0
        assert children[-1]["char_end"] == len(parent_text)
        for left, right in zip(children, children[1:]):
            assert left["char_end"] <= right["char_start"]
            assert not parent_text[left["char_end"]:right["char_start"]].strip()
        assert words(" ".join(child["text"] for child in children)) == words(parent_text)


@pytest.mark.parametrize("key,value", [("char_start", 1), ("char_end", 1),
    ("parent_passage_id", "wrong-parent")])
def test_audit_rejects_invalid_exact_parent_offsets_or_reference(isolated_resegmentation, key, value):
    module, _, _, _ = isolated_resegmentation
    paths = _stage(module)
    rows = read_jsonl(paths["passages"])
    rows[0][key] = value
    write_jsonl(paths["passages"], rows)
    _audit_rejects(module)


def test_unsplit_parent_is_retained_exactly_and_not_relabelled(isolated_resegmentation):
    module, data, original, _ = isolated_resegmentation
    rows = read_jsonl(data / "passages.jsonl")
    rows[1]["text"] = rows[1]["text"].split(" Petroleum exports", 1)[0]
    rows[1]["word_count"] = len(words(rows[1]["text"]))
    assert 80 <= rows[1]["word_count"] < 160
    write_jsonl(data / "passages.jsonl", rows)
    expansion_report = _read(data / "reports/corpus_expansion.json")
    expansion_report["corpus_sha256"] = digest((data / "passages.jsonl").read_bytes())
    write_json(data / "reports/corpus_expansion.json", expansion_report)
    paths = _stage(module)
    active = read_jsonl(paths["passages"])
    assert [row for row in active if row["passage_id"] == "parent-b"] == [rows[1]]
    assert sum(row["word_count"] for row in rows) == (
        sum(row["word_count"] for row in active) +
        sum(row["parent"]["word_count"] for row in read_jsonl(paths["quarantine"])))
    assert not module.audit()["errors"]


@pytest.mark.parametrize("field,value", [("text", "Fabricated crude oil evidence."),
    ("word_count", 999), ("available_by", "2026-10-06"),
    ("published_date", "2024-01-01"), ("page", 99), ("regions", ["Mars"]),
    ("document_id", "unrelated-document"), ("url", "https://example.test/fabricated"),
    ("labels", {"answerability": "sufficient"})])
def test_audit_rejects_modified_child_text_metadata_counts_or_labels(isolated_resegmentation, field, value):
    module, _, _, _ = isolated_resegmentation
    paths = _stage(module)
    rows = read_jsonl(paths["passages"])
    rows[0][field] = value
    write_jsonl(paths["passages"], rows)
    _audit_rejects(module)


def test_audit_rejects_same_tokens_with_changed_exact_source_text(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    paths = _stage(module)
    rows = read_jsonl(paths["passages"])
    before = words(rows[0]["text"])
    rows[0]["text"] = rows[0]["text"].replace("Crude", "CRUDE", 1)
    assert words(rows[0]["text"]) == before
    write_jsonl(paths["passages"], rows)
    _audit_rejects(module)


def test_audit_rejects_parent_plus_child_double_counting(isolated_resegmentation):
    module, _, original, _ = isolated_resegmentation
    paths = _stage(module)
    write_jsonl(paths["passages"], read_jsonl(paths["passages"]) + [original[0]])
    _audit_rejects(module)


def test_audit_rejects_lost_parent_words_hidden_as_arbitrary_quarantine(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    paths = _stage(module)
    rows = read_jsonl(paths["quarantine"])
    rows[0]["reason"] = "arbitrary_discard_to_improve_metrics"
    write_jsonl(paths["quarantine"], rows)
    _audit_rejects(module)


def test_audit_rejects_missing_child_or_duplicate_row(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    paths = _stage(module)
    rows = read_jsonl(paths["passages"])
    write_jsonl(paths["passages"], [rows[0], rows[0], *rows[2:]])
    _audit_rejects(module)


@pytest.mark.parametrize("requirement", ["passages", "words"])
def test_publish_refuses_small_corpus_despite_valid_index_and_audit(isolated_resegmentation, requirement):
    module, data, _, monkeypatch = isolated_resegmentation
    _ready_for_publish(module, monkeypatch)
    monkeypatch.setattr(module, "MIN_PASSAGES", 10000 if requirement == "passages" else 4)
    monkeypatch.setattr(module, "MIN_WORDS", 100000 if requirement == "words" else 1)
    before = _live_bytes(data)
    with pytest.raises((ValueError, RuntimeError), match="(?i)size|requirement|10000|100000|threshold|minimum|count"):
        module.publish()
    assert _live_bytes(data) == before


@pytest.mark.parametrize("damage", ["changed", "order", "shape", "nonfinite", "manifest-hash"])
def test_index_guard_reads_actual_vectors_and_order(isolated_resegmentation, damage):
    module, _, _, monkeypatch = isolated_resegmentation
    _lower_only_size_gates(module, monkeypatch)
    _stage(module)
    paths = _fake_index(module)
    module.verify_index()
    if damage == "changed":
        paths["vectors"].write_bytes(paths["vectors"].read_bytes() + b"corruption")
    elif damage in {"shape", "nonfinite"}:
        matrix = np.load(paths["vectors"], allow_pickle=False)
        matrix = matrix[:-1] if damage == "shape" else np.full_like(matrix, np.nan)
        np.save(paths["vectors"], matrix, allow_pickle=False)
        manifest = _read(paths["index_manifest"])
        manifest["vectors_sha256"] = digest(paths["vectors"].read_bytes())
        write_json(paths["index_manifest"], manifest)
    else:
        manifest = _read(paths["index_manifest"])
        if damage == "order":
            manifest["passage_ids"] = list(reversed(manifest["passage_ids"]))
        else:
            manifest["corpus_sha256"] = "wrong-fixture-hash"
        write_json(paths["index_manifest"], manifest)
    with pytest.raises((ValueError, RuntimeError)):
        module.verify_index()


@pytest.mark.parametrize("target", ["passages", "quarantine", "report", "index_manifest", "vectors"])
def test_publication_requires_fresh_audit_after_any_staged_artifact_changes(isolated_resegmentation, target):
    module, data, _, monkeypatch = isolated_resegmentation
    paths = _ready_for_publish(module, monkeypatch)
    before = _live_bytes(data)
    paths[target].write_bytes(paths[target].read_bytes() + b"\n")
    with pytest.raises((ValueError, RuntimeError)):
        module.require_fresh_audit()
    with pytest.raises((ValueError, RuntimeError)):
        module.publish()
    assert _live_bytes(data) == before


def test_implementation_change_invalidates_audit_and_publication(isolated_resegmentation):
    module, data, _, monkeypatch = isolated_resegmentation
    _ready_for_publish(module, monkeypatch)
    before = _live_bytes(data)
    path = data.parent / "guard_implementation.py"
    path.write_text("# changed implementation\n", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        module.require_fresh_audit()
    with pytest.raises((ValueError, RuntimeError)):
        module.publish()
    assert _live_bytes(data) == before


@pytest.mark.parametrize("target", ["passages.jsonl", "sources.jsonl", "documents.jsonl",
    "annotations/codex.jsonl", "annotations/passage_labels.jsonl", "annotations/passage_review.jsonl",
    "raw/document-a.html"])
def test_publication_fails_closed_if_live_baseline_was_changed(isolated_resegmentation, target):
    module, data, _, monkeypatch = isolated_resegmentation
    _ready_for_publish(module, monkeypatch)
    path = data / target
    path.write_bytes(path.read_bytes() + b"\n")
    before = _live_bytes(data)
    with pytest.raises((ValueError, RuntimeError)):
        module.publish()
    assert _live_bytes(data) == before


@pytest.mark.parametrize("unsafe", ["../outside/file", "../../private.json", "D:\\outside\\file"])
def test_provenance_paths_cannot_escape_fixture_data(isolated_resegmentation, unsafe):
    module, _, _, _ = isolated_resegmentation
    with pytest.raises(ValueError, match="escapes data"):
        module.within_data(unsafe)


def test_original_annotation_archive_is_checked_independently(isolated_resegmentation):
    module, data, _, _ = isolated_resegmentation
    module.snapshot()
    archive = data / "expansion_runs/original/baseline/passages.jsonl"
    archive.write_bytes(archive.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="annotation archive"):
        module.verify_baseline()


def test_shared_corpus_writer_lock_is_exclusive_and_foreign_lock_not_removed(isolated_resegmentation):
    module, data, _, _ = isolated_resegmentation
    lock = data / ".corpus_writer.lock"
    with module.pipeline_lock():
        assert _read(lock)["run"] == str(module.ROOT)
        with pytest.raises(RuntimeError):
            with module.pipeline_lock():
                pytest.fail("Concurrent corpus writer acquired the same lock")
    assert not lock.exists()
    lock.write_text('{"pid":0,"run":"foreign"}', encoding="utf-8")
    before = lock.read_bytes()
    with pytest.raises(RuntimeError):
        with module.pipeline_lock():
            pytest.fail("An interrupted or foreign writer must fail closed")
    assert lock.read_bytes() == before


def test_publication_recomputes_evidence_even_if_audit_file_claims_success(isolated_resegmentation):
    module, data, _, monkeypatch = isolated_resegmentation
    paths = _ready_for_publish(module, monkeypatch)
    before = _live_bytes(data)
    rows = read_jsonl(paths["passages"])
    rows[0]["char_start"] = 1  # Actual source slice no longer equals child text.
    write_jsonl(paths["passages"], rows)
    current_hash = digest(paths["passages"].read_bytes())
    report = _read(paths["report"])
    report["corpus_sha256"] = current_hash
    write_json(paths["report"], report)
    manifest = _read(paths["index_manifest"])
    manifest["corpus_sha256"] = current_hash
    write_json(paths["index_manifest"], manifest)
    audit = _read(paths["audit"])
    audit["corpus_sha256"] = current_hash
    audit["stage_hashes"] = module.stage_hashes(True)
    audit["errors"] = []
    audit["index_verified"] = True
    write_json(paths["audit"], audit)
    # File freshness is valid. The independent semantic/provenance gate must
    # still reject this modified child rather than trust success booleans.
    module.require_fresh_audit()
    with pytest.raises(ValueError, match="exact original substring"):
        module.publish()
    assert _live_bytes(data) == before


def test_successful_isolated_publication_updates_index_but_preserves_annotations(isolated_resegmentation):
    module, data, _, monkeypatch = isolated_resegmentation
    paths = _ready_for_publish(module, monkeypatch)
    before = _live_bytes(data)
    staged = paths["passages"].read_bytes()
    output = module.publish()
    assert (data / "passages.jsonl").read_bytes() == staged
    assert (data / "index/vectors.npy").read_bytes() == paths["vectors"].read_bytes()
    assert (data / "index/manifest.json").read_bytes() == paths["index_manifest"].read_bytes()
    assert output["stage"] == "published_with_rebuilt_index"
    frozen = data / output["published_snapshot"]
    assert frozen.is_relative_to(data)
    assert (frozen / "passages.jsonl").read_bytes() == staged
    assert (module.ROOT / "baseline/passages.jsonl").read_bytes() == before["passages.jsonl"]
    for name in ("documents.jsonl", "sources.jsonl", "annotations/codex.jsonl",
                 "annotations/passage_labels.jsonl", "annotations/passage_review.jsonl"):
        assert (data / name).read_bytes() == before[name]
    journal = _read(module.ROOT / "publication_transaction/journal.json")
    assert journal["state"] == "committed"
    assert set(journal["replaced"]) == set(journal["files"])
    assert _read(data / "reports/corpus_resegmentation.json") == output


def test_conflicting_frozen_publication_never_overwrites_existing_bytes(isolated_resegmentation):
    module, data, _, monkeypatch = isolated_resegmentation
    paths = _ready_for_publish(module, monkeypatch)
    before = _live_bytes(data)
    frozen = module.ROOT / "published" / _read(paths["report"])["corpus_sha256"] / "passages.jsonl"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b"immutable other release\n")
    with pytest.raises(ValueError, match="Frozen publication cannot be overwritten"):
        module.publish()
    assert frozen.read_bytes() == b"immutable other release\n"
    assert _live_bytes(data) == before


def test_existing_publication_journal_requires_explicit_recovery_review(isolated_resegmentation):
    module, data, _, monkeypatch = isolated_resegmentation
    _ready_for_publish(module, monkeypatch)
    before = _live_bytes(data)
    transaction = module.ROOT / "publication_transaction"
    transaction.mkdir()
    write_json(transaction / "journal.json", {"state": "committing", "files": ["passages.jsonl"]})
    journal = (transaction / "journal.json").read_bytes()
    with pytest.raises(ValueError, match="explicit recovery review"):
        module.publish()
    assert _live_bytes(data) == before
    assert (transaction / "journal.json").read_bytes() == journal


def test_mid_commit_failure_rolls_back_every_live_file_and_keeps_journal(isolated_resegmentation):
    module, data, _, monkeypatch = isolated_resegmentation
    _ready_for_publish(module, monkeypatch)
    before = _live_bytes(data)
    corpus_report_before = (data / "reports/corpus.json").read_bytes()
    replace = Path.replace
    failure = []

    def fail_once(path, destination):
        if Path(destination) == data / "index/manifest.json" and not failure:
            failure.append(True)
            raise OSError("isolated midway commit failure")
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(OSError, match="midway commit failure"):
        module.publish()
    assert failure == [True]
    assert _live_bytes(data) == before
    assert (data / "reports/corpus.json").read_bytes() == corpus_report_before
    assert not (data / "reports/corpus_resegmentation.json").exists()
    assert not (data / "reports/resegmentation_quality.json").exists()
    transaction = module.ROOT / "publication_transaction"
    journal = _read(transaction / "journal.json")
    assert journal["state"] == "rolled_back"
    assert journal["replaced"]
    for name, expected in journal["old_hashes"].items():
        if expected is not None:
            assert digest((data / name).read_bytes()) == expected


def test_prepare_failure_changes_no_live_file_and_leaves_recoverable_backups(isolated_resegmentation):
    module, data, _, monkeypatch = isolated_resegmentation
    _ready_for_publish(module, monkeypatch)
    before = _live_bytes(data)
    copy = module.shutil.copy2

    def fail_preparing(source, destination, *args, **kwargs):
        target = Path(destination)
        if target == module.ROOT / "publication_transaction/new/index/manifest.json":
            raise OSError("isolated preparation copy failure")
        return copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "copy2", fail_preparing)
    with pytest.raises(OSError, match="preparation copy failure"):
        module.publish()
    assert _live_bytes(data) == before
    assert (module.ROOT / "publication_transaction/old/passages.jsonl").read_bytes() == before["passages.jsonl"]
    journal = _read(module.ROOT / "publication_transaction/journal.json")
    assert journal["state"] == "preparing"
    assert journal["replaced"] == []
    assert journal["old_hashes"]["passages.jsonl"] == digest(before["passages.jsonl"])


def test_us_abbreviation_before_lowercase_crude_oil_is_not_scrambled_text(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    parent = _passage("abbreviation-parent", "abbreviation-document", 17)
    parent["text"] = parent["text"].replace("Crude oil production", "U.S. crude oil production", 1)
    parent["word_count"] = len(words(parent["text"]))
    assert module.child_quality(parent["text"]) is None
    children, reason = module.split_parent(parent)
    assert reason is None
    assert len(children) == 2
    assert children[0]["text"].startswith("U.S. crude oil production")


def test_us_abbreviation_at_final_parent_edge_is_still_rejected(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    parent = _passage("incomplete-parent", "incomplete-document", 18)
    parent["text"] = parent["text"].rstrip(".") + " in the U.S."
    parent["word_count"] = len(words(parent["text"]))
    assert module.child_quality(parent["text"]) == "abbreviation_at_incomplete_final_edge"
    children, reason = module.split_parent(parent)
    assert children is None
    assert reason == "parent_incomplete_abbreviation_edge"


def test_dollars_per_barrel_unit_at_sentence_end_is_not_incomplete_abbreviation(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    parent = _passage("price-parent", "price-document", 19)
    parent["text"] = parent["text"].rstrip(".") + " at $72/b."
    parent["word_count"] = len(words(parent["text"]))
    assert module.child_quality(parent["text"]) is None
    children, reason = module.split_parent(parent)
    assert reason is None
    assert len(children) == 2
    assert children[-1]["text"].endswith("$72/b.")
    assert words(" ".join(child["text"] for child in children)) == words(parent["text"])


def test_changed_implementation_can_restage_without_changing_frozen_inputs(isolated_resegmentation):
    module, data, _, _ = isolated_resegmentation
    _stage(module)
    baseline_before = (module.ROOT / "baseline/manifest.json").read_bytes()
    live_before = _live_bytes(data)
    implementation = data.parent / "guard_implementation.py"
    implementation.write_text("# revised safe splitting gates\n", encoding="utf-8")
    module.verify_baseline()
    module.stage()
    report = _read(module.stage_paths()["report"])
    assert report["implementation_hashes"] == module.implementation_hashes()
    assert module.audit()["errors"] == []
    assert (module.ROOT / "baseline/manifest.json").read_bytes() == baseline_before
    assert _live_bytes(data) == live_before


def test_changed_implementation_requires_restage_before_new_audit(isolated_resegmentation):
    module, data, _, _ = isolated_resegmentation
    _stage(module)
    implementation = data.parent / "guard_implementation.py"
    implementation.write_text("# changed splitting gates without restaging\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)implementation|stage"):
        module.audit()


def test_bibliography_heavy_text_is_not_factual_evidence(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    bibliography = (
        "References: EIA petroleum report, https://www.eia.gov/oil/report-a. "
        "Crude oil statistical yearbook, https://www.eia.gov/oil/report-b. "
        "Accessed January 2025. Bibliography: oil market tables, "
        "https://www.eia.gov/oil/report-c.")
    assert module.child_quality(bibliography) is not None


def test_factual_sentence_with_source_url_is_not_a_pure_reference_list(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    sentence = _passage("url-parent", "url-document", 21)["text"].split(" Petroleum exports", 1)[0]
    sentence += " Source: https://www.eia.gov/analysis/oil-report.pdf."
    assert module.child_quality(sentence) is None


@pytest.mark.parametrize("field,value", [
    ("labels_coverage", 4500), ("archived_machine_labels", 9000),
    ("replaced_parents", 99), ("child_passages", 99),
    ("source_passages_path", "expansion_runs/unrelated/passages.jsonl"),
    ("source_passages_sha256", "wrong-source-corpus-sha"),
    ("annotation_baseline_sha256", "wrong-annotation-sha"),
    ("by_source", {"fabricated-source": 4}), ("documents", 999),
    ("unique_words", 999999), ("pending_near_duplicate_parents_excluded", 0)])
def test_audit_recomputes_report_facts_instead_of_trusting_claims(isolated_resegmentation, field, value):
    module, _, _, _ = isolated_resegmentation
    paths = _stage(module)
    report = _read(paths["report"])
    assert report[field] != value
    report[field] = value
    write_json(paths["report"], report)
    _audit_rejects(module)


def test_st_james_name_is_not_a_sentence_boundary_or_incomplete_unit(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    parent = _passage("saint-parent", "saint-document", 23)
    parent["text"] = parent["text"].replace("in field", "near St. James field", 1)
    parent["word_count"] = len(words(parent["text"]))
    assert module.child_quality(parent["text"]) is None
    children, reason = module.split_parent(parent)
    assert reason is None
    assert len(children) == 2
    assert "St. James" in children[0]["text"]
    assert all(not child["text"].endswith("St.") for child in children)


def test_st_abbreviation_at_incomplete_final_edge_is_rejected(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    parent = _passage("saint-incomplete", "saint-incomplete-document", 24)
    parent["text"] = parent["text"].rstrip(".") + " near St."
    parent["word_count"] = len(words(parent["text"]))
    assert module.child_quality(parent["text"]) == "abbreviation_at_incomplete_final_edge"
    assert module.split_parent(parent) == (None, "parent_incomplete_abbreviation_edge")


def test_generic_sanctions_without_mineral_oil_anchor_are_not_oil_evidence(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    text = "Sanctions affecting diplomatic relations in Syria were revised after negotiations."
    assert module.child_quality(text) == "no_mineral_oil_evidence"


def test_yet_it_opening_requires_parent_context_and_is_not_promoted(isolated_resegmentation):
    module, _, _, _ = isolated_resegmentation
    text = "Yet it reduced crude oil production after the pipeline closure."
    assert module.child_quality(text) == "dependent_opening"


@pytest.mark.parametrize("damage", ["stale-corpus", "active-critical"])
def test_optional_semantic_report_must_match_active_version_and_have_no_critical_children(isolated_resegmentation, damage):
    module, _, _, _ = isolated_resegmentation
    paths = _stage(module)
    semantic = module.ROOT / "stage/reports/semantic_spotcheck.json"
    write_json(semantic, {
        "checked_corpus_sha256": ("different-corpus-hash" if damage == "stale-corpus"
            else digest(paths["passages"].read_bytes())),
        "critical_active_child_ids": ([read_jsonl(paths["passages"])[0]["passage_id"]]
            if damage == "active-critical" else [])})
    with pytest.raises(ValueError, match="Semantic spotcheck"):
        module.audit()


def test_semantic_report_is_bound_to_audit_and_preserved_in_frozen_publication(isolated_resegmentation):
    module, data, _, monkeypatch = isolated_resegmentation
    paths = _ready_for_publish(module, monkeypatch)
    semantic = module.ROOT / "stage/reports/semantic_spotcheck.json"
    write_json(semantic, {
        "checked_corpus_sha256": digest(paths["passages"].read_bytes()),
        "critical_active_child_ids": [], "provenance": "model_only_not_human"})
    with pytest.raises(ValueError, match="stale"):
        module.require_fresh_audit()
    result = module.audit(require_index=True)
    assert result["stage_hashes"]["semantic_spotcheck"] == digest(semantic.read_bytes())
    frozen_bytes = semantic.read_bytes()
    output = module.publish()
    assert (data / output["published_snapshot"] / "reports/semantic_spotcheck.json").read_bytes() == frozen_bytes


def test_semantic_report_changed_after_audit_invalidates_publication(isolated_resegmentation):
    module, data, _, monkeypatch = isolated_resegmentation
    paths = _ready_for_publish(module, monkeypatch)
    semantic = module.ROOT / "stage/reports/semantic_spotcheck.json"
    write_json(semantic, {
        "checked_corpus_sha256": digest(paths["passages"].read_bytes()),
        "critical_active_child_ids": []})
    module.audit(require_index=True)
    before = _live_bytes(data)
    semantic.write_bytes(semantic.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="stale"):
        module.publish()
    assert _live_bytes(data) == before
