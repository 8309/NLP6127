import json

import pytest

from oil_evidence.corpus_versions import annotation_passages, active_label_coverage, active_corpus_version, original_parent
from oil_evidence.store import digest


def rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return digest(path.read_bytes())


def report(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def versions(tmp_path):
    original = dict(passage_id="old", document_id="document", page=3, source="EIA", url="https://www.eia.gov/report",
                    published_date="2020-01-01", available_by="2020-01-01", date_precision="day", date_source="cover",
                    text="Oil supplies declined. Stocks also declined.", word_count=7)
    child = dict(original, passage_id="child", parent_passage_id="old", text="Oil supplies declined.", word_count=3)
    original_path = tmp_path / "expansion_runs/first/baseline/passages.jsonl"
    original_hash = rows(original_path, [original])
    source_path = tmp_path / "resegmentation_runs/semantic/baseline/passages.jsonl"
    source_hash = rows(source_path, [original])
    live_hash = rows(tmp_path / "passages.jsonl", [child])
    report(tmp_path / "reports/passage_annotation.json", {"corpus_sha256": original_hash})
    manifest = dict(corpus_sha256=live_hash, source_corpus_sha256=source_hash,
                    source_passages_path=str(source_path.relative_to(tmp_path)), source_passages_sha256=source_hash,
                    annotation_baseline_path=str(original_path.relative_to(tmp_path)), annotation_baseline_sha256=original_hash,
                    source_passages=1, passages=1, labels_coverage=0, archived_machine_labels=1, unlabelled_corpus_passages=1)
    report(tmp_path / "reports/corpus_resegmentation.json", manifest)
    return tmp_path, original, child, manifest


def test_resegmentation_keeps_annotation_text_and_does_not_relabel_children(versions):
    data, original, child, _ = versions
    assert annotation_passages(data) == [original]
    assert active_label_coverage([{"passage_id": "old"}], data) == 0
    with pytest.raises(ValueError, match="frozen annotation"):
        active_label_coverage([{"passage_id": "child"}], data)
    assert original_parent(dict(child, rank=1, score=0.2), data) == original


def test_initial_corpus_without_annotation_report(tmp_path):
    rows(tmp_path / "passages.jsonl", [dict(passage_id="one", text="Oil")])
    assert annotation_passages(tmp_path)[0]["passage_id"] == "one"


def test_current_annotation_version_needs_no_archive(versions):
    data, original, _, _ = versions
    rows(data / "passages.jsonl", [original])
    assert annotation_passages(data) == [original]
    assert active_label_coverage([{"passage_id": "old"}], data) == 1


def test_additive_expansion_supports_existing_absolute_baseline_paths(versions):
    data, original, child, manifest = versions
    (data / "reports/corpus_resegmentation.json").unlink()
    expanded_hash = rows(data / "passages.jsonl", [original, child])
    report(data / "reports/corpus_expansion.json", dict(corpus_sha256=expanded_hash,
           annotation_baseline_path=str((data / manifest["annotation_baseline_path"]).resolve()),
           annotation_baseline_sha256=manifest["annotation_baseline_sha256"]))
    assert annotation_passages(data) == [original]
    assert active_label_coverage([{"passage_id": "old"}], data) == 1


def test_legacy_expansion_uses_snapshot_baseline(versions):
    data, original, child, manifest = versions
    (data / "reports/corpus_resegmentation.json").unlink()
    expanded_hash = rows(data / "passages.jsonl", [original, child])
    report(data / "reports/corpus_expansion.json", dict(corpus_sha256=expanded_hash,
           snapshot_root=str((data / "expansion_runs/first").resolve()), baseline_sha256=manifest["annotation_baseline_sha256"]))
    assert annotation_passages(data) == [original]


@pytest.mark.parametrize("path", ["../passages.jsonl", "folder/../../passages.jsonl", "..\\passages.jsonl", "C:/outside/passages.jsonl", "C:passages.jsonl", "/outside/passages.jsonl"])
@pytest.mark.parametrize("field", ["annotation_baseline_path", "source_passages_path"])
def test_resegmentation_paths_cannot_escape_data(versions, path, field):
    data, _, _, manifest = versions
    manifest[field] = path
    report(data / "reports/corpus_resegmentation.json", manifest)
    with pytest.raises(ValueError, match="path"):
        annotation_passages(data)


@pytest.mark.parametrize("field", ["corpus_sha256", "source_passages_sha256", "source_corpus_sha256", "annotation_baseline_sha256"])
@pytest.mark.parametrize("bad_hash", [None, "not-a-hash", "0" * 64])
def test_resegmentation_hashes_fail_closed(versions, field, bad_hash):
    data, _, _, manifest = versions
    manifest[field] = bad_hash
    report(data / "reports/corpus_resegmentation.json", manifest)
    with pytest.raises(ValueError):
        annotation_passages(data)


def test_modified_source_parent_hash_rejected(versions):
    data, _, child, manifest = versions
    rows(data / manifest["source_passages_path"], [{"passage_id": "changed", "text": "changed"}])
    with pytest.raises(ValueError, match="hash mismatch"):
        original_parent(child, data)


def test_missing_archive_fails_closed(versions):
    data, _, _, manifest = versions
    (data / manifest["annotation_baseline_path"]).unlink()
    with pytest.raises(ValueError, match="missing"):
        annotation_passages(data)


@pytest.mark.parametrize("value", [[], None, "invalid"])
def test_malformed_version_report(versions, value):
    data, _, _, _ = versions
    report(data / "reports/corpus_resegmentation.json", value)
    with pytest.raises(ValueError, match="object"):
        annotation_passages(data)


def test_invalid_json_report(versions):
    data, _, _, _ = versions
    (data / "reports/corpus_resegmentation.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read"):
        annotation_passages(data)


def test_same_id_with_changed_text_is_not_annotation_coverage(versions):
    data, original, child, manifest = versions
    manifest["corpus_sha256"] = rows(data / "passages.jsonl", [dict(original, text=child["text"])])
    report(data / "reports/corpus_resegmentation.json", manifest)
    with pytest.raises(ValueError, match="reused.*changed"):
        annotation_passages(data)


def test_same_id_with_changed_metadata_is_not_annotation_coverage(versions):
    data, original, _, manifest = versions
    manifest["corpus_sha256"] = rows(data / "passages.jsonl", [dict(original, available_by="2020-02-01")])
    report(data / "reports/corpus_resegmentation.json", manifest)
    with pytest.raises(ValueError, match="reused.*changed"):
        annotation_passages(data)


def test_duplicate_label_ids_fail_closed(versions):
    data, _, _, _ = versions
    with pytest.raises(ValueError, match="uniquely"):
        active_label_coverage([{"passage_id": "old"}, {"passage_id": "old"}], data)


def test_duplicate_source_ids_fail_closed(versions):
    data, original, _, manifest = versions
    new_hash = rows(data / manifest["source_passages_path"], [original, original])
    manifest.update(source_passages_sha256=new_hash, source_corpus_sha256=new_hash)
    report(data / "reports/corpus_resegmentation.json", manifest)
    with pytest.raises(ValueError, match="unique"):
        annotation_passages(data)


def test_incorrect_source_count_fail_closed(versions):
    data, _, _, manifest = versions
    manifest["source_passages"] = 2
    report(data / "reports/corpus_resegmentation.json", manifest)
    with pytest.raises(ValueError, match="count"):
        annotation_passages(data)


def test_source_path_symlink_cannot_escape_data(versions, tmp_path):
    data, _, _, manifest = versions
    outside = tmp_path.parent / (tmp_path.name + "-outside.jsonl")
    outside.write_bytes((data / manifest["source_passages_path"]).read_bytes())
    linked = data / "link.jsonl"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("Host disallows symlink creation")
    manifest["source_passages_path"] = "link.jsonl"
    report(data / "reports/corpus_resegmentation.json", manifest)
    with pytest.raises(ValueError, match="outside DATA"):
        annotation_passages(data)


def test_nonsegmented_passage_has_no_parent(versions):
    data, original, _, _ = versions
    assert original_parent(original, data) is None


@pytest.mark.parametrize("change", [dict(document_id="other"), dict(page=9), dict(available_by="2021-01-01"), dict(text="Unrelated sentence."), dict(parent_passage_id="absent")])
def test_parent_context_rejects_modified_or_unrelated_passage(versions, change):
    data, _, child, _ = versions
    with pytest.raises(ValueError):
        original_parent(dict(child, **change), data)


def test_active_child_must_be_contiguous_within_parent(versions):
    data, _, child, manifest = versions
    changed = dict(child, text="Oil declined. Stocks supplies.")
    manifest["corpus_sha256"] = rows(data / "passages.jsonl", [changed])
    report(data / "reports/corpus_resegmentation.json", manifest)
    with pytest.raises(ValueError, match="contiguous"):
        original_parent(changed, data)


def test_parent_metadata_not_silently_changed(versions):
    data, _, child, manifest = versions
    changed = dict(child, page=7)
    manifest["corpus_sha256"] = rows(data / "passages.jsonl", [changed])
    report(data / "reports/corpus_resegmentation.json", manifest)
    with pytest.raises(ValueError, match="provenance"):
        original_parent(changed, data)


def test_additive_expansion_cannot_change_original_record(versions):
    data, original, child, manifest = versions
    (data / "reports/corpus_resegmentation.json").unlink()
    new_hash = rows(data / "passages.jsonl", [dict(original, text="Changed"), child])
    report(data / "reports/corpus_expansion.json", dict(corpus_sha256=new_hash,
           annotation_baseline_path=manifest["annotation_baseline_path"], annotation_baseline_sha256=manifest["annotation_baseline_sha256"]))
    with pytest.raises(ValueError, match="changed or removed"):
        annotation_passages(data)


@pytest.mark.parametrize("offsets", [dict(char_start=0, char_end=2), dict(char_start=-1, char_end=21), dict(char_start=True, char_end=21), dict(char_start=0)])
def test_incorrect_parent_character_offsets_fail_closed(versions, offsets):
    data, _, child, manifest = versions
    changed = dict(child, **offsets)
    manifest["corpus_sha256"] = rows(data / "passages.jsonl", [changed])
    report(data / "reports/corpus_resegmentation.json", manifest)
    with pytest.raises(ValueError, match="character offsets"):
        original_parent(changed, data)


def test_exact_character_offsets_retain_parent(versions):
    data, original, child, manifest = versions
    changed = dict(child, char_start=0, char_end=len(child["text"]))
    manifest["corpus_sha256"] = rows(data / "passages.jsonl", [changed])
    report(data / "reports/corpus_resegmentation.json", manifest)
    assert original_parent(changed, data) == original


def test_existing_labels_require_annotation_version_manifest(versions):
    data, _, _, _ = versions
    (data / "reports/passage_annotation.json").unlink()
    rows(data / "annotations/passage_labels.jsonl", [{"passage_id": "old"}])
    with pytest.raises(ValueError, match="no trusted annotation"):
        annotation_passages(data)


@pytest.fixture
def unified_versions(versions):
    data, original, _, legacy = versions
    source_page = {key: value for key, value in original.items() if key not in ["passage_id", "word_count"]}
    source_page.update(source_page_id="document-page-3-cleaned", regions=["Europe", "Asia"])
    child = dict(original, passage_id="unified-child", source_page_id=source_page["source_page_id"],
                 text="Oil supplies declined.", word_count=3, regions=["Europe"],
                 char_start=0, char_end=len("Oil supplies declined."),
                 segmentation={"version": "semantic-page-v1", "overlap_words": 0})
    source_path = data / "unified_runs/semantic-page-v1/published/version/source_pages.jsonl"
    source_hash = rows(source_path, [source_page])
    live_hash = rows(data / "passages.jsonl", [child])
    manifest = dict(version="semantic-page-v1", corpus_sha256=live_hash,
                    source_pages_path=str(source_path.relative_to(data)), source_pages_sha256=source_hash,
                    baseline_passages_path=legacy["source_passages_path"],
                    baseline_passages_sha256=legacy["source_passages_sha256"],
                    annotation_baseline_path=legacy["annotation_baseline_path"],
                    annotation_baseline_sha256=legacy["annotation_baseline_sha256"],
                    passages=1, source_pages=1, labels_coverage=0,
                    archived_machine_labels=1, unlabelled_corpus_passages=1)
    report(data / "reports/corpus_unified.json", manifest)
    return data, original, source_page, child, manifest


def update_unified_child(fixture, **changes):
    data, _, _, child, manifest = fixture
    changed = dict(child, **changes)
    manifest["corpus_sha256"] = rows(data / "passages.jsonl", [changed])
    report(data / "reports/corpus_unified.json", manifest)
    return changed


def test_unified_version_precedes_stale_resegmentation_and_preserves_annotation_archive(unified_versions):
    data, original, source_page, child, manifest = unified_versions
    assert active_corpus_version(data) == manifest
    assert annotation_passages(data) == [original]
    assert active_label_coverage([{"passage_id": "old"}], data) == 0
    with pytest.raises(ValueError, match="frozen annotation"):
        active_label_coverage([{"passage_id": child["passage_id"]}], data)
    assert original_parent(dict(child, rank=1, score=0.2), data) == source_page
    assert child["regions"] != source_page["regions"]  # local regions must not be inherited


def test_source_page_context_is_not_an_active_passage_or_label(unified_versions):
    data, _, source_page, child, _ = unified_versions
    assert "passage_id" not in original_parent(child, data)
    assert source_page["source_page_id"] not in {p["passage_id"] for p in annotation_passages(data)}
    with pytest.raises(ValueError, match="frozen annotation"):
        active_label_coverage([{"passage_id": source_page["source_page_id"]}], data)


@pytest.mark.parametrize("field", ["source_pages_path", "baseline_passages_path", "annotation_baseline_path"])
@pytest.mark.parametrize("path", ["../passages.jsonl", "..\\passages.jsonl", "C:/outside/file.jsonl", "/outside/file.jsonl"])
def test_unified_paths_cannot_escape_data(unified_versions, field, path):
    data, _, _, _, manifest = unified_versions
    manifest[field] = path
    report(data / "reports/corpus_unified.json", manifest)
    with pytest.raises(ValueError, match="path"):
        annotation_passages(data)


@pytest.mark.parametrize("field", ["corpus_sha256", "source_pages_sha256", "baseline_passages_sha256", "annotation_baseline_sha256"])
@pytest.mark.parametrize("bad_hash", [None, "invalid", "0" * 64])
def test_unified_hashes_fail_closed(unified_versions, field, bad_hash):
    data, _, _, _, manifest = unified_versions
    manifest[field] = bad_hash
    report(data / "reports/corpus_unified.json", manifest)
    with pytest.raises(ValueError):
        annotation_passages(data)


@pytest.mark.parametrize("field", ["source_pages_path", "baseline_passages_path", "annotation_baseline_path"])
def test_unified_missing_immutable_input_fails_closed(unified_versions, field):
    data, _, _, _, manifest = unified_versions
    (data / manifest[field]).unlink()
    with pytest.raises(ValueError, match="missing"):
        annotation_passages(data)


def test_unified_source_pages_cannot_duplicate_ids(unified_versions):
    data, _, page, _, manifest = unified_versions
    manifest["source_pages_sha256"] = rows(data / manifest["source_pages_path"], [page, page])
    report(data / "reports/corpus_unified.json", manifest)
    with pytest.raises(ValueError, match="unique"):
        annotation_passages(data)


@pytest.mark.parametrize("change", [dict(text=""), dict(text=None), dict(document_id=""), dict(page=True), dict(page=0)])
def test_unified_malformed_source_page_fails_closed(unified_versions, change):
    data, _, page, _, manifest = unified_versions
    manifest["source_pages_sha256"] = rows(data / manifest["source_pages_path"], [dict(page, **change)])
    report(data / "reports/corpus_unified.json", manifest)
    with pytest.raises(ValueError, match="malformed"):
        annotation_passages(data)


@pytest.mark.parametrize("field", ["passages", "source_pages"])
@pytest.mark.parametrize("count", [0, True, "1"])
def test_unified_report_counts_are_verified(unified_versions, field, count):
    data, _, _, _, manifest = unified_versions
    manifest[field] = count
    report(data / "reports/corpus_unified.json", manifest)
    with pytest.raises(ValueError, match="count"):
        annotation_passages(data)


@pytest.mark.parametrize("change", [dict(document_id="different"), dict(page=7), dict(available_by="2020-02-01"),
                                     dict(source="other"), dict(source_page_id="missing"),
                                     dict(char_start=0, char_end=2), dict(char_start=True), dict(char_end=None),
                                     dict(text="Oil stocks declined."), dict(segmentation={"version": "old-policy"})])
def test_unified_active_child_provenance_offsets_and_policy_verified(unified_versions, change):
    data, _, _, _, _ = unified_versions
    changed = update_unified_child(unified_versions, **change)
    with pytest.raises(ValueError):
        annotation_passages(data)
    with pytest.raises(ValueError):
        original_parent(changed, data)


def test_unified_context_rejects_nonactive_or_modified_caller_passage(unified_versions):
    data, _, _, child, _ = unified_versions
    for change in [dict(passage_id="not-active"), dict(regions=["Africa"]), dict(text="Changed")]:
        with pytest.raises(ValueError, match="changed or non-active"):
            original_parent(dict(child, **change), data)


def test_unified_must_not_reuse_old_baseline_passage_id(unified_versions):
    data, original, _, _, _ = unified_versions
    update_unified_child(unified_versions, passage_id=original["passage_id"])
    with pytest.raises(ValueError, match="reuse baseline IDs"):
        annotation_passages(data)


@pytest.mark.parametrize("version", [None, "", "../../bad", True])
def test_unified_version_name_invalid(unified_versions, version):
    data, _, _, _, manifest = unified_versions
    manifest["version"] = version
    report(data / "reports/corpus_unified.json", manifest)
    with pytest.raises(ValueError, match="version is invalid"):
        annotation_passages(data)


def test_unified_passages_require_current_version_report(unified_versions):
    data, _, _, child, _ = unified_versions
    (data / "reports/corpus_unified.json").unlink()
    with pytest.raises(ValueError, match="no trusted version report"):
        annotation_passages(data)
    with pytest.raises(ValueError, match="no trusted version report"):
        original_parent(child, data)


def test_stale_unified_report_does_not_shadow_valid_legacy_version(unified_versions):
    data, original, _, _, _ = unified_versions
    legacy_child = dict(original, passage_id="child", parent_passage_id="old", text="Oil supplies declined.", word_count=3)
    rows(data / "passages.jsonl", [legacy_child])
    assert annotation_passages(data) == [original]
    assert active_corpus_version(data)["source_passages"] == 1
    assert original_parent(legacy_child, data) == original


def test_unified_annotation_hash_must_match_published_label_version(unified_versions):
    data, _, _, _, manifest = unified_versions
    alternate_path = data / "unified_runs/semantic-page-v1/baseline/alternate_annotations.jsonl"
    alternate_hash = rows(alternate_path, [dict(passage_id="unrelated", text="Different oil text.")])
    manifest.update(annotation_baseline_path=str(alternate_path.relative_to(data)), annotation_baseline_sha256=alternate_hash)
    report(data / "reports/corpus_unified.json", manifest)
    with pytest.raises(ValueError, match="does not match the label report"):
        annotation_passages(data)
