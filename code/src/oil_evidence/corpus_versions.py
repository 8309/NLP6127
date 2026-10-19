"""Read-only, fail-closed access to frozen annotation and parent corpus versions.

Old labels belong to their original text, not to newly segmented children.  These
helpers never relabel a passage or attach parent context to generated claims.
"""
from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
import re

from .store import DATA, digest, read_jsonl


def _report(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read corpus version report: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Corpus version report must be an object: {path}")
    return value


def _sha(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid corpus version hash: {field}")
    return value


def _data_path(data, value, *, allow_absolute=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Corpus version path is missing")
    # Reject traversal before resolution, including Windows syntax on other hosts.
    normalized = value.replace("\\", "/")
    if ".." in normalized.split("/"):
        raise ValueError("Corpus version path traversal is not allowed")
    path = Path(value)
    if (path.is_absolute() or PureWindowsPath(value).drive) and not allow_absolute:
        raise ValueError("Corpus version path must be relative to DATA")
    target = (path if path.is_absolute() else data / path).resolve()
    try:
        target.relative_to(data.resolve())
    except ValueError as exc:
        raise ValueError("Corpus version path is outside DATA") from exc
    return target


def _rows(path, expected=None, *, id_field="passage_id"):
    if not path.is_file():
        raise ValueError(f"Frozen corpus version is missing: {path}")
    if expected and digest(path.read_bytes()) != _sha(expected, "passages_sha256"):
        raise ValueError(f"Frozen corpus version hash mismatch: {path}")
    try:
        rows = read_jsonl(path)
        ids = [row[id_field] for row in rows]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Malformed frozen corpus: {path}") from exc
    if not rows or any(not isinstance(pid, str) or not pid for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Frozen corpus must have nonempty unique {id_field} IDs: {path}")
    return rows


_PROVENANCE = ["document_id", "page", "source", "title", "url", "published_date", "available_by",
               "date_precision", "date_source", "event_date", "event_status", "redistribution"]


def _source_page_child(passage, source_page, version):
    """Validate exact source-page ancestry, without inheriting whole-page regions."""
    if passage.get("source_page_id") != source_page["source_page_id"]:
        raise ValueError("Unified passage does not refer to its source page")
    for key in _PROVENANCE:
        if passage.get(key) != source_page.get(key):
            raise ValueError(f"Unified passage changed source page provenance: {key}")
    segmentation = passage.get("segmentation")
    if not isinstance(segmentation, dict) or segmentation.get("version") != version:
        raise ValueError("Unified passage segmentation version is incorrect")
    start, end = passage.get("char_start"), passage.get("char_end")
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source_page["text"])
            or source_page["text"][start:end] != passage.get("text")):
        raise ValueError("Unified passage character offsets do not match source page text")


def _unified(data, active_hash, active, report):
    if _sha(report.get("corpus_sha256"), "corpus_sha256") != active_hash:
        raise ValueError("Unified corpus report is stale for the active corpus")
    version = report.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", version):
        raise ValueError("Unified corpus version is invalid")
    source_pages = _rows(_data_path(data, report.get("source_pages_path")),
                         _sha(report.get("source_pages_sha256"), "source_pages_sha256"),
                         id_field="source_page_id")
    for page in source_pages:
        if (not isinstance(page.get("text"), str) or not page["text"].strip()
                or not isinstance(page.get("document_id"), str) or not page["document_id"]
                or type(page.get("page")) is not int or page["page"] < 1):
            raise ValueError("Unified source page has malformed text or provenance")
    pages_by_id = {p["source_page_id"]: p for p in source_pages}
    for passage in active:
        source_page = pages_by_id.get(passage.get("source_page_id"))
        if source_page is None:
            raise ValueError("Unified passage source page is missing")
        _source_page_child(passage, source_page, version)
    baseline = _rows(_data_path(data, report.get("baseline_passages_path")),
                     _sha(report.get("baseline_passages_sha256"), "baseline_passages_sha256"))
    if {p["passage_id"] for p in active} & {p["passage_id"] for p in baseline}:
        raise ValueError("Unified corpus must use new passage IDs, not reuse baseline IDs")
    annotated = _rows(_data_path(data, report.get("annotation_baseline_path")),
                      _sha(report.get("annotation_baseline_sha256"), "annotation_baseline_sha256"))
    if {p["passage_id"] for p in active} & {p["passage_id"] for p in annotated}:
        raise ValueError("Unified corpus reused an annotation passage ID")
    for field, count in [("passages", len(active)), ("source_pages", len(source_pages))]:
        if field in report and (type(report[field]) is not int or report[field] != count):
            raise ValueError(f"Unified corpus {field} count is incorrect")
    return report


def _active_unified(data, active_hash, active):
    """A matching unified version takes precedence over stale legacy reports."""
    path = data / "reports/corpus_unified.json"
    if path.exists():
        report = _report(path)
        report_hash = _sha(report.get("corpus_sha256"), "corpus_sha256")
        if report_hash == active_hash:
            return _unified(data, active_hash, active, report)
        if any(p.get("source_page_id") for p in active):
            raise ValueError("Unified corpus report is stale for the active corpus")
    elif any(p.get("source_page_id") for p in active):
        raise ValueError("Unified corpus passages have no trusted version report")
    return None


def active_corpus_version(data_dir=None):
    """Return a verified active segmentation report, never a stale legacy one."""
    data = Path(data_dir or DATA)
    path = data / "passages.jsonl"
    active = _rows(path)
    active_hash = digest(path.read_bytes())
    unified = _active_unified(data, active_hash, active)
    if unified is not None:
        return unified
    if (data / "reports/corpus_resegmentation.json").exists():
        return _resegmentation(data, active_hash)
    return None


def _resegmentation(data, active_hash):
    report = _report(data / "reports/corpus_resegmentation.json")
    if _sha(report.get("corpus_sha256"), "corpus_sha256") != active_hash:
        raise ValueError("Resegmentation report is stale for the active corpus")
    source_hash = _sha(report.get("source_passages_sha256"), "source_passages_sha256")
    if _sha(report.get("source_corpus_sha256"), "source_corpus_sha256") != source_hash:
        raise ValueError("Resegmentation source hashes disagree")
    source = _rows(_data_path(data, report.get("source_passages_path")), source_hash)
    if type(report.get("source_passages")) is not int or report["source_passages"] != len(source):
        raise ValueError("Resegmentation source passage count is incorrect")
    return report


def annotation_passages(data_dir=None):
    """Return the exact text version to which existing annotation IDs refer.

    An additive expansion must still contain byte-equivalent original records.
    A resegmentation or unified page rebuild returns the hash-verified frozen
    annotation corpus; old IDs are never translated into new passage IDs.
    """
    data = Path(data_dir or DATA)
    live_path = data / "passages.jsonl"
    active = _rows(live_path)
    active_hash = digest(live_path.read_bytes())
    unified = _active_unified(data, active_hash, active)
    annotation_report_path = data / "reports/passage_annotation.json"
    if not annotation_report_path.exists():
        if read_jsonl(data / "annotations/passage_labels.jsonl"):
            raise ValueError("Existing passage labels have no trusted annotation version report")
        return active
    expected = _sha(_report(annotation_report_path).get("corpus_sha256"), "annotation corpus_sha256")
    if active_hash == expected:
        return active
    if unified is not None:
        if _sha(unified.get("annotation_baseline_sha256"), "annotation_baseline_sha256") != expected:
            raise ValueError("Unified annotation version does not match the label report")
        archived = _rows(_data_path(data, unified.get("annotation_baseline_path")), expected)
        if {p["passage_id"] for p in active} & {p["passage_id"] for p in archived}:
            raise ValueError("Unified corpus reused an annotation passage ID")
        return archived
    if (data / "reports/corpus_resegmentation.json").exists():
        report = _resegmentation(data, active_hash)
        if _sha(report.get("annotation_baseline_sha256"), "annotation_baseline_sha256") != expected:
            raise ValueError("Resegmentation annotation version does not match the label report")
        path = _data_path(data, report.get("annotation_baseline_path"))
        archived = _rows(path, expected)
        # A reused ID with changed text/metadata is not valid label coverage.
        active_by_id = {p["passage_id"]: p for p in active}
        if any(p["passage_id"] in active_by_id and active_by_id[p["passage_id"]] != p for p in archived):
            raise ValueError("Resegmentation reused an annotation passage ID with changed content")
        return archived
    report = _report(data / "reports/corpus_expansion.json")
    if _sha(report.get("corpus_sha256"), "corpus_sha256") != active_hash:
        raise ValueError("Expansion report is stale for the active corpus")
    expected_baseline = report.get("annotation_baseline_sha256", report.get("baseline_sha256"))
    if _sha(expected_baseline, "annotation_baseline_sha256") != expected:
        raise ValueError("Expansion annotation version does not match the label report")
    value = report.get("annotation_baseline_path")
    if value is None:
        value = str(_data_path(data, report.get("snapshot_root"), allow_absolute=True) / "baseline/passages.jsonl")
    archived = _rows(_data_path(data, value, allow_absolute=True), expected)
    active_by_id = {p["passage_id"]: p for p in active}
    if any(active_by_id.get(p["passage_id"]) != p for p in archived):
        raise ValueError("Expansion changed or removed frozen annotation passages")
    return archived


def active_label_coverage(labels, data_dir=None):
    """Count only existing labels whose exact archived passage is still active."""
    data = Path(data_dir or DATA)
    original = {p["passage_id"]: p for p in annotation_passages(data)}
    active = {p["passage_id"]: p for p in _rows(data / "passages.jsonl")}
    ids = [row["passage_id"] for row in labels]
    if len(ids) != len(set(ids)) or any(pid not in original for pid in ids):
        raise ValueError("Labels do not refer uniquely to the frozen annotation corpus")
    return sum(active.get(pid) == original[pid] for pid in ids)


def original_parent(passage, data_dir=None):
    """Return verified parent passage or cleaned source page for display only.

    Source-page regions need not match a child's local regions. This context is
    never an extra retrieval result, citation, label mapping, or answer input.
    """
    if passage.get("source_page_id"):
        data = Path(data_dir or DATA)
        active_path = data / "passages.jsonl"
        active = _rows(active_path)
        report = _active_unified(data, digest(active_path.read_bytes()), active)
        if report is None:
            raise ValueError("Unified source-page context has no trusted version report")
        actual = next((row for row in active if row["passage_id"] == passage.get("passage_id")), None)
        if actual is None or any(passage.get(key) != value for key, value in actual.items()):
            raise ValueError("Source page context requested for a changed or non-active passage")
        source = _rows(_data_path(data, report["source_pages_path"]), report["source_pages_sha256"],
                       id_field="source_page_id")
        parent = next((row for row in source if row["source_page_id"] == passage["source_page_id"]), None)
        if parent is None:
            raise ValueError("Unified passage source page is missing")
        _source_page_child(passage, parent, report["version"])
        return parent
    pid = passage.get("parent_passage_id")
    if not pid:
        return None
    data = Path(data_dir or DATA)
    active_hash = digest((data / "passages.jsonl").read_bytes())
    report = _resegmentation(data, active_hash)
    source_path = _data_path(data, report.get("source_passages_path"))
    source = _rows(source_path, report["source_passages_sha256"])
    parent = next((row for row in source if row["passage_id"] == pid), None)
    if parent is None:
        raise ValueError(f"Original parent passage is missing: {pid}")
    actual = next((row for row in _rows(data / "passages.jsonl") if row["passage_id"] == passage.get("passage_id")), None)
    if actual is None or any(passage.get(key) != value for key, value in actual.items()):
        raise ValueError("Parent context requested for a changed or non-active child passage")
    for key in ["document_id", "page", "source", "title", "url", "published_date", "available_by", "date_precision", "date_source",
                "event_date", "event_status", "redistribution", "regions"]:
        if passage.get(key) != parent.get(key):
            raise ValueError(f"Child passage changed parent provenance: {key}")
    child_text = " ".join(passage.get("text", "").split())
    if not child_text or child_text not in " ".join(parent["text"].split()):
        raise ValueError("Child passage is not a contiguous part of its original parent")
    if "char_start" in passage or "char_end" in passage:
        start, end = passage.get("char_start"), passage.get("char_end")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(parent["text"])
                or parent["text"][start:end] != passage["text"]):
            raise ValueError("Child passage character offsets do not match original parent text")
    return parent
