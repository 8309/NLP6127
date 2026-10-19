"""Read-only raw-PDF scope audit; writes an isolated machine audit report only."""

import hashlib
import io
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from oil_evidence import pdf_scope


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "expansion_runs/official-2018-2026-v2/pdf_scope_audit.json"
SOURCES = {"AU-REQ", "UK-DUKES", "UK-ENERGY-TRENDS"}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _diagnostics(pages, source, scope):
    if source == "AU-REQ":
        toc = pdf_scope._au_toc_ranges(pages)
        headings = [
            {"page": number, "title": title, "line": line,
             "printed_page": pdf_scope._au_printed_page(text)}
            for number, text in enumerate(pages, 1)
            for title, line in pdf_scope._au_headings(text)
        ]
        candidates = []
        for heading in headings:
            if heading["title"] != "oil":
                continue
            text = pages[heading["page"] - 1]
            strong = bool(re.match(r"^\d{1,2}\s*\.?\s*Oil", heading["line"], re.I))
            strong |= bool(re.search(r"Australia[’']s\s+oil\s+sector", text, re.I))
            confirmed = any(heading["printed_page"] == low for low, _ in toc)
            if strong or confirmed:
                candidates.append(heading)
        detail = {"toc_printed_ranges": toc, "major_headings": headings,
                  "verified_oil_start_candidates": candidates}
        boundary_labels = {label for pair in toc for label in pair}
        detail["toc_boundary_text_layer_probes"] = []
        for physical, text in enumerate(pages, 1):
            printed = pdf_scope._au_printed_page(text)
            if printed is None and re.fullmatch(r"\d{1,3}", text.strip()):
                printed = int(text.strip())
            if printed in boundary_labels:
                detail["toc_boundary_text_layer_probes"].append({
                    "physical_page": physical, "observed_printed_page": printed,
                    "raw_text_excerpt": text[:3000],
                    "note": "Diagnostic text-layer observation only; page-number/TOC matches do not authorize scope.",
                })
        if len(candidates) != 1:
            reason = ("no_verified_oil_start" if not candidates
                      else "ambiguous_multiple_oil_starts")
        else:
            start = candidates[0]
            ends = [h for h in headings if h["page"] > start["page"] and h["title"] != "oil"]
            if not ends:
                reason = "next_major_chapter_missing"
            else:
                end = ends[0]
                related = [(low, high) for low, high in toc if low == start["printed_page"]]
                if related and not any(high == end["printed_page"] for _, high in related):
                    reason = "next_chapter_disagrees_with_toc"
                else:
                    reason = "verified_standalone_oil_and_next_major_chapter"
                detail["next_major_chapter"] = end
        # These are diagnostic hints only, never used to widen the range.
        detail["oil_heading_lookalikes"] = [
            {"page": number, "line": line}
            for number, text in enumerate(pages, 1)
            if not pdf_scope._contents_page(text)
            for line in pdf_scope._lines(text)[:12]
            if re.fullmatch(r"(?:\d{1,2}\s*\.?\s*)?Oil(?:\s+Oil)?", line, re.I)
        ]
    elif source == "UK-DUKES":
        headings = [
            {"page": number, "chapter": chapter, "title": title}
            for number, text in enumerate(pages, 1)
            for chapter, title in pdf_scope._uk_headings(text)
        ]
        allowed = {"oil and oil products", "petroleum", "petroleum and petroleum products"}
        candidates = [h for h in headings if h["chapter"] == 3 and h["title"] in allowed]
        detail = {"chapter_headings": headings, "verified_oil_start_candidates": candidates}
        if len(candidates) != 1:
            reason = ("no_verified_chapter_3_start" if not candidates
                      else "ambiguous_multiple_chapter_3_starts")
        else:
            start = candidates[0]
            later = [h for h in headings if h["page"] > start["page"] and h["chapter"] > 3]
            if not later:
                reason = "next_chapter_missing"
            elif later[0]["chapter"] == 4 and later[0]["title"] in {"natural gas", "gas"}:
                reason = "verified_chapter_3_to_chapter_4_boundary"
                detail["next_chapter"] = later[0]
            else:
                reason = "next_chapter_not_recognized_gas_chapter_4"
                detail["next_chapter"] = later[0]
        detail["chapter_heading_lookalikes"] = [
            {"page": number, "line": line}
            for number, text in enumerate(pages, 1)
            if not pdf_scope._contents_page(text)
            for line in pdf_scope._lines(text)
            if re.match(r"^Chapter\s+[34]\b", line, re.I)
        ]
    else:
        return {"reason": "no_special_chapter_scope_oil_topic_and_prose_filters_still_required"}
    if bool(scope) != reason.startswith("verified_"):
        reason = "diagnostic_inconsistency_recheck_required"
    return {"reason": reason, **detail}


def audit():
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    manifest_path = DATA / "documents.jsonl"
    initial_bytes = manifest_path.read_bytes()
    docs = [json.loads(line) for line in initial_bytes.decode("utf-8").splitlines() if line.strip()]
    docs = [doc for doc in docs if doc.get("source") in SOURCES]
    records = []
    for position, doc in enumerate(docs, 1):
        year = re.search(r"\b(20\d{2})\b", doc.get("title", ""))
        record = {key: doc.get(key) for key in (
            "document_id", "source", "title", "url", "published_date", "available_by",
            "date_precision", "date_source", "raw_path", "raw_sha256", "redistribution",
            "ingest_exclusion_reason")}
        record["edition_year"] = year[1] if year else (doc.get("available_by") or "unknown")[:4]
        record["eligible_by_current_manifest"] = not bool(doc.get("ingest_exclusion_reason"))
        try:
            raw = (DATA / doc["raw_path"]).read_bytes()
            record["observed_raw_sha256"] = _sha(raw)
            if record["observed_raw_sha256"] != doc["raw_sha256"]:
                raise ValueError("raw PDF hash does not match manifest")
            if not raw.startswith(b"%PDF-"):
                raise ValueError("raw file is not a PDF")
            pages = [page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages]
            scope = pdf_scope.oil_page_numbers(pages, doc["source"])
            record.update(
                physical_page_count=len(pages),
                scope_mode="unrestricted" if scope is None else "scoped" if scope else "fail_closed",
                oil_physical_pages=None if scope is None else sorted(scope),
                selected_page_count=None if scope is None else len(scope),
                text_layer_nonempty_pages=sum(bool(page.strip()) for page in pages),
                diagnostics=_diagnostics(pages, doc["source"], scope),
            )
        except Exception as error:
            record.update(scope_mode="audit_error", error=f"{type(error).__name__}: {error}")
        records.append(record)
        print(f"Scope audit {position}/{len(docs)}: {doc['source']} {record['edition_year']} "
              f"{record['scope_mode']} eligible={record['eligible_by_current_manifest']}", flush=True)

    final_bytes = manifest_path.read_bytes()
    final_hash = _sha(final_bytes)
    initial_hash = _sha(initial_bytes)
    final_docs = [json.loads(line) for line in final_bytes.decode("utf-8").splitlines() if line.strip()]
    final_docs = [doc for doc in final_docs if doc.get("source") in SOURCES]
    subset_start = _sha(json.dumps(docs, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    subset_end = _sha(json.dumps(final_docs, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    report = {
        "provenance": "machine_not_human",
        "auditor": "Codex automated raw PDF chapter-boundary audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "All AU-REQ, UK-DUKES and UK-ENERGY-TRENDS raw PDFs in the current documents snapshot",
        "method": "pypdf raw text extraction and the unchanged production oil_page_numbers function; diagnostic hints never expand scope",
        "documents_manifest_sha256_start": initial_hash,
        "documents_manifest_sha256_end": final_hash,
        "documents_snapshot_stable": initial_hash == final_hash,
        "source_documents_snapshot_sha256_start": subset_start,
        "source_documents_snapshot_sha256_end": subset_end,
        "source_documents_snapshot_stable": subset_start == subset_end,
        "scope_module_sha256": _sha(Path(pdf_scope.__file__).read_bytes()),
        "limitations": [
            "Automated audit is not independent human or visual verification of every chapter boundary.",
            "Current manifest ingest exclusions remain in force, regardless of a successful range match.",
            "A selected chapter page is not automatically a retained oil passage; prose, topic, table and duplicate filters still apply.",
            "No scope was widened, and no documents, passages, extraction caches or annotations were modified.",
        ],
        "counts": {
            "documents": len(records),
            "by_source": dict(Counter(row["source"] for row in records)),
            "scope_mode": dict(Counter(row["scope_mode"] for row in records)),
            "eligible_scope_mode": dict(Counter(row["scope_mode"] for row in records if row["eligible_by_current_manifest"])),
            "excluded_by_manifest": sum(not row["eligible_by_current_manifest"] for row in records),
            "fail_closed_by_year": dict(Counter(row["edition_year"] for row in records if row["scope_mode"] == "fail_closed")),
            "fail_closed_by_reason": dict(Counter(row["diagnostics"]["reason"] for row in records if row["scope_mode"] == "fail_closed")),
        },
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT), "counts": report["counts"],
                      "documents_snapshot_stable": report["documents_snapshot_stable"]}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    audit()
