"""Record eight machine QA cases without changing the corpus or annotations."""

import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from oil_evidence.corpus import clean_page
from oil_evidence.store import words


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/expansion_runs/official-2018-2026-v2"
CASES = {
    "c66494af2ccae33b4d25-140fb23929a5226f": {
        "verdict": "oil_evidence_with_furniture_warning",
        "observation": "Substantive primary-oil production, net imports and refinery output facts; the chapter heading, employee contacts and telephone/email details precede the evidence.",
    },
    "c66494af2ccae33b4d25-4fef12069be09cd5": {
        "verdict": "oil_evidence_with_numeric_warning",
        "observation": "Substantive refinery output/import narrative. Visual PDF comparison confirms 1998 superscript-footnote 1 becomes 19981 and the chart's 2023 superscript 2 becomes 20232. These are not five-digit years.",
        "visual_check": "discovery/scope-qa/dukes-2024-p14.png",
    },
    "a14377a9532a20811e3a-eba1d164728ff052": {
        "verdict": "oil_evidence_with_furniture_warning",
        "observation": "Substantive oil production, imports, refinery output and stocks facts. Employee contacts remain; primary oils must not be silently equated with crude only.",
    },
    "a14377a9532a20811e3a-8443267688e47956": {
        "verdict": "requires_numeric_cleaning_repair",
        "observation": "Substantive jet-fuel/heating-oil demand prose, but raw page text ends its first paragraph 'pre-pandemic demand in\\n2019.'. The numeric-only cleaning rule drops 2019., leaving an incomplete comparison and joined chart caption.",
        "raw_numeric_line_lost": "2019.",
    },
    "a27a5ee8526c96fd4cf4-b096c5ebbb86daeb": {
        "verdict": "mixed_energy_with_real_oil_fact",
        "observation": "Actual oil output rose 0.2 percent relative to Q3 2018; most remaining lines concern other energy sources. Not a false oil glossary hit, but oil-specific evidence density is low.",
    },
    "4b18b4617931040ed876-962a0bbaf849b813": {
        "verdict": "mixed_energy_with_real_oil_fact",
        "observation": "Actual Q2 2020 oil output -1.4 percent, crude -1.6 percent and NGL +1.1 percent; distinctions are retained. Other fuels dominate the remainder and should not answer oil-only questions.",
    },
    "ef5b3e4469054a9e2d17-6cc8a3b858c965a0": {
        "verdict": "reject_navigation_dominated_fragment",
        "observation": "Selected fragment is About this release / In this release / Data tables navigation; oil appears in chapter-link labels, while the final sentence is total final energy consumption, not oil evidence. Other substantive oil statements elsewhere on the page do not justify this fragment.",
        "visual_check": "discovery/scope-qa/energy-trends-2021-p1.png",
    },
    "040c390c0dd0711a82b4-4f9e1974c6aa7d98": {
        "verdict": "reject_navigation_dominated_fragment",
        "observation": "Most of the fragment is the same cover/sidebar navigation. Its final oil-output -16 percent claim is real, but the navigation-heavy passage should be rejected or carefully re-extracted as prose, not counted as clean evidence.",
    },
}


def audit():
    stage = RUN / "stage/passages.jsonl"
    initial_bytes = stage.read_bytes()
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()
    passages = {row["passage_id"]: row for row in
                (json.loads(line) for line in initial_bytes.decode("utf-8").splitlines() if line.strip())}
    docs = {row["document_id"]: row for row in
            (json.loads(line) for line in (ROOT / "data/documents.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    readers, records = {}, []
    for passage_id, assessment in CASES.items():
        if passage_id not in passages:
            records.append({"passage_id": passage_id, **assessment,
                            "present_in_audited_stage": False,
                            "note": "Observed in the first 8846-passage stage; a later rerun may have removed it."})
            continue
        passage = passages[passage_id]
        doc = docs[passage["document_id"]]
        if doc["document_id"] not in readers:
            raw = (ROOT / "data" / doc["raw_path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != doc["raw_sha256"]:
                raise ValueError("Raw PDF changed: " + doc["document_id"])
            readers[doc["document_id"]] = PdfReader(io.BytesIO(raw))
        raw_page = readers[doc["document_id"]].pages[passage["page"] - 1].extract_text() or ""
        cleaned = " ".join(words(clean_page(raw_page)))
        fragment = " ".join(words(passage["text"]))
        records.append({
            **assessment, "passage_id": passage_id,
            "present_in_audited_stage": True, "passage": passage,
            "raw_path": doc["raw_path"], "raw_sha256": doc["raw_sha256"],
            "physical_page": passage["page"],
            "available_by_within_window": "2018-01-01" <= passage["available_by"] <= "2026-10-05",
            "document_not_quarantined": not bool(doc.get("ingest_exclusion_reason")),
            "fragment_in_current_cleaned_raw_page": fragment in cleaned,
            "raw_page_text": raw_page,
            "assessment_provenance": "Codex machine_not_human",
        })
    report = {
        "provenance": "machine_not_human",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection": "Four DUKES passages from two eligible annual reports and four Energy Trends passages from distinct years; includes true failure cases, not only successes.",
        "stage_passages_sha256_start": initial_hash,
        "stage_passages_sha256_end": hashlib.sha256(stage.read_bytes()).hexdigest(),
        "method": "Read-only stored passage comparison with original PDF text; two layout-sensitive pages also rendered and visually inspected by Codex.",
        "limitations": "Eight machine-reviewed cases are not formal human annotation or a representative error-rate estimate. No documents, passages or labels were changed.",
        "counts": {"cases": len(records), "by_verdict": dict(Counter(row["verdict"] for row in records))},
        "records": records,
    }
    output = RUN / "uk_oil_qa.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "counts": report["counts"],
                      "present": sum(row["present_in_audited_stage"] for row in records)}, ensure_ascii=False))


if __name__ == "__main__":
    audit()
