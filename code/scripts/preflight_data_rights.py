"""Metadata-only rights inventory; never a legal clearance or corpus rewrite.

Policies were inspected through official primary web pages on 2026-10-13.
The script does not refresh those policies, infer rights from a domain alone,
export source text, alter an old redistribution flag, or contact a rights holder.
Candidates remain subject to document/component checks. All raw text is withheld.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
import zipfile

from oil_evidence import machine_evaluation as protocol
from oil_evidence.store import now

OBSERVED_ON = "2026-10-13"
CORPUS_SHA256 = "62dde606a50bda9c7404231498c5dc83f0cd4a36f1259540f2ae377f7ca767cd"
WB_OCTOBER = "https://thedocs.worldbank.org/en/doc/55ed8b04df8228fea1607a97b5561f73-0050012025/original/CMO-October-2025.pdf"
AU_MARCH = "https://www.industry.gov.au/sites/default/files/2025-03/resources-and-energy-quarterly-march-2025.pdf"
EVIDENCE = {
    "eia": dict(url="https://www.eia.gov/about/copyrights_reuse.php", title="EIA Copyrights and Reuse",
        observation="official_page_opened", short_quote="Protected materials",
        finding="EIA information products may be reused with dated acknowledgement; contributed/licensed resources, photographs and logos have separate restrictions."),
    "boem_copy": dict(url="https://www.boem.gov/about-boem/disclaimer-liability-and-endorsement",
        title="BOEM Disclaimer", observation="official_page_opened", short_quote="may be distributed or copied",
        finding="Site information may be copied with appropriate credit; the disclaimer does not guarantee third-party non-infringement."),
    "boem_copyright": dict(url="https://www.boem.gov/about-boem/copyright-restrictions-and-permissions",
        title="BOEM Copyright, Restrictions and Permissions", observation="official_page_opened", short_quote="not all materials appearing",
        finding="Federal-agency output is generally public domain, but externally obtained material can be restricted and individual copyright status is not guaranteed."),
    "bsee": dict(url="https://www.bsee.gov/bsee.gov/privacy-disclaimer", title="BSEE Disclaimer",
        observation="official_page_opened", short_quote="may be distributed or copied",
        finding="BSEE site information may be copied with byline/image credit; references to external material do not establish external reuse rights."),
    "doe": dict(url="https://www.energy.gov/web-policies", title="DOE Web Policies",
        observation="official_page_opened", short_quote="protected by U.S. and foreign",
        finding="DOE government information may be copied with acknowledgement; contributed or licensed works require their own permission beyond applicable exceptions."),
    "treasury": dict(url="https://home.treasury.gov/subfooter/privacy-policy", title="Treasury Privacy Policy",
        observation="official_page_opened_no_copyright_grant_found", short_quote="",
        finding="The inspected privacy/legal notice did not establish a blanket copyright licence for the collected press releases or OFAC records."),
    "us_statute": dict(url="https://www.copyright.gov/title17/92chap1.html#105", title="US Copyright Office: 17 USC 101 and 105",
        observation="official_statute_opened", short_quote="part of that person’s official",
        finding="The statutory government-work category depends on employee/officer authorship within official duties, not merely website hosting; transferred copyrights and statutory exceptions also exist."),
    "dallas": dict(url="https://www.dallasfed.org/fed/disclaimer", title="Dallas Fed Terms of Use",
        observation="official_page_opened", short_quote="purpose of private gain",
        finding="Bank-owned material may be reproduced with attribution and without private gain; third-party content and proprietary graphics/logos are excluded."),
    "dallas_comments": dict(url="https://www.dallasfed.org/research/surveys/des/2019/1901", title="Dallas Fed Energy Survey: first quarter 2019",
        observation="official_document_opened", short_quote="",
        finding="Survey pages combine Bank reporting and respondent comments; this inventory does not certify ownership of respondent statements."),
    "wb_general": dict(url="https://www.worldbank.org/ext/en/legal/terms-conditions/reproduction-and-use",
        title="World Bank Reproduction and Use", observation="official_page_opened", short_quote="unless explicitly stated otherwise",
        finding="Educational access with attribution is distinguished from reproduction/redistribution, which requires permission absent a specific applicable licence."),
    "wb_okr": dict(url="https://www.worldbank.org/ext/en/legal/terms-conditions/open-knowledge-repository",
        title="World Bank Open Knowledge Repository Terms", observation="official_page_opened", short_quote="Third Party Content",
        finding="Repository works can carry different licences; component ownership, attribution and translation/adaptation notices require separate attention."),
    "wb_october": dict(url=WB_OCTOBER, title="World Bank Commodity Markets Outlook: October 2025, copyright page",
        observation="official_pdf_copyright_page_opened", short_quote="CC BY 3.0 IGO",
        finding="This specific report states CC BY 3.0 IGO, with attribution/adaptation requirements and a third-party-component exception."),
    "wb_april2021": dict(url="https://thedocs.worldbank.org/en/doc/c5de1ea3b3276cf54e7a1dff4e95362b-0350012021/original/CMO-April-2021.pdf",
        title="World Bank Commodity Markets Outlook: April 2021", observation="pdf_opened_but_rights_not_confirmed_later_timeout",
        short_quote="", finding="Do not transfer the October 2025 licence to this different report; its complete rights notice still needs confirmation."),
    "wb_april2025": dict(url="https://thedocs.worldbank.org/en/doc/1b388949805c9a0ae3736bdacb32ea94-0050012025/original/CMO-April-2025.pdf",
        title="World Bank Commodity Markets Outlook: April 2025", observation="pdf_opened_but_rights_not_confirmed_later_timeout",
        short_quote="", finding="The specific rights notice was not reliably extracted during this attempt; report-level verification remains pending."),
    "uk": dict(url="https://www.gov.uk/help/terms-conditions", title="GOV.UK Terms and Conditions",
        observation="official_page_opened", short_quote="published under the",
        finding="Most GOV.UK content is Crown-copyright material published under OGL; credited non-Crown and exempt content need separate checks."),
    "ogl": dict(url="https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
        title="Open Government Licence v3.0", observation="web_fetch_403_no_bypass", short_quote="",
        finding="The controlling licence URL returned 403 to the web tool; no fresh full-text licence verification is claimed."),
    "au": dict(url=AU_MARCH, title="Resources and Energy Quarterly: March 2025, copyright notice",
        observation="official_search_index_notice_full_pdf_exceeds_web_tool_size_limit", short_quote="content supplied by third parties",
        finding="The publisher-indexed notice specifies CC BY 4.0 excluding third-party content, coat of arms, logos and trademarked/noted material; full PDF fetch hit the tool size cap."),
}

FAMILIES = {
    "EIA": ("candidate_conditional", "dated_attribution_and_third_party_exclusions", ["eia"], {"www.eia.gov", "eia.gov"}),
    "BOEM-EIS": ("held", "document_component_authorship_not_cleared", ["boem_copy", "boem_copyright"], {"www.boem.gov"}),
    "BOEM-POLICY": ("held", "document_component_authorship_not_cleared", ["boem_copy", "boem_copyright"], {"www.boem.gov"}),
    "BSEE-HURRICANE": ("candidate_conditional", "agency_text_credit_external_content_excluded", ["bsee"], {"www.bsee.gov"}),
    "DOE-SPR": ("candidate_conditional", "government_information_only_third_party_review", ["doe"], {"www.energy.gov"}),
    "DOE-SPR-NEWS": ("candidate_conditional", "government_information_only_third_party_review", ["doe"], {"www.energy.gov"}),
    "TREASURY-OIL": ("held", "agency_authorship_or_document_permission_confirmation_needed", ["treasury", "us_statute"], {"home.treasury.gov"}),
    "OFAC": ("held", "agency_authorship_or_document_permission_confirmation_needed", ["treasury", "us_statute"], {"ofac.treasury.gov"}),
    "DALLAS-OIL": ("held", "separate_bank_owned_reporting_from_respondent_comments", ["dallas", "dallas_comments"], {"www.dallasfed.org"}),
    "WB-CMO": ("held", "specific_report_licence_pending", ["wb_general", "wb_okr"], {"thedocs.worldbank.org"}),
    "UK": ("candidate_conditional", "OGL_notice_and_excluded_components_pending", ["uk", "ogl"], {"assets.publishing.service.gov.uk", "www.gov.uk"}),
    "AU-REQ": ("candidate_conditional", "indexed_CC_BY_4_notice_component_checks_pending", ["au"], {"www.industry.gov.au"}),
}
REGISTRY_FIELDS = ("passage_id", "document_id", "source", "url", "title", "published_date", "available_by",
                   "page", "source_page_id", "char_start", "char_end", "word_count")


def decision(row):
    source = row.get("source", "")
    family = "EIA" if source == "EIA" or source.startswith("EIA-") else "UK" if source.startswith("UK-") else source
    status, reason, evidence, hosts = FAMILIES.get(family, ("held", "unreviewed_source_family", [], set()))
    parsed = urlsplit(row.get("url", ""))
    if parsed.scheme != "https" or parsed.hostname not in hosts or parsed.username or parsed.password:
        return family, "held", "official_source_identity_not_matched", evidence
    if family == "EIA" and row.get("redistribution") != "eia_attribution":
        status, reason = "held", "original_attribution_basis_missing"
    if family == "UK" and row.get("redistribution") != "OGL_with_exclusions":
        status, reason = "held", "original_document_licence_basis_missing"
    if family == "AU-REQ" and (row.get("url") != AU_MARCH or row.get("redistribution") != "cc_by_4_attribution_with_exclusions"):
        status, reason = "held", "this_Australian_document_notice_not_reviewed"
    if family == "WB-CMO" and row.get("url") == WB_OCTOBER:
        status, reason, evidence = "candidate_conditional", "CC_BY_3_IGO_third_party_component_checks_pending", evidence + ["wb_october"]
    return family, status, reason, evidence


def inventory(rows):
    if not rows:
        raise ValueError("A nonempty original corpus is required")
    seen, registry, documents, counts = set(), [], {}, defaultdict(Counter)
    for row in rows:
        pid, doc = row.get("passage_id"), row.get("document_id")
        if not isinstance(pid, str) or not pid or pid in seen or not isinstance(doc, str) or not doc:
            raise ValueError("Original corpus IDs are missing or duplicated")
        if not isinstance(row.get("text"), str) or not row["text"]:
            raise ValueError("Original source text must exist for hashing, never export")
        seen.add(pid)
        family, status, reason, evidence = decision(row)
        record = {key:row.get(key) for key in REGISTRY_FIELDS}
        record.update(original_redistribution=row.get("redistribution"), source_family=family,
            rights_preflight_status=status, reason=reason, policy_evidence_ids=evidence,
            export_text_cleared=False, text_sha256=protocol.sha256(row["text"].encode("utf-8")))
        registry.append(record)
        counts[row.get("source", "unknown")][status] += 1
        if doc not in documents:
            documents[doc] = dict(document_id=doc, source=row.get("source"), url=row.get("url"), title=row.get("title"),
                published_date=row.get("published_date"), available_by=row.get("available_by"),
                passage_ids=[], status_counts=Counter(), policy_evidence_ids=set(), export_text_cleared=False)
        document = documents[doc]
        if document["url"] != row.get("url"):
            raise ValueError("One frozen document ID maps to conflicting source URLs")
        document["passage_ids"].append(pid)
        document["status_counts"][status] += 1
        document["policy_evidence_ids"].update(evidence)
    for document in documents.values():
        document["status_counts"] = dict(document["status_counts"])
        document["policy_evidence_ids"] = sorted(document["policy_evidence_ids"])
    totals = Counter(row["rights_preflight_status"] for row in registry)
    summary = dict(status="rights_policy_preflight_not_legal_or_document_clearance", corpus_passages=len(registry),
        unique_documents=len(documents), conditional_candidate_passages=totals["candidate_conditional"],
        held_passages=totals["held"], text_export_cleared_passages=0, all_raw_text_withheld_from_package=True,
        original_redistribution_flags_preserved=True, family_counts={key:dict(value) for key,value in sorted(counts.items())},
        official_policy_observed_on=OBSERVED_ON, inventory_is_not_a_corpus_filter=True,
        limitations=["A conditional candidate is not an export permission or proof of ownership of every component.",
            "No legal certification, fair-use determination, foreign-law analysis, rights-holder consent or course-rights approval is inferred.",
            "Original corpus, splits, labels and index remain unchanged; rights status is a separate metadata ledger.",
            "Public-domain hosting does not establish the authorship/status of contractor, donated, licensed or respondent content.",
            "Live URLs may change; reconstruction must match original content hashes rather than rebinding the experiment."])
    return registry, list(documents.values()), summary


def write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(raw if isinstance(raw, bytes) else protocol.canonical_bytes(raw) + b"\n")


def run(root, corpus_path, expected_sha256=CORPUS_SHA256):
    root, corpus_path = Path(root).resolve(), Path(corpus_path).resolve()
    if not corpus_path.is_relative_to(root / "data") or corpus_path.name != "corpus.jsonl":
        raise ValueError("Read the explicitly frozen canonical corpus.jsonl, not arbitrary host files")
    raw = corpus_path.read_bytes()
    original_hash = protocol.sha256(raw)
    if expected_sha256 is not None and original_hash != expected_sha256:
        raise ValueError("Original frozen corpus hash differs")
    rows = protocol.parse_jsonl(raw, "frozen corpus")
    registry, documents, summary = inventory(rows)
    output = root / "data/machine_runs/data-rights" / (now().replace(":", "-") + "-" + uuid4().hex[:8])
    evidence = {key:dict(value, observed_on=OBSERVED_ON, observation_source="official_primary_web_source",
        observation_is_not_document_clearance=True) for key,value in EVIDENCE.items()}
    write_new(output / "policy_evidence.json", evidence)
    write_new(output / "passage_registry.jsonl", b"".join(protocol.canonical_bytes(row)+b"\n" for row in registry))
    write_new(output / "document_registry.jsonl", b"".join(protocol.canonical_bytes(row)+b"\n" for row in documents))
    summary.update(corpus_sha256=original_hash, created_utc=now(), corpus_source=str(corpus_path))
    write_new(output / "summary.json", summary)
    readme = ("Registry-only reconstruction package\n\n"
        "This archive contains metadata, IDs, source URLs, hashes and policy pointers only.\n"
        "It contains no source text, PDFs, images, evidence quotes, model prompts, answers, weights or credentials.\n"
        "Conditional candidate status is not document/component clearance. Retrieve originals from source URLs only where authorised,\n"
        "retain attribution/notices and compare original document/paragraph hashes. Quarantine mismatches; never rebind frozen experiments.\n"
        "Rights-holder or instructor decisions are not supplied by this inventory.\n")
    write_new(output / "REGISTRY_README.txt", readme.encode())
    allowed = ("passage_registry.jsonl", "document_registry.jsonl", "REGISTRY_README.txt")
    # Only generated metadata files enter the optional local ZIP; not the corpus
    # path, policy quotes, source documents or arbitrary supplied directories.
    with zipfile.ZipFile(output / "source-registry-only.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in allowed:
            archive.writestr(name, (output / name).read_bytes())
    if protocol.sha256(corpus_path.read_bytes()) != original_hash:
        raise ValueError("Original corpus changed during inventory; output retained but uncertified")
    receipt = dict(status=summary["status"], original_corpus_unchanged=True, original_corpus_sha256=original_hash,
        conditional_candidate_passages=summary["conditional_candidate_passages"], held_passages=summary["held_passages"],
        text_export_cleared_passages=0, original_corpus_count=len(rows), output=str(output),
        registry_zip_contents=list(allowed), registry_only_zip=True, raw_text_exported=False, model_calls=0,
        legal_certification=False, copyright_owner_permission_obtained=False, instructor_rights_approval_obtained=False,
        implementation_sha256=protocol.sha256(Path(__file__).read_bytes()),
        artifacts_sha256={p.name:protocol.sha256(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()})
    write_new(output / "receipt.json", receipt)
    return output, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    args = parser.parse_args()
    output, receipt = run(args.canonical_root, args.corpus)
    print(json.dumps(dict(output=str(output), **{key:receipt[key] for key in
        ("original_corpus_count", "conditional_candidate_passages", "held_passages", "text_export_cleared_passages",
         "raw_text_exported", "original_corpus_unchanged")}), indent=2))
