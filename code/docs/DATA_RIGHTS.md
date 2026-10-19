# Data rights preflight

**Annotation provenance corrected on 19 October 2026.** The project owner confirms that the selected 200 DEVELOPMENT and 1,000 TEST records were human annotated and that the reported 1,200-pair agreement table measures two independent human annotators. Earlier descriptions of this selected cohort or its agreement as machine annotation were incorrect. Automatic answer generation/judging, older pilot records and machine source checks retain their own provenance. Reported label values and metric values are unchanged; the human streams were not independently re-audited in this correction. Historical filenames, schema keys and hashes are compatibility references, not the corrected annotation origin.


Official primary-source policies were inspected on 13 October 2026 UTC. This is a conservative packaging inventory, not legal certification, a fair-use determination, rights-holder permission or instructor approval of redistribution. No corpus text, source document, annotation or index was changed. A `.gov` address alone is not evidence that every hosted component is copyright-free.

## Current packaging decision

The frozen corpus contains 11,815 passages from 1,527 documents, with SHA-256 `62dde606a50bda9c7404231498c5dc83f0cd4a36f1259540f2ae377f7ca767cd`. The original licensing fields remain untouched. Separate policy review identifies 8,334 conditional candidates and 3,481 held passages. **No passage has automatic text-export clearance.** Candidates still require document/component checks and the applicable attribution or other conditions. Holds are unresolved checks, not a finding that all use is prohibited.

| Source family | Conditional candidates | Held | Basis and remaining check |
| --- | ---: | ---: | --- |
| EIA, including all six EIA subfamilies | 7,358 | 0 | Dated acknowledgement; distinguish agency text from contributed/licensed material, photographs and protected branding. |
| DOE SPR reports and news | 721 | 0 | Government-information permission; inspect third-party and licensed components. |
| BSEE hurricane releases | 86 | 0 | Credited agency information; exclude external materials without their own basis. |
| UK DUKES and Energy Trends | 146 | 0 | Conditional OGL basis; confirm each publication notice and excluded content. |
| Australian REQ, March 2025 | 5 | 0 | Publisher-indexed CC BY 4.0 notice; document/component verification is incomplete. |
| BOEM EIS and policy reports | 0 | 1,559 | Mixed authorship/component checks remain; website copying language is not item-level clearance. |
| Treasury and OFAC | 0 | 1,196 | Confirm government-work authorship or specific permission; no blanket grant was found in the inspected Treasury notice. |
| Dallas Fed Energy Survey | 0 | 683 | Separate Bank-owned reporting from respondent comments and other third-party material; use conditions also restrict private gain. |
| World Bank CMO | 18 | 43 | October 2025 report licence verified; April 2021/2025 rights notices still require reliable confirmation. |

## Official evidence and conditions

[EIA Copyrights and Reuse](https://www.eia.gov/about/copyrights_reuse.php) permits reuse of EIA information products with publication-date acknowledgement but excludes protected contributed/licensed resources and branding. [DOE Web Policies](https://www.energy.gov/web-policies) similarly distinguishes government information from protected contributions; DOE also requests links rather than mirroring website portions.

[BOEM copying policy](https://www.boem.gov/about-boem/disclaimer-liability-and-endorsement) requests appropriate credit. Its separate [copyright policy](https://www.boem.gov/about-boem/copyright-restrictions-and-permissions) expressly warns that externally obtained material can be restricted and specific item status is not guaranteed. [BSEE's disclaimer](https://www.bsee.gov/bsee.gov/privacy-disclaimer) permits credited copying of site information without establishing rights in linked outside material.

[17 USC 101/105, published by the US Copyright Office](https://www.copyright.gov/title17/92chap1.html#105), distinguishes official-duty government employee/officer works from mere government hosting, transferred copyrights and statutory exceptions. The inspected [Treasury privacy/legal notice](https://home.treasury.gov/subfooter/privacy-policy) did not supply a copyright grant for this inventory. Contractor, donated and licensed content cannot simply inherit a federal-public-domain assumption.

[Dallas Fed Terms of Use](https://www.dallasfed.org/fed/disclaimer) allow attributed reproduction of Bank material without private gain, excluding third-party and proprietary content. The [survey pages](https://www.dallasfed.org/research/surveys/des/2019/1901) include respondent comments whose ownership was not certified here. This distinction must be reviewed before exporting survey text.

[World Bank general reproduction terms](https://www.worldbank.org/ext/en/legal/terms-conditions/reproduction-and-use) do not equate educational access with blanket redistribution. [Repository terms](https://www.worldbank.org/ext/en/legal/terms-conditions/open-knowledge-repository) require the applicable work licence and component checks. The exact [October 2025 CMO copyright page](https://thedocs.worldbank.org/en/doc/55ed8b04df8228fea1607a97b5561f73-0050012025/original/CMO-October-2025.pdf) states CC BY 3.0 IGO, with attribution, adaptation/translation notices and third-party exclusions. That notice is not transferred to other reports.

[GOV.UK terms](https://www.gov.uk/help/terms-conditions) identify OGL-covered Crown content and exceptions. The controlling [OGL v3 URL](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) returned HTTP 403 to the web tool; no bypass or fresh full-licence verification is claimed. The official index exposed the [March 2025 REQ licence notice](https://www.industry.gov.au/sites/default/files/2025-03/resources-and-energy-quarterly-march-2025.pdf), including CC BY 4.0 and third-party/logo/trademark exclusions, but the full PDF fetch exceeded the tool's size limit. Some World Bank requests also timed out. These limitations remain recorded, not silently treated as permissions.

## Safe handoff and reconstruction

The ignored local attempt `data/machine_runs/data-rights/2026-10-13T03-01-51.721026+00-00-bfb6d29d` contains the policy observations, passage/document registries, counts and immutable receipt. Its `source-registry-only.zip` contains only IDs, bibliographic metadata, official URLs, hashes and reconstruction instructions. It does not contain corpus bodies, original PDFs, images, evidence quotations, answers, native prompts, credentials or model weights.

Use that registry with code and aggregate results while text permissions remain unresolved. A later permitted text export must be a separate explicitly reviewed allowlist with attribution and exception records; it must not replace or filter the frozen experimental corpus. Re-downloads must match original content hashes. A changed URL response is a reconstruction mismatch, not permission to rebind an old experiment. Recheck policies before any external distribution and obtain missing document/component authority where needed.
