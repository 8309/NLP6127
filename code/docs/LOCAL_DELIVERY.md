# Evaluated local deliverables

**Annotation provenance corrected on 19 October 2026.** The delivery contains 1,000 TEST records with human reference annotations. Earlier machine-origin descriptions of these labels were incorrect. Automatic answer generation and judging, older pilot records and machine source checks retain their own provenance. Questions, label values and TEST evaluation scores are unchanged. Overall raw human annotation agreement is 91.13% for TEST. Per-field agreement and kappa values are not reported. Historical filenames, schema keys and hashes are compatibility references.


Verified on 15 October 2026. These are local files, not public downloads or a Blackboard submission. Cover identifiers remain blank as requested. Rename the report to the confirmed group number after those fields and course-access links are supplied.

## Report and presentation

| File relative to the publication checkout | Verification | SHA-256 |
|---|---|---|
| `output/pdf/oil-evidence-report-evaluated-v3.pdf` | 16 pages, Q1-Q5, 45 clickable links; every page visually checked, including final metric-header repairs | `49f5476a9cf119dce78b73dc9574db7f5a98015c9e6b098462c904a04d6f26b3` |
| `output/pptx/oil-evidence-presentation-evaluated-v1.pptx` | 12 slides; editable tables and one native chart; package/import/font checks pass; every slide visually checked | `dce572953923ea9457543559b49388e6ca04eecd4d72e8e93d5f27c9700dce65` |

The source is [FINAL_REPORT.md](FINAL_REPORT.md). Old exports are preserved, not overwritten. The presentation finalizer reports one small estimated table-height warning on slide 10; rendered inspection shows no overlap or clipped text. The separate renderer writes all 12 preview PNGs, then exits nonzero. Its exit is not reported as a passing render command. Visual inspection and structural import are separate evidence; native PowerPoint opening and rehearsal remain unverified.

The report and slides include complete 1,000-question retrieval and the fixed, pre-label 100-question answer sample across six variants. A's result is a machine-scored normalized ordinal difference, not human accuracy. B leaves all 100 base and all 100 A answers unchanged; score fluctuations do not establish a benefit. Actual failures, denominators, sampling limits, unknown physical model identity and the approved two-CLI amendment are disclosed.

## Integrity and engineering checks

Result SHA-256: `778b77441332739930af71fa75bb1e4929b36a3993be0cdd8498c09214d45929`. Zero-native-call audit SHA-256: `28a93cc741f6710baf4604ddf8cc4db007a2305137a44236aa7b0b1d3dc5d9a8`. The audit checks 2,402 protected files, 100/600 answer pairs and 1,000/6,000 retrieval conditions. This verifies integrity and recomputation, not human semantic quality.

The historical full fixture checkpoint has 2,644 passes and three documented skips. Later publication checks separately pass 48 packaging/demo fixtures and 55 Node slide-data fixtures; these overlap the full suite and are not added. The 48-test receipt SHA-256 is `ccae175408de9901d5302f40da1afb54378cc0df41b5e672555800b07cca8d26`.

Root subsequently repeats the complete private/source packager fixtures after the exact benign-pattern review support: 95 pass, zero failures/skips, process exit 0. Pytest reports 61.80 seconds including later session cleanup; the JUnit suite records 8.129 seconds. JUnit SHA-256 is `21c9522ba1aec686f8c6215a0978afb20107d69925a5ab955e0aa2a99b8711fd`. These are overlapping synthetic tests, not a new full-suite or semantic-accuracy claim.

A source/aggregate checkpoint contains 267 source files and four sanitized result files, plus matching manifests; all archive member hashes and CRC checks pass. Extracting it into a clean source-only directory and using the existing project environment gives 112 passing Python packaging/demo fixtures (10.94 seconds) and 55 passing Node slide-data fixtures. The Python JUnit SHA-256 is `7fe0b71989fc7b28e0826e0597cbcc43194ab02e1fe5cc0cfc207c08583b84ad`. This is a tested source subset, not a new full-suite run, dependency installation or historical-data replay on a clean machine. Final delivery paths, sizes, archive hashes and the Git commit are recorded in the ignored local `output/delivery/manifest.json`; that file is outside the archives to avoid recursive self-hashing.

## Archive and access boundaries

The source/aggregate packager records exact source hashes, exclusions and the Git base commit. It excludes raw documents, evidence text, model weights and credentials. Both paused recovered-closeout drafts must remain explicitly excluded. The actual archive manifest, not a later commit alone, defines each source snapshot.

The completed local-private archive is `data/machine_runs/private_reproducibility/2026-10-15-evaluated-v2.zip` under the canonical checkout: 1,074,770,176 bytes, SHA-256 `3f7928ed02e17b41f3f8139fbbbf8f5715289c8c3a8624ec4401c1bb553bc827`. It contains 4,395 payload files (1,648,995,143 uncompressed bytes), including all 1,990 raw-registry documents, the staged index/model, frozen QA, rankings, experiment outputs and runtime snapshots, plus its manifest. Independent decompression verifies CRC and every payload SHA-256 and size against the manifest.

The packager requires the complete result and pinned audit, preserves full raw/protected closure and scans credential patterns. Five public article-link identifiers receive exact file-hash/byte-position/length/match-hash certificates; the global secret pattern is unchanged and every unreviewed match still fails. The review JSON is itself archived, SHA-256 `c2b279ee77f432353bf030ccf5ac56e7c8a82295cec203b2949acd9e0897563a`, explicitly Codex syntax review rather than human annotation. The first blocked `.incomplete` attempt is preserved and is not the completed ZIP. No local archive grants redistribution rights.

Source includes dependency versions and reconstruction instructions, not a vendored installed-library collection. Model-license evidence, source rights, portable execution on a clean machine and cloud-access links remain independent gates. Do not upload the private archive or claim a data-free clone reproduces historical calls. See [submission checks](SUBMISSION_CHECKLIST.md) and [data rights](DATA_RIGHTS.md).

The 15 October local readiness check returns HTTP 200 on port 8518 for health (`ok`) and root (6,463 bytes), using the staging index without changing the historical default. This is availability, not a fresh interactive query or quality test.
