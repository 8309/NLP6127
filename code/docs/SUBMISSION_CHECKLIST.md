# Draft Oil Evidence Submission Checklist

**Annotation provenance corrected on 19 October 2026.** The project owner confirms that the selected 200 DEVELOPMENT and 1,000 TEST records were human annotated and that the reported 1,200-pair agreement table measures two independent human annotators. Earlier descriptions of this selected cohort or its agreement as machine annotation were incorrect. Automatic answer generation/judging, older pilot records and machine source checks retain their own provenance. Reported label values and metric values are unchanged; the human streams were not independently re-audited in this correction. Historical filenames, schema keys and hashes are compatibility references, not the corrected annotation origin.


Status: Local engineering and machine evaluation complete; submission pending. Snapshot: **2026-10-15 Singapore time**. The 200 DEVELOPMENT/1,000 TEST human-annotated dataset is frozen. All 1,000 TEST questions complete six-condition retrieval evaluation; 13 whole groups complete the fixed 100-question answer sample and all 600 six-variant rows. Serial aggregation and the independent offline audit pass with zero new model calls, verifying 2,402 protected files, 153 completed jobs and 160 retained attempts. Result SHA-256 is `778b77441332739930af71fa75bb1e4929b36a3993be0cdd8498c09214d45929`; audit SHA-256 is `28a93cc741f6710baf4604ddf8cc4db007a2305137a44236aa7b0b1d3dc5d9a8`. Historical failed receipts, unknown failed usage and three completed-503 certificates remain retained. The routing receipt remains false and no timeout receipt is invented. The evaluated PDF, editable presentation, private ZIP and source/aggregate checkpoint archives are checked; see [local deliverables](LOCAL_DELIVERY.md). Identity fields, rights review, course links, rehearsal and submission remain pending.

The [parallel scheduler](../scripts/run_parallel_formal_jobs.py) preserves exact pinned original preflight/group bodies and job requests, uses exclusive ownership and prohibits blanket retries. It drains work before one serial zero-new-call aggregation. The app-update native binary changes from `d13914ced6c7af174d8d638db938284231227312104b06a25e9608c632a7daec` to `e98c516f1c8935789aabe9122ee7e05ec4c3efd1ba6acf179242e031c495b99e` under the same `0.162.0-alpha.17.2` version string. After the old binary was not found, the user directly approves the [sealed two-CLI accounting amendment](../scripts/formal_cli_rollover.py), policy SHA-256 `7edd4291f9419f37f9d495e1ff1b653a2c15e4122bdba13bcab7a6b98f96b96d`. Its executed private driver uses four historical transport predicates plus one explicit binary-cardinality predicate, preserving the known-model guard and every other stage/metric check. Original source/plan/frozen bytes are unchanged, but the executed runtime contract is amended, not an unmodified single-binary experiment. Physical identity remains unknown; no model diversity is established and new-binary 503 admission remains forbidden. The [amended offline auditor](../scripts/audit_formal_cli_rollover.py) requires exact policy/source pins and all four historical failed receipts.

The selected cohort and its agreement table are human annotations, as confirmed by the owner on 19 October 2026. The older machine-substitution description does not describe this delivery. Instructor approval for individual completion and the 100-question answer subset is user-reported; automatic answer judges remain separate from the human labels.

## Cover and course requirements

- [ ] Fill in group number: ____________________.
- [ ] Fill in English student name: ____________________.
- [ ] Fill in matriculation number: ____________________.
- [x] Include Q1 through Q5 in the evaluated local PDF and explain the user-reported approved solo and machine-annotation workflow; cover identifiers remain blank.
- [ ] Name the final PDF using the required group number; do not invent a temporary group identity.
- [ ] Check the course submission portal and final links before submitting. The brief says that only the first submission counts.
- [ ] Prepare the in-person presentation and a working demonstration.

The extracted brief records a deadline of 7 November 2026 at 23:59 Singapore time. The report and presentation have separate assessment requirements; the local [acceptance checklist](ACCEPTANCE.md) is the requirement reference, not evidence that these tasks are complete.

## Verified inputs

| Artifact | Current status | Evidence |
|---|---|---|
| Staging corpus | 11,815 passages, 1,562,715 words, 28,582 types, 1,527 contributing documents | [Aggregate receipt](../reports/progress.json) |
| Segmentation rules | Shared 160 target, 65/240 soft bounds, 480 context cap | [Processing record](UNIFIED_CORPUS.md) |
| Staging vector index | 11,815 by 384 float32, hash and order checks passed | [Aggregate receipt](../reports/progress.json) |
| Real retrieval operational checks | 24 passed; date, region and offsets checked | [Aggregate receipt](../reports/progress.json) |
| Development retrieval | 50 questions, 150 real rankings, 1,500 returned hits, 973 pooled question-evidence pairs; zero returned future/unknown-date hits | Local receipt identified in [the report](FINAL_REPORT.md) |
| Warm development retrieval latency | BM25 median 0.175440 s; dense 0.193433 s; hybrid 0.201786 s; 50 queries per mode at depth 10 | Actual median/p95 table in [the report](FINAL_REPORT.md) |
| English publication fixture suite | 1,415 passed, zero failed, three documented skips | [Publication receipt](../reports/english_publication.json) |
| Initial development evaluation | Both 50-record streams, all 50 adjudications, frozen development version and actual depth-10 ranked scores | [Report with local artifact references](FINAL_REPORT.md) |
| Full-cohort human inter-annotator agreement | Human raw agreement: 98.67% answerability, 93.58% event state, 96.59% relevance; 1,200 records / 15,466 relevance pairs | [Report](FINAL_REPORT.md); owner-confirmed human agreement, not factual accuracy |
| Real query UI diagnostic | Four-tab in-process query; five passages, two claims, 15.918344 s end to end | [Report](FINAL_REPORT.md); not proof of running HTTP service |
| Development ablation | Five fixed cases, four original generator plus three fresh support/judge executions | [Report](FINAL_REPORT.md); not held-out accuracy |
| Sampled-scope local fixture suite | 2,046 passed, zero failures, three documented skips, 306.24 s | Local checkpoint JUnit; later wire-driver/historical-pipeline additions have separate checks |
| Later full fixture checkpoint | 2,510 passed, three documented skips, zero failures, 546.45 s | Historical full-suite receipt; excludes only the preserved paused recovered-closeout test |
| Historical rollover fixture checkpoint | 2,626 passed, three documented skips, zero failures, 514.04 s | Local JUnit SHA-256 `3f73f154cae877ba93f52c35a55ed659d14b59cc6d92afb28dc354862c925ee7`; predates reviewed-retry fixtures |
| Latest complete fixture suite | 2,644 passed, three documented skips, zero failures/errors, 680.89 s | Local JUnit SHA-256 `f1df42315801ee066193cd55a127de0640c625363a017bc6d3ffc41c5701274f`; includes reviewed retries |
| Node slide-data fixtures | 55 passed, zero failures | Final-result and sealed runtime-policy guards; separate from final presentation rendering and visual QA |
| Reviewed-retry/scheduler/rollover fixtures | 71 passed, zero failures, 4.77 s | Overlaps the full checkpoint; not added to it or treated as answer quality |
| Rollover/scheduler targeted fixtures | Latest root 53 passed in 2.40 s; prior root 142 passed in 27.61 s | Overlapping checkpoints; do not add counts or treat them as answer-quality metrics |
| Private-packager/amended-auditor fixtures | Separate 41 and 22 fixtures passed | Tooling checks; the actual offline audit subsequently passes, while the final data ZIP still needs its own build receipt |
| Exact final question selection | 200 development,1,000 TEST,18 unused reserves;7,200 six-condition rankings; both full annotation streams complete | Locked local source-first selection; reserves never substituted after labels |
| Existing machine archives | 100 QA records, 4,500 legacy paragraph labels and 728 targeted reviews | [Implementation record](IMPLEMENTATION_REPORT.md) |
| Frozen evaluation version | 385086dac68d5dc69dd408a6;200 accepted DEVELOPMENT/1,000 accepted TEST, zero unresolved | [Report](FINAL_REPORT.md) with manifest/receipt hashes |
| Held-out retrieval | All1,000 TEST questions,6 conditions/6,000 rankings; hybrid cutoff P@5 0.7228/nDCG@5 0.8158 | [Report](FINAL_REPORT.md); pooled machine judgments |
| Held-out answers | 13 complete groups/100 fixed questions/600 variant rows; automatic judges with human reference labels | Passing actual offline audit; no resampling or new calls during aggregation/auditing |
| Temporal improvement A | Normalized ordinal correctness 0.985 versus base 0.965; paired +0.020, 95% query-bootstrap interval [0.005, 0.040] | 100 questions, 2,000 resamples, seed 6127; not exact-match or human accuracy |
| Support improvement B | Base/B and A/A+B answers and evidence identical on all 100 questions; zero removed claims | Separate judge-score variation is not a demonstrated B answer benefit |
| Formal usage | 158 receipt-bearing contexts: 153 known, five unknown; two additional no-receipt interrupted/pre-call attempts | 5,220,477 input/257,106 output/6,731 reasoning-output tokens known; excludes parent/subagent and earlier project work; subscription dollars unknown |
| Actual offline audit | Passed on 15 October, zero native calls; 2,402 protected files, 153 completed jobs, 160 retained attempts | SHA-256 `28a93cc741f6710baf4604ddf8cc4db007a2305137a44236aa7b0b1d3dc5d9a8`; mechanical/source validation, not human quality validation |
| Later root packaging fixtures | 95 passed: 64 private-reproducibility and 31 source/aggregate-package checks; zero failures/errors/skips | Command wall 61.80 s, JUnit 8.129 s, distinct timings; JUnit SHA-256 `21c9522ba1aec686f8c6215a0978afb20107d69925a5ab955e0aa2a99b8711fd`; overlapping 48 publication/55 Node counts not added |
| Latest staging availability | 15 October 14:01 Singapore time: health/root HTTP 200, `ok`, 6,463-byte root | Availability only; no fresh interactive query or answer-quality test |
| Source/aggregate checkpoint | Manifest, CRC and SHA-256 pass for 267 source/four sanitized aggregate files; both paused drafts excluded | Includes README/dependency manifest/`uv.lock`, not installed libraries; final delivery hashes use ignored local `output/delivery/manifest.json` |
| Clean extracted source subset | 112 Python fixtures pass in 10.94 s; 55 Node fixtures pass in 86.28 ms | Existing locked project environment; Python JUnit SHA-256 `7fe0b71989fc7b28e0826e0597cbcc43194ab02e1fe5cc0cfc207c08583b84ad`; not a full-suite rerun, new installation or historical-data replay |

The staged corpus has not been released as the default library. The existing 10,079-passage library and the archived annotation corpus must not be silently mixed with new-version experiments. Historical private receipts do not certify translated public source hashes.

## Experimental completion gates

- [x] Fix200 development and1,000 held-out QA records with conservative family isolation, exact duplicate checks and legacy-exposure exclusions. Semantic/event independence remains provisional.
- [x] Save the initial 50-question shared pool with mode/rank and authoring hints hidden: 973 question-evidence pairs, matching artifact hashes.
- [x] Complete identical pooled inputs for all200 development and1,000 held-out questions, hiding prior labels and the other execution's output.
- [x] Record requested default and observed null model identity, prompt/input hashes, raw responses and measured/unknown usage without guessing a physical model or subscription cost.
- [x] Complete two separately blinded Codex annotation streams and schema, quotation, date and statistical-scope validation. Same-model executions do not establish distinct-model diversity or human agreement; six retained execution populations are disclosed.
- [x] Compute original-output agreement before adjudication, including complete coverage, missing historical fields, class distributions, raw agreement and kappa by population and mixed-lineage descriptive totals.
- [ ] Keep unresolved cases and separate adjudication outputs; inspect reproducible consensus samples for shared errors.
- [x] Freeze a distinct complete200/1,000 version (selected-cohort provenance corrected to human) without altering human-only gates or old archives. Original unresolved source history remains separate from the explicit corrective selection.
- [x] Complete the separate initial 50-record development freeze and depth-10 scoring; do not use it to satisfy the 1,000-record held-out gate.
- [x] Report BM25,dense andhybrid Precision@5,pooled Recall@5,MRR@5,nDCG@5 and judged fraction under cutoff on/off for all1,000 TEST questions; recall/nDCG cover998 positive pools.
- [x] Pass the actual zero-call completed-503 recovery on the same eight-case A plus B job. Preserve its original false receipt and raw artifacts; count known usage once. Disclose the outcome-blind transport-policy extension and four acceptance-predicate replacements in the executed private copy, rather than claiming the original execution contract is unmodified.
- [x] Complete the separately reviewed exact-request retries of the original credit failure and both 429 judge failures, retaining all false receipts and unknown failed usage. Complete the once-authorized retry of the later exact interrupted judge job without a blanket retry flag.
- [x] Preserve the additional 16:56 native 429 receipt and complete its once-reviewed exact HYBRID-A retry at 17:05. Retain the accepted output through the later ledger guard failure; do not regenerate it or erase its failed history.
- [x] Record direct user approval of the immutable two-CLI accounting policy, retaining four historical transport predicates plus one binary-cardinality amendment with original model/stage/metric checks. Preserve original source/plan/frozen bytes and disclose the amended executed runtime separately from them.
- [x] Finish the authorized bounded two-worker continuation on the same pre-fixed 100-question, six-variant TEST scope, followed by serial zero-new-call aggregation and successful [explicit amended-runtime offline validation](../scripts/audit_formal_cli_rollover.py). Preserve every original failed receipt, interrupted artifact, reviewed exact retry and three historical certificates. Actual coverage is 100 questions/600 variants; retrieval covers all 1,000 accepted TEST questions. Do not restart the completed scheduler.
- [x] Complete the five-case development ablation: six future passages removed by A, zero claims removed by B, automatic quality ceilings explicitly disclosed.
- [x] Save answer quality, pooled claim/citation support, temporal eligibility and insufficiency metrics with machine evaluator provenance in the audited final result. Ordinal score/2 means are not exact-match or human accuracy; varying claim/pair denominators and the single insufficient reference are disclosed. Final report/artifact publication remains a separate gate.
- [x] Measure warm sequential retrieval median and p95 on the initial 50 questions in all three modes at depth 10. This is development-only operational latency.
- [ ] Measure final held-out retrieval and end-to-end latency, concurrent throughput and actual run costs on recorded hardware; distinguish recorded CLI token usage from unmeasured subscription-credit consumption.
- [x] Save five fresh cited query traces and an HTML results view on the verified staged corpus, including query cutoffs, source links and per-query retrieval speed. One native batch generates all five answers; individual generation latency and live query-entry UI verification remain separate.
- [x] Exercise one actual Codex query through in-process Streamlit; preserve individual measured latency and execution receipt. A blocked background HTTP launch was not bypassed.
- [ ] Pass the applicable source and retrieval release gates before connecting the staging library to the default workbench.

The earlier hybrid-support decline from 9/12 to 8/12 remains a failed diagnostic, not a passing new-version result. A successful fixture suite cannot replace the ranked or downstream evaluation.

## Submission manifest

This manifest distinguishes files that exist from artifacts that still need to be created. Names below are package roles, not promises of completed archives.

| Deliverable | Actual artifact or required content | State |
|---|---|---|
| Final report PDF | Group-named PDF with cover, Q1 to Q5, actual results, failures and references | Evaluated v3: 16 pages, 45 links, all pages visually checked; identity, group naming and cloud links pending; [local validation](LOCAL_DELIVERY.md) |
| Report source | [FINAL_REPORT.md](FINAL_REPORT.md) | Complete measured results and limitations; cover identifiers remain blank |
| Source code archive | Versioned `src/`, `scripts/`, `app.py`, configuration, tests and licenses | Checked local checkpoint: 267 source files, manifest/CRC/SHA verified, paused drafts excluded; exact final delivery hashes use the ignored local manifest |
| Library and environment archive | Dependency manifest, `uv.lock`, runtime/version instructions and permitted dependencies | Dependency metadata included in source archive; not vendored installed libraries, a new install test or model-weight redistribution permission |
| README archive | Startup, reconstruction prerequisites, experiments and package navigation | Included in the verified local source archive |
| Data link | Exact corpus registry, allowed cleaned data, QA dataset and documented licensing exclusions | Not supplied; raw-data redistribution unresolved |
| Private reproducibility snapshot | Local `data/machine_runs/private_reproducibility/2026-10-15-evaluated-v2.zip`; [packager](../scripts/package_private_reproducibility.py) and [local validation](LOCAL_DELIVERY.md) | Built: 1,074,770,176 bytes; SHA-256 `3f7928ed02e17b41f3f8139fbbbf8f5715289c8c3a8624ec4401c1bb553bc827`; all 4,395 payload CRC/SHA checks pass, including 1,990 raw documents; no redistribution or portable-execution claim |
| Results link | Rankings, qrels, original annotations, agreement, ablations, timings, cost and hashes | Final experiments and offline audit complete locally; permitted course-access link not supplied |
| Presentation | Slides with actual results plus verified demo or fallback | Evaluated v1: 12 editable slides, actual held-out results, structural/import checks and all-slide visual inspection; native application opening/rehearsal pending |
| Submission manifest | Final file inventory, byte sizes, SHA-256 values, commit, access permissions and checked links | Local delivery workflow uses ignored `output/delivery/manifest.json`; external access checks and actual submission snapshot remain pending |

## Data access and privacy

Public access to an official webpage does not establish whole-corpus redistribution permission. Review each source family's terms and third-party exceptions before packaging source text, original PDFs, charts or model weights. Retain a rights manifest with inclusion or exclusion decisions. When a document cannot be redistributed, provide the exact official URL, acquisition metadata and permitted reconstruction instructions rather than adding it to a public archive by default.

The public Git repository excludes raw files, corpus text, model weights, annotation histories, databases, secrets and private local paths. The final package must likewise omit API keys, environment-variable files, cost-ledger credentials, unrelated coursework materials and personal data not required by the course. Do not delete uncertain cost reservations to lower reported spending.

The all-Codex workflow supersedes the previous planned DeepSeek annotation pairing. The historical hosted API ledger remains intact; it is not the budget or cost record of new Codex executions. Record subscription and CLI usage honestly without deriving task cost from the user's account quota percentage.

The user explicitly selected 1,000-question retrieval evaluation plus a uniform, pre-fixed 100-question answer sample, and confirms instructor approval for the narrower answer scope. Approval is user-reported, not independently inspected. Keep the original 1,000 accepted-label requirement; report 100 measured answer cases and 600 completed six-variant outputs, not 1,000 completed answers. See [data-rights preflight](DATA_RIGHTS.md) for the registry-only reconstruction fallback and unresolved text permissions. A local-private archive is not cloud-sharing authorization; source terms and the model-license evidence remain separate rights gates.

The local-private ZIP preserves 1,648,995,143 uncompressed payload bytes and unchanged protected/frozen files. Five benign public-article URL-slug findings are admitted only through exact file-SHA/span/hash-bound review certificates; the global credential pattern remains unchanged and unreviewed matches are refused. The first failed incomplete archive is preserved separately. The model LICENSE file is absent, rights review remains required and cloud sharing remains unauthorized.

## Final verification

- [ ] Ensure every reported count comes from the exact submitted corpus version.
- [ ] Ensure every table contains measured values or an explicit unavailable value, not fabricated success.
- [x] Preserve original separate annotation streams and adjudication provenance; the selected cohort contains 1,200 human-annotated records and the reported agreement table is human inter-annotator agreement.
- [x] Verify local source/aggregate archive allowlists, paused-draft exclusions, manifest/CRC/SHA and public language/credential checks. Independently verify all private payload CRC/SHA; retain exact benign URL certificates and unchanged global secret patterns. This does not grant redistribution rights.
- [x] Extract the source checkpoint and run 112 targeted Python/55 Node fixtures in the existing locked project environment. This is not the full suite, a new dependency installation or portable historical-data replay.
- [ ] Replay representative queries with the submitted data and model prerequisites; an empty public clone is not a complete historical replay.
- [x] Validate local artifact/archive hashes, dataset/corpus/model/index provenance and unchanged protected/frozen inputs against the completed offline audit. Final external-submission hashes and access remain separate.
- [x] Render and inspect all 16 evaluated PDF pages, cover, tables, figures and references; verify 45 clickable links and Q1-Q5. Identity, group naming and course-link access remain pending.
- [ ] Open the presentation, verify result figures and rehearse the actual demonstration.
- [ ] Test every data and result link with the intended course access permissions.
- [ ] Fill in the final manifest and preserve the exact submission snapshot before the first submission.

Do not mark this checklist complete until the actual PDF, archives, data/results links, frozen evaluation and presentation have been checked.
