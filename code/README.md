# Oil Evidence

**Annotation provenance corrected on 19 October 2026.** The project owner confirms that the selected 200 DEVELOPMENT and 1,000 TEST records were human annotated and that the reported 1,200-pair agreement table measures two independent human annotators. Earlier descriptions of this selected cohort or its agreement as machine annotation were incorrect. Automatic answer generation/judging, older pilot records and machine source checks retain their own provenance. Reported label values and metric values are unchanged; the human streams were not independently re-audited in this correction. Historical filenames, schema keys and hashes are compatibility references, not the corrected annotation origin.


A local evidence-retrieval and annotation workbench for crude-oil supply, policy and transport events. See [TASKS.md](TASKS.md) for implementation progress and [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) for coursework acceptance requirements.

## Public repository scope

This repository contains implementation code, tests, configuration, source seeds and progress documentation. Start with [current progress](docs/GITHUB_PROGRESS.md); [reports/progress.json](reports/progress.json) provides an aggregate machine-readable summary. Original PDF/HTML files, corpora, extraction caches, annotations, SQLite ledgers, indexes, model files and the original coursework assignment PDF are excluded. Paths under `data/` below refer to artifacts retained locally.

This is the English publication edition. Historical private receipts verify the preserved canonical bilingual implementation and its local artifacts, not the newly translated public interface or code. Exact historical replay requires the matching original implementation and the excluded raw-data snapshot. The publication edition does not rebuild or certify a new corpus.

The English edition passed 1,415 fixture tests with zero failures and three documented skips. Its synthetic workbench smoke and separate read-only official PDF sample check also passed. The current public tree contains no literal Han text. See [publication boundaries](docs/PUBLICATION_EDITION.md) and [the publication verification receipt](reports/english_publication.json). These local checks are separate from historical corpus results and the language-only GitHub Actions workflow.

A fresh clone can install the locked dependencies and run unit tests against fixed fixtures. Reconstructing historical corpora, querying an existing knowledge base and replaying annotations also require the corresponding local originals and frozen snapshots. Publishing the code does not publish a complete data package or establish coursework acceptance.

The current uniform staging corpus has a local 384-dimensional index for 11,815 passages, but has not been published or selected as the default library. It still contains 1,562,715 words, with an 18.15% passage-count margin; the 160/65/240/480 segmentation parameters are unchanged. Byte-identical source reconstruction passed before and after indexing, with matching vector, corpus, passage-order and model hashes. Related tests passed 268 checks, and 24 real keyword, dense and hybrid retrieval runs passed cutoff-date, region and source-offset checks. These verify operation and integrity, not retrieval accuracy or release approval. The historical source-closeout full suite remains 1,171 passed with one host-permission skip. Fixed machine-review warnings and the historical 9,452-passage release rejection after hybrid support fell from 9/12 to 8/12 remain recorded. The default 10,079-passage library, labels and ledgers are unchanged. The historical index-build step did not start the workbench; the later staging service has a separate availability check below. See [uniform corpus implementation](docs/UNIFIED_CORPUS.md). `oil build --run RUN` creates a staging version without directly replacing the default library.

Fifty initial development questions are fixed: 16 factual, 14 temporal, 10 comparison and 10 evidence-sufficiency challenges. They produced 150 real three-mode retrieval runs, 1,500 ranked hits and a blinded union of 973 question-evidence pairs. Both fresh Codex annotation streams and all 50 source-based adjudications are complete. Their machine-only development version is frozen and scored at depth 10: hybrid pooled recall 0.691 and nDCG 0.729, versus dense precision 0.470 and MRR 0.853. This is development evidence, not a completed held-out test. Authoring anchors remain separate from annotation inputs, and question categories are not answer labels. See [development question preparation](docs/DEVELOPMENT_QUESTIONS.md) and [the draft report](docs/FINAL_REPORT.md) for exact inputs, machine agreement and metric conventions.

## Current subscription-backed workbench

Current formal checkpoint: both annotation streams and source review cover the fixed200 DEVELOPMENT/1,000 TEST bank. Human-annotated cohort (historical version `385086dac68d5dc69dd408a6`) is frozen with zero unresolved records. Retrieval evaluation completes all1,000 TEST questions in six top-five conditions. Hybrid with cutoff filtering scores Precision@5 0.7228, pooled Recall@5 0.6960, MRR@5 0.9442 and nDCG@5 0.8158. These use human pooled judgments; recall/nDCG cover998 positive pools.

Snapshot: **2026-10-15 Singapore time**. Local engineering and the formal experiment with human reference labels and automatic judges are complete: 13 whole groups cover the preselected 100 TEST questions and all 600 six-variant answer rows. Serial aggregation and the independent offline audit pass with zero additional model calls. Result SHA-256 is `778b77441332739930af71fa75bb1e4929b36a3993be0cdd8498c09214d45929`; audit SHA-256 is `28a93cc741f6710baf4604ddf8cc4db007a2305137a44236aa7b0b1d3dc5d9a8`, covering 2,402 protected files, 153 completed jobs and 160 retained attempts. All original failures and three historical completed-503 certificates remain preserved. The evaluated PDF, presentation, local-private ZIP and source/aggregate checkpoint archives are checked. Cover identifiers, redistribution review, course links, rehearsal and submission remain pending. See [current progress](docs/GITHUB_PROGRESS.md), [submission checklist](docs/SUBMISSION_CHECKLIST.md) and [the report source](docs/FINAL_REPORT.md).

Temporal filtering A raises mean normalized ordinal machine correctness from 0.965 to 0.985 on the fixed 100 questions (paired difference +0.020, query-bootstrap 95% interval [0.005, 0.040], 2,000 resamples, seed 6127). This is not exact-match accuracy or human accuracy. B changes no answers or evidence: base/B and A/A+B are identical on all 100 questions, with zero removed claims. Small independent judge-score differences do not establish a B benefit. Only one sampled reference is insufficient, so its 1/1 correct-abstention result cannot establish general refusal quality.

The evaluated report (16 pages) and editable presentation (12 slides) now contain the complete results and have passed page-by-page/slide-by-slide visual inspection. Exact local filenames, hashes and validation limits are in [local deliverables](docs/LOCAL_DELIVERY.md). Output files are excluded from public Git. Cover identifiers, group-specific naming, permitted cloud links and course submission remain pending.

The later root packaging checkpoint passes 95 fixtures: 64 private-reproducibility and 31 source/aggregate-package checks. Its recorded command wall time is 61.80 seconds, while JUnit reports 8.129 seconds; these are distinct timing measurements, not model latency. JUnit SHA-256 is `21c9522ba1aec686f8c6215a0978afb20107d69925a5ab955e0aa2a99b8711fd`. The separate 48 publication and 55 Node fixtures overlap other checkpoints and are not added. Fixtures alone do not certify an actual final ZIP.

The actual local-private ZIP is built and independently checked: 1,074,770,176 bytes, 4,395 payload files including 1,990 raw documents, SHA-256 `3f7928ed02e17b41f3f8139fbbbf8f5715289c8c3a8624ec4401c1bb553bc827`. All payload CRC/SHA checks pass and protected/frozen files are unchanged. Five benign public-article URL-slug findings have exact file-hash/span/hash-bound review certificates; the global credential pattern is unchanged and unreviewed matches are refused. The failed incomplete archive is preserved separately. This archive remains local: the model LICENSE file is absent, rights review is required, cloud sharing is unauthorized and portable execution is unproven. See [local deliverables](docs/LOCAL_DELIVERY.md).

The source/aggregate checkpoint passes manifest, CRC and SHA-256 verification for 267 source files and four sanitized aggregate files, with both paused drafts excluded. Source includes the README, dependency manifest and `uv.lock`, not vendored installed libraries. Its clean extracted source passes 112 targeted Python fixtures in 10.94 seconds and 55 Node fixtures in 86.28 milliseconds using the existing locked project environment. Python JUnit SHA-256 is `7fe0b71989fc7b28e0826e0597cbcc43194ab02e1fe5cc0cfc207c08583b84ad`. This is not a new dependency installation, full-suite run or portable historical-data replay. The delivery workflow records final archive names, commit and hashes in ignored local `output/delivery/manifest.json`, avoiding a recursive public archive-hash binding.

The [bounded parallel scheduler](scripts/run_parallel_formal_jobs.py) retains the pinned original preflight, groups, batching, requests and stage processing. An exclusive owner and per-job ownership guard prevent duplicate execution. A hash-bound manifest permits one retry for each explicitly approved failed job only after checking its entire original artifact inventory and descriptor; other failures remain blocked. Completed jobs reuse saved outputs. After workers finish, one serial, zero-new-call pass validates and aggregates the saved jobs. The historical outcome-blind terminal-503 policy still changes four acceptance predicates in a private driver copy without rewriting the original driver or false receipts.

An app update changed the native CLI binary from `d13914ced6c7af174d8d638db938284231227312104b06a25e9608c632a7daec` to `e98c516f1c8935789aabe9122ee7e05ec4c3efd1ba6acf179242e031c495b99e`, despite the same `0.162.0-alpha.17.2` version string. The old binary was not found locally. The user directly approved the [sealed two-CLI accounting amendment](scripts/formal_cli_rollover.py), policy SHA-256 `7edd4291f9419f37f9d495e1ff1b653a2c15e4122bdba13bcab7a6b98f96b96d`. The executed private driver retains four historical transport predicates and changes one additional usage-binary-cardinality predicate; the known-model guard and all other stage/metric checks remain. Original files, plan and frozen input bytes are unchanged, but the executed runtime contract is explicitly amended, not an unmodified single-binary experiment. Requested model remains null and physical model identity unknown. New-binary 503 outputs remain ineligible for the old transport policy. The [amended offline auditor](scripts/audit_formal_cli_rollover.py) requires explicit policy/source pins and all four historical failed receipts; it cannot promote partial data or create transport certificates.

The latest full fixture checkpoint, recorded on 14 October, passes 2,644 tests with three documented skips and zero failures/errors in 680.89 seconds, including the reviewed-retry fixtures. The separate Node slide-data suite passes 55 fixtures. Earlier full and targeted checkpoints overlap and are not added. These are engineering checks, separate from the completed machine answer evaluation and 15 October offline audit. Formal usage retains 158 receipt-bearing execution contexts: 153 with known usage and five unknown, plus two interrupted/pre-call attempts without receipts. Known totals are 5,220,477 input, 257,106 output and 6,731 reasoning-output tokens, counted once; parent/subagent and earlier project usage are excluded. No new hosted API calls are made; physical model identity and subscription dollar cost remain unknown.

On the original local checkout, retrieval alone can be reproduced without new model calls. This creates a new result directory and does not retry the failed answer job:

```powershell
$env:PYTHONPATH='D:/coursework/AI6127/oil-evidence-english/src;D:/coursework/AI6127/oil-evidence-english/scripts'
& 'D:/coursework/AI6127/oil-evidence/.venv/Scripts/python.exe' -B -X utf8 scripts/run_machine_experiments.py --canonical-root 'D:/coursework/AI6127/oil-evidence' --corpus-run semantic-page-v2-buffer run --frozen-dir 'D:/coursework/AI6127/oil-evidence/data/machine_runs/frozen_evaluation/385086dac68d5dc69dd408a6' --retrieval-run 'D:/coursework/AI6127/oil-evidence/data/machine_runs/candidate-retrieval/2026-10-14T04-17-34.486189+00-00-e3bbbc24bb' --max-new-calls 0
```

Successful retrieval-only output ends with `partial_new_attempt_limit_reached`, 1,000 completed retrieval questions and zero native attempts; that separate command's status does not replace the completed formal result. The formal run remains under `data/machine_runs/formal-experiments/2026-10-14T05-29-35.082875+00-00-406a2e1b`. Its original credit failure and all three later native 429 failures received separately reviewed, exact-request retries; all failed receipts remain intact. The parallel continuation is finished: do not restart it or reissue completed jobs. Do not resample, alter the model request, delete history, purchase credits automatically or infer a reset schedule.

The user reports instructor approval for solo completion and machine annotation, and selected Codex instead of a hosted API for the new workflow. The English workbench defaults to Codex using the native CLI's existing ChatGPT login. It never reads authentication files or copies API keys. Fresh structured calls preserve original request/output bytes, exact-source checks, emitted token usage and execution receipts. An unknown runtime model identifier remains null; separate contexts do not prove model diversity or human agreement.

The staging UI uses the verified 11,815-passage index explicitly, without replacing the historical default corpus. On the existing local Windows checkout, launch it with the project interpreter:

```powershell
$env:PYTHONPATH='D:/coursework/AI6127/oil-evidence-english/src'
& 'D:/coursework/AI6127/oil-evidence/.venv/Scripts/python.exe' -X utf8 scripts/start_codex_workbench.py --canonical-root 'D:/coursework/AI6127/oil-evidence' --port 8518
```

This command is for the existing local data checkout, not a data-free clone. Keep the process running and visit [the staging workbench](http://127.0.0.1:8518). In-process Streamlit testing completed one actual Codex query: five evidence passages, two cited claims, 15.918344 seconds end to end. The earlier denied background launch and 14 October availability check remain historical. The latest readiness check at 14:01 Singapore time on 15 October returns HTTP 200 for health (`ok`) and root (6,463 bytes). This is availability, not a fresh interactive query or quality test. No paid external API was called; subscription dollar/credit cost is unknown, not zero.

New machine-cohort freezing and scoring use separate scripts: `adjudicate_machine_annotations.py`, `freeze_machine_dataset.py` and `report_machine_retrieval.py`. They do not relax the legacy human-freeze path or count old passage-topic labels as QA records. The completed formal evaluation retains 1,000 accepted TEST records. The earlier `run_codex_ablations.py` five-case development diagnostic remains separate: four saved generator executions plus three fresh support/judge calls are counted once. A five-query cited demonstration, verified evaluated report, editable 12-slide presentation and checked local archives exist. Local engineering, formal evaluation, offline audit and artifact inspection are complete; external submission is not.

## Windows startup

After installing the environment, double-click `start-workbench.cmd` in the project root. Keep that window open while using the application: closing it or pressing Ctrl+C stops the local server. If the browser reports that the page is unavailable, check whether the server is running. Opening a browser tab does not start the backend.

Open PowerShell in the project checkout:

```powershell
uv sync --locked --group dev
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8517 --browser.gatherUsageStats false
```

Visit [the local workbench](http://127.0.0.1:8517). The project uses its own Python 3.12 `.venv`, not an environment from another project. `uv.lock` pins the dependencies.

The sidebar accepts an API key without echoing it; the key is available only to the current local server process. Evidence-only preview is also available without an API. The server binds only to 127.0.0.1. Do not expose this unauthenticated local application directly to the public internet.

## Default library and historical replay

The sentence-based resegmentation version was published locally on 2026-10-11: 10,079 passages, 1,406,498 words and 26,431 unique words from 1,217 contributing documents. The original download count remains 1,434. That round added no source documents or Hugging Face data. Passage growth came from complete-sentence segmentation, not additional information. It met the numerical requirements of >=10,000 passages and >=100,000 words, but did not establish corpus-wide semantic quality, human annotation, formal evaluation or overall assignment acceptance. Its main report is `data/reports/corpus_resegmentation.json`, not the historical v2 `corpus_expansion.json`.

The process replaced 2,030 parent passages with 4,060 non-overlapping, 80-160-word children. Parents remain in frozen versions and interface context, but are not indexed or counted alongside their children. It quarantined 762 original passages: 521 involved in pending near-duplicate candidates, 143 from finance catalog sources awaiting extraction, 81 with structural defects and 17 with observed defects. Original text and exclusion reasons were retained. All retained word order, exact character offsets, sources, pages and dates are unchanged. Subtracting the quarantined 126,002 words from the original 1,532,500 gives the current word count.

The original 4,500 paragraph labels, 100 QA records, 728 reviews and their hashes were not rewritten and still refer to their original text version. Children do not inherit parent labels. Actual label coverage is 3,135 / 10,079, with 6,944 active passages lacking individual labels. The interface distinguishes the old label archive from current coverage. Source files, quarantine records, index and checks are frozen under `data/resegmentation_runs/semantic-80-160-v1/published/<corpus_sha256>/`; the 8,811-passage parent version is in that run's `baseline/`. The original annotation corpus remains in the v1 `baseline/`.

The following is a historical execution record. Do not rerun `stage` or `publish` for a published run. Single-writer locking, source/annotation/hash gates, the publication journal and exception rollback were verified. Later versions must explicitly preserve the text referenced by labels and parent-child mappings; do not overwrite the current version with an ordinary `oil build` or the old workflow.

```powershell
# Historical order: snapshot -> stage -> index -> audit --require-index -> publish
# These writes are complete. Do not rerun stage/audit/publish for semantic-80-160-v1.
# Replay passed with the historical frozen implementation; current code has changed.
# A matching old implementation is required, and hash mismatches fail closed.
.\.venv\Scripts\python.exe -X utf8 scripts/reproduce_resegmentation.py
# Workbench and retrieval-rule checks, without paid model calls
.\.venv\Scripts\python.exe -X utf8 scripts/check_workbench.py
.\.venv\Scripts\python.exe -X utf8 scripts/check_resegmentation_queries.py
```

The recorded replay passed. Outputs under `reproductions/<corpus_sha256>/` match the published corpus and quarantine records byte for byte. Source integrity, index order and 384-dimensional vectors match. The historical checks passed 332 tests, with one Windows symlink-permission skip; 30 BM25/dense/hybrid cutoff and lineage checks, four application tabs and parent-context display also passed. Reports are `data/reports/resegmentation_quality.json`, `resegmentation_queries.json`, `resegmentation_reproduction.json` and `workbench_checks.json`. Fixed machine spot checks preserved initial failures and withdrew bad partitions; this was not exhaustive passage-level human semantic review. Mixed topics, partial truncation, footnotes and temporal upper bounds remain limitations. That round made zero external model API calls and created zero new human labels. Codex subscription usage was not measured.

### Historical second official expansion

The following records the 2026-10-10 v2 expansion to 8,811 passages. Its counts and reports are not the current state. Existing machine labels refer to the original fixed corpus, not evaluation ground truth for the expanded library.

Do not rerun `publish` for an already published version. To reproduce previously downloaded corpus and index artifacts, the historical chain starts at `stage`, writing into that run's `stage/` without changing the default library. Future expansion must snapshot the current library under a new `--run` name instead of reusing v1/v2 baselines; default run names exist only for old-workflow compatibility.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/expand_corpus.py snapshot --run official-2018-2026-v2
.\.venv\Scripts\python.exe -X utf8 scripts/expand_corpus.py discover-extra --run official-2018-2026-v2
.\.venv\Scripts\python.exe -X utf8 scripts/expand_corpus.py discover-eia --run official-2018-2026-v2
.\.venv\Scripts\python.exe -X utf8 scripts/add_wb_observed.py
# Collect sequentially with per-series download limits; do not start concurrent writers.
.\.venv\Scripts\python.exe -X utf8 scripts/collect_expansion_batch.py --run official-2018-2026-v2 BSEE-HURRICANE DOE-SPR BOEM-POLICY AU-REQ UK-DUKES UK-ENERGY-TRENDS EIA-METHODS EIA-FINANCE EIA-IEO WB-CMO
.\.venv\Scripts\python.exe -X utf8 scripts/apply_source_page_exclusions.py
.\.venv\Scripts\python.exe -X utf8 scripts/validate_new_sources.py --run official-2018-2026-v2
.\.venv\Scripts\python.exe -X utf8 scripts/expand_corpus.py stage --run official-2018-2026-v2
.\.venv\Scripts\python.exe -X utf8 scripts/audit_expansion.py --run official-2018-2026-v2
.\.venv\Scripts\python.exe -X utf8 scripts/expand_corpus.py index --run official-2018-2026-v2
# Already executed once and frozen. Do not republish this run.
.\.venv\Scripts\python.exe -X utf8 scripts/expand_corpus.py publish --run official-2018-2026-v2
.\.venv\Scripts\python.exe -X utf8 scripts/check_workbench.py
.\.venv\Scripts\python.exe -X utf8 scripts/check_expansion_queries.py
```

`data/expansion_runs/official-2018-2026-v1/baseline/` retains the original 4,500-passage annotation corpus; the v2 `baseline/` retains the previous 6,392 passages and index. Each `published/<corpus_sha256>/` holds that version's corpus, source manifest, index manifest and quality reports, and must not be overwritten with different contents. Original PDF/HTML files remain under `data/raw/`. Discovered web pages may change; frozen local originals and implementation hashes define historical reproducibility.

That expansion added US Strategic Petroleum Reserve material, offshore hurricane shutdowns, BOEM policy/production analysis, UK energy statistics, Australian oil chapters, EIA oil-model methods/finance and World Bank 2025 reports. Methods reports explain supply mechanisms; they are not evidence that an event occurred or that production was observed. It quarantined 117 new originals with uncertain historical versions rather than admitting them by assigning later publication dates. Originals were retained. Visual PDF checks led to oil-chapter restrictions, table/dictionary/reference exclusions and year-preservation fixes. OPEC access and bulk-use restrictions were not bypassed. Whole-corpus redistribution, near-duplicate adjudication and human spot checks remain unresolved.

```powershell
.\.venv\Scripts\oil.exe discover
.\.venv\Scripts\oil.exe collect --limit 6
# Current build creates a uniform staging version. Use a new run; do not replace the default.
.\.venv\Scripts\oil.exe build --run new-uniform-run
.\.venv\Scripts\oil.exe index
.\.venv\Scripts\oil.exe query "What factors affect crude oil supply and inventories?" --as-of 2026-08-01 --verify
.\.venv\Scripts\oil.exe pilot --limit 50
.\.venv\Scripts\oil.exe evaluate data/annotations/pilot.jsonl --smoke --limit 5
.\.venv\Scripts\oil.exe status
.\.venv\Scripts\python.exe -m pytest -q
```

`discover` and `collect` are source entry points; the current `build` creates a uniform staging version. Existing labels must not be automatically inherited or old chunks mixed in through a legacy incremental path. Replace `new-uniform-run` with an explicit new run name and follow the uniform workflow's audit, comparison and release gates. `collect --limit N` adds up to N new documents; previously downloaded files with matching hashes are skipped. `pilot` preserves an existing review queue. Cleaning changes the corpus hash, so an old vector index will refuse to load and must be rebuilt. Downloads, models and data stay in this project rather than a global environment.

## Model services

The canonical local archive contains 100 Codex machine annotations: review of 50 Flash candidates and completion of 50 original pilot questions. The annotation-review view defaults to the completed Codex records, with filters for corrections, insufficient evidence and records not recommended for formal evaluation. Review notes, English reference answers and source quotes are separate; raw JSON remains available in a collapsed view. The historical canonical review notes are in Chinese and are excluded from the public repository.

Those historical judgments were produced by Codex reading fixed evidence, without an external API. They used Codex subscription credits; this application does not measure that usage. Online query settings below remain separate. Provenance is explicitly `codex_model_annotated`, not human ground truth or a second human annotator.

Outputs are in `data/annotations/codex.jsonl`, immutable audit versions in `data/annotations/codex_runs/`, and statistics in `data/reports/codex_annotation.json`. `scripts/codex_labels.py` displays fixed reading material and validates/publishes decisions already written. It does not invoke Codex, produce semantic labels or convert subscription credits into API calls. Original `pilot.jsonl`, `candidates.jsonl` and human review history remain separate.

The historical API workflow used DeepSeek V4.1 Flash (interface alias `deepseek-flash`) for all roles, with a cumulative USD 5 ceiling. It remains optional and separate from the current subscription-backed Codex workflow. Without a configured API, the historical hosted path offers evidence-excerpt preview rather than presenting it as a synthesized answer or an answerability judgment.

`config.toml` defines a `/chat/completions`-compatible endpoint, generation/verification role models, the spending ceiling and per-million input/output token rates. Separate requests to the same Flash model do not establish independent-model agreement. Remote endpoints must use HTTPS. Read keys only from `OIL_LLM_API_KEY`; do not put secrets in code or chat. `scripts/run_model_pilot.py` also accepts a non-echoing key that exists only for the current process.

The SQLite cost ledger reserves conservative upper bounds transactionally and retains reservations for failed requests to prevent duplicate spending when billing is uncertain. Usage accumulates across runs; deleting the ledger must not erase the accounting history. The recorded conservative estimate uses official peak, cache-miss Flash rates checked on 2026-10-06: USD 0.30/M input and USD 1.20/M output tokens. Final provider billing takes precedence. See [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/). Thinking is explicitly disabled by default so short structured outputs do not silently spend the output allowance on reasoning.

```powershell
# Run only after configuring the service and local environment variable.
.\.venv\Scripts\oil.exe query "What does the evidence establish about oil supply?" --generation llm --verify
.\.venv\Scripts\oil.exe generate --limit 50
.\.venv\Scripts\oil.exe export-reviews --dataset pilot
.\.venv\Scripts\oil.exe agreement
# After reconciling multiple annotator exports, submit a unique adjudicated record list.
# Insufficient counts, unreviewed records or unconfirmed groups fail the freezing gate.
.\.venv\Scripts\oil.exe freeze data/annotations/adjudicated.jsonl
.\.venv\Scripts\oil.exe evaluate data/frozen/VERSION/test.jsonl --generation llm
```

## Data and interfaces

Paragraph labels identify source-text topics, entities, oil-task relevance, forecast/observed/mixed status, quantities/units and dates. They are separate from the 100 QA annotations. The interface counts only validated outputs; progress is in `data/reports/passage_annotation_progress.json`. Fixed inputs, batch decisions, outputs, audits and revisions are under `data/annotations/passage_label_runs/e546fc6c7f1e48b7/`. A partial run must not be published as all 4,500 passages, and the original corpus is not rewritten. Empty quantity/event arrays mean none were selected, not that the passage is unlabelled. Machine outputs do not count as human labels.

The first 4,500-passage publication is version `e4158423d4e7b3a5`, whose immutable snapshot remains unchanged. All 728 flagged passages received a machine second pass: 452 confirmed, 20 corrected and 256 unresolved source checks. The merged version is `034b1bc8d6575322`; complete labels are in `data/annotations/passage_labels.jsonl`, 728 adjudications in `data/annotations/passage_review.jsonl`, and statistics/hashes in `data/reports/passage_review.json`. The paragraph-label view exposes second-pass results and an ambiguity/review filter.

Second-pass inputs, blinded labels, comparisons, adjudications and snapshots are in `data/annotations/passage_review_runs/e4158423d4e7b3a5/`; `published/034b1bc8d6575322/` is the complete published version. The parent agent reread two coarse-policy topic judgments and appended revisions without overwriting original decisions or audits. The 256 unresolved items are not 256 confirmed errors: they include source-version, statistical-scope, table-extraction and conservatively retained warnings about missing non-oil rows. The 200 advisory strict-evaluation exclusions overlap those statuses; they do not delete source text or automatically filter retrieval. That review made zero external API calls; Codex subscription usage was unmeasured and human review remained zero.

- `data/sources.jsonl`: discovered sources, titles, date evidence and redistribution flags.
- `data/raw/` and `data/documents.jsonl`: originals, URLs, SHA-256, collection time and byte counts.
- `data/passages.jsonl`: text, document/passage IDs, pages, dates and regions. Region tags are keyword heuristics, not human factual labels.
- `data/index/`: 384-dimensional English semantic vectors and corpus/model/vector hashes, using native NumPy similarity. Reciprocal rank fusion combines BM25 and vector candidates.
- `data/annotations/` and `data/annotations.sqlite3`: candidate questions, review queues and append-only human edits.
- `data/runs/` and `data/experiments/`: query outputs and hashes of experiment configuration, corpus, code and dataset.

The Python entry point `oil_evidence.query.query(question, mode, as_of, region, temporal, verify, generation)` returns `claims`, `evidence`, `removed_claims`, `verification_method`, `model_calls`, `retrieval_seconds` and `total_seconds`. `temporal` and `verify` control improvements A and B respectively.

Data uses JSON/JSONL. Model suggestions, web pages and PDFs are untrusted source material, not execution instructions. SQLite stores annotation and cost ledgers, not the text-retrieval index.

## Experiment constraints

Chinese-query performance of the English BGE model is unverified; formal questions use English. Retrieval hits are not answer-correctness judgments. Improvement B in excerpt-preview mode checks literal quotations only and does not establish semantic verification effectiveness.

The original human and machine freeze implementations remain separate. The delivered selected cohort has 200 DEVELOPMENT and 1,000 TEST human-annotated records, and its reported agreement table compares two independent human annotators. Historical machine-named paths and manifests are retained compatibility references. Automatic answer judges remain machine outputs. Recall covers judged candidates rather than all relevant corpus passages; report judged_fraction. Develop on the development set and do not repeatedly tune using test results.

## Access and copyright

Honor robots.txt, per-host rate limits, failure records, document-size caps and per-invocation download budgets. Do not change identities to bypass 403 responses. Preserve EIA attribution and third-party chart copyright notices. Public access to OPEC or policy documents does not grant whole-corpus redistribution rights. Check licenses before submitting originals; a coursework cloud-link requirement does not itself grant permission to redistribute.

## Outstanding acceptance requirements

The student reports instructor approval for solo work and the fixed 100-question answer subset; this is not independently verified correspondence. The selected dataset is human annotated per the owner correction. Local engineering, the human-annotated dataset, 1,000-question retrieval experiment, 100-question six-variant answer experiment, offline audit and local artifact/archive checks are complete. Data-rights review, accessible course submission links and native presentation rehearsal remain separate gates. Group number, name and matriculation number remain intentionally blank. No coursework submission has been made.
