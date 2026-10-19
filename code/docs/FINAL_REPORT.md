# Oil Evidence Coursework Report

Course: AI6127 Natural Language Processing

Group number: ____________________

Student name: ____________________

Matriculation number: ____________________

## Abstract

Oil Evidence is a local retrieval augmented generation (RAG) workbench for questions about crude-oil supply, production policies, sanctions and transport disruptions. Its central requirement is to explain what the available evidence establishes at a specified cutoff date. The system combines keyword and semantic retrieval, presents traceable evidence, and optionally generates cited claims with a second support-checking pass.

The verified staging knowledge base contains 11,815 passages, 1,562,715 words and 28,582 unique word types from 1,527 contributing documents. A local 384-dimensional index supports three retrieval modes. The delivered dataset contains 1,000 TEST records with human annotations. On all 1,000 TEST questions, hybrid retrieval with cutoff filtering achieves Precision@5 0.7228, pooled Recall@5 0.6960, MRR@5 0.9442 and nDCG@5 0.8158, with zero returned post-cutoff or unknown-date passages. In the separate, pre-label uniform sample of 100 TEST questions, automatic correctness rises from 0.965 for hybrid base to 0.985 with cutoff filtering. The support checker changes none of the 100 base or 100 cutoff-filtered answers; its small score difference is judge variability, not demonstrated answer improvement. Retrieval uses human reference judgments; answer-quality scores use automatic judges and are not human-rated answer accuracy. Original frozen files remain unchanged.

The 1,000 TEST records have human reference annotations. Questions, labels and TEST evaluation scores remain unchanged.

## Q1 Knowledge base and application

### Application scope

The workbench supports factual, temporal, comparative and evidence-sufficiency questions. Example tasks include retrieving an official production outlook, separating an announced policy from evidence of implementation, comparing supply disruptions, and identifying when the documents cannot support a causal or quantitative conclusion. A query includes an English question, an optional region filter and an information cutoff date.

The fixed collection window is 1 January 2018 through 5 October 2026. Publication date, conservative availability date, event date and the period described by a statistic are separate fields. This distinction is important because a forecast for a future year can be legitimate evidence before that year, whereas a later publication cannot establish what was available at an earlier cutoff. [2]

### Sources and collection

The corpus draws on official energy and policy material. Its source families include the US Energy Information Administration's outlooks, oil articles, country analyses and methodology; US Treasury and Office of Foreign Assets Control policy announcements; Strategic Petroleum Reserve material; offshore policy and production reports; Dallas Federal Reserve energy surveys; UK and Australian oil statistics; and World Bank commodity reports. Different source families provide outlooks, reported observations, legal actions and mechanism background; these roles are not interchangeable. Exact current contribution counts by source should accompany the final source-registry export, rather than be inferred from earlier expansion counts. [2]

Three source-registry examples are EIA's crude-price and refinery-margin article dated 5 October 2026, Treasury's announcement concerning an oil-smuggling network dated 26 February 2018, and DOE's Strategic Petroleum Reserve crude-sale notice dated 28 February 2019. Their exact official URLs are retained below. They identify actual documents admitted to the selected corpus, not examples of completed answer evaluation. [13][14][15]

Collection records the requested URL, source identity, title, publication and availability evidence, collection time, bytes and SHA-256 hash. Requests respect robots restrictions, a one-second minimum interval per host, a 25 MB document limit, a 300 MB invocation limit and limited retries. [2]

### Cleaning and passage construction

Cleaning extracts approved article bodies and oil-related PDF sections, removes repeated navigation and obvious nonbody content, and quarantines uncertain tables, charts, reference lists and source versions. Every admitted body uses the same complete-sentence segmentation policy: a 160-word target, soft bounds of 65 and 240 words, and a 480-word context safety ceiling. These are structural targets rather than a requirement that every passage have the same length. Short complete statements and long indivisible sentences can differ in length while retaining consistent admission rules.

Passages do not overlap. Each passage retains its document and source-page identifiers, physical page or article body, exact character offsets, text, source URL, dates and regions. Exact deduplication is performed; semantic near-duplicate review remains limited. Historical parent passages are not indexed or counted alongside their replacement children. Newly created passage IDs do not inherit old machine labels. [2]

### Corpus statistics

| Quantity | Verified staging value |
|---|---:|
| Admitted passages | 11,815 |
| Word tokens | 1,562,715 |
| Unique word types | 28,582 |
| Contributing documents | 1,527 |
| Cleaned source pages | 5,703 |
| Passage margin above 10,000 | 1,815, or 18.15% |
| Vector dimensions | 384 |

Word statistics use the project's lowercased alphanumeric tokenizer, which retains selected internal punctuation. The unique-type count is a tokenizer-dependent vocabulary statistic, not the number of distinct concepts. The passage and word thresholds are met numerically. Fixed machine source checks preserve 68 original cases and diagnostics, with no remaining known critical issue in that sample and 34 retained warnings; this does not establish a corpus-wide error rate. [1]

The 11,815-passage corpus is a verified staging candidate and has not replaced the 10,079-passage default workbench library. Corpus-size claims, demonstration outputs and experimental results must name the corpus they actually use. Raw-document redistribution rights remain unresolved, so no public raw-data archive is provided.

## Q2 Retrieval system and interface

### Retrieval methods

The sparse baseline uses BM25 keyword retrieval over tokenized passage text. This provides inspectable matches for names, units and policy terms. The dense baseline uses the local English model `BAAI/bge-small-en-v1.5`. Passage and query vectors are normalized, and their dot product supplies cosine-similarity ranking. The verified index stores an 11,815 by 384 float32 matrix; model-file, corpus, vector and passage-order hashes guard against stale or mismatched indexes. [1][2]

Hybrid retrieval combines sparse and dense ranks using reciprocal rank fusion (RRF). Each candidate receives a contribution `1 / (60 + rank)` from each ranking in which it appears. This follows Cormack, Clarke and Buttcher (2009), without assuming that BM25 and cosine-similarity scores share a scale. The candidate pool is 50 per mode and the displayed result count is five. Eligibility filters precede final ranking. The 1,000-question TEST evaluation below compares methods using fixed pooled human judgments. [4][24][31]

### User interface and five demonstration questions

The Streamlit interface provides query entry, cutoff and region controls, evidence cards, original-source links, machine-annotation review and passage-label inspection. Evidence cards expose identifiers, source dates and full text so that users can inspect claims rather than rely on a fluent answer alone. The new staging interface defaults to subscription-backed Codex using its existing native ChatGPT login, with no API-key input. It reads the selected 11,815-passage index without changing the historical default library. The application binds to localhost and is not an authenticated public service. [3][17]

Five fresh demonstration queries were run on the verified 11,815-passage staging corpus using hybrid retrieval at depth five. The following questions come from the public seeds; they are demonstration choices, not test-set ground truth. [5][16]

| Question | Cutoff | Demonstration purpose |
|---|---|---|
| What does EIA project for US crude oil production, and in what units? | 2026-10-05 | Preserve quantities, units and forecast periods |
| As of 2025-06-30, what oil supply outlook had EIA published? | 2025-06-30 | Retrieve historical information rather than the latest report |
| What evidence shows that an announced OPEC production adjustment was implemented? | 2026-10-05 | Separate announcement from implementation and possible insufficiency |
| Which sanctions affecting Russian oil were effective as of 2025-02-01? | 2025-02-01 | Inspect legal dates and the scope of the policy evidence |
| What exact Brent price will occur next month, according to verified evidence? | 2026-10-05 | Avoid presenting a forecast as a certain future observation |

The five cases retrieved 25 passages and generated five cited answers in one fresh Codex execution. Retrieval took 0.087971, 0.062281, 0.068672, 0.064153 and 0.081456 seconds respectively, after an excluded warm-up. Shared batch generation took 29.931725 seconds and emitted 27,281 input and 1,240 output tokens. That batch duration is not five independently measured generation latencies; per-query generation and end-to-end values remain unmeasured. All 14 saved artifact hashes matched their receipt. [16]

The first answer cites dated EIA production forecasts and explicitly says they are not a current 2026 outlook. The historical-outlook answer also states that retrieval did not establish the latest June 2025 outlook. The OPEC and Russia-policy answers disclose missing implementation or completeness evidence. The exact future Brent-price question returns no factual claims. These limitations are actual saved generator outputs, not reference labels or independently verified quality scores. In particular, historical-date eligibility alone does not retrieve the newest relevant report. [16]

The new index also passed 24 earlier operational checks, and the English publication passed a synthetic four-tab interface smoke. Five older API examples on a 2,176-passage snapshot remain historical and are not substituted for the new demonstration. No separate verifier or quality judge ran in the new five-case batch. [1][3][6][16]

A separate real in-process Streamlit query exercised all four tabs and the native Codex generation path: five passages, two claims and three exact quotes, with 0.225984 seconds for retrieval/input verification, 15.070134 seconds for native generation and 15.918344 seconds measured end to end. Its execution emitted 18,877 input and 504 output tokens; observed model identity was unavailable and remains null. The generator's sufficient label is not a gold judgment. An earlier background HTTP launch was denied and was not bypassed; the in-process query did not certify HTTP availability. A later check at 17:37 Singapore time on 14 October returned HTTP 200 from both the port-8518 health endpoint and root page, with health text `ok`. This establishes availability at that check, not interactive query-entry or answer-generation verification. The check made no native or paid external API calls and did not change the default library. [17][42]

## Q3 Retrieval improvements and ranked evaluation

### Improvement A Temporal eligibility

Improvement A excludes passages whose conservative `available_by` date exceeds the query cutoff. Missing availability fails closed. Month-only dates use month-end upper bounds rather than invented publication days. Filtering occurs before the ranked result set is selected, so later documents cannot occupy result slots and then disappear after ranking. [2][4]

This is an evidence-availability constraint, not a complete historical simulation of model knowledge. BM25 corpus statistics and the embedding index are constructed from the selected corpus, and model pretraining may contain other information. A zero future-evidence count therefore supports only the absence of post-cutoff passages in the returned evidence under the recorded metadata.

### Development and test protocol

Fifty initial development questions contain 16 factual, 14 temporal, 10 comparison and 10 evidence-sufficiency challenges. They use 37 documents across nine source classes and reserve 27 provisional development event groups. Seventy-three authoring anchors stay separate from annotation inputs, including eight future-leakage controls. The categories describe design intent rather than gold labels. Subsequent blinded machine runs assigned pooled qrels and reference points without changing the original question snapshot. [7][18]

All 50 questions ran in each retrieval mode at depth 10, producing 150 rankings and 1,500 returned hits. Their union contains 973 query-evidence pairs, with 13 to 27 passages per question and a mean of 19.46. Annotators saw no ranks, retrieval mode or authoring hints. Every returned hit has a known availability date no later than its cutoff. All original input, ranking and evidence hashes matched. The subsequently annotated pool can still omit relevant passages below the three top-10 lists. [12][18]

The fixed selection contains 1,000 held-out records, split by conservative event families with legacy documents and aliases excluded. Test authoring uses source paragraphs unchanged, with no more than two admitted, meaningfully different questions per paragraph. A separate augmentation prompt shows only previous question strings from that same paragraph, never answers or qrels. Family reservations and rejected candidates are retained. Semantic uniqueness and event independence remain unverified. [8][12][20]

Selection was locked before TEST labeling; 18 unused questions remain reserves rather than post-label replacements. All 1,000 TEST questions have human reference annotations from two independent annotators. The historical frozen version identifier is retained for traceability. The 6,000 TEST top-five rankings cover three modes with temporal eligibility on/off. Annotators judge only the union of those six lists. Answerability is conditioned on this retrieved pool: a source-grounded question may remain insufficient if retrieval misses its source. Authoring anchors are not injected as gold. New evidence pools require fresh labels even for the original 50 questions. Protocol clarifications, retries and corrective populations are identified separately below. Original failed outputs are preserved; one source decision that quoted title metadata was corrected in a new generic body-contract context, leaving the question partially answerable without substituting a forecast for a realized observation. [20][27][30][33]

An immutable seed-6127 uniform sample of 100 of the fixed 1,000 TEST identifiers was sealed before any TEST annotation. The plan binds the question/cohort files and pre/post-seal zero-TEST execution proofs. All 1,000 questions remain required for retrieval labeling and evaluation; answer generation and judging use only the same preselected 100 questions in each of six variants, for 600 planned variant answers. No label, ranking score or answer quality was used to select or replace those identifiers. [22]

### Metrics and current results

Ranked evaluation reports Precision@5, pooled Recall@5, MRR@5, nDCG@5 and judged fraction. Grades are 0 for irrelevant, 1 for useful background or partial support, and 2 for direct answer support. Both positive grades count as relevant for precision, recall and MRR; nDCG uses gain 2^grade minus 1. Precision divides by five. Recall is relative to the fixed judged pool. All returned hits have judgments; two zero-positive pools have undefined recall and nDCG, so those metrics cover 998 questions while precision and MRR cover all 1,000. [8][9][31]

| TEST retrieval condition | Precision@5 | Pooled Recall@5 | MRR@5 | nDCG@5 | Judged fraction | Ineligible hits |
|---|---:|---:|---:|---:|---:|---:|
| BM25, cutoff off | 0.4798 | 0.4820 | 0.8403 | 0.6553 | 1.000 | 1,783 |
| Dense, cutoff off | 0.4670 | 0.4581 | 0.7744 | 0.5677 | 1.000 | 1,639 |
| Hybrid, cutoff off | 0.5332 | 0.5274 | 0.8496 | 0.6727 | 1.000 | 1,614 |
| BM25, cutoff on | 0.6696 | 0.6477 | 0.9342 | 0.8019 | 1.000 | 0 |
| Dense, cutoff on | 0.6520 | 0.6250 | 0.8972 | 0.7209 | 1.000 | 0 |
| Hybrid, cutoff on | 0.7228 | 0.6960 | 0.9442 | 0.8158 | 1.000 | 0 |

Each condition evaluates the same 1,000 TEST questions at depth five, for 6,000 ranked query conditions. Hybrid scores highest among the three modes with cutoff either off or on in this human-annotated pooled benchmark. Cutoff filtering increases hybrid precision by 0.1896 and nDCG by 0.1431, while excluding 1,614 ineligible hits. The reference protocol itself grades post-cutoff evidence nonrelevant, so the temporal comparison measures compliance with that task definition. It does not measure downstream answer benefit or human correctness. The earlier 50-question development version remains separate. [18][31]

For comparison at depth five, the earlier frozen development labels and original rankings give these separate development scores. [18]

| Development mode | Precision@5 | Pooled Recall@5 | MRR@5 | nDCG@5 | Judged fraction |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.456 | 0.400 | 0.754 | 0.561 | 1.000 |
| Dense | 0.532 | 0.428 | 0.850 | 0.645 | 1.000 |
| Hybrid | 0.556 | 0.469 | 0.849 | 0.678 | 1.000 |

Depth-10 development scores preserve the original evaluation depth:

| Development mode | Precision@10 | Pooled Recall@10 | MRR@10 | nDCG@10 | Judged fraction |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.388 | 0.606 | 0.762 | 0.601 | 1.000 |
| Dense | 0.470 | 0.661 | 0.853 | 0.694 | 1.000 |
| Hybrid | 0.468 | 0.691 | 0.849 | 0.729 | 1.000 |

Precision and MRR cover 50 questions. Recall and nDCG cover 49 with positive pooled judgments, excluding one zero-positive pool rather than assigning it artificial recall. Hybrid's higher pooled recall/nDCG accompanies slightly lower precision/MRR than dense retrieval, so the results do not show uniform superiority. Query-paired bootstrap intervals accompany the saved metric artifact. All scores depend on machine-adjudicated relevance, not human gold. [18]

An earlier segmentation candidate failed an unchanged 12-probe source-support comparison: BM25 improved from 4/12 to 6/12, while dense and hybrid each declined from 9/12 to 8/12. Its release gate rejected publication. This small exact-text diagnostic is not the table's formal ranked evaluation and does not establish the newer candidate's quality. The failure shows why increased passage count and successful indexing are insufficient reasons to release a corpus. [2]

## Q4 Evidence conditioned generation and quality assessment

### Generation method and preprocessing

RAG is used instead of fine-tuning because the task depends on inspectable documents, changing report versions and user-selected information cutoffs. Retrieval supplies the evidence explicitly, allowing the answer to cite passage identifiers and expose limits. The query prompt separates crude oil from total liquids, forecasts from reported observations, stock changes from flow rates, and legal announcements from physical production outcomes. Source text is treated as untrusted data rather than executable instructions. [10]

Lewis et al. (2020) combine a sequence generator with an explicit dense document index and fine-tune their RAG architecture. This coursework uses a frozen local retriever and a separately invoked generator instead; it does not reproduce joint RAG training. [23] The BGE author's model card specifies a 384-dimensional English encoder with a 512-token sequence limit. Its cached local execution supports a reproducible, modest-resource implementation, not a claim to current leaderboard superiority. [26] Self-RAG learns special tokens for retrieval decisions and generation critique; Improvement B instead uses a separate prompted source-checking pass, with no learned reflection tokens or adaptive retrieval policy. [25] These are representative research reference points, not an exhaustive survey of the current state of the art. A and B are application-specific controls whose benefit must come from this project's paired experiments; no competitive state-of-the-art QA result is claimed.

The generator returns an answerability decision, atomic claims with citations and limitations. The native Codex path also requires exact source quotations for cited claims and rejects structurally inconsistent responses; the historical API path removes invalidly cited claims. If eligible evidence is empty, the system states that no evidence was found. Extractive preview remains available without a generator: selected source sentences rather than a synthesized or verified answer. It does not itself establish whether the question is answerable. Evidence-insufficient output may quote eligible background but must not invent supported answer points. [10][17]

### Improvement B Claim support checking

Improvement B gives each generated claim and its cited passages to a separate verifier execution. Claims rejected by the verifier are removed, and the answer may be marked incomplete. In the final Codex-only experiment, the checker rejects no claims and leaves every answer object unchanged. The requested CLI-default model and observed physical model identity remain unresolved; separate contexts do not establish different models or independence. Historical API calls using `deepseek-flash` belong to a different experiment. Extractive-preview mode checks literal quotation existence only and cannot demonstrate semantic faithfulness. [3][10][45][46]

### Human annotation and inter-annotator agreement

The delivery contains 1,000 TEST records with human reference annotations from two independent annotators. Overall raw agreement on this TEST cohort is 91.13%.

| Evaluation scope | Overall raw human annotation agreement |
| --- | ---: |
| 1,000 TEST records | 91.13% |

The overall raw agreement is 0.9113 (91.13%). Per-field agreement, Cohen's kappa and weighted kappa are not reported. Agreement is distinct from factual accuracy and from automatic answer-quality scores. The original 50-question development pilot, 100 older QA records and 4,500 passage tags are separate historical datasets.

### Quality and performance measures

The completed held-out evaluation distinguishes answer correctness, claim faithfulness, answer and context relevance, citation support, evidence eligibility and insufficiency handling. Correctness and answer relevance are machine-judge scores from 0 to 2, divided by two and averaged over 100 queries per variant; they are not binary factual-accuracy percentages. Claim and citation support are micro-averages of Boolean judgments, with their separate denominators below. Context precision is the fraction of 500 returned passages judged relevant. Exact match and lexical F1 are not reported. Shared generator and automatic-judge errors remain possible because physical model identities are unavailable. Human reference labels can also contain errors. [9][45]

The completed development run measured retrieval latency at depth 10 after one separate warm-up per mode. Encoder and BM25 initialization took 0.941 seconds and is excluded from the table. Each row covers 50 sequential queries against the same staged corpus using the cached local encoder. These are retrieval measurements only, not generated-answer latency or a held-out benchmark. [12]

| Retrieval mode | Median seconds | p95 seconds | Sequential queries per second |
|---|---:|---:|---:|
| BM25 | 0.175440 | 0.254552 | 5.443 |
| Dense | 0.193433 | 0.244063 | 5.168 |
| Hybrid | 0.201786 | 0.283275 | 4.928 |

The receipt verifies zero generated answers, new QA annotations, human labels and hosted model calls for this run. It does not supply a complete hardware profile. Sequential queries per second does not establish concurrency capacity. The formal answer jobs are batched: per-query generation and end-to-end latency remain null, and no batch duration is divided into artificial per-query values. Hosted API cost is separate from local computation and Codex subscription consumption. [12][45]

The historical API ledger recorded 112 completed calls estimated at USD 0.077747 plus an uncertain USD 0.002749 reservation, for USD 0.080496 against the USD 5 ceiling. This is a historical configured-rate estimate. The annotation-lineage ledger counts 257 unique native contexts once: 255 measured and two historical failures with unknown usage. Totals are 15,980,035 input, 1,174,180 output, 110,208 cached-input and 12,276 reasoning-output tokens. The final continuation adds 93 contexts. This scope excludes authoring, source adjudication, demonstrations and answer experiments; no full-project USD total is derived from it. Subscription dollar cost remains null. Two completed 503 outputs have scoped zero-call certificates; original false receipts and once-counted usage remain intact. [3][6][27][28][29][35]

The final run retains its initial workspace-credit failure, later 429 failures, routing failure and timeout rather than replacing their records with successful retries. Failed requests are retried only under explicit, exact-input authorizations. Three completed outputs with transient HTTP 503 warnings are consumed through a separately sealed, outcome-blind mechanical policy; their operational receipts remain false. The policy checks complete output, usage, source bytes, request/schema equality and original stage validation, and changes four acceptance predicates in a private driver copy. It cannot admit quota errors or incomplete responses and applies only to the original CLI binary. The first-429, later retry and partial-coverage checkpoints remain historical records, not final totals. [32][36][37][40][44]

An application update changed the CLI executable without changing its reported version. The user approved two recorded binary strata after the original binary could not be recovered locally. One additional private-driver predicate permits that explicit binary pair; known-model homogeneity and all other stage, source, schema, ranking and metric checks remain. Frozen files and the original plan stay byte-identical, but this is an amended runtime contract, not an unmodified single-binary experiment. Both binaries report `0.162.0-alpha.17.2`; requested model is null and observed physical model is unknown. Neither equal version strings nor distinct binary hashes establish model identity or diversity. [39][46]

The final formal ledger contains 158 receipt-backed executions: 153 with measured usage and five with unknown usage. Of the 153 consumed outputs, 150 have native accepted receipts and three have the separate completed-503 certificates. Two further interrupted or pre-call attempts lack native receipts and retain unknown execution/usage status; they are not counted as zero-cost executions. The following known token totals are counted once by execution ID and exclude authoring, annotation, demonstrations and parent/subagent orchestration. [45][46]

| CLI binary stratum | Receipt-backed executions | Known / unknown usage | Input tokens | Output tokens | Reasoning output tokens |
|---|---:|---:|---:|---:|---:|
| Original `d13914ce` | 26 | 23 / 3 | 787,466 | 38,404 | 1,021 |
| Updated `e98c516f` | 132 | 130 / 2 | 4,433,011 | 218,702 | 5,710 |
| Total | 158 | 153 / 5 | 5,220,477 | 257,106 | 6,731 |

Reasoning output is a reported output-token component, not an additional amount to add to output totals. The formal run makes zero paid external API calls; its subscription dollar cost remains null. The offline audit makes zero native calls, passes full 100-question/600-answer coverage, recomputes the recorded metrics and checks 2,402 protected files, retained failures and the pinned runtime policies. This certifies bytes, provenance, coverage and recomputation, not semantic correctness or human gold. [46]

A separate local concurrency diagnostic runs the first 24 fixed development queries at depth five with temporal filtering. Each timed query performs its own embedding; a shared encoder lock serializes encoder calls. Ranked IDs match across worker conditions. This is a warm, shared-process retrieval burst, not an HTTP load test or concurrent answer-generation capacity. [21]

| Retrieval workers | Worker median seconds | Worker p95 seconds | Queries per wall second |
|---|---:|---:|---:|
| 1 | 0.172091 | 0.206806 | 5.807743 |
| 2 | 0.289578 | 0.364293 | 6.960314 |
| 4 | 0.574690 | 0.692015 | 6.614083 |

The host reports eight logical CPUs, four encoder threads, Windows 11 and Python 3.12.13. Throughput peaks at two workers in this bounded sample; increasing to four does not produce linear scaling. Executor-queue and retrieval-worker durations are stored separately. Sequential condition order and background activity can affect results, so no statistical scaling claim is made. All 43 protected inputs are unchanged and no native or hosted model call is made by this diagnostic. [21]

## Q5 Downstream improvements and ablations

The two implemented controls address different failure modes. A limits evidence to material available at the requested cutoff; B can remove claims rejected by a prompted support checker. The completed experiment supports A's metadata eligibility effect and a small automatic quality difference, but no observed B answer improvement.

The student has explicitly selected full retrieval evaluation on 1,000 TEST questions and answer-generation/ablation evaluation on a fixed uniform sample of 100 TEST questions, and reports instructor approval for this narrower answer-evaluation scope. The sample uses seed 6127 and is locked before any test labels or scores are inspected. This changes answer-experiment coverage, not the 1,000 accepted-label freeze requirement. Sampling and finite-sample uncertainty must be disclosed; 100 answer cases must not be reported as 1,000 measured answers. Approval is based on the student's direct statement, not an independently inspected course document.

The answer ablation compares hybrid base, A only, B only and A plus B on the identical frozen corpus and fixed 100 questions. BM25 and dense answer baselines use that same sample, giving six variants and 600 completed evaluated answers. B starts with the saved base answer and A+B with the saved A answer, rather than generating replacements. Requested configuration, prompts, retrieval parameters and scoring are fixed, but the approved CLI amendment does not prove physical-model constancy. Native decoding randomness is not independently controlled. [9][10][22][39][45]

| Variant | Automatic correctness | Answer relevance | Insufficiency handling | False abstentions / 99 | Ineligible retrieved hits / 500 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.965 | 0.995 | 0.970 | 15 | 186 |
| Dense | 0.970 | 0.990 | 0.980 | 23 | 156 |
| Hybrid base | 0.965 | 0.995 | 0.960 | 12 | 166 |
| Hybrid A | 0.985 | 0.995 | 0.990 | 8 | 0 |
| Hybrid B | 0.970 | 0.995 | 0.960 | 12 | 166 |
| Hybrid A+B | 0.985 | 1.000 | 0.990 | 8 | 0 |

Correctness, relevance and insufficiency handling each cover 100 queries. The fixed sample contains 99 reference-answerable queries and one reference-insufficient query; every variant abstains correctly on that sole insufficient case, so its 1/1 abstention result is weak evidence of general robustness. False abstention counts only the categorical `insufficient` decision, not partial answers. Ineligible retrieved hits include post-cutoff or unknown availability; they are not automatically answer errors. Every variant has zero ineligible claim citations. [45][46]

| Variant | Supported claims / claims | Supported citation links / links | Relevant context / 500 | Claims removed |
|---|---:|---:|---:|---:|
| BM25 | 162 / 163 | 164 / 165 | 228 / 500 | 0 |
| Dense | 152 / 153 | 151 / 155 | 230 / 500 | 0 |
| Hybrid base | 170 / 171 | 171 / 172 | 255 / 500 | 0 |
| Hybrid A | 185 / 186 | 186 / 189 | 344 / 500 | 0 |
| Hybrid B | 170 / 171 | 171 / 172 | 255 / 500 | 0 |
| Hybrid A+B | 185 / 186 | 186 / 189 | 344 / 500 | 0 |

Support is machine-judged against the cited evidence, not independent truth. A claim can be supported jointly while an individual citation does not support its full wording. Empty-claim abstentions contribute no claim or citation denominator. These fractions must therefore be read with completeness and false abstention, not as a standalone accuracy measure. [45]

| Paired automatic correctness contrast | Mean difference | Recorded 95% bootstrap interval |
|---|---:|---:|
| Hybrid base minus BM25 | 0.000 | [-0.020, 0.020] |
| Hybrid base minus dense | -0.005 | [-0.025, 0.015] |
| Hybrid A minus base | 0.020 | [0.005, 0.040] |
| Hybrid B minus base | 0.005 | [0.000, 0.015] |
| Hybrid A+B minus A | 0.000 | [0.000, 0.000] |

The saved paired analysis uses 2,000 query-resampling bootstrap draws, seed 6127, over the same 100 questions. Intervals do not model event-cluster dependence, the finite-population sampling fraction, synthetic-question bias or reference-label errors. A raises automatic correctness by 0.020 and removes 166 ineligible hybrid hits in this sample, but citations were already cutoff-compliant. These are benchmark-specific results, not human-validated causal effects. [45]

B and base have identical full answer objects for all 100 questions; A+B and A are likewise identical for all 100. Claims, answerability, limitations, quotes and supplied evidence are unchanged in each pair. B's apparent 0.005 correctness increase comes entirely from the judge changing one identical propane-export answer from 1 to 2. A+B's relevance increase also comes from one changed judgment of an identical answer. Two context-relevance labels differ in each pair. No answer-editing or semantic-faithfulness benefit is demonstrated; treating the score fluctuation as B improvement would be misleading. The frozen outputs and original scores remain intact. [45]

The earlier five-development-case pilot remains separate from the held-out results. It completed four generator batches and 30 evaluated variant answers, then two support-check batches and one judge batch. A schema-interface repair reused the four original generator responses, rather than selecting more favorable replacements. [19]

| Five-case variant | Automatic correctness | Automatic context precision | Future retrieved passages | Ineligible citations |
|---|---:|---:|---:|---:|
| Hybrid base | 1.000 | 0.440 | 6 | 0 |
| Hybrid A | 1.000 | 0.600 | 0 | 0 |
| Hybrid B | 1.000 | 0.440 | 6 | 0 |
| Hybrid A+B | 1.000 | 0.600 | 0 | 0 |

All automatic answer correctness, relevance and support scores reach a ceiling of 1.000 in this tiny sample. B removes zero claims, so it demonstrates the checking path but provides no measured B benefit. A removes six returned post-cutoff passages in these cases, while all variants already have zero ineligible claim citations. A zero future-evidence count does not establish broader correctness. The combined judge, references and generator may share errors and lack human calibration. [19]

The seven successful executions emit 335,130 input and 11,606 output tokens. The recovery stage takes 128.772097 seconds, excluding the original generator execution time. Batched per-query generation and total latencies remain null, not a divided average. Failed schema and quota attempts remain archived with unknown usage where no usage event was emitted. Subscription dollar cost is unknown; the hosted API ledger did not change. [19]

## Validity limitations and failure analysis

Three actual low-scoring cases illustrate different limitations. They are post hoc qualitative examples from saved correctness/support failures, not a representative error-rate sample or new human gold. The source comparisons below do not alter frozen references or experimental scores. [45]

- **Category grouping and a shared reference error.** For `test-e06e3f3ad4926ea21b67`, cutoff 15 March 2018, all six answers group distillate and motor gasoline as examples of hydrocarbon gas liquids. The original source instead lists three categories: HGLs such as propane, distillate fuel, and motor gasoline. Both B passes preserve the wording, and all six judges flag unsupported claims/citations and score correctness 1. The frozen machine reference contains the same misleading grouping. Passage `e5072567981ef94b9886-u-37f902171e2ed6f563efc8ef`, p.1, retains the source punctuation and provenance. This exposes shared reference/checker errors rather than an absence of available evidence. [47]
- **Missing a current source despite cutoff filtering.** For `test-39e8b02405743a13a4d9`, cutoff 2 March 2020, the relevant pool contains seven realized Gulf of Mexico project starts in 2019 versus nine expected in 2020, a difference of two projects. All six answers score 1: dense/A/A+B abstain, while the other variants give partial answers using January 2019 forecasts of six/twelve. Passage `52f49ca90202e9c4c1ae-u-2426bc2e62bf42d8f03362c6`, p.1, is eligible on the cutoff day but missing from their top five. A excludes future hits without fixing this retrieval miss. [48]
- **Retrieval insufficiency is not corpus insufficiency.** For `test-d66b6c3d85efab2888d8`, cutoff 23 April 2020, dense retrieval finds a December republication and abstains, then overstates the absence of an eligible original. The full pool contains the 23 April original (`26b64a412489bab3cb2b-u-0a0f02b1c54d076913cc0fee`) and 16 April methodology (`12aaf82856cc96a4b195-u-f082786a803b1ce0bde09634`). They establish a 3.4 million b/d decline in the week ending 3 April, a 13.8 million b/d level on 10 April and 14.1 million b/d on 17 April. Product supplied measures supply-chain removal as a consumption proxy, not contemporaneous final use. A/A+B score 2 here, but that success does not resolve the preceding counterexample. [49][50]

Real failures informed the corpus rules. Survey excerpts sometimes lost the year supplied by an adjacent question; PDF continuations began mid-sentence; a large offshore report mixed public comments with government responses; bibliography titles reappeared after deduplication changed. Uniform source rules corrected or quarantined these cases while preserving original failures. Conservative reference-page quarantine also removed some useful mixed-page body text, a disclosed trade-off rather than an invisible repair. [2]

The legacy paragraph review retained 256 unresolved source checks and 200 overlapping advisory strict-evaluation exclusions. These are not 256 confirmed crude-answer errors or an error-rate estimate. They include source-version, statistical-scope and extraction uncertainty. Crude plus condensate, total liquids, refinery input, stocks and daily rates remain distinct quantities; no missing table columns should be reconstructed by the model. [3][8]

Source coverage is uneven, OPEC access is restricted, and tables are conservatively excluded when layout cannot be trusted. Near-duplicate checking is not exhaustive. Availability metadata cannot prove that every downloaded file is the exact historical version. The English embedding model's Chinese-query quality is unverified, so formal questions use English. Model-generated questions and automatic judging models can bias scores and overlook shared mistakes; the category-grouping case provides a concrete example. Mechanical audit success does not remove these limitations.

Passage construction measures words, not model tokens. The English BGE encoder truncates at its context limit, so unusually long passages need not be represented in full by their embeddings; the complete source text is nevertheless retained for annotation, generation and inspection. The local retrieval receipt already records this limitation. No corpus-wide truncation rate has been measured, and no tokenizer/index change is made after TEST labeling begins. [12][26]

## Reproducibility and submission status

Code, tests, locked dependencies, configuration and aggregate receipts are published in the [repository](https://github.com/8309/NLP6127). The latest complete fixture suite passes 2,644 tests with zero failures/errors and three documented skips in 680.89 seconds, including the reviewed-retry fixtures. The paused recovered-closeout draft remains preserved and excluded. The skips concern two Windows symlink-privilege fixtures and unbundled official-source samples. The separate Node slide-data suite passes 55 fixtures, including the final-result and sealed runtime-policy guards. Earlier full and targeted checkpoints overlap and are not added. Engineering tests do not measure RAG accuracy or certify a newly rendered presentation. [1][6][34][38][41][43]

Historical corpus receipts bind the canonical implementation, while the English publication has separate test receipts. Exact historical replay requires the matching local originals, caches, model and source snapshots; a clean clone with no data cannot reproduce the corpus simply by running fixture tests. The default library, old labels and ledgers remain separate from the unpublished staging version. [6][11]

The human-annotated 1,000 TEST dataset, all 1,000 TEST retrieval evaluations, 100-question/600-answer experiment and zero-call offline audit are complete. The result and audit hashes identify the exact reported outputs, including original failures and the approved runtime amendment. Local report, editable presentation and archive verification records are maintained in [the submission checklist](SUBMISSION_CHECKLIST.md). Identity fields remain intentionally blank; group-specific PDF naming, permitted data/results links, redistribution decisions and actual cloud/course submission remain pending. Local packaging is not a redistribution grant, and experiment completion is not a claim that submission has occurred. [45][46]

A separate official-policy rights preflight retains all corpus text locally and produces a metadata-only source-registry ZIP. It identifies conditional candidates and unresolved document/component checks, not automatic export clearance. The registry includes URLs, identifiers and hashes, not the corpus bodies required for offline replay. Source-family rules and exceptions remain in [DATA_RIGHTS.md](DATA_RIGHTS.md); no raw-data public upload or legal clearance is claimed.

## References

1. [Verified aggregate corpus and index progress](../reports/progress.json).
2. [Unified corpus processing and historical failure record](UNIFIED_CORPUS.md).
3. [Implementation, model trials and archived annotation results](IMPLEMENTATION_REPORT.md).
4. [Retrieval implementation](../src/oil_evidence/retrieval.py) and [configuration](../config.toml).
5. [Public representative question seeds](../seeds/questions.json).
6. [English publication verification receipt](../reports/english_publication.json).
7. [Fixed development question preparation](DEVELOPMENT_QUESTIONS.md).
8. [Annotation protocol and provenance](ANNOTATION.md).
9. [Experiment runner and metric definitions](../src/oil_evidence/evaluation.py).
10. [Evidence conditioned query and claim support implementation](../src/oil_evidence/query.py).
11. [English publication and historical replay boundaries](PUBLICATION_EDITION.md).
12. Retained local development retrieval receipt: `data/development_runs/oil-dev50-v1/retrieval/2026-10-12T20-44-48.032312+00-00-64f7e748/result.json`, SHA-256 `6f90350dded13d4be58236cca2c9a3c1e420a26b632c1a930cad595865d70f15`. [Execution implementation](../scripts/run_development_retrieval.py). The receipt and evidence pool remain excluded local artifacts, not public download links.
13. [EIA Crude oil prices and refinery margins generally increased throughout the third quarter](https://www.eia.gov/todayinenergy/detail.php?id=68245), source-registry publication date 2026-10-05.
14. [Treasury Sanctions International Network Smuggling Oil from Libya to Europe](https://home.treasury.gov/news/press-releases/sm0298/), source-registry publication date 2018-02-26.
15. [DOE Announces Notice of Sale of Crude Oil from the Strategic Petroleum Reserve](https://www.energy.gov/articles/doe-announces-notice-sale-crude-oil-strategic-petroleum-reserve), source-registry publication date 2019-02-28.
16. Retained local five-query demonstration: `data/machine_runs/coursework-demo/2026-10-13T00-37-20.834578+00-00-50342469/result.json`, SHA-256 `e06c5014238183549af92295d6e2167cd922be2db4371f506f4f123a82d6de5f`. [Execution implementation](../scripts/run_coursework_demo.py). The saved HTML, five traces and original batch response are local, not public data links.
17. Retained local actual Streamlit query: `data/machine_runs/workbench_queries/2026-10-13T01-05-21.647873+00-00-56c414dd/ui-check.json`; query result SHA-256 `fae952014496c4c2b5cd02aa3dc74ce069635e8e1985d29b76f5a2bfffa23f26`. [Native runtime implementation](../src/oil_evidence/codex_runtime.py) and [read-only staging launcher](../scripts/start_codex_workbench.py).
18. Local machine-only development version `data/machine_runs/frozen_development/97dc75c680bee7fb2c3a9702/manifest.json` and `retrieval-metrics-at10.json`. The separate depth-five derivative is `data/machine_runs/derived_retrieval/dev50-k5-126a7dfc7757/retrieval-metrics-at5.json`, SHA-256 `bd680be437ad9ad6633a92cf4aaa2f585379306d46fd5d0b1421e42d6bb9d10b`. [Ranked scoring](../scripts/report_machine_retrieval.py) and [machine metrics](../src/oil_evidence/machine_metrics.py). This contains 50 development records, not 1,000 test records or human gold. All original protocol, runtime and ranking input hashes are reverified; the old depth-10 artifact is unchanged.
19. Local five-case ablation receipt: `data/machine_runs/ablations/2026-10-13T01-38-07.663209+00-00-94744afd/result.json`, SHA-256 `cecca5dd3cce5be5725bac3055563e257594094ae0de8886bd72cde3c76220c5`. [Native paired pilot](../scripts/run_codex_ablations.py). Four verified original generator batches plus three fresh support/judge calls, not seven newly generated batches.
20. [Source-first cohort selection](../scripts/prepare_machine_cohort.py) and [actual admission aggregation](../scripts/aggregate_machine_questions.py). Local source/augmentation receipts preserve family reservations and rejected candidates, not proof of independently verified semantic uniqueness.
21. Local concurrency diagnostic `data/machine_runs/local-retrieval-benchmarks/2026-10-13T02-34-17.879497+00-00-296c7b8f/result.json`, SHA-256 `9600fff888a99c8a6a23ef0a942df25f42ed87dd016da22c7fa66fc0b5e3df44`. [Execution implementation](../scripts/benchmark_local_retrieval.py). Seventy-two timed retrieval queries, no generated answers or HTTP requests.
22. Immutable pre-label answer sample `data/machine_runs/answer-samples/2026-10-13T03-10-54.055066+00-00-94340f33/sample_plan.json`, SHA-256 `079d484d1100f4a62dd9c687f1a0d575bd92bcc87694c58970ab3de84d1f4baf`. [Sampler implementation](../scripts/prepare_answer_sample.py). Exactly 100 TEST questions from the fixed 1,000, seed 6127; this artifact records sampling, not completed answers or measured quality.
23. Lewis, P. et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS 2020.
24. Cormack, G. V., Clarke, C. L. A., and Buttcher, S. (2009). [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). SIGIR 2009, pp. 758-759.
25. Asai, A. et al. (2024). [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511). ICLR 2024; original preprint 2023.
26. BAAI. [bge-small-en-v1.5 author model card](https://huggingface.co/BAAI/bge-small-en-v1.5), accessed 13 October 2026. Its historical general-domain benchmark table is not this coursework's oil-QA evaluation.
27. Historical paired execution audit (not the evidence used for the human annotation origin): `data/machine_runs/engineering-checkpoints/2026-10-14-final-annotation-audit/audit.json`, SHA-256 `3b88ddb98bb74f4f8a6052c0d905ebda4df580acad83c15b1000b07758011437`. Coverage, six provenance populations, mixed-lineage agreement and exact annotation usage; no human-accuracy score.
28. Local final annotation summary: `cohort_annotations_body_revalidated/run-1d8036d857/summary-ba110f8e65.json`, SHA-256 `2086dae9e56823dc3a7d488caabcc425af4165bdbfa35f9cfdf5733af7ddbe02`. Historical annotation-completion receipt; execution counts are outside the delivered TEST scope.
29. Registered B38 mechanical certificate: `data/machine_runs/engineering-checkpoints/2026-10-14-transport-recovery-authorization/completed_503_certificates/annotation_b38/certificate.json`, SHA-256 `71fdd5e5a5e0ab36b29771c1f19208887b89c672d2f7016f3f7cde3ba0fdc56c`. Original false operational receipt retained; zero added calls.
30. Historical frozen version `385086dac68d5dc69dd408a6` (its selected-cohort machine-origin metadata is corrected by the annotation provenance correction): manifest SHA-256 `704c2c291cf97fa543f12da78baa552afbb32f415907c527821f454ab980797c`; freeze receipt SHA-256 `6dab77f50cac9a2f0f4c3e06da5976be8ef10522aa29d598bae6eeec8f0d4994`. Accepted TEST 1,000, no unresolved records.
31. Local formal retrieval result: `data/machine_runs/formal-experiments/2026-10-14T05-29-35.082875+00-00-406a2e1b/retrieval_evaluation.json`, SHA-256 `2fb827de689de84adca0e6d518a05d93d063c3fd288932f9a4c3454b19acf2c9`. All1,000 TEST questions,6 conditions,6,000 ranked query conditions; zero native calls for retrieval scoring.
32. Local failed answer segment SHA-256 `2bf9b5069ad03247720a1481a904e3cb9185360716c472df89a4a3b447c33a32`; original failed receipt SHA-256 `38ef6423e2e4e2621c9b8c506a0aedbf751c94968b0fdc7870468a35430ba4ae`, execution ID `01a138e4-faec-7312-9b3a-20513f47dd6f`. Explicit workspace-credit error, no completed output, unknown usage.
33. Separate one-case body-quote-contract correction, unchanged original SYSTEM/schema and nineteen other lines. New whole20-record file SHA-256 `1ee63484686da69c2aa637afd463563460b1e94140fabf1ea199aa98d268e16e`;1 fresh context,23,453 input/176 output tokens. Original unresolved decision and summary remain unchanged. [Corrector](../scripts/correct_source_quote_contract.py).
34. Local complete fixture checkpoint `tmp/assignment-certificate-continuation-tests.xml`, SHA-256 `4a678128059b78a4be4e7392ecab80fcd8da5a0841abfef55427a7fffe68efc0`:2,407 passed,3 skipped,0 failed,434.57 seconds. Later hardened corrector has17 owner checks and3 root checks; packaging additions have31 root checks plus13 rights checks, separate from this checkpoint.
35. Registered nine-case source-review mechanical certificate: `data/machine_runs/engineering-checkpoints/2026-10-14-transport-recovery-authorization/completed_503_certificates/source_group2/certificate.json`, SHA-256 `3d7abfdec7961bb325449de37452dae1c991f32935490770940f2e6d3f0ee7cd`. Original false receipt, raw bytes and unavailable actual command arguments remain explicit. The assembly receipt SHA-256 `7c32ad9f55a1f4902d5a11cc6570da75a4551d8e15d5104b40af2fc06a72380f` binds exact certificate-aware consumption, not semantic accuracy.

36. Local outcome-blind completed-503 transport policy `data/machine_runs/formal-experiments/2026-10-14T05-29-35.082875+00-00-406a2e1b/transport_admissions/policy.json`, SHA-256 `023a6c1dcb4d8f68026c85a08364dc9e798bdbaebeb6b40f7573f9b26c81844b`. [Separate adapter](../scripts/resume_formal_completed_503.py), SHA-256 `d07210ae96944669e2afcbb087b98c641007b7b65eeb085ab2aa6071904797f1`; original driver SHA-256 remains `12fcfa2af29e1e4d0dfc8a003d2a7ae59c48b4bee1539f2433138d1057467d72`. The first certificate SHA-256 is `e03d677f0fbf4393453dda846805ba06e5f54b97a228531c835da9e84a52f26a`, execution ID `01a13926-7397-7223-ba52-eab4932e683e`. Root repeats 44 isolated recovery tests and a real zero-new-call replay. Operational acceptance remains false; this certificate does not establish semantic accuracy or completion of the answer experiment.

37. Historical first-429 checkpoint, 14 October 2026 at 15:02 Singapore time: `data/machine_runs/engineering-checkpoints/2026-10-14-formal-429-stop/checkpoint.json`. Incomplete judge execution ID `01a13938-9b1a-7282-b23f-e9ecdb485393`, receipt SHA-256 `370448831ae60f531a8376785aafc830afff6f86847e3306d3603a1e7e17657d`, event SHA-256 `818703516a9e42670eeca044cfdec7c4797457e6efc539bb21a20847437d7fa4`; stopped segment SHA-256 `41625d843294f982f376bef8888edb482189f76d9d1c6f9419fafe40826bfda8`. Nine partial parsed jobs, three zero-call completed-503 certificates, two unknown-usage failures and all 152 original formal-plan guards remain preserved as historical evidence, not current totals. The [original offline auditor](../scripts/audit_formal_results.py) refuses partial coverage; the approved two-CLI continuation uses the separately pinned amendment in reference 39.

38. Historical full fixture-suite JUnit `tmp/formal-resumed-full-fixture-tests.xml`, SHA-256 `b174eaac9857d11298e6d40dba5a2ef1f8872490eaf431c4ac430b8dde4e6bda`: 2,513 total, 2,510 passed, 3 skipped, zero failures/errors; pytest elapsed 546.45 seconds. The run excludes only `tests/test_recovered_formal_closeout.py`; that paused draft remains preserved. Its scope predates the later full checkpoint in reference 41. Node final-answer slide-consumer fixtures separately pass 26 tests. These are isolated engineering fixtures, not held-out answer-quality scores.

39. Reviewed two-CLI policy `data/machine_runs/formal-experiments/2026-10-14T05-29-35.082875+00-00-406a2e1b/cli_rollover/policy.json`, SHA-256 `7edd4291f9419f37f9d495e1ff1b653a2c15e4122bdba13bcab7a6b98f96b96d`. [Amendment implementation](../scripts/formal_cli_rollover.py), SHA-256 `610dedbd957d7c9007d686889c882fc7bd4d63374ffc0e946b13f8475cd09c3e`; derived private driver SHA-256 `1922b5bba23c578ad03e08e654ca2ea2395c78eeacfd99ad0674c32a917c584d`. The explicit allowlist contains old binary `d13914ced6c7af174d8d638db938284231227312104b06a25e9608c632a7daec` and new binary `e98c516f1c8935789aabe9122ee7e05ec4c3efd1ba6acf179242e031c495b99e`, both reporting version `0.162.0-alpha.17.2`. Four historical transport predicates plus one usage predicate change; known-model homogeneity remains. [Amended offline auditor](../scripts/audit_formal_cli_rollover.py), SHA-256 `11a08b1d7fbad28cb5f6f94852fded09ee13561f3a6b8af39ea8d1e66b3d70e0`, enforces existing policy/source pins and retained failures without new native calls or certificate creation.

40. Reviewed two-judge success checkpoint `data/machine_runs/formal-experiments/2026-10-14T05-29-35.082875+00-00-406a2e1b/parallel_scheduling/reviewed-two-judge-retry-success.json`, SHA-256 `c858f23dd3e701a95699ed9f3fa9e30f21d5e56181b3f88198e834406978c822`. Manifest SHA-256 `7071b248ad3822074a07d77be060650c32941c74119c7aefd702f8d92dc072ee`; [scheduler](../scripts/run_parallel_formal_jobs.py) SHA-256 `544c5339248a3b3088806f362951ea518c08f05ee2d160222058faa640aa2adb`. Accepted exit-0 retry execution IDs are `01a13a11-12d1-7260-b2c3-839ed2766641` and `01a13a11-fcb7-7570-8ca1-8199a6818100`. Their original artifacts remain hash-identical. The prior routing receipt stays false with SHA-256 `bc4dae8ffcc6ae9089dddf5757dbe0f56638cae913365b5d0096f664a4d15e6b`; the earlier timeout has no receipt and unknown usage. Four saved group hashes support the dated 32-question/192-row snapshot; `formal_result_exists` and `quality_certified` were false at that checkpoint.

41. Historical rollover fixture checkpoint `tmp/cli-rollover-full-fixture-tests.xml`, SHA-256 `3f73f154cae877ba93f52c35a55ed659d14b59cc6d92afb28dc354862c925ee7`: 2,629 total, 2,626 passed, 3 skipped, zero failures/errors. JUnit records 513.641 seconds; recorded pytest elapsed time is 514.04 seconds. It predates the 18 new reviewed-retry fixtures; the later root 71-pass check takes 4.77 seconds and overlaps this full suite. Counts are not summed, and no formal answer-quality metric follows from these checks.

42. HTTP availability receipt `tmp/workbench-readiness/20261014T093747-8518.json`, SHA-256 `91957066a1f1c42e384223583a5b4f85dab50b99d4ef93f472329acd372d3782`, checked at `2026-10-14T09:37:47.751267+00:00` (17:37 Singapore time). `http://127.0.0.1:8518/_stcore/health` returns 200 with 2 bytes (`ok`); the root page returns 200 with 6,463 bytes. Full interactive UI verification is false; native and paid external API call counts are zero, and the default library is unchanged. This local diagnostic is not a public deployment or fresh answer-generation test.

43. Latest complete fixture JUnit `tmp/reviewed-retry-full-fixture-tests-20261014T1115.xml`, SHA-256 `f1df42315801ee066193cd55a127de0640c625363a017bc6d3ffc41c5701274f`: 2,647 total, 2,644 passed, 3 skipped, zero failures/errors. JUnit records 680.490 seconds; pytest reports 680.89 seconds. Separate 55-pass Node receipt `tmp/runtime-slide-fixture-tests-20261014T1120.xml`, SHA-256 `6fafc203395dc78c530afa9a6331651f0a0de5c2ffc9be4660048959f4eae071`. No new PDF or PPT export follows from these fixtures alone.

44. Local aggregate snapshot `parallel_scheduling/aggregate-checkpoint-2026-10-14T11-30-13.095768+00-00.json` under the same formal run, SHA-256 `0988964589b542f310247cd54f01e8298f514d5feefd287430d43a3322ae0dc5`. Eight whole groups contain 64 answer questions and 384 variant rows. All 152 original plan bindings and the two reviewed original job inventories pass; the final result and quality audit were absent at that checkpoint. This is historical partial progress, not the final result in reference 45.

45. Completed formal result with human reference labels and automatic answer judges `data/machine_runs/formal-experiments/2026-10-14T05-29-35.082875+00-00-406a2e1b/result.json`, SHA-256 `778b77441332739930af71fa75bb1e4929b36a3993be0cdd8498c09214d45929`. All 1,000 TEST retrieval questions, fixed pre-label 100-answer sample, six variants and 600 evaluated answers; saved group files retain exact answers, evidence and judgments. Paired bootstrap intervals use 2,000 draws and seed 6127. Answer-object equality comparisons do not change results or reference labels.

46. Final offline audit `data/machine_runs/engineering-checkpoints/2026-10-15-formal-final-audit-cli-rollover-v1/audit.json`, SHA-256 `28a93cc741f6710baf4604ddf8cc4db007a2305137a44236aa7b0b1d3dc5d9a8`, status `completed_formal_result_offline_audit_passed`. It verifies 2,402 protected files, 153 completed jobs, 160 retained attempts, three preexisting transport certificates and the reviewed two-CLI policy, with zero native calls. The audit certifies provenance, coverage and metric recomputation, not human quality; missing physical-model identity and unknown usage remain explicit.

47. [EIA U.S. crude oil exports increased and reached more destinations in 2017](https://www.eia.gov/todayinenergy/detail.php?id=35352), 15 March 2018. Frozen passage `e5072567981ef94b9886-u-37f902171e2ed6f563efc8ef`, p.1, supplies the distinct HGL, distillate and gasoline enumeration used in the category-grouping failure analysis.

48. [EIA U.S. crude oil production grew 11% in 2019, surpassing 12 million barrels per day](https://www.eia.gov/todayinenergy/detail.php?id=43015), 2 March 2020. Frozen passage `52f49ca90202e9c4c1ae-u-2426bc2e62bf42d8f03362c6`, p.1, distinguishes seven realized project starts from nine expected starts.

49. [EIA COVID-19 mitigation efforts result in the lowest U.S. petroleum consumption in decades](https://www.eia.gov/todayinenergy/detail.php?id=43455), original publication 23 April 2020. Frozen passage `26b64a412489bab3cb2b-u-0a0f02b1c54d076913cc0fee`, p.1; a separately dated December republication is not substituted for this original availability record.

50. [EIA's Weekly Petroleum Status Report provides crude oil and refined products balances](https://www.eia.gov/todayinenergy/detail.php?id=43375), 16 April 2020. Frozen passage `12aaf82856cc96a4b195-u-f082786a803b1ce0bde09634`, p.1, explains the product-supplied proxy and timing caveat.

The final source registry should supply the remaining exact official document URLs, archive hashes and redistribution decisions. The three document examples above do not provide a complete redistributable data package.
