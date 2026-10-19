# Draft Oil Evidence Coursework Presentation Outline

**Annotation provenance corrected on 19 October 2026.** The delivery contains 1,000 TEST records with human reference annotations. Earlier machine-origin descriptions of these labels were incorrect. Automatic answer generation and judging, older pilot records and machine source checks retain their own provenance. Questions, label values and TEST evaluation scores are unchanged. Overall raw human annotation agreement is 91.13% for TEST. Per-field agreement and kappa values are not reported. Historical filenames, schema keys and hashes are compatibility references.


Status: DRAFT CONTENT OUTLINE. No slide deck has been produced, and result slides must use actual completed experiments rather than the pending values in the report draft.

The presentation explains a crude-oil evidence question-answering application, demonstrates its source traceability, compares retrieval and generation settings, and describes the corrected human-annotation provenance and automatic answer evaluation. Group number, student name and matriculation number remain blank until supplied. [FINAL_REPORT.md](FINAL_REPORT.md) contains the report narrative, and [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) tracks delivery requirements.

## Slide 1 Task and research question

Title: Oil Evidence Question Answering

Show the blank identity fields and the application objective: answer crude-oil supply, policy, sanctions and transport questions using evidence available by a selected date. State that the task is research and evidence retrieval, not price prediction or investment advice. Introduce the question of whether temporal filtering and claim support checking improve useful, supported answers.

## Slide 2 Knowledge base and coverage

Show 11,815 staged passages, 1,562,715 words, 28,582 word types and 1,527 contributing documents. Identify official source families and the 2018-01-01 through 2026-10-05 window. Explain that the stage is not automatically the default workbench library and that the demonstration must name its actual corpus. Use source-family counts only after the final registry supplies them.

## Slide 3 Collection and cleaning

Show the smallest useful pipeline: official documents, verified source pages, shared sentence segmentation, admitted passages and local index. Explain URL/date/hash provenance, nonoverlap, exact offsets, complete units and time qualifiers. State the unchanged 160-word target, 65/240 soft bounds and 480 context cap. Include one real before-and-after extraction case with a recorded disposition rather than a fabricated clean example.

## Slide 4 Sparse dense and hybrid retrieval

Explain BM25 lexical matching, 384-dimensional BGE semantic similarity and reciprocal rank fusion. The initial 50 development questions now have 150 actual top-10 rankings and 1,500 hits, pooled into 973 question-evidence pairs. Show one query's three actual ranked lists with the same cutoff and region filters. Distinguish a retrieved hit from a relevance judgment. The final retrieval plot should contain Precision@5, pooled Recall@5, MRR@5, nDCG@5 and judged fraction from the frozen evaluation; those quality values remain unmeasured.

## Slide 5 Live evidence demonstration

Run a production-outlook question and open a full evidence card. Point to the publication date, forecast period, unit, passage ID and original source. Save the answer and timings before the presentation. Do not claim that old API examples on a smaller corpus are new staging-corpus outputs.

## Slide 6 Temporal filtering

Demonstrate the same historical question with A off and on. Show only an actual recorded case where the rankings change, and mark the post-cutoff evidence. Explain that conservative availability dates can exclude unknown versions and that evidence filtering is not a complete historical simulation of model pretraining or corpus statistics. Show measured future-evidence counts in the final slide.

## Slide 7 Evidence conditioned generation and support checking

Show the evidence-conditioned generator, citation-ID validation and B's separate claim-support check. Use one actual retained/removed claim pair with its cited text and verifier reason. Explain why crude versus total liquids, forecasts versus observations and announced versus effective events matter. Identify same-model verifier bias and the extra call's latency and cost.

## Slide 8 Human annotation and evaluation protocol

State the human origin of all 1,000 TEST records. Report overall raw human annotation agreement of 91.13% for 1,000 TEST records. The per-field breakdown and kappa values are not supplied. Do not assign broader-cohort statistics to TEST. Automatic answer judges remain separate.

## Slide 9 Retrieval and downstream results

Present the completed ranked retrieval comparison and base/A/B/A plus B ablation with the corpus, dataset, generator and evaluator fixed. Include answer correctness, faithfulness, relevance, citation support and temporal errors. If one setting loses quality, show it. Pending experiment cells are not result charts and must not appear as zero scores.

## Slide 10 Performance and cost

The completed development-only retrieval medians are 0.175440 seconds for BM25, 0.193433 for dense and 0.201786 for hybrid, each over 50 top-10 queries after an excluded per-mode warm-up. Use the corresponding p95 values in the report, not a fabricated throughput chart. Hardware profiling, end-to-end answer latency, concurrent throughput and final experiment cost still need measurement. Separate historical API estimates, uncertain reservations, local computation, recorded Codex CLI tokens and unmeasured subscription-credit consumption. Do not use the historical USD 0.080496 trial accounting or an account quota percentage as the final task cost.

## Slide 11 Failure cases and limits

Include the retained historical hybrid-support regression from 9/12 to 8/12 as a small diagnostic, not the new formal result. Add a real extraction or source-version failure and an evidence-insufficient question. Discuss uneven source coverage, restricted OPEC access, table exclusions, nonexhaustive near-duplicate checking and reference-label errors and automatic-judge bias. State which limitations survive the final experiments.

## Slide 12 Reproducibility and conclusion

Provide the checked code, permitted data and results links, frozen version hashes and startup instructions. Summarize the measured findings only after the actual experiments finish. Distinguish fixture test success from retrieval quality and explain which raw files cannot be redistributed. End with the application boundary and one concrete remaining limitation, not a claim of complete factual reliability.

## Demonstration sequence and fallback

Use three saved cases: a supported factual query, a historical-cutoff query and an evidence-insufficiency challenge. For each, show question, cutoff, actual corpus identity, answer, evidence, source link and latency. Use the remaining two saved queries to satisfy the five-query report requirement and prepare them for questions from the assessor.

Start the local workbench before the presentation and check its health, selected corpus and index hashes. Have a recorded trace or screenshots of the exact saved outputs available if the live service or API fails. Label fallback material as recorded outputs; do not present it as a new live run. Keep API keys and ledgers out of projected windows, and do not expose the unauthenticated local service to the internet.

## Evidence required before deck production

- Five fresh query traces and screenshots on the selected corpus, with timing and sources.
- Ranked retrieval metrics, judged coverage and per-mode examples.
- Completed independent machine annotations, original agreement and adjudication records.
- Base/A/B/A plus B results with quality, temporal errors, latency and cost.
- Real failures and unchanged historical regression records.
- Checked cover identifiers, report, source/library/README archives and permitted data/results links.

Engineering receipts already support corpus counts, index integrity, the 24 initial operational checks and the 150 real development retrieval runs with warm latency. They do not fill any missing quality or ablation result. All final figures and captions must identify their measured scope.
