# AI6127 assignment acceptance

**Annotation provenance corrected on 19 October 2026.** The delivery contains 1,000 TEST records with human reference annotations. Earlier machine-origin descriptions of these labels were incorrect. Automatic answer generation and judging, older pilot records and machine source checks retain their own provenance. Questions, label values and TEST evaluation scores are unchanged. Overall raw human annotation agreement is 91.13% for TEST. Per-field agreement and kappa values are not reported. Historical filenames, schema keys and hashes are compatibility references.


Source of truth: ../Assignment(1).pdf in the course root (four pages, 2026). This checklist records requirements, not proof of completion.

| Item | Requirement | Evidence to produce |
|---|---|---|
| Knowledge base, Q1 / 20 points | >=10,000 documents/passages AND >=100,000 words; report documents/chunks/words/types, gathering/cleaning/storage and application examples | corpus.json plus source registry, cleaning configuration, manual QA |
| Retrieval / 40 points | sparse + dense + hybrid; not SQL LIKE search | three separately runnable modes and comparable results |
| Q2 | simple friendly UI, five queries/results and query speed | UI verification and saved five-query runs |
| Q3 | motivated retrieval/ranking improvements, examples, explicit ranked retrieval evaluation | development comparison, pooled qrels, Precision@K, Recall@K, MRR, nDCG |
| Generation / 40 points | task answers conditioned on retrieved evidence, justified method and preprocessing | cited answer traces and method explanation |
| Q4 annotation | >=1,000 manually labelled records; >=80% inter-annotator agreement; 3 recommended, 2 allowed | independent human judgments, coverage, raw agreement and supplementary kappa |
| Q4 quality | task-appropriate quality and answer faithfulness/relevance/context relevance; latency, throughput, cost, scalability | formal results, human/automatic distinction, performance records |
| Q5 | downstream improvement; ablations for multiple innovations | base, A, B, A+B under identical test settings |
| Submission | one PDF named group number, names and matriculation numbers on first page, Q1–Q5, links to data/results and source/library/README archives | checked submission manifest and links |
| Presentation | in-person week 13 | slides and demonstrable application |

Deadline: 7 November 2026 23:59 Singapore. Only first submission counts. Report weight 35%; local assessment_criteria.md separately lists presentation 30%. Late penalty: 5 percentage points per rounded-up day.

## Historical course-permission note (not current label provenance)

On 2026-10-12, the user stated that the instructor has agreed to individual completion and replacement of manual annotation with machine annotation. This is a user-provided report of course permission, not independently verified written correspondence. No instructor message has been sent by this project. Group number, name and matriculation number are intentionally blank pending the user's final details.

This historical machine-pipeline plan does not describe the corrected human-annotated selected cohort. Preserve the original rubric above and label every machine output truthfully. Permission does not itself produce the required 1,000 QA records, measured agreement, experiment results or submission links; these must still be delivered. The exact agreement interpretation for two independent executions of one model is not specified by the reported permission and must not be presented as independent-human or distinct-model agreement.

Do not contact the instructor, submit on Blackboard, purchase credits or increase the API budget without explicit authorization. Keep the final cover fields blank until supplied.

## Historical implementation plan (2026-10-10; not current label provenance)

The user selected a machine-only evaluation workflow because manual annotation is not feasible: Codex Sol 6.1 and DeepSeek Flash 4.1 independently label the same QA records, followed by model-agreement measurement and separate adjudication. The user accepted describing this accurately as model annotation, not manual annotation. This is a planned implementation choice, not instructor approval or satisfaction of the original Q4 human-labeling requirement. Preserve the requirement above and disclose the deviation in the final report. The dependency-ordered work is in `TASKS.md`; no annotation was launched when recording this decision.

## Historical automated-pipeline selection (superseded for selected-cohort origin)

On 2026-10-12 the user selected Codex alone for the new annotation runs instead of waiting for DeepSeek credentials. Use two fresh, blinded executions with identical evidence and rules, retain both raw outputs and compute agreement before adjudication. Repeated executions of the same model are not different-model diversity. Codex subscription usage is separate from API billing; record CLI usage when exposed and otherwise mark it unmeasured. Do not reuse credentials previously pasted into chat or convert subscription authentication into a third-party API key.
