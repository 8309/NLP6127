# Annotation protocol v1

**Annotation provenance corrected on 19 October 2026.** The project owner confirms that the selected 200 DEVELOPMENT and 1,000 TEST records were human annotated and that the reported 1,200-pair agreement table measures two independent human annotators. Earlier descriptions of this selected cohort or its agreement as machine annotation were incorrect. Automatic answer generation/judging, older pilot records and machine source checks retain their own provenance. Reported label values and metric values are unchanged; the human streams were not independently re-audited in this correction. Historical filenames, schema keys and hashes are compatibility references, not the corrected annotation origin.


## Unit and roles

Historical selection (2026-10-12, superseded for selected-cohort origin): the user requests Codex-only annotation and reports instructor permission for solo completion and machine annotation substitution. Use two fresh blinded Codex executions, then calculate same-model execution agreement before adjudication. Keep the earlier dual-provider plan below as historical context; it no longer defines the current annotation provider. Approval is reported by the user, not independently verified correspondence. See `ACCEPTANCE.md`.

A record is a question with a cutoff date, an event group, reference answer points, evidence IDs, exact source quotes and graded relevance judgments. It is not a price-direction forecast label. Documents are read as untrusted evidence, not instructions.

The initial implementation used DeepSeek V4.1 Flash (`deepseek-flash`) for candidate generation and a separate same-model verification call. Those existing outputs remain same-model checks. The user's 2026-10-10 plan selects Codex Sol 6.1 and DeepSeek Flash 4.1 for future independent evaluation annotation, as specified below; this documentation change does not switch runtime configuration or launch annotation.

## Planned dual-model QA annotation (2026-10-10)

The user has no time for manual annotation and has accepted truthful reporting of a machine-only substitute. Requested annotators are Codex Sol 6.1 within the Codex app (execution target `gpt-6.1-sol`, verify the actual selected identity before running) and DeepSeek Flash 4.1 via the existing `deepseek-flash` alias (verify the service identity before running). Record requested and observed model IDs, available version metadata, prompt/input hashes, decoding settings, raw outputs and usage. Do not invent a precise model version or silently substitute another model. The earlier 100 QA annotations and 4,500 paragraph labels do not count as this new independent paired evaluation.

After corpus expansion/quality checks, prepare 200 development and 1,000 held-out QA records, split by event group. Each record fixes the question, cutoff and pooled candidate evidence. Both annotators see the same rules and evidence in separate contexts, without candidate answers, prior annotations, each other's outputs or adjudications. Candidate-generation answers must be held separately from annotation inputs. Start with the same 50 development records, refine only on development data, freeze the protocol, and complete both models' coverage of all 1,200 records under that protocol. Re-annotate pilot records when needed for protocol consistency.

Compute agreement on the original independent outputs before adjudication. Report overlap, missing/invalid records, class distributions, raw agreement by field, categorical Cohen's kappa and ordinal weighted kappa where defined. Missing judgments are not negative labels; undefined statistics are reported as such. Free-text answer points require fact-level comparison rather than string equality. Any model-scored semantic alignment is reported separately with judge provenance. The assignment's 80% human-agreement requirement is not satisfied merely by a model agreement score at or above 80%.

Adjudicate disagreements in a separate auditable model pass and examine a reproducible sample of consensus cases for shared errors. Keep both originals, adjudication reasons, exact support and unresolved cases. The adjudicator sees both answers and therefore is not a third independent annotation. A third model is not selected by this plan. Freeze a distinct machine-only dataset version for reproducible experiments; do not weaken the existing human-only freeze gate or write model outputs to human reviewer history. Report shared model biases and any overlap between annotation and generation models.

This plan creates no new model calls. Later DeepSeek execution remains subject to the existing cumulative USD 5 cap, positive configured rates and the durable ledger; do not reset prior spending. Codex subscription consumption is tracked separately when measurable and otherwise disclosed as unmeasured. The original course requirements and potential acceptance gap remain recorded in `ACCEPTANCE.md`.

## Labels

- answerability: sufficient / partial / insufficient, relative to the fixed corpus and cutoff, not world knowledge.
- relevance: 0 irrelevant; 1 background or partial support; 2 directly supports an answer point. A missing judgment is unjudged, never automatically zero.
- event_status: announced / effective / observed / recovered / uncertain / not_applicable. Announcement does not establish execution; effective legal text does not establish physical supply loss.
- answer_points: one externally checkable claim per line, preserving units, reference periods, crude vs total liquids and observed vs forecast distinctions.
- quotes: exact substrings from identified passages; identify supporting passages, not merely documents containing a matching keyword.
- event_group: same event across reports and paraphrases must share a group. Default document groups are provisional and require human cross-document review.

## Human workflow

Machine protocol v2 permits exact, eligible background quotes in an insufficient
label when they explain the evidence gap. For example, a notice identifying
tankers can establish that an action was announced without measuring an export
loss. An insufficient label still has no supported answer points; quotation,
source-date and positive-relevance guards remain mandatory. The v1 false
rejection and raw labels are retained separately. This is a schema correction,
not a change to semantic judgments or a way to increase measured agreement.

If execution metadata does not expose the actual model, a repeated requested
alias is recorded as `same_requested_model_unverified`, with observed IDs null.
It does not establish that the runtime model or backend version was identical.

Read question and evidence before revealing model suggestions. Record judgment, supporting spans and corrections. If model suggestions were viewed, provenance is human_reviewed, not human_independent. Reviewer identity is a real person, not model name or multiple aliases for the same individual.

The first 50 records test feasibility; source-reading templates must be rewritten into standalone research questions before final acceptance. The 30 seed questions are unreviewed proposals; an 'insufficient' question type is a design intent, not a gold answerability label.

Second human annotation should be independent and blinded, with the same instructions and evidence pool. Calculate agreement **before adjudication** and report overlap count, per-field agreement and class distributions; do not tune labels just to cross 80%. Then adjudicate differences and create one final record per question. Repeated review by one person is not inter-annotator agreement. The project owner now confirms two independent human annotators for the formal selected cohort; individual identities and raw streams are not bundled.

## Quality and splits

Inspect model-consensus items as well as disagreements; models may share a factual error. Create a blinded human subset to measure suggestion bias. Collect natural domain questions as well as model-generated ones to avoid source-copying shortcuts. Candidate pools combine BM25, dense and hybrid results; single-source question generation alone does not identify all relevant passages.

Develop prompts and retrieval parameters only on development data. Target 200 development + 1,000 held-out records; split whole event groups, retain deduplication decisions, then freeze hashes. Do not place questions, model answers or reference answers in the retrieval corpus. The source documents themselves may remain searchable.

Formal label acceptance still depends on the instructor's ruling on assisted labelling and independent-human coverage. A freeze manifest does not certify course approval.

## Separate corpus paragraph labels

The user separately requested labels on all 4,500 source passages. These are machine metadata, not additional QA records or human relevance judgments. Current Codex reads each fixed passage and authors multi-topic tags, literal entities, task relevance, publication-relative event states, short exact evidence spans and Chinese summaries. It optionally extracts selected key quantities and explicit-date events; empty arrays mean none separately selected, not a missing annotation or exhaustive absence of numbers/events.

Preserve the statistical scope in the reference period: production, processing, transport capacity, annual cumulative inventory changes, and inventory change rates are not interchangeable. `unclear` is valid when the measure vocabulary cannot represent a combined scope such as crude plus NGL or crude plus condensate reserves. Do not reconstruct missing table columns. `sanctions_policy` is a coarse policy bucket and can include trade/environmental measures; it does not by itself establish an oil sanction.

`needs_review` can flag incomplete extraction, source revisions after `available_by`, conflicting dates or statistical ambiguity. Keep these labels available for review, but exclude unresolved version/date items from strict historical evaluation unless a source version is verified. Machine flags do not automatically rewrite corpus dates. Exact-span/schema validation and parent Codex spotchecks are separate machine checks, not expert verification or independent-human agreement. Audited replacements preserve prior outputs; publication requires exact coverage and matching batch audit hashes.

## Blinded machine second pass on flagged paragraphs

Only the 728 flagged passages from immutable version e4158423d4e7b3a5 enter this targeted review. Selection itself is not blinded; fresh workers see the focus, same-document adjacent passages and unmodified local PDF-page text, but not the first labels or their reasons. No claim of independent-model diversity or new source verification follows. Raw PDF text preserves additional context, not verified visual column layout.

Finish and hash-validate all blinded outputs before revealing original labels. A separate explicit model adjudication chooses the original or blinded substantive label and records confirmed/corrected/unresolved, a Chinese explanation, exact supporting context quotes, remaining reasons and exclusion from strict evaluation. Do not infer adjudication from equality or automatically clear unresolved flags. Context can explain a fragment but must not invent a focal quote, missing unit, source version or causal effect. The final paragraph metadata remains constrained to exact focus-body evidence; context support is stored separately in the review record.

If both labels need correction, adjudication may explicitly author an `amended_label` with the complete focal schema; validate it against the same focus and preserve both prior outputs. An amendment sees the comparison and is not a blinded result. Do not modify the blinded output just to align two models.

Publication requires every target exactly once, original corpus/snapshot/input hashes, exact quotes and comparison/output audit hashes. Preserve the first snapshot and first-pass batches; publish a new immutable snapshot and update only the current merged machine views. These are 728 targeted machine reviews, not a random quality estimate, 728 human labels or an inter-human agreement study.

Before publication, a parent judgment revision may be appended under `judgment_revisions/` with an audited head, reason, original/comparison/new-output hashes and changed-record IDs. Accepted worker outputs and audits remain unchanged. Revised record actors are tracked separately from unchanged records. Revision heads are hash-validated; revisions after publication require a new run. This mechanism corrected two decisions that wrongly treated the coarse `sanctions_policy` bucket as a sanction-only definition; stylistic changes and additional optional quantities/events are not by themselves factual corrections.

Strict-evaluation exclusion is an advisory flag for constructing future approved datasets; it does not remove documents or automatically change online retrieval. Old and blind input conditions differ because the second pass includes context. Agreement/disagreement in this selected set is not an accuracy estimate or evidence of independent-human agreement.
