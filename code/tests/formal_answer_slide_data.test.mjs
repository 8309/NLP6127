import test from 'node:test';
import assert from 'node:assert/strict';
import { FORMAL_ANSWER_VARIANTS, validateFormalAnswerResult } from '../scripts/formal_answer_slide_data.mjs';

function fixture() {
  const sampleSha256 = 'a'.repeat(64);
  const retrievalSha256 = 'b'.repeat(64);
  const binding = {manifest: {version: 'frozen-fixture'}, retrieval: {dataset_version: 'frozen-fixture'},
    sampleSha256, retrievalSha256};
  const summary = Object.fromEntries(FORMAL_ANSWER_VARIANTS.map(variant => [variant, {
    questions: 100, automatic_correctness: 0.5, automatic_relevance: 0.5,
    automatic_insufficiency_handling: 0.5, claims_removed: 0,
    future_evidence_count: 3, ineligible_citation_count: 1,
    reference_insufficient_questions: 20, reference_answerable_questions: 80,
    categorical_correct_abstention_rate: 0.5, categorical_false_abstention_rate: 0.5,
    automatic_claim_faithfulness: 0.5, automatic_claim_faithfulness_denominator: 200,
    automatic_citation_support_rate: 0.5, automatic_citation_support_rate_denominator: 300,
    automatic_context_precision: 0.5, automatic_context_precision_denominator: 500,
  }]));
  return {binding, result: {status: 'completed_formal_machine_only_experiment', human_gold: false,
    formal_heldout_test: true, original_frozen_inputs_unchanged: true, error: null,
    dataset_version: 'frozen-fixture', required_test_records: 1000, planned_test_questions: 1000,
    completed_retrieval_questions: 1000, planned_answer_questions: 100,
    completed_answer_questions: 100, completed_variant_answers: 600,
    answer_sample_plan_sha256: sampleSha256,
    retrieval_evaluation: {sha256: retrievalSha256, test_questions: 1000, ranked_query_conditions: 6000}, summary}};
}

test('complete synthetic fixture preserves values and all six variants', () => {
  const {binding, result} = fixture();
  assert.equal(validateFormalAnswerResult(result, binding), result);
  assert.equal(result.summary.hybrid_AB.future_evidence_count, 3);
});

for (const [name, edit] of [
  ['failed run', r => { r.status = 'failed_not_complete'; }],
  ['human gold', r => { r.human_gold = true; }],
  ['not held out', r => { r.formal_heldout_test = false; }],
  ['changed inputs', r => { r.original_frozen_inputs_unchanged = false; }],
  ['final error', r => { r.error = 'failure'; }],
  ['different dataset', r => { r.dataset_version = 'other'; }],
  ['different sample', r => { r.answer_sample_plan_sha256 = 'c'.repeat(64); }],
  ['different retrieval', r => { r.retrieval_evaluation.sha256 = 'c'.repeat(64); }],
  ['incomplete retrieval', r => { r.completed_retrieval_questions = 999; }],
  ['incomplete questions', r => { r.completed_answer_questions = 99; }],
  ['incomplete variants', r => { r.completed_variant_answers = 599; }],
  ['missing variant', r => { delete r.summary.dense; }],
  ['extra variant', r => { r.summary.extra = r.summary.dense; }],
  ['incomplete variant', r => { r.summary.dense.questions = 99; }],
  ['nonfinite score', r => { r.summary.dense.automatic_correctness = NaN; }],
  ['out-of-range score', r => { r.summary.dense.automatic_correctness = 1.01; }],
  ['string score', r => { r.summary.dense.automatic_correctness = '0.5'; }],
  ['missing support score', r => { delete r.summary.dense.automatic_claim_faithfulness; }],
  ['negative denominator', r => { r.summary.dense.automatic_citation_support_rate_denominator = -1; }],
  ['fractional count', r => { r.summary.dense.future_evidence_count = 1.5; }],
  ['reference mismatch', r => { r.summary.dense.reference_answerable_questions = 79; }],
  ['nonempty null score', r => { r.summary.dense.automatic_claim_faithfulness = null; }],
  ['empty fabricated score', r => { r.summary.dense.automatic_claim_faithfulness_denominator = 0; }],
]) {
  test(`rejects ${name}`, () => {
    const {binding, result} = fixture();
    edit(result);
    assert.throws(() => validateFormalAnswerResult(result, binding), /Invalid formal answer result/);
  });
}

test('empty support population remains undefined instead of zero or one', () => {
  const {binding, result} = fixture();
  result.summary.hybrid_B.automatic_claim_faithfulness = null;
  result.summary.hybrid_B.automatic_claim_faithfulness_denominator = 0;
  result.summary.hybrid_B.automatic_citation_support_rate = null;
  result.summary.hybrid_B.automatic_citation_support_rate_denominator = 0;
  validateFormalAnswerResult(result, binding);
});

test('requires separately bound retrieval and sample files', () => {
  const {binding, result} = fixture();
  for (const key of ['manifest', 'retrieval', 'sampleSha256', 'retrievalSha256']) {
    assert.throws(() => validateFormalAnswerResult(result, {...binding, [key]: null}));
  }
});
