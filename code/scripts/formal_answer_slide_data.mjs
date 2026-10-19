/** Read-only validation of complete formal answer metrics for slide consumers. */
export const FORMAL_ANSWER_VARIANTS = Object.freeze([
  'bm25', 'dense', 'hybrid_base', 'hybrid_A', 'hybrid_B', 'hybrid_AB',
]);
export const FORMAL_ANSWER_NAMES = Object.freeze([
  'BM25', 'Dense', 'Hybrid base', 'Hybrid + A', 'Hybrid + B', 'Hybrid + A + B',
]);

function requireValue(condition, message) {
  if (!condition) throw new Error(`Invalid formal answer result: ${message}`);
}
function ratio(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
}
function count(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

export function validateFormalAnswerResult(result, {manifest, retrieval, sampleSha256, retrievalSha256}) {
  requireValue(Boolean(manifest && retrieval), 'bind the frozen manifest and formal retrieval');
  requireValue(typeof sampleSha256 === 'string' && /^[a-f0-9]{64}$/.test(sampleSha256), 'bind the sample file SHA256');
  requireValue(typeof retrievalSha256 === 'string' && /^[a-f0-9]{64}$/.test(retrievalSha256), 'bind the retrieval file SHA256');
  requireValue(result?.status === 'completed_formal_machine_only_experiment', 'completion status');
  requireValue(result.human_gold === false && result.formal_heldout_test === true,
    'machine-only held-out provenance');
  requireValue(result.original_frozen_inputs_unchanged === true && result.error === null, 'unchanged inputs and no final error');
  requireValue(result.dataset_version === manifest.version && result.dataset_version === retrieval.dataset_version,
    'same frozen dataset version');
  requireValue(result.required_test_records === 1000 && result.planned_test_questions === 1000
    && result.completed_retrieval_questions === 1000, 'all 1,000 TEST retrieval questions');
  requireValue(result.planned_answer_questions === 100 && result.completed_answer_questions === 100
    && result.completed_variant_answers === 600, '100 complete answer questions and 600 variants');
  requireValue(result.answer_sample_plan_sha256 === sampleSha256, 'same original sample file');
  requireValue(result.retrieval_evaluation?.sha256 === retrievalSha256
    && result.retrieval_evaluation.test_questions === 1000
    && result.retrieval_evaluation.ranked_query_conditions === 6000, 'same completed retrieval result');
  const keys = Object.keys(result.summary ?? {}).sort();
  requireValue(JSON.stringify(keys) === JSON.stringify([...FORMAL_ANSWER_VARIANTS].sort()), 'exactly six variant summaries');
  for (const variant of FORMAL_ANSWER_VARIANTS) {
    const row = result.summary[variant];
    requireValue(row?.questions === 100, `${variant} question coverage`);
    for (const key of ['automatic_correctness', 'automatic_relevance', 'automatic_insufficiency_handling']) {
      requireValue(ratio(row[key]), `${variant} ${key}`);
    }
    for (const key of ['claims_removed', 'future_evidence_count', 'ineligible_citation_count',
      'reference_insufficient_questions', 'reference_answerable_questions']) {
      requireValue(count(row[key]), `${variant} ${key}`);
    }
    requireValue(row.reference_insufficient_questions + row.reference_answerable_questions === 100,
      `${variant} reference coverage`);
    for (const [key, denominator] of [
      ['categorical_correct_abstention_rate', 'reference_insufficient_questions'],
      ['categorical_false_abstention_rate', 'reference_answerable_questions'],
    ]) {
      requireValue(row[denominator] === 0 ? row[key] === null : ratio(row[key]), `${variant} ${key} denominator`);
    }
    for (const key of ['automatic_claim_faithfulness', 'automatic_citation_support_rate', 'automatic_context_precision']) {
      const denominator = row[`${key}_denominator`];
      requireValue(count(denominator), `${variant} ${key} denominator`);
      requireValue(denominator === 0 ? row[key] === null : ratio(row[key]), `${variant} ${key}`);
    }
  }
  return result;
}
