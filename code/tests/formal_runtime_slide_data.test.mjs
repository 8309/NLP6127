import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { validateFormalRuntimePolicy, runtimePolicySpeakerNotes, RUNTIME_POLICY_SHA256,
  RUNTIME_BINARY_SHA256, RUNTIME_LIMITATION_TEXT, RUNTIME_PENDING_TEXT } from '../scripts/formal_runtime_slide_data.mjs';

function fixture() {
  const policy = {
    version: 'formal-reviewed-two-cli-binary-rollover-v1', user_authorization_confirmed: true, outcome_blind: true,
    allowed_native_cli_sha256: [...RUNTIME_BINARY_SHA256],
    preserved_native_cli_sha256: RUNTIME_BINARY_SHA256[0], continuation_native_cli_sha256: RUNTIME_BINARY_SHA256[1],
    native_cli_version: 'codex-cli 0.162.0-alpha.17.2', original_driver_file_unchanged: true,
    original_operational_receipts_unchanged: true, historical_transport_predicate_changes: 4,
    additional_usage_predicate_changes: 1, observed_physical_model: null, requested_model: null,
    physical_model_diversity_claimed: false, new_binary_completed503_admitted: false,
    unknown_or_third_binary_allowed: false, human_gold: false, quality_certified: false,
    new_native_calls_for_policy: 0, answer_sample_plan_sha256: 'a'.repeat(64), frozen_manifest_sha256: 'f'.repeat(64),
  };
  const binding = {policySha256: RUNTIME_POLICY_SHA256,
    sampleSha256: policy.answer_sample_plan_sha256, frozenManifestSha256: policy.frozen_manifest_sha256,
    formalAnswerResult: {answer_sample_plan_sha256: policy.answer_sample_plan_sha256}};
  return {policy, binding};
}

test('approved policy retains hashes and complete-answer source bindings', () => {
  const {policy, binding} = fixture();
  assert.equal(validateFormalRuntimePolicy(policy, binding), policy);
});

test('approved runtime policy can accompany a partial draft without asserting completion', () => {
  const {policy, binding} = fixture();
  assert.equal(validateFormalRuntimePolicy(policy, {...binding, formalAnswerResult: null}), policy);
  assert.match(RUNTIME_PENDING_TEXT, /require a separately verified result/);
  assert.doesNotMatch(RUNTIME_PENDING_TEXT, /0 of 100|600 complete/);
});

for (const [name, mutate] of [
  ['unapproved policy SHA', (_p, b) => { b.policySha256 = '0'.repeat(64); }],
  ['policy version', p => { p.version = 'different'; }],
  ['user approval', p => { p.user_authorization_confirmed = false; }],
  ['outcome selection', p => { p.outcome_blind = false; }],
  ['third binary', p => { p.allowed_native_cli_sha256.push('0'.repeat(64)); }],
  ['duplicate binary', p => { p.allowed_native_cli_sha256[1] = p.allowed_native_cli_sha256[0]; }],
  ['reversed lineage assignment', p => { p.preserved_native_cli_sha256 = p.continuation_native_cli_sha256; }],
  ['different CLI version', p => { p.native_cli_version = 'different'; }],
  ['changed original source', p => { p.original_driver_file_unchanged = false; }],
  ['changed original receipt', p => { p.original_operational_receipts_unchanged = false; }],
  ['wrong transport count', p => { p.historical_transport_predicate_changes = 3; }],
  ['wrong additional predicate count', p => { p.additional_usage_predicate_changes = 2; }],
  ['claimed physical identity', p => { p.observed_physical_model = 'guessed'; }],
  ['changed requested model', p => { p.requested_model = 'different'; }],
  ['model diversity', p => { p.physical_model_diversity_claimed = true; }],
  ['new-binary503 admission', p => { p.new_binary_completed503_admitted = true; }],
  ['unknown binary allowed', p => { p.unknown_or_third_binary_allowed = true; }],
  ['human gold', p => { p.human_gold = true; }],
  ['policy quality certificate', p => { p.quality_certified = true; }],
  ['policy model call', p => { p.new_native_calls_for_policy = 1; }],
  ['different loaded sample', (_p, b) => { b.sampleSha256 = 'b'.repeat(64); }],
  ['different loaded frozen manifest', (_p, b) => { b.frozenManifestSha256 = 'b'.repeat(64); }],
  ['missing loaded sample for complete answers', (_p, b) => { b.sampleSha256 = null; }],
  ['missing loaded manifest for complete answers', (_p, b) => { b.frozenManifestSha256 = null; }],
  ['different complete-answer sample', (_p, b) => { b.formalAnswerResult.answer_sample_plan_sha256 = 'b'.repeat(64); }],
]) {
  test(`rejects ${name}`, () => {
    const {policy, binding} = fixture();
    mutate(policy, binding);
    assert.throws(() => validateFormalRuntimePolicy(policy, binding), /Invalid formal runtime policy/);
  });
}

test('visible and note disclosures distinguish accounting from physical identity and quality', () => {
  const notes = runtimePolicySpeakerNotes();
  assert.match(RUNTIME_LIMITATION_TEXT, /Two CLI binary lineages/);
  assert.match(RUNTIME_LIMITATION_TEXT, /Physical model identity is unknown/);
  assert.match(RUNTIME_LIMITATION_TEXT, /Human calibration remains unmeasured/);
  assert.match(notes, /directly approved/);
  assert.match(notes, /four historical completed-503 transport predicates/);
  assert.match(notes, /one explicit two-binary accounting predicate/);
  assert.match(notes, /redistribution rights remain unresolved/);
  assert.match(notes, /final result validator remains mandatory/);
  assert.match(notes, /credit-stop record remains historical/);
});

test('builder keeps twelve slides and mandatory final-answer validation', () => {
  const source = fs.readFileSync(new URL('../scripts/render_coursework_slides.mjs', import.meta.url), 'utf8');
  assert.match(source, /if \(formalAnswers\) validateFormalAnswerResult\(formalAnswers/);
  assert.match(source, /readSource\('formal_cli_rollover_policy', false\)/);
  assert.match(source, /formalRuntimePolicy \? RUNTIME_PENDING_TEXT/);
  assert.match(source, /explicitTotalSlideCount: 12/);
});
