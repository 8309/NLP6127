/** Validate the sealed runtime amendment without generating a deck or answers. */
export const RUNTIME_POLICY_SHA256 = '7edd4291f9419f37f9d495e1ff1b653a2c15e4122bdba13bcab7a6b98f96b96d';
export const RUNTIME_BINARY_SHA256 = Object.freeze([
  'd13914ced6c7af174d8d638db938284231227312104b06a25e9608c632a7daec',
  'e98c516f1c8935789aabe9122ee7e05ec4c3efd1ba6acf179242e031c495b99e',
]);
export const RUNTIME_LIMITATION_TEXT = 'Two CLI binary lineages. Physical model identity is unknown. Human calibration remains unmeasured.';
export const RUNTIME_PENDING_TEXT = 'Final answer coverage and quality require a separately verified result.';

function requireValue(condition, message) {
  if (!condition) throw new Error(`Invalid formal runtime policy: ${message}`);
}
function sha256(value) {
  return typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
}

export function validateFormalRuntimePolicy(policy, {
  policySha256, sampleSha256 = null, frozenManifestSha256 = null, formalAnswerResult = null,
}) {
  requireValue(policySha256 === RUNTIME_POLICY_SHA256, 'exact approved policy file SHA256');
  requireValue(policy?.version === 'formal-reviewed-two-cli-binary-rollover-v1', 'policy version');
  requireValue(policy.user_authorization_confirmed === true && policy.outcome_blind === true,
    'direct user approval and outcome-blind scope');
  requireValue(Array.isArray(policy.allowed_native_cli_sha256)
    && JSON.stringify([...policy.allowed_native_cli_sha256].sort()) === JSON.stringify([...RUNTIME_BINARY_SHA256].sort()),
  'exactly the two approved executable hashes');
  requireValue(policy.preserved_native_cli_sha256 === RUNTIME_BINARY_SHA256[0]
    && policy.continuation_native_cli_sha256 === RUNTIME_BINARY_SHA256[1], 'original and continuation lineages');
  requireValue(policy.native_cli_version === 'codex-cli 0.162.0-alpha.17.2', 'unchanged disclosed CLI version');
  requireValue(policy.original_driver_file_unchanged === true
    && policy.original_operational_receipts_unchanged === true, 'preserved original source and receipts');
  requireValue(policy.historical_transport_predicate_changes === 4
    && policy.additional_usage_predicate_changes === 1, 'four transport predicates plus one accounting predicate');
  requireValue(policy.observed_physical_model === null && policy.requested_model === null
    && policy.physical_model_diversity_claimed === false, 'unknown physical identity and no diversity claim');
  requireValue(policy.new_binary_completed503_admitted === false
    && policy.unknown_or_third_binary_allowed === false, 'unchanged historical 503 and binary exclusions');
  requireValue(policy.human_gold === false && policy.quality_certified === false
    && policy.new_native_calls_for_policy === 0, 'policy approval does not certify answer quality');
  requireValue(sha256(policy.answer_sample_plan_sha256) && sha256(policy.frozen_manifest_sha256),
    'policy binds its original sample and frozen manifest');
  if (sampleSha256 !== null) {
    requireValue(sha256(sampleSha256) && policy.answer_sample_plan_sha256 === sampleSha256,
      'same loaded answer sample file');
  }
  if (frozenManifestSha256 !== null) {
    requireValue(sha256(frozenManifestSha256) && policy.frozen_manifest_sha256 === frozenManifestSha256,
      'same loaded frozen manifest file');
  }
  if (formalAnswerResult !== null) {
    requireValue(sha256(sampleSha256) && sha256(frozenManifestSha256),
      'complete answers require both loaded sample and frozen-manifest hashes');
    requireValue(formalAnswerResult.answer_sample_plan_sha256 === policy.answer_sample_plan_sha256,
      'complete answer result uses the policy sample');
  }
  return policy;
}

export function runtimePolicySpeakerNotes() {
  return '\nThe user directly approved preserving existing results and recording both native CLI executable lineages after an app update. '
    + `The sealed accounting policy SHA256 is ${RUNTIME_POLICY_SHA256}. `
    + `The historical executable SHA256 is ${RUNTIME_BINARY_SHA256[0]}. `
    + `The continuation executable SHA256 is ${RUNTIME_BINARY_SHA256[1]}. `
    + 'Both report codex-cli 0.162.0-alpha.17.2. The requested model remains null and physical model identity remains unknown. '
    + 'The private driver retains four historical completed-503 transport predicates and adds one explicit two-binary accounting predicate. '
    + 'The known-model guard and original stage, input and metric checks remain in place. Original source files and operational receipts remain unchanged. '
    + 'The historical 503 policy excludes the new executable. Separate executable lineages establish no model diversity or human accuracy. '
    + 'Policy approval establishes neither completed answer coverage nor calibrated answer quality. The final result validator remains mandatory for complete answer figures. '
    + 'Human reference labels and automatic judges may contain errors. Automatic judge calibration is unmeasured; source/model redistribution rights remain unresolved. '
    + 'The original credit-stop record remains historical evidence rather than a statement of current answer coverage.';
}
