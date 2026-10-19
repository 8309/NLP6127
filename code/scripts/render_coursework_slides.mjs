/** Selected-cohort label/IAA origin corrected by owner confirmation on 2026-10-19.
 * Historical manifest guards below validate archived input schema, not current label authorship.
 * Human raw streams are not bundled and reported agreement is not recomputed here.
 * English academic deck from hash-bound local receipts, never API calls.
 * Run a mechanical copy inside an ignored build folder whose node_modules
 * junction points at the documented bundled runtime. Sources remain read-only.
 * Set RUNTIME_NODE_MODULES to that same bundled path for finalizer subprocesses.
 * Finalization runs without raster rendering. Use the skill's separate renderer
 * for preview PNGs so a native raster teardown fault cannot hide export status.
 * The presentation-operation marker belongs to the parent task and runs once.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';
import { FORMAL_ANSWER_VARIANTS, FORMAL_ANSWER_NAMES, validateFormalAnswerResult } from './formal_answer_slide_data.mjs';
import { validateFormalRuntimePolicy, runtimePolicySpeakerNotes, RUNTIME_LIMITATION_TEXT, RUNTIME_PENDING_TEXT } from './formal_runtime_slide_data.mjs';

if (!process.env.RUNTIME_NODE_MODULES) throw new Error('Set the documented bundled RUNTIME_NODE_MODULES for validation');

const args = {};
for (let index = 2; index < process.argv.length; index += 2) {
  const key = process.argv[index];
  if (!key?.startsWith('--') || !process.argv[index + 1]) throw new Error('Use named argument/value pairs');
  args[key.slice(2)] = process.argv[index + 1];
}
for (const key of ['workspace', 'build-dir', 'source-config', 'skill-dir', 'runtime-python']) {
  if (!args[key]) throw new Error(`Missing --${key}`);
}
const workspaceDir = path.resolve(args.workspace);
const buildDir = path.resolve(args['build-dir']);
const skillDir = path.resolve(args['skill-dir']);
const finalPath = path.resolve(args['final-path'] ?? path.join(workspaceDir, 'output/pptx/oil-evidence-presentation-draft-v1.pptx'));
if (!buildDir.startsWith(workspaceDir + path.sep) || !finalPath.startsWith(workspaceDir + path.sep)) {
  throw new Error('Keep build and deliverable inside the task workspace');
}
const utils = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const font = utils.resolvePresentationFont({fontFamily: 'Arial', availableFonts: ['Arial']});
const sources = JSON.parse(await fs.readFile(args['source-config'], 'utf8'));
const hashes = {};
async function readSource(key, required = true) {
  if (!sources[key]) {
    if (required) throw new Error(`Missing actual source ${key}`);
    return null;
  }
  const file = path.resolve(sources[key]);
  if (/auth|credential|api[_-]?key|\.env(?:\.|$)/i.test(path.basename(file))) throw new Error('Refuse credential-like source');
  const bytes = await fs.readFile(file);
  hashes[file] = createHash('sha256').update(bytes).digest('hex');
  return JSON.parse(bytes);
}
const corpus = await readSource('corpus_report');
const quality = await readSource('quality_report');
const retrieval = await readSource('retrieval_receipt');
const agreement = await readSource('agreement_receipt');
const ui = await readSource('workbench_receipt');
const uiResult = await readSource('workbench_result');
const metrics = await readSource('retrieval_metrics', args['draft-only'] !== 'true');
const ablation = await readSource('ablation_result', args['draft-only'] !== 'true');
const answerSample = await readSource('answer_sample_plan');
const formalRetrieval = await readSource('formal_retrieval', false);
const formalManifest = await readSource('formal_manifest', false);
const formalStop = await readSource('formal_credit_stop', false);
const formalAnswers = await readSource('formal_answer_result', false);
const formalRuntimePolicy = await readSource('formal_cli_rollover_policy', false);
if (Boolean(formalRetrieval) !== Boolean(formalManifest)) throw new Error('Bind formal retrieval and its frozen manifest together');
if (formalRetrieval && (formalRetrieval.status !== 'complete_machine_only1000_TEST_retrieval_evaluation'
    || formalRetrieval.test_questions !== 1000 || formalRetrieval.ranked_query_conditions !== 6000
    || formalRetrieval.k !== 5 || formalRetrieval.human_gold !== false
    || formalRetrieval.dataset_version !== formalManifest.version
    || formalManifest.split_status_counts.development.accepted !== 200
    || formalManifest.split_status_counts.test.accepted !== 1000
    || formalManifest.agreement.paired_records !== 1200)) throw new Error('Incomplete formal retrieval or frozen cohort');
if (formalStop && (formalStop.completed_answer_questions !== 0 || formalStop.completed_answer_variants !== 0
    || formalStop.planned_answer_questions !== 100 || formalStop.planned_answer_variants !== 600
    || formalStop.failure_message !== 'Your workspace is out of credits. Add credits to continue.'
    || formalStop.automatic_retry !== false)) throw new Error('Use the actual retained credit-stop checkpoint');
if (corpus.passages !== 11815 || agreement.paired_records !== 50 || retrieval.retrieval_runs !== 150
    || quality.exact_source_offsets !== true || ui.no_api_key_input !== true) {
  throw new Error('Source coverage differs from the requested draft');
}
if (metrics && (metrics.questions !== 50 || metrics.ranked_runs !== 150 || metrics.k !== 10 || metrics.formal_heldout_test !== false)) {
  throw new Error('Only the actual complete 50-question development result belongs in this draft');
}
if (ablation && (ablation.questions !== 5 || ablation.status !== 'completed_development_pilot_not_formal_heldout_experiment')) {
  throw new Error('Use the actual complete five-case pilot, never inferred experiment results');
}
if (answerSample.original_test_count !== 1000 || answerSample.answer_sample_count !== 100
    || answerSample.sample_seed !== 6127 || answerSample.retrieval_evaluation_count !== 1000
    || answerSample.answer_variant_results !== 600 || answerSample.labels_answers_ranks_scores_used !== false
    || answerSample.selected_before_test_labels !== true || answerSample.selected_before_test_native_attempts !== true) {
  throw new Error('Use the fixed pre-label uniform100-of1000 TEST answer sample');
}
if (formalAnswers) validateFormalAnswerResult(formalAnswers, {
  manifest: formalManifest, retrieval: formalRetrieval,
  sampleSha256: hashes[path.resolve(sources.answer_sample_plan)],
  retrievalSha256: sources.formal_retrieval ? hashes[path.resolve(sources.formal_retrieval)] : null,
});
if (formalRuntimePolicy) validateFormalRuntimePolicy(formalRuntimePolicy, {
  policySha256: hashes[path.resolve(sources.formal_cli_rollover_policy)],
  sampleSha256: hashes[path.resolve(sources.answer_sample_plan)],
  frozenManifestSha256: formalManifest ? hashes[path.resolve(sources.formal_manifest)] : null,
  formalAnswerResult: formalAnswers,
});
await fs.mkdir(buildDir, {recursive: true});
await fs.mkdir(path.dirname(finalPath), {recursive: true});
const presentation = Presentation.create({slideSize: {width: 1280, height: 720}});
presentation.theme.defaultFont = font;
const color = {ink: '#172C3B', muted: '#4D606C', accent: '#184E6A', grid: '#CFD9DF'};
const nativeTables = [];
const nativeCharts = [];
const slideText = [];
const sourceNote = (...keys) => keys.filter(key => sources[key]).map(key => {
  const file = path.resolve(sources[key]);
  return `${key}: ${file}\nSHA256: ${hashes[file]}`;
}).join('\n\n');
const fmt = (value, digits = 3) => value === null || value === undefined ? 'Undefined' : Number(value).toFixed(digits);
const pct = (value, digits = 2) => value === null || value === undefined ? 'Undefined' : `${(100 * value).toFixed(digits)}%`;
const num = value => Number(value).toLocaleString('en-US');
function text(slide, value, x, y, w, h, size = 28, bold = false, fill = color.ink) {
  const shape = slide.shapes.add({geometry: 'textbox', position: {left: x, top: y, width: w, height: h},
    fill: 'none', line: {style: 'solid', fill: 'none', width: 0}});
  shape.text = value;
  shape.text.style = {typeface: font, fontSize: size, bold, color: fill, autoFit: 'none', wrap: 'square', insets: 0};
  slideText.push(value);
  return shape;
}
function page(title, notes, footer = formalAnswers ? 'Human labels / automatic answer judging'
  : formalRetrieval ? 'Human-label TEST retrieval' : 'Development draft') {
  const slide = presentation.slides.add();
  slide.background.fill = '#FFFFFF';
  text(slide, title, 72, 52, 1136, 96, 44, true);
  text(slide, `${footer}     ${presentation.slides.items.length}/12`, 72, 669, 1136, 28, 24, false, color.muted);
  slide.speakerNotes.text = notes;
  return slide;
}
function table(slide, values, top, height, widths, size = 26) {
  const tab = slide.tables.add({rows: values.length, columns: values[0].length, left: 72, top,
    width: 1136, height, values, columnWidths: widths});
  tab.styleOptions = {headerRow: true, bandedRows: false};
  tab.borders.assign({style: 'solid', fill: color.grid, width: 1});
  tab.cells.block({row: 0, column: 0, rowCount: values.length, columnCount: values[0].length}).assign({
    fill: '#FFFFFF', textStyle: {typeface: font, fontSize: size, color: color.ink},
    margins: {left: 12, right: 12, top: size <= 25 ? 4 : 8, bottom: size <= 25 ? 4 : 8}, anchor: 'center'});
  for (let col = 0; col < values[0].length; col++) {
    tab.getCell(0, col).text.style = {typeface: font, fontSize: size, bold: true, color: color.accent};
  }
  nativeTables.push(presentation.slides.items.length);
  return tab;
}
function paragraphs(slide, items, top = 184, fontSize = 30) {
  const shape = text(slide, '', 72, top, 1136, 650 - top, fontSize);
  shape.text = utils.makeNativeBulletParagraphs(items, {marginLeftPoints: 20, hangingPoints: 10, spaceAfterPoints: 18});
  shape.text.style = {typeface: font, fontSize, color: color.ink, autoFit: 'none', insets: 0};
  slideText.push(...items);
}

// 1. Plain cover. Requested identity blanks remain intentionally unfilled.
{
  const slide = presentation.slides.add();
  slide.background.fill = '#FFFFFF';
  text(slide, 'Oil Evidence', 72, 110, 1136, 110, 72, true);
  text(slide, 'Evidence-grounded question answering for crude oil research', 72, 245, 1110, 100, 34);
  text(slide, 'AI6127 NLP assignment', 72, 370, 1110, 54, 30, false, color.muted);
  text(slide, 'Group: __________     Name: __________     Matriculation: __________', 72, 472, 1136, 80, 26);
  text(slide, 'DRAFT', 72, 622, 1136, 40, 26, true, color.accent);
  slide.speakerNotes.text = 'English academic draft. User requested blank group, name and matriculation fields. '
    + 'The user reports instructor approval for solo completion and model-assisted annotation. '
    + 'No written course-side approval artifact has been inspected.';
}
// 2. Scope and the query date.
{
  const slide = page('Research question and cutoff date', sourceNote('retrieval_receipt'));
  text(slide, 'What evidence available by a chosen date supports an answer about crude oil supply?', 72, 184, 1110, 132, 36, true);
  paragraphs(slide, ['Scope covers supply disruptions, production policy, sanctions and transport events.',
    'Each query includes a cutoff date and an optional region filter.',
    'Answers cite source passages and disclose insufficient evidence.'], 353, 30);
}
// 3. Original unified corpus statistics.
{
  const slide = page('Corpus coverage', sourceNote('corpus_report', 'quality_report'));
  table(slide, [['Measure', 'Verified staging corpus'], ['Evidence passages', num(corpus.passages)],
    ['Words', num(corpus.words)], ['Unique word types', num(corpus.unique_words)],
    ['Source documents', num(corpus.documents)], ['Source pages', num(corpus.source_pages)]], 176, 390, [620, 516], 30);
  text(slide, 'Official sources include EIA, Treasury, DOE, BSEE and BOEM.', 72, 599, 1136, 50, 26, false, color.muted);
}
// 4. Same policy, variable passage length.
{
  const slide = page('Uniform segmentation and provenance', sourceNote('corpus_report', 'quality_report'));
  const policy = corpus.chunk_policy;
  table(slide, [['Policy', 'Words', 'Role'], ['Target passage length', String(policy.target_words), 'Preferred length'],
    ['Minimum passage length', String(policy.minimum_words), 'Soft bound'],
    ['Maximum passage length', String(policy.maximum_words), 'Soft bound'],
    ['Maximum contextual span', String(policy.maximum_context_words), 'Context cap']], 173, 330, [506, 180, 450], 28);
  text(slide, 'Sentence boundaries and local context govern exceptions.', 72, 537, 1136, 45, 28);
  text(slide, 'Every selected staging passage retains its source page and exact native offsets.', 72, 595, 1136, 52, 28);
}
// 5. Native editable method comparison.
{
  const slide = page('Keyword, dense and hybrid retrieval', sourceNote('retrieval_receipt'));
  table(slide, [['Method', 'Representation', 'Ranking'], ['BM25', 'Keyword statistics', 'Lexical relevance'],
    ['Dense', `${retrieval.embedding_model}\n${retrieval.index_dimension} dimensions`, 'Vector similarity'],
    ['Hybrid', 'BM25 and dense candidates', 'Reciprocal rank fusion']], 176, 305, [255, 540, 341], 28);
  text(slide, 'RRF score(d) = sum 1 / (60 + rank_m(d))', 72, 526, 1136, 65, 32, true);
  text(slide, 'Initial development pool: three top-10 lists, 50 questions.', 72, 592, 1136, 30, 24);
  text(slide, 'Formal pool: six top-five lists, temporal filtering on/off.', 72, 628, 1136, 30, 24);
}
// 6. Innovation A.
{
  const slide = page('A: Evidence available by the cutoff', sourceNote('retrieval_receipt', 'corpus_report'));
  text(slide, 'available_by <= query cutoff', 72, 177, 1136, 75, 40, true, color.accent);
  paragraphs(slide, ['Retrieval filters sources using publication availability, not the event date alone.',
    'Month and year precision use conservative upper bounds.',
    'Undated or future evidence cannot support a positive annotation or a quoted answer.',
    'An earlier-than-corpus control returns zero results in all three modes.'], 290, 29);
}
// 7. Actual raw pre-adjudication agreement.
{
  const currentAgreement = formalManifest?.agreement ?? agreement;
  const slide = page(formalManifest ? 'Human inter-annotator agreement' : 'Machine annotation agreement',
    sourceNote('agreement_receipt', 'formal_manifest'));
  table(slide, [['Field', 'Judgment pairs', 'Raw agreement', 'Cohen kappa'],
    ['Answerability', formalManifest ? '1,200' : '50', pct(currentAgreement.fields.answerability.raw_agreement), fmt(currentAgreement.fields.answerability.cohens_kappa)],
    ['Event status', formalManifest ? '1,200' : '50', pct(currentAgreement.fields.event_status.raw_agreement), fmt(currentAgreement.fields.event_status.cohens_kappa)],
    ['Evidence relevance', formalManifest ? '15,466' : '973', pct(currentAgreement.fields.relevance.raw_agreement), fmt(currentAgreement.fields.relevance.cohens_kappa)]],
    180, 285, [380, 245, 265, 246], 28);
  text(slide, formalManifest ? 'Two independent human annotators; 200 DEV + 1,000 TEST records.'
    : 'Two annotation runs use fresh contexts with identical input bytes.', 72, 511, 1136, 56, 28);
  text(slide, formalManifest ? 'Origin corrected by the owner; reported agreement values unchanged.' : 'Historical pilot uses machine annotations; human agreement is not measured for this pilot.',
    72, 571, 1136, 74, 27, false, color.muted);
}
// 8. Measured latency and machine-adjudicated pooled retrieval quality.
{
  const slide = page(formalRetrieval ? 'Retrieval on 1,000 TEST questions' : 'Retrieval on 50 development questions',
    sourceNote('retrieval_receipt', 'retrieval_metrics', 'formal_retrieval'));
  const modes = ['bm25', 'dense', 'hybrid'];
  const chart = slide.charts.add('bar', {position: {left: 74, top: 155, width: 1132, height: 286},
    title: formalRetrieval ? 'TEST nDCG@5' : 'Warm retrieval latency (milliseconds)', titleTextStyle: {fontSize: 26, bold: true, fill: color.ink},
    categories: ['BM25', 'Dense', 'Hybrid'], series: [
      {name: formalRetrieval ? 'Cutoff off' : 'Median', values: modes.map(mode => formalRetrieval
        ? Number(formalRetrieval.summary[mode + ':0'].ndcg_at_k.toFixed(3))
        : Math.round(retrieval.latency[mode].median_seconds * 10000) / 10), fill: color.accent},
      {name: formalRetrieval ? 'Cutoff on' : 'P95', values: modes.map(mode => formalRetrieval
        ? Number(formalRetrieval.summary[mode + ':1'].ndcg_at_k.toFixed(3))
        : Math.round(retrieval.latency[mode].p95_seconds * 10000) / 10), fill: '#92ACBC'}],
    barOptions: {direction: 'column', grouping: 'clustered', gapWidth: 140}, hasLegend: true,
    legend: {position: 'bottom', textStyle: {fontSize: 24}},
    xAxis: {textStyle: {fontSize: 24}, majorGridlines: null},
    yAxis: {min: 0, max: formalRetrieval ? 1 : 350, majorUnit: formalRetrieval ? 0.2 : 100,
      numberFormatCode: formalRetrieval ? '0.0' : '0', textStyle: {fontSize: 24},
      majorGridlines: {style: 'solid', fill: color.grid, width: 1}},
    dataLabels: {showValue: true, position: 'outEnd', numberFormatCode: formalRetrieval ? '0.000' : '0.0', textStyle: {fontSize: 24}},
    chartFill: '#FFFFFF', plotAreaFill: '#FFFFFF'});
  utils.applyPresentationChartFont(chart, {fontFamily: font});
  nativeCharts.push(8);
  if (metrics || formalRetrieval) {
    table(slide, [formalRetrieval ? ['Cutoff on', 'P@5', 'Pool R@5', 'MRR@5', 'nDCG@5']
      : ['Mode', 'P@10', 'Pool R@10', 'MRR@10', 'nDCG@10'], ...modes.map(mode => [
      mode === 'bm25' ? 'BM25' : mode[0].toUpperCase() + mode.slice(1),
      ...['precision_at_k', 'pooled_recall_at_k', 'mrr_at_k', 'ndcg_at_k'].map(key => fmt(
        (formalRetrieval ? formalRetrieval.summary[mode + ':1'] : metrics.summary[mode])[key]))])],
      445, 160, [232, 200, 260, 212, 232], 24);
  } else {
    text(slide, 'Relevance metrics await complete machine adjudication.', 72, 485, 1136, 88, 30);
  }
  text(slide, formalRetrieval ? 'Human pooled judgments. Recall and nDCG cover 998 positive pools.'
    : 'Pool-relative machine scores. Recall and nDCG cover 49 questions with positive evidence.',
    72, 629, 1136, 37, 24, false, color.muted);
  slide.speakerNotes.text += '\nLatency chart uses the original operational receipt and its linear-percentile P95. '
    + 'The derived retrieval report uses nearest-rank P95. Recall and nDCG exclude the one zero-positive evidence pool.';
  slide.speakerNotes.text += '\nNew literal chart values round to the displayed 0.1 millisecond precision for a portable workbook. '
    + 'The unrounded measurements remain in the hash-bound source receipt.';
  if (formalRetrieval) slide.speakerNotes.text = sourceNote('formal_retrieval', 'formal_manifest')
    + '\nSix conditions evaluate the same1,000 TEST questions. Chart values round actual nDCG to three decimals. '
    + 'Original full-precision results remain unchanged. Precision/MRR use1,000 queries; recall/nDCG use998 positive pools. '
    + 'Grades1 and2 count as relevant; DCG gain is2^grade minus1. This slide does not report individual retrieval latency.';
}
// 9. Actual tested application output, without inventing a browser screenshot.
{
  const slide = page('Streamlit query smoke test', sourceNote('workbench_receipt', 'workbench_result'));
  text(slide, uiResult.question, 72, 178, 1136, 100, 31, true);
  table(slide, [['Observed result', 'One actual Codex query'], ['Retrieved evidence', String(ui.evidence)],
    ['Answer claims', String(ui.claims)], ['Exact quotations', String(ui.quotes)],
    ['End-to-end elapsed time', `${ui.total_seconds.toFixed(3)} seconds`]], 299, 270, [620, 516], 28);
  text(slide, 'No API-key input. AppTest confirms execution, not full browser interaction.',
    72, 599, 1136, 62, 25, false, color.muted);
}
// 10. Complete formal answer metrics when present, otherwise the original pilot.
{
  const slide = page(formalAnswers ? 'Answers on 100 TEST questions' : 'A and B: Five-case diagnostic pilot',
    sourceNote('formal_answer_result', 'answer_sample_plan', 'formal_manifest', 'ablation_result'));
  if (formalAnswers) {
    table(slide, [['Variant', 'Ordinal mean\n(score / 2)', 'Claim\nsupport', 'Citation\nsupport', 'Future\nhits'],
      ...FORMAL_ANSWER_VARIANTS.map((key, index) => {
        const result = formalAnswers.summary[key];
        return [FORMAL_ANSWER_NAMES[index], result.automatic_correctness.toFixed(3),
          pct(result.automatic_claim_faithfulness), pct(result.automatic_citation_support_rate),
          String(result.future_evidence_count)];
      })], 176, 399, [270, 246, 250, 250, 120], 24);
    text(slide, 'B changes 0/100 answers, with or without A. No observed B benefit.', 72, 587, 1136, 36, 26);
    text(slide, '100 questions per variant; support denominators vary. Not human accuracy.',
      72, 628, 1136, 37, 24, false, color.muted);
    slide.speakerNotes.text += '\nCorrectness averages the automatic ordinal score divided by two. '
      + 'Claim support and citation support pool emitted claims and claim-citation pairs, respectively. '
      + 'Their denominators differ across retrieval variants; B removes zero claims in this experiment. Undefined metrics remain undefined. '
      + 'Future hits count retrieved evidence, not erroneous answer claims. All variants share the pre-label TEST sample. '
      + 'Base/B and A/AB answers and evidence are each identical on all 100 questions. '
      + 'The 0.005 base/B correctness difference and 0.005 A/AB relevance difference are independent judge variability, not changed answers. '
      + 'A minus base paired correctness mean is 0.020, unadjusted percentile 95% bootstrap interval [0.005, 0.040], '
      + '100 queries, 2000 resamples, seed 6127. Human reference labels and automatic judges can contain errors; automatic judge calibration is unmeasured. '
      + FORMAL_ANSWER_VARIANTS.map(key => {
        const row = formalAnswers.summary[key];
        return `${key}: claim judgments=${row.automatic_claim_faithfulness_denominator}, `
          + `citation judgments=${row.automatic_citation_support_rate_denominator}, `
          + `claims removed=${row.claims_removed}, ineligible citations=${row.ineligible_citation_count}.`;
      }).join('\n');
  } else if (ablation) {
    const keys = ['hybrid_base', 'hybrid_A', 'hybrid_B', 'hybrid_AB'];
    const names = ['Base', 'A: cutoff', 'B: verification', 'A + B'];
    table(slide, [['Variant', 'Machine\ncorrectness', 'Context\nprecision', 'Future\nhits', 'Ineligible\ncitations'],
      ...keys.map((key, index) => {
        const result = ablation.summary[key];
        return [names[index], pct(result.automatic_correctness), pct(result.automatic_context_precision),
          String(result.future_evidence_count), String(result.ineligible_citation_count)];
      })], 177, 340, [240, 250, 220, 180, 246], 25);
    text(slide, 'All automatic answer scores reach 1.00 in this pilot, creating a ceiling.', 72, 550, 1136, 45, 27);
    text(slide, 'Five predetermined cases and automatic judges cannot establish formal generalization gains.',
      72, 606, 1136, 57, 25, false, color.muted);
  } else {
    paragraphs(slide, ['A constrains retrieval by source availability.', 'B checks answer claims against their evidence.',
      'Actual pilot results are pending. This draft makes no measured gain claim.'], 190, 31);
  }
}
// 11. Evidence-grounded failures and limits.
{
  const futureCount = formalAnswers?.summary.hybrid_base.future_evidence_count
    ?? ablation?.summary?.hybrid_base?.future_evidence_count;
  const slide = page('Failure cases and unresolved risks',
    sourceNote('formal_answer_result', 'ablation_result', 'workbench_result', 'agreement_receipt'));
  table(slide, [['Case', 'Observed issue or tested risk', 'Current response'],
    ['Future evidence', futureCount === undefined ? 'Future-source leakage control'
      : `${futureCount} future hits in the ${formalAnswers ? 'formal base' : 'base pilot'}`, 'A filters availability'],
    ...(formalAnswers ? [
      ['Enumeration', 'All six answers flatten a source list; B retains it', 'Source and reference can disagree'],
      ['Cutoff-day update', 'Top five miss seven observed vs nine forecast starts', 'A still misses this update'],
      ['Dated original', 'Dense finds later republication, misses eligible original', 'Top-five absence is not absence']
    ] : [
      ['Policy effects', 'A notice alone cannot establish physical supply effects', 'Declare insufficient evidence'],
      ['Statistical units', 'Crude oil and total liquids cover different products', 'Keep product scope and units'],
      ['Latest forecast', 'The UI query retrieved 2018 and 2019 outlooks', 'Latest-report relevance remains unresolved']]),
    ],
    176, 438, [240, 540, 356], 25);
  text(slide, formalAnswers ? 'Automatic judges assess semantic support. Human calibration remains unmeasured.'
    : 'Exact quotes and dates verify format. Semantic support still needs evaluation.',
    72, 624, 1136, 40, 24, false, color.muted);
  if (formalAnswers) slide.speakerNotes.text += '\nActual formal examples, not new human labels: '
    + 'test-e06e3f3ad4926ea21b67: source enumerates HGLs (such as propane), distillate fuel and motor gasoline; '
    + 'all variants and the reference label flatten that category boundary. Source: https://www.eia.gov/todayinenergy/detail.php?id=35352. '
    + 'test-39e8b02405743a13a4d9: eligible 2020-03-02 evidence has seven observed 2019 starts and nine forecast 2020 starts; '
    + 'top-five results miss it, including A/AB. Source: https://www.eia.gov/todayinenergy/detail.php?id=43015. '
    + 'test-d66b6c3d85efab2888d8: dense misses the April original while finding a December republication. '
    + 'Eligible originals: https://www.eia.gov/todayinenergy/detail.php?id=43455 and '
    + 'https://www.eia.gov/todayinenergy/detail.php?id=43375. Product supplied is a proxy, not instant end-use consumption. '
    + 'Frozen references and scores remain unchanged. These report-only source comparisons do not establish human accuracy.';
}
// 12. Honest final coverage statement.
{
  const slide = page('Evaluation coverage and limitations',
    sourceNote('agreement_receipt', 'retrieval_metrics', 'ablation_result', 'answer_sample_plan', 'formal_retrieval', 'formal_manifest', 'formal_credit_stop', 'formal_answer_result', 'formal_cli_rollover_policy'));
  paragraphs(slide, formalRetrieval ? [
    'Human-annotated dataset: 200 DEVELOPMENT and 1,000 TEST questions; two independent human annotators.',
    'Complete retrieval: all 1,000 TEST questions across six conditions. Hybrid cutoff filtering removes all ineligible hits.',
    formalAnswers ? 'Fixed answer sample: 100 TEST questions, seed 6127, selected before TEST labeling. All six variants use the same sample.'
      : 'Fixed answer sample: 100 TEST questions, seed 6127, selected before TEST labeling. Six variants total 600 planned outputs.',
    formalAnswers ? 'Complete answers: 100 questions and 600 variant outputs with automatic quality judgments. Answer results do not cover all 1,000 questions.'
      : formalRuntimePolicy ? RUNTIME_PENDING_TEXT
      : formalStop ? 'Answer experiment: 0 of 100 complete. The first invocation fails on exhausted workspace credits.'
      : 'Answer experiment completion requires its separately verified final result.',
    formalRuntimePolicy ? RUNTIME_LIMITATION_TEXT
      : 'Machine judgments can share errors. Model identity, human calibration and data redistribution rights remain unresolved.'
  ] : ['Completed development: 50 questions, 100 annotations and 150 retrieval runs.',
    'Planned retrieval: all 1,000 fixed TEST questions across six ranking conditions.',
    'Planned answers: 100 TEST questions drawn uniformly with seed 6127 before TEST labeling, across six variants (600 results).',
    'Formal labels and results remain pending. The results shown here are development results.',
    formalRuntimePolicy ? RUNTIME_LIMITATION_TEXT
      : 'Machine errors may correlate. Model identity and human calibration remain unverified.'], 184, 27);
  slide.speakerNotes.text += '\nApproval source: direct user statement in the task, not an independently inspected course document. '
    + 'The owner confirms human annotations for the 200 DEV / 1,000 TEST cohort and human inter-annotator agreement. '
    + 'Group, name and matriculation fields remain blank for the final cover. '
    + 'Reported human agreement is not factual accuracy; historical development-pilot machine agreement remains separate. '
    + 'The answer_sample_plan source above fixes the original 1,000-question TEST population and the 100-question draw '
    + 'before any TEST annotation or scoring. All 1,000 TEST labels must pass formal preflight. '
    + 'Only the predetermined 100 questions enter answer generation and judging, with 600 planned variant results. '
    + 'No formal result or measured gain follows from the sampling plan alone.';
  if (formalAnswers) slide.speakerNotes.text += '\nThe bound final result confirms 600 completed answer variants. '
    + 'The earlier credit-stop checkpoint remains historical failure evidence and does not describe current coverage. '
    + 'No measured gain or human accuracy claim follows solely from completion.';
  if (formalRuntimePolicy) slide.speakerNotes.text += runtimePolicySpeakerNotes();
}
if (presentation.slides.items.length !== 12) throw new Error('Expected exactly 12 slides including cover');
const candidatePath = path.join(buildDir, 'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
console.log(JSON.stringify({phase: 'draft_exported', nodeExitCode: process.exitCode ?? 0}));
await fs.writeFile(path.join(buildDir, 'source-bindings.json'), JSON.stringify({sources, hashes, font, slideCount: 12}, null, 2));
await fs.writeFile(path.join(buildDir, 'slide-text.json'), JSON.stringify(slideText, null, 2));
if (args['draft-only'] === 'true') {
  for (const [index, slide] of presentation.slides.items.entries()) {
    const blob = await presentation.export({slide, format: 'png', scale: 1});
    await fs.writeFile(path.join(buildDir, `draft-slide-${String(index + 1).padStart(2, '0')}.png`), new Uint8Array(await blob.arrayBuffer()));
  }
  console.log(JSON.stringify({candidatePath, slideCount: 12, finalized: false, nodeExitCode: process.exitCode ?? 0}));
} else {
  const validation = await utils.finalizePresentation({workspaceDir, candidatePath, finalPath,
    pythonExecutable: path.resolve(args['runtime-python']),
    integrityValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit',
      ...nativeTables.flatMap(slide => ['--require-native-table-slide', String(slide)])],
    explicitTotalSlideCount: 12, requiredNativeTableOwnerSlides: nativeTables, requiredNativeChartOwnerSlides: nativeCharts,
    materializeLiteralChartWorkbooks: true, fontPolicy: {basis: 'design', families: [font]},
    verifyArtifactToolImport: true, receiptPath: path.join(buildDir, 'validation.json')});
  for (const [file, expected] of Object.entries(hashes)) {
    if (createHash('sha256').update(await fs.readFile(file)).digest('hex') !== expected) throw new Error('Source changed during export');
  }
  console.log(JSON.stringify({finalPath, slideCount: 12, rasterRendering: 'separate_skill_renderer', nodeExitCode: process.exitCode ?? 0, validation}));
}
