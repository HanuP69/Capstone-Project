import { cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import Papa from 'papaparse';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dashboard = path.resolve(here, '..');
const repo = path.resolve(dashboard, '..');
const out = path.join(dashboard, 'public', 'data');
await mkdir(path.join(out, 'figures'), { recursive: true });

const sources = {
  'metrics.json': 'master_evaluation_metrics.csv',
  'primary-audit.json': 'primary_audit_trail.csv',
  'secondary-audit.json': 'secondary_audit_trail.csv',
  'mfaq-comparison.json': 'mfaq_4condition_comparison.csv',
  'retrieval-comparison.json': 'hybrid_retrieval_comparison.csv',
  'weights-comparison.json': 'meta_learner_weights_comparison.csv',
  'mfaq-audit.json': 'mfaq_agentic_audit_trail.csv',
};
for (const [jsonName, csvName] of Object.entries(sources)) {
  const csv = await readFile(path.join(repo, csvName), 'utf8');
  const parsed = Papa.parse(csv, { header: true, skipEmptyLines: true, dynamicTyping: true });
  const fatal = parsed.errors.filter((error) => error.type === 'Quotes' || error.type === 'Delimiter');
  if (fatal.length) throw new Error(`${csvName}: ${fatal.map((error) => error.message).join('; ')}`);
  await writeFile(path.join(out, jsonName), JSON.stringify(parsed.data, null, 2));
}
const humanEval = await readFile(path.join(repo, 'human_eval_report.json'), 'utf8');
JSON.parse(humanEval);
await writeFile(path.join(out, 'human-eval.json'), humanEval);
const pipelineSource = await readFile(path.join(repo, 'verification_pipeline.py'), 'utf8');
const docsStart = pipelineSource.indexOf('SYNTHETIC_NITJ_DOCS = [');
const docsEnd = pipelineSource.indexOf('\n# Fully data-driven structured records', docsStart);
if (docsStart < 0 || docsEnd < 0) throw new Error('Could not locate the source synthetic NITJ evidence document block.');
const docsBlock = pipelineSource.slice(docsStart, docsEnd);
const documentPattern = /\{"id":\s*"([^"]+)",\s*"title":\s*"([^"]+)",\s*"category":\s*"([^"]+)",\s*"is_synthetic":\s*True,\s*"text":\s*"([^"]+)"\s*\}/gs;
const documents = [...docsBlock.matchAll(documentPattern)].map((match) => ({ id: match[1], title: match[2], category: match[3], is_synthetic: true, text: match[4] }));
if (!documents.length) throw new Error('No source evidence documents were parsed from verification_pipeline.py.');
await writeFile(path.join(out, 'synthetic-docs.json'), JSON.stringify(documents, null, 2));
for (const figure of ['roc_curve_comparison.png', 'confusion_matrix_groundedness.png']) {
  await cp(path.join(repo, 'evaluation_results', figure), path.join(out, 'figures', figure));
}
console.log(`Prepared source benchmark data and ${documents.length} synthetic evidence documents in ${path.relative(repo, out)}.`);
