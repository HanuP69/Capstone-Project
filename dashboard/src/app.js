const DATA = JSON.parse(document.getElementById('summary').textContent);
const $ = (selector) => document.querySelector(selector);

const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[character]));

const fmt = (value, digits = 3) => {
  if (value === null || value === undefined || value === '') return '—';
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : String(value);
};

const badge = (decision) => {
  const name = String(decision || '').toLowerCase();
  const cls = name === 'pass' ? 'pass' : name === 'block' ? 'block' : name === 'flag' ? 'flag' : 'neutral';
  return `<span class="badge ${cls}">${esc(decision || 'n/a')}</span>`;
};

const table = (columns, rows) => {
  if (!rows || !rows.length) return '<p class="muted">No rows generated.</p>';
  const head = columns.map((column) => `<th>${esc(column.label)}</th>`).join('');
  const body = rows.map((row) => `<tr>${columns.map((column) => `<td>${column.render ? column.render(row) : esc(row[column.key])}</td>`).join('')}</tr>`).join('');
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
};

const metricRows = (rows) => (rows || []).map((row) => ({
  ...row,
  condition: DATA.condition_names[row.condition] || row.condition,
}));

const metricsTable = (rows) => table([
  { key: 'condition', label: 'Condition' },
  { key: 'n', label: 'n' },
  { key: 'accuracy', label: 'Accuracy', render: (row) => fmt(row.accuracy, 4) },
  { key: 'macro_f1', label: 'Macro F1', render: (row) => fmt(row.macro_f1, 4) },
  { key: 'missed_falsehood_rate', label: 'Missed falsehoods', render: (row) => fmt(row.missed_falsehood_rate, 4) },
  { key: 'abstention_rate', label: 'Abstention', render: (row) => fmt(row.abstention_rate, 4) },
], metricRows(rows));

const qualityMap = Object.fromEntries((DATA.quality || []).map((row) => [row.metric, row.value]));
const findMetric = (rows, condition) => (rows || []).find((row) => row.condition === condition) || {};
const allD = findMetric(DATA.metrics, 'pred_d');
const realD = findMetric((DATA.subsets || []).filter((row) => row.subset === 'real_chunk'), 'pred_d');

$('#dataset-mode').textContent = DATA.dataset_mode || '';
$('#generated-at').textContent = DATA.generated_at ? `Generated ${DATA.generated_at}` : '';

const cards = [
  ['Indexed chunks', DATA.corpus.chunks_dedup, `${DATA.corpus.pdfs_parsed} parsed PDFs`],
  ['Test claims', qualityMap.test_claims, `${qualityMap.test_real_chunk || 0} real-chunk`],
  ['D accuracy · all', `${(Number(allD.accuracy) * 100).toFixed(1)}%`, `${allD.n || 0} claims`],
  ['D accuracy · real-chunk', `${(Number(realD.accuracy) * 100).toFixed(1)}%`, `${realD.n || 0} claims`],
  ['Hedge / absent claims', qualityMap.test_hedge, `${qualityMap.test_unverifiable || 0} unverifiable`],
  ['Records in use', (DATA.records || []).length, 'support ≥ 2'],
];
$('#overview-cards').innerHTML = cards.map(([label, value, note]) => `
  <article class="card">
    <div class="label">${esc(label)}</div>
    <div class="value">${esc(value ?? '—')}</div>
    <div class="note">${esc(note || '')}</div>
  </article>`).join('');

$('#quality-panel').innerHTML = `<h2>Dataset audit</h2>${table([
  { key: 'metric', label: 'Metric' },
  { key: 'value', label: 'Value' },
], DATA.quality || [])}`;

$('#metrics-all').innerHTML = metricsTable(DATA.metrics);
$('#metrics-real').innerHTML = metricsTable((DATA.subsets || []).filter((row) => row.subset === 'real_chunk'));
$('#ablation').innerHTML = table([
  { key: 'rule', label: 'Rule' },
  { key: 'accuracy', label: 'Accuracy', render: (row) => fmt(row.accuracy, 4) },
  { key: 'macro_f1', label: 'Macro F1', render: (row) => fmt(row.macro_f1, 4) },
  { key: 'missed_falsehood', label: 'Missed falsehoods', render: (row) => fmt(row.missed_falsehood, 4) },
  { key: 'abstention', label: 'Abstention', render: (row) => fmt(row.abstention, 4) },
], DATA.ablation);
$('#subgroup').innerHTML = table([
  { key: 'group', label: 'Group' },
  { key: 'n', label: 'n' },
  { key: 'accuracy', label: 'Accuracy', render: (row) => fmt(row.accuracy, 4) },
  { key: 'macro_f1', label: 'Macro F1', render: (row) => fmt(row.macro_f1, 4) },
  { key: 'missed_falsehood', label: 'Missed falsehoods', render: (row) => fmt(row.missed_falsehood, 4) },
  { key: 'abstention', label: 'Abstention', render: (row) => fmt(row.abstention, 4) },
], DATA.subgroup);
$('#unsupervised').innerHTML = table([
  { key: 'condition', label: 'Condition' },
  { key: 'unsupported_per_answer', label: 'Unsupported / answer', render: (row) => fmt(row.unsupported_per_answer, 3) },
  { key: 'faithful_rate', label: 'Faithful rate', render: (row) => fmt(row.faithful_rate, 3) },
  { key: 'context_relevance', label: 'Context relevance', render: (row) => fmt(row.context_relevance, 3) },
  { key: 'query_entailment', label: 'Query entailment', render: (row) => fmt(row.query_entailment, 3) },
], DATA.unsupervised);

$('#records-table').innerHTML = table([
  { key: 'key', label: 'Key' },
  { key: 'value', label: 'Value' },
  { key: 'operator', label: 'Operator' },
  { key: 'support', label: 'Support' },
  { key: 'source_doc', label: 'Source' },
  { key: 'source_date', label: 'Date' },
], DATA.records || []);

const renderClaims = (filter = '') => {
  const needle = filter.trim().toLowerCase();
  const rows = (DATA.claims || []).filter((row) => !needle || [
    row.claim_id, row.claim, row.label, row.pred_d, row.decision, row.fact_note,
  ].some((value) => String(value || '').toLowerCase().includes(needle)));
  $('#claims-table').innerHTML = table([
    { key: 'claim_id', label: 'ID' },
    { key: 'claim', label: 'Claim', render: (row) => esc(row.claim).slice(0, 180) },
    { key: 'label', label: 'Label' },
    { key: 'pred_d', label: 'Predicted' },
    { key: 'decision', label: 'Decision', render: (row) => badge(row.decision) },
    { key: 'F', label: 'F', render: (row) => fmt(row.F, 3) },
    { key: 'D', label: 'D', render: (row) => fmt(row.D, 3) },
    { key: 'C', label: 'C', render: (row) => fmt(row.C, 3) },
  ], rows.slice(0, 300));
};
renderClaims();
$('#claim-filter').addEventListener('input', (event) => renderClaims(event.target.value));

$('#limitations').innerHTML = (DATA.limitations || []).map((item) => `<li>${esc(item)}</li>`).join('');

document.querySelectorAll('.tabs button').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach((other) => other.classList.remove('active'));
    document.querySelectorAll('.tab').forEach((section) => section.classList.remove('active'));
    button.classList.add('active');
    $(`#tab-${button.dataset.tab}`).classList.add('active');
  });
});

$('#verify-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = $('#query').value.trim();
  const output = $('#verify-output');
  const button = event.target.querySelector('button[type="submit"]');
  if (!query) return;
  output.innerHTML = '<p class="muted">Verifying against the local pipeline…</p>';
  button.disabled = true;
  try {
    const response = await fetch('/api/verify-query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Verification failed (${response.status}).`);
    const claims = (payload.claims || []).map((claim) => `
      <article class="claim-card">
        <p>${badge(claim.decision)} <strong>${esc(claim.claim_text)}</strong></p>
        <p class="scores">
          <span>F ${fmt(claim.F_score, 3)}</span>
          <span>P ${fmt(claim.P_score, 3)}</span>
          <span>D ${fmt(claim.D_score, 3)}</span>
          <span>G ${fmt(claim.G_score, 3)}</span>
        </p>
        <p class="muted">${esc(claim.fact_explanation || '')}</p>
        ${claim.regenerated_claim ? `<p class="muted">Rewritten: ${esc(claim.regenerated_claim)}</p>` : ''}
      </article>`).join('');
    output.innerHTML = `
      <div class="answer"><strong>Grounded answer</strong><br>${esc(payload.final_grounded_answer || '')}</div>
      <p class="muted">Overall groundedness ${fmt(payload.overall_groundedness, 3)} · ${esc(payload.total_claims || 0)} claims · ${fmt(payload.latency_seconds, 2)} s · weighting ${esc(claims[0] ? 'fixed weights' : 'n/a')}</p>
      ${claims || '<p class="muted">No claims were extracted from the draft.</p>'}`;
  } catch (error) {
    output.innerHTML = `<div class="error">${esc(error.message)}</div>`;
  } finally {
    button.disabled = false;
  }
});
