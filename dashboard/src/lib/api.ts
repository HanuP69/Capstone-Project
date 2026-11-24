import type { AuditRow, ClaimVerifyResponse, DataBundle, QueryResponse } from '../types';

export const USE_MOCK = import.meta.env.USE_MOCK.toLowerCase() !== 'false';
export const EXAMPLE_QUERIES = [
  { label: 'Exam dates', query: 'When are mid-semester examinations scheduled?' },
  { label: 'Fee deadline', query: 'What is the B.Tech tuition fee and its payment deadline?' },
  { label: 'Hostel curfew', query: 'What time is hostel curfew?' },
  { label: 'Attendance rule', query: 'What attendance percentage is required to sit end-semester exams?' },
  { label: 'Test a false claim', query: 'Verify this claim: the semester registration deadline is 20 August 2026.' },
] as const;

export async function loadDatasets(): Promise<DataBundle> {
  const names = ['metrics', 'primary-audit', 'secondary-audit', 'mfaq-comparison', 'retrieval-comparison', 'weights-comparison', 'mfaq-audit', 'human-eval', 'synthetic-docs'] as const;
  const values = await Promise.all(names.map(async (name) => {
    const response = await fetch(`/data/${name}.json`);
    if (!response.ok) throw new Error(`Could not load ${name} benchmark data (${response.status}).`);
    return response.json();
  }));
  return {
    metrics: values[0] as DataBundle['metrics'], primaryAudit: values[1] as DataBundle['primaryAudit'],
    secondaryAudit: values[2] as DataBundle['secondaryAudit'], mfaqComparison: values[3] as DataBundle['mfaqComparison'],
    retrievalComparison: values[4] as DataBundle['retrievalComparison'], weightsComparison: values[5] as DataBundle['weightsComparison'],
    mfaqAudit: values[6] as DataBundle['mfaqAudit'], humanEval: values[7] as DataBundle['humanEval'], documents: values[8] as DataBundle['documents'],
  };
}

export async function checkBackend(): Promise<boolean> {
  if (USE_MOCK) return false;
  try { return (await fetch('/api/metrics', { signal: AbortSignal.timeout(2500) })).ok; }
  catch { return false; }
}

export async function verifyQuery(query: string, useMetaLearner = true, data?: DataBundle): Promise<QueryResponse> {
  if (!query.trim()) throw new Error('Enter a question to verify.');
  if (USE_MOCK) {
    if (!data) throw new Error('Benchmark data is not available yet.');
    return mockQuery(query.trim(), data);
  }
  const response = await fetch('/api/verify-query', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: query.trim(), institution: 'NITJ_REAL', use_meta_learner: useMetaLearner, retrieval_mode: 'hybrid' }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Verification failed (${response.status}).`);
  }
  return response.json() as Promise<QueryResponse>;
}

function fromAudit(row: AuditRow, documents: DataBundle['documents']): ClaimVerifyResponse {
  const fact = row.fact_explanation || '[Synthetic Record] No record explanation was stored.';
  const policy = row.policy_explanation || '[Synthetic Policy] No policy explanation was stored.';
  const stop = new Set(['the', 'and', 'that', 'this', 'with', 'from', 'into', 'what', 'when', 'where', 'which', 'does', 'have', 'are', 'for', 'not', 'its', 'is', 'at', 'to', 'of', 'in', 'on', 'by', 'a', 'an']);
  const terms = (row.claim.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((term) => term.length > 2 && !stop.has(term));
  const matchScore = (document: DataBundle['documents'][number]) => terms.reduce((score, term) => score + (document.text.toLowerCase().includes(term) || document.title.toLowerCase().includes(term) ? 1 : 0), 0);
  const document = [...documents].sort((a, b) => matchScore(b) - matchScore(a)).find((item) => item.category.toLowerCase() === row.category.toLowerCase() && matchScore(item) > 0)
    ?? [...documents].sort((a, b) => matchScore(b) - matchScore(a))[0];
  const passage = document ? `[Synthetic Context] ${document.title} — ${document.text}` : '';
  return {
    claim_id: row.id, claim_text: row.claim, category: row.category,
    F_score: row.F, P_score: row.P, D_score: row.D, G_score: row.G, decision: row.decision,
    predicted_label: row.pred, fact_explanation: fact, policy_explanation: policy,
    retrieved_contexts: [passage, fact, policy].filter((item) => item.includes('[Synthetic')),
    regenerated_claim: row.regenerated_claim || null, weighting_mode: 'meta_learner_mle',
    retrieval_method: 'hybrid_rrf', is_synthetic_benchmark: true,
  };
}

function mockQuery(query: string, data: DataBundle): QueryResponse {
  const q = query.toLowerCase();
  const rows = data.primaryAudit;
  let candidates: AuditRow[];
  if (/hallucin|false|fabricat|20 august|wrong|incorrect/.test(q)) {
    candidates = rows.filter((row) => row.decision === 'BLOCK' && /registration deadline|20 august/i.test(row.claim));
  } else if (/curfew|hostel/.test(q)) {
    candidates = rows.filter((row) => /hostel rules/i.test(row.category) && row.decision === 'PASS');
  } else if (/attendance|eligible to sit|end-semester exam/.test(q)) {
    candidates = rows.filter((row) => /attendance/i.test(row.claim) && row.decision === 'PASS');
  } else if (/fee|tuition|payment|deadline/.test(q)) {
    candidates = rows.filter((row) => /fee deadlines/i.test(row.category) && row.decision === 'PASS');
  } else {
    candidates = rows.filter((row) => /exam dates/i.test(row.category) && row.decision === 'PASS');
  }
  if (!candidates.length) candidates = rows.slice(0, 1);
  const ignoredTerms = new Set(['the', 'and', 'that', 'this', 'with', 'from', 'into', 'what', 'when', 'where', 'which', 'does', 'have', 'are', 'for', 'not', 'its', 'is', 'at', 'to', 'of', 'in', 'on', 'by', 'how']);
  const queryTerms = (q.match(/[a-z0-9]+/g) ?? []).filter((term) => term.length > 2 && !ignoredTerms.has(term));
  const relevance = (row: AuditRow) => queryTerms.reduce((score, term) => score + (row.claim.toLowerCase().includes(term) ? 2 : row.fact_explanation.toLowerCase().includes(term) ? 1 : 0), 0);
  candidates.sort((a, b) => relevance(b) - relevance(a));
  const isDeliberateFalsehood = /hallucin|false|fabricat|20 august|wrong|incorrect/.test(q);
  const claims = candidates.slice(0, isDeliberateFalsehood ? 1 : 2).map((row) => fromAudit(row, data.documents));
  const finalLines = claims.map((claim) => claim.decision === 'BLOCK'
    ? claim.regenerated_claim ? `${claim.regenerated_claim} [Synthetic benchmark context]` : 'Blocked claim omitted from the grounded answer because it did not meet the decision threshold.'
    : `${claim.claim_text}${claim.decision === 'FLAG' ? ' [Caution: benchmark evidence is incomplete]' : ' [Synthetic benchmark context]'}`);
  return {
    query, draft_response: claims.map((claim) => claim.claim_text).join(' '), final_grounded_answer: finalLines.join(' '),
    overall_groundedness: claims.reduce((total, claim) => total + claim.G_score, 0) / claims.length,
    total_claims: claims.length, passed_claims: claims.filter((c) => c.decision === 'PASS').length,
    flagged_claims: claims.filter((c) => c.decision === 'FLAG').length, blocked_claims: claims.filter((c) => c.decision === 'BLOCK').length,
    latency_seconds: 0, claims, dataset_mode: '[Synthetic Benchmark] — mock mode using repository audit records',
  };
}
