import { useState } from 'react';
import { ArrowUpRight, CircleHelp, LoaderCircle, Send } from 'lucide-react';
import { ClaimDrawer } from '../components/ClaimDrawer';
import { Button, MetricCard, PageFrame, PageHeader, Panel, VerdictBadge } from '../components/ui';
import { EXAMPLE_QUERIES, USE_MOCK, verifyQuery } from '../lib/api';
import type { AuditRow, ClaimVerifyResponse, DataBundle, QueryResponse } from '../types';

export function VerifyPage({ data }: { data: DataBundle }) {
  const [result, setResult] = useState<QueryResponse>(() => createExample(data));
  const [query, setQuery] = useState<string>(EXAMPLE_QUERIES[0].query);
  const [selected, setSelected] = useState<ClaimVerifyResponse | null>(null);
  const [useMeta, setUseMeta] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function run(text = query) {
    if (!text.trim() || busy) return;
    setQuery(text);
    setError('');
    setBusy(true);
    try { setResult(await verifyQuery(text, useMeta, data)); }
    catch (err) { setError(err instanceof Error ? err.message : 'Verification failed. Please try again.'); }
    finally { setBusy(false); }
  }

  const isExample = result.dataset_mode === 'Prefilled example';
  return <PageFrame>
    <PageHeader title="Verify a claim" description="Enter a question about an academic date, fee or rule. The example below is ready to inspect." />
    <Panel className="query-panel">
      <form onSubmit={(event) => { event.preventDefault(); void run(); }}>
        <label className="query-label" htmlFor="query"><span>Your question</span><span className="query-hint">NIT Jamshedpur</span></label>
        <textarea id="query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask about an academic date, rule or fee…" rows={3} maxLength={1000} aria-describedby="query-help" />
        <div className="query-controls"><div className="query-controls-left"><span className="institution-chip">NIT Jamshedpur</span><span className="control-divider" />
          <label className="meta-toggle"><input type="checkbox" checked={useMeta} onChange={(event) => setUseMeta(event.target.checked)} /><span className="custom-check" /><span>Use learned weights</span><span className="tooltip-wrap"><CircleHelp size={16} /><span className="tooltip">Learned weights: fact 0.508, policy 0.315 and grounding 0.177.</span></span></label>
        </div>
        <Button type="submit" className="button-primary run-button" disabled={busy || !query.trim()}>{busy ? <><LoaderCircle className="spin" size={17} /> Checking…</> : <><Send size={16} /> Verify answer</>}</Button></div>
        <span id="query-help" className="sr-only">Enter a student question. Each claim will be compared with the available records.</span>
      </form>
      {error && <div className="inline-error" role="alert"><span>{error}</span><button className="text-button" onClick={() => void run()}>Try again</button></div>}
      <div className="examples-row"><span className="example-label">Try another example</span>{EXAMPLE_QUERIES.map((item) => <button className="example-chip" key={item.label} onClick={() => void run(item.query)} disabled={busy}>{item.label}<ArrowUpRight size={14} /></button>)}</div>
    </Panel>

    <div className="result-summary">
      <MetricCard label="Claims extracted" value={String(result.total_claims)} note="Statements checked separately" />
      <MetricCard label="Overall groundedness" value={result.overall_groundedness.toFixed(3)} note="Weighted claim score G(c)" />
      <MetricCard label="Passed" value={String(result.passed_claims)} note={`${result.flagged_claims} flagged · ${result.blocked_claims} blocked`} />
      <MetricCard label="Latency" value={isExample || USE_MOCK ? '—' : `${result.latency_seconds.toFixed(2)} s`} note={isExample || USE_MOCK ? 'Not measured for this result' : 'Time returned by the API'} />
    </div>

    <div className="verify-results-grid">
      <Panel title="Draft answer" subtitle="Before the claim checks" className="answer-panel"><div className="answer-copy">{result.draft_response}</div><div className="answer-meta"><span>{result.total_claims} claims extracted</span></div></Panel>
      <Panel title="Claim review" subtitle="Select a claim to see its scores and evidence" className="claim-list-panel"><div className="claim-chip-list">{result.claims.map((claim, index) => <button key={claim.claim_id} className={`claim-chip claim-chip-${claim.decision.toLowerCase()}`} onClick={() => setSelected(claim)}><span className="claim-index">{index + 1}</span><span className="claim-chip-text">{claim.claim_text}</span><span className="claim-score">{claim.G_score.toFixed(2)}</span><VerdictBadge decision={claim.decision} compact /><ArrowUpRight className="claim-open" size={16} aria-hidden="true" /></button>)}</div><div className="gate-legend"><strong>Decision thresholds</strong><span>PASS ≥ 0.72</span><span>FLAG ≥ 0.50</span><span>BLOCK &lt; 0.50</span></div></Panel>
    </div>

    <Panel className="final-answer-panel" title="Answer after verification" subtitle="The final response reflects the claim decisions"><div className="answer-transform"><div className="final-copy">{cleanAnswer(result.final_grounded_answer)}</div></div><div className="answer-foot">Select a claim above to inspect the evidence used for its decision.</div></Panel>
    <ClaimDrawer claim={selected} open={Boolean(selected)} onClose={() => setSelected(null)} useMeta={useMeta} />
  </PageFrame>;
}

function createExample(data: DataBundle): QueryResponse {
  const matching = data.primaryAudit.filter((row) => /exam dates/i.test(row.category) && /mid-semester/i.test(row.claim));
  const rows = (matching.length ? matching : data.primaryAudit.filter((row) => row.decision === 'PASS')).slice(0, 2);
  const claims = rows.map((row) => claimFromAudit(row, data));
  const finalAnswer = claims.map((claim) => claim.claim_text).join(' ');
  return {
    query: EXAMPLE_QUERIES[0].query,
    draft_response: finalAnswer,
    final_grounded_answer: finalAnswer,
    overall_groundedness: claims.length ? claims.reduce((total, claim) => total + claim.G_score, 0) / claims.length : 0,
    total_claims: claims.length,
    passed_claims: claims.filter((claim) => claim.decision === 'PASS').length,
    flagged_claims: claims.filter((claim) => claim.decision === 'FLAG').length,
    blocked_claims: claims.filter((claim) => claim.decision === 'BLOCK').length,
    latency_seconds: 0,
    claims,
    dataset_mode: 'Prefilled example',
  };
}

function claimFromAudit(row: AuditRow, data: DataBundle): ClaimVerifyResponse {
  const ignored = new Set(['the', 'and', 'that', 'this', 'with', 'from', 'into', 'what', 'when', 'where', 'which', 'does', 'have', 'are', 'for', 'not', 'its', 'is', 'at', 'to', 'of', 'in', 'on', 'by', 'a', 'an']);
  const terms = (row.claim.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((term) => term.length > 2 && !ignored.has(term));
  const matches = [...data.documents].sort((a, b) => scoreDocument(b) - scoreDocument(a));
  function scoreDocument(document: DataBundle['documents'][number]) {
    return terms.reduce((score, term) => score + (document.text.toLowerCase().includes(term) || document.title.toLowerCase().includes(term) ? 1 : 0), 0) + (document.category.toLowerCase() === row.category.toLowerCase() ? 2 : 0);
  }
  const document = matches.find((item) => scoreDocument(item) > 0);
  const evidence = document ? `[Synthetic Context] ${document.title} — ${document.text}` : '';
  return {
    claim_id: row.id,
    claim_text: row.claim,
    category: row.category,
    F_score: row.F,
    P_score: row.P,
    D_score: row.D,
    G_score: row.G,
    decision: row.decision,
    predicted_label: row.pred,
    fact_explanation: row.fact_explanation,
    policy_explanation: row.policy_explanation,
    retrieved_contexts: [evidence, row.fact_explanation, row.policy_explanation].filter((text) => text.includes('[Synthetic')),
    regenerated_claim: row.regenerated_claim || null,
    weighting_mode: 'meta_learner_mle',
    retrieval_method: 'hybrid_rrf',
    is_synthetic_benchmark: true,
  };
}

function cleanAnswer(text: string) {
  return text.replace(/\s*\[Synthetic benchmark context\]/gi, '').replace(/\s*\[Caution: benchmark evidence is incomplete\]/gi, ' (benchmark evidence is incomplete)');
}
