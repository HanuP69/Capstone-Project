import { useState } from 'react';
import { ArrowUpRight, CircleHelp, LoaderCircle, Send } from 'lucide-react';
import { ClaimDrawer } from '../components/ClaimDrawer';
import { Button, PageFrame, PageHeader, Panel, VerdictBadge } from '../components/ui';
import { EXAMPLE_QUERIES, verifyQuery } from '../lib/api';
import type { ClaimVerifyResponse, DataBundle, QueryResponse } from '../types';

export function VerifyPage({ data }: { data: DataBundle }) {
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [query, setQuery] = useState('');
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
    catch (err) { setError(err instanceof TypeError ? 'Cannot reach the local NIT JSR backend. Start the root FastAPI app on port 8000 and check that Ollama is running.' : err instanceof Error ? err.message : 'Verification failed. Please try again.'); }
    finally { setBusy(false); }
  }

  return <PageFrame>
    <PageHeader title="Verify a claim" description="Ask a question about NIT Jamshedpur. The answer is checked against the ingested archive in ChromaDB." />
    <Panel className="query-panel">
      <form onSubmit={(event) => { event.preventDefault(); void run(); }}>
        <label className="query-label" htmlFor="query"><span>Your question</span><span className="query-hint">NIT Jamshedpur</span></label>
        <textarea id="query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask about an academic date, rule or fee…" rows={3} maxLength={1000} aria-describedby="query-help" />
        <div className="query-controls"><div className="query-controls-left"><span className="institution-chip">NIT Jamshedpur</span><span className="control-divider" />
          <label className="meta-toggle"><input type="checkbox" checked={useMeta} onChange={(event) => setUseMeta(event.target.checked)} /><span className="custom-check" /><span>Use learned weights</span><span className="tooltip-wrap"><CircleHelp size={16} /><span className="tooltip">Applies the learned weighting model when assessing each claim.</span></span></label>
        </div>
        <Button type="submit" className="button-primary run-button" disabled={busy || !query.trim()}>{busy ? <><LoaderCircle className="spin" size={17} /> Checking…</> : <><Send size={16} /> Verify answer</>}</Button></div>
        <span id="query-help" className="sr-only">Enter a student question. Each claim will be compared with the available records.</span>
      </form>
      {error && <div className="inline-error" role="alert"><span>{error}</span><button className="text-button" onClick={() => void run()}>Try again</button></div>}
      <div className="examples-row"><span className="example-label">Try another example</span>{EXAMPLE_QUERIES.map((item) => <button className="example-chip" key={item.label} onClick={() => void run(item.query)} disabled={busy}>{item.label}<ArrowUpRight size={14} /></button>)}</div>
    </Panel>

    {result ? <>
      <div className="verify-results-grid">
        <Panel title="Draft answer" subtitle="Before the claim checks" className="answer-panel"><div className="answer-copy">{result.draft_response}</div></Panel>
        <Panel title="Claim review" subtitle="Select a claim to see its verdict and evidence" className="claim-list-panel"><div className="claim-chip-list">{result.claims.map((claim, index) => <button key={claim.claim_id} className={`claim-chip claim-chip-${claim.decision.toLowerCase()}`} onClick={() => setSelected(claim)}><span className="claim-index">{index + 1}</span><span className="claim-chip-text">{claim.claim_text}</span><VerdictBadge decision={claim.decision} compact /><ArrowUpRight className="claim-open" size={16} aria-hidden="true" /></button>)}</div></Panel>
      </div>
      <Panel className="final-answer-panel" title="Answer after verification" subtitle="The final response reflects the claim decisions"><div className="answer-transform"><div className="final-copy">{cleanAnswer(result.final_grounded_answer)}</div></div><div className="answer-foot">Select a claim above to inspect the evidence used for its decision.</div></Panel>
    </> : <Panel title="Ready to verify" subtitle="Your response will appear here after you ask a question."><p className="panel-copy">The default path uses the archived NIT Jamshedpur corpus indexed in ChromaDB. Source freshness depends on the documents already ingested; this is not a live institute feed.</p></Panel>}
    <ClaimDrawer claim={selected} open={Boolean(selected)} onClose={() => setSelected(null)} />
  </PageFrame>;
}

function cleanAnswer(text: string) {
  return text.replace(/\s*\[Synthetic benchmark context\]/gi, '').replace(/\s*\[Caution: benchmark evidence is incomplete\]/gi, ' (benchmark evidence is incomplete)');
}
