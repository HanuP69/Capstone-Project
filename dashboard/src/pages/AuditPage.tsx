import { useMemo, useState } from 'react';
import { Download, FileSearch, Search, SlidersHorizontal, ArrowDownAZ } from 'lucide-react';
import { ClaimDrawer } from '../components/ClaimDrawer';
import { PageFrame, PageHeader, Panel, VerdictBadge } from '../components/ui';
import type { AuditRow, ClaimVerifyResponse, DataBundle } from '../types';

export function AuditPage({ data }: { data: DataBundle }) {
  const [search, setSearch] = useState('');
  const [verdict, setVerdict] = useState('ALL');
  const [category, setCategory] = useState('ALL');
  const [sort, setSort] = useState<'category' | 'G'>('category');
  const [selected, setSelected] = useState<ClaimVerifyResponse | null>(null);
  const rows = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return data.primaryAudit.filter((row) => (verdict === 'ALL' || row.decision === verdict) && (category === 'ALL' || row.category === category) && (!needle || `${row.claim} ${row.id} ${row.category}`.toLowerCase().includes(needle)))
      .sort((a, b) => sort === 'G' ? b.G - a.G : a.category.localeCompare(b.category) || a.id.localeCompare(b.id));
  }, [data.primaryAudit, search, verdict, category, sort]);
  const categories = [...new Set(data.primaryAudit.map((row) => row.category))].sort();
  const selectRow = (row: AuditRow) => setSelected({
    claim_id: row.id, claim_text: row.claim, category: row.category, F_score: row.F, P_score: row.P, D_score: row.D, G_score: row.G,
    decision: row.decision, predicted_label: row.pred, fact_explanation: row.fact_explanation, policy_explanation: row.policy_explanation,
    retrieved_contexts: [row.fact_explanation, row.policy_explanation].filter((text) => text.includes('[Synthetic')),
    regenerated_claim: row.regenerated_claim || null, weighting_mode: 'audit_source', retrieval_method: 'hybrid_rrf', is_synthetic_benchmark: true,
  });
  function exportCsv() {
    const columns = ['id', 'category', 'claim', 'decision', 'F', 'P', 'D', 'G', 'pred', 'regenerated_claim'];
    const quote = (value: unknown) => `"${String(value ?? '').replaceAll('"', '""')}"`;
    const csv = [columns.join(','), ...rows.map((row) => columns.map((column) => quote(row[column as keyof AuditRow])).join(','))].join('\r\n');
    const href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = href; link.download = 'capstone-audit-filtered.csv'; link.click(); URL.revokeObjectURL(href);
  }
  return <PageFrame>
    <PageHeader title="Audit trail" description="Search the claim records and inspect how each decision was reached." action={<button className="button button-primary" onClick={exportCsv}><Download />Export filtered CSV</button>} />
    <div className="audit-summary"><div><strong>{rows.length}</strong><span>of {data.primaryAudit.length} records</span></div><span className="audit-sep" /><div><span className="audit-label">PASS</span><strong>{rows.filter((row) => row.decision === 'PASS').length}</strong></div><div><span className="audit-label">FLAG</span><strong>{rows.filter((row) => row.decision === 'FLAG').length}</strong></div><div><span className="audit-label">BLOCK</span><strong>{rows.filter((row) => row.decision === 'BLOCK').length}</strong></div></div>
    <Panel className="audit-panel"><div className="audit-toolbar"><label className="search-field"><Search /><input type="search" placeholder="Search claims, IDs or categories" value={search} onChange={(event) => setSearch(event.target.value)} aria-label="Search audit claims" /></label><div className="filter-controls"><label><SlidersHorizontal /><span>Verdict</span><select value={verdict} onChange={(event) => setVerdict(event.target.value)} aria-label="Filter by verdict"><option value="ALL">All</option><option>PASS</option><option>FLAG</option><option>BLOCK</option></select></label><label><span>Category</span><select value={category} onChange={(event) => setCategory(event.target.value)} aria-label="Filter by category"><option value="ALL">All categories</option>{categories.map((item) => <option key={item}>{item}</option>)}</select></label><label><ArrowDownAZ /><span>Sort</span><select value={sort} onChange={(event) => setSort(event.target.value as 'category' | 'G')} aria-label="Sort audit records"><option value="category">Category</option><option value="G">G score, high to low</option></select></label></div></div>
      <div className="audit-table-wrap"><table className="audit-table"><thead><tr><th>Claim</th><th>Category</th><th>Verdict</th><th>F</th><th>P</th><th>D</th><th>G(c)</th><th><span className="sr-only">Details</span></th></tr></thead><tbody>{rows.map((row) => <tr key={row.id} onClick={() => selectRow(row)} tabIndex={0} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') selectRow(row); }} aria-label={`Open claim ${row.id}, ${row.decision}, G ${row.G.toFixed(3)}`}><td><span className="audit-claim-id">{row.id}</span><span className="audit-claim-text">{row.claim}</span></td><td><span className="category-label">{row.category}</span></td><td><VerdictBadge decision={row.decision} compact /></td><td className="score-cell mono" data-label="Fact score">{row.F.toFixed(2)}</td><td className="score-cell mono" data-label="Policy score">{row.P.toFixed(2)}</td><td className="score-cell mono" data-label="Evidence score">{row.D.toFixed(2)}</td><td className="score-cell mono score-strong" data-label="Groundedness">{row.G.toFixed(3)}</td><td><span className="row-arrow" aria-hidden="true">↗</span></td></tr>)}</tbody></table>{rows.length === 0 && <div className="audit-empty"><FileSearch /><strong>No claims match these filters</strong><span>Try another search, verdict or category.</span></div>}</div>
      <div className="table-foot"><span>Showing {rows.length} of {data.primaryAudit.length} records</span><span>Select a row to view its evidence and scoring details.</span></div>
    </Panel>
    <ClaimDrawer claim={selected} open={Boolean(selected)} onClose={() => setSelected(null)} />
  </PageFrame>;
}
