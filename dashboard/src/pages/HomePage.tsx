import { Link } from 'react-router-dom';
import type { AuditRow, DataBundle } from '../types';
import { PageFrame, VerdictBadge } from '../components/ui';

export function HomePage({ data }: { data: DataBundle }) {
  const exampleRows = data.primaryAudit.filter((row) => /exam dates/i.test(row.category) && /mid-semester/i.test(row.claim)).slice(0, 2);
  const previewRows = exampleRows.length ? exampleRows : data.primaryAudit.slice(0, 2);
  return <PageFrame>
    <section className="home-hero">
      <div className="home-intro">
        <h1>Academic answers, checked claim by claim</h1>
        <p>Language models can invent exam dates, fees and rules. This project checks each statement against institutional records and policy before presenting an answer.</p>
        <Link className="button button-primary" to="/verify">Ask about NIT Jamshedpur <span aria-hidden="true">→</span></Link>
      </div>
      <ExamplePreview rows={previewRows} />
    </section>

    <section className="home-results" aria-labelledby="home-results-title">
      <div className="section-heading-block"><div><h2 id="home-results-title">Results overview</h2><p>Benchmark performance and supporting evaluation measures are collected on the Results page.</p></div><Link to="/evaluation">View results <span aria-hidden="true">→</span></Link></div>
    </section>

    <section className="home-method" aria-labelledby="home-method-title">
      <div className="section-heading-block"><div><h2 id="home-method-title">How the checks work</h2><p>Three signals are combined for each extracted claim.</p></div><Link to="/pipeline">See the pipeline <span aria-hidden="true">→</span></Link></div>
      <div className="method-columns">
        <article><h3>Extract claims</h3><p>A draft answer is split into statements that can be checked separately.</p></article>
        <article><h3>Check three sources</h3><p>Fact records, policy rules and retrieved passages each provide a score.</p></article>
        <article><h3>Apply the decision gate</h3><p>Claims pass, receive a caution, or are blocked when evidence is too weak.</p></article>
      </div>
    </section>
  </PageFrame>;
}

function ExamplePreview({ rows }: { rows: AuditRow[] }) {
  return <section className="example-preview" aria-label="Example verification result">
    <div className="preview-head"><span>Benchmark sample</span><span>NIT Jamshedpur</span></div>
    <p className="preview-query">“When are mid-semester examinations scheduled?”</p>
    <div className="preview-claims">
      {rows.map((row) => <div className="preview-claim" key={row.id}><div><span className="preview-category">{row.category}</span><p>{row.claim}</p></div><VerdictBadge decision={row.decision} compact /></div>)}
      {!rows.length && <p>No example records are available.</p>}
    </div>
    <div className="preview-foot"><span>Evidence review</span><Link to="/verify">Try a question <span aria-hidden="true">→</span></Link></div>
  </section>;
}
