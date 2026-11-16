import { useEffect, useState } from 'react';
import { Check, CircleCheck, CircleDot, Play, RotateCcw } from 'lucide-react';
import { PageFrame, PageHeader, Panel } from '../components/ui';

const stages = [
  { id: 'query', title: 'Student question', sub: 'A campus-related request' },
  { id: 'generator', title: 'Draft answer', sub: 'A candidate response' },
  { id: 'extractor', title: 'Extract claims', sub: 'Separate checkable statements' },
  { id: 'agents', title: 'Parallel checks', sub: 'Fact · policy · evidence' },
  { id: 'scoring', title: 'Combine scores', sub: 'Learned weighted G(c)' },
  { id: 'gate', title: 'Apply thresholds', sub: 'Pass · flag · block' },
  { id: 'regenerate', title: 'Revise answer', sub: 'Remove or qualify weak claims' },
  { id: 'final', title: 'Return answer', sub: 'Keep claim-level evidence' },
];

export function PipelinePage() {
  const [step, setStep] = useState(-1);
  const [running, setRunning] = useState(false);
  const start = () => { if (!running) { setStep(0); setRunning(true); } };
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setStep((current) => {
      if (current >= stages.length - 1) { window.clearInterval(id); setRunning(false); return current; }
      return current + 1;
    }), 620);
    return () => window.clearInterval(id);
  }, [running]);

  return <PageFrame>
    <PageHeader title="Pipeline" description="A draft answer is split into claims, checked in parallel, scored and returned with evidence." />
    <Panel className="pipeline-panel" title="Verification flow" subtitle="The three checks run at the same time for each claim." action={<button className="button button-primary" onClick={start} disabled={running}><Play size={16} />{running ? 'Running example…' : step === stages.length - 1 ? 'Run again' : 'Run example'}</button>}>
      <div className="flow-map" aria-label="Claim verification process">
        <div className="flow-row flow-row-input">{stages.slice(0, 3).map((stage, index) => <Stage key={stage.id} title={stage.title} sub={stage.sub} active={step === index} done={step > index} />)}</div>
        <div className="flow-arrow-line"><span className="flow-down" aria-hidden="true" />Each claim is checked separately</div>
        <div className={`agent-branch ${step >= 3 ? 'agent-branch-active' : ''}`}>
          <div className="agent-branch-label">Three checks run in parallel</div>
          <div className="agent-cards">
            <AgentCard label="Fact check" code="F" description="Compare with recorded dates, fees and facts" active={step >= 3} />
            <AgentCard label="Policy check" code="P" description="Apply rules such as attendance and eligibility" active={step >= 3} />
            <AgentCard label="Evidence check" code="D" description="Find relevant passages in source material" active={step >= 3} />
          </div>
        </div>
        <div className="flow-row flow-row-ending">{stages.slice(4).map((stage, offset) => { const index = offset + 4; return <Stage key={stage.id} title={stage.title} sub={stage.sub} active={step === index} done={step > index} />; })}</div>
      </div>
      <div className="pipeline-foot"><span><span className={`activity-dot ${running ? 'activity-pulse' : ''}`} />{running ? `Current step: ${stages[step]?.title}` : step === stages.length - 1 ? 'Example complete. Each claim keeps its decision and evidence.' : 'Run the example to follow a claim through the checks.'}</span></div>
    </Panel>
    <div className="pipeline-bottom-grid">
      <Panel title="Groundedness score" subtitle="Weighted fact, policy and evidence scores"><p className="panel-copy">The three signals are combined into one score for each claim.</p><div className="score-equation">G(c) <span>=</span> <b>0.508</b> F <span>+</span> <b>0.315</b> P <span>+</span> <b>0.177</b> D</div><div className="weight-bars"><Weight label="Fact check" value={50.8} /><Weight label="Policy" value={31.5} /><Weight label="Evidence" value={17.7} /></div></Panel>
      <Panel title="Decision thresholds" subtitle="The same score limits are used for each claim"><div className="threshold-stack"><div><span className="threshold-sign threshold-pass"><CircleCheck size={17} /></span><span>G(c) ≥ 0.72</span><strong>PASS</strong><small>Keep the claim</small></div><div><span className="threshold-sign threshold-flag"><CircleDot size={17} /></span><span>0.50 ≤ G(c) &lt; 0.72</span><strong>FLAG</strong><small>Add a caution</small></div><div><span className="threshold-sign threshold-block"><RotateCcw size={17} /></span><span>G(c) &lt; 0.50</span><strong>BLOCK</strong><small>Revise or omit</small></div></div></Panel>
    </div>
  </PageFrame>;
}

function Stage({ title, sub, active, done }: { title: string; sub: string; active: boolean; done: boolean }) {
  return <div className={`flow-node ${active ? 'flow-node-active' : ''} ${done ? 'flow-node-done' : ''}`} aria-current={active ? 'step' : undefined}><div className="flow-node-top"><span className="node-state">{done ? <Check size={16} /> : null}</span></div><strong>{title}</strong><span>{sub}</span></div>;
}
function AgentCard({ label, code, description, active }: { label: string; code: string; description: string; active: boolean }) {
  return <div className={`agent-card ${active ? 'agent-active' : ''}`}><span className="agent-code">{code}</span><strong>{label}</strong><span>{description}</span></div>;
}
function Weight({ label, value }: { label: string; value: number }) { return <div className="weight-row"><span>{label}</span><div><span style={{ width: `${value}%` }} /></div><strong>{value.toFixed(1)}%</strong></div>; }
