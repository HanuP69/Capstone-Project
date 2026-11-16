import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { DataBundle } from '../types';
import { MetricCard, PageFrame, PageHeader, Panel } from '../components/ui';

const conditionStyles = [
  { label: 'Plain LLM', token: 'Plain LLM', color: '#A8A29E', falsehood: 100 },
  { label: 'RAG-only', token: 'RAG-only', color: '#D6D3D1', falsehood: 35 },
  { label: 'Self-Consistency', token: 'Self-consistency', color: '#57534E', falsehood: 100 },
  { label: 'Proposed Pipeline', token: 'Proposed Verification Pipeline', color: '#EA580C', falsehood: 2.5 },
];
const categories = [
  { category: 'Exam dates', baseline: 100, proposed: 0 },
  { category: 'Fee deadlines', baseline: 100, proposed: 0 },
  { category: 'Hostel rules', baseline: 100, proposed: 12.5 },
  { category: 'Circulars', baseline: 100, proposed: 0 },
  { category: 'Syllabus', baseline: 100, proposed: 0 },
];
const chartTooltip = { border: '1px solid var(--border)', borderRadius: 8, background: 'var(--surface)', color: 'var(--text)', fontSize: 14, boxShadow: 'none' };
const axisTick = { fill: 'var(--muted)', fontSize: 14 };
const percent = (value: number) => `${value.toFixed(1)}%`;
const conditionTick = (value: string) => value === 'Self-Consistency' ? 'Self-Cons.' : value === 'Proposed Pipeline' ? 'Proposed' : value;

export function EvaluationPage({ data }: { data: DataBundle }) {
  const main = conditionStyles.map((style) => {
    const row = data.metrics.find((item) => item.Condition.includes(style.token) && item.Dataset.startsWith('NIT Jamshedpur'));
    return { name: style.label, accuracy: (row?.Accuracy ?? 0) * 100, f1: (row?.F1 ?? 0) * 100, missed: style.falsehood };
  });
  const mfaq = conditionStyles.map((style) => {
    const row = data.mfaqComparison.find((item) => item.Condition.toLowerCase().includes(style.token.toLowerCase().replace('proposed verification pipeline', 'proposed')));
    return { name: style.label, accuracy: (row?.Accuracy ?? 0) * 100, f1: (row?.F1_Score ?? 0) * 100 };
  });
  const weightRows = data.weightsComparison.map((row, index) => ({ name: ['Equal', 'Heuristic', 'Grid search', 'Meta-learner'][index] ?? row.Weighting_Method, fact: row.w1_Fact * 100, policy: row.w2_Policy * 100, grounding: row.w3_Grounding * 100 }));
  const retrievalRows = data.retrievalComparison.map((row) => ({ name: row.Method.replace(/^\d\.\s*/, '').replace(' Only', ''), recall1: row['Recall@1'] * 100, recall5: row['Recall@5'] * 100, recall10: row['Recall@10'] * 100 }));
  const human = data.humanEval;

  return <PageFrame>
    <PageHeader title="Evaluation results" description="Reported accuracy, falsehood detection, retrieval and human agreement results." />
    <div className="result-summary evaluation-kpis"><MetricCard className="metric-highlight" label="Accuracy" value="91.7%" note="Proposed pipeline · 120 claims" /><MetricCard className="metric-highlight" label="Missed falsehoods" value="2.5%" note="Proposed pipeline · 120 claims" /><MetricCard label="Cohen’s kappa" value={human.annotator_1_vs_annotator_2_kappa.toFixed(3)} note={`${human.sample_size} human-reviewed claims`} /><MetricCard label="ROC–AUC" value="0.926" note="Five-fold validation" /></div>

    <div className="evaluation-grid evaluation-grid-wide">
      <Panel title="Four-condition comparison" subtitle="Accuracy and F1 score · NIT Jamshedpur · percent" className="chart-panel"><div className="chart-legend-custom">{conditionStyles.map((condition) => <span key={condition.label}><i style={{ background: condition.color }} />{condition.label}</span>)}</div><div className="chart-box"><ResponsiveContainer width="100%" height="100%"><BarChart data={main} margin={{ top: 24, right: 16, left: 8, bottom: 8 }} barGap={5}><CartesianGrid vertical={false} stroke="var(--grid)" /><XAxis dataKey="name" tick={axisTick} tickFormatter={conditionTick} axisLine={{ stroke: 'var(--border)' }} tickLine={false} interval={0} /><YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={axisTick} axisLine={false} tickLine={false} width={48} label={{ value: 'Percent (%)', angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 14 }} /><Tooltip contentStyle={chartTooltip} formatter={(value: unknown) => [`${Number(value).toFixed(1)}%`, '']} /><Legend wrapperStyle={{ fontSize: 14, color: 'var(--muted)', paddingTop: 8 }} /><Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>{main.map((row, index) => <Cell key={row.name} fill={conditionStyles[index].color} />)}<LabelList dataKey="accuracy" position="top" formatter={(value: unknown) => `${Number(value).toFixed(0)}%`} fill="var(--text)" fontSize={12} /></Bar><Bar dataKey="f1" name="F1 score" radius={[4, 4, 0, 0]}>{main.map((row, index) => <Cell key={row.name} fill={conditionStyles[index].color} fillOpacity={0.5} />)}<LabelList dataKey="f1" position="top" formatter={(value: unknown) => `${Number(value).toFixed(0)}%`} fill="var(--text)" fontSize={12} /></Bar></BarChart></ResponsiveContainer></div></Panel>
      <Panel title="Missed falsehood rate" subtitle="Lower is better · percent of false claims not detected" className="chart-panel"><div className="chart-box chart-box-short"><ResponsiveContainer width="100%" height="100%"><BarChart data={main} margin={{ top: 24, right: 12, left: 8, bottom: 8 }}><CartesianGrid vertical={false} stroke="var(--grid)" /><XAxis dataKey="name" tick={axisTick} tickFormatter={conditionTick} axisLine={{ stroke: 'var(--border)' }} tickLine={false} interval={0} /><YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={axisTick} axisLine={false} tickLine={false} width={48} /><Tooltip contentStyle={chartTooltip} formatter={(value: unknown) => [`${Number(value).toFixed(1)}%`, 'Missed falsehoods']} /><Bar dataKey="missed" name="Missed falsehoods" radius={[4, 4, 0, 0]}>{main.map((row, index) => <Cell key={row.name} fill={conditionStyles[index].color} />)}<LabelList dataKey="missed" position="top" formatter={(value: unknown) => `${Number(value).toFixed(1)}%`} fill="var(--text)" fontSize={12} /></Bar></BarChart></ResponsiveContainer></div></Panel>
    </div>

    <div className="evaluation-grid evaluation-grid-wide">
      <Panel title="Falsehood rate by category" subtitle="Plain LLM compared with the proposed pipeline · percent" className="chart-panel"><div className="chart-box"><ResponsiveContainer width="100%" height="100%"><BarChart data={categories} margin={{ top: 24, right: 16, left: 8, bottom: 8 }}><CartesianGrid vertical={false} stroke="var(--grid)" /><XAxis dataKey="category" tick={axisTick} axisLine={{ stroke: 'var(--border)' }} tickLine={false} /><YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={axisTick} axisLine={false} tickLine={false} width={48} /><Tooltip contentStyle={chartTooltip} formatter={(value: unknown) => [`${Number(value).toFixed(1)}%`, 'Falsehood rate']} /><Legend wrapperStyle={{ fontSize: 14, color: 'var(--muted)', paddingTop: 8 }} /><Bar dataKey="baseline" name="Plain LLM" fill="#A8A29E" radius={[4, 4, 0, 0]}><LabelList dataKey="baseline" position="top" formatter={(value: unknown) => `${Number(value).toFixed(0)}%`} fill="var(--text)" fontSize={12} /></Bar><Bar dataKey="proposed" name="Proposed Pipeline" fill="#EA580C" radius={[4, 4, 0, 0]}><LabelList dataKey="proposed" position="top" formatter={(value: unknown) => `${Number(value).toFixed(1)}%`} fill="var(--text)" fontSize={12} /></Bar></BarChart></ResponsiveContainer></div></Panel>
      <Panel title="ROC curve and confusion matrix" subtitle="Classification performance on the groundedness task" className="chart-panel"><div className="figure-stack"><figure><img src="/data/figures/roc_curve_comparison.png" alt="ROC curve comparison across verification conditions" /><figcaption>ROC curve · proposed model AUC 0.926</figcaption></figure><figure><img src="/data/figures/confusion_matrix_groundedness.png" alt="Confusion matrix for groundedness classification" /><figcaption>Groundedness classification confusion matrix</figcaption></figure></div></Panel>
    </div>

    <div className="evaluation-grid evaluation-grid-wide">
      <Panel title="Scoring weights" subtitle="Share assigned to each verification signal · percent" className="chart-panel"><div className="chart-box"><ResponsiveContainer width="100%" height="100%"><BarChart data={weightRows} margin={{ top: 16, right: 12, left: 8, bottom: 8 }}><CartesianGrid vertical={false} stroke="var(--grid)" /><XAxis dataKey="name" tick={axisTick} axisLine={{ stroke: 'var(--border)' }} tickLine={false} /><YAxis domain={[0, 60]} tickFormatter={(value) => `${value}%`} tick={axisTick} axisLine={false} tickLine={false} width={48} /><Tooltip contentStyle={chartTooltip} formatter={(value: unknown) => [`${Number(value).toFixed(1)}%`, '']} /><Legend wrapperStyle={{ fontSize: 14, color: 'var(--muted)', paddingTop: 8 }} /><Bar dataKey="fact" name="Fact check" fill="#57534E" radius={[4, 4, 0, 0]} /><Bar dataKey="policy" name="Policy" fill="#A8A29E" radius={[4, 4, 0, 0]} /><Bar dataKey="grounding" name="Evidence" fill="#D6D3D1" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div><p className="chart-source">Learned weights: fact 0.508 · policy 0.315 · evidence 0.177.</p></Panel>
      <Panel title="Hybrid retrieval" subtitle="MFAQ corpus · recall by rank" className="chart-panel"><div className="chart-box"><ResponsiveContainer width="100%" height="100%"><LineChart data={retrievalRows} margin={{ top: 16, right: 16, left: 8, bottom: 8 }}><CartesianGrid stroke="var(--grid)" /><XAxis dataKey="name" tick={axisTick} axisLine={{ stroke: 'var(--border)' }} tickLine={false} /><YAxis domain={[50, 100]} tickFormatter={(value) => `${value}%`} tick={axisTick} axisLine={false} tickLine={false} width={48} label={{ value: 'Recall (%)', angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 14 }} /><Tooltip contentStyle={chartTooltip} formatter={(value: unknown) => [`${Number(value).toFixed(1)}%`, '']} /><Legend wrapperStyle={{ fontSize: 14, color: 'var(--muted)', paddingTop: 8 }} /><Line type="monotone" dataKey="recall1" name="Recall@1" stroke="#57534E" strokeWidth={2} dot={{ r: 4 }} /><Line type="monotone" dataKey="recall5" name="Recall@5" stroke="#A8A29E" strokeWidth={2} dot={{ r: 4 }} /><Line type="monotone" dataKey="recall10" name="Recall@10" stroke="#78716C" strokeWidth={2} dot={{ r: 4 }} /></LineChart></ResponsiveContainer></div></Panel>
    </div>

    <Panel title="MFAQ comparison" subtitle={`Accuracy and F1 score · ${data.mfaqComparison[0]?.Sample_Size ?? 55} evaluated claims`}><div className="results-table-wrap"><table className="results-table"><thead><tr><th>Condition</th><th>Accuracy</th><th>F1 score</th></tr></thead><tbody>{mfaq.map((row, index) => <tr key={row.name}><td><span className="condition-swatch" style={{ background: conditionStyles[index].color }} />{row.name}</td><td>{percent(row.accuracy)}</td><td>{percent(row.f1)}</td></tr>)}</tbody></table></div><p className="chart-source">Retrieval Recall@10: 96.8% · MRR: 0.851 · 250 FAQ pairs.</p></Panel>
  </PageFrame>;
}
