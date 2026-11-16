export function ScoreGauge({ label, shortLabel, value, color }: { label: string; shortLabel: string; value: number; color: string }) {
  const percent = Math.max(0, Math.min(100, value * 100));
  return <div className="score-gauge" aria-label={`${label}: ${percent.toFixed(1)} percent`}>
    <div className="gauge-ring" style={{ '--gauge-value': `${percent * 3.6}deg`, '--gauge-color': color } as React.CSSProperties}><div className="gauge-center"><strong>{value.toFixed(2)}</strong><span>{shortLabel}</span></div></div>
    <div className="gauge-caption"><strong>{label}</strong><span>Agent score</span></div>
  </div>;
}
