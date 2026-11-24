export function ScoreGauge({ label, shortLabel, value, color }: { label: string; shortLabel: string; value: number; color: string }) {
  const percent = Math.max(0, Math.min(100, value * 100));
  const radius = 23;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - percent / 100);
  return <div className="score-gauge" role="img" aria-label={`${label}: ${percent.toFixed(1)} percent`}>
    <div className="gauge-ring"><svg className="gauge-ring-svg" viewBox="0 0 56 56" aria-hidden="true"><circle className="gauge-track" cx="28" cy="28" r={radius} /><circle className="gauge-progress" cx="28" cy="28" r={radius} stroke={color} strokeDasharray={circumference} strokeDashoffset={offset} /></svg><div className="gauge-center"><strong>{value.toFixed(2)}</strong><span>{shortLabel}</span></div></div>
    <div className="gauge-caption"><strong>{label}</strong><span>Agent score</span></div>
  </div>;
}
