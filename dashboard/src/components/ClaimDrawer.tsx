import { useEffect, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowDown, BookOpen, Calculator, X } from 'lucide-react';
import type { ClaimVerifyResponse, Decision } from '../types';
import { ScoreGauge } from './ScoreGauge';
import { SourceTag, VerdictBadge } from './ui';

export function ClaimDrawer({ claim, open, onClose, useMeta = true }: { claim: ClaimVerifyResponse | null; open: boolean; onClose: () => void; useMeta?: boolean }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const handle = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [open, onClose]);
  const weights = useMeta ? [0.508, 0.315, 0.177] : [0.4, 0.3, 0.3];
  const terms = claim ? [claim.F_score * weights[0], claim.P_score * weights[1], claim.D_score * weights[2]] : [0, 0, 0];
  const reason = claim ? gateReason(claim.decision, claim.G_score) : '';
  return <AnimatePresence>
    {open && claim && <>
      <motion.button className="drawer-scrim" aria-label="Close claim details" onClick={onClose} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
      <motion.aside className="claim-drawer" role="dialog" aria-modal="true" aria-labelledby="drawer-title" initial={{ x: 32, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 32, opacity: 0 }} transition={{ duration: 0.2, ease: 'easeOut' }}>
        <div className="drawer-head"><div><span className="drawer-kicker">{claim.category}</span><h2 id="drawer-title">Claim details</h2></div><button ref={closeRef} className="icon-button" onClick={onClose} aria-label="Close details"><X size={19} /></button></div>
        <div className="drawer-claim"><span className="label-overline">Claim</span><p>{claim.claim_text}</p><VerdictBadge decision={claim.decision} /></div>
        <div className="drawer-section"><div className="section-heading"><h3>Verification scores</h3><span>Range 0–1</span></div><div className="gauge-row">
          <ScoreGauge label="Fact check" shortLabel="F" value={claim.F_score} color="#57534E" />
          <ScoreGauge label="Policy" shortLabel="P" value={claim.P_score} color="#A8A29E" />
          <ScoreGauge label="Evidence" shortLabel="D" value={claim.D_score} color="#D6D3D1" />
        </div></div>
        <div className="formula-card"><div className="formula-title"><Calculator size={17} /><span>Groundedness calculation</span></div><div className="formula-row"><span>{weights[0].toFixed(3)} × F</span><span>+</span><span>{weights[1].toFixed(3)} × P</span><span>+</span><span>{weights[2].toFixed(3)} × D</span><span>=</span><strong>{claim.G_score.toFixed(3)}</strong></div><div className="contribution-row"><span>{terms[0].toFixed(3)}</span><span>{terms[1].toFixed(3)}</span><span>{terms[2].toFixed(3)}</span><span>reported G(c)</span></div><div className="weight-note">{useMeta ? 'Learned weights' : 'Default weights'} · PASS ≥ 0.72 · FLAG ≥ 0.50 · BLOCK &lt; 0.50</div></div>
        <div className="drawer-section"><div className="section-heading"><h3>Retrieved evidence</h3><BookOpen size={17} /></div>
          {claim.retrieved_contexts.length ? <div className="evidence-list">{claim.retrieved_contexts.map((text, index) => <article className="evidence-item" key={`${index}-${text}`}><div><SourceTag>{text.match(/^\[([^\]]+)\]/)?.[0] ?? 'Evidence'}</SourceTag><span className="evidence-number">Source {index + 1}</span></div><p><EvidenceText text={text} claimText={claim.claim_text} /></p></article>)}</div> : <div className="empty-evidence"><BookOpen size={17} /><span>No evidence passage is available for this claim.</span></div>}
        </div>
        <div className={`gate-reason gate-${claim.decision.toLowerCase()}`}><div className="section-heading"><h3>Decision</h3><VerdictBadge decision={claim.decision} compact /></div><p>{reason}</p><small>Predicted label · {claim.predicted_label}</small></div>
        {claim.regenerated_claim && <div className="regenerated-box"><div className="formula-title"><ArrowDown size={17} /><span>Regenerated claim</span></div><p>{claim.regenerated_claim}</p></div>}
      </motion.aside>
    </>}
  </AnimatePresence>;
}
function gateReason(decision: Decision, score: number) {
  if (decision === 'PASS') return `G(c) = ${score.toFixed(3)} meets the groundedness threshold of 0.72. The claim is kept with its available evidence.`;
  if (decision === 'FLAG') return `G(c) = ${score.toFixed(3)} is between 0.50 and 0.72. The claim is marked for caution because support is incomplete or uncertain.`;
  return `G(c) = ${score.toFixed(3)} is below 0.50. The claim is blocked and, when available, revised using grounded context.`;
}
function EvidenceText({ text, claimText }: { text: string; claimText: string }) {
  const clean = text.replace(/^\[[^\]]+\]\s*(?:\[[^\]]+\]\s*)?/, '');
  const ignored = new Set(['the', 'and', 'that', 'this', 'with', 'from', 'into', 'what', 'when', 'where', 'which', 'does', 'have', 'are', 'for', 'not', 'its', 'is', 'at', 'to', 'of', 'in', 'on', 'by']);
  const terms = [...new Set((claimText.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((term) => term.length > 2 && !ignored.has(term)))].sort((a, b) => b.length - a.length);
  if (!terms.length) return <>{clean}</>;
  const pattern = new RegExp(`(${terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  return <>{clean.split(pattern).map((part, index) => terms.includes(part.toLowerCase()) ? <mark key={index}>{part}</mark> : part)}</>;
}
