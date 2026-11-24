import { useEffect, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowDown, BookOpen, X } from 'lucide-react';
import type { ClaimVerifyResponse, Decision } from '../types';
import { SourceTag, VerdictBadge } from './ui';

export function ClaimDrawer({ claim, open, onClose }: { claim: ClaimVerifyResponse | null; open: boolean; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const handle = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [open, onClose]);
  const reason = claim ? gateReason(claim.decision) : '';
  return <AnimatePresence>
    {open && claim && <>
      <motion.button className="drawer-scrim" aria-label="Close claim details" onClick={onClose} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
      <motion.aside className="claim-drawer" role="dialog" aria-modal="true" aria-labelledby="drawer-title" initial={{ x: 32, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 32, opacity: 0 }} transition={{ duration: 0.2, ease: 'easeOut' }}>
        <div className="drawer-head"><div><span className="drawer-kicker">{claim.category}</span><h2 id="drawer-title">Claim details</h2></div><button ref={closeRef} className="icon-button" onClick={onClose} aria-label="Close details"><X size={19} /></button></div>
        <div className="drawer-claim"><span className="label-overline">Claim</span><p>{claim.claim_text}</p><VerdictBadge decision={claim.decision} /></div>
        <div className="drawer-section"><div className="section-heading"><h3>Retrieved evidence</h3><BookOpen size={17} /></div>
          {claim.retrieved_contexts.length ? <div className="evidence-list">{claim.retrieved_contexts.map((text, index) => <article className="evidence-item" key={`${index}-${text}`}><div><SourceTag>{text.match(/^\[([^\]]+)\]/)?.[0] ?? 'Evidence'}</SourceTag><span className="evidence-number">Source {index + 1}</span></div><p><EvidenceText text={text} claimText={claim.claim_text} /></p></article>)}</div> : <div className="empty-evidence"><BookOpen size={17} /><span>No evidence passage is available for this claim.</span></div>}
        </div>
        <div className={`gate-reason gate-${claim.decision.toLowerCase()}`}><div className="section-heading"><h3>Decision</h3><VerdictBadge decision={claim.decision} compact /></div><p>{reason}</p><small>Predicted label · {claim.predicted_label}</small></div>
        {claim.regenerated_claim && <div className="regenerated-box"><div className="formula-title"><ArrowDown size={17} /><span>Regenerated claim</span></div><p>{claim.regenerated_claim}</p></div>}
      </motion.aside>
    </>}
  </AnimatePresence>;
}
function gateReason(decision: Decision) {
  if (decision === 'PASS') return 'The claim is supported by the available evidence and is retained.';
  if (decision === 'FLAG') return 'The evidence is incomplete or uncertain, so the claim is flagged for caution.';
  return 'The evidence did not meet the required standard, so the claim is blocked and may be regenerated using grounded context.';
}
function EvidenceText({ text, claimText }: { text: string; claimText: string }) {
  const clean = text.replace(/^\[[^\]]+\]\s*(?:\[[^\]]+\]\s*)?/, '');
  const ignored = new Set(['the', 'and', 'that', 'this', 'with', 'from', 'into', 'what', 'when', 'where', 'which', 'does', 'have', 'are', 'for', 'not', 'its', 'is', 'at', 'to', 'of', 'in', 'on', 'by']);
  const terms = [...new Set((claimText.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((term) => term.length > 2 && !ignored.has(term)))].sort((a, b) => b.length - a.length);
  if (!terms.length) return <>{clean}</>;
  const pattern = new RegExp(`(${terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  return <>{clean.split(pattern).map((part, index) => terms.includes(part.toLowerCase()) ? <mark key={index}>{part}</mark> : part)}</>;
}
