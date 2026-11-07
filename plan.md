# Capstone Notebook Plan — Groundedness Verification Pipeline

## 1. Project recap (from proposal)
- **Goal:** A domain-agnostic, multi-agent groundedness verification pipeline for
  institutional/academic LLM answers.
- **Core contribution:** claim-level verification + groundedness scoring
  `G(c) = w1·F(c) + w2·P(c) + w3·D(c)` + decision gate (pass / flag / block-and-regenerate).
- **Validation:** cross-institution (NITJ primary + public secondary), baselines
  (Plain LLM, RAG-only, Self-consistency), metrics (accuracy, P/R/F1, ROC/PR,
  confusion matrices, latency, hallucination reduction).

## 2. Current notebook state (`capstone_verification.ipynb`)
- **60 cells**, runs end-to-end on local Ollama (`gemma4:e4b` + `nomic-embed-text`).
- Datasets:
  - Primary: synthetic NITJ institutional KB + 60 labeled claims (true/false/unverifiable).
  - Secondary: SciFact (scientific claim verification, 60 claims).
  - Extra: MFAQ retrieval/QA benchmark (60 questions).
- Pipeline:
  - Retrieval: nomic-embed-text + FAISS.
  - Agents: LLM fact-check (record-aware), LLM policy-compliance, grounding
    (cosine for primary, LLM entailment for secondary).
  - Scoring + decision gate (fact-authoritative when record-backed).
  - LangGraph agentic demo with regeneration.
- Baselines: Plain LLM, LLM+RAG, Self-consistency.
- Evaluation: metrics tables, confusion matrices, ROC/PR (proposed), calibration
  (weight grid + threshold sensitivity), cross-institution comparison, latency,
  hallucination/missed-falsehood reduction, MFAQ retrieval + answer F1.
- Achieved: primary accuracy ~0.95, secondary ~1.00, MFAQ Recall@5 ~0.967.

## 3. Gaps vs proposal (to close in notebook / 50% scope)
1. **Real institutional data:** only synthetic NITJ KB used; no ingestion of actual
   circulars/handbook/PDFs.
2. **Claim extraction:** demonstrated on a sample, not integrated into the full
   evaluation loop (response → claims → verification).
3. **Regeneration feedback loop:** LangGraph demo only; not evaluated at scale.
4. **Cross-institution:** secondary is SciFact (scientific), not a university dataset.
5. **Human evaluation / annotation protocol:** not implemented (no agreement study).
6. **ROC/PR for baselines:** only proposed has ROC/PR.
7. **Audit trail export:** no per-claim JSON/CSV export.
8. **Scoring transfer:** weights calibrated on primary; transfer shown but not
   re-tuned/validated systematically.

## 4. Proposed next steps (prioritized)
### Phase A — strengthen core (notebook, high value)
- [ ] Integrate claim extraction into evaluation: generate LLM responses for
      queries, extract atomic claims, verify each, aggregate per-response score.
- [ ] Run regeneration loop on blocked claims and re-score (measure correction rate).
- [ ] Add per-claim audit-trail export (CSV/JSON) for the proposed pipeline.
- [ ] Add baseline ROC/PR (where a score exists) for fair comparison.

### Phase B — data & generalizability
- [ ] Ingest real NITJ documents (PDFs/circulars) into the KB; rebuild claims.
- [ ] Add a public university FAQ/policy dataset as an additional secondary
      (or keep MFAQ as the real-world retrieval benchmark).
- [ ] Expand synthetic NITJ dataset to ~200 claims with multi-hop/contradictory
      cases for harder evaluation.

### Phase C — evaluation depth (paper-ready)
- [ ] Human evaluation: sample 50 claims, annotate true/false/unverifiable,
      compute Cohen's kappa vs pipeline decisions.
- [ ] Ablation: remove fact-check / policy / grounding one at a time and
      measure accuracy drop.
- [ ] Error analysis: categorize false positives/negatives by claim type.
- [ ] Final consolidated results table for the paper (LaTeX export).

### Phase D — engineering (outside notebook, future)
- [ ] FastAPI backend + MongoDB claim logs (per proposal tech stack).
- [ ] Chroma/FAISS persistent store; Ollama/NIM integration.
- [ ] Frontend dashboard for claim-level confidence.

## 5. Immediate next action
1. Add claim-extraction + regeneration evaluation (Phase A) to the notebook.
2. Export per-claim audit trail.
3. Re-run end-to-end and refresh metrics.

## 6. Files
- Notebook: `capstone_verification.ipynb` (main, 60 cells)
- Extra: `averitec_verification.ipynb` (AVeriTeC stress test)
- Plan: `plan.md` (this file)
