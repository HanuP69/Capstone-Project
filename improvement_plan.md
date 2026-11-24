# Improvement Plan — Honest, Reproducible, Baseline-Beating Capstone

**Status:** planning document only. No code changed by this pass.
**Environment confirmed:** Ollama 0.31.2 with `gemma4:e4b` (9.6 GB) and `nomic-embed-text:latest` (274 MB) installed locally.
**Goal:** replace every simulated/fabricated result with a real, reproducible experiment; fix the agents, gate, regeneration, retrieval, API and repo; then beat the baselines **on measured data**, with statistics that survive review.

---

## 0. Non-negotiable ground rules

1. **No number in any document may be typed by a human.** Every metric must be computed by a committed script and written to a CSV/JSON by that script. README, dashboard, LaTeX and `study_plan.md` must derive from those files or be deleted.
2. **No author-written prediction.** Baselines A/B/C must call the real LLM / real retriever. `if true_label == ...` branches are banned from evaluation code.
3. **No simulated annotators.** Human agreement comes from two real people labelling independently; raw labels are committed.
4. **Group-split dev/test by source document before any tuning.** All thresholds, weights, k, prompts, and judge calibration are tuned on **dev only**. Test is frozen until the final run.
5. **Same budget for every condition:** same model, same context size, same number of LLM calls (or call count reported), same retrieval corpus, same prompt except for the layer under test.
6. **Publish failures too.** If the proposed pipeline does not beat RAG on the frozen test set, we say so. We do not move goalposts after seeing results. "Winning" means a statistically supported paired improvement on held-out data.
7. **Everything is re-runnable in one command from a clean checkout**, with the notebook either re-executed top-to-bottom or stripped of pasted outputs.

---

## 1. Master flaw register (Claude's findings + deep-dive findings)

Severity: **P0** = invalidates results, **P1** = invalidates claims, **P2** = engineering correctness, **P3** = hygiene.

### 1.1 Evaluation integrity

| ID | Flaw | Evidence | Sev | Fixed in |
|---|---|---|---|---|
| F01 | NITJSR baselines A/B/C never call an LLM or retriever; they are `if true_label == ...` branches | `run_nitjsr_case_study.py:267-305` | P0 | §6 |
| F02 | Baselines A/B/C in the 150-claim suite are simulated; plain LLM is `np.random.rand() > 0.12` | `run_complete_ai_evaluation.py:275-295` | P0 | §6 |
| F03 | Master metrics table is typed literals (accuracy, precision, recall, F1, hallucination, latency) | `run_complete_ai_evaluation.py:574-579`; `master_evaluation_metrics.csv` | P0 | §6, §10 |
| F04 | Human eval = two RNG annotators (93%/91% correct, else random wrong); κ=0.812 is a property of those constants | `run_complete_ai_evaluation.py:540-566`; reproduced exactly with seed 42 | P0 | §5 |
| F05 | `consensus_human` falls back to ground truth on disagreement, so "pipeline-to-human agreement" is agreement with the answer key | `run_complete_ai_evaluation.py:553` | P0 | §5 |
| F06 | README calls F04 a "Human Annotator Agreement Study" | `README.md:87-91` | P0 | §10 |
| F07 | RAG Triad is not computed: faithfulness = 0.94 if first 3 query words appear in context else 0.82; answer relevance = 0.92 if answer >30 chars | `run_nitjsr_case_study.py:432-436` | P0 | §7 |
| F08 | Context relevance is `clip(12 × RRF + 0.35, 0.40, 1.0)` with no validation | `run_nitjsr_case_study.py:419-420` | P0 | §7 |
| F09 | Per-category chart arrays typed by hand (`[0,0,12.5,0,0]`); real distribution is Circulars 50%, Syllabus 12.5%, others 0% | `run_nitjsr_case_study.py:501-502`; recomputed from `nitjsr_audit_trail.csv` | P0 | §10 |
| F10 | Faithfulness reported as 0.936 (README/CSV), 94.2% (script print), 0.942 (app.py) | `README.md:208`, `run_nitjsr_case_study.py:456`, `app.py:362` | P0 | §10 |
| F11 | Context relevance 0.906 (README/CSV) vs 1.000 (app.py) | `app.py:363` | P1 | §10 |
| F12 | Proposed row: P=0.965, R=0.917, F1=0.958 is arithmetically inconsistent; recomputed macro-F1 = 0.915, binary F1 = 0.889, missed falsehoods = 2.5% | `master_evaluation_metrics.csv:5`; audit CSV | P0 | §6, §10 |
| F13 | README says 2.5% missed falsehoods, CSV says 0.0% | `README.md:76` vs `master_evaluation_metrics.csv:5` | P1 | §10 |
| F14 | "Macro-F1" in the NITJSR table is binary F1 for the `true` class; true macro-F1 is 0.9035 vs reported 0.918 (RAG 0.696 vs 0.721) | `run_nitjsr_case_study.py:348-352` | P0 | §6 |
| F15 | `recall = p_correct / 40.0` hardcodes the number of true claims | `run_nitjsr_case_study.py:351` | P1 | §6 |
| F16 | AUC = 0.942 appears in dashboard and viva notes but is computed nowhere | `dashboard.html:240`, `study_plan.md:249` | P1 | §10 |
| F17 | MFAQ baselines use salted Python `hash()` → predictions change per process (measured 47.3–60.0% across seeds); cannot be reproduced | `run_mfaq_full_agentic_pipeline.py:179,203`; seed sweep | P0 | §6 |
| F18 | MFAQ baseline latencies (0.85/1.10/2.75/1.45) are literals | `run_mfaq_full_agentic_pipeline.py:279-282` | P0 | §6 |
| F19 | ROC baseline scores are generated from ground truth (`uniform(0.15,0.45) if label=="false"`) → label leakage in the AUC plot | `run_complete_ai_evaluation.py:282` | P0 | §6 |
| F20 | "Dense RAG" baseline reads `rrf_score` and branches on keyword lists keyed by `true_label`; it is neither dense nor RAG; `top_sim` unused | `run_nitjsr_case_study.py:279-293` | P0 | §6 |
| F21 | MFAQ "multi-agent" sets P=0.5 for all 55 claims; false claims are token surgery; unverifiable = 7 fixed templates | `run_mfaq_full_agentic_pipeline.py:130-157,219` | P0 | §6 |
| F22 | Retrieval latency excludes query embedding and BM25 tokenization; "Dense 0.02 ms" is an argsort timing | `run_bm25_rrf_benchmark.py:111-115` | P1 | §6 |
| F23 | The only honest run (notebook) shows RAG 96.7% vs proposed 95.0% on primary, 98.3% vs 97.5% averaged, at ~2× latency — contradicting README's RAG=65% | notebook cells 26-29, 58 | P0 | §6, §12 |
| F24 | Plain LLM and self-consistency output a single class (acc 0.333, precision 0.111); "misses 100% of falsehoods" is largely abstention, not hallucination | notebook cells 29, 31; `README.md:73,75` | P1 | §6, §13 |
| F25 | Self-consistency uses the identical prompt to plain LLM, so its comparison is not meaningful | notebook cells 23, 26 | P1 | §6 |

### 1.2 Scoring, gate, retrieval

| ID | Flaw | Evidence | Sev | Fixed in |
|---|---|---|---|---|
| F26 | G(c) decides **zero** claims: 120/120 primary rows and 100/100 NITJSR rows are resolved by F/P overrides or the F=P=0.5→D branch | gate logic at `verification_pipeline.py:988-1014`; replayed on both audit CSVs | P0 | §8 |
| F27 | All four weighting methods give identical 91.7/100/78.2 — even `(1,0,0)` and `(0,1,0)`; only `(0,0,1)` changes anything | `meta_learner_weights_comparison.csv`; replay | P0 | §8 |
| F28 | Meta-learner drops the intercept (−4.337) and normalises only positive coefficients, then presents them as weights of G | `meta_learner_grounding.py:56-64`; `app.py:292-307` | P1 | §8 |
| F29 | The 5-fold CV accuracy (93.3%) is for the logistic regression, a different model from the G-gate whose thresholds it supposedly justifies | `meta_learner_grounding.py:76-81` | P1 | §8 |
| F30 | Ablation uses fixed 0.4/0.3/0.3 regardless of deployed weights | `run_complete_ai_evaluation.py:440` | P1 | §8 |
| F31 | Chroma similarity normalised twice: `1-dist` clipped to [0,1], then `(s+1)/2` → D ∈ [0.5,1]; FAISS path maps to [0,1]. Same metric, different semantics | `verification_pipeline.py:830,913,959-965` | P1 | §7 |
| F32 | D thresholds 0.84/0.70 are magic numbers; D distributions overlap heavily (false 0.827, true 0.837, unverifiable 0.810) | `verification_pipeline.py:1001-1006`; `nitjsr_audit_trail.csv` | P0 | §7, §8 |
| F33 | Cosine similarity is not entailment; false claims retrieve at ~0.82 | `README.md:199` | P0 | §7 |
| F34 | Grid search tuned and evaluated on the same 60 primary claims; best 0.95 at (0.4,0.5,0.1) in notebook but README claims (0.5,0.3,0.2) | notebook cell 38; `README.md:134` | P1 | §5, §8 |
| F35 | RRF k=20 is advertised but the case study runs with default k=60; `app.py` still reports k=20 | `run_nitjsr_case_study.py:112-113`, `app.py:329-332` | P1 | §7 |
| F36 | No temporal decay / recency weighting exists; only an `is_latest` column is declared and never used | grep across repo; `nitjsr_ingestion_engine.py:146` | P1 | §7 |
| F37 | Prompt date is a stale literal ("Monday, 23 November 2026"), not wall-clock | `run_nitjsr_case_study.py:425`, `README.md:180`, `app.py:358` | P1 | §7 |

### 1.3 Agents and pipeline logic

| ID | Flaw | Evidence | Sev | Fixed in |
|---|---|---|---|---|
| F38 | Policy agent is a substring lookup containing benchmark claims verbatim ("tata motors training centre", "vibranium", "sleeping in the library", "lifelong imprisonment", "notice nitjsr/acd/2024/183", "lecture hall complex") | `verification_pipeline.py:649-691` | P0 | §4 |
| F39 | Docstrings say "zero hardcoded rules" while F38 is hardcoded | `verification_pipeline.py:546,635`, `README.md:19-20` | P1 | §4, §10 |
| F40 | Real case study passes `records=[]` → F=0.5 for all 100 claims; the fact-check agent contributes nothing | `run_nitjsr_case_study.py:104-109`; CSV | P0 | §4 |
| F41 | P=1.0 returned early for substrings "14 days", "overdue fine", "academic probation", "10:00 pm" → wrong values PASS via P≥0.9 | `verification_pipeline.py:677-691` | P0 | §4 |
| F42 | Library/revaluation checks use `or` where both constraints must hold ("4 books for 21 days" passes) | `verification_pipeline.py:716-734` | P0 | §4 |
| F43 | `"10" in nums` compares a string to floats — always False | `verification_pipeline.py:739` | P1 | §4 |
| F44 | "Wrong value" lists `[14,21,30,1000,1500]`, `[6,8,10,21]`, `[3.0,3.5,4.0,7.0,8.0]` are the test perturbations themselves | `verification_pipeline.py:703,723,733,741` | P0 | §4 |
| F45 | FactCheck returns on the first keyword hit; multi-fact claims verify only one fact | `verification_pipeline.py:594-630` | P0 | §4 |
| F46 | Any month word without the exact canonical string is a contradiction; "August 15, 2026" is BLOCKed. Years are parsed as numbers | `verification_pipeline.py:604-610` | P0 | §4 |
| F47 | `process_query` draft prompt contains no retrieved context → generator is a plain LLM; RAG exists only inside the grounding agent, while the triad script does inject context | `verification_pipeline.py:1137-1144` vs `run_nitjsr_case_study.py:423-430` | P0 | §4, §7 |
| F48 | Claim extractor is regex sentence split + `split(" and ")`; the LLM is passed in but unused; no decontextualisation ("it closes on 15 Aug"); claims are not atomic | `verification_pipeline.py:548-568` | P0 | §4 |
| F49 | Regeneration is single-shot and never re-verified | `verification_pipeline.py:1111-1115`; notebook cell 66 | P0 | §4 |
| F50 | If every claim is BLOCKed and regeneration is empty, `final_answer = draft` — the ungrounded draft is returned | `verification_pipeline.py:1160` | P0 | §4 |
| F51 | Errors swallowed: LLM failure returns `""`; embedding failure returns `[0.0]*768` → silent D=0.5 | `verification_pipeline.py:505-507,533-540` | P0 | §9 |
| F52 | `_save_cache` rewrites the entire ~32 MB embedding JSON on every generate call | `verification_pipeline.py:471-478,503,541` | P2 | §9 |
| F53 | Notebook's regenerated preamble ("Here are a few options for the grounded rewrite…") re-PASSes with G=0.66–0.83; re-verification cannot tell a rewrite from filler | notebook cell 66 | P0 | §4 |
| F54 | Notebook cell 64: the LLM's refusal ("I do not have real-time access…") is scored F=1, P=1 and PASSes | notebook cell 64 | P0 | §4 |

### 1.4 Benchmarks and dataset design

| ID | Flaw | Evidence | Sev | Fixed in |
|---|---|---|---|---|
| F55 | False/unverifiable claims are cartoonish (vibranium, hieroglyphics, imprisonment, gold leaf, helicopter rides); any LLM catches them | `run_complete_ai_evaluation.py:148-184`; `run_nitjsr_case_study.py:189-240` | P0 | §5 |
| F56 | No claim links to a source chunk; "true" labels are unsupported; proposal promised timestamped records | both claim lists; proposal §6 | P0 | §5 |
| F57 | Index covers ~1,585 of 18,853 chunks (~8.4%) via `[:400]`/`[:300]`/`[:500]` slices | `run_nitjsr_case_study.py:91-98`, `build_chroma_db.py:55-62` | P1 | §7 |
| F58 | "Secondary public dataset" is 5 self-written docs and 30 self-written claims, scoring 100% | `verification_pipeline.py:354-443`; `run_complete_ai_evaluation.py:194-228` | P0 | §5 |
| F59 | MFAQ is a retrieval benchmark, not an institutional-claims corpus; notebook uses it as "cross-institution" | `README.md:93-112`, notebook §14 | P1 | §5 |
| F60 | Thresholds, weights and metrics are all derived from the same data; no held-out split | all evaluation scripts | P0 | §5 |
| F61 | SciFact load failed silently in notebook; fallback = 3 unique claims repeated 20×, all conditions 100% | notebook cell 9 | P0 | §5 |
| F62 | Playwright scrape crashed (asyncio loop error); "Real NIT JSR data (scraped)" section has no data | notebook cell 62 | P1 | §10 |
| F63 | LangGraph agent evaluated on 6 claims only | notebook cell 36 | P2 | §6 |

### 1.5 Notebook and artifact hygiene

| ID | Flaw | Evidence | Sev | Fixed in |
|---|---|---|---|---|
| F64 | Cell 70 source uses 60 claims but output says 120 and adds a "Plain LLM (No verif)" line the code never computes | notebook cell 70 | P0 | §10 |
| F65 | Cell 72 source computes misclassifications; output is a typed 120-claim summary | notebook cell 72 | P0 | §10 |
| F66 | Cell 76 source only writes a protocol ("no live annotation"); output reports κ=0.812 and 91.2% | notebook cell 76 | P0 | §10 |
| F67 | Cell 82 source reads CSV; output claims 94.0% NITJ accuracy while CSV says 91.7% | notebook cell 82 | P0 | §10 |
| F68 | Cell 85 "Method Comparison" lines are not printed by the source | notebook cell 85 | P0 | §10 |
| F69 | Cell 48/49 source samples at most 60 MFAQ pairs; output claims 250; cell 50 output reports Recall@3/@10 that the source never computes | notebook cells 48-50 | P0 | §10 |
| F70 | Cell 53 uses `sampled[:10]` but output says "30 evaluated questions"; cell 55 prints 7 metrics where source builds 5 | notebook cells 53, 55 | P0 | §10 |
| F71 | Cell 41 prints mean absolute accuracy gap **0.354** while the surrounding narrative claims cross-institution transfer | notebook cell 41 | P1 | §10 |
| F72 | Cell 68 writes a 60-row `primary_audit_trail.csv`; the committed 120-row file comes from the rule-based script — two systems share one filename | notebook cell 68 vs `run_complete_ai_evaluation.py` | P0 | §10 |
| F73 | Execution counts repeat (50,50,50 / 38,39) and Section 32 cells 91-104 have no outputs | notebook metadata | P1 | §10 |
| F74 | Notebook's "proposed" uses gate 0.35/0.65 and D=LLM entailment on secondary with F=P=0.5 hardcoded; repo uses 0.50/0.72 and regex rules — two systems reported as one | notebook cells 19, 21, 24 vs `verification_pipeline.py` | P0 | §4, §10 |
| F75 | README latency numbers 0.85/1.15/2.9/1.45 are typed; notebook measured 0.82/0.88/1.74 | README vs notebook cell 58 | P1 | §10 |

### 1.6 API, docs, repo

| ID | Flaw | Evidence | Sev | Fixed in |
|---|---|---|---|---|
| F76 | `app.py` mutates shared `pipeline.scorer.weights` per request → threadpool race | `app.py:176-182,227-233` | P1 | §9 |
| F77 | CORS `allow_origins=["*"]` with credentials | `app.py:35-41` | P2 | §9 |
| F78 | Audit log is a JSON file; proposal §9 specifies MongoDB | `app.py:67-82` | P2 | §9 |
| F79 | `/api/meta-learner` and `/api/nitjsr-case-study` return hardcoded coefficients/metrics | `app.py:284-318,342-366` | P1 | §9 |
| F80 | ~90 MB of DB/Chroma/cache committed, plus a 32 MB untracked zip | tracked file sizes | P2 | §9 |
| F81 | Docs contradict each other: plan.md 2,273/1,094 vs README 2,241/1,056; study_plan κ=0.887 vs 0.812; heuristic weights 0.40/0.35/0.25 vs 0.40/0.30/0.30 | `plan.md:24-28`, `study_plan.md:120,262`, `README.md:10` | P1 | §10 |
| F82 | `study_plan.md` is a viva script that asserts non-existent temporal decay, an invented AUC, and a "bug fixed" story while the zero-vector fallback is still in code | `study_plan.md:159-161,237-241`, `verification_pipeline.py:540` | P1 | §10 |
| F83 | Proposal §8 explicitly labels its figures as placeholder/illustrative, but README presents those figures as empirical findings | proposal §8; `README.md:79-85` | P0 | §10 |

---

## 2. Target architecture

```
Student query
  -> retrieve top-k chunks (real 18,853-chunk index, recency-weighted RRF, k tuned on dev)
  -> draft LLM answer WITH retrieved context (same context available to all conditions)
  -> LLM claim extractor (atomic + decontextualised, prompted, logged)
  -> per claim, three independent signals:
       F: structured-record check (records loaded from data file with source_chunk_id)
       P: policy-rule engine (rules loaded from data file, structured predicates, no claim strings)
       D: NLI / LLM-as-judge entailment over retrieved evidence (not cosine)
  -> G = calibrated score; G-driven gate (overrides only if validated on dev)
  -> BLOCK: rewrite -> re-extract -> re-verify (max 2 rounds)
  -> if still ungrounded: explicit ABSTAIN with reasons (never return the raw draft)
  -> audit record: claim, source pointers, all signals, all prompts/responses, latency
```

Data files (new): `data/nitj_records.json`, `data/nitj_policies.json`, `data/public_records.json`, `data/public_policies.json`, `data/claims_primary.json`, `data/claims_secondary.json`, `data/annotations.json`, `data/splits.json`.

---

## 3. Phase 0 — Triage and safety (half a day)

- [ ] Add `check_artifacts.py`: scans README, dashboard, LaTeX, study_plan for metric literals and fails if a number is not present in a committed CSV/JSON produced by a script.
- [ ] Add to `.gitignore`: `nitjsr_documents.db`, `nitjsr_chroma_db/`, `nb_cache/`, `*.zip`, `documents/`. Keep only small fixtures; write a `data/README.md` explaining how to rebuild.
- [ ] Temporarily replace hardcoded API metrics with `501 Not Implemented — regenerate artifacts` so nothing false is served while we work.
- [ ] Freeze `data/splits.json` with a group split by `source_chunk_id` before generating any new claims.
- [ ] Rename `README.md` claims to "prototype under re-evaluation" until Phase 7 completes.

**Exit criterion:** no script can emit a metric literal, and no stale number is served by the API.

---

## 4. Phase 1 — Rewrite the verification layer (2–3 days)

**Files:** `verification_pipeline.py` (lines 143–443, 548–744, 968–1183).

1. **Records and rules become data.** Move `SYNTHETIC_NITJ_RECORDS`, `SYNTHETIC_NITJ_POLICIES`, and their public counterparts into JSON with fields: `id`, `description`, `canonical_value`, `val_type`, `category`, `source_chunk_id`, `source_document`, `issue_date`, `valid_from`, `valid_to`. Policy rules: `rule_id`, `predicate` (structured: field/operator/value), `source_ref`. **No claim text, no "wrong value" lists, no substring lists** (fixes F38, F39, F44).
2. **FactCheckAgent:** parse all dates and numbers in a claim (ISO, `DD Month YYYY`, `Month DD, YYYY`, `Rs 1,00,000`), compare **every** detected entity against the relevant canonical record, return a list of per-entity verdicts plus aggregated F. No early return on first keyword (F45, F46). Multi-fact claims are only PASS if **all** detected facts verify.
3. **PolicyComplianceAgent:** evaluate all applicable predicates; combine conjunctive constraints with AND; return P=1 only when every applicable rule is satisfied. Remove all early-return substring shortcuts and all "known wrong value" arrays (F41–F44). Handle `10:00 AM` correctly.
4. **GroundingAgent:** D comes from an NLI/LLM judge (`SUPPORTED`/`CONTRADICTED`/`INSUFFICIENT`), with a confidence calibrated on dev. Retrieval supplies evidence only. Remove double normalisation; keep one documented D definition for all backends (F31–F33).
5. **DecisionGate:** G must drive the decision. Either (a) pure `G >= τ_pass / <= τ_flag`, or (b) overrides retained **only** if dev data shows they improve accuracy, with an explicit G-only ablation. Accepted only if weights matter in the comparison table (F26–F30).
6. **ClaimExtractor:** prompt the LLM to produce atomic, self-contained claims; include a decontextualisation pass; log raw output; keep regex version as an ablated baseline (F48).
7. **Generator:** inject retrieved context into the draft prompt; the same context is available to RAG and proposed conditions (F47, F74).
8. **Regeneration loop:** rewrite → re-extract → re-verify; up to 2 rounds; require the rewrite to be a single factual sentence (parsed, not a preamble); if still ungrounded, return `ABSTAIN` with reasons, never `draft` (F49, F50, F53, F54).
9. **Error handling:** raise on LLM/embedding failure with an explicit `ERROR` status; no zero vectors; no silent `""` (F51).
10. **Caching:** per-key files or SQLite instead of rewriting a 32 MB JSON; cache key includes model digest, seed, prompt, all parameters (F52).

**Acceptance tests (write them first):**
- "Revaluation within 14 days with fee Rs 500" → F=0 (wrong window), not PASS.
- "4 books for 21 days" → P=0.
- "August 15, 2026" → F=1.
- "Fee is Rs 10,000 per day" vs canonical 5 → F=0.
- Multi-fact claim with one wrong date → at least one F=0, overall not PASS.
- LLM returns preamble → rejected by rewrite validator.
- All-BLOCK query → `ABSTAIN`, never the draft.

---

## 5. Phase 2 — Dataset, annotation, split (3–5 days)

1. **Primary claims from real chunks.** For each of the 5 categories, sample chunks from SQLite (all 18,853, stratified, not `[:400]`). Create:
   - **True:** verbatim or strictly entailed sentence, with `source_chunk_id`.
   - **False/plausible-wrong:** perturb exactly one fact in a true claim — off-by-a-week date, superseded circular (old fee/ref number), wrong amount, wrong eligibility threshold, swapped morning/afternoon shift. No vibranium/hieroglyphics (F55, F56).
   - **Unverifiable:** plausible but genuinely absent from the corpus, verified by retrieval + manual check (e.g. "Library fines are transferred to a student welfare fund").
   - Target: **~240 claims** (80/label), balanced across categories and source documents.
2. **Secondary dataset — real, not synthetic.** Use an openly available university FAQ/QA corpus (non-NITJ), filtered to academic/administrative questions, with documented source, licence, and date. If a genuinely suitable university corpus cannot be obtained, **drop the "cross-institution validation" claim** and rename this "secondary out-of-domain FAQ stress test" (F58, F59, F61).
3. **Human annotation.** Two teammates label independently against timestamped source records. Store raw labels + disagreements + adjudicated final label. Report raw agreement and Cohen's κ **with a confidence interval**. Delete all simulated annotator code (F04–F06).
4. **Group split.** Split by `source_chunk_id`/source document into dev (40%) and test (60%), stratified by label and category. Freeze test. All tuning in Phases 1, 4, 7 uses dev only (F60, F34).
5. **Record provenance** for every claim: document, date, ref number, chunk id, perturbation description (F56).

**Exit criterion:** `data/claims_*.json` with source pointers, `data/annotations.json` with two real annotators, `data/splits.json` frozen.

---

## 6. Phase 3 — Real evaluation harness (2–3 days)

New file `run_honest_evaluation.py`, replacing the three simulation-heavy runners.

| Condition | Real implementation |
|---|---|
| A: Plain LLM | `gemma4:e4b`, prompt only, no context. Record abstention rate separately from "missed falsehoods" (F24). |
| B: RAG-only | Real retrieval, k tuned on dev, context injected into the same LLM, one call. |
| C: Self-consistency | Same prompt as A (or B, documented), T=3 at T=0.7, majority vote. If it collapses to one class, report that explicitly (F25). |
| D: Proposed | Full pipeline from §2, with the same retrieval corpus and LLM budget. |

- Fix the seed with Ollama's `seed` option and log the model digest; set `num_predict`, temperature, and context identically across conditions.
- Measure **end-to-end latency** including embedding and LLM calls; report mean ± std (F18, F22, F75).
- Compute accuracy, macro-P/R/F1 **with correct multi-class definitions**, per-label metrics, confusion matrix, missed-falsehood rate, **abstention rate**, ROC/PR only where scores are real and meaningful, and paired bootstrap CIs (F12, F14, F15, F19).
- Store raw prompts, responses, predictions, signals, and latency in `honest_audit_trail.csv`.
- Statistical test: **McNemar** on paired D-vs-B predictions; report p-value and effect size. Power check: with ~150 test claims, a 10-point difference is detectable; if test is smaller, say so (F23).
- Delete simulation code from `run_nitjsr_case_study.py:267-305`, `run_complete_ai_evaluation.py:275-295,574-579`, and `run_mfaq_full_agentic_pipeline.py:174-204` (F01–F03, F17, F20, F21).
- Keep the genuinely measured MFAQ retrieval/QA runner (`run_full_mfaq_benchmark.py`) — those numbers are real and internally consistent.

**Exit criterion:** one command regenerates every metric from real Ollama calls; no `if true_label`, no `np.random.rand()`, no `hash()`, no literal latencies.

---

## 7. Phase 4 — Retrieval and temporal grounding (1–2 days)

- Index **all** 18,853 chunks (or a documented, seeded stratified sample with reported coverage); remove `[:400]` sampling (F57).
- Tune RRF `k` and `w_dense:w_bm25` on **dev only**; report the selected values and the dev curve. Use the selected k everywhere, including the API (F35).
- Implement real recency: parse `issue_date`, compute `age_years`, apply `exp(-λ·age)` in RRF with λ tuned on dev; actually compute and store `is_latest` at ingestion (F36).
- Inject `datetime.now()` into prompts at runtime, with the academic session derived from the date (F37).
- Re-implement the RAG Triad as real measurements: faithfulness = fraction of extracted claims entailed by retrieved evidence by the NLI judge; context relevance = judged relevance of retrieved chunks to the query; answer relevance = judged directness. No keyword/length shortcuts (F07, F08).

---

## 8. Phase 5 — Weights and gate that actually matter (1–2 days)

- Train the meta-learner on dev only, using out-of-fold predictions; **report the intercept and use the full linear predictor (or its calibrated probability) as G**. Do not present normalised positive coefficients as the weights of a gate that ignores them (F28, F29).
- Compare weighting methods **including a G-only gate and the override cascade** on the frozen test set. Acceptance: the table must show non-identical numbers where weights differ; if all methods tie, state that weights are not identifiable for this feature distribution and stop claiming calibration (F26, F27).
- Re-run the ablation with the deployed weights and with the G-only gate (F30).
- Calibrate τ_pass/τ_flag on dev via ROC/PR; report Brier score and calibration curve, not just accuracy (F32, F34).

---

## 9. Phase 6 — API, repo, engineering hygiene (1–2 days)

- Pass weights explicitly per request; never mutate shared pipeline state. Use locks only if sharing is unavoidable (F76).
- Restrict CORS to configured origins; set `allow_credentials` only when origins are explicit (F77).
- Switch audit to MongoDB per proposal §9, or to SQLite with a clear documented deviation (F78).
- Make `/api/meta-learner`, `/api/metrics`, `/api/nitjsr-case-study`, `/api/human-eval`, and the dashboard read from regenerated artifacts or live computation — zero literals (F79, F16, F10, F11).
- Remove large binaries from git; add rebuild instructions (F80). Delete the 32 MB zip.
- Fix docstrings and README wording to match the code ("rule-based policy engine", not "zero hardcoded rules") (F39).

---

## 10. Phase 7 — Notebook, docs and final run (2 days)

- Clear **all** outputs from `capstone_verification.ipynb`, restructure it as a thin narrative wrapper that calls the committed scripts, and re-execute top-to-bottom in one kernel. Commit only outputs that the source actually produces. Delete every pasted summary cell (F64–F73, F75).
- Either implement the live scrape properly with async Playwright and log the result, or delete the "scraped data" section entirely (F62).
- Regenerate `README.md`, `dashboard.html`, `plan.md`, `study_plan.md`, all `.tex` tables, and `evaluation_results/*.png` **from the CSVs**. Remove `study_plan.md` model answers that assert non-existent features or invented numbers; replace with honest answers to the same questions (F81–F83, F09, F16).
- Add `artifacts_manifest.json` mapping each published number to its generating script, commit hash, and CSV row.
- Run `check_artifacts.py` in CI / locally; it must pass.
- **Final experiment:** freeze everything, run the honest evaluation once on test, report paired results with CIs. Then write the abstract/conclusion to match.

---

## 11. How we actually beat the baselines (without cheating)

The honest notebook run is the starting point: **RAG 96.7 vs proposed 95.0** on the current 60-claim set. That set is too easy and too synthetic for a claim-level verifier to show value. A fair win requires a test set that targets the failure mode the method was designed for:

1. **Construct plausible-wrong claims, not cartoon ones.** Off-by-one-week dates, superseded circulars (older fee/ref number still plausible), swapped thresholds, wrong payment portal, wrong loan period. RAG retrieves a topically perfect chunk and still says "true"; record/rule checks catch it. This is the method's designed advantage (F55).
2. **Use multi-fact claims.** "Registration is 15 August and the late fine is Rs 1,000." A response-level verifier or RAG judge often passes it because half is right; atomic verification should catch the wrong fine. Report this as a pre-registered subgroup analysis.
3. **Use temporal conflicts.** Two circulars for the same topic with different dates/amounts; the correct answer depends on `is_latest`. RAG without recency gets this wrong; recency-weighted RRF + records with validity windows gets it right.
4. **Keep the comparison fair.** Identical model, prompt budget, and corpus for B and D; D's extra structured records are the declared contribution, not hidden oracle labels. Never paste ground-truth labels into prompts (the notebook did this and it is not defensible).
5. **Power and statistics.** ~150 test claims per dataset, paired McNemar test, bootstrap CI on the accuracy delta, per-category breakdown with counts. Pre-register the primary metric on dev before opening test.
6. **Report the whole truth:** accuracy, macro-F1, missed-falsehood, **abstention**, latency. If D's win comes with 3× latency or high abstention, say so.
7. **If we still lose:** that is a valid, publishable negative result ("claim-level verification did not outperform RAG on institutional FAQ claims"). We deliver the working system and the honest comparison. We do **not** relabel, re-split, or re-tune after seeing test.

---

## 12. Definition of done

- [ ] No `if true_label`, `np.random.rand()`, `hash(claim)`, or typed metric literals anywhere in evaluation code.
- [ ] Every published number traceable to a CSV generated by a committed script (`check_artifacts.py` passes).
- [ ] Two real annotators, raw labels committed, κ with CI reported; no simulated annotators.
- [ ] Group dev/test split frozen before tuning; all tuning logged from dev only.
- [ ] G demonstrably changes decisions; weight comparison table is not identical across methods, or the claim is withdrawn.
- [ ] D is NLI/judge-based, calibrated, single definition across backends; no double normalisation.
- [ ] Policy rules and records are data files with source pointers; no benchmark strings in source.
- [ ] Multi-fact, date-format, and wrong-value acceptance tests pass.
- [ ] Regeneration re-verifies and can abstain; no preamble rewrites accepted; never falls back to raw draft.
- [ ] Real end-to-end latency including embedding and LLM calls.
- [ ] RAG Triad computed by judge, not keyword/length heuristics.
- [ ] Notebook re-executed with matching outputs, or outputs stripped; no pasted summaries.
- [ ] Secondary dataset is real or the generalizability claim is removed.
- [ ] API reads live/regenerated data; no CORS `*`; no shared mutable weights; audit in MongoDB or documented SQLite deviation.
- [ ] Large binaries removed from git; rebuild instructions present.
- [ ] README/dashboard/LaTeX/study_plan agree with CSVs; proposal placeholder figures are not presented as results.
- [ ] Paired statistical test and CI reported for proposed vs RAG on frozen test.
- [ ] Final write-up matches the measured outcome, whichever way it goes.

---

## 13. Suggested order and effort

| Phase | Effort | Depends on |
|---|---|---|
| 0 Triage | 0.5 day | — |
| 1 Agents | 2–3 days | 0 |
| 2 Dataset + annotation | 3–5 days | 0 (can run in parallel with 1) |
| 3 Evaluation harness | 2–3 days | 1, 2 |
| 4 Retrieval + temporal | 1–2 days | 2 |
| 5 Weights + gate | 1–2 days | 3, 4 |
| 6 API + repo | 1–2 days | 3 |
| 7 Notebook + docs + final run | 2 days | 5, 6 |

**Total ≈ 12–18 working days.** If time is shorter, do Phases 0, 1, 2, 3, 7 and explicitly downgrade the claims in §5/§8/§11; an honest prototype beats a fabricated validation every time.

---

## 14. Files to delete, rewrite, keep, add

**Delete / quarantine:** simulated branches in `run_nitjsr_case_study.py`, `run_complete_ai_evaluation.py`, `run_mfaq_full_agentic_pipeline.py`; typed chart arrays; pasted notebook outputs; `Capstone-Project-main.zip`; large committed binaries.

**Rewrite:** `verification_pipeline.py` (agents, gate, generator, regeneration, caching, errors), `run_nitjsr_case_study.py` (real conditions + real triad), `run_complete_ai_evaluation.py` (real baselines, real annotation, derived master table), `app.py`, `README.md`, `dashboard.html`, `plan.md`, `study_plan.md`, all `.tex` exports.

**Keep (they are genuinely measured):** `run_full_mfaq_benchmark.py` and its CSVs; `nitjsr_ingestion_engine.py` (with `is_latest` added); `scan_documents_metadata.py`; `build_chroma_db.py` (with full-corpus indexing and recency).

**Add:** `run_honest_evaluation.py`, `check_artifacts.py`, `data/*.json`, `honest_audit_trail.csv`, `artifacts_manifest.json`, `tests/test_agents.py` (acceptance tests from §4), `data/README.md`.

---

## 15. Deep-dive pass 2 — new findings (F84–F108)

**Method:** read every Python artifact, **replayed the committed gate row-by-row over the committed audit trails**, and queried `nitjsr_documents.db` / `nitjsr_chroma_db/chroma.sqlite3`. Every number below was recomputed from files in the repo, not copied from a document. §1 stands unchanged; this section adds what the first pass missed and corrects three conclusions (§15.7).

### 15.1 The headline 91.7% is not reproducible from the committed code (P0)

| Metric | Stored in `primary_audit_trail.csv` | Replaying `DecisionGate.decide` on its own F/P/D |
|---|---|---|
| PASS / FLAG / BLOCK | 32 / 44 / 44 | 46 / 30 / 44 |
| Accuracy | 91.67% | **87.50%** |

14 of 120 rows disagree, all of them `F=P=0.5, D>=0.84`: the file records FLAG, the committed gate returns PASS (e.g. `SYN-EX-T06` D=0.958, `SYN-EX-U04` D=0.921, `SYN-FD-T07` D=0.889). The artifact was produced by an **earlier gate** (one that flagged every neutral-F/P claim), and every document — README 91.7%, `plan.md`, `master_evaluation_metrics.csv`, dashboard — quotes that earlier run. Nothing in the repo can regenerate 91.7%.

**Gate identifiability (measured, not argued):** a pure `G>=0.72 / >=0.50` gate scores **91.67%** on the same rows (10 errors, different rows); the cascade with the whole D-branch removed also scores **91.67%**; the committed cascade scores 91.67% on the stored decisions. Four different decision rules, three different weight vectors, one number. The problem is not only that G is bypassed (F26) — on this benchmark the gate, the weights and the D-branch are **jointly non-identifiable**.

### 15.2 What the "real NITJSR" 91% actually is
The audit trail decomposes exactly as follows (100 rows, `records=[]` so F is 0.5 everywhere):

* **74/100** decided by the policy keyword lists: 38 rows PASS via `P>=0.9`, 36 rows BLOCK via `P<=0.1`.
* **26/100** reach the D branch: 19 FLAG, 7 PASS, **5 of the 10 total errors**.
* **0/40** false claims were ever caught by a structured record.
* **3/20** "unverifiable" claims are returned as `true`.

### 15.3 Many "true" claims contain nothing checkable

| | has a digit | no digit |
|---|---|---|
| true (40) | 14 | **26** |
| false (40) | 19 | 21 |
| unverifiable (20) | 5 | 15 |

26 of 40 "true" claims ("Hostel curfew hours require boarders to be inside hostel premises by the specified night cutoff") contain no date, number, name or reference at all. **23 of those 26, and 37 of all 40, pass solely because the claim contains a string from `policy_compliances`.** They are not verified; they are recognised.

### 15.4 D does not measure grounding, and the corpus explains why
* Mean D: false **0.918**, true **0.923**, unverifiable 0.810 — a true/false gap of **0.004**.
* **40/40** false claims and **10/40** unverifiable claims score above the 0.84 "PASS" threshold.
* Therefore the published 0.84/0.70 cut-offs are not thresholds on grounding; they sit inside the noise band of a cosine score between a claim and a one-token perturbation of the sentence it was built from.
* The code already contains its own RAG baseline (`rag_preds` from D at 0.80/0.60) and **discards it**; recomputed it scores **40.0%**, against the 65.0% typed into the master table.

### 15.5 The retrieval layer is worse than any document admits
Top retrieved chunk category equals the claim's category **49% of the time** (Fee Deadlines 14/20, Hostel 20/20, Exam Dates **3/20**, Syllabus 11/20). Cause: `build_chroma_db.py` indexes `table_chunks[:500] + hostel + fee[:400] + exam[:400] + dean/academic[:400] + syll[:300]` over `sorted(os.listdir(...))`, i.e. an **alphabetical prefix** of the archive — which is why the same `dean_office_notices_*` files return for every query and why the 0.82 cosine for false claims is really "everything is close to everything".

### 15.6 Corpus metadata is largely invented
| Field | Reality |
|---|---|
| `issue_date` | **7,657 / 18,853 chunks (41%)** carry the hardcoded fallback `2024-08-01` from `extract_document_metadata` |
| `ref_number` | falls back to synthesized `NITJSR/<Category>/2024` |
| `is_latest` | `1` written for **every** row at insert; never computed, so any decay/latest logic is meaningless before it is rebuilt |
| date formats | mixed `01-01-2024`, `01.01.2024`, `2024-08-01`; never normalised, so no parser could apply recency as stored |
| corpus counts | summary says 2,241 PDFs / 1,056 digital / 35.04 MB; plan.md says 2,273 / 1,094; the tree holds **2,174** PDFs and the DB **1,048** distinct document names |

### 15.7 New flaw rows

| ID | Flaw | Evidence | Sev | Fixed in |
|---|---|---|---|---|
| F84 | Stored audit trail does not match the committed gate; 14/120 rows disagree and the headline 91.7% replays as 87.5% | §15.1; replay of `DecisionGate` over `primary_audit_trail.csv` | P0 | §3, §10 |
| F85 | 26/40 "true" NITJSR claims contain no checkable fact; 23 pass purely on a `policy_compliances` substring | §15.3 | P0 | §4, §5 |
| F86 | D has no discriminative power: true 0.923 vs false 0.918; 40/40 false above the PASS threshold | §15.4 | P0 | §7 |
| F87 | The code's own D-based RAG baseline scores 40.0%; the master table types 65.0% instead of using it | `run_complete_ai_evaluation.py:283-293` (unused) | P0 | §6 |
| F88 | 41% of chunk dates are the hardcoded fallback; ref numbers synthesized; `is_latest` constant; date formats unnormalised | §15.6 | P0 | §7 |
| F89 | Ingestion counts contradict each other and the tree (2,241 / 2,273 / 2,174; 1,056 / 1,048) | §15.6 | P1 | §10 |
| F90 | Retrieval top-1 category matches the claim category only 49% of the time; Exam Dates 3/20 | §15.5 | P1 | §7 |
| F91 | Deployed weights and all thresholds are literals (`LEARNED_META_WEIGHTS`, 0.72/0.50/0.84/0.70); re-running the meta-learner cannot change the pipeline | `verification_pipeline.py:36-53,995-1010` | P1 | §8 |
| F92 | LLM/embedding cache key excludes the model (and seed/num_ctx); switching `gemma4` or `nomic-embed-text` silently replays the previous model's outputs | `verification_pipeline.py:462-464` | P1 | §9 |
| F93 | Fact-check **false positives** on correct claims: 4/40 true claims blocked (`SYN-FD-T04`, `CR-T04`, `CR-T05`, `SY-T06`); "Rs 5 lakh" parses as 5 vs 500000; a fine claim matches the *deadline* record and any month triggers contradiction | `primary_audit_trail.csv` log columns | P0 | §4 |
| F94 | "Unverifiable" is scored as an error in every binary metric and mapped to "Hallucinated" in the confusion matrix, although BLOCK/FLAG design emits it for 39/40 unverifiable claims | `run_complete_ai_evaluation.py:346-350,395` | P1 | §6 |
| F95 | Regeneration is never re-scored, and in the synthetic path its output is the record text verbatim (43/120 rows); delivered answer ≠ scored claim, and the "closed loop" is a lookup of the answer key | §15.2; `secondary/primary_audit_trail.csv` `regenerated_claim` | P0 | §4 |
| F96 | Six runners execute at import (no `__main__` guard), so no importable harness and `pytest`/import fires 100 LLM calls; `nitjsr_ingestion_engine.py` hardcodes absolute Windows paths | `nitjsr_ingestion_engine.py:19-20`; grep | P2 | §9 |
| F97 | README install line omits `chromadb`, `datasets`, `ollama`, `PyMuPDF`; no `requirements.txt`/lockfile | `README.md:283` | P2 | §9 |
| F98 | README says 400-word windows with 50-word overlap; code uses `max_words=220, overlap=35` | `nitjsr_ingestion_engine.py:96` vs `README.md` | P2 | §10 |
| F99 | `is_synthetic_benchmark=True` and `dataset_mode="[Synthetic Benchmark]"` are hardcoded, so the API labels the real NITJSR corpus synthetic | `verification_pipeline.py:1108-1116,1160` | P2 | §9 |
| F100 | `classify_claim_category` is order-dependent ("hostel mess fee" → Fee Deadlines) and never used for scoring, so the reported per-category table tests the authors' labels, not the classifier | `verification_pipeline.py:1068-1079` | P3 | §6 |
| F101 | The "balanced 120" repeats claims (T01/T07, CR-T02/T08, FD-T01/T05, SY-T01/T03); effective independent N is well below 120 | `run_complete_ai_evaluation.py:27-250` | P2 | §5 |
| F102 | "Unverifiable" conflates *not in the 12-doc synthetic KB* with *unknowable*: "There are 12 student hostels" is a checkable fact labelled unverifiable | `run_complete_ai_evaluation.py:92-99` | P1 | §5 |
| F103 | "Cross-dataset mean accuracy" averages three different systems on three label distributions (MFAQ row comes from an entailment-only pipeline with P=0.5) | `meta_learner_grounding.py:143-176` | P1 | §6 |
| F104 | Proposal §5.1 requires weights calibrated **against the human-evaluation dataset**; the implementation fits on the same 120 claims it evaluates, and the human study produces no labels at all | proposal §5.1 vs `meta_learner_grounding.py:60-80` | P0 | §5, §8 |
| F105 | The audit trail carries no prompt, response, model name/digest, seed, chunk id or timestamp — only a doc title string — so claims of auditability cannot be checked | `nitjsr_audit_trail.csv` columns | P1 | §9 |
| F106 | `/api/audit-trail` serves the synthetic 120-claim benchmark CSV while presenting it as the request audit log | `app.py:377-384` | P3 | §9 |
| F107 | `dashboard.html` types 91.0%, Macro-F1 0.918, 0.936/0.906/0.920, "AUC = 0.942" and "1,585 vectors" | `dashboard.html:62,240,250,280,290,359-369` | P1 | §10 |
| F108 | `scan_documents_metadata.py` counts are quoted as corpus facts but the corpus now on disk differs from every quoted total | §15.6 | P2 | §10 |

### 15.8 Corrections to this plan's earlier sections

1. **§14 "Keep" is too generous.** Keep the *code paths* of `nitjsr_ingestion_engine.py` and `build_chroma_db.py`, not their outputs. The DB must be rebuilt: 41% of dates are fallbacks, every `ref_number` may be synthesized, `is_latest` is constant, and the Chroma index is an alphabetical prefix (F88, F90). §7 needs a **metadata-repair task** before any decay/latest work; the current §7 only says "implement recency".
2. **§11's starting point (RAG 96.7 vs proposed 95.0) is not the current system.** The notebook's D is an LLM judge fed the 12 ground-truth records; the repo's D is cosine over a Chroma prefix. Neither is the other (F74). Do not carry either number into the rewrite as a baseline.
3. **F26 wording.** "G decides zero claims" is correct for the code path, but a G-only gate ties the cascade at 91.7%. State the flaw as *the decision rule is not identifiable on this data*, and let the §8 acceptance test (weights must change outcomes, or the claim is withdrawn) decide the rest.
4. **F04/F05 understate the human-eval problem.** Beyond the RNG: the study produces no label file, so it also fails proposal §5.1's calibration input (F104). If any calibration is to be defended, the two real annotators must label the same claims the weights are fitted on.

### 15.9 If only one day is available

Replay the committed gate over the committed artifacts once, publish **87.5%** (not 91.7%), and say in one line that the 91% NITJSR figure is 74% substring lookup and 26% D-thresholding on a near-zero-separation score. That single act removes the two claims most likely to collapse under a supervisor's first question, without touching a line of agent code.
