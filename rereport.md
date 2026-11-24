# Re-audit: bugs, hardcodedness, and false or misleading claims

**Scope:** every active Python module, generated `data/` and `results/` artifact, `README.md`, `TODO.md`, `artifacts_manifest.json`, `check_artifacts.py`, and the shipped dashboard (`dashboard/src`, `dashboard/dist`).
**Method:** code read, artifact cross-checking, recomputation of the reported test metrics from `results/test_predictions.csv`, targeted probes of `NumericCheck`, and a replay of the gate with an alternative rule order.
**Status:** no application code was changed. This file is the only new file.

---

## Executive verdict

The measured core is real: `/api/verify-query` genuinely calls Ollama, Chroma, BM25, a cross-encoder NLI judge, and the numeric/policy checks. The main test CSV is internally consistent: recomputing `accuracy`, `macro-F1`, `missed_falsehood_rate`, and `abstention_rate` from `results/test_predictions.csv` reproduces `results/condition_metrics.csv` exactly (D = 0.8807 accuracy / 0.8801 macro-F1, n = 109).

That said, the result is **not yet safe to present as-is**, for four independent reasons:

1. **The shipped frontend still shows the old fabricated synthetic benchmark**: 91.7% accuracy, 2.5% missed falsehoods, Cohen's kappa 0.812, ROC-AUC 0.926, 120 claims, 80 "human-reviewed" claims, MFAQ 78.2%, and four weighting methods that all score 91.7/100/78.2. These are the exact numbers the improvement plan was supposed to remove.
2. **The gate lets a numeric match override a strong contradiction.** Three false test claims are predicted `true` with `F = 1.0`, `D = 0.0`, and `C = 0.88–0.98` because `F >= 0.95` returns PASS before `C` is checked.
3. **The "unverifiable" class is circular and mostly not claims.** 57 of 80 absent claims are hedges/refusals ("check your portal", "varies by institution"), 2 are template placeholders, only 3 mention NIT Jamshedpur, and the class was selected by the same NLI judge used to evaluate it. 30 of the 38 test-unverifiable claims are of this kind.
4. **The structured records used by the fact check are wrong or misapplied.** `library_fine = 1000` comes from a late-registration notice; `revaluation_fee = 5000` comes from a fee-payment table; the record `operator` is never used; and keyword matching is so loose that a claim about a tuition or hostel fee can be checked against `late_registration_fee`.

A second, independent set of documentation claims is false or stale: `README.md` says the multi-fact accuracy is 0.40 (artifact says 0.7333), says 153 claims / 0 unverifiable / 82 dev / 71 test (the experiment actually used 233 claims / 38 test-unverifiable / 124 dev / 109 test), says weights were tuned on dev (they are fixed), and says record values were used only where extraction agreed across documents (several have `support = 1`).

**Honest headline:** on the 71 real-chunk claims alone (36 true, 35 false), D scores 0.8451 accuracy / 0.8449 macro-F1; B scores 0.7606 / 0.7535. On the 79 non-hedge claims, D scores 0.8481 / 0.8706; B scores 0.6835 / 0.4790. D still beats B, but the headline 0.8807 is inflated by easy unsupported claims and by the gate short-circuit.

---

## 1. P0 — The dashboard is still the old fabricated system

### 1.1 Static benchmark files are the old synthetic artifacts

`dashboard/dist/data/` is tracked and served by the Vite preview. It still contains:

- `dashboard/dist/data/metrics.json`: `"Dataset": "NIT Jamshedpur [Synthetic]"`, `"Sample_Size": 120`, proposed `Accuracy: 0.917`, `Precision: 0.965`, `Recall: 0.917`, `F1: 0.958`, `Hallucination_Rate_%: 0`; a second proposed row on `"Public University [Synthetic]"` with `Accuracy: 1`, `F1: 1`, n = 30.
- `dashboard/dist/data/human-eval.json`: `annotator_1_vs_annotator_2_kappa: 0.812`, `pipeline_to_human_consensus_agreement_pct: 91.2`, `sample_size: 80`, `dataset_mode: "[Synthetic Benchmark]"`.
- `dashboard/dist/data/weights-comparison.json`: all four methods (equal, heuristic, grid search, "Logistic Regression Meta-Learner (Ours)") have `NITJ_Accuracy: 91.7`, `PublicUniv_Accuracy: 100`, `MFAQ_Accuracy: 78.2`. This is exactly flaw F27 from the improvement plan.
- `dashboard/dist/data/mfaq-comparison.json`: 55 MFAQ claims, proposed 78.2%, hallucination 83.3%. No MFAQ code exists anywhere in the active repository.
- `dashboard/dist/data/retrieval-comparison.json`: Recall@1/3/5/10 and MRR values; no retrieval-benchmark script exists in the active repository.
- `dashboard/dist/data/primary-audit.json`: rows such as `SYN-EX-T01` with `[Synthetic Record]` explanations.
- `dashboard/dist/data/secondary-audit.json` and `synthetic-docs.json`: synthetic public-university documents and claims.
- `dashboard/dist/data/figures/roc_curve_comparison.png` and `confusion_matrix_groundedness.png`: plots from the old fabricated run.

None of these files are produced by the current pipeline. `make_docs.py` writes only `README.md` and `dashboard.html`; it never copies current `results/` data into `dashboard/dist/data/`.

### 1.2 The Evaluation page hardcodes the old numbers

`dashboard/dist/assets/EvaluationPage-hfblF8h_.js` contains literal strings:

- `"Accuracy", value: "91.7%", note: "Proposed pipeline · 120 claims"`
- `"Missed falsehoods", value: "2.5%", note: "Proposed pipeline · 120 claims"`
- `"ROC–AUC", value: "0.926", note: "Five-fold validation"`
- `"Learned weights: fact 0.508 · policy 0.315 · evidence 0.177."`

`kappa` is read from the old `human-eval.json` and displayed as `${k.sample_size} human-reviewed claims`.

The **Home page** uses `i.primaryAudit` to render the "Benchmark sample" card; the shipped `primary-audit.json` is the synthetic file. The **Audit page** also filters and displays `primaryAudit`. Opening the dashboard therefore still shows synthetic exam dates and PASS labels as if they were a real benchmark.

### 1.3 The frontend data layer never asks the backend for results

`dashboard/src/lib/api.ts:8-20` loads these names from `/data/`:

```ts
['metrics', 'primary-audit', 'secondary-audit', 'mfaq-comparison',
 'retrieval-comparison', 'weights-comparison', 'mfaq-audit',
 'human-eval', 'synthetic-docs']
```

It never calls `/api/metrics`, `/api/stats`, `/api/corpus`, `/api/records`, or `/api/unsupervised`. Only `POST /api/verify-query` is live.

The built bundle currently sets `USE_MOCK` to `false` (`Ty="false".toLowerCase()!=="false"`), so the Verify page does hit the live endpoint. But the Home, Results, Audit, and Evaluation pages still read the old files. The source expression `import.meta.env.USE_MOCK.toLowerCase()` also throws if the environment variable is undefined, so a rebuild without `USE_MOCK` can fail at module load.

### 1.4 Two dashboards now contradict each other

- Backend `GET /` serves `dashboard.html`, generated from current `results/`, and shows D = 0.8807.
- Vite on `:5176` serves `dashboard/dist/`, which shows 91.7% and 2.5%.

The notice banner that used to say "Results use a synthetic benchmark … this is not a live institute feed" was removed in commit `982018f`. That removal is now dangerous: the dashboard still serves synthetic benchmark data, but the only warning is gone. The banner should not have been removed until `dashboard/dist/data` and the hardcoded Evaluation page were replaced.

### 1.5 The dashboard cannot be rebuilt from this repository

- `dashboard/src/` contains only `lib/api.ts`; the components, pages, types, and styles are absent. `api.ts` imports `../types`, which does not exist.
- `dashboard/package.json` has no `dependencies`, no `devDependencies`, no `build` script, and no lockfile. It only runs `vite preview` against prebuilt files.
- `vite` exists only because `dashboard/node_modules/` is committed: 10,226 tracked files, including `.vite` cache and `@esbuild/win32-x64/esbuild.exe`.
- `.gitignore` does not ignore `dashboard/node_modules` or `dashboard/dist`.
- `vite.config.ts` has no React plugin and no build input, so the missing source could not be compiled as-is.

This is both a hygiene problem and a reproducibility problem: the "source" of the frontend is a minified bundle.

---

## 2. P0 — The gate returns PASS before checking contradiction

`pipeline.py:378-390`:

```python
if F <= FACT_BLOCK: return G, "BLOCK", "false"
if P <= FACT_BLOCK: return G, "BLOCK", "false"
if F >= FACT_PASS: return G, "PASS", "true"
if C >= self.tau_contra: return G, "BLOCK", "false"
if D >= self.tau_pass: return G, "PASS", "true"
return G, "FLAG", "unverifiable"
```

`FACT_PASS = 0.95`, `tau_contra = 0.62`. A claim with a matching structured record is therefore PASS even when the NLI evidence strongly contradicts it.

Concrete test failures:

| claim | label | predicted | F | D | C |
|---|---|---|---|---|---|
| `F041` | false | **true** | 1.0 | 0.0 | 0.982 |
| `MF007F` | false | **true** | 1.0 | 0.0 | 0.883 |
| `MF008F` | false | **true** | 1.0 | 0.0 | 0.970 |

`F041` is the malformed claim "Semester Registration Schedule for Autumn Semester AY 2024-50 …". `MF007F` is a multi-fact claim whose second fact changes `1000/-` to `1005/-`; the `1005/-` is not parsed as money (see §6), so the attendance record matches and `F = 1.0`.

This also breaks the acceptance criterion from the improvement plan: a multi-fact claim with one wrong fact must not PASS.

A naive fix — check `C` before `F` — is not enough. On the same 109 rows it changes D to **0.7706 accuracy / 0.7511 macro-F1**, because `C >= 0.62` is also true for 15 true claims. There are 18 rows with `F >= 0.95` and `C >= 0.62`: 15 true, 3 false. The NLI contradiction signal is not calibrated well enough to be a veto, and the F short-circuit is hiding that. The gate needs a calibrated `C` (or better evidence selection), not another hardcoded order.

---

## 3. P0 — The "unverifiable" class is circular and mostly not a claim

`absent.py` generates the unsupported class by asking the LLM generic questions, then keeping sentences with `ent < 0.35 and contra < 0.5` (`absent.py:47`). The same `Judge` (the NLI cross-encoder) is used later by `pipeline.py` and `evaluate.py`. The test labels are therefore selected by the very signal used to score them.

The generated set is also poor:

- 57 of 80 claims match hedge/refusal patterns (`varies`, `check`, `contact`, `typically`, `generally`, `should`, `recommend`, `depends`, `consult`, `different`). Examples:
  - "The specific minimum attendance percentage … varies by institution and program; you should check your course …"
  - "Please check your student portal or contact the university's Bursar/Finance office directly …"
  - "This ensures you get the most accurate information regarding borrowing rules."
- 2 are literal placeholders: `U017` "The current mess fee for one semester is \$X" and `U040` "The hostel is scheduled to reopen on **[Insert Date Here]**".
- Only 3 of 80 mention NIT Jamshedpur; 30 mention a generic institution/university/college.
- 30 of the 38 test-unverifiable claims are hedge-like.
- `absent.py:50` assigns `split` randomly per claim, not by source document. The same question appears in both dev and test for 13 questions.
- The "absence audit" in `evaluate.py` excludes claims with `max_entail >= 0.6`, but the generator already filtered at `0.35`; it excludes 0 claims. The claimed contamination repair does nothing.

The effect on the headline is large:

| subset | n | D accuracy | D macro-F1 | B accuracy | B macro-F1 |
|---|---|---|---|---|---|
| all test claims | 109 | 0.8807 | 0.8801 | 0.4954 | 0.4036 |
| non-hedge claims | 79 | 0.8481 | 0.8706 | 0.6835 | 0.4790 |
| real-chunk claims only | 71 | 0.8451 | 0.8449 | 0.7606 | 0.7535 |

D still beats B on the harder subsets, but the 0.8807 headline is inflated by 30 easy refusals.

---

## 4. P0 — Structured records are wrong, low-confidence, and misapplied

`records.py` builds `data/records.json` by BM25-searching a handful of hardcoded keyword queries, taking the first plausible value from the top 8 chunks, and majority-voting. The output is then used by `NumericCheck` as ground truth.

Current records:

| key | value | support | source | problem |
|---|---|---|---|---|
| `attendance_min` | 75 | 3 | `notices_6074.pdf` | plausible |
| `library_fine` | 1000 | 3 | `notices_7125.pdf` | source is a **late registration** notice, not a library fine |
| `revaluation_fee` | 5000 | 1 | `dean_office_notices_10.pdf` | source is a **fee-payment table**; 5000 is a JoSAA processing fee |
| `late_registration_fee` | 1000 | 3 | `notices_5572.pdf` | plausible for one condition |
| `tuition_fee` | 106500 | 1 | `dean_office_notices_19.pdf` | single vote; likely a total fee, not tuition |
| `hostel_fee` | 17000 | 1 | `dean_office_notices_10.pdf` | single vote; same document as mess |
| `mess_fee` | 17000 | 2 | `dean_office_notices_6.pdf` | candidates include 520, 5000, 15000, 17000, 18000 |
| `registration_deadline` | 2026-08-15 | 2 | `notices_7618.pdf` | no source date; candidates span 2024–2026 |

### 4.1 The `operator` field is never used

`records.py:156` stores `operator` (`>=`, `<=`, `=`). `pipeline.py:268-296` never reads it. `NumericCheck` only tests equality/tolerance. A record meaning "library fine is at most 1000" is treated as "must equal 1000". `support` is also never used by the pipeline.

### 4.2 Keyword matching is too loose

`pipeline.py:275-277`:

```python
key_tokens = rec["key"].replace("_", " ").split()
if not any(t in claim.lower() for t in key_tokens):
    continue
```

Any claim containing the word "fee" can match `revaluation_fee` or `late_registration_fee`. Any claim containing "min" can match `attendance_min` (including "admin"). This produces concrete errors:

- `T012` (true) is predicted false: `record revaluation_fee 5000 vs [18000.0] [dean_office_notices_10.pdf]`. The claim is about a hostel/mess fee; the revaluation record wins because the claim contains "Fee".
- `MF004T` (true) is predicted false for the same reason.
- A probe with evidence chunk `c00096` (late registration) and the claim "The tuition fee is Rs 1000" returns `F = 1.0, record late_registration_fee matches`. The same happens for "The hostel fee is Rs 1000" and "The revaluation fee is Rs 1000".

Only 10 of 109 test claims use a record at all; 2 true claims are false negatives caused by the wrong revaluation record.

### 4.3 Record types that are silently ignored

`pipeline.py:279` maps only `currency`, `percent`, `count_days`, and `date`. `records.py` also emits `count_books`, `number` (CGPA), and `time`; those records are never used. The `min_cgpa`, `curfew_time`, and `library_books` records therefore have no effect.

### 4.4 Evidence for records is sometimes mojibake

`data/policies.json` `ATT_75` evidence begins with garbled Devanagari text (`\u0905...`). `source_chunk_id` exists, but the stored evidence is not readable and is not the same document as `records.json`'s `attendance_min` (`notices_5688.pdf` vs `notices_6074.pdf`). Provenance is inconsistent.

---

## 5. P0 — README and TODO contain false or stale numbers

`check_artifacts.py` passes, but it only checks whether a numeric token appears anywhere in an artifact. It does not check semantics, and it explicitly allowlists numbers such as `1147`, `2241`, and `1365` (`check_artifacts.py:11-12`) without tracing them. It also does not scan `dashboard/dist/data`.

Specific contradictions:

| statement | location | actual |
|---|---|---|
| "Claims: 153" | `README.md:13` | 153 in `claims.jsonl` **plus 80** in `claims_absent.jsonl` = 233 used by evaluation |
| "Labels: … 'unverifiable': 0" | `README.md:14` | test has 38 unverifiable (`results/test_predictions.csv`) |
| "83 dev documents (82 claims), 84 test documents (71 claims)" | `README.md:16` | 124 dev claims and 109 test claims actually evaluated |
| "Thresholds and weights were tuned on dev only" | `README.md:17`, `make_docs.py:52` | thresholds tuned; weights fixed at 0.40/0.25/0.35 in `evaluate.py:319` and `config.py:33` |
| "D reaches 0.40 accuracy vs 0.91 on single-fact" | `README.md:84`, `make_docs.py:105`, `TODO.md:28` | `results/subgroup_metrics.csv`: 0.7333 multi-fact, 0.9043 single-fact |
| "cascade and F-only are within 1 point" | `README.md:49` | 0.8807 − 0.8624 = 0.0183, i.e. 1.83 percentage points |
| "values are used only where the extraction agreed across documents" | `README.md:70`, `make_docs.py:99` | `revaluation_fee`, `tuition_fee`, `hostel_fee` have `support = 1` |
| "180 of 2241 PDFs chosen by content coverage" | `README.md:86`, `make_docs.py:107` | `data/selection.json` `pool_size = 463`; 180 came from 463 filtered candidates, not directly from 2241 |
| "Every number … is generated by make_docs.py" | `README.md:2` | `make_docs.py` contains typed literals (0.40, 0.91, "one third", "only one rule"), so this is false |

`README.md:4` also prints "from  PDFs" because `data/corpus_stats.json` has no `pdfs_total` field; `make_docs.py:38` asks for it and gets an empty string.

`TODO.md`:

- `TODO.md:24`: "Audit logs are written to SQLite (`cache/audit.sqlite`)" — no active code writes any audit log. `grep -rn "audit.sqlite" --include=*.py .` returns nothing. `app.py` has no audit middleware.
- `TODO.md:16` says "0 chunks use the fallback date"; `TODO.md:38` says "41% of chunk dates are fallbacks". Both cannot be true.
- `TODO.md:28` repeats the false 0.40 vs 0.91.
- `TODO.md:34` says atomic decomposition is still a TODO, but `pipeline.py:419-440` already splits claims on `", and "` / `"; and "`. The TODO is stale (the implementation is partial, not absent).
- `TODO.md:14` says extending to the full text-bearing set is "a configuration change plus one embedding run; no code change". `select_corpus.py` filters by `ACADEMIC`, `JUNK`, and `TOPICS` quotas; ingesting all 1094 text-bearing PDFs would require changing that selection code or bypassing `selection.json`.

---

## 6. P1 — Baselines are unfair and normalization is buggy

`evaluate.py:54-63`:

- Baseline A: "Is the following claim true? Answer YES or NO only."
- Baseline B: "Is the claim supported by the context? Answer YES or NO only."
- Baseline C: the **same B prompt** at temperature 0.7, three samples, majority vote.

All three baselines are binary. They can never output `unverifiable`. D is three-way and abstains 33%. Comparing three-way accuracy when the baselines are structurally forbidden from using one of the three labels is not a fair test of the verification layer; it is partly a test of task framing. The baseline `abstention_rate = 0.0` rows in `results/condition_metrics.csv` are therefore by construction.

`evaluate.py:68-73` normalizes answers with substring tests:

```python
if "yes" in x and "no" not in x: return "true"
if "no" in x: return "false"
```

`"no"` is a substring of `"know"` and `"unknown"`. "I don't know" becomes `false`; "unknown" becomes `false`; "cannot" contains `"no"` and becomes `false`. This biases A/B/C toward `false` and inflates D's apparent advantage on the `false` class.

With the same labels, the honest comparison is still favorable but smaller:

| subset | A acc | B acc | C acc | D acc |
|---|---|---|---|---|
| all 109 | 0.3211 | 0.4954 | 0.4862 | 0.8807 |
| non-hedge 79 | 0.4430 | 0.6835 | 0.6709 | 0.8481 |
| real-chunk 71 | 0.4930 | 0.7606 | 0.7465 | 0.8451 |

D still wins on the harder subsets; the headline margin is not purely an artifact.

---

## 7. P1 — Claim construction produces fragments and corrupted multi-fact claims

`claims.py` selects sentences with `sentences()`, strips a few prefixes, and accepts anything with 12–42 words and a numeric entity. It does not require a subject, a verb, or a self-contained statement.

- 58 of 153 `claims.jsonl` rows start with a non-letter (bullet, digit, parenthesis). Examples:
  - `T012`: "15000/-) and Mess Fee (Rs.18000/-) for first semester before wish to opt hostel and mess, they can approach to chief warden office."
  - `T009`: "2nd, 4th, 6th and 8th Semester Timing of Examination: A-Shift …"
  - `T019`: "1000/- by visiting the institute on or before 08th August 2025 …"
- 13 claims start with a lowercase letter.
- These fragments are not atomic, self-contained claims; a verifier cannot know what "15000/-)" refers to. They cause false negatives (T012, T009, etc.) that are data problems, not model problems.

Multi-fact construction (`claims.py:219-233`):

```python
nums = re.findall(r"\d+(?:\.\d+)?", b["claim"])
old = nums[0]
new = str(int(old) + rng2.choice([1, 2, 5, 10]))
perturbed_b = b["claim"].replace(old, new, 1)
```

It takes the **first** number in the second sentence, not the number that carries the fact, and does a raw string replacement. If the first number is part of a year, a room number, or a larger token, the replacement can corrupt a different fact or produce `AY 2024-50` (as in `F041`). The base `source_chunk_id` is copied from `a` only; the second source is not recorded. Multi-fact metadata is therefore incomplete.

`claims.py` also has dead logic:

- `if False: pass` at line 205.
- `used` set at 176/193 is never read.
- `SWAPS` at line 21 and `swap_entity()` at 155 are never called.
- The loop break at 178 requires `counts["unverifiable"] >= 60`, but the unverifiable counter is never incremented, so the condition can never be true.
- `split.json` counts only `claims.jsonl` (lines 249-255); it does not include the 80 absent claims that the evaluation actually uses.

`absent.py` adds the 80 unverifiable claims with random splits (line 50), so `split.json` is not the evaluation split.

---

## 8. P1 — Regeneration, audit trail, and API are inconsistent

`pipeline.py:462-487`:

- `claims = [self.verify(...) for ...]` stores the original claims.
- `kept` is built separately; if a BLOCK claim is rewritten successfully, `kept.append(cur)` appends the **new** claim, but `AnswerResult.claims=claims` still returns the **original** list.
- `ClaimResult.regen` is declared (`pipeline.py:34`) but never assigned anywhere. `app.py:137` maps `"regenerated_claim": (c.regen or None)`, so the API always returns `null`.
- If a claim is `FLAG` (unverifiable), it is added to `kept` without regeneration (`pipeline.py:466-468`). `abstained = not kept`, so a response consisting only of FLAG claims is not considered abstained, and the final answer contains unverified statements.

`app.py:138-139` returns:

```python
"weighting_mode": "meta_learner_mle",
"is_synthetic_benchmark": False,
```

There is no meta-learner in the active code. Weights are fixed in `config.py:33` / `evaluate.py:319` / `app.py:31-33`. The label is false.

`dashboard/src/lib/api.ts:41` sends `use_meta_learner` and `retrieval_mode`:

```ts
body: JSON.stringify({ query, institution: 'NITJ_REAL',
  use_meta_learner: useMetaLearner, retrieval_mode: 'hybrid' })
```

`app.py:37-38` `QueryIn` has only `query`. FastAPI/Pydantic ignores the extra fields, so the "Use learned weights" checkbox in the Verify page has no effect. The backend never reads it.

`app.py` also has no audit log, despite `TODO.md:24` claiming SQLite audit logging.

---

## 9. P1 — Reproducibility is broken from a clean checkout

- No `requirements.txt`, `pyproject.toml`, `environment.yml`, or lockfile. Imports include `chromadb`, `numpy`, `ollama`, `rank_bm25`, `fitz` (PyMuPDF), `pandas`, `sklearn`, `scipy`, `fastapi`, `pydantic`, `uvicorn`, `sentence_transformers`, `torch`.
- `README.md` has no install, run, or rebuild instructions. `grep -niE "pip|npm|uvicorn|install|run" README.md` returns only the word "run" in a table cell.
- `.gitignore` ignores the exact artifacts needed to reproduce: `documents/`, `index/`, `cache/`, `results/`, `data/chunks.jsonl`, `data/docs.jsonl`, `data/records.json`, `data/thresholds.json`. A fresh clone has no corpus, no index, no thresholds, no records, and no results.
- `data/claims.jsonl`, `claims_absent.jsonl`, `corpus_stats.json`, `index_manifest.json`, `policies.json`, and `split.json` are tracked, but the files that explain or regenerate them are not all tracked.
- `pipeline.py:236` recency uses `age_years(... or date.today())`; `pipeline_text.age_years` defaults to `date.today()`. Retrieval rankings therefore change with the calendar date. The 2026-11-24 results are not exactly reproducible in 2027 without a frozen evaluation date.
- `artifacts_manifest.json` records `git_commit: daf16b8…`; current HEAD is `982018f`. The manifest is stale, and each entry has no per-entry commit hash (TODO acknowledges this).
- `TODO.md:14` says the full-corpus extension needs "no code change", but selection is code (`select_corpus.py`).
- `dashboard` cannot be rebuilt (see §1.5).

`python check_artifacts.py` does pass:

```
all numbers in docs trace to generated artifacts
```

That is not evidence of correctness. The check:
- scans only `README.md`, `dashboard.html`, `TODO.md` (`check_artifacts.py:10`);
- accepts a token if it appears anywhere in any CSV/JSON, regardless of meaning;
- allowlists 1147/2241/1365 without an artifact;
- does not scan `dashboard/dist/data`, where the fake numbers live.

---

## 10. P1 — Numeric parsing gaps make some false claims invisible

`pipeline_text.py`:

```python
MONEY = re.compile(r"(?:rs\.?|inr|rupees?)\s*([\d][\d,]*(?:\.\d+)?)(?:\s*/-)?", re.I)
```

It requires `Rs`/`INR`/`rupees`. NITJ notices frequently write bare `1000/-`, `15000/-`, `18000/-`. Those values are not parsed as money. `records.py` has the same prefix requirement.

Consequences:

- `MF007F` changes `1000/-` to `1005/-`; `NumericCheck` does not see either as money, so the attendance record (75%) matches and the claim gets `F = 1.0`, `D = 0.0`, and PASS.
- `T012` contains `15000/-` and `Rs.18000/-`; only the `Rs.18000` is parsed, the `15000/-` is invisible, and the wrong `revaluation_fee` record wins.
- `replace_money` in `claims.py` also requires a prefix, so `claims.py` cannot perturb bare `1000/-` through the normal path; only the multi-fact raw `.replace()` creates those false claims, and the verifier cannot see them.

`records.py:98-104` parses a date as `YYYY-MM-DD` from `DD/MM/YYYY` without checking whether the source is `MM/DD/YYYY`; it can emit an invalid date. `records.py:84-87` only recognizes CGPA as `x/10` or `x point`, so a plain "CGPA 6.5" record is missed.

---

## 11. P1 — Unsupervised metrics are misdefined

`evaluate.py:235-240`:

```python
ar_ent, _, _ = judge.entail(txt, [{"text": q, "doc_name": "query"}]) if False else (0.0, 0.0, "")
...
ar_ent, _, _ = judge.entail(q, [{"text": txt[:UNSUP_CTX_CHAR], "doc_name": "answer"}])
```

`judge.entail(claim, evidence)` builds pairs `(evidence, claim)`. Calling `judge.entail(q, answer)` tests whether the **answer** entails the **query**, not whether the query is answered by the answer. The resulting `answer_relevance` for D is 0.155, below A's 0.743; the metric is not measuring relevance in the usual direction.

`answers["C"]` (`evaluate.py:222`) chooses the sample with the highest sum of token overlap to all samples, including itself. That favors long samples and is not a standard self-consistency vote. C uses the B prompt, not the A prompt, so it is self-consistency of RAG, not of the plain LLM.

`context_relevance` is computed once per query and reused for all four conditions, so the identical 0.096 in the table is a property of retrieval, not of the answer condition. The table layout implies a comparison that does not exist.

The `unsupervised_answers.csv` shows 24 queries, no gold answers, and entailment-only scoring. That is a sanity check, not a benchmark; the README does label it as entailment-based, which is fair, but the `answer_relevance` column is misleading.

---

## 12. P2 — Engineering and hygiene bugs

- `index.py:62-64` loops by `CHROMA_ADD_BATCH` but slices by a hardcoded `400`. If `CHROMA_ADD_BATCH` is changed, the batch boundaries and slices disagree.
- `config.py:34-35` defines `TAU_PASS` and `TAU_BLOCK`, but `Gate` never imports them; `Gate` reads `data/thresholds.json` or defaults to 0.60. The config values are dead.
- `config.py:19` `PDF_TARGET = 150`, but the corpus has 180 PDFs because `select_corpus.py` uses `PDF_TARGET + 30`. The config does not describe the actual selection.
- `config.py:43` has a comment; `evaluate.py` has comments at 161, 186, 189, 191, 193, 195, 197, 223, 235, 292, 351, 359, 372, 377. This violates the "no comments" requirement.
- `evaluate.py:211` takes `conditions` but never uses it.
- `evaluate.py:292` initializes `cfg = {}` and overwrites it at 317; dead assignment.
- `evaluate.py:351-358` builds `sg` and then rebuilds `sg_rows` at 359-369; the first loop is dead.
- `evaluate.py:236` contains `if False else (0.0, 0.0, "")`; dead branch.
- `results/test_predictions.csv` stores `pred_a`, `pred_b`, `pred_c` as booleans (`True`/`False`) while `label` and `pred_d` are strings. Recomputing metrics from the CSV requires manual type normalization; a naive `accuracy_score(df.label, df.pred_a)` returns 0.0.
- `pipeline.py:106` `Cache.key` includes model, temperature, num_predict, seed, and prompt, but not `num_ctx`. Changing `LLM_NUM_CTX` reuses stale cached generations.
- `pipeline.py` `LLM.call` raises `RuntimeError("empty generation")` with no retry or fallback; a single empty Ollama response aborts a long evaluation.
- `Judge.__init__` loads the NLI model with `local_files_only=True`; if the model is missing, it silently falls back to the LLM judge and stores `load_error`, which is never surfaced in API output or `stats.json`. `results/stats.json` only reports the first row's `judge_backend`.
- `Reranker.__init__` silently disables reranking on any exception and stores `self.error`, which is never surfaced. Retrieval quality can change without notice.
- `Retriever.stats` counts dense/rerank calls but is never reported.
- `pipeline.py` `best_sentences` has an unused `ct = tokenize(claim)` at line 232.
- `pipeline.py` `NumericCheck.compare` uses a hardcoded `0.5` for day tolerance and `max(1.0, abs(rvn))`; these are not config constants.
- `pipeline.py` recency uses hardcoded `0.55 + 0.45 * exp(...)`, not config.
- `pipeline.py` `best_sentences` dedups on `s[:80]`, so two different sentences sharing an 80-character prefix are treated as duplicates.
- `pipeline.py` `verify` recursively retrieves for every atomic part, so a two-part claim does two full retrievals plus one for the parent. This is slow and can produce inconsistent evidence across parts.
- `app.py` `get_pipeline` protects creation with a lock, but `Retriever.stats` is mutated without a lock; concurrent requests can race on the counters (minor, counters only).
- `app.py` has no request ID, no structured logging, and no audit record; `TODO.md:24` claims otherwise.
- `check_artifacts.py` `ALLOW` includes 1147, 2241, 1365, etc.; a wrong document number equal to one of those passes.
- `make_manifest.py` imports `hashlib` and `pathlib` but uses neither.
- `make_manifest.py` writes a single `git_commit` at generation time; `artifacts_manifest.json` is stale (`daf16b8` vs HEAD `982018f`).
- `make_docs.py` writes `README.md` and `dashboard.html`, but never updates `dashboard/dist/data`, so the frontend cannot pick up new results.
- The backend serves `dashboard.html` at `/` while the Vite app serves `dashboard/dist` at `:5176`; the two show different numbers.

---

## 13. Hardcodedness inventory

The previous pass moved many constants into `config.py`, but a lot of domain and tuning logic remains hardcoded:

### 13.1 Evaluation and weights

- `config.py:33` `DEFAULT_WEIGHTS = {"w_fact": 0.40, "w_policy": 0.25, "w_ground": 0.35}` — not env-overridable.
- `evaluate.py:319` writes the same weights into `thresholds.json` as literals.
- `app.py:31-33` repeats the same weights as Pydantic defaults.
- `evaluate.py:294-295` tunes only `tau_pass`/`tau_contra` over a hardcoded 5×5 grid; weights, `FACT_BLOCK`, `FACT_PASS`, and `GROUND_PENALTY` are never tuned.
- `evaluate.py:300-314` reimplements the gate instead of calling `Gate.decide`; a future gate change will silently diverge from the ablation.
- `evaluate.py:186-207` hardcodes the D-only/F-only/P-only/G-only thresholds.
- `evaluate.py:68-73` hardcodes the string normalization.
- `evaluate.py:252-278` hardcodes 24 `QUERIES`.
- `absent.py:9-21` hardcodes 12 `EXTRA` questions; `absent.py:47` hardcodes `ent < 0.35 and contra < 0.5`; `absent.py:50` hardcodes the 0.5 split.

### 13.2 Data construction

- `ingest.py:14` hardcodes `EXCLUDE`; `ingest.py:15` hardcodes `QUOTA`; `ingest.py:17-25` hardcodes `CATEGORY_HINTS`.
- `select_corpus.py:9-12` hardcodes `EXCLUDE`, `ACADEMIC`, `JUNK`; `select_corpus.py:14-27` hardcodes 12 `TOPICS` quotas; `select_corpus.py:53` hardcodes `PDF_TARGET + 30`.
- `claims.py:11-19` hardcodes `BOILER`; `claims.py:20-32` hardcodes `SWAPS`; `claims.py:178` hardcodes the 60/60/60 target via `CLAIMS_PER_LABEL`; `claims.py:207` hardcodes 20 iterations and 15 multi-fact pairs; `claims.py:228` hardcodes the +1/+2/+5/+10 perturbation choices.
- `records.py:13-31` hardcodes 16 `SPEC` entries, their descriptions, value types, operators, and keyword queries; `records.py:117-137` hardcodes `sane()` ranges.
- `pipeline.py:419` hardcodes the atomic split regex; `pipeline.py` recency constants; `NumericCheck` day tolerance 0.5.

### 13.3 Runtime and models

- `pipeline.py:64` `CrossEncoder(NLI_MODEL, max_length=512, device="cpu")`; `Reranker` `max_length=512`, `device="cpu"`.
- `pipeline.py:99` `Judge._llm_judge` uses `evidence[:3]`; `evidence` limit is hardcoded.
- `pipeline.py` `best_sentences` uses `s[:80]` and `BEST_SENT_MIN_WORDS`.
- `index.py:63-64` hardcodes `400`.
- `app.py:20` defaults CORS to `*`.
- `dashboard/src/lib/api.ts:8-11` hardcodes the nine dataset names; `EvaluationPage` hardcodes 91.7/2.5/0.926; `HomePage` hardcodes the "Benchmark sample" query and labels.

### 13.4 Config constants that are dead or misleading

- `TAU_PASS` and `TAU_BLOCK` are never used.
- `PDF_TARGET` does not match the 180-PDF corpus.
- `DEFAULT_WEIGHTS` is duplicated and not the single source of truth.
- `SPLIT_DEV` only controls `claims.py`; `absent.py` ignores it.

---

## 14. Lies and misleading statements

| # | statement | where | why it is false/misleading |
|---|---|---|---|
| 1 | "Proposed pipeline · 120 claims · 91.7% accuracy · 2.5% missed falsehoods" | `dashboard/dist/assets/EvaluationPage-hfblF8h_.js`; `dashboard/dist/data/metrics.json` | The current test set has 109 scored claims and 0.8807 accuracy / 0.0367 missed falsehoods. 91.7/2.5 are the old synthetic numbers. |
| 2 | "Cohen's kappa 0.812, 80 human-reviewed claims" | `dashboard/dist/data/human-eval.json`; Evaluation page | There is no human annotation in the new code. The old `human-eval.json` is the simulated annotator artifact. |
| 3 | "ROC-AUC 0.926 · Five-fold validation" | Evaluation page hardcoded string | No ROC/AUC or five-fold validation is computed by the active code. The old `roc_curve_comparison.png` is from the synthetic run. |
| 4 | "Logistic Regression Meta-Learner (Ours)" and "Learned weights 0.508/0.315/0.177" | `weights-comparison.json`; Evaluation page | No meta-learner exists in the active code. All four weight methods in the file have identical 91.7/100/78.2 — the exact non-identifiability flaw. |
| 5 | "MFAQ 78.2%, 55 claims" | `mfaq-comparison.json`; Evaluation page | No MFAQ code or data exists in the active repository. |
| 6 | "Hybrid retrieval Recall@1 79.2, Recall@5 94.8" | `retrieval-comparison.json`; Evaluation page | No retrieval benchmark exists in the active repository. |
| 7 | "Audit logs are written to SQLite (cache/audit.sqlite)" | `TODO.md:24` | No active code writes an audit log; `grep` finds no `audit.sqlite`. |
| 8 | "Claims: 153 … unverifiable: 0" | `README.md:13-14` | The evaluation uses 233 claims (153 + 80) and 38 test-unverifiable claims. |
| 9 | "82 dev claims / 71 test claims" | `README.md:16` | 124 dev / 109 test predictions. |
| 10 | "D reaches 0.40 multi-fact vs 0.91 single-fact" | `README.md:84`, `TODO.md:28`, `make_docs.py:105` | Artifacts say 0.7333 and 0.9043. |
| 11 | "cascade and F-only are within 1 point" | `README.md:49` | 1.83 points. |
| 12 | "values are used only where extraction agreed across documents" | `README.md:70`, `make_docs.py:99` | `revaluation_fee`, `tuition_fee`, `hostel_fee` have support 1. |
| 13 | "Thresholds and weights were tuned on dev only" | `README.md:17`, `make_docs.py:52` | Only thresholds are tuned; weights are fixed. |
| 14 | "Every number in this file is generated by make_docs.py" | `README.md:2` | `make_docs.py` contains typed literals (0.40, 0.91, "one third", "only one rule"). |
| 15 | "180 of 2241 PDFs chosen by content coverage" | `README.md:86`, `make_docs.py:107` | `selection.json` pool is 463; the 180 were chosen from 463 filtered candidates. |
| 16 | "Audit logs … SQLite rather than MongoDB … deliberate deviation" | `TODO.md:24` | Same as #7; no audit log at all. |
| 17 | "0 chunks use the fallback date" vs "41% of chunk dates are fallbacks" | `TODO.md:16` vs `TODO.md:38` | Mutually contradictory. |
| 18 | "Atomic decomposition … still a TODO" | `TODO.md:34`, `README.md:84` | `pipeline.py` already splits on `, and ` / `; and `. The implementation is partial, but the statement is stale. |
| 19 | "Extending to the full text-bearing set … no code change" | `TODO.md:14` | `select_corpus.py` filters and quotas; bypassing them is a code change. |
| 20 | "Weighting mode: meta_learner_mle" | `app.py:138` | Fixed weights; no meta-learner. |
| 21 | "Use learned weights" checkbox | Verify page | Backend `QueryIn` has only `query`; the flag is ignored. |
| 22 | "Synthetic benchmark" notice removed | commit `982018f` | The dashboard still serves the synthetic benchmark files, so the warning was removed while the data remained synthetic. |
| 23 | "git_commit: daf16b8…" | `artifacts_manifest.json` | Current HEAD is `982018f`; the manifest is stale. |
| 24 | "All numbers in docs trace to artifacts" | `check_artifacts.py` output | The checker is token-based, allowlists untraced numbers, and does not scan the dashboard. |

---

## 15. What is actually measured and credible

Keep these parts:

- The corpus is real: 2,241 PDFs in `documents/`, 180 selected/parsed, 1,423 deduplicated chunks, 1,423 Chroma vectors at 768 dimensions.
- The NLI judge is real (`cross-encoder/nli-MiniLM2-L6-H768`); `results/stats.json` reports `judge_backend: nli`.
- The test predictions are computed from real model calls; recomputing the metrics from `test_predictions.csv` matches `condition_metrics.csv` exactly.
- D still beats B on the 71 real-chunk claims (0.8451 vs 0.7606 accuracy; 0.8449 vs 0.7535 macro-F1) and on the 79 non-hedge claims (0.8481 vs 0.6835; 0.8706 vs 0.4790).
- The live `/api/verify-query` path is real and returns a grounded answer with per-claim F/P/D/G, decisions, and evidence.
- McNemar D vs B on all 109 is 45 better / 3 worse, p ≈ 0; bootstrap delta 0.3853 (0.2844–0.4862). These are computed. They are not, however, a substitute for fixing the benchmark and gate.

The honest claim after the fixes should be something like:

> On 71 claims drawn verbatim or perturbed from held-out NITJ documents (36 true, 35 false), the verified pipeline reaches 0.845 accuracy / 0.845 macro-F1 versus 0.761 / 0.754 for RAG. On 79 claims excluding model-generated refusals, it reaches 0.848 / 0.871 versus 0.684 / 0.479. An additional 38 model-generated "unverifiable" claims are excluded from the headline because they are selected by the same NLI judge and are mostly hedges/refusals.

That is a defensible result, and still a win.

---

## 16. Recommended fix order

1. **Replace the dashboard data, not just the banner.** Delete or regenerate `dashboard/dist/data/*.json` from `results/`; remove the hardcoded 91.7/2.5/0.926/120/80/0.508 strings; make Home/Results/Audit read `/api/metrics`, `/api/stats`, `/api/corpus`, and a new `/api/audit` endpoint. Do not show any human-eval/MFAQ/retrieval panel until there is a script that produces it.
2. **Fix the gate.** Do not let `F >= 0.95` override a high `C`. Either calibrate `C` on dev or require `D` to be above a floor before PASS. Add a regression test from `F041`, `MF007F`, `MF008F`, `T012`, `MF004T`.
3. **Replace the unverifiable set.** Generate it from real absent questions with human review; require the claim to be a self-contained assertion about NITJ; deduplicate; split by source question; never filter with the evaluation judge. Drop the 57 hedge/refusal sentences. Re-run.
4. **Rebuild `records.json`.** Verify each value against its cited chunk by hand; drop records with support < 2; use an exact key/phrase match, not any token; honor `operator`; use the record's `val_type` for all types; add a unit test that a tuition claim cannot match `late_registration_fee`.
5. **Fix numeric parsing.** Accept bare `1000/-`, `Rs 1,00,000`, `₹`, and `lakh`; make `replace_money` and `parse_money` use the same grammar; add a regression test for `MF007F`.
6. **Fix the claim set.** Require a subject/verb or reject fragments; strip bullets; record both source chunks for multi-fact claims; for multi-fact false claims, perturb the value that belongs to the selected fact, not the first number in the string.
7. **Fix the baselines.** Give A/B/C the same three-way label space and an explicit `UNVERIFIABLE` option; fix `norm` to parse tokens, not substrings; use the same prompt for C as the condition it is testing; report the full prompt and call budget.
8. **Fix the docs.** Regenerate `README.md`/`TODO.md` from artifacts, remove typed limitations, replace "153/0/82/71" with the real 233/38/124/109, replace 0.40/0.91 with 0.7333/0.9043, replace "within 1 point" with 1.83, and delete the audit-log claim until audit logging exists.
9. **Fix reproducibility.** Add `requirements.txt` (or `pyproject.toml`) with pinned versions; add README run/rebuild steps; stop ignoring the artifacts needed to run or document how to regenerate them; freeze an evaluation date (or remove `date.today()` from retrieval/prompting); regenerate `artifacts_manifest.json` and include per-entry commit hashes.
10. **Fix the frontend repo.** Restore the real React/TypeScript source or remove the claim that the dashboard is source-reproducible; add `dependencies`/`devDependencies` and a `build` script; stop tracking `node_modules`; add `dashboard/node_modules` to `.gitignore`; either rebuild `dist` from source or document that `dist` is the only artifact.
11. **Fix the API.** Actually read `use_meta_learner` or remove the checkbox; set `weighting_mode` to `"fixed_weights"`; populate `regenerated_claim` and return the regenerated claims in `AnswerResult.claims`; do not call a response "not abstained" when every claim is FLAG.
12. **Fix the checker.** Make `check_artifacts.py` parse each sentence/row and verify the number against the named artifact; scan `dashboard/dist/data`; remove the untraced allowlist entries; fail on a stale `git_commit`.
13. **Remove dead code/comments.** `if False`, unused `used`, `SWAPS`, `conditions`, `cfg`, duplicate subgroup loop, `ct`, comments in `evaluate.py`/`config.py`.
14. **Add tests.** At minimum: gate order, record matching, money parsing, fragment rejection, baseline normalization, and a clean-checkout rebuild.

---

## 17. Verification performed

- `python check_artifacts.py` → `all numbers in docs trace to generated artifacts` (exit 0). This is the weak check described in §9.
- Recomputed all four conditions from `results/test_predictions.csv`: A 0.3211/0.2160, B 0.4954/0.4036, C 0.4862/0.3933, D 0.8807/0.8801 — matches `results/condition_metrics.csv`.
- Recomputed D on subsets: non-hedge 79 = 0.8481/0.8706; real-chunk 71 = 0.8451/0.8449; false-missed on real-chunk = 0.0563.
- Replayed the gate with `C` before `F`: 0.7706/0.7511 (vs 0.8807/0.8801), showing the short-circuit is load-bearing, not cosmetic.
- Counted 18 rows with `F >= 0.95` and `C >= 0.62`: 15 true, 3 false; the 3 false are `F041`, `MF007F`, `MF008F`.
- Probed `NumericCheck` with evidence `c00096`: "The tuition fee is Rs 1000" and "The hostel fee is Rs 1000" both return `record late_registration_fee matches`.
- Counted absent claims: 57/80 hedge-like, 30/38 test unverifiable hedge-like, 3/80 NITJ mentions, 2 placeholders, 13 questions in both splits.
- Counted claim fragments: 58/153 start with a non-letter.
- Reviewed `dashboard/dist/data/*.json`, `dashboard/dist/assets/EvaluationPage-*.js`, `HomePage-*.js`, `AuditPage-*.js`, and `dashboard/src/lib/api.ts`.
- `git ls-files`: 10,226 dashboard/node_modules files tracked; `dashboard/src` contains only `api.ts`; `dashboard/package.json` has no dependencies; no requirements file; `artifacts_manifest.json` commit `daf16b8` vs HEAD `982018f`.

**Not done:** no model was re-run, no code was changed, and no live visual check of the dashboard was performed in this pass. The numbers above are read from the committed/generated artifacts and recomputed from `test_predictions.csv`.
