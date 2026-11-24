# Deferred work

## Supervised NITJSR evaluation
The real-corpus evaluation is unsupervised by design. A supervised version needs human labels that do not exist yet:

1. Sample claims from `data/claims.jsonl` and `data/claims_absent.jsonl`.
2. Two annotators label each claim against the cited chunk (`source_chunk_id`), blind to the other's labels.
3. Store raw labels, compute Cohen's kappa with a bootstrap confidence interval, adjudicate disagreements.
4. Add the annotated file to `evaluate.py` as a labelled test set and report both the constructed-label and human-label results.

Until that exists, the headline remains constructed-label accuracy on real-chunk claims, not a human-verified accuracy.

## Corpus
- `documents/` contains 2241 PDFs; `select_corpus.py` filters them to a pool of 463 and all 463 are ingested (`PDF_TARGET=463`). Extending to every text-bearing PDF requires changing the selection quotas in `select_corpus.py` or bypassing `data/selection.json`; it is not only a configuration change.
- PDFs without a text layer still need OCR before ingestion.
- Date metadata is noisy. The fallback-date share is not reported as a single number here because the previous two figures contradicted each other; use `data/corpus_stats.json` and `data/quality_audit.csv` as the source of truth.
- Very wide tables can still lose column alignment when split at the table word limit.

## Models
- The NLI cross-encoder (`cross-encoder/nli-MiniLM2-L6-H768`) is expected to be cached; `GET /api/health` reports the active judge backend and any load error.
- The reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is optional; `/api/health` reports whether it loaded.

## API
- SQLite audit logging is implemented in `app.py` at `cache/audit.sqlite` for `/verify-claim`, `/ask`, and `/api/verify-query`. The stored response is truncated, so it is an audit summary rather than a full prompt/response archive.
- No authentication; the service is intended to bind to localhost.
- `use_meta_learner` is not part of the API. Weights are fixed and normalised per request; the response says `fixed_weights`.

## Evaluation
- Claims are single-fact perturbations, verbatim sentences, and concatenated multi-fact claims. A fragment filter exists and rejects them at construction time: the source-backed test set currently contains 0 fragmentary claims (`results/quality_audit.csv`).
- Multi-fact performance is reported in `results/subgroup_metrics.csv`; atomic decomposition is a heuristic split on `, and` / `; and`, not a parser.
- A/B/C predictions were rerun with three-way TRUE/FALSE/UNVERIFIABLE prompts. `results/baseline_rerun.json` records the model, seed, sample count and prompt hashes.
- The 38 test unverifiable claims are model-generated and filtered by the same NLI judge used for scoring. They are audited in `results/quality_audit.csv` and excluded from the source-backed headline; replacing them needs human-written absent questions.
- Weights are fixed, not learned. Only `tau_pass` and `tau_contra` are tuned on dev.
- The regeneration loop rewrites blocked claims up to the configured rounds and abstains when no claim passes.

## Open gaps
- Full prompt/response audit trail with model digest, seed and per-claim chunk ids; the current SQLite audit stores request and response summaries.
- Calibration reporting is a Brier score plus a binned CSV; there is no reliability plot yet.
- `artifacts_manifest.json` records a per-entry artifact commit and the HEAD at generation time, but it does not capture uncommitted working-tree changes.
- Independent human review of the generated absent claims and the fragmentary constructed claims.
