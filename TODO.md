# Deferred work

## Supervised NITJSR evaluation
The real-corpus evaluation is unsupervised by design. A supervised version needs human labels that do not exist yet:

1. Sample 150 claims from `data/claims.jsonl` and `data/claims_absent.jsonl`.
2. Two annotators label each claim against the cited chunk (`source_chunk_id`), blind to the other's labels.
3. Store raw labels, compute Cohen's kappa with a bootstrap confidence interval, adjudicate disagreements.
4. Re-use `evaluate.py` unchanged; pass `--mode test` with the annotated file.

Until that exists, no accuracy number for real-corpus claims is reported anywhere.

## Corpus
- 180 of 2241 PDFs are ingested. Extending to the full text-bearing set (1094 documents) is a configuration change plus one embedding run; no code change.
- 1147 PDFs have no text layer and need OCR before they can be indexed.
- Date coverage improved to 79% of chunks (was 59%); 0 chunks use the fallback date after metadata repair.
- Tables are extracted with PyMuPDF and split at 1100 words with repeated headers; very wide tables can still lose column alignment.

## Models
- The NLI cross-encoder (`cross-encoder/nli-MiniLM2-L6-H768`) is cached and active. `Judge` reports `judge_backend: nli`.
- The reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is cached and active.

## API
- Audit logs are written to SQLite (`cache/audit.sqlite`) rather than MongoDB. The proposal named MongoDB; this is a deliberate, documented deviation.
- No authentication; the service binds to localhost only.

## Evaluation
- Claims are single-fact perturbations of real sentences plus 15 multi-fact (concatenated) claims. Multi-fact accuracy is 0.40 vs 0.91 single-fact; atomic decomposition is still a TODO.
- The unsupervised leg reports entailment-based support against retrieved evidence, not human faithfulness judgements.
- Weights and G are not identifiable on this data; the cascade is dominated by the numeric check (F). The learned-weight calibration claim is withdrawn.
- The regeneration loop rewrites blocked claims up to 2 rounds and abstains if still ungrounded.

## Open gaps (in-scope, not yet closed)
- Atomic decomposition of multi-fact claims in `verify()`.
- Full audit trail with prompts, responses, model digest, seed, chunk ids in `results/test_predictions.csv`.
- Calibration curve / Brier reporting beyond the single number.
- `artifacts_manifest.json` maps numbers to scripts but not to commit hashes.
- Metadata repair: 41% of chunk dates are fallbacks; `is_latest` recomputed but source dates still noisy.