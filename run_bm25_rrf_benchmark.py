"""
Comprehensive BM25 Sparse vs Dense vs Hybrid Reciprocal Rank Fusion (RRF) Benchmark
===================================================================================
Compares:
1. Sparse Lexical Search: BM25Okapi
2. Dense Semantic Vector Search: nomic-embed-text + FAISS (Cosine Similarity)
3. Standard Hybrid RRF: Reciprocal Rank Fusion (k=60, equal 1.0:1.0 weights)
4. Calibrated Hybrid RRF: Reciprocal Rank Fusion (k=20, w_dense=2.0, w_bm25=1.0)

Evaluated across:
A. Real-World MFAQ Corpus (250 QA pairs from clips/mfaq)
B. Full Academic Verification Dataset (150 claims: NIT Jamshedpur + Public University)
"""

import os
import re
import json
import time
import random
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import faiss

from verification_pipeline import (
    AcademicVerificationPipeline,
    SYNTHETIC_NITJ_DOCS, SYNTHETIC_NITJ_RECORDS, SYNTHETIC_NITJ_POLICIES,
    SYNTHETIC_PUBLIC_DOCS, SYNTHETIC_PUBLIC_RECORDS, SYNTHETIC_PUBLIC_POLICIES,
    tokenize_text, LLMClient
)

sns.set_theme(style="whitegrid", font_scale=1.1)
FIG_DIR = "evaluation_results"
os.makedirs(FIG_DIR, exist_ok=True)
CACHE_DIR = "nb_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 80)
print("HYBRID RETRIEVAL BENCHMARK: BM25 vs DENSE vs RECIPROCAL RANK FUSION (RRF)")
print("=" * 80)

# ============================================================================
# PART 1: Real-World MFAQ Retrieval Benchmark (250 FAQ Pairs)
# ============================================================================
print("\n[PART 1] Loading MFAQ Real-World FAQ Benchmark (250 pairs)...")
stream = load_dataset("clips/mfaq", "en_flat", split="train", streaming=True)
pairs = []
for row in stream:
    q = row.get("question", "").strip()
    a = row.get("answer", "").strip()
    if len(q) > 15 and len(a) > 25:
        pairs.append({"question": q, "answer": a})
        if len(pairs) >= 250:
            break

unique_docs = []
seen_answers = set()
q_to_doc = {}
for i, p in enumerate(pairs):
    ans = p["answer"]
    if ans not in seen_answers:
        seen_answers.add(ans)
        unique_docs.append(ans)
        q_to_doc[i] = len(unique_docs) - 1
    else:
        q_to_doc[i] = unique_docs.index(ans)

print(f"Loaded {len(pairs)} QA pairs mapped to {len(unique_docs)} unique corpus passages.")

# Load embeddings from cache
llm = LLMClient()
corpus_vecs = llm.embed_texts(unique_docs)
c_norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
c_norms[c_norms == 0] = 1.0
norm_corpus_vecs = corpus_vecs / c_norms

questions = [p["question"] for p in pairs]
q_vecs = llm.embed_texts(questions)
q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
q_norms[q_norms == 0] = 1.0
norm_q_vecs = q_vecs / q_norms

# Dense Index
d_dim = norm_corpus_vecs.shape[1]
faiss_index = faiss.IndexFlatIP(d_dim)
faiss_index.add(norm_corpus_vecs)

# Dense Similarity Matrix
dense_sim_matrix = np.dot(norm_q_vecs, norm_corpus_vecs.T)

# BM25 Index
tokenized_corpus = [tokenize_text(d) for d in unique_docs]
bm25_model = BM25Okapi(tokenized_corpus)

bm25_rankings = []
t0_bm25 = time.time()
for p in pairs:
    toks = tokenize_text(p["question"])
    scores = bm25_model.get_scores(toks) if toks else np.zeros(len(unique_docs))
    bm25_rankings.append(list(np.argsort(-scores)))
bm25_latency_ms = ((time.time() - t0_bm25) / len(pairs)) * 1000.0

t0_dense = time.time()
dense_rankings = []
for i in range(len(pairs)):
    dense_rankings.append(list(np.argsort(-dense_sim_matrix[i])))
dense_latency_ms = ((time.time() - t0_dense) / len(pairs)) * 1000.0


def compute_retrieval_metrics(get_ranking_fn, n_items=250):
    r1, r3, r5, r10 = 0, 0, 0, 0
    mrr_list = []
    for i in range(n_items):
        gold = q_to_doc[i]
        ranking = get_ranking_fn(i)
        if gold in ranking[:1]: r1 += 1
        if gold in ranking[:3]: r3 += 1
        if gold in ranking[:5]: r5 += 1
        if gold in ranking[:10]: r10 += 1
        if gold in ranking:
            mrr_list.append(1.0 / (ranking.index(gold) + 1))
        else:
            mrr_list.append(0.0)
    return {
        "Recall@1": round((r1 / n_items) * 100.0, 2),
        "Recall@3": round((r3 / n_items) * 100.0, 2),
        "Recall@5": round((r5 / n_items) * 100.0, 2),
        "Recall@10": round((r10 / n_items) * 100.0, 2),
        "MRR": round(float(np.mean(mrr_list)), 4)
    }


def make_rrf_ranking(i, k_rrf=60, w_dense=1.0, w_bm25=1.0, candidate_depth=50):
    d_rank = dense_rankings[i][:candidate_depth]
    b_rank = bm25_rankings[i][:candidate_depth]
    rrf_scores = {}
    for r, d_idx in enumerate(d_rank):
        rrf_scores[d_idx] = rrf_scores.get(d_idx, 0.0) + w_dense / (k_rrf + r + 1)
    for r, d_idx in enumerate(b_rank):
        rrf_scores[d_idx] = rrf_scores.get(d_idx, 0.0) + w_bm25 / (k_rrf + r + 1)
    return sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)


print("Computing retrieval performance metrics...")
mfaq_results = []

# 1. Sparse BM25
m_bm25 = compute_retrieval_metrics(lambda i: bm25_rankings[i])
m_bm25.update({"Method": "1. Sparse BM25 Only", "Latency_ms": round(bm25_latency_ms, 2)})
mfaq_results.append(m_bm25)

# 2. Dense Semantic
m_dense = compute_retrieval_metrics(lambda i: dense_rankings[i])
m_dense.update({"Method": "2. Dense Only (FAISS)", "Latency_ms": round(dense_latency_ms, 2)})
mfaq_results.append(m_dense)

# 3. Standard Hybrid RRF (k=60, 1.0:1.0)
t0_rrf60 = time.time()
m_rrf60 = compute_retrieval_metrics(lambda i: make_rrf_ranking(i, k_rrf=60, w_dense=1.0, w_bm25=1.0))
rrf60_latency_ms = ((time.time() - t0_rrf60) / len(pairs)) * 1000.0 + bm25_latency_ms + dense_latency_ms
m_rrf60.update({"Method": "3. Standard Hybrid RRF (k=60, 1:1)", "Latency_ms": round(rrf60_latency_ms, 2)})
mfaq_results.append(m_rrf60)

# 4. Calibrated Hybrid RRF (k=20, 2.0:1.0)
t0_rrf20 = time.time()
m_rrf20 = compute_retrieval_metrics(lambda i: make_rrf_ranking(i, k_rrf=20, w_dense=2.0, w_bm25=1.0))
rrf20_latency_ms = ((time.time() - t0_rrf20) / len(pairs)) * 1000.0 + bm25_latency_ms + dense_latency_ms
m_rrf20.update({"Method": "4. Calibrated Hybrid RRF (k=20, 2:1)", "Latency_ms": round(rrf20_latency_ms, 2)})
mfaq_results.append(m_rrf20)

df_mfaq_retrieval = pd.DataFrame(mfaq_results)
print("\n=== MFAQ Retrieval Benchmark Results (250 FAQ Pairs) ===")
print(df_mfaq_retrieval[["Method", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "Latency_ms"]])

# ============================================================================
# PART 2: Full Academic Verification Dataset (150 Claims: NITJ + Public Univ)
# ============================================================================
print("\n[PART 2] Evaluating Full Academic Verification Dataset (150 Claims) under Retrieval Backends...")

df_primary = pd.read_csv("primary_audit_trail.csv")
df_secondary = pd.read_csv("secondary_audit_trail.csv")

claims_eval = []
for _, row in df_primary.iterrows():
    claims_eval.append({
        "claim": row["claim"],
        "true_label": row["label"],
        "category": row["category"],
        "institution": "NITJ"
    })
for _, row in df_secondary.iterrows():
    claims_eval.append({
        "claim": row["claim"],
        "true_label": row["label"],
        "category": row["category"],
        "institution": "PUBLIC"
    })

print(f"Total claims to verify across institutions: {len(claims_eval)} (120 NITJ + 30 Public)")

# Setup Pipelines for each mode
backends = ["dense", "bm25", "hybrid"]
backend_metrics = []

for backend in backends:
    p_nitj = AcademicVerificationPipeline(
        docs=SYNTHETIC_NITJ_DOCS,
        records=SYNTHETIC_NITJ_RECORDS,
        policies=SYNTHETIC_NITJ_POLICIES,
        use_meta_learner=True,
        retrieval_mode=backend
    )
    p_pub = AcademicVerificationPipeline(
        docs=SYNTHETIC_PUBLIC_DOCS,
        records=SYNTHETIC_PUBLIC_RECORDS,
        policies=SYNTHETIC_PUBLIC_POLICIES,
        use_meta_learner=True,
        retrieval_mode=backend
    )

    preds = []
    d_scores_true = []
    d_scores_false = []
    d_scores_unverifiable = []

    for item in claims_eval:
        pip = p_nitj if item["institution"] == "NITJ" else p_pub
        res = pip.verify_single_claim(item["claim"], retrieval_mode=backend, skip_regeneration=True)
        preds.append(res.predicted_label)
        if item["true_label"] == "true":
            d_scores_true.append(res.D_score)
        elif item["true_label"] == "false":
            d_scores_false.append(res.D_score)
        else:
            d_scores_unverifiable.append(res.D_score)

    # Compute metrics
    labels = [c["true_label"] for c in claims_eval]
    correct = sum(1 for p, y in zip(preds, labels) if p == y)
    acc = (correct / len(labels)) * 100.0

    # False claims missed
    false_idx = [i for i, y in enumerate(labels) if y == "false"]
    missed_false = sum(1 for idx in false_idx if preds[idx] != "false")
    missed_rate = (missed_false / len(false_idx)) * 100.0 if false_idx else 0.0

    backend_metrics.append({
        "Retrieval_Backend": f"{backend.capitalize()} Search",
        "Total_Claims": len(labels),
        "Overall_Accuracy": round(acc, 2),
        "Missed_Falsehood_Rate": round(missed_rate, 2),
        "Mean_D_True_Claims": round(float(np.mean(d_scores_true)), 3),
        "Mean_D_False_Claims": round(float(np.mean(d_scores_false)), 3),
        "Mean_D_Unverifiable_Claims": round(float(np.mean(d_scores_unverifiable)), 3),
        "Grounding_Separation_Gap": round(float(np.mean(d_scores_true) - np.mean(d_scores_unverifiable)), 3)
    })

df_verification_impact = pd.DataFrame(backend_metrics)
print("\n=== Verification Accuracy & Grounding Score Separation across Backends ===")
print(df_verification_impact)

# Save Comparison CSVs
df_mfaq_retrieval.to_csv("hybrid_retrieval_comparison.csv", index=False)
df_verification_impact.to_csv("hybrid_verification_impact.csv", index=False)
print("\nSaved hybrid_retrieval_comparison.csv and hybrid_verification_impact.csv")

# Save LaTeX Table
with open("hybrid_retrieval_results_table.tex", "w", encoding="utf-8") as f:
    f.write(df_mfaq_retrieval.to_latex(index=False, caption="Retrieval Performance Comparison: Sparse BM25 vs Dense vs Reciprocal Rank Fusion (RRF) on MFAQ (250 FAQ Pairs)", label="tab:hybrid_retrieval"))
print("Saved hybrid_retrieval_results_table.tex")

# ============================================================================
# PART 3: Publication-Quality Visualizations
# ============================================================================
print("\n>>> Generating Publication-Quality Figures...")

# Figure 1: MFAQ Retrieval Metrics Comparison
fig, ax = plt.subplots(figsize=(10, 5.5))
metrics = ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR (x100)"]
x = np.arange(len(metrics))
width = 0.20

colors = ["#78909c", "#1976d2", "#f57c00", "#2e7d32"]

for idx, row in df_mfaq_retrieval.iterrows():
    vals = [row["Recall@1"], row["Recall@3"], row["Recall@5"], row["Recall@10"], row["MRR"] * 100.0]
    rects = ax.bar(x + idx * width - (1.5 * width), vals, width, label=row["Method"], color=colors[idx], edgecolor="black", alpha=0.9)
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, weight="bold")

ax.set_ylabel("Score (%)", fontsize=12, weight="bold")
ax.set_title("Retrieval Performance Comparison: Sparse BM25 vs Dense Vector vs Hybrid RRF\n(clips/mfaq Real-World FAQ Benchmark, N=250 Pairs)", fontsize=13, weight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, weight="bold")
ax.set_ylim(50, 108)
ax.legend(loc="upper left", frameon=True, fontsize=10)
plt.tight_layout()

retrieval_plot_path = os.path.join(FIG_DIR, "hybrid_rrf_retrieval_comparison.png")
plt.savefig(retrieval_plot_path, dpi=300)
plt.close()
print(f"Saved {retrieval_plot_path}")

# Figure 2: Impact of Retrieval Backend on Academic Verification
fig, ax = plt.subplots(figsize=(8, 4.8))
methods_plot = df_verification_impact["Retrieval_Backend"].tolist()
accs = df_verification_impact["Overall_Accuracy"].tolist()
gaps = [g * 100.0 for g in df_verification_impact["Grounding_Separation_Gap"].tolist()]

x_bar = np.arange(len(methods_plot))
w_bar = 0.35

b1 = ax.bar(x_bar - w_bar / 2, accs, w_bar, label="Verification Accuracy (%)", color="#1565c0", edgecolor="black")
b2 = ax.bar(x_bar + w_bar / 2, gaps, w_bar, label="True vs Unverifiable Gap (D x100)", color="#2e7d32", edgecolor="black")

for b in b1:
    h = b.get_height()
    ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, weight="bold")
for b in b2:
    h = b.get_height()
    ax.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, weight="bold")

ax.set_ylabel("Percentage / Scaled Score", fontsize=11, weight="bold")
ax.set_title("Academic Verification Performance Across Retrieval Backends (N=150 Claims)", fontsize=12, weight="bold")
ax.set_xticks(x_bar)
ax.set_xticklabels(methods_plot, fontsize=11, weight="bold")
ax.set_ylim(0, 115)
ax.legend(loc="upper left", frameon=True)
plt.tight_layout()

verif_plot_path = os.path.join(FIG_DIR, "hybrid_rrf_verification_impact.png")
plt.savefig(verif_plot_path, dpi=300)
plt.close()
print(f"Saved {verif_plot_path}")

print("\n" + "=" * 80)
print("HYBRID BM25 + RECIPROCAL RANK FUSION (RRF) BENCHMARK COMPLETE!")
print("=" * 80)
