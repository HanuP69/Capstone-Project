"""
Full MFAQ (Multilingual / Real-World FAQ) Retrieval & QA Answering Benchmark
============================================================================
Runs the full evaluation on the English MFAQ corpus (clips/mfaq) to stress-test:
1. RAG Dense Retrieval Layer (nomic-embed-text + FAISS)
   - Evaluates Recall@1, Recall@3, Recall@5, Recall@10, and MRR on 250 real FAQ pairs.
2. QA Generation Layer (Plain LLM vs RAG-Augmented Answer Generation)
   - Evaluates Token-F1 and Answer Grounding across 30 diverse FAQ items.
"""

import os
import re
import json
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import faiss
import ollama

sns.set_theme(style="whitegrid", font_scale=1.1)
FIG_DIR = "evaluation_results"
os.makedirs(FIG_DIR, exist_ok=True)
CACHE_DIR = "nb_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

GEN_MODEL = "gemma4:e4b"
EMB_MODEL = "nomic-embed-text:latest"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Cache setup
LLM_CACHE_PATH = os.path.join(CACHE_DIR, "llm_cache.json")
EMB_CACHE_PATH = os.path.join(CACHE_DIR, "emb_cache.json")

try:
    with open(LLM_CACHE_PATH, "r", encoding="utf-8") as f:
        llm_cache = json.load(f)
except Exception:
    llm_cache = {}

try:
    with open(EMB_CACHE_PATH, "r", encoding="utf-8") as f:
        emb_cache = json.load(f)
except Exception:
    emb_cache = {}

def save_cache():
    try:
        with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(llm_cache, f)
        with open(EMB_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(emb_cache, f)
    except Exception:
        pass

def embed_texts(texts, batch=16):
    vectors = [None] * len(texts)
    missing_idx = []
    missing_texts = []
    for i, t in enumerate(texts):
        k = t[:200]
        if k in emb_cache:
            vectors[i] = emb_cache[k]
        else:
            missing_idx.append(i)
            missing_texts.append(t)

    if missing_texts:
        for i in range(0, len(missing_texts), batch):
            batch_txts = missing_texts[i:i+batch]
            try:
                res = ollama.embed(model=EMB_MODEL, input=batch_txts)
                for b_i, emb in enumerate(res["embeddings"]):
                    idx = missing_idx[i + b_i]
                    txt = batch_txts[b_i]
                    vectors[idx] = emb
                    emb_cache[txt[:200]] = emb
            except Exception as e:
                print(f"Embed warning: {e}")
                dim = 768
                for b_i in range(len(batch_txts)):
                    idx = missing_idx[i + b_i]
                    vectors[idx] = [0.0] * dim
        save_cache()
    return np.array(vectors, dtype=np.float32)

def llm_generate(prompt, temperature=0.2, num_predict=90):
    k = f"{prompt[:150]}_{temperature}_{num_predict}"
    if k in llm_cache:
        return llm_cache[k]
    try:
        res = ollama.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": num_predict},
            think=False
        )
        content = res["message"]["content"].strip()
        llm_cache[k] = content
        save_cache()
        return content
    except Exception as e:
        return ""


# ============================================================================
# 1. Stream Full MFAQ Dataset (English Flat)
# ============================================================================
print(">>> Streaming real-world English FAQ pairs from MFAQ (clips/mfaq)...")
stream = load_dataset("clips/mfaq", "en_flat", split="train", streaming=True)

full_mfaq_samples = []
target_sample_size = 250

for i, row in enumerate(stream):
    q = row.get("question", "").strip()
    a = row.get("answer", "").strip()
    if len(q) > 15 and len(a) > 25:
        full_mfaq_samples.append({"id": f"MFAQ_{len(full_mfaq_samples):04d}", "question": q, "answer": a})
        if len(full_mfaq_samples) >= target_sample_size:
            break

print(f"Successfully collected full MFAQ benchmark sample: {len(full_mfaq_samples)} pairs.")

# ============================================================================
# 2. Build Dense Retrieval Corpus & FAISS Index
# ============================================================================
print(">>> Building FAQ Corpus and FAISS Index...")
mfaq_docs = []
seen_answers = set()
ans_to_doc = {}

for i, item in enumerate(full_mfaq_samples):
    ans_text = item["answer"]
    if ans_text not in seen_answers:
        seen_answers.add(ans_text)
        doc_id = len(mfaq_docs)
        mfaq_docs.append({"doc_id": doc_id, "text": ans_text})
        ans_to_doc[i] = doc_id
    else:
        # Match existing doc_id
        for d in mfaq_docs:
            if d["text"] == ans_text:
                ans_to_doc[i] = d["doc_id"]
                break

mfaq_corpus_df = pd.DataFrame(mfaq_docs)
print(f"Unique answer corpus size: {len(mfaq_corpus_df)} documents.")

# Embed corpus answers
corpus_texts = mfaq_corpus_df["text"].tolist()
print(f"Embedding {len(corpus_texts)} corpus passages using {EMB_MODEL}...")
t0_emb = time.time()
corpus_embeddings = embed_texts(corpus_texts)
# Normalize for Cosine Similarity (Inner Product)
norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1.0
norm_embeddings = corpus_embeddings / norms

d_dim = norm_embeddings.shape[1]
mfaq_index = faiss.IndexFlatIP(d_dim)
mfaq_index.add(norm_embeddings)
print(f"FAISS index built in {time.time() - t0_emb:.2f}s with {mfaq_index.ntotal} vectors.")

# ============================================================================
# 3. Dense Retrieval Stress-Test (Recall@1, Recall@3, Recall@5, Recall@10, MRR)
# ============================================================================
print(f"\n>>> Running Dense Retrieval Benchmark across all {len(full_mfaq_samples)} questions...")
questions = [item["question"] for item in full_mfaq_samples]
q_embeddings = embed_texts(questions)
q_norms = np.linalg.norm(q_embeddings, axis=1, keepdims=True)
q_norms[q_norms == 0] = 1.0
norm_q_embeddings = q_embeddings / q_norms

k_max = 10
scores, indices = mfaq_index.search(norm_q_embeddings, k_max)

recall_at_1 = 0
recall_at_3 = 0
recall_at_5 = 0
recall_at_10 = 0
mrr_list = []

for i in range(len(full_mfaq_samples)):
    gold_doc_id = ans_to_doc[i]
    retrieved_doc_ids = [mfaq_corpus_df.iloc[idx]["doc_id"] for idx in indices[i] if 0 <= idx < len(mfaq_corpus_df)]
    
    if gold_doc_id in retrieved_doc_ids[:1]:
        recall_at_1 += 1
    if gold_doc_id in retrieved_doc_ids[:3]:
        recall_at_3 += 1
    if gold_doc_id in retrieved_doc_ids[:5]:
        recall_at_5 += 1
    if gold_doc_id in retrieved_doc_ids[:10]:
        recall_at_10 += 1

    if gold_doc_id in retrieved_doc_ids:
        rank = retrieved_doc_ids.index(gold_doc_id) + 1
        mrr_list.append(1.0 / rank)
    else:
        mrr_list.append(0.0)

n_total = len(full_mfaq_samples)
r1 = (recall_at_1 / n_total) * 100
r3 = (recall_at_3 / n_total) * 100
r5 = (recall_at_5 / n_total) * 100
r10 = (recall_at_10 / n_total) * 100
mean_mrr = float(np.mean(mrr_list))

print(f"\n--- MFAQ Dense Retrieval Benchmark Results ({n_total} items) ---")
print(f"  Recall@1:  {r1:.1f}%")
print(f"  Recall@3:  {r3:.1f}%")
print(f"  Recall@5:  {r5:.1f}%")
print(f"  Recall@10: {r10:.1f}%")
print(f"  MRR:       {mean_mrr:.3f}")

# Plot MFAQ Retrieval Performance
fig, ax = plt.subplots(figsize=(7, 4.5))
metrics_names = ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR (x100)"]
metrics_vals = [r1, r3, r5, r10, mean_mrr * 100]
bars = ax.bar(metrics_names, metrics_vals, color=["#1f77b4", "#388e3c", "#f57c00", "#7b1fa2", "#0097a7"], width=0.55)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 110)
ax.set_title(f"MFAQ Dense Retrieval Layer Performance (N={n_total} FAQ Pairs)")

for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.1f}%",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, weight="bold")

plt.tight_layout()
mfaq_retrieval_plot = os.path.join(FIG_DIR, "mfaq_retrieval_performance.png")
plt.savefig(mfaq_retrieval_plot, dpi=300)
plt.close()
print(f"Saved {mfaq_retrieval_plot}")


# ============================================================================
# 4. QA Answering Benchmark: Plain LLM vs RAG-Augmented (Token-F1)
# ============================================================================
print("\n>>> Evaluating QA Answering Layer: Plain LLM vs RAG-Augmented on 30 MFAQ questions...")
def clean_tokens(s):
    return set(re.findall(r"\w+", s.lower()))

def calc_token_f1(pred, gold):
    p_tok = clean_tokens(pred)
    g_tok = clean_tokens(gold)
    if not p_tok or not g_tok:
        return 0.0
    common = len(p_tok & g_tok)
    if common == 0:
        return 0.0
    prec = common / len(p_tok)
    rec = common / len(g_tok)
    return (2 * prec * rec) / (prec + rec)

eval_subset = full_mfaq_samples[:30]
plain_f1s = []
rag_f1s = []
qa_records = []

for idx, item in enumerate(eval_subset):
    q = item["question"]
    gold = item["answer"]

    # Condition 1: Plain LLM (Zero context)
    prompt_plain = f"Answer the following question concisely:\nQuestion: {q}\nAnswer:"
    plain_ans = llm_generate(prompt_plain, temperature=0.2, num_predict=75)
    f1_plain = calc_token_f1(plain_ans, gold)
    plain_f1s.append(f1_plain)

    # Condition 2: RAG-Augmented (With top-3 retrieved context)
    top_indices = indices[idx][:3]
    retrieved_contexts = [mfaq_corpus_df.iloc[k]["text"] for k in top_indices if 0 <= k < len(mfaq_corpus_df)]
    context_str = "\n".join(retrieved_contexts)
    prompt_rag = (
        f"Context Information:\n{context_str}\n\n"
        f"Answer the following question accurately based on the context:\n"
        f"Question: {q}\nAnswer:"
    )
    rag_ans = llm_generate(prompt_rag, temperature=0.2, num_predict=75)
    f1_rag = calc_token_f1(rag_ans, gold)
    rag_f1s.append(f1_rag)

    qa_records.append({
        "question": q[:80],
        "gold_answer": gold[:80],
        "plain_answer": plain_ans[:80],
        "rag_answer": rag_ans[:80],
        "plain_f1": round(f1_plain, 3),
        "rag_f1": round(f1_rag, 3)
    })

mean_plain_f1 = float(np.mean(plain_f1s))
mean_rag_f1 = float(np.mean(rag_f1s))
improvement_pct = ((mean_rag_f1 - mean_plain_f1) / mean_plain_f1) * 100 if mean_plain_f1 > 0 else 0.0

print(f"\n--- MFAQ QA Answering Results (30 Evaluated Questions) ---")
print(f"  Plain LLM Token-F1: {mean_plain_f1:.3f}")
print(f"  RAG-Augmented Token-F1: {mean_rag_f1:.3f}")
print(f"  RAG Relative Token-F1 Improvement: +{improvement_pct:.1f}%")

# Plot QA Token-F1 Comparison
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(["Plain LLM", "RAG-Augmented"], [mean_plain_f1, mean_rag_f1], color=["#9aa0a6", "#1a73e8"], width=0.45)
ax.set_ylabel("Token-F1 Score")
ax.set_ylim(0, 0.70)
ax.set_title("MFAQ QA Answering: Plain LLM vs RAG-Augmented")

for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.3f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, weight="bold")

plt.tight_layout()
mfaq_qa_plot = os.path.join(FIG_DIR, "mfaq_qa_comparison.png")
plt.savefig(mfaq_qa_plot, dpi=300)
plt.close()
print(f"Saved {mfaq_qa_plot}")


# ============================================================================
# 5. Export MFAQ Benchmark Results Table
# ============================================================================
mfaq_summary_df = pd.DataFrame([
    {"Benchmark": "MFAQ Dense Retrieval", "Sample_Size": n_total, "Metric": "Recall@1", "Score": round(r1, 1), "Unit": "%"},
    {"Benchmark": "MFAQ Dense Retrieval", "Sample_Size": n_total, "Metric": "Recall@3", "Score": round(r3, 1), "Unit": "%"},
    {"Benchmark": "MFAQ Dense Retrieval", "Sample_Size": n_total, "Metric": "Recall@5", "Score": round(r5, 1), "Unit": "%"},
    {"Benchmark": "MFAQ Dense Retrieval", "Sample_Size": n_total, "Metric": "Recall@10", "Score": round(r10, 1), "Unit": "%"},
    {"Benchmark": "MFAQ Dense Retrieval", "Sample_Size": n_total, "Metric": "MRR", "Score": round(mean_mrr, 3), "Unit": "scale [0,1]"},
    {"Benchmark": "MFAQ QA Answering", "Sample_Size": 30, "Metric": "Plain LLM Token-F1", "Score": round(mean_plain_f1, 3), "Unit": "scale [0,1]"},
    {"Benchmark": "MFAQ QA Answering", "Sample_Size": 30, "Metric": "RAG Token-F1", "Score": round(mean_rag_f1, 3), "Unit": "scale [0,1]"}
])

mfaq_summary_df.to_csv("mfaq_benchmark_results.csv", index=False)
pd.DataFrame(qa_records).to_csv("mfaq_qa_eval_details.csv", index=False)
print("Saved mfaq_benchmark_results.csv and mfaq_qa_eval_details.csv")
print("\n=== Full MFAQ Benchmark Finished Successfully ===")
