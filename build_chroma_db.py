"""
Build Persistent ChromaDB Vector Database for NIT Jamshedpur Documents
======================================================================
Extracts high-yield structured tables and prose chunks from SQLite `nitjsr_documents.db`,
computes/loads 768-dim embeddings via `nomic-embed-text`, and indexes them into
a persistent ChromaDB collection (`./nitjsr_chroma_db`) with cosine distance metric
and rich institutional metadata filtering.
"""

import os
import sqlite3
import time
import chromadb
from chromadb.config import Settings
import numpy as np

from verification_pipeline import LLMClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "nitjsr_documents.db")
CHROMA_DIR = os.path.join(BASE_DIR, "nitjsr_chroma_db")

print("=" * 80)
print("BUILDING PERSISTENT CHROMADB VECTOR DATABASE FOR NIT JAMSHEDPUR")
print("=" * 80)

# 1. Load chunks from SQLite
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
SELECT chunk_id, document_name, category, issue_date, academic_session, ref_number, page_number, chunk_type, content
FROM document_chunks
""")
all_rows = cursor.fetchall()
conn.close()

print(f"Loaded {len(all_rows)} total chunks from SQLite database.")

corpus_docs = []
for r in all_rows:
    corpus_docs.append({
        "id": r[0],
        "title": f"{r[1]} (Ref: {r[5]}, Date: {r[3]}, Category: {r[2]})",
        "doc_name": r[1],
        "category": r[2],
        "issue_date": str(r[3] or "Unknown"),
        "academic_session": str(r[4] or "Unknown"),
        "ref_number": str(r[5] or "Unknown"),
        "page_number": int(r[6] or 1),
        "chunk_type": str(r[7] or "prose"),
        "text": r[8]
    })

# Sample high-yield chunks: tables, hostel, fee, exam, circulars, syllabus
table_chunks = [d for d in corpus_docs if d["chunk_type"] == "table"]
hostel_chunks = [d for d in corpus_docs if d["category"] == "Hostel Rules"]
fee_chunks = [d for d in corpus_docs if d["category"] == "Fee Deadlines"][:400]
exam_chunks = [d for d in corpus_docs if d["category"] == "Exam Dates"][:400]
circ_chunks = [d for d in corpus_docs if "dean" in d["doc_name"].lower() or "academic" in d["doc_name"].lower()][:400]
syll_chunks = [d for d in corpus_docs if d["category"] == "Syllabus Queries"][:300]

sample_corpus = list({d["id"]: d for d in (table_chunks[:500] + hostel_chunks + fee_chunks + exam_chunks + circ_chunks + syll_chunks)}.values())
print(f"Selected {len(sample_corpus)} high-yield core institutional chunks for ChromaDB index.")

# 2. Compute/Load Embeddings
llm = LLMClient()
doc_texts = [f"{d['title']}: {d['text']}" for d in sample_corpus]
t0 = time.time()
embeddings = llm.embed_texts(doc_texts)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1.0
normalized_embs = embeddings / norms
t1 = time.time()
print(f"Vector embeddings ready: {normalized_embs.shape} in {t1 - t0:.2f}s.")

# 3. Initialize Persistent ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DIR)
# Using cosine space: cosine distance = 1 - cosine_similarity
collection = client.get_or_create_collection(
    name="nitjsr_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

# 4. Ingest Chunks into ChromaDB in batches
batch_size = 500
total_added = 0
for i in range(0, len(sample_corpus), batch_size):
    b_slice = sample_corpus[i:i + batch_size]
    b_ids = [d["id"] for d in b_slice]
    b_texts = [d["text"] for d in b_slice]
    b_embs = normalized_embs[i:i + batch_size].tolist()
    b_metas = [{
        "doc_name": d["doc_name"],
        "category": d["category"],
        "chunk_type": d["chunk_type"],
        "issue_date": d["issue_date"],
        "academic_session": d["academic_session"],
        "ref_number": d["ref_number"],
        "page_number": d["page_number"]
    } for d in b_slice]

    collection.upsert(
        ids=b_ids,
        documents=b_texts,
        embeddings=b_embs,
        metadatas=b_metas
    )
    total_added += len(b_ids)
    print(f"  Ingested batch {i // batch_size + 1}: {total_added}/{len(sample_corpus)} chunks into ChromaDB.")

count = collection.count()
print(f"\nChromaDB Collection 'nitjsr_knowledge_base' populated successfully with {count} chunks!")
print(f"Persistent Storage Path: {CHROMA_DIR}")

# 5. Sanity Check Test Query with Metadata Filtering
test_query = "What is the tuition fee for B.Tech semester and payment portal?"
q_emb = llm.embed_texts([test_query])
q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
q_norm[q_norm == 0] = 1.0
q_emb = (q_emb / q_norm).tolist()

results = collection.query(
    query_embeddings=q_emb,
    n_results=3,
    where={"category": "Fee Deadlines"}
)

print("\n--- ChromaDB Sanity Check Query Result ---")
print(f"Query: '{test_query}' (with where={{'category': 'Fee Deadlines'}})")
for rank in range(len(results["ids"][0])):
    cid = results["ids"][0][rank]
    meta = results["metadatas"][0][rank]
    dist = results["distances"][0][rank]
    sim = 1.0 - dist  # Cosine similarity = 1 - cosine distance
    doc_snippet = results["documents"][0][rank][:150].replace("\n", " ")
    print(f"Rank {rank + 1} [Sim: {sim:.4f}, Dist: {dist:.4f}]: ID={cid} | Doc={meta['doc_name']} | Type={meta['chunk_type']}")
    print(f"       Snippet: {doc_snippet}...")

print("\nChromaDB build complete and verified!")
