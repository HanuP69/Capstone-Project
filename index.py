import json
import pickle
import time

import chromadb
import numpy as np
import ollama
from rank_bm25 import BM25Okapi

from config import CHROMA_ADD_BATCH, DATA, EMB_MODEL, EMBED_BATCH, EMBED_CHAR_LIMIT, ROOT
from pipeline_text import tokenize

INDEX_DIR = ROOT / "index"
COLLECTION = "nitj_chunks"


def load_chunks():
    return [json.loads(l) for l in open(DATA / "chunks.jsonl", encoding="utf-8")]


def embed(texts, prefix):
    out = []
    batch = EMBED_BATCH
    for i in range(0, len(texts), batch):
        part = [f"{prefix}: {t[:EMBED_CHAR_LIMIT]}" for t in texts[i:i + batch]]
        res = ollama.embed(model=EMB_MODEL, input=part)
        out.extend(res["embeddings"])
        if (i // batch) % 10 == 0:
            print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    arr = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def main():
    chunks = load_chunks()
    docs = [c["text"] for c in chunks]
    t0 = time.time()
    embs = embed(docs, "search_document")
    print(f"embeddings {embs.shape} in {time.time() - t0:.1f}s", flush=True)

    INDEX_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    ids = [c["chunk_id"] for c in chunks]
    metas = [{
        "doc_name": c["doc_name"],
        "category": c["category"],
        "issue_date": c["issue_date"] or "",
        "ref_number": c["ref_number"] or "",
        "session": c["session"] or "",
        "chunk_type": c["chunk_type"],
        "page": int(c["page"]),
        "is_latest": int(c["is_latest"]),
        "n_words": int(c["n_words"]),
    } for c in chunks]
    for i in range(0, len(ids), CHROMA_ADD_BATCH):
        col.add(ids=ids[i:i + 400], embeddings=embs[i:i + 400].tolist(),
                documents=docs[i:i + 400], metadatas=metas[i:i + 400])
    print("chroma count", col.count(), flush=True)

    tokens = [tokenize(d) for d in docs]
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": BM25Okapi(tokens), "tokens": tokens, "ids": ids}, f)

    manifest = {
        "chunks": len(chunks),
        "dim": int(embs.shape[1]),
        "emb_model": EMB_MODEL,
        "collection": COLLECTION,
        "categories": {},
        "with_date": sum(1 for c in chunks if c["issue_date"]),
        "latest": sum(int(c["is_latest"]) for c in chunks),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }
    for c in chunks:
        manifest["categories"][c["category"]] = manifest["categories"].get(c["category"], 0) + 1
    with open(DATA / "index_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
