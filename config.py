import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DOCS = ROOT / "documents"
CACHE = ROOT / "cache"

for p in (DATA, RESULTS, CACHE):
    p.mkdir(exist_ok=True)

GEN_MODEL = "gemma4:e4b"
EMB_MODEL = "nomic-embed-text:latest"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SEED = 42
PDF_TARGET = 150
CHUNK_WORDS = 240
CHUNK_OVERLAP_SENTENCES = 1
TABLE_MAX_WORDS = 1100
MIN_CHUNK_WORDS = 35

DENSE_WEIGHT = 1.6
SPARSE_WEIGHT = 1.0
RRF_K = 20
RECENCY_LAMBDA = 0.35
RETRIEVE_K = 12
RERANK_TOP = 6
JUDGE_TOP = 3

DEFAULT_WEIGHTS = {"w_fact": 0.40, "w_policy": 0.25, "w_ground": 0.35}
TAU_PASS = 0.62
TAU_BLOCK = 0.40

SPLIT_DEV = 0.5
CLAIMS_PER_LABEL = 60
