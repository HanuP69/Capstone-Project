import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DOCS = ROOT / "documents"
CACHE = ROOT / "cache"

for p in (DATA, RESULTS, CACHE):
    p.mkdir(exist_ok=True)

GEN_MODEL = os.environ.get("GEN_MODEL", "gemma4:e4b")
EMB_MODEL = os.environ.get("EMB_MODEL", "nomic-embed-text:latest")
NLI_MODEL = os.environ.get("NLI_MODEL", "cross-encoder/nli-MiniLM2-L6-H768")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

SEED = int(os.environ.get("SEED", "42"))
PDF_TARGET = int(os.environ.get("PDF_TARGET", "150"))
CHUNK_WORDS = int(os.environ.get("CHUNK_WORDS", "240"))
CHUNK_OVERLAP_SENTENCES = int(os.environ.get("CHUNK_OVERLAP_SENTENCES", "1"))
TABLE_MAX_WORDS = int(os.environ.get("TABLE_MAX_WORDS", "1100"))
MIN_CHUNK_WORDS = int(os.environ.get("MIN_CHUNK_WORDS", "35"))

DENSE_WEIGHT = float(os.environ.get("DENSE_WEIGHT", "1.6"))
SPARSE_WEIGHT = float(os.environ.get("SPARSE_WEIGHT", "1.0"))
RRF_K = int(os.environ.get("RRF_K", "20"))
RECENCY_LAMBDA = float(os.environ.get("RECENCY_LAMBDA", "0.35"))
RETRIEVE_K = int(os.environ.get("RETRIEVE_K", "12"))
RERANK_TOP = int(os.environ.get("RERANK_TOP", "6"))
JUDGE_TOP = int(os.environ.get("JUDGE_TOP", "3"))

DEFAULT_WEIGHTS = {"w_fact": 0.40, "w_policy": 0.25, "w_ground": 0.35}
TAU_PASS = float(os.environ.get("TAU_PASS", "0.62"))
TAU_BLOCK = float(os.environ.get("TAU_BLOCK", "0.40"))
FACT_BLOCK = float(os.environ.get("FACT_BLOCK", "0.05"))
FACT_PASS = float(os.environ.get("FACT_PASS", "0.95"))
GROUND_PENALTY = float(os.environ.get("GROUND_PENALTY", "0.35"))

SPLIT_DEV = float(os.environ.get("SPLIT_DEV", "0.5"))
CLAIMS_PER_LABEL = int(os.environ.get("CLAIMS_PER_LABEL", "60"))