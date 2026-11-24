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

# --- tuning constants (env-overridable) ---
LLM_NUM_PREDICT = int(os.environ.get("LLM_NUM_PREDICT", "220"))
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "4096"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
REGEN_ROUNDS = int(os.environ.get("REGEN_ROUNDS", "2"))
REGEN_TEMPERATURE = float(os.environ.get("REGEN_TEMPERATURE", "0.2"))
REGEN_NUM_PREDICT = int(os.environ.get("REGEN_NUM_PREDICT", "120"))
BEST_SENT_MIN_WORDS = int(os.environ.get("BEST_SENT_MIN_WORDS", "6"))
ATOMIC_MIN_PART_WORDS = int(os.environ.get("ATOMIC_MIN_PART_WORDS", "6"))
NUMERIC_TOL_MONEY = float(os.environ.get("NUMERIC_TOL_MONEY", "0.02"))
NUMERIC_TOL_PCT = float(os.environ.get("NUMERIC_TOL_PCT", "0.1"))
OVERLAP_GATE = float(os.environ.get("OVERLAP_GATE", "0.5"))
EVIDENCE_CHAR_LIMIT = int(os.environ.get("EVIDENCE_CHAR_LIMIT", "1800"))
LLM_CTX_CHAR = int(os.environ.get("LLM_CTX_CHAR", "700"))
RERANK_CHAR_LIMIT = int(os.environ.get("RERANK_CHAR_LIMIT", "1500"))
ANSWER_CTX_CHAR = int(os.environ.get("ANSWER_CTX_CHAR", "1200"))
EVIDENCE_SNIPPET_CHAR = int(os.environ.get("EVIDENCE_SNIPPET_CHAR", "300"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "32"))
EMBED_CHAR_LIMIT = int(os.environ.get("EMBED_CHAR_LIMIT", "6000"))
CHROMA_ADD_BATCH = int(os.environ.get("CHROMA_ADD_BATCH", "400"))
RECENCY_DEFAULT_AGE = float(os.environ.get("RECENCY_DEFAULT_AGE", "3.0"))
CONTEXT_DOCS = int(os.environ.get("CONTEXT_DOCS", "4"))
CONTEXT_CHAR = int(os.environ.get("CONTEXT_CHAR", "900"))
UNSUP_SAMPLES = int(os.environ.get("UNSUP_SAMPLES", "3"))
UNSUP_NUM_PREDICT = int(os.environ.get("UNSUP_NUM_PREDICT", "200"))
UNSUP_EVIDENCE_CHAR = int(os.environ.get("UNSUP_EVIDENCE_CHAR", "1500"))
UNSUP_CTX_CHAR = int(os.environ.get("UNSUP_CTX_CHAR", "500"))
UNSUP_CLAIM_MIN_WORDS = int(os.environ.get("UNSUP_CLAIM_MIN_WORDS", "5"))
UNSUP_ENTAIL_THRESHOLD = float(os.environ.get("UNSUP_ENTAIL_THRESHOLD", "0.5"))
BASELINE_NUM_PREDICT = int(os.environ.get("BASELINE_NUM_PREDICT", "10"))
D_ONLY_PASS = float(os.environ.get("D_ONLY_PASS", "0.6"))
D_ONLY_FAIL = float(os.environ.get("D_ONLY_FAIL", "0.4"))
G_ONLY_PASS = float(os.environ.get("G_ONLY_PASS", "0.6"))
G_ONLY_FAIL = float(os.environ.get("G_ONLY_FAIL", "0.4"))
ANSWER_NUM_PREDICT = int(os.environ.get("ANSWER_NUM_PREDICT", "240"))
DRAFT_NUM_PREDICT = int(os.environ.get("DRAFT_NUM_PREDICT", "240"))
ABSENCE_ENTAIL_THRESHOLD = float(os.environ.get("ABSENCE_ENTAIL_THRESHOLD", "0.6"))
CALIBRATION_BINS = int(os.environ.get("CALIBRATION_BINS", "10"))
BOOTSTRAP_N = int(os.environ.get("BOOTSTRAP_N", "2000"))
CLAIM_SENT_MIN_WORDS = int(os.environ.get("CLAIM_SENT_MIN_WORDS", "12"))
INGEST_HEAD_LIMIT = int(os.environ.get("INGEST_HEAD_LIMIT", "6000"))
INGEST_REF_LIMIT = int(os.environ.get("INGEST_REF_LIMIT", "2500"))
INGEST_SESSION_LIMIT = int(os.environ.get("INGEST_SESSION_LIMIT", "3000"))
INGEST_CLASSIFY_LIMIT = int(os.environ.get("INGEST_CLASSIFY_LIMIT", "4000"))
CLAIM_SENT_MAX_WORDS = int(os.environ.get("CLAIM_SENT_MAX_WORDS", "42"))
MAX_ENTAIL_FOR_UNSUPPORTED = float(os.environ.get("MAX_ENTAIL_FOR_UNSUPPORTED", "0.35"))
WIDE_SENT_LIMIT = int(os.environ.get("WIDE_SENT_LIMIT", "8"))
JUDGE_LLM_NUM_PREDICT = int(os.environ.get("JUDGE_LLM_NUM_PREDICT", "8"))