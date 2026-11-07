"""
FastAPI Backend Microservice for Academic Groundedness Verification
=================================================================
Per Capstone Proposal Section 9:
"Backend: FastAPI, MongoDB (claim logs, scores, audit trail)"
Primary responsibility: Ankit Sharma (Backend API & Verification Agents)

NOTE: Operating on [Synthetic Benchmark] datasets derived from institutional rules.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from verification_pipeline import (
    AcademicVerificationPipeline,
    DEFAULT_WEIGHTS, LEARNED_META_WEIGHTS,
    SYNTHETIC_NITJ_DOCS, SYNTHETIC_NITJ_RECORDS, SYNTHETIC_NITJ_POLICIES,
    SYNTHETIC_PUBLIC_DOCS, SYNTHETIC_PUBLIC_RECORDS, SYNTHETIC_PUBLIC_POLICIES
)

app = FastAPI(
    title="Institutional Academic Assistant Verification API [Synthetic Benchmark]",
    description="Multi-Agent Groundedness Verification Pipeline (NIT Jamshedpur & Cross-Institution Benchmark)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipelines initialized in memory using synthetic benchmark schemas
pipelines = {
    "NITJ": AcademicVerificationPipeline(
        docs=SYNTHETIC_NITJ_DOCS,
        records=SYNTHETIC_NITJ_RECORDS,
        policies=SYNTHETIC_NITJ_POLICIES
    ),
    "PUBLIC": AcademicVerificationPipeline(
        docs=SYNTHETIC_PUBLIC_DOCS,
        records=SYNTHETIC_PUBLIC_RECORDS,
        policies=SYNTHETIC_PUBLIC_POLICIES
    )
}

AUDIT_LOGS_FILE = "api_audit_log.json"
audit_logs = []
if os.path.exists(AUDIT_LOGS_FILE):
    try:
        with open(AUDIT_LOGS_FILE, "r", encoding="utf-8") as f:
            audit_logs = json.load(f)
    except Exception:
        audit_logs = []

def save_audit(log_entry: Dict[str, Any]):
    audit_logs.append(log_entry)
    try:
        with open(AUDIT_LOGS_FILE, "w", encoding="utf-8") as f:
            json.dump(audit_logs[-500:], f, indent=2)
    except Exception:
        pass

# ============================================================================
# Schemas
# ============================================================================
class ClaimVerifyRequest(BaseModel):
    claim: str = Field(..., example="The semester registration deadline at NIT Jamshedpur is 15 August 2026.")
    institution: str = Field(default="NITJ", example="NITJ")
    use_meta_learner: bool = Field(default=False, description="Use MLE Logistic Regression derived weights (w1=0.508, w2=0.315, w3=0.177) instead of default heuristic")
    retrieval_mode: str = Field(default="hybrid", description="Retrieval backend: 'hybrid' (BM25 + Dense RRF), 'dense', or 'bm25'")

class ClaimVerifyResponse(BaseModel):
    claim_id: str
    claim_text: str
    category: str
    F_score: float
    P_score: float
    D_score: float
    G_score: float
    decision: str
    predicted_label: str
    fact_explanation: str
    policy_explanation: str
    retrieved_contexts: List[str]
    regenerated_claim: Optional[str] = None
    weighting_mode: str = "default_heuristic"
    retrieval_method: str = "hybrid_rrf"
    is_synthetic_benchmark: bool = True

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the semester registration deadline and tuition fee for B.Tech?")
    institution: str = Field(default="NITJ", example="NITJ")
    use_meta_learner: bool = Field(default=False, description="Use MLE Logistic Regression derived weights")
    retrieval_mode: str = Field(default="hybrid", description="Retrieval backend: 'hybrid', 'dense', or 'bm25'")

class QueryResponse(BaseModel):
    query: str
    draft_response: str
    final_grounded_answer: str
    overall_groundedness: float
    total_claims: int
    passed_claims: int
    flagged_claims: int
    blocked_claims: int
    latency_seconds: float
    claims: List[ClaimVerifyResponse]
    dataset_mode: str = "[Synthetic Benchmark]"

# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Institutional Groundedness Verification Engine",
        "dataset_mode": "[Synthetic Benchmark] (Derived from institutional ordinances)",
        "supported_institutions": ["NITJ", "PUBLIC"],
        "endpoints": {
            "verify_claim": "/api/verify-claim",
            "verify_query": "/api/verify-query",
            "audit_trail": "/api/audit-trail",
            "metrics": "/api/metrics",
            "meta_learner": "/api/meta-learner",
            "retrieval_comparison": "/api/retrieval-comparison",
            "human_eval": "/api/human-eval"
        }
    }

@app.post("/api/verify-claim", response_model=ClaimVerifyResponse)
def verify_claim(req: ClaimVerifyRequest):
    inst = req.institution.upper()
    if inst not in pipelines:
        raise HTTPException(status_code=400, detail=f"Institution '{req.institution}' not supported. Use 'NITJ' or 'PUBLIC'.")
    pipeline = pipelines[inst]

    orig_weights = pipeline.scorer.weights
    pipeline.scorer.weights = LEARNED_META_WEIGHTS if req.use_meta_learner else DEFAULT_WEIGHTS

    try:
        result = pipeline.verify_single_claim(req.claim, retrieval_mode=req.retrieval_mode)
    finally:
        pipeline.scorer.weights = orig_weights

    log_item = {
        "timestamp": time.time(),
        "type": "claim",
        "institution": inst,
        "weighting_mode": "meta_learner_mle" if req.use_meta_learner else "default_heuristic",
        "retrieval_mode": req.retrieval_mode,
        "claim": req.claim,
        "F": result.F_score,
        "P": result.P_score,
        "D": result.D_score,
        "G": result.G_score,
        "decision": result.decision,
        "pred": result.predicted_label,
        "dataset_mode": "[Synthetic Benchmark]"
    }
    save_audit(log_item)

    return ClaimVerifyResponse(
        claim_id=result.claim_id,
        claim_text=result.claim_text,
        category=result.category,
        F_score=result.F_score,
        P_score=result.P_score,
        D_score=result.D_score,
        G_score=result.G_score,
        decision=result.decision,
        predicted_label=result.predicted_label,
        fact_explanation=result.fact_explanation,
        policy_explanation=result.policy_explanation,
        retrieved_contexts=result.retrieved_contexts,
        regenerated_claim=result.regenerated_claim,
        weighting_mode="meta_learner_mle" if req.use_meta_learner else "default_heuristic",
        retrieval_method=result.retrieval_method,
        is_synthetic_benchmark=True
    )

@app.post("/api/verify-query", response_model=QueryResponse)
def verify_query(req: QueryRequest):
    inst = req.institution.upper()
    if inst not in pipelines:
        raise HTTPException(status_code=400, detail=f"Institution '{req.institution}' not supported.")
    pipeline = pipelines[inst]

    orig_weights = pipeline.scorer.weights
    pipeline.scorer.weights = LEARNED_META_WEIGHTS if req.use_meta_learner else DEFAULT_WEIGHTS

    try:
        resp = pipeline.process_query(req.query)
    finally:
        pipeline.scorer.weights = orig_weights

    log_item = {
        "timestamp": time.time(),
        "type": "query",
        "institution": inst,
        "weighting_mode": "meta_learner_mle" if req.use_meta_learner else "default_heuristic",
        "query": req.query,
        "overall_G": resp.overall_groundedness,
        "claims_total": resp.total_claims,
        "passed": resp.passed_claims,
        "blocked": resp.blocked_claims,
        "latency": resp.latency_seconds,
        "dataset_mode": "[Synthetic Benchmark]"
    }
    save_audit(log_item)

    claim_responses = [
        ClaimVerifyResponse(
            claim_id=c.claim_id,
            claim_text=c.claim_text,
            category=c.category,
            F_score=c.F_score,
            P_score=c.P_score,
            D_score=c.D_score,
            G_score=c.G_score,
            decision=c.decision,
            predicted_label=c.predicted_label,
            fact_explanation=c.fact_explanation,
            policy_explanation=c.policy_explanation,
            retrieved_contexts=c.retrieved_contexts,
            regenerated_claim=c.regenerated_claim,
            weighting_mode="meta_learner_mle" if req.use_meta_learner else "default_heuristic",
            is_synthetic_benchmark=True
        ) for c in resp.claims
    ]

    return QueryResponse(
        query=resp.query,
        draft_response=resp.draft_response,
        final_grounded_answer=resp.final_grounded_answer,
        overall_groundedness=resp.overall_groundedness,
        total_claims=resp.total_claims,
        passed_claims=resp.passed_claims,
        flagged_claims=resp.flagged_claims,
        blocked_claims=resp.blocked_claims,
        latency_seconds=resp.latency_seconds,
        claims=claim_responses,
        dataset_mode="[Synthetic Benchmark]"
    )

@app.get("/api/meta-learner")
def get_meta_learner():
    results = {
        "status": "calibrated",
        "method": "Logistic Regression Maximum Likelihood Estimation (MLE)",
        "mathematical_formulation": "logit(P(Grounded)) = beta_0 + beta_F * F(c) + beta_P * P(c) + beta_D * D(c)",
        "derived_weights": LEARNED_META_WEIGHTS,
        "default_heuristic_weights": DEFAULT_WEIGHTS,
        "raw_coefficients": {
            "beta_0_intercept": -4.337,
            "beta_F_fact": 3.091,
            "beta_P_policy": 1.914,
            "beta_D_grounding": 1.074
        },
        "odds_ratios": {
            "exp_beta_F": 22.0,
            "exp_beta_P": 6.78,
            "exp_beta_D": 2.93
        },
        "relative_importance_percentages": {
            "w1_fact": 50.8,
            "w2_policy": 31.5,
            "w3_grounding": 17.7
        },
        "cross_validation_5fold": {
            "cv_accuracy": "93.3% (+/- 4.2%)",
            "cv_roc_auc": 0.926,
            "multinomial_3class_cv_accuracy": "89.2% (+/- 6.2%)"
        },
        "dataset_mode": "[Synthetic Benchmark]"
    }
    if os.path.exists("meta_learner_weights_comparison.csv"):
        df = pd.read_csv("meta_learner_weights_comparison.csv")
        results["comparison_table"] = df.to_dict(orient="records")
    return results

@app.get("/api/retrieval-comparison")
def get_retrieval_comparison():
    resp = {
        "status": "calibrated",
        "methods": [
            "1. Sparse Lexical Search (BM25Okapi)",
            "2. Dense Semantic Vector Search (nomic-embed-text + FAISS)",
            "3. Hybrid Reciprocal Rank Fusion (RRF)"
        ],
        "rrf_parameters": {
            "standard_k": 60,
            "calibrated_k": 20,
            "calibrated_weights": {"w_dense": 2.0, "w_bm25": 1.0}
        },
        "formula": "RRF(d) = w_dense / (k + rank_dense(d)) + w_bm25 / (k + rank_bm25(d))",
        "dataset_mode": "[Synthetic Benchmark]"
    }
    if os.path.exists("hybrid_retrieval_comparison.csv"):
        df_ret = pd.read_csv("hybrid_retrieval_comparison.csv")
        resp["mfaq_retrieval_benchmark"] = df_ret.to_dict(orient="records")
    return resp

@app.get("/api/audit-trail")
def get_audit_trail(limit: int = Query(50, ge=1, le=200)):
    if os.path.exists("primary_audit_trail.csv"):
        df = pd.read_csv("primary_audit_trail.csv")
        records = df.head(limit).to_dict(orient="records")
        return {"total_records": len(df), "returned": len(records), "dataset_mode": "[Synthetic Benchmark]", "data": records}
    return {"total_records": len(audit_logs), "dataset_mode": "[Synthetic Benchmark]", "data": audit_logs[-limit:]}

@app.get("/api/metrics")
def get_metrics():
    if os.path.exists("master_evaluation_metrics.csv"):
        df = pd.read_csv("master_evaluation_metrics.csv")
        return {"dataset_mode": "[Synthetic Benchmark]", "metrics": df.to_dict(orient="records")}
    return {"error": "Metrics table not found."}

@app.get("/api/human-eval")
def get_human_eval():
    if os.path.exists("human_eval_report.json"):
        with open("human_eval_report.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            data["dataset_mode"] = "[Synthetic Benchmark]"
            return data
    return {"error": "Human evaluation report not found."}
