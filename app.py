import json
import os
import threading
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config import DATA, RESULTS, ROOT
from pipeline import Pipeline

app = FastAPI(title="NITJ Groundedness Verification", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_lock = threading.Lock()
_pipeline = None


class ClaimIn(BaseModel):
    claim: str = Field(..., min_length=8, max_length=1200)
    w_fact: float = Field(0.40, ge=0, le=1)
    w_policy: float = Field(0.25, ge=0, le=1)
    w_ground: float = Field(0.35, ge=0, le=1)


class QueryIn(BaseModel):
    query: str = Field(..., min_length=4, max_length=600)


def get_pipeline():
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = Pipeline()
    return _pipeline


def read_csv(name):
    path = RESULTS / name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"{name} has not been generated")
    return pd.read_csv(path).to_dict(orient="records")


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"{path.name} has not been generated")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def dashboard():
    path = ROOT / "dashboard.html"
    if not path.exists():
        return "<h1>NITJ verification</h1><p>Run make_docs.py to generate the dashboard.</p>"
    return path.read_text(encoding="utf-8")


@app.post("/verify-claim")
def verify_claim(req: ClaimIn):
    total = req.w_fact + req.w_policy + req.w_ground
    if total <= 0:
        raise HTTPException(status_code=400, detail="weights must sum to a positive number")
    weights = {"w_fact": req.w_fact / total, "w_policy": req.w_policy / total, "w_ground": req.w_ground / total}
    res = get_pipeline().verify(req.claim, weights=weights)
    payload = asdict(res)
    payload["weights"] = weights
    return payload


@app.post("/ask")
def ask(req: QueryIn):
    res = get_pipeline().answer(req.query)
    payload = asdict(res)
    payload["claims"] = [asdict(c) for c in res.claims]
    return payload


@app.get("/api/metrics")
def metrics():
    return read_csv("condition_metrics.csv")


@app.get("/api/stats")
def stats():
    return read_json(RESULTS / "stats.json")


@app.get("/api/unsupervised")
def unsup():
    return read_csv("unsupervised_metrics.csv")


@app.get("/api/corpus")
def corpus():
    return {
        "corpus": read_json(DATA / "corpus_stats.json"),
        "index": read_json(DATA / "index_manifest.json"),
        "claims": read_json(DATA / "split.json"),
    }


@app.get("/api/records")
def records():
    path = DATA / "records.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="records.json missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _claim_to_api(c, idx):
    ctxs = []
    for e in (c.evidence or []):
        if isinstance(e, dict):
            ctxs.append(f"[{e.get('doc_name', '')} | {e.get('date', '')}] {e.get('text', '')}")
        else:
            ctxs.append(str(e))
    if not ctxs and c.fact_note:
        ctxs.append(c.fact_note)
    return {
        "claim_id": f"claim-{idx}", "claim_text": c.claim, "category": c.category,
        "F_score": c.F, "P_score": c.P, "D_score": c.D, "G_score": c.G,
        "decision": c.decision, "predicted_label": c.label,
        "fact_explanation": c.fact_note, "policy_explanation": c.policy_note,
        "retrieved_contexts": ctxs, "regenerated_claim": (c.regen or None),
        "weighting_mode": "meta_learner_mle", "retrieval_method": "hybrid_rrf",
        "is_synthetic_benchmark": False,
    }


@app.post("/api/verify-query")
def verify_query(req: QueryIn):
    try:
        res = get_pipeline().answer(req.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    claims = [_claim_to_api(c, i) for i, c in enumerate(res.claims)]
    g_scores = [c["G_score"] for c in claims]
    overall = (sum(g_scores) / len(g_scores)) if g_scores else 0.0
    return {
        "query": res.query, "draft_response": res.draft,
        "final_grounded_answer": res.answer, "overall_groundedness": round(overall, 4),
        "total_claims": len(claims),
        "passed_claims": sum(1 for c in claims if c["decision"] == "PASS"),
        "flagged_claims": sum(1 for c in claims if c["decision"] == "FLAG"),
        "blocked_claims": sum(1 for c in claims if c["decision"] == "BLOCK"),
        "latency_seconds": res.latency, "claims": claims,
        "dataset_mode": "Live pipeline — NIT Jamshedpur circulars",
    }
