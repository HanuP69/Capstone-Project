import json
import os
import sqlite3
import threading
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config import CACHE, DATA, DEFAULT_WEIGHTS, RESULTS, ROOT
from pipeline import Pipeline

app = FastAPI(title="NITJ Groundedness Verification", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://127.0.0.1:5173,http://127.0.0.1:5176").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_lock = threading.Lock()
_audit_lock = threading.Lock()
_pipeline = None
AUDIT_DB = CACHE / "audit.sqlite"


class ClaimIn(BaseModel):
    claim: str = Field(..., min_length=8, max_length=1200)
    w_fact: float = Field(DEFAULT_WEIGHTS["w_fact"], ge=0, le=1)
    w_policy: float = Field(DEFAULT_WEIGHTS["w_policy"], ge=0, le=1)
    w_ground: float = Field(DEFAULT_WEIGHTS["w_ground"], ge=0, le=1)


class QueryIn(BaseModel):
    query: str = Field(..., min_length=4, max_length=600)


def get_pipeline():
    global _pipeline
    with _lock:
        if _pipeline is None:
            _pipeline = Pipeline()
    return _pipeline


def audit_event(endpoint, request, response):
    try:
        with _audit_lock:
            connection = sqlite3.connect(AUDIT_DB)
            connection.execute("create table if not exists audit (id integer primary key autoincrement, at text, endpoint text, request text, response text)")
            connection.execute("insert into audit (at, endpoint, request, response) values (?, ?, ?, ?)",
                               (datetime.now().isoformat(timespec="seconds"), endpoint,
                                json.dumps(request, ensure_ascii=False),
                                json.dumps(response, ensure_ascii=False)[:20000]))
            connection.commit()
            connection.close()
    except Exception:
        pass


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
    path = ROOT / "dashboard" / "dist" / "index.html"
    if not path.exists():
        path = ROOT / "dashboard.html"
    if not path.exists():
        return "<h1>NITJ verification</h1><p>Run make_docs.py and dashboard/build.py to generate the dashboard.</p>"
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
    audit_event("verify-claim", {"claim": req.claim, "weights": weights},
                {"decision": res.decision, "label": res.label, "F": res.F, "D": res.D})
    return payload


@app.post("/ask")
def ask(req: QueryIn):
    res = get_pipeline().answer(req.query)
    payload = asdict(res)
    payload["claims"] = [asdict(c) for c in res.claims]
    payload["evaluated_claims"] = [asdict(c) for c in res.evaluated_claims]
    audit_event("ask", {"query": req.query},
                {"answer": res.answer, "abstained": res.abstained, "claims": len(res.claims)})
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
        "weighting_mode": "fixed_weights", "retrieval_method": "hybrid_rrf",
        "is_synthetic_benchmark": False,
    }


@app.post("/api/verify-query")
def verify_query(req: QueryIn):
    try:
        res = get_pipeline().answer(req.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    source = res.evaluated_claims or res.claims
    claims = [_claim_to_api(c, i) for i, c in enumerate(source)]
    g_scores = [c["G_score"] for c in claims]
    overall = (sum(g_scores) / len(g_scores)) if g_scores else 0.0
    payload = {
        "request_id": uuid.uuid4().hex, "query": res.query, "draft_response": res.draft,
        "final_grounded_answer": res.answer, "overall_groundedness": round(overall, 4),
        "total_claims": len(claims),
        "passed_claims": sum(1 for c in claims if c["decision"] == "PASS"),
        "flagged_claims": sum(1 for c in claims if c["decision"] == "FLAG"),
        "blocked_claims": sum(1 for c in claims if c["decision"] == "BLOCK"),
        "latency_seconds": res.latency, "claims": claims,
        "weighting_mode": "fixed_weights",
        "retrieval_stats": res.retrieval_stats,
        "dataset_mode": "Live pipeline — NIT Jamshedpur circulars",
    }
    audit_event("verify-query", {"query": req.query},
                {"answer": res.answer, "claims": len(claims), "overall": payload["overall_groundedness"]})
    return payload


@app.get("/api/health")
def health():
    pipe = get_pipeline()
    return {"status": "ok", "pipeline": pipe.health()}
