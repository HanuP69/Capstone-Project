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
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(","),
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
