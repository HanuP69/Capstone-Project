import sqlite3

import pytest
from fastapi.testclient import TestClient

import app as api
from pipeline import AnswerResult, ClaimResult


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.result = ClaimResult(
            claim="The minimum attendance is 75 percent.", category="Circulars",
            F=1.0, P=1.0, D=0.93, C=0.0, G=0.97, decision="PASS", label="true",
            fact_note="attendance_min matches", policy_note="consistent",
            evidence=[{"doc_name": "notice.pdf", "chunk_id": "c1",
                       "date": "2025-01-01", "text": "The minimum attendance is 75 percent."}],
            regen="The minimum attendance is 75 percent.",
            judge_backend="nli",
        )
        self.answer_result = AnswerResult(
            query="attendance", draft="The minimum attendance is 75 percent.",
            answer="The minimum attendance is 75 percent.", claims=[self.result],
            latency=0.2, citations=["notice.pdf"], evaluated_claims=[self.result],
            retrieval_stats={"retrieve_calls": 1, "rerank_calls": 0},
        )

    def verify(self, claim, weights=None):
        self.calls.append(("verify", claim, weights))
        return self.result

    def answer(self, query):
        self.calls.append(("answer", query))
        return self.answer_result

    def health(self):
        return {"judge_backend": "nli", "records": 4, "reranker_enabled": True,
                "thresholds": {"tau_pass": 0.6, "tau_contra": 0.62}}


@pytest.fixture
def fake():
    return FakePipeline()


@pytest.fixture
def client(monkeypatch, tmp_path, fake):
    monkeypatch.setattr(api, "get_pipeline", lambda: fake)
    monkeypatch.setattr(api, "AUDIT_DB", tmp_path / "audit.sqlite")
    return TestClient(api.app)


def test_verify_claim_returns_contract_and_normalizes_weights(client, fake):
    response = client.post("/verify-claim", json={
        "claim": "The minimum attendance is 75 percent.",
        "w_fact": 0.8, "w_policy": 0.1, "w_ground": 0.1,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "PASS"
    assert body["label"] == "true"
    assert body["evidence"][0]["chunk_id"] == "c1"
    assert body["weights"] == pytest.approx({"w_fact": 0.8, "w_policy": 0.1, "w_ground": 0.1})
    assert fake.calls[0][2] == pytest.approx(body["weights"])


def test_verify_claim_rejects_zero_weights(client):
    response = client.post("/verify-claim", json={
        "claim": "The minimum attendance is 75 percent.",
        "w_fact": 0.0, "w_policy": 0.0, "w_ground": 0.0,
    })
    assert response.status_code == 400


def test_verify_claim_validates_short_claim(client):
    response = client.post("/verify-claim", json={"claim": "short"})
    assert response.status_code == 422


def test_verify_query_returns_counts_and_claim_details(client):
    response = client.post("/api/verify-query", json={
        "query": "What is the minimum attendance?",
    })
    assert response.status_code == 200
    body = response.json()
    assert len(body["request_id"]) == 32
    assert body["total_claims"] == 1
    assert body["passed_claims"] == 1
    assert body["flagged_claims"] == 0
    assert body["blocked_claims"] == 0
    assert body["overall_groundedness"] == 0.97
    assert body["claims"][0]["claim_id"] == "claim-0"
    assert body["claims"][0]["regenerated_claim"] == "The minimum attendance is 75 percent."
    assert body["claims"][0]["is_synthetic_benchmark"] is False
    assert body["weighting_mode"] == "fixed_weights"
    assert body["dataset_mode"].startswith("Live pipeline")


def test_verify_query_generates_unique_request_ids(client):
    first = client.post("/api/verify-query", json={"query": "attendance policy"}).json()
    second = client.post("/api/verify-query", json={"query": "attendance policy"}).json()
    assert first["request_id"] != second["request_id"]


def test_verify_query_surfaces_pipeline_failure(client, monkeypatch, fake):
    def fail(query):
        raise RuntimeError("generation unavailable")

    monkeypatch.setattr(fake, "answer", fail)
    response = client.post("/api/verify-query", json={"query": "attendance policy"})
    assert response.status_code == 500
    assert "generation unavailable" in response.json()["detail"]


def test_ask_returns_claims_and_evaluated_claims(client):
    response = client.post("/ask", json={"query": "attendance policy"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["claims"]) == 1
    assert len(body["evaluated_claims"]) == 1
    assert body["abstained"] is False


def test_health_reports_pipeline_state(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["pipeline"]["judge_backend"] == "nli"
    assert body["pipeline"]["records"] == 4


def test_artifact_endpoints_return_generated_data(client):
    metrics = client.get("/api/metrics")
    records = client.get("/api/records")
    corpus = client.get("/api/corpus")
    assert metrics.status_code == 200
    assert len(metrics.json()) == 4
    assert records.status_code == 200
    assert len(records.json()) == 4
    assert corpus.status_code == 200
    assert corpus.json()["claims"]["total"] == 233


def test_missing_artifact_returns_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "RESULTS", tmp_path)
    response = client.get("/api/metrics")
    assert response.status_code == 503


def test_audit_event_is_persisted(client, tmp_path):
    client.post("/verify-claim", json={"claim": "The minimum attendance is 75 percent."})
    connection = sqlite3.connect(tmp_path / "audit.sqlite")
    rows = connection.execute("select endpoint, request, response from audit").fetchall()
    connection.close()
    assert len(rows) == 1
    assert rows[0][0] == "verify-claim"
    assert "attendance" in rows[0][1]
