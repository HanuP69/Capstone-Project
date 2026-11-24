import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_query_regenerates_and_blocks_wrong_claims():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = (await c.post("/api/v1/query", json={"query": "when are mids?"})).json()
        assert r["iterations"] == 2                      # first draft blocked, regenerated
        assert "15 October" in r["final_answer"] and "60%" not in r["final_answer"]
        got = await c.get(f"/api/v1/queries/{r['id']}")
        assert got.status_code == 200


@pytest.mark.asyncio
async def test_verify_flags_contradictions():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = (await c.post("/api/v1/verify", json={"text": "Mid-semester examinations start on 20 October."})).json()
        assert r["claims"][0]["decision"] == "block"
        assert r["claims"][0]["fact"]["verdict"] == "contradicted"
