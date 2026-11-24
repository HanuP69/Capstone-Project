from fastapi import APIRouter, HTTPException

from app.deps import pipeline, store
from app.schemas import QueryRequest, QueryResponse, VerifyRequest, VerifyResponse

router = APIRouter(prefix="/api/v1")


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Full pipeline: generate -> extract -> verify -> score -> gate -> (regenerate)."""
    result = await pipeline.run(req.query)
    await store.save(result)
    return result


@router.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest):
    """Verify arbitrary text (no generation). Used to score baseline outputs and datasets."""
    return VerifyResponse(claims=await pipeline.verify_text(req.text))


@router.get("/queries", response_model=list[QueryResponse])
async def list_queries(limit: int = 50):
    return await store.list(limit)


@router.get("/queries/{query_id}", response_model=QueryResponse)
async def get_query(query_id: str):
    r = await store.get(query_id)
    if not r:
        raise HTTPException(404, "Query not found")
    return r
