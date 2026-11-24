from abc import ABC, abstractmethod

from app.config import settings
from app.schemas import QueryResponse


class Store(ABC):
    @abstractmethod
    async def save(self, r: QueryResponse) -> None: ...
    @abstractmethod
    async def get(self, id: str) -> QueryResponse | None: ...
    @abstractmethod
    async def list(self, limit: int = 50) -> list[QueryResponse]: ...


class MemoryStore(Store):
    def __init__(self):
        self._d: dict[str, QueryResponse] = {}

    async def save(self, r): self._d[r.id] = r
    async def get(self, id): return self._d.get(id)
    async def list(self, limit=50): return list(self._d.values())[-limit:][::-1]


class MongoStore(Store):
    """Audit trail of every query, claim, per-agent result and score."""

    def __init__(self, uri: str, db: str):
        from motor.motor_asyncio import AsyncIOMotorClient
        self.col = AsyncIOMotorClient(uri)[db]["queries"]

    async def save(self, r):
        await self.col.insert_one({"_id": r.id, **r.model_dump(mode="json")})

    async def get(self, id):
        doc = await self.col.find_one({"_id": id})
        return QueryResponse(**doc) if doc else None

    async def list(self, limit=50):
        return [QueryResponse(**d) async for d in self.col.find().sort("created_at", -1).limit(limit)]


def make_store() -> Store:
    return MongoStore(settings.mongo_uri, settings.mongo_db) if settings.mongo_uri else MemoryStore()
