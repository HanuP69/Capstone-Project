from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class Decision(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    BLOCK = "block"


class Claim(BaseModel):
    id: int
    text: str


class AgentResult(BaseModel):
    agent: str
    verdict: Verdict
    # Normalised [0,1]; None means "this agent has no opinion on this claim"
    score: Optional[float] = Field(default=None, ge=0, le=1)
    evidence: list[str] = []
    reason: str = ""


class ClaimResult(BaseModel):
    claim: Claim
    fact: AgentResult
    policy: AgentResult
    grounding: AgentResult
    groundedness: float
    decision: Decision
    failure_reason: str = ""


class VerifyRequest(BaseModel):
    text: str


class VerifyResponse(BaseModel):
    claims: list[ClaimResult]


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    query: str
    final_answer: str
    iterations: int
    claims: list[ClaimResult]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
