"""Wrap the team's real agents so the pipeline can call them uniformly.

Your agent can be sync or async, and can return a dict, a (score, ...) tuple, a bare
float, or your own object: edit `default_mapper` (or pass a custom `to_result`) once you
know the real return shape.
"""
import asyncio
import inspect
from typing import Any, Callable

from app.agents.base import VerificationAgent
from app.schemas import AgentResult, Claim, Verdict


def default_mapper(name: str, raw: Any) -> AgentResult:
    # TODO: adjust to the real agents' output. Assumes a dict like
    # {"verdict": "supported|contradicted|unverifiable", "score": 0.9, "evidence": [...], "reason": "..."}
    if isinstance(raw, AgentResult):
        return raw
    if isinstance(raw, (int, float)):
        return AgentResult(agent=name, verdict=Verdict.SUPPORTED if raw >= 0.5 else Verdict.CONTRADICTED,
                           score=float(raw))
    return AgentResult(agent=name, verdict=Verdict(raw["verdict"]), score=raw.get("score"),
                       evidence=raw.get("evidence", []), reason=raw.get("reason", ""))


class AgentAdapter(VerificationAgent):
    def __init__(self, name: str, fn: Callable[[str], Any],
                 to_result: Callable[[str, Any], AgentResult] = default_mapper):
        self.name, self.fn, self.to_result = name, fn, to_result

    async def verify(self, claim: Claim) -> AgentResult:
        if inspect.iscoroutinefunction(self.fn):
            raw = await self.fn(claim.text)
        else:
            raw = await asyncio.to_thread(self.fn, claim.text)
        return self.to_result(self.name, raw)
