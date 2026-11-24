import asyncio

from app.agents.base import VerificationAgent
from app.config import settings
from app.pipeline import gate, scoring
from app.pipeline.claim_extractor import ClaimExtractor
from app.pipeline.generator import ResponseGenerator
from app.schemas import Claim, ClaimResult, Decision, QueryResponse


class Pipeline:
    def __init__(self, generator: ResponseGenerator, extractor: ClaimExtractor,
                 fact: VerificationAgent, policy: VerificationAgent, grounding: VerificationAgent):
        self.generator, self.extractor = generator, extractor
        self.fact, self.policy, self.grounding = fact, policy, grounding

    async def verify_claim(self, claim: Claim) -> ClaimResult:
        f, p, d = await asyncio.gather(
            self.fact.verify(claim), self.policy.verify(claim), self.grounding.verify(claim))
        g = scoring.groundedness(f, p, d)
        decision, reason = gate.decide(g, [f, p, d])
        return ClaimResult(claim=claim, fact=f, policy=p, grounding=d,
                           groundedness=g, decision=decision, failure_reason=reason)

    async def verify_text(self, text: str) -> list[ClaimResult]:
        claims = self.extractor.extract(text)
        return list(await asyncio.gather(*(self.verify_claim(c) for c in claims)))

    async def run(self, query: str) -> QueryResponse:
        feedback: list[str] | None = None
        for attempt in range(settings.max_regenerations + 1):
            draft = await self.generator.generate(query, feedback)
            results = await self.verify_text(draft)
            blocked = [r for r in results if r.decision == Decision.BLOCK]
            if not blocked or attempt == settings.max_regenerations:
                break
            feedback = [f'"{r.claim.text}" -> {r.failure_reason}' for r in blocked]

        # Final answer: keep passed/flagged claims only; blocked claims never reach the student.
        kept = [r.claim.text for r in results if r.decision != Decision.BLOCK]
        answer = " ".join(kept) or "I could not verify an answer to this; please contact the academic office."
        return QueryResponse(query=query, final_answer=answer, iterations=attempt + 1, claims=results)
