from abc import ABC, abstractmethod

from app.schemas import AgentResult, Claim


class VerificationAgent(ABC):
    """Every verifier takes one atomic claim and returns a normalised AgentResult."""

    name: str

    @abstractmethod
    async def verify(self, claim: Claim) -> AgentResult: ...
