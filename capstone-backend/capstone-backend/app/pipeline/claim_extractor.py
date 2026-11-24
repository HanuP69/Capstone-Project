import re

from app.schemas import Claim


class ClaimExtractor:
    """PLACEHOLDER: sentence splitter. Replace `extract` with an LLM call that returns
    atomic, self-contained claims (resolve pronouns, one fact per claim)."""

    def extract(self, text: str) -> list[Claim]:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 3]
        return [Claim(id=i, text=p) for i, p in enumerate(parts)]
