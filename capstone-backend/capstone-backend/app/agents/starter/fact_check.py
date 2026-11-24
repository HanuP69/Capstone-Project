import json
from pathlib import Path

from app.agents.base import VerificationAgent
from app.schemas import AgentResult, Claim, Verdict
from app.textutils import fact_tokens, tokens

RECORDS_PATH = Path(__file__).parent.parent.parent / "data" / "records.json"


class FactCheckAgent(VerificationAgent):
    """Compares a claim's checkable values (dates/numbers) with official records.

    Records come from a JSON file now; swap `_load` for a Mongo collection later.
    """

    name = "fact_check"

    def __init__(self, records_path: Path = RECORDS_PATH, min_overlap: int = 2):
        self.records = json.loads(Path(records_path).read_text())
        self.min_overlap = min_overlap

    async def verify(self, claim: Claim) -> AgentResult:
        claim_toks = tokens(claim.text)
        best, best_overlap = None, 0
        for rec in self.records:
            overlap = len(claim_toks & set(rec["keywords"]))
            if overlap > best_overlap:
                best, best_overlap = rec, overlap

        if best is None or best_overlap < self.min_overlap:
            return AgentResult(agent=self.name, verdict=Verdict.UNVERIFIABLE, score=None,
                               reason="No matching official record")

        claim_vals, rec_vals = fact_tokens(claim.text), fact_tokens(best["statement"])
        evidence = [f'{best["id"]}: {best["statement"]}']
        if not claim_vals:
            return AgentResult(agent=self.name, verdict=Verdict.UNVERIFIABLE, score=None,
                               evidence=evidence, reason="Claim has no checkable date/number")
        if claim_vals <= rec_vals:
            return AgentResult(agent=self.name, verdict=Verdict.SUPPORTED, score=1.0, evidence=evidence)
        return AgentResult(agent=self.name, verdict=Verdict.CONTRADICTED, score=0.0, evidence=evidence,
                           reason=f"Claim says {sorted(claim_vals - rec_vals)}, record says {sorted(rec_vals)}")
