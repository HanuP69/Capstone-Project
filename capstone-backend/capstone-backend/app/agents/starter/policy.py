import json
import re
from pathlib import Path

from app.agents.base import VerificationAgent
from app.schemas import AgentResult, Claim, Verdict
from app.textutils import tokens

RULES_PATH = Path(__file__).parent.parent.parent / "data" / "policies.json"

OPS = {
    "eq": lambda x, v: x == v,
    "le": lambda x, v: x <= v,
    "ge": lambda x, v: x >= v,
}


class PolicyComplianceAgent(VerificationAgent):
    """Deterministic rule base. A rule triggers when its keywords appear in the claim,
    then the first number in the claim is compared against the rule (eq / le / ge)."""

    name = "policy"

    def __init__(self, rules_path: Path = RULES_PATH):
        self.rules = json.loads(Path(rules_path).read_text())

    async def verify(self, claim: Claim) -> AgentResult:
        claim_toks = tokens(claim.text)
        nums = re.findall(r"\d+(?:\.\d+)?", claim.text)
        for rule in self.rules:
            if len(claim_toks & set(rule["keywords"])) < rule.get("min_keywords", 1):
                continue
            if not nums:
                continue
            ok = OPS[rule["op"]](float(nums[0]), float(rule["value"]))
            evidence = [f'{rule["id"]}: {rule["description"]}']
            if ok:
                return AgentResult(agent=self.name, verdict=Verdict.SUPPORTED, score=1.0, evidence=evidence)
            return AgentResult(agent=self.name, verdict=Verdict.CONTRADICTED, score=0.0, evidence=evidence,
                               reason=f'Violates rule {rule["id"]}: {rule["description"]}')
        return AgentResult(agent=self.name, verdict=Verdict.UNVERIFIABLE, score=None,
                           reason="No policy rule applies")
