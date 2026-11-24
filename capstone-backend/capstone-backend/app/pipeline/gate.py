from app.config import settings
from app.schemas import AgentResult, Decision, Verdict


def decide(g: float, results: list[AgentResult]) -> tuple[Decision, str]:
    # Hard rule: a contradiction from official records / policy always blocks.
    hard = [r for r in results if r.agent in ("fact_check", "policy") and r.verdict == Verdict.CONTRADICTED]
    if hard:
        return Decision.BLOCK, "; ".join(r.reason for r in hard)
    if g >= settings.pass_threshold:
        return Decision.PASS, ""
    if g >= settings.flag_threshold:
        return Decision.FLAG, "Low confidence: weak supporting evidence"
    return Decision.BLOCK, "Ungrounded: no sufficient evidence"
