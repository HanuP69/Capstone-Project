from app.config import settings
from app.schemas import AgentResult


def groundedness(fact: AgentResult, policy: AgentResult, grounding: AgentResult,
                 w: tuple[float, float, float] | None = None) -> float:
    """G(c) = w1*F + w2*P + w3*D, renormalised over agents that returned a score.

    Mohit owns this: calibrate weights, decide how to treat 'no opinion' (here: drop the
    agent and renormalise) and replace with the final formulation.
    """
    w = w or (settings.w_fact, settings.w_policy, settings.w_ground)
    pairs = [(wi, r.score) for wi, r in zip(w, (fact, policy, grounding)) if r.score is not None]
    total = sum(wi for wi, _ in pairs)
    return round(sum(wi * s for wi, s in pairs) / total, 4) if total else 0.0
