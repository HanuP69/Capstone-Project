"""PLACEHOLDERS for the team's finished agents.

Fill in each loader to return an agent (use AgentAdapter around your function/class).
While a loader returns None, the pipeline falls back to the starter agent in
agents/starter/, so the API keeps working.

Example:
    from my_agents.fact import check_fact          # your code
    def load_fact_check():
        return AgentAdapter("fact_check", check_fact)
"""
from app.agents.adapters import AgentAdapter  # noqa: F401
from app.agents.base import VerificationAgent


def load_fact_check() -> VerificationAgent | None:
    return None  # TODO: plug in the real Fact-Check agent


def load_policy() -> VerificationAgent | None:
    return None  # TODO: plug in the real Policy-Compliance agent


def load_grounding() -> VerificationAgent | None:
    return None  # TODO: plug in the real Grounding agent
