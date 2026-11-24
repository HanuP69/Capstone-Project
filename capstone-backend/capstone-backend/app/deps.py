from app.agents import team
from app.agents.starter.fact_check import FactCheckAgent
from app.agents.starter.grounding import GroundingAgent
from app.agents.starter.policy import PolicyComplianceAgent
from app.db import make_store
from app.pipeline.claim_extractor import ClaimExtractor
from app.pipeline.generator import MockGenerator
from app.pipeline.orchestrator import Pipeline

store = make_store()
pipeline = Pipeline(
    MockGenerator(), ClaimExtractor(),
    team.load_fact_check() or FactCheckAgent(),
    team.load_policy() or PolicyComplianceAgent(),
    team.load_grounding() or GroundingAgent(),
)
