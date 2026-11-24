import json
from pathlib import Path

from app.agents.base import VerificationAgent
from app.schemas import AgentResult, Claim, Verdict
from app.textutils import tokens

DOCS_PATH = Path(__file__).parent.parent.parent / "data" / "documents.json"


class GroundingAgent(VerificationAgent):
    """Retrieval-grounding check.

    PLACEHOLDER: lexical overlap over a tiny JSON store so the pipeline runs end-to-end.
    Tanishk's RAG layer (Chroma/FAISS + embeddings) should replace `_retrieve` and keep
    the same AgentResult contract (score = similarity/entailment in [0,1]).
    """

    name = "grounding"

    def __init__(self, docs_path: Path = DOCS_PATH, support_threshold: float = 0.6):
        self.docs = json.loads(Path(docs_path).read_text())
        self.support_threshold = support_threshold

    def _retrieve(self, claim: Claim, k: int = 1):
        ctoks = tokens(claim.text)
        scored = []
        for d in self.docs:
            dtoks = tokens(d["text"])
            sim = len(ctoks & dtoks) / max(len(ctoks), 1)
            scored.append((sim, d))
        return sorted(scored, key=lambda x: -x[0])[:k]

    async def verify(self, claim: Claim) -> AgentResult:
        (sim, doc), = self._retrieve(claim, k=1)
        evidence = [f'{doc["source"]}: {doc["text"]}']
        if sim >= self.support_threshold:
            return AgentResult(agent=self.name, verdict=Verdict.SUPPORTED, score=min(1.0, sim), evidence=evidence)
        return AgentResult(agent=self.name, verdict=Verdict.UNVERIFIABLE, score=sim, evidence=evidence,
                           reason="Weak support in retrieved documents")
