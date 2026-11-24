from abc import ABC, abstractmethod


class ResponseGenerator(ABC):
    @abstractmethod
    async def generate(self, query: str, feedback: list[str] | None = None) -> str:
        """`feedback` holds verification-failure reasons from the previous attempt."""


class MockGenerator(ResponseGenerator):
    """Dev stub. Replace with Tanishk's LLM+RAG generator (same interface)."""

    DRAFT = ("Mid-semester examinations start on 20 October. "
             "Students need a minimum attendance of 60%. "
             "Timetables are posted on the department notice board.")
    CORRECTED = ("Mid-semester examinations start on 15 October. "
                 "Students need a minimum attendance of 75%.")

    async def generate(self, query: str, feedback: list[str] | None = None) -> str:
        return self.CORRECTED if feedback else self.DRAFT
