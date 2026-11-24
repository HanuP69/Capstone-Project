import pandas as pd
import pytest

import evaluate
from evaluate import majority_label, parse_label, prompt_a, prompt_b, run_conditions
from pipeline import ClaimResult


@pytest.mark.parametrize("text,expected", [
    ("TRUE", "true"),
    ("true.", "true"),
    ("yes", "true"),
    ("supported", "true"),
    ("FALSE", "false"),
    ("no", "false"),
    ("not supported", "false"),
    ("not   supported", "false"),
    ("contradicted", "false"),
    ("UNVERIFIABLE", "unverifiable"),
    ("I cannot determine", "unverifiable"),
    ("", "unverifiable"),
])
def test_parse_label_contract(text, expected):
    assert parse_label(text) == expected


def test_prompt_a_is_three_way_and_not_binary():
    prompt = prompt_a("The fee is Rs 1000.")
    assert "TRUE, FALSE, or UNVERIFIABLE" in prompt
    assert "Answer YES" not in prompt.upper()
    assert "Answer NO" not in prompt.upper()


def test_prompt_b_is_three_way_and_contains_context():
    prompt = prompt_b("The fee is Rs 1000.", "The fee is Rs 1000.")
    assert "TRUE, FALSE, or UNVERIFIABLE" in prompt
    assert "Context:" in prompt
    assert "The fee is Rs 1000." in prompt


def test_majority_label_returns_unique_winner():
    assert majority_label(["TRUE", "TRUE", "FALSE"]) == "true"
    assert majority_label(["FALSE", "false", "TRUE"]) == "false"


def test_majority_label_tie_is_unverifiable_and_deterministic():
    samples = ["TRUE", "FALSE", "UNVERIFIABLE"]
    assert majority_label(samples) == "unverifiable"
    assert majority_label(samples) == majority_label(samples)


def test_majority_label_empty_is_unverifiable():
    assert majority_label([]) == "unverifiable"


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def call(self, prompt, temperature=0.0, num_predict=0):
        self.prompts.append(prompt)
        return "TRUE"


class FakeRetriever:
    def retrieve(self, query):
        return [{"chunk_id": "c1", "text": "The minimum attendance is 75 percent.",
                 "doc_name": "notice.pdf", "issue_date": "2025-01-01",
                 "ref_number": "R1", "category": "Circulars"}]


class FakePipeline:
    def __init__(self):
        self.llm = FakeLLM()
        self.retriever = FakeRetriever()
        self.judge = None

    def verify(self, claim):
        return ClaimResult(claim=claim, category="Circulars", F=1.0, P=1.0, D=0.9,
                           C=0.0, G=0.9, decision="PASS", label="true",
                           fact_note="match", policy_note="ok", evidence=[])


def test_run_conditions_uses_three_way_prompts(monkeypatch):
    pipe = FakePipeline()
    monkeypatch.setattr(evaluate, "Pipeline", lambda: pipe)
    claims = [{"claim_id": "T1", "claim": "The minimum attendance is 75 percent.",
               "label": "true", "split": "test", "category": "Circulars"}]
    result = run_conditions(claims, use_llm=True, split="test")
    assert len(result) == 1
    assert result.iloc[0]["pred_a"] == "true"
    assert result.iloc[0]["pred_b"] == "true"
    assert result.iloc[0]["pred_c"] == "true"
    assert result.iloc[0]["pred_d"] == "true"
    assert len(pipe.llm.prompts) == 5
    assert all("TRUE, FALSE, or UNVERIFIABLE" in prompt for prompt in pipe.llm.prompts)
    assert not any("ANSWER YES" in prompt.upper() for prompt in pipe.llm.prompts)
    assert not any("ANSWER NO" in prompt.upper() for prompt in pipe.llm.prompts)
