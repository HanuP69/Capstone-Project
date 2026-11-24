import json
from pathlib import Path

from claims import replace_money
from config import DATA
from pipeline import Gate, NumericCheck
from pipeline_text import parse_money
from quality import has_placeholder, is_hedge, is_well_formed

ROOT = Path(__file__).resolve().parents[1]


def test_grounding_floor_blocks_false_high_fact():
    gate = Gate(tau_pass=0.6, tau_contra=0.62)
    assert gate.decide(1.0, 0.5, 0.0, 0.982)[2] == "false"
    assert gate.decide(1.0, 0.5, 0.0, 0.883)[2] == "false"
    assert gate.decide(1.0, 0.5, 0.0, 0.970)[2] == "false"


def test_grounding_floor_keeps_supported_high_fact():
    gate = Gate(tau_pass=0.6, tau_contra=0.62)
    assert gate.decide(1.0, 0.5, 0.692, 0.900)[2] == "true"
    assert gate.decide(1.0, 0.5, 0.534, 0.768)[2] == "false"


def test_bare_and_prefixed_money():
    assert parse_money("pay 1005/- now") == [1005.0]
    assert parse_money("Rs. 1,00,000/-") == [100000.0]
    assert parse_money("₹500") == [500.0]
    assert parse_money("1.5 lakh") == [150000.0]


def test_replace_bare_money():
    assert replace_money("pay 1000/- now", 1000, 1005) == "pay 1005/- now"


def test_tuition_claim_does_not_match_late_fee_record():
    records = json.loads((DATA / "records.json").read_text(encoding="utf-8"))
    checker = NumericCheck(records=records)
    chunk = next(json.loads(line) for line in open(DATA / "chunks.jsonl", encoding="utf-8")
                 if json.loads(line)["chunk_id"] == "c00096")
    evidence = [{"text": chunk["text"], "chunk_id": chunk["chunk_id"],
                 "doc_name": chunk["doc_name"], "issue_date": chunk["issue_date"]}]
    score, note = checker.compare("The tuition fee is Rs 1000", evidence)
    assert "late_registration_fee" not in note
    assert score != 1.0


def test_quality_filters():
    assert is_hedge("The fee varies by institution; please check your portal")
    assert has_placeholder("The fee is $X")
    assert not is_well_formed("15000/-) and Mess Fee (Rs.18000/-) for first semester")
    assert is_well_formed("The minimum attendance required for end semester examinations is 75 percent.")
