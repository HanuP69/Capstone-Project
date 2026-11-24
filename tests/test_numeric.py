import pytest

from claims import replace_money
from pipeline import NumericCheck
from pipeline_text import (entities, parse_cgpa, parse_dates, parse_days, parse_money,
                           parse_percent, parse_time)


def evidence(text="", chunk_id="c1", doc_name="notice.pdf"):
    return [{"text": text, "chunk_id": chunk_id, "doc_name": doc_name,
             "issue_date": "2025-01-01"}]


@pytest.mark.parametrize("text,expected", [
    ("pay 1000/- now", [1000.0]),
    ("Rs. 1,00,000/-", [100000.0]),
    ("₹500", [500.0]),
    ("1.5 lakh", [150000.0]),
    ("2 crore", [20000000.0]),
])
def test_money_grammar(text, expected):
    assert parse_money(text) == expected


def test_money_below_ten_is_not_a_fee():
    assert parse_money("Rs 5") == []


@pytest.mark.parametrize("text,expected", [
    ("75%", [75.0]),
    ("75 percent", [75.0]),
    ("12.5 percent", [12.5]),
    ("150%", []),
])
def test_percent_grammar(text, expected):
    assert parse_percent(text) == expected


def test_dates_in_supported_formats():
    assert parse_dates("07/03/2025") == ["2025-03-07"]
    assert parse_dates("7 March 2025") == ["2025-03-07"]
    assert parse_dates("March 7, 2025") == ["2025-03-07"]


def test_invalid_date_is_ignored():
    assert parse_dates("31/02/2025") == []


def test_days_and_cgpa_and_time():
    assert parse_days("15 working days") == [15]
    assert parse_days("401 days") == []
    assert parse_cgpa("CGPA of 8.5") == [8.5]
    assert parse_cgpa("8.5/10") == [8.5]
    assert parse_time("2:30 pm") == ["14:30"]
    assert parse_time("12:00 am") == ["00:00"]
    assert parse_time("10.15 hrs") == ["10:15"]


def test_entities_has_all_numeric_kinds():
    found = entities("Rs 1000, 75 percent, 15 days, 07/03/2025, 2:30 pm, CGPA of 8.5")
    assert found["money"] == [1000.0]
    assert found["percent"] == [75.0]
    assert found["days"] == [15]
    assert found["dates"] == ["2025-03-07"]
    assert found["times"] == ["14:30"]
    assert found["cgpa"] == [8.5]


def test_replace_money_preserves_bare_format():
    assert replace_money("pay 1000/- now", 1000, 1005) == "pay 1005/- now"


def test_replace_money_preserves_grouped_format():
    assert replace_money("pay Rs 1,00,000/- now", 100000, 100005) == "pay Rs 100,005/- now"


def test_replace_money_returns_none_when_value_is_absent():
    assert replace_money("pay 1000/- now", 2000, 2005) is None


def test_record_matches_attendance_and_rejects_wrong_value():
    record = {"key": "attendance_min", "aliases": ["minimum attendance"],
              "val_type": "percent", "operator": ">=", "value": 75, "support": 3,
              "source_chunk_id": "c1", "source_doc": "notice.pdf"}
    checker = NumericCheck(records=[record])
    good, good_note = checker.compare("The minimum attendance is 75 percent.",
                                      evidence("The minimum attendance is 75 percent."))
    bad, bad_note = checker.compare("The minimum attendance is 70 percent.",
                                    evidence("The minimum attendance is 75 percent."))
    assert good == 1.0 and "attendance_min" in good_note
    assert bad == 0.0 and "attendance_min" in bad_note


def test_percent_tolerance_is_absolute_not_ten_percent_relative():
    checker = NumericCheck(records=[])
    close, _ = checker.compare("The minimum attendance is 75 percent.",
                               evidence("The minimum attendance is 75.05 percent."))
    far, _ = checker.compare("The minimum attendance is 75 percent.",
                             evidence("The minimum attendance is 80 percent."))
    assert close == 1.0
    assert far == 0.0


def test_money_tolerance_is_exact():
    checker = NumericCheck(records=[])
    same, _ = checker.compare("The fee is Rs 1000.", evidence("The fee is Rs 1000."))
    different, note = checker.compare("The fee is Rs 1000.", evidence("The fee is Rs 1005."))
    assert same == 1.0
    assert different == 0.0
    assert "1000.0" in note and "1005.0" in note


def test_record_below_support_is_skipped():
    record = {"key": "attendance_min", "aliases": ["minimum attendance"],
              "val_type": "percent", "operator": ">=", "value": 75, "support": 1,
              "source_chunk_id": "c1", "source_doc": "notice.pdf"}
    checker = NumericCheck(records=[record])
    score, note = checker.compare("The minimum attendance is 75 percent.",
                                  evidence("The minimum attendance is 75 percent."))
    assert score == 1.0
    assert "record" not in note


def test_record_is_skipped_when_claim_has_another_numeric_kind():
    record = {"key": "attendance_min", "aliases": ["minimum attendance"],
              "val_type": "percent", "operator": ">=", "value": 75, "support": 3,
              "source_chunk_id": "c1", "source_doc": "notice.pdf"}
    checker = NumericCheck(records=[record])
    claim = "The minimum attendance is 75 percent and the late fee is Rs 1000."
    text = "The minimum attendance is 75 percent and the late fee is Rs 1000."
    _, note = checker.compare(claim, evidence(text))
    assert "record" not in note


def test_wrong_topic_does_not_match_generic_money():
    checker = NumericCheck(records=[])
    score, note = checker.compare("The tuition fee is Rs 1000.",
                                  evidence("The late registration fee is Rs 1000."))
    assert score == 0.5
    assert "overlapping" in note


def test_claim_without_checkable_value_is_unverifiable():
    checker = NumericCheck(records=[])
    score, note = checker.compare("The policy is strict.", evidence("The policy is strict."))
    assert score == 0.5
    assert "no checkable value" in note
