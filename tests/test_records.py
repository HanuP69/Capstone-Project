import json

from config import DATA, RECORD_MIN_SUPPORT
from records import extract_all, load_chunks, sane, values_near


def test_extract_all_supports_record_types():
    assert extract_all("currency", "The fee is Rs 1000.") == [1000.0]
    assert extract_all("percent", "Attendance must be 75 percent.") == [75.0]
    assert extract_all("count_days", "Submit within 15 working days.") == [15]
    assert extract_all("date", "Dated 07/03/2025.") == ["2025-03-07"]


def test_values_near_finds_value_after_phrase():
    found = list(values_near("The late registration fee is Rs 1000.", "late registration fee", "currency"))
    assert found[0][0] == 1000.0


def test_values_near_returns_nothing_for_missing_phrase():
    assert list(values_near("The hostel fee is Rs 15000.", "late fee", "currency")) == []


def test_sane_bounds():
    assert not sane("attendance_min", 49)
    assert sane("attendance_min", 50)
    assert sane("attendance_min", 100)
    assert not sane("attendance_min", 101)
    assert not sane("hostel_fee", 99)
    assert sane("hostel_fee", 100)


def test_every_record_is_supported_and_traceable():
    records = json.loads((DATA / "records.json").read_text(encoding="utf-8"))
    chunks = {chunk["chunk_id"]: chunk for chunk in load_chunks()}
    assert records
    for record in records:
        assert record["support"] >= RECORD_MIN_SUPPORT
        assert record["source_chunk_id"] in chunks
        chunk = chunks[record["source_chunk_id"]]
        assert chunk["doc_name"] == record["source_doc"]
        assert record["source_phrase"].lower() in record["evidence"].lower()
        assert record["value"] in record["candidates"]
        assert record["aliases"]


def test_record_keys_are_unique_and_sane():
    records = json.loads((DATA / "records.json").read_text(encoding="utf-8"))
    keys = [record["key"] for record in records]
    assert len(keys) == len(set(keys))
    for record in records:
        assert sane(record["key"], record["value"])
