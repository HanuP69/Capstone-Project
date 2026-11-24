import pandas as pd

from reproduce import check, claim_report, ledger


def test_ledger_covers_every_source_backed_claim():
    rows = ledger()
    assert len(rows) == 153
    assert rows["claim_id"].is_unique
    assert (rows["source_chunk_id"] != "").all()
    assert rows["source_traceable"].all()


def test_claim_report_exposes_source_text_and_prediction():
    rows = ledger()
    claim_id = rows[(rows["split"] == "test") & (rows["source_chunk_id"] != "")].iloc[0]["claim_id"]
    report = claim_report(claim_id)
    assert report["claim_id"] == claim_id
    assert report["source_chunk_id"]
    assert report["source_text"]
    assert report["prediction"] in {"true", "false", "unverifiable"}


def test_check_recomputes_metrics_and_baseline_metadata():
    errors, rows = check()
    assert not errors
    assert len(rows) == 153
