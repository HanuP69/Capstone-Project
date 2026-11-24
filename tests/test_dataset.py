import json

import pandas as pd

from config import DATA, RESULTS
from evaluate import load_claims
from quality import has_placeholder, is_fragment, is_hedge, is_well_formed


def load_split():
    return json.loads((DATA / "split.json").read_text(encoding="utf-8"))


def load_chunks():
    return {json.loads(line)["chunk_id"]: json.loads(line)
            for line in open(DATA / "chunks.jsonl", encoding="utf-8")}


def test_split_counts_reconcile():
    split = load_split()
    assert split["total"] == 233
    assert split["constructed_claims"] + split["absent_claims"] == split["total"]
    assert split["dev_claims"] + split["test_claims"] == split["total"]
    assert split["test_absent"] == 38


def test_claim_ids_are_unique_and_partitioned():
    claims = load_claims()
    ids = [claim["claim_id"] for claim in claims]
    assert len(ids) == len(set(ids))
    split = load_split()
    assert len(claims) == split["total"]


def test_prediction_ids_are_unique_and_traceable():
    claims = load_claims()
    known = {claim["claim_id"] for claim in claims}
    for name in ("test_predictions.csv", "dev_predictions.csv"):
        predictions = pd.read_csv(RESULTS / name)
        assert predictions["claim_id"].is_unique
        assert set(predictions["claim_id"]) <= known


def test_every_constructed_claim_has_a_traceable_source():
    chunks = load_chunks()
    claims = [claim for claim in load_claims() if claim.get("source_chunk_id")]
    assert len(claims) == 153
    for claim in claims:
        chunk = chunks[claim["source_chunk_id"]]
        assert chunk["doc_name"] == claim["doc_name"]
        assert claim["label"] in {"true", "false", "unverifiable"}
        assert claim["split"] in {"dev", "test"}


def test_absent_claims_are_unverifiable_and_unsourced():
    absent = [claim for claim in load_claims() if claim.get("method") == "generated_absent"]
    assert len(absent) == 80
    for claim in absent:
        assert claim["label"] == "unverifiable"
        assert claim["source_chunk_id"] == ""
        assert claim["doc_name"] == ""
        assert claim["split"] in {"dev", "test"}


def test_quality_audit_matches_in_memory_counts():
    test_claims = [claim for claim in load_claims() if claim.get("split") == "test"]
    audit = pd.read_csv(RESULTS / "quality_audit.csv").set_index("metric")["value"]
    constructed = [claim for claim in test_claims if claim.get("source_chunk_id")]
    assert len(test_claims) == 109
    assert len(constructed) == 71
    assert sum(is_fragment(claim["claim"]) for claim in test_claims) == audit["test_fragment"]
    assert sum(is_hedge(claim["claim"]) for claim in test_claims) == audit["test_hedge"]
    assert sum(has_placeholder(claim["claim"]) for claim in test_claims) == audit["test_placeholder"]
    assert sum(is_well_formed(claim["claim"]) for claim in constructed) == audit["test_real_chunk_well_formed"]


def test_test_and_dev_document_counts_match_split_artifact():
    split = load_split()
    claims = [claim for claim in load_claims() if claim.get("source_chunk_id")]
    for name in ("dev", "test"):
        docs = {claim["doc_name"] for claim in claims if claim["split"] == name}
        assert len(docs) == split[f"{name}_docs"]
