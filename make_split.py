import json
from collections import Counter

from config import DATA


def load(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main():
    claims = load(DATA / "claims.jsonl")
    absent = load(DATA / "claims_absent.jsonl")
    rows = claims + absent
    docs = {claim.get("doc_name", "") for claim in claims if claim.get("doc_name")}
    dev_docs = {claim.get("doc_name", "") for claim in claims if claim.get("split") == "dev" and claim.get("doc_name")}
    info = {
        "total": len(rows),
        "constructed_claims": len(claims),
        "absent_claims": len(absent),
        "dev_docs": len(dev_docs),
        "test_docs": len(docs - dev_docs),
        "dev_claims": sum(1 for row in rows if row.get("split") == "dev"),
        "test_claims": sum(1 for row in rows if row.get("split") == "test"),
        "dev_absent": sum(1 for row in absent if row.get("split") == "dev"),
        "test_absent": sum(1 for row in absent if row.get("split") == "test"),
        "source_dev_claims": sum(1 for row in claims if row.get("split") == "dev"),
        "source_test_claims": sum(1 for row in claims if row.get("split") == "test"),
        "labels": dict(Counter(row.get("label", "") for row in rows)),
        "methods": dict(Counter(row.get("method", "") for row in rows)),
        "categories": dict(Counter(row.get("category", "") for row in rows)),
    }
    (DATA / "split.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
