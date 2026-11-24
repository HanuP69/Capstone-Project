import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config import DATA, RESULTS, ROOT
from evaluate import metrics

DOCS = ["README.md", "dashboard.html", "TODO.md"]
OPS = {"8001", "5173", "5176", "127", "600", "1200"}
NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")
FORBIDDEN = ["[Synthetic]", "MFAQ", "meta_learner_mle", "91.7%", "2.5%", "78.2%"]


def add_number(values, number):
    if isinstance(number, bool):
        return
    try:
        value = float(number)
    except (TypeError, ValueError):
        return
    values.add(str(number))
    for digits in (1, 2, 3, 4):
        values.add(f"{value:.{digits}f}")
    if 0 <= value <= 1:
        values.add(f"{value * 100:.2f}")


def collect():
    values = set()
    for path in list(RESULTS.glob("*.csv")) + list(DATA.glob("*.csv")):
        frame = pd.read_csv(path)
        for column in frame.columns:
            for value in frame[column].dropna():
                for token in NUM.findall(str(value)):
                    add_number(values, token)
    for path in list(RESULTS.glob("*.json")) + list(DATA.glob("*.json")):
        if path.name == "artifacts_manifest.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, (int, float)):
                add_number(values, item)
            elif isinstance(item, str):
                for token in NUM.findall(item):
                    add_number(values, token)
    return values


def allowed(token, values):
    if token in OPS or re.fullmatch(r"20\d{2}", token):
        return True
    if token in values:
        return True
    try:
        number = float(token)
    except ValueError:
        return False
    return any(abs(number - float(value)) < 5e-4 for value in values
               if re.fullmatch(r"-?\d+(?:\.\d+)?", value))


def scan(path, values):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S)
    text = re.sub(r"https?://\S+|\b\d{1,3}(?:\.\d{1,3}){3}\b", " ", text)
    bad = []
    for match in NUM.finditer(text):
        token = match.group(1)
        if "." not in token and len(token) < 2:
            continue
        if not allowed(token, values):
            start = max(0, match.start() - 55)
            bad.append((token, text[start:match.end() + 55].replace("\n", " ")))
    return bad


def check_records(errors):
    records = json.loads((DATA / "records.json").read_text(encoding="utf-8"))
    chunks = {chunk["chunk_id"]: chunk for chunk in
              (json.loads(line) for line in open(DATA / "chunks.jsonl", encoding="utf-8"))}
    for record in records:
        if int(record.get("support", 0)) < 2:
            errors.append(f"record {record.get('key')} has support below 2")
        chunk = chunks.get(record.get("source_chunk_id"))
        if not chunk:
            errors.append(f"record {record.get('key')} cites a missing chunk")
            continue
        if chunk["doc_name"] != record.get("source_doc"):
            errors.append(f"record {record.get('key')} source document mismatch")
        phrase = record.get("source_phrase", "")
        if phrase and phrase.lower() not in record.get("evidence", "").lower():
            errors.append(f"record {record.get('key')} evidence does not contain its phrase")


def check_split(errors):
    split = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
    claims = sum(1 for _ in open(DATA / "claims.jsonl", encoding="utf-8"))
    absent = sum(1 for _ in open(DATA / "claims_absent.jsonl", encoding="utf-8"))
    if split.get("total") != claims + absent:
        errors.append("split.json total does not equal claims plus absent claims")
    if split.get("dev_claims", 0) + split.get("test_claims", 0) != split.get("total"):
        errors.append("split.json dev/test counts do not sum to total")


def check_metrics(errors):
    predictions = pd.read_csv(RESULTS / "test_predictions.csv",
                              dtype={"pred_a": "string", "pred_b": "string",
                                     "pred_c": "string", "pred_d": "string"})
    conditions = pd.read_csv(RESULTS / "condition_metrics.csv").set_index("condition")
    for column in ("pred_a", "pred_b", "pred_c", "pred_d"):
        if not pd.api.types.is_string_dtype(predictions[column]):
            errors.append(f"{column} is not stored as strings")
    result = metrics(predictions, "pred_d")
    expected = float(conditions.loc["pred_d", "accuracy"])
    if abs(result["accuracy"] - expected) > 1e-4:
        errors.append("condition_metrics accuracy does not match test_predictions")


def check_dashboard(errors):
    root = ROOT / "dashboard" / "dist"
    summary_path = root / "data" / "summary.json"
    index_path = root / "index.html"
    if not summary_path.exists() or not index_path.exists():
        errors.append("dashboard build output is missing")
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    text = index_path.read_text(encoding="utf-8")
    for marker in FORBIDDEN:
        if marker.lower() in text.lower():
            errors.append(f"dashboard contains forbidden marker {marker}")
    if summary.get("dataset_mode", "").lower().find("no synthetic benchmark") < 0:
        errors.append("dashboard dataset mode is not explicit")
    predictions = pd.read_csv(RESULTS / "test_predictions.csv")
    source_backed = predictions[predictions["source_chunk_id"].fillna("").astype(str).str.len() > 0]
    if len(summary.get("claims", [])) != len(source_backed):
        errors.append("dashboard claim count does not match source-backed test claims")
    if len(summary.get("metrics", [])) != 4:
        errors.append("dashboard metrics are incomplete")


def check_manifest(errors):
    path = ROOT / "artifacts_manifest.json"
    if not path.exists():
        errors.append("artifacts_manifest.json is missing")
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("entries"):
        errors.append("manifest has no entries")
    for entry in manifest.get("entries", []):
        if not entry.get("artifact_commit"):
            errors.append("manifest entry has no artifact commit")
            break
    head = manifest.get("head_at_generation")
    if head:
        try:
            subprocess.run(["git", "merge-base", "--is-ancestor", head, "HEAD"],
                           cwd=ROOT, check=True, capture_output=True)
        except Exception:
            errors.append("manifest head_at_generation is not an ancestor of HEAD")


def main():
    values = collect()
    errors = []
    for name in DOCS:
        for token, context in scan(ROOT / name, values):
            errors.append(f"{name}: untraceable number {token} | ...{context}...")
    check_records(errors)
    check_split(errors)
    check_metrics(errors)
    check_dashboard(errors)
    check_manifest(errors)
    if errors:
        print(f"{len(errors)} artifact checks failed:")
        for error in errors[:60]:
            print(" -", error)
        sys.exit(1)
    print("all artifact checks passed")


if __name__ == "__main__":
    main()
