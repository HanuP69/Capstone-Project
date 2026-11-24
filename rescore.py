import json
import re
import time
from datetime import datetime

import pandas as pd
from sklearn.metrics import confusion_matrix

from config import (ABSENCE_ENTAIL_THRESHOLD, ATOMIC_MIN_PART_WORDS, ATOMIC_SPLIT,
                    DATA, DEFAULT_WEIGHTS, MAX_ENTAIL_FOR_UNSUPPORTED, RESULTS)
from evaluate import (LABELS, ablation, bootstrap_delta, brier_score,
                      calibration_bins, mcnemar, metrics, normalize_label,
                      normalize_predictions)
from pipeline import Gate, NumericCheck
from pipeline_text import sentences
from quality import has_placeholder, is_hedge, is_well_formed

GRID_PASS = [0.5, 0.55, 0.6, 0.65, 0.7]
GRID_CONTRA = [0.55, 0.6, 0.62, 0.65, 0.7]


def recompute_numeric(frame):
    path = DATA / "records.json"
    if not path.exists():
        return frame
    chunks = {chunk["chunk_id"]: chunk for chunk in
              (json.loads(line) for line in open(DATA / "chunks.jsonl", encoding="utf-8"))}
    checker = NumericCheck(records=json.loads(path.read_text(encoding="utf-8")))
    scores = []
    notes = []
    for row in frame.itertuples():
        evidence = []
        for chunk_id in str(getattr(row, "evidence_chunks", "")).split(";"):
            chunk = chunks.get(chunk_id)
            if not chunk:
                continue
            for sentence in sentences(chunk["text"]):
                evidence.append({"text": sentence, "chunk_id": chunk_id,
                                 "doc_name": chunk["doc_name"], "issue_date": chunk["issue_date"]})
        parts = [part.strip() for part in re.split(ATOMIC_SPLIT, row.claim) if part.strip()]
        if len(parts) >= 2 and all(len(part.split()) >= ATOMIC_MIN_PART_WORDS for part in parts):
            results = [checker.compare(part, evidence) for part in parts]
            score = min(result[0] for result in results)
            note = " | ".join(result[1] for result in results)
        else:
            score, note = checker.compare(row.claim, evidence)
        scores.append(score)
        notes.append(note)
    frame["F"] = scores
    frame["fact_note"] = notes
    return frame


def apply_gate(frame, gate):
    decisions = []
    labels = []
    scores = []
    for row in frame.itertuples():
        score, decision, label = gate.decide(row.F, row.P, row.D, row.C)
        scores.append(round(score, 3))
        decisions.append(decision)
        labels.append(label)
    frame["G"] = scores
    frame["decision"] = decisions
    frame["pred_d"] = labels
    return frame


def tune_thresholds(dev):
    truth = [normalize_label(value) for value in dev["label"]]
    best = None
    for tau_pass in GRID_PASS:
        for tau_contra in GRID_CONTRA:
            gate = Gate(tau_pass=tau_pass, tau_contra=tau_contra)
            preds = []
            for row in dev.itertuples():
                preds.append(gate.decide(row.F, row.P, row.D, row.C)[2])
            result = metrics(pd.DataFrame({"label": truth, "pred": preds}), "pred")
            if best is None or result["macro_f1"] > best["dev_macro_f1"]:
                best = {
                    "tau_pass": tau_pass,
                    "tau_contra": tau_contra,
                    "dev_macro_f1": result["macro_f1"],
                    "dev_accuracy": result["accuracy"],
                    "weights": DEFAULT_WEIGHTS,
                }
    return best


def masks(frame):
    real = frame["source_chunk_id"].fillna("").astype(str).str.len() > 0
    hedge = frame["claim"].map(is_hedge)
    placeholder = frame["claim"].map(has_placeholder)
    well = frame["claim"].map(is_well_formed)
    return {
        "all": pd.Series(True, index=frame.index),
        "real_chunk": real,
        "non_hedge": ~hedge & ~placeholder,
        "well_formed": well,
        "real_chunk_well_formed": real & well,
        "clean_absent": ~real & ~hedge & ~placeholder & well,
    }


def subset_table(frame):
    rows = []
    for name, mask in masks(frame).items():
        sub = frame[mask]
        if sub.empty:
            continue
        for condition in ("pred_a", "pred_b", "pred_c", "pred_d"):
            result = metrics(sub, condition)
            rows.append({"subset": name, "condition": condition, "n": result["n"],
                         "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
                         "missed_falsehood_rate": result["missed_falsehood_rate"],
                         "abstention_rate": result["abstention_rate"]})
    return pd.DataFrame(rows)


def absent_rows():
    path = DATA / "claims_absent.jsonl"
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in open(path, encoding="utf-8")])


def quality_table(frame):
    real = frame["source_chunk_id"].fillna("").astype(str).str.len() > 0
    hedge = frame["claim"].map(is_hedge)
    placeholder = frame["claim"].map(has_placeholder)
    well = frame["claim"].map(is_well_formed)
    absent = absent_rows()
    overlap = 0
    if not absent.empty and "source_question" in absent.columns:
        seen = absent.groupby("source_question")["split"].nunique()
        overlap = int((seen > 1).sum())
    rows = [
        ("test_claims", len(frame)),
        ("test_real_chunk", int(real.sum())),
        ("test_absent", int((~real).sum())),
        ("test_hedge", int(hedge.sum())),
        ("test_placeholder", int(placeholder.sum())),
        ("test_fragment", int((~well).sum())),
        ("test_well_formed", int(well.sum())),
        ("test_real_chunk_well_formed", int((real & well).sum())),
        ("test_real_chunk_fragment", int((real & ~well).sum())),
        ("test_real_chunk_hedge", int((real & hedge).sum())),
        ("test_real_chunk_placeholder", int((real & placeholder).sum())),
        ("test_clean_absent", int((~real & ~hedge & ~placeholder & well).sum())),
        ("test_unverifiable", int((frame["label"] == "unverifiable").sum())),
        ("test_unverifiable_hedge", int(((frame["label"] == "unverifiable") & hedge).sum())),
        ("absent_claims", len(absent)),
        ("absent_questions_in_both_splits", overlap),
        ("absent_nitj_mentions", int(absent["claim"].str.contains("NIT Jamshedpur", case=False, na=False).sum()) if not absent.empty else 0),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def write_artifacts(dev, test, clean, contaminated=0, audit=None):
    conditions = pd.DataFrame([metrics(clean, column) for column in ("pred_a", "pred_b", "pred_c", "pred_d")])
    conditions.to_csv(RESULTS / "condition_metrics.csv", index=False)

    old_stats = {}
    if (RESULTS / "stats.json").exists():
        old_stats = json.loads((RESULTS / "stats.json").read_text(encoding="utf-8"))
    stats = {
        "n_scored": len(clean),
        "n_excluded_contaminated": contaminated,
        "n_absence_above_generation_threshold": int(audit["above_generation_threshold"].sum()) if audit is not None and "above_generation_threshold" in audit else 0,
        "mcnemar_d_vs_b": mcnemar(clean["pred_d"], clean["pred_b"], clean["label"]),
        "mcnemar_d_vs_a": mcnemar(clean["pred_d"], clean["pred_a"], clean["label"]),
        "bootstrap_d_minus_b": bootstrap_delta(clean["pred_d"], clean["pred_b"], clean["label"]),
        "confusion_d": confusion_matrix(clean["label"], clean["pred_d"], labels=LABELS).tolist(),
        "labels": LABELS,
        "judge_backend": ",".join(sorted({str(value) for value in clean["judge_backend"]})),
        "seconds": old_stats.get("seconds", 0),
        "brier_d": brier_score(clean),
        "calibration_d": calibration_bins(clean),
        "headline_subset": "real_chunk_well_formed",
        "rescored_at": datetime.now().isoformat(timespec="seconds"),
    }
    (RESULTS / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    pd.DataFrame(stats["calibration_d"]).to_csv(RESULTS / "calibration.csv", index=False)

    headline = clean[clean["source_chunk_id"].fillna("").astype(str).str.len() > 0]
    headline_stats = {
        "n_scored": len(headline),
        "n_excluded_generated": len(clean) - len(headline),
        "label_counts": {label: int((headline["label"] == label).sum()) for label in LABELS},
        "mcnemar_d_vs_b": mcnemar(headline["pred_d"], headline["pred_b"], headline["label"]),
        "bootstrap_d_minus_b": bootstrap_delta(headline["pred_d"], headline["pred_b"], headline["label"]),
        "confusion_d": confusion_matrix(headline["label"], headline["pred_d"], labels=LABELS).tolist(),
        "brier_d": brier_score(headline),
        "judge_backend": ",".join(sorted({str(value) for value in headline["judge_backend"]})),
    }
    (RESULTS / "headline_stats.json").write_text(json.dumps(headline_stats, indent=2), encoding="utf-8")
    headline_rules = ablation(headline)
    base_accuracy = float(headline_rules.loc[headline_rules["rule"] == "cascade", "accuracy"].iloc[0])
    base_f1 = float(headline_rules.loc[headline_rules["rule"] == "cascade", "macro_f1"].iloc[0])
    headline_rules["delta_accuracy_vs_cascade"] = (headline_rules["accuracy"] - base_accuracy).round(4)
    headline_rules["delta_macro_f1_vs_cascade"] = (headline_rules["macro_f1"] - base_f1).round(4)
    headline_rules.to_csv(RESULTS / "headline_ablation.csv", index=False)
    headline_subgroup = []
    headline_multi = headline["multi_fact"].astype(str).str.lower().isin(["true", "1"])
    for name, mask in (("single_fact", ~headline_multi), ("multi_fact", headline_multi)):
        sub = headline[mask]
        if sub.empty:
            continue
        result = metrics(sub, "pred_d")
        headline_subgroup.append({"group": name, "n": result["n"], "accuracy": result["accuracy"],
                                  "macro_f1": result["macro_f1"],
                                  "missed_falsehood": result["missed_falsehood_rate"],
                                  "abstention": result["abstention_rate"]})
    pd.DataFrame(headline_subgroup).to_csv(RESULTS / "headline_subgroup.csv", index=False)
    headline_latency = headline[[column for column in ("lat_a", "lat_b", "lat_c", "lat_d") if column in headline.columns]].mean().round(3).to_dict()
    (RESULTS / "headline_latency.json").write_text(json.dumps(headline_latency, indent=2), encoding="utf-8")

    subgroup = []
    multi = clean["multi_fact"].astype(str).str.lower().isin(["true", "1"])
    for name, mask in (("single_fact", ~multi), ("multi_fact", multi)):
        sub = clean[mask]
        if sub.empty:
            continue
        result = metrics(sub, "pred_d")
        subgroup.append({"group": name, "n": result["n"], "accuracy": result["accuracy"],
                         "macro_f1": result["macro_f1"],
                         "missed_falsehood": result["missed_falsehood_rate"],
                         "abstention": result["abstention_rate"]})
    pd.DataFrame(subgroup).to_csv(RESULTS / "subgroup_metrics.csv", index=False)

    rules = ablation(clean)
    base_accuracy = float(rules.loc[rules["rule"] == "cascade", "accuracy"].iloc[0])
    base_f1 = float(rules.loc[rules["rule"] == "cascade", "macro_f1"].iloc[0])
    rules["delta_accuracy_vs_cascade"] = (rules["accuracy"] - base_accuracy).round(4)
    rules["delta_macro_f1_vs_cascade"] = (rules["macro_f1"] - base_f1).round(4)
    rules.to_csv(RESULTS / "ablation.csv", index=False)
    subset_table(test).to_csv(RESULTS / "subset_metrics.csv", index=False)
    quality_table(test).to_csv(RESULTS / "quality_audit.csv", index=False)

    latency_columns = [column for column in ("lat_a", "lat_b", "lat_c", "lat_d") if column in clean.columns]
    if latency_columns:
        latency = clean[latency_columns].mean().round(3).to_dict()
        (RESULTS / "latency.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")


def rescore():
    t0 = time.time()
    dev = recompute_numeric(normalize_predictions(pd.read_csv(RESULTS / "dev_predictions.csv")))
    test = recompute_numeric(normalize_predictions(pd.read_csv(RESULTS / "test_predictions.csv")))
    config = tune_thresholds(dev)
    gate = Gate(tau_pass=config["tau_pass"], tau_contra=config["tau_contra"])
    apply_gate(dev, gate)
    apply_gate(test, gate)
    dev.to_csv(RESULTS / "dev_predictions.csv", index=False)
    test.to_csv(RESULTS / "test_predictions.csv", index=False)
    (DATA / "thresholds.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    audit = pd.read_csv(RESULTS / "absence_audit.csv") if (RESULTS / "absence_audit.csv").exists() else None
    clean = test.copy()
    contaminated = 0
    if audit is not None and "max_entail" in audit.columns:
        flagged = audit[audit["max_entail"] >= ABSENCE_ENTAIL_THRESHOLD]["claim_id"]
        mask = (clean["label"] == "unverifiable") & clean["claim_id"].isin(flagged)
        contaminated = int(mask.sum())
        clean = clean[~mask].copy()
    clean.to_csv(RESULTS / "test_scored.csv", index=False)
    write_artifacts(dev, test, clean, contaminated, audit)

    answers_path = RESULTS / "unsupervised_answers.csv"
    if answers_path.exists():
        answers = pd.read_csv(answers_path)
        if "answer_relevance" in answers.columns:
            answers = answers.rename(columns={"answer_relevance": "query_entailment"})
            answers.to_csv(answers_path, index=False)
        aggregate = answers.groupby("condition").agg(
            unsupported_per_answer=("unsupported_per_answer", "mean"),
            answered_rate=("answered", "mean"),
            faithful_rate=("faithful", "mean"),
            claims_per_answer=("n_claims", "mean"),
            context_relevance=("context_relevance", "mean"),
            query_entailment=("query_entailment", "mean"),
        ).round(3).reset_index()
        aggregate.to_csv(RESULTS / "unsupervised_metrics.csv", index=False)

    print(json.dumps({"thresholds": config, "n_test": len(test),
                      "n_scored": len(clean), "seconds": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    rescore()
