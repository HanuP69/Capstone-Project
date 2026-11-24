import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from config import DATA, RESULTS, SEED
from pipeline import (JUDGE_TOP, Pipeline, RERANK_TOP, RETRIEVE_K, RRF_K,
                      Cache, LLM)
from pipeline_text import sentences, tokenize

LABELS = ["true", "false", "unverifiable"]


def load_claims():
    rows = [json.loads(l) for l in open(DATA / "claims.jsonl", encoding="utf-8")]
    if (DATA / "claims_absent.jsonl").exists():
        for l in open(DATA / "claims_absent.jsonl", encoding="utf-8"):
            r = json.loads(l)
            r.setdefault("multi_fact", False)
            r.setdefault("n_facts", 1)
            rows.append(r)
    for r in rows:
        r.setdefault("multi_fact", False)
        r.setdefault("n_facts", 1)
    return rows


def context_of(docs, n=4):
    return "\n\n".join(f"[{d['doc_name']} | {d['issue_date']} | {d['ref_number']}] {d['text'][:900]}" for d in docs[:n])


def run_conditions(claims, use_llm=True, split=None):
    if split is not None:
        claims = [c for c in claims if c.get("split", "test") == split]
    pipe = Pipeline()
    llm = pipe.llm
    judge = pipe.judge
    out = []
    t0 = time.time()
    for i, c in enumerate(claims):
        claim = c["claim"]
        label = c["label"]
        docs = pipe.retriever.retrieve(claim)
        ctx = context_of(docs, 4)
        t_a = time.time()
        a = llm.call(f"Question: Is the following claim true? {claim}\nAnswer YES or NO only.", 0.0, 10) if use_llm else "UNVERIFIABLE"
        lat_a = time.time() - t_a
        t_b = time.time()
        b = llm.call(f"Context:\n{ctx}\n\nClaim: {claim}\nIs the claim supported by the context? Answer YES or NO only.", 0.0, 10) if use_llm else "UNVERIFIABLE"
        lat_b = time.time() - t_b
        t_c = time.time()
        samples = [llm.call(f"Context:\n{ctx}\n\nClaim: {claim}\nIs the claim supported? Answer YES or NO only.", 0.7, 10) for _ in range(3)] if use_llm else ["UNVERIFIABLE"]
        c_ans = max(set(samples), key=samples.count) if samples else "UNVERIFIABLE"
        lat_c = time.time() - t_c
        t_d = time.time()
        res = pipe.verify(claim)
        lat_d = time.time() - t_d
        def norm(x):
            x = (x or "").lower()
            if "yes" in x and "no" not in x:
                return "true"
            if "no" in x:
                return "false"
            return "unverifiable"
        out.append({
            "claim_id": c["claim_id"], "claim": claim, "label": label,
            "category": c.get("category", ""), "split": c.get("split", "test"),
            "multi_fact": bool(c.get("multi_fact", False)), "n_facts": int(c.get("n_facts", 1)),
            "construction": c.get("method", c.get("construction", "")),
            "source_chunk_id": c.get("source_chunk_id", ""),
            "pred_a": norm(a), "pred_b": norm(b), "pred_c": norm(c_ans), "pred_d": res.label,
            "F": res.F, "P": res.P, "D": res.D, "C": res.C, "G": res.G,
            "decision": res.decision, "fact_note": res.fact_note, "policy_note": res.policy_note,
            "judge_backend": res.judge_backend,
            "lat_a": round(lat_a, 3), "lat_b": round(lat_b, 3),
            "lat_c": round(lat_c, 3), "lat_d": round(lat_d, 3),
            "evidence_chunks": ";".join(e["chunk_id"] for e in res.evidence),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(claims)}", flush=True)
    df = pd.DataFrame(out)
    print("run_conditions done in", round(time.time() - t0, 1), "s", flush=True)
    return df


def metrics(df, col):
    y = df["label"].tolist()
    p = df[col].tolist()
    acc = accuracy_score(y, p)
    mp = precision_score(y, p, average="macro", zero_division=0)
    mr = recall_score(y, p, average="macro", zero_division=0)
    mf = f1_score(y, p, average="macro", zero_division=0)
    mfr = float(((df["label"] == "false") & (df[col] != "false")).mean()) if (df["label"] == "false").any() else 0.0
    abst = float((df[col] == "unverifiable").mean())
    pc = precision_score(y, p, average=None, labels=LABELS, zero_division=0)
    rc = recall_score(y, p, average=None, labels=LABELS, zero_division=0)
    fc = f1_score(y, p, average=None, labels=LABELS, zero_division=0)
    return {
        "condition": col, "n": len(df), "accuracy": round(acc, 4),
        "macro_precision": round(mp, 4), "macro_recall": round(mr, 4),
        "macro_f1": round(mf, 4), "missed_falsehood_rate": round(mfr, 4),
        "abstention_rate": round(abst, 4),
        "per_class_precision": {LABELS[i]: round(float(pc[i]), 3) for i in range(3)},
        "per_class_recall": {LABELS[i]: round(float(rc[i]), 3) for i in range(3)},
        "per_class_f1": {LABELS[i]: round(float(fc[i]), 3) for i in range(3)},
    }


def mcnemar(pred_a, pred_b, truth):
    a = (pred_a == truth).astype(int).values if hasattr(pred_a, "values") else np.array(pred_a == truth, dtype=int)
    b = (pred_b == truth).astype(int).values if hasattr(pred_b, "values") else np.array(pred_b == truth, dtype=int)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    from math import sqrt
    if n01 + n10 == 0:
        return {"d_better": 0, "b_better": 0, "p_value": 1.0}
    chi = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    from scipy.stats import chi2
    p = float(1 - chi2.cdf(chi, 1))
    return {"d_better": n10, "b_better": n01, "p_value": round(p, 6)}


def bootstrap_delta(pred_a, pred_b, truth, n=2000, seed=SEED):
    a = (pred_a == truth).astype(int).values if hasattr(pred_a, "values") else np.array(pred_a == truth, dtype=int)
    b = (pred_b == truth).astype(int).values if hasattr(pred_b, "values") else np.array(pred_b == truth, dtype=int)
    rng = np.random.default_rng(seed)
    deltas = []
    n0 = len(a)
    for _ in range(n):
        idx = rng.integers(0, n0, size=n0)
        deltas.append(a[idx].mean() - b[idx].mean())
    deltas = np.array(deltas)
    return {"delta": round(float(deltas.mean()), 4),
            "ci_low": round(float(np.percentile(deltas, 2.5)), 4),
            "ci_high": round(float(np.percentile(deltas, 97.5)), 4)}


def absence_audit(test_df, pipe):
    aud = []
    for r in test_df[test_df["label"] == "unverifiable"].itertuples():
        docs = pipe.retriever.retrieve(r.claim)
        ev = pipe.retriever.best_sentences(r.claim, docs, 4)
        ent, contra, _ = pipe.judge.entail(r.claim, ev or docs[:2])
        aud.append({"claim_id": r.claim_id, "max_entail": round(float(ent), 3), "contra": round(float(contra), 3)})
    a = pd.DataFrame(aud)
    contaminated = int((a["max_entail"] >= 0.6).sum()) if len(a) else 0
    return a, contaminated


def brier_score(df):
    # binary true/false only
    sub = df[df["label"].isin(["true", "false"])].copy()
    if len(sub) == 0:
        return None
    y = (sub["label"] == "true").astype(int).values
    d = sub["D"].astype(float).values
    return round(float(np.mean((d - y) ** 2)), 4)


def calibration_bins(df, n_bins=10):
    sub = df[df["label"].isin(["true", "false"])].copy()
    if len(sub) == 0:
        return []
    sub["bin"] = pd.cut(sub["D"].astype(float), bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    rows = []
    for b, g in sub.groupby("bin", observed=True):
        if len(g) == 0:
            continue
        rows.append({"bin": str(b), "n": len(g),
                     "mean_pred": round(float(g["D"].mean()), 3),
                     "frac_true": round(float((g["label"] == "true").mean()), 3)})
    return rows


def ablation(df):
    # compare decision rules
    y = df["label"].tolist()
    rules = {}
    # cascade (current)
    rules["cascade"] = df["pred_d"].tolist()
    # D-only
    rules["d_only"] = ["true" if d >= 0.6 else "false" if d < 0.4 else "unverifiable" for d in df["D"]]
    # F-only
    rules["f_only"] = ["true" if f >= 0.95 else "false" if f <= 0.05 else "unverifiable" for f in df["F"]]
    # P-only
    rules["p_only"] = ["true" if p >= 0.95 else "false" if p <= 0.05 else "unverifiable" for p in df["P"]]
    # G-only (using score as single threshold)
    g = df["G"].astype(float).values
    rules["g_only"] = ["true" if v >= 0.6 else "false" if v < 0.4 else "unverifiable" for v in g]
    rows = []
    for name, preds in rules.items():
        acc = accuracy_score(y, preds)
        mf = f1_score(y, preds, average="macro", zero_division=0)
        mfr = float(((df["label"] == "false") & (pd.Series(preds) != "false").values).mean()) if (df["label"] == "false").any() else 0.0
        abst = float((pd.Series(preds) == "unverifiable").mean())
        rows.append({"rule": name, "accuracy": round(acc, 4), "macro_f1": round(mf, 4),
                     "missed_falsehood": round(mfr, 4), "abstention": round(abst, 4)})
    return pd.DataFrame(rows)


def unsupervised(pipe, judge, conditions):
    rows = []
    for q in QUERIES:
        docs = pipe.retriever.retrieve(q)
        ctx = context_of(docs, 4)
        evidence = [{"text": d["text"][:1500], "doc_name": d["doc_name"]} for d in docs[:4]]
        answers = {}
        answers["A"] = pipe.llm.call(f"Question: {q}\nAnswer in at most three sentences.", 0.0, 200)
        answers["B"] = pipe.llm.call(f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer in at most three sentences using only the context.", 0.0, 200)
        samples = [pipe.llm.call(f"Question: {q}\nAnswer in at most three sentences.", 0.7, 200) for _ in range(3)]
        answers["D"] = pipe.answer(q).answer
        answers["C"] = max(samples, key=lambda s: sum(len(set(tokenize(s)) & set(tokenize(x))) for x in samples))
        # RAG triad components
        ctx_rel = []
        for d in docs[:4]:
            e, c, _ = judge.entail(q, [{"text": d["text"][:500], "doc_name": d["doc_name"]}])
            ctx_rel.append(float(e))
        for cond, txt in answers.items():
            cl = [s for s in sentences(txt) if len(s.split()) >= 5]
            unsupported = 0
            for s in cl:
                ent, contra, _ = judge.entail(s, evidence)
                if ent < 0.5:
                    unsupported += 1
            # answer relevance
            ar_ent, _, _ = judge.entail(txt, [{"text": q, "doc_name": "query"}]) if False else (0.0, 0.0, "")
            try:
                ar_ent, _, _ = judge.entail(q, [{"text": txt[:500], "doc_name": "answer"}])
            except Exception:
                ar_ent = 0.0
            rows.append({"query": q, "condition": cond, "answer": txt, "n_claims": len(cl),
                         "unsupported": unsupported,
                         "unsupported_per_answer": round(unsupported / max(1, len(cl)), 3),
                         "answered": int(len(cl) > 0 and "do not know" not in txt.lower() and "cannot" not in txt.lower()),
                         "faithful": int(len(cl) > 0 and unsupported == 0),
                         "context_relevance": round(float(np.mean(ctx_rel)) if ctx_rel else 0.0, 3),
                         "answer_relevance": round(float(ar_ent), 3)})
        print("  q done", flush=True)
    return pd.DataFrame(rows)


QUERIES = [
    "What is the minimum attendance required for end semester examinations?",
    "What is the tuition fee for B.Tech first semester?",
    "What is the last date for fee payment?",
    "How many books can a student borrow from the library?",
    "What is the hostel curfew time?",
    "What is the revaluation fee?",
    "When does the autumn semester registration begin?",
    "What is the late registration fine?",
    "What is the minimum CGPA for scholarship?",
    "How many credits are required for graduation?",
    "What is the examination duration?",
    "What is the hostel fee per semester?",
    "What are the library opening hours?",
    "What is the process for medical leave?",
    "When are the mid semester examinations?",
    "What is the minimum attendance for hostel residents?",
    "What is the fee for backlog examination?",
    "What is the hostel mess charge?",
    "What documents are needed for registration?",
    "What is the grading system?",
    "What is the convocation fee?",
    "What is the maximum number of backlogs allowed?",
    "What is the process for transcript request?",
    "What is the hostel allotment procedure?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    claims = load_claims()
    t0 = time.time()

    if args.mode in ("dev", "all"):
        dev = run_conditions(claims, use_llm=not args.no_llm, split="dev")
        dev.to_csv(RESULTS / "dev_predictions.csv", index=False)
        cfg = {}
        # tune thresholds on dev
        best = None
        for tp in [0.5, 0.55, 0.6, 0.65, 0.7]:
            for tc in [0.55, 0.6, 0.62, 0.65, 0.7]:
                tmp = dev.copy()
                preds = []
                for r in tmp.itertuples():
                    F, P, D, C = r.F, r.P, r.D, r.C
                    if F <= 0.05:
                        lab = "false"
                    elif P <= 0.05:
                        lab = "false"
                    elif F >= 0.95:
                        lab = "true"
                    elif C >= tc:
                        lab = "false"
                    elif D >= tp:
                        lab = "true"
                    else:
                        lab = "unverifiable"
                    preds.append(lab)
                acc = accuracy_score(tmp["label"], preds)
                mf = f1_score(tmp["label"], preds, average="macro", zero_division=0)
                if best is None or mf > best[0]:
                    best = (mf, tp, tc, acc)
        cfg = {"tau_pass": best[1], "tau_contra": best[2],
               "dev_macro_f1": round(best[0], 4), "dev_accuracy": round(best[3], 4),
               "weights": {"w_fact": 0.40, "w_policy": 0.25, "w_ground": 0.35}}
        (DATA / "thresholds.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print("tuned:", cfg, flush=True)

    if args.mode in ("test", "all"):
        test = run_conditions(claims, use_llm=not args.no_llm, split="test")
        test.to_csv(RESULTS / "test_predictions.csv", index=False)
        pipe = Pipeline()
        audit, contaminated = absence_audit(test, pipe)
        audit.to_csv(RESULTS / "absence_audit.csv", index=False)
        mask = (test["label"] == "unverifiable") & test["claim_id"].isin(audit[audit["max_entail"] >= 0.6]["claim_id"])
        clean = test[~mask].copy()
        clean.to_csv(RESULTS / "test_scored.csv", index=False)

        table = [metrics(clean, c) for c in ("pred_a", "pred_b", "pred_c", "pred_d")]
        pd.DataFrame(table).to_csv(RESULTS / "condition_metrics.csv", index=False)
        stats = {
            "n_scored": len(clean), "n_excluded_contaminated": contaminated,
            "mcnemar_d_vs_b": mcnemar(clean["pred_d"], clean["pred_b"], clean["label"]),
            "mcnemar_d_vs_a": mcnemar(clean["pred_d"], clean["pred_a"], clean["label"]),
            "bootstrap_d_minus_b": bootstrap_delta(clean["pred_d"], clean["pred_b"], clean["label"]),
            "confusion_d": confusion_matrix(clean["label"], clean["pred_d"], labels=LABELS).tolist(),
            "labels": LABELS,
            "judge_backend": clean["judge_backend"].iloc[0] if len(clean) else "",
            "seconds": round(time.time() - t0, 1),
            "brier_d": brier_score(clean),
            "calibration_d": calibration_bins(clean),
        }
        (RESULTS / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        pd.DataFrame(stats["calibration_d"]).to_csv(RESULTS / "calibration.csv", index=False)
        print(json.dumps(stats, indent=2), flush=True)

        # subgroup: multi-fact vs single
        sg = []
        for mf in (False, True):
            sub = clean[clean["multi_fact"] == mf]
            if len(sub) == 0:
                continue
            sg.append({"group": "multi_fact" if mf else "single_fact", "n": len(sub),
                       **{k: metrics(sub, c)[k] for c in ("pred_d",) for k in ("accuracy", "macro_f1", "missed_falsehood_rate", "abstention_rate")}})
        # flatten
        sg_rows = []
        for mf in (False, True):
            sub = clean[clean["multi_fact"] == mf]
            if len(sub) == 0:
                continue
            m = metrics(sub, "pred_d")
            sg_rows.append({"group": "multi_fact" if mf else "single_fact", "n": len(sub),
                            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                            "missed_falsehood": m["missed_falsehood_rate"],
                            "abstention": m["abstention_rate"]})
        pd.DataFrame(sg_rows).to_csv(RESULTS / "subgroup_metrics.csv", index=False)

        # ablation
        ab = ablation(clean)
        ab.to_csv(RESULTS / "ablation.csv", index=False)
        print("ablation:\n", ab.to_string(index=False), flush=True)

        # latency
        lat = clean[["lat_a", "lat_b", "lat_c", "lat_d"]].mean().round(3).to_dict()
        (RESULTS / "latency.json").write_text(json.dumps(lat, indent=2), encoding="utf-8")

    if args.mode in ("unsup", "all"):
        pipe = Pipeline()
        judge = pipe.judge
        u = unsupervised(pipe, judge, None)
        u.to_csv(RESULTS / "unsupervised_answers.csv", index=False)
        agg = u.groupby("condition").agg(
            unsupported_per_answer=("unsupported_per_answer", "mean"),
            answered_rate=("answered", "mean"),
            faithful_rate=("faithful", "mean"),
            claims_per_answer=("n_claims", "mean"),
            context_relevance=("context_relevance", "mean"),
            answer_relevance=("answer_relevance", "mean")).round(3).reset_index()
        agg.to_csv(RESULTS / "unsupervised_metrics.csv", index=False)
        print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
