import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from config import DATA, RESULTS


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_frame(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def records(frame):
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def main():
    corpus = read_json(DATA / "corpus_stats.json")
    split = read_json(DATA / "split.json")
    stats = read_json(RESULTS / "headline_stats.json")
    headline_ablation = read_frame(RESULTS / "headline_ablation.csv")
    headline_subgroup = read_frame(RESULTS / "headline_subgroup.csv")
    headline_latency = read_json(RESULTS / "headline_latency.json")
    records_data = read_json(DATA / "records.json")
    quality_frame = read_frame(RESULTS / "quality_audit.csv")
    quality = records(quality_frame)
    quality_map = {row["metric"]: row["value"] for row in quality}
    claims_frame = read_frame(RESULTS / "test_predictions.csv")
    if not claims_frame.empty:
        claims_frame = claims_frame[claims_frame["source_chunk_id"].fillna("").astype(str).str.len() > 0]
    columns = ["claim_id", "claim", "label", "pred_d", "F", "P", "D", "C", "G",
               "decision", "fact_note", "source_chunk_id", "multi_fact"]
    claims = records(claims_frame[[column for column in columns if column in claims_frame.columns]]) if not claims_frame.empty else []
    subset_frame = read_frame(RESULTS / "subset_metrics.csv")
    metrics_frame = subset_frame[subset_frame["subset"] == "real_chunk"] if not subset_frame.empty else pd.DataFrame()

    limitations = [
        f"{int(quality_map.get('test_real_chunk_fragment', 0))} of {int(quality_map.get('test_real_chunk', 0))} source-backed test claims are fragments (digit, bullet or unmatched bracket).",
        f"{int(quality_map.get('test_real_chunk_hedge', 0))} source-backed test claims match hedge/refusal patterns.",
        "A/B/C predictions were rerun with three-way TRUE/FALSE/UNVERIFIABLE prompts; metadata is in results/baseline_rerun.json.",
        "G-only accuracy exceeds cascade on the source-backed subset; the cascade is not established as the best rule.",
        f"McNemar D vs B p = {stats.get('mcnemar_d_vs_b', {}).get('p_value', '')}; this is not below 0.05.",
        f"The corpus is {corpus.get('pdfs_selected', 0)} of {len(list((ROOT / 'documents').glob('*.pdf'))) if (ROOT / 'documents').exists() else 0} PDFs. Scanned PDFs without a text layer are skipped.",
        "Records are limited to values with support at least 2 and an exact source phrase; ambiguous records are dropped.",
        "Unsupervised scores are entailment-based, not human judgements. Only one policy rule is active.",
        "Supervised NITJSR evaluation with independent human labels is deferred.",
    ]

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_mode": "Source-backed NITJ circular claims · generated probes excluded · no synthetic benchmark",
        "corpus": corpus,
        "split": split,
        "stats": stats,
        "metrics": records(metrics_frame),
        "subsets": records(subset_frame),
        "ablation": records(headline_ablation),
        "subgroup": records(headline_subgroup),
        "unsupervised": records(read_frame(RESULTS / "unsupervised_metrics.csv")),
        "quality": quality,
        "records": records_data,
        "claims": claims,
        "latency": headline_latency,
        "limitations": limitations,
        "condition_names": {
            "pred_a": "A plain LLM",
            "pred_b": "B RAG",
            "pred_c": "C RAG self-consistency",
            "pred_d": "D verified pipeline",
        },
    }

    dist = HERE / "dist"
    for obsolete in (dist / "assets", dist / "data", dist / "manus-routes.json"):
        if obsolete.is_dir():
            shutil.rmtree(obsolete)
        elif obsolete.exists():
            obsolete.unlink()
    (dist / "data").mkdir(parents=True, exist_ok=True)
    (dist / "data" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    template = (HERE / "src" / "template.html").read_text(encoding="utf-8")
    styles = (HERE / "src" / "styles.css").read_text(encoding="utf-8")
    app = (HERE / "src" / "app.js").read_text(encoding="utf-8")
    payload = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace("__STYLES__", styles).replace("__APP__", app).replace("__SUMMARY__", payload)
    (dist / "index.html").write_text(html, encoding="utf-8")
    print("wrote dashboard/dist/index.html and dashboard/dist/data/summary.json")


if __name__ == "__main__":
    main()
