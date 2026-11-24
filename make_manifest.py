import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"


def artifact_commit(source):
    path = source.split(" ")[0]
    commit = git("log", "-1", "--format=%H", "--", path)
    return HEAD if commit in ("", "unknown") else commit


HEAD = git("rev-parse", "HEAD")
DIRTY = bool(git("status", "--porcelain"))


def main():
    manifest = {"generated_by": "make_manifest.py", "head_at_generation": HEAD,
                "dirty_at_generation": DIRTY, "entries": []}

    def add(number, source, script, note=""):
        try:
            number = number.item()
        except AttributeError:
            pass
        manifest["entries"].append({"number": number, "source": source, "script": script,
                                    "artifact_commit": artifact_commit(source), "note": note})

    corpus = json.loads((DATA / "corpus_stats.json").read_text(encoding="utf-8"))
    add(corpus.get("pdfs_total"), "data/corpus_stats.json", "ingest.py")
    add(corpus["pdfs_selected"], "data/corpus_stats.json", "select_corpus.py")
    add(corpus["chunks_dedup"], "data/corpus_stats.json", "ingest.py")
    add(corpus["duplicates_removed"], "data/corpus_stats.json", "ingest.py")
    selection = json.loads((DATA / "selection.json").read_text(encoding="utf-8"))
    add(selection["pool_size"], "data/selection.json", "select_corpus.py")

    index = json.loads((DATA / "index_manifest.json").read_text(encoding="utf-8"))
    add(index["chunks"], "data/index_manifest.json", "index.py")
    add(index["dim"], "data/index_manifest.json", "index.py")

    split = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
    for key in ("total", "constructed_claims", "absent_claims", "dev_claims", "test_claims"):
        add(split[key], "data/split.json", "make_split.py")

    thresholds = json.loads((DATA / "thresholds.json").read_text(encoding="utf-8"))
    add(thresholds["tau_pass"], "data/thresholds.json", "rescore.py (dev)")
    add(thresholds["tau_contra"], "data/thresholds.json", "rescore.py (dev)")

    conditions = pd.read_csv(RESULTS / "condition_metrics.csv")
    for row in conditions.itertuples():
        for column in ("n", "accuracy", "macro_f1", "missed_falsehood_rate", "abstention_rate"):
            add(getattr(row, column), f"results/condition_metrics.csv ({row.condition})", "rescore.py")

    stats = json.loads((RESULTS / "stats.json").read_text(encoding="utf-8"))
    add(stats["n_scored"], "results/stats.json", "rescore.py")
    add(stats["bootstrap_d_minus_b"]["delta"], "results/stats.json", "rescore.py")
    add(stats["brier_d"], "results/stats.json", "rescore.py")

    for name in ("subset_metrics.csv", "ablation.csv", "subgroup_metrics.csv",
                 "quality_audit.csv", "unsupervised_metrics.csv"):
        frame = pd.read_csv(RESULTS / name)
        for column in frame.columns:
            for value in frame[column].dropna():
                if isinstance(value, (int, float)):
                    add(value, f"results/{name}", "rescore.py")

    for record in json.loads((DATA / "records.json").read_text(encoding="utf-8")):
        add(record["value"], "data/records.json", "records.py")

    (ROOT / "artifacts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("entries:", len(manifest["entries"]), "dirty:", DIRTY)


if __name__ == "__main__":
    main()
