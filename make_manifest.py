import json, hashlib, pathlib, subprocess
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
try:
    _commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
except Exception:
    _commit = "unknown"
manifest = {"generated_by": "make_manifest.py", "git_commit": _commit, "entries": []}
def add(number, source, script, note=""):
    manifest["entries"].append({"number": number, "source": source, "script": script, "note": note})
cs = json.loads((DATA / "corpus_stats.json").read_text())
add(cs["pdfs_selected"], "data/corpus_stats.json", "select_corpus.py")
add(cs["chunks_dedup"], "data/corpus_stats.json", "ingest.py")
add(cs["duplicates_removed"], "data/corpus_stats.json", "ingest.py")
im = json.loads((DATA / "index_manifest.json").read_text())
add(im["chunks"], "data/index_manifest.json", "index.py")
add(im["dim"], "data/index_manifest.json", "index.py")
sp = json.loads((DATA / "split.json").read_text())
add(sp["total"], "data/split.json", "claims.py")
add(sp["dev_claims"], "data/split.json", "claims.py")
add(sp["test_claims"], "data/split.json", "claims.py")
th = json.loads((DATA / "thresholds.json").read_text())
add(th["tau_pass"], "data/thresholds.json", "evaluate.py (dev)")
add(th["tau_contra"], "data/thresholds.json", "evaluate.py (dev)")
cm = pd.read_csv(RESULTS / "condition_metrics.csv")
for _, r in cm.iterrows():
    for col in ["n", "accuracy", "macro_f1", "missed_falsehood_rate", "abstention_rate"]:
        add(r[col], f"results/condition_metrics.csv ({r['condition']})", "evaluate.py (test)")
st = json.loads((RESULTS / "stats.json").read_text())
add(st["n_scored"], "results/stats.json", "evaluate.py (test)")
add(st["bootstrap_d_minus_b"]["delta"], "results/stats.json", "evaluate.py (test)")
add(st["brier_d"], "results/stats.json", "evaluate.py (test)")
ab = pd.read_csv(RESULTS / "ablation.csv")
for _, r in ab.iterrows():
    add(r["accuracy"], f"results/ablation.csv ({r['rule']})", "evaluate.py (test)")
sg = pd.read_csv(RESULTS / "subgroup_metrics.csv")
for _, r in sg.iterrows():
    add(r["accuracy"], f"results/subgroup_metrics.csv ({r['group']})", "evaluate.py (test)")
um = pd.read_csv(RESULTS / "unsupervised_metrics.csv")
for _, r in um.iterrows():
    add(r["unsupported_per_answer"], f"results/unsupervised_metrics.csv ({r['condition']})", "evaluate.py (unsup)")
(ROOT / "artifacts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("entries:", len(manifest["entries"]))