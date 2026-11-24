import json
import re
import sys
from pathlib import Path

import pandas as pd

from config import DATA, RESULTS, ROOT

DOCS = ["README.md", "dashboard.html", "TODO.md"]
ALLOW = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "16", "24", "30", "40", "41", "50",
         "59", "60", "79", "80", "100", "120", "150", "180", "200", "240", "512", "768", "1100", "1147", "1365", "2048", "2241"}
NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")


def collect():
    values = set(ALLOW)
    for p in list(RESULTS.glob("*.csv")) + list(DATA.glob("*.csv")):
        df = pd.read_csv(p)
        for col in df.columns:
            for v in df[col].dropna():
                if df[col].dtype.kind in "if":
                    values.add(f"{v}")
                    values.add(f"{v:.1f}")
                    values.add(f"{v:.2f}")
                    values.add(f"{v:.3f}")
                    values.add(f"{v:.4f}")
                else:
                    for m in NUM.findall(str(v)):
                        values.add(m)
    for p in list(RESULTS.glob("*.json")) + list(DATA.glob("*.json")):
        raw = p.read_text(encoding="utf-8")
        for m in NUM.findall(raw):
            values.add(m)
    return values


def scan(path, allowed):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    bad = []
    for m in NUM.finditer(text):
        token = m.group(1)
        if token in allowed:
            continue
        if re.fullmatch(r"20\d{2}", token):
            continue
        if "." in token:
            try:
                if any(abs(float(token) - float(a)) < 5e-4 for a in allowed if "." in a):
                    continue
            except ValueError:
                pass
        start = max(0, m.start() - 45)
        bad.append((token, text[start:m.end() + 45].replace("\n", " ")))
    return bad


def main():
    allowed = collect()
    problems = []
    for name in DOCS:
        bad = scan(ROOT / name, allowed)
        if bad:
            problems.extend((name, tok, ctx) for tok, ctx in bad)
    if problems:
        print(f"{len(problems)} unsupported numbers:")
        for name, tok, ctx in problems[:40]:
            print(f"  {name}: {tok} | ...{ctx}...")
        sys.exit(1)
    print("all numbers in docs trace to generated artifacts")


if __name__ == "__main__":
    main()
