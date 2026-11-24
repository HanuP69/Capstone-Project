import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import BASELINE_NUM_PREDICT, DATA, GEN_MODEL, RESULTS, SEED, UNSUP_SAMPLES
from evaluate import context_of, load_claims, majority_label, parse_label, prompt_a, prompt_b
from pipeline import Pipeline

VERSION = "three_way_v1"
DEFAULT_OUTPUT = RESULTS / "test_predictions_baseline.csv"
DEFAULT_META = RESULTS / "baseline_rerun.json"


def prompt_hashes():
    return {
        "prompt_a_sha256": hashlib.sha256(prompt_a("CLAIM").encode("utf-8")).hexdigest(),
        "prompt_b_sha256": hashlib.sha256(prompt_b("CLAIM", "CONTEXT").encode("utf-8")).hexdigest(),
    }


def save(frame, path, meta, metadata_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    claims = {claim["claim_id"]: claim for claim in load_claims() if claim.get("split") == "test"}
    if args.output.exists() and not args.force:
        frame = pd.read_csv(args.output)
    else:
        frame = pd.read_csv(RESULTS / "test_predictions.csv")
    frame["baseline_prompt"] = frame.get("baseline_prompt", "")
    frame["baseline_prompt"] = frame["baseline_prompt"].fillna("").astype(str)
    for column in ("pred_a", "pred_b", "pred_c", "pred_d"):
        frame[column] = frame[column].astype(object)
    if args.force:
        frame["baseline_prompt"] = ""

    pending = [index for index, row in frame.iterrows()
               if row["claim_id"] in claims and row["baseline_prompt"] != VERSION]
    if args.limit:
        pending = pending[:args.limit]

    pipe = Pipeline()
    started = time.time()
    meta = {
        "version": VERSION,
        "model": GEN_MODEL,
        "seed": SEED,
        "temperature_a_b": 0.0,
        "temperature_c": 0.7,
        "samples_c": UNSUP_SAMPLES,
        "num_predict": BASELINE_NUM_PREDICT,
        **prompt_hashes(),
        "split": "test",
        "claims_total": len(claims),
        "claims_pending": len(pending),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": "",
    }

    for count, index in enumerate(pending, start=1):
        row = frame.loc[index]
        claim = claims[row["claim_id"]]["claim"]
        docs = pipe.retriever.retrieve(claim)
        context = context_of(docs)
        t_a = time.time()
        answer_a = pipe.llm.call(prompt_a(claim), 0.0, BASELINE_NUM_PREDICT)
        lat_a = time.time() - t_a
        t_b = time.time()
        answer_b = pipe.llm.call(prompt_b(claim, context), 0.0, BASELINE_NUM_PREDICT)
        lat_b = time.time() - t_b
        t_c = time.time()
        samples = [pipe.llm.call(prompt_b(claim, context), 0.7, BASELINE_NUM_PREDICT)
                   for _ in range(UNSUP_SAMPLES)]
        lat_c = time.time() - t_c
        frame.at[index, "pred_a"] = parse_label(answer_a)
        frame.at[index, "pred_b"] = parse_label(answer_b)
        frame.at[index, "pred_c"] = majority_label(samples)
        frame.at[index, "lat_a"] = round(lat_a, 3)
        frame.at[index, "lat_b"] = round(lat_b, 3)
        frame.at[index, "lat_c"] = round(lat_c, 3)
        frame.at[index, "baseline_prompt"] = VERSION
        if count % 5 == 0 or count == len(pending):
            meta["claims_completed"] = count
            meta["elapsed_seconds"] = round(time.time() - started, 1)
            save(frame, args.output, meta, args.metadata)
            print(f"{count}/{len(pending)} elapsed={meta['elapsed_seconds']}s", flush=True)

    meta["claims_completed"] = len(pending)
    meta["elapsed_seconds"] = round(time.time() - started, 1)
    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    save(frame, args.output, meta, args.metadata)
    if args.replace and not args.limit:
        os.replace(args.output, RESULTS / "test_predictions.csv")
        print("replaced results/test_predictions.csv", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
