import json
import random

from config import DATA, SEED
from evaluate import QUERIES
from pipeline import Judge, LLM, Pipeline, Retriever
from pipeline_text import sentences

EXTRA = [
    "What is the fine for late hostel fee payment at NIT Jamshedpur?",
    "How many credits are required for a minor degree?",
    "What is the dress code for the convocation ceremony?",
    "How can a student appeal against a disciplinary fine?",
    "What is the maximum duration of a Ph.D. programme?",
    "When is the next senate meeting scheduled?",
    "What is the fee for a duplicate degree certificate?",
    "How many working days does a semester have?",
    "What is the procedure for changing a branch after first year?",
    "What scholarship is available for students from Jharkhand?",
    "What is the penalty for plagiarism in a project report?",
    "How many internal evaluations are conducted per semester?",
]


def main():
    rng = random.Random(SEED)
    llm = LLM()
    retriever = Retriever(llm)
    pipe = Pipeline(retriever=retriever, llm=llm)
    questions = QUERIES + EXTRA
    kept = []
    tried = 0
    for q in questions:
        for temp in (0.8, 1.0):
            tried += 1
            try:
                ans = llm.call(f"Student question: {q}\nAnswer confidently in two sentences.", temp, 120)
            except Exception:
                continue
            for s in sentences(ans):
                s = s.strip().rstrip(".")
                if len(s.split()) < 8:
                    continue
                docs = retriever.retrieve(s, k=8)
                ev = retriever.best_sentences(s, docs, limit=3)
                ent, contra, _ = pipe.judge.entail(s, ev or docs[:2])
                if ent < 0.35 and contra < 0.5:
                    kept.append({"claim": s, "label": "unverifiable", "method": "generated_absent",
                                 "source_question": q, "max_entail": round(float(ent), 3),
                                 "split": "dev" if rng.random() < 0.5 else "test"})
            print(f"  {len(kept)} kept / {tried} answers", flush=True)
    seen = set()
    out = []
    for c in kept:
        if c["claim"] in seen:
            continue
        seen.add(c["claim"])
        out.append({**c, "claim_id": f"U{len(out):03d}", "category": "Unsupported",
                    "doc_name": "", "source_chunk_id": "", "source_date": "", "ref_number": "", "kind": "none", "value": None})
    with open(DATA / "claims_absent.jsonl", "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("absent claims:", len(out))


if __name__ == "__main__":
    main()
