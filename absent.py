import hashlib
import json
import re

from config import ABSENCE_CONTRA_MAX, DATA, MAX_ENTAIL_FOR_UNSUPPORTED
from evaluate import QUERIES
from pipeline import LLM, Pipeline, Retriever
from pipeline_text import sentences
from quality import has_placeholder, is_hedge, is_well_formed

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


def split_for(question):
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def main():
    llm = LLM()
    retriever = Retriever(llm)
    pipe = Pipeline(retriever=retriever, llm=llm)
    questions = QUERIES + EXTRA
    kept = []
    tried = 0
    for question in questions:
        for temperature in (0.8, 1.0):
            tried += 1
            try:
                answer = llm.call(f"Student question: {question}\nAnswer confidently in two sentences.", temperature, 120)
            except Exception:
                continue
            for sentence in sentences(answer):
                sentence = sentence.strip().rstrip(".")
                if len(sentence.split()) < 8:
                    continue
                if is_hedge(sentence) or has_placeholder(sentence) or not is_well_formed(sentence):
                    continue
                docs = retriever.retrieve(sentence, k=8)
                evidence = retriever.best_sentences(sentence, docs, limit=3)
                entail, contra, _ = pipe.judge.entail(sentence, evidence or docs[:2])
                if entail < MAX_ENTAIL_FOR_UNSUPPORTED and contra < ABSENCE_CONTRA_MAX:
                    kept.append({"claim": sentence, "label": "unverifiable", "method": "generated_absent",
                                 "source_question": question, "max_entail": round(float(entail), 3),
                                 "split": split_for(question)})
            print(f"  {len(kept)} kept / {tried} answers", flush=True)
    seen = set()
    out = []
    for claim in kept:
        key = re.sub(r"\W+", " ", claim["claim"].lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append({**claim, "claim_id": f"U{len(out):03d}", "category": "Unsupported",
                    "doc_name": "", "source_chunk_id": "", "source_date": "", "ref_number": "",
                    "kind": "none", "value": None})
    with open(DATA / "claims_absent.jsonl", "w", encoding="utf-8") as handle:
        for claim in out:
            handle.write(json.dumps(claim, ensure_ascii=False) + "\n")
    print("absent claims:", len(out))


if __name__ == "__main__":
    main()
