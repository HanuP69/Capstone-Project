import json
import re
from collections import Counter, defaultdict

from rank_bm25 import BM25Okapi

from config import DATA, RECORD_MIN_SUPPORT
from pipeline_text import parse_dates, parse_days, parse_money, parse_percent, tokenize

SPEC = [
    {
        "key": "attendance_min",
        "description": "Minimum attendance required for end-semester examinations",
        "val_type": "percent",
        "operator": ">=",
        "phrases": ["minimum attendance", "attendance requirement"],
        "aliases": ["minimum attendance", "attendance requirement"],
    },
    {
        "key": "late_registration_fee",
        "description": "Late semester registration fee",
        "val_type": "currency",
        "operator": "=",
        "phrases": ["late fine", "late fee", "late registration fee"],
        "aliases": ["late registration fee", "late fee", "late fine"],
    },
    {
        "key": "btech_total_fee",
        "description": "B.Tech first-semester total fee (tuition, institute and one-time payment)",
        "val_type": "currency",
        "operator": "=",
        "phrases": ["Total Fee (Tuition Fee"],
        "aliases": ["b.tech total fee", "total fee"],
    },
    {
        "key": "hostel_fee",
        "description": "Hostel fee per semester",
        "val_type": "currency",
        "operator": "=",
        "phrases": ["hostel fee"],
        "aliases": ["hostel fee"],
    },
]


def load_chunks():
    return [json.loads(line) for line in open(DATA / "chunks.jsonl", encoding="utf-8")]


def extract_all(kind, text):
    if kind == "currency":
        return parse_money(text)
    if kind == "percent":
        return parse_percent(text)
    if kind == "count_days":
        return parse_days(text)
    if kind == "date":
        return parse_dates(text)
    return []


def phrase_regex(phrase):
    return re.compile(r"\s+".join(re.escape(part) for part in phrase.split()), re.I)


def values_near(text, phrase, kind, window=180):
    pattern = phrase_regex(phrase)
    for match in pattern.finditer(text):
        segment = text[match.end():match.end() + window]
        values = extract_all(kind, segment)
        if values:
            yield values[0], match.start(), match.end()


def sane(key, value):
    if key == "attendance_min":
        return isinstance(value, (int, float)) and 50 <= value <= 100
    if key in ("late_registration_fee", "btech_total_fee", "hostel_fee"):
        return isinstance(value, (int, float)) and value >= 100
    return True


def main():
    chunks = load_chunks()
    corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    records = []
    for spec in SPEC:
        hits = []
        for index, chunk in enumerate(chunks):
            for phrase in spec["phrases"]:
                for value, start, _ in values_near(chunk["text"], phrase, spec["val_type"]):
                    if sane(spec["key"], value):
                        query_score = max((bm25.get_scores(tokenize(phrase))[index] for phrase in spec["phrases"]), default=0.0)
                        hits.append((value, chunk, query_score, start, phrase))
                        break
                else:
                    continue
                break
        if not hits:
            continue
        votes = Counter(value for value, _, _, _, _ in hits)
        value, support = votes.most_common(1)[0]
        if support < RECORD_MIN_SUPPORT:
            continue
        if len(votes) > 1 and votes.most_common(2)[0][1] == votes.most_common(2)[1][1]:
            continue
        best = max((hit for hit in hits if hit[0] == value), key=lambda hit: hit[2])
        _, chunk, _, start, phrase = best
        snippet = re.sub(r"\s+", " ", chunk["text"][max(0, start - 80):start + 260])
        records.append({
            "key": spec["key"],
            "description": spec["description"],
            "value": value,
            "operator": spec["operator"],
            "val_type": spec["val_type"],
            "aliases": spec["aliases"],
            "support": support,
            "candidates": sorted(votes),
            "source_chunk_id": chunk["chunk_id"],
            "source_doc": chunk["doc_name"],
            "source_date": chunk["issue_date"],
            "source_phrase": phrase,
            "evidence": snippet,
        })
    with open(DATA / "records.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    for record in records:
        print(f"{record['key']:24s} {record['operator']:2s} {str(record['value']):10s} "
              f"support={record['support']} doc={record['source_doc']} date={record['source_date']}")
    print("records:", len(records), "of", len(SPEC))


if __name__ == "__main__":
    main()
