import json
import re
from collections import Counter

from rank_bm25 import BM25Okapi

from config import DATA
from pipeline_text import tokenize

MONEY = re.compile(r"(?:rs\.?|inr|rupees?)\s*([\d][\d,]*)(?:\s*/-)?", re.I)
PERCENT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
TIME = re.compile(r"\b([01]?\d|2[0-3])[:.](\d{2})\s*(am|pm|hrs|hours)?\b", re.I)
DATE_ANY = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b")
DAYS = re.compile(r"(\d{1,3})\s*(?:working\s*)?days?\b", re.I)
BOOKS = re.compile(r"(\d{1,2})\s*(?:books?|volumes?)", re.I)

SPEC = [
    ("attendance_min", "Minimum attendance required", "percent", ">=",
     ["minimum attendance", "attendance of", "attendance requirement", "condonation"]),
    ("library_books", "Books issued per student", "count_books", "<=",
     ["books", "borrow", "issued"]),
    ("library_days", "Library loan period", "count_days", "<=",
     ["loan period", "issued for", "days", "overdue"]),
    ("library_fine", "Library overdue fine per day", "currency", "<=",
     ["overdue", "fine per day", "late return"]),
    ("revaluation_days", "Revaluation application window", "count_days", "<=",
     ["revaluation", "re-evaluation", "scrutiny"]),
    ("revaluation_fee", "Revaluation fee per subject", "currency", "=",
     ["revaluation fee", "fee for revaluation", "scrutiny fee"]),
    ("late_registration_fee", "Late registration fee", "currency", "=",
     ["late fee", "late registration", "fine for late"]),
    ("tuition_fee", "B.Tech tuition fee", "currency", "=",
     ["tuition fee", "semester fee", "academic fee"]),
    ("hostel_fee", "Hostel fee", "currency", "=",
     ["hostel fee", "hostel charge", "room rent"]),
    ("mess_fee", "Mess fee", "currency", "=",
     ["mess fee", "mess charge", "mess advance"]),
    ("min_cgpa", "Minimum CGPA requirement", "number", ">=",
     ["minimum cgpa", "cgpa of", "cgpa required"]),
    ("probation_cgpa", "CGPA triggering probation", "number", "<=",
     ["probation", "academic probation"]),
    ("scholarship_income", "Scholarship income ceiling", "currency", "=",
     ["family income", "income ceiling", "income of"]),
    ("medical_leave_days", "Medical leave allowance", "count_days", ">=",
     ["medical leave", "sick leave"]),
    ("curfew_time", "Hostel curfew time", "time", "=",
     ["curfew", "entry time", "gate closing"]),
    ("registration_deadline", "Semester registration deadline", "date", "=",
     ["registration", "last date", "deadline"]),
]


def load_chunks():
    return [json.loads(l) for l in open(DATA / "chunks.jsonl", encoding="utf-8")]


def money_value(text):
    vals = []
    for m in MONEY.finditer(text):
        raw = m.group(1).replace(",", "")
        if raw.isdigit():
            v = int(raw)
            if 10 <= v <= 10_000_000:
                vals.append(v)
    return vals


def extract(kind, text):
    if kind == "currency":
        vals = money_value(text)
        return vals[0] if vals else None
    if kind == "percent":
        vals = [float(x) for x in PERCENT.findall(text)]
        vals = [v for v in vals if 0 < v <= 100]
        return vals[0] if vals else None
    if kind == "count_days":
        vals = [int(x) for x in DAYS.findall(text)]
        vals = [v for v in vals if 0 < v <= 400]
        return vals[0] if vals else None
    if kind == "count_books":
        vals = [int(x) for x in BOOKS.findall(text)]
        vals = [v for v in vals if 0 < v <= 40]
        return vals[0] if vals else None
    if kind == "number":
        vals = re.findall(r"\b(\d{1,2}(?:\.\d{1,2})?)\s*(?:/\s*10|point)", text)
        vals = [float(x) for x in vals if 0 < float(x) <= 10]
        return vals[0] if vals else None
    if kind == "time":
        hits = TIME.findall(text)
        for h, mi, ap in hits:
            hh, mm = int(h), int(mi)
            if ap.lower() == "pm" and hh < 12:
                hh += 12
            if ap.lower() == "am" and hh == 12:
                hh = 0
            if 0 <= hh <= 23:
                return f"{hh:02d}:{mm}"
        return None
    if kind == "date":
        m = DATE_ANY.search(text)
        if m:
            return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
        from ingest import parse_date
        return parse_date(text)
    return None


def sane(key, val):
    if val is None:
        return False
    if key in ("attendance_min",):
        return isinstance(val, (int, float)) and 50 <= val <= 100
    if key in ("library_books",):
        return isinstance(val, int) and 1 <= val <= 20
    if key in ("library_days", "revaluation_days", "medical_leave_days"):
        return isinstance(val, int) and 7 <= val <= 120
    if key in ("library_fine", "revaluation_fee", "late_registration_fee",
               "tuition_fee", "hostel_fee", "mess_fee", "scholarship_income"):
        return isinstance(val, int) and val >= 100
    if key in ("min_cgpa", "probation_cgpa"):
        return isinstance(val, (int, float)) and 0 < val <= 10
    if key in ("curfew_time",):
        return isinstance(val, str) and ":" in val
    if key in ("registration_deadline",):
        return isinstance(val, str) and len(val) == 10
    return True


def main():
    chunks = load_chunks()
    corpus = [(tokenize(c["text"]), c) for c in chunks]
    bm25 = BM25Okapi([t for t, _ in corpus])
    records = []
    for key, desc, kind, op, queries in SPEC:
        scores = None
        for q in queries:
            s = bm25.get_scores(tokenize(q))
            scores = s if scores is None else scores + s
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:8]
        found = []
        for i in order:
            c = chunks[i]
            val = extract(kind, c["text"])
            if val is None or not sane(key, val):
                continue
            found.append((val, c, scores[i]))
        if not found or len(found) < 2:
            continue
        votes = Counter(v for v, _, _ in found)
        value, count = votes.most_common(1)[0]
        best = max((f for f in found if f[0] == value), key=lambda f: f[2])
        records.append({
            "key": key,
            "description": desc,
            "value": value,
            "operator": op,
            "val_type": kind,
            "support": count,
            "candidates": sorted({v for v, _, _ in found}),
            "source_chunk_id": best[1]["chunk_id"],
            "source_doc": best[1]["doc_name"],
            "source_date": best[1]["issue_date"],
            "evidence": best[1]["text"][:600],
        })
    with open(DATA / "records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    for r in records:
        print(f"{r['key']:24s} {r['operator']:2s} {str(r['value']):12s} support={r['support']} doc={r['source_doc']} date={r['source_date']}")
    print("records:", len(records), "of", len(SPEC))


if __name__ == "__main__":
    main()
