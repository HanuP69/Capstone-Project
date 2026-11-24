import json
import re
from collections import Counter

import fitz

from config import DATA, DOCS, PDF_TARGET
from ingest import classify, parse_date, parse_ref, parse_session

EXCLUDE = ("tender", "boq", "specification", "static_pdfs", "corrigend", "quotation", "bid", "award")
ACADEMIC = re.compile(
    r"attendance|examination|semester|registration|tuition|library|hostel|mess|scholarship|"
    r"cgpa|grade|curriculum|syllabus|placement|medical leave|ordinance|academic|credit", re.I)
JUNK = re.compile(
    r"purchase|retail|mrp|tender|quotation|\bbid\b|supply|rate contract|vendor|consignee|"
    r"indent|registrar|recruitment|security deposit|e-?auction", re.I)

TOPICS = {
    "attendance": (r"attendance|condonation|detention", 14),
    "library": (r"library|book bank|borrow|overdue", 10),
    "hostel": (r"hostel|warden|curfew|hall of residence|mess", 16),
    "fee": (r"tuition fee|semester fee|fee structure|fee payable|academic fee", 16),
    "revaluation": (r"revaluation|re-evaluation|scrutiny", 8),
    "grading": (r"cgpa|grade point|ordinance|academic regulation", 12),
    "scholarship": (r"scholarship|merit.cum.means|fee waiver", 10),
    "placement": (r"placement|internship|recruit", 8),
    "exam": (r"examination|end semester|mid semester|datesheet|date sheet", 16),
    "medical": (r"medical leave|sick leave|medical certificate", 6),
    "registration": (r"registration|late fee|enrolment|enrollment", 10),
    "calendar": (r"academic calendar|semester schedule|commencement", 10),
}


def scan(path):
    doc = fitz.open(path)
    text = ""
    for i in range(min(3, len(doc))):
        text += "\n" + doc[i].get_text()
    doc.close()
    return text


def main():
    files = [p for p in sorted(DOCS.glob("*.pdf")) if not any(x in p.name.lower() for x in EXCLUDE)]
    print(f"scanning {len(files)} pdfs", flush=True)
    rows = []
    for i, p in enumerate(files, start=1):
        try:
            text = scan(p)
        except Exception:
            continue
        if len(text.strip()) < 200:
            continue
        low = text.lower()
        acad = len(ACADEMIC.findall(text))
        junk = len(JUNK.findall(text))
        if acad < 3 or junk > acad:
            continue
        hits = {k: len(re.findall(pat, low)) for k, (pat, _) in TOPICS.items()}
        rows.append({
            "doc_name": p.name,
            "issue_date": parse_date(text),
            "ref_number": parse_ref(text, p.name),
            "session": parse_session(text),
            "category": classify(text, p.name),
            "hits": hits,
            "distinct_topics": sum(1 for v in hits.values() if v > 0),
        })
        if i % 200 == 0:
            print(f"  {i}/{len(files)} clean={len(rows)}", flush=True)

    with open(DATA / "scan.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    chosen = {}
    for topic, (_, quota) in TOPICS.items():
        cands = [r for r in rows if r["hits"][topic] > 0 and r["doc_name"] not in chosen]
        cands.sort(key=lambda r: (r["hits"][topic], r["issue_date"], r["distinct_topics"]), reverse=True)
        for r in cands[:quota]:
            chosen[r["doc_name"]] = topic

    rest = [r for r in rows if r["doc_name"] not in chosen]
    rest.sort(key=lambda r: (r["distinct_topics"], r["issue_date"]), reverse=True)
    target = min(PDF_TARGET, len(rows))
    for r in rest:
        if len(chosen) >= target:
            break
        chosen[r["doc_name"]] = "fill"

    selected = sorted(chosen)
    coverage = Counter(chosen.values())
    payload = {
        "selected": selected,
        "assignment": chosen,
        "topic_coverage": dict(coverage),
        "pool_size": len(rows),
        "docs_with_date": sum(1 for r in rows if r["issue_date"]),
    }
    with open(DATA / "selection.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({k: v for k, v in payload.items() if k != "assignment"}, indent=2))
    print(json.dumps({k: coverage[k] for k in sorted(coverage)}, indent=2))


if __name__ == "__main__":
    main()
