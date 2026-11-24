import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import date

import fitz

from config import CHUNK_OVERLAP_SENTENCES, CHUNK_WORDS, DATA, DOCS, INGEST_CLASSIFY_LIMIT, INGEST_HEAD_LIMIT, INGEST_REF_LIMIT, INGEST_SESSION_LIMIT, MIN_CHUNK_WORDS, PDF_TARGET, TABLE_MAX_WORDS

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}

EXCLUDE = ("tender", "boq", "specification", "static_pdfs", "corrigend", "quotation", "bid")
QUOTA = {"dean_office_notices": 40, "notices": 85, "department_notices": 30}

CATEGORY_HINTS = {
    "Exam Dates": ["examination", "exam", "mid sem", "end sem", "datesheet", "date sheet", "admit card", "result"],
    "Fee Deadlines": ["fee", "tuition", "dues", "payment", "fine", "refund", "mess charge", "scholarship amount"],
    "Hostel Rules": ["hostel", "curfew", "warden", "hall of residence", "boarder", "mess", "gate pass"],
    "Syllabus Queries": ["syllabus", "credit", "cgpa", "curriculum", "course structure", "branch", "programme", "program"],
    "Circulars": ["circular", "notice", "ordinance", "guideline", "attendance", "library", "placement"],
}

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
DATE_NUM = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b")
DATE_DMY = re.compile(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s*(20\d{2})\b")
DATE_MDY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})\s*,?\s*(20\d{2})\b")
REF_PATTERNS = [
    re.compile(r"(NITJSR\s*/\s*[A-Za-z0-9]+(?:\s*/\s*[A-Za-z0-9.\-]+){1,4})"),
    re.compile(r"(NIT\s*/\s*[A-Za-z0-9]+(?:\s*/\s*[A-Za-z0-9.\-]+){1,4})"),
    re.compile(r"(NOTICE\s*[-\u2013]\s*[A-Z0-9/\-\u2013 ]{2,40})"),
]
SESSION = re.compile(r"\b(20\d{2})\s*[-\u2013/]\s*(20\d{2}|\d{2})\b")


def select_pdfs():
    sel = DATA / "selection.json"
    if sel.exists():
        return json.loads(sel.read_text(encoding="utf-8"))["selected"]
    names = sorted(p.name for p in DOCS.glob("*.pdf"))
    names = [n for n in names if not any(x in n.lower() for x in EXCLUDE)]
    buckets = defaultdict(list)
    for n in names:
        for key in QUOTA:
            if n.startswith(key):
                buckets[key].append(n)
                break
    chosen = []
    for key, quota in QUOTA.items():
        lst = buckets.get(key, [])
        if not lst:
            continue
        step = max(1, len(lst) // quota)
        chosen.extend(lst[::step][:quota])
    rest = [n for n in names if n not in chosen]
    for n in rest:
        if len(chosen) >= PDF_TARGET:
            break
        chosen.append(n)
    chosen = sorted(set(chosen))[:PDF_TARGET]
    return chosen


def norm_ws(text):
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


def parse_date(text):
    head = text[:INGEST_HEAD_LIMIT]
    m = DATE_NUM.search(head)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mo > 12 and d <= 12:
            d, mo = mo, d
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            pass
    m = DATE_DMY.search(head)
    if m and m.group(2).lower()[:3] in {k[:3] for k in MONTHS}:
        mo = next(v for k, v in MONTHS.items() if k[:3] == m.group(2).lower()[:3])
        try:
            return date(int(m.group(3)), mo, int(m.group(1))).isoformat()
        except ValueError:
            pass
    m = DATE_MDY.search(head)
    if m and m.group(1).lower()[:3] in {k[:3] for k in MONTHS}:
        mo = next(v for k, v in MONTHS.items() if k[:3] == m.group(1).lower()[:3])
        try:
            return date(int(m.group(3)), mo, int(m.group(2))).isoformat()
        except ValueError:
            pass
    return ""


def parse_ref(text, name):
    head = text[:INGEST_REF_LIMIT]
    for pat in REF_PATTERNS:
        m = pat.search(head)
        if m:
            return norm_ws(m.group(1)).replace(" ", "")
    m = re.search(r"([A-Z]{2,}/\S*\d{2,}\S*)", head)
    if m:
        return m.group(1)[:40]
    return ""


def parse_session(text):
    m = SESSION.search(text[:INGEST_SESSION_LIMIT])
    if not m:
        return ""
    a, b = m.group(1), m.group(2)
    if len(b) == 2:
        b = a[:2] + b
    return f"{a}-{b}"


def classify(text, name):
    low = (text[:INGEST_CLASSIFY_LIMIT] + " " + name).lower()
    scores = {k: sum(low.count(h) for h in hints) for k, hints in CATEGORY_HINTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


def table_markdown(rows):
    rows = [[norm_ws(str(c)) if c is not None else "" for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = [c or f"col{i + 1}" for i, c in enumerate(rows[0])]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def split_sentences(text):
    parts = []
    for block in text.split("\n"):
        block = norm_ws(block)
        if not block:
            continue
        parts.extend(s for s in SENT_SPLIT.split(block) if s.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(sentences, target_words, overlap_sentences):
    chunks = []
    cur = []
    words = 0
    for s in sentences:
        n = len(s.split())
        if words + n > target_words and cur:
            chunks.append(" ".join(cur))
            keep = cur[-overlap_sentences:] if overlap_sentences else []
            cur = list(keep)
            words = sum(len(x.split()) for x in cur)
        cur.append(s)
        words += n
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) >= MIN_CHUNK_WORDS]


def repeated_lines(pages):
    counter = Counter()
    for lines in pages:
        for line in set(lines):
            counter[line] += 1
    limit = max(2, int(0.4 * len(pages)))
    return {line for line, n in counter.items() if n >= limit and len(line) < 90}


def doc_date_family(ref, name):
    if ref:
        parts = ref.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
    base = re.sub(r"_\d+\.pdf$", "", name)
    return base


def process_pdf(path):
    doc = fitz.open(path)
    pages_text = []
    tables = []
    for pno in range(len(doc)):
        page = doc[pno]
        raw = page.get_text()
        pages_text.append([norm_ws(x) for x in raw.split("\n") if norm_ws(x)])
        table_boxes = []
        try:
            found = page.find_tables()
            for t in getattr(found, "tables", []):
                md = table_markdown(t.extract())
                if md:
                    tables.append((pno + 1, md, t.bbox))
                    table_boxes.append(t.bbox)
        except Exception:
            pass
        blocks = []
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
            if not text.strip():
                continue
            inside = False
            for (tx0, ty0, tx1, ty1) in table_boxes:
                ix = max(0, min(x1, tx1) - max(x0, tx0))
                iy = max(0, min(y1, ty1) - max(y0, ty0))
                if ix * iy > 0.6 * max(1, (x1 - x0) * (y1 - y0)):
                    inside = True
                    break
            if not inside:
                blocks.append((y0, text))
        blocks.sort(key=lambda x: x[0])
        pages_text[pno] = [norm_ws(x[1]) for x in blocks if norm_ws(x[1])]
    first = "\n".join(pages_text[0]) if pages_text else ""
    meta = {
        "doc_name": path.name,
        "issue_date": parse_date(first),
        "ref_number": parse_ref(first, path.name),
        "session": parse_session(first),
        "category": classify(first, path.name),
        "n_pages": len(pages_text),
    }
    doc.close()
    boiler = repeated_lines(pages_text)
    chunks = []
    for pno, lines in enumerate(pages_text, start=1):
        kept = [ln for ln in lines if ln not in boiler and len(ln) > 2]
        sentences = split_sentences("\n".join(kept))
        for idx, body in enumerate(chunk_sentences(sentences, CHUNK_WORDS, CHUNK_OVERLAP_SENTENCES), start=1):
            chunks.append({
                "doc_name": path.name,
                "page": pno,
                "chunk_type": "prose",
                "text": body,
                "n_words": len(body.split()),
            })
    for pno, md, _ in tables:
        words = md.split()
        if len(words) > TABLE_MAX_WORDS:
            rows = md.split("\n")
            head = rows[:2]
            block = []
            for row in rows[2:]:
                block.append(row)
                if len(" ".join(head + block).split()) > TABLE_MAX_WORDS:
                    chunks.append({"doc_name": path.name, "page": pno, "chunk_type": "table",
                                   "text": "\n".join(head + block[:-1]), "n_words": len(" ".join(head + block[:-1]).split())})
                    block = [row]
            if block:
                chunks.append({"doc_name": path.name, "page": pno, "chunk_type": "table",
                               "text": "\n".join(head + block), "n_words": len(" ".join(head + block).split())})
        else:
            chunks.append({"doc_name": path.name, "page": pno, "chunk_type": "table", "text": md, "n_words": len(words)})
    for c in chunks:
        c.update({
            "category": meta["category"],
            "issue_date": meta["issue_date"],
            "ref_number": meta["ref_number"],
            "session": meta["session"],
            "family": doc_date_family(meta["ref_number"], path.name),
        })
    return meta, chunks


def main():
    names = select_pdfs()
    print(f"selected {len(names)} pdfs", flush=True)
    docs, all_chunks, failures = [], [], []
    for i, name in enumerate(names, start=1):
        try:
            meta, chunks = process_pdf(DOCS / name)
        except Exception as exc:
            failures.append({"doc_name": name, "error": repr(exc)})
            continue
        meta["n_chunks"] = len(chunks)
        docs.append(meta)
        all_chunks.extend(chunks)
        if i % 25 == 0:
            print(f"  {i}/{len(names)} docs, {len(all_chunks)} chunks", flush=True)

    seen = {}
    dedup_chunks = []
    duplicates = 0
    for c in all_chunks:
        body = re.sub(r"\s+", " ", c["text"].lower()).strip()
        h = hashlib.sha1(body.encode("utf-8")).hexdigest()
        if h in seen:
            duplicates += 1
            continue
        seen[h] = True
        c["hash"] = h[:12]
        dedup_chunks.append(c)

    latest = {}
    for c in dedup_chunks:
        key = (c["family"], c["category"])
        if c["issue_date"] and (key not in latest or c["issue_date"] > latest[key]):
            latest[key] = c["issue_date"]
    for c in dedup_chunks:
        key = (c["family"], c["category"])
        same = [x for x in dedup_chunks if (x["family"], x["category"]) == key]
        dates = sorted({x["issue_date"] for x in same if x["issue_date"]})
        c["is_latest"] = int(bool(c["issue_date"]) and c["issue_date"] == latest.get(key, ""))
        c["family_docs"] = len({x["doc_name"] for x in same})
        c["family_date_span"] = [dates[0], dates[-1]] if dates else []

    for i, c in enumerate(dedup_chunks):
        c["chunk_id"] = f"c{i:05d}"

    with open(DATA / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in dedup_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(DATA / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    words = [c["n_words"] for c in dedup_chunks if c["chunk_type"] == "prose"]
    stats = {
        "pdfs_total": len(list(DOCS.glob("*.pdf"))),
        "pdfs_selected": len(names),
        "pdfs_parsed": len(docs),
        "failures": failures,
        "chunks_raw": len(all_chunks),
        "chunks_dedup": len(dedup_chunks),
        "duplicates_removed": duplicates,
        "prose_chunks": sum(1 for c in dedup_chunks if c["chunk_type"] == "prose"),
        "table_chunks": sum(1 for c in dedup_chunks if c["chunk_type"] == "table"),
        "with_date": sum(1 for c in dedup_chunks if c["issue_date"]),
        "with_ref": sum(1 for c in dedup_chunks if c["ref_number"]),
        "latest_chunks": sum(c["is_latest"] for c in dedup_chunks),
        "median_prose_words": statistics.median(words) if words else 0,
        "max_prose_words": max(words) if words else 0,
        "categories": dict(Counter(c["category"] for c in dedup_chunks)),
        "date_range": [min((c["issue_date"] for c in dedup_chunks if c["issue_date"]), default=""),
                       max((c["issue_date"] for c in dedup_chunks if c["issue_date"]), default="")],
    }
    with open(DATA / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2)[:1500])


if __name__ == "__main__":
    main()
