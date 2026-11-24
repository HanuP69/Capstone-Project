import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import date, timedelta

from config import CLAIMS_PER_LABEL, CLAIM_SENT_MAX_WORDS, CLAIM_SENT_MIN_WORDS, DATA, SEED, SPLIT_DEV
from pipeline_text import entities, money_spans, parse_dates, sentences
from quality import has_placeholder, is_well_formed

BOILER = re.compile(
    r"dated|notice|ref\.?\s*no|national institute|advertis|recruitment|registrar|tender|quotation|"
    r"\bbid\b|application (is|are) invited|eligible candidates|walk-?in|annexure|page \d|fax|"
    r"e-?mail|www\.|tele|phone|corrigend|press release|code no|roll no|application form",
    re.I)
POLICY = re.compile(
    r"attendance|condonation|fee|fine|penalt|library|book|hostel|mess|curfew|examination|exam|"
    r"semester|registration|deadline|last date|due|scholarship|cgpa|credit|marks|duration|days|"
    r"hours|percent|%|eligible|required|mandatory|permitted|allowed|prohibited",
    re.I)

def clean(sentence):
    s = re.sub(r"\s+", " ", sentence).strip()
    s = re.sub(r"^[|:\-–\s]+", "", s)
    s = re.sub(r"^(NITJSR|NIT)\S*\s*", "", s)
    return s


def pick_sentences(chunks):
    out = []
    seen = set()
    for c in chunks:
        if c["chunk_type"] != "prose" or not c["issue_date"]:
            continue
        for raw in sentences(c["text"]):
            s = clean(raw)
            w = s.split()
            if not (CLAIM_SENT_MIN_WORDS <= len(w) <= CLAIM_SENT_MAX_WORDS):
                continue
            if BOILER.search(s) or not POLICY.search(s):
                continue
            if s.count("|") > 1 or "..." in s:
                continue
            if has_placeholder(s) or not is_well_formed(s):
                continue
            ent = entities(s)
            if not (ent["money"] or ent["percent"] or ent["days"] or ent["dates"]):
                continue
            key = hashlib.sha1(re.sub(r"[^a-z0-9 ]", "", s.lower()).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append((c, s, ent))
    return out


def money_variant(v, rng):
    for factor in rng.sample([0.5, 0.75, 1.25, 1.5, 2.0, 0.8], 6):
        new = round(v * factor)
        if new != v and new >= 10:
            return new
    return v + 500


def percent_variant(v, rng):
    opts = [o for o in (v - 15, v - 10, v + 10, v + 15) if 0 < o <= 100 and o != v]
    return rng.choice(opts) if opts else (v + 5 if v + 5 <= 100 else max(1, v - 5))


def day_variant(v, rng):
    opts = [o for o in (v - 7, v - 3, v + 3, v + 7, v * 2) if 1 <= o <= 60 and o != v]
    return rng.choice(opts) if opts else v + 5


def date_variant(iso, rng):
    d = date.fromisoformat(iso)
    return (d + timedelta(days=rng.choice([-14, -7, -3, 3, 7, 14]))).isoformat()


def render_date(iso, template):
    d = date.fromisoformat(iso)
    if re.search(r"\d{1,2}[./-]\d{1,2}[./-]\d{4}", template):
        sep = re.search(r"\d{1,2}([./-])\d{1,2}", template).group(1)
        return f"{d.day:02d}{sep}{d.month:02d}{sep}{d.year}"
    if re.search(r"[A-Za-z]{3,9}\s+\d{1,2}", template):
        return d.strftime("%B %d, %Y")
    return d.strftime("%d %B %Y")


def replace_money(sentence, old, new):
    for start, end, value in money_spans(sentence):
        if abs(value - old) < 1e-6:
            rendered = f"{new:,.0f}" if "," in sentence[start:end] else f"{new:.0f}"
            return sentence[:start] + rendered + sentence[end:]
    return None


def replace_percent(sentence, old, new):
    for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*%", sentence):
        if abs(float(m.group(1)) - old) < 1e-6:
            return sentence[:m.start(1)] + f"{new:g}" + sentence[m.end(1):]
    return None


def replace_days(sentence, old, new):
    for m in re.finditer(r"(\d{1,3})\s*(?:working\s*)?days?\b", sentence, re.I):
        if int(m.group(1)) == old:
            return sentence[:m.start(1)] + str(new) + sentence[m.end(1):]
    return None


def replace_date(sentence, old, new):
    for m in re.finditer(
            r"\d{1,2}[./-]\d{1,2}[./-]20\d{2}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s*20\d{2}|[A-Za-z]{3,9}\s+\d{1,2},?\s*20\d{2}",
            sentence):
        parsed = parse_dates(m.group(0))
        if parsed and parsed[0] == old:
            return sentence[:m.start()] + render_date(new, m.group(0)) + sentence[m.end():]
    return None


def perturb(sentence, ent, rng):
    for kind, key, fn in (("money", "money", replace_money), ("percent", "percent", replace_percent),
                          ("days", "days", replace_days), ("date", "dates", replace_date)):
        if not ent[key]:
            continue
        old = ent[key][0]
        if kind == "money":
            new = money_variant(old, rng)
        elif kind == "percent":
            new = percent_variant(old, rng)
        elif kind == "days":
            new = day_variant(old, rng)
        else:
            new = date_variant(old, rng)
        out = fn(sentence, old, new)
        if out and out != sentence:
            return kind, old, new, out
    return None



def main():
    rng = random.Random(SEED)
    chunks = [json.loads(l) for l in open(DATA / "chunks.jsonl", encoding="utf-8")]
    docs = sorted({c["doc_name"] for c in chunks})
    rng.shuffle(docs)
    cut = int(len(docs) * SPLIT_DEV)
    dev_docs = set(docs[:cut])
    pool = pick_sentences(chunks)
    rng.shuffle(pool)
    print("candidate policy sentences:", len(pool))

    claims = []
    counts = defaultdict(int)
    for chunk, sent, ent in pool:
        if counts["true"] >= CLAIMS_PER_LABEL and counts["false"] >= CLAIMS_PER_LABEL:
            break
        base = {
            "category": chunk["category"],
            "doc_name": chunk["doc_name"],
            "source_chunk_id": chunk["chunk_id"],
            "source_date": chunk["issue_date"],
            "ref_number": chunk["ref_number"],
            "split": "dev" if chunk["doc_name"] in dev_docs else "test",
        }
        if counts["true"] < CLAIMS_PER_LABEL:
            claims.append({**base, "claim_id": f"T{counts['true']:03d}", "claim": sent,
                           "label": "true", "method": "verbatim", "kind": "none",
                           "value": None})
            counts["true"] += 1
        if counts["false"] < CLAIMS_PER_LABEL:
            p = perturb(sent, ent, rng)
            if p:
                kind, old, new, text = p
                nums = {round(float(x), 2) for x in re.findall(r"\d+(?:\.\d+)?", chunk["text"])}
                clash = kind != "date" and round(float(new), 2) in nums
                if not clash:
                    claims.append({**base, "claim_id": f"F{counts['false']:03d}", "claim": text,
                                   "label": "false", "method": f"perturbed_{kind}",
                                   "kind": kind, "original_value": old, "value": new})
                    counts["false"] += 1

    true_pool = [c for c in claims if c["label"] == "true"]
    rng2 = random.Random(SEED + 7)
    mf_t = 0
    mf_f = 0
    for i in range(20):
        if mf_t >= 15:
            break
        a = true_pool[(2 * i) % len(true_pool)]
        b = true_pool[(2 * i + 1) % len(true_pool)]
        if a["source_chunk_id"] == b["source_chunk_id"]:
            continue
        sent = a["claim"].rstrip(".") + ", and " + b["claim"][0].lower() + b["claim"][1:].rstrip(".") + "."
        base = dict(a)
        base.update({"claim_id": f"MF{mf_t:03d}T", "claim": sent, "label": "true",
                     "method": "multi_fact_verbatim", "multi_fact": True, "n_facts": 2,
                     "source_chunk_id_b": b["source_chunk_id"], "doc_name_b": b["doc_name"],
                     "original_value": None, "value": None, "kind": "none"})
        claims.append(base)
        perturbed = perturb(b["claim"], entities(b["claim"]), rng2)
        if perturbed:
            kind, old, new, perturbed_b = perturbed
            sent2 = a["claim"].rstrip(".") + ", and " + perturbed_b[0].lower() + perturbed_b[1:].rstrip(".") + "."
            base2 = dict(a)
            base2.update({"claim_id": f"MF{mf_f:03d}F", "claim": sent2, "label": "false",
                          "method": "multi_fact_perturbed", "multi_fact": True, "n_facts": 2,
                          "source_chunk_id_b": b["source_chunk_id"], "doc_name_b": b["doc_name"],
                          "kind": kind, "original_value": old, "value": new})
            claims.append(base2)
            mf_f += 1
        mf_t += 1

    for c in claims:
        c.setdefault("multi_fact", False)
        c.setdefault("n_facts", 1)

    with open(DATA / "claims.jsonl", "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    info = {
        "total": len(claims),
        "dev_docs": len(dev_docs),
        "test_docs": len(docs) - len(dev_docs),
        "dev_claims": sum(1 for c in claims if c["split"] == "dev"),
        "test_claims": sum(1 for c in claims if c["split"] == "test"),
        "labels": {k: sum(1 for c in claims if c["label"] == k) for k in ("true", "false", "unverifiable")},
        "methods": {m: sum(1 for c in claims if c["method"] == m) for m in sorted({c["method"] for c in claims})},
        "categories": {m: sum(1 for c in claims if c["category"] == m) for m in sorted({c["category"] for c in claims})},
    }
    print(json.dumps(info, indent=2))
    for c in claims[:8]:
        print(c["label"], "|", c["claim"][:160])


if __name__ == "__main__":
    main()
