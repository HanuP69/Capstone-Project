import re
from datetime import date

from config import AS_OF_DATE

TOKEN = re.compile(r"[a-z0-9]+")

CURRENCY = r"(?:rs\.?|inr|rupees?|\u20b9)"
MONEY_TOKEN = re.compile(
    rf"(?:(?:{CURRENCY})\s*([\d][\d,]*(?:\.\d+)?)(?:\s*/-)?"
    rf"|(?<![\d.])([\d][\d,]*(?:\.\d+)?)\s*/-)",
    re.I,
)
PERCENT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent\b)", re.I)
DATE_NUM = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b")
DATE_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*,?\s*(20\d{2})\b", re.I)
DATE_MDY = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(20\d{2})\b", re.I)
YEAR = re.compile(r"\b(20\d{2})\b")
TIME = re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\s*(am|pm|hrs|hours)?\b", re.I)
DAYS = re.compile(r"(\d{1,3})\s*(?:working\s*)?days?\b", re.I)
LPA = re.compile(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|lpa|crore|crores)", re.I)
CGPA = re.compile(r"\b(\d(?:\.\d{1,2})?)\s*(?:/\s*10|point\s+cgpa|cgpa)\b", re.I)
CGPA_PREFIX = re.compile(r"\bcgpa\s*(?:of|:|=)?\s*(\d(?:\.\d{1,2})?)\b", re.I)

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
MULT = {"lakh": 100000, "lakhs": 100000, "lac": 100000, "lacs": 100000,
        "lpa": 100000, "crore": 10000000, "crores": 10000000}


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if len(t) > 1]


def strip_punct(text):
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def sentences(text):
    out = []
    for part in re.split(r"(?<=[.!?])\s+", text):
        part = part.strip()
        if part:
            out.append(part)
    return out


def money_spans(text):
    spans = []
    for m in MONEY_TOKEN.finditer(text):
        group = 1 if m.group(1) is not None else 2
        raw = m.group(group).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if not 10 <= value <= 10_000_000:
            continue
        spans.append((m.start(group), m.end(group), value))
    return spans


def parse_money(text):
    vals = [v for _, _, v in money_spans(text)]
    for m in LPA.finditer(text):
        vals.append(float(m.group(1)) * MULT[m.group(2).lower()])
    return vals


def parse_percent(text):
    return [float(x) for x in PERCENT.findall(text) if 0 < float(x) <= 100]


def parse_dates(text):
    found = []
    for m in DATE_NUM.finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mo > 12 and d <= 12:
            d, mo = mo, d
        try:
            found.append(date(y, mo, d).isoformat())
        except ValueError:
            pass
    for m in DATE_DMY.finditer(text):
        try:
            found.append(date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1))).isoformat())
        except ValueError:
            pass
    for m in DATE_MDY.finditer(text):
        try:
            found.append(date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))).isoformat())
        except ValueError:
            pass
    return found


def parse_days(text):
    return [int(x) for x in DAYS.findall(text) if 0 < int(x) <= 400]


def parse_cgpa(text):
    vals = []
    for m in CGPA.finditer(text):
        v = float(m.group(1))
        if 0 < v <= 10:
            vals.append(v)
    for m in CGPA_PREFIX.finditer(text):
        v = float(m.group(1))
        if 0 < v <= 10:
            vals.append(v)
    return vals


def parse_time(text):
    vals = []
    for h, mi, ap in TIME.findall(text):
        hh = int(h)
        ap = (ap or "").lower()
        if ap == "pm" and hh < 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0
        if 0 <= hh <= 23:
            vals.append(f"{hh:02d}:{mi}")
    return vals


def parse_years(text):
    return [int(y) for y in YEAR.findall(text)]


def entities(text):
    return {
        "money": parse_money(text),
        "percent": parse_percent(text),
        "days": parse_days(text),
        "dates": parse_dates(text),
        "times": parse_time(text),
        "cgpa": parse_cgpa(text),
        "years": parse_years(text),
    }


def reference_date():
    if AS_OF_DATE:
        try:
            return date.fromisoformat(AS_OF_DATE)
        except ValueError:
            pass
    return date.today()


def age_years(iso, ref=None):
    if not iso:
        return None
    ref = ref or reference_date()
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return None
    return max(0.0, (ref - d).days / 365.25)


def money_fmt(v):
    if v is None:
        return "n/a"
    return f"Rs {v:,.0f}"


def overlap_terms(claim, text):
    a = set(tokenize(claim))
    b = set(tokenize(text))
    if not a:
        return 0.0
    return len(a & b) / len(a)
