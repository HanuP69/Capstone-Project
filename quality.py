import re

HEDGE = re.compile(
    r"\b(varies?|vary|typically|generally|usually|should|recommend\w*|depends?|consult\w*|"
    r"check your|contact|different (?:institution|university|college|programme|program)|"
    r"may vary|ensure|most accurate|please check|it is advisable|as per (?:the )?(?:institute|university)|"
    r"specific (?:minimum|fee|date|amount|percentage))\b",
    re.I,
)
PLACEHOLDER = re.compile(r"(?:\$X|\[insert|insert date|xxx|<[^>]+>|\bTBD\b)", re.I)
NOTE_TAIL = re.compile(r"\bNote\s*:\s*-?\s*1\.?\s*$", re.I)

def is_hedge(text):
    return bool(HEDGE.search(text or ""))


def has_placeholder(text):
    return bool(PLACEHOLDER.search(text or ""))


def is_fragment(text):
    text = (text or "").strip()
    if not text:
        return True
    if not text[0].isalpha():
        return True
    if text[0].islower():
        return True
    if NOTE_TAIL.search(text):
        return True
    if text.count("(") != text.count(")") or text.count("[") != text.count("]"):
        return True
    if len(text.split()) < 8:
        return True
    if re.match(r"^(?:and|or|but|in|on|at|by|for|to|with|from)\b", text, re.I):
        return True
    return False


def is_well_formed(text):
    return not is_fragment(text)


def is_clean_absent(text):
    return not is_hedge(text) and not has_placeholder(text) and not is_fragment(text)
