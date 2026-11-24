import re

STOP = {"the", "a", "an", "of", "is", "are", "to", "for", "in", "on", "and", "or",
        "be", "will", "at", "by", "with", "as", "it", "that", "this", "your", "you"}
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"}


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP}


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def fact_tokens(text: str) -> set[str]:
    """Numbers + month names: the 'checkable' part of a claim (dates, fees, limits)."""
    return numbers(text) | {t for t in re.findall(r"[a-z]+", text.lower()) if t in MONTHS}
