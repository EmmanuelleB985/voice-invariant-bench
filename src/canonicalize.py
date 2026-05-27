"""Canonicalize spoken surface forms into typed canonical values.

This is the lynchpin of scoring: if "four fifty p.m." doesn't normalize
to 16:50, you cannot deterministically compare predicted vs expected.

Also provides the inverse helpers (int_to_words, int_to_spelled_digits)
that the renderer uses to build natural surface forms.
"""
from __future__ import annotations
import re
from datetime import datetime
from dateutil import parser as dateparser
from word2number import w2n


_DIGIT_WORDS_MAP = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

_DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five",
                "six", "seven", "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
         "sixty", "seventy", "eighty", "ninety"]


# ---------- Forward: spoken → canonical --------------------------------------

def words_to_int(text: str) -> int | None:
    """Handle 'fourteen', 'one four', 'forty', 'one hundred and twenty'."""
    text = text.lower().strip().replace("-", " ")
    tokens = text.split()
    if all(t in _DIGIT_WORDS_MAP for t in tokens) and len(tokens) > 1:
        return int("".join(str(_DIGIT_WORDS_MAP[t]) for t in tokens))
    try:
        return w2n.word_to_num(text)
    except (ValueError, IndexError):
        if text.isdigit():
            return int(text)
        return None


def canonicalize_time(text: str) -> str | None:
    """'four fifty p.m.' -> '16:50'. Handles 'nine o'clock a.m.' too."""
    text = text.lower().strip()
    # 'nine o'clock a.m.' -> hour-only on-the-hour
    m_oc = re.match(
        r"(?P<h>\w+)\s+o['\u2019]?clock\s*"
        r"(?P<ap>a\.?m\.?|p\.?m\.?|am|pm)?$",
        text,
    )
    if m_oc:
        h = words_to_int(m_oc.group("h"))
        ap = (m_oc.group("ap") or "").replace(".", "")
        if h is not None and 0 <= h <= 23:
            if ap == "pm" and h < 12:
                h += 12
            if ap == "am" and h == 12:
                h = 0
            return f"{h:02d}:00"
    m = re.match(
        r"(?P<h>\w+)\s+(?P<m>\w+)?\s*(?P<ap>a\.?m\.?|p\.?m\.?|am|pm)?$",
        text,
    )
    if m:
        h = words_to_int(m.group("h"))
        mn = words_to_int(m.group("m") or "zero") or 0
        ap = (m.group("ap") or "").replace(".", "")
        if h is not None and 0 <= h <= 23 and 0 <= mn <= 59:
            if ap == "pm" and h < 12:
                h += 12
            if ap == "am" and h == 12:
                h = 0
            return f"{h:02d}:{mn:02d}"
    try:
        dt = dateparser.parse(text)
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return None


def canonicalize_date(text: str, ref: datetime | None = None) -> str | None:
    try:
        dt = dateparser.parse(text, default=ref or datetime(2026, 1, 1))
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def canonicalize_amount(text: str) -> float | None:
    text = text.lower().strip()
    text = re.sub(r"[$£€,]", "", text)
    text = re.sub(r"\b(dollars?|pounds?|euros?|usd|gbp|eur)\b", "", text)
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    n = words_to_int(text)
    return float(n) if n is not None else None


def canonicalize_postcode(text: str) -> str | None:
    """UK / US postcodes. 'S W one A, one A A' -> 'SW1A 1AA'."""
    tokens = re.split(r"[\s,.]+", text.lower().strip())
    out = []
    for t in tokens:
        if t in _DIGIT_WORDS_MAP:
            out.append(str(_DIGIT_WORDS_MAP[t]))
        elif t == "oh":
            out.append("0")
        else:
            out.append(t.upper())
    cleaned = "".join(out)
    if re.match(r"^[A-Z]{1,2}\d[A-Z\d]?\d[A-Z]{2}$", cleaned):
        return cleaned[:-3] + " " + cleaned[-3:]
    if re.match(r"^\d{5}(-?\d{4})?$", cleaned):
        return cleaned
    return None


def canonicalize_code(text: str) -> str | None:
    """'A B dash one two nine four dash Z' -> 'AB-1294-Z'."""
    out = []
    for tok in re.split(r"[\s,]+", text.lower()):
        if tok in {"dash", "hyphen", "minus"}:
            out.append("-")
        elif tok in _DIGIT_WORDS_MAP:
            out.append(str(_DIGIT_WORDS_MAP[tok]))
        elif len(tok) == 1 and tok.isalpha():
            out.append(tok.upper())
        elif tok.isalnum():
            out.append(tok.upper())
    s = "".join(out)
    return s if s else None


def canonicalize(inv_type: str, text: str,
                 ref_date: datetime | None = None):
    fn = {
        "time": canonicalize_time,
        "date": lambda t: canonicalize_date(t, ref_date),
        "amount": canonicalize_amount,
        "quantity": words_to_int,
        "street_number": words_to_int,
        "postcode": canonicalize_postcode,
        "code": canonicalize_code,
        "recipient_id": canonicalize_code,
        "order_id": canonicalize_code,
    }.get(inv_type)
    if fn is None:
        return text.strip()
    return fn(text)


# ---------- Inverse: canonical → spoken --------------------------------------

def int_to_words(n: int) -> str:
    if n < 0:
        return f"negative {int_to_words(-n)}"
    if n < 10:
        return _DIGIT_WORDS[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        if n % 10 == 0:
            return _TENS[n // 10]
        return f"{_TENS[n // 10]} {_DIGIT_WORDS[n % 10]}"
    if n < 1000:
        h, r = divmod(n, 100)
        if r == 0:
            return f"{_DIGIT_WORDS[h]} hundred"
        return f"{_DIGIT_WORDS[h]} hundred and {int_to_words(r)}"
    return str(n)


def int_to_spelled_digits(n: int) -> str:
    return " ".join(_DIGIT_WORDS[int(d)] for d in str(abs(n)))


def time_to_spoken(h24: int, m: int) -> str:
    """16:50 -> 'four fifty p.m.'"""
    ampm = "p.m." if h24 >= 12 else "a.m."
    h = h24 % 12 or 12
    if m == 0:
        return f"{int_to_words(h)} o'clock {ampm}"
    if m == 15:
        return f"quarter past {int_to_words(h)} {ampm}"
    if m == 30:
        return f"half past {int_to_words(h)} {ampm}"
    return f"{int_to_words(h)} {int_to_words(m)} {ampm}"
