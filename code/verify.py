"""Answer extraction + correctness checking.

Checking order (first hit wins):
  1. normalised string equality   (handles "(a+5)(b+2)" vs "(a + 5)(b + 2)")
  2. numeric equality             (handles "0.5" vs "1/2", "1600" vs "1,600")
  3. math_verify symbolic equality, with LaTeX delimiters added

NOTE: math_verify.parse() returns an EMPTY result for bare strings like "9901" (it expects LaTeX
delimiters), and verify([], []) is False -- which silently scored every correct MATH-500 answer as
wrong. Hence the string/numeric fast paths first, and the "$...$" wrapping for math_verify.
"""
from __future__ import annotations
import re

_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_FALLBACK_INT = re.compile(r"(?:final answer|answer)\b[^0-9\-]{0,15}(-?\d{1,4})", re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    """Last \\boxed{...}; fall back to 'answer is <int>' for integer tasks."""
    boxes = _BOXED.findall(text)
    if boxes:
        return boxes[-1].strip()
    m = _FALLBACK_INT.findall(text)
    return m[-1] if m else None


def _norm(s: str) -> str:
    s = str(s).strip()
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\ ", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("$", ""), ("\\$", "")):
        s = s.replace(a, b)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".").lstrip("+")
    if s.endswith("^\\circ") or s.endswith("^{\\circ}"):
        s = s.split("^")[0]
    if s.startswith("\\text{") and s.endswith("}"):
        s = s[6:-1]
    # Base subscripts: gold "204_5" vs model "204" (the question already fixes the base).
    # Audit found this as the only false-negative class in a 20-sample manual check.
    s = re.sub(r"_\{?\d+\}?$", "", s)
    return s


def _as_num(s: str):
    # normalise LaTeX first: "\$32,\!348" -> "32348" (stray backslashes otherwise break float())
    t = _norm(s)
    t = re.sub(r"[,\s\$]|\\%|%|\\", "", t)
    try:
        return float(t)
    except ValueError:
        pass
    m = re.fullmatch(r"-?\\?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", t)
    if m:
        try:
            v = float(m.group(1)) / float(m.group(2))
            return -v if t.startswith("-") else v
        except ZeroDivisionError:
            return None
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", t)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except ZeroDivisionError:
            return None
    return None


def _as_int(s: str):
    t = re.sub(r"[,\s\$]", "", str(s))
    m = re.search(r"-?\d+", t)
    return int(m.group()) if m else None


def check_answer(pred: str | None, gold: str, benchmark: str = "") -> bool:
    if pred is None:
        return False
    if benchmark.startswith("aime"):          # AIME answers are integers 0-999
        p, g = _as_int(pred), _as_int(gold)
        return p is not None and g is not None and p == g

    if _norm(pred) == _norm(gold):            # 1. normalised string
        return True
    pn, gn = _as_num(pred), _as_num(gold)     # 2. numeric
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6 * max(1.0, abs(gn))
    try:                                       # 3. symbolic, WITH latex delimiters
        from math_verify import parse, verify
        g = parse(f"${gold}$")
        p = parse(f"${pred}$")
        if g and p:
            return bool(verify(g, p))
    except Exception:
        pass
    return False
