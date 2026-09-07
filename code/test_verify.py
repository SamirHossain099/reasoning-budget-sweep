"""Unit tests for answer checking. Cases drawn from the manual audit of real generations.
Run: python -m src.test_verify
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from src.verify import check_answer, extract_answer

CASES = [
    # (pred, gold, benchmark, expected)
    ("204", "204_5", "math500", True),                 # base subscript (audit false-negative)
    ("(2, 4)", "(2,4)", "math500", True),              # whitespace
    ("32348", r"\$32,\!348", "math500", True),         # currency + thousands separator
    ("0.5", r"\frac{1}{2}", "math500", True),          # decimal vs fraction
    ("1/2", r"\frac{1}{2}", "math500", True),
    (r"\dfrac{3}{4}", r"\frac{3}{4}", "math500", True),
    ("9901", "9901", "math500", True),
    ("2", "2", "math500", True),
    # true negatives
    ("3", "-3", "math500", False),
    ("-35", r"-\frac{35}{9}", "math500", False),
    (r"24\sqrt{3}", "216", "math500", False),
    ("147", "735", "aime25", False),
    ("18", "13", "math500", False),
    # aime integer handling
    ("009", "9", "aime24", True),
    ("42", "042", "aime24", True),
    ("40", "149", "aime24", False),
]

EXTRACT = [
    (r"so the answer is \boxed{42}.", "42"),
    (r"first \boxed{1} then \boxed{7}", "7"),           # last box wins
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),           # nested braces
    ("no box here at all", None),
]


def main():
    fails = 0
    for pred, gold, bench, exp in CASES:
        got = check_answer(pred, gold, bench)
        if got != exp:
            fails += 1
            print(f"  FAIL check_answer({pred!r}, {gold!r}, {bench}) -> {got}, expected {exp}")
    for text, exp in EXTRACT:
        got = extract_answer(text)
        if got != exp:
            fails += 1
            print(f"  FAIL extract_answer({text!r}) -> {got!r}, expected {exp!r}")
    total = len(CASES) + len(EXTRACT)
    print(f"verify tests: {total - fails}/{total} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
