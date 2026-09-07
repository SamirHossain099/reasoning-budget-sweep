"""Recompute `correct` for every cached response using the current verify.check_answer.

Answers and gold labels are cached per response, so scoring bugs can be fixed WITHOUT regenerating
anything. Run after any change to src/verify.py.

  python -m src.rescore            # report what would change
  python -m src.rescore --apply    # rewrite the parquet files
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from src.verify import check_answer

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    import time
    files = sorted(GEN.rglob("*.parquet"))
    tot = changed = skipped = 0
    per_bench = {}
    for f in files:
        # never touch a file the running sweep may still be writing
        if time.time() - f.stat().st_mtime < 120:
            skipped += 1
            continue
        try:
            d = pd.read_parquet(f)
        except Exception:
            skipped += 1
            continue
        if "answer" not in d.columns:
            continue
        new = [check_answer(a, g, b) for a, g, b in zip(d["answer"], d["gold"], d["benchmark"])]
        n_ch = int((pd.Series(new, index=d.index) != d["correct"]).sum())
        tot += len(d); changed += n_ch
        b = d["benchmark"].iloc[0]
        s = per_bench.setdefault(b, {"n": 0, "old": 0, "new": 0})
        s["n"] += len(d); s["old"] += int(d["correct"].sum()); s["new"] += int(sum(new))
        if args.apply and n_ch:
            d["correct"] = new
            d.to_parquet(f)

    print(f"responses scanned: {tot}   changed: {changed}   files skipped (in-flight): {skipped}   "
          f"({'APPLIED' if args.apply else 'dry run'})")
    for b, s in sorted(per_bench.items()):
        if s["n"]:
            print(f"  {b:9s} n={s['n']:5d}  accuracy {s['old']/s['n']:.3f} -> {s['new']/s['n']:.3f}")


if __name__ == "__main__":
    main()
