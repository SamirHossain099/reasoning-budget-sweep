"""Show sweep progress for a given grid: cells done / in-progress / not-started, items generated,
and total tokens. Mirror the same --models/--benchmarks/--budgets/--seeds you run generate with.

  python -m src.status --models r1-1.5b --budgets 1024 2048 4096 8192 16384 32768 --seeds 0 1 2
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "generations"


def cell_state(mkey, bench, budget, seed):
    base = OUT / mkey / bench
    legacy = base / f"b{budget}_s{seed}.parquet"
    cdir = base / f"b{budget}_s{seed}"
    if legacy.exists() or (cdir / "_DONE").exists():
        return "done"
    if cdir.exists() and list(cdir.glob("part_*.parquet")):
        return "partial"
    return "todo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["r1-1.5b"])
    ap.add_argument("--benchmarks", nargs="+", default=["aime24", "aime25", "math500"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    counts = {"done": 0, "partial": 0, "todo": 0}
    per_model = {}
    for m in args.models:
        d = {"done": 0, "partial": 0, "todo": 0}
        for b in args.benchmarks:
            for bud in args.budgets:
                for s in args.seeds:
                    st = cell_state(m, b, bud, s)
                    counts[st] += 1; d[st] += 1
        per_model[m] = d
    total = sum(counts.values())

    # responses + tokens generated so far
    files = list(OUT.rglob("*.parquet"))
    n_resp = n_tok = 0
    if files:
        df = pd.concat([pd.read_parquet(f)[["item_id", "seed", "budget", "n_new_tokens"]] for f in files],
                       ignore_index=True)
        n_resp, n_tok = len(df), int(df.n_new_tokens.sum())

    print(f"GRID: {args.models}")
    print(f"      benchmarks={args.benchmarks} budgets={args.budgets} seeds={args.seeds}")
    print(f"CELLS: {counts['done']} done / {counts['partial']} partial / {counts['todo']} todo "
          f"= {counts['done']}/{total} ({100*counts['done']/max(total,1):.0f}%)")
    for m, d in per_model.items():
        print(f"   {m:16s} done {d['done']:3d}  partial {d['partial']:2d}  todo {d['todo']:3d}")
    print(f"RESPONSES generated so far: {n_resp:,}  ({n_tok:,} tokens)")


if __name__ == "__main__":
    main()
