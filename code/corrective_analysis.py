"""Diagnostics for budget-swept reasoning evaluations.

 Q1. Verify the definitional identity accuracy == answer_rate * P(correct|answer), and quantify how
     much of a regression of accuracy on answer_rate is definitional rather than empirical.
 Q2. DECOMPOSITION: at a fixed benchmark and budget, does between-model variance in accuracy come
     from differences in answer_rate (how often an answer is produced) or in P(correct|answer)?
 Q3. SPLIT-HALF NOISE TEST: how much does the model ranking change from sampling noise alone
     (two halves of the same budget) compared with between two different budgets?
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(0)
NBOOT = 2000
KEY = ["model", "benchmark", "budget", "item_id", "seed"]


def load():
    frames = []
    for f in (ROOT / "results" / "generations").rglob("*.parquet"):
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=KEY)
    return df


def complete_cells_only(df, min_frac=0.95):
    """Drop cells with incomplete coverage (review found an incomplete cell topping a leaderboard)."""
    n = df.groupby(["model", "benchmark", "budget"]).size().rename("n").reset_index()
    target = n.groupby(["benchmark", "budget"])["n"].max().rename("target").reset_index()
    n = n.merge(target, on=["benchmark", "budget"])
    keep = n[n.n >= min_frac * n.target][["model", "benchmark", "budget"]]
    before = df.groupby(["model", "benchmark", "budget"]).ngroups
    df = df.merge(keep, on=["model", "benchmark", "budget"])
    print(f"[cells] kept {len(keep)} of {before} cells (dropped incomplete ones)")
    return df


def q1_identity(df):
    print("\n=== Q1: is the headline an identity? ===")
    viol = ((df.correct) & (~df.has_answer)).sum()
    print(f"  rows with correct & not has_answer: {viol}  -> identity {'HOLDS' if viol==0 else 'BROKEN'}")
    c = df.groupby(["model", "benchmark", "budget"]).agg(
        acc=("correct", "mean"), ar=("has_answer", "mean")).reset_index()
    c["cond"] = np.where(c.ar > 0, c.acc / c.ar, np.nan)
    print(f"  accuracy == answer_rate * P(correct|answer) max abs error: "
          f"{np.nanmax(np.abs(c.acc - c.ar * c.cond)):.2e}")
    print(f"  P(correct|answer) across cells: mean={c.cond.mean():.3f} sd={c.cond.std():.3f} "
          f"min={c.cond.min():.3f} max={c.cond.max():.3f}")
    print("  -> R^2 of accuracy~answer_rate measures how CONSTANT P(correct|answer) is, "
          "not a causal effect.")
    return c


def q2_decomposition(df):
    """At fixed benchmark+budget, decompose between-model variance of accuracy."""
    print("\n=== Q2: between-model variance -- verbosity vs reasoning quality ===")
    print("  (log accuracy = log answer_rate + log P(correct|answer); shares sum to 1)")
    rows = []
    for (b, bud), g in df.groupby(["benchmark", "budget"]):
        c = g.groupby("model").agg(acc=("correct", "mean"), ar=("has_answer", "mean")).reset_index()
        c = c[(c.acc > 0) & (c.ar > 0)]
        if len(c) < 4:
            continue
        la, lr = np.log(c.acc), np.log(c.ar)
        lc = la - lr
        va = la.var(ddof=1)
        if va <= 0:
            continue
        # var(la) = cov(la,lr) + cov(la,lc)  -> exact additive shares
        s_ar = np.cov(la, lr)[0, 1] / va
        s_cond = np.cov(la, lc)[0, 1] / va
        rows.append({"benchmark": b, "budget": bud, "n_models": len(c),
                     "share_answer_rate": s_ar, "share_conditional": s_cond})
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    if len(out):
        print(f"\n  MEAN share from answer_rate (verbosity): {out.share_answer_rate.mean():.3f}")
        print(f"  MEAN share from P(correct|answer) (quality): {out.share_conditional.mean():.3f}")
    return out


def _leaderboard(g, item_seed_pairs=None):
    if item_seed_pairs is not None:
        g = g.merge(item_seed_pairs, on=["item_id", "seed"])
    return g.groupby("model")["correct"].mean()


def q3_split_half(df, lo=4096, hi=8192):
    """Compare between-budget rank movement to within-budget sampling noise."""
    print(f"\n=== Q3: rank instability vs sampling noise ({lo} vs {hi}) ===")
    for bench in sorted(df.benchmark.unique()):
        sub = df[df.benchmark == bench]
        a, b = sub[sub.budget == lo], sub[sub.budget == hi]
        models = sorted(set(a.model) & set(b.model))
        if len(models) < 4:
            continue
        a, b = a[a.model.isin(models)], b[b.model.isin(models)]
        obs_tau = kendalltau(_leaderboard(a)[models], _leaderboard(b)[models])[0]

        pairs = a[["item_id", "seed"]].drop_duplicates().reset_index(drop=True)
        boot_between, boot_within = [], []
        for _ in range(NBOOT):
            idx = RNG.integers(0, len(pairs), len(pairs))
            samp = pairs.iloc[idx]
            # between-budget tau under resampling
            try:
                t = kendalltau(_leaderboard(a, samp)[models], _leaderboard(b, samp)[models])[0]
                if not np.isnan(t):
                    boot_between.append(t)
            except Exception:
                pass
            # within-budget NULL: split the SAME budget in half -> tau from noise alone
            half = len(pairs) // 2
            p1, p2 = pairs.iloc[RNG.permutation(len(pairs))[:half]], pairs.iloc[RNG.permutation(len(pairs))[:half]]
            try:
                t2 = kendalltau(_leaderboard(a, p1)[models], _leaderboard(a, p2)[models])[0]
                if not np.isnan(t2):
                    boot_within.append(t2)
            except Exception:
                pass
        bb, bw = np.array(boot_between), np.array(boot_within)
        print(f"  {bench}: n={len(models)} models  observed tau({lo} vs {hi}) = {obs_tau:.3f}")
        if len(bb):
            print(f"     between-budget tau: median {np.median(bb):.3f}  95% CI [{np.percentile(bb,2.5):.3f}, {np.percentile(bb,97.5):.3f}]")
        if len(bw):
            print(f"     WITHIN-budget (noise only) tau: median {np.median(bw):.3f}  95% CI [{np.percentile(bw,2.5):.3f}, {np.percentile(bw,97.5):.3f}]")
        if len(bb) and len(bw):
            p = float((bw <= np.median(bb)).mean())
            verdict = ("SUPPORTED: between-budget reshuffling exceeds noise"
                       if p < 0.05 else
                       "NOT SUPPORTED: between-budget reshuffling is within sampling noise")
            print(f"     P(noise tau <= observed between tau) = {p:.3f}  -> {verdict}")


def main():
    df = load()
    print(f"loaded {len(df):,} responses, {df.model.nunique()} models")
    df = complete_cells_only(df)
    q1_identity(df)
    q2_decomposition(df)
    q3_split_half(df)


if __name__ == "__main__":
    main()
