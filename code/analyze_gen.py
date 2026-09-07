"""Analyze Phase-2 generations: completion-rate + truncated-vs-genuine-error decomposition, and
(with >=2 models) budget-dependent leaderboard rank instability. Works on partial data (safe to run
while the sweep is in progress).

Definitions per response:
  reached_cap  = generation hit max_new_tokens (budget)
  truncated    = reached_cap AND no answer extracted  (ran out of budget mid-reasoning)
  genuine_err  = has an answer (or finished) but wrong, and NOT truncated
  correct      = answer matches gold
Cell metrics (per model,benchmark,budget, averaged over items then seeds):
  accuracy, completion_rate = 1 - mean(reached_cap), P(correct | completed) = acc among not-reached_cap.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results" / "generations"
OUT = ROOT / "results"; FIG = ROOT / "figures"


def load() -> pd.DataFrame:
    files = list(GEN.rglob("*.parquet"))
    if not files:
        print("no generations yet"); return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # legacy flat-file + shard cells could overlap; keep one row per response
    df = df.drop_duplicates(subset=["model", "benchmark", "budget", "item_id", "seed"], keep="last")
    df["truncated"] = df.reached_cap & (~df.has_answer)
    df["completed"] = ~df.reached_cap
    return df


def per_seed_cell(df):
    """One row per (model,benchmark,budget,seed)."""
    g = df.groupby(["model", "benchmark", "budget", "seed"])
    out = g.agg(n=("correct", "size"),
                accuracy=("correct", "mean"),
                completion_rate=("completed", "mean"),
                answer_rate=("has_answer", "mean"),      # extractable answer (scorable at all)
                truncated_rate=("truncated", "mean"),
                mean_tokens=("n_new_tokens", "mean")).reset_index()
    # P(correct | completed) per cell
    pcc = (df[df.completed].groupby(["model", "benchmark", "budget", "seed"])["correct"]
           .mean().rename("acc_given_completed").reset_index())
    return out.merge(pcc, on=["model", "benchmark", "budget", "seed"], how="left")


def mean_ci(s):
    from scipy import stats
    x = s.dropna().to_numpy()
    if len(x) == 0: return (np.nan, np.nan)
    if len(x) < 2: return (float(x[0]), 0.0)
    return (float(x.mean()), float(stats.sem(x) * stats.t.ppf(0.975, len(x) - 1)))


def aggregate(cells):
    rows = []
    for (m, b, bud), g in cells.groupby(["model", "benchmark", "budget"]):
        rec = {"model": m, "benchmark": b, "budget": bud, "n_seeds": g.seed.nunique()}
        for col in ["accuracy", "completion_rate", "answer_rate", "truncated_rate",
                    "acc_given_completed", "mean_tokens"]:
            mu, ci = mean_ci(g[col]); rec[col] = mu; rec[col + "_ci"] = ci
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["benchmark", "model", "budget"])


def rank_instability(agg):
    """Kendall tau of the per-budget model leaderboard vs the max-budget leaderboard, per benchmark."""
    from scipy.stats import kendalltau
    rows = []
    for b, g in agg.groupby("benchmark"):
        budgets = sorted(g.budget.unique()); ref = budgets[-1]
        piv = g.pivot_table(index="model", columns="budget", values="accuracy")
        if ref not in piv or piv.shape[0] < 3:
            continue
        for bud in budgets[:-1]:
            common = piv[[bud, ref]].dropna()
            if len(common) < 3:
                continue
            tau, p = kendalltau(common[bud], common[ref])
            rows.append({"benchmark": b, "budget": bud, "n_models": len(common),
                         "kendall_tau_vs_max": tau, "p": p})
    return pd.DataFrame(rows)


def figures(agg):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for b, g in agg.groupby("benchmark"):
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for m, gm in g.groupby("model"):
            gm = gm.sort_values("budget")
            ax[0].errorbar(gm.budget, gm.accuracy, yerr=gm.accuracy_ci, marker="o", capsize=2, label=m)
            ax[1].errorbar(gm.budget, gm.completion_rate, yerr=gm.completion_rate_ci, marker="o", capsize=2, label=m)
            ax[2].errorbar(gm.budget, gm.acc_given_completed, yerr=gm.acc_given_completed_ci, marker="o", capsize=2, label=m)
        for a, t in zip(ax, ["accuracy", "completion rate", "P(correct | completed)"]):
            a.set_xscale("log", base=2); a.set_xlabel("token budget"); a.set_ylabel(t); a.legend(fontsize=7)
        ax[0].set_title(f"{b}: accuracy vs budget")
        ax[1].set_title(f"{b}: completion rate vs budget")
        ax[2].set_title(f"{b}: is the gain just completion? (P correct | completed)")
        fig.tight_layout(); fig.savefig(FIG / f"gen_{b}.png", dpi=130); plt.close(fig)


def main():
    df = load()
    if df.empty: return
    cells = per_seed_cell(df)
    cells.to_csv(OUT / "gen_cells_by_seed.csv", index=False)
    agg = aggregate(cells)
    agg.to_csv(OUT / "gen_aggregate.csv", index=False)
    print(f"generations: {len(df)} responses; {df.model.nunique()} models; "
          f"budgets {sorted(df.budget.unique())}")
    show = ["model", "benchmark", "budget", "n_seeds", "accuracy", "completion_rate",
            "truncated_rate", "acc_given_completed", "mean_tokens"]
    print(agg[show].to_string(index=False))
    ri = rank_instability(agg)
    if not ri.empty:
        ri.to_csv(OUT / "gen_rank_instability.csv", index=False)
        print("\nrank instability (needs >=3 models):")
        print(ri.to_string(index=False))
    else:
        print("\n(rank instability needs >=3 models at shared budgets; add models)")
    try:
        figures(agg); print(f"wrote figures/gen_*.png")
    except Exception as e:
        print(f"figures skipped: {e}")


if __name__ == "__main__":
    main()
