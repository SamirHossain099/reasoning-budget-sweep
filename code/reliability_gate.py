"""Split-half rank reliability with a test-length correction.

Split-half reliability characterises a HALF-length instrument, while the benchmark in use is full
length, so the uncorrected coefficient understates full-length reliability. Correct handling:
  * Spearman-Brown applies to PEARSON correlations of half-test SCORES, not to Kendall tau.
  * So: correlate the two halves' model-accuracy vectors (Pearson and Spearman), apply the
    Spearman-Brown step-up to those, and report Kendall tau separately as a descriptive statistic.
  * Also estimate full-pool reliability directly by parametric bootstrap (resample item x seed with
    replacement to full size, twice, then correlate), which needs no Spearman-Brown assumption.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

RNG = np.random.default_rng(0); NBOOT = 2000
KEY = ["model","benchmark","budget","item_id","seed"]

def load():
    fr=[]
    for f in Path("results/generations").rglob("*.parquet"):
        try: fr.append(pd.read_parquet(f))
        except Exception: pass
    return pd.concat(fr,ignore_index=True).drop_duplicates(subset=KEY)

def acc_on(g, pairs):
    return g.merge(pairs,on=["item_id","seed"]).groupby("model")["correct"].mean()

def sb(r,k=2.0):
    return k*r/(1+(k-1)*r) if r>-1 else np.nan

def run(df, bench, budget):
    g = df[(df.benchmark==bench)&(df.budget==budget)]
    models = sorted(g.model.unique())
    if len(models)<4: return None
    g = g[g.model.isin(models)]
    pairs = g[["item_id","seed"]].drop_duplicates().reset_index(drop=True)
    n = len(pairs)
    taus, pear, spear, full_pear = [], [], [], []
    for _ in range(NBOOT):
        perm = RNG.permutation(n); h1, h2 = pairs.iloc[perm[:n//2]], pairs.iloc[perm[n//2:]]
        a1, a2 = acc_on(g,h1).reindex(models), acc_on(g,h2).reindex(models)
        if a1.isna().any() or a2.isna().any(): continue
        if a1.std()==0 or a2.std()==0: continue
        taus.append(kendalltau(a1,a2)[0]); pear.append(pearsonr(a1,a2)[0]); spear.append(spearmanr(a1,a2)[0])
        # full-length: two independent full-size resamples (no SB needed)
        b1 = pairs.iloc[RNG.integers(0,n,n)]; b2 = pairs.iloc[RNG.integers(0,n,n)]
        f1, f2 = acc_on(g,b1).reindex(models), acc_on(g,b2).reindex(models)
        if not (f1.isna().any() or f2.isna().any() or f1.std()==0 or f2.std()==0):
            full_pear.append(pearsonr(f1,f2)[0])
    if not taus: return None
    mp = float(np.nanmean(pear))
    return dict(bench=bench, budget=budget, n_models=len(models), n_pairs=n,
        tau_half=float(np.nanmean(taus)),
        pearson_half=mp, spearman_half=float(np.nanmean(spear)),
        pearson_SB_full=sb(mp),
        pearson_full_boot=float(np.nanmean(full_pear)) if full_pear else np.nan,
        full_boot_lo=float(np.nanpercentile(full_pear,2.5)) if full_pear else np.nan,
        full_boot_hi=float(np.nanpercentile(full_pear,97.5)) if full_pear else np.nan)

df = load()
print(f"loaded {len(df):,} responses\n")
rows=[]
for bench in ["aime24","aime25","math500"]:
    for bud in sorted(df[df.benchmark==bench].budget.unique()):
        sub=df[(df.benchmark==bench)&(df.budget==bud)]
        if sub.model.nunique()>=5 and sub.groupby('model').size().min()>=60:
            r=run(df,bench,bud)
            if r: rows.append(r)
out=pd.DataFrame(rows)
pd.set_option("display.width",200)
print(out.round(3).to_string(index=False))
print("\n--- GATE ---")
print(f"mean HALF-length tau (what we reported): {out.tau_half.mean():.3f}")
print(f"mean FULL-length reliability, SB-corrected Pearson: {out.pearson_SB_full.mean():.3f}")
print(f"mean FULL-length reliability, direct bootstrap:     {out.pearson_full_boot.mean():.3f}")
print("\npublished AIME band (Hariri et al., ACL 2026): 0.78-0.95")
