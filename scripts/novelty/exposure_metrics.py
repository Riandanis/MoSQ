#!/usr/bin/env python3
"""File-backed drug-centred R2 and RNA gain for the exposure arms (transductive, size-matched,
inductive) -- numbers first computed in chat on 2026-09-30 and quoted in the drafts.
Conventions: R2 = per-seed mean of fold R2; drug-centred R2, rho, top-10 via analyse_run pooled
per seed; RNA gain pooled per seed (rna_subset.csv method). Mean +- across-seed SD.
Output: reports/novelty/tables/exposure_metrics{,_per_seed}.csv
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from per_cellline_ranking import analyse_run  # noqa: E402

ARMS = {
    "transductive (multiseed s0-9)": ("reports/revision/multiseed/preds/ncc__multitoken_dra__seed{s}__fold*__test.parquet", range(10)),
    "transductive (s0-2)": ("reports/revision/multiseed/preds/ncc__multitoken_dra__seed{s}__fold*__test.parquet", range(3)),
    "size-matched leaky (s0-2)": ("reports/final_experiments/smlc_strict_ncc/preds/strict_ncc__multitoken_dra__seed{s}__fold*__test.parquet", range(3)),
    "inductive (s0-2)": ("reports/revision/leakfree_strict_ncc/preds/strict_ncc__multitoken_dra__seed{s}__fold*__test.parquet", range(3)),
    "random init (s0-2)": ("reports/revision/multiseed/preds/ncc__qformer_randinit__seed{s}__fold*__test.parquet", range(3)),
}

def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()

rows = []
for arm, (pat, seeds) in ARMS.items():
    for s in seeds:
        fs = sorted(glob.glob(str(ROOT / pat.format(s=s))))
        assert len(fs) == 5, (arm, s, len(fs))
        d = pd.concat(pd.read_parquet(f) for f in fs)
        rk = analyse_run(d, 20, 10)
        rows.append(dict(arm=arm, seed=s, r2=np.mean([r2(g.y_true, g.y_pred) for _, g in d.groupby("fold")]),
                         dc_r2=rk["r2_drug_centered"], rho=rk["per_cellline_spearman_mean"],
                         rna_gain=r2(d[d.has_rna].y_true, d[d.has_rna].y_pred) - r2(d[~d.has_rna].y_true, d[~d.has_rna].y_pred)))
t = pd.DataFrame(rows)
out = ROOT / "reports/novelty/tables"
t.to_csv(out / "exposure_metrics_per_seed.csv", index=False)
g = t.groupby("arm", sort=False)
s = g[["r2", "dc_r2", "rho", "rna_gain"]].agg(["mean", "std"]).round(4)
s["n_seeds"] = g.size()
s.to_csv(out / "exposure_metrics.csv")
print(s.to_string())
