#!/usr/bin/env python3
"""Panel-completion gate (novelty plan step 1). CPU only, reads existing prediction dumps.

For each held-out cell line, a fixed random half of its measured drugs is the evaluation
set; k drugs from the other half are "revealed". Every model is post-hoc calibrated with a
per-line offset = mean residual over the revealed drugs, then scored on the evaluation half.

Transductive DR-A + offset is a (heuristic) upper bound for any DR-A trained with k revealed
drugs, since its pretraining saw every label of the line, evaluation half included. If it
does not beat no-pretraining + offset, the panel-completion use case is closed. If it does,
that is necessary -- not sufficient -- for a leak-free DR-A niche; experiment C decides.

Arms within a seed share folds (verified: identical row_index per fold), so differences are
seed-paired. Output: reports/novelty/tables/panel_completion_gate{,_summary}.csv
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
PREDS = ROOT / "reports/revision/multiseed/preds"
OUT = ROOT / "reports/novelty/tables"
ARMS = ["multitoken_dra", "qformer_randinit", "xgboost", "standalone_mlp"]
KS = [0, 5, 10, 20, "all"]
SEEDS = range(10)
DRAWS = range(3)  # evaluation-half draws per seed


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def load(arm, seed):
    fs = sorted(glob.glob(str(PREDS / f"ncc__{arm}__seed{seed}__fold*__test.parquet")))
    df = pd.concat(pd.read_parquet(f) for f in fs)
    return df[["row_index", "cellline_id", "drug_id", "has_rna", "y_true", "y_pred", "fold"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        base = load(ARMS[0], seed).sort_values("row_index").reset_index(drop=True)
        preds = {ARMS[0]: base.y_pred.to_numpy()}
        for arm in ARMS[1:]:
            other = load(arm, seed).sort_values("row_index").reset_index(drop=True)
            assert (other.row_index.values == base.row_index.values).all(), (arm, seed)
            assert np.allclose(other.y_true.values, base.y_true.values), (arm, seed)
            preds[arm] = other.y_pred.to_numpy()

        # Out-of-fold drug means (training cell lines only) for drug-centring.
        dmean = np.full(len(base), np.nan)
        for f in base.fold.unique():
            m = base.fold == f
            means = base[~m].groupby("drug_id").y_true.mean()
            dmean[m.to_numpy()] = base.loc[m, "drug_id"].map(means).to_numpy()

        for draw in DRAWS:
            rng = np.random.default_rng(10_000 * seed + draw)
            u = rng.random(len(base))
            rk = pd.Series(u).groupby(base.cellline_id).rank(method="first").to_numpy()
            n = base.groupby("cellline_id").cellline_id.transform("size").to_numpy()
            is_eval = rk <= np.ceil(n / 2)
            pool_rank = pd.Series(np.where(is_eval, np.inf, u)).groupby(
                base.cellline_id).rank(method="first").to_numpy()

            for k in KS:
                revealed = ~is_eval & (np.isfinite(pool_rank) if k == "all" else pool_rank <= k)
                ev = is_eval & np.isfinite(dmean)
                for arm in ARMS + ["additive"]:
                    yp = dmean if arm == "additive" else preds[arm]
                    resid = pd.Series(base.y_true.to_numpy() - yp)[revealed]
                    off = resid.groupby(base.cellline_id[revealed]).mean()
                    cal = yp + base.cellline_id.map(off).fillna(0).to_numpy()
                    for stratum, mask in [("all", ev), ("rna", ev & base.has_rna.to_numpy()),
                                          ("no_rna", ev & ~base.has_rna.to_numpy())]:
                        y, f, d = base.y_true.to_numpy()[mask], cal[mask], dmean[mask]
                        rows.append(dict(seed=seed, draw=draw, k=str(k), arm=arm,
                                         stratum=stratum, r2=r2(y, f), dc_r2=r2(y - d, f - d)))
        print(f"seed {seed} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "panel_completion_gate.csv", index=False)

    # Average draws within seed, then pair across seeds.
    per_seed = res.groupby(["seed", "k", "arm", "stratum"])[["r2", "dc_r2"]].mean().reset_index()
    summ = []
    for (k, stratum), g in per_seed.groupby(["k", "stratum"]):
        piv = g.pivot(index="seed", columns="arm")
        for arm in ["multitoken_dra", "qformer_randinit", "xgboost", "standalone_mlp", "additive"]:
            rec = dict(k=k, stratum=stratum, arm=arm)
            for met in ["r2", "dc_r2"]:
                rec[met] = piv[met][arm].mean()
                rec[f"{met}_sd"] = piv[met][arm].std()
            if arm == "multitoken_dra":
                best = max(["qformer_randinit", "xgboost", "standalone_mlp"],
                           key=lambda a: piv["dc_r2"][a].mean())
                for met in ["r2", "dc_r2"]:
                    d = piv[met]["multitoken_dra"] - piv[met][best]
                    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
                    rec[f"delta_{met}_vs_{'best'}"] = d.mean()
                    rec[f"delta_{met}_ci"] = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                    rec[f"delta_{met}_pos"] = f"{int((d > 0).sum())}/{len(d)}"
                    rec[f"wilcoxon_{met}"] = stats.wilcoxon(d).pvalue
                rec["best_leakfree"] = best
            summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "panel_completion_gate_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for stratum in ["all", "rna", "no_rna"]:
        s = summ[summ.stratum == stratum]
        print(f"\n=== stratum={stratum}: dc_r2 (mean over 10 seeds x 3 draws) ===")
        print(s.pivot(index="k", columns="arm", values="dc_r2").round(4).to_string())
        print(s[s.arm == "multitoken_dra"][["k", "best_leakfree", "delta_dc_r2_vs_best",
              "delta_dc_r2_ci", "delta_dc_r2_pos", "delta_r2_vs_best", "delta_r2_ci"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
