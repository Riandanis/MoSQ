#!/usr/bin/env python3
"""POST-HOC: S4 (per-drug ridge) on the 544-line Level 2 subset.

Pre-specified in the pre-registration record, "Post-hoc item — 2026-10-02". S4 is used
unchanged (`ridge_predict`, `cell_features` imported from baseline_sweep.py); run once.
Output: reports/novelty/sweep/tables/s4_common_subset{,_per_seed,_deltas}.csv
"""
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from per_cellline_ranking import analyse_run  # noqa: E402

spec = importlib.util.spec_from_file_location("bs", ROOT / "scripts/novelty/baseline_sweep.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

SRC = {
    "MoSQ, no pretraining": "reports/letter_audit/probe/mosq_common/preds/ncc__qformer_randinit__seed{s}__fold*__test.parquet",
    "MoSQ (DR-A)": "reports/revision/mosq_common/preds/ncc__multitoken_dra__seed{s}__fold*__test.parquet",
    "XGBoost": "reports/revision/mosq_common/preds/ncc__xgboost__seed{s}__fold*__test.parquet",
    "Standalone MLP": "reports/revision/mosq_common/preds/ncc__standalone_mlp__seed{s}__fold*__test.parquet",
    "tCNNS": "reports/revision/comparators/preds/ncc__tcnns__seed{s}__fold*__test.parquet",
    "DeepCDR": "reports/revision/comparators/preds/ncc__deepcdr__seed{s}__fold*__test.parquet",
    "DrugCell": "reports/revision/comparators/preds/ncc__drugcell__seed{s}__fold*__test.parquet",
}
KEY = ["cellline_id", "drug_id"]


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def load(pat, s):
    fs = sorted(glob.glob(str(ROOT / pat.format(s=s))))
    if len(fs) != 5:
        return None
    return pd.concat(pd.read_parquet(f) for f in fs).sort_values(KEY).reset_index(drop=True)


def metrics(d):
    rk = analyse_run(d, 20, 10)
    return dict(r2_foldmean=np.mean([r2(g.y_true, g.y_pred) for _, g in d.groupby("fold")]),
                r2_pooled=r2(d.y_true, d.y_pred), dc_r2=rk["r2_drug_centered"],
                rho=rk["per_cellline_spearman_mean"], top10=rk["top10_recall_mean"])


def main():
    cf = bs.cell_features()
    feat = np.concatenate([
        np.stack(cf.rna.to_numpy()), cf.rna_available.to_numpy(float)[:, None],
        bs.onehot(cf.cancer_type_id.to_numpy(), 30), bs.onehot(cf.tissue_id.to_numpy(), 26)], axis=1)
    assert feat.shape[1] == 313
    fpos = {c: i for i, c in enumerate(cf.index)}

    def X_of(lines):
        return feat[[fpos[c] for c in lines]]

    rows, alphas = [], []
    for s in range(3):
        base = load(SRC["MoSQ, no pretraining"], s)
        assert base.cellline_id.nunique() == 544
        p = np.full(len(base), np.nan)
        for f in sorted(base.fold.unique()):
            te_m = (base.fold == f).to_numpy()
            pf, a = bs.ridge_predict(base[~te_m], base[te_m], X_of)
            p[te_m] = pf
            alphas += a
        assert np.isfinite(p).all()
        s4 = base.copy()
        s4["y_pred"] = p
        rows.append(dict(seed=s, model="S4 per-drug ridge", **metrics(s4)))
        for name, pat in SRC.items():
            d = load(pat, s)
            if d is None:
                continue
            assert (d[KEY].values == base[KEY].values).all() and (d.fold.values == base.fold.values).all(), name
            assert np.allclose(d.y_true.values, base.y_true.values, atol=1e-5), name
            rows.append(dict(seed=s, model=name, **metrics(d)))
        print(f"seed {s} done", flush=True)

    t = pd.DataFrame(rows)
    out = ROOT / "reports/novelty/sweep/tables"
    t.to_csv(out / "s4_common_subset_per_seed.csv", index=False)
    g = t.groupby("model")
    summ = g[["r2_foldmean", "r2_pooled", "dc_r2", "rho", "top10"]].agg(["mean", "std"]).round(4)
    summ["n_seeds"] = g.size()
    summ = summ.sort_values(("r2_foldmean", "mean"), ascending=False)
    summ.to_csv(out / "s4_common_subset.csv")
    print(summ.to_string())

    piv = t.pivot(index="seed", columns="model", values="r2_foldmean")
    deltas = []
    for other in [m for m in piv.columns if m != "S4 per-drug ridge"]:
        d = (piv["S4 per-drug ridge"] - piv[other]).dropna()
        if len(d) > 1:
            lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
            ci = f"[{lo:+.4f}, {hi:+.4f}]"
        else:
            ci = "single seed, no CI"
        deltas.append(dict(comparison=f"S4 - {other}", n_seeds=len(d), delta=round(d.mean(), 4),
                           ci95=ci, positive=f"{int((d > 0).sum())}/{len(d)}"))
    dt = pd.DataFrame(deltas)
    dt.to_csv(out / "s4_common_subset_deltas.csv", index=False)
    print(dt.to_string(index=False))
    a = np.array(alphas)
    print(f"ridge alpha: median {np.median(a):.3g}, share at grid edge "
          f"{np.mean(np.isin(a, [bs.RIDGE_ALPHAS[0], bs.RIDGE_ALPHAS[-1]])):.3f}")


if __name__ == "__main__":
    main()
