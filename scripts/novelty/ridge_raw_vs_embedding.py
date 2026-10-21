#!/usr/bin/env python3
"""POST-HOC control: per-drug ridge (S4) on raw expression vs on BulkRNABERT embeddings.

Pre-specified in the pre-registration record ("Post-hoc control — 2026-10-02"). Run once.
Lines: the 517 with both DeepCDR 697-gene expression and BulkRNABERT; NCC partitions seeds 0-9
from the multiseed random-init dumps; training AND test restricted to those lines.
Arms: E = embedding + one-hots, X = raw 697-gene expression + one-hots, C = one-hots only.
Output: reports/novelty/sweep/tables/ridge_raw_vs_embedding{,_per_seed,_deltas}.csv
"""
import glob
import importlib.util
import sys
from multiprocessing import Pool
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

ARMS = ["E_embedding", "X_raw697", "C_context"]


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def build_features():
    cf = bs.cell_features()                                   # BulkRNABERT, tissue, cancer type
    raw = pd.read_csv(ROOT / "data/matched/rna_expression697.csv").set_index("cellline_id")
    raw = raw[raw.abs().sum(1) > 0]                           # lines actually covered
    lines = sorted(set(cf.index[cf.rna_available]) & set(raw.index))
    ctx = np.concatenate([bs.onehot(cf.loc[lines, "cancer_type_id"].to_numpy(), 30),
                          bs.onehot(cf.loc[lines, "tissue_id"].to_numpy(), 26)], axis=1)
    emb = np.stack(cf.loc[lines, "rna"].to_numpy())
    feats = {"E_embedding": np.concatenate([emb, ctx], 1),
             "X_raw697": np.concatenate([raw.loc[lines].to_numpy(float), ctx], 1),
             "C_context": ctx}
    return lines, feats


LINES, FEATS = build_features()
POS = {c: i for i, c in enumerate(LINES)}


def run_seed(seed):
    base = bs.load_dumps(bs.MULTI / f"ncc__qformer_randinit__seed{seed}__fold*__test.parquet")
    base = base[base.cellline_id.isin(POS)].reset_index(drop=True)
    preds = {a: np.full(len(base), np.nan) for a in ARMS}
    for f in sorted(base.fold.unique()):
        te = (base.fold == f).to_numpy()
        for a in ARMS:
            F = FEATS[a]
            p, _ = bs.ridge_predict(base[~te], base[te], lambda ls: F[[POS[c] for c in ls]])
            preds[a][te] = p
    rows = []
    for a in ARMS:
        d = base[["cellline_id", "drug_id", "fold", "y_true"]].copy()
        d["y_pred"] = preds[a]
        assert np.isfinite(d.y_pred).all(), (seed, a)
        rk = analyse_run(d, 20, 10)
        rows.append(dict(seed=seed, arm=a, n_rows=len(d), n_lines=d.cellline_id.nunique(),
                         r2=np.mean([r2(g.y_true, g.y_pred) for _, g in d.groupby("fold")]),
                         dc_r2=rk["r2_drug_centered"], rho=rk["per_cellline_spearman_mean"],
                         top10=rk["top10_recall_mean"]))
    print(f"seed {seed} done", flush=True)
    return rows


def main():
    print(f"{len(LINES)} lines with both representations; dims "
          + ", ".join(f"{a}={FEATS[a].shape[1]}" for a in ARMS), flush=True)
    with Pool(5) as pool:
        rows = [r for rs in pool.map(run_seed, range(10)) for r in rs]
    t = pd.DataFrame(rows)
    out = ROOT / "reports/novelty/sweep/tables"
    t.to_csv(out / "ridge_raw_vs_embedding_per_seed.csv", index=False)
    g = t.groupby("arm")
    summ = g[["r2", "dc_r2", "rho", "top10"]].agg(["mean", "std"]).round(4)
    summ["n_seeds"] = g.size()
    summ.to_csv(out / "ridge_raw_vs_embedding.csv")
    print(summ.to_string())
    deltas = []
    for a, b in [("E_embedding", "X_raw697"), ("E_embedding", "C_context"), ("X_raw697", "C_context")]:
        for met in ["r2", "dc_r2", "rho", "top10"]:
            p = t.pivot(index="seed", columns="arm", values=met)
            d = p[a] - p[b]
            lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
            deltas.append(dict(comparison=f"{a} - {b}", metric=met, delta=round(d.mean(), 4),
                               ci_lo=round(lo, 4), ci_hi=round(hi, 4), positive=f"{int((d > 0).sum())}/10",
                               wilcoxon_p=stats.wilcoxon(d).pvalue))
    dt = pd.DataFrame(deltas)
    dt.to_csv(out / "ridge_raw_vs_embedding_deltas.csv", index=False)
    print(dt.to_string(index=False))


if __name__ == "__main__":
    main()
