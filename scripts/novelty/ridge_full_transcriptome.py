#!/usr/bin/env python3
"""POST-HOC control, amendment (third arm F): per-drug ridge (S4) on full-transcriptome raw
expression (DepMap 24Q4, protein-coding log2(TPM+1)) vs BulkRNABERT embedding vs DeepCDR
697-gene expression vs context only.

Pre-specified in the pre-registration record ("Amendment — 2026-10-02, third arm F").
Lines: intersection of E, X and F, excluding any line whose DepMap ModelID is shared with
another of our lines (ambiguous mapping). Splits: NCC partitions seeds 0-9 (multiseed
random-init dumps); train and test restricted to those lines. Run once.
Output: reports/novelty/sweep/tables/ridge_full_transcriptome{,_per_seed,_deltas}.csv
"""
import importlib.util
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from gastro_transformer.comparators.data import normalise_cellline  # noqa: E402
from per_cellline_ranking import analyse_run  # noqa: E402

spec = importlib.util.spec_from_file_location("bs", ROOT / "scripts/novelty/baseline_sweep.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

DEPMAP = ROOT / "data/external/depmap"
ARMS = ["E_embedding", "F_full_transcriptome", "X_raw697", "C_context"]


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def build():
    cf = bs.cell_features()
    model = pd.read_csv(DEPMAP / "Model.csv")
    name2ach = {}
    for col in ["CellLineName", "StrippedCellLineName"]:
        for n, a in zip(model[col], model.ModelID):
            k = normalise_cellline(n)
            if k:
                name2ach.setdefault(k, a)
    expr = pd.read_csv(DEPMAP / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
    raw = pd.read_csv(ROOT / "data/matched/rna_expression697.csv").set_index("cellline_id")
    raw = raw[raw.abs().sum(1) > 0]
    ach = {c: name2ach.get(normalise_cellline(c)) for c in cf.index}
    counts = Counter(a for a in ach.values() if a)
    ambiguous = {c for c, a in ach.items() if a and counts[a] > 1}
    lines = sorted(c for c in cf.index[cf.rna_available]
                   if c in raw.index and ach[c] in expr.index and c not in ambiguous)
    ctx = np.concatenate([bs.onehot(cf.loc[lines, "cancer_type_id"].to_numpy(), 30),
                          bs.onehot(cf.loc[lines, "tissue_id"].to_numpy(), 26)], axis=1)
    F = expr.loc[[ach[c] for c in lines]].to_numpy(np.float32)
    F = F[:, F.std(0) > 0]  # drop genes constant across these lines
    feats = {"E_embedding": np.concatenate([np.stack(cf.loc[lines, "rna"].to_numpy()), ctx], 1),
             "F_full_transcriptome": np.concatenate([F, ctx], 1).astype(np.float32),
             "X_raw697": np.concatenate([raw.loc[lines].to_numpy(float), ctx], 1),
             "C_context": ctx}
    return lines, feats, sorted(ambiguous)


LINES, FEATS, AMBIG = build()
POS = {c: i for i, c in enumerate(LINES)}


def run_seed(seed):
    base = bs.load_dumps(bs.MULTI / f"ncc__qformer_randinit__seed{seed}__fold*__test.parquet")
    base = base[base.cellline_id.isin(POS)].reset_index(drop=True)
    preds = {a: np.full(len(base), np.nan) for a in ARMS}
    for f in sorted(base.fold.unique()):
        te = (base.fold == f).to_numpy()
        for a in ARMS:
            Fm = FEATS[a]
            preds[a][te], _ = bs.ridge_predict(base[~te], base[te], lambda ls: Fm[[POS[c] for c in ls]])
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
    print(f"{len(LINES)} lines (ambiguous DepMap mapping dropped: {AMBIG}); dims "
          + ", ".join(f"{a}={FEATS[a].shape[1]}" for a in ARMS), flush=True)
    with Pool(5) as pool:
        rows = [r for rs in pool.map(run_seed, range(10)) for r in rs]
    t = pd.DataFrame(rows)
    out = ROOT / "reports/novelty/sweep/tables"
    t.to_csv(out / "ridge_full_transcriptome_per_seed.csv", index=False)
    g = t.groupby("arm")
    summ = g[["r2", "dc_r2", "rho", "top10"]].agg(["mean", "std"]).round(4)
    summ["n_seeds"] = g.size()
    summ.to_csv(out / "ridge_full_transcriptome.csv")
    print(summ.to_string())
    deltas = []
    for a, b in [("E_embedding", "F_full_transcriptome"), ("F_full_transcriptome", "X_raw697"),
                 ("E_embedding", "X_raw697"), ("E_embedding", "C_context"),
                 ("F_full_transcriptome", "C_context"), ("X_raw697", "C_context")]:
        for met in ["r2", "dc_r2", "rho", "top10"]:
            p = t.pivot(index="seed", columns="arm", values=met)
            d = p[a] - p[b]
            lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
            deltas.append(dict(comparison=f"{a} - {b}", metric=met, delta=round(d.mean(), 4),
                               ci_lo=round(lo, 4), ci_hi=round(hi, 4), positive=f"{int((d > 0).sum())}/10",
                               wilcoxon_p=stats.wilcoxon(d).pvalue))
    dt = pd.DataFrame(deltas)
    dt.to_csv(out / "ridge_full_transcriptome_deltas.csv", index=False)
    print(dt.to_string(index=False))


if __name__ == "__main__":
    main()
