#!/usr/bin/env python3
"""POST-HOC drug-side control: per-cell-line ridge on drug representations, novel-drug splits.

Pre-specified in the pre-registration record ("Post-hoc control — 2026-10-04: drug side").
Splits: NCD and scaffold, primary partition (seed 0) + fold_seed 1, 2, 3 (the encoder sweep's
partitions), regenerated with splits.build_folds and verified against existing dumps.
One RidgeCV per cell line on that line's training-drug rows; arms differ only in drug features.
Output: reports/novelty/sweep/tables/ridge_drug_side{_per_fold,,_deltas}.csv
"""
import glob
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from gastro_transformer import splits  # noqa: E402
from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset  # noqa: E402

spec = importlib.util.spec_from_file_location("bs", ROOT / "scripts/novelty/baseline_sweep.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

ARMS = {"CB": "chemberta", "MTR": "chemberta_mtr", "MG": "morgan", "DS": "descriptors"}
PARTITIONS = [None, 1, 2, 3]
REGIMES = ["ncd", "scaffold"]
MIN_TRAIN = 10


def load():
    ds = IC50Dataset(str(ROOT / "data/ic50_data.csv"),
                     DrugEmbeddingDataset(str(ROOT / "data/drug_embeddings.csv"), drug_dim=768),
                     rna_csv_path=str(ROOT / "data/ccle_rna_for_ic50.csv"), rna_dim=256, add_tissue_ids=True)
    sm = pd.read_csv(ROOT / "data/drug_smiles.csv")
    smiles = dict(zip(sm.drug_id.astype(str), sm.smiles))
    feats = {}
    for arm, f in ARMS.items():
        t = pd.read_csv(ROOT / f"data/drug_embeddings_{f}.csv").set_index("drug_id")
        feats[arm] = t
    return ds, smiles, feats


DS, SMILES, FEATS = load()
Y = DS.ic50_values.numpy().astype(np.float64)
LINE = np.asarray(DS.cellline_ids, dtype=object)
DRUG = np.asarray(DS.drug_ids, dtype=object)


def folds(regime, fs):
    return splits.build_folds(regime, DS, n_folds=5, seed=0, smiles_by_drug=SMILES, fold_seed=fs)


def verify():
    """Regenerated primary partitions must equal the existing dumps' drug->fold map."""
    for regime in REGIMES:
        sf = folds(regime, None)
        regen = pd.Series(sf, index=DRUG).groupby(level=0).first()
        d = pd.concat(pd.read_parquet(f) for f in glob.glob(
            str(ROOT / f"reports/revision/encoder_descriptors/preds/{regime}__qformer_randinit__seed0__fold*__test.parquet")))
        dump = d.groupby("drug_id").fold.first()
        agree = (regen.reindex(dump.index) == dump).mean()
        print(f"verify {regime} primary partition vs dumps: {agree:.3f} drugs agree", flush=True)
        assert agree == 1.0, regime


def run(task):
    regime, fs = task
    sf = folds(regime, fs)
    rows = []
    for f in range(5):
        te = sf == f
        test_drugs = set(DRUG[te])
        assert not test_drugs & set(DRUG[~te]), "drug leak"
        preds = {a: np.full(te.sum(), np.nan) for a in list(ARMS) + ["LM"]}
        lm_all = np.full(te.sum(), np.nan)
        te_idx = np.flatnonzero(te)
        tr_by_line = pd.Series(np.flatnonzero(~te)).groupby(LINE[~te]).apply(list)
        te_by_line = pd.Series(np.arange(len(te_idx))).groupby(LINE[te_idx]).apply(list)
        n_fallback = 0
        for line, pos in te_by_line.items():
            tr = np.array(tr_by_line.get(line, []), dtype=int)
            m = Y[tr].mean() if len(tr) else Y[~te].mean()
            lm_all[pos] = m
            preds["LM"][pos] = m
            rows_te = te_idx[pos]
            for arm, F in FEATS.items():
                if len(tr) < MIN_TRAIN:
                    preds[arm][pos] = m
                    n_fallback += arm == "CB"
                    continue
                Xtr = F.loc[DRUG[tr]].to_numpy(float)
                Xte = F.loc[DRUG[rows_te]].to_numpy(float)
                # Bug fix 2026-10-04: drop columns constant up to rounding (SD < 1e-8); StandardScaler
                # only guards exactly-zero variance and otherwise divides by ~1e-16.
                mu, sd = Xtr.mean(0), Xtr.std(0)
                keep = sd > 1e-8
                if not keep.any():
                    preds[arm][pos] = m
                    continue
                Ztr = (Xtr[:, keep] - mu[keep]) / sd[keep]
                Zte = (Xte[:, keep] - mu[keep]) / sd[keep]
                # Second fix 2026-10-04: no extrapolation beyond the training range of any feature
                # (singleton-owned descriptor columns otherwise give |z| ~ 1e5 for a held-out drug).
                Zte = np.clip(Zte, Ztr.min(0), Ztr.max(0))
                mdl = RidgeCV(alphas=bs.RIDGE_ALPHAS).fit(Ztr, Y[tr])
                preds[arm][pos] = mdl.predict(Zte)
        y = Y[te_idx]
        for arm, p in preds.items():
            assert np.isfinite(p).all(), (regime, fs, f, arm)
            yc, pc = y - lm_all, p - lm_all
            rho = pd.DataFrame({"d": DRUG[te_idx], "y": y, "p": p}).groupby("d").apply(
                lambda g: stats.spearmanr(g.y, g.p)[0] if g.p.std() > 0 and len(g) > 2 else np.nan)
            rows.append(dict(regime=regime, partition="primary" if fs is None else f"fs{fs}", fold=f, arm=arm,
                             r2=bs_r2(y, p), line_centred_r2=bs_r2(yc, pc), drug_rho=float(np.nanmean(rho)),
                             n_test=len(y), n_fallback_lines=n_fallback))
    print(f"{regime} {fs} done", flush=True)
    return rows


def bs_r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    verify()
    tasks = [(r, fs) for r in REGIMES for fs in PARTITIONS]
    with Pool(8) as pool:
        rows = [r for rs in pool.map(run, tasks) for r in rs]
    t = pd.DataFrame(rows)
    out = ROOT / "reports/novelty/sweep/tables"
    t.to_csv(out / "ridge_drug_side_per_fold.csv", index=False)
    pm = t.groupby(["regime", "partition", "arm"])[["r2", "line_centred_r2", "drug_rho"]].mean().reset_index()
    summ = pm.groupby(["regime", "arm"])[["r2", "line_centred_r2", "drug_rho"]].agg(["mean", "std"]).round(4)
    summ.to_csv(out / "ridge_drug_side.csv")
    print(summ.to_string())
    deltas = []
    for regime in REGIMES:
        for a, b in [("CB", "DS"), ("MTR", "DS"), ("MG", "DS"), ("CB", "LM"), ("MTR", "LM"), ("MG", "LM"), ("DS", "LM")]:
            for met in ["r2", "line_centred_r2"]:
                p = t[t.regime == regime].pivot_table(index=["partition", "fold"], columns="arm", values=met)
                d = p[a] - p[b]
                lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
                pp = pm[pm.regime == regime].pivot(index="partition", columns="arm", values=met)
                dp = pp[a] - pp[b]
                deltas.append(dict(regime=regime, comparison=f"{a} - {b}", metric=met, delta=round(d.mean(), 4),
                                   ci_lo=round(lo, 4), ci_hi=round(hi, 4), positive_folds=f"{int((d > 0).sum())}/{len(d)}",
                                   wilcoxon_p=stats.wilcoxon(d).pvalue,
                                   partition_deltas=" ".join(f"{x:+.4f}" for x in dp.to_numpy())))
    dt = pd.DataFrame(deltas)
    dt.to_csv(out / "ridge_drug_side_deltas.csv", index=False)
    print(dt.to_string(index=False))
    print("fallback lines (fewer than 10 training drugs), per regime/partition/fold:",
          t[t.arm == "CB"].n_fallback_lines.describe()[["min", "max"]].to_dict())


if __name__ == "__main__":
    main()
