#!/usr/bin/env python3
"""Exploratory MF/low-rank panel-completion gate (NOT pre-registered; a peek to judge a plan).

Question: do k revealed (drug, lnIC50) pairs of a held-out line carry drug-SPECIFIC information
beyond a per-line offset?

Model: a line's residual vector r (y - base) over drugs ~ N(0, Sigma), Sigma = low-rank(rank r)
drug-drug covariance estimated on the fold's TRAINING lines (drug-centred, soft-impute) + a*s2*I.
Prediction on unrevealed drugs = base + Sigma_ER (Sigma_RR)^-1 r_R  (GP / probabilistic MF).

Bases:  drugmean  -> pure MF completion (no cell features)
        qformer   -> random-init Q-Former prediction (features) + low-rank correction
Offsets (the existing gate) are reproduced exactly for reference.
Hyper-parameters (rank, a) picked per (seed, fold, k) by simulated reveals on an inner split of
TRAINING lines only, using the drugmean base. Evaluation halves identical to
scripts/novelty/panel_completion_gate.py (rng 10000*seed+draw).
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/workspace/volume/Gastro_transformers/MoSQ")
PREDS = ROOT / "reports/revision/multiseed/preds"
KS = [5, 10, 20, "all"]
RANKS = [1, 2, 5, 10, 20, 40, 60, 100]
ALPHAS = [0.5, 1.0, 2.0, 4.0]
SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else range(10)
DRAWS = range(3)


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def load(arm, seed):
    fs = sorted(glob.glob(str(PREDS / f"ncc__{arm}__seed{seed}__fold*__test.parquet")))
    df = pd.concat(pd.read_parquet(f) for f in fs)
    return df.sort_values("row_index").reset_index(drop=True)


def soft_impute(R, rank=60, iters=15):
    """R: lines x drugs with NaN. Returns filled matrix (low-rank completion of missing)."""
    obs = ~np.isnan(R)
    X = np.where(obs, R, 0.0)
    for _ in range(iters):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        L = (U[:, :rank] * S[:rank]) @ Vt[:rank]
        X = np.where(obs, R, L)
    return X


def make_sigma(R):
    """Return function rank -> (lowrank cov [D,D], noise var s2)."""
    obs = ~np.isnan(R)
    X = soft_impute(R)
    n = X.shape[0]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    cache = {}
    for rk in RANKS:
        C = (Vt[:rk].T * (S[:rk] ** 2 / n)) @ Vt[:rk]
        rec = (U[:, :rk] * S[:rk]) @ Vt[:rk]
        s2 = ((R - rec)[obs] ** 2).mean()
        cache[rk] = (C, s2)
    return cache


def gp_predict(C, s2, a, rev_idx, ev_idx, r_rev):
    K = C[np.ix_(rev_idx, rev_idx)] + a * s2 * np.eye(len(rev_idx))
    return C[np.ix_(ev_idx, rev_idx)] @ np.linalg.solve(K, r_rev)


def matrix(df, lines, drugs_index):
    sub = df[df.cellline_id.isin(lines)]
    M = sub.pivot(index="cellline_id", columns="drug_id", values="resid")
    return M.reindex(index=lines, columns=drugs_index).to_numpy()


def select_hparams(train_df, drugs, k, rng):
    """Inner 80/20 split of training lines; simulate reveals; pick (rank, a) on drugmean base."""
    lines = np.array(sorted(train_df.cellline_id.unique()))
    rng.shuffle(lines)
    cut = int(0.8 * len(lines))
    fit_l, val_l = lines[:cut], lines[cut:]
    sig = make_sigma(matrix(train_df, fit_l, drugs))
    dpos = {d: i for i, d in enumerate(drugs)}
    scores = {(rk, a): [0.0, 0.0] for rk in RANKS for a in ALPHAS}
    off_sse = 0.0
    for l, g in train_df[train_df.cellline_id.isin(val_l)].groupby("cellline_id"):
        n = len(g)
        kk = (n // 2) if k == "all" else k
        if n <= kk + 1:
            continue
        perm = rng.permutation(n)
        rv, ev = perm[:kk], perm[kk:]
        ri = np.array([dpos[d] for d in g.drug_id.values])
        r = g.resid.to_numpy()
        off_sse += ((r[ev] - r[rv].mean()) ** 2).sum()
        for (rk, a) in scores:
            C, s2 = sig[rk]
            p = gp_predict(C, s2, a, ri[rv], ri[ev], r[rv])
            scores[(rk, a)][0] += ((r[ev] - p) ** 2).sum()
    best = min(scores, key=lambda key: scores[key][0])
    return best, scores[best][0] / off_sse  # <1 means beats offset on training-line episodes


def main():
    rows, sel_rows = [], []
    for seed in SEEDS:
        base = load("qformer_randinit", seed)
        drugs = np.array(sorted(base.drug_id.unique()))
        dpos = {d: i for i, d in enumerate(drugs)}
        base["dpos"] = base.drug_id.map(dpos)
        dmean = np.full(len(base), np.nan)
        for f in base.fold.unique():
            m = (base.fold == f).to_numpy()
            means = base[~m].groupby("drug_id").y_true.mean()
            dmean[m] = base.loc[m, "drug_id"].map(means).to_numpy()
        base["dmean"] = dmean

        # per-fold covariance + hparam selection (independent of the evaluation draw)
        fold_sig, fold_hp = {}, {}
        fold_perm = {f: np.random.default_rng(55 + f).permutation(len(drugs)) for f in base.fold.unique()}
        for f in sorted(base.fold.unique()):
            tr = base[base.fold != f].copy()
            tr_means = tr.groupby("drug_id").y_true.mean()
            tr["resid"] = tr.y_true - tr.drug_id.map(tr_means)
            fold_sig[f] = make_sigma(matrix(tr, np.array(sorted(tr.cellline_id.unique())), drugs))
            for k in KS:
                hp, ratio = select_hparams(tr, drugs, k, np.random.default_rng(777 + 100 * seed + f))
                fold_hp[(f, k)] = hp
                sel_rows.append(dict(seed=seed, fold=f, k=str(k), rank=hp[0], a=hp[1],
                                     inner_sse_ratio_vs_offset=ratio))

        y = base.y_true.to_numpy()
        q = base.y_pred.to_numpy()
        has_rna = base.has_rna.to_numpy()
        for draw in DRAWS:
            rng = np.random.default_rng(10_000 * seed + draw)
            u = rng.random(len(base))
            rk_ = pd.Series(u).groupby(base.cellline_id).rank(method="first").to_numpy()
            n = base.groupby("cellline_id").cellline_id.transform("size").to_numpy()
            is_eval = rk_ <= np.ceil(n / 2)
            pool_rank = pd.Series(np.where(is_eval, np.inf, u)).groupby(
                base.cellline_id).rank(method="first").to_numpy()
            for k in KS:
                revealed = ~is_eval & (np.isfinite(pool_rank) if k == "all" else pool_rank <= k)
                preds = {}
                for name, b in [("drugmean", dmean), ("qformer", q)]:
                    resid = pd.Series(y - b)[revealed]
                    off = resid.groupby(base.cellline_id[revealed]).mean()
                    preds[f"{name}+offset"] = b + base.cellline_id.map(off).fillna(0).to_numpy()
                    gp_sel = b.copy()
                    gp_grid = {key: b.copy() for key in [(rk, a) for rk in RANKS for a in ALPHAS]}
                    for (f, l), g in base.groupby(["fold", "cellline_id"]):
                        idx = g.index.to_numpy()
                        rv = idx[revealed[idx]]
                        ev = idx[is_eval[idx]]
                        if len(rv) == 0 or len(ev) == 0:
                            continue
                        rr = (y - b)[rv]
                        ri, ei = base.dpos.to_numpy()[rv], base.dpos.to_numpy()[ev]
                        for key in gp_grid:
                            C, s2 = fold_sig[f][key[0]]
                            gp_grid[key][ev] = b[ev] + gp_predict(C, s2, key[1], ri, ei, rr)
                        gp_sel[ev] = gp_grid[fold_hp[(f, k)]][ev]
                    preds[f"{name}+lowrank(sel)"] = gp_sel
                    pl = b.copy()
                    for (f, l), g in base.groupby(["fold", "cellline_id"]):
                        idx = g.index.to_numpy()
                        rv = idx[revealed[idx]]
                        ev = idx[is_eval[idx]]
                        if len(rv) == 0 or len(ev) == 0:
                            continue
                        rk_sel, a_sel = fold_hp[(f, k)]
                        C, s2 = fold_sig[f][rk_sel]
                        perm = fold_perm[f]
                        ri, ei = perm[base.dpos.to_numpy()[rv]], perm[base.dpos.to_numpy()[ev]]
                        pl[ev] = b[ev] + gp_predict(C, s2, a_sel, ri, ei, (y - b)[rv])
                    preds[f"{name}+lowrank(placebo)"] = pl
                    for key, p in gp_grid.items():
                        preds[f"{name}+lowrank(r={key[0]},a={key[1]})"] = p
                ev_mask = is_eval & np.isfinite(dmean)
                for arm, p in preds.items():
                    for stratum, mm in [("all", ev_mask), ("rna", ev_mask & has_rna),
                                        ("no_rna", ev_mask & ~has_rna)]:
                        rows.append(dict(seed=seed, draw=draw, k=str(k), arm=arm, stratum=stratum,
                                         r2=r2(y[mm], p[mm]),
                                         dc_r2=r2(y[mm] - dmean[mm], p[mm] - dmean[mm])))
        print(f"seed {seed} done", flush=True)
        out = Path("/tmp/mf_gate")
        pd.DataFrame(rows).to_csv(out / f"mf_gate_rows_s{SEEDS[0]}.csv", index=False)
        pd.DataFrame(sel_rows).to_csv(out / f"mf_gate_selection_s{SEEDS[0]}.csv", index=False)


if __name__ == "__main__":
    main()
