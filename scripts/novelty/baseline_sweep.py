#!/usr/bin/env python3
"""Baseline sweep (pre-registered; see the pre-registration record)
(Appendix 2026-10-01 15:26, §2). Each item runs once; do not change grids or conventions here
without a dated note in that file.

Inductive NCC (also scored for ranking), seeds 0-9, partitions read from the qformer_randinit dumps:
  S1 per-drug mean | S2 drug x tissue mean (shrunk) | S3 drug x cancer-type mean (shrunk)
  S4 per-drug ridge on 313-d cell features (GCV alpha) | S5 kNN cell lines on RNA (S2 fallback)
Panel completion, same evaluation halves / reveals as reports/novelty/kriging:
  P1 biased ALS matrix factorisation, (rank, lambda) by training-line reveal simulations

Usage:
  baseline_sweep.py inductive <seed>     -> reports/novelty/sweep/rows/inductive_s<seed>.csv (+ preds)
  baseline_sweep.py panel <seed>         -> reports/novelty/sweep/rows/panel_s<seed>.csv
  baseline_sweep.py summarize            -> reports/novelty/sweep/tables/*.csv (+ printed)
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from gastro_transformer.metrics import topk_recall  # noqa: E402
from per_cellline_ranking import analyse_run  # noqa: E402

MULTI = ROOT / "reports/revision/multiseed/preds"
INDUCTIVE_DRA = ROOT / "reports/revision/leakfree_strict_ncc/preds"
import os  # noqa: E402

SMOKE = os.environ.get("SWEEP_SMOKE") == "1"  # random labels, fold 0, /tmp output: code-path test only
OUT = Path("/tmp/sweep_smoke") if SMOKE else ROOT / "reports/novelty/sweep"
KRIG = ROOT / "reports/novelty/kriging/rows"

# Grids fixed in the pre-registration.
LAMBDAS_MEAN = [0, 1, 2, 5, 10, 20, 50]
RIDGE_ALPHAS = 10.0 ** np.arange(-2, 4.01, 0.5)  # 13 values
KNN_KS = [5, 10, 20, 50]
MF_RANKS = [2, 5, 10, 20, 40]
MF_LAMBDAS = [0.1, 1.0, 10.0]
MF_SWEEPS = 20
PANEL_KS = [5, 10, 20, "all"]
N_INNER = 5


def r2(y, f):
    return float(1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def load_dumps(pattern):
    fs = sorted(glob.glob(str(pattern)))
    assert fs, pattern
    df = pd.concat(pd.read_parquet(f) for f in fs).sort_values("row_index").reset_index(drop=True)
    if SMOKE:
        df["y_true"] = np.random.default_rng(0).normal(2.0, 2.7, len(df)).astype(np.float32)
    return df


def cell_features():
    """Per-cell-line table: tissue, cancer type, RNA availability, 256-d RNA embedding."""
    cache = OUT / "cache_cell_features.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset

    ds = IC50Dataset(str(ROOT / "data/ic50_data.csv"),
                     DrugEmbeddingDataset(str(ROOT / "data/drug_embeddings.csv"), drug_dim=768),
                     rna_csv_path=str(ROOT / "data/ccle_rna_for_ic50.csv"), rna_dim=256,
                     add_tissue_ids=True)
    first = {}
    for i, c in enumerate(ds.cellline_ids):
        first.setdefault(c, i)
    recs = []
    for c, i in first.items():
        it = ds[i]
        recs.append(dict(cellline_id=c, tissue_id=int(it["tissue_id"]),
                         cancer_type_id=int(it["cancer_type_id"]),
                         rna_available=bool(it["rna_available"]),
                         rna=it["rna_embed"].numpy().astype(np.float32)))
    cf = pd.DataFrame(recs).set_index("cellline_id")
    OUT.mkdir(parents=True, exist_ok=True)
    cf.to_pickle(cache)
    return cf


def inner_folds(lines, seed, fold):
    rng = np.random.default_rng(1234 + 100 * seed + fold)
    lines = np.array(sorted(lines))
    rng.shuffle(lines)
    return np.array_split(lines, N_INNER)


# ---------------------------------------------------------------- inductive items
def group_mean_predict(tr, te, key, lam):
    """Shrunk drug x group mean. tr/te: DataFrames with drug_id, key, y_true."""
    gmean = tr.y_true.mean()
    m_d = tr.groupby("drug_id").y_true.mean()
    agg = tr.groupby(["drug_id", key]).y_true.agg(["mean", "size"])
    idx = pd.MultiIndex.from_arrays([te.drug_id, te[key]])
    m_dg = agg["mean"].reindex(idx).to_numpy()
    n = agg["size"].reindex(idx).fillna(0).to_numpy()
    md = te.drug_id.map(m_d).fillna(gmean).to_numpy()
    m_dg = np.where(n > 0, m_dg, md)
    denom = n + lam
    return np.where(denom > 0, (n * m_dg + lam * md) / np.where(denom > 0, denom, 1), md)


def select_lambda(tr, key, seed, fold):
    sse = {lam: 0.0 for lam in LAMBDAS_MEAN}
    for val_lines in inner_folds(tr.cellline_id.unique(), seed, fold):
        v = tr.cellline_id.isin(val_lines)
        for lam in LAMBDAS_MEAN:
            p = group_mean_predict(tr[~v], tr[v], key, lam)
            sse[lam] += ((tr[v].y_true.to_numpy() - p) ** 2).sum()
    return min(sse, key=sse.get)


def ridge_predict(tr, te, X_of):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    pred = np.full(len(te), np.nan)
    alphas_used = []
    te_pos = {d: np.flatnonzero(te.drug_id.to_numpy() == d) for d in te.drug_id.unique()}
    for d, g in tr.groupby("drug_id"):
        if d not in te_pos:
            continue
        X = X_of(g.cellline_id.to_numpy())
        sc = StandardScaler().fit(X)
        m = RidgeCV(alphas=RIDGE_ALPHAS).fit(sc.transform(X), g.y_true.to_numpy())
        alphas_used.append(m.alpha_)
        rows = te_pos[d]
        pred[rows] = m.predict(sc.transform(X_of(te.cellline_id.to_numpy()[rows])))
    return pred, alphas_used


def knn_matrix(tr, query_lines, cf, ks, drugs):
    """Predictions [len(ks), n_query, n_drugs] from RNA-available training lines; NaN if none."""
    tr = tr[tr.cellline_id.map(cf.rna_available)]
    tlines = np.array(sorted(tr.cellline_id.unique()))
    Y = tr.pivot(index="cellline_id", columns="drug_id", values="y_true").reindex(
        index=tlines, columns=drugs).to_numpy(np.float32)
    A = np.stack(cf.loc[tlines, "rna"].to_numpy())
    Q = np.stack(cf.loc[query_lines, "rna"].to_numpy())
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    order = np.argsort(-(Q @ A.T), axis=1, kind="stable")
    mask = ~np.isnan(Y)[order]                     # [q, ntr, D]
    vals = np.nan_to_num(Y)[order]
    cnt = np.cumsum(mask, axis=1)
    out = []
    for k in ks:
        sel = mask & (cnt <= k)
        num, den = (vals * sel).sum(1), sel.sum(1)
        out.append(np.where(den > 0, num / np.maximum(den, 1), np.nan))
    return np.stack(out)


def knn_predict(tr, te, cf, ks, drugs):
    dpos = {d: i for i, d in enumerate(drugs)}
    qlines = np.array(sorted(te.cellline_id.unique()))
    lpos = {c: i for i, c in enumerate(qlines)}
    M = knn_matrix(tr, qlines, cf, ks, drugs)
    li = te.cellline_id.map(lpos).to_numpy()
    di = te.drug_id.map(dpos).to_numpy()
    return M[:, li, di]                            # [len(ks), n_rows]


def select_k(tr, cf, drugs, seed, fold):
    sse = np.zeros(len(KNN_KS))
    for val_lines in inner_folds(tr.cellline_id.unique(), seed, fold):
        v = tr.cellline_id.isin(val_lines) & tr.cellline_id.map(cf.rna_available)
        if not v.any():
            continue
        inner_tr = tr[~tr.cellline_id.isin(val_lines)]
        P = knn_predict(inner_tr, tr[v], cf, KNN_KS, drugs)
        y = tr[v].y_true.to_numpy()
        ok = ~np.isnan(P).any(0)
        sse += ((P[:, ok] - y[ok]) ** 2).sum(1)
    return KNN_KS[int(np.argmin(sse))]


def onehot(ids, n):
    out = np.zeros((len(ids), n))
    ok = (ids >= 0) & (ids < n)
    out[np.flatnonzero(ok), ids[ok]] = 1.0
    return out


def run_inductive(seed):
    cf = cell_features()
    base = load_dumps(MULTI / f"ncc__qformer_randinit__seed{seed}__fold*__test.parquet")
    base["tissue_id"] = base.cellline_id.map(cf.tissue_id)
    base["cancer_type_id"] = base.cellline_id.map(cf.cancer_type_id)
    assert (base.has_rna.to_numpy() == base.cellline_id.map(cf.rna_available).to_numpy()).all()
    drugs = np.array(sorted(base.drug_id.unique()))

    rna_mat = np.stack(cf.rna.to_numpy())
    feat = np.concatenate([
        rna_mat, cf.rna_available.to_numpy(float)[:, None],
        onehot(cf.cancer_type_id.to_numpy(), 30), onehot(cf.tissue_id.to_numpy(), 26)], axis=1)
    fpos = {c: i for i, c in enumerate(cf.index)}

    def X_of(lines):
        return feat[[fpos[c] for c in lines]]

    preds = {a: np.full(len(base), np.nan) for a in ["S1", "S2", "S3", "S4", "S5"]}
    sel = []
    for f in sorted(base.fold.unique())[: 1 if SMOKE else None]:
        te_m = (base.fold == f).to_numpy()
        tr, te = base[~te_m], base[te_m]
        preds["S1"][te_m] = te.drug_id.map(tr.groupby("drug_id").y_true.mean()).to_numpy()
        lam_t = select_lambda(tr, "tissue_id", seed, f)
        lam_c = select_lambda(tr, "cancer_type_id", seed, f)
        preds["S2"][te_m] = group_mean_predict(tr, te, "tissue_id", lam_t)
        preds["S3"][te_m] = group_mean_predict(tr, te, "cancer_type_id", lam_c)
        p4, alphas = ridge_predict(tr, te, X_of)
        preds["S4"][te_m] = p4
        k = select_k(tr, cf, drugs, seed, f)
        s5 = preds["S2"][te_m].copy()
        rna_rows = te.has_rna.to_numpy()
        kp = knn_predict(tr, te[rna_rows], cf, [k], drugs)[0]
        s5[rna_rows] = np.where(np.isnan(kp), s5[rna_rows], kp)
        preds["S5"][te_m] = s5
        sel.append(dict(seed=seed, fold=f, lambda_tissue=lam_t, lambda_cancer=lam_c, knn_k=k,
                        ridge_alpha_median=float(np.median(alphas)),
                        ridge_alpha_at_edge=float(np.mean([a in (RIDGE_ALPHAS[0], RIDGE_ALPHAS[-1])
                                                           for a in alphas]))))
        print(f"seed {seed} fold {f}: lambda_t={lam_t} lambda_c={lam_c} k={k}", flush=True)

    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    rows = []
    for arm, p in preds.items():
        assert SMOKE or np.isfinite(p).all(), arm
        if SMOKE:
            continue
        df = base[["row_index", "cellline_id", "drug_id", "fold", "has_rna", "y_true"]].copy()
        df["y_pred"] = p.astype(np.float32)
        df.to_parquet(OUT / "preds" / f"ncc__{arm}__seed{seed}.parquet", index=False)
        rows.append(dict(seed=seed, arm=arm, **score_inductive(df)))
    pd.DataFrame(rows).to_csv(OUT / "rows" / f"inductive_s{seed}.csv", index=False)
    pd.DataFrame(sel).to_csv(OUT / "rows" / f"inductive_selection_s{seed}.csv", index=False)


def score_inductive(df):
    y, p, rna = df.y_true.to_numpy(), df.y_pred.to_numpy(), df.has_rna.to_numpy()
    ar = analyse_run(df, 20, 10)
    r_rna, r_no = r2(y[rna], p[rna]), r2(y[~rna], p[~rna])
    return dict(r2=r2(y, p), dc_r2=ar["r2_drug_centered"], rho=ar["per_cellline_spearman_mean"],
                top10=ar["top10_recall_mean"], r2_rna=r_rna, r2_no_rna=r_no, rna_gain=r_rna - r_no)


def score_references():
    rows = []
    for arm, pat, seeds in [
        ("qformer_randinit", MULTI / "ncc__qformer_randinit__seed{s}__fold*__test.parquet", range(10)),
        ("dra_transductive", MULTI / "ncc__multitoken_dra__seed{s}__fold*__test.parquet", range(10)),
        ("dra_inductive", INDUCTIVE_DRA / "strict_ncc__multitoken_dra__seed{s}__fold*__test.parquet",
         range(3)),
    ]:
        for s in seeds:
            df = load_dumps(str(pat).format(s=s))
            rows.append(dict(seed=s, arm=arm, **score_inductive(df)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- panel item (P1)
def eval_halves(base, seed, draw):
    """Identical to reports/novelty/kriging (mf_gate.py)."""
    rng = np.random.default_rng(10_000 * seed + draw)
    u = rng.random(len(base))
    rk = pd.Series(u).groupby(base.cellline_id).rank(method="first").to_numpy()
    n = base.groupby("cellline_id").cellline_id.transform("size").to_numpy()
    is_eval = rk <= np.ceil(n / 2)
    pool = pd.Series(np.where(is_eval, np.inf, u)).groupby(base.cellline_id).rank(
        method="first").to_numpy()
    return is_eval, pool


def als(Y, mask, md, rank, lam, rng):
    """Biased MF y = md[d] + b[c] + u[c].v[d] on observed entries. Returns prediction matrix."""
    nl, nd = Y.shape
    W = mask.astype(np.float64)
    R0 = np.where(mask, Y - md[None, :], 0.0)
    V = rng.normal(0, 0.1, (nd, rank))
    b = np.zeros(nl)
    U = np.zeros((nl, rank))
    I1 = lam * np.eye(rank + 1)
    I0 = lam * np.eye(rank)
    for _ in range(MF_SWEEPS):
        A = np.concatenate([np.ones((nd, 1)), V], axis=1)                       # [nd, r+1]
        G = (W @ (A[:, :, None] * A[:, None, :]).reshape(nd, -1)).reshape(nl, rank + 1, rank + 1)
        x = np.linalg.solve(G + I1, (R0 @ A)[..., None])[..., 0]               # R0 is 0 off-mask
        b, U = x[:, 0], x[:, 1:]
        R1 = np.where(mask, R0 - b[:, None], 0.0)
        G = (W.T @ (U[:, :, None] * U[:, None, :]).reshape(nl, -1)).reshape(nd, rank, rank)
        V = np.linalg.solve(G + I0, (R1.T @ U)[..., None])[..., 0]
    return md[None, :] + b[:, None] + U @ V.T


def mf_complete(train_rows, reveal_rows, lines, drugs, md, rank, lam, seed_rng):
    lpos = {c: i for i, c in enumerate(lines)}
    dpos = {d: i for i, d in enumerate(drugs)}
    Y = np.zeros((len(lines), len(drugs)))
    M = np.zeros_like(Y, dtype=bool)
    for df in (train_rows, reveal_rows):
        li = df.cellline_id.map(lpos).to_numpy()
        di = df.drug_id.map(dpos).to_numpy()
        Y[li, di] = df.y_true.to_numpy()
        M[li, di] = True
    return als(Y, M, md, rank, lam, np.random.default_rng(seed_rng)), lpos, dpos


def drug_means(rows, drugs):
    m = rows.groupby("drug_id").y_true.mean().reindex(drugs).to_numpy()
    return np.where(np.isnan(m), rows.y_true.mean(), m)


def select_mf(tr, drugs, k, seed, fold):
    rng = np.random.default_rng(777 + 100 * seed + fold)
    lines = np.array(sorted(tr.cellline_id.unique()))
    rng.shuffle(lines)
    cut = int(0.8 * len(lines))
    fit_l, val_l = lines[:cut], lines[cut:]
    rev, ev = [], []
    for _, g in tr[tr.cellline_id.isin(val_l)].groupby("cellline_id"):
        n = len(g)
        kk = (n // 2) if k == "all" else k
        if n <= kk + 1:
            continue
        perm = rng.permutation(n)
        rev.append(g.iloc[perm[:kk]])
        ev.append(g.iloc[perm[kk:]])
    rev, ev = pd.concat(rev), pd.concat(ev)
    all_lines = np.array(sorted(tr.cellline_id.unique()))
    fit_rows = tr[tr.cellline_id.isin(fit_l)]
    md = drug_means(fit_rows, drugs)
    best, best_sse = None, np.inf
    for rank in MF_RANKS:
        for lam in MF_LAMBDAS:
            P, lpos, dpos = mf_complete(fit_rows, rev, all_lines, drugs, md,
                                        rank, lam, 99 + 100 * seed + fold)
            p = P[ev.cellline_id.map(lpos).to_numpy(), ev.drug_id.map(dpos).to_numpy()]
            sse = ((ev.y_true.to_numpy() - p) ** 2).sum()
            if sse < best_sse:
                best, best_sse = (rank, lam), sse
    return best


def per_line_rank(y, p, groups):
    rhos, recs = [], []
    for idx in groups:
        if len(idx) < 20:
            continue
        if np.std(p[idx]) > 0 and np.std(y[idx]) > 0:
            rhos.append(stats.spearmanr(y[idx], p[idx])[0])
        recs.append(topk_recall(y[idx], p[idx], k=10, sensitive_is_low=True))
    return float(np.mean(rhos)), float(np.nanmean(recs))


def run_panel(seed):
    base = load_dumps(MULTI / f"ncc__qformer_randinit__seed{seed}__fold*__test.parquet")
    drugs = np.array(sorted(base.drug_id.unique()))
    y = base.y_true.to_numpy()
    has_rna = base.has_rna.to_numpy()
    dmean = np.full(len(base), np.nan)
    md_fold, hp = {}, {}
    for f in sorted(base.fold.unique()):
        m = (base.fold == f).to_numpy()
        means = base[~m].groupby("drug_id").y_true.mean()
        dmean[m] = base.loc[m, "drug_id"].map(means).to_numpy()
        md_fold[f] = drug_means(base[~m], drugs)
        tr = base[~m]
        for k in PANEL_KS:
            hp[(f, k)] = select_mf(tr, drugs, k, seed, f)
            print(f"seed {seed} fold {f} k={k}: (rank, lambda)={hp[(f, k)]}", flush=True)

    groups = list(base.groupby(["fold", "cellline_id"]).indices.items())
    rows, sel = [], [dict(seed=seed, fold=f, k=str(k), rank=v[0], lam=v[1]) for (f, k), v in hp.items()]
    for draw in range(3):
        is_eval, pool = eval_halves(base, seed, draw)
        ev_mask = is_eval & np.isfinite(dmean)
        for k in PANEL_KS:
            revealed = ~is_eval & (np.isfinite(pool) if k == "all" else pool <= k)
            pred = np.full(len(base), np.nan)
            for f in sorted(base.fold.unique()):
                m = (base.fold == f).to_numpy()
                rank, lam = hp[(f, k)]
                lines = np.array(sorted(base.cellline_id.unique()))
                P, lpos, dpos = mf_complete(base[~m], base[m & revealed], lines, drugs, md_fold[f],
                                            rank, lam, 99 + 100 * seed + f)
                pred[m] = P[base.cellline_id[m].map(lpos).to_numpy(),
                            base.drug_id[m].map(dpos).to_numpy()]
            for stratum, mm in [("all", ev_mask), ("rna", ev_mask & has_rna),
                                ("no_rna", ev_mask & ~has_rna)]:
                lines_idx = [idx[mm[idx]] for _, idx in groups]
                rho, rec = per_line_rank(y, pred, lines_idx)
                rows.append(dict(seed=seed, draw=draw, k=str(k), arm="P1_mf_als", stratum=stratum,
                                 r2=r2(y[mm], pred[mm]),
                                 dc_r2=r2(y[mm] - dmean[mm], pred[mm] - dmean[mm]),
                                 rho=rho, top10=rec))
        print(f"seed {seed} draw {draw} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rows" / f"panel_s{seed}.csv", index=False)
    pd.DataFrame(sel).to_csv(OUT / "rows" / f"panel_selection_s{seed}.csv", index=False)


# ---------------------------------------------------------------- summary
def ci(d):
    d = np.asarray(d)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] {int((d > 0).sum())}/{len(d)}"


def summarize():
    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    ind = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "rows/inductive_s[0-9]*.csv"))))
    ref = score_references()
    allr = pd.concat([ind, ref])
    allr.to_csv(tables / "inductive_per_seed.csv", index=False)
    mets = ["r2", "dc_r2", "rho", "top10", "r2_rna", "r2_no_rna", "rna_gain"]
    out = []
    for subset, seeds in [("seeds0-9", range(10)), ("seeds0-2", range(3))]:
        a = allr[allr.seed.isin(seeds)]
        g = a.groupby("arm")[mets]
        s = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
        s.insert(0, "n_seeds", a.groupby("arm").seed.nunique())
        s.insert(0, "seeds", subset)
        out.append(s)
    summ = pd.concat(out)
    summ.to_csv(tables / "inductive_summary.csv")
    print(summ.to_string())
    piv = allr.pivot(index="seed", columns="arm", values="r2")
    best = piv[["S1", "S2", "S3", "S4", "S5"]].mean().idxmax()
    print(f"\nBest simple baseline (seed-mean R2, seeds 0-9): {best} = {piv[best].mean():.4f}")
    lines = []
    for met in ["r2", "rho", "top10", "dc_r2"]:
        p = allr.pivot(index="seed", columns="arm", values=met)
        for arm in ["S2", "S3", "S4", "S5"]:
            lines.append(dict(metric=met, comparison=f"{arm} - S1", seeds="0-9",
                              delta=ci(p[arm] - p["S1"])))
        for arm in ["S1", "S2", "S3", "S4", "S5"]:
            lines.append(dict(metric=met, comparison=f"qformer_randinit - {arm}", seeds="0-9",
                              delta=ci(p["qformer_randinit"] - p[arm])))
            q = p.loc[[0, 1, 2]]
            lines.append(dict(metric=met, comparison=f"dra_inductive - {arm}", seeds="0-2",
                              delta=ci(q["dra_inductive"] - q[arm])))
    d = pd.DataFrame(lines)
    d.to_csv(tables / "inductive_deltas.csv", index=False)
    print(d.to_string(index=False))

    pf = sorted(glob.glob(str(OUT / "rows/panel_s[0-9]*.csv")))
    if pf:
        pan = pd.concat(pd.read_csv(f) for f in pf)
        fu = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(KRIG / "fu_rows_s*.csv"))))
        fu["arm"] = fu.base + "+" + fu.arm
        fu = fu[fu.arm.isin(["drugmean+offset", "drugmean+krig_shared", "randinit+offset",
                             "randinit+krig_own", "dra+krig_own"])]
        comb = pd.concat([pan, fu[pan.columns]])
        ps = comb.groupby(["seed", "k", "arm", "stratum"])[["r2", "dc_r2", "rho", "top10"]].mean(
        ).reset_index()
        ps.to_csv(tables / "panel_per_seed.csv", index=False)
        g = ps[ps.stratum == "all"].groupby(["arm", "k"])[["r2", "dc_r2", "rho", "top10"]]
        t = (g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)).unstack("k")
        t.to_csv(tables / "panel_summary.csv")
        print("\n" + t.to_string())
        lines = []
        for met in ["r2", "rho"]:
            for k in ["5", "10", "20", "all"]:
                p = ps[(ps.stratum == "all") & (ps.k == k)].pivot(index="seed", columns="arm",
                                                                    values=met)
                for other in ["drugmean+krig_shared", "randinit+krig_own", "randinit+offset"]:
                    lines.append(dict(metric=met, k=k, comparison=f"P1 - {other}",
                                      delta=ci(p["P1_mf_als"] - p[other])))
        d = pd.DataFrame(lines)
        d.to_csv(tables / "panel_deltas.csv", index=False)
        print(d.to_string(index=False))


if __name__ == "__main__":
    (OUT / "rows").mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "inductive":
        run_inductive(int(sys.argv[2]))
    elif sys.argv[1] == "panel":
        run_panel(int(sys.argv[2]))
    elif sys.argv[1] == "summarize":
        summarize()
