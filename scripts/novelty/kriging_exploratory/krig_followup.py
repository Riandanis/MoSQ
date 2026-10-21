#!/usr/bin/env python3
"""Follow-up to mf_gate_dra.py (exploratory, not pre-registered).

(3) Sigma source: 'shared' = covariance of drug-centred training-line residuals (as before);
    'own' = covariance of THIS model's residuals on training lines (from the other folds'
    test dumps). 'own' has a mild indirect exposure for random init (the fold-g model that
    produced the residuals was trained on fold-f lines); transductive DR-A is leaky anyway.
    Hyper-parameters are re-selected on training-line reveal simulations with the same base.
(4) Ranking: per-line Spearman rho and top-10 recall on the evaluation half (>= 20 drugs),
    same definitions as scripts/analysis/per_cellline_ranking.py.
(5) Strata for RNA gain; across-seed SDs come from the aggregation step.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import mf_gate as G  # noqa: E402
from gastro_transformer.metrics import topk_recall  # noqa: E402

KS = [5, 10, 20, "all"]
SEED = int(sys.argv[1])


def per_line_rank(df_y, df_p, lines):
    rhos, recs = [], []
    for idx in lines:
        if len(idx) < 20:
            continue
        y, p = df_y[idx], df_p[idx]
        if np.std(p) > 0 and np.std(y) > 0:
            rhos.append(stats.spearmanr(y, p)[0])
        recs.append(topk_recall(y, p, k=10, sensitive_is_low=True))
    return np.mean(rhos), np.nanmean(recs)


def main():
    base = G.load("qformer_randinit", SEED)
    dra = G.load("multitoken_dra", SEED)
    assert (dra.row_index.values == base.row_index.values).all()
    drugs = np.array(sorted(base.drug_id.unique()))
    base["dpos"] = base.drug_id.map({d: i for i, d in enumerate(drugs)})
    dmean = np.full(len(base), np.nan)
    for f in base.fold.unique():
        m = (base.fold == f).to_numpy()
        dmean[m] = base.loc[m, "drug_id"].map(base[~m].groupby("drug_id").y_true.mean()).to_numpy()
    y = base.y_true.to_numpy()
    bases = {"drugmean": dmean, "randinit": base.y_pred.to_numpy(), "dra": dra.y_pred.to_numpy()}
    has_rna = base.has_rna.to_numpy()

    # Sigma + hparams per (fold, base-for-sigma)
    sig, hp = {}, {}
    for f in sorted(base.fold.unique()):
        tr = base[base.fold != f].copy()
        tr_lines = np.array(sorted(tr.cellline_id.unique()))
        trm = tr.groupby("drug_id").y_true.mean()
        sources = {"shared": (tr.y_true - tr.drug_id.map(trm)).to_numpy()}
        for name in ["randinit", "dra"]:
            sources[name] = (y - bases[name])[tr.index.to_numpy()]
        for src, res in sources.items():
            t = tr.copy()
            t["resid"] = res
            sig[(f, src)] = G.make_sigma(G.matrix(t, tr_lines, drugs))
            for k in KS:
                hp[(f, src, k)], _ = G.select_hparams(
                    t, drugs, k, np.random.default_rng(777 + 100 * SEED + f))

    groups = list(base.groupby(["fold", "cellline_id"]).indices.items())
    dpos = base.dpos.to_numpy()
    rows = []
    for draw in range(3):
        rng = np.random.default_rng(10_000 * SEED + draw)
        u = rng.random(len(base))
        rk = pd.Series(u).groupby(base.cellline_id).rank(method="first").to_numpy()
        n = base.groupby("cellline_id").cellline_id.transform("size").to_numpy()
        is_eval = rk <= np.ceil(n / 2)
        pool = pd.Series(np.where(is_eval, np.inf, u)).groupby(base.cellline_id).rank(
            method="first").to_numpy()
        ev_mask = is_eval & np.isfinite(dmean)
        for k in KS:
            revealed = ~is_eval & (np.isfinite(pool) if k == "all" else pool <= k)
            for bname, b in bases.items():
                arms = {}
                resid = pd.Series(y - b)[revealed]
                off = resid.groupby(base.cellline_id[revealed]).mean()
                arms["offset"] = b + base.cellline_id.map(off).fillna(0).to_numpy()
                srcs = ["shared"] + ([bname] if bname != "drugmean" else [])
                for src in srcs:
                    p = b.copy()
                    for (f, _l), idx in groups:
                        rv, ev = idx[revealed[idx]], idx[is_eval[idx]]
                        if len(rv) == 0 or len(ev) == 0:
                            continue
                        rkk, a = hp[(f, src, k)]
                        C, s2 = sig[(f, src)][rkk]
                        p[ev] = b[ev] + G.gp_predict(C, s2, a, dpos[rv], dpos[ev], (y - b)[rv])
                    arms["krig_" + ("shared" if src == "shared" else "own")] = p
                for aname, p in arms.items():
                    for stratum, mm in [("all", ev_mask), ("rna", ev_mask & has_rna),
                                        ("no_rna", ev_mask & ~has_rna)]:
                        lines = [idx[mm[idx]] for _, idx in groups]
                        rho, rec = per_line_rank(y, p, lines)
                        rows.append(dict(seed=SEED, draw=draw, k=str(k), base=bname, arm=aname,
                                         stratum=stratum, r2=G.r2(y[mm], p[mm]),
                                         dc_r2=G.r2(y[mm] - dmean[mm], p[mm] - dmean[mm]),
                                         rho=rho, top10=rec))
    pd.DataFrame(rows).to_csv(G.ROOT / f"reports/novelty/kriging/rows/fu_rows_s{SEED}.csv", index=False)
    print(f"seed {SEED} done", flush=True)


if __name__ == "__main__":
    main()
