#!/usr/bin/env python3
"""Score Experiment C-transfer as pre-registered (the pre-registration record, amendment
2026-10-01 14:35, which supersedes the offset primary).

Primary, k = 20, evaluation half, fold-paired over completed (seed, fold), per learner:
  (a) F1+krig - R1+krig  and  (b) F1+krig - pure kriging, each with 95% CI above zero.
POSITIVE = (a) and (b) for both learners; PARTIAL = one learner; else NULL.
Kriging = kriging_exploratory/mf_gate.py unchanged: shared Sigma from drug-centred
training-line responses; (rank, a) selected on inner training-line reveals.
Secondary (decide nothing): offset arms, F2/R2, k = 10, drug-centred R2, RNA strata.
CPU only; writes reports/novelty/tables/c_transfer_{folds,summary}.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
EXP = ROOT / "reports/novelty/novelty_c_transfer"
OUT = ROOT / "reports/novelty/tables"
LEARNERS = ["xgboost", "standalone_mlp"]
sys.path.insert(0, str(ROOT / "scripts/novelty/kriging_exploratory"))
import mf_gate as K  # noqa: E402  (exploratory kriging code, used unchanged)


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def load(arm, seed, fold):
    p = EXP / "preds" / f"ncc__{arm}__seed{seed}__fold{fold}__test.parquet"
    return pd.read_parquet(p).set_index("row_index") if p.exists() else None


def main():
    ic50 = pd.read_csv(ROOT / "data/ic50_data.csv", dtype={"cellline_id": str, "drug_id": str})
    ic50 = ic50.drop_duplicates(["cellline_id", "drug_id"], keep="first")
    rows = []
    for roles_path in sorted((EXP / "roles").glob("seed*_fold*.parquet")):
        seed, fold = (int(x) for x in roles_path.stem.replace("seed", "").split("_fold"))
        roles = pd.read_parquet(roles_path).set_index("row_index")
        test_lines = set(roles.cellline_id)
        tr = ic50[~ic50.cellline_id.isin(test_lines)].copy()
        dmean = tr.groupby("drug_id").ic50_value.mean()
        ev = roles[roles.is_eval]
        # Kriging covariance from drug-centred TRAINING-line responses only.
        tr["resid"] = tr.ic50_value - tr.drug_id.map(dmean)
        drugs = np.array(sorted(ic50.drug_id.unique()))
        dpos = {d: i for i, d in enumerate(drugs)}
        sig = K.make_sigma(K.matrix(tr, np.array(sorted(tr.cellline_id.unique())), drugs))
        hps = {}
        for k in (10, 20):
            rev = roles[roles.pool_rank <= k]
            if k not in hps:
                hps[k], _ = K.select_hparams(tr, drugs, k, np.random.default_rng(777 + 100 * seed + fold))
            rank_k, a_k = hps[k]
            C, s2 = sig[rank_k]

            def krig(base_all):
                """base_all: Series over all test rows. Returns kriged eval-half predictions."""
                base_all = base_all.astype(np.float64)
                out = base_all.loc[ev.index].copy()
                for line, g in roles.groupby("cellline_id"):
                    rv = g.index[g.pool_rank <= k]
                    evi = g.index[g.is_eval]
                    if len(rv) == 0 or len(evi) == 0:
                        continue
                    r = (roles_y.loc[rv] - base_all.loc[rv]).to_numpy()
                    out.loc[evi] = base_all.loc[evi].to_numpy() + K.gp_predict(
                        C, s2, a_k, g.drug_id.loc[rv].map(dpos).to_numpy(),
                        g.drug_id.loc[evi].map(dpos).to_numpy(), r)
                return out
            for learner in LEARNERS:
                arms = {"F1": load(f"F1_k{k}_{learner}", seed, fold),
                        "R1": load(f"R1_{learner}", seed, fold),
                        "F2": load(f"F2_k{k}_{learner}", seed, fold),
                        "R2": load(f"R2_k{k}_{learner}", seed, fold)}
                if any(v is None for v in arms.values()):
                    continue
                preds = {}
                roles_y = arms["R1"].y_true
                preds["F1K"] = krig(arms["F1"].y_pred)
                preds["R1K"] = krig(arms["R1"].y_pred)
                preds["PK"] = krig(roles.drug_id.map(dmean).astype(float))
                for name in ("F1", "R1"):
                    d = arms[name]
                    resid = (d.y_true - d.y_pred).loc[rev.index]
                    off = resid.groupby(rev.cellline_id).mean()
                    preds[name] = d.y_pred.loc[ev.index] + ev.cellline_id.map(off).fillna(0)
                for name in ("F2", "R2"):
                    preds[name] = arms[name].y_pred.loc[ev.index]
                y = arms["R1"].y_true.loc[ev.index]
                dm = ev.drug_id.map(dmean)
                has_rna = arms["R1"].has_rna.loc[ev.index].astype(bool)
                for name, f in preds.items():
                    for stratum, m in (("all", np.ones(len(y), bool)), ("rna", has_rna.values),
                                       ("no_rna", ~has_rna.values)):
                        ok = m & dm.notna().values
                        rows.append(dict(seed=seed, fold=fold, k=k, learner=learner, arm=name,
                                         stratum=stratum, r2=r2(y[m], f[m]),
                                         dc_r2=r2((y - dm)[ok], (f - dm)[ok])))
    if not rows:
        sys.exit("no completed folds yet")
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "c_transfer_folds.csv", index=False)

    summ = []
    for (k, learner, stratum), g in res.groupby(["k", "learner", "stratum"]):
        p = g.pivot_table(index=["seed", "fold"], columns="arm", values=["r2", "dc_r2"])
        for a, b in (("F1K", "R1K"), ("F1K", "PK"), ("R1K", "PK"), ("F1", "R1"), ("F2", "R2")):
            d = p["r2"][a] - p["r2"][b]
            ci = (stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
                  if len(d) > 1 else (np.nan, np.nan))
            summ.append(dict(
                k=k, learner=learner, stratum=stratum, test=f"{a}-{b}", n_pairs=len(d),
                seeds=",".join(map(str, sorted(g.seed.unique()))),
                r2_a=p["r2"][a].mean(), r2_b=p["r2"][b].mean(), delta_r2=d.mean(),
                ci_lo=ci[0], ci_hi=ci[1], positive=f"{int((d > 0).sum())}/{len(d)}",
                wilcoxon_p=stats.wilcoxon(d).pvalue if len(d) >= 5 else np.nan,
                dc_a=p["dc_r2"][a].mean(), dc_b=p["dc_r2"][b].mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "c_transfer_summary.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(summ.round(4).to_string(index=False))

    prim = summ[(summ.k == 20) & (summ.stratum == "all")]
    wins, n = {}, 0
    for learner in LEARNERS:
        a = prim[(prim.learner == learner) & (prim.test == "F1K-R1K")]
        b = prim[(prim.learner == learner) & (prim.test == "F1K-PK")]
        if len(a) and len(b):
            n = int(a.n_pairs.iloc[0])
            wins[learner] = dict(a_vs_raw_krig=bool(a.ci_lo.iloc[0] > 0),
                                 b_vs_pure_krig=bool(b.ci_lo.iloc[0] > 0))
    ok = [all(v.values()) for v in wins.values()]
    branch = "POSITIVE" if len(ok) == 2 and all(ok) else "PARTIAL" if any(ok) else "NULL"
    print(f"\nPRIMARY (amended; k=20; (a) F1K-R1K and (b) F1K-PK; {n} fold-pairs"
          f"{'; FINAL' if n == 15 else '; INTERIM -- not a verdict'}): {wins} -> {branch}")


if __name__ == "__main__":
    main()
