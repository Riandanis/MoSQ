#!/usr/bin/env python3
"""PASS / FAIL verdict for novelty steps 2 (A1) and 3 (aux SupCon), as pre-registered.

PASS requires BOTH:
  (a) R2 gain over the random-init Q-Former (reports/revision/multiseed, qformer_randinit,
      same seed => same folds) has a 95% CI above zero, fold-paired over seeds 0-2 x 5 folds;
  (b) drug-centred R2 >= 0.1226 (random-init's), mean over complete seeds.
Either failing => NULL, reported as a negative result; no further variants.

Usage: score_pass.py a1 | aux
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from per_cellline_ranking import analyse_run  # noqa: E402

STEPS = {
    "a1": ("Step 2 (A1): inductive DR-A, per-drug rank bins",
           "reports/novelty/novelty_a1_per_drug_rank/preds/strict_ncc__multitoken_dra__seed{s}__fold*__test.parquet"),
    "aux": ("Step 3: DR-A as auxiliary fine-tuning loss (lambda=0.5)",
            "reports/novelty/novelty_aux_supcon/preds/ncc__qformer_randinit_auxsupcon__seed{s}__fold*__test.parquet"),
}
REF = "reports/revision/multiseed/preds/ncc__qformer_randinit__seed{s}__fold{f}__test.parquet"
# Criterion (b)'s bar = random-init Q-Former's drug-centred R2, 10-seed NCC mean, read from the
# repo table (it is 0.1226). Never hard-coded.
_rank = pd.read_csv(ROOT / "reports/revision/tables/per_cellline_ranking.csv")
DC_BAR = round(float(_rank[(_rank.experiment == "multiseed") & (_rank.regime == "ncc")
                           & (_rank.arm == "qformer_randinit")].r2_drug_centered.mean()), 4)


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def rna_gain(df):
    """R2 on RNA-available rows minus R2 on RNA-missing rows, pooled over a seed's folds."""
    return (r2(df[df.has_rna].y_true, df[df.has_rna].y_pred)
            - r2(df[~df.has_rna].y_true, df[~df.has_rna].y_pred))


def main(step):
    title, pattern = STEPS[step]
    pairs, dcs, gains, ref_gains = [], [], [], []
    for s in range(3):
        files = sorted(glob.glob(str(ROOT / pattern.format(s=s))))
        if not files:
            continue
        df = pd.concat(pd.read_parquet(f) for f in files)
        for f, sub in df.groupby("fold"):
            ref = pd.read_parquet(ROOT / REF.format(s=s, f=f))
            assert set(ref.row_index) == set(sub.row_index), (s, f)
            pairs.append(dict(seed=s, fold=f, model=r2(sub.y_true, sub.y_pred),
                              randinit=r2(ref.y_true, ref.y_pred)))
        if df.fold.nunique() == 5:
            dcs.append(analyse_run(df, 20, 10)["r2_drug_centered"])
            gains.append(rna_gain(df))
            # Reference: random init on the SAME seed, same pooled-per-seed method as
            # rna_subset.csv. Never hard-code it: a hard-coded +0.0192 (NCC and random-split
            # rows averaged together) was quoted on 2026-10-01; correct is +0.0156 for s0-2.
            ref = pd.concat(pd.read_parquet(ROOT / REF.format(s=s, f=f)) for f in range(5))
            ref_gains.append(rna_gain(ref))
    if not pairs:
        print(f"{title}: no results yet")
        return
    t = pd.DataFrame(pairs)
    d = t.model - t.randinit
    lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
    dc = float(np.mean(dcs)) if dcs else float("nan")
    a, b = lo > 0, dc >= DC_BAR
    final = len(d) == 15 and len(dcs) == 3
    print(f"{title}  [{'FINAL' if final else 'INTERIM'}: {len(d)}/15 fold-pairs, {len(dcs)}/3 complete seeds]")
    print(f"  R2 {t.model.mean():.4f} vs random init {t.randinit.mean():.4f}")
    print(f"  (a) R2 gain {d.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], {int((d > 0).sum())}/{len(d)} positive, "
          f"Wilcoxon p={stats.wilcoxon(d).pvalue:.2e}  -> {'PASS' if a else 'FAIL'}")
    print(f"  (b) drug-centred R2 {dc:.4f} vs bar {DC_BAR}  -> {'PASS' if b else 'FAIL'}")
    print(f"  RNA gain {np.mean(gains):+.4f} vs random init {np.mean(ref_gains):+.4f} on the same "
          f"{len(gains)} seed(s), pooled per seed (secondary)" if gains else "  RNA gain: n/a")
    verdict = "PASS" if (a and b) else ("HARMFUL (null, negative result)" if hi < 0 else "NULL (negative result)")
    print(f"  VERDICT: {verdict}{'' if final else ' -- provisional'}")
    out = ROOT / "reports/novelty/tables"
    t.assign(delta=d).to_csv(out / f"score_pass_{step}_fold_pairs.csv", index=False)
    pd.DataFrame([dict(step=step, final=final, n_fold_pairs=len(d), n_complete_seeds=len(dcs),
                       r2_model=t.model.mean(), r2_randinit=t.randinit.mean(), delta=d.mean(),
                       ci_lo=lo, ci_hi=hi, n_positive=int((d > 0).sum()),
                       wilcoxon_p=stats.wilcoxon(d).pvalue, dc_r2=dc, dc_bar=DC_BAR,
                       rna_gain=np.mean(gains) if gains else np.nan,
                       rna_gain_randinit=np.mean(ref_gains) if ref_gains else np.nan,
                       crit_a=bool(a), crit_b=bool(b), verdict=verdict)]).to_csv(
        out / f"score_pass_{step}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
