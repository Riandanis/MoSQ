#!/usr/bin/env python
"""Does the reconstruction term help where a missing-modality objective plausibly
should -- on the 406 cell lines with no RNA-seq? CPU only, no GPU, no retraining.

THE HYPOTHESIS
--------------
DR-A's reconstruction term literally trains the model to infer one modality from the
others. 406 of 998 cell lines have no RNA-seq and receive a learned `rna_fallback`
token (gastro_transformer/model.py). If the term earns its place anywhere, it is
there -- and the revision recommends removing it without ever having checked.

THE COMPARISON
--------------
`multiseed` (two-term, recon_weight=0.5) vs `dra_contrastive_only` (one-term,
recon_weight=0), both at 10 seeds on identical NCC folds. These are not merely
seed-paired: the two arms produce identical row sets with identical `y_true`, so this
is a ROW-paired, full-dataset, out-of-fold comparison. Asserted, not assumed.

PRIMARY TEST -- pre-registered before looking
---------------------------------------------
Difference-in-differences: per seed, (Delta_noRNA - Delta_RNA), Wilcoxon signed-rank
over 10 seeds. That is the actual test of "helps specifically where RNA is missing".
The three stratum-wise deltas are reported as descriptive only.

R2 LEVELS ARE NOT COMPARABLE ACROSS STRATA. The no-RNA rows are ~41% of the data and
have different target variance, so only the paired within-stratum delta and the DiD
are interpretable. Never compare has_rna vs no_rna R2 levels directly.

OUTPUTS (reports/recon_investigation/tables/)
    no_rna_paired_seedwise.csv   per (arm, seed, stratum) metrics
    no_rna_paired_tests.csv      the DiD and the stratum-wise paired tests
    no_rna_paired.md             narrative
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import ROOT, guard_artifacts, write_table, write_text  # noqa: E402
from gastro_transformer.metrics import compute_regression_metrics, paired_difference_ci  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REVISION = ROOT / "reports" / "revision"
TWO_TERM, ONE_TERM = "multiseed", "dra_contrastive_only"


def load_arm(experiment: str, arm: str, regime: str) -> pd.DataFrame:
    """Read this arm's test dumps directly.

    `predlog.load_all_predictions` globs */preds/*.parquet across all 30 experiment
    dirs (1530 files) and filters in memory. Reading the two directories we need is
    equivalent and ~15x faster; the schema is whatever predlog.dump_predictions wrote,
    and we assert the columns we rely on rather than trusting that.
    """
    files = sorted((REVISION / experiment / "preds").glob(f"{regime}__{arm}__seed*__fold*__test.parquet"))
    files = [f for f in files if "__fs" not in f.name]  # exclude partition-resampled runs
    if not files:
        raise FileNotFoundError(f"no dumps for {experiment}/{regime}/{arm}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    for col in ("seed", "fold", "cellline_id", "drug_id", "has_rna", "y_true", "y_pred"):
        if col not in df.columns:
            raise KeyError(f"{experiment}: dump missing required column {col!r}")
    return df


def assert_row_paired(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """The two arms must cover identical rows with identical targets."""
    keys = ["seed", "fold", "cellline_id", "drug_id"]
    a2 = a.sort_values(keys).reset_index(drop=True)
    b2 = b.sort_values(keys).reset_index(drop=True)
    if len(a2) != len(b2):
        raise AssertionError(f"row count differs: {len(a2)} vs {len(b2)}")
    for k in keys:
        if not a2[k].equals(b2[k]):
            raise AssertionError(f"key column {k!r} differs between arms")
    maxdiff = float(np.abs(a2.y_true.to_numpy() - b2.y_true.to_numpy()).max())
    if maxdiff > 1e-6:
        raise AssertionError(f"y_true differs between arms (max {maxdiff:.3g})")

    per_seed = a.groupby("seed").apply(
        lambda g: g[["cellline_id", "drug_id"]].drop_duplicates().shape[0], include_groups=False)
    facts = dict(
        n_rows_per_arm=len(a2), n_seeds=a.seed.nunique(),
        unique_pairs_per_seed=int(per_seed.iloc[0]),
        unique_pairs_consistent=bool((per_seed == per_seed.iloc[0]).all()),
        n_no_rna_rows=int((~a.has_rna).sum() / a.seed.nunique()),
        n_no_rna_celllines=int(a.loc[~a.has_rna, "cellline_id"].nunique()),
        n_rna_celllines=int(a.loc[a.has_rna, "cellline_id"].nunique()),
        y_true_maxdiff=maxdiff,
    )
    logger.info(f"pairing verified: {facts}")
    return facts


def seedwise_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Pool folds within a seed -> out-of-fold predictions over the whole dataset."""
    rows = []
    for seed, g in df.groupby("seed"):
        for stratum, sub in (("all", g), ("has_rna", g[g.has_rna]), ("no_rna", g[~g.has_rna])):
            m = compute_regression_metrics(sub.y_true.to_numpy(), sub.y_pred.to_numpy())
            rows.append(dict(arm=label, seed=int(seed), stratum=stratum, n=len(sub),
                             n_celllines=sub.cellline_id.nunique(),
                             r2=m["r2"], rmse=m["rmse"], mae=m["mae"]))
    return pd.DataFrame(rows)


def paired_test(a: np.ndarray, b: np.ndarray, name: str) -> dict:
    """a - b, paired over seeds. Wilcoxon plus a bootstrap CI on the difference."""
    d = a - b
    ci = paired_difference_ci(a, b, n_boot=20000, seed=0)
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    n = len(d)
    return dict(comparison=name, n_seeds=n, mean_diff=float(d.mean()),
                sd_diff=float(d.std(ddof=1)), ci_low=ci["ci_low"], ci_high=ci["ci_high"],
                ci_excludes_zero=bool(ci["ci_low"] > 0 or ci["ci_high"] < 0),
                wilcoxon_p=p, wilcoxon_floor=2 / 2 ** n if n <= 20 else 0.0,
                n_positive=int((d > 0).sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="multitoken_dra")
    ap.add_argument("--regime", default="ncc")
    args = ap.parse_args()
    guard_artifacts("entry")

    two = load_arm(TWO_TERM, args.arm, args.regime)
    one = load_arm(ONE_TERM, args.arm, args.regime)
    facts = assert_row_paired(two, one)

    m = pd.concat([seedwise_metrics(two, "two_term_w0.5"),
                   seedwise_metrics(one, "one_term_w0")], ignore_index=True)
    write_table(m, "no_rna_paired_seedwise.csv")

    piv = m.pivot_table(index=["stratum", "seed"], columns="arm", values="r2").reset_index()
    tests = []
    for stratum in ("all", "has_rna", "no_rna"):
        s = piv[piv.stratum == stratum].sort_values("seed")
        tests.append(paired_test(s["two_term_w0.5"].to_numpy(), s["one_term_w0"].to_numpy(),
                                 f"two_term - one_term ({stratum})"))

    # PRIMARY, pre-registered: difference-in-differences.
    d_no = (piv[piv.stratum == "no_rna"].sort_values("seed")["two_term_w0.5"].to_numpy()
            - piv[piv.stratum == "no_rna"].sort_values("seed")["one_term_w0"].to_numpy())
    d_yes = (piv[piv.stratum == "has_rna"].sort_values("seed")["two_term_w0.5"].to_numpy()
             - piv[piv.stratum == "has_rna"].sort_values("seed")["one_term_w0"].to_numpy())
    did = paired_test(d_no, d_yes, "PRIMARY DiD: (two-one)|no_rna - (two-one)|has_rna")
    tests.insert(0, did)

    tdf = pd.DataFrame(tests)
    write_table(tdf, "no_rna_paired_tests.csv")

    overall = tdf[tdf.comparison.str.contains(r"\(all\)")].iloc[0]
    L = ["# Does the reconstruction term help where RNA is missing?", "",
         f"Row-paired over **{facts['n_seeds']} seeds**; "
         f"{facts['unique_pairs_per_seed']:,} unique (cell line, drug) pairs per seed; "
         f"{facts['n_no_rna_rows']:,} no-RNA rows over "
         f"{facts['n_no_rna_celllines']} cell lines vs {facts['n_rna_celllines']} with RNA. "
         f"`y_true` identical between arms (max diff {facts['y_true_maxdiff']:.2g}).", "",
         "## Per-seed R2 by stratum", "",
         m.groupby(["stratum", "arm"]).r2.agg(["mean", "std"]).round(4).to_markdown(), "",
         "## Paired tests", "",
         tdf.to_markdown(index=False, floatfmt=".4f"), "",
         "## Reading", "",
         "**The primary test is the difference-in-differences**, first row: it asks "
         "whether the two-term objective helps *more* where RNA is absent than where "
         "it is present. The stratum-wise rows are descriptive.", "",
         "R2 levels must not be compared across strata -- the no-RNA rows are a "
         "different subpopulation with different target variance. Only within-stratum "
         "paired deltas and the DiD are interpretable.", "",
         f"**Verification gate:** the `(all)` row should reproduce the previously "
         f"computed 10-seed paired result (delta = -0.0004, CI [-0.0030, +0.0022]). "
         f"Measured here: {overall.mean_diff:+.4f}, CI "
         f"[{overall.ci_low:+.4f}, {overall.ci_high:+.4f}].", ""]
    write_text("\n".join(L), "no_rna_paired.md")

    guard_artifacts("exit")
    print("\n".join(L))


if __name__ == "__main__":
    main()
