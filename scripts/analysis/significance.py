#!/usr/bin/env python3
"""Paired significance testing across seeds and folds.

Wilcoxon signed-rank tests on seed- and fold-wise paired results across the random,
NCC and NCD splits. Several of the differences of interest are small relative to
fold-to-fold variation, so the pairing matters.

Two tests are reported, deliberately:

**Primary** -- Wilcoxon on per-seed mean R^2, paired across arms. Seeds are
independent runs, so these are genuinely independent paired observations. The
cost is power: at n=5 a two-sided signed-rank test has a minimum attainable p of
0.0625, so it *cannot* reach p<0.05 no matter how large the effect. Where that
happens, the bootstrap CI on the paired difference is the honest statistic to
quote, and the limitation should be stated rather than worked around.

**Secondary** -- the same test over all (seed, fold) pairs. Higher power, but
folds within a seed share training data, so these are not independent
observations. Labelled as pseudo-replicated wherever it appears.

Switching to a paired t-test to obtain a smaller p-value would be test-shopping;
the t-test is reported only as a third column for continuity with the previous
analysis, never as the headline.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer.metrics import paired_difference_ci  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WILCOXON_MIN_P = {3: 0.25, 4: 0.125, 5: 0.0625, 6: 0.03125}


def stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided signed-rank p, or NaN where the test is not defined."""
    diff = a - b
    if diff.size < 3 or np.allclose(diff, 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(a, b)[1])
    except ValueError:
        return float("nan")


def compare(metrics: pd.DataFrame, metric: str = "r2") -> pd.DataFrame:
    rows = []

    for (experiment, regime), sub in metrics.groupby(["experiment", "regime"]):
        arms = sorted(sub.arm.unique())

        # Per-seed means: the independent unit.
        seed_means = sub.groupby(["arm", "seed"])[metric].mean().unstack("arm")
        # Per (seed, fold): pseudo-replicated but higher powered.
        fold_level = sub.set_index(["seed", "fold", "arm"])[metric].unstack("arm")

        for arm_a, arm_b in itertools.combinations(arms, 2):
            paired_seeds = seed_means[[arm_a, arm_b]].dropna()
            paired_folds = fold_level[[arm_a, arm_b]].dropna()
            if len(paired_seeds) < 2:
                continue

            a_seed = paired_seeds[arm_a].to_numpy()
            b_seed = paired_seeds[arm_b].to_numpy()
            a_fold = paired_folds[arm_a].to_numpy()
            b_fold = paired_folds[arm_b].to_numpy()

            n_seeds = len(paired_seeds)
            p_seed = _wilcoxon(a_seed, b_seed)
            ci = paired_difference_ci(a_seed, b_seed)

            rows.append(
                {
                    "experiment": experiment,
                    "regime": regime,
                    "metric": metric,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "mean_a": round(float(a_seed.mean()), 4),
                    "mean_b": round(float(b_seed.mean()), 4),
                    "delta": round(float(a_seed.mean() - b_seed.mean()), 4),
                    "n_seeds": n_seeds,
                    "wilcoxon_p_seedwise": p_seed,
                    "sig_seedwise": stars(p_seed),
                    "wilcoxon_min_attainable_p": WILCOXON_MIN_P.get(n_seeds, np.nan),
                    "seedwise_test_underpowered": bool(
                        n_seeds in WILCOXON_MIN_P and WILCOXON_MIN_P[n_seeds] >= 0.05
                    ),
                    "boot_ci_low": round(ci["ci_low"], 4),
                    "boot_ci_high": round(ci["ci_high"], 4),
                    "ci_excludes_zero": bool(ci["ci_low"] > 0 or ci["ci_high"] < 0),
                    "n_fold_pairs": len(paired_folds),
                    "wilcoxon_p_foldwise_pseudorep": _wilcoxon(a_fold, b_fold),
                    "ttest_p_seedwise_secondary": float(stats.ttest_rel(a_seed, b_seed)[1])
                    if n_seeds > 1
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


# Significance testing across the random, NCC and NCD data splits: NCC and random fall out of compare() directly, but MoSQ's drug-blind
# result lives in the ``leakfree_ncd`` experiment while its comparators live in
# ``ncd_baselines``, and compare() groups by experiment so the two never meet.
#
# Pairing them is legitimate here specifically because drug-blind fold assignment
# is deterministic: ``create_ncd_sample_folds`` ignores the seed, so fold k holds
# out the same compounds in both experiments. That quirk (drug-blind folds do not
# vary with --seed; use --fold_seed) is used here to advantage. It would NOT be valid for a regime
# whose partition varies with the seed.
CROSS_EXPERIMENT = [
    # (regime, experiment_a, arm_a, experiment_b, arm_b)
    ("ncd", "leakfree_ncd", "multitoken_dra", "ncd_baselines", "standalone_mlp"),
    ("ncd", "leakfree_ncd", "multitoken_dra", "ncd_baselines", "xgboost"),
    ("ncd", "leakfree_ncd", "multitoken_dra", "ncd_baselines", "tcnn_style"),
    ("ncd", "leakfree_ncd", "multitoken_dra", "ncd_baselines", "qformer_randinit"),
]


def compare_cross_experiment(metrics: pd.DataFrame, metric: str = "r2") -> pd.DataFrame:
    """Paired tests for arms that live in different experiments but share folds."""
    rows = []
    for regime, exp_a, arm_a, exp_b, arm_b in CROSS_EXPERIMENT:
        sel_a = metrics[(metrics.experiment == exp_a) & (metrics.regime == regime)
                        & (metrics.arm == arm_a)]
        sel_b = metrics[(metrics.experiment == exp_b) & (metrics.regime == regime)
                        & (metrics.arm == arm_b)]
        if sel_a.empty or sel_b.empty:
            continue
        a = sel_a.groupby("seed")[metric].mean()
        b = sel_b.groupby("seed")[metric].mean()
        shared = sorted(set(a.index) & set(b.index))
        if len(shared) < 2:
            continue
        av, bv = a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
        p = _wilcoxon(av, bv)
        ci = paired_difference_ci(av, bv)
        rows.append({
            "experiment": f"{exp_a}|{exp_b}", "regime": regime, "metric": metric,
            "arm_a": arm_a, "arm_b": arm_b,
            "mean_a": round(float(av.mean()), 4), "mean_b": round(float(bv.mean()), 4),
            "delta": round(float(av.mean() - bv.mean()), 4),
            "n_seeds": len(shared), "wilcoxon_p_seedwise": p, "sig_seedwise": stars(p),
            "wilcoxon_min_attainable_p": WILCOXON_MIN_P.get(len(shared), np.nan),
            "seedwise_test_underpowered": bool(
                len(shared) in WILCOXON_MIN_P and WILCOXON_MIN_P[len(shared)] >= 0.05),
            "boot_ci_low": round(ci["ci_low"], 4), "boot_ci_high": round(ci["ci_high"], 4),
            "ci_excludes_zero": bool(ci["ci_low"] > 0 or ci["ci_high"] < 0),
            "n_fold_pairs": np.nan, "wilcoxon_p_foldwise_pseudorep": np.nan,
            "ttest_p_seedwise_secondary": np.nan,
        })
    return pd.DataFrame(rows)


# The mean-based baseline also needs paired significance testing, and the
# intersection was missing: compare() pairs across seeds, and the mean
# baselines are deterministic -- one value per fold, no seed variation -- so they never
# entered it. Folds are the paired unit here, which is the correct instrument for a
# deterministic comparator rather than a workaround for one.
#
# n = 5 folds floors the two-sided signed-rank test at p = 0.0625, so it cannot reach
# p < 0.05 however large the effect. The bootstrap CI is the statistic to quote, and the
# floor is emitted alongside it rather than left for the reader to infer.
def compare_vs_mean_baseline(
    metrics: pd.DataFrame, tables: Path, metric: str = "r2",
    baseline_arm: str = "drug_mean",
) -> pd.DataFrame:
    rows = []

    for regime in ("ncc", "ncd"):
        baseline_path = tables / f"mean_baselines_{regime}.csv"
        if not baseline_path.exists():
            continue
        base = pd.read_csv(baseline_path)
        base = base[base.arm == baseline_arm]
        if base.empty:
            continue
        base_by_fold = base.set_index("fold")[metric].sort_index()

        sub = metrics[metrics.regime == regime]
        for (experiment, arm), grp in sub.groupby(["experiment", "arm"]):
            # Average over seeds within each fold: the baseline has no seed axis, so
            # the fold is the only unit both sides share.
            model_by_fold = grp.groupby("fold")[metric].mean().sort_index()
            shared = model_by_fold.index.intersection(base_by_fold.index)
            if len(shared) < 3:
                continue

            a = model_by_fold.loc[shared].to_numpy()
            b = base_by_fold.loc[shared].to_numpy()
            p = _wilcoxon(a, b)
            ci = paired_difference_ci(a, b)
            n_folds = len(shared)

            rows.append({
                "experiment": experiment,
                "regime": regime,
                "metric": metric,
                "arm_a": arm,
                "arm_b": f"{baseline_arm} (deterministic baseline)",
                "mean_a": round(float(a.mean()), 4),
                "mean_b": round(float(b.mean()), 4),
                "delta": round(float(a.mean() - b.mean()), 4),
                "n_folds_paired": n_folds,
                "n_seeds_averaged": int(grp.seed.nunique()),
                "wilcoxon_p_foldwise": p,
                "sig_foldwise": stars(p),
                "wilcoxon_min_attainable_p": WILCOXON_MIN_P.get(n_folds, np.nan),
                "foldwise_test_underpowered": bool(
                    n_folds in WILCOXON_MIN_P and WILCOXON_MIN_P[n_folds] >= 0.05
                ),
                "boot_ci_low": round(ci["ci_low"], 4),
                "boot_ci_high": round(ci["ci_high"], 4),
                "ci_excludes_zero": bool(ci["ci_low"] > 0 or ci["ci_high"] < 0),
                "n_folds_positive": int((a > b).sum()),
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    parser.add_argument("--metric", default="r2")
    args = parser.parse_args()

    tables = Path(args.out_root) / "tables"
    metrics_path = tables / "metrics_full.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run scripts/analysis/aggregate_metrics.py first."
        )

    metrics = pd.read_csv(metrics_path)
    result = compare(metrics, metric=args.metric)
    if result.empty:
        logger.warning("Not enough paired runs to test yet (need >= 2 seeds per arm).")
        return

    out_path = tables / f"significance_{args.metric}.csv"
    cross = compare_cross_experiment(metrics)
    if len(cross):
        result = pd.concat([result, cross], ignore_index=True)
        logger.info(f'added {len(cross)} cross-experiment NCD comparisons')
    result.to_csv(out_path, index=False)

    view = result[
        [
            "regime", "arm_a", "arm_b", "mean_a", "mean_b", "delta",
            "n_seeds", "wilcoxon_p_seedwise", "sig_seedwise",
            "boot_ci_low", "boot_ci_high", "ci_excludes_zero",
        ]
    ]
    logger.info("\n" + view.to_string(index=False))

    # The mean baseline, significance-tested. Separate file because the
    # paired unit is the fold, not the seed, and mixing the two in one table invites
    # reading a fold-paired p as if it were seed-paired.
    vs_mean = compare_vs_mean_baseline(metrics, tables, metric=args.metric)
    if len(vs_mean):
        vs_mean_path = tables / f"significance_{args.metric}_vs_mean_baseline.csv"
        vs_mean.to_csv(vs_mean_path, index=False)
        logger.info(f"wrote {len(vs_mean)} mean-baseline comparisons to {vs_mean_path.name}")
        headline = vs_mean[
            (vs_mean.experiment == "multiseed") & (vs_mean.regime == "ncc")
        ][["arm_a", "mean_a", "mean_b", "delta", "n_folds_paired",
           "wilcoxon_p_foldwise", "boot_ci_low", "boot_ci_high", "n_folds_positive"]]
        if not headline.empty:
            logger.info("\nvs per-drug mean (NCC, fold-paired):\n"
                        + headline.to_string(index=False))

    underpowered = result[result.seedwise_test_underpowered]
    if not underpowered.empty:
        n_seeds = int(underpowered.n_seeds.iloc[0])
        logger.warning(
            f"\nWith {n_seeds} seeds the two-sided signed-rank test cannot go below "
            f"p={WILCOXON_MIN_P.get(n_seeds)}, so it can never reach p<0.05. "
            "Quote the bootstrap CI on the paired difference and state this limit "
            "explicitly rather than substituting a different test."
        )

    logger.info(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
