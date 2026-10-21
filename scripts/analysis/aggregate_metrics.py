#!/usr/bin/env python3
"""Recompute all metrics from prediction dumps.

Beyond R^2 this reports absolute-error metrics (RMSE, MAE, MALE), an
experimental-variability-aware statistic (EVAPA), and the comparison between the
592 cell lines with RNA-seq and the 406 without.

All of them fall out of the prediction dumps with no GPU time, because every dump
row carries its cell-line id, drug id and RNA-availability flag.

Outputs (under ``reports/revision/tables/``):
  metrics_full.csv    per (experiment, regime, arm, seed, fold)
  metrics_summary.csv mean +- SD across folds and seeds
  rna_subset.csv      the 592-with-RNA vs 406-without breakdown
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer.metrics import compute_regression_metrics  # noqa: E402
from gastro_transformer.predlog import load_all_predictions  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GROUP_KEYS = ["experiment", "regime", "arm", "seed", "fold_seed", "fold"]
METRIC_COLS = ["r2", "rmse", "mae", "male", "evapa", "pearson_r", "spearman_r"]


def per_fold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(GROUP_KEYS, sort=True):
        metrics = compute_regression_metrics(sub.y_true.values, sub.y_pred.values)
        metrics.update(dict(zip(GROUP_KEYS, keys)))
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over folds and seeds.

    ``n_folds`` and ``n_seeds`` are reported alongside to make clear what the
    spread is over.
    """
    # Only the primary partition (fold_seed = -1) belongs in the headline summary.
    # Runs with fold_seed >= 0 resample the *split*, which is a different design
    # answering a different question; averaging the two would report a mean over a
    # mixture of partitions as though it were replication over seeds.
    # ``partition_spread`` covers those separately.
    metrics = metrics[metrics.fold_seed == -1]
    grouped = metrics.groupby(["experiment", "regime", "arm"])
    summary = grouped[METRIC_COLS].agg(["mean", "std"]).round(4)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_folds"] = grouped.fold.nunique()
    summary["n_seeds"] = grouped.seed.nunique()
    summary["n_runs"] = grouped.size()

    # ``r2_std`` above is the spread over every fold-run, so it mixes fold-to-fold
    # and seed-to-seed variation. Reproducibility across *seeds* is the quantity of
    # interest, and the paired tests operate on per-seed means,
    # so report that dispersion separately rather than letting one number stand in
    # for both. For MoSQ under NCC the two differ by more than 3x (0.0097 pooled
    # against 0.0028 across seed means), which is large enough to mislead.
    seed_means = metrics.groupby(["experiment", "regime", "arm", "seed"])["r2"].mean()
    per_seed = seed_means.groupby(["experiment", "regime", "arm"]).agg(
        r2_seed_mean="mean", r2_seed_std="std"
    ).round(4)
    summary = summary.join(per_seed)
    return summary.reset_index()


def rna_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Performance split by RNA-seq availability.

    The 406 cell lines without RNA-seq are zero-filled and flagged by a boolean
    mask; MoSQ substitutes a learned fallback token for them while the baselines
    see literal zeros. This table is what quantifies the resulting gap.
    """
    rows = []
    for keys, sub in df.groupby(["experiment", "regime", "arm", "seed", "has_rna"], sort=True):
        metrics = compute_regression_metrics(sub.y_true.values, sub.y_pred.values)
        metrics.update(
            dict(zip(["experiment", "regime", "arm", "seed", "has_rna"], keys))
        )
        metrics["n_celllines"] = sub.cellline_id.nunique()
        rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    parser.add_argument("--experiment", default=None, help="Limit to one experiment")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    tables = out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    df = load_all_predictions(out_root, experiment=args.experiment)
    logger.info(
        f"Loaded {len(df):,} predictions across "
        f"{df.groupby(['experiment', 'regime', 'arm', 'seed']).ngroups} runs"
    )

    metrics = per_fold_metrics(df)
    metrics.to_csv(tables / "metrics_full.csv", index=False)

    summary = summarise(metrics)
    summary.to_csv(tables / "metrics_summary.csv", index=False)

    subset = rna_subset(df)
    subset.to_csv(tables / "rna_subset.csv", index=False)

    logger.info("\n" + summary[
        ["regime", "arm", "r2_mean", "r2_std", "rmse_mean", "mae_mean", "evapa_mean", "n_runs"]
    ].to_string(index=False))

    if subset.has_rna.nunique() > 1:
        pivot = subset.pivot_table(index=["regime", "arm"], columns="has_rna", values="r2")
        pivot.columns = ["no_rna_406", "has_rna_592"]
        pivot["delta"] = (pivot.has_rna_592 - pivot.no_rna_406).round(4)
        logger.info("\nRNA-availability breakdown (R^2):\n" + pivot.round(4).to_string())

    logger.info(f"\nWrote tables to {tables}")


if __name__ == "__main__":
    main()
