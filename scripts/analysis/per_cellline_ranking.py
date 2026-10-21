#!/usr/bin/env python3
"""Within-cell-line drug ranking.

Two questions: can the model rank candidate drugs for an individual unseen cell
line (what precision oncology needs), and is it learning drug-cell interaction or
memorising baseline drug toxicity?

The measurement that settles both is comparison against the per-drug-mean
baseline, not against zero. That baseline assigns every cell line the *same* drug
ordering, and it already achieves a per-cell-line Spearman of about 0.85 -- so a
high absolute rank correlation is not by itself evidence of interaction modelling.
Two statistics separate the claims:

``per_cellline_spearman``  absolute ranking quality. Compare to the drug-mean
                           baseline's value, never to zero.
``r2_drug_centered``       R^2 after subtracting each drug's training-fold mean
                           from predictions and targets alike. The drug-mean
                           baseline scores exactly 0 here by construction, so
                           this isolates what the model adds beyond knowing which
                           drug it is.

Runs entirely on prediction dumps; no GPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer.metrics import compute_regression_metrics, topk_recall  # noqa: E402
from gastro_transformer.predlog import load_all_predictions  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RUN_KEYS = ["experiment", "regime", "arm", "seed"]


def drug_mean_lookup(df: pd.DataFrame) -> pd.Series:
    """Each drug's mean true IC50 over the rows *outside* its own test fold.

    Using out-of-fold rows keeps the centring free of test-fold information, which
    matters because this offset is what the drug-centred R^2 is measured against.
    """
    totals = df.groupby("drug_id").y_true.sum()
    counts = df.groupby("drug_id").y_true.size()
    fold_totals = df.groupby(["drug_id", "fold"]).y_true.sum()
    fold_counts = df.groupby(["drug_id", "fold"]).y_true.size()

    out_of_fold = (
        (totals - fold_totals.unstack("fold").T).T / (counts - fold_counts.unstack("fold").T).T
    )
    return out_of_fold


def analyse_run(df: pd.DataFrame, min_drugs: int, k: int) -> dict:
    rhos, recalls = [], []
    for _, sub in df.groupby("cellline_id"):
        if len(sub) < min_drugs:
            continue
        y_true, y_pred = sub.y_true.to_numpy(), sub.y_pred.to_numpy()
        if np.std(y_pred) > 0 and np.std(y_true) > 0:
            rhos.append(stats.spearmanr(y_true, y_pred)[0])
        recalls.append(topk_recall(y_true, y_pred, k=k, sensitive_is_low=True))

    offsets = drug_mean_lookup(df)
    offset_per_row = np.array(
        [
            offsets.loc[d, f] if (d in offsets.index and f in offsets.columns) else np.nan
            for d, f in zip(df.drug_id, df.fold)
        ]
    )
    valid = np.isfinite(offset_per_row)
    centred = compute_regression_metrics(
        df.y_true.to_numpy()[valid] - offset_per_row[valid],
        df.y_pred.to_numpy()[valid] - offset_per_row[valid],
    )

    return {
        "n_celllines_evaluated": int(len(recalls)),
        "n_celllines_with_ordering": int(len(rhos)),
        "per_cellline_spearman_mean": float(np.mean(rhos)) if rhos else np.nan,
        "per_cellline_spearman_std": float(np.std(rhos)) if rhos else np.nan,
        "per_cellline_spearman_median": float(np.median(rhos)) if rhos else np.nan,
        f"top{k}_recall_mean": float(np.nanmean(recalls)) if recalls else np.nan,
        f"top{k}_recall_std": float(np.nanstd(recalls)) if recalls else np.nan,
        "r2_drug_centered": centred["r2"],
        "spearman_drug_centered": centred["spearman_r"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--min_drugs", type=int, default=20)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    tables = out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    df = load_all_predictions(out_root, experiment=args.experiment)
    logger.info(f"Loaded {len(df):,} predictions")

    rows = []
    for keys, sub in df.groupby(RUN_KEYS, sort=True):
        record = analyse_run(sub, args.min_drugs, args.topk)
        record.update(dict(zip(RUN_KEYS, keys)))
        rows.append(record)

    result = pd.DataFrame(rows)
    out_path = tables / "per_cellline_ranking.csv"
    result.to_csv(out_path, index=False)

    summary = (
        result.groupby(["regime", "arm"])[
            [
                "per_cellline_spearman_mean",
                f"top{args.topk}_recall_mean",
                "r2_drug_centered",
            ]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    logger.info("\n" + summary.to_string())

    baseline = result[result.arm == "drug_mean"]
    if not baseline.empty:
        rho_bar = baseline.per_cellline_spearman_mean.mean()
        logger.info(
            f"\nPer-drug-mean baseline reference: per-cell-line rho = {rho_bar:.4f}, "
            "drug-centred R^2 = 0 by construction. Any arm must be read against "
            "these, not against zero."
        )

    logger.info(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
