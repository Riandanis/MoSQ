#!/usr/bin/env python3
"""Mean-based baselines and within-cell-line ranking.

A mean-based baseline asks whether machine learning helps at all (as in Garai et
al.). It also tests a specific concern: global IC50 quintiles may encode drug
identity rather than biological sensitivity, so a model may be memorising baseline
drug toxicity.

A predictor that ignores the cell line entirely and returns each drug's
training-fold mean reaches R^2 = 0.725 under cell-line-disjoint CV. That is the
number the model has to be judged against, not zero.

The per-drug-mean predictor assigns every cell line the *same* drug ordering, so
its per-cell-line Spearman is a fixed value that a model capturing real drug-cell
interaction must beat. This script computes both framings, and writes predictions
in the standard dump format so the baselines behave as ordinary arms in every
downstream analysis.

Runs on CPU in a couple of minutes; no checkpoint required.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer import splits  # noqa: E402
from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset  # noqa: E402
from gastro_transformer.metrics import compute_regression_metrics, topk_recall  # noqa: E402
from gastro_transformer.predlog import dump_predictions  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DRUG_CSV = str(ROOT / "data/drug_embeddings.csv")
IC50_CSV = str(ROOT / "data/ic50_data.csv")
RNA_CSV = str(ROOT / "data/ccle_rna_for_ic50.csv")

BASELINES = ("global_mean", "drug_mean", "cellline_mean")


def fit_predict(kind: str, frame: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray) -> np.ndarray:
    """Fit a mean baseline on the training rows and predict the test rows.

    Unseen keys fall back to the training global mean. Under cell-line-disjoint
    folds every test cell line is unseen by construction, so ``cellline_mean``
    degenerates to the global mean -- which is itself the point worth reporting.
    """
    train, test = frame[train_mask], frame[test_mask]
    global_mean = float(train.y.mean())

    if kind == "global_mean":
        return np.full(len(test), global_mean, dtype=np.float64)
    if kind == "drug_mean":
        lookup = train.groupby("drug_id").y.mean()
        return test.drug_id.map(lookup).fillna(global_mean).to_numpy(dtype=np.float64)
    if kind == "cellline_mean":
        lookup = train.groupby("cellline_id").y.mean()
        return test.cellline_id.map(lookup).fillna(global_mean).to_numpy(dtype=np.float64)
    raise ValueError(f"unknown baseline {kind!r}")


def ranking_metrics(frame: pd.DataFrame, min_drugs: int = 20, k: int = 10) -> dict:
    """Per-cell-line Spearman and top-k recall, averaged over cell lines.

    Ranking within a cell line removes drug-level marginal effects, which is
    exactly what a test of drug-toxicity memorisation calls for. Cell lines with
    fewer than ``min_drugs`` test rows are excluded: a rank correlation over a
    handful of drugs is too noisy to average.
    """
    rhos, recalls, n_used = [], [], 0
    for _, sub in frame.groupby("cellline_id"):
        if len(sub) < min_drugs:
            continue
        n_used += 1
        y_true, y_pred = sub.y_true.to_numpy(), sub.y_pred.to_numpy()
        # A constant prediction has no ordering, so Spearman is undefined, not zero.
        if np.std(y_pred) > 0 and np.std(y_true) > 0:
            from scipy import stats

            rhos.append(stats.spearmanr(y_true, y_pred)[0])
        recalls.append(topk_recall(y_true, y_pred, k=k, sensitive_is_low=True))

    return {
        "n_celllines_evaluated": n_used,
        "n_celllines_with_ordering": len(rhos),
        "per_cellline_spearman_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "per_cellline_spearman_std": float(np.std(rhos)) if rhos else float("nan"),
        "per_cellline_spearman_median": float(np.median(rhos)) if rhos else float("nan"),
        f"top{k}_recall_mean": float(np.nanmean(recalls)) if recalls else float("nan"),
        f"top{k}_recall_std": float(np.nanstd(recalls)) if recalls else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", default="ncc", choices=[r for r in splits.REGIMES if r != "scaffold"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--min_drugs", type=int, default=20)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--experiment", default="mean_baselines")
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    parser.add_argument("--dump_preds", action="store_true", default=True)
    parser.add_argument("--no_dump_preds", dest="dump_preds", action="store_false")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset (same construction as the benchmark scripts)...")
    drug_ds = DrugEmbeddingDataset(DRUG_CSV, drug_dim=768)
    ic50_ds = IC50Dataset(IC50_CSV, drug_ds, rna_csv_path=RNA_CSV, rna_dim=256, add_tissue_ids=True)
    logger.info(f"{len(ic50_ds)} pairs, {ic50_ds.num_celllines} cell lines")

    frame = pd.DataFrame(
        {
            "cellline_id": ic50_ds.cellline_ids,
            "drug_id": ic50_ds.drug_ids,
            "y": ic50_ds.ic50_values.numpy().astype(np.float64),
        }
    )

    rows, ranking_rows = [], []

    for seed in args.seeds:
        sample_fold = splits.build_folds(args.regime, ic50_ds, n_folds=args.n_folds, seed=seed)

        for kind in BASELINES:
            fold_metrics, dumps = [], []

            for fold in range(args.n_folds):
                train_mask = sample_fold != fold
                test_mask = sample_fold == fold
                test_idx = np.where(test_mask)[0]

                y_pred = fit_predict(kind, frame, train_mask, test_mask)
                y_true = frame.y.to_numpy()[test_idx]

                metrics = compute_regression_metrics(y_true, y_pred)

                # Drug-centred targets: subtract each drug's training-fold mean from
                # both sides. On this scale the drug_mean baseline is exactly 0 by
                # construction, so R^2 measures only what a model adds beyond
                # knowing which drug it is.
                drug_lookup = frame[train_mask].groupby("drug_id").y.mean()
                global_mean = float(frame[train_mask].y.mean())
                offset = frame.drug_id.iloc[test_idx].map(drug_lookup).fillna(global_mean).to_numpy()
                centred = compute_regression_metrics(y_true - offset, y_pred - offset)
                metrics["r2_drug_centered"] = centred["r2"]

                metrics.update({"arm": kind, "seed": seed, "fold": fold, "regime": args.regime})
                fold_metrics.append(metrics)

                dump = pd.DataFrame(
                    {
                        "cellline_id": frame.cellline_id.iloc[test_idx].to_numpy(),
                        "y_true": y_true,
                        "y_pred": y_pred,
                    }
                )
                dumps.append(dump)

                if args.dump_preds:
                    dump_predictions(
                        out_root,
                        experiment=args.experiment,
                        regime=args.regime,
                        arm=kind,
                        seed=seed,
                        fold=fold,
                        split="test",
                        dataset=ic50_ds,
                        row_indices=test_idx.tolist(),
                        y_true=y_true,
                        y_pred=y_pred,
                    )

            rows.extend(fold_metrics)

            pooled = pd.concat(dumps, ignore_index=True)
            rank = ranking_metrics(pooled, min_drugs=args.min_drugs, k=args.topk)
            rank.update({"arm": kind, "seed": seed, "regime": args.regime})
            ranking_rows.append(rank)

            r2s = [m["r2"] for m in fold_metrics]
            logger.info(
                f"[seed {seed}] {kind:15s} R2={np.mean(r2s):.4f}+-{np.std(r2s):.4f}  "
                f"drug-centered R2={np.mean([m['r2_drug_centered'] for m in fold_metrics]):+.4f}  "
                f"per-CL rho={rank['per_cellline_spearman_mean']:.4f}  "
                f"top{args.topk}={rank[f'top{args.topk}_recall_mean']:.4f}"
            )

    metrics_df = pd.DataFrame(rows)
    ranking_df = pd.DataFrame(ranking_rows)

    metrics_path = tables_dir / f"mean_baselines_{args.regime}.csv"
    ranking_path = tables_dir / f"mean_baselines_ranking_{args.regime}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    summary = (
        metrics_df.groupby("arm")[["r2", "rmse", "mae", "male", "evapa", "r2_drug_centered"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_path = tables_dir / f"mean_baselines_summary_{args.regime}.json"
    with open(summary_path, "w") as handle:
        json.dump(json.loads(summary.to_json()), handle, indent=2)

    logger.info(f"\n{summary}\n")
    logger.info(f"Wrote {metrics_path}")
    logger.info(f"Wrote {ranking_path}")
    logger.info(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
