"""Numpy metrics superset used by the post-hoc revision analyses.

``losses.compute_ic50_metrics`` stays exactly as it is: it is the torch-tensor
function every benchmark script calls and its outputs are baked into every
committed report JSON. This module adds further metrics (RMSE, MAE, MALE, EVAPA) on top
of the same six definitions, for use on the prediction dumps.

Target convention: ``ic50_value`` in this project is ``LN_IC50`` (natural log of
micromolar IC50), spanning roughly -10 to +13.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def male(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute log10 error on the micromolar scale.

    Targets are already natural-log IC50, so a "log error" computed directly on
    them is numerically identical to MAE. To report something distinct we
    exponentiate back to micromolar and take a base-10 log-error with a +1
    offset for numerical safety near zero concentration.

    Caveat worth stating in the manuscript: because LN_IC50 reaches +13
    (~4.4e5 microM), this statistic is dominated by the potent tail.
    """
    true_um = np.exp(np.asarray(y_true, dtype=np.float64))
    pred_um = np.exp(np.asarray(y_pred, dtype=np.float64))
    return float(np.mean(np.abs(np.log10(1.0 + true_um) - np.log10(1.0 + pred_um))))


def evapa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Explained variance, ``1 - Var(residual) / Var(y_true)``.

    Reported alongside R^2. The two differ by exactly the squared prediction
    bias, which is informative on cell-line-disjoint folds where predictions can
    carry a systematic offset. EVAPA is not standardised in the DRP literature,
    so the formula must be given in the caption.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.var(y_true)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / denom)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Superset of ``losses.compute_ic50_metrics`` with MALE, EVAPA and n.

    The first six keys use identical definitions to the torch version so numbers
    stay comparable with the committed reports.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = y_true.size

    resid = y_true - y_pred
    mse = float(np.mean(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    out = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1.0 - np.sum(resid**2) / (ss_tot + 1e-8)),
        "male": male(y_true, y_pred),
        "evapa": evapa(y_true, y_pred),
        "n": int(n),
    }

    # Correlations are undefined for degenerate inputs; report NaN rather than raising.
    if n > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["pearson_r"] = float(stats.pearsonr(y_true, y_pred)[0])
        out["spearman_r"] = float(stats.spearmanr(y_true, y_pred)[0])
    else:
        out["pearson_r"] = float("nan")
        out["spearman_r"] = float("nan")

    return out


def per_group_metrics(
    df: pd.DataFrame,
    group_col: str,
    min_n: int = 20,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Metrics computed within each level of ``group_col``.

    Groups with fewer than ``min_n`` rows are dropped: a Spearman correlation
    over a handful of drugs is too noisy to average. Cell lines in this dataset
    carry between 1 and 986 drugs (median 177), so the filter matters.
    """
    rows = []
    for key, sub in df.groupby(group_col, sort=True):
        if len(sub) < min_n:
            continue
        metrics = compute_regression_metrics(sub[y_true_col].values, sub[y_pred_col].values)
        metrics[group_col] = key
        rows.append(metrics)
    return pd.DataFrame(rows)


def topk_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    sensitive_is_low: bool = True,
) -> float:
    """Fraction of the true top-k most sensitive items recovered in the predicted top-k.

    Sensitivity means *low* IC50, hence ``sensitive_is_low=True`` by default.
    Returns NaN when fewer than k items are available.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size < k:
        return float("nan")

    sign = 1.0 if sensitive_is_low else -1.0
    true_top = set(np.argsort(sign * y_true)[:k].tolist())
    pred_top = set(np.argsort(sign * y_pred)[:k].tolist())
    return len(true_top & pred_top) / float(k)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for ``fn(y_true, y_pred)``."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    n = y_true.size

    stats_boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats_boot[i] = fn(y_true[idx], y_pred[idx])

    lo, hi = np.nanpercentile(stats_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_difference_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap CI for the mean paired difference ``a - b``.

    Used where Wilcoxon cannot reach significance for want of samples: at n=5 a
    two-sided signed-rank test has a minimum p of 0.0625.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must have equal shape, got {a.shape} and {b.shape}")

    diff = a - b
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(diff, size=diff.size, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_pairs": int(diff.size),
    }
