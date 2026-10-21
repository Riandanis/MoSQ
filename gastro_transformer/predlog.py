"""Per-sample prediction dumping.

Most of the evaluation -- significance testing, per-cell-line drug ranking,
RNA-subset comparison, additional error metrics, predicted-vs-true scatter plots,
scaffold-distance analysis -- are post-hoc analyses over predictions. Nothing in
this repo saved predictions, so each of those required a full GPU rerun. Dumping
once removes that coupling entirely.

Row identity comes for free: every evaluation loader in the codebase is
``DataLoader(Subset(ds, test_idx), shuffle=False)``, so prediction order equals
``test_idx`` order exactly. No change to ``IC50Dataset`` or the batch dict is
needed -- the caller just passes the indices it already had.

Layout::

    reports/revision/
      manifest.jsonl
      <experiment>/preds/<regime>__<arm>__seed<S>__fold<F>__<split>.parquet
      tables/
      figures/
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRED_SCHEMA = [
    "experiment",
    "regime",
    "arm",
    "seed",
    "fold",
    "split",
    "row_index",
    "cellline_id",
    "drug_id",
    "has_rna",
    "y_true",
    "y_pred",
]


def _unwrap(dataset) -> Tuple[object, Optional[List[int]]]:
    """Return (base IC50Dataset, valid_indices) for a dataset or filtered wrapper."""
    base = getattr(dataset, "ic50_ds", None) or getattr(dataset, "base", None)
    if base is None:
        return dataset, None
    return base, list(getattr(dataset, "valid_indices", range(len(dataset))))


def resolve_row_ids(
    dataset, row_indices: Sequence[int]
) -> Tuple[List[str], List[str], np.ndarray]:
    """Map dataset-level row indices to (cellline_id, drug_id, has_rna).

    ``row_indices`` are the indices handed to ``Subset``, i.e. they index
    ``dataset``. For a filtered wrapper they are translated through
    ``valid_indices`` onto the underlying ``IC50Dataset`` first.
    """
    base, valid = _unwrap(dataset)
    base_indices = [valid[i] for i in row_indices] if valid is not None else list(row_indices)

    cellline_ids = [base.cellline_ids[i] for i in base_indices]
    drug_ids = [base.drug_ids[i] for i in base_indices]

    cl_idx = base.cellline_indices.numpy()
    has_rna_by_cl = base.cellline_has_rna.numpy()
    has_rna = np.array([bool(has_rna_by_cl[cl_idx[i]]) for i in base_indices], dtype=bool)

    return cellline_ids, drug_ids, has_rna


def dump_predictions(
    out_root: Union[str, Path],
    *,
    experiment: str,
    regime: str,
    arm: str,
    seed: int,
    fold: int,
    split: str,
    dataset,
    row_indices: Sequence[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_seed: Optional[int] = None,
    extra: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """Write one parquet of per-sample predictions and return its path.

    ``fold_seed`` resamples the *partition* rather than the weight init, so two
    runs sharing a seed but differing in ``fold_seed`` are different experiments
    that happen to carry the same fold numbers. It must therefore appear in both
    the filename and the frame: the filename so concurrent partitions do not
    overwrite one another, and the column so ``groupby`` does not merge them.

    Omitting it from the filename silently destroyed 40 dumps in the
    ``encoder_partitions`` run -- three partitions raced on one path and the
    surviving mixture scored R2=0.228 where no contributing run exceeded 0.061,
    because folds drawn from different partitions overlap and leak train into
    test. The JSON path was already ``__fs<N>``-aware, which is what made the
    corruption survive review: the summary files looked right.
    """
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()
    row_indices = list(row_indices)

    if not (len(row_indices) == y_true.size == y_pred.size):
        raise ValueError(
            f"length mismatch: {len(row_indices)} indices, "
            f"{y_true.size} targets, {y_pred.size} predictions. "
            "Predictions must come from a shuffle=False loader over Subset(ds, row_indices)."
        )

    cellline_ids, drug_ids, has_rna = resolve_row_ids(dataset, row_indices)

    df = pd.DataFrame(
        {
            "experiment": experiment,
            "regime": regime,
            "arm": arm,
            "seed": np.int16(seed),
            "fold_seed": np.int16(-1 if fold_seed is None else fold_seed),
            "fold": np.int16(fold),
            "split": split,
            "row_index": np.asarray(row_indices, dtype=np.int32),
            "cellline_id": cellline_ids,
            "drug_id": drug_ids,
            "has_rna": has_rna,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    for key, values in (extra or {}).items():
        df[key] = np.asarray(values).ravel()

    pred_dir = Path(out_root) / experiment / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    fs = "" if fold_seed is None else f"__fs{fold_seed}"
    path = pred_dir / f"{regime}__{arm}__seed{seed}{fs}__fold{fold}__{split}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Wrote {len(df)} predictions to {path}")
    return path


def load_all_predictions(out_root: Union[str, Path], **filters) -> pd.DataFrame:
    """Concatenate every prediction dump under ``out_root``, optionally filtered.

    Filter values may be a scalar or any container, e.g.
    ``load_all_predictions(root, regime='ncc', arm=['xgboost', 'multitoken_dra'])``.
    """
    paths = sorted(Path(out_root).glob("*/preds/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No prediction dumps found under {out_root}")

    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    # Dumps written before fold_seed was threaded through carry no such column.
    # They are all single-partition runs, so -1 ("default partition") is correct
    # for them and keeps the grouping keys uniform across old and new files.
    if "fold_seed" not in df.columns:
        df["fold_seed"] = np.int16(-1)
    else:
        df["fold_seed"] = df["fold_seed"].fillna(-1).astype(np.int16)

    for column, wanted in filters.items():
        if wanted is None:
            continue
        if isinstance(wanted, (str, bytes, int, np.integer, bool, np.bool_)):
            df = df[df[column] == wanted]
        else:
            df = df[df[column].isin(list(wanted))]

    return df.reset_index(drop=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def file_sha256(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """Hash a checkpoint so a run can be traced back to exact weights."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_manifest(out_root: Union[str, Path], record: dict) -> None:
    """Append one provenance line. Written only on clean exit, so its presence
    is what ``--skip_existing`` treats as proof a job finished."""
    import torch

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    enriched = {
        **record,
        "git_sha": _git_sha(),
        "hostname": os.uname().nodename,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    with open(out_root / "manifest.jsonl", "a") as handle:
        handle.write(json.dumps(enriched, default=str) + "\n")


def load_manifest(out_root: Union[str, Path]) -> pd.DataFrame:
    path = Path(out_root) / "manifest.jsonl"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return pd.DataFrame(records)
