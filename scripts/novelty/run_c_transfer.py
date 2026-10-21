#!/usr/bin/env python3
"""Experiment C-transfer: are DR-A representations model-agnostic for panel completion?

Pre-registered (pre-registration record, appendix, 2026-10-01). One (seed, k) per
invocation. Per NCC fold:

1. Each test cell line's drugs are split into a fixed evaluation half and a reveal pool
   (rng 10000*seed + fold); the first k of the pool are revealed (nested across k).
2. DR-A is pretrained on training cell lines + revealed pairs only (same schedule as every
   other DR-A run). A guard checks the corpus size in the pretraining log.
3. Frozen query features feed XGBoost and the standalone MLP; raw 1080-d features are the
   comparator. Arms (prediction dumps under reports/novelty/<experiment>/preds):

   F1_k{k}_{learner}  DR-A(k) features, learner fit on training-line rows; predicts ALL
                      test rows, so the per-line offset can be estimated out of sample on
                      the revealed rows during analysis.          <- primary, with R1
   R1_{learner}       raw features, same; k-independent.
   F2_k{k}_{learner}  DR-A(k) features, learner fit on training + revealed rows; eval half.
   R2_k{k}_{learner}  raw features, same.

Roles (evaluation half, pool rank) are saved per fold so the analysis can rebuild the
revealed sets exactly. Evaluation-half pairs never reach any encoder or learner.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer import splits  # noqa: E402
from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset  # noqa: E402
from gastro_transformer.predlog import append_manifest, dump_predictions  # noqa: E402
from gastro_transformer.seeding import set_all_seeds  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_by_path(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_abl = _load_by_path("_abl_ct", "scripts/benchmark_5fold_ncc_ablation.py")
_tr = _load_by_path("_tr_ct", "scripts/benchmark_transfer.py")

LEARNERS = ["xgboost", "standalone_mlp"]


def fold_roles(dataset, test_idx, seed: int, fold: int) -> pd.DataFrame:
    """Evaluation half + reveal-pool order for every test row. Deterministic per (seed, fold)."""
    test_idx = np.sort(np.asarray(test_idx))
    roles = pd.DataFrame({
        "row_index": test_idx,
        "cellline_id": np.asarray(dataset.cellline_ids, dtype=object)[test_idx],
        "drug_id": np.asarray(dataset.drug_ids, dtype=object)[test_idx],
    })
    u = np.random.default_rng(10_000 * seed + fold).random(len(roles))
    rank = pd.Series(u).groupby(roles.cellline_id.values).rank(method="first").to_numpy()
    n = roles.groupby("cellline_id").cellline_id.transform("size").to_numpy()
    roles["is_eval"] = rank <= np.ceil(n / 2)
    pool_u = np.where(roles.is_eval, np.inf, u)
    roles["pool_rank"] = (
        pd.Series(pool_u).groupby(roles.cellline_id.values).rank(method="first").to_numpy()
    )
    roles.loc[roles.is_eval, "pool_rank"] = np.nan
    return roles


def pretrain(args, fold_dir: Path, test_lines, revealed: pd.DataFrame, expected_rows: int,
             fold: int) -> Path:
    ckpt = fold_dir / "pretrained_dra.pt"
    if ckpt.exists():
        logger.info(f"  reusing {ckpt}")
        return ckpt
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "held_out.txt").write_text("\n".join(sorted(test_lines)) + "\n")
    revealed[["cellline_id", "drug_id"]].to_csv(fold_dir / "reveal_pairs.csv", index=False)
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts/pretrain_phase3.py"),
        "--checkpoint", args.clrna_ckpt, "--output_dir", str(fold_dir),
        "--device", args.device, "--epochs", str(args.pretrain_epochs),
        "--batch_size", "256", "--bin_mode", "global",
        "--seed", str(args.seed * 1000 + fold),
        "--held_out_celllines", str(fold_dir / "held_out.txt"),
        "--reveal_pairs", str(fold_dir / "reveal_pairs.csv"),
    ]
    started = time.time()
    with open(fold_dir / "pretrain.log", "w") as handle:
        rc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT).returncode
    log = (fold_dir / "pretrain.log").read_text(errors="replace")
    if rc != 0:
        raise RuntimeError(f"fold {fold} pretraining failed; see {fold_dir / 'pretrain.log'}")
    match = re.search(r"Total: (\d+) samples", log)
    if not match or int(match.group(1)) != expected_rows:
        raise RuntimeError(
            f"fold {fold}: pretraining corpus {match.group(1) if match else '?'} rows != "
            f"expected {expected_rows} (training lines + revealed). Leak guard tripped."
        )
    logger.info(f"  pretraining done in {(time.time() - started) / 60:.1f} min, "
                f"corpus {expected_rows} rows verified")
    return ckpt


def standardise(X_fit, *others):
    mean, std = X_fit.mean(axis=0), X_fit.std(axis=0) + 1e-8
    return [(X_fit - mean) / std] + [(X - mean) / std for X in others]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--pretrain_epochs", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=10, help="downstream MLP epochs")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--clrna_ckpt", default=str(ROOT / "saved_checkpoints/pretrained_clrna.pt"))
    parser.add_argument("--experiment", default="novelty_c_transfer")
    parser.add_argument("--out_root", default=str(ROOT / "reports/novelty"))
    parser.add_argument("--keep_checkpoints", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    exp_dir = out_root / args.experiment
    result_path = exp_dir / f"c_transfer__seed{args.seed}__k{args.k}.json"
    if result_path.exists():
        logger.info(f"{result_path.name} exists, skipping")
        return
    set_all_seeds(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    started = time.time()

    drug_ds = DrugEmbeddingDataset(_abl.DRUG_CSV, drug_dim=768)
    dataset = IC50Dataset(_abl.IC50_CSV, drug_ds, rna_csv_path=_abl.RNA_CSV, rna_dim=256,
                          add_tissue_ids=True)
    sample_fold = splits.build_folds("ncc", dataset, n_folds=args.n_folds, seed=args.seed)
    all_lines = np.asarray(dataset.cellline_ids, dtype=object)

    def dump(arm, fold, idx, y_true, y_pred):
        dump_predictions(out_root, experiment=args.experiment, regime="ncc", arm=arm,
                         seed=args.seed, fold=fold, split="test", dataset=dataset,
                         row_indices=idx, y_true=y_true, y_pred=y_pred)

    def done(arm, fold):
        return (exp_dir / "preds" / f"ncc__{arm}__seed{args.seed}__fold{fold}__test.parquet").exists()

    folds = args.folds if args.folds is not None else list(range(args.n_folds))
    for fold in folds:
        arms_needed = [f"{p}_k{args.k}_{l}" for p in ("F1", "F2", "R2") for l in LEARNERS] + \
                      [f"R1_{l}" for l in LEARNERS]
        if all(done(a, fold) for a in arms_needed):
            logger.info(f"fold {fold}: all dumps present, skipping")
            continue
        set_all_seeds(args.seed * 1000 + fold)
        train_idx, _, test_idx = splits.get_fold_splits(sample_fold, fold, seed=args.seed)
        _, test_lines = splits.fold_celllines(sample_fold, dataset.cellline_ids, fold)

        roles = fold_roles(dataset, test_idx, args.seed, fold)
        (exp_dir / "roles").mkdir(parents=True, exist_ok=True)
        roles.to_parquet(exp_dir / "roles" / f"seed{args.seed}_fold{fold}.parquet", index=False)
        revealed = roles[roles.pool_rank <= args.k]
        eval_rows = roles[roles.is_eval]
        assert not set(revealed.row_index) & set(eval_rows.row_index)
        expected = int((~np.isin(all_lines, list(test_lines))).sum()) + len(revealed)
        logger.info(f"fold {fold}: {len(test_lines)} test lines, {len(revealed)} revealed pairs "
                    f"(k={args.k}), {len(eval_rows)} evaluation pairs")

        fold_dir = exp_dir / "checkpoints" / f"seed{args.seed}_k{args.k}_fold{fold}"
        ckpt = pretrain(args, fold_dir, test_lines, revealed, expected, fold)
        encoder = _tr.build_encoder("dra_leakfree", args, dataset, device,
                                    checkpoint_override=str(ckpt))

        test_all = roles.row_index.to_numpy()
        rev_idx = revealed.row_index.to_numpy()
        ev_idx = eval_rows.row_index.to_numpy()
        pos = {r: i for i, r in enumerate(test_all)}
        rev_pos = np.array([pos[r] for r in rev_idx], dtype=int)
        ev_pos = np.array([pos[r] for r in ev_idx], dtype=int)

        feats = {
            "F": (_tr.extract_queries(encoder, dataset, train_idx, device),
                  _tr.extract_queries(encoder, dataset, test_all, device)),
            "R": (_abl.extract_fair_features(dataset, train_idx),
                  _abl.extract_fair_features(dataset, test_all)),
        }
        del encoder
        torch.cuda.empty_cache()

        for kind, ((X_tr, y_tr), (X_te, y_te)) in feats.items():
            for learner in LEARNERS:
                # Train-only fit, predictions for every test row (offset computed later
                # out of sample on revealed rows). R1 is k-independent; skip if present.
                arm1 = f"F1_k{args.k}_{learner}" if kind == "F" else f"R1_{learner}"
                if not done(arm1, fold):
                    set_all_seeds(args.seed * 1000 + fold)
                    A, B = standardise(X_tr, X_te)
                    pred = _tr.fit_downstream(learner, A, y_tr, B, y_te, args.seed, device, args.epochs)
                    dump(arm1, fold, test_all, y_te, pred)
                    logger.info(f"  {arm1}: eval-half R2 (no offset) "
                                f"{1 - ((y_te[ev_pos] - pred[ev_pos])**2).sum() / ((y_te[ev_pos] - y_te[ev_pos].mean())**2).sum():.4f}")
                # Fit on training + revealed rows, predict evaluation half.
                arm2 = f"{'F2' if kind == 'F' else 'R2'}_k{args.k}_{learner}"
                if not done(arm2, fold):
                    set_all_seeds(args.seed * 1000 + fold)
                    X_fit = np.vstack([X_tr, X_te[rev_pos]])
                    y_fit = np.concatenate([y_tr, y_te[rev_pos]])
                    A, B = standardise(X_fit, X_te[ev_pos])
                    pred = _tr.fit_downstream(learner, A, y_fit, B, y_te[ev_pos], args.seed,
                                              device, args.epochs)
                    dump(arm2, fold, ev_idx, y_te[ev_pos], pred)
                    logger.info(f"  {arm2}: eval-half R2 "
                                f"{1 - ((y_te[ev_pos] - pred)**2).sum() / ((y_te[ev_pos] - y_te[ev_pos].mean())**2).sum():.4f}")

        if not args.keep_checkpoints:
            ckpt.unlink(missing_ok=True)

    elapsed = time.time() - started
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"seed": args.seed, "k": args.k, "folds": folds,
                                       "wall_clock_s": round(elapsed, 1), "config": vars(args)},
                                      indent=2))
    append_manifest(out_root, {"experiment": args.experiment, "seed": args.seed, "k": args.k,
                               "wall_clock_s": round(elapsed, 1), "args": vars(args)})
    logger.info(f"=== C-transfer seed{args.seed} k{args.k} done in {elapsed / 60:.1f} min ===")


if __name__ == "__main__":
    main()
