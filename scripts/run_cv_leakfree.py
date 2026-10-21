#!/usr/bin/env python3
"""Inductive (leak-free) cross-validation with per-fold DR-A pretraining.

The transductive NCC protocol partitions cell lines for fine-tuning but reuses a
DR-A checkpoint that was pretrained on *all* 998 cell lines, so test cell lines
were seen during pretraining. The inductive protocol re-runs DR-A pretraining per
fold on the training cell lines only, making test cell lines unseen at every
stage. The same distinction applies to NCD with drugs.

That is the difference between R^2 = 0.791 and R^2 = 0.700, and quantifying it is
one of the manuscript's claims -- so it needs to be reproducible from this repo,
which it previously was not: the committed ``reports/ncc_leakfree_5fold/`` results
had no generating script here.

Cost: one DR-A pretraining per fold per seed, which dominates the campaign. The
pretraining is cached under ``<out_root>/<experiment>/checkpoints/`` and reused on
resume, so a killed run does not repeat finished folds.

Note on epochs: the sibling implementation defaulted to 30 pretraining epochs
while the headline DR-A checkpoint used 15. That was never like-for-like, and the
default here is 15 for both.

Example::

    python scripts/run_cv_leakfree.py --regime strict_ncc --seed 0 --device cuda:0
    python scripts/run_cv_leakfree.py --regime ncd --seed 0 --bin_mode global
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer import splits  # noqa: E402
from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset  # noqa: E402
from gastro_transformer.predlog import append_manifest, dump_predictions  # noqa: E402
from gastro_transformer.seeding import seed_worker, set_all_seeds, torch_generator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_by_path(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_abl = _load_by_path("_abl_lf", "scripts/benchmark_5fold_ncc_ablation.py")


def size_matched_exclusion(all_ids, test_ids, seed: int) -> set:
    """Pick ``len(test_ids)`` ids to exclude from pretraining, none of them held out.

    The size-matched leaky control (`--pretrain_exclude size_matched`). The inductive
    arm differs from the transductive one in two ways at once -- no label exposure
    *and* a 20% smaller pretraining corpus -- so its deficit cannot be attributed to
    either. This exclusion holds the corpus size fixed while putting the test items
    back in, which isolates label exposure with nothing else moving.

    Pre-registered as Experiment A (size-matched leaky control).
    """
    candidates = sorted(set(all_ids) - set(test_ids))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=len(test_ids), replace=False)
    return {candidates[i] for i in chosen}


def pretrain_fold(args, held_out, fold: int) -> Path:
    """Run DR-A pretraining excluding ``held_out``.

    Shells out to ``pretrain_phase3.py`` rather than importing it, so the leak-free
    checkpoints are produced by exactly the same code path (and CLI defaults) as
    the headline checkpoint. Divergence there would undermine the comparison.

    ``held_out`` is the fold's test items under the default ``--pretrain_exclude
    test``, and a same-sized draw of *non-test* items under ``size_matched``.
    """
    fs = "" if args.fold_seed is None else f"_fs{args.fold_seed}"
    fold_dir = (Path(args.out_root) / args.experiment / "checkpoints"
                / f"seed{args.seed}{fs}_fold{fold}")
    checkpoint = fold_dir / "pretrained_dra.pt"
    if checkpoint.exists() and args.skip_existing:
        logger.info(f"  fold {fold}: reusing cached pretraining at {checkpoint}")
        return checkpoint

    fold_dir.mkdir(parents=True, exist_ok=True)
    held_out_path = fold_dir / "held_out.txt"
    held_out_path.write_text("\n".join(sorted(held_out)) + "\n")

    flag = "--held_out_celllines" if args.regime == "strict_ncc" else "--held_out_drugs"
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts/pretrain_phase3.py"),
        "--checkpoint", args.clrna_ckpt,
        "--output_dir", str(fold_dir),
        "--device", args.device,
        "--epochs", str(args.pretrain_epochs),
        "--batch_size", str(args.batch_size),
        "--bin_mode", args.bin_mode,
        "--seed", str(args.seed * 1000 + fold),
        flag, str(held_out_path),
    ]
    logger.info(f"  fold {fold}: pretraining DR-A on {len(held_out)} held-out items excluded...")
    started = time.time()
    log_path = fold_dir / "pretrain.log"
    with open(log_path, "w") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"fold {fold} pretraining failed; see {log_path}")

    logger.info(f"  fold {fold}: pretraining done in {(time.time() - started) / 60:.1f} min")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", default="strict_ncc", choices=["strict_ncc", "ncd"])
    parser.add_argument("--arm", default="multitoken_dra")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold_seed", type=int, default=None,
                        help="Resample the CV partition independently of --seed; "
                             "see scripts/run_cv.py for the rationale.")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=10, help="fine-tuning epochs")
    parser.add_argument("--pretrain_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bin_mode", default="global")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clrna_ckpt", default=str(ROOT / "saved_checkpoints/pretrained_clrna.pt"))
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--keep_checkpoints", action="store_true",
                        help="Retain per-fold DR-A checkpoints (~300 MB each) after fine-tuning")
    parser.add_argument("--pretrain_exclude", default="test", choices=["test", "size_matched"],
                        help="What to exclude from each fold's pretraining corpus. "
                             "'test' is the inductive protocol. 'size_matched' excludes a "
                             "same-sized draw of NON-test items instead, holding corpus size "
                             "fixed while restoring label exposure -- the control that "
                             "separates the two (pre-registered Experiment A).")
    args = parser.parse_args()

    # The exclusion mode must reach the experiment key, or the size-matched control
    # would overwrite the inductive arm's result JSONs and prediction dumps. A path
    # collision of exactly this kind corrupted 40 dumps on 2026-08-24.
    _suffix = "" if args.pretrain_exclude == "test" else f"_{args.pretrain_exclude}"
    args.experiment = args.experiment or f"leakfree_{args.regime}{_suffix}"
    out_root = Path(args.out_root)
    fs = "" if args.fold_seed is None else f"__fs{args.fold_seed}"
    result_path = (out_root / args.experiment
                   / f"{args.regime}__{args.arm}__seed{args.seed}{fs}.json")
    if args.skip_existing and result_path.exists():
        logger.info(f"{result_path.name} exists, skipping")
        return

    set_all_seeds(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    drug_ds = DrugEmbeddingDataset(_abl.DRUG_CSV, drug_dim=768)
    dataset = IC50Dataset(
        _abl.IC50_CSV, drug_ds, rna_csv_path=_abl.RNA_CSV, rna_dim=256, add_tissue_ids=True
    )
    sample_fold = splits.build_folds(args.regime, dataset, n_folds=args.n_folds,
                                     seed=args.seed, fold_seed=args.fold_seed)

    folds_to_run = args.folds if args.folds is not None else list(range(args.n_folds))
    fold_metrics = []
    started = time.time()

    from torch.utils.data import DataLoader, Subset

    from gastro_transformer.model import ModalitySlotQFormer

    for fold in folds_to_run:
        set_all_seeds(args.seed * 1000 + fold)

        if args.regime == "strict_ncc":
            _, test_items = splits.fold_celllines(sample_fold, dataset.cellline_ids, fold)
            all_items = dataset.cellline_ids
        else:
            _, test_items = splits.fold_drugs(sample_fold, dataset.drug_ids, fold)
            all_items = dataset.drug_ids

        if args.pretrain_exclude == "size_matched":
            excluded = size_matched_exclusion(all_items, test_items, args.seed * 1000 + fold)
            assert not (excluded & set(test_items)), (
                f"fold {fold}: size-matched exclusion overlaps the test set; the control "
                f"is only valid if every test item is RETAINED in the pretraining corpus"
            )
            logger.info(
                f"  fold {fold}: size-matched control -- excluding {len(excluded)} non-test "
                f"items, retaining all {len(test_items)} test items in the corpus"
            )
        else:
            excluded = test_items

        held_out = excluded
        checkpoint = pretrain_fold(args, held_out, fold)

        train_idx, val_idx, test_idx = splits.get_fold_splits(sample_fold, fold, seed=args.seed)
        logger.info(
            f"  fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}, "
            f"{len(held_out)} held-out {'cell lines' if args.regime == 'strict_ncc' else 'drugs'}"
        )

        common = dict(num_workers=args.num_workers, pin_memory=True)
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
            worker_init_fn=seed_worker, generator=torch_generator(args.seed), **common,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, **common
        )
        test_loader = DataLoader(
            Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, **common
        )

        config = _abl.make_qformer_config(
            device,
            use_multitoken_cellline=True, use_qformer=True,
            use_qformer_for_ic50=True, use_ic50_attn_pool=True,
            num_cell_lines=max(dataset.num_celllines, 1000),
        )
        model = ModalitySlotQFormer(config).to(device)
        model = _abl.load_pretrained_weights(model, str(checkpoint), device)
        model = _abl.finetune_foundation(
            model, train_loader, val_loader, config, device, epochs=args.epochs, use_tricks=False
        )
        metrics, preds, targets = _abl.evaluate_foundation(
            model, test_loader, device, return_preds=True
        )
        fold_metrics.append(metrics)

        dump_predictions(
            out_root,
            experiment=args.experiment, regime=args.regime, arm=args.arm,
            seed=args.seed, fold=fold, split="test",
            dataset=dataset, row_indices=test_idx, y_true=targets, y_pred=preds,
        )
        logger.info(f"  fold {fold}: R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f}")

        if not args.keep_checkpoints:
            checkpoint.unlink(missing_ok=True)

    average = _abl.aggregate_metrics(fold_metrics)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as handle:
        json.dump(
            _abl.convert_numpy(
                {
                    "arm": args.arm, "regime": args.regime, "seed": args.seed,
                    "fold_metrics": fold_metrics, "average": average, "config": vars(args),
                }
            ),
            handle, indent=2,
        )

    elapsed = time.time() - started
    append_manifest(
        out_root,
        {
            "experiment": args.experiment, "regime": args.regime, "arm": args.arm,
            "seed": args.seed, "n_folds": args.n_folds, "bin_mode": args.bin_mode,
            "pretrain_epochs": args.pretrain_epochs,
            "pretrain_exclude": args.pretrain_exclude,
            "r2_mean": average["r2"], "r2_std": average["r2_std"],
            "wall_clock_s": round(elapsed, 1), "args": vars(args),
        },
    )
    logger.info(
        f"=== {args.regime} {args.arm} seed{args.seed}: "
        f"R2={average['r2']:.4f}+-{average['r2_std']:.4f} in {elapsed / 60:.1f} min ==="
    )


if __name__ == "__main__":
    main()
