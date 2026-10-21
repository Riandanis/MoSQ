#!/usr/bin/env python3
"""Frozen-representation transfer, with target-free and leak-free controls.

A transfer result can be circular: if the representations come from an encoder that
already observed the evaluated cell lines' IC50 labels, a downstream model trained on
them is expected to do well. The arms below separate that exposure from transfer.

Feature arms, in increasing order of label exposure:

  raw            the 1080-d input vector; no learned encoder at all
  randinit       Q-Former queries from an untrained encoder; measures what the
                 architecture contributes before any training
  clrna          Phase 1 encoder; pretrained on images and RNA, never on IC50
  recon_only     DR-A pretraining with the contrastive term switched off, so the
                 encoder is trained on this data but never sees an IC50 label
  dra            the full DR-A encoder

``recon_only`` is the tightest control: same architecture, same data, same
optimiser, same number of steps, and no IC50 supervision.

Standardisation: every downstream predictor receives per-fold z-scored features
fitted on training rows only, shares the folds, and shares its training procedure;
this script emits a standardisation table documenting it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer import splits  # noqa: E402
from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset  # noqa: E402
from gastro_transformer.metrics import compute_regression_metrics  # noqa: E402
from gastro_transformer.predlog import append_manifest, dump_predictions  # noqa: E402
from gastro_transformer.seeding import set_all_seeds  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_by_path(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_abl = _load_by_path("_abl_tr", "scripts/benchmark_5fold_ncc_ablation.py")
_lit = _load_by_path("_lit_tr", "scripts/benchmark_literature_models.py")
_run_cv = _load_by_path("_run_cv_tr", "scripts/run_cv.py")

FEATURE_ARMS = ["raw", "randinit", "clrna", "recon_only", "dra", "dra_leakfree"]
DOWNSTREAM = ["xgboost", "standalone_mlp", "tcnn_style", "deepcdr_style", "drugcell_style"]
# Predictors that take a flat feature vector with no assumed drug/cell partition.
UNSTRUCTURED = {"xgboost", "standalone_mlp"}
# Arms whose encoder is rebuilt per fold rather than shared across folds. The `dra`
# arm loads one globally-pretrained checkpoint for every fold, so a held-out cell
# line's frozen features encode DR-A supervision on that cell line -- which is the
# label-exposure channel, which the reconstruction-only control does not close.
# `dra_leakfree` loads each fold's own inductive checkpoint instead
# (pre-registered Experiment B).
PER_FOLD_ARMS = {"dra_leakfree"}


@torch.no_grad()
def extract_queries(model, dataset, indices, device, batch_size=512):
    """Mean-pooled Q-Former query vectors, 768-d.

    Rebuilds the token stack the way ``model.forward`` does for MultiToken mode,
    then pools the queries. The IC50 head is never invoked, so no part of the
    regression head leaks into the features.
    """
    from torch.utils.data import DataLoader, Subset

    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)
    out, targets = [], []

    for batch in loader:
        drug = batch["drug_embed"].to(device)
        cancer = batch["cancer_type_id"].to(device)
        tissue = batch["tissue_id"].to(device)
        rna = batch["rna_embed"].to(device)
        available = batch["rna_available"].to(device)

        drug_tok = model.projectors["drug"](drug).unsqueeze(1)
        drug_tok = drug_tok + model.modality_type_embeddings["drug"]

        cl = model.cellline_encoder(
            cancer_type_ids=cancer, tissue_ids=tissue,
            rna_embeds=rna, rna_available=available,
        )
        cl[:, 0:1] = cl[:, 0:1] + model.modality_type_embeddings["cancer"]
        cl[:, 1:2] = cl[:, 1:2] + model.modality_type_embeddings["tissue"]
        cl[:, 2:3] = cl[:, 2:3] + model.modality_type_embeddings["cellline_rna"]

        combined = torch.cat([drug_tok, cl], dim=1)
        queries = model.qformer(combined, modality_mask=None)
        out.append(queries.cpu().numpy())
        targets.append(batch["ic50"].numpy())

    return np.vstack(out).astype(np.float32), np.concatenate(targets).astype(np.float32)


def fold_checkpoint(args, fold: int) -> Path:
    """Path to the per-fold inductive DR-A checkpoint for the `dra_leakfree` arm."""
    return (Path(args.dra_ckpt_per_fold) / f"seed{args.seed}_fold{fold}"
            / "pretrained_dra.pt")


def build_encoder(arm: str, args, dataset, device, checkpoint_override=None):
    from gastro_transformer.model import ModalitySlotQFormer

    config = _abl.make_qformer_config(
        device, use_multitoken_cellline=True, use_qformer=True,
        use_qformer_for_ic50=True, use_ic50_attn_pool=True,
        num_cell_lines=max(dataset.num_celllines, 1000),
    )
    model = ModalitySlotQFormer(config).to(device)
    checkpoint = checkpoint_override or {
        "clrna": args.clrna_ckpt, "dra": args.dra_ckpt,
        "recon_only": args.recon_ckpt,
    }.get(arm)
    if checkpoint:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"arm {arm!r} needs {checkpoint}, which does not exist")
        model = _abl.load_pretrained_weights(model, checkpoint, device)
    return model


def fit_downstream(name, X_train, y_train, X_test, y_test, seed, device, epochs):
    """Train one downstream predictor on frozen features. Returns predictions."""
    if name == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbosity=0,
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)

    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    cls = {"standalone_mlp": _lit.StandaloneMLP, "tcnn_style": _lit.tCNNStyle,
           "deepcdr_style": _lit.DeepCDRStyle, "drugcell_style": _lit.DrugCellStyle}[name]
    # Frozen Q-Former queries have no drug/cell split, so the branch models get an
    # even halving of the feature vector rather than the raw 768/312 layout.
    model = _run_cv.build_feature_arm(
        name, cls, X_train.shape[1], drug_dim=_abl.DRUG_DIM
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=256, shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        preds = []
        for start in range(0, len(X_test), 4096):
            chunk = torch.from_numpy(X_test[start:start + 4096]).to(device)
            preds.append(model(chunk).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=FEATURE_ARMS, choices=FEATURE_ARMS)
    parser.add_argument("--downstream", nargs="+", default=["xgboost"], choices=DOWNSTREAM)
    parser.add_argument("--regime", default="ncc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment", default="transfer")
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    parser.add_argument("--dra_ckpt", default=str(ROOT / "saved_checkpoints/pretrained_dra.pt"))
    parser.add_argument("--clrna_ckpt", default=str(ROOT / "saved_checkpoints/pretrained_clrna.pt"))
    parser.add_argument("--recon_ckpt",
                        default=str(ROOT / "saved_checkpoints/recon_only/pretrained_dra.pt"))
    parser.add_argument("--dra_ckpt_per_fold",
                        default=str(ROOT / "reports/final_experiments/leakfree_strict_ncc_ckpt"
                                           "/checkpoints"),
                        help="Directory of per-fold inductive DR-A checkpoints "
                             "(seed<S>_fold<F>/pretrained_dra.pt) for the dra_leakfree arm. "
                             "Each fold's encoder never saw that fold's test cell lines.")
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out_root)

    drug_ds = DrugEmbeddingDataset(_abl.DRUG_CSV, drug_dim=768)
    dataset = IC50Dataset(
        _abl.IC50_CSV, drug_ds, rna_csv_path=_abl.RNA_CSV, rna_dim=256, add_tissue_ids=True
    )
    sample_fold = splits.build_folds(args.regime, dataset, n_folds=args.n_folds, seed=args.seed)

    rows, standardisation = [], []
    started = time.time()

    for arm in args.arms:
        encoder = None
        if arm != "raw" and arm not in PER_FOLD_ARMS:
            try:
                encoder = build_encoder(arm, args, dataset, device)
            except FileNotFoundError as exc:
                logger.warning(f"skipping arm {arm!r}: {exc}")
                continue

        for fold in range(args.n_folds):
            set_all_seeds(args.seed * 1000 + fold)
            train_idx, _, test_idx = splits.get_fold_splits(sample_fold, fold, seed=args.seed)

            if arm in PER_FOLD_ARMS:
                ckpt = fold_checkpoint(args, fold)
                if not ckpt.exists():
                    logger.warning(
                        f"skipping arm {arm!r}: fold {fold} checkpoint missing at {ckpt}. "
                        f"Run the inductive pretraining with --keep_checkpoints first."
                    )
                    break
                encoder = build_encoder(arm, args, dataset, device,
                                        checkpoint_override=str(ckpt))
                logger.info(f"  arm {arm} fold {fold}: encoder from {ckpt}")

            if arm == "raw":
                X_train, y_train = _abl.extract_fair_features(dataset, train_idx)
                X_test, y_test = _abl.extract_fair_features(dataset, test_idx)
            else:
                X_train, y_train = extract_queries(encoder, dataset, train_idx, device)
                X_test, y_test = extract_queries(encoder, dataset, test_idx, device)

            # Standardise on training rows only, per fold. Same for every arm and
            # every downstream predictor, and it is the only preprocessing applied.
            mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
            X_train_s = (X_train - mean) / std
            X_test_s = (X_test - mean) / std

            # tCNN/DeepCDR/DrugCell-style predictors slice their input into a drug
            # branch and a cell branch. The raw 1080-d vector has that structure
            # (768 drug + 312 cell); a fused Q-Former representation does not, and
            # any split point would be arbitrary. So the structured predictors run
            # only on the raw arm, and the frozen-feature comparison uses predictors
            # that consume an unstructured vector, rather than a fabricated partition.
            usable = args.downstream if arm == "raw" else [
                d for d in args.downstream if d in UNSTRUCTURED
            ]
            for name in usable:
                y_pred = fit_downstream(
                    name, X_train_s, y_train, X_test_s, y_test, args.seed, device, args.epochs
                )
                metrics = compute_regression_metrics(y_test, y_pred)
                metrics.update({"features": arm, "downstream": name,
                                "fold": fold, "seed": args.seed})
                rows.append(metrics)

                dump_predictions(
                    out_root,
                    experiment=args.experiment, regime=args.regime,
                    arm=f"{arm}__{name}", seed=args.seed, fold=fold, split="test",
                    dataset=dataset, row_indices=test_idx, y_true=y_test, y_pred=y_pred,
                )
                logger.info(
                    f"  {arm:11s} -> {name:15s} fold {fold}: R2={metrics['r2']:.4f}"
                )

            if fold == 0:
                standardisation.append(
                    {
                        "features": arm,
                        "dimensionality": int(X_train.shape[1]),
                        "preprocessing": "per-fold z-score, training-fold statistics only",
                        "folds": f"{args.n_folds}-fold {args.regime}, identical across arms",
                        "downstream_training": "AdamW lr=1e-3 wd=0.01 batch=256 MSE "
                                               f"{args.epochs} epochs (neural); "
                                               "XGBRegressor 200 trees depth 6 lr 0.1",
                    }
                )

    frame = pd.DataFrame(rows)
    tables = out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / f"transfer_{args.regime}_seed{args.seed}.csv", index=False)
    pd.DataFrame(standardisation).to_csv(tables / "transfer_standardisation.csv", index=False)

    summary = frame.groupby(["features", "downstream"])[
        ["r2", "rmse", "mae", "male", "evapa"]
    ].agg(["mean", "std"]).round(4)
    logger.info("\n" + summary.to_string())

    append_manifest(
        out_root,
        {"experiment": args.experiment, "regime": args.regime, "seed": args.seed,
         "arms": args.arms, "downstream": args.downstream,
         "wall_clock_s": round(time.time() - started, 1)},
    )
    logger.info(f"Wrote {tables / f'transfer_{args.regime}_seed{args.seed}.csv'}")


if __name__ == "__main__":
    main()
