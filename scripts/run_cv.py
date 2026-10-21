#!/usr/bin/env python3
"""Seeded cross-validation runner for the revision campaign.

Replaces the practice of retrofitting a ``--seed`` flag into four divergent
benchmark scripts. One entry point covers every (regime, arm, seed) cell of the
matrix and always writes per-sample predictions, so the downstream analyses
(significance testing, ranking, subset breakdowns, scatter plots) never require a
GPU rerun.

The seed varies the split *and* the weight initialisation; varying only the
initialisation would understate the variance across independent runs. Because all arms within a seed share the same folds, comparisons stay
fold-paired, which is what the signed-rank test needs.

Example::

    python scripts/run_cv.py --regime ncc --arm multitoken_dra --seed 0
    python scripts/run_cv.py --regime ncd --arm standalone_mlp --seed 3 --n_folds 5
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
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer import splits  # noqa: E402
from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset  # noqa: E402
from gastro_transformer.predlog import append_manifest, dump_predictions  # noqa: E402
from gastro_transformer.seeding import seed_worker, set_all_seeds, torch_generator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_by_path(name: str, relative: str):
    """Load a script module by file path, matching the existing repo convention."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_abl = _load_by_path("_abl", "scripts/benchmark_5fold_ncc_ablation.py")
_lit = _load_by_path("_lit", "scripts/benchmark_literature_models.py")

DRUG_CSV = _abl.DRUG_CSV
IC50_CSV = _abl.IC50_CSV
RNA_CSV = _abl.RNA_CSV
FAIR_INPUT_DIM = _abl.FAIR_INPUT_DIM

# Arms that consume the shared 1080-d feature vector rather than the Q-Former.
FEATURE_ARMS = {
    "standalone_mlp": _lit.StandaloneMLP,
    "tcnn_style": _lit.tCNNStyle,
    "deepcdr_style": _lit.DeepCDRStyle,
    "drugcell_style": _lit.DrugCellStyle,
}
# Q-Former arms, mapped to the checkpoint they initialise from (None = random init).
QFORMER_ARMS = {
    "multitoken_dra": "dra",
    "qformer_clrna": "clrna",
    "qformer_randinit": None,
    # Novelty step 3: random init + MSE + 0.5 * SupCon on global-quintile bins whose
    # edges come from the fold's training rows. Leak-free by construction.
    "qformer_randinit_auxsupcon": None,
}
AUX_SUPCON_WEIGHT = 0.5  # fixed before the run (pre-registered); no sweep
ARMS = tuple(FEATURE_ARMS) + tuple(QFORMER_ARMS) + ("xgboost",)


def build_dataset(which: str, drug_csv: str = DRUG_CSV, drug_dim: int = 768,
                  rna_csv: str = RNA_CSV, rna_dim: int = 256):
    drug_ds = DrugEmbeddingDataset(drug_csv, drug_dim=drug_dim)
    ic50_ds = IC50Dataset(IC50_CSV, drug_ds, rna_csv_path=rna_csv, rna_dim=rna_dim,
                          add_tissue_ids=True)
    if which == "rna_filtered":
        return _lit.RNAFilteredDataset(ic50_ds)
    return ic50_ds


class _SubsetView:
    """Row subset that still exposes the attributes folds and dumps need."""

    def __init__(self, base, rows):
        self.base = base
        self.rows = list(rows)
        self.valid_indices = self.rows
        self.cellline_ids = [base.cellline_ids[i] for i in self.rows]
        self.drug_ids = [base.drug_ids[i] for i in self.rows]
        self.cellline_to_idx = {c: i for i, c in enumerate(sorted(set(self.cellline_ids)))}
        self.num_celllines = base.num_celllines

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.base[self.rows[i]]


def build_feature_arm(arm: str, cls, feature_dim: int, drug_dim: int):
    """Construct a feature-vector comparator.

    StandaloneMLP consumes the concatenated vector; tCNNStyle / DeepCDRStyle /
    DrugCellStyle split it internally into a drug branch and a cell branch, so they
    are parameterised by the two widths instead. Getting this wrong fails at
    construction with a TypeError, which has already cost two silent stage failures
    -- hence one shared builder rather than the same conditional in every script.
    """
    if arm == "standalone_mlp":
        return cls(input_dim=feature_dim)
    return cls(drug_dim=drug_dim, cell_dim=feature_dim - drug_dim)


def infer_drug_dim(drug_csv: str) -> int:
    """Read the embedding width from the CSV header, so swapping encoders is a
    data change rather than a code change."""
    header = pd.read_csv(drug_csv, nrows=0).columns
    return sum(1 for c in header if c.startswith("emb_"))


def make_loaders(dataset, train_idx, val_idx, test_idx, batch_size, seed, num_workers=8):
    """Training loader is shuffled and seeded; eval loaders must stay unshuffled
    so predictions line up with the row indices they were built from."""
    common = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=torch_generator(seed),
        **common,
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def resolve_checkpoint(kind: str, args) -> str:
    if kind == "dra":
        return args.dra_ckpt
    if kind == "clrna":
        return args.clrna_ckpt
    raise ValueError(kind)


def run_fold(args, dataset, sample_fold, fold, device):
    """Train and evaluate one arm on one fold. Returns (metrics, preds, targets, test_idx)."""
    train_idx, val_idx, test_idx = splits.get_fold_splits(sample_fold, fold, seed=args.seed)
    logger.info(
        f"  fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )

    if args.arm == "xgboost":
        import xgboost as xgb

        X_train, y_train = _abl.extract_fair_features(dataset, train_idx)
        X_test, y_test = _abl.extract_fair_features(dataset, test_idx)
        mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit((X_train - mean) / std, y_train)
        y_pred = model.predict((X_test - mean) / std)
        return _abl.compute_metrics_np(y_test, y_pred), y_pred, y_test, test_idx

    train_loader, val_loader, test_loader = make_loaders(
        dataset, train_idx, val_idx, test_idx, args.batch_size, args.seed, args.num_workers
    )

    if args.arm in FEATURE_ARMS:
        # StandaloneMLP consumes the concatenated vector; the three comparators
        # split it internally into a drug branch and a cell-line branch, so they
        # are parameterised by the two widths instead.
        model = build_feature_arm(
            args.arm, FEATURE_ARMS[args.arm], args.feature_dim, args.drug_dim
        ).to(device)
        model = _abl.train_standalone_mlp(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.mlp_lr, model_name=args.arm,
        )
        metrics, preds, targets = _abl.evaluate_mlp(model, test_loader, device, return_preds=True)
        return metrics, preds, targets, test_idx

    ckpt_kind = QFORMER_ARMS[args.arm]
    config = _abl.make_qformer_config(
        device,
        drug_dim=args.drug_dim,
        rna_dim=args.rna_dim,
        num_query_tokens=args.num_query_tokens,
        qformer_layers=args.qformer_layers,
        use_multitoken_cellline=True,
        use_qformer=True,
        use_qformer_for_ic50=True,
        use_ic50_attn_pool=True,
        num_cell_lines=max(dataset.num_celllines, 1000),
    )
    from gastro_transformer.model import ModalitySlotQFormer

    model = ModalitySlotQFormer(config).to(device)
    if ckpt_kind is not None:
        model = _abl.load_pretrained_weights(model, resolve_checkpoint(ckpt_kind, args), device)

    aux = {}
    if args.arm == "qformer_randinit_auxsupcon":
        _p3 = _load_by_path("_p3", "scripts/pretrain_phase3.py")
        y_train = dataset.ic50_values[train_idx].numpy().astype(np.float64)
        edges = np.percentile(y_train, [20, 40, 60, 80])
        logger.info(f"  aux SupCon: weight={AUX_SUPCON_WEIGHT} train-row bin edges={edges.round(4)}")
        aux = dict(
            aux_loss_fn=_p3.IC50SupervisedContrastiveLoss(temperature=0.1),
            aux_weight=AUX_SUPCON_WEIGHT,
            aux_bin_edges=torch.tensor(edges, dtype=torch.float32, device=device),
        )

    model = _abl.finetune_foundation(
        model, train_loader, val_loader, config, device, epochs=args.epochs, use_tricks=False,
        **aux,
    )
    metrics, preds, targets = _abl.evaluate_foundation(model, test_loader, device, return_preds=True)
    return metrics, preds, targets, test_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", required=True, choices=list(splits.REGIMES))
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold_seed", type=int, default=None,
                        help="Resample the CV partition independently of --seed. "
                             "Needed for regime=ncd, whose partition is otherwise "
                             "fixed no matter what --seed is. Omit to keep the "
                             "historical split. Results are written to a separate "
                             "'__fs<N>' file so partitions never overwrite one another.")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Run only these fold indices (default: all). The partition is still "
             "built over --n_folds, so results stay comparable to a full run.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="full", choices=["full", "rna_filtered"])
    parser.add_argument("--num_query_tokens", type=int, default=32)
    parser.add_argument("--qformer_layers", type=int, default=6)
    parser.add_argument("--rna_csv", default=RNA_CSV,
                        help="Alternative cell-line representation for the matched-input "
                             "supplementary analysis (e.g. data/matched/rna_expression697.csv). "
                             "Width inferred from the emb_* columns.")
    parser.add_argument("--cellline_subset", default=None,
                        choices=[None, "comparator_common"],
                        help="Restrict to the cell lines the faithful comparators can "
                             "also consume, so MoSQ is evaluated on identical rows.")
    parser.add_argument("--drug_csv", default=DRUG_CSV,
                        help="Alternative drug embedding table for the encoder sweep. "
                             "Width is inferred from the emb_* columns.")
    parser.add_argument("--dra_ckpt", default=str(ROOT / "saved_checkpoints/pretrained_dra.pt"))
    parser.add_argument("--clrna_ckpt", default=str(ROOT / "saved_checkpoints/pretrained_clrna.pt"))
    parser.add_argument("--experiment", default="multiseed")
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    # Keep partitions in separate result files. Without this a resampled run
    # would collide with the deterministic one and --skip_existing would treat a
    # different partition as already computed.
    fs = "" if args.fold_seed is None else f"__fs{args.fold_seed}"
    tag = f"{args.regime}__{args.arm}__seed{args.seed}{fs}"
    result_path = out_root / args.experiment / f"{tag}.json"

    if args.skip_existing and result_path.exists():
        logger.info(f"{tag}: result exists, skipping")
        return

    if args.regime in ("strict_ncc",) or (args.regime == "ncd" and args.arm in QFORMER_ARMS
                                          and QFORMER_ARMS[args.arm] == "dra"):
        # These need DR-A pretraining re-run per fold on training cell lines/drugs
        # only; that path lives in run_cv_leakfree.py.
        logger.warning(
            f"regime={args.regime} with arm={args.arm} requires per-fold leak-free "
            "pretraining. Use scripts/run_cv_leakfree.py for that combination."
        )

    set_all_seeds(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    started = time.time()

    logger.info(f"=== {tag} | dataset={args.dataset} device={device} ===")
    args.drug_dim = infer_drug_dim(args.drug_csv)
    args.rna_dim = infer_drug_dim(args.rna_csv)  # same emb_* convention
    if args.rna_csv != RNA_CSV:
        logger.info(f"cell representation: {Path(args.rna_csv).name} ({args.rna_dim}-d)")
    # The shared feature vector is drug + RNA(256) + cancer one-hot(30) + tissue one-hot(26).
    args.feature_dim = args.drug_dim + args.rna_dim + 30 + 26
    if args.drug_dim != 768:
        logger.info(f"drug encoder: {Path(args.drug_csv).name} ({args.drug_dim}-d)")
        if args.arm == "xgboost" or args.arm in FEATURE_ARMS:
            raise SystemExit(
                f"--drug_csv with arm={args.arm} is not supported: the shared feature "
                "extractor in benchmark_5fold_ncc_ablation.py hardcodes a 768-d drug "
                "block. The encoder sweep uses the Q-Former arms only."
            )
    dataset = build_dataset(args.dataset, args.drug_csv, args.drug_dim,
                            args.rna_csv, args.rna_dim)

    if args.cellline_subset == "comparator_common":
        from gastro_transformer.comparators.data import common_celllines
        from torch.utils.data import Subset as _Subset

        keep, _ = common_celllines(dataset.cellline_ids, require=("deepcdr", "drugcell"))
        rows = [i for i, c in enumerate(dataset.cellline_ids) if c in keep]
        logger.info(f"comparator_common subset: {len(rows):,} of {len(dataset):,} rows")
        dataset = _SubsetView(dataset, rows)
    smiles_by_drug = None
    if args.regime == "scaffold":
        smiles_path = ROOT / "data/drug_smiles.csv"
        if not smiles_path.exists():
            raise SystemExit(
                f"{smiles_path} not found; run scripts/build_drug_encoders.py first."
            )
        smiles = pd.read_csv(smiles_path)
        smiles_by_drug = dict(zip(smiles.drug_id.astype(str), smiles.smiles))

    sample_fold = splits.build_folds(
        args.regime, dataset, n_folds=args.n_folds, seed=args.seed,
        smiles_by_drug=smiles_by_drug, fold_seed=args.fold_seed,
    )

    folds_to_run = args.folds if args.folds is not None else list(range(args.n_folds))
    if any(f >= args.n_folds for f in folds_to_run):
        raise ValueError(f"--folds {folds_to_run} out of range for --n_folds {args.n_folds}")

    fold_metrics = []
    for fold in folds_to_run:
        # Re-seed per fold so a resumed or reordered run reproduces exactly.
        set_all_seeds(args.seed * 1000 + fold)
        metrics, preds, targets, test_idx = run_fold(args, dataset, sample_fold, fold, device)
        fold_metrics.append(metrics)

        dump_predictions(
            out_root,
            experiment=args.experiment,
            regime=args.regime,
            arm=args.arm,
            seed=args.seed,
            fold_seed=args.fold_seed,
            fold=fold,
            split="test",
            dataset=dataset,
            row_indices=test_idx,
            y_true=targets,
            y_pred=preds,
        )
        logger.info(f"  fold {fold}: R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f}")

    average = _abl.aggregate_metrics(fold_metrics)
    result = {
        "arm": args.arm,
        "regime": args.regime,
        "seed": args.seed,
        "dataset": args.dataset,
        "fold_metrics": fold_metrics,
        "average": average,
        "config": vars(args),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as handle:
        json.dump(_abl.convert_numpy(result), handle, indent=2)

    elapsed = time.time() - started
    append_manifest(
        out_root,
        {
            "experiment": args.experiment,
            "regime": args.regime,
            "arm": args.arm,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "dataset": args.dataset,
            "r2_mean": average["r2"],
            "r2_std": average["r2_std"],
            "wall_clock_s": round(elapsed, 1),
            "args": vars(args),
        },
    )
    logger.info(
        f"=== {tag}: R2={average['r2']:.4f}+-{average['r2_std']:.4f} "
        f"in {elapsed / 60:.1f} min -> {result_path} ==="
    )


if __name__ == "__main__":
    main()
