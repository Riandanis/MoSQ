#!/usr/bin/env python3
"""
Random Split 5-Fold CV Benchmark - RNA-filtered dataset (592 cell-lines).

This runs random (row-based) splits on the RNA-available cell-lines only,
matching the RNA-filtered condition used in the literature comparison.

For context on split types:
- Random (R): Standard row-based random split (this script)
- NCC/CL-CV: No common cell-line (cell-line generalization)
- NCD: No common drug (drug generalization)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from gastro_transformer.config import GastroTransformerConfig
from gastro_transformer.model import ModalitySlotQFormer
from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset

# Reuse utilities from sample_efficiency.py
import importlib.util
_se_spec = importlib.util.spec_from_file_location(
    "sample_efficiency",
    str(Path(__file__).resolve().parent / "sample_efficiency.py")
)
_se = importlib.util.module_from_spec(_se_spec)
_se_spec.loader.exec_module(_se)

load_pretrained_weights = _se.load_pretrained_weights
evaluate_foundation = _se.evaluate_foundation
finetune_foundation = _se.finetune_foundation
aggregate_metrics = _se.aggregate_metrics
convert_numpy = _se.convert_numpy

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DRUG_CSV = str(ROOT / 'data/drug_embeddings.csv')
IC50_CSV = str(ROOT / 'data/ic50_data.csv')
RNA_CSV  = str(ROOT / 'data/ccle_rna_for_ic50.csv')
DEFAULT_BASELINE = str(ROOT / 'saved_checkpoints/pretrained_clrna.pt')
DEFAULT_PHASE3 = str(ROOT / 'saved_checkpoints/pretrained_dra.pt')


# RNA-Filtered Dataset Wrapper (from benchmark_literature_models.py)
class RNAFilteredDataset:
    """
    Wraps an IC50Dataset and filters to only samples from cell-lines with RNA.
    This removes the zero-filling confound for fair comparison with Garai et al.
    """
    def __init__(self, ic50_dataset):
        self.ic50_ds = ic50_dataset
        self._build_rna_filter()

    def _build_rna_filter(self):
        has_rna = self.ic50_ds.cellline_has_rna
        cl_indices = self.ic50_ds.cellline_indices
        rna_mask = has_rna[cl_indices]
        self.valid_indices = np.where(rna_mask.numpy())[0].tolist()
        self._n_rna_cl = has_rna.sum().item()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.ic50_ds[self.valid_indices[idx]]

    @property
    def num_celllines(self):
        return self.ic50_ds.num_celllines

    @property
    def cellline_ids(self):
        return [self.ic50_ds.cellline_ids[i] for i in self.valid_indices]


def make_config(device):
    cfg = GastroTransformerConfig()
    cfg.num_query_tokens = 32
    cfg.qformer_layers = 6
    cfg.use_multitoken_cellline = True
    cfg.use_qformer = True
    cfg.use_qformer_for_ic50 = True
    cfg.use_ic50_attn_pool = True
    cfg.qformer_finetune_lr_ratio = 0.2
    cfg.device = device
    return cfg


def create_random_folds(n_samples, n_folds=5, seed=42):
    """
    Create random row-based folds (standard KFold, NOT cell-line-aware).
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_size = n_samples // n_folds
    sample_fold = np.zeros(n_samples, dtype=int)

    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_samples
        sample_fold[indices[start:end]] = i

    logger.info(f"Random {n_folds}-fold split: {np.bincount(sample_fold)} samples per fold")
    return sample_fold


def get_random_fold_splits(sample_fold, fold_idx):
    """
    Return (train_indices, val_indices, test_indices) for random fold.
    """
    train_mask = sample_fold != fold_idx
    test_mask = sample_fold == fold_idx
    train_indices = np.where(train_mask)[0].tolist()
    test_indices = np.where(test_mask)[0].tolist()

    rng = np.random.default_rng(42 + fold_idx)
    rng.shuffle(train_indices)

    val_size = len(train_indices) // 4
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]

    return train_indices, val_indices, test_indices


def run_fold(checkpoint_path, ic50_ds, train_indices, val_indices, test_indices,
             config, device, epochs, batch_size):
    """Run a single fold: load checkpoint, fine-tune, evaluate."""
    train_loader = DataLoader(
        Subset(ic50_ds, train_indices),
        batch_size=batch_size, shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(ic50_ds, val_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(ic50_ds, test_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=0,
    )

    model = ModalitySlotQFormer(config).to(device)
    model = load_pretrained_weights(model, checkpoint_path, device)

    model = finetune_foundation(
        model, train_loader, val_loader, config, device,
        epochs=epochs, patience=5, use_tricks=False,
    )

    metrics = evaluate_foundation(model, test_loader, device)
    del model
    torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Random Split 5-Fold CV - RNA-filtered dataset (592 cell-lines)'
    )
    parser.add_argument('--checkpoint_baseline', type=str, default=DEFAULT_BASELINE)
    parser.add_argument('--checkpoint_phase3', type=str, default=DEFAULT_PHASE3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--output_dir', type=str,
                        default=str(ROOT / 'reports/random_split_5fold_rnafiltered'))
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RANDOM SPLIT 5-FOLD CV - RNA-Filtered (592 cell-lines)")
    logger.info("=" * 70)
    logger.info(f"Baseline: {args.checkpoint_baseline}")
    logger.info(f"Phase 3:  {args.checkpoint_phase3}")
    logger.info(f"Epochs: {args.epochs}, Folds: {args.n_folds}, Device: {device}")

    # Load data
    logger.info("Loading data...")
    drug_ds = DrugEmbeddingDataset(DRUG_CSV, drug_dim=768)
    ic50_ds = IC50Dataset(IC50_CSV, drug_ds, rna_csv_path=RNA_CSV, rna_dim=256, add_tissue_ids=True)
    logger.info(f"Full dataset: {len(ic50_ds)} samples, {ic50_ds.num_celllines} cell-lines")

    # Apply RNA filter
    rna_ds = RNAFilteredDataset(ic50_ds)
    n_rna_cl = rna_ds._n_rna_cl
    logger.info(f"RNA-filtered: {len(rna_ds)} samples, {n_rna_cl} cell-lines with RNA")

    # Create RANDOM folds (not cell-line-aware)
    sample_fold = create_random_folds(len(rna_ds), n_folds=args.n_folds, seed=42)

    # Report cell-line overlap
    for fold_idx in range(args.n_folds):
        test_mask = sample_fold == fold_idx
        test_cls = set(rna_ds.cellline_ids[i] for i in np.where(test_mask)[0])

        train_mask = sample_fold != fold_idx
        train_cls = set()
        for i in np.where(train_mask)[0]:
            train_cls.add(rna_ds.cellline_ids[i])

        overlap = len(test_cls & train_cls)
        logger.info(f"  Fold {fold_idx}: test cell-lines={len(test_cls)}, "
                    f"train cell-lines={len(train_cls)}, "
                    f"overlap={overlap} ({100*overlap/len(test_cls):.1f}% of test)")

    config = make_config(device)

    results = {}

    for model_name, ckpt_path in [
        ('baseline_clrna', args.checkpoint_baseline),
        ('phase3_pretrained', args.checkpoint_phase3),
    ]:
        logger.info(f"\n--- {model_name} ---")
        fold_metrics = []

        for fold_idx in range(args.n_folds):
            train_idx, val_idx, test_idx = get_random_fold_splits(sample_fold, fold_idx)
            logger.info(f"  Fold {fold_idx+1}: train={len(train_idx)}, "
                        f"val={len(val_idx)}, test={len(test_idx)}")

            metrics = run_fold(
                checkpoint_path=ckpt_path,
                ic50_ds=rna_ds,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                config=config,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            fold_metrics.append(metrics)
            logger.info(f"  Fold {fold_idx+1}: R²={metrics['r2']:.4f}, "
                        f"Pearson={metrics['pearson_r']:.4f}, "
                        f"Spearman={metrics['spearman_r']:.4f}")

        results[model_name] = {
            'fold_metrics': fold_metrics,
            'average': aggregate_metrics(fold_metrics),
        }

        avg = results[model_name]['average']
        logger.info(f"  AVG: R²={avg['r2']:.4f}±{avg['r2_std']:.4f}, "
                    f"Pearson={avg['pearson_r']:.4f}±{avg['pearson_r_std']:.4f}")

    # ---- Save results ----
    results_json = convert_numpy(results)
    results_json['config'] = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'n_folds': args.n_folds,
        'seed': 42,
        'device': device,
        'split_type': 'random (row-based, RNA-filtered 592 cell-lines)',
        'checkpoint_baseline': args.checkpoint_baseline,
        'checkpoint_phase3': args.checkpoint_phase3,
        'finetune_config': 'MultiToken, attn_pool, qformer_lr_ratio=0.2, MSE, no tricks',
        'dataset': f'RNA-filtered ({n_rna_cl} cell-lines out of {ic50_ds.num_celllines})',
    }

    out_path = output_dir / 'random_split_rnafiltered_results.json'
    with open(out_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # ---- Print comparison table ----
    logger.info("\n" + "=" * 80)
    logger.info("RANDOM SPLIT 5-FOLD CV RESULTS (RNA-Filtered, 592 cell-lines)")
    logger.info("=" * 80)
    logger.info("Comparison with full 998 cell-line results:")
    logger.info("  Full (998 CL) DR-A Random: R²=0.805")
    logger.info("-" * 80)

    base = results['baseline_clrna']['average']
    p3 = results['phase3_pretrained']['average']
    delta_r2 = p3['r2'] - base['r2']
    sign = '+' if delta_r2 >= 0 else ''

    logger.info(f"{'Model':<25} {'R²':>18} {'Pearson R':>18} {'Spearman R':>18}")
    logger.info("-" * 80)
    logger.info(f"{'Baseline (CLRNA)':<25} "
                f"{base['r2']:.4f}±{base['r2_std']:.4f}     "
                f"{base['pearson_r']:.4f}±{base['pearson_r_std']:.4f}     "
                f"{base['spearman_r']:.4f}±{base['spearman_r_std']:.4f}")
    logger.info(f"{'Phase 3 (IC50-Aware)':<25} "
                f"{p3['r2']:.4f}±{p3['r2_std']:.4f}     "
                f"{p3['pearson_r']:.4f}±{p3['pearson_r_std']:.4f}     "
                f"{p3['spearman_r']:.4f}±{p3['spearman_r_std']:.4f}")
    logger.info(f"\nDelta R²: {sign}{delta_r2:.4f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
