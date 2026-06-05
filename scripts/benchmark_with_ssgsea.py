#!/usr/bin/env python3
"""
Benchmark: MultiToken (RNA-BERT only) vs MultiToken + ssGSEA.

5-fold NCC (No Common Cell-Line) cross-validation comparing:
    1. MultiToken (RNA-BERT only): DR-A checkpoint → fine-tune → evaluate
    2. MultiToken + ssGSEA: DR-A checkpoint → fine-tune (with ssGSEA) → evaluate

Reads existing ablation results for baseline comparison.
Runs new experiments for ssGSEA variant.
"""

import sys
sys.path.insert(0, '/workspace/volume/Gastro_transformers/gastro_v5')

import argparse
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold

from gastro_transformer.config import GastroTransformerConfig
from gastro_transformer.model import ModalitySlotQFormer
from gastro_transformer.model_with_ssgsea import ModalitySlotQFormerWithSsgsea
from gastro_transformer.data import DrugEmbeddingDataset
from gastro_transformer.data_with_ssgsea import IC50DatasetWithSsgsea
from gastro_transformer.utils import get_tissue_id_for_cellline

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

ROOT = '/workspace/volume/Gastro_transformers/gastro_v5/'
DRUG_CSV = ROOT + 'data/processed/drug_embeddings_20260224.csv'
IC50_CSV = ROOT + 'data/processed/ic50_data_20260224.csv'
RNA_CSV = ROOT + 'data/processed/ccle_rna_for_ic50.csv'
SSGSEA_TSV = ROOT + 'data/CCLE_20260324_ssGSEA_ccle_RNABert_sample_x_768Geneset.tsv'
DEFAULT_CHECKPOINT = ROOT + 'checkpoints_save/checkpoints_phase3/pretrained_phase3.pt'
SSGSEA_CHECKPOINT = ROOT + 'checkpoints_save/checkpoints_phase3_ssgsea/pretrained_phase3_ssgsea.pt'

# Existing ablation results for RNA-only MultiToken baseline
EXISTING_RESULTS_PATH = ROOT + 'reports/5fold_ncc_ablation/ablation_results.json'


def load_existing_results():
    """Load existing ablation results for RNA-only MultiToken baseline."""
    if Path(EXISTING_RESULTS_PATH).exists():
        with open(EXISTING_RESULTS_PATH) as f:
            data = json.load(f)
        return data
    return None


def train_fold(
    model,
    train_loader,
    val_loader,
    device,
    config,
    num_epochs=10,
    lr=1e-4,
    qformer_lr_ratio=0.2,
):
    """Train a single fold and return best model + metrics."""
    model = model.to(device)

    # Freeze projectors and type embeddings (only train Q-Former + heads)
    for name, p in model.named_parameters():
        if name.startswith('projectors.') or name.startswith('modality_type_embeddings.'):
            p.requires_grad = False

    # Build optimizer with differential LR
    base_lr = lr
    low_lr = base_lr * qformer_lr_ratio

    low_lr_params = []
    high_lr_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('qformer.'):
            low_lr_params.append(p)
        else:
            high_lr_params.append(p)

    param_groups = [
        {'params': low_lr_params, 'lr': low_lr},
        {'params': high_lr_params, 'lr': base_lr},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()

    best_val_r2 = -float('inf')
    best_state = None

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            # Move batch to device
            drug = batch['drug_embed'].to(device)
            cancer = batch.get('cancer_type_id')
            if cancer is not None:
                cancer = cancer.to(device)
            tissue = batch.get('tissue_id')
            if tissue is not None:
                tissue = tissue.to(device)
            rna = batch.get('rna_embed')
            if rna is not None:
                rna = rna.to(device)
            rna_avail = batch.get('rna_available')
            if rna_avail is not None:
                rna_avail = rna_avail.to(device)
            ic50 = batch['ic50'].to(device)

            # ssGSEA fields
            ssgsea = batch.get('ssgsea_embed')
            if ssgsea is not None:
                ssgsea = ssgsea.to(device)
            ssgsea_avail = batch.get('ssgsea_available')
            if ssgsea_avail is not None:
                ssgsea_avail = ssgsea_avail.to(device)

            outputs = model(
                drug_embeds=drug,
                cancer_type_ids=cancer,
                tissue_ids=tissue,
                cellline_rna_embeds=rna,
                rna_available=rna_avail,
                ssgsea_embeds=ssgsea,
                ssgsea_available=ssgsea_avail,
            )
            pred = outputs['ic50_pred']
            loss = criterion(pred, ic50)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                drug = batch['drug_embed'].to(device)
                cancer = batch.get('cancer_type_id')
                if cancer is not None:
                    cancer = cancer.to(device)
                tissue = batch.get('tissue_id')
                if tissue is not None:
                    tissue = tissue.to(device)
                rna = batch.get('rna_embed')
                if rna is not None:
                    rna = rna.to(device)
                rna_avail = batch.get('rna_available')
                if rna_avail is not None:
                    rna_avail = rna_avail.to(device)
                ic50 = batch['ic50'].to(device)

                # ssGSEA fields
                ssgsea = batch.get('ssgsea_embed')
                if ssgsea is not None:
                    ssgsea = ssgsea.to(device)
                ssgsea_avail = batch.get('ssgsea_available')
                if ssgsea_avail is not None:
                    ssgsea_avail = ssgsea_avail.to(device)

                outputs = model(
                    drug_embeds=drug,
                    cancer_type_ids=cancer,
                    tissue_ids=tissue,
                    cellline_rna_embeds=rna,
                    rna_available=rna_avail,
                    ssgsea_embeds=ssgsea,
                    ssgsea_available=ssgsea_avail,
                )
                val_preds.append(outputs['ic50_pred'].cpu())
                val_targets.append(ic50.cpu())

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        val_r2 = 1 - F.mse_loss(val_preds, val_targets) / val_targets.var()
        val_r2 = val_r2.item()

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best state
    model.load_state_dict(best_state)
    return model, best_val_r2


def compute_metrics(preds, targets):
    """Compute regression metrics."""
    preds = np.array(preds)
    targets = np.array(targets)

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))

    # Pearson correlation
    centered_pred = preds - preds.mean()
    centered_target = targets - targets.mean()
    pearson_r = np.sum(centered_pred * centered_target) / (
        np.sqrt(np.sum(centered_pred ** 2)) * np.sqrt(np.sum(centered_target ** 2))
    )

    # Spearman correlation (rank-based)
    from scipy.stats import pearsonr, spearmanr
    spearman_r, _ = spearmanr(preds, targets)
    pearson_r_val, _ = pearsonr(preds, targets)

    # R-squared
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        'r2': float(r2),
        'pearson_r': float(pearson_r_val),
        'spearman_r': float(spearman_r),
        'rmse': float(rmse),
        'mae': float(mae),
    }


def run_benchmark_rna_only(folds=5, epochs=10, batch_size=256, device='cuda:0'):
    """Run benchmark for RNA-only MultiToken (reads existing results or runs new)."""
    logger.info("=" * 60)
    logger.info("Benchmark: MultiToken (RNA-BERT only)")
    logger.info("=" * 60)

    # Try to load existing results
    existing = load_existing_results()
    if existing:
        logger.info(f"Using existing RNA-only results from {EXISTING_RESULTS_PATH}")
        return existing

    # Run new experiment (RNA-only baseline)
    logger.info("No existing results found. Running RNA-only baseline...")

    config = GastroTransformerConfig()
    config.num_query_tokens = 32
    config.qformer_layers = 6
    config.use_multitoken_cellline = True
    config.use_qformer = True
    config.use_qformer_for_ic50 = True
    config.use_ic50_attn_pool = True

    drug_ds = DrugEmbeddingDataset(DRUG_CSV, drug_dim=768)
    ic50_ds = IC50DatasetWithSsgsea(
        IC50_CSV, drug_ds,
        rna_csv_path=RNA_CSV,
        rna_dim=256,
        ssgsea_tsv_path=None,  # RNA-only mode
        ssgsea_dim=768,
        add_tissue_ids=True,
    )

    # Get unique cell-lines for NCC splitting
    unique_cls = sorted(set(ic50_ds.cellline_ids))
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_cl_idx, val_cl_idx) in enumerate(kf.split(unique_cls)):
        train_cls = set([unique_cls[i] for i in train_cl_idx])
        val_cls = set([unique_cls[i] for i in val_cl_idx])

        # Filter dataset to train/val cell-lines
        from gastro_transformer.data_with_ssgsea import IC50DatasetWithSsgsea as ICDS
        train_ds = ICDS(
            IC50_CSV, drug_ds,
            rna_csv_path=RNA_CSV,
            rna_dim=256,
            ssgsea_tsv_path=None,
            ssgsea_dim=768,
            add_tissue_ids=True,
            allowed_celllines=train_cls,
        )
        val_ds = ICDS(
            IC50_CSV, drug_ds,
            rna_csv_path=RNA_CSV,
            rna_dim=256,
            ssgsea_tsv_path=None,
            ssgsea_dim=768,
            add_tissue_ids=True,
            allowed_celllines=val_cls,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = ModalitySlotQFormer(config)

        # Load pretrained weights
        ckpt = torch.load(DEFAULT_CHECKPOINT, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model.load_state_dict(filtered, strict=False)

        model, _ = train_fold(model, train_loader, val_loader, device, config, num_epochs=epochs)

        # Evaluate
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                drug = batch['drug_embed'].to(device)
                cancer = batch.get('cancer_type_id')
                if cancer is not None:
                    cancer = cancer.to(device)
                tissue = batch.get('tissue_id')
                if tissue is not None:
                    tissue = tissue.to(device)
                rna = batch.get('rna_embed')
                if rna is not None:
                    rna = rna.to(device)
                rna_avail = batch.get('rna_available')
                if rna_avail is not None:
                    rna_avail = rna_avail.to(device)
                ic50 = batch['ic50']

                outputs = model(
                    drug_embeds=drug,
                    cancer_type_ids=cancer,
                    tissue_ids=tissue,
                    cellline_rna_embeds=rna,
                    rna_available=rna_avail,
                )
                val_preds.append(outputs['ic50_pred'].cpu())
                val_targets.append(ic50)

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        metrics = compute_metrics(val_preds, val_targets)
        fold_results.append(metrics)
        logger.info(f"Fold {fold_idx + 1}/{folds}: R²={metrics['r2']:.4f}, "
                    f"Pearson={metrics['pearson_r']:.4f}, Spearman={metrics['spearman_r']:.4f}")

    # Aggregate
    result = {
        'model': 'MultiToken (RNA-BERT only)',
        'folds': fold_results,
        'mean_r2': np.mean([r['r2'] for r in fold_results]),
        'std_r2': np.std([r['r2'] for r in fold_results]),
        'mean_pearson_r': np.mean([r['pearson_r'] for r in fold_results]),
        'mean_spearman_r': np.mean([r['spearman_r'] for r in fold_results]),
        'mean_rmse': np.mean([r['rmse'] for r in fold_results]),
        'mean_mae': np.mean([r['mae'] for r in fold_results]),
    }

    return result


def run_benchmark_with_ssgsea(folds=5, epochs=10, batch_size=256, device='cuda:0', ssgsea_dim=768):
    """Run benchmark for MultiToken + ssGSEA."""
    logger.info("=" * 60)
    logger.info("Benchmark: MultiToken + ssGSEA")
    logger.info("=" * 60)

    config = GastroTransformerConfig()
    config.num_query_tokens = 32
    config.qformer_layers = 6
    config.use_multitoken_cellline = True
    config.use_qformer = True
    config.use_qformer_for_ic50 = True
    config.use_ic50_attn_pool = True

    drug_ds = DrugEmbeddingDataset(DRUG_CSV, drug_dim=768)
    ic50_ds = IC50DatasetWithSsgsea(
        IC50_CSV, drug_ds,
        rna_csv_path=RNA_CSV,
        rna_dim=256,
        ssgsea_tsv_path=SSGSEA_TSV,
        ssgsea_dim=ssgsea_dim,
        add_tissue_ids=True,
    )

    logger.info(f"ssGSEA availability: {ic50_ds.cellline_has_ssgsea.sum().item()}/{ic50_ds.num_celllines}")

    # Get unique cell-lines for NCC splitting
    unique_cls = sorted(set(ic50_ds.cellline_ids))
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_cl_idx, val_cl_idx) in enumerate(kf.split(unique_cls)):
        train_cls = set([unique_cls[i] for i in train_cl_idx])
        val_cls = set([unique_cls[i] for i in val_cl_idx])

        # Filter dataset to train/val cell-lines
        train_ds = IC50DatasetWithSsgsea(
            IC50_CSV, drug_ds,
            rna_csv_path=RNA_CSV,
            rna_dim=256,
            ssgsea_tsv_path=SSGSEA_TSV,
            ssgsea_dim=ssgsea_dim,
            add_tissue_ids=True,
            allowed_celllines=train_cls,
        )
        val_ds = IC50DatasetWithSsgsea(
            IC50_CSV, drug_ds,
            rna_csv_path=RNA_CSV,
            rna_dim=256,
            ssgsea_tsv_path=SSGSEA_TSV,
            ssgsea_dim=ssgsea_dim,
            add_tissue_ids=True,
            allowed_celllines=val_cls,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = ModalitySlotQFormerWithSsgsea(config, ssgsea_dim=ssgsea_dim)

        # Load pretrained weights from DR-A + ssGSEA checkpoint
        ssgsea_ckpt_path = SSGSEA_CHECKPOINT
        if not Path(ssgsea_ckpt_path).exists():
            logger.warning(f"ssGSEA checkpoint not found at {ssgsea_ckpt_path}, using base DR-A checkpoint")
            ssgsea_ckpt_path = DEFAULT_CHECKPOINT
        ckpt = torch.load(ssgsea_ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        model_dict = model.state_dict()

        # Filter params that exist in both
        filtered = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered[k] = v

        logger.info(f"Fold {fold_idx + 1}: Loaded {len(filtered)}/{len(model_dict)} params from DR-A checkpoint")

        model.load_state_dict(filtered, strict=False)

        model, _ = train_fold(model, train_loader, val_loader, device, config, num_epochs=epochs)

        # Evaluate
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                drug = batch['drug_embed'].to(device)
                cancer = batch.get('cancer_type_id')
                if cancer is not None:
                    cancer = cancer.to(device)
                tissue = batch.get('tissue_id')
                if tissue is not None:
                    tissue = tissue.to(device)
                rna = batch.get('rna_embed')
                if rna is not None:
                    rna = rna.to(device)
                rna_avail = batch.get('rna_available')
                if rna_avail is not None:
                    rna_avail = rna_avail.to(device)
                ic50 = batch['ic50']

                # ssGSEA
                ssgsea = batch.get('ssgsea_embed')
                if ssgsea is not None:
                    ssgsea = ssgsea.to(device)
                ssgsea_avail = batch.get('ssgsea_available')
                if ssgsea_avail is not None:
                    ssgsea_avail = ssgsea_avail.to(device)

                outputs = model(
                    drug_embeds=drug,
                    cancer_type_ids=cancer,
                    tissue_ids=tissue,
                    cellline_rna_embeds=rna,
                    rna_available=rna_avail,
                    ssgsea_embeds=ssgsea,
                    ssgsea_available=ssgsea_avail,
                )
                val_preds.append(outputs['ic50_pred'].cpu())
                val_targets.append(ic50)

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        metrics = compute_metrics(val_preds, val_targets)
        fold_results.append(metrics)
        logger.info(f"Fold {fold_idx + 1}/{folds}: R²={metrics['r2']:.4f}, "
                    f"Pearson={metrics['pearson_r']:.4f}, Spearman={metrics['spearman_r']:.4f}")

    # Aggregate
    result = {
        'model': 'MultiToken + ssGSEA',
        'folds': fold_results,
        'mean_r2': np.mean([r['r2'] for r in fold_results]),
        'std_r2': np.std([r['r2'] for r in fold_results]),
        'mean_pearson_r': np.mean([r['pearson_r'] for r in fold_results]),
        'mean_spearman_r': np.mean([r['spearman_r'] for r in fold_results]),
        'mean_rmse': np.mean([r['rmse'] for r in fold_results]),
        'mean_mae': np.mean([r['mae'] for r in fold_results]),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Benchmark: MultiToken vs MultiToken + ssGSEA')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ssgsea_dim', type=int, default=768)
    parser.add_argument('--output_dir', type=str,
                        default=ROOT + 'reports/benchmark_with_ssgsea')
    parser.add_argument('--skip_rna_only', action='store_true',
                        help='Skip RNA-only baseline (use existing results only)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # RNA-only baseline
    if args.skip_rna_only:
        rna_only_results = load_existing_results()
        if rna_only_results is None:
            logger.warning("No existing RNA-only results found. Running RNA-only baseline...")
            rna_only_results = run_benchmark_rna_only(
                folds=args.folds, epochs=args.epochs,
                batch_size=args.batch_size, device=args.device
            )
    else:
        rna_only_results = run_benchmark_rna_only(
            folds=args.folds, epochs=args.epochs,
            batch_size=args.batch_size, device=args.device
        )

    # MultiToken + ssGSEA
    ssgsea_results = run_benchmark_with_ssgsea(
        folds=args.folds, epochs=args.epochs,
        batch_size=args.batch_size, device=args.device,
        ssgsea_dim=args.ssgsea_dim
    )

    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS COMPARISON")
    logger.info("=" * 60)
    logger.info(f"MultiToken (RNA-BERT only): R² = {rna_only_results['mean_r2']:.4f} ± {rna_only_results['std_r2']:.4f}")
    logger.info(f"MultiToken + ssGSEA:         R² = {ssgsea_results['mean_r2']:.4f} ± {ssgsea_results['std_r2']:.4f}")
    delta_r2 = ssgsea_results['mean_r2'] - rna_only_results['mean_r2']
    logger.info(f"Delta R²: {'+' if delta_r2 > 0 else ''}{delta_r2:.4f}")

    # Save results
    results = {
        'rna_only': rna_only_results,
        'ssgsea': ssgsea_results,
        'delta_r2': delta_r2,
        'config': {
            'folds': args.folds,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'ssgsea_dim': args.ssgsea_dim,
        }
    }

    results_path = output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
