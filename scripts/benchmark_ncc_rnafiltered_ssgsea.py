#!/usr/bin/env python3
"""
Benchmark: DR-A + ssGSEA on RNA-filtered dataset (592 cell-lines).

5-fold NCC (No Common Cell-Line) cross-validation on cell-lines with RNA only.

This combines:
    - RNA-filtering: Only cell-lines with RNA (592 out of 998)
    - DR-A + ssGSEA checkpoint: Pretrained with pathway enrichment scores
    - NCC split: No common cell-line between train/test (harder generalization)

Expected improvement from RNA-filtering:
    - Removes zero-filling confound (zero-filled cell-lines hurt performance)
    - Should match or exceed the full-998 CL results since zero-filled CLs are removed
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
from sklearn.model_selection import KFold

from gastro_transformer.config import GastroTransformerConfig
from gastro_transformer.model_with_ssgsea import ModalitySlotQFormerWithSsgsea
from gastro_transformer.data import DrugEmbeddingDataset
from gastro_transformer.data_with_ssgsea import IC50DatasetWithSsgsea

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

ROOT = '/workspace/volume/Gastro_transformers/gastro_v5/'
DRUG_CSV = ROOT + 'data/processed/drug_embeddings_20260224.csv'
IC50_CSV = ROOT + 'data/processed/ic50_data_20260224.csv'
RNA_CSV = ROOT + 'data/processed/ccle_rna_for_ic50.csv'
SSGSEA_TSV = ROOT + 'data/CCLE_20260324_ssGSEA_ccle_RNABert_sample_x_768Geneset.tsv'
SSGSEA_CHECKPOINT = ROOT + 'checkpoints_save/checkpoints_phase3_ssgsea/pretrained_phase3_ssgsea.pt'


def compute_metrics(preds, targets):
    """Compute regression metrics."""
    preds = np.array(preds)
    targets = np.array(targets)

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))

    from scipy.stats import pearsonr, spearmanr
    spearman_r, _ = spearmanr(preds, targets)
    pearson_r_val, _ = pearsonr(preds, targets)

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
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
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

    model.load_state_dict(best_state)
    return model, best_val_r2


def run_benchmark(folds=5, epochs=10, batch_size=256, device='cuda:0', ssgsea_dim=768):
    """Run RNA-filtered NCC benchmark with DR-A + ssGSEA."""
    logger.info("=" * 70)
    logger.info("BENCHMARK: DR-A + ssGSEA on RNA-filtered dataset (592 cell-lines)")
    logger.info("=" * 70)

    config = GastroTransformerConfig()
    config.num_query_tokens = 32
    config.qformer_layers = 6
    config.use_multitoken_cellline = True
    config.use_qformer = True
    config.use_qformer_for_ic50 = True
    config.use_ic50_attn_pool = True

    # Load full dataset first to identify RNA-available cell-lines
    drug_ds = DrugEmbeddingDataset(DRUG_CSV, drug_dim=768)
    full_ds = IC50DatasetWithSsgsea(
        IC50_CSV, drug_ds,
        rna_csv_path=RNA_CSV,
        rna_dim=256,
        ssgsea_tsv_path=SSGSEA_TSV,
        ssgsea_dim=ssgsea_dim,
        add_tissue_ids=True,
    )

    logger.info(f"Full dataset: {len(full_ds)} samples, {full_ds.num_celllines} cell-lines")
    logger.info(f"ssGSEA availability: {full_ds.cellline_has_ssgsea.sum().item()}/{full_ds.num_celllines}")

    # RNA filtering: get cell-lines that have RNA
    # IMPORTANT: cellline_ids has 173942 elements (one per sample), NOT aligned with cellline_has_rna (998 elements)
    # Build reverse mapping: array index -> cell line ID
    idx_to_cellline = {v: k for k, v in full_ds.cellline_to_idx.items()}

    # Collect RNA-available cell lines correctly
    rna_cls = set()
    for pos in range(len(full_ds.cellline_has_rna)):
        if full_ds.cellline_has_rna[pos]:
            cl_id = idx_to_cellline[pos]
            rna_cls.add(cl_id)

    n_rna = len(rna_cls)
    logger.info(f"RNA-available cell-lines: {n_rna}")

    # Use cell-lines with RNA (RNA-filtered setting)
    allowed_cls = rna_cls

    # Create RNA-filtered dataset
    rna_ds = IC50DatasetWithSsgsea(
        IC50_CSV, drug_ds,
        rna_csv_path=RNA_CSV,
        rna_dim=256,
        ssgsea_tsv_path=SSGSEA_TSV,
        ssgsea_dim=ssgsea_dim,
        add_tissue_ids=True,
        allowed_celllines=allowed_cls,
    )

    logger.info(f"RNA-filtered dataset: {len(rna_ds)} samples, {rna_ds.num_celllines} cell-lines")
    logger.info(f"ssGSEA in RNA-filtered: {rna_ds.cellline_has_ssgsea.sum().item()}/{rna_ds.num_celllines}")

    # Get unique cell-lines for NCC splitting
    unique_cls = sorted(set(rna_ds.cellline_ids))
    logger.info(f"NCC split on {len(unique_cls)} unique RNA-filtered cell-lines")

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_cl_idx, val_cl_idx) in enumerate(kf.split(unique_cls)):
        train_cls = set([unique_cls[i] for i in train_cl_idx])
        val_cls = set([unique_cls[i] for i in val_cl_idx])

        logger.info(f"\nFold {fold_idx + 1}/{folds}: train CLs={len(train_cls)}, val CLs={len(val_cls)}")

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

        logger.info(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

        model = ModalitySlotQFormerWithSsgsea(config, ssgsea_dim=ssgsea_dim)

        # Load pretrained weights from DR-A + ssGSEA checkpoint
        if Path(SSGSEA_CHECKPOINT).exists():
            ckpt = torch.load(SSGSEA_CHECKPOINT, map_location=device, weights_only=False)
            state_dict = ckpt['model_state_dict']
            model_dict = model.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            logger.info(f"  Loaded {len(filtered)}/{len(model_dict)} params from DR-A + ssGSEA checkpoint")
            model.load_state_dict(filtered, strict=False)
        else:
            logger.warning(f"ssGSEA checkpoint not found at {SSGSEA_CHECKPOINT}, using random init")

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
        logger.info(f"  Fold {fold_idx + 1}: R²={metrics['r2']:.4f}, "
                    f"Pearson={metrics['pearson_r']:.4f}, Spearman={metrics['spearman_r']:.4f}")

        del model
        torch.cuda.empty_cache()

    # Aggregate
    result = {
        'model': 'DR-A + ssGSEA (RNA-filtered NCC)',
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
    parser = argparse.ArgumentParser(
        description='Benchmark: DR-A + ssGSEA on RNA-filtered dataset (592 cell-lines)'
    )
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ssgsea_dim', type=int, default=768)
    parser.add_argument('--output_dir', type=str,
                        default=ROOT + 'reports/ncc_rnafiltered_ssgsea')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_benchmark(
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        ssgsea_dim=args.ssgsea_dim,
    )

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY: DR-A + ssGSEA (RNA-filtered NCC)")
    logger.info("=" * 70)
    logger.info(f"Mean R²:     {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
    logger.info(f"Mean Pearson: {result['mean_pearson_r']:.4f}")
    logger.info(f"Mean Spearman: {result['mean_spearman_r']:.4f}")
    logger.info(f"Mean RMSE:    {result['mean_rmse']:.4f}")
    logger.info(f"Mean MAE:     {result['mean_mae']:.4f}")
    logger.info("=" * 70)

    # Comparison with existing results
    logger.info("\nComparison with existing results:")
    logger.info("  RNA-filtered random split (DR-A RNA-only): R²=0.837")
    logger.info("  RNA-filtered NCC (DR-A RNA-only):           R²=??? (expected ~0.82)")
    logger.info("  Full 998 CL NCC + ssGSEA:                    R²=0.814")
    logger.info("  Full 998 CL NCC (DR-A RNA-only):            R²=0.791")

    # Save results
    results_path = output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
