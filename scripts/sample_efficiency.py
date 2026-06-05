#!/usr/bin/env python3
"""
Sample Efficiency Curve: How gracefully do models degrade with less training data?

Compares 3 models at 4 data fractions (10%, 25%, 50%, 100%):
1. Standalone MLP — task-specific baseline (no foundation model)
2. Foundation (rand. init) — architecture cost without pretraining
3. Foundation (pretrained, MultiToken + v3) — best config, shows pretraining value

Uses nested subsets: 10% ⊂ 25% ⊂ 50% ⊂ 100%.
Test/val splits stay identical across all fractions.
"""

import sys
sys.path.insert(0, '/workspace/volume/Gastro_transformers/gastro_v5')

import argparse
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gastro_transformer.config import GastroTransformerConfig
from gastro_transformer.model import ModalitySlotQFormer
from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset
from gastro_transformer.losses import compute_ic50_metrics
from gastro_transformer.train import EMAModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

ROOT = '/workspace/volume/Gastro_transformers/gastro_v5/'
DRUG_CSV = ROOT + 'data/processed/drug_embeddings_20260224.csv'
IC50_CSV = ROOT + 'data/processed/ic50_data_20260224.csv'
RNA_CSV  = ROOT + 'data/processed/ccle_rna_for_ic50.csv'
CHECKPOINT = ROOT + 'checkpoints_save/checkpoints_CLRNA/pretrained.pt'

NUM_CANCER_TYPES = 30
NUM_TISSUE_TYPES = 26
FRACTIONS = [0.10, 0.25, 0.50, 1.00]


# ---------------------------------------------------------------------------
# Standalone MLP (copied from run_standalone_mlp.py)
# ---------------------------------------------------------------------------

class StandaloneMLP(nn.Module):
    def __init__(self, input_dim=1080, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def batch_to_features(batch, device):
    drug = batch['drug_embed'].to(device)
    B = drug.shape[0]
    rna = batch.get('rna_embed')
    rna = rna.to(device) if rna is not None else torch.zeros(B, 256, device=device)

    cancer_oh = torch.zeros(B, NUM_CANCER_TYPES, device=device)
    cancer_ids = batch.get('cancer_type_id')
    if cancer_ids is not None:
        cancer_ids = cancer_ids.to(device)
        valid = (cancer_ids >= 0) & (cancer_ids < NUM_CANCER_TYPES)
        if valid.any():
            cancer_oh[valid] = cancer_oh[valid].scatter_(1, cancer_ids[valid].unsqueeze(1), 1.0)

    tissue_oh = torch.zeros(B, NUM_TISSUE_TYPES, device=device)
    tissue_ids = batch.get('tissue_id')
    if tissue_ids is not None:
        tissue_ids = tissue_ids.to(device)
        valid = (tissue_ids >= 0) & (tissue_ids < NUM_TISSUE_TYPES)
        if valid.any():
            tissue_oh[valid] = tissue_oh[valid].scatter_(1, tissue_ids[valid].unsqueeze(1), 1.0)

    return torch.cat([drug, rna, cancer_oh, tissue_oh], dim=-1)


# ---------------------------------------------------------------------------
# Shared utilities (from benchmark_ablations_v5.py)
# ---------------------------------------------------------------------------

def create_cellline_aware_folds(ic50_dataset, n_folds=3, seed=42):
    rng = np.random.default_rng(seed)
    unique_celllines = sorted(ic50_dataset.cellline_to_idx.keys())
    rng.shuffle(unique_celllines)
    n = len(unique_celllines)
    fold_size = n // n_folds
    cellline_folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        cellline_folds.append(set(unique_celllines[start:end]))

    sample_fold = np.zeros(len(ic50_dataset), dtype=int)
    for i in range(len(ic50_dataset)):
        cl_id = ic50_dataset.cellline_ids[i]
        for fold_idx, fold_cl in enumerate(cellline_folds):
            if cl_id in fold_cl:
                sample_fold[i] = fold_idx
                break
    logger.info(f"CV folds: {np.bincount(sample_fold)} samples per fold")
    return sample_fold


def load_pretrained_weights(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape}
    logger.info(f"Loaded {len(filtered)}/{len(model_dict)} params from checkpoint")
    model.load_state_dict(filtered, strict=False)
    return model


def evaluate_foundation(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            drug_embeds = batch['drug_embed'].to(device)
            ic50_targets = batch['ic50'].to(device)
            cellline_ids = batch['cellline_id'].to(device)
            cancer_type_ids = batch.get('cancer_type_id')
            if cancer_type_ids is not None:
                cancer_type_ids = cancer_type_ids.to(device)
            tissue_ids = batch.get('tissue_id')
            if tissue_ids is not None:
                tissue_ids = tissue_ids.to(device)
            cellline_rna = batch.get('rna_embed')
            if cellline_rna is not None:
                cellline_rna = cellline_rna.to(device)
            rna_available = batch.get('rna_available')
            if rna_available is not None:
                rna_available = rna_available.to(device)

            outputs = model(
                drug_embeds=drug_embeds,
                cellline_ids=cellline_ids,
                cancer_type_ids=cancer_type_ids,
                tissue_ids=tissue_ids,
                cellline_rna_embeds=cellline_rna,
                rna_available=rna_available,
            )
            all_preds.extend(outputs['ic50_pred'].cpu().numpy())
            all_targets.extend(ic50_targets.cpu().numpy())

    return compute_ic50_metrics(torch.tensor(all_preds), torch.tensor(all_targets))


def evaluate_mlp(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            features = batch_to_features(batch, device)
            ic50_targets = batch['ic50'].to(device)
            preds = model(features)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(ic50_targets.cpu().numpy())
    return compute_ic50_metrics(torch.tensor(all_preds), torch.tensor(all_targets))


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_standalone_mlp(model, train_loader, val_loader, device, epochs=10, patience=5, lr=1e-3):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_epochs = min(2, max(1, epochs // 5))
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss, n_batches = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            features = batch_to_features(batch, device)
            ic50_targets = batch['ic50'].to(device)
            with autocast('cuda', enabled=True):
                preds = model(features)
                loss = nn.functional.mse_loss(preds, ic50_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        val_loss, val_n = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch_to_features(batch, device)
                ic50_targets = batch['ic50'].to(device)
                preds = model(features)
                val_loss += nn.functional.mse_loss(preds, ic50_targets).item()
                val_n += 1
        val_loss /= max(val_n, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            logger.info(f"    Early stop epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def finetune_foundation(model, train_loader, val_loader, config, device,
                        epochs=10, patience=5, use_tricks=True):
    base_lr = config.learning_rate
    low_lr = base_lr * config.qformer_finetune_lr_ratio

    param_groups = []
    assigned = set()

    # Q-Former: low LR
    qf_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and 'qformer' in name:
            qf_params.append(p)
            assigned.add(name)
    if qf_params:
        param_groups.append({'params': qf_params, 'lr': low_lr})

    # Pretrained projectors: low LR
    for name, p in model.named_parameters():
        if p.requires_grad and name not in assigned and 'projectors' in name:
            if 'projectors.image' in name or 'projectors.rna' in name:
                param_groups.append({'params': [p], 'lr': low_lr})
            else:
                param_groups.append({'params': [p], 'lr': base_lr})
            assigned.add(name)

    # Pretrained type embeds: low LR
    for name, p in model.named_parameters():
        if p.requires_grad and name not in assigned and 'modality_type_embeddings' in name:
            if 'image' in name or ('rna' in name and 'cellline_rna' not in name):
                param_groups.append({'params': [p], 'lr': low_lr})
            else:
                param_groups.append({'params': [p], 'lr': base_lr})
            assigned.add(name)

    # Everything else: full LR
    other_params = [p for name, p in model.named_parameters()
                    if p.requires_grad and name not in assigned]
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr})

    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
    warmup_epochs = min(2, max(1, epochs // 5))
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    ema = EMAModel(model, decay=0.999) if use_tricks else None
    ic50_loss_fn = (lambda pred, tgt: nn.functional.huber_loss(pred, tgt, delta=1.5)) if use_tricks else nn.functional.mse_loss

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss, n_batches = 0, 0

        for batch in train_loader:
            optimizer.zero_grad()
            drug_embeds = batch['drug_embed'].to(device)
            ic50_targets = batch['ic50'].to(device)
            cellline_ids = batch['cellline_id'].to(device)
            cancer_type_ids = batch.get('cancer_type_id')
            if cancer_type_ids is not None:
                cancer_type_ids = cancer_type_ids.to(device)
            tissue_ids = batch.get('tissue_id')
            if tissue_ids is not None:
                tissue_ids = tissue_ids.to(device)
            cellline_rna = batch.get('rna_embed')
            if cellline_rna is not None:
                cellline_rna = cellline_rna.to(device)
            rna_available = batch.get('rna_available')
            if rna_available is not None:
                rna_available = rna_available.to(device)

            with autocast('cuda', enabled=True):
                outputs = model(
                    drug_embeds=drug_embeds,
                    cellline_ids=cellline_ids,
                    cancer_type_ids=cancer_type_ids,
                    tissue_ids=tissue_ids,
                    cellline_rna_embeds=cellline_rna,
                    rna_available=rna_available,
                )
                loss = ic50_loss_fn(outputs['ic50_pred'], ic50_targets)

                if use_tricks and tissue_ids is not None:
                    tissue_logits = outputs.get('tissue_logits')
                    if tissue_logits is not None:
                        tissue_loss = nn.functional.cross_entropy(tissue_logits, tissue_ids)
                        loss = loss + 0.1 * tissue_loss

                if use_tricks:
                    outputs2 = model(
                        drug_embeds=drug_embeds,
                        cellline_ids=cellline_ids,
                        cancer_type_ids=cancer_type_ids,
                        tissue_ids=tissue_ids,
                        cellline_rna_embeds=cellline_rna,
                        rna_available=rna_available,
                    )
                    loss2 = ic50_loss_fn(outputs2['ic50_pred'], ic50_targets)
                    rdrop_loss = nn.functional.mse_loss(outputs['ic50_pred'], outputs2['ic50_pred'])
                    loss = 0.5 * (loss + loss2) + 0.5 * rdrop_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        eval_model_fn = lambda: evaluate_foundation(model, val_loader, device)
        if ema is not None:
            with ema.apply(model):
                val_metrics = eval_model_fn()
                val_loss = val_metrics['rmse']
        else:
            val_metrics = eval_model_fn()
            val_loss = val_metrics['rmse']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if ema is not None:
                with ema.apply(model):
                    best_state = deepcopy(model.state_dict())
            else:
                best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            logger.info(f"    Early stop epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def make_config(device, **overrides):
    cfg = GastroTransformerConfig()
    cfg.num_query_tokens = 32
    cfg.qformer_layers = 6
    cfg.use_feature_cellline_encoder = True
    cfg.qformer_finetune_lr_ratio = 0.2
    cfg.device = device
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def aggregate_metrics(fold_metrics):
    agg = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        agg[key] = float(np.mean(values))
        agg[f'{key}_std'] = float(np.std(values))
    return agg


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Fold split helper
# ---------------------------------------------------------------------------

def get_fold_splits(sample_fold, fold_idx):
    """Return (train_indices, val_indices, test_indices) for a given fold.

    train_indices are shuffled with seed=42+fold_idx (same as all other scripts).
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


def subsample_train(train_indices, fraction):
    """Take first int(frac * len) entries — nested subsets since indices are pre-shuffled."""
    n = int(fraction * len(train_indices))
    return train_indices[:n]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Sample Efficiency Curve Experiment')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default=ROOT + 'reports/ablations_v5')
    parser.add_argument('--fractions', type=float, nargs='+', default=FRACTIONS)
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SAMPLE EFFICIENCY CURVE EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Fractions: {args.fractions}")
    logger.info(f"Epochs: {args.epochs}, Device: {device}")

    # Load data once
    logger.info("Loading data...")
    drug_ds = DrugEmbeddingDataset(DRUG_CSV, drug_dim=768)
    ic50_ds = IC50Dataset(IC50_CSV, drug_ds, rna_csv_path=RNA_CSV, rna_dim=256, add_tissue_ids=True)
    logger.info(f"Total: {len(ic50_ds)} samples, {ic50_ds.num_celllines} cell-lines")

    sample_fold = create_cellline_aware_folds(ic50_ds, n_folds=3, seed=42)

    results = {}

    for frac in args.fractions:
        frac_key = f"{frac:.2f}"
        logger.info(f"\n{'='*70}")
        logger.info(f"FRACTION: {frac_key} ({int(frac*100)}% of training data)")
        logger.info(f"{'='*70}")

        frac_results = {}

        # ---- Model 1: Standalone MLP ----
        logger.info(f"\n--- Standalone MLP @ {frac_key} ---")
        mlp_fold_metrics = []
        for fold_idx in range(3):
            train_idx, val_idx, test_idx = get_fold_splits(sample_fold, fold_idx)
            sub_train_idx = subsample_train(train_idx, frac)
            logger.info(f"  Fold {fold_idx+1}: train={len(sub_train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

            train_loader = DataLoader(Subset(ic50_ds, sub_train_idx),
                                      batch_size=args.batch_size, shuffle=True,
                                      num_workers=4, persistent_workers=True)
            val_loader = DataLoader(Subset(ic50_ds, val_idx),
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=4, persistent_workers=True)
            test_loader = DataLoader(Subset(ic50_ds, test_idx),
                                     batch_size=args.batch_size, shuffle=False,
                                     num_workers=4, persistent_workers=True)

            model = StandaloneMLP(input_dim=1080, hidden_dim=512).to(device)
            model = train_standalone_mlp(model, train_loader, val_loader, device,
                                         epochs=args.epochs, patience=5, lr=1e-3)
            metrics = evaluate_mlp(model, test_loader, device)
            mlp_fold_metrics.append(metrics)
            logger.info(f"  Fold {fold_idx+1}: R²={metrics['r2']:.4f}, Pearson={metrics['pearson_r']:.4f}")
            del model; torch.cuda.empty_cache()

        frac_results['standalone_mlp'] = {
            'fold_metrics': mlp_fold_metrics,
            'average': aggregate_metrics(mlp_fold_metrics),
        }

        # ---- Model 2: Foundation (random init, no tricks) ----
        logger.info(f"\n--- Foundation (rand. init) @ {frac_key} ---")
        randinit_fold_metrics = []
        config_randinit = make_config(
            device,
            use_qformer=True,
            use_qformer_for_ic50=False,  # detached head
            use_ic50_attn_pool=False,
        )
        for fold_idx in range(3):
            train_idx, val_idx, test_idx = get_fold_splits(sample_fold, fold_idx)
            sub_train_idx = subsample_train(train_idx, frac)
            logger.info(f"  Fold {fold_idx+1}: train={len(sub_train_idx)}")

            train_loader = DataLoader(Subset(ic50_ds, sub_train_idx),
                                      batch_size=args.batch_size, shuffle=True,
                                      num_workers=4, persistent_workers=True)
            val_loader = DataLoader(Subset(ic50_ds, val_idx),
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=4, persistent_workers=True)
            test_loader = DataLoader(Subset(ic50_ds, test_idx),
                                     batch_size=args.batch_size, shuffle=False,
                                     num_workers=4, persistent_workers=True)

            model = ModalitySlotQFormer(config_randinit).to(device)
            # NO pretrained weights
            model = finetune_foundation(model, train_loader, val_loader, config_randinit,
                                        device, epochs=args.epochs, patience=5, use_tricks=False)
            metrics = evaluate_foundation(model, test_loader, device)
            randinit_fold_metrics.append(metrics)
            logger.info(f"  Fold {fold_idx+1}: R²={metrics['r2']:.4f}, Pearson={metrics['pearson_r']:.4f}")
            del model; torch.cuda.empty_cache()

        frac_results['foundation_randinit'] = {
            'fold_metrics': randinit_fold_metrics,
            'average': aggregate_metrics(randinit_fold_metrics),
        }

        # ---- Model 3: Foundation (pretrained, MultiToken + v3 tricks) ----
        logger.info(f"\n--- Foundation (pretrained, MultiToken) @ {frac_key} ---")
        pretrained_fold_metrics = []
        config_pretrained = make_config(
            device,
            use_multitoken_cellline=True,
            use_qformer=True,
            use_qformer_for_ic50=True,
            use_ic50_attn_pool=True,
        )
        for fold_idx in range(3):
            train_idx, val_idx, test_idx = get_fold_splits(sample_fold, fold_idx)
            sub_train_idx = subsample_train(train_idx, frac)
            logger.info(f"  Fold {fold_idx+1}: train={len(sub_train_idx)}")

            train_loader = DataLoader(Subset(ic50_ds, sub_train_idx),
                                      batch_size=args.batch_size, shuffle=True,
                                      num_workers=4, persistent_workers=True)
            val_loader = DataLoader(Subset(ic50_ds, val_idx),
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=4, persistent_workers=True)
            test_loader = DataLoader(Subset(ic50_ds, test_idx),
                                     batch_size=args.batch_size, shuffle=False,
                                     num_workers=4, persistent_workers=True)

            model = ModalitySlotQFormer(config_pretrained).to(device)
            model = load_pretrained_weights(model, CHECKPOINT, device)
            model = finetune_foundation(model, train_loader, val_loader, config_pretrained,
                                        device, epochs=args.epochs, patience=5, use_tricks=True)
            metrics = evaluate_foundation(model, test_loader, device)
            pretrained_fold_metrics.append(metrics)
            logger.info(f"  Fold {fold_idx+1}: R²={metrics['r2']:.4f}, Pearson={metrics['pearson_r']:.4f}")
            del model; torch.cuda.empty_cache()

        frac_results['foundation_pretrained'] = {
            'fold_metrics': pretrained_fold_metrics,
            'average': aggregate_metrics(pretrained_fold_metrics),
        }

        results[frac_key] = frac_results

        # Print fraction summary
        logger.info(f"\n--- Summary @ {frac_key} ---")
        for model_name, res in frac_results.items():
            m = res['average']
            logger.info(f"  {model_name:<25} R²={m['r2']:.4f}±{m['r2_std']:.4f}  "
                        f"Pearson={m['pearson_r']:.4f}±{m['pearson_r_std']:.4f}")

    # ========== Save results ==========
    results_json = convert_numpy(results)
    results_json['config'] = {
        'fractions': args.fractions,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'n_folds': 3,
        'seed': 42,
        'device': device,
        'checkpoint': CHECKPOINT,
        'models': {
            'standalone_mlp': 'StandaloneMLP(1080→512→256→1), MSE, lr=1e-3',
            'foundation_randinit': 'ModalitySlotQFormer(feature-based, detached head), MSE, no tricks, no pretrain',
            'foundation_pretrained': 'ModalitySlotQFormer(multitoken, attn pool), pretrained, Huber+EMA+RDrop+tissue',
        }
    }

    out_path = output_dir / 'sample_efficiency_results.json'
    with open(out_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # ========== Generate plot ==========
    generate_plot(results, args.fractions, output_dir)

    # ========== Final summary table ==========
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE EFFICIENCY RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Fraction':<10} {'Standalone MLP':>18} {'Foundation (rand)':>18} {'Foundation (pre)':>18}")
    logger.info("-" * 66)
    for frac in args.fractions:
        frac_key = f"{frac:.2f}"
        r = results[frac_key]
        mlp_r2 = r['standalone_mlp']['average']['r2']
        rand_r2 = r['foundation_randinit']['average']['r2']
        pre_r2 = r['foundation_pretrained']['average']['r2']
        logger.info(f"{frac_key:<10} {mlp_r2:>18.4f} {rand_r2:>18.4f} {pre_r2:>18.4f}")


def generate_plot(results, fractions, output_dir):
    """Generate sample efficiency curve plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    model_labels = {
        'standalone_mlp': ('Standalone MLP (687K params)', '#2196F3', 's'),
        'foundation_randinit': ('Foundation (rand. init, ~78M)', '#FF9800', '^'),
        'foundation_pretrained': ('Foundation (pretrained, MultiToken)', '#4CAF50', 'o'),
    }

    pct_labels = [f"{int(f*100)}%" for f in fractions]

    for model_key, (label, color, marker) in model_labels.items():
        means = []
        stds = []
        for frac in fractions:
            frac_key = f"{frac:.2f}"
            avg = results[frac_key][model_key]['average']
            means.append(avg['r2'])
            stds.append(avg['r2_std'])

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(fractions, means, marker=marker, label=label, color=color,
                linewidth=2, markersize=8, zorder=3)
        ax.fill_between(fractions, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel('Fraction of Training Data', fontsize=12)
    ax.set_ylabel('R² (3-fold CV)', fontsize=12)
    ax.set_title('Sample Efficiency: Pretrained Foundation Model\nvs Task-Specific Baselines', fontsize=13)
    ax.set_xscale('log')
    ax.set_xticks(fractions)
    ax.set_xticklabels(pct_labels)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.08, 1.15)

    plt.tight_layout()
    plot_path = output_dir / 'sample_efficiency_curve.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
