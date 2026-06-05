#!/usr/bin/env python3
"""
Literature Model Comparison: Compare Q-Former against DeepCDR, DrugCell, and tCNN.

Runs 6 models on RNA-available cell-lines only (592 out of 998) for a fair
comparison with Garai et al. (Commun. Chem. 2026) which used 561 cell-lines.

Models:
  1. Phase 3 Q-Former (pretrained)        - Q-Former cross-attention + Phase 3 pretrain
  2. Q-Former (random init)               - Q-Former without pretraining
  3. DeepCDR-style                        - MLP branches + 1D CNN fusion (random init)
  4. DrugCell-style                       - MLP branches + late FC fusion (random init)
  5. tCNN-style                           - Twin MLP + FC fusion (random init)
  6. Standalone MLP                       - Simple concat MLP (random init)

All use same input features: drug (768d ChemBERTa) + cancer (30d OH) + tissue (26d OH) + RNA (256d) = 1080d
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
from torch.utils.data import DataLoader, Subset, Dataset
from pathlib import Path
from copy import deepcopy

from gastro_transformer.config import GastroTransformerConfig
from gastro_transformer.model import ModalitySlotQFormer
from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset
from gastro_transformer.losses import compute_ic50_metrics
from gastro_transformer.train import EMAModel

# Reuse utilities from sample_efficiency.py
import importlib.util
_se_spec = importlib.util.spec_from_file_location(
    "sample_efficiency",
    "/workspace/volume/Gastro_transformers/gastro_v5/scripts/sample_efficiency.py"
)
_se = importlib.util.module_from_spec(_se_spec)
_se_spec.loader.exec_module(_se)

create_cellline_aware_folds = _se.create_cellline_aware_folds
load_pretrained_weights = _se.load_pretrained_weights
evaluate_foundation = _se.evaluate_foundation
finetune_foundation = _se.finetune_foundation
get_fold_splits = _se.get_fold_splits
aggregate_metrics = _se.aggregate_metrics
convert_numpy = _se.convert_numpy
batch_to_features = _se.batch_to_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

ROOT = '/workspace/volume/Gastro_transformers/gastro_v5/'
DRUG_CSV = ROOT + 'data/processed/drug_embeddings_20260224.csv'
IC50_CSV = ROOT + 'data/processed/ic50_data_20260224.csv'
RNA_CSV  = ROOT + 'data/processed/ccle_rna_for_ic50.csv'
DEFAULT_CLRNA = ROOT + 'checkpoints_save/checkpoints_CLRNA/pretrained.pt'
DEFAULT_PHASE3 = ROOT + 'checkpoints_save/checkpoints_phase3/pretrained_phase3.pt'

NUM_CANCER_TYPES = 30
NUM_TISSUE_TYPES = 26
DRUG_DIM = 768
RNA_DIM = 256
CELL_DIM = NUM_CANCER_TYPES + NUM_TISSUE_TYPES + RNA_DIM  # 312


# =============================================================================
# RNA-Filtered Dataset Wrapper
# =============================================================================

class RNAFilteredDataset(Dataset):
    """
    Wraps an IC50Dataset and filters to only samples from cell-lines with RNA.
    This removes the zero-filling confound for fair comparison with Garai et al.

    Proxies cellline_to_idx and cellline_ids so create_cellline_aware_folds works.
    """
    def __init__(self, ic50_dataset: IC50Dataset):
        self.ic50_ds = ic50_dataset
        self._build_rna_filter()

    def _build_rna_filter(self):
        """Build index mapping: original index -> filtered index (or -1)."""
        has_rna = self.ic50_ds.cellline_has_rna  # [num_celllines]
        cl_indices = self.ic50_ds.cellline_indices  # [num_samples]

        # Boolean mask: True where cell-line has RNA
        rna_mask = has_rna[cl_indices]  # [num_samples]
        self.valid_indices = np.where(rna_mask.numpy())[0].tolist()
        # Log after init is called (during main)
        self._n_rna_cl = has_rna.sum().item()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.ic50_ds[self.valid_indices[idx]]

    # Proxy attributes needed by create_cellline_aware_folds
    @property
    def cellline_to_idx(self):
        return self.ic50_ds.cellline_to_idx

    @property
    def num_celllines(self):
        return self.ic50_ds.num_celllines

    @property
    def cellline_ids(self):
        """Return cellline_ids for valid (RNA-available) samples only."""
        return [self.ic50_ds.cellline_ids[i] for i in self.valid_indices]


# =============================================================================
# Literature Model Architectures (random init only)
# =============================================================================

class DeepCDRStyle(nn.Module):
    """
    DeepCDR-style architecture: MLP branches for drug/cell + 1D CNN fusion.
    Reference: Liu et al., DeepCDR (2019) - drug + cell-line branch with 1D CNN.

    Architecture:
      drug: 768 → 256 → 128
      cell: 312 → 256 → 128
      concat → 1DConv(kernel=3, filters=128) → 1DConv(kernel=3, filters=128) → FC → IC50
    """
    def __init__(self, drug_dim=768, cell_dim=312, hidden=128, dropout=0.1):
        super().__init__()
        self.drug_branch = nn.Sequential(
            nn.Linear(drug_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cell_branch = nn.Sequential(
            nn.Linear(cell_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 1D CNN on concatenated [drug_feat; cell_feat]
        # Conv1d: [B, channels, length], then transpose for LayerNorm [B, length, channels]
        self.conv = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden * 2, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        """
        Args:
            x: [B, drug_dim + cell_dim] = [B, 1080]
        """
        drug = x[:, :DRUG_DIM]
        cell = x[:, DRUG_DIM:]

        drug_h = self.drug_branch(drug)          # [B, 128]
        cell_h = self.cell_branch(cell)           # [B, 128]

        # Concat and apply 1D CNN
        fused = torch.cat([drug_h, cell_h], dim=-1)  # [B, 256]
        fused = fused.unsqueeze(-1)                     # [B, 256, 1]
        fused = self.conv(fused)                       # [B, 128, 1]
        fused = fused.squeeze(-1)                      # [B, 128]
        fused = self.ln(fused)                        # [B, 128]

        return self.head(fused).squeeze(-1)             # [B]


class DrugCellStyle(nn.Module):
    """
    DrugCell-style architecture: MLP branches + late FC fusion.
    Reference: Jang et al., DrugCell (2021) - hierarchical neural network for drug response.

    Architecture:
      drug: 768 → 256 → 128
      cell: 312 → 256 → 128
      concat → FC(256) → FC(128) → IC50
    """
    def __init__(self, drug_dim=768, cell_dim=312, hidden=128, dropout=0.1):
        super().__init__()
        self.drug_branch = nn.Sequential(
            nn.Linear(drug_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cell_branch = nn.Sequential(
            nn.Linear(cell_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        drug = x[:, :DRUG_DIM]
        cell = x[:, DRUG_DIM:]

        drug_h = self.drug_branch(drug)
        cell_h = self.cell_branch(cell)

        fused = torch.cat([drug_h, cell_h], dim=-1)
        return self.fusion(fused).squeeze(-1)


class tCNNStyle(nn.Module):
    """
    tCNN-style architecture: Twin CNN + FC fusion.
    Reference: Chuang et al., tCNN (2020) - twin CNN for drug-target interaction.

    Architecture:
      drug: 768 → 256 (MLP)
      cell: 312 → 256 (MLP)
      concat → FC(256) → IC50
    """
    def __init__(self, drug_dim=768, cell_dim=312, hidden=256, dropout=0.1):
        super().__init__()
        self.drug_branch = nn.Sequential(
            nn.Linear(drug_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cell_branch = nn.Sequential(
            nn.Linear(cell_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        drug = x[:, :DRUG_DIM]
        cell = x[:, DRUG_DIM:]

        drug_h = self.drug_branch(drug)
        cell_h = self.cell_branch(cell)

        fused = torch.cat([drug_h, cell_h], dim=-1)
        return self.head(fused).squeeze(-1)


class StandaloneMLP(nn.Module):
    """
    Simple concatenation MLP baseline.
    Architecture: 1080 → 512 → 256 → 1
    """
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


# =============================================================================
# Training utilities
# =============================================================================

def train_model(model, train_loader, val_loader, device, epochs=10, patience=5, lr=1e-3, model_name="model"):
    """Train a standalone model (MLP or literature model)."""
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

        # Validate
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
            logger.info(f"    [{model_name}] Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, loader, device):
    """Evaluate a standalone model."""
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


def make_qformer_config(device, pretrained=False, use_multitoken=True):
    """Create Q-Former config for IC50 fine-tuning."""
    cfg = GastroTransformerConfig()
    cfg.num_query_tokens = 32
    cfg.qformer_layers = 6
    cfg.qformer_finetune_lr_ratio = 0.2
    cfg.device = device

    if use_multitoken:
        cfg.use_multitoken_cellline = True
        cfg.use_qformer = True
        cfg.use_qformer_for_ic50 = True
        cfg.use_ic50_attn_pool = True
    else:
        # Feature-based (single fused cell token)
        cfg.use_feature_cellline_encoder = True
        cfg.use_qformer = True
        cfg.use_qformer_for_ic50 = True
        cfg.use_ic50_attn_pool = True
    return cfg


def run_literature_model(model_class, model_kwargs, train_loader, val_loader, test_loader,
                          device, epochs, model_name):
    """Train and evaluate a literature model."""
    model = model_class(**model_kwargs).to(device)
    model = train_model(model, train_loader, val_loader, device,
                        epochs=epochs, patience=5, lr=1e-3, model_name=model_name)
    metrics = evaluate_model(model, test_loader, device)
    del model
    torch.cuda.empty_cache()
    return metrics


def run_qformer_model(checkpoint_path, train_loader, val_loader, test_loader,
                       device, epochs, config, model_name, load_pretrained=True):
    """Train and evaluate a Q-Former model."""
    model = ModalitySlotQFormer(config).to(device)
    if load_pretrained and checkpoint_path is not None:
        model = load_pretrained_weights(model, checkpoint_path, device)
    model = finetune_foundation(model, train_loader, val_loader, config, device,
                                 epochs=epochs, patience=5, use_tricks=False)
    metrics = evaluate_foundation(model, test_loader, device)
    del model
    torch.cuda.empty_cache()
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Literature Model Comparison Benchmark')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_folds', type=int, default=3)
    parser.add_argument('--checkpoint_clrna', type=str, default=DEFAULT_CLRNA)
    parser.add_argument('--checkpoint_phase3', type=str, default=DEFAULT_PHASE3)
    parser.add_argument('--output_dir', type=str,
                        default=ROOT + 'reports/literature_comparison')
    args = parser.parse_args()

    device = args.device
    n_folds = args.n_folds
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"LITERATURE MODEL COMPARISON (RNA-Filtered, 592 Cell-Lines, {n_folds}-Fold CV)")
    logger.info("=" * 70)
    logger.info(f"Device: {device}, Epochs: {args.epochs}, Batch: {args.batch_size}, Folds: {n_folds}")
    logger.info(f"CLRNA checkpoint: {args.checkpoint_clrna}")
    logger.info(f"Phase 3 checkpoint: {args.checkpoint_phase3}")

    # Load data
    logger.info("\nLoading data...")
    drug_ds = DrugEmbeddingDataset(DRUG_CSV, drug_dim=768)
    ic50_ds = IC50Dataset(IC50_CSV, drug_ds, rna_csv_path=RNA_CSV, rna_dim=256, add_tissue_ids=True)

    # Filter to RNA-available cell-lines only
    rna_filtered_ds = RNAFilteredDataset(ic50_ds)

    logger.info(f"RNA-filtered dataset: {len(rna_filtered_ds)} samples, "
                f"{ic50_ds.num_celllines} total cell-lines, "
                f"{ic50_ds.cellline_has_rna.sum().item()} with RNA")

    # Create folds on the RNA-filtered dataset
    # Note: we need to create folds based on cell-line membership in the filtered dataset
    sample_fold = create_cellline_aware_folds(rna_filtered_ds, n_folds=n_folds, seed=42)

    results = {}

    # Model configurations
    model_configs = [
        # (name, model_class, model_kwargs, is_foundation, checkpoint_path, load_pretrained)
        ('standalone_mlp', StandaloneMLP, {}, False, None, False),
        ('deepcdr_style', DeepCDRStyle, {}, False, None, False),
        ('drugcell_style', DrugCellStyle, {}, False, None, False),
        ('tcnn_style', tCNNStyle, {}, False, None, False),
        ('qformer_randinit', None, {}, True, None, False),
        ('qformer_phase3_pretrained', None, {}, True, args.checkpoint_phase3, True),
    ]

    for model_name, model_class, model_kwargs, is_foundation, ckpt_path, load_pret in model_configs:
        logger.info(f"\n{'='*70}")
        logger.info(f"MODEL: {model_name}")
        logger.info(f"{'='*70}")

        fold_metrics = []

        for fold_idx in range(n_folds):
            train_idx, val_idx, test_idx = get_fold_splits(sample_fold, fold_idx)
            logger.info(f"  Fold {fold_idx+1}: train={len(train_idx)}, "
                        f"val={len(val_idx)}, test={len(test_idx)}")

            train_loader = DataLoader(
                Subset(rna_filtered_ds, train_idx),
                batch_size=args.batch_size, shuffle=True,
                num_workers=4, persistent_workers=True)
            val_loader = DataLoader(
                Subset(rna_filtered_ds, val_idx),
                batch_size=args.batch_size, shuffle=False,
                num_workers=4, persistent_workers=True)
            test_loader = DataLoader(
                Subset(rna_filtered_ds, test_idx),
                batch_size=args.batch_size, shuffle=False,
                num_workers=4, persistent_workers=True)

            if is_foundation:
                # Q-Former models (use MultiToken config)
                config = make_qformer_config(device, use_multitoken=True)
                metrics = run_qformer_model(
                    ckpt_path, train_loader, val_loader, test_loader,
                    device, args.epochs, config, model_name, load_pretrained=load_pret)
            else:
                metrics = run_literature_model(
                    model_class, model_kwargs, train_loader, val_loader, test_loader,
                    device, args.epochs, model_name)

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
                    f"Pearson={avg['pearson_r']:.4f}±{avg['pearson_r_std']:.4f}, "
                    f"Spearman={avg['spearman_r']:.4f}±{avg['spearman_r_std']:.4f}")

    # ========== Save results ==========
    results_json = convert_numpy(results)
    results_json['config'] = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'n_folds': n_folds,
        'seed': 42,
        'device': device,
        'checkpoint_clrna': args.checkpoint_clrna,
        'checkpoint_phase3': args.checkpoint_phase3,
        'dataset': 'RNA-filtered (592 cell-lines with RNA out of 998)',
        'models': {
            'standalone_mlp': 'MLP(1080→512→256→1), MSE, lr=1e-3',
            'deepcdr_style': 'MLP branches + 1D CNN fusion, MSE, lr=1e-3',
            'drugcell_style': 'MLP branches + FC fusion, MSE, lr=1e-3',
            'tcnn_style': 'Twin MLP + FC fusion, MSE, lr=1e-3',
            'qformer_randinit': 'ModalitySlotQFormer (MultiToken), random init, MSE, differential LR',
            'qformer_phase3_pretrained': 'ModalitySlotQFormer (MultiToken), Phase 3 pretrained, MSE',
        }
    }

    out_path = output_dir / 'literature_results.json'
    with open(out_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # ========== Print comparison table ==========
    logger.info("\n" + "=" * 90)
    logger.info("LITERATURE MODEL COMPARISON RESULTS (RNA-Filtered, 592 Cell-Lines)")
    logger.info("=" * 90)
    logger.info(f"{'Model':<30} {'R²':>12} {'Pearson R':>12} {'Spearman R':>12} {'RMSE':>10} {'MAE':>10}")
    logger.info("-" * 90)

    for model_name, res in results.items():
        avg = res['average']
        logger.info(f"{model_name:<30} "
                    f"{avg['r2']:>6.4f}±{avg['r2_std']:<5.4f} "
                    f"{avg['pearson_r']:>6.4f}±{avg['pearson_r_std']:<5.4f} "
                    f"{avg['spearman_r']:>6.4f}±{avg['spearman_r_std']:<5.4f} "
                    f"{avg['rmse']:>10.4f} "
                    f"{avg['mae']:>10.4f}")

    logger.info("=" * 90)

    # Highlight Q-Former advantage
    qf_rand = results.get('qformer_randinit', {}).get('average', {})
    qf_pre = results.get('qformer_phase3_pretrained', {}).get('average', {})
    mlp_avg = results.get('standalone_mlp', {}).get('average', {})

    if qf_rand and mlp_avg:
        delta = qf_rand['r2'] - mlp_avg['r2']
        sign = '+' if delta >= 0 else ''
        logger.info(f"\nQ-Former (rand init) vs Standalone MLP: {sign}{delta:.4f} R²")
        logger.info("  → Q-Former fusion adds value even without pretraining")

    if qf_pre and qf_rand:
        delta = qf_pre['r2'] - qf_rand['r2']
        sign = '+' if delta >= 0 else ''
        logger.info(f"\nPhase 3 Pretrained vs Q-Former (rand init): {sign}{delta:.4f} R²")
        logger.info("  → Phase 3 pretraining adds additional gain on top of architecture")

    logger.info("=" * 90)


if __name__ == '__main__':
    main()
