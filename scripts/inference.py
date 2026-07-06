#!/usr/bin/env python3
"""
Inference script for Gastro-Transformer v2.
Loads trained model and generates predictions on test set.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from gastro_transformer.config import GastroTransformerConfig
from gastro_transformer.model import ModalitySlotQFormer
from gastro_transformer.model_v21 import ModalitySlotQFormer as ModalitySlotQFormerV21
from gastro_transformer.data import (
    DrugEmbeddingDataset,
    IC50Dataset,
    create_data_loaders,
    split_ic50_dataset_cellline_aware,
)
from gastro_transformer.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Gastro-Transformer v2 Inference')
    parser.add_argument('--checkpoint', type=str, default='saved_checkpoints/pretrained_dra.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--paired_image_csv', type=str,
                        default='data/paired_image_ms-bcpp.csv')
    parser.add_argument('--paired_rna_csv', type=str,
                        default='data/paired_rna_ms-bcpp.csv')
    parser.add_argument('--unpaired_image_csv', type=str,
                        default='data/unpaired_image.csv')
    parser.add_argument('--unpaired_rna_csv', type=str,
                        default='data/unpaired_rna.csv')
    parser.add_argument('--drug_embeddings_csv', type=str,
                        default='data/drug_embeddings.csv')
    parser.add_argument('--ic50_csv', type=str,
                        default='data/ic50_data.csv')
    parser.add_argument('--cellline_rna_csv', type=str,
                        default='data/ccle_rna_for_ic50.csv')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--output_dir', type=str, default='reports')
    parser.add_argument('--model_version', type=str, default='v2',
                        choices=['v2', 'v21'],
                        help='Model version: v2 = standard, v21 = attention pool + gated fusion')
    return parser.parse_args()


def load_data(config):
    """Load and prepare data."""
    drug_dataset = DrugEmbeddingDataset(config.drug_embeddings_csv)
    ic50_dataset = IC50Dataset(
        config.ic50_csv,
        drug_dataset,
        rna_csv_path=config.cellline_rna_csv,
        add_tissue_ids=True
    )

    # Use cell-line-aware split
    ic50_train, ic50_val, ic50_test = split_ic50_dataset_cellline_aware(
        ic50_dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )

    return ic50_test, ic50_dataset


def run_inference(model, test_loader, device):
    """Run inference on test set."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_drug_ids = []
    all_cellline_ids = []

    with torch.no_grad():
        for batch in test_loader:
            drug_embeds = batch['drug_embed'].to(device)
            ic50_targets = batch['ic50'].to(device)
            cellline_ids = batch['cellline_id'].to(device)

            cancer_type_ids = batch.get('cancer_type_id')
            if cancer_type_ids is not None:
                cancer_type_ids = cancer_type_ids.to(device)

            tissue_ids_batch = batch.get('tissue_id')
            if tissue_ids_batch is not None:
                tissue_ids_batch = tissue_ids_batch.to(device)

            cellline_rna_embeds = batch.get('rna_embed')
            if cellline_rna_embeds is not None:
                cellline_rna_embeds = cellline_rna_embeds.to(device)

            rna_available = batch.get('rna_available')
            if rna_available is not None:
                rna_available = rna_available.to(device)

            outputs = model(
                drug_embeds=drug_embeds,
                cellline_ids=cellline_ids,
                cancer_type_ids=cancer_type_ids,
                tissue_ids=tissue_ids_batch,
                cellline_rna_embeds=cellline_rna_embeds,
                rna_available=rna_available
            )

            all_predictions.extend(outputs['ic50_pred'].cpu().numpy())
            all_targets.extend(ic50_targets.cpu().numpy())
            all_cellline_ids.extend(cellline_ids.cpu().numpy())

    return np.array(all_predictions), np.array(all_targets), np.array(all_cellline_ids)


def compute_detailed_metrics(predictions, targets):
    """Compute detailed metrics."""
    from scipy import stats

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(predictions, targets)
    spearman_r, spearman_p = stats.spearmanr(predictions, targets)

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Percentile errors
    percentiles = [50, 75, 90, 95]
    errors = np.abs(predictions - targets)
    percentile_errors = {f'p{p}': np.percentile(errors, p) for p in percentiles}

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'r2': float(r2),
        'percentile_errors': percentile_errors,
    }


def main():
    args = parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load config
    config = GastroTransformerConfig()
    config.drug_embeddings_csv = args.drug_embeddings_csv
    config.ic50_csv = args.ic50_csv
    config.cellline_rna_csv = args.cellline_rna_csv
    config.device = args.device

    print("Loading data...")
    ic50_test, ic50_dataset = load_data(config)

    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        ic50_test,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, weights_only=False)

    # Load config from checkpoint if available, otherwise use default
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        # Update config with checkpoint values (checkpoint_config can be dict or object)
        if isinstance(checkpoint_config, dict):
            for key, value in checkpoint_config.items():
                if hasattr(config, key):
                    try:
                        setattr(config, key, value)
                    except:
                        pass
        else:
            for key in dir(checkpoint_config):
                if not key.startswith('_') and hasattr(config, key):
                    try:
                        setattr(config, key, getattr(checkpoint_config, key))
                    except:
                        pass
        print(f"Loaded config from checkpoint: use_qformer={config.use_qformer}")

    # Create model based on version
    if args.model_version == 'v21':
        print("Using model_v21: Attention pool + gated fusion IC50 head")
        model = ModalitySlotQFormerV21(config)
    else:
        model = ModalitySlotQFormer(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(args.device)

    print(f"Model parameters: {model.count_parameters()}")

    print("Running inference...")
    predictions, targets, cellline_ids = run_inference(model, test_loader, args.device)

    print("Computing metrics...")
    metrics = compute_detailed_metrics(predictions, targets)

    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"Pearson R:  {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"Spearman R: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.2e})")
    print("\nPercentile Errors:")
    for k, v in metrics['percentile_errors'].items():
        print(f"  {k}: {v:.4f}")

    # Save predictions
    results_df = pd.DataFrame({
        'cellline_id': cellline_ids,
        'actual_ic50': targets,
        'predicted_ic50': predictions,
        'error': np.abs(predictions - targets)
    })
    results_df.to_csv(f'{args.output_dir}/predictions.csv', index=False)

    # Save metrics
    import json
    with open(f'{args.output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")

    return metrics


if __name__ == '__main__':
    main()
