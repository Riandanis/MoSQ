#!/usr/bin/env python3
"""
Main Entry Point for Gastro-Transformer v2 Training.

Usage:
    # Demo with synthetic data
    python scripts/main.py --mode demo

    # Full training pipeline
    python scripts/main.py --mode train \
        --paired_image_csv data/paired_image_ms-bcpp.csv \
        --paired_rna_csv data/paired_rna_ms-bcpp.csv \
        --unpaired_image_csv data/unpaired_image.csv \
        --unpaired_rna_csv data/unpaired_rna.csv \
        --drug_embeddings_csv data/drug_embeddings.csv \
        --ic50_csv data/ic50_data.csv \
        --cellline_rna_csv data/ccle_rna_for_ic50.csv

    # Individual training stages
    python scripts/main.py --mode pretrain
    python scripts/main.py --mode finetune --checkpoint saved_checkpoints/pretrained_dra.pt

    # Evaluate
    python scripts/main.py --mode evaluate --checkpoint saved_checkpoints/pretrained_dra.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gastro_transformer.config import GastroTransformerConfig
from gastro_transformer.model import ModalitySlotQFormer
from gastro_transformer.model_v21 import ModalitySlotQFormer as ModalitySlotQFormerV21
from gastro_transformer.train import GastroTransformerTrainer, evaluate_ncd


def create_model(config, model_version='v2'):
    """Create model based on version."""
    if model_version == 'v21':
        logger.info("Using model_v21: Attention pool + gated fusion IC50 head")
        return ModalitySlotQFormerV21(config)
    else:
        return ModalitySlotQFormer(config)
from gastro_transformer.data import (
    PairedMultiModalDataset,
    UnpairedModalityDataset,
    DrugEmbeddingDataset,
    IC50Dataset,
    create_data_loaders,
    split_ic50_dataset_cellline_aware,
)
from gastro_transformer.utils import load_checkpoint

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Gastro-Transformer v2: Modality-Slot Q-Former'
    )

    # Mode
    parser.add_argument(
        '--mode', type=str, default='demo',
        choices=['demo', 'train', 'pretrain', 'pretrain_dcl', 'finetune', 'evaluate', 'evaluate_ncd', 'analyze'],
        help='Training mode: pretrain_dcl = Drug-CellLine Contrastive Pretraining'
    )

    # Drug-CellLine Pretraining specific
    parser.add_argument('--dcl_pretrain_epochs', type=int, default=10,
                        help='Epochs for drug-cellline contrastive pretraining')
    parser.add_argument('--dcl_temperature', type=float, default=0.1,
                        help='Temperature for drug-cellline contrastive loss')
    parser.add_argument('--lambda_dcl', type=float, default=1.0,
                        help='Weight for drug-cellline contrastive loss')
    parser.add_argument('--use_masked_modality_pred', action='store_true',
                        help='Use masked modality prediction loss')
    parser.add_argument('--lambda_mmp', type=float, default=0.5,
                        help='Weight for masked modality prediction loss')

    # Config
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config JSON file'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Use fast mode preset (fewer epochs, smaller model)'
    )

    # Data paths
    parser.add_argument('--paired_image_csv', type=str, help='Paired image embeddings CSV')
    parser.add_argument('--paired_rna_csv', type=str, help='Paired RNA embeddings CSV')
    parser.add_argument('--unpaired_image_csv', type=str, help='Unpaired image embeddings CSV')
    parser.add_argument('--unpaired_rna_csv', type=str, help='Unpaired RNA embeddings CSV')
    parser.add_argument('--drug_embeddings_csv', type=str, help='Drug embeddings CSV')
    parser.add_argument('--ic50_csv', type=str, help='IC50 data CSV')
    parser.add_argument('--cellline_rna_csv', type=str, default=None,
                        help='Cell-line RNA embeddings CSV')

    # Training parameters
    parser.add_argument('--pretrain_epochs', type=int, help='Pre-training epochs')
    parser.add_argument('--finetune_epochs', type=int, help='Fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Base learning rate')
    parser.add_argument('--qformer_finetune_lr_ratio', type=float, default=0.1,
                        help='Q-Former LR = base LR * this ratio during fine-tuning')
    parser.add_argument('--freeze_qformer', action='store_true',
                        help='Freeze Q-Former during fine-tuning (ablation)')
    parser.add_argument('--freeze_projectors', action='store_true',
                        help='Freeze modality projectors during fine-tuning')
    parser.add_argument('--freeze_type_embeddings', action='store_true',
                        help='Freeze type embeddings during fine-tuning')
    parser.add_argument('--num_query_tokens', type=int, default=32,
                        help='Number of Q-Former query tokens')
    parser.add_argument('--qformer_layers', type=int, default=None,
                        help='Number of Q-Former layers (default: config default)')
    parser.add_argument('--use_qformer', type=lambda x: x.lower() == 'true', default=True,
                        help='Use Q-Former for fusion (False = simple concatenation baseline)')
    parser.add_argument('--use_ic50_skip_connection', action='store_true',
                        help='Use skip connection in IC50 head: concat fused + projected drug + cellline')
    parser.add_argument('--use_ic50_attn_pool', action='store_true',
                        help='Use attention pooling + gated residual fusion for IC50 head (v2.1 architecture)')
    parser.add_argument('--use_feature_cellline_encoder', action='store_true',
                        help='Use feature-based cell encoder (RNA+cancer+tissue) instead of ID embeddings')
    parser.add_argument('--use_multitoken_cellline', action='store_true',
                        help='Decompose cell-line into 3 separate typed Q-Former tokens (cancer+tissue+rna)')

    # Training improvements
    parser.add_argument('--use_huber_loss', action='store_true',
                        help='Use Huber loss instead of MSE for IC50 (robust to outliers)')
    parser.add_argument('--huber_delta', type=float, default=1.5,
                        help='Huber loss delta threshold')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use Exponential Moving Average of model weights')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay rate')
    parser.add_argument('--use_multitask_finetune', action='store_true',
                        help='Add tissue classification auxiliary loss during fine-tuning')
    parser.add_argument('--lambda_tissue_finetune', type=float, default=0.1,
                        help='Weight for auxiliary tissue classification loss')
    parser.add_argument('--use_rdrop', action='store_true',
                        help='Use R-Drop consistency regularization')
    parser.add_argument('--rdrop_alpha', type=float, default=0.5,
                        help='Weight for R-Drop KL divergence loss')
    parser.add_argument('--model_version', type=str, default='v2',
                        choices=['v2', 'v21'],
                        help='Model version: v2 = standard, v21 = attention pool + gated fusion')

    # Data splitting
    parser.add_argument('--split_ic50', type=bool, default=True,
                        help='Use cell-line-aware split (default True)')
    parser.add_argument('--no_split_ic50', action='store_true',
                        help='Disable cell-line-aware split')

    # Model dimensions
    parser.add_argument('--image_dim', type=int, default=512, help='Image embedding dimension')
    parser.add_argument('--rna_dim', type=int, default=256, help='RNA embedding dimension')
    parser.add_argument('--drug_dim', type=int, default=768, help='Drug embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension')

    # Checkpointing
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')

    # NCD evaluation
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for NCD CV')
    parser.add_argument('--finetune_lr', type=float, default=1e-4, help='Fine-tuning learning rate for NCD')
    parser.add_argument('--output_dir', type=str, default='reports/ncd_evaluation', help='Output directory for NCD results')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda/cpu)')
    parser.add_argument('--no_mixed_precision', action='store_true', help='Disable mixed precision')

    # Experiment tracking
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='gastro-transformer-v2')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')

    return parser.parse_args()


def create_config_from_args(args) -> GastroTransformerConfig:
    """Create config from command line arguments."""
    if args.fast:
        config = GastroTransformerConfig.fast_mode()
        logger.info("Using fast mode preset")
    elif args.config:
        config = GastroTransformerConfig.load(args.config)
    else:
        config = GastroTransformerConfig()

    # Override with CLI arguments
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.qformer_finetune_lr_ratio:
        config.qformer_finetune_lr_ratio = args.qformer_finetune_lr_ratio
    if args.freeze_qformer:
        config.freeze_qformer_in_finetune = True
    if args.freeze_projectors:
        config.freeze_projectors_in_finetune = True
    if args.freeze_type_embeddings:
        config.freeze_type_embeddings_in_finetune = True
    if args.num_query_tokens:
        config.num_query_tokens = args.num_query_tokens
    if args.qformer_layers is not None:
        config.qformer_layers = args.qformer_layers
    if args.use_qformer is not None:
        config.use_qformer = args.use_qformer
    if args.use_ic50_skip_connection:
        config.use_ic50_skip_connection = True
    if args.use_ic50_attn_pool:
        config.use_ic50_attn_pool = True
    if args.use_feature_cellline_encoder:
        config.use_feature_cellline_encoder = True
    if args.use_multitoken_cellline:
        config.use_multitoken_cellline = True

    # Training improvements
    if args.use_huber_loss:
        config.use_huber_loss = True
        config.huber_delta = args.huber_delta
    if args.use_ema:
        config.use_ema = True
        config.ema_decay = args.ema_decay
    if args.use_multitask_finetune:
        config.use_multitask_finetune = True
        config.lambda_tissue_finetune = args.lambda_tissue_finetune
    if args.use_rdrop:
        config.use_rdrop = True
        config.rdrop_alpha = args.rdrop_alpha

    # Drug-CellLine Pretraining config
    if args.dcl_pretrain_epochs:
        config.dcl_pretrain_epochs = args.dcl_pretrain_epochs
    if args.dcl_temperature:
        config.dcl_temperature = args.dcl_temperature
    if args.lambda_dcl:
        config.lambda_dcl = args.lambda_dcl
    if args.use_masked_modality_pred:
        config.use_masked_modality_pred = True
    if args.lambda_mmp:
        config.lambda_mmp = args.lambda_mmp

    # Data paths
    if args.paired_image_csv:
        config.paired_image_csv = args.paired_image_csv
    if args.paired_rna_csv:
        config.paired_rna_csv = args.paired_rna_csv
    if args.unpaired_image_csv:
        config.unpaired_image_csv = args.unpaired_image_csv
    if args.unpaired_rna_csv:
        config.unpaired_rna_csv = args.unpaired_rna_csv
    if args.drug_embeddings_csv:
        config.drug_embeddings_csv = args.drug_embeddings_csv
    if args.ic50_csv:
        config.ic50_csv = args.ic50_csv
    if args.cellline_rna_csv:
        config.cellline_rna_csv = args.cellline_rna_csv

    # Epochs
    if args.pretrain_epochs:
        config.pretrain_epochs = args.pretrain_epochs
    if args.finetune_epochs:
        config.finetune_epochs = args.finetune_epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    # Model dimensions
    if args.image_dim:
        config.image_dim = args.image_dim
    if args.rna_dim:
        config.rna_dim = args.rna_dim
    if args.drug_dim:
        config.drug_dim = args.drug_dim
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim

    # Output dirs
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir

    # Device
    config.device = args.device
    if args.no_mixed_precision:
        config.mixed_precision = False

    # Data splitting
    if args.no_split_ic50:
        config.split_ic50 = False

    # Experiment tracking
    config.use_wandb = args.use_wandb
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    # Ensure output directories exist (after CLI overrides)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    return config


def create_synthetic_data(config: GastroTransformerConfig):
    """Create synthetic data for demo mode."""
    logger.info("Creating synthetic data for demo...")

    n_paired = 20
    n_unpaired_img = 100
    n_unpaired_rna = 100
    n_ic50 = 500

    # Synthetic paired data
    paired_data = {
        'image': np.random.randn(n_paired, config.image_dim).astype(np.float32),
        'rna': np.random.randn(n_paired, config.rna_dim).astype(np.float32),
        'tissue_label': np.random.randint(0, config.num_tissue_types, n_paired)
    }

    # Synthetic unpaired data
    unpaired_image_data = {
        'embeddings': np.random.randn(n_unpaired_img, config.image_dim).astype(np.float32),
        'tissue_label': np.random.randint(0, config.num_tissue_types, n_unpaired_img)
    }
    unpaired_rna_data = {
        'embeddings': np.random.randn(n_unpaired_rna, config.rna_dim).astype(np.float32),
        'tissue_label': np.random.randint(0, config.num_tissue_types, n_unpaired_rna)
    }

    # Synthetic IC50 data
    ic50_data = {
        'cellline_id': np.random.randint(0, 100, n_ic50),
        'drug_id': np.random.randint(0, 20, n_ic50),
        'ic50': np.random.randn(n_ic50).astype(np.float32),
        'cancer_type': np.random.randint(0, config.num_cancer_types, n_ic50),
        'tissue_type': np.random.randint(0, config.num_tissue_types, n_ic50)
    }

    # Synthetic drug embeddings
    drug_embeddings = np.random.randn(20, config.drug_dim).astype(np.float32)

    return {
        'paired': paired_data,
        'unpaired_image': unpaired_image_data,
        'unpaired_rna': unpaired_rna_data,
        'ic50': ic50_data,
        'drug_embeddings': drug_embeddings
    }


def run_demo(config: GastroTransformerConfig, model_version: str = 'v2'):
    """Run demo with synthetic data."""
    logger.info("=" * 60)
    logger.info("DEMO MODE: Testing pipeline with synthetic data")
    logger.info("=" * 60)

    # Use CPU for demo
    config.device = 'cpu'
    config.mixed_precision = False

    # Create model
    model = create_model(config, model_version)
    logger.info(f"Model parameters: {model.count_parameters()}")

    # Create synthetic data loaders
    synthetic = create_synthetic_data(config)

    from torch.utils.data import Dataset, DataLoader

    # Custom dataset that returns dictionaries
    class DictDataset(Dataset):
        def __init__(self, data_dict):
            self.data = data_dict
            self.length = len(next(iter(data_dict.values())))

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    # Paired dataset (returns dict with keys)
    paired_data = {
        'image_embed': torch.from_numpy(synthetic['paired']['image']),
        'rna_embed': torch.from_numpy(synthetic['paired']['rna']),
        'tissue_label': torch.from_numpy(synthetic['paired']['tissue_label']).long()
    }
    paired_dataset = DictDataset(paired_data)

    # IC50 dataset
    ic50_data = {
        'cellline_id': torch.from_numpy(synthetic['ic50']['cellline_id']).long(),
        'drug_embed': torch.from_numpy(synthetic['drug_embeddings'])[synthetic['ic50']['drug_id']],
        'ic50': torch.from_numpy(synthetic['ic50']['ic50']),
        'cancer_type_id': torch.from_numpy(synthetic['ic50']['cancer_type']).long(),
        'tissue_id': torch.from_numpy(synthetic['ic50']['tissue_type']).long(),
    }
    # Add synthetic RNA data for feature-based cell encoder
    if config.use_feature_cellline_encoder:
        n_ic50 = len(synthetic['ic50']['ic50'])
        ic50_data['rna_embed'] = torch.randn(n_ic50, config.rna_dim)
        ic50_data['rna_available'] = torch.ones(n_ic50, dtype=torch.bool)
    ic50_dataset = DictDataset(ic50_data)

    # Data loaders
    data_loaders = {
        'paired': DataLoader(paired_dataset, batch_size=config.batch_size, shuffle=True),
        'unpaired_image': DataLoader(paired_dataset, batch_size=config.batch_size, shuffle=True),
        'unpaired_rna': DataLoader(paired_dataset, batch_size=config.batch_size, shuffle=True),
        'ic50': DataLoader(ic50_dataset, batch_size=config.batch_size, shuffle=True),
    }

    # Create trainer
    trainer = GastroTransformerTrainer(
        model=model,
        config=config,
        data_loaders=data_loaders,
        device=config.device,
        paired_dataset=paired_dataset
    )

    # Quick training test
    logger.info("Testing pre-training (1 epoch)...")
    try:
        trainer.pretrain(epochs=1, log_every=5)
        logger.info("Pre-training test passed!")
    except Exception as e:
        logger.error(f"Pre-training failed: {e}")
        raise

    logger.info("Testing IC50 fine-tuning (1 epoch)...")
    try:
        trainer.finetune_ic50(epochs=1, log_every=5)
        logger.info("IC50 fine-tuning test passed!")
    except Exception as e:
        logger.error(f"IC50 fine-tuning failed: {e}")
        raise

    logger.info("=" * 60)
    logger.info("DEMO COMPLETE - All tests passed!")
    logger.info("=" * 60)


def load_real_data(config: GastroTransformerConfig):
    """Load real data from CSV files."""
    logger.info("Loading data from CSV files...")

    datasets = {}

    # Paired datasets
    if config.paired_image_csv and config.paired_rna_csv:
        datasets['paired'] = PairedMultiModalDataset(
            config.paired_image_csv,
            config.paired_rna_csv
        )
        logger.info(f"Loaded paired dataset: {len(datasets['paired'])} samples")

    # Unpaired datasets
    if config.unpaired_image_csv:
        datasets['unpaired_image'] = UnpairedModalityDataset(
            config.unpaired_image_csv,
            modality='image',
            embedding_dim=config.image_dim
        )
        logger.info(f"Loaded unpaired image dataset: {len(datasets['unpaired_image'])} samples")

    if config.unpaired_rna_csv:
        datasets['unpaired_rna'] = UnpairedModalityDataset(
            config.unpaired_rna_csv,
            modality='rna',
            embedding_dim=config.rna_dim
        )
        logger.info(f"Loaded unpaired RNA dataset: {len(datasets['unpaired_rna'])} samples")

    # IC50 dataset
    if config.ic50_csv and config.drug_embeddings_csv:
        drug_dataset = DrugEmbeddingDataset(config.drug_embeddings_csv)
        ic50_dataset = IC50Dataset(
            config.ic50_csv,
            drug_dataset,
            rna_csv_path=config.cellline_rna_csv,
            add_tissue_ids=True
        )
        logger.info(f"Loaded IC50 dataset: {len(ic50_dataset)} samples")

        # Cell-line aware split
        if config.split_ic50:
            ic50_train, ic50_val, ic50_test = split_ic50_dataset_cellline_aware(
                ic50_dataset,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42
            )
            datasets['ic50_train'] = ic50_train
            datasets['ic50_val'] = ic50_val
            datasets['ic50_test'] = ic50_test
            logger.info(f"IC50 split: train={len(ic50_train)}, val={len(ic50_val)}, test={len(ic50_test)}")
        else:
            datasets['ic50'] = ic50_dataset

    return datasets


def run_full_training(config: GastroTransformerConfig, model_version: str = 'v2'):
    """Run full training pipeline."""
    logger.info("=" * 60)
    logger.info("FULL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load data
    datasets = load_real_data(config)

    # Create data loaders
    # If IC50 data is already split (when split_ic50=True in load_real_data),
    # use the pre-split datasets and don't split again
    ic50_already_split = 'ic50_train' in datasets and 'ic50_val' in datasets and 'ic50_test' in datasets

    if ic50_already_split:
        # Use pre-split IC50 datasets
        data_loaders = create_data_loaders(
            config,
            paired_dataset=datasets.get('paired'),
            unpaired_image_dataset=datasets.get('unpaired_image'),
            unpaired_rna_dataset=datasets.get('unpaired_rna'),
            ic50_train=datasets.get('ic50_train'),
            ic50_val=datasets.get('ic50_val'),
            ic50_test=datasets.get('ic50_test'),
            split_ic50=False,  # Already split
        )
    else:
        data_loaders = create_data_loaders(
            config,
            paired_dataset=datasets.get('paired'),
            unpaired_image_dataset=datasets.get('unpaired_image'),
            unpaired_rna_dataset=datasets.get('unpaired_rna'),
            ic50_dataset=datasets.get('ic50'),
            split_ic50=config.split_ic50,
            ic50_split_seed=42
        )

    # Create model
    model = create_model(config, model_version)
    logger.info(f"Model parameters: {model.count_parameters()}")

    # Create trainer
    trainer = GastroTransformerTrainer(
        model=model,
        config=config,
        data_loaders=data_loaders,
        device=config.device,
        paired_dataset=datasets.get('paired')
    )

    # Pre-training
    if config.pretrain_epochs > 0:
        logger.info("Starting pre-training...")
        trainer.pretrain(epochs=config.pretrain_epochs)

        # Save pretrained checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
        }, f"{config.checkpoint_dir}/pretrained.pt")
        logger.info(f"Saved pretrained model to {config.checkpoint_dir}/pretrained.pt")

    # IC50 fine-tuning
    if config.finetune_epochs > 0:
        logger.info("Starting IC50 fine-tuning...")
        trainer.finetune_ic50(epochs=config.finetune_epochs)
        logger.info(f"Saved best model to {config.checkpoint_dir}/best_ic50_model.pt")

    logger.info("Training complete!")


def main():
    """Main entry point."""
    args = parse_args()
    config = create_config_from_args(args)

    logger.info(f"Running in mode: {args.mode}")
    logger.info(f"Config: {config}")

    if args.mode == 'demo':
        run_demo(config, args.model_version)
    elif args.mode == 'train':
        run_full_training(config, args.model_version)
    elif args.mode == 'pretrain':
        datasets = load_real_data(config)
        # Handle pre-split IC50 datasets
        ic50_already_split = 'ic50_train' in datasets and 'ic50_val' in datasets and 'ic50_test' in datasets

        if ic50_already_split:
            data_loaders = create_data_loaders(
                config,
                paired_dataset=datasets.get('paired'),
                unpaired_image_dataset=datasets.get('unpaired_image'),
                unpaired_rna_dataset=datasets.get('unpaired_rna'),
                ic50_train=datasets.get('ic50_train'),
                ic50_val=datasets.get('ic50_val'),
                ic50_test=datasets.get('ic50_test'),
            )
        else:
            data_loaders = create_data_loaders(
                config,
                paired_dataset=datasets.get('paired'),
                unpaired_image_dataset=datasets.get('unpaired_image'),
                unpaired_rna_dataset=datasets.get('unpaired_rna'),
                ic50_dataset=datasets.get('ic50'),
            )
        model = create_model(config, args.model_version)
        trainer = GastroTransformerTrainer(model, config, data_loaders, config.device)
        trainer.pretrain(epochs=config.pretrain_epochs)
    elif args.mode == 'pretrain_dcl':
        """Drug-CellLine Contrastive Pretraining"""
        if not args.checkpoint:
            logger.error("Checkpoint required for DCL pretraining (pretrained model)")
            return

        datasets = load_real_data(config)

        # Handle pre-split IC50 datasets
        ic50_already_split = 'ic50_train' in datasets and 'ic50_val' in datasets and 'ic50_test' in datasets

        if ic50_already_split:
            data_loaders = create_data_loaders(
                config,
                paired_dataset=datasets.get('paired'),
                unpaired_image_dataset=datasets.get('unpaired_image'),
                unpaired_rna_dataset=datasets.get('unpaired_rna'),
                ic50_train=datasets.get('ic50_train'),
                ic50_val=datasets.get('ic50_val'),
                ic50_test=datasets.get('ic50_test'),
                split_ic50=False,
            )
        else:
            data_loaders = create_data_loaders(
                config,
                paired_dataset=datasets.get('paired'),
                unpaired_image_dataset=datasets.get('unpaired_image'),
                unpaired_rna_dataset=datasets.get('unpaired_rna'),
                ic50_dataset=datasets.get('ic50'),
            )

        model = create_model(config, args.model_version)

        # Load pretrained weights
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        trainer = GastroTransformerTrainer(model, config, data_loaders, config.device)
        trainer.pretrain_drug_cellline(epochs=config.dcl_pretrain_epochs)

    elif args.mode == 'finetune':
        if not args.checkpoint:
            logger.error("Checkpoint required for fine-tuning")
            return
        datasets = load_real_data(config)

        # Handle pre-split IC50 datasets (same fix as in run_full_training)
        ic50_already_split = 'ic50_train' in datasets and 'ic50_val' in datasets and 'ic50_test' in datasets

        if ic50_already_split:
            data_loaders = create_data_loaders(
                config,
                paired_dataset=datasets.get('paired'),
                unpaired_image_dataset=datasets.get('unpaired_image'),
                unpaired_rna_dataset=datasets.get('unpaired_rna'),
                ic50_train=datasets.get('ic50_train'),
                ic50_val=datasets.get('ic50_val'),
                ic50_test=datasets.get('ic50_test'),
                split_ic50=False,
            )
        else:
            data_loaders = create_data_loaders(
                config,
                paired_dataset=datasets.get('paired'),
                unpaired_image_dataset=datasets.get('unpaired_image'),
                unpaired_rna_dataset=datasets.get('unpaired_rna'),
                ic50_dataset=datasets.get('ic50'),
                split_ic50=config.split_ic50,
            )

        model = create_model(config, args.model_version)
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        trainer = GastroTransformerTrainer(model, config, data_loaders, config.device)
        trainer.finetune_ic50(epochs=config.finetune_epochs)
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            logger.error("Checkpoint required for evaluation")
            return
        logger.info("Evaluation not yet implemented")
    elif args.mode == 'evaluate_ncd':
        if not args.checkpoint:
            logger.error("Checkpoint required for NCD evaluation")
            return
        if not args.ic50_csv:
            logger.error("ic50_csv required for NCD evaluation")
            return
        if not args.drug_embeddings_csv:
            logger.error("drug_embeddings_csv required for NCD evaluation")
            return
        if not args.cellline_rna_csv:
            logger.error("cellline_rna_csv required for NCD evaluation")
            return

        logger.info("Starting NCD (No Common Drug) evaluation...")

        # Run NCD evaluation
        results = evaluate_ncd(
            checkpoint_path=args.checkpoint,
            config=config,
            ic50_csv=args.ic50_csv,
            drug_embeddings_csv=args.drug_embeddings_csv,
            cellline_rna_csv=args.cellline_rna_csv,
            n_folds=args.n_folds,
            finetune_epochs=args.finetune_epochs,
            device=args.device,
            finetune_lr=args.finetune_lr,
            huber_delta=config.huber_delta,
            early_stopping_patience=3,
            output_dir=args.output_dir
        )

        logger.info("NCD evaluation complete!")
    elif args.mode == 'analyze':
        logger.info("Analysis mode not yet implemented")
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
