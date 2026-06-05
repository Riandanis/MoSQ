"""
Configuration for Gastro-Transformer v2: Modality-Slot Q-Former.

This module contains the configuration dataclass for the Gastro-Transformer v2 model,
which uses a modality-slot architecture with typed token embeddings and Q-Former fusion.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import json
from pathlib import Path


@dataclass
class GastroTransformerConfig:
    """Configuration for Gastro-Transformer v2 model and training."""

    # ==========================================================================
    # Encoder dimensions (from frozen pre-trained models)
    # ==========================================================================
    image_dim: int = 512          # Image embeddings dimension (e.g., from CONCH)
    rna_dim: int = 256            # RNA embeddings dimension (e.g., from BulkRNABERT)
    drug_dim: int = 768           # Drug embeddings dimension (e.g., from ChemBERTa)
    endo_dim: int = 0             # Endoscopy embeddings dimension (0 = disabled)

    # ==========================================================================
    # Model architecture dimensions
    # ==========================================================================
    hidden_dim: int = 768         # Common projection dimension for all modalities
    num_query_tokens: int = 48    # Q-Former learnable query tokens (increased from 32)
    qformer_layers: int = 8       # Number of Q-Former transformer blocks (increased from 6)
    qformer_heads: int = 12       # Number of attention heads in Q-Former (unchanged)
    dropout: float = 0.1          # Dropout rate

    # ==========================================================================
    # Modality configuration
    # ==========================================================================
    num_modality_types: int = 4   # Number of modality types (image, rna, drug, cellline)

    # ==========================================================================
    # Task-specific configuration
    # ==========================================================================
    num_tissue_types: int = 26    # Number of tissue type labels
    num_cancer_types: int = 30     # Cancer subtype classification
    num_drug_classes: int = 10     # Drug mechanism of action (MoA) categories

    # ==========================================================================
    # Cell-line configuration (for IC50 prediction)
    # ==========================================================================
    num_cell_lines: int = 1200    # Number of unique cell-lines in IC50 dataset
    use_cellline_embeddings: bool = True
    use_feature_cellline_encoder: bool = False  # Use feature-based cell encoder (RNA+cancer+tissue) instead of ID embeddings

    # ==========================================================================
    # Training configuration
    # ==========================================================================
    pretrain_epochs: int = 100     # Pre-training epochs (Stage 1 & 2)
    finetune_epochs: int = 50      # Fine-tuning epochs (Stage 3)
    learning_rate: float = 1e-4    # Base learning rate
    batch_size: int = 256          # Batch size for training
    num_workers: int = 24          # DataLoader workers
    prefetch_factor: int = 4       # Prefetch batches per worker
    persistent_workers: bool = True  # Keep workers alive between epochs
    gradient_accumulation_steps: int = 1

    # Fine-tuning specific
    qformer_finetune_lr_ratio: float = 0.1  # Q-Former LR = this × base LR
    freeze_qformer_in_finetune: bool = False  # Freeze Q-Former during IC50 fine-tuning
    freeze_projectors_in_finetune: bool = False  # Freeze modality projectors during fine-tuning
    freeze_type_embeddings_in_finetune: bool = False  # Freeze type embeddings during fine-tuning
    use_qformer: bool = True  # Use Q-Former for fusion (False = simple concatenation)
    use_qformer_for_ic50: bool = True  # Q-detached: False = bypass Q-Former for IC50 (use drug + cellline directly)
    use_ic50_skip_connection: bool = False  # Add skip connection: concat fused + projected drug + cellline
    use_ic50_attn_pool: bool = False  # IC50 attention pooling + gated residual fusion (v2.1 architecture)
    use_multitoken_cellline: bool = False  # Decompose cell-line into 3 separate typed Q-Former tokens (cancer+tissue+rna) instead of 1 fused token

    # Drug-CellLine Contrastive Pretraining (Phase 1b)
    dcl_pretrain_epochs: int = 10  # Number of epochs for drug-cellline contrastive pretraining
    dcl_temperature: float = 0.1  # Temperature for contrastive loss
    lambda_dcl: float = 1.0  # Weight for drug-cellline contrastive loss
    use_masked_modality_pred: bool = False  # Use masked modality prediction loss
    lambda_mmp: float = 0.5  # Weight for masked modality prediction loss

    # Fine-tuning loss and regularization
    use_huber_loss: bool = False       # Huber loss instead of MSE (robust to IC50 outliers)
    huber_delta: float = 1.5           # Huber loss threshold (linear beyond this)
    use_ema: bool = False              # Exponential Moving Average of model weights
    ema_decay: float = 0.999           # EMA decay rate (higher = smoother)
    use_multitask_finetune: bool = False  # Tissue classification auxiliary loss during IC50 fine-tuning
    lambda_tissue_finetune: float = 0.1   # Weight for auxiliary tissue loss
    use_rdrop: bool = False            # R-Drop consistency regularization
    rdrop_alpha: float = 0.5           # Weight for R-Drop KL divergence loss

    # Optimizer settings
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # ==========================================================================
    # Loss weights
    # ==========================================================================
    lambda_intra: float = 1.0     # Intra-modal contrastive loss weight
    lambda_cross: float = 2.0     # Cross-modal contrastive (paired data)
    lambda_proto: float = 0.5     # Prototypical alignment loss weight
    lambda_recon: float = 0.1     # Reconstruction loss weight
    lambda_ortho: float = 0.05    # Orthogonality loss weight (disabled in code)

    unpaired_loss_weight: float = 0.5  # Scale factor for unpaired modality losses

    # IC50 data scale configuration
    ic50_already_log_transformed: bool = True  # GDSC LN_IC50 is pre-transformed

    # Early stopping
    early_stopping_patience: int = 10

    # Contrastive loss temperature
    temperature: float = 0.07

    # Data splitting: Cell-line-aware train/val/test split
    split_ic50: bool = True  # Default to True for proper evaluation

    # ==========================================================================
    # Data paths
    # ==========================================================================
    paired_image_csv: Optional[str] = None
    paired_rna_csv: Optional[str] = None
    unpaired_image_csv: Optional[str] = None
    unpaired_rna_csv: Optional[str] = None
    drug_embeddings_csv: Optional[str] = None
    ic50_csv: Optional[str] = None
    cellline_rna_csv: Optional[str] = None
    cellline_metadata_csv: Optional[str] = None

    # ==========================================================================
    # Output paths
    # ==========================================================================
    checkpoint_dir: str = "checkpoints_save"
    log_dir: str = "logs"

    # ==========================================================================
    # Experiment tracking
    # ==========================================================================
    use_wandb: bool = False
    wandb_project: str = "gastro-transformer-v2"
    wandb_entity: Optional[str] = None
    experiment_name: Optional[str] = None

    # ==========================================================================
    # Device configuration
    # ==========================================================================
    device: str = "cuda:1"
    mixed_precision: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Create output directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Validate dimensions
        assert self.hidden_dim % self.qformer_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by qformer_heads ({self.qformer_heads})"

        # Validate tissue types
        try:
            from .utils import TISSUE_TYPES
            if self.num_tissue_types != len(TISSUE_TYPES):
                import warnings
                warnings.warn(
                    f"config.num_tissue_types={self.num_tissue_types} but "
                    f"len(TISSUE_TYPES)={len(TISSUE_TYPES)} in utils.py",
                    UserWarning
                )
        except ImportError:
            pass

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> 'GastroTransformerConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    @classmethod
    def fast_mode(cls) -> 'GastroTransformerConfig':
        """Create a fast-training configuration for quick experimentation."""
        return cls(
            pretrain_epochs=50,
            finetune_epochs=25,
            batch_size=64,
            num_workers=4,
            hidden_dim=512,
            qformer_layers=4,
            qformer_heads=8,
            num_query_tokens=16,
        )

    @classmethod
    def default_mode(cls) -> 'GastroTransformerConfig':
        """Create default configuration for production training."""
        return cls()

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()
