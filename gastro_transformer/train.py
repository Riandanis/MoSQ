"""
Training Pipeline for Gastro-Transformer v2: Modality-Slot Q-Former.

Implements the staged training approach:
    Stage 1 & 2: Pre-training with intra-modal and cross-modal losses
    Stage 3: Fine-tuning for downstream tasks (IC50 prediction)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, SequentialLR, LinearLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from pathlib import Path
from tqdm import tqdm
import logging
import json
import warnings
from collections import defaultdict

from .config import GastroTransformerConfig
from .model import ModalitySlotQFormer
from .losses import GastroTransformerLoss, compute_ic50_metrics, DrugCellLineContrastiveLoss, MaskedModalityPredictionLoss
from .utils import save_checkpoint, load_checkpoint

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of weights: ema_param = decay * ema_param + (1 - decay) * param.
    Use context manager `ema.apply(model)` to temporarily swap in EMA weights for evaluation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Context manager: swap in EMA weights, then restore originals."""
        return _EMAContext(self, model)


class _EMAContext:
    """Context manager for temporarily applying EMA weights."""

    def __init__(self, ema: EMAModel, model: nn.Module):
        self.ema = ema
        self.model = model

    def __enter__(self):
        self.ema.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.ema.shadow:
                self.ema.backup[name] = param.data.clone()
                param.data.copy_(self.ema.shadow[name])
        return self.model

    def __exit__(self, *args):
        for name, param in self.model.named_parameters():
            if name in self.ema.backup:
                param.data.copy_(self.ema.backup[name])
        self.ema.backup = {}


class GastroTransformerTrainer:
    """
    Trainer for Gastro-Transformer v2 with staged training pipeline.

    Training Stages:
    1. Pre-training (Stage 1 & 2): Learn intra-modal and cross-modal representations
       - Uses paired data (limited) for cross-modal alignment
       - Uses unpaired data (large-scale) for intra-modal learning
       - Prototypical alignment extends paired learning to unpaired data

    2. Fine-tuning (Stage 3): IC50 prediction with differential learning rates
       - Q-Former uses lower LR (0.1× base LR) to preserve pretrained knowledge
       - IC50 head and CellLine encoder use full LR for new task
    """

    def __init__(
        self,
        model: ModalitySlotQFormer,
        config: GastroTransformerConfig,
        data_loaders: Dict[str, DataLoader],
        device: Optional[str] = None,
        paired_dataset=None
    ):
        """
        Args:
            model: ModalitySlotQFormer model
            config: Training configuration
            data_loaders: Dictionary of DataLoaders
            device: Device to train on
            paired_dataset: PairedMultiModalDataset instance (for prototype init)
        """
        self.config = config
        self.data_loaders = data_loaders
        self.paired_dataset = paired_dataset

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.model = model.to(device)

        # Initialize loss function
        self.loss_fn = GastroTransformerLoss(config).to(device)

        # Training state
        self.history = defaultdict(list)
        self.current_epoch = 0

        # Gradient scaler for mixed precision
        self._device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        self.scaler = GradScaler(self._device_type) if config.mixed_precision and self._device_type == 'cuda' else None

        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"AMP (mixed precision) enabled: {self.scaler is not None}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def _get_num_batches(self, stage: str = 'pretrain') -> int:
        """Get number of batches per epoch for a given stage."""
        if stage == 'pretrain':
            # Use the largest unpaired dataset
            for key in ['unpaired_image', 'unpaired_rna']:
                if key in self.data_loaders and self.data_loaders[key] is not None:
                    return len(self.data_loaders[key])
            if 'paired' in self.data_loaders and self.data_loaders['paired'] is not None:
                return len(self.data_loaders['paired'])
            return 100  # Default
        else:
            # IC50 fine-tuning
            for key in ['ic50_train', 'ic50']:
                if key in self.data_loaders and self.data_loaders[key] is not None:
                    return len(self.data_loaders[key])
            return 100

    def _get_infinite_iterator(self, key: str):
        """Get infinite iterator for a data loader."""
        loader = self.data_loaders.get(key)
        if loader is None:
            return iter([])  # Return empty iterator if no loader
        while True:
            for batch in loader:
                yield batch

    def _forward_paired(self, batch: Dict) -> Dict:
        """Forward pass for paired data."""
        image_embeds = batch.get('image_embed')
        rna_embeds = batch.get('rna_embed')

        if image_embeds is None or rna_embeds is None:
            return {'total': torch.tensor(0.0, device=self.device)}

        image_embeds = image_embeds.to(self.device)
        rna_embeds = rna_embeds.to(self.device)
        tissue_labels = batch.get('tissue_label')
        if tissue_labels is not None:
            tissue_labels = tissue_labels.to(self.device)

        outputs = self.model(
            image_embeds=image_embeds,
            rna_embeds=rna_embeds,
            return_embeddings=True
        )

        losses = self.loss_fn(
            outputs,
            tissue_labels=tissue_labels,
            is_paired=True,
            image_prototypes=self.model.image_prototypes,
            rna_prototypes=self.model.rna_prototypes
        )

        return losses

    def _forward_unpaired(self, batch: Dict, modality: str) -> Dict:
        """Forward pass for unpaired data."""
        if modality == 'image':
            embeds = batch.get('image_embed')
        elif modality == 'rna':
            embeds = batch.get('rna_embed')
        else:
            return {'total': torch.tensor(0.0, device=self.device)}

        if embeds is None:
            return {'total': torch.tensor(0.0, device=self.device)}

        embeds = embeds.to(self.device)
        tissue_labels = batch.get('tissue_label')
        if tissue_labels is not None:
            tissue_labels = tissue_labels.to(self.device)

        if modality == 'image':
            outputs = self.model(image_embeds=embeds, return_embeddings=True)
        else:
            outputs = self.model(rna_embeds=embeds, return_embeddings=True)

        losses = self.loss_fn(
            outputs,
            tissue_labels=tissue_labels,
            is_paired=False,
            image_prototypes=self.model.image_prototypes,
            rna_prototypes=self.model.rna_prototypes
        )

        return losses

    def _initialize_prototypes(self):
        """Initialize prototypes from paired data."""
        if self.paired_dataset is None:
            logger.warning("No paired dataset provided, skipping prototype initialization")
            return

        from collections import defaultdict
        paired_data = defaultdict(list)

        # Collect paired samples by tissue type
        for i in range(len(self.paired_dataset)):
            item = self.paired_dataset[i]
            tissue_id = item['tissue_label'].item()
            paired_data[tissue_id].append((
                item['image_embed'],
                item['rna_embed']
            ))

        self.model.initialize_prototypes(paired_data)

    def pretrain(
        self,
        epochs: Optional[int] = None,
        log_every: int = 10,
        save_every: int = 10
    ) -> Dict[str, List[float]]:
        """
        Stage 1 & 2: Pre-training with intra-modal and cross-modal losses.
        """
        epochs = epochs or self.config.pretrain_epochs

        logger.info("=" * 60)
        logger.info("STAGE 1 & 2: Pre-training (Modality-Slot Q-Former)")
        logger.info("=" * 60)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        total_steps = epochs * self._get_num_batches('pretrain')
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1
        )

        # Initialize prototypes from paired data
        self._initialize_prototypes()

        for epoch in range(epochs):
            self.current_epoch = epoch
            self.model.train()
            epoch_losses = defaultdict(float)
            num_batches = 0

            paired_iter = self._get_infinite_iterator('paired')
            unpaired_img_iter = self._get_infinite_iterator('unpaired_image')
            unpaired_rna_iter = self._get_infinite_iterator('unpaired_rna')

            num_batches_per_epoch = self._get_num_batches('pretrain')
            pbar = tqdm(range(num_batches_per_epoch), desc=f"Pretrain Epoch {epoch+1}/{epochs}")

            accumulation_steps = 3

            for batch_idx in pbar:
                optimizer.zero_grad()
                total_loss = 0.0

                # Paired data forward
                paired_batch = next(paired_iter)
                paired_loss_dict = self._forward_paired(paired_batch)
                loss = paired_loss_dict.get('total', 0)
                total_loss = total_loss + loss

                scaled_loss = loss / accumulation_steps
                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Unpaired image forward
                unpaired_img_batch = next(unpaired_img_iter)
                unpaired_img_loss_dict = self._forward_unpaired(unpaired_img_batch, 'image')
                loss = unpaired_img_loss_dict.get('total', 0)
                total_loss = total_loss + loss

                scaled_loss = loss / accumulation_steps
                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Unpaired RNA forward
                unpaired_rna_batch = next(unpaired_rna_iter)
                unpaired_rna_loss_dict = self._forward_unpaired(unpaired_rna_batch, 'rna')
                loss = unpaired_rna_loss_dict.get('total', 0)
                total_loss = total_loss + loss

                scaled_loss = loss / accumulation_steps
                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Gradient update
                if self.scaler:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                scheduler.step()

                # Accumulate losses
                for k, v in paired_loss_dict.items():
                    epoch_losses[k] += v.item() if hasattr(v, 'item') else v
                for k, v in unpaired_img_loss_dict.items():
                    epoch_losses[k] += v.item() if hasattr(v, 'item') else v
                for k, v in unpaired_rna_loss_dict.items():
                    epoch_losses[k] += v.item() if hasattr(v, 'item') else v

                num_batches += 1

                # Logging
                if batch_idx % log_every == 0:
                    avg_loss = total_loss.item() if hasattr(total_loss, 'item') else total_loss
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

                    if WANDB_AVAILABLE and self.config.use_wandb:
                        wandb.log({'pretrain_loss': avg_loss, 'step': epoch * num_batches_per_epoch + batch_idx})

            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= num_batches

            self.history['pretrain_loss'].append(epoch_losses.get('total', 0))

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_losses.get('total', 0):.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=epoch_losses.get('total', 0),
                    config=self.config,
                    path=f"{self.config.checkpoint_dir}/pretrain_epoch_{epoch+1}.pt"
                )

        logger.info("Pre-training complete!")
        return dict(self.history)

    def pretrain_drug_cellline(
        self,
        epochs: Optional[int] = None,
        log_every: int = 10,
        save_every: int = 10
    ) -> Dict[str, List[float]]:
        """
        Stage 1b: Drug-CellLine Contrastive Pretraining

        This stage teaches Q-Former about drug-cellline interactions before IC50 fine-tuning.
        Uses contrastive learning between drug embeddings and cell-line RNA embeddings.

        The goal is to close the gap between:
        - Pretraining on image-RNA pairs
        - Fine-tuning on drug-cellline IC50 prediction
        """
        epochs = epochs or self.config.dcl_pretrain_epochs

        logger.info("=" * 60)
        logger.info("STAGE 1b: Drug-CellLine Contrastive Pretraining")
        logger.info("=" * 60)

        # Initialize losses
        dcl_loss_fn = DrugCellLineContrastiveLoss(
            temperature=self.config.dcl_temperature
        )
        mmp_loss_fn = None
        if self.config.use_masked_modality_pred:
            mmp_loss_fn = MaskedModalityPredictionLoss(
                hidden_dim=self.config.hidden_dim
            )

        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        total_steps = epochs * self._get_num_batches('ic50')
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1
        )

        for epoch in range(epochs):
            self.current_epoch = epoch
            self.model.train()
            epoch_losses = defaultdict(float)
            num_batches = 0

            # Get IC50 dataloader
            ic50_loader = self.data_loaders.get('ic50_train')
            if ic50_loader is None:
                # Try alternative key
                ic50_loader = self.data_loaders.get('ic50')
            if ic50_loader is None:
                raise ValueError("IC50 data loader not found for DCL pretraining")

            ic50_iter = iter(ic50_loader)

            num_batches_per_epoch = len(ic50_loader)
            pbar = tqdm(range(num_batches_per_epoch), desc=f"DCL Epoch {epoch+1}/{epochs}")

            for batch_idx in pbar:
                try:
                    batch = next(ic50_iter)
                except StopIteration:
                    ic50_iter = iter(ic50_loader)
                    batch = next(ic50_iter)

                optimizer.zero_grad()

                # Get data
                drug_embeds = batch['drug_embed'].to(self.device)
                cellline_rna = batch.get('rna_embed')
                if cellline_rna is not None:
                    cellline_rna = cellline_rna.to(self.device)
                cellline_ids = batch['cellline_id'].to(self.device)
                cancer_type_ids = batch.get('cancer_type_id')
                if cancer_type_ids is not None:
                    cancer_type_ids = cancer_type_ids.to(self.device)
                tissue_ids = batch.get('tissue_id')
                if tissue_ids is not None:
                    tissue_ids = tissue_ids.to(self.device)
                rna_available = batch.get('rna_available')
                if rna_available is not None:
                    rna_available = rna_available.to(self.device)

                # Forward pass - need gradients for contrastive learning
                outputs = self.model(
                    drug_embeds=drug_embeds,
                    cellline_ids=cellline_ids,
                    cancer_type_ids=cancer_type_ids,
                    tissue_ids=tissue_ids,
                    cellline_rna_embeds=cellline_rna,
                    rna_available=rna_available,
                    return_embeddings=True
                )

                projected = outputs.get('projected', {})

                if 'drug' not in projected or 'cellline' not in projected:
                    logger.warning(f"Missing projected embeddings in batch {batch_idx}")
                    continue

                drug_proj = projected['drug']
                cellline_proj = projected['cellline']

                # Drug-CellLine Contrastive Loss
                dcl_loss = dcl_loss_fn(drug_proj, cellline_proj)

                total_loss = self.config.lambda_dcl * dcl_loss

                # Optional: Masked Modality Prediction
                if mmp_loss_fn is not None and cellline_rna is not None:
                    mmp_loss = mmp_loss_fn(cellline_proj, drug_proj)
                    total_loss = total_loss + self.config.lambda_mmp * mmp_loss
                    epoch_losses['mmp'] += mmp_loss.item()

                # Backward
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(optimizer)
                else:
                    total_loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                if self.scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                scheduler.step()

                # Logging
                epoch_losses['total'] += total_loss.item()
                epoch_losses['dcl'] += dcl_loss.item()
                num_batches += 1

                if batch_idx % log_every == 0:
                    pbar.set_postfix({
                        'loss': f"{epoch_losses['total']/num_batches:.4f}",
                        'dcl': f"{epoch_losses['dcl']/num_batches:.4f}"
                    })

            # Validation
            val_metrics = self._validate_dcl()

            logger.info(f"DCL Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {epoch_losses['total']/num_batches:.4f}, "
                       f"DCL Loss: {epoch_losses['dcl']/num_batches:.4f}, "
                       f"Val DCL: {val_metrics.get('dcl_loss', 0):.4f}")

            self.history['train_loss'].append(epoch_losses['total']/num_batches)
            self.history['val_loss'].append(val_metrics.get('dcl_loss', 0))

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    self.model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    loss=epoch_losses.get('total', 0),
                    config=self.config,
                    path=f"{self.config.checkpoint_dir}/dcl_epoch_{epoch+1}.pt"
                )

        # Save best model
        save_checkpoint(
            self.model,
            optimizer,
            scheduler,
            epoch=epochs,
            loss=epoch_losses.get('total', 0),
            config=self.config,
            path=f"{self.config.checkpoint_dir}/pretrained_dcl.pt"
        )

        logger.info("Drug-CellLine Contrastive Pretraining complete!")
        return dict(self.history)

    def _validate_dcl(self) -> Dict[str, float]:
        """Validation for DCL pretraining."""
        self.model.eval()

        dcl_loss_fn = DrugCellLineContrastiveLoss(
            temperature=self.config.dcl_temperature
        )

        ic50_loader = self.data_loaders.get('ic50_val')
        if ic50_loader is None:
            ic50_loader = self.data_loaders.get('ic50')

        dcl_losses = []

        with torch.no_grad():
            for batch in ic50_loader:
                drug_embeds = batch['drug_embed'].to(self.device)
                cellline_rna = batch.get('rna_embed')
                if cellline_rna is not None:
                    cellline_rna = cellline_rna.to(self.device)
                cellline_ids = batch['cellline_id'].to(self.device)
                cancer_type_ids = batch.get('cancer_type_id')
                if cancer_type_ids is not None:
                    cancer_type_ids = cancer_type_ids.to(self.device)
                tissue_ids = batch.get('tissue_id')
                if tissue_ids is not None:
                    tissue_ids = tissue_ids.to(self.device)
                rna_available = batch.get('rna_available')
                if rna_available is not None:
                    rna_available = rna_available.to(self.device)

                outputs = self.model(
                    drug_embeds=drug_embeds,
                    cellline_ids=cellline_ids,
                    cancer_type_ids=cancer_type_ids,
                    tissue_ids=tissue_ids,
                    cellline_rna_embeds=cellline_rna,
                    rna_available=rna_available,
                    return_embeddings=True
                )

                projected = outputs.get('projected', {})

                if 'drug' not in projected or 'cellline' not in projected:
                    continue

                drug_proj = projected['drug']
                cellline_proj = projected['cellline']

                dcl_loss = dcl_loss_fn(drug_proj, cellline_proj)
                dcl_losses.append(dcl_loss.item())

        self.model.train()

        return {
            'dcl_loss': np.mean(dcl_losses) if dcl_losses else 0.0
        }

    def finetune_ic50(
        self,
        epochs: Optional[int] = None,
        log_every: int = 10
    ) -> Dict[str, List[float]]:
        """
        Stage 3: Fine-tune for IC50 prediction with differential learning rates.

        Key change from v4: Uses differential LR for Q-Former vs task heads.
        - Q-Former backbone: lower LR (learning_rate * qformer_finetune_lr_ratio)
        - IC50 head + CellLine encoder: full learning_rate
        """
        epochs = epochs or self.config.finetune_epochs

        train_loader = self.data_loaders.get('ic50_train', self.data_loaders.get('ic50'))
        val_loader = self.data_loaders.get('ic50_val')
        test_loader = self.data_loaders.get('ic50_test')

        if train_loader is None:
            raise ValueError("IC50 data loader not provided for fine-tuning")

        logger.info("=" * 60)
        logger.info("STAGE 3: Fine-tuning for IC50 prediction (Differential LR)")
        logger.info("=" * 60)

        # Handle freeze options
        if self.config.freeze_qformer_in_finetune:
            logger.info("Freezing Q-Former during fine-tuning (ablation)")
            for param in self.model.qformer.parameters():
                param.requires_grad = False

        if self.config.freeze_type_embeddings_in_finetune:
            logger.info("Freezing type embeddings during fine-tuning (ablation)")
            for name, param in self.model.named_parameters():
                if 'modality_type_embeddings' in name:
                    param.requires_grad = False

        # Build parameter groups with differential learning rates
        # Key insight: drug projector and drug/cellline type embeddings were NEVER
        # pretrained (pretraining only uses image+RNA), so they need full LR.
        # Pretrained components (image/rna projectors, Q-Former) get lower LR.
        param_groups = []
        assigned_names = set()

        base_lr = self.config.learning_rate
        low_lr = base_lr * self.config.qformer_finetune_lr_ratio

        # Q-Former backbone: lower LR (pretrained)
        qformer_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'qformer' in name:
                qformer_params.append(param)
                assigned_names.add(name)

        if qformer_params:
            param_groups.append({'params': qformer_params, 'lr': low_lr})
            logger.info(f"Q-Former LR: {low_lr} (pretrained, {len(qformer_params)} params)")

        # Split projectors: pretrained (image, rna) get low LR; new (drug) gets full LR
        pretrained_proj_params = []
        new_proj_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in assigned_names and 'projectors' in name:
                if 'projectors.image' in name or 'projectors.rna' in name:
                    pretrained_proj_params.append(param)
                else:  # drug projector — never pretrained
                    new_proj_params.append(param)
                assigned_names.add(name)

        if pretrained_proj_params and not self.config.freeze_projectors_in_finetune:
            param_groups.append({'params': pretrained_proj_params, 'lr': low_lr})
            logger.info(f"Pretrained Projectors (image, rna) LR: {low_lr}")
        if new_proj_params and not self.config.freeze_projectors_in_finetune:
            param_groups.append({'params': new_proj_params, 'lr': base_lr})
            logger.info(f"New Projectors (drug) LR: {base_lr} (never pretrained)")
        if self.config.freeze_projectors_in_finetune:
            logger.info("Projectors frozen")

        # Split type embeddings: pretrained (image, rna) get low LR; new (drug, cellline) get full LR
        pretrained_type_params = []
        new_type_params = []
        if not self.config.freeze_type_embeddings_in_finetune:
            for name, param in self.model.named_parameters():
                if param.requires_grad and name not in assigned_names and 'modality_type_embeddings' in name:
                    if 'image' in name or 'rna' in name:
                        pretrained_type_params.append(param)
                    else:  # drug, cellline — never pretrained
                        new_type_params.append(param)
                    assigned_names.add(name)

            if pretrained_type_params:
                param_groups.append({'params': pretrained_type_params, 'lr': low_lr})
                logger.info(f"Pretrained Type Embeds (image, rna) LR: {low_lr}")
            if new_type_params:
                param_groups.append({'params': new_type_params, 'lr': base_lr})
                logger.info(f"New Type Embeds (drug, cellline) LR: {base_lr} (never pretrained)")
        else:
            # Mark frozen type embed names as assigned so they don't leak to "other"
            for name, param in self.model.named_parameters():
                if 'modality_type_embeddings' in name:
                    assigned_names.add(name)
            logger.info("Type Embeds frozen")

        # IC50 attn pool head: 5x base LR (brand new, needs fast learning)
        attn_pool_params = []
        ic50_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in assigned_names:
                if 'ic50_attn_pool_head' in name:
                    attn_pool_params.append(param)
                    assigned_names.add(name)
                elif 'ic50_head' in name or 'cellline_encoder' in name:
                    ic50_params.append(param)
                    assigned_names.add(name)

        if attn_pool_params:
            attn_pool_lr = base_lr * 5.0
            param_groups.append({'params': attn_pool_params, 'lr': attn_pool_lr})
            logger.info(f"IC50 AttnPool Head LR: {attn_pool_lr} (5x base)")
        if ic50_params:
            param_groups.append({'params': ic50_params, 'lr': base_lr})
            logger.info(f"IC50 Head + CellLine Encoder LR: {base_lr}")

        # Remaining params (tissue_head, cancer_head, etc.): full LR
        other_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in assigned_names:
                other_params.append(param)

        if other_params:
            param_groups.append({'params': other_params, 'lr': base_lr})

        optimizer = AdamW(param_groups, weight_decay=self.config.weight_decay)

        # Warmup + Cosine Decay scheduler
        # 2-epoch linear warmup prevents destabilizing pretrained weights
        warmup_epochs = min(2, max(1, epochs // 5))
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        logger.info(f"Scheduler: {warmup_epochs}-epoch warmup + cosine decay")

        # --- Training improvements ---

        # EMA: Exponential Moving Average of model weights
        ema = None
        if self.config.use_ema:
            ema = EMAModel(self.model, decay=self.config.ema_decay)
            logger.info(f"EMA enabled (decay={self.config.ema_decay})")

        # Loss function selection
        if self.config.use_huber_loss:
            ic50_loss_fn = lambda pred, tgt: nn.functional.huber_loss(pred, tgt, delta=self.config.huber_delta)
            logger.info(f"IC50 loss: Huber (delta={self.config.huber_delta})")
        else:
            ic50_loss_fn = nn.functional.mse_loss
            logger.info("IC50 loss: MSE")

        # Multi-task and R-Drop flags
        if self.config.use_multitask_finetune:
            logger.info(f"Multi-task finetune: tissue classification (lambda={self.config.lambda_tissue_finetune})")
        if self.config.use_rdrop:
            logger.info(f"R-Drop enabled (alpha={self.config.rdrop_alpha})")

        finetune_history = defaultdict(list)
        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{epochs}")

            all_predictions = []
            all_targets = []

            for batch in pbar:
                optimizer.zero_grad()

                # Move batch to device
                drug_embeds = batch['drug_embed'].to(self.device)
                ic50_targets = batch['ic50'].to(self.device)
                cellline_ids = batch['cellline_id'].to(self.device)

                cancer_type_ids = batch.get('cancer_type_id')
                if cancer_type_ids is not None:
                    cancer_type_ids = cancer_type_ids.to(self.device)

                tissue_ids_batch = batch.get('tissue_id')
                if tissue_ids_batch is not None:
                    tissue_ids_batch = tissue_ids_batch.to(self.device)

                cellline_rna_embeds = batch.get('rna_embed')
                if cellline_rna_embeds is not None:
                    cellline_rna_embeds = cellline_rna_embeds.to(self.device)

                rna_available = batch.get('rna_available')
                if rna_available is not None:
                    rna_available = rna_available.to(self.device)

                fwd_kwargs = dict(
                    drug_embeds=drug_embeds,
                    cellline_ids=cellline_ids,
                    cancer_type_ids=cancer_type_ids,
                    tissue_ids=tissue_ids_batch,
                    cellline_rna_embeds=cellline_rna_embeds,
                    rna_available=rna_available,
                )

                # Forward pass
                with autocast(self._device_type, enabled=self.scaler is not None):
                    outputs = self.model(**fwd_kwargs)

                    # Primary IC50 loss (Huber or MSE)
                    loss = ic50_loss_fn(outputs['ic50_pred'], ic50_targets)

                    # Multi-task: auxiliary tissue classification loss
                    if self.config.use_multitask_finetune and tissue_ids_batch is not None:
                        tissue_logits = outputs.get('tissue_logits')
                        if tissue_logits is not None:
                            tissue_loss = nn.functional.cross_entropy(tissue_logits, tissue_ids_batch)
                            loss = loss + self.config.lambda_tissue_finetune * tissue_loss

                    # R-Drop: forward twice, penalize prediction divergence
                    if self.config.use_rdrop:
                        outputs2 = self.model(**fwd_kwargs)
                        loss2 = ic50_loss_fn(outputs2['ic50_pred'], ic50_targets)
                        # Symmetric MSE between the two stochastic predictions
                        rdrop_loss = nn.functional.mse_loss(outputs['ic50_pred'], outputs2['ic50_pred'])
                        loss = 0.5 * (loss + loss2) + self.config.rdrop_alpha * rdrop_loss

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                # EMA update after each optimizer step
                if ema is not None:
                    ema.update(self.model)

                total_loss += loss.item()
                num_batches += 1

                all_predictions.extend(outputs['ic50_pred'].detach().cpu().numpy())
                all_targets.extend(ic50_targets.cpu().numpy())

                if num_batches % log_every == 0:
                    pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})

            # Compute metrics
            avg_loss = total_loss / num_batches
            metrics = compute_ic50_metrics(
                torch.tensor(all_predictions),
                torch.tensor(all_targets)
            )

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, MSE: {metrics['mse']:.4f}, Pearson R: {metrics['pearson_r']:.4f}")

            finetune_history['train_loss'].append(avg_loss)
            finetune_history['train_mse'].append(metrics['mse'])
            finetune_history['train_pearson_r'].append(metrics['pearson_r'])

            if WANDB_AVAILABLE and self.config.use_wandb:
                wandb.log({
                    'finetune_loss': avg_loss,
                    'finetune_mse': metrics['mse'],
                    'finetune_r2': metrics['r2'],
                    'epoch': epoch
                })

            # Validation (use EMA weights if available)
            if val_loader is not None:
                if ema is not None:
                    with ema.apply(self.model):
                        val_loss, val_metrics = self._validate_ic50(val_loader)
                else:
                    val_loss, val_metrics = self._validate_ic50(val_loader)

                finetune_history['val_loss'].append(val_loss)
                finetune_history['val_mse'].append(val_metrics['mse'])
                finetune_history['val_r2'].append(val_metrics['r2'])

                logger.info(f"Validation - Loss: {val_loss:.4f}, MSE: {val_metrics['mse']:.4f}, R2: {val_metrics['r2']:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save EMA weights if using EMA, otherwise save regular weights
                    if ema is not None:
                        with ema.apply(self.model):
                            save_checkpoint(
                                model=self.model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch,
                                loss=val_loss,
                                config=self.config,
                                path=f"{self.config.checkpoint_dir}/best_ic50_model.pt"
                            )
                    else:
                        save_checkpoint(
                            model=self.model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            loss=val_loss,
                            config=self.config,
                            path=f"{self.config.checkpoint_dir}/best_ic50_model.pt"
                        )

            scheduler.step()

        # Test evaluation (use EMA weights if available)
        if test_loader is not None:
            if ema is not None:
                with ema.apply(self.model):
                    test_loss, test_metrics = self._validate_ic50(test_loader)
            else:
                test_loss, test_metrics = self._validate_ic50(test_loader)
            finetune_history['test_loss'] = test_loss
            finetune_history['test_mse'] = test_metrics['mse']
            finetune_history['test_r2'] = test_metrics['r2']
            logger.info(f"Test - Loss: {test_loss:.4f}, MSE: {test_metrics['mse']:.4f}, R2: {test_metrics['r2']:.4f}")

        logger.info("IC50 fine-tuning complete!")
        return dict(finetune_history)

    def _validate_ic50(self, val_loader: DataLoader) -> tuple:
        """Validate on IC50 data."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                drug_embeds = batch['drug_embed'].to(self.device)
                ic50_targets = batch['ic50'].to(self.device)
                cellline_ids = batch['cellline_id'].to(self.device)

                cancer_type_ids = batch.get('cancer_type_id')
                if cancer_type_ids is not None:
                    cancer_type_ids = cancer_type_ids.to(self.device)

                tissue_ids_batch = batch.get('tissue_id')
                if tissue_ids_batch is not None:
                    tissue_ids_batch = tissue_ids_batch.to(self.device)

                cellline_rna_embeds = batch.get('rna_embed')
                if cellline_rna_embeds is not None:
                    cellline_rna_embeds = cellline_rna_embeds.to(self.device)

                rna_available = batch.get('rna_available')
                if rna_available is not None:
                    rna_available = rna_available.to(self.device)

                outputs = self.model(
                    drug_embeds=drug_embeds,
                    cellline_ids=cellline_ids,
                    cancer_type_ids=cancer_type_ids,
                    tissue_ids=tissue_ids_batch,
                    cellline_rna_embeds=cellline_rna_embeds,
                    rna_available=rna_available
                )

                loss = nn.functional.mse_loss(outputs['ic50_pred'], ic50_targets)
                total_loss += loss.item()

                all_predictions.extend(outputs['ic50_pred'].cpu().numpy())
                all_targets.extend(ic50_targets.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        metrics = compute_ic50_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_targets)
        )

        self.model.train()
        return avg_loss, metrics


def evaluate_ncd(
    checkpoint_path: str,
    config: GastroTransformerConfig,
    ic50_csv: str,
    drug_embeddings_csv: str,
    cellline_rna_csv: str,
    n_folds: int = 5,
    finetune_epochs: int = 10,
    device: str = 'cuda:0',
    seed: int = 42,
    finetune_lr: float = 1e-4,
    huber_delta: float = 1.5,
    early_stopping_patience: int = 3,
    output_dir: str = 'reports/ncd_evaluation'
) -> Dict:
    """
    Run NCD (No Common Drug) 5-fold CV evaluation.

    This evaluates generalization to completely unseen drugs, which is a harder
    task than the standard cell-line-aware CV. The model must predict IC50 for
    drugs it has never seen during training.

    Args:
        checkpoint_path: Path to pretrained checkpoint
        config: GastroTransformerConfig
        ic50_csv: Path to IC50 data CSV
        drug_embeddings_csv: Path to drug embeddings CSV
        cellline_rna_csv: Path to cell-line RNA embeddings CSV
        n_folds: Number of folds (default 5)
        finetune_epochs: Number of fine-tuning epochs per fold (default 10)
        device: Device to use
        seed: Random seed
        finetune_lr: Learning rate for fine-tuning
        huber_delta: Huber loss delta
        early_stopping_patience: Early stopping patience
        output_dir: Output directory for results

    Returns:
        Dictionary with per-fold metrics and summary statistics
    """
    from torch.utils.data import Subset
    from .data import (
        DrugEmbeddingDataset,
        IC50Dataset,
        create_ncd_folds,
        create_data_loaders
    )
    from .losses import compute_ic50_metrics
    from .utils import save_checkpoint

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("NCD (No Common Drug) Cross-Validation Evaluation")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    drug_dataset = DrugEmbeddingDataset(drug_embeddings_csv, drug_dim=config.drug_dim)

    # Load full IC50 dataframe for fold creation
    ic50_df = pd.read_csv(ic50_csv)
    valid_drugs = set(drug_dataset.drug_ids)
    ic50_df = ic50_df[ic50_df['drug_id'].isin(valid_drugs)]
    ic50_df = ic50_df.drop_duplicates(subset=['cellline_id', 'drug_id'], keep='first')
    logger.info(f"Total IC50 samples: {len(ic50_df)}, Unique drugs: {ic50_df['drug_id'].nunique()}")

    # Create NCD folds
    logger.info(f"Creating {n_folds} NCD folds...")
    ncd_folds = create_ncd_folds(ic50_df, n_folds=n_folds, seed=seed)

    # Storage for results
    all_fold_results = []
    all_predictions = []
    all_targets = []

    for fold_idx, (train_indices, test_indices) in enumerate(ncd_folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx + 1}/{n_folds}")
        logger.info(f"{'='*60}")

        # Create dataset for this fold
        ic50_dataset = IC50Dataset(
            ic50_csv_path=ic50_csv,
            drug_embeddings=drug_dataset,
            rna_csv_path=cellline_rna_csv,
            add_tissue_ids=True
        )

        # Create train/val/test subsets
        # Further split train into train/val (90/10)
        rng = np.random.default_rng(seed + fold_idx)
        train_indices = train_indices.copy()
        rng.shuffle(train_indices)

        val_size = int(len(train_indices) * 0.1)
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]

        train_subset = Subset(ic50_dataset, train_indices)
        val_subset = Subset(ic50_dataset, val_indices)
        test_subset = Subset(ic50_dataset, test_indices)

        logger.info(f"Train: {len(train_indices)} samples, "
                   f"Val: {len(val_indices)} samples, "
                   f"Test: {len(test_indices)} samples")

        # Check train/test drug overlap (should be none)
        train_drugs = set(ic50_df.iloc[train_indices]['drug_id'])
        test_drugs = set(ic50_df.iloc[test_indices]['drug_id'])
        overlap = train_drugs & test_drugs
        logger.info(f"Drug overlap check: {len(overlap)} (should be 0)")

        # Create data loaders
        data_loaders = create_data_loaders(
            config,
            ic50_train=train_subset,
            ic50_val=val_subset,
            ic50_test=test_subset,
            split_ic50=False
        )

        # Create fresh model for each fold
        logger.info("Creating fresh model...")
        model = ModalitySlotQFormer(config)

        # Load pretrained checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        # Create trainer
        trainer = GastroTransformerTrainer(
            model=model,
            config=config,
            data_loaders=data_loaders,
            device=device
        )

        # Fine-tune with custom settings for NCD
        logger.info(f"Fine-tuning for {finetune_epochs} epochs...")
        # Use Huber loss for robustness
        config.use_huber_loss = True
        config.huber_delta = huber_delta
        config.learning_rate = finetune_lr

        # Fine-tune
        history = trainer.finetune_ic50(epochs=finetune_epochs, log_every=50)

        # Evaluate on test set
        test_loader = data_loaders['ic50_test']
        test_loss, test_metrics = trainer._validate_ic50(test_loader)

        logger.info(f"Fold {fold_idx + 1} Test Metrics:")
        logger.info(f"  R2: {test_metrics['r2']:.4f}")
        logger.info(f"  Pearson r: {test_metrics['pearson_r']:.4f}")
        logger.info(f"  Spearman r: {test_metrics['spearman_r']:.4f}")
        logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  MAE: {test_metrics['mae']:.4f}")

        # Warn about low R2
        if test_metrics['r2'] < 0:
            warnings.warn(
                f"Fold {fold_idx + 1}: R² = {test_metrics['r2']:.4f} < 0. "
                f"NCD is a hard generalization task - negative R² is expected.",
                UserWarning
            )

        fold_result = {
            'fold': fold_idx + 1,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices),
            'train_drugs': len(train_drugs),
            'test_drugs': len(test_drugs),
            'r2': test_metrics['r2'],
            'pearson_r': test_metrics['pearson_r'],
            'spearman_r': test_metrics['spearman_r'],
            'rmse': test_metrics['rmse'],
            'mae': test_metrics['mae'],
            'mse': test_metrics['mse']
        }
        all_fold_results.append(fold_result)

        # Save fold predictions
        model.eval()
        fold_preds = []
        fold_targets = []
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

                fold_preds.extend(outputs['ic50_pred'].cpu().numpy())
                fold_targets.extend(ic50_targets.cpu().numpy())

        all_predictions.extend(fold_preds)
        all_targets.extend(fold_targets)

        # Save fold checkpoint
        save_checkpoint(
            model=model,
            optimizer=None,
            scheduler=None,
            epoch=fold_idx,
            loss=test_loss,
            config=config,
            path=f"{output_dir}/fold_{fold_idx+1}_model.pt"
        )

    # Compute aggregate statistics
    r2_values = [r['r2'] for r in all_fold_results]
    pearson_values = [r['pearson_r'] for r in all_fold_results]
    spearman_values = [r['spearman_r'] for r in all_fold_results]
    rmse_values = [r['rmse'] for r in all_fold_results]
    mae_values = [r['mae'] for r in all_fold_results]

    results = {
        'n_folds': n_folds,
        'finetune_epochs': finetune_epochs,
        'finetune_lr': finetune_lr,
        'huber_delta': huber_delta,
        'fold_results': all_fold_results,
        'summary': {
            'r2_mean': np.mean(r2_values),
            'r2_std': np.std(r2_values),
            'pearson_r_mean': np.mean(pearson_values),
            'pearson_r_std': np.std(pearson_values),
            'spearman_r_mean': np.mean(spearman_values),
            'spearman_r_std': np.std(spearman_values),
            'rmse_mean': np.mean(rmse_values),
            'rmse_std': np.std(rmse_values),
            'mae_mean': np.mean(mae_values),
            'mae_std': np.std(mae_values),
        }
    }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("NCD EVALUATION SUMMARY")
    logger.info("=" * 60)

    # Comparison table with CL-CV baseline
    cl_cv_r2 = 0.757  # Hardcoded from CLAUDE.md
    cl_cv_r2_std = 0.007

    logger.info("\n| Split   | R²               | Pearson r        | RMSE             |")
    logger.info("|---------|------------------|------------------|------------------|")
    logger.info(f"| CL-CV   | {cl_cv_r2:.3f} ± {cl_cv_r2_std:.3f}      | 0.872            | 1.37             |")
    logger.info(f"| NCD     | {results['summary']['r2_mean']:.3f} ± {results['summary']['r2_std']:.3f}      | "
                f"{results['summary']['pearson_r_mean']:.3f}             | "
                f"{results['summary']['rmse_mean']:.3f}             |")

    logger.info("\nNote: NCD is a harder generalization task than CL-CV.")
    logger.info("The model must predict IC50 for drugs not seen during training.")

    if results['summary']['r2_mean'] < 0:
        logger.warning("\n⚠️  Negative R² detected - this is EXPECTED for NCD!")
        logger.warning("The model cannot generalize to completely unseen drugs without")
        logger.warning("molecular representations of those drugs.")

    # Save results
    results_path = f"{output_dir}/ncd_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            return obj

        json.dump(convert_to_native(results), f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    return results
