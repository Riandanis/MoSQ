#!/usr/bin/env python3
"""
Phase 3: IC50-Aware Pretraining

Replaces Stage 2b's trivially-easy classification objectives with two IC50-aware
objectives that directly align with downstream regression:

A. IC50-Aware Supervised Contrastive Loss:
   - Compute IC50 quintile boundaries (5 bins) from full dataset
   - Positives = same IC50 bin, temperature = 0.1
   - Follows IntraModalContrastiveLoss pattern (losses.py)

C. Cross-Modal Reconstruction:
   - Randomly mask drug OR cellline tokens (50/50)
   - Reconstruct masked token's projected embedding via MSE
   - Decoder: Linear(768,768) → GELU → Linear(768,768)

Loss: 1.0 * L_contrastive + 0.5 * L_reconstruction

Loads from CLRNA checkpoint (Stage 1), saves pretrained_dra.pt.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from gastro_transformer.config import GastroTransformerConfig
from gastro_transformer.model import ModalitySlotQFormer
from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DRUG_CSV = str(ROOT / 'data/drug_embeddings.csv')
IC50_CSV = str(ROOT / 'data/ic50_data.csv')
RNA_CSV  = str(ROOT / 'data/ccle_rna_for_ic50.csv')
DEFAULT_CHECKPOINT = str(ROOT / 'saved_checkpoints/pretrained_clrna.pt')


def load_pretrained_weights(model, checkpoint_path, device):
    """Load pretrained weights with partial matching (strict=False)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape}
    logger.info(f"Loaded {len(filtered)}/{len(model_dict)} params from checkpoint")
    model.load_state_dict(filtered, strict=False)
    return model


def build_tokens_for_batch(model, batch, device):
    """
    Build the 4 KV tokens (drug + cancer + tissue + cellline_rna) exactly as
    model.forward() does for MultiToken mode.

    Returns:
        tokens_list: list of 4 tensors, each [B, 1, D] (drug, cancer, tissue, rna)
        combined: [B, 4, D] concatenated
    """
    drug_embeds = batch['drug_embed'].to(device)
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

    # Drug token
    drug_proj = model.projectors['drug'](drug_embeds).unsqueeze(1)  # [B, 1, D]
    drug_proj = drug_proj + model.modality_type_embeddings['drug']

    # Cell-line tokens
    cl_tokens = model.cellline_encoder(
        cancer_type_ids=cancer_type_ids,
        tissue_ids=tissue_ids,
        rna_embeds=cellline_rna,
        rna_available=rna_available,
    )  # [B, 3, D]
    # Add type embeddings
    cl_tokens[:, 0:1, :] = cl_tokens[:, 0:1, :] + model.modality_type_embeddings['cancer']
    cl_tokens[:, 1:2, :] = cl_tokens[:, 1:2, :] + model.modality_type_embeddings['tissue']
    cl_tokens[:, 2:3, :] = cl_tokens[:, 2:3, :] + model.modality_type_embeddings['cellline_rna']

    cancer_tok = cl_tokens[:, 0:1, :]
    tissue_tok = cl_tokens[:, 1:2, :]
    rna_tok = cl_tokens[:, 2:3, :]

    tokens_list = [drug_proj, cancer_tok, tissue_tok, rna_tok]
    combined = torch.cat(tokens_list, dim=1)  # [B, 4, D]
    return tokens_list, combined


def compute_ic50_quintiles(ic50_ds):
    """Compute quintile boundaries from full IC50 dataset."""
    all_ic50 = []
    for i in range(len(ic50_ds)):
        sample = ic50_ds[i]
        all_ic50.append(sample['ic50'].item())
    all_ic50 = np.array(all_ic50)
    boundaries = np.percentile(all_ic50, [20, 40, 60, 80])
    logger.info(f"IC50 quintile boundaries: {boundaries}")
    logger.info(f"IC50 range: [{all_ic50.min():.2f}, {all_ic50.max():.2f}], "
                f"mean={all_ic50.mean():.2f}, std={all_ic50.std():.2f}")
    return boundaries


def assign_quintile_bins(ic50_values, boundaries):
    """Assign IC50 values to quintile bins (0-4)."""
    bins = torch.zeros_like(ic50_values, dtype=torch.long)
    for i, b in enumerate(boundaries):
        bins = bins + (ic50_values > b).long()
    return bins


class IC50SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss where positives = same IC50 quintile bin.
    Follows IntraModalContrastiveLoss pattern from losses.py.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [B, D] fused representations
            labels: [B] quintile bin labels (0-4)
        Returns:
            Scalar loss
        """
        embeddings = F.normalize(embeddings, dim=-1)
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Similarity matrix [B, B]
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Positive mask: same bin, exclude diagonal
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)

        pos_count = pos_mask.sum(dim=1)
        valid_mask = pos_count > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=embeddings.device)

        # Stable log-softmax (same as losses.py M9 fix)
        log_softmax_sim = F.log_softmax(sim, dim=1)
        log_prob = (log_softmax_sim * pos_mask).sum(dim=1)
        loss = -log_prob[valid_mask] / pos_count[valid_mask].clamp(min=1)

        return loss.mean()


def main():
    parser = argparse.ArgumentParser(description='Phase 3: IC50-Aware Pretraining')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='CLRNA pretrained checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default=str(ROOT / 'saved_checkpoints/phase3'))
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--qformer_lr_ratio', type=float, default=0.2,
                        help='LR ratio for Q-Former (differential LR)')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                        help='Weight for IC50 supervised contrastive loss')
    parser.add_argument('--recon_weight', type=float, default=0.5,
                        help='Weight for cross-modal reconstruction loss')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Contrastive loss temperature')
    parser.add_argument('--held_out_drugs', type=str, default=None,
                        help='Path to text file with held-out drug IDs to exclude from training')
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 3: IC50-Aware Pretraining")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    logger.info(f"Base LR: {args.base_lr}, Q-Former LR ratio: {args.qformer_lr_ratio}")
    logger.info(f"Contrastive weight: {args.contrastive_weight}, "
                f"Reconstruction weight: {args.recon_weight}")
    logger.info(f"Temperature: {args.temperature}")

    # ---- Create model ----
    config = GastroTransformerConfig()
    config.num_query_tokens = 32
    config.qformer_layers = 6
    config.use_multitoken_cellline = True
    config.use_qformer = True
    config.use_qformer_for_ic50 = True
    config.use_ic50_attn_pool = True
    config.device = device

    model = ModalitySlotQFormer(config).to(device)
    model = load_pretrained_weights(model, args.checkpoint, device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Phase 3 components ----
    hidden_dim = config.hidden_dim

    # Contrastive loss
    contrastive_loss_fn = IC50SupervisedContrastiveLoss(temperature=args.temperature)

    # Reconstruction decoder + mask tokens
    # Drug mask: replaces 1 drug token
    drug_mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim, device=device) * 0.02)
    # Cell-line mask: replaces 3 cell-line tokens with 1 mask token
    cellline_mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim, device=device) * 0.02)

    recon_decoder = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    ).to(device)

    # ---- Load data ----
    logger.info("Loading data...")
    drug_ds = DrugEmbeddingDataset(DRUG_CSV, drug_dim=768)

    # Parse held-out drugs for leak-free NCD evaluation
    allowed_drugs = None
    if args.held_out_drugs:
        held_out_path = Path(args.held_out_drugs)
        if held_out_path.exists():
            held_out_drugs = set(line.strip() for line in held_out_path.open().readlines() if line.strip())
            all_drugs = set(drug_ds.drug_id_to_idx.keys())
            allowed_drugs = all_drugs - held_out_drugs
            logger.info(f"Held-out drugs: {len(held_out_drugs)}, Training drugs: {len(allowed_drugs)}")

    ic50_ds = IC50Dataset(IC50_CSV, drug_ds, rna_csv_path=RNA_CSV, rna_dim=256,
                           add_tissue_ids=True, allowed_drugs=allowed_drugs)
    logger.info(f"Total: {len(ic50_ds)} samples, {ic50_ds.num_celllines} cell-lines")

    # Compute IC50 quintile boundaries
    boundaries = compute_ic50_quintiles(ic50_ds)
    boundaries_tensor = torch.tensor(boundaries, dtype=torch.float32, device=device)

    loader = DataLoader(
        ic50_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
    )

    # ---- Freeze irrelevant components ----
    frozen_prefixes = ['projectors.image.', 'projectors.rna.']
    frozen_names_exact = {'modality_type_embeddings.image', 'modality_type_embeddings.rna'}
    for name, p in model.named_parameters():
        if any(name.startswith(pf) for pf in frozen_prefixes) or name in frozen_names_exact:
            p.requires_grad = False

    # ---- Build optimizer with 2 param groups ----
    base_lr = args.base_lr
    low_lr = base_lr * args.qformer_lr_ratio

    low_lr_prefixes = ['qformer.']

    low_lr_params = []
    high_lr_params = []
    low_lr_names = []
    high_lr_names = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in low_lr_prefixes):
            low_lr_params.append(p)
            low_lr_names.append(name)
        else:
            high_lr_params.append(p)
            high_lr_names.append(name)

    # Phase 3 components at full LR
    high_lr_params.append(drug_mask_token)
    high_lr_params.append(cellline_mask_token)
    for p in recon_decoder.parameters():
        high_lr_params.append(p)

    param_groups = [
        {'params': low_lr_params, 'lr': low_lr},
        {'params': high_lr_params, 'lr': base_lr},
    ]

    logger.info(f"Low LR ({low_lr:.1e}): {len(low_lr_names)} param tensors "
                f"({sum(p.numel() for p in low_lr_params):,} params)")
    logger.info(f"High LR ({base_lr:.1e}): {len(high_lr_names) + 3 + sum(1 for _ in recon_decoder.parameters())} param tensors "
                f"({sum(p.numel() for p in high_lr_params):,} params)")

    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)

    total_steps = args.epochs * len(loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[low_lr, base_lr],
        total_steps=total_steps,
        pct_start=0.1,
    )

    # ---- Training loop ----
    history = {
        'contrastive_loss': [], 'recon_loss': [], 'total_loss': [],
        'mean_cosine_sim': [], 'bin_distribution': [],
    }

    for epoch in range(args.epochs):
        model.train()
        recon_decoder.train()

        epoch_losses = {'contrastive': 0., 'recon': 0., 'total': 0.}
        epoch_cosine_sims = []
        num_batches = 0

        pbar = tqdm(loader, desc=f"Phase 3 Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            # ---- Build tokens ----
            tokens_list, combined = build_tokens_for_batch(model, batch, device)
            # tokens_list: [drug, cancer, tissue, rna], each [B, 1, D]
            # combined: [B, 4, D]
            B = combined.size(0)

            # ---- Objective A: IC50-Aware Supervised Contrastive ----
            # Full forward: all 4 tokens → Q-Former → fused [B, D]
            fused = model.qformer(combined, modality_mask=None)  # [B, D]

            # Assign IC50 quintile bins
            ic50_vals = batch['ic50'].to(device)
            bins = assign_quintile_bins(ic50_vals, boundaries_tensor)

            loss_contrastive = contrastive_loss_fn(fused, bins)

            # Collapse detection: mean cosine similarity
            with torch.no_grad():
                fused_norm = F.normalize(fused, dim=-1)
                cos_sim_matrix = torch.matmul(fused_norm, fused_norm.T)
                # Exclude diagonal
                mask_diag = ~torch.eye(B, dtype=torch.bool, device=device)
                mean_cos_sim = cos_sim_matrix[mask_diag].mean().item()
                epoch_cosine_sims.append(mean_cos_sim)

            # ---- Objective C: Cross-Modal Reconstruction ----
            # 50/50 chance: mask drug or mask cellline
            mask_drug = torch.rand(1).item() < 0.5

            if mask_drug:
                # Mask drug token, keep 3 CL tokens, reconstruct drug projected embedding
                drug_target = tokens_list[0].squeeze(1).detach()  # [B, D]

                masked_input = torch.cat([
                    drug_mask_token.expand(B, -1, -1),  # [B, 1, D] mask
                    tokens_list[1],  # cancer [B, 1, D]
                    tokens_list[2],  # tissue [B, 1, D]
                    tokens_list[3],  # rna [B, 1, D]
                ], dim=1)  # [B, 4, D]

                fused_masked = model.qformer(masked_input, modality_mask=None)  # [B, D]
                predicted = recon_decoder(fused_masked)  # [B, D]
                loss_recon = F.mse_loss(predicted, drug_target)
            else:
                # Mask 3 CL tokens with single mask, keep drug, reconstruct mean of CL embeddings
                cl_target = torch.cat([
                    tokens_list[1], tokens_list[2], tokens_list[3]
                ], dim=1).mean(dim=1).detach()  # [B, D]

                masked_input = torch.cat([
                    tokens_list[0],  # drug [B, 1, D]
                    cellline_mask_token.expand(B, -1, -1),  # [B, 1, D] mask
                ], dim=1)  # [B, 2, D]

                fused_masked = model.qformer(masked_input, modality_mask=None)  # [B, D]
                predicted = recon_decoder(fused_masked)  # [B, D]
                loss_recon = F.mse_loss(predicted, cl_target)

            # ---- Total loss ----
            total_loss = (args.contrastive_weight * loss_contrastive +
                          args.recon_weight * loss_recon)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + [drug_mask_token, cellline_mask_token] +
                list(recon_decoder.parameters()),
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            epoch_losses['contrastive'] += loss_contrastive.item()
            epoch_losses['recon'] += loss_recon.item()
            epoch_losses['total'] += total_loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                pbar.set_postfix({
                    'loss': f"{epoch_losses['total']/num_batches:.4f}",
                    'con': f"{epoch_losses['contrastive']/num_batches:.4f}",
                    'rec': f"{epoch_losses['recon']/num_batches:.4f}",
                    'cos': f"{mean_cos_sim:.3f}",
                })

        # Epoch summary
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        mean_epoch_cosine = np.mean(epoch_cosine_sims)

        history['contrastive_loss'].append(epoch_losses['contrastive'])
        history['recon_loss'].append(epoch_losses['recon'])
        history['total_loss'].append(epoch_losses['total'])
        history['mean_cosine_sim'].append(float(mean_epoch_cosine))

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} — "
            f"Total: {epoch_losses['total']:.4f}, "
            f"Contrastive: {epoch_losses['contrastive']:.4f}, "
            f"Reconstruction: {epoch_losses['recon']:.4f}, "
            f"Mean cosine sim: {mean_epoch_cosine:.4f}"
        )

        # Collapse warning
        if mean_epoch_cosine > 0.95:
            logger.warning(
                f"COLLAPSE WARNING: Mean cosine similarity = {mean_epoch_cosine:.4f} > 0.95! "
                f"Representations may be collapsing."
            )

    # ---- Save checkpoint ----
    checkpoint_path = output_dir / 'pretrained_dra.pt'
    checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'drug_mask_token': drug_mask_token.data,
        'cellline_mask_token': cellline_mask_token.data,
        'recon_decoder_state_dict': recon_decoder.state_dict(),
        'loss': epoch_losses['total'],
        'config': config.__dict__,
        'history': history,
        'stage': 'phase3',
        'source_checkpoint': args.checkpoint,
        'ic50_quintile_boundaries': boundaries.tolist(),
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Save training history
    history_path = output_dir / 'phase3_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final contrastive loss: {history['contrastive_loss'][-1]:.4f}")
    logger.info(f"Final reconstruction loss: {history['recon_loss'][-1]:.4f}")
    logger.info(f"Final mean cosine sim: {history['mean_cosine_sim'][-1]:.4f}")
    logger.info(f"Checkpoint: {checkpoint_path}")


if __name__ == '__main__':
    main()
