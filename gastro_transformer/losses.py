"""
Loss Functions for Gastro-Transformer Foundation Model.

Implements the hybrid loss strategy for training with limited paired data:
    LOSS = λ1 * L_intra + λ2 * L_cross_paired + λ3 * L_proto + λ4 * L_recon + λ5 * L_ortho

Loss Components:
    - L_intra: Intra-modal supervised contrastive loss
    - L_cross: Cross-modal CLIP-style contrastive loss (paired data)
    - L_proto: Prototypical alignment loss (extends to unpaired data)
    - L_recon: Reconstruction loss for regularization
    - L_ortho: Orthogonality loss for modality disentanglement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from .config import GastroTransformerConfig


class IntraModalContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss within a single modality.

    Pulls together samples with the same tissue label (positives)
    and pushes apart samples with different labels (negatives).

    This loss helps learn good representations WITHIN each modality
    using tissue type labels as supervision.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, D]
        labels: torch.Tensor       # [B] tissue type labels
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Normalized embeddings [B, D]
            labels: Tissue type labels [B]
        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute similarity matrix [B, B]
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same label, excluding diagonal)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)  # Exclude self

        # Count positives per sample
        pos_count = pos_mask.sum(dim=1)

        # Skip samples with no positives
        valid_mask = pos_count > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=embeddings.device)

        # M9 fix: use log_softmax for numerical stability at low temperature (0.07).
        # The old implementation computed log(sum_exp_pos / sum_exp_denom + eps) which
        # is numerically equivalent but bypasses PyTorch's stable log-sum-exp kernel.
        # At T=0.07, similarities are magnified 14x, making precision critical.
        #
        # log P(j is positive | anchor i) = sim[i,j] - logsumexp(sim[i, all j≠i])
        # Summed over all positives j and averaged = supervised contrastive objective.

        # log_softmax includes the diagonal; pos_mask already zeros the diagonal,
        # so the diagonal contributes 0 to the positive sum correctly.
        log_softmax_sim = F.log_softmax(sim, dim=1)  # [B, B], stable log-probabilities

        # Sum log P over positive pairs, average by positive count per anchor
        log_prob = (log_softmax_sim * pos_mask).sum(dim=1)  # [B]
        loss = -log_prob[valid_mask] / pos_count[valid_mask].clamp(min=1)

        return loss.mean()


class CrossModalContrastiveLoss(nn.Module):
    """
    CLIP-style cross-modal contrastive loss for paired data.

    For paired samples (same patient has both modalities), the diagonal
    of the similarity matrix represents positive pairs, and off-diagonal
    elements are negatives.

    This is the CRITICAL loss for learning cross-modal alignment from
    limited paired patient data (10-50 samples).
    """

    def __init__(self, temperature: float = 0.07, learnable_temperature: bool = True):
        super().__init__()
        self.temperature = temperature

        if learnable_temperature:
            # Learnable log temperature (CLIP-style)
            self.logit_scale = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1 / temperature))
            )
        else:
            self.logit_scale = None

    def forward(
        self,
        embed_a: torch.Tensor,  # [N, D] - e.g., image embeddings
        embed_b: torch.Tensor   # [N, D] - e.g., RNA embeddings (paired)
    ) -> torch.Tensor:
        """
        Args:
            embed_a: First modality embeddings (normalized)
            embed_b: Second modality embeddings (paired, normalized)
        Returns:
            Scalar loss value
        """
        batch_size = embed_a.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=embed_a.device)

        # Normalize embeddings
        embed_a = F.normalize(embed_a, dim=-1)
        embed_b = F.normalize(embed_b, dim=-1)

        # Get temperature scale.
        # M5 fix: clamp on both ends. max=100 prevents overflow; min=1.0 prevents
        # the logit scale from collapsing toward zero (which silently zeroes gradients,
        # making cross-modal alignment appear "stable" while actually learning nothing).
        if self.logit_scale is not None:
            logit_scale = self.logit_scale.exp().clamp(min=1.0, max=100)
        else:
            logit_scale = 1.0 / self.temperature

        # Compute similarity matrix [N, N]
        logits = logit_scale * torch.matmul(embed_a, embed_b.T)

        # Labels: diagonal is positive (same patient index)
        labels = torch.arange(batch_size, device=logits.device)

        # Symmetric cross-entropy loss
        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.T, labels)

        return (loss_a2b + loss_b2a) / 2


class PrototypicalAlignmentLoss(nn.Module):
    """
    Prototypical alignment loss for extending cross-modal alignment
    to unpaired data using tissue-type prototypes.

    Key Insight: Paired samples teach the model how embeddings from the
    SAME TISSUE should relate. Prototypes (mean embeddings per tissue)
    then guide alignment of unpaired samples.

    For unpaired data:
    - Image embeddings should be closer to RNA prototypes of the same tissue
    - RNA embeddings should be closer to image prototypes of the same tissue
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,       # [B, D]
        tissue_ids: torch.Tensor,       # [B]
        prototypes: torch.Tensor,       # [num_tissues, D]
        is_image: bool = True           # Whether embeddings are from image modality
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Modality embeddings (will be compared to cross-modal prototypes)
            tissue_ids: Tissue type labels for each sample
            prototypes: Prototypes of the OTHER modality
                       (if embeddings are images, prototypes should be RNA prototypes)
            is_image: Whether input embeddings are from image modality
        Returns:
            Scalar loss value
        """
        if prototypes is None:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize
        embeddings = F.normalize(embeddings, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)

        # Similarity to all prototypes [B, num_tissues]
        sim = torch.matmul(embeddings, prototypes.T) / self.temperature

        # Classification loss: predict correct tissue from cross-modal prototypes
        loss = F.cross_entropy(sim, tissue_ids)

        return loss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for regularization.

    Decodes the fused embedding back to original modality dimensions.
    Prevents representation collapse and encourages the fused embedding
    to preserve information from all modalities.
    """

    def __init__(self, hidden_dim: int, output_dims: Dict[str, int]):
        """
        Args:
            hidden_dim: Dimension of fused embedding
            output_dims: Dictionary mapping modality name to original dimension
                        e.g., {'image': 512, 'rna': 256}
        """
        super().__init__()
        self.decoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            )
            for name, dim in output_dims.items()
        })

    def forward(
        self,
        fused: torch.Tensor,                        # [B, D] where D=hidden_dim
        original_embeds: Dict[str, torch.Tensor]    # {modality: [B, ORIGINAL_dim]}
    ) -> torch.Tensor:
        """
        Args:
            fused: Fused embedding from Q-Former [B, hidden_dim]
            original_embeds: Dictionary of ORIGINAL (pre-projection) embeddings per modality
                           Must be in original input dimensions (e.g., 512 for image, 256 for RNA)
        Returns:
            Average MSE reconstruction loss
        """
        total_loss = 0
        count = 0

        for name, embed in original_embeds.items():
            if name in self.decoders:
                # Decode fused embedding back to original modality dimension
                recon = self.decoders[name](fused)  # [B, original_dim]
                # Verify dimensions match
                if recon.shape != embed.shape:
                    raise ValueError(
                        f"Reconstruction dimension mismatch: decoder output {recon.shape} "
                        f"vs original embedding {embed.shape} for modality '{name}'. "
                        f"Ensure original_embeds contains pre-projection embeddings "
                        f"(original dimensions), not projected ones."
                    )
                total_loss += F.mse_loss(recon, embed)
                count += 1

        return total_loss / max(count, 1)


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality loss for modality disentanglement.

    Encourages different modality embeddings to be orthogonal (independent),
    which helps the model learn complementary information from each modality
    rather than redundant representations.
    """

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: Dictionary of modality embeddings {name: [B, D]}
        Returns:
            Scalar orthogonality loss
        """
        keys = list(embeddings.keys())

        if len(keys) < 2:
            device = list(embeddings.values())[0].device
            return torch.tensor(0.0, device=device)

        total_loss = 0
        count = 0

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                # Normalize embeddings
                e1 = F.normalize(embeddings[keys[i]], dim=-1)
                e2 = F.normalize(embeddings[keys[j]], dim=-1)

                # Minimize absolute cosine similarity (want orthogonal)
                cos_sim = torch.abs(torch.sum(e1 * e2, dim=-1))
                total_loss += cos_sim.mean()
                count += 1

        return total_loss / max(count, 1)


class DrugCellLineContrastiveLoss(nn.Module):
    """
    Contrastive loss between drug and cell-line embeddings.

    Positive pairs: drug + its corresponding cell-line (from IC50 data)
    Negative pairs: drug + random other cell-lines in batch

    This teaches Q-Former that drugs and cell-lines have structured
    relationships before IC50 regression. It bridges the gap between
    pretraining (image-RNA) and fine-tuning (drug-cellline).

    Temperature controls how sharp the similarity distributions are.
    Lower temperature = sharper distributions = harder contrastive learning.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        drug_embeds: torch.Tensor,      # [B, D] - projected drug embeddings
        cellline_embeds: torch.Tensor   # [B, D] - projected cell-line embeddings
    ) -> torch.Tensor:
        """
        Args:
            drug_embeds: Projected drug embeddings [B, D]
            cellline_embeds: Projected cell-line embeddings [B, D]
        Returns:
            Scalar contrastive loss
        """
        # Normalize embeddings
        drug_embeds = F.normalize(drug_embeds, dim=-1)
        cellline_embeds = F.normalize(cellline_embeds, dim=-1)

        # Similarity matrix [B, B]
        # Each row i: similarity of drug_i to all cell-lines
        sim = torch.matmul(drug_embeds, cellline_embeds.T) / self.temperature

        # Labels are diagonal (positive pairs: drug_i should match cellline_i)
        labels = torch.arange(drug_embeds.shape[0], device=drug_embeds.device)

        # Symmetric loss: drug->cellline AND cellline->drug
        loss_d2cl = F.cross_entropy(sim, labels)
        loss_cl2d = F.cross_entropy(sim.T, labels)

        return (loss_d2cl + loss_cl2d) / 2


class MaskedModalityPredictionLoss(nn.Module):
    """
    Masked modality prediction: predict drug embedding from cell-line RNA (and vice versa).

    This is a form of denoising autoencoder where:
    - Input: cell-line RNA embedding
    - Target: drug embedding

    If Q-Former can predict drug from cell-line, it has learned meaningful
    drug-cellline relationships from the IC50 training data.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Predict drug from cell-line
        self.drug_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        cellline_embeds: torch.Tensor,   # [B, D]
        drug_embeds: torch.Tensor        # [B, D]
    ) -> torch.Tensor:
        """Predict drug from cell-line embedding."""
        predicted_drug = self.drug_predictor(cellline_embeds)
        # MSE loss between predicted and actual drug embeddings
        return F.mse_loss(predicted_drug, drug_embeds)


class GastroTransformerLoss(nn.Module):
    """
    Combined loss for Gastro-Transformer training.

    Implements the hybrid loss strategy:
        LOSS = λ1 * L_intra + λ2 * L_cross + λ3 * L_proto + λ4 * L_recon + λ5 * L_ortho
               + L_ic50 + L_cancer

    The weights (λ) are configured in GastroTransformerConfig.
    """

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config

        # Pre-training losses
        self.intra_loss = IntraModalContrastiveLoss(temperature=config.temperature)
        self.cross_loss = CrossModalContrastiveLoss(temperature=config.temperature)
        self.proto_loss = PrototypicalAlignmentLoss(temperature=0.1)
        self.recon_loss = ReconstructionLoss(
            config.hidden_dim,
            {'image': config.image_dim, 'rna': config.rna_dim}
        )
        self.ortho_loss = OrthogonalityLoss()

        # Task-specific losses
        self.ic50_loss = nn.MSELoss()
        self.cancer_loss = nn.CrossEntropyLoss()
        self.tissue_loss = nn.CrossEntropyLoss()
        self.drug_class_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        # Labels
        tissue_labels: Optional[torch.Tensor] = None,
        ic50_labels: Optional[torch.Tensor] = None,
        cancer_labels: Optional[torch.Tensor] = None,
        drug_class_labels: Optional[torch.Tensor] = None,
        # Paired data flag
        is_paired: bool = False,
        # Prototypes for unpaired data
        image_prototypes: Optional[torch.Tensor] = None,
        rna_prototypes: Optional[torch.Tensor] = None,
        # Original embeddings (for reconstruction)
        original_embeds: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all relevant losses based on available data.

        Args:
            model_outputs: Dictionary from model forward pass
            tissue_labels: Tissue type labels [B]
            ic50_labels: IC50 values [B]
            cancer_labels: Cancer type labels [B]
            drug_class_labels: Drug mechanism of action labels [B]
            is_paired: Whether this batch contains paired data
            image_prototypes: Image prototypes for prototypical loss
            rna_prototypes: RNA prototypes for prototypical loss
            original_embeds: Original embeddings for reconstruction loss

        Returns:
            Dictionary of individual losses and weighted total
        """
        losses = {}
        projected = model_outputs.get('projected', {})

        # =======================================================================
        # 1. INTRA-MODAL CONTRASTIVE LOSSES (for both paired and unpaired)
        # =======================================================================
        if 'image' in projected and tissue_labels is not None:
            losses['intra_image'] = self.intra_loss(projected['image'], tissue_labels)

        if 'rna' in projected and tissue_labels is not None:
            losses['intra_rna'] = self.intra_loss(projected['rna'], tissue_labels)

        # =======================================================================
        # 2. CROSS-MODAL CONTRASTIVE LOSS (only for paired data)
        # =======================================================================
        if is_paired and 'image' in projected and 'rna' in projected:
            losses['cross_modal'] = self.cross_loss(projected['image'], projected['rna'])

        # =======================================================================
        # 3. PROTOTYPICAL ALIGNMENT LOSS (paired AND unpaired data)
        # =======================================================================
        # H2 fix: previously only applied to unpaired data. Paired embeddings also
        # need to stay near prototype centroids — without this, paired data can drift
        # away from the prototypes that guide unpaired data, creating a growing
        # divergence between the anchor space and the unpaired alignment space.
        if 'image' in projected and tissue_labels is not None and rna_prototypes is not None:
            losses['proto_image'] = self.proto_loss(
                projected['image'], tissue_labels, rna_prototypes, is_image=True
            )

        if 'rna' in projected and tissue_labels is not None and image_prototypes is not None:
            losses['proto_rna'] = self.proto_loss(
                projected['rna'], tissue_labels, image_prototypes, is_image=False
            )

        # =======================================================================
        # 4. RECONSTRUCTION LOSS
        # =======================================================================
        if original_embeds is not None and 'fused_embedding' in model_outputs:
            losses['reconstruction'] = self.recon_loss(
                model_outputs['fused_embedding'], original_embeds
            )

        # =======================================================================
        # 5. ORTHOGONALITY LOSS — DISABLED (H6 fix)
        # =======================================================================
        # H6: OrthogonalityLoss pushes image/RNA embeddings toward cosine_sim=0,
        # directly opposing the CrossModalContrastiveLoss which pushes paired
        # image+RNA embeddings toward cosine_sim=1. This injects a persistent
        # counter-gradient that slows cross-modal alignment — especially harmful
        # with only 10-50 paired samples where every gradient step matters.
        # lambda_ortho is preserved in config for potential future disentanglement
        # experiments using separate shared/private subspaces.
        # losses['orthogonality'] = self.ortho_loss(projected)  # disabled

        # =======================================================================
        # 6. TASK-SPECIFIC LOSSES
        # =======================================================================
        # IC50 regression
        if ic50_labels is not None and 'ic50_pred' in model_outputs:
            losses['ic50'] = self.ic50_loss(model_outputs['ic50_pred'], ic50_labels)

        # Cancer type classification
        if cancer_labels is not None and 'cancer_logits' in model_outputs:
            losses['cancer'] = self.cancer_loss(model_outputs['cancer_logits'], cancer_labels)

        # Tissue type classification (auxiliary task during pre-training)
        if tissue_labels is not None and 'tissue_logits' in model_outputs:
            losses['tissue'] = self.tissue_loss(model_outputs['tissue_logits'], tissue_labels)

        # Drug class (MoA) classification
        if drug_class_labels is not None and 'drug_logits' in model_outputs:
            losses['drug_class'] = self.drug_class_loss(
                model_outputs['drug_logits'], drug_class_labels
            )

        # =======================================================================
        # WEIGHTED TOTAL LOSS
        # =======================================================================
        total = torch.tensor(0.0, device=model_outputs['fused_embedding'].device)

        # Pre-training losses with configured weights
        total = total + self.config.lambda_intra * (
            losses.get('intra_image', 0) + losses.get('intra_rna', 0)
        )
        total = total + self.config.lambda_cross * losses.get('cross_modal', 0)
        total = total + self.config.lambda_proto * (
            losses.get('proto_image', 0) + losses.get('proto_rna', 0)
        )
        total = total + self.config.lambda_recon * losses.get('reconstruction', 0)
        total = total + self.config.lambda_ortho * losses.get('orthogonality', 0)

        # Task losses (weight = 1.0)
        total = total + losses.get('ic50', 0)
        total = total + losses.get('cancer', 0)
        total = total + 0.5 * losses.get('tissue', 0)  # Auxiliary task, lower weight
        total = total + losses.get('drug_class', 0)

        losses['total'] = total

        return losses


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def compute_ic50_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """Compute IC50 regression metrics."""
    from scipy import stats as _stats

    mse = F.mse_loss(predictions, targets).item()
    mae = F.l1_loss(predictions, targets).item()

    # Pearson correlation
    pred_centered = predictions - predictions.mean()
    target_centered = targets - targets.mean()
    corr = (pred_centered * target_centered).sum() / (
        pred_centered.norm() * target_centered.norm() + 1e-8
    )

    # Spearman rank correlation
    spearman_r, spearman_p = _stats.spearmanr(
        predictions.detach().cpu().numpy(),
        targets.detach().cpu().numpy()
    )

    # R-squared (coefficient of determination)
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        'mse': mse,
        'rmse': mse ** 0.5,
        'mae': mae,
        'pearson_r': corr.item(),
        'spearman_r': float(spearman_r),
        'r2': r2.item()
    }


# =============================================================================
# NOTES FOR USER - LOSS CONFIGURATION
# =============================================================================
"""
LOSS WEIGHT TUNING GUIDE:
=========================

The default weights are based on the implementation guide's recommendations
for training with limited paired data (10-50 samples):

    lambda_intra = 1.0   # Base weight for intra-modal learning
    lambda_cross = 2.0   # HIGHER weight because paired data is precious
    lambda_proto = 0.5   # Moderate weight for prototype-based extension
    lambda_recon = 0.1   # Low weight, mainly for regularization
    lambda_ortho = 0.05  # Very low weight, encourages diversity

TUNING RECOMMENDATIONS:

1. If cross-modal alignment is poor:
   - Increase lambda_cross (try 3.0 or 4.0)
   - Ensure paired data is loaded correctly

2. If model overfits to paired data:
   - Decrease lambda_cross
   - Increase lambda_recon and lambda_ortho

3. If intra-modal clustering is poor:
   - Increase lambda_intra
   - Verify tissue labels are correct

4. If IC50 predictions are poor:
   - During fine-tuning, the IC50 loss has weight 1.0
   - Consider unfreezing more layers during fine-tuning

TEMPERATURE:
============
- temperature = 0.07 is standard for contrastive learning (CLIP, etc.)
- Lower temperature makes the loss more sensitive to hard negatives
- Higher temperature makes the loss smoother
"""
