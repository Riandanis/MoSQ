"""
ssGSEA Model Extension for ModalitySlotQFormer.

Extends ModalitySlotQFormer to support ssGSEA as a 4th cell-line Q-Former token:
[cancer, tissue, RNA-BERT, ssGSEA] = 4 cell-line tokens + 1 drug token = 5 total KV tokens.

Key design:
- ssGSEA gets its own typed token embedding (type='ssgsea')
- ssGSEA projector: 768 → hidden_dim
- Learnable fallback for missing ssGSEA (same pattern as RNA fallback)
- Forward pass accepts ssgsea_embeds, ssgsea_available → 5 KV tokens to Q-Former
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .model import (
    ModalityProjector,
    QFormer,
    MultiTokenCellLineEncoder,
    IC50ResidualFusionHead,
    GastroTransformerConfig,
)


class MultiTokenCellLineEncoderWithSsgsea(nn.Module):
    """
    Multi-token cell-line encoder WITH ssGSEA support.

    Extends MultiTokenCellLineEncoder to output 4 tokens instead of 3:
    [cancer, tissue, RNA-BERT, ssGSEA] → [B, 4, D]

    Falls back to 3 tokens when ssGSEA is unavailable (all zeros).
    """

    def __init__(self, config: GastroTransformerConfig, ssgsea_dim: int = 768):
        super().__init__()
        self.config = config
        self.ssgsea_dim = ssgsea_dim

        # Cancer and tissue embeddings (same as parent)
        self.cancer_type_embed = nn.Embedding(
            config.num_cancer_types, config.hidden_dim
        )
        self.tissue_embed = nn.Embedding(
            config.num_tissue_types, config.hidden_dim
        )

        # RNA projector: raw RNA → hidden_dim (same as parent)
        self.rna_projector = nn.Sequential(
            nn.Linear(config.rna_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )

        # RNA fallback (same as parent)
        self.rna_fallback = nn.Parameter(torch.randn(1, config.hidden_dim) * 0.02)

        # ssGSEA projector: ssGSEA (768d) → hidden_dim
        self.ssgsea_projector = nn.Sequential(
            nn.Linear(ssgsea_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )

        # ssGSEA fallback (learnable) for missing data
        self.ssgsea_fallback = nn.Parameter(torch.randn(1, config.hidden_dim) * 0.02)

    def forward(
        self,
        cancer_type_ids: Optional[torch.Tensor] = None,
        tissue_ids: Optional[torch.Tensor] = None,
        rna_embeds: Optional[torch.Tensor] = None,
        rna_available: Optional[torch.Tensor] = None,
        ssgsea_embeds: Optional[torch.Tensor] = None,
        ssgsea_available: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            tokens: [B, 4, D] — cancer, tissue, rna, ssgsea tokens

        Args:
            cancer_type_ids: Cancer type IDs [B]
            tissue_ids: Tissue type IDs [B]
            rna_embeds: RNA-BERT embeddings [B, rna_dim]
            rna_available: Boolean mask for RNA availability [B]
            ssgsea_embeds: ssGSEA pathway scores [B, ssgsea_dim]
            ssgsea_available: Boolean mask for ssGSEA availability [B]
        """
        batch_size = cancer_type_ids.size(0) if cancer_type_ids is not None else (
            rna_embeds.size(0) if rna_embeds is not None else ssgsea_embeds.size(0)
        )
        device = (cancer_type_ids.device if cancer_type_ids is not None else
                  rna_embeds.device if rna_embeds is not None else ssgsea_embeds.device)

        # Cancer token
        if cancer_type_ids is not None:
            max_idx = self.cancer_type_embed.num_embeddings - 1
            safe_ids = cancer_type_ids.clamp(0, max_idx)
            cancer_tok = self.cancer_type_embed(safe_ids)  # [B, D]
        else:
            cancer_tok = torch.zeros(batch_size, self.config.hidden_dim, device=device)

        # Tissue token
        if tissue_ids is not None:
            max_idx = self.tissue_embed.num_embeddings - 1
            safe_ids = tissue_ids.clamp(0, max_idx)
            tissue_tok = self.tissue_embed(safe_ids)  # [B, D]
        else:
            tissue_tok = torch.zeros(batch_size, self.config.hidden_dim, device=device)

        # RNA token (with fallback for missing)
        if rna_embeds is not None:
            rna_tok = self.rna_projector(rna_embeds)  # [B, D]
            if rna_available is not None and not rna_available.all():
                fallback = self.rna_fallback.expand(batch_size, -1)
                mask = rna_available.unsqueeze(-1).float()
                rna_tok = mask * rna_tok + (1 - mask) * fallback
        else:
            rna_tok = self.rna_fallback.expand(batch_size, -1)

        # ssGSEA token (with fallback for missing)
        if ssgsea_embeds is not None:
            ssgsea_tok = self.ssgsea_projector(ssgsea_embeds)  # [B, D]
            if ssgsea_available is not None and not ssgsea_available.all():
                fallback = self.ssgsea_fallback.expand(batch_size, -1)
                mask = ssgsea_available.unsqueeze(-1).float()
                ssgsea_tok = mask * ssgsea_tok + (1 - mask) * fallback
        else:
            ssgsea_tok = self.ssgsea_fallback.expand(batch_size, -1)

        # Stack: [B, 4, D]
        tokens = torch.stack([cancer_tok, tissue_tok, rna_tok, ssgsea_tok], dim=1)
        return tokens


class ModalitySlotQFormerWithSsgsea(nn.Module):
    """
    ModalitySlotQFormer extended with ssGSEA support.

    Extends ModalitySlotQFormer to accept ssGSEA as a 4th cell-line token:
    - Adds ssgsea_projector (ModalityProjector: ssgsea_dim → hidden_dim)
    - Adds modality_type_embeddings['ssgsea']
    - Extends MultiTokenCellLineEncoder → MultiTokenCellLineEncoderWithSsgsea
    - Forward pass accepts ssgsea_embeds, ssgsea_available → 5 KV tokens to Q-Former

    Architecture:
        Drug (1) → Q-Former → IC50
        CellLine (4 tokens): Cancer + Tissue + RNA-BERT + ssGSEA
        Total: 5 KV tokens to Q-Former (drug + 4 cell-line)
    """

    def __init__(
        self,
        config: GastroTransformerConfig,
        ssgsea_dim: int = 768,
    ):
        # Initialize parent ModalitySlotQFormer with MultiToken enabled
        # (We manually construct to avoid parent __init__ side effects)
        super().__init__()
        self.config = config
        self.ssgsea_dim = ssgsea_dim

        # =========================================================================
        # Modality projectors (from parent pattern)
        # =========================================================================
        self.projectors = nn.ModuleDict({
            'image': ModalityProjector(config.image_dim, config.hidden_dim, config.dropout),
            'rna': ModalityProjector(config.rna_dim, config.hidden_dim, config.dropout),
            'drug': ModalityProjector(config.drug_dim, config.hidden_dim, config.dropout),
        })

        # Optional endoscopy projector
        if config.endo_dim > 0:
            self.projectors['endo'] = ModalityProjector(
                config.endo_dim, config.hidden_dim, config.dropout
            )

        # ssGSEA projector (NEW)
        self.projectors['ssgsea'] = ModalityProjector(
            ssgsea_dim, config.hidden_dim, config.dropout
        )

        # =========================================================================
        # Modality type embeddings (typed tokens)
        # =========================================================================
        self.modality_type_embeddings = nn.ParameterDict({
            'image': nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02),
            'rna': nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02),
            'drug': nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02),
            'cellline': nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02),
        })

        if config.endo_dim > 0:
            self.modality_type_embeddings['endo'] = nn.Parameter(
                torch.randn(1, 1, config.hidden_dim) * 0.02
            )

        # Multi-token cell-line type embeddings (cancer, tissue, cellline_rna)
        if config.use_multitoken_cellline:
            self.modality_type_embeddings['cancer'] = nn.Parameter(
                torch.randn(1, 1, config.hidden_dim) * 0.02
            )
            self.modality_type_embeddings['tissue'] = nn.Parameter(
                torch.randn(1, 1, config.hidden_dim) * 0.02
            )
            self.modality_type_embeddings['cellline_rna'] = nn.Parameter(
                torch.randn(1, 1, config.hidden_dim) * 0.02
            )
            # ssGSEA type embedding (NEW)
            self.modality_type_embeddings['ssgsea'] = nn.Parameter(
                torch.randn(1, 1, config.hidden_dim) * 0.02
            )

        # =========================================================================
        # Q-Former for multi-modal fusion
        # =========================================================================
        self.qformer = QFormer(config)

        # =========================================================================
        # Cell-line encoder with ssGSEA support
        # =========================================================================
        self.use_multitoken_cellline = config.use_multitoken_cellline
        if config.use_multitoken_cellline:
            # Use extended encoder with ssGSEA support
            self.cellline_encoder = MultiTokenCellLineEncoderWithSsgsea(
                config, ssgsea_dim=ssgsea_dim
            )
            self.use_feature_cellline_encoder = True
        else:
            raise ValueError(
                "ModalitySlotQFormerWithSsgsea requires use_multitoken_cellline=True. "
                "ssGSEA integration is only supported with MultiToken cell-line encoding."
            )

        # =========================================================================
        # IC50 regression heads
        # =========================================================================
        self.ic50_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        # Q-detached IC50 head
        self.ic50_head_detached = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        # IC50 attention pooling + gated residual fusion head
        if config.use_ic50_attn_pool:
            self.ic50_attn_pool_head = IC50ResidualFusionHead(
                hidden_dim=config.hidden_dim,
                num_heads=config.qformer_heads,
                dropout=config.dropout,
            )

        # Classification heads
        self.tissue_head = nn.Linear(config.hidden_dim, config.num_tissue_types)
        self.cancer_head = nn.Linear(config.hidden_dim, config.num_cancer_types)
        self.drug_class_head = nn.Linear(config.hidden_dim, config.num_drug_classes)

    def forward(
        self,
        # Any combination of these (at least one required):
        image_embeds: Optional[torch.Tensor] = None,
        rna_embeds: Optional[torch.Tensor] = None,
        drug_embeds: Optional[torch.Tensor] = None,
        endo_embeds: Optional[torch.Tensor] = None,
        # Cell-line inputs (for IC50):
        cellline_ids: Optional[torch.Tensor] = None,
        cancer_type_ids: Optional[torch.Tensor] = None,
        tissue_ids: Optional[torch.Tensor] = None,
        cellline_rna_embeds: Optional[torch.Tensor] = None,
        rna_available: Optional[torch.Tensor] = None,
        # ssGSEA inputs (NEW)
        ssgsea_embeds: Optional[torch.Tensor] = None,
        ssgsea_available: Optional[torch.Tensor] = None,
        # Output control:
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ModalitySlotQFormerWithSsgsea.

        Extends parent ModalitySlotQFormer.forward with:
            - ssgsea_embeds: ssGSEA pathway scores [B, 768]
            - ssgsea_available: Boolean mask [B]

        Returns:
            Dictionary containing all outputs (same as parent)
        """
        # STEP 1: Build modality token list
        tokens = []
        projected = {}

        # Image tokens
        if image_embeds is not None:
            if image_embeds.dim() == 2:
                image_embeds = image_embeds.unsqueeze(1)
            img_proj = self.projectors['image'](image_embeds)
            img_proj = img_proj + self.modality_type_embeddings['image']
            tokens.append(img_proj)
            projected['image'] = img_proj.mean(dim=1)

        # RNA tokens (patient RNA for pretraining)
        if rna_embeds is not None:
            rna_proj = self.projectors['rna'](rna_embeds).unsqueeze(1)
            rna_proj = rna_proj + self.modality_type_embeddings['rna']
            tokens.append(rna_proj)
            projected['rna'] = rna_proj.squeeze(1)

        # Drug token
        if drug_embeds is not None:
            drug_proj = self.projectors['drug'](drug_embeds).unsqueeze(1)
            drug_proj = drug_proj + self.modality_type_embeddings['drug']
            tokens.append(drug_proj)
            projected['drug'] = drug_proj.squeeze(1)

        # Cell-line tokens (MultiToken with ssGSEA: 4 tokens)
        has_cellline = False
        if self.use_multitoken_cellline and self.cellline_encoder is not None:
            if cancer_type_ids is not None or cellline_rna_embeds is not None or ssgsea_embeds is not None:
                cl_tokens = self.cellline_encoder(
                    cancer_type_ids=cancer_type_ids,
                    tissue_ids=tissue_ids,
                    rna_embeds=cellline_rna_embeds,
                    rna_available=rna_available,
                    ssgsea_embeds=ssgsea_embeds,
                    ssgsea_available=ssgsea_available,
                )  # [B, 4, D]

                # Add type embeddings to each token
                cl_tokens[:, 0:1, :] = cl_tokens[:, 0:1, :] + self.modality_type_embeddings['cancer']
                cl_tokens[:, 1:2, :] = cl_tokens[:, 1:2, :] + self.modality_type_embeddings['tissue']
                cl_tokens[:, 2:3, :] = cl_tokens[:, 2:3, :] + self.modality_type_embeddings['cellline_rna']
                cl_tokens[:, 3:4, :] = cl_tokens[:, 3:4, :] + self.modality_type_embeddings['ssgsea']

                tokens.append(cl_tokens)
                projected['cellline'] = cl_tokens.mean(dim=1)
                has_cellline = True

        # Endoscopy tokens
        if endo_embeds is not None and 'endo' in self.projectors:
            endo_proj = self.projectors['endo'](endo_embeds).unsqueeze(1)
            endo_proj = endo_proj + self.modality_type_embeddings['endo']
            tokens.append(endo_proj)
            projected['endo'] = endo_proj.squeeze(1)

        assert len(tokens) > 0, "At least one modality must be provided"
        combined = torch.cat(tokens, dim=1)  # [B, N_total, D]

        # STEP 2: Fusion (Q-Former or simple concatenation)
        all_queries = None
        if self.config.use_qformer:
            if self.config.use_ic50_attn_pool:
                all_queries = self.qformer(combined, modality_mask=None, return_all_queries=True)
                fused = all_queries.mean(dim=1)
            else:
                fused = self.qformer(combined, modality_mask=None)
        else:
            fused = combined.mean(dim=1)

        # STEP 3: Outputs
        outputs = {'fused_embedding': fused}
        if return_embeddings:
            outputs['projected'] = projected

        # Classification heads
        outputs['tissue_logits'] = self.tissue_head(fused)
        outputs['cancer_logits'] = self.cancer_head(fused)

        if 'drug' in projected:
            outputs['drug_logits'] = self.drug_class_head(fused)

        # IC50 head
        if drug_embeds is not None and has_cellline:
            if self.config.use_ic50_attn_pool and all_queries is not None and 'drug' in projected and 'cellline' in projected:
                outputs['ic50_pred'] = self.ic50_attn_pool_head(
                    qformer_queries=all_queries,
                    drug_proj=projected['drug'],
                    cellline_proj=projected['cellline'],
                )
            elif self.config.use_qformer_for_ic50:
                outputs['ic50_pred'] = self.ic50_head(fused).squeeze(-1)
            else:
                if 'drug' in projected and 'cellline' in projected:
                    drug_cl_concat = torch.cat(
                        [projected['drug'], projected['cellline']], dim=-1
                    )
                    outputs['ic50_pred'] = self.ic50_head_detached(drug_cl_concat).squeeze(-1)
                else:
                    outputs['ic50_pred'] = self.ic50_head(fused).squeeze(-1)

        return outputs

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }
