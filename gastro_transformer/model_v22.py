"""
Gastro-Transformer v2.2: Feature-Based Cell Line Encoder

This model replaces the learnable cell line ID embeddings with a feature-based approach.
Key difference from v2 (model.py):
- No learnable cellline_embed table
- Uses cell line RNA embeddings + cancer type + tissue type as features
- Projects features to hidden dimension - generalizes to ANY cell line

This enables true generalization to unseen cell lines (cold start).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from .config import GastroTransformerConfig


class ModalityProjector(nn.Module):
    """
    Project each modality embedding to common hidden dimension.

    Two-layer MLP with LayerNorm and GELU activation.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, input_dim] or [B, N, input_dim]
        Returns:
            Projected tensor of shape [B, output_dim] or [B, N, output_dim]
        """
        return self.proj(x)


class QFormerBlock(nn.Module):
    """
    Single Q-Former block with self-attention on queries and
    cross-attention to modality embeddings.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,          # [B, N, D]
        key_value: torch.Tensor,       # [B, M, D]
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention on queries
        residual = query
        query = self.self_attn_norm(query)
        attn_out, _ = self.self_attn(query, query, query)
        query = residual + attn_out

        # Cross-attention to modality tokens
        residual = query
        query = self.cross_attn_norm(query)
        attn_out, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=key_padding_mask)
        query = residual + attn_out

        # Feed-forward
        residual = query
        query = self.ffn_norm(query)
        ffn_out = self.ffn(query)
        query = residual + ffn_out

        return query


class QFormer(nn.Module):
    """
    Q-Former: Learnable query tokens that attend to modality embeddings.

    Uses num_query_tokens learnable queries that undergo L layers of:
    1. Self-attention among queries
    2. Cross-attention to modality token embeddings
    3. Feed-forward network
    """

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_query_tokens, config.hidden_dim) * 0.02
        )

        # Q-Former blocks
        self.blocks = nn.ModuleList([
            QFormerBlock(config.hidden_dim, config.qformer_heads, config.dropout)
            for _ in range(config.qformer_layers)
        ])

        self.query_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        modality_embeds: torch.Tensor,  # [B, M, D] - modality token embeddings
        modality_mask: Optional[torch.Tensor] = None  # [B, M] - padding mask
    ) -> torch.Tensor:
        """
        Args:
            modality_embeds: Modality embeddings [B, M, D]
            modality_mask: Padding mask [B, M]
        Returns:
            Query outputs [B, num_query_tokens, D]
        """
        batch_size = modality_embeds.size(0)

        # Initialize query tokens
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)  # [B, N, D]

        # Apply Q-Former blocks
        for block in self.blocks:
            query_tokens = block(query_tokens, modality_embeds, modality_mask)

        query_tokens = self.query_norm(query_tokens)

        return query_tokens


class FeatureBasedCellEncoder(nn.Module):
    """
    Feature-based cell line encoder (v2.2).

    Instead of learnable cell line ID embeddings, this encoder uses:
    - Cell line RNA embeddings (gene expression)
    - Cancer type embeddings (biological prior)
    - Tissue type embeddings (biological prior)

    This enables TRUE generalization to unseen cell lines (cold start).
    Unlike v2 which clamps unseen IDs to last trained embedding.

    Input features:
    - RNA: [B, rna_dim] (256) - gene expression
    - cancer_type_id: [B] - cancer type category
    - tissue_id: [B] - tissue of origin

    Output: [B, hidden_dim] - cell line representation
    """

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config

        # Cancer type embeddings (biological prior)
        self.cancer_type_embed = nn.Embedding(
            config.num_cancer_types, config.hidden_dim
        )

        # Tissue type embeddings (biological prior)
        self.tissue_embed = nn.Embedding(
            config.num_tissue_types, config.hidden_dim
        )

        # RNA projector: maps RNA dim -> hidden dim
        self.rna_projector = nn.Sequential(
            nn.Linear(config.rna_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )

        # Fallback projector for when RNA is NOT available
        # Projects tissue + cancer embeddings to hidden dim
        self.fallback_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )

        # Final fusion layer: RNA projector output + cancer + tissue
        self.final_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )

    def forward(
        self,
        cancer_type_ids: Optional[torch.Tensor] = None,  # [B]
        tissue_ids: Optional[torch.Tensor] = None,       # [B]
        rna_embeds: Optional[torch.Tensor] = None,      # [B, rna_dim]
        rna_available: Optional[torch.Tensor] = None    # [B] bool
    ) -> torch.Tensor:
        """
        Args:
            cancer_type_ids: Cancer type IDs [B]
            tissue_ids: Tissue type IDs [B]
            rna_embeds: RNA expression embeddings [B, rna_dim]
            rna_available: Boolean mask [B], True = RNA available

        Returns:
            Cell line embedding [B, hidden_dim]
        """
        batch_size = cancer_type_ids.size(0) if cancer_type_ids is not None else rna_embeds.size(0)

        # Initialize feature components
        cancer_emb = None
        tissue_emb = None
        rna_proj = None

        # Cancer type embedding
        if cancer_type_ids is not None:
            max_idx = self.cancer_type_embed.num_embeddings - 1
            safe_ids = cancer_type_ids.clamp(0, max_idx)
            cancer_emb = self.cancer_type_embed(safe_ids)  # [B, D]

        # Tissue type embedding
        if tissue_ids is not None:
            max_idx = self.tissue_embed.num_embeddings - 1
            safe_ids = tissue_ids.clamp(0, max_idx)
            tissue_emb = self.tissue_embed(safe_ids)  # [B, D]

        # RNA projection
        if rna_embeds is not None:
            rna_proj = self.rna_projector(rna_embeds)  # [B, D]

        # Fusion strategy based on what's available
        if rna_embeds is not None and rna_available is not None:
            # Selective fusion: use RNA where available, fallback where not
            if rna_available.all():
                # All have RNA - fuse all three
                features = torch.cat([rna_proj, cancer_emb, tissue_emb], dim=-1)
                cellline_emb = self.final_fusion(features)
            else:
                # Some missing RNA - use fallback for those
                # RNA fusion
                rna_features = torch.cat([rna_proj, cancer_emb, tissue_emb], dim=-1)
                rna_fused = self.final_fusion(rna_features)

                # Fallback fusion (no RNA)
                fallback_features = torch.cat([cancer_emb, tissue_emb], dim=-1)
                fallback_emb = self.fallback_fusion(fallback_features)

                # Select based on RNA availability
                mask = rna_available.unsqueeze(-1).float()  # [B, 1]
                cellline_emb = mask * rna_fused + (1 - mask) * fallback_emb
        elif rna_embeds is not None:
            # RNA provided, no mask - fuse all
            features = torch.cat([rna_proj, cancer_emb, tissue_emb], dim=-1)
            cellline_emb = self.final_fusion(features)
        else:
            # No RNA - use fallback
            features = torch.cat([cancer_emb, tissue_emb], dim=-1)
            cellline_emb = self.fallback_fusion(features)

        return cellline_emb


class IC50ResidualFusionHead(nn.Module):
    """
    Attention pooling + gated residual fusion for IC50 prediction.
    Similar to v2.1 but works with feature-based cell encoder.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Attention pooling over Q-Former queries
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)

        # Gated residual: balance Q-Former fusion vs raw concatenated features
        self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_sigmoid = nn.Sigmoid()

    def forward(
        self,
        qformer_output: torch.Tensor,  # [B, num_queries, D]
        drug_embed: torch.Tensor,        # [B, D]
        cellline_embed: torch.Tensor    # [B, D]
    ) -> torch.Tensor:
        """
        Args:
            qformer_output: Q-Former output [B, num_queries, D]
            drug_embed: Projected drug embedding [B, D]
            cellline_embed: Cell line embedding [B, D]

        Returns:
            IC50 prediction logits [B]
        """
        # Attention pooling over queries
        Q = self.query_proj(qformer_output)  # [B, N, D]
        K = self.key_proj(qformer_output)    # [B, N, D]
        V = self.value_proj(qformer_output)  # [B, N, D]

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_pooled = torch.matmul(attn_weights, V)  # [B, N, D]
        attn_pooled = attn_pooled.mean(dim=1)  # [B, D]

        # Gated residual fusion
        concat_raw = torch.cat([drug_embed, cellline_embed], dim=-1)  # [B, 2D]
        gate = self.gate_sigmoid(self.gate_fc(concat_raw))  # [B, D]

        # Balance: gate * Q-Former_pooled + (1-gate) * raw_concat
        fused = gate * attn_pooled + (1 - gate) * (drug_embed + cellline_embed) * 0.5

        return fused


class ModalitySlotQFormerV22(nn.Module):
    """
    Gastro-Transformer v2.2: Feature-Based Cell Line Encoder

    Key difference from v2 (model.py):
    - Uses FeatureBasedCellEncoder instead of CellLineEncoder
    - No learnable cellline_embed table - truly generalizable to ANY cell line
    - Cell line representation based on: RNA + cancer type + tissue type

    Architecture:
    - Modality projectors: image, RNA, drug → hidden_dim
    - Typed tokens: add type embedding to each modality
    - Q-Former: cross-attention fusion
    - FeatureBasedCellEncoder: RNA + cancer + tissue → cell line embedding
    - Task heads: tissue, cancer, drug classification, IC50 prediction
    """

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config

        # Modality projectors
        self.projectors = nn.ModuleDict({
            'image': ModalityProjector(config.image_dim, config.hidden_dim, config.dropout),
            'rna': ModalityProjector(config.rna_dim, config.hidden_dim, config.dropout),
            'drug': ModalityProjector(config.drug_dim, config.hidden_dim, config.dropout),
        })

        if config.endo_dim > 0:
            self.projectors['endo'] = ModalityProjector(
                config.endo_dim, config.hidden_dim, config.dropout
            )

        # Modality type embeddings (typed tokens)
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

        # Q-Former for multi-modal fusion
        self.qformer = QFormer(config)

        # Feature-based cell-line encoder (v2.2 - key difference!)
        self.cellline_encoder = FeatureBasedCellEncoder(config)

        # IC50 regression head - takes fused [B, D] only
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

        # Q-detached IC50 head: bypass Q-Former, use drug + cellline embeddings only
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

        # Q-Former with Skip Connection
        self.ic50_head_skip = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
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

        # Prototypes for prototypical alignment
        self.register_buffer('image_prototypes', None)
        self.register_buffer('rna_prototypes', None)
        self.register_buffer('prototypes_initialized', torch.tensor(False))

    def forward(
        self,
        # Any combination of these (at least one required):
        image_embeds: Optional[torch.Tensor] = None,       # [B, 512] or [B, N, 512]
        rna_embeds: Optional[torch.Tensor] = None,         # [B, 256]
        drug_embeds: Optional[torch.Tensor] = None,        # [B, 768]
        endo_embeds: Optional[torch.Tensor] = None,        # [B, endo_dim]
        # Cell-line inputs (for IC50) - FEATURE BASED (v2.2):
        # NOTE: No cellline_ids - uses features instead!
        cancer_type_ids: Optional[torch.Tensor] = None,     # [B] — integer IDs
        tissue_ids: Optional[torch.Tensor] = None,         # [B] — integer IDs
        cellline_rna_embeds: Optional[torch.Tensor] = None, # [B, 256] — cell-line RNA
        rna_available: Optional[torch.Tensor] = None,       # [B] bool — RNA availability mask
        # Output control:
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ModalitySlotQFormerV22.

        Key difference from v2:
        - No cellline_ids parameter needed
        - Uses feature-based cellline_encoder

        Args:
            image_embeds: Image embeddings [B, 512] or [B, N, 512]
            rna_embeds: RNA embeddings [B, 256]
            drug_embeds: Drug embeddings [B, 768]
            endo_embeds: Endoscopy embeddings [B, endo_dim]
            cancer_type_ids: Cancer type IDs [B]
            tissue_ids: Tissue type IDs [B]
            cellline_rna_embeds: Cell-line RNA embeddings [B, 256]
            rna_available: Boolean mask [B], True = cell-line has RNA data
            return_embeddings: Whether to return projected per-modality embeddings

        Returns:
            Dictionary containing:
                - fused_embedding: Fused representation [B, D]
                - projected: Dict of per-modality projected embeddings
                - tissue_logits: Tissue type classification logits
                - cancer_logits: Cancer type classification logits
                - drug_logits: Drug class logits
                - ic50_pred: IC50 prediction
        """
        # STEP 1: Build modality token list
        tokens = []
        projected = {}

        # Image tokens
        if image_embeds is not None:
            if image_embeds.dim() == 2:
                image_embeds = image_embeds.unsqueeze(1)  # [B, 1, 512]
            img_proj = self.projectors['image'](image_embeds)  # [B, N, D]
            img_proj = img_proj + self.modality_type_embeddings['image']
            tokens.append(img_proj)
            projected['image'] = img_proj.mean(dim=1)  # [B, D]

        # RNA tokens
        if rna_embeds is not None:
            rna_proj = self.projectors['rna'](rna_embeds).unsqueeze(1)  # [B, 1, D]
            rna_proj = rna_proj + self.modality_type_embeddings['rna']
            tokens.append(rna_proj)
            projected['rna'] = rna_proj.squeeze(1)

        # Drug token
        if drug_embeds is not None:
            drug_proj = self.projectors['drug'](drug_embeds).unsqueeze(1)  # [B, 1, D]
            drug_proj = drug_proj + self.modality_type_embeddings['drug']
            tokens.append(drug_proj)
            projected['drug'] = drug_proj.squeeze(1)

        # Cell-line token (FEATURE BASED - v2.2)
        # Uses: cancer_type + tissue + RNA (instead of cellline_id)
        if self.cellline_encoder is not None:
            cl_emb = self.cellline_encoder(
                cancer_type_ids=cancer_type_ids,
                tissue_ids=tissue_ids,
                rna_embeds=cellline_rna_embeds,
                rna_available=rna_available
            ).unsqueeze(1)  # [B, 1, D]
            cl_emb = cl_emb + self.modality_type_embeddings['cellline']
            tokens.append(cl_emb)
            projected['cellline'] = cl_emb.squeeze(1)

        # Endoscopy token (if enabled)
        if endo_embeds is not None and 'endo' in self.projectors:
            endo_proj = self.projectors['endo'](endo_embeds).unsqueeze(1)
            endo_proj = endo_proj + self.modality_type_embeddings['endo']
            tokens.append(endo_proj)

        # STEP 2: Stack tokens and run Q-Former
        if tokens:
            modality_tokens = torch.cat(tokens, dim=1)  # [B, M, D]
            # No padding needed - we don't have padding in this setup
            query_outputs = self.qformer(modality_tokens, modality_mask=None)
        else:
            raise ValueError("At least one modality must be provided")

        # Pool query outputs for classification
        fused = query_outputs.mean(dim=1)  # [B, D]

        # STEP 3: Task-specific outputs
        outputs = {'fused_embedding': fused}

        if return_embeddings:
            outputs['projected'] = projected

        # Classification heads (on pooled Q-Former output)
        if image_embeds is not None or rna_embeds is not None:
            outputs['tissue_logits'] = self.tissue_head(fused)
            outputs['cancer_logits'] = self.cancer_head(fused)

        if drug_embeds is not None:
            outputs['drug_logits'] = self.drug_class_head(fused)

        # IC50 prediction (requires drug + cell-line)
        if drug_embeds is not None and self.cellline_encoder is not None:
            # Use feature-based cell line embedding
            if self.config.use_qformer_for_ic50:
                if self.config.use_ic50_attn_pool:
                    # Attention pool + gated fusion
                    fused_ic50 = self.ic50_attn_pool_head(
                        query_outputs, projected['drug'], projected['cellline']
                    )
                    outputs['ic50_pred'] = self.ic50_head(fused_ic50).squeeze(-1)
                elif self.config.use_ic50_skip_connection:
                    # Skip connection: concat fused + drug + cellline
                    skip_concat = torch.cat([fused, projected['drug'], projected['cellline']], dim=-1)
                    outputs['ic50_pred'] = self.ic50_head_skip(skip_concat).squeeze(-1)
                else:
                    outputs['ic50_pred'] = self.ic50_head(fused).squeeze(-1)
            else:
                # Q-detached: bypass Q-Former, use projected drug + cellline
                drug_cl_concat = torch.cat(
                    [projected['drug'], projected['cellline']], dim=-1
                )
                outputs['ic50_pred'] = self.ic50_head_detached(drug_cl_concat).squeeze(-1)

        return outputs

    def initialize_prototypes(self, paired_data: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]):
        """
        Initialize prototypes from paired patient data.

        Args:
            paired_data: Dict mapping cancer_type_id to list of (image_embed, rna_embed) pairs
        """
        if self.prototypes_initialized:
            return

        all_images = []
        all_rnas = []
        for cancer_type, pairs in paired_data.items():
            for img_emb, rna_emb in pairs:
                all_images.append(img_emb)
                all_rnas.append(rna_emb)

        if all_images:
            self.image_prototypes = torch.stack(all_images).mean(dim=0, keepdim=True)
            self.rna_prototypes = torch.stack(all_rnas).mean(dim=0, keepdim=True)
            self.prototypes_initialized = torch.tensor(True)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
