"""
Gastro-Transformer v2: Modality-Slot Q-Former Model Architecture.

Multi-modal foundation model for gastric cancer drug response prediction.
Uses a modality-slot architecture with typed token embeddings and Q-Former fusion.
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

        # Self-attention on queries
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention from queries to modality embeddings
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        # Normalize modality K/V before cross-attention
        self.norm_kv = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,           # [B, Q, D]
        modality_embeds: torch.Tensor,   # [B, N, D]
        modality_mask: Optional[torch.Tensor] = None  # [B, N] True = masked
    ) -> torch.Tensor:
        """
        Args:
            queries: Learnable query tokens [B, Q, D]
            modality_embeds: Concatenated modality embeddings [B, N, D]
            modality_mask: Padding mask for modality embeddings
        Returns:
            Updated query tokens [B, Q, D]
        """
        # Self-attention on queries (pre-norm)
        q = self.norm1(queries)
        q = queries + self.self_attn(q, q, q)[0]

        # Cross-attention to modality embeddings (pre-norm on both Q and K/V)
        q2 = self.norm2(q)
        kv = self.norm_kv(modality_embeds)
        q = q + self.cross_attn(
            q2, kv, kv,
            key_padding_mask=modality_mask
        )[0]

        # Feed-forward (pre-norm)
        q = q + self.ffn(self.norm3(q))

        return q


class QFormer(nn.Module):
    """
    Q-Former: Transformer with learnable query tokens for multi-modal fusion.

    Based on BLIP-2 architecture, adapted for medical multi-modal data.
    """

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_query_tokens, config.hidden_dim) * 0.02
        )

        # Stack of Q-Former blocks
        self.blocks = nn.ModuleList([
            QFormerBlock(config.hidden_dim, config.qformer_heads, config.dropout)
            for _ in range(config.qformer_layers)
        ])

        self.output_proj = None

    def forward(
        self,
        modality_embeds: torch.Tensor,   # [B, N, D]
        modality_mask: Optional[torch.Tensor] = None,
        return_all_queries: bool = False
    ) -> torch.Tensor:
        """
        Args:
            modality_embeds: Concatenated projected modality embeddings [B, N, D]
            modality_mask: Padding mask [B, N]
            return_all_queries: If True, return all query tokens instead of mean pooling
        Returns:
            Fused embedding [B, D] or all queries [B, Q, D] if return_all_queries=True
        """
        B = modality_embeds.shape[0]

        # Expand query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)  # [B, Q, D]

        # Pass through Q-Former blocks
        for block in self.blocks:
            queries = block(queries, modality_embeds, modality_mask)

        if return_all_queries:
            return queries  # [B, Q, D]

        # Mean pooling over query tokens
        output = queries.mean(dim=1)  # [B, D]

        return output


class CellLineEncoder(nn.Module):
    """
    Encode cell-lines for IC50 prediction.

    Uses learnable embeddings for each cell-line, optionally enhanced
    with cancer type priors, tissue-type FiLM modulation, and RNA expression data.
    """

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config

        # Learnable cell-line embeddings
        self.cellline_embed = nn.Embedding(
            config.num_cell_lines, config.hidden_dim
        )

        # Cancer type embeddings (biological prior)
        self.cancer_type_embed = nn.Embedding(
            config.num_cancer_types, config.hidden_dim
        )

        # FiLM tissue-type modulation
        self.tissue_gamma = nn.Embedding(
            config.num_tissue_types, config.hidden_dim
        )
        self.tissue_beta = nn.Embedding(
            config.num_tissue_types, config.hidden_dim
        )

        # Fusion layer when RNA is available
        self.rna_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )

    def forward(
        self,
        cellline_ids: torch.Tensor,                     # [B]
        cancer_type_ids: Optional[torch.Tensor] = None, # [B]
        tissue_ids: Optional[torch.Tensor] = None,      # [B]
        rna_embeds: Optional[torch.Tensor] = None,      # [B, D]
        rna_mask: Optional[torch.Tensor] = None         # [B] bool: True = RNA available
    ) -> torch.Tensor:
        """
        Args:
            cellline_ids: Cell-line IDs [B]
            cancer_type_ids: Cancer type IDs for biological prior [B]
            tissue_ids: Tissue type IDs for FiLM modulation [B]
            rna_embeds: RNA expression embeddings [B, D]
            rna_mask: Boolean mask [B], True where RNA is available. When False,
                      skip RNA fusion for that sample to avoid corrupting with zeros.
        Returns:
            Cell-line embedding [B, D]
        """
        # Safety clamp: prevent out-of-bounds indexing
        max_idx = self.cellline_embed.num_embeddings - 1
        safe_cellline_ids = cellline_ids.clamp(0, max_idx)

        # Base cell-line embedding
        cellline_emb = self.cellline_embed(safe_cellline_ids)  # [B, D]

        # Add cancer type prior if available
        # Fix: cancer_type_id=-1 means unknown — don't add wrong embedding
        if cancer_type_ids is not None:
            valid_cancer = cancer_type_ids >= 0  # [B] bool mask
            if valid_cancer.any():
                max_cancer_idx = self.cancer_type_embed.num_embeddings - 1
                safe_cancer_ids = cancer_type_ids.clamp(0, max_cancer_idx)
                cancer_emb = self.cancer_type_embed(safe_cancer_ids)
                # Zero out contribution for invalid cancer types
                cancer_emb = cancer_emb * valid_cancer.unsqueeze(-1).float()
                cellline_emb = cellline_emb + 0.5 * cancer_emb

        # Apply FiLM tissue-type modulation
        if tissue_ids is not None:
            max_tissue_idx = self.tissue_gamma.num_embeddings - 1
            safe_tissue_ids = tissue_ids.clamp(0, max_tissue_idx)
            gamma = self.tissue_gamma(safe_tissue_ids)  # [B, D]
            beta = self.tissue_beta(safe_tissue_ids)     # [B, D]
            cellline_emb = gamma * cellline_emb + beta   # FiLM modulation

        # Fuse with RNA if available, respecting per-sample RNA mask
        if rna_embeds is not None:
            if rna_mask is not None and not rna_mask.all():
                # Selective fusion: only fuse RNA for samples that have it
                fused = self.rna_fusion(torch.cat([cellline_emb, rna_embeds], dim=-1))
                # Where RNA is available, use fused; where missing, keep raw cellline_emb
                cellline_emb = torch.where(
                    rna_mask.unsqueeze(-1), fused, cellline_emb
                )
            else:
                # All samples have RNA (or no mask provided) — fuse all
                combined = torch.cat([cellline_emb, rna_embeds], dim=-1)
                cellline_emb = self.rna_fusion(combined)

        return cellline_emb


class FeatureBasedCellEncoder(nn.Module):
    """
    Feature-based cell line encoder.

    Instead of learnable cell line ID embeddings, uses:
    - Cell line RNA embeddings (gene expression)
    - Cancer type embeddings (biological prior)
    - Tissue type embeddings (biological prior)

    Enables TRUE generalization to unseen cell lines (cold start).
    """

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config

        self.cancer_type_embed = nn.Embedding(
            config.num_cancer_types, config.hidden_dim
        )
        self.tissue_embed = nn.Embedding(
            config.num_tissue_types, config.hidden_dim
        )
        self.rna_projector = nn.Sequential(
            nn.Linear(config.rna_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        self.fallback_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )
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
        cancer_type_ids: Optional[torch.Tensor] = None,
        tissue_ids: Optional[torch.Tensor] = None,
        rna_embeds: Optional[torch.Tensor] = None,
        rna_available: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = cancer_type_ids.size(0) if cancer_type_ids is not None else rna_embeds.size(0)

        cancer_emb = None
        tissue_emb = None
        rna_proj = None

        if cancer_type_ids is not None:
            max_idx = self.cancer_type_embed.num_embeddings - 1
            safe_ids = cancer_type_ids.clamp(0, max_idx)
            cancer_emb = self.cancer_type_embed(safe_ids)

        if tissue_ids is not None:
            max_idx = self.tissue_embed.num_embeddings - 1
            safe_ids = tissue_ids.clamp(0, max_idx)
            tissue_emb = self.tissue_embed(safe_ids)

        if rna_embeds is not None:
            rna_proj = self.rna_projector(rna_embeds)

        if rna_embeds is not None and rna_available is not None:
            if rna_available.all():
                features = torch.cat([rna_proj, cancer_emb, tissue_emb], dim=-1)
                cellline_emb = self.final_fusion(features)
            else:
                rna_features = torch.cat([rna_proj, cancer_emb, tissue_emb], dim=-1)
                rna_fused = self.final_fusion(rna_features)
                fallback_features = torch.cat([cancer_emb, tissue_emb], dim=-1)
                fallback_emb = self.fallback_fusion(fallback_features)
                mask = rna_available.unsqueeze(-1).float()
                cellline_emb = mask * rna_fused + (1 - mask) * fallback_emb
        elif rna_embeds is not None:
            features = torch.cat([rna_proj, cancer_emb, tissue_emb], dim=-1)
            cellline_emb = self.final_fusion(features)
        else:
            features = torch.cat([cancer_emb, tissue_emb], dim=-1)
            cellline_emb = self.fallback_fusion(features)

        return cellline_emb


class MultiTokenCellLineEncoder(nn.Module):
    """
    Multi-token cell-line encoder for Q-Former.

    Instead of fusing cancer+tissue+RNA into 1 token, decomposes into 3 separate
    typed tokens so Q-Former cross-attention sees 4 KV tokens (drug+cancer+tissue+rna)
    instead of 2 (drug+cellline), making cross-attention actually useful.

    Returns 3 tokens: [B, 3, D] = [cancer_embed, tissue_embed, rna_proj]
    Falls back to 2 tokens [B, 2, D] = [cancer_embed, tissue_embed] when RNA unavailable.
    """

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config

        self.cancer_type_embed = nn.Embedding(
            config.num_cancer_types, config.hidden_dim
        )
        self.tissue_embed = nn.Embedding(
            config.num_tissue_types, config.hidden_dim
        )
        # RNA projector: raw RNA → hidden_dim
        self.rna_projector = nn.Sequential(
            nn.Linear(config.rna_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        # Learnable fallback token for missing RNA
        self.rna_fallback = nn.Parameter(torch.randn(1, config.hidden_dim) * 0.02)

    def forward(
        self,
        cancer_type_ids: Optional[torch.Tensor] = None,
        tissue_ids: Optional[torch.Tensor] = None,
        rna_embeds: Optional[torch.Tensor] = None,
        rna_available: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns:
            tokens: [B, 3, D] — cancer, tissue, rna tokens
        """
        batch_size = cancer_type_ids.size(0) if cancer_type_ids is not None else rna_embeds.size(0)
        device = cancer_type_ids.device if cancer_type_ids is not None else rna_embeds.device

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
                # Use fallback for samples without RNA
                fallback = self.rna_fallback.expand(batch_size, -1)
                mask = rna_available.unsqueeze(-1).float()
                rna_tok = mask * rna_tok + (1 - mask) * fallback
        else:
            rna_tok = self.rna_fallback.expand(batch_size, -1)

        # Stack: [B, 3, D]
        tokens = torch.stack([cancer_tok, tissue_tok, rna_tok], dim=1)
        return tokens


class IC50AttentionPool(nn.Module):
    """
    Attention-pooling over Q-Former query tokens for IC50 prediction.

    A single learnable query attends to all Q-Former output tokens via cross-attention,
    producing a task-specific weighted combination instead of mean-pooling.

    Input:  [B, Q, D]  (Q query tokens from Q-Former)
    Output: [B, D]     (single IC50-relevant representation)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ic50_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

    def forward(self, query_tokens: torch.Tensor) -> torch.Tensor:
        B = query_tokens.shape[0]
        q = self.norm_q(self.ic50_query.expand(B, -1, -1))
        kv = self.norm_kv(query_tokens)
        pooled, _ = self.cross_attn(q, kv, kv)
        return pooled.squeeze(1)


class IC50ResidualFusionHead(nn.Module):
    """
    IC50 prediction head with attention pooling + gated residual fusion.

    Combines:
    1. Attention-pooled Q-Former output (cross-modal context)    [B, D]
    2. Raw drug projection (drug-specific features)              [B, D]
    3. Raw cell-line projection (cell-line-specific features)    [B, D]

    A learned gate balances Q-Former fusion vs raw modality features.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_pool = IC50AttentionPool(hidden_dim, num_heads, dropout)
        self.raw_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, qformer_queries, drug_proj, cellline_proj):
        qf_pooled = self.attn_pool(qformer_queries)
        raw_concat = torch.cat([drug_proj, cellline_proj], dim=-1)
        raw_features = self.raw_proj(raw_concat)
        gate_input = torch.cat([qf_pooled, raw_features], dim=-1)
        g = self.gate(gate_input)
        fused = g * qf_pooled + (1 - g) * raw_features
        return self.head(fused).squeeze(-1)


class ModalitySlotQFormer(nn.Module):
    """
    Gastro-Transformer v2: Modality-Slot Q-Former.

    Architecture:
    1. Each modality is projected and receives a type embedding (typed token)
    2. All typed modality tokens are concatenated into a single sequence
    3. Q-Former processes the sequence and produces a single fused output
    4. Task heads operate on the fused output only

    Key design rules:
    - No boolean flags controlling architecture variants
    - Cell-line RNA uses the shared RNA projector
    - IC50 head takes [B, D] fused output (not concatenated inputs)
    """

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config

        # Modality projectors (trainable)
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

        # Q-Former for multi-modal fusion
        self.qformer = QFormer(config)

        # Cell-line encoder for IC50
        self.use_multitoken_cellline = config.use_multitoken_cellline
        if config.use_multitoken_cellline:
            self.cellline_encoder = MultiTokenCellLineEncoder(config)
            self.use_feature_cellline_encoder = True  # flag for forward pass routing
        elif config.use_feature_cellline_encoder:
            self.cellline_encoder = FeatureBasedCellEncoder(config)
            self.use_feature_cellline_encoder = True
        elif config.use_cellline_embeddings:
            self.cellline_encoder = CellLineEncoder(config)
            self.use_feature_cellline_encoder = False
        else:
            self.cellline_encoder = None
            self.use_feature_cellline_encoder = False

        # IC50 regression head - takes fused [B, D] only (not concatenated)
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
        # Takes [B, 2*hidden_dim] = concatenated projected drug + cellline embeddings
        # NOTE: Same input as Q-Former version for fair comparison (cellline RNA fusion happens inside Q-Former)
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

        # Q-Former with Skip Connection: concatenate fused + projected drug + projected cellline
        # Takes [B, 3*hidden_dim] = fused + drug + cellline
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

        # IC50 attention pooling + gated residual fusion head (v2.1 architecture)
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

        # Prototypes for prototypical alignment (initialized from paired data)
        self.register_buffer('image_prototypes', None)
        self.register_buffer('rna_prototypes', None)
        self.register_buffer('prototypes_initialized', torch.tensor(False))

    def forward(
        self,
        # Any combination of these (at least one required):
        image_embeds: Optional[torch.Tensor] = None,       # [B, 512] or [B, N, 512]
        rna_embeds: Optional[torch.Tensor] = None,         # [B, 256]
        drug_embeds: Optional[torch.Tensor] = None,         # [B, 768]
        endo_embeds: Optional[torch.Tensor] = None,         # [B, endo_dim]
        # Cell-line inputs (for IC50):
        cellline_ids: Optional[torch.Tensor] = None,        # [B] — integer IDs
        cancer_type_ids: Optional[torch.Tensor] = None,     # [B] — integer IDs
        tissue_ids: Optional[torch.Tensor] = None,          # [B] — integer IDs
        cellline_rna_embeds: Optional[torch.Tensor] = None, # [B, 256] — cell-line RNA
        rna_available: Optional[torch.Tensor] = None,     # [B] bool — RNA availability mask
        # Output control:
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ModalitySlotQFormer.

        Args:
            image_embeds: Image embeddings [B, 512] or [B, N, 512]
            rna_embeds: RNA embeddings [B, 256]
            drug_embeds: Drug embeddings [B, 768]
            endo_embeds: Endoscopy embeddings [B, endo_dim]
            cellline_ids: Cell-line IDs [B]
            cancer_type_ids: Cancer type IDs [B]
            tissue_ids: Tissue type IDs [B]
            cellline_rna_embeds: Cell-line RNA embeddings [B, 256]
            rna_available: Boolean mask [B], True where cell-line has RNA data
            return_embeddings: Whether to return projected per-modality embeddings

        Returns:
            Dictionary containing:
                - fused_embedding: Fused representation [B, D]
                - projected: Dict of per-modality projected embeddings (if return_embeddings=True)
                - tissue_logits: Tissue type classification logits
                - cancer_logits: Cancer type classification logits
                - drug_logits: Drug class logits (if drug provided)
                - ic50_pred: IC50 prediction (if drug + cellline provided)
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
            projected['image'] = img_proj.mean(dim=1)  # [B, D] for losses

        # RNA tokens (patient RNA for pretraining)
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

        # Cell-line token (encoded via CellLineEncoder, FeatureBasedCellEncoder, or MultiToken)
        has_cellline = False
        if self.use_multitoken_cellline and self.cellline_encoder is not None:
            # Multi-token: decompose cell-line into 3 separate typed tokens
            if cancer_type_ids is not None or cellline_rna_embeds is not None:
                cl_tokens = self.cellline_encoder(
                    cancer_type_ids=cancer_type_ids,
                    tissue_ids=tissue_ids,
                    rna_embeds=cellline_rna_embeds,
                    rna_available=rna_available
                )  # [B, 3, D]
                # Add type embeddings to each token
                cl_tokens[:, 0:1, :] = cl_tokens[:, 0:1, :] + self.modality_type_embeddings['cancer']
                cl_tokens[:, 1:2, :] = cl_tokens[:, 1:2, :] + self.modality_type_embeddings['tissue']
                cl_tokens[:, 2:3, :] = cl_tokens[:, 2:3, :] + self.modality_type_embeddings['cellline_rna']
                tokens.append(cl_tokens)
                # Mean pool for projected representation (used by gated residual, detached heads)
                projected['cellline'] = cl_tokens.mean(dim=1)
                has_cellline = True
        elif self.use_feature_cellline_encoder and self.cellline_encoder is not None:
            # Feature-based: uses cancer_type + tissue + RNA (no cellline_ids needed)
            if cancer_type_ids is not None or cellline_rna_embeds is not None:
                cl_emb = self.cellline_encoder(
                    cancer_type_ids=cancer_type_ids,
                    tissue_ids=tissue_ids,
                    rna_embeds=cellline_rna_embeds,
                    rna_available=rna_available
                ).unsqueeze(1)  # [B, 1, D]
                cl_emb = cl_emb + self.modality_type_embeddings['cellline']
                tokens.append(cl_emb)
                projected['cellline'] = cl_emb.squeeze(1)
                has_cellline = True
        elif cellline_ids is not None and self.cellline_encoder is not None:
            # ID-based CellLineEncoder: lookup + cancer prior + FiLM + RNA fusion
            rna_for_cl = None
            if cellline_rna_embeds is not None:
                # Reuse RNA projector for cell-line RNA (shared embedding space)
                rna_for_cl = self.projectors['rna'](cellline_rna_embeds)
            cl_emb = self.cellline_encoder(
                cellline_ids, cancer_type_ids, tissue_ids,
                rna_embeds=rna_for_cl, rna_mask=rna_available
            ).unsqueeze(1)  # [B, 1, D]
            cl_emb = cl_emb + self.modality_type_embeddings['cellline']
            tokens.append(cl_emb)
            projected['cellline'] = cl_emb.squeeze(1)
            has_cellline = True

        # Endoscopy tokens (future)
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
                # Get all query tokens for attention pooling IC50 head
                all_queries = self.qformer(combined, modality_mask=None, return_all_queries=True)  # [B, Q, D]
                fused = all_queries.mean(dim=1)  # [B, D] for classification heads
            else:
                fused = self.qformer(combined, modality_mask=None)  # [B, D]
        else:
            fused = combined.mean(dim=1)  # [B, D]

        # STEP 3: Outputs
        outputs = {'fused_embedding': fused}
        if return_embeddings:
            outputs['projected'] = projected

        # Classification heads (always computed from fused)
        outputs['tissue_logits'] = self.tissue_head(fused)
        outputs['cancer_logits'] = self.cancer_head(fused)

        # Drug class head (only if drug present)
        if 'drug' in projected:
            outputs['drug_logits'] = self.drug_class_head(fused)

        # IC50 head (only if drug + cellline present)
        if drug_embeds is not None and has_cellline:
            if self.config.use_ic50_attn_pool and all_queries is not None and 'drug' in projected and 'cellline' in projected:
                # Attention pooling + gated residual fusion (v2.1)
                outputs['ic50_pred'] = self.ic50_attn_pool_head(
                    qformer_queries=all_queries,
                    drug_proj=projected['drug'],
                    cellline_proj=projected['cellline'],
                )
            elif self.config.use_qformer_for_ic50:
                # Q-Former path: use fused embedding
                if self.config.use_ic50_skip_connection and 'drug' in projected and 'cellline' in projected:
                    skip_concat = torch.cat([fused, projected['drug'], projected['cellline']], dim=-1)
                    outputs['ic50_pred'] = self.ic50_head_skip(skip_concat).squeeze(-1)
                else:
                    outputs['ic50_pred'] = self.ic50_head(fused).squeeze(-1)
            else:
                # Q-detached path: bypass Q-Former, use projected drug + cellline
                if 'drug' in projected and 'cellline' in projected:
                    drug_cl_concat = torch.cat(
                        [projected['drug'], projected['cellline']], dim=-1
                    )
                    outputs['ic50_pred'] = self.ic50_head_detached(drug_cl_concat).squeeze(-1)
                else:
                    outputs['ic50_pred'] = self.ic50_head(fused).squeeze(-1)

        return outputs

    def initialize_prototypes(self, paired_data: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]):
        """
        Initialize prototypes from paired patient data.

        Args:
            paired_data: Dictionary mapping tissue_id to list of (image_embed, rna_embed) tuples
        """
        num_tissues = self.config.num_tissue_types
        embed_dim = self.config.hidden_dim

        device = next(self.parameters()).device
        image_protos = torch.zeros(num_tissues, embed_dim, device=device)
        rna_protos = torch.zeros(num_tissues, embed_dim, device=device)
        counts = torch.zeros(num_tissues, device=device)

        with torch.no_grad():
            for tissue_id, pairs in paired_data.items():
                for img_emb, rna_emb in pairs:
                    # Move embeddings to device
                    img_emb = img_emb.to(device)
                    rna_emb = rna_emb.to(device)
                    # Project embeddings
                    img_proj = self.projectors['image'](img_emb.unsqueeze(0)).squeeze(0)
                    rna_proj = self.projectors['rna'](rna_emb.unsqueeze(0)).squeeze(0)

                    image_protos[tissue_id] += img_proj.detach()
                    rna_protos[tissue_id] += rna_proj.detach()
                    counts[tissue_id] += 1

            # Average prototypes
            mask = counts > 0
            image_protos[mask] /= counts[mask].unsqueeze(-1)
            rna_protos[mask] /= counts[mask].unsqueeze(-1)

        self.image_prototypes = image_protos
        self.rna_prototypes = rna_protos
        self.prototypes_initialized = torch.tensor(True)

        print(f"Initialized prototypes for {mask.sum().item()} tissue types from paired data")

    def get_embedding(
        self,
        image_embeds: Optional[torch.Tensor] = None,
        rna_embeds: Optional[torch.Tensor] = None,
        drug_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get fused embedding for inference.

        Convenience method for getting the fused representation without task heads.
        """
        outputs = self.forward(
            image_embeds=image_embeds,
            rna_embeds=rna_embeds,
            drug_embeds=drug_embeds,
            return_embeddings=False
        )
        return outputs['fused_embedding']

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }
