"""
Gastro-Transformer v2.1: Residual Fusion IC50 Head.

Changes from v2:
- IC50 head now receives ALL 32 query tokens (not mean-pooled) via attention pooling
- Raw drug + cell-line projections are concatenated as residual features
- Gated fusion lets the model learn how much to trust Q-Former vs raw features
- Other task heads (tissue, cancer, drug classification) remain unchanged

Architecture for IC50 path:
    Q-Former [B, 32, D] ──→ AttentionPool ──→ [B, D]  ─┐
                                                         ├─→ Gate ──→ IC50 Head ──→ IC50
    Drug proj [B, D] + CellLine proj [B, D] ──→ [B, 2D] ─┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from .config import GastroTransformerConfig


# ============================================================
# Unchanged components: ModalityProjector, QFormerBlock, QFormer, CellLineEncoder
# (kept identical to v2 — copy from original model.py)
# ============================================================

class ModalityProjector(nn.Module):
    """Project each modality embedding to common hidden dimension."""

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
        return self.proj(x)


class QFormerBlock(nn.Module):
    """Single Q-Former block with self-attention + cross-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

    def forward(self, queries, modality_embeds, modality_mask=None):
        q = self.norm1(queries)
        q = queries + self.self_attn(q, q, q)[0]
        q2 = self.norm2(q)
        kv = self.norm_kv(modality_embeds)
        q = q + self.cross_attn(q2, kv, kv, key_padding_mask=modality_mask)[0]
        q = q + self.ffn(self.norm3(q))
        return q


class QFormer(nn.Module):
    """Q-Former with learnable query tokens for multi-modal fusion."""

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_query_tokens, config.hidden_dim) * 0.02
        )
        self.blocks = nn.ModuleList([
            QFormerBlock(config.hidden_dim, config.qformer_heads, config.dropout)
            for _ in range(config.qformer_layers)
        ])
        self.output_proj = None

    def forward(self, modality_embeds, modality_mask=None, return_all_queries=False):
        B = modality_embeds.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        for block in self.blocks:
            queries = block(queries, modality_embeds, modality_mask)
        if return_all_queries:
            return queries  # [B, Q, D]
        return queries.mean(dim=1)  # [B, D]


class CellLineEncoder(nn.Module):
    """Encode cell-lines with learnable embeddings + cancer prior + FiLM + RNA fusion."""

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config
        self.cellline_embed = nn.Embedding(config.num_cell_lines, config.hidden_dim)
        self.cancer_type_embed = nn.Embedding(config.num_cancer_types, config.hidden_dim)
        self.tissue_gamma = nn.Embedding(config.num_tissue_types, config.hidden_dim)
        self.tissue_beta = nn.Embedding(config.num_tissue_types, config.hidden_dim)
        self.rna_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )

    def forward(self, cellline_ids, cancer_type_ids=None, tissue_ids=None, rna_embeds=None):
        max_idx = self.cellline_embed.num_embeddings - 1
        cellline_emb = self.cellline_embed(cellline_ids.clamp(0, max_idx))
        if cancer_type_ids is not None:
            max_cancer_idx = self.cancer_type_embed.num_embeddings - 1
            cellline_emb = cellline_emb + 0.5 * self.cancer_type_embed(
                cancer_type_ids.clamp(0, max_cancer_idx)
            )
        if tissue_ids is not None:
            max_tissue_idx = self.tissue_gamma.num_embeddings - 1
            safe_ids = tissue_ids.clamp(0, max_tissue_idx)
            cellline_emb = self.tissue_gamma(safe_ids) * cellline_emb + self.tissue_beta(safe_ids)
        if rna_embeds is not None:
            cellline_emb = self.rna_fusion(torch.cat([cellline_emb, rna_embeds], dim=-1))
        return cellline_emb


# ============================================================
# NEW: IC50-specific attention pooling + gated residual fusion
# ============================================================

class IC50AttentionPool(nn.Module):
    """
    Attention-pooling over Q-Former query tokens for IC50 prediction.

    Instead of mean-pooling 32 query tokens into [B, D], this learns a
    task-specific weighted combination. A single learnable query attends
    to all 32 Q-Former output tokens via cross-attention.

    Input:  [B, Q, D]  (Q=32 query tokens from Q-Former)
    Output: [B, D]     (single IC50-relevant representation)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Learnable IC50 query (single token that asks "what's relevant for IC50?")
        self.ic50_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

    def forward(self, query_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_tokens: All Q-Former output tokens [B, Q, D]
        Returns:
            IC50-relevant pooled representation [B, D]
        """
        B = query_tokens.shape[0]
        q = self.norm_q(self.ic50_query.expand(B, -1, -1))   # [B, 1, D]
        kv = self.norm_kv(query_tokens)                        # [B, Q, D]
        pooled, _ = self.cross_attn(q, kv, kv)                # [B, 1, D]
        return pooled.squeeze(1)                                # [B, D]


class IC50ResidualFusionHead(nn.Module):
    """
    IC50 prediction head with gated residual fusion.

    Combines:
    1. Attention-pooled Q-Former output (cross-modal context)    [B, D]
    2. Raw drug projection (drug-specific features)              [B, D]
    3. Raw cell-line projection (cell-line-specific features)    [B, D]

    A learned gate balances Q-Former fusion vs raw modality features.

    Architecture:
        Q-Former tokens [B,32,D] → AttentionPool → [B, D] = qf_pooled
        Drug proj [B, D]                                    = drug_raw
        CellLine proj [B, D]                                = cl_raw

        raw_concat  = [drug_raw; cl_raw]                     → [B, 2D]
        raw_proj    = Linear(2D → D)                         → [B, D]

        gate        = σ(Linear([qf_pooled; raw_proj]))       → [B, D]
        fused       = gate * qf_pooled + (1 - gate) * raw_proj  → [B, D]

        ic50        = MLP(fused)                             → [B, 1]
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Attention pooling over Q-Former queries
        self.attn_pool = IC50AttentionPool(hidden_dim, num_heads, dropout)

        # 2. Raw feature projection: [drug_proj; cl_proj] → [B, D]
        self.raw_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. Gating mechanism: decides per-dimension how much to use
        #    Q-Former context vs raw modality features
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 4. Final IC50 regression MLP
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

    def forward(
        self,
        qformer_queries: torch.Tensor,  # [B, Q, D] — all 32 query tokens
        drug_proj: torch.Tensor,         # [B, D]   — projected drug embedding
        cellline_proj: torch.Tensor,     # [B, D]   — projected cell-line embedding
    ) -> torch.Tensor:
        """
        Args:
            qformer_queries: All Q-Former output tokens [B, Q, D]
            drug_proj: Projected drug embedding [B, D]
            cellline_proj: Projected cell-line embedding [B, D]
        Returns:
            IC50 prediction [B]
        """
        # Attention-pool the Q-Former output
        qf_pooled = self.attn_pool(qformer_queries)         # [B, D]

        # Project raw modality features
        raw_concat = torch.cat([drug_proj, cellline_proj], dim=-1)  # [B, 2D]
        raw_features = self.raw_proj(raw_concat)                     # [B, D]

        # Gated fusion
        gate_input = torch.cat([qf_pooled, raw_features], dim=-1)   # [B, 2D]
        g = self.gate(gate_input)                                     # [B, D]
        fused = g * qf_pooled + (1 - g) * raw_features              # [B, D]

        # Predict IC50
        return self.head(fused).squeeze(-1)                          # [B]


# ============================================================
# Modified ModalitySlotQFormer
# ============================================================

class ModalitySlotQFormer(nn.Module):
    """
    Gastro-Transformer v2.1: Modality-Slot Q-Former with Residual Fusion.

    Key change from v2:
    - IC50 prediction uses IC50ResidualFusionHead which receives:
      (a) All 32 Q-Former query tokens (not mean-pooled)
      (b) Raw drug + cell-line projections as residual features
    - Other task heads (tissue, cancer, drug) still use mean-pooled fused [B, D]
    """

    def __init__(self, config: GastroTransformerConfig):
        super().__init__()
        self.config = config

        # --- Unchanged: projectors, type embeddings, Q-Former, cell-line encoder ---
        self.projectors = nn.ModuleDict({
            'image': ModalityProjector(config.image_dim, config.hidden_dim, config.dropout),
            'rna': ModalityProjector(config.rna_dim, config.hidden_dim, config.dropout),
            'drug': ModalityProjector(config.drug_dim, config.hidden_dim, config.dropout),
        })
        if config.endo_dim > 0:
            self.projectors['endo'] = ModalityProjector(
                config.endo_dim, config.hidden_dim, config.dropout
            )

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

        self.qformer = QFormer(config)

        if config.use_cellline_embeddings:
            self.cellline_encoder = CellLineEncoder(config)
        else:
            self.cellline_encoder = None

        # --- CHANGED: IC50 head is now the residual fusion version ---
        self.ic50_head = IC50ResidualFusionHead(
            hidden_dim=config.hidden_dim,
            num_heads=config.qformer_heads,
            dropout=config.dropout,
        )

        # --- Unchanged: classification heads ---
        self.tissue_head = nn.Linear(config.hidden_dim, config.num_tissue_types)
        self.cancer_head = nn.Linear(config.hidden_dim, config.num_cancer_types)
        self.drug_class_head = nn.Linear(config.hidden_dim, config.num_drug_classes)

        # --- Unchanged: prototypes ---
        self.register_buffer('image_prototypes', None)
        self.register_buffer('rna_prototypes', None)
        self.register_buffer('prototypes_initialized', torch.tensor(False))

    def forward(
        self,
        image_embeds: Optional[torch.Tensor] = None,
        rna_embeds: Optional[torch.Tensor] = None,
        drug_embeds: Optional[torch.Tensor] = None,
        endo_embeds: Optional[torch.Tensor] = None,
        cellline_ids: Optional[torch.Tensor] = None,
        cancer_type_ids: Optional[torch.Tensor] = None,
        tissue_ids: Optional[torch.Tensor] = None,
        cellline_rna_embeds: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:

        # STEP 1: Build modality token list (UNCHANGED)
        tokens = []
        projected = {}

        if image_embeds is not None:
            if image_embeds.dim() == 2:
                image_embeds = image_embeds.unsqueeze(1)
            img_proj = self.projectors['image'](image_embeds)
            img_proj = img_proj + self.modality_type_embeddings['image']
            tokens.append(img_proj)
            projected['image'] = img_proj.mean(dim=1)

        if rna_embeds is not None:
            rna_proj = self.projectors['rna'](rna_embeds).unsqueeze(1)
            rna_proj = rna_proj + self.modality_type_embeddings['rna']
            tokens.append(rna_proj)
            projected['rna'] = rna_proj.squeeze(1)

        if drug_embeds is not None:
            drug_proj = self.projectors['drug'](drug_embeds).unsqueeze(1)
            drug_proj = drug_proj + self.modality_type_embeddings['drug']
            tokens.append(drug_proj)
            projected['drug'] = drug_proj.squeeze(1)

        if cellline_ids is not None and self.cellline_encoder is not None:
            rna_for_cl = None
            if cellline_rna_embeds is not None:
                rna_for_cl = self.projectors['rna'](cellline_rna_embeds)
            cl_emb = self.cellline_encoder(
                cellline_ids, cancer_type_ids, tissue_ids, rna_embeds=rna_for_cl
            ).unsqueeze(1)
            cl_emb = cl_emb + self.modality_type_embeddings['cellline']
            tokens.append(cl_emb)
            projected['cellline'] = cl_emb.squeeze(1)

        if endo_embeds is not None and 'endo' in self.projectors:
            endo_proj = self.projectors['endo'](endo_embeds).unsqueeze(1)
            endo_proj = endo_proj + self.modality_type_embeddings['endo']
            tokens.append(endo_proj)
            projected['endo'] = endo_proj.squeeze(1)

        assert len(tokens) > 0, "At least one modality must be provided"
        combined = torch.cat(tokens, dim=1)  # [B, N_total, D]

        # STEP 2: Q-Former fusion
        # CHANGED: Always get all query tokens for IC50 path
        if self.config.use_qformer:
            all_queries = self.qformer(combined, modality_mask=None,
                                       return_all_queries=True)  # [B, Q, D]
            fused = all_queries.mean(dim=1)  # [B, D] for classification heads
        else:
            all_queries = None
            fused = combined.mean(dim=1)

        # STEP 3: Outputs
        outputs = {'fused_embedding': fused}
        if return_embeddings:
            outputs['projected'] = projected

        # Classification heads — still use mean-pooled fused [B, D] (UNCHANGED)
        outputs['tissue_logits'] = self.tissue_head(fused)
        outputs['cancer_logits'] = self.cancer_head(fused)
        if 'drug' in projected:
            outputs['drug_logits'] = self.drug_class_head(fused)

        # IC50 head — CHANGED: uses residual fusion
        if drug_embeds is not None and cellline_ids is not None:
            if self.config.use_qformer and all_queries is not None:
                # v2.1: Residual fusion with all query tokens + raw features
                outputs['ic50_pred'] = self.ic50_head(
                    qformer_queries=all_queries,           # [B, Q, D]
                    drug_proj=projected['drug'],           # [B, D]
                    cellline_proj=projected['cellline'],   # [B, D]
                )
            else:
                # Fallback for no-QFormer ablation: use concat baseline
                raw_concat = torch.cat(
                    [projected['drug'], projected['cellline']], dim=-1
                )  # [B, 2D]
                # Simple fallback head for ablation (2D → 1)
                fallback_input = fused  # mean-pooled [B, D]
                outputs['ic50_pred'] = self.ic50_head.head(
                    self.ic50_head.raw_proj(raw_concat)
                ).squeeze(-1)

        return outputs

    # --- Unchanged methods below ---

    def initialize_prototypes(self, paired_data: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]):
        num_tissues = self.config.num_tissue_types
        embed_dim = self.config.hidden_dim
        device = next(self.parameters()).device
        image_protos = torch.zeros(num_tissues, embed_dim, device=device)
        rna_protos = torch.zeros(num_tissues, embed_dim, device=device)
        counts = torch.zeros(num_tissues, device=device)
        with torch.no_grad():
            for tissue_id, pairs in paired_data.items():
                for img_emb, rna_emb in pairs:
                    img_emb = img_emb.to(device)
                    rna_emb = rna_emb.to(device)
                    img_proj = self.projectors['image'](img_emb.unsqueeze(0)).squeeze(0)
                    rna_proj = self.projectors['rna'](rna_emb.unsqueeze(0)).squeeze(0)
                    image_protos[tissue_id] += img_proj.detach()
                    rna_protos[tissue_id] += rna_proj.detach()
                    counts[tissue_id] += 1
            mask = counts > 0
            image_protos[mask] /= counts[mask].unsqueeze(-1)
            rna_protos[mask] /= counts[mask].unsqueeze(-1)
        self.image_prototypes = image_protos
        self.rna_prototypes = rna_protos
        self.prototypes_initialized = torch.tensor(True)
        print(f"Initialized prototypes for {mask.sum().item()} tissue types from paired data")

    def get_embedding(self, image_embeds=None, rna_embeds=None, drug_embeds=None):
        outputs = self.forward(
            image_embeds=image_embeds, rna_embeds=rna_embeds,
            drug_embeds=drug_embeds, return_embeddings=False
        )
        return outputs['fused_embedding']

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}
