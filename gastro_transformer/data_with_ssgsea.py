"""
ssGSEA Data Extension for IC50 Dataset.

Extends IC50Dataset with ssGSEA pathway enrichment scores loading.
Each cell-line gets its own typed Q-Former token for cross-modal learning.

ssGSEA TSV format expected:
    - Path: data/CCLE_20260324_ssGSEA_ccle_RNABert_sample_x_768Geneset.tsv
    - Columns: sample (e.g., SK-ES-1_BONE) + Geneset_0 ... Geneset_767
    - Normalization: normalize_name(sample.split('_')[0]) → SKES1
"""

from __future__ import annotations

import torch
import pandas as pd
import numpy as np
from typing import Optional, Dict, Set
import warnings

from .data import IC50Dataset, DrugEmbeddingDataset
from .config import GastroTransformerConfig


def normalize_name(n):
    """Normalize cell line name for matching."""
    return str(n).replace('-', '').replace(' ', '').replace('_', '').upper()


class IC50DatasetWithSsgsea(IC50Dataset):
    """
    Extended IC50Dataset with ssGSEA pathway enrichment scores.

    Inherits all IC50Dataset behavior and adds:
    - ssGSEA embeddings per cell-line (768d pathway scores)
    - ssGSEA availability mask
    - ssGSEA fallback for missing cell-lines

    The ssGSEA token is added as a 4th cell-line Q-Former token:
    [cancer, tissue, RNA-BERT, ssGSEA] = 4 cell-line tokens + 1 drug token = 5 total KV tokens
    """

    def __init__(
        self,
        ic50_csv_path: str,
        drug_embeddings: DrugEmbeddingDataset,
        config: Optional[GastroTransformerConfig] = None,
        cellline_id_col: str = 'cellline_id',
        drug_id_col: str = 'drug_id',
        ic50_col: str = 'ic50_value',
        cancer_type_col: Optional[str] = 'cancer_type_id',
        rna_csv_path: Optional[str] = None,
        rna_dim: int = 256,
        add_tissue_ids: bool = False,
        rna_knn_impute: bool = False,
        log_transform: bool = False,
        max_cellline_idx: Optional[int] = None,
        allowed_drugs: Optional[Set[str]] = None,
        allowed_celllines: Optional[Set[str]] = None,
        # ssGSEA-specific arguments
        ssgsea_tsv_path: Optional[str] = None,
        ssgsea_dim: int = 768,
    ):
        """
        Args:
            All IC50Dataset args (see IC50Dataset docs)
            ssgsea_tsv_path: Path to ssGSEA TSV file (None = RNA-only mode, backward compatible)
            ssgsea_dim: Dimension of ssGSEA embeddings (default 768)
        """
        # Initialize parent class (loads IC50, RNA, etc.)
        super().__init__(
            ic50_csv_path=ic50_csv_path,
            drug_embeddings=drug_embeddings,
            config=config,
            cellline_id_col=cellline_id_col,
            drug_id_col=drug_id_col,
            ic50_col=ic50_col,
            cancer_type_col=cancer_type_col,
            rna_csv_path=rna_csv_path,
            rna_dim=rna_dim,
            add_tissue_ids=add_tissue_ids,
            rna_knn_impute=rna_knn_impute,
            log_transform=log_transform,
            max_cellline_idx=max_cellline_idx,
            allowed_drugs=allowed_drugs,
            allowed_celllines=allowed_celllines,
        )

        # Load ssGSEA embeddings if provided
        self.ssgsea_dim = ssgsea_dim
        self.ssgsea_tsv_path = ssgsea_tsv_path
        self.cellline_ssgsea_embeds = None
        self.cellline_has_ssgsea = None

        if ssgsea_tsv_path is not None:
            self._load_cellline_ssgsea(ssgsea_tsv_path, ssgsea_dim)
        else:
            # Backward compatible: no ssGSEA, all zeros with availability=False
            warnings.warn(
                "No ssGSEA embeddings provided (ssgsea_tsv_path=None). "
                "ssGSEA token will be zero-filled with availability=False. "
                "This is the same behavior as RNA-only mode.",
                UserWarning
            )
            self.cellline_ssgsea_embeds = torch.zeros(
                self.num_celllines, ssgsea_dim, dtype=torch.float32
            )
            self.cellline_has_ssgsea = torch.zeros(
                self.num_celllines, dtype=torch.bool
            )

    def _load_cellline_ssgsea(self, tsv_path: str, ssgsea_dim: int):
        """
        Load ssGSEA pathway enrichment scores from TSV.

        Args:
            tsv_path: Path to ssGSEA TSV file
            ssgsea_dim: Expected dimension (768 genesets)

        TSV format:
            - First column: 'sample' (e.g., 'SK-ES-1_BONE')
            - Remaining columns: 'Geneset_0' ... 'Geneset_767'
        """
        print(f"Loading ssGSEA embeddings from {tsv_path}...")
        ssgsea_df = pd.read_csv(tsv_path, sep='\t')

        # Identify embedding columns (columns are 'Geneset1', 'Geneset2', ... 'Geneset768')
        emb_cols = [c for c in ssgsea_df.columns if c.startswith('Geneset') and c != 'sample']
        actual_dim = len(emb_cols)
        print(f"  Found {actual_dim} ssGSEA geneset columns")

        if actual_dim != ssgsea_dim:
            warnings.warn(
                f"ssGSEA dimension mismatch: expected {ssgsea_dim}, found {actual_dim}. "
                f"Using actual dimension {actual_dim}.",
                UserWarning
            )
            self.ssgsea_dim = actual_dim

        # Ensure embedding columns are float
        ssgsea_df[emb_cols] = ssgsea_df[emb_cols].astype(np.float32)

        # Create mapping from normalized cell-line name to embedding
        # Sample names are like 'SK-ES-1_BONE' → normalize to 'SKES1'
        ssgsea_dict = {}
        for _, row in ssgsea_df.iterrows():
            raw_sample = row['sample']
            # Extract cell-line name (before underscore)
            cl_name = raw_sample.split('_')[0]
            normalized = normalize_name(cl_name)
            emb = torch.tensor(
                row[emb_cols].values.astype(np.float32), dtype=torch.float32
            )
            ssgsea_dict[normalized] = emb

        print(f"  Loaded ssGSEA for {len(ssgsea_dict)} cell-lines")

        # Match to our dataset's cell-lines using normalized names
        # Build normalized → idx mapping for our dataset
        normalized_to_idx = {}
        for cl in self.cellline_to_idx:
            normalized = normalize_name(cl)
            normalized_to_idx[normalized] = self.cellline_to_idx[cl]

        # Allocate tensors
        self.cellline_ssgsea_embeds = torch.zeros(
            self.num_celllines, self.ssgsea_dim, dtype=torch.float32
        )
        self.cellline_has_ssgsea = torch.zeros(
            self.num_celllines, dtype=torch.bool
        )

        # Fill in embeddings
        matched = 0
        unmatched = []
        for normalized, emb in ssgsea_dict.items():
            if normalized in normalized_to_idx:
                idx = normalized_to_idx[normalized]
                self.cellline_ssgsea_embeds[idx] = emb
                self.cellline_has_ssgsea[idx] = True
                matched += 1
            else:
                unmatched.append(normalized)

        n_with = self.cellline_has_ssgsea.sum().item()
        n_without = self.num_celllines - n_with

        print(f"  ssGSEA matched to {matched}/{len(ssgsea_dict)} TSV cell-lines")
        print(f"  ssGSEA availability: {n_with}/{self.num_celllines} dataset cell-lines have ssGSEA, {n_without} missing")

        if unmatched and len(unmatched) <= 20:
            print(f"  Unmatched cell-lines (first 20): {unmatched[:20]}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns item with additional ssGSEA fields.

        Adds:
            - ssgsea_embed: ssGSEA embedding for this cell-line [768] or [ssgsea_dim]
            - ssgsea_available: Boolean indicating if ssGSEA was available [1]
        """
        # Get base item from parent
        item = super().__getitem__(idx)

        # Add ssGSEA embedding
        if self.cellline_ssgsea_embeds is not None:
            cl_idx = self.cellline_indices[idx]
            item['ssgsea_embed'] = self.cellline_ssgsea_embeds[cl_idx]
            item['ssgsea_available'] = self.cellline_has_ssgsea[cl_idx]

        return item
