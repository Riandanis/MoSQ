"""
Data Loading and Dataset Classes for Gastro-Transformer.

Handles CSV-based loading of embeddings from frozen encoders.
Supports paired data (limited), unpaired data (large-scale), and IC50 data.
"""

from __future__ import annotations  # Enable postponed evaluation of annotations

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import warnings
from sklearn.neighbors import NearestNeighbors
from .config import GastroTransformerConfig
from .utils import get_tissue_id_for_cellline


class PairedMultiModalDataset(Dataset):
    """
    Dataset for paired patient samples where BOTH image and RNA embeddings
    are available from the SAME patient.

    This is the CRITICAL but LIMITED data (10-50 samples expected).
    Used for direct cross-modal alignment learning.

    Expected CSV format for image embeddings:
        sample_id, tissue_label, emb_0, emb_1, ..., emb_{image_dim-1}

    Expected CSV format for RNA embeddings:
        sample_id, tissue_label, emb_0, emb_1, ..., emb_{rna_dim-1}

    The sample_ids must match between the two CSVs for pairing.
    """

    def __init__(
        self,
        image_csv_path: str,
        rna_csv_path: str,
        image_dim: int = 512,
        rna_dim: int = 256,
        sample_id_col: str = 'sample_id',
        tissue_label_col: str = 'tissue_label',
        cancer_label_col: Optional[str] = 'cancer_label'
    ):
        """
        Args:
            image_csv_path: Path to image embeddings CSV
            rna_csv_path: Path to RNA embeddings CSV
            image_dim: Dimension of image embeddings
            rna_dim: Dimension of RNA embeddings
            sample_id_col: Column name for sample IDs
            tissue_label_col: Column name for tissue type labels
            cancer_label_col: Column name for cancer type labels (optional)
        """
        self.image_dim = image_dim
        self.rna_dim = rna_dim

        # Load CSVs
        image_df = pd.read_csv(image_csv_path)
        rna_df = pd.read_csv(rna_csv_path)

        # Find common sample IDs (paired samples)
        image_ids = set(image_df[sample_id_col].values)
        rna_ids = set(rna_df[sample_id_col].values)
        common_ids = image_ids.intersection(rna_ids)

        if len(common_ids) == 0:
            raise ValueError(
                "No matching sample_ids found between image and RNA CSVs. "
                "Ensure sample_id columns contain matching values for paired samples."
            )

        print(f"Found {len(common_ids)} paired samples")

        # Filter to common IDs
        image_df = image_df[image_df[sample_id_col].isin(common_ids)]
        rna_df = rna_df[rna_df[sample_id_col].isin(common_ids)]

        # Sort by sample_id for consistent ordering
        image_df = image_df.sort_values(sample_id_col).reset_index(drop=True)
        rna_df = rna_df.sort_values(sample_id_col).reset_index(drop=True)

        # CRITICAL: Verify alignment after sorting - prevents cross-modal misalignment
        if not (image_df[sample_id_col].values == rna_df[sample_id_col].values).all():
            raise ValueError(
                "Sample ID misalignment after sorting! Paired data corrupted. "
                "Check for duplicate sample_ids or inconsistent data."
            )

        # Store sample IDs
        self.sample_ids = image_df[sample_id_col].values.tolist()

        # Extract embedding columns
        emb_cols_image = [c for c in image_df.columns if c.startswith('emb_')]
        emb_cols_rna = [c for c in rna_df.columns if c.startswith('emb_')]

        if len(emb_cols_image) != image_dim:
            warnings.warn(
                f"Expected {image_dim} image embedding columns, found {len(emb_cols_image)}. "
                "Using found columns."
            )
        if len(emb_cols_rna) != rna_dim:
            warnings.warn(
                f"Expected {rna_dim} RNA embedding columns, found {len(emb_cols_rna)}. "
                "Using found columns."
            )

        # Convert to tensors
        self.image_embeds = torch.tensor(
            image_df[emb_cols_image].values, dtype=torch.float32
        )
        self.rna_embeds = torch.tensor(
            rna_df[emb_cols_rna].values, dtype=torch.float32
        )

        # Tissue labels
        self.tissue_labels = torch.tensor(
            image_df[tissue_label_col].values, dtype=torch.long
        )

        # Cancer labels (optional)
        if cancer_label_col and cancer_label_col in image_df.columns:
            self.cancer_labels = torch.tensor(
                image_df[cancer_label_col].values, dtype=torch.long
            )
        else:
            self.cancer_labels = None

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'sample_id': self.sample_ids[idx],
            'image_embed': self.image_embeds[idx],
            'rna_embed': self.rna_embeds[idx],
            'tissue_label': self.tissue_labels[idx],
            'is_paired': True
        }

        if self.cancer_labels is not None:
            item['cancer_label'] = self.cancer_labels[idx]

        return item

    def get_paired_data_by_tissue(self) -> Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get paired data organized by tissue type for prototype initialization.

        Returns:
            Dictionary mapping tissue_id to list of (image_embed, rna_embed) tuples
        """
        paired_data = {}

        for i in range(len(self)):
            tissue_id = self.tissue_labels[i].item()
            img_emb = self.image_embeds[i]
            rna_emb = self.rna_embeds[i]

            if tissue_id not in paired_data:
                paired_data[tissue_id] = []
            paired_data[tissue_id].append((img_emb, rna_emb))

        return paired_data


class UnpairedModalityDataset(Dataset):
    """
    Dataset for unpaired single-modality samples (large-scale data).

    Expected CSV format:
        sample_id, tissue_label, emb_0, emb_1, ..., emb_{dim-1}
    """

    def __init__(
        self,
        csv_path: str,
        modality: str,  # 'image' or 'rna'
        embedding_dim: int,
        sample_id_col: str = 'sample_id',
        tissue_label_col: str = 'tissue_label'
    ):
        """
        Args:
            csv_path: Path to embeddings CSV
            modality: 'image' or 'rna'
            embedding_dim: Expected dimension of embeddings
            sample_id_col: Column name for sample IDs
            tissue_label_col: Column name for tissue labels
        """
        self.modality = modality
        self.embedding_dim = embedding_dim

        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} {modality} samples from {csv_path}")

        # Handle NaN tissue labels
        tissue_labels = df[tissue_label_col].values
        nan_count = pd.isna(tissue_labels).sum()
        if nan_count > 0:
            warnings.warn(
                f"Dropping {nan_count} samples with NaN tissue labels",
                UserWarning
            )
            valid_mask = ~pd.isna(tissue_labels)
            df = df[valid_mask]
            print(f"After dropping NaN: {len(df)} samples")

        # Store sample IDs
        self.sample_ids = df[sample_id_col].values.tolist()

        # Extract embedding columns
        emb_cols = [c for c in df.columns if c.startswith('emb_')]
        if len(emb_cols) != embedding_dim:
            warnings.warn(
                f"Expected {embedding_dim} embedding columns, found {len(emb_cols)}. "
                "Using found columns."
            )

        # Convert to tensor
        self.embeddings = torch.tensor(df[emb_cols].values, dtype=torch.float32)
        self.tissue_labels = torch.tensor(df[tissue_label_col].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'sample_id': self.sample_ids[idx],
            f'{self.modality}_embed': self.embeddings[idx],
            'tissue_label': self.tissue_labels[idx],
            'is_paired': False
        }


class DrugEmbeddingDataset(Dataset):
    """
    Dataset for drug embeddings.

    Expected CSV format:
        drug_id, [moa_label], emb_0, emb_1, ..., emb_{drug_dim-1}
    """

    def __init__(
        self,
        csv_path: str,
        drug_dim: int = 768,
        drug_id_col: str = 'drug_id',
        moa_label_col: Optional[str] = 'moa_label'
    ):
        self.drug_dim = drug_dim

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} drug embeddings from {csv_path}")

        # Drug IDs
        self.drug_ids = df[drug_id_col].values.tolist()

        # Create drug_id to index mapping
        self.drug_id_to_idx = {did: i for i, did in enumerate(self.drug_ids)}

        # Embeddings
        emb_cols = [c for c in df.columns if c.startswith('emb_')]
        self.embeddings = torch.tensor(df[emb_cols].values, dtype=torch.float32)

        # MoA labels (optional) - only process if numeric
        if moa_label_col and moa_label_col in df.columns:
            # Check if the column contains numeric data
            try:
                # Try to convert to numeric first
                numeric_labels = pd.to_numeric(df[moa_label_col], errors='coerce')
                # If we have any non-NaN values, use numeric labels
                if numeric_labels.notna().any():
                    self.moa_labels = torch.tensor(numeric_labels.fillna(-1).values, dtype=torch.long)
                else:
                    # String labels - encode to integers
                    unique_labels = df[moa_label_col].unique()
                    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                    self.moa_labels = torch.tensor([label_to_idx.get(label, -1) for label in df[moa_label_col]], dtype=torch.long)
                    self.moa_label_names = unique_labels.tolist()
            except (ValueError, TypeError):
                # If conversion fails, just skip moa_labels
                self.moa_labels = None
        else:
            self.moa_labels = None

    def __len__(self) -> int:
        return len(self.drug_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'drug_id': self.drug_ids[idx],
            'drug_embed': self.embeddings[idx]
        }
        if self.moa_labels is not None:
            item['moa_label'] = self.moa_labels[idx]
        return item

    def get_embedding_by_id(self, drug_id: str) -> Optional[torch.Tensor]:
        """Get embedding for a specific drug ID."""
        if drug_id in self.drug_id_to_idx:
            return self.embeddings[self.drug_id_to_idx[drug_id]]
        # Return None explicitly - caller should handle this case
        return None


def create_ncd_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    drug_id_col: str = 'drug_id'
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create No Common Drug (NCD) folds for cross-validation.

    Groups by drug_id and splits drugs into n_folds disjoint sets.
    Returns train/test indices such that NO drug appears in both train and test.

    This is a harder generalization task than cell-line-aware CV, as the model
    must predict response to completely unseen drugs.

    Args:
        df: DataFrame with IC50 data containing at least drug_id_col
        n_folds: Number of folds (default 5)
        seed: Random seed for reproducibility
        drug_id_col: Column name for drug IDs

    Returns:
        List of (train_indices, test_indices) tuples, one per fold

    Example:
        >>> folds = create_ncd_folds(ic50_df, n_folds=5, seed=42)
        >>> for train_idx, test_idx in folds:
        ...     train_drugs = set(df.iloc[train_idx]['drug_id'])
        ...     test_drugs = set(df.iloc[test_idx]['drug_id'])
        ...     assert len(train_drugs & test_drugs) == 0  # No overlap!
    """
    rng = np.random.default_rng(seed)

    # Get unique drugs and their frequency for stratified splitting
    drug_counts = df[drug_id_col].value_counts()
    unique_drugs = drug_counts.index.tolist()

    # Handle edge case with too few drugs
    if len(unique_drugs) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} unique drugs for {n_folds}-fold NCD split, "
            f"but found only {len(unique_drugs)}"
        )

    # Sort by frequency (descending) for more balanced splits
    unique_drugs = sorted(unique_drugs, key=lambda d: drug_counts[d], reverse=True)

    # Assign drugs to folds using round-robin based on frequency
    # This ensures each fold gets a similar distribution of drug frequencies
    bins = {i: [] for i in range(n_folds)}
    for idx, drug in enumerate(unique_drugs):
        bins[idx % n_folds].append(drug)

    # Shuffle within each bin
    for i in range(n_folds):
        rng.shuffle(bins[i])

    # Assign drugs to folds - each fold gets one bin as test
    fold_drugs = []
    for test_bin in range(n_folds):
        test_drugs = set(bins[test_bin])
        train_drugs = set()
        for train_bin in range(n_folds):
            if train_bin != test_bin:
                train_drugs.update(bins[train_bin])
        fold_drugs.append((train_drugs, test_drugs))

    # Convert drug sets to indices
    folds = []
    drug_id_to_indices = df.groupby(drug_id_col).indices

    for train_drugs, test_drugs in fold_drugs:
        train_indices = []
        test_indices = []

        for drug_id in train_drugs:
            if drug_id in drug_id_to_indices:
                train_indices.extend(drug_id_to_indices[drug_id].tolist())

        for drug_id in test_drugs:
            if drug_id in drug_id_to_indices:
                test_indices.extend(drug_id_to_indices[drug_id].tolist())

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Shuffle indices
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        folds.append((train_indices, test_indices))

    # Verify no drug overlap
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_drugs = set(df.iloc[train_idx][drug_id_col])
        test_drugs = set(df.iloc[test_idx][drug_id_col])
        overlap = train_drugs & test_drugs
        if len(overlap) > 0:
            raise ValueError(f"Fold {fold_idx}: Drug overlap detected: {overlap}")

    print(f"Created {n_folds} NCD folds:")
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_drugs = len(set(df.iloc[train_idx][drug_id_col]))
        test_drugs = len(set(df.iloc[test_idx][drug_id_col]))
        print(f"  Fold {fold_idx+1}: {train_drugs} train drugs, {test_drugs} test drugs, "
              f"{len(train_idx)} train samples, {len(test_idx)} test samples")

    return folds


def split_ic50_dataset_cellline_aware(
    ic50_dataset: IC50Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Split IC50 dataset by CELL-LINE (not by row) to prevent data leakage.

    CRITICAL: This split ensures no cell-line appears in multiple splits.
    Naive row-based splitting causes 99%+ cell-line overlap, invalidating metrics.

    Args:
        ic50_dataset: IC50Dataset to split
        train_ratio: Fraction of cell-lines for training
        val_ratio: Fraction of cell-lines for validation
        test_ratio: Fraction of cell-lines for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_subset, val_subset, test_subset) as torch Subset objects

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    from torch.utils.data import Subset

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    rng = np.random.default_rng(seed)

    # Get unique cell-lines from the dataset
    unique_celllines = sorted(ic50_dataset.cellline_to_idx.keys())
    rng.shuffle(unique_celllines)

    n_total = len(unique_celllines)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Remaining go to test

    train_celllines = set(unique_celllines[:n_train])
    val_celllines = set(unique_celllines[n_train:n_train + n_val])
    test_celllines = set(unique_celllines[n_train + n_val:])

    # Verify no overlap
    assert len(train_celllines & val_celllines) == 0, "Cell-line leakage: train/val overlap"
    assert len(train_celllines & test_celllines) == 0, "Cell-line leakage: train/test overlap"
    assert len(val_celllines & test_celllines) == 0, "Cell-line leakage: val/test overlap"

    # Create indices for each split based on cell-line membership
    train_indices = []
    val_indices = []
    test_indices = []

    for i in range(len(ic50_dataset)):
        cellline_id = ic50_dataset.cellline_ids[i]  # Original string ID
        if cellline_id in train_celllines:
            train_indices.append(i)
        elif cellline_id in val_celllines:
            val_indices.append(i)
        elif cellline_id in test_celllines:
            test_indices.append(i)

    train_subset = Subset(ic50_dataset, train_indices)
    val_subset = Subset(ic50_dataset, val_indices)
    test_subset = Subset(ic50_dataset, test_indices)

    print(f"Cell-line-aware split:")
    print(f"  Train: {len(train_celllines)} cell-lines, {len(train_indices)} IC50 pairs")
    print(f"  Val:   {len(val_celllines)} cell-lines, {len(val_indices)} IC50 pairs")
    print(f"  Test:  {len(test_celllines)} cell-lines, {len(test_indices)} IC50 pairs")
    print(f"  No cell-line overlap between splits ✓")

    return train_subset, val_subset, test_subset


class IC50Dataset(Dataset):
    """
    Dataset for IC50 prediction task.

    Expected CSV format:
        cellline_id, drug_id, ic50_value, [cancer_type_id]

    Requires drug embeddings to be loaded separately.
    """

    def __init__(
        self,
        ic50_csv_path: str,
        drug_embeddings: DrugEmbeddingDataset,
        config: Optional[GastroTransformerConfig] = None,  # PHASE 2: needed for KNN imputation
        cellline_id_col: str = 'cellline_id',
        drug_id_col: str = 'drug_id',
        ic50_col: str = 'ic50_value',
        cancer_type_col: Optional[str] = 'cancer_type_id',
        rna_csv_path: Optional[str] = None,  # Optional cell-line RNA data
        rna_dim: int = 256,
        add_tissue_ids: bool = False,  # Resolve cell-line → tissue_id for tissue bridge
        rna_knn_impute: bool = False,  # PHASE 2: KNN impute missing RNA from tissue centroids
        log_transform: bool = False,  # Set True ONLY if ic50_col is raw μM (not already log-scale)
        max_cellline_idx: Optional[int] = None,  # Validate cell-line count fits model embedding table
        allowed_drugs: Optional[Set[str]] = None,  # If set, only include these drugs (for leak-free NCD)
        allowed_celllines: Optional[Set[str]] = None  # If set, only include these cell-lines (for leak-free NCC)
    ):
        """
        Args:
            ic50_csv_path: Path to IC50 data CSV
            drug_embeddings: DrugEmbeddingDataset with drug embeddings
            config: GastroTransformerConfig (needed for tissue type count in KNN imputation)
            cellline_id_col: Column name for cell-line IDs
        Args:
            ic50_csv_path: Path to IC50 data CSV
            drug_embeddings: DrugEmbeddingDataset with drug embeddings
            cellline_id_col: Column name for cell-line IDs
            drug_id_col: Column name for drug IDs
            ic50_col: Column name for IC50 values
            cancer_type_col: Column name for cancer type (optional)
            rna_csv_path: Path to cell-line RNA embeddings (optional)
            rna_dim: Dimension of RNA embeddings
            add_tissue_ids: If True, resolve each cell-line → tissue type ID via
                CELLLINE_TO_TISSUE and store as a tensor. Resolved at __init__
                (not __getitem__) to avoid 185K dict lookups per epoch.
            log_transform: If True, applies log10(ic50 + 1e-9) before storing.
                Set False (default) when data is already log-scaled (e.g., GDSC
                LN_IC50 values, which span -10 to +13 with negative values for
                potent drugs). Setting True on pre-log data causes double-transform.
            max_cellline_idx: If set, raises ValueError when the dataset contains
                more unique cell-lines than this value, catching embedding table
            allowed_drugs: If set, only include samples with drug_id in this set
                size mismatches at dataset load time rather than at forward pass.
            allowed_celllines: If set, only include samples with cellline_id in this set.
                Used for leak-free NCC evaluation where test cell-lines are held out during pretraining.
        """
        self.drug_embeddings = drug_embeddings
        self.config = config  # PHASE 2: stored for KNN imputation

        # Load IC50 data
        df = pd.read_csv(ic50_csv_path)
        print(f"Loaded {len(df)} IC50 entries from {ic50_csv_path}")

        # Filter to drugs with embeddings
        valid_drugs = set(drug_embeddings.drug_ids)
        df = df[df[drug_id_col].isin(valid_drugs)]
        print(f"After filtering to available drugs: {len(df)} entries")

        # Filter to allowed_drugs if specified (for leak-free NCD)
        if allowed_drugs is not None:
            df = df[df[drug_id_col].isin(allowed_drugs)]
            print(f"After filtering to allowed drugs: {len(df)} entries")

        # Filter to allowed_celllines if specified (for leak-free NCC)
        if allowed_celllines is not None:
            df = df[df[cellline_id_col].isin(allowed_celllines)]
            print(f"After filtering to allowed celllines: {len(df)} entries")

        # Detect and remove duplicate drug-cellline pairs
        dup_count = df.duplicated(subset=[cellline_id_col, drug_id_col]).sum()
        if dup_count > 0:
            warnings.warn(
                f"Found {dup_count} duplicate drug-cellline pairs. "
                f"Keeping first occurrence of each pair.",
                UserWarning
            )
            df = df.drop_duplicates(subset=[cellline_id_col, drug_id_col], keep='first')
            print(f"After removing duplicates: {len(df)} entries")

        # Store data
        self.cellline_ids = df[cellline_id_col].values.tolist()
        self.drug_ids = df[drug_id_col].values.tolist()

        # IC50 value handling — controlled by log_transform flag.
        # GDSC LN_IC50 data is already log-scaled (range ≈ -10 to +13, negative = potent).
        # Applying log10(abs(...)) on pre-log data destroys sign information for 20% of
        # rows and compresses std from ~2.7 to ~0.4, making the regression target nearly flat.
        ic50_raw = df[ic50_col].values.astype(np.float32)
        if log_transform:
            # Only for raw μM inputs — fails loudly on pre-log data.
            if (ic50_raw < 0).any():
                raise ValueError(
                    f"log_transform=True but {(ic50_raw < 0).sum()} negative values found "
                    f"(min={ic50_raw.min():.2f}). Negative values indicate the data is already "
                    "log-transformed. Set log_transform=False (the default)."
                )
            self.ic50_values = torch.tensor(
                np.log10(ic50_raw + 1e-9), dtype=torch.float32
            )
            print(f"IC50 log10-transformed: raw [{ic50_raw.min():.2f}, {ic50_raw.max():.2f}] "
                  f"→ log [{self.ic50_values.min():.2f}, {self.ic50_values.max():.2f}]")
        else:
            # Data is already on a log scale — store as-is.
            if (ic50_raw > 100).any():
                warnings.warn(
                    f"IC50 values appear large (max={ic50_raw.max():.2f}) for log-scale input. "
                    "If data is in raw μM, set log_transform=True. Storing as-is.",
                    UserWarning
                )
            self.ic50_values = torch.tensor(ic50_raw, dtype=torch.float32)
            print(f"IC50 values stored as-is (log_transform=False): "
                  f"range [{self.ic50_values.min():.2f}, {self.ic50_values.max():.2f}], "
                  f"std={self.ic50_values.std():.3f}")

        # Create cell-line ID to integer index mapping
        unique_celllines = sorted(set(self.cellline_ids))
        self.cellline_to_idx = {cl: i for i, cl in enumerate(unique_celllines)}
        self.num_celllines = len(unique_celllines)
        print(f"Number of unique cell-lines: {self.num_celllines}")

        # Validate against model embedding table size (H8 guard)
        if max_cellline_idx is not None and self.num_celllines > max_cellline_idx:
            raise ValueError(
                f"IC50 dataset has {self.num_celllines} unique cell-lines but "
                f"max_cellline_idx={max_cellline_idx}. The model embedding table would be "
                "out-of-bounds. Increase config.num_cell_lines or reduce the dataset."
            )

        # Cell-line indices as tensor
        self.cellline_indices = torch.tensor(
            [self.cellline_to_idx[cl] for cl in self.cellline_ids],
            dtype=torch.long
        )

        # Cancer types (optional)
        if cancer_type_col and cancer_type_col in df.columns:
            self.cancer_type_ids = torch.tensor(
                df[cancer_type_col].values, dtype=torch.long
            )
        else:
            self.cancer_type_ids = None

        # Tissue IDs for tissue bridge: resolve cell-line name → tissue type ID
        # Resolved here (not in __getitem__) to avoid repeated dict lookups.
        self.tissue_ids = None
        if add_tissue_ids:
            unique_cl_list = sorted(self.cellline_to_idx.keys(), key=lambda cl: self.cellline_to_idx[cl])
            tissue_id_per_cellline = torch.tensor(
                [get_tissue_id_for_cellline(cl) for cl in unique_cl_list],
                dtype=torch.long
            )
            # Map each row to its tissue ID via the pre-computed per-cell-line tensor
            self.tissue_ids = tissue_id_per_cellline[self.cellline_indices]
            print(f"Resolved tissue IDs for {len(unique_cl_list)} unique cell-lines")

        # Store RNA KNN imputation flag
        self.rna_knn_impute = rna_knn_impute

        # Load cell-line RNA embeddings if available
        self.cellline_rna_embeds = None
        if rna_csv_path is not None:
            self._load_cellline_rna(rna_csv_path, rna_dim)
        else:
            warnings.warn(
                "No cell-line RNA embeddings provided (rna_csv_path=None). "
                "IC50 prediction will use learnable cell-line embeddings only "
                "(no gene expression signal). Provide --cellline_rna_csv to enable "
                "drug-RNA interaction modeling.",
                UserWarning
            )

    def _load_cellline_rna(self, rna_csv_path: str, rna_dim: int):
        """Load cell-line RNA embeddings with optional KNN imputation."""
        rna_df = pd.read_csv(rna_csv_path)
        emb_cols = [c for c in rna_df.columns if c.startswith('emb_')]

        # Ensure embedding columns are float type
        rna_df[emb_cols] = rna_df[emb_cols].astype(np.float32)

        # Create mapping from cell-line to RNA embedding
        rna_dict = {}
        for _, row in rna_df.iterrows():
            cl_id = row['cellline_id']
            if cl_id in self.cellline_to_idx:
                rna_dict[cl_id] = torch.tensor(
                    row[emb_cols].values.astype(np.float32), dtype=torch.float32
                )

        print(f"Loaded RNA embeddings for {len(rna_dict)} cell-lines")

        # Create tensor indexed by cell-line index
        self.cellline_rna_embeds = torch.zeros(
            self.num_celllines, len(emb_cols), dtype=torch.float32
        )
        # Track which cell-lines actually have RNA data (vs zero-filled)
        self.cellline_has_rna = torch.zeros(self.num_celllines, dtype=torch.bool)
        for cl_id, emb in rna_dict.items():
            idx = self.cellline_to_idx[cl_id]
            self.cellline_rna_embeds[idx] = emb
            self.cellline_has_rna[idx] = True

        n_with = self.cellline_has_rna.sum().item()
        n_without = self.num_celllines - n_with
        print(f"RNA availability: {n_with}/{self.num_celllines} cell-lines have RNA, {n_without} missing")

        # PHASE 2: KNN Imputation for missing RNA embeddings
        # For cell-lines without RNA, impute using tissue-type centroids from similar cell-lines
        if self.rna_knn_impute:
            self._impute_missing_rna(len(emb_cols))

    def _impute_missing_rna(self, rna_dim: int):
        """
        PHASE 2: Impute missing RNA embeddings using tissue-type KNN.

        For cell-lines without RNA data, finds tissue-similar cell-lines with RNA
        and uses their centroids as imputed embeddings. This increases effective
        RNA coverage from ~592 to ~998 cell-lines.
        """
        # Identify cell-lines with and without RNA
        has_rna_mask = self.cellline_rna_embeds.abs().sum(dim=1) > 1e-6  # Non-zero embeddings
        has_rna_indices = has_rna_mask.nonzero(as_tuple=True)[0]
        missing_rna_indices = (~has_rna_mask).nonzero(as_tuple=True)[0]

        if len(missing_rna_indices) == 0:
            print(f"  All {len(has_rna_indices)} cell-lines have RNA, no imputation needed")
            return

        print(f"  PHASE 2 KNN Imputation: {len(has_rna_indices)} with RNA, {len(missing_rna_indices)} missing")

        # Get tissue IDs for all cell-lines (need add_tissue_ids=True for this to work)
        if self.tissue_ids is None:
            warnings.warn(
                "PHASE 2 KNN imputation requires add_tissue_ids=True. "
                "Falling back to zero embeddings for missing RNA.",
                UserWarning
            )
            return

        # Compute tissue-type centroids from cell-lines WITH RNA
        tissue_centroids = {}  # tissue_id -> list of embeddings
        for idx in has_rna_indices:
            tissue_id = self.tissue_ids[idx].item()
            if tissue_id not in tissue_centroids:
                tissue_centroids[tissue_id] = []
            tissue_centroids[tissue_id].append(self.cellline_rna_embeds[idx])

        # Average to get centroid per tissue type
        tissue_centroid_tensor = torch.zeros(self.config.num_tissue_types, rna_dim)
        tissue_has_centroid = torch.zeros(self.config.num_tissue_types, dtype=torch.bool)
        for tissue_id, emb_list in tissue_centroids.items():
            tissue_centroid_tensor[tissue_id] = torch.stack(emb_list).mean(dim=0)
            tissue_has_centroid[tissue_id] = True

        # For missing cell-lines: assign nearest tissue centroid
        # If no centroid available for that tissue, use global mean of available centroids
        available_centroids = tissue_centroid_tensor[tissue_has_centroid]
        if len(available_centroids) > 0:
            global_centroid = available_centroids.mean(dim=0)
        else:
            global_centroid = torch.zeros(rna_dim)

        # Find tissue ID for each missing cell-line and assign centroid
        imputed_count = 0
        for idx in missing_rna_indices:
            tissue_id = self.tissue_ids[idx].item()
            if tissue_has_centroid[tissue_id]:
                self.cellline_rna_embeds[idx] = tissue_centroid_tensor[tissue_id].clone()
            else:
                # Use global centroid for unknown tissues
                self.cellline_rna_embeds[idx] = global_centroid.clone()
            imputed_count += 1

        print(f"  Imputed RNA embeddings for {imputed_count} cell-lines using tissue centroids")

    def __len__(self) -> int:
        return len(self.ic50_values)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        drug_id = self.drug_ids[idx]
        drug_embed = self.drug_embeddings.get_embedding_by_id(drug_id)

        # Explicit error handling for missing drug embeddings
        if drug_embed is None:
            raise ValueError(
                f"Drug embedding not found for drug_id='{drug_id}' at index {idx}. "
                f"Ensure all drugs in IC50 CSV exist in drug embeddings CSV."
            )

        item = {
            'cellline_id': self.cellline_indices[idx],
            'drug_id': drug_id,
            'drug_embed': drug_embed,
            'ic50': self.ic50_values[idx]
        }

        if self.cancer_type_ids is not None:
            item['cancer_type_id'] = self.cancer_type_ids[idx]

        if self.cellline_rna_embeds is not None:
            cl_idx = self.cellline_indices[idx]
            item['rna_embed'] = self.cellline_rna_embeds[cl_idx]
            item['rna_available'] = self.cellline_has_rna[cl_idx]

        if self.tissue_ids is not None:
            item['tissue_id'] = self.tissue_ids[idx]

        return item


def create_data_loaders(
    config: GastroTransformerConfig,
    paired_dataset: Optional[PairedMultiModalDataset] = None,
    unpaired_image_dataset: Optional[UnpairedModalityDataset] = None,
    unpaired_rna_dataset: Optional[UnpairedModalityDataset] = None,
    ic50_dataset: Optional[IC50Dataset] = None,
    ic50_train: Optional[Dataset] = None,
    ic50_val: Optional[Dataset] = None,
    ic50_test: Optional[Dataset] = None,
    split_ic50: Optional[bool] = None,  # None defaults to config.split_ic50
    ic50_split_seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create data loaders for all datasets.

    Args:
        config: GastroTransformerConfig
        paired_dataset: Paired multi-modal dataset (limited, precious)
        unpaired_image_dataset: Large-scale unpaired image data
        unpaired_rna_dataset: Large-scale unpaired RNA data
        ic50_dataset: IC50 prediction dataset
        ic50_train: Pre-split IC50 training dataset (optional)
        ic50_val: Pre-split IC50 validation dataset (optional)
        ic50_test: Pre-split IC50 test dataset (optional)
        split_ic50: If None, uses config.split_ic50 (default: True). Set False only for debugging.

    Returns:
        Dictionary of DataLoaders
    """
    # Default to config setting
    if split_ic50 is None:
        split_ic50 = getattr(config, 'split_ic50', True)
    loaders = {}

    # Get DataLoader options from config
    worker_init = (config.num_workers > 0)
    prefetch = getattr(config, 'prefetch_factor', 4) if worker_init else None
    persistent = getattr(config, 'persistent_workers', True) if worker_init else False

    if paired_dataset is not None:
        # M1 fix: drop_last=True with a small paired dataset (e.g., 48 samples) and a
        # large batch_size (256) produces zero batches, silently disabling cross-modal
        # learning. Use drop_last=False; the min() clamp below ensures batch_size ≤ N.
        paired_batch_size = min(config.batch_size, len(paired_dataset))
        if len(paired_dataset) < 2:
            warnings.warn(
                f"Paired dataset has only {len(paired_dataset)} samples. "
                "Cross-modal contrastive loss requires ≥2 samples per batch.",
                UserWarning
            )
        elif len(paired_dataset) < config.batch_size:
            warnings.warn(
                f"Paired dataset ({len(paired_dataset)} samples) is smaller than "
                f"batch_size ({config.batch_size}). Using batch_size={paired_batch_size} "
                "for paired loader. Cross-modal learning is active on all samples.",
                UserWarning
            )
        loaders['paired'] = DataLoader(
            paired_dataset,
            batch_size=paired_batch_size,
            shuffle=True,
            num_workers=config.num_workers if worker_init else 0,
            drop_last=False,  # Fixed: was True, which emptied the loader when N < batch_size
            pin_memory=True,
            prefetch_factor=prefetch if worker_init else 2,
            persistent_workers=persistent if worker_init else False
        )

    if unpaired_image_dataset is not None:
        loaders['unpaired_image'] = DataLoader(
            unpaired_image_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers if worker_init else 0,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=prefetch if worker_init else 2,
            persistent_workers=persistent if worker_init else False
        )

    if unpaired_rna_dataset is not None:
        loaders['unpaired_rna'] = DataLoader(
            unpaired_rna_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers if worker_init else 0,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=prefetch if worker_init else 2,
            persistent_workers=persistent if worker_init else False
        )

    # Handle pre-split IC50 datasets
    if ic50_train is not None and ic50_val is not None and ic50_test is not None:
        # Use pre-split datasets
        loaders['ic50_train'] = DataLoader(
            ic50_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers if worker_init else 0,
            pin_memory=True,
            prefetch_factor=prefetch if worker_init else 2,
            persistent_workers=persistent if worker_init else False
        )
        loaders['ic50_val'] = DataLoader(
            ic50_val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers if worker_init else 0,
            pin_memory=True,
            prefetch_factor=prefetch if worker_init else 2,
            persistent_workers=persistent if worker_init else False
        )
        loaders['ic50_test'] = DataLoader(
            ic50_test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers if worker_init else 0,
            pin_memory=True,
            prefetch_factor=prefetch if worker_init else 2,
            persistent_workers=persistent if worker_init else False
        )
        loaders['ic50'] = loaders['ic50_train']
    elif ic50_dataset is not None:
        if split_ic50:
            # Cell-line-aware split to prevent data leakage
            train_set, val_set, test_set = split_ic50_dataset_cellline_aware(
                ic50_dataset,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=ic50_split_seed
            )
            loaders['ic50_train'] = DataLoader(
                train_set,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers if worker_init else 0,
                pin_memory=True,
                prefetch_factor=prefetch if worker_init else 2,
                persistent_workers=persistent if worker_init else False
            )
            loaders['ic50_val'] = DataLoader(
                val_set,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers if worker_init else 0,
                pin_memory=True,
                prefetch_factor=prefetch if worker_init else 2,
                persistent_workers=persistent if worker_init else False
            )
            loaders['ic50_test'] = DataLoader(
                test_set,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers if worker_init else 0,
                pin_memory=True,
                prefetch_factor=prefetch if worker_init else 2,
                persistent_workers=persistent if worker_init else False
            )
            # Keep backward compatibility: 'ic50' maps to train for existing code
            loaders['ic50'] = loaders['ic50_train']
        else:
            loaders['ic50'] = DataLoader(
                ic50_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers if worker_init else 0,
                pin_memory=True,
                prefetch_factor=prefetch if worker_init else 2,
                persistent_workers=persistent if worker_init else False
            )

    return loaders


def load_datasets_from_config(config: GastroTransformerConfig) -> Dict[str, Dataset]:
    """
    Load all datasets based on paths specified in config.

    Returns:
        Dictionary of loaded datasets
    """
    datasets = {}

    # Paired data
    if config.paired_image_csv and config.paired_rna_csv:
        datasets['paired'] = PairedMultiModalDataset(
            image_csv_path=config.paired_image_csv,
            rna_csv_path=config.paired_rna_csv,
            image_dim=config.image_dim,
            rna_dim=config.rna_dim
        )

    # Unpaired image data
    if config.unpaired_image_csv:
        datasets['unpaired_image'] = UnpairedModalityDataset(
            csv_path=config.unpaired_image_csv,
            modality='image',
            embedding_dim=config.image_dim
        )

    # Unpaired RNA data
    if config.unpaired_rna_csv:
        datasets['unpaired_rna'] = UnpairedModalityDataset(
            csv_path=config.unpaired_rna_csv,
            modality='rna',
            embedding_dim=config.rna_dim
        )

    # Drug embeddings
    if config.drug_embeddings_csv:
        datasets['drugs'] = DrugEmbeddingDataset(
            csv_path=config.drug_embeddings_csv,
            drug_dim=config.drug_dim
        )

        # IC50 data (requires drug embeddings)
        if config.ic50_csv:
            datasets['ic50'] = IC50Dataset(
                ic50_csv_path=config.ic50_csv,
                drug_embeddings=datasets['drugs'],
                rna_csv_path=config.cellline_rna_csv  # Cell-line RNA embeddings for Q-Former
            )

    return datasets


# =============================================================================
# NOTES FOR USER - CSV FORMAT REQUIREMENTS
# =============================================================================
"""
CSV FORMAT SPECIFICATIONS:
==========================

1. PAIRED IMAGE EMBEDDINGS CSV:
   Required columns:
   - sample_id: Unique patient/sample identifier (MUST match RNA CSV)
   - tissue_label: Integer tissue type label (0, 1, 2, ...)
   - emb_0, emb_1, ..., emb_511: Embedding values (512 dimensions)

   Optional columns:
   - cancer_label: Cancer subtype label

   Example:
   sample_id,tissue_label,cancer_label,emb_0,emb_1,...,emb_511
   patient_001,5,2,0.123,-0.456,...,0.789
   patient_002,5,2,0.234,-0.567,...,0.890

2. PAIRED RNA EMBEDDINGS CSV:
   Same format as image, but with 256 embedding columns:
   sample_id,tissue_label,emb_0,emb_1,...,emb_255

3. UNPAIRED IMAGE EMBEDDINGS CSV:
   Same format, but sample_ids don't need to match any other modality

4. UNPAIRED RNA EMBEDDINGS CSV:
   Same format as paired RNA

5. DRUG EMBEDDINGS CSV:
   Required columns:
   - drug_id: Unique drug identifier
   - emb_0, emb_1, ..., emb_767: Drug embedding values (768 dimensions)

   Optional columns:
   - moa_label: Mechanism of action category (integer)

   Example:
   drug_id,moa_label,emb_0,emb_1,...,emb_767
   drug_001,3,0.111,0.222,...,0.333

6. IC50 DATA CSV:
   Required columns:
   - cellline_id: Cell-line identifier (string or integer)
   - drug_id: Drug identifier (must match drug embeddings CSV)
   - ic50_value: IC50 value (float, typically log-transformed)

   Optional columns:
   - cancer_type_id: Cancer type of the cell-line (integer)

   Example:
   cellline_id,drug_id,ic50_value,cancer_type_id
   AGS,drug_001,2.5,0
   KATOIII,drug_002,3.1,0

7. CELL-LINE METADATA CSV (optional, for IC50):
   Required columns:
   - cellline_id: Cell-line identifier
   Optional:
   - cellline_name: Human-readable name
   - cancer_type_id: Cancer type category
   - tissue_type: Tissue of origin

TISSUE LABEL ENCODING:
======================
You need a consistent integer encoding across all modalities:
    0: Stomach (Gastric)
    1: Esophagus
    2: Breast
    3: Lung
    4: Colon
    ...

Store this mapping in a separate file or in utils.py for reference.

IMPORTANT NOTES:
================
1. For PAIRED data, sample_ids MUST match between image and RNA CSVs
2. Embedding columns MUST be named emb_0, emb_1, etc.
3. tissue_label should be consistent integers across all files
4. IC50 values should be pre-processed (e.g., log-transformed, normalized)
"""
