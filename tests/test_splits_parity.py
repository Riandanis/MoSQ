"""Parity gate for the splits refactor.

``gastro_transformer.splits`` replaces three duplicated copies of the fold
construction code. The committed results in ``reports/`` were produced by those
copies, so any behavioural drift silently invalidates them.

The reference implementations below are frozen verbatim from
``scripts/benchmark_5fold_ncc_ablation.py`` (lines 125-162) and
``scripts/benchmark_random_split_rnafiltered.py`` as they stood before the
refactor. They exist only to be compared against and must never be "fixed".
"""

from __future__ import annotations

import numpy as np
import pytest

from gastro_transformer import splits


# ---------------------------------------------------------------------------
# Frozen pre-refactor reference implementations - do not modify
# ---------------------------------------------------------------------------

def _ref_create_cellline_aware_folds(ic50_dataset, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    unique_celllines = sorted(ic50_dataset.cellline_to_idx.keys())
    rng.shuffle(unique_celllines)
    n = len(unique_celllines)
    fold_size = n // n_folds
    cellline_folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        cellline_folds.append(set(unique_celllines[start:end]))

    sample_fold = np.zeros(len(ic50_dataset), dtype=int)
    for i in range(len(ic50_dataset)):
        cl_id = ic50_dataset.cellline_ids[i]
        for fold_idx, fold_cl in enumerate(cellline_folds):
            if cl_id in fold_cl:
                sample_fold[i] = fold_idx
                break
    return sample_fold


def _ref_get_fold_splits(sample_fold, fold_idx):
    train_mask = sample_fold != fold_idx
    test_mask = sample_fold == fold_idx
    train_indices = np.where(train_mask)[0].tolist()
    test_indices = np.where(test_mask)[0].tolist()

    rng = np.random.default_rng(42 + fold_idx)
    rng.shuffle(train_indices)
    val_size = len(train_indices) // 4
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]

    return train_indices, val_indices, test_indices


def _ref_create_random_folds(n_samples, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    sample_fold = np.zeros(n_samples, dtype=int)
    fold_size = n_samples // n_folds
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_samples
        sample_fold[indices[start:end]] = i
    return sample_fold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeIC50Dataset:
    """Minimal stand-in exposing the attributes the split functions touch."""

    def __init__(self, cellline_ids, drug_ids):
        self.cellline_ids = list(cellline_ids)
        self.drug_ids = list(drug_ids)
        unique = sorted(set(self.cellline_ids))
        self.cellline_to_idx = {c: i for i, c in enumerate(unique)}

    def __len__(self):
        return len(self.cellline_ids)


@pytest.fixture
def dataset():
    """Uneven cell-line and drug frequencies, mirroring the real data."""
    rng = np.random.default_rng(0)
    n_cl, n_drug = 137, 41
    celllines, drugs = [], []
    for c in range(n_cl):
        # Deliberately ragged: some cell lines carry far more drugs than others.
        n_for_cl = int(rng.integers(3, n_drug + 1))
        for d in rng.choice(n_drug, size=n_for_cl, replace=False):
            celllines.append(f"CL{c:03d}")
            drugs.append(f"DRUG{d:03d}")
    return FakeIC50Dataset(celllines, drugs)


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_folds", [3, 5, 10])
@pytest.mark.parametrize("seed", [0, 42])
def test_cellline_folds_match_reference(dataset, n_folds, seed):
    got = splits.create_cellline_aware_folds(dataset, n_folds=n_folds, seed=seed)
    want = _ref_create_cellline_aware_folds(dataset, n_folds=n_folds, seed=seed)
    np.testing.assert_array_equal(got, want)


def test_get_fold_splits_match_reference(dataset):
    sample_fold = splits.create_cellline_aware_folds(dataset, n_folds=5, seed=42)
    for fold_idx in range(5):
        got = splits.get_fold_splits(sample_fold, fold_idx)
        want = _ref_get_fold_splits(sample_fold, fold_idx)
        for got_part, want_part in zip(got, want):
            assert got_part == want_part


@pytest.mark.parametrize("n_folds", [3, 5])
def test_random_folds_match_reference(n_folds):
    got = splits.create_random_folds(1000, n_folds=n_folds, seed=42)
    want = _ref_create_random_folds(1000, n_folds=n_folds, seed=42)
    np.testing.assert_array_equal(got, want)


def test_refactored_scripts_reexport_identical_functions():
    """The importlib-by-path consumers bind these names off the script module."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for script in ("sample_efficiency.py", "benchmark_5fold_ncc_ablation.py"):
        spec = importlib.util.spec_from_file_location(
            f"_parity_{script.replace('.py', '')}", root / "scripts" / script
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.create_cellline_aware_folds is splits.create_cellline_aware_folds
        assert module.get_fold_splits is splits.get_fold_splits


# ---------------------------------------------------------------------------
# Disjointness invariants for every regime
# ---------------------------------------------------------------------------

def test_ncc_folds_are_cellline_disjoint(dataset):
    sample_fold = splits.create_cellline_aware_folds(dataset, n_folds=5, seed=42)
    ids = np.asarray(dataset.cellline_ids, dtype=object)
    for fold_idx in range(5):
        train, test = splits.fold_celllines(sample_fold, ids, fold_idx)
        assert not (train & test)


def test_ncd_folds_are_drug_disjoint(dataset):
    sample_fold = splits.create_ncd_sample_folds(dataset.drug_ids, n_folds=5, seed=42)
    assert set(np.unique(sample_fold)) == set(range(5))
    for fold_idx in range(5):
        train, test = splits.fold_drugs(sample_fold, dataset.drug_ids, fold_idx)
        assert not (train & test)


def test_ncd_drug_assignment_matches_data_create_ncd_folds(dataset):
    """The sample_fold adapter must agree with the original tuple-returning API."""
    import pandas as pd

    from gastro_transformer.data import create_ncd_folds

    df = pd.DataFrame({"drug_id": dataset.drug_ids})
    sample_fold = splits.create_ncd_sample_folds(dataset.drug_ids, n_folds=5, seed=42)

    for fold_idx, (train_idx, test_idx) in enumerate(create_ncd_folds(df, n_folds=5, seed=42)):
        assert set(df.iloc[test_idx].drug_id) == set(
            np.asarray(dataset.drug_ids, dtype=object)[sample_fold == fold_idx].tolist()
        )
        assert set(df.iloc[train_idx].drug_id) == set(
            np.asarray(dataset.drug_ids, dtype=object)[sample_fold != fold_idx].tolist()
        )


def test_scaffold_folds_are_scaffold_disjoint():
    pytest.importorskip("rdkit")
    smiles_by_drug = {
        "A": "c1ccccc1C(=O)O",
        "B": "c1ccccc1CC(=O)O",
        "C": "c1ccncc1C(=O)N",
        "D": "c1ccncc1CCN",
        "E": "C1CCCCC1CO",
        "F": "C1CCCCC1CCO",
        "G": "CCCCO",          # acyclic: no Murcko scaffold
        "H": "c1ccc2ccccc2c1",
        "I": "c1ccc2ccccc2c1C",
        "J": "C1CCNCC1",
    }
    drug_ids = [d for d in smiles_by_drug for _ in range(10)]
    sample_fold = splits.create_scaffold_folds(drug_ids, smiles_by_drug, n_folds=3, seed=42)

    scaffolds = splits.murcko_scaffolds(smiles_by_drug)
    row_scaffolds = np.array([scaffolds[d] for d in drug_ids], dtype=object)
    for fold_idx in range(3):
        test = set(row_scaffolds[sample_fold == fold_idx].tolist())
        train = set(row_scaffolds[sample_fold != fold_idx].tolist())
        assert not (train & test)


def test_acyclic_molecules_get_unique_scaffold_keys():
    pytest.importorskip("rdkit")
    scaffolds = splits.murcko_scaffolds({"X": "CCCCO", "Y": "CCCCCO", "Z": "not_a_smiles"})
    assert len({scaffolds["X"], scaffolds["Y"], scaffolds["Z"]}) == 3


def test_build_folds_rejects_unknown_regime(dataset):
    with pytest.raises(ValueError, match="Unknown regime"):
        splits.build_folds("nonsense", dataset)


def test_build_folds_requires_smiles_for_scaffold(dataset):
    with pytest.raises(ValueError, match="smiles_by_drug"):
        splits.build_folds("scaffold", dataset)


def test_strict_ncc_shares_the_ncc_partition(dataset):
    """Strict NCC differs only in per-fold pretraining, never in the partition."""
    np.testing.assert_array_equal(
        splits.build_folds("ncc", dataset, n_folds=5, seed=7),
        splits.build_folds("strict_ncc", dataset, n_folds=5, seed=7),
    )
