"""Canonical cross-validation split construction.

Before this module, ``create_cellline_aware_folds`` / ``get_fold_splits`` existed
as three byte-identical copies (``scripts/benchmark_5fold_ncc_ablation.py``,
``scripts/sample_efficiency.py``, and a fourth in an earlier internal codebase),
differing only in a default argument that every call site overrode anyway. All
copies now re-export from here.

Fold semantics used across the revision:

``random``      row-level KFold; cell lines and drugs appear on both sides.
``ncc``         cell-line-disjoint folds; DR-A pretraining still saw every cell
                line, so test cell lines are known to the pretrained encoder
                (transductive).
``strict_ncc``  the same cell-line partition, but DR-A pretraining is re-run per
                fold on the training cell lines only, via
                ``IC50Dataset(allowed_celllines=...)``. Inductive.
``ncd``         drug-disjoint folds. Leak-free variant additionally excludes the
                held-out drugs from pretraining via ``allowed_drugs``.
``scaffold``    Bemis-Murcko scaffold-disjoint folds; strictly harder than NCD.

The functions returning a per-row ``sample_fold`` array compose with
``get_fold_splits`` so every regime shares one downstream code path.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIMES = ("random", "ncc", "strict_ncc", "ncd", "scaffold")


# ---------------------------------------------------------------------------
# NCC / strict NCC - cell-line-disjoint folds
# ---------------------------------------------------------------------------

def create_cellline_aware_folds(ic50_dataset, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Assign every row to a fold such that cell lines are disjoint across folds.

    Verbatim port of the implementation that produced the committed reports; the
    ordering (``sorted`` then ``rng.shuffle``, contiguous chunks, last fold
    absorbing the remainder) must not change or those results stop reproducing.

    Note that ``RNAFilteredDataset`` proxies ``cellline_to_idx`` for all 998 cell
    lines while exposing only the 592 with RNA through ``cellline_ids``. Folds
    stay disjoint and valid but come out unevenly sized. That behaviour is
    intentional and preserved.
    """
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
    logger.info(f"CV folds: {np.bincount(sample_fold)} samples per fold")
    return sample_fold


def get_fold_splits(
    sample_fold: np.ndarray, fold_idx: int, seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """Split one fold into (train, val, test) row indices.

    Validation is the first 25% of the shuffled training rows. The default
    ``seed=42`` reproduces the previously hardcoded ``default_rng(42 + fold_idx)``.
    """
    train_mask = sample_fold != fold_idx
    test_mask = sample_fold == fold_idx
    train_indices = np.where(train_mask)[0].tolist()
    test_indices = np.where(test_mask)[0].tolist()

    rng = np.random.default_rng(seed + fold_idx)
    rng.shuffle(train_indices)
    val_size = len(train_indices) // 4
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]

    return train_indices, val_indices, test_indices


def fold_celllines(
    sample_fold: np.ndarray, cellline_ids: Sequence[str], fold_idx: int
) -> Tuple[Set[str], Set[str]]:
    """(train_celllines, test_celllines) for a fold.

    Feeds ``IC50Dataset(allowed_celllines=...)`` so the strict/inductive NCC
    pretraining never sees a held-out cell line.
    """
    cellline_ids = np.asarray(cellline_ids, dtype=object)
    test = set(cellline_ids[sample_fold == fold_idx].tolist())
    train = set(cellline_ids[sample_fold != fold_idx].tolist())
    return train, test


# ---------------------------------------------------------------------------
# Random - row-level folds
# ---------------------------------------------------------------------------

def create_random_folds(n_samples: int, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Row-level KFold assignment. Cell lines and drugs straddle the split."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    sample_fold = np.zeros(n_samples, dtype=int)
    fold_size = n_samples // n_folds
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_samples
        sample_fold[indices[start:end]] = i
    logger.info(f"Random folds: {np.bincount(sample_fold)} samples per fold")
    return sample_fold


# ---------------------------------------------------------------------------
# NCD - drug-disjoint folds
# ---------------------------------------------------------------------------

def create_ncd_sample_folds(
    drug_ids: Sequence[str],
    n_folds: int = 5,
    seed: int = 42,
    fold_seed: Optional[int] = None,
) -> np.ndarray:
    """Per-row fold ids for drug-disjoint CV.

    Mirrors ``data.create_ncd_folds``: drugs are sorted by frequency descending
    and dealt round-robin into bins, so each fold holds a comparable mix of
    common and rare drugs. That function returns (train, test) index tuples; this
    returns the ``sample_fold`` shape so NCD composes with ``get_fold_splits``
    like every other regime.

    **``seed`` does not affect the partition, and never did.** The original
    implementation drew an rng and called ``rng.shuffle(bins[i])``, but that
    permutes the order of drugs *within* bin ``i`` while every drug in bin ``i`` is
    assigned fold ``i`` regardless -- the shuffle could not change any assignment.
    The call is removed here rather than left in place, because a seeded-looking
    no-op is what caused three "independent seeds" to evaluate one identical
    partition without anyone noticing.

    Pass ``fold_seed`` to genuinely resample the partition. Drugs are shuffled
    within consecutive frequency blocks of ``n_folds`` before the round-robin
    deal, which preserves the frequency balancing the deterministic version was
    designed for while randomising *which* fold receives each drug.

    ``fold_seed=None`` (the default) reproduces the historical assignment exactly,
    so committed results and ``tests/test_splits_parity.py`` are unaffected.
    """
    series = pd.Series(list(drug_ids))
    drug_counts = series.value_counts()
    unique_drugs = sorted(drug_counts.index.tolist(), key=lambda d: drug_counts[d], reverse=True)

    if len(unique_drugs) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} unique drugs for {n_folds}-fold NCD split, "
            f"but found only {len(unique_drugs)}"
        )

    if fold_seed is not None:
        # Shuffle within each frequency block, then deal. Each fold still receives
        # one drug from every block, so the common/rare mix is preserved.
        rng = np.random.default_rng(fold_seed)
        shuffled: List[str] = []
        for start in range(0, len(unique_drugs), n_folds):
            block = unique_drugs[start:start + n_folds]
            rng.shuffle(block)
            shuffled.extend(block)
        unique_drugs = shuffled

    drug_to_fold: Dict[str, int] = {}
    bins: Dict[int, List[str]] = {i: [] for i in range(n_folds)}
    for idx, drug in enumerate(unique_drugs):
        bins[idx % n_folds].append(drug)
    for i in range(n_folds):
        for drug in bins[i]:
            drug_to_fold[drug] = i

    sample_fold = series.map(drug_to_fold).to_numpy(dtype=int)
    partition = "deterministic" if fold_seed is None else f"fold_seed={fold_seed}"
    logger.info(f"NCD folds ({partition}): {np.bincount(sample_fold)} samples per fold")
    return sample_fold


def fold_drugs(
    sample_fold: np.ndarray, drug_ids: Sequence[str], fold_idx: int
) -> Tuple[Set[str], Set[str]]:
    """(train_drugs, test_drugs) for a fold; feeds ``allowed_drugs``."""
    drug_ids = np.asarray(drug_ids, dtype=object)
    test = set(drug_ids[sample_fold == fold_idx].tolist())
    train = set(drug_ids[sample_fold != fold_idx].tolist())
    return train, test


# ---------------------------------------------------------------------------
# Scaffold - Bemis-Murcko scaffold-disjoint folds
# ---------------------------------------------------------------------------

def murcko_scaffolds(
    smiles_by_drug: Dict[str, str], include_chirality: bool = False
) -> Dict[str, str]:
    """Map drug id to its Bemis-Murcko scaffold SMILES.

    Molecules that fail to parse, and acyclic molecules (which have an empty
    scaffold), fall back to a per-drug unique key so they are never silently
    pooled into one giant pseudo-scaffold.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")

    scaffolds: Dict[str, str] = {}
    n_failed = 0
    for drug_id, smiles in smiles_by_drug.items():
        scaffold = ""
        if isinstance(smiles, str) and smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=include_chirality
                )
        if not scaffold:
            scaffold = f"__unscaffolded__{drug_id}"
            n_failed += 1
        scaffolds[drug_id] = scaffold

    if n_failed:
        logger.warning(f"{n_failed} drugs had no Murcko scaffold; assigned unique keys")
    return scaffolds


def create_scaffold_folds(
    drug_ids: Sequence[str],
    smiles_by_drug: Dict[str, str],
    n_folds: int = 5,
    seed: int = 42,
    fold_seed: Optional[int] = None,
) -> np.ndarray:
    """Per-row fold ids for scaffold-disjoint CV.

    Scaffold groups are packed largest-first into the currently smallest fold,
    which balances row counts deterministically. Every drug sharing a scaffold
    lands in the same fold, so this is strictly harder than NCD.

    Unlike :func:`create_ncd_sample_folds`, the partition here **does** respond to
    the seed: groups are shuffled before a *stable* size sort, so equal-sized
    groups are ordered randomly and land in different folds. Row counts per fold
    come out the same, but the assignment differs. ``fold_seed`` is accepted so the
    partition can be varied independently of weight initialisation; when omitted it
    falls back to ``seed``, which preserves existing behaviour exactly.
    """
    scaffolds = murcko_scaffolds(smiles_by_drug)
    series = pd.Series(list(drug_ids))
    missing = set(series.unique()) - set(scaffolds)
    if missing:
        raise ValueError(f"No SMILES for {len(missing)} drugs, e.g. {sorted(missing)[:5]}")

    row_scaffold = series.map(scaffolds)
    group_sizes = row_scaffold.value_counts()

    rng = np.random.default_rng(seed if fold_seed is None else fold_seed)
    # Shuffle before the stable size sort so equal-sized groups are ordered randomly.
    groups = list(group_sizes.index)
    rng.shuffle(groups)
    groups = sorted(groups, key=lambda g: group_sizes[g], reverse=True)

    fold_loads = np.zeros(n_folds, dtype=int)
    scaffold_to_fold: Dict[str, int] = {}
    for group in groups:
        target = int(np.argmin(fold_loads))
        scaffold_to_fold[group] = target
        fold_loads[target] += group_sizes[group]

    sample_fold = row_scaffold.map(scaffold_to_fold).to_numpy(dtype=int)
    logger.info(
        f"Scaffold folds: {np.bincount(sample_fold)} samples per fold "
        f"across {len(groups)} scaffolds"
    )
    return sample_fold


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def build_folds(
    regime: str,
    ic50_dataset,
    n_folds: int = 5,
    seed: int = 42,
    smiles_by_drug: Optional[Dict[str, str]] = None,
    fold_seed: Optional[int] = None,
) -> np.ndarray:
    """Return the per-row fold assignment for a regime.

    ``strict_ncc`` shares NCC's partition; the two differ only in whether DR-A
    pretraining is re-run per fold, which is the runner's concern, not the
    split's.

    ``fold_seed`` separates *which rows are held out* from *how the model is
    initialised*, so a stage can replicate over partitions rather than over
    initialisations. It matters most for ``ncd``, whose partition is otherwise
    fixed regardless of ``seed`` -- three "seeds" there evaluate one identical
    split. ``fold_seed=None`` leaves every regime exactly as it was.

    Partition sensitivity to the seed, measured on this dataset:

    ==========  =============================================================
    regime      does ``seed`` alone change the partition?
    ==========  =============================================================
    ncc         yes
    strict_ncc  yes
    random      yes
    scaffold    yes (via tie-breaking among equal-sized scaffold groups)
    ncd         **no** -- use ``fold_seed``
    ==========  =============================================================
    """
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime {regime!r}; expected one of {REGIMES}")

    # For regimes whose partition already follows the seed, fold_seed simply
    # overrides it; for ncd it is the only way to move the partition at all.
    split_seed = seed if fold_seed is None else fold_seed

    if regime in ("ncc", "strict_ncc"):
        return create_cellline_aware_folds(ic50_dataset, n_folds=n_folds, seed=split_seed)
    if regime == "random":
        return create_random_folds(len(ic50_dataset), n_folds=n_folds, seed=split_seed)
    if regime == "ncd":
        return create_ncd_sample_folds(
            dataset_drug_ids(ic50_dataset), n_folds=n_folds, seed=seed, fold_seed=fold_seed
        )
    if regime == "scaffold":
        if not smiles_by_drug:
            raise ValueError("regime='scaffold' requires smiles_by_drug")
        return create_scaffold_folds(
            dataset_drug_ids(ic50_dataset), smiles_by_drug, n_folds=n_folds,
            seed=seed, fold_seed=fold_seed,
        )
    raise AssertionError(f"unhandled regime {regime!r}")


def dataset_drug_ids(ic50_dataset) -> List[str]:
    """Per-row drug id strings for an ``IC50Dataset`` or a filtered wrapper.

    ``IC50Dataset`` exposes ``drug_ids``; wrappers such as ``RNAFilteredDataset``
    may only expose ``valid_indices`` into a wrapped base dataset.
    """
    if hasattr(ic50_dataset, "drug_ids"):
        return list(ic50_dataset.drug_ids)
    base = getattr(ic50_dataset, "ic50_ds", None) or getattr(ic50_dataset, "base", None)
    valid = getattr(ic50_dataset, "valid_indices", None)
    if base is not None and valid is not None and hasattr(base, "drug_ids"):
        return [base.drug_ids[i] for i in valid]
    raise AttributeError(
        f"{type(ic50_dataset).__name__} exposes no per-row drug ids "
        "(need .drug_ids, or .valid_indices into a wrapped dataset)"
    )
