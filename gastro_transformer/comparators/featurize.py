"""Molecular featurisation for the faithful comparator reimplementations.

Two encodings, each reproducing what the original paper used:

``molecular_graphs``  DeepCDR's drug input — 75-dimensional atom features and a
                      dense adjacency matrix, matching deepchem's
                      ``ConvMolFeaturizer`` field-for-field.
``smiles_onehot``     tCNNS's drug input — a character one-hot matrix over the
                      SMILES string.

The 75 dimensions decompose as 44 atom-type + 11 degree + 7 implicit-valence
+ 1 formal charge + 1 radical electrons + 5 hybridisation + 1 aromatic
+ 5 total-H. deepchem is not installed here, so the encoding is rebuilt from
rdkit primitives; the widths are pinned by the tests in
``tests/test_comparators.py`` so drift is caught rather than silently absorbed.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# deepchem's ConvMolFeaturizer atom-type vocabulary, in its original order. The
# trailing "Unknown" bucket absorbs anything outside the list.
ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As",
    "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se",
    "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr",
    "Pt", "Hg", "Pb", "Unknown",
]
DEGREES = list(range(11))
IMPLICIT_VALENCES = list(range(7))
TOTAL_HS = list(range(5))
ATOM_FEATURE_DIM = (
    len(ATOM_TYPES) + len(DEGREES) + len(IMPLICIT_VALENCES) + 1 + 1 + 5 + 1 + len(TOTAL_HS)
)  # 75


def _one_hot(value, allowed: Sequence, unknown_ok: bool = False) -> List[float]:
    """One-hot over ``allowed``; out-of-vocabulary folds into the last slot."""
    if value not in allowed:
        if not unknown_ok:
            return [0.0] * len(allowed)
        value = allowed[-1]
    return [float(value == item) for item in allowed]


def atom_features(atom) -> List[float]:
    """75-dimensional atom descriptor."""
    from rdkit.Chem import rdchem

    hybridisations = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]
    return (
        _one_hot(atom.GetSymbol(), ATOM_TYPES, unknown_ok=True)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetImplicitValence(), IMPLICIT_VALENCES)
        + [float(atom.GetFormalCharge()), float(atom.GetNumRadicalElectrons())]
        + _one_hot(atom.GetHybridization(), hybridisations)
        + [float(atom.GetIsAromatic())]
        + _one_hot(atom.GetTotalNumHs(), TOTAL_HS)
    )


def molecular_graphs(
    smiles_by_drug: Dict[str, str], max_atoms: int | None = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Dense padded molecular graphs for every drug.

    Returns ``(features[N, max_atoms, 75], adjacency[N, max_atoms, max_atoms], drug_ids)``.

    Dense rather than sparse deliberately. There are only a couple of hundred
    unique drugs of a few dozen atoms each, so the whole tensor is a few MB and a
    batched matmul beats scatter-gather. It also matches DeepCDR's own
    formulation, which is a dense ``batch_dot`` over the adjacency.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    drug_ids = sorted(smiles_by_drug)
    mols = {}
    for drug_id in drug_ids:
        mol = Chem.MolFromSmiles(smiles_by_drug[drug_id])
        if mol is None:
            raise ValueError(f"unparseable SMILES for drug {drug_id!r}")
        mols[drug_id] = mol

    observed_max = max(mol.GetNumAtoms() for mol in mols.values())
    if max_atoms is None:
        max_atoms = observed_max
    elif observed_max > max_atoms:
        raise ValueError(f"max_atoms={max_atoms} too small; largest molecule has {observed_max}")

    features = np.zeros((len(drug_ids), max_atoms, ATOM_FEATURE_DIM), dtype=np.float32)
    adjacency = np.zeros((len(drug_ids), max_atoms, max_atoms), dtype=np.float32)

    for i, drug_id in enumerate(drug_ids):
        mol = mols[drug_id]
        for atom in mol.GetAtoms():
            features[i, atom.GetIdx()] = atom_features(atom)
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adjacency[i, a, b] = 1.0
            adjacency[i, b, a] = 1.0

    logger.info(
        f"molecular graphs: {len(drug_ids)} drugs, max {max_atoms} atoms, "
        f"{ATOM_FEATURE_DIM}-d atom features"
    )
    return features, adjacency, drug_ids


def smiles_charset(smiles_by_drug: Dict[str, str]) -> Tuple[List[str], int]:
    """Character vocabulary and maximum length, derived from the data.

    tCNNS derives both from its own drug set (reporting 28 characters and length
    188 for their 223 GDSC compounds), so deriving them here rather than hardcoding
    their numbers is what reproduces the method.
    """
    chars = sorted({c for s in smiles_by_drug.values() for c in s})
    length = max(len(s) for s in smiles_by_drug.values())
    return chars, length


def smiles_onehot(
    smiles_by_drug: Dict[str, str],
    chars: Sequence[str] | None = None,
    length: int | None = None,
) -> Tuple[np.ndarray, List[str], List[str], int]:
    """One-hot SMILES tensor ``[N, length, n_chars]`` plus the vocabulary used.

    Note the axis order: tCNNS feeds ``[batch, length, channels]`` to a 1-D
    convolution over the character axis, so characters are the channel dimension.
    """
    if chars is None or length is None:
        derived_chars, derived_length = smiles_charset(smiles_by_drug)
        chars = chars or derived_chars
        length = length or derived_length

    index = {c: i for i, c in enumerate(chars)}
    drug_ids = sorted(smiles_by_drug)
    tensor = np.zeros((len(drug_ids), length, len(chars)), dtype=np.float32)

    n_truncated = 0
    for i, drug_id in enumerate(drug_ids):
        smiles = smiles_by_drug[drug_id]
        if len(smiles) > length:
            n_truncated += 1
            smiles = smiles[:length]
        for position, char in enumerate(smiles):
            if char in index:
                tensor[i, position, index[char]] = 1.0

    if n_truncated:
        logger.warning(f"{n_truncated} SMILES truncated to {length} characters")
    logger.info(f"SMILES one-hot: {len(drug_ids)} drugs, {length} x {len(chars)}")
    return tensor, drug_ids, list(chars), length
