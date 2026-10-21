"""Input assembly for the faithful comparator models.

Two jobs: parse DrugCell's Gene Ontology file into the wiring its visible neural
network needs, and reconcile our GDSC-style cell-line names with the DepMap
``ACH-`` identifiers that DeepCDR's omics matrices are keyed on.

The identifier reconciliation is the fiddly part and the reason the comparison
runs on a subset. Our 998 cell lines are GDSC names (``22RV1``, ``42-MG-BA``);
DeepCDR is keyed on DepMap ACH ids; DrugCell on CCLE ``NAME_TISSUE`` strings.
Normalising to alphanumeric-uppercase reconciles ~97%, and the binding constraint
is DeepCDR's three-omics roster of 561 lines rather than the mapping itself.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
COMPARATOR_DIR = ROOT / "data" / "comparators"


def normalise_cellline(name: str) -> str:
    """Strip punctuation and case so GDSC/CCLE/DepMap spellings collide."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


# ---------------------------------------------------------------------------
# DrugCell: ontology -> visible neural network wiring
# ---------------------------------------------------------------------------

def load_drugcell_ontology(
    ontology_path: Path | None = None, gene2ind_path: Path | None = None
) -> Tuple[Dict[str, List[str]], Dict[str, List[int]], List[str], str, Dict[str, int]]:
    """Parse ``drugcell_ont.txt`` into (children, genes, topological order, root, gene index).

    The file holds one edge per line, ``parent <TAB> child <TAB> relation``, where
    relation is ``default`` for term-term edges and ``gene`` for term-gene
    annotations. Terms are returned leaves-first so a single forward pass can
    evaluate them in order.
    """
    ontology_path = ontology_path or COMPARATOR_DIR / "drugcell_drugcell_ont.txt"
    gene2ind_path = gene2ind_path or COMPARATOR_DIR / "drugcell_gene2ind.txt"

    gene_index = {}
    for line in gene2ind_path.read_text().splitlines():
        if not line.strip():
            continue
        idx, gene = line.split("\t")
        gene_index[gene.strip()] = int(idx)

    children: Dict[str, List[str]] = {}
    genes: Dict[str, List[int]] = {}
    all_terms: Set[str] = set()
    has_parent: Set[str] = set()

    for line in ontology_path.read_text().splitlines():
        if not line.strip():
            continue
        parent, child, relation = line.split("\t")
        parent, child, relation = parent.strip(), child.strip(), relation.strip()
        all_terms.add(parent)
        children.setdefault(parent, [])
        genes.setdefault(parent, [])
        if relation == "default":
            children[parent].append(child)
            has_parent.add(child)
            all_terms.add(child)
            children.setdefault(child, [])
            genes.setdefault(child, [])
        else:
            if child in gene_index:
                genes[parent].append(gene_index[child])

    roots = sorted(all_terms - has_parent)
    if len(roots) != 1:
        raise ValueError(f"expected exactly one ontology root, found {len(roots)}: {roots[:5]}")
    root = roots[0]

    # Leaves-first topological order via iterative post-order, so deep ontologies
    # cannot blow the recursion limit.
    order: List[str] = []
    seen: Set[str] = set()
    for start in sorted(all_terms):
        if start in seen:
            continue
        stack = [(start, False)]
        while stack:
            term, expanded = stack.pop()
            if expanded:
                if term not in seen:
                    seen.add(term)
                    order.append(term)
                continue
            if term in seen:
                continue
            stack.append((term, True))
            for child in children.get(term, []):
                if child not in seen:
                    stack.append((child, False))

    logger.info(
        f"DrugCell ontology: {len(all_terms)} terms, root={root}, "
        f"{sum(len(v) for v in genes.values())} gene annotations over {len(gene_index)} genes"
    )
    return children, genes, order, root, gene_index


def load_drugcell_mutations(
    cell2ind_path: Path | None = None, cell2mutation_path: Path | None = None
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Binary mutation matrix and a normalised-name lookup into its rows."""
    cell2ind_path = cell2ind_path or COMPARATOR_DIR / "drugcell_cell2ind.txt"
    cell2mutation_path = cell2mutation_path or COMPARATOR_DIR / "drugcell_cell2mutation.txt"

    names = {}
    for line in cell2ind_path.read_text().splitlines():
        if not line.strip():
            continue
        idx, name = line.split("\t")
        # CCLE ids are NAME_TISSUE; the tissue suffix is not in our identifiers.
        names[normalise_cellline(name.split("_")[0])] = int(idx)

    matrix = np.loadtxt(cell2mutation_path, delimiter=",", dtype=np.float32)
    logger.info(f"DrugCell mutations: {matrix.shape[0]} cell lines x {matrix.shape[1]} genes")
    return matrix, names


# ---------------------------------------------------------------------------
# DeepCDR: three omics keyed on DepMap ACH ids
# ---------------------------------------------------------------------------

def load_depmap_id_map(annotation_path: Path | None = None) -> Dict[str, str]:
    """Normalised cell-line name -> DepMap ``ACH-`` identifier."""
    annotation_path = annotation_path or COMPARATOR_DIR / "deepcdr_annot.txt"
    annotations = pd.read_csv(annotation_path, sep="\t", low_memory=False)

    mapping: Dict[str, str] = {}
    for _, row in annotations.iterrows():
        ach = row.get("depMapID")
        if not isinstance(ach, str) or not ach.startswith("ACH"):
            continue
        for candidate in (str(row.get("CCLE_ID", "")).split("_")[0], row.get("Name", "")):
            key = normalise_cellline(candidate)
            if key:
                mapping.setdefault(key, ach)
    logger.info(f"DepMap id map: {len(mapping)} normalised names")
    return mapping


def load_deepcdr_omics(
    which: Sequence[str] = ("expression", "methylation", "mutation"),
) -> Dict[str, pd.DataFrame]:
    """Load the published DeepCDR omics matrices, indexed by ACH id."""
    files = {
        "expression": "deepcdr_expr.csv",
        "methylation": "deepcdr_meth.csv",
        "mutation": "deepcdr_mut.csv",
    }
    omics = {}
    for name in which:
        path = COMPARATOR_DIR / files[name]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/fetch_comparator_data.sh")
        frame = pd.read_csv(path, index_col=0)
        omics[name] = frame
        logger.info(f"DeepCDR {name}: {frame.shape[0]} cell lines x {frame.shape[1]} features")
    return omics


# ---------------------------------------------------------------------------
# The common subset
# ---------------------------------------------------------------------------

def common_celllines(
    our_celllines: Sequence[str],
    require: Sequence[str] = ("deepcdr", "drugcell"),
) -> Tuple[Set[str], Dict[str, str]]:
    """Cell lines we can feed to every requested comparator.

    Returns the surviving original names plus the name -> ACH mapping. Restricting
    to the intersection is what makes the comparison like-for-like; the cost is
    that DeepCDR's three-omics roster of 561 lines caps it well below our 998, and
    MoSQ has to be re-evaluated on the same subset for the numbers to be
    comparable.
    """
    keep = {normalise_cellline(c) for c in our_celllines}
    ach_map = load_depmap_id_map()

    if "deepcdr" in require:
        omics = load_deepcdr_omics(("expression", "methylation", "mutation"))
        roster = set.intersection(*(set(frame.index) for frame in omics.values()))
        keep = {c for c in keep if ach_map.get(c) in roster}

    if "drugcell" in require:
        _, names = load_drugcell_mutations()
        keep &= set(names)

    survivors = {c for c in our_celllines if normalise_cellline(c) in keep}
    logger.info(
        f"common subset: {len(survivors)} of {len(set(our_celllines))} cell lines "
        f"(requiring {', '.join(require)})"
    )
    return survivors, ach_map
