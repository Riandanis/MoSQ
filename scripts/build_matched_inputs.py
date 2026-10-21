#!/usr/bin/env python3
"""Build matched cell-line representations for the supplementary comparison.

The main comparator table gives every model the inputs its authors designed, which
is the fair test of the published pipelines. It does, however, vary the
cell-line representation alongside the architecture. This script produces the
inputs for a supplementary analysis that separates those two factors.

Emitted in MoSQ's RNA-slot format (``cellline_id, emb_0 .. emb_{d-1}``), so
``run_cv.py --rna_csv`` can feed them straight into the model's RNA token:

  rna_expression697.csv   DeepCDR's 697-gene expression matrix
  rna_mutation3008.csv    DrugCell's 3,008-gene binary mutation matrix

Both are dimension-agnostic drop-ins: MoSQ's RNA projector is
``Linear(config.rna_dim, hidden_dim)`` and DeepCDR's expression branch is
``Linear(expression_dim, 256)``, so the substitution needs no architectural change
in either direction.

Cell lines outside a source matrix get a zero row and are flagged, exactly as the
406 without BulkRNABERT already are.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer.comparators.data import (  # noqa: E402
    load_deepcdr_omics,
    load_depmap_id_map,
    load_drugcell_mutations,
    normalise_cellline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = ROOT / "data" / "matched"


def _write(name: str, celllines: list[str], matrix: np.ndarray, covered: int) -> Path:
    frame = pd.DataFrame(matrix, columns=[f"emb_{i}" for i in range(matrix.shape[1])])
    frame.insert(0, "cellline_id", celllines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    frame.to_csv(path, index=False)
    logger.info(
        f"{name}: {matrix.shape[0]} cell lines x {matrix.shape[1]} features, "
        f"{covered} covered ({100 * covered / max(matrix.shape[0], 1):.0f}%) -> {path}"
    )
    return path


def build_expression(celllines: list[str]) -> Path:
    """DeepCDR's 697-gene expression, keyed to our cell-line names."""
    expression = load_deepcdr_omics(("expression",))["expression"]
    ach = load_depmap_id_map()

    matrix = np.zeros((len(celllines), expression.shape[1]), dtype=np.float32)
    covered = 0
    for i, name in enumerate(celllines):
        key = ach.get(normalise_cellline(name))
        if key in expression.index:
            matrix[i] = expression.loc[key].to_numpy(dtype=np.float32)
            covered += 1
    return _write("rna_expression697.csv", celllines, matrix, covered)


def build_mutation(celllines: list[str]) -> Path:
    """DrugCell's 3,008-gene binary mutation vector, keyed to our cell-line names."""
    mutations, name_to_row = load_drugcell_mutations()

    matrix = np.zeros((len(celllines), mutations.shape[1]), dtype=np.float32)
    covered = 0
    for i, name in enumerate(celllines):
        row = name_to_row.get(normalise_cellline(name))
        if row is not None:
            matrix[i] = mutations[row]
            covered += 1
    return _write("rna_mutation3008.csv", celllines, matrix, covered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", nargs="+", default=["expression", "mutation"],
                        choices=["expression", "mutation"])
    args = parser.parse_args()

    ic50 = pd.read_csv(ROOT / "data/ic50_data.csv", usecols=["cellline_id"])
    celllines = sorted(ic50.cellline_id.astype(str).unique())
    logger.info(f"{len(celllines)} cell lines in the IC50 table")

    if "expression" in args.which:
        build_expression(celllines)
    if "mutation" in args.which:
        build_mutation(celllines)


if __name__ == "__main__":
    main()
