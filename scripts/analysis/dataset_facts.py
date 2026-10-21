#!/usr/bin/env python3
"""Measure dataset-level quantities the paper states, rather than asserting them.

  bin agreement   "the two bin schemes agree on only X% of pairs" -- evidence that
                  global and per-drug binning are substantively different
                  objectives rather than a relabelling.
  completeness    "the drug-cell-line matrix is X% complete (median M of N
                  compounds per cell line)" -- the mechanism explaining why the
                  cross-modal reconstruction target is close to uninformative.
  batch composition
                  for the supervised contrastive loss: positives per anchor and
                  the fraction of anchors with no positive. The published
                  checkpoint predates the logging of these fields, so they are
                  read from the runs that share its binning and batch size.

Output: reports/revision/tables/dataset_facts.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Runs whose phase3_history.json carries the batch-composition fields AND uses
# the published global-quintile binning at the published batch size. These
# differ from the deployed checkpoint only in initialisation or in the
# reconstruction weight, neither of which touches how bins are formed.
GLOBAL_BIN_RUNS = ("dra_noclrna", "contrastive_only", "reconw3p0", "reconw11p0", "recon_only")


def _load_by_path(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bin_agreement(ic50_ds, p3) -> dict:
    """Fraction of pairs receiving the same bin index under two schemes.

    Chance agreement is 1/n_bins because both schemes are equal-mass by
    construction, so the comparison worth reporting is against 0.20, not 0.
    """
    out = {}
    labels = {}
    for mode in ("global", "per_drug_rank", "per_drug_z", "random"):
        labels[mode], _ = p3.compute_bin_labels(ic50_ds, mode=mode, seed=42)
        labels[mode] = np.asarray(labels[mode])

    n = len(labels["global"])
    for other in ("per_drug_rank", "per_drug_z", "random"):
        agree = float((labels["global"] == labels[other]).mean())
        out[f"global_vs_{other}"] = round(agree, 4)
        logger.info(f"bin agreement global vs {other}: {agree:.4f}")
    out["n_pairs"] = int(n)
    out["chance"] = 0.2

    # Per-drug bins are equal-mass within each compound, so a bin label should
    # carry no information about drug identity. Verify rather than assert it:
    # the maximum over drugs of the largest bin share should sit near 1/5.
    drug_ids = np.asarray(ic50_ds.drug_ids)
    worst = 0.0
    for d in np.unique(drug_ids):
        m = drug_ids == d
        if m.sum() < 5:
            continue
        counts = np.bincount(labels["per_drug_rank"][m], minlength=5)
        worst = max(worst, counts.max() / m.sum())
    out["per_drug_rank_max_bin_share"] = round(float(worst), 4)
    logger.info(f"per-drug-rank worst-case bin share: {worst:.4f} (uniform = 0.2)")
    return out


def matrix_completeness(ic50_ds) -> dict:
    """How complete is the drug x cell-line screening matrix?

    The reconstruction objective masks one modality and rebuilds it from the
    others. If nearly every cell line was screened against nearly every
    compound, knowing the compound says little about which cell line it was
    applied to, and the best available solution is to emit the mean embedding.
    """
    df = pd.DataFrame({"cell": ic50_ds.cellline_ids, "drug": ic50_ds.drug_ids})
    n_cells = df["cell"].nunique()
    n_drugs = df["drug"].nunique()
    observed = len(df.drop_duplicates())
    completeness = observed / (n_cells * n_drugs)
    per_cell = df.groupby("cell")["drug"].nunique()
    out = dict(
        n_celllines=int(n_cells),
        n_drugs=int(n_drugs),
        observed_pairs=int(observed),
        completeness=round(float(completeness), 4),
        median_drugs_per_cellline=int(per_cell.median()),
        min_drugs_per_cellline=int(per_cell.min()),
        max_drugs_per_cellline=int(per_cell.max()),
    )
    logger.info(
        f"matrix {completeness:.1%} complete; median {out['median_drugs_per_cellline']} "
        f"of {n_drugs} compounds per cell line"
    )
    return out


def batch_composition() -> dict:
    """Measured positives per anchor, pooled over runs sharing the published bins."""
    per_run = {}
    for run in GLOBAL_BIN_RUNS:
        path = ROOT / "saved_checkpoints" / run / "phase3_history.json"
        if not path.exists():
            continue
        hist = json.loads(path.read_text())
        if "pos_per_anchor" not in hist:
            continue
        pos = np.asarray(hist["pos_per_anchor"], dtype=float)
        empty = np.asarray(hist["frac_anchors_no_positive"], dtype=float)
        per_run[run] = dict(
            pos_per_anchor_mean=round(float(pos.mean()), 3),
            pos_per_anchor_min=round(float(pos.min()), 3),
            frac_anchors_no_positive_max=float(empty.max()),
            n_epochs=int(len(pos)),
        )
    if not per_run:
        return {"available": False}
    means = [v["pos_per_anchor_mean"] for v in per_run.values()]
    out = dict(
        available=True,
        runs=per_run,
        pooled_pos_per_anchor=round(float(np.mean(means)), 1),
        pooled_pos_per_anchor_min=round(float(min(v["pos_per_anchor_min"] for v in per_run.values())), 1),
        max_frac_anchors_no_positive=float(max(v["frac_anchors_no_positive_max"] for v in per_run.values())),
        analytic_expectation=256 / 5,
    )
    logger.info(
        f"measured positives/anchor = {out['pooled_pos_per_anchor']} "
        f"(analytic {out['analytic_expectation']}), "
        f"max empty-anchor fraction = {out['max_frac_anchors_no_positive']}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    args = parser.parse_args()

    _abl = _load_by_path("_abl_df", "scripts/benchmark_5fold_ncc_ablation.py")
    p3 = _load_by_path("_p3_df", "scripts/pretrain_phase3.py")

    from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset

    drug_ds = DrugEmbeddingDataset(_abl.DRUG_CSV, drug_dim=768)
    ic50_ds = IC50Dataset(_abl.IC50_CSV, drug_ds, rna_csv_path=_abl.RNA_CSV,
                          rna_dim=256, add_tissue_ids=True)

    facts = {
        "bin_agreement": bin_agreement(ic50_ds, p3),
        "matrix_completeness": matrix_completeness(ic50_ds),
        "batch_composition": batch_composition(),
    }

    out_dir = Path(args.out_root) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_facts.json").write_text(json.dumps(facts, indent=2))
    logger.info(f"Wrote {out_dir / 'dataset_facts.json'}")


if __name__ == "__main__":
    main()
