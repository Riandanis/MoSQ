#!/usr/bin/env python3
"""Estimate assay reproducibility and the resulting ceiling on R².

R², Pearson and Spearman do not show whether prediction errors are experimentally
acceptable; that needs an estimate of measurement noise. The GDSC table we start from
contains **11,352 duplicated (cell line, drug) pairs**, which preprocessing discards by
keeping the first occurrence (see ``dataset_provenance.py``). Each such pair is two
measurements of the same quantity, so the spread between them estimates assay
reproducibility directly.

Method: for each duplicated pair with two entries, take the difference d = x1 - x2.
If both are measurements of a common true value with independent noise sigma, then
SD(d) = sigma*sqrt(2). The mean of d estimates any systematic offset between first
and second occurrence and is reported separately rather than folded into the noise.

**Two limits:**

1. The mean signed difference is not zero, so the two occurrences are not
   guaranteed to be pure technical replicates -- they may come from different
   screens or assay versions. Any such heterogeneity inflates SD(d), so the sigma
   here is an **upper bound** on measurement noise and the implied R² ceiling is a
   **lower bound** on what a perfect model could reach.
2. The estimate comes from the subset of pairs that happen to be duplicated, which
   need not be representative of the full matrix.

Writes reports/revision/tables/experimental_ceiling.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "reports" / "revision"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

KEY = ["cellline_id", "drug_id"]


def main() -> None:
    df = pd.read_csv(ROOT / "data/ic50_data.csv")
    # FIXED 2026-10-02. The previous version took grouped.nth(0) and grouped.nth(1)
    # separately. In pandas 2.x `nth` returns the selected rows in the frame's original
    # order, not aligned by group, so a[i] and b[i] came from DIFFERENT (cell line, drug)
    # pairs (0.000 of positions aligned). That produced r = 0.2235, SD(d) = 2.1484 and
    # sigma = 1.5191 -- all artefacts. The earlier comment's "0.5553 sorted vs 0.2235 in
    # source order" was this bug. Aggregating first/last per group keeps each pair together.
    # Pre-fix output kept as experimental_ceiling__prefix_misaligned_2026-10-02.json.
    dup = df[df.duplicated(KEY, keep=False)]
    grouped = dup.groupby(KEY, sort=False)["ic50_value"]
    sizes = grouped.size().value_counts().to_dict()

    pairs = grouped.agg(["first", "last"])
    a = pairs["first"].to_numpy()  # source-order first occurrence (what preprocessing keeps)
    b = pairs["last"].to_numpy()
    d = a - b
    n = len(d)

    # Bias-removed spread is the noise estimate; the mean is reported separately.
    sigma = float(d.std() / np.sqrt(2))
    bias = float(d.mean())

    var_y = float(df.ic50_value.var())
    ceiling_r2 = 1.0 - sigma ** 2 / var_y

    # Concentration of the duplication across drugs is the diagnostic that shows
    # these are not technical replicates: scattered duplicates would implicate many
    # drugs, whereas a handful of drugs each spanning most cell lines indicates the
    # same compound screened under two protocols.
    n_drugs_dup = int(dup.drug_id.nunique())
    n_drugs_all = int(df.drug_id.nunique())

    result = {
        "n_replicate_pairs": int(n),
        "n_drugs_involved": n_drugs_dup,
        "n_drugs_total": n_drugs_all,
        "pairs_per_involved_drug": round(n / max(n_drugs_dup, 1)),
        "interpretation": (
            "NOT technical replicates: duplication concentrated in "
            f"{n_drugs_dup}/{n_drugs_all} drugs, each spanning ~{round(n/max(n_drugs_dup,1))} "
            "cell lines. Mechanism confirmed against the source release: these are "
            "exactly the compounds carrying two DRUG_IDs in GDSC v17.3, i.e. two "
            "independent screens over different concentration ranges. The spread "
            "bounds between-screen disagreement, not within-assay reproducibility. "
            "(An earlier version of this note attributed the duplication to the "
            "GDSC1/GDSC2 split; v17.3 predates that split, so that was wrong -- see "
            "scripts/analysis/dataset_provenance.py.)"
        ),
        "group_sizes": {str(k): int(v) for k, v in sizes.items()},
        "mean_signed_difference": round(bias, 4),
        "sd_of_difference": round(float(d.std()), 4),
        "sigma_upper_bound": round(sigma, 4),
        "replicate_pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 4),
        "var_ln_ic50": round(var_y, 4),
        "rmse_noise_floor": round(sigma, 4),
        "max_attainable_r2": round(ceiling_r2, 4),
        "note": (
            "sigma is an UPPER bound (duplicates may not be pure technical "
            "replicates; mean signed difference is non-zero), so max_attainable_r2 "
            "is a LOWER bound on the achievable ceiling."
        ),
    }

    # Where our results sit relative to the ceiling, and how much of the headroom
    # above the per-drug-mean baseline is actually being used.
    baseline = 0.7257
    for name, r2, rmse in (
        ("ncc_full_998cl", 0.7867, 1.2820),
        ("ncc_rna_subset_592cl", 0.8102, None),
    ):
        entry = {
            "r2": r2,
            "headroom_used_pct": round((r2 - baseline) / (ceiling_r2 - baseline) * 100, 1),
        }
        if rmse is not None:
            entry["rmse"] = rmse
            entry["model_only_error"] = round(float(np.sqrt(max(rmse ** 2 - sigma ** 2, 0))), 4)
            entry["noise_share_of_squared_error_pct"] = round(sigma ** 2 / rmse ** 2 * 100, 1)
        result[name] = entry

    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    path = tables / "experimental_ceiling.json"
    path.write_text(json.dumps(result, indent=2))

    logger.info(f"{n:,} replicate pairs; sigma <= {sigma:.4f} LN units "
                f"(bias {bias:+.4f}); ceiling R2 >= {ceiling_r2:.4f}")
    logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
