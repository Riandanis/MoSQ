#!/usr/bin/env python3
"""Pin the GDSC release and characterise duplicates and censoring.

Documents the exact GDSC release and input table, the inclusion criteria, and how
duplicate and censored/extrapolated IC50 values arise. Note that the duplicated
pairs are *not* a GDSC1/GDSC2 artefact: the release used here **predates that split**.

**Release, established by byte-level identity rather than inference.** Our extract
is GDSC **v17.3, release 7.0** (`v17.3_fitted_dose_response.xlsx`). Every one of
our rows matches a v17.3 row exactly, in the original row order; v17.3 restricted to
our drug list reproduces our table exactly. Release 8.5 was excluded first -- only 5
exact matches in 194,750 rows, and 118 of our cell lines are absent from it,
including two withdrawn on authentication failure. In modern terms v17.3 predates
the GDSC1/GDSC2 split and is the screen GDSC1 later expanded, so the accurate
description is "GDSC1 lineage, release 7.0 (v17.3)", citing Iorio et al. 2016
alongside Yang et al. 2013.

**Duplicates.** Exactly 15 of 251 compounds carry *two* DRUG_IDs in v17.3 -- two
independent screens of the same compound, generally over different concentration
ranges (Afatinib at 0.5 uM and at 10 uM, a 20-fold difference). Our preprocessing
keys on drug *name*, so those two screens collapse into duplicate (cell line, drug)
pairs. This is why the two values correlate only moderately (r = 0.56, see
``experimental_ceiling.py``): they are not replicate measurements of one quantity but
separate experiments with different dynamic ranges.

**Censoring.** v17.3 records MIN/MAX_CONC_MICROMOLAR, so censoring *is* derivable
even though no explicit flag exists. A
fitted LN_IC50 above the maximum tested concentration is an extrapolation, not a
measurement, and the fraction is large. This script quantifies it.

Writes reports/revision/tables/dataset_provenance.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "reports" / "revision"
V173 = ROOT / "data" / "external" / "v17.3_fitted_dose_response.xlsx"  # see README, "External data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ours = pd.read_csv(ROOT / "data/ic50_data.csv")
    result = {
        "release": "GDSC v17.3, release 7.0",
        "source_file": "v17.3_fitted_dose_response.xlsx",
        "archive": "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-7.0/",
        "predates_gdsc1_gdsc2_split": True,
        "modern_description": "GDSC1 lineage, release 7.0 (v17.3)",
        "cite": ["Yang et al. 2013", "Iorio et al. 2016"],
        "our_rows": int(len(ours)),
        "our_drugs": int(ours.drug_id.nunique()),
        "our_celllines": int(ours.cellline_id.nunique()),
    }

    dup_mask = ours.duplicated(["cellline_id", "drug_id"], keep=False)
    result["duplicate_rows"] = int(dup_mask.sum())
    result["duplicate_drugs"] = int(ours[dup_mask].drug_id.nunique())

    if not V173.exists():
        logger.warning(f"{V173} absent -- writing release facts without censoring stats")
    else:
        v = pd.read_excel(V173)
        result["v173_rows"] = int(len(v))
        result["v173_drugs"] = int(v.DRUG_NAME.nunique())
        result["v173_celllines"] = int(v.CELL_LINE_NAME.nunique())

        # The duplicate mechanism: same DRUG_NAME, two DRUG_IDs, different ranges.
        per_name = v.groupby("DRUG_NAME").DRUG_ID.nunique()
        multi = sorted(per_name[per_name > 1].index.tolist())
        result["drugs_with_two_drug_ids"] = len(multi)
        result["duplicate_mechanism_confirmed"] = (
            sorted(ours[dup_mask].drug_id.unique()) == multi
        )

        sub = v[v.DRUG_NAME.isin(set(ours.drug_id.unique()))].copy()
        sub["e"] = sub.LN_IC50 > np.log(sub.MAX_CONC_MICROMOLAR)
        above = sub.e
        below = sub.LN_IC50 < np.log(sub.MIN_CONC_MICROMOLAR)
        result["restricted_rows"] = int(len(sub))
        result["frac_above_max_conc"] = round(float(above.mean()), 4)
        result["n_above_max_conc"] = int(above.sum())
        result["frac_below_min_conc"] = round(float(below.mean()), 4)
        result["frac_within_tested_range"] = round(float((~above & ~below).mean()), 4)
        # Is censoring a drug-level property or a cell-line-level one? This decides
        # whether the between-stratum variance is just drug identity in disguise
        # (in which case drug-centred R2 already controls for it) or carries
        # genuine cell-line-specific signal. It is substantially both, so the
        # paper must not claim it is "largely a property of the compound".
        per_drug = sub.groupby("DRUG_NAME").e.agg(["mean", "size"])
        per_drug = per_drug[per_drug["size"] >= 50]
        within = float(sub.groupby("DRUG_NAME").e.var().mean())
        total = float(sub.e.var())
        result["extrapolation_between_drug_pct"] = round(100 * (total - within) / total, 1)
        result["drugs_scored"] = int(len(per_drug))
        result["drugs_over_90pct_extrapolated"] = int((per_drug["mean"] > 0.9).sum())
        result["drugs_under_10pct_extrapolated"] = int((per_drug["mean"] < 0.1).sum())
        result["per_drug_rate_median"] = round(float(per_drug["mean"].median()), 3)
        result["censoring_note"] = (
            "No explicit censoring flag exists, but MIN/MAX_CONC_MICROMOLAR make "
            "censoring derivable. A fitted LN_IC50 above the maximum tested "
            "concentration is an extrapolation of the dose-response curve, not a "
            "measurement."
        )

    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    path = tables / "dataset_provenance.json"
    path.write_text(json.dumps(result, indent=2))

    logger.info(f"release: {result['release']}")
    logger.info(f"duplicate mechanism confirmed: {result.get('duplicate_mechanism_confirmed')}")
    if "frac_above_max_conc" in result:
        logger.info(f"extrapolated above max conc: {result['frac_above_max_conc']:.1%}")
    logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
