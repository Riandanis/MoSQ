#!/usr/bin/env python3
"""EVAPA as Garai et al. (2026) define it.

EVAPA is an experimental-variability-aware statistic introduced for IC50 prediction.
It is *not* ``1 - Var(residual)/Var(y)`` (a variance-explained ratio with no
experimental variability in it); it has a precise published definition.

**Garai's definition.** EVAPA is the percentage of predictions falling within a
fixed multiplicative factor of the experimental IC50. The factor is derived, not
chosen: from ChEMBL_35 they collated IC50 measurements sharing drug, cell line and
assay (Resazurin / CellTiter-Glo, 72 h incubation), formed 272 pairwise pIC50
differences, fitted a normal, and obtained

    sigma(delta pIC50) = 0.734  ->  sigma(pIC50) = 0.734 / sqrt(2) = 0.519
    90% of measurements within 1.645 * 0.519 = 0.853 log10 units
    10 ** 0.853 = 7.129

so 90% of repeat IC50 measurements agree within a factor of 7.129, and EVAPA is the
share of predictions inside that band.

**Why no replicate data of our own is needed.** The variability estimate is Garai's,
taken from ChEMBL replicates under GDSC-compatible assay conditions. Our targets are
LN(IC50), so the criterion is a threshold on absolute error:

    |log10(pred) - log10(true)| <= 0.853
    |ln(pred)   - ln(true)|     <= 0.853 * ln(10) = 1.9644

i.e. EVAPA = mean(|y_true - y_pred| <= 1.9644) * 100 on the LN scale, with no
back-transform and no rescaling -- Garai are explicit that the metric is computed
on unnormalised IC50.

**Comparability caveat.** Garai filtered out IC50 values above the maximum tested
concentration, training on ~51,000 measured pairs. We retain them (63% of our data;
see ``censoring_strata.py``). Their EVAPA and ours are therefore computed over
different populations, and we report ours split by stratum so the measured-only
figure is comparable to theirs.

Writes reports/revision/tables/evapa.{csv,md}
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer.predlog import load_all_predictions  # noqa: E402

OUT = ROOT / "reports" / "revision"

# The GDSC release table is a 16 MB external download, not a repository artefact
# (see README, "External data"). Without it the run falls back to pooled-only EVAPA.
V173 = ROOT / "data" / "external" / "v17.3_fitted_dose_response.xlsx"

# Garai et al. 2026, Communications Chemistry 9:106, doi 10.1038/s42004-026-01916-9
SIGMA_DELTA_PIC50 = 0.734
LOG10_TOLERANCE = 0.853          # 1.645 * (0.734 / sqrt(2))
FACTOR = 10 ** LOG10_TOLERANCE   # 7.129
LN_TOLERANCE = LOG10_TOLERANCE * math.log(10)   # 1.9644

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evapa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of predictions within Garai's factor-7.129 band, on the LN scale."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= LN_TOLERANCE) * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", default="multiseed,mean_baselines,transfer",
        help="Comma-separated experiments to score. The mean baselines matter "
             "here specifically: Garai report EVAPA for their mean-based "
             "baseline, so ours has to be on the same table or the model rows "
             "are being compared against nothing.",
    )
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiment.split(",") if e.strip()]
    df = load_all_predictions(OUT, experiment=experiments)

    # Stratify by censoring so the measured-only column is comparable with Garai,
    # who excluded extrapolated values entirely.
    if V173.exists():
        v = pd.read_excel(V173)
        k = lambda c, d, x: (c.astype(str).str.strip() + "|" + d.astype(str).str.strip()
                             + "|" + x.round(6).astype(str))
        v["k"] = k(v.CELL_LINE_NAME, v.DRUG_NAME, v.LN_IC50)
        v["e"] = v.LN_IC50 > np.log(v.MAX_CONC_MICROMOLAR)
        lookup = v.drop_duplicates("k").set_index("k").e
        df["extrapolated"] = k(df.cellline_id, df.drug_id, df.y_true).map(lookup)
    else:
        logger.warning(f"{V173} absent -- reporting pooled EVAPA only")
        df["extrapolated"] = np.nan

    rows = []
    for keys, sub in df.groupby(["experiment", "regime", "arm", "seed", "fold"], sort=True):
        entry = dict(zip(["experiment", "regime", "arm", "seed", "fold"], keys))
        entry["evapa_all"] = evapa(sub.y_true.values, sub.y_pred.values)
        meas = sub[sub.extrapolated == False]  # noqa: E712 -- NaN must not match
        extr = sub[sub.extrapolated == True]   # noqa: E712
        entry["evapa_measured"] = evapa(meas.y_true.values, meas.y_pred.values) if len(meas) else np.nan
        entry["evapa_extrapolated"] = evapa(extr.y_true.values, extr.y_pred.values) if len(extr) else np.nan
        entry["mae"] = float(np.mean(np.abs(sub.y_true.values - sub.y_pred.values)))
        rows.append(entry)
    per_fold = pd.DataFrame(rows)

    summary = (
        per_fold.groupby(["experiment", "regime", "arm"])
        .agg(evapa_all=("evapa_all", "mean"), evapa_all_sd=("evapa_all", "std"),
             evapa_measured=("evapa_measured", "mean"),
             evapa_extrapolated=("evapa_extrapolated", "mean"))
        .round(2).reset_index()
    )

    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(tables / "evapa_per_fold.csv", index=False)
    summary.to_csv(tables / "evapa.csv", index=False)

    lines = [
        "# EVAPA (Garai et al. 2026 definition)",
        "",
        f"Share of predictions within a factor of {FACTOR:.3f} of the experimental "
        f"IC50, i.e. |Δln(IC50)| ≤ {LN_TOLERANCE:.4f}.",
        "",
        "`measured` excludes values above the maximum tested concentration and is "
        "the column comparable with Garai, who filtered those out.",
        "",
        "| experiment | regime | arm | EVAPA all | SD | EVAPA measured | EVAPA extrapolated |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {r.experiment} | {r.regime} | {r.arm} | {r.evapa_all:.2f} | "
                     f"{r.evapa_all_sd:.2f} | {r.evapa_measured:.2f} | {r.evapa_extrapolated:.2f} |")
    (tables / "evapa.md").write_text("\n".join(lines))

    logger.info(f"EVAPA band: factor {FACTOR:.3f} (|Δln| <= {LN_TOLERANCE:.4f})")
    for _, r in summary[summary.regime == "ncc"].iterrows():
        logger.info(f"  ncc {r.experiment:14s} {r.arm:18s} all={r.evapa_all:5.2f}%  "
                    f"measured={r.evapa_measured:5.2f}%  extrap={r.evapa_extrapolated:5.2f}%")
    logger.info(f"wrote {tables / 'evapa.csv'}")


if __name__ == "__main__":
    main()
