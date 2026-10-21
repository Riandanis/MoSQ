#!/usr/bin/env python3
"""Performance split by whether the target is a measurement or an extrapolation.

No censoring-specific handling is applied in training. Censoring is, however, derivable
from the source release even though no flag is distributed, so the question can be
*measured* rather than merely disclosed.

GDSC v17.3 records MIN/MAX_CONC_MICROMOLAR per row. A fitted LN_IC50 above
ln(MAX_CONC) is an extrapolation of the dose-response curve beyond the tested range
rather than an observed inhibition point. 63% of our pairs are of that kind.

This script joins every existing prediction dump back to v17.3 -- the join is exact
on (cell line, drug, LN_IC50), 185,294/185,294 rows, with MAX_CONC unique per key --
and reports metrics separately for the two strata. No GPU, no rerun.

**Reading the output.** R2 is normalised by the variance *within* each stratum, so
comparing R2 across strata conflates model skill with target spread. The variance of
each stratum is reported alongside for that reason, and RMSE is the metric to
compare across strata since it is on the original scale.

Writes reports/revision/tables/censoring_strata.{csv,md}
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer.metrics import compute_regression_metrics  # noqa: E402
from gastro_transformer.predlog import load_all_predictions  # noqa: E402

OUT = ROOT / "reports" / "revision"
V173 = ROOT / "data" / "external" / "v17.3_fitted_dose_response.xlsx"  # see README, "External data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def key(cell: pd.Series, drug: pd.Series, value: pd.Series) -> pd.Series:
    return (cell.astype(str).str.strip() + "|" + drug.astype(str).str.strip()
            + "|" + value.round(6).astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="multiseed")
    parser.add_argument("--regime", default="ncc")
    args = parser.parse_args()

    if not V173.exists():
        raise SystemExit(f"{V173} not found; needed to derive censoring")

    v = pd.read_excel(V173)
    v["k"] = key(v.CELL_LINE_NAME, v.DRUG_NAME, v.LN_IC50)
    v["extrapolated"] = v.LN_IC50 > np.log(v.MAX_CONC_MICROMOLAR)
    lookup = v.drop_duplicates("k").set_index("k").extrapolated

    df = load_all_predictions(OUT, experiment=args.experiment, regime=args.regime)
    df["k"] = key(df.cellline_id, df.drug_id, df.y_true)
    df["extrapolated"] = df.k.map(lookup)

    unmatched = int(df.extrapolated.isna().sum())
    if unmatched:
        logger.warning(f"{unmatched:,} predictions ({unmatched/len(df):.2%}) did not "
                       "match v17.3 and are dropped from this table")
        df = df[df.extrapolated.notna()]
    df["stratum"] = np.where(df.extrapolated, "extrapolated", "measured")

    rows = []
    for keys, sub in df.groupby(["arm", "seed", "fold", "stratum"], sort=True):
        metrics = compute_regression_metrics(sub.y_true.values, sub.y_pred.values)
        metrics.update(dict(zip(["arm", "seed", "fold", "stratum"], keys)))
        metrics["var_y"] = float(np.var(sub.y_true.values))
        metrics["n"] = int(len(sub))
        rows.append(metrics)
    per_fold = pd.DataFrame(rows)

    summary = (
        per_fold.groupby(["arm", "stratum"])
        .agg(r2=("r2", "mean"), r2_sd=("r2", "std"),
             rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
             mae=("mae", "mean"), var_y=("var_y", "mean"), n=("n", "mean"))
        .round(4).reset_index()
    )

    # Variance decomposition. A large between-stratum share means the pooled R2 is
    # substantially rewarding separation of potent-in-range from not, which is a
    # drug-level property -- so this number belongs beside the pooled headline.
    head = df[df.arm == "multitoken_dra"] if (df.arm == "multitoken_dra").any() else df
    y = head.y_true.values
    mask = head.extrapolated.values.astype(bool)
    share_e = float(mask.mean())
    total = float(np.var(y))
    within = share_e * float(np.var(y[mask])) + (1 - share_e) * float(np.var(y[~mask]))
    decomp = {
        "extrapolated_share": round(share_e, 4),
        "total_var_y": round(total, 4),
        "within_stratum_var": round(within, 4),
        "between_stratum_var": round(total - within, 4),
        "between_stratum_pct": round(100 * (total - within) / total, 1),
        "mean_measured": round(float(y[~mask].mean()), 4),
        "mean_extrapolated": round(float(y[mask].mean()), 4),
        "mean_gap": round(float(y[mask].mean() - y[~mask].mean()), 4),
    }

    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(tables / "censoring_strata_per_fold.csv", index=False)
    summary.to_csv(tables / "censoring_strata.csv", index=False)
    import json as _json
    (tables / "censoring_decomposition.json").write_text(_json.dumps(decomp, indent=2))
    logger.info(f"between-stratum variance: {decomp['between_stratum_pct']}% of total; "
                f"mean gap {decomp['mean_gap']:+.2f} LN")

    share = float(df.extrapolated.mean())
    lines = [
        "# Performance by censoring stratum",
        "",
        f"Experiment `{args.experiment}`, regime `{args.regime}`. "
        f"{share:.1%} of evaluated pairs are extrapolations above the maximum "
        "tested concentration.",
        "",
        "R2 is normalised by within-stratum variance, so compare RMSE across strata "
        "and R2 only within one.",
        "",
        "| arm | stratum | R2 | RMSE | MAE | Var(y) | n/fold |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {r.arm} | {r.stratum} | {r.r2:.4f} | {r.rmse:.4f} | "
                     f"{r.mae:.4f} | {r.var_y:.3f} | {int(r.n):,} |")
    (tables / "censoring_strata.md").write_text("\n".join(lines))

    logger.info(f"extrapolated share of evaluated pairs: {share:.1%}")
    for _, r in summary.iterrows():
        logger.info(f"  {r.arm:18s} {r.stratum:13s} R2={r.r2:+.4f} RMSE={r.rmse:.4f} "
                    f"Var(y)={r.var_y:.3f}")
    logger.info(f"wrote {tables / 'censoring_strata.csv'}")


if __name__ == "__main__":
    main()
