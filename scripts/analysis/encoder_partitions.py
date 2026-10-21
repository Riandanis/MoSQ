#!/usr/bin/env python3
"""Encoder sweep with partition replication (novel-drug / NCD setting).

Reads the per-run **result JSONs** rather than the prediction parquets. That is
deliberate: the parquet dumps for encoder seed0 were destroyed by the fold_seed
collision (an output-path clash between runs), while every JSON survived. R² is recoverable in full from the JSONs; only per-sample
post-hoc analysis was lost.

Two designs are reported separately because they answer different questions:

* **Primary partition** (``fold_seed = None``, seeds 0/1/2) varies weight
  initialisation on one fixed split of the 215 compounds. Its spread is seed
  noise.
* **Partition replication** (``fold_seed = 1/2/3``, seed 0) resamples the split.
  Its spread is partition noise.

For ``ncd`` and ``scaffold`` the partition is fixed regardless of ``seed``, so the
primary design scores a *single draw* of the compound split. Any comparative claim
between encoders rests on the replication, not on the primary spread -- which is
the whole reason this preset exists.

Writes reports/revision/tables/encoder_partitions.{csv,md}
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "reports" / "revision"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def collect() -> pd.DataFrame:
    rows = []
    for path in sorted(OUT.glob("encoder_*/*.json")):
        name = path.stem
        match = re.match(r"(?P<regime>\w+?)__(?P<arm>\w+?)__seed(?P<seed>\d+)(?:__fs(?P<fs>\d+))?$", name)
        if not match:
            continue
        blob = json.loads(path.read_text())
        average = blob["average"]
        rows.append(
            {
                "encoder": path.parent.name.replace("encoder_", ""),
                "regime": match["regime"],
                "seed": int(match["seed"]),
                "fold_seed": int(match["fs"]) if match["fs"] else -1,
                "design": "partition" if match["fs"] else "primary",
                "r2": average["r2"],
                "r2_fold_std": average.get("r2_std"),
                "pearson_r": average.get("pearson_r"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    runs = collect()
    if runs.empty:
        raise SystemExit("no encoder result JSONs found")

    summary = (
        runs.groupby(["regime", "encoder", "design"])["r2"]
        .agg(mean="mean", sd="std", n="size")
        .round(4)
        .reset_index()
        .sort_values(["regime", "design", "mean"], ascending=[True, True, False])
    )

    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    runs.to_csv(tables / "encoder_partitions_runs.csv", index=False)
    summary.to_csv(tables / "encoder_partitions.csv", index=False)

    lines = [
        "# Drug-encoder sweep: seed replication vs partition replication",
        "",
        "`primary` = one fixed compound split, 3 weight seeds (spread = seed noise).",
        "`partition` = 3 resampled compound splits, seed 0 (spread = partition noise).",
        "",
    ]
    for regime in sorted(summary.regime.unique()):
        lines += [f"## {regime}", "", "| encoder | design | R2 | SD | n |", "|---|---|---|---|---|"]
        block = summary[summary.regime == regime]
        for _, r in block.iterrows():
            sd = "n/a" if pd.isna(r.sd) else f"{r.sd:.4f}"
            lines.append(f"| {r.encoder} | {r.design} | {r['mean']:+.4f} | {sd} | {int(r.n)} |")
        lines.append("")

    (tables / "encoder_partitions.md").write_text("\n".join(lines))

    for regime in sorted(summary.regime.unique()):
        logger.info(f"--- {regime} ---")
        for design in ("primary", "partition"):
            block = summary[(summary.regime == regime) & (summary.design == design)]
            order = " > ".join(f"{r.encoder}({r['mean']:+.4f})" for _, r in block.iterrows())
            logger.info(f"  {design:10s}: {order}")

    logger.info(f"wrote {tables / 'encoder_partitions.csv'}")


if __name__ == "__main__":
    main()
