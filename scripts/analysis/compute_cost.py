#!/usr/bin/env python3
"""Compute-cost accounting for the experiment campaign.

Two traps make this less trivial than summing durations, and both are handled here.

**Only append-only sources are read.** ``reports/revision/matrix_*_summary.json`` is
rewritten on every rerun, and a ``--skip_existing`` job records its *skip* time
rather than its compute time: the multiseed summary reports 0.7 minutes for a stage
that consumed roughly 15 GPU-hours. We therefore read the campaign stage logs, whose
``OK <tag> in X min`` lines are appended and never rewritten.

**Contended durations are not single-model cost.** The scheduler runs several jobs
per GPU, so a per-job wall-clock is inflated by contention, and summing per-job
durations *overstates* device-hours. Leak-free DR-A pretraining logs ~173 min/fold
with three concurrent seeds against roughly 58 min run alone. Presenting the former
as "the cost of training one model" would be wrong by about 3x.

So two figures are emitted and clearly distinguished:

  (a) ``single_model``  — the cheapest observed run of a configuration, which is the
      closest available estimate of uncontended cost, and the one to report.
  (b) ``campaign_total`` — summed occupancy across all jobs, labelled as a
      shared-device figure and not a single-model claim.

Writes ``reports/revision/tables/compute_cost.csv``.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT = ROOT / "reports" / "revision"
TABLES = OUT / "tables"

# "[slot 2 cuda:1] OK multiseed/ncc__xgboost__seed0 in 12.3 min"
OK_LINE = re.compile(
    r"\[slot \d+ (?P<device>cuda:\d+)\]\s+OK\s+(?P<tag>\S+)\s+in\s+(?P<minutes>[\d.]+)\s+min"
)

# Below this a "run" is a --skip_existing no-op, not real compute.
SKIP_THRESHOLD_MIN = 0.5


def parse_stage_logs() -> pd.DataFrame:
    rows = []
    for log in sorted(OUT.glob("campaign_*.log")):
        stage = log.stem.replace("campaign_", "")
        for line in log.read_text(errors="replace").splitlines():
            m = OK_LINE.search(line)
            if not m:
                continue
            rows.append({
                "stage": stage,
                "tag": m.group("tag"),
                "device": m.group("device"),
                "minutes": float(m.group("minutes")),
            })
    return pd.DataFrame(rows)


def parse_manifest() -> pd.DataFrame:
    path = OUT / "manifest.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "wall_clock_s" not in rec:
            continue
        rows.append({
            "experiment": rec.get("experiment"),
            "regime": rec.get("regime"),
            "arm": rec.get("arm"),
            "seed": rec.get("seed"),
            "minutes": float(rec["wall_clock_s"]) / 60.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    jobs = parse_stage_logs()
    if jobs.empty:
        logger.warning("no OK lines found in campaign_*.log - nothing to account for")
        return

    real = jobs[jobs.minutes >= SKIP_THRESHOLD_MIN]
    skipped = len(jobs) - len(real)
    logger.info(f"{len(jobs)} OK lines, {len(real)} with real compute "
                f"({skipped} were --skip_existing no-ops, excluded)")

    # Same tag can appear several times across reruns; the cheapest real run is the
    # least contended one we observed, and the best available proxy for solo cost.
    per_config = (real.groupby(["stage", "tag"])
                      .agg(runs=("minutes", "size"),
                           single_model_min=("minutes", "min"),
                           slowest_min=("minutes", "max"),
                           total_occupancy_min=("minutes", "sum"))
                      .reset_index())

    per_stage = (per_config.groupby("stage")
                 .agg(configs=("tag", "size"),
                      single_model_min=("single_model_min", "median"),
                      campaign_total_h=("total_occupancy_min", lambda s: s.sum() / 60.0))
                 .reset_index()
                 .sort_values("campaign_total_h", ascending=False))

    TABLES.mkdir(parents=True, exist_ok=True)
    per_config.to_csv(TABLES / "compute_cost_per_config.csv", index=False)
    per_stage.to_csv(TABLES / "compute_cost.csv", index=False)

    total_h = per_stage.campaign_total_h.sum()
    summary = {
        "campaign_total_gpu_h_shared_device": round(float(total_h), 2),
        "n_jobs_with_real_compute": int(len(real)),
        "n_skip_existing_noops_excluded": int(skipped),
        "devices": sorted(real.device.unique().tolist()),
        "note": (
            "campaign_total is summed job occupancy on shared devices; jobs ran "
            "concurrently so this overstates true device-hours and must not be "
            "presented as single-model training cost. Use single_model_min, and "
            "state that it was measured under a 300 W power cap."
        ),
    }
    (TABLES / "compute_cost_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(f"campaign total (shared-device occupancy): {total_h:.1f} GPU-h")
    logger.info(f"Wrote {TABLES / 'compute_cost.csv'}")


if __name__ == "__main__":
    main()
