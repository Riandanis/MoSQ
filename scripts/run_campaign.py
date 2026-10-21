#!/usr/bin/env python3
"""Drive the full revision campaign, stage by stage.

Python rather than shell deliberately: bash reads a script incrementally by byte
offset, so editing a running campaign script corrupts its execution. Python loads
the file once, which makes the driver safe to edit while it runs.

Every stage is idempotent (``--skip_existing``), so this can be killed and
re-invoked at any point without repeating finished work. If a stage fails the
driver logs it and continues, because a single failed cell should not block the
rest of a multi-day campaign.

Usage::

    python scripts/run_campaign.py                 # all stages
    python scripts/run_campaign.py --from encoder_sweep
    python scripts/run_campaign.py --only analysis
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/revision"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# GPU 1 is preferred: GPU 0 carries unrelated processes, so the campaign keeps
# its footprint there small. Three of the four scheduler slots and every
# single-device stage (pretraining, transfer, figures) run on GPU 1.
# Pass --slots "cuda:1,cuda:1" to stay off GPU 0 entirely.
SLOTS = "cuda:1,cuda:1,cuda:1,cuda:0"
SERIAL_DEVICE = "cuda:1"
SLOTS_IN_USE = SLOTS

# Seed counts per stage. Five where the headline comparisons live; three where
# each run is expensive and three independent runs suffice; two
# for the 10-fold check, which only has to answer "does k change the conclusion".
STAGES = [
    # Ordered by decisiveness, not convenience. If the campaign is cut short, the
    # stages that can change what the paper claims have already run.
    ("multiseed", [0, 1, 2, 3, 4]),   # significance, the headline table
    ("binmode", [0, 1, 2]),           # bin-mode ablation: can change the DR-A story
    ("encoder_sweep", [0, 1, 2]),     # NCD ceiling across chemical encoders
    ("recon_pretrain", []),           # target-free control checkpoint
    ("transfer", [0, 1, 2]),          # frozen-feature transfer
    ("mosq_common", [0, 1, 2]),       # MoSQ on the comparator subset, like-for-like
    ("comparators", [0, 1, 2]),       # faithful DeepCDR / DrugCell / tCNNS
    # Moved ahead of the leak-free stages on 2026-08-19. Every headline comparison
    # currently reports p = 0.0625, because a two-sided signed-rank test at n=5
    # cannot go below that whatever the effect size. Ten seeds drop
    # the floor to about 0.002. This costs ~13 GPU-h against the ~30 h the two
    # leak-free stages need, and it decides whether the paper's central claim can
    # be called significant at all, so it goes first.
    ("wilcoxon_power", [5, 6, 7, 8, 9]),  # lift n to 10 so Wilcoxon can resolve
    ("leakfree_ncc", [0, 1, 2]),      # the inductive/transductive gap
    ("leakfree_ncd", [0, 1, 2]),      # the leak-free NCD number
    ("ncd_baselines", [0, 1, 2]),     # is the NCD ceiling setting-wide
    ("rnafiltered", [0, 1, 2]),       # the 592-cell-line headline subset
    ("capacity", [0, 1, 2]),          # capacity ladder: is the parameter count justified
    ("tenfold", [0, 1, 2]),           # 10-fold CV check (3 seeds)
    ("matched_inputs", [0, 1, 2]),        # supplementary: separates architecture from
                                          # cell-line representation (see docs)
    ("analysis", []),
]

BIN_MODES = ["global", "per_drug_rank", "per_drug_z", "random"]


def run(cmd, log_name: str, cwd=ROOT) -> int:
    """Run a command, appending to a stage log. Returns the exit code."""
    log_path = OUT / f"campaign_{log_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "a") as handle:
        handle.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        handle.flush()
        return subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                              cwd=str(cwd), env=env).returncode


def wait_for_running_matrix() -> None:
    """Block while an externally-started run_matrix is still going.

    Starting a second scheduler over the same jobs would double up, because
    --skip_existing only sees a job as finished once its result JSON is written.
    """
    while True:
        probe = subprocess.run(
            ["bash", "-c", "ps -eo args | grep -c '[r]un_matrix.py'"],
            capture_output=True, text=True,
        )
        if probe.stdout.strip() in ("0", ""):
            return
        logger.info("waiting for an already-running run_matrix to finish...")
        time.sleep(120)


def stage_matrix(preset: str, seeds) -> None:
    rc = run(
        [sys.executable, "-u", str(ROOT / "scripts/run_matrix.py"),
         "--preset", preset, "--seeds", *[str(s) for s in seeds], "--slots", SLOTS_IN_USE],
        preset,
    )
    if rc:
        logger.warning(f"stage {preset} exited {rc}; continuing")


def stage_binmode(seeds) -> None:
    """DR-A pretraining once per bin mode, then fine-tuned through the standard path.

    The bin mode is the only thing that differs between arms, which is what makes
    this a controlled test of the drug-identity hypothesis.
    """
    for mode in BIN_MODES:
        if mode == "global":
            continue  # the shipped checkpoint already is bin_mode=global
        ckpt_dir = ROOT / f"saved_checkpoints/binmode_{mode}"
        if (ckpt_dir / "pretrained_dra.pt").exists():
            logger.info(f"binmode {mode}: pretrained checkpoint present, reusing")
            continue
        logger.info(f"binmode {mode}: pretraining DR-A (15 epochs)")
        rc = run(
            [sys.executable, "-u", str(ROOT / "scripts/pretrain_phase3.py"),
             "--checkpoint", str(ROOT / "saved_checkpoints/pretrained_clrna.pt"),
             "--output_dir", str(ckpt_dir), "--bin_mode", mode,
             "--epochs", "15", "--seed", "0", "--device", SERIAL_DEVICE],
            "binmode",
        )
        if rc:
            logger.warning(f"binmode {mode} pretraining exited {rc}")

    # Fine-tuning is scheduled through the matrix so it uses every GPU slot;
    # only the pretraining above is inherently serial.
    stage_matrix("binmode_ft", seeds)


def stage_recon_pretrain(_seeds) -> None:
    """DR-A pretraining with the contrastive term off: an encoder trained on this
    data that never sees an IC50 label. The tightest target-free control."""
    ckpt_dir = ROOT / "saved_checkpoints/recon_only"
    if (ckpt_dir / "pretrained_dra.pt").exists():
        logger.info("recon-only checkpoint present, reusing")
        return
    rc = run(
        [sys.executable, "-u", str(ROOT / "scripts/pretrain_phase3.py"),
         "--checkpoint", str(ROOT / "saved_checkpoints/pretrained_clrna.pt"),
         "--output_dir", str(ckpt_dir),
         "--contrastive_weight", "0.0", "--recon_weight", "1.0",
         "--epochs", "15", "--seed", "0", "--device", SERIAL_DEVICE],
        "transfer",
    )
    if rc:
        logger.warning(f"recon-only pretraining exited {rc}")


def stage_transfer(seeds) -> None:
    for seed in seeds:
        rc = run(
            [sys.executable, "-u", str(ROOT / "scripts/benchmark_transfer.py"),
             "--seed", str(seed), "--regime", "ncc",
             "--downstream", "xgboost", "standalone_mlp", "tcnn_style",
             "deepcdr_style", "drugcell_style",
             "--device", SERIAL_DEVICE],
            "transfer",
        )
        if rc:
            logger.warning(f"transfer seed {seed} exited {rc}")


def stage_mosq_common(seeds) -> None:
    """MoSQ restricted to the 544 cell lines the faithful comparators can consume.

    Without this the comparator table has no MoSQ row that is measured on the same
    rows, and the comparison is not like-for-like -- which is the whole point of
    rebuilding the comparators.
    """
    for arm in ("multitoken_dra", "standalone_mlp", "xgboost"):
        for seed in seeds:
            rc = run(
                [sys.executable, "-u", str(ROOT / "scripts/run_cv.py"),
                 "--regime", "ncc", "--arm", arm, "--seed", str(seed),
                 "--cellline_subset", "comparator_common",
                 "--experiment", "mosq_common", "--device", SERIAL_DEVICE, "--skip_existing"],
                "comparators",
            )
            if rc:
                logger.warning(f"mosq_common {arm} seed {seed} exited {rc}")


def stage_comparators(seeds) -> None:
    """The faithful reimplementations, on the common subset.

    DrugCell evaluates 2,086 ontology terms sequentially per forward pass and is
    markedly slower than the others, so it runs last and at fewer seeds if the
    queue is still going at daybreak.
    """
    # DrugCell's forward pass walks 2,086 ontology terms sequentially, so an epoch
    # costs roughly an order of magnitude more than the others. It runs last and at
    # a single seed; if that proves affordable the seed count can be raised, and if
    # it does not the table reports n=1 for that row rather than silently stalling
    # every stage behind it.
    per_arm_seeds = {"tcnns": seeds, "deepcdr": seeds, "drugcell": seeds[:1]}
    for arm in ("tcnns", "deepcdr", "drugcell"):
        for seed in per_arm_seeds[arm]:
            rc = run(
                [sys.executable, "-u", str(ROOT / "scripts/benchmark_comparators.py"),
                 "--arm", arm, "--subset", "common", "--seed", str(seed),
                 "--regime", "ncc", "--experiment", "comparators",
                 "--device", SERIAL_DEVICE, "--skip_existing"],
                "comparators",
            )
            if rc:
                logger.warning(f"comparator {arm} seed {seed} exited {rc}")


def stage_wilcoxon_power(seeds) -> None:
    """Extra seeds on the arms that carry the headline comparisons.

    At n=5 a two-sided signed-rank test bottoms out at p = 0.0625 and can never
    reach p < 0.05, whatever the effect size. Ten seeds drop that floor to about
    0.002, so it is worth the few GPU-hours to make the test capable of answering.

    Runs into the existing ``multiseed`` experiment so the analysis layer pools
    all ten seeds automatically.
    """
    for arm in ("multitoken_dra", "qformer_randinit", "standalone_mlp", "xgboost"):
        for seed in seeds:
            rc = run(
                [sys.executable, "-u", str(ROOT / "scripts/run_cv.py"),
                 "--regime", "ncc", "--arm", arm, "--seed", str(seed),
                 "--experiment", "multiseed", "--device", SERIAL_DEVICE, "--skip_existing"],
                "wilcoxon_power",
            )
            if rc:
                logger.warning(f"wilcoxon_power {arm} seed {seed} exited {rc}")


MATCHED_SOURCES = {
    "expression697": "data/matched/rna_expression697.csv",   # DeepCDR's cell input
    "mutation3008": "data/matched/rna_mutation3008.csv",     # DrugCell's cell input
}


def stage_matched_inputs(seeds) -> None:
    """Supplementary analysis: hold one factor fixed, vary the other.

    The main comparator table gives every model the inputs its authors designed,
    which is the fair test of the published pipelines. It does vary cell-line
    representation alongside architecture, though, so this stage separates them:

      MoSQ on 697-gene expression / 3,008-gene mutations  -> architecture fixed,
                                                            representation varied
      DeepCDR on MoSQ's BulkRNABERT embedding             -> representation fixed,
                                                            architecture varied

    Both substitutions are drop-ins: MoSQ's RNA projector and DeepCDR's expression
    branch are both parameterised by input width. Reported as supplementary, not as
    the headline -- crippling a competitor's input is not a fairer comparison.
    """
    for name, path in MATCHED_SOURCES.items():
        if not (ROOT / path).exists():
            logger.warning(f"{path} missing; run scripts/build_matched_inputs.py")
            continue
        for seed in seeds:
            rc = run(
                [sys.executable, "-u", str(ROOT / "scripts/run_cv.py"),
                 "--regime", "ncc", "--arm", "multitoken_dra", "--seed", str(seed),
                 "--rna_csv", str(ROOT / path),
                 "--cellline_subset", "comparator_common",
                 "--experiment", f"matched_mosq_{name}",
                 "--device", SERIAL_DEVICE, "--skip_existing"],
                "matched_inputs",
            )
            if rc:
                logger.warning(f"matched mosq {name} seed {seed} exited {rc}")

    for seed in seeds:
        rc = run(
            [sys.executable, "-u", str(ROOT / "scripts/benchmark_comparators.py"),
             "--arm", "deepcdr", "--subset", "common", "--seed", str(seed),
             "--regime", "ncc", "--cell_source", "mosq_rna",
             "--experiment", "matched_deepcdr_mosqrna",
             "--device", SERIAL_DEVICE, "--skip_existing"],
            "matched_inputs",
        )
        if rc:
            logger.warning(f"matched deepcdr seed {seed} exited {rc}")


def stage_analysis(_seeds) -> None:
    steps = [
        ["scripts/analysis/mean_baselines.py", "--regime", "ncc"],
        ["scripts/analysis/mean_baselines.py", "--regime", "ncd"],
        ["scripts/analysis/aggregate_metrics.py"],
        ["scripts/analysis/significance.py"],
        ["scripts/analysis/per_cellline_ranking.py"],
        ["scripts/analysis/compute_cost.py"],
        ["scripts/analysis/experimental_ceiling.py"],
        ["scripts/analysis/make_figures.py"],
    ]
    for step in steps:
        rc = run([sys.executable, "-u", str(ROOT / step[0]), *step[1:]], "analysis")
        if rc:
            logger.warning(f"{step[0]} exited {rc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", default=None, help="Start at this stage")
    parser.add_argument("--only", default=None, help="Run just this stage")
    parser.add_argument("--slots", default=SLOTS,
                        help="Comma-separated device list, one entry per concurrent slot")
    parser.add_argument("--skip_wait", action="store_true",
                        help="Do not wait for an externally-started run_matrix")
    args = parser.parse_args()

    global SLOTS_IN_USE, SERIAL_DEVICE
    SLOTS_IN_USE = args.slots
    # Serial stages follow whichever device the slots favour.
    devices = [d.strip() for d in args.slots.split(",") if d.strip()]
    SERIAL_DEVICE = max(set(devices), key=devices.count)
    logger.info(f"slots={SLOTS_IN_USE}  serial stages on {SERIAL_DEVICE}")

    if not args.skip_wait:
        wait_for_running_matrix()

    names = [name for name, _ in STAGES]
    if args.only:
        selected = [s for s in STAGES if s[0] == args.only]
    elif args.start:
        selected = STAGES[names.index(args.start):]
    else:
        selected = STAGES

    for name, seeds in selected:
        logger.info(f"=== STAGE {name} (seeds: {seeds or 'n/a'}) ===")
        started = time.time()
        if name == "binmode":
            stage_binmode(seeds)
        elif name == "recon_pretrain":
            stage_recon_pretrain(seeds)
        elif name == "transfer":
            stage_transfer(seeds)
        elif name == "mosq_common":
            stage_mosq_common(seeds)
        elif name == "comparators":
            stage_comparators(seeds)
        elif name == "wilcoxon_power":
            stage_wilcoxon_power(seeds)
        elif name == "matched_inputs":
            stage_matched_inputs(seeds)
        elif name == "analysis":
            stage_analysis(seeds)
        else:
            stage_matrix(name, seeds)
        logger.info(f"=== STAGE {name} done in {(time.time() - started) / 60:.1f} min ===")

    logger.info("=== CAMPAIGN COMPLETE ===")


if __name__ == "__main__":
    main()
