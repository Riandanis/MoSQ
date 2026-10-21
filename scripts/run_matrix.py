#!/usr/bin/env python3
"""Schedule the multi-seed experiment matrix across GPUs.

A fixed pool of device-pinned worker slots, each running ``run_cv.py`` as a
subprocess. No Ray, no Slurm -- the matrix is a few hundred independent jobs and
a process pool is the whole requirement.

Two slots per GPU by default: the model is ~73M parameters at batch 256, nowhere
near 48 GB, so a second process usefully overlaps the first one's dataloader and
kernel-launch gaps. Expect roughly 1.5x per GPU rather than 2x.

Jobs are idempotent. ``--skip_existing`` is passed through to ``run_cv.py``, which
writes its result JSON only on clean exit, so a killed run resumes without
redoing finished work.

Example::

    python scripts/run_matrix.py --preset multiseed --dry-run
    python scripts/run_matrix.py --preset multiseed --slots cuda:0,cuda:0,cuda:1,cuda:1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Job:
    regime: str
    arm: str
    seed: int
    experiment: str
    extra: Dict[str, str] = field(default_factory=dict)
    # Bare store_true flags. ``extra`` always emits "--key value", so a boolean
    # switch cannot be expressed there.
    flags: List[str] = field(default_factory=list)

    @property
    def tag(self) -> str:
        return f"{self.experiment}/{self.regime}__{self.arm}__seed{self.seed}"

    runner: str = "run_cv.py"

    def command(self, device: str, out_root: str, skip_existing: bool) -> List[str]:
        cmd = [
            sys.executable, "-u",
            str(ROOT / "scripts" / self.runner),
            "--regime", self.regime,
            "--arm", self.arm,
            "--seed", str(self.seed),
            "--experiment", self.experiment,
            "--out_root", out_root,
            "--device", device,
        ]
        for key, value in self.extra.items():
            cmd += [f"--{key}", str(value)]
        for flag in self.flags:
            cmd.append(f"--{flag}")
        if skip_existing:
            cmd.append("--skip_existing")
        return cmd


# Presets. Keep these explicit rather than clever: every reported table is built
# on the matrix, so it should be readable at a glance.
QFORMER_ARMS = ["multitoken_dra", "qformer_clrna", "qformer_randinit"]
BASELINE_ARMS = ["standalone_mlp", "xgboost"]
COMPARATOR_ARMS = ["tcnn_style", "deepcdr_style", "drugcell_style"]


def preset_multiseed(seeds: List[int]) -> List[Job]:
    """5 seeds x {random, ncc} x every arm, on identical folds per seed."""
    arms = QFORMER_ARMS + BASELINE_ARMS + COMPARATOR_ARMS
    return [
        Job(regime=regime, arm=arm, seed=seed, experiment="multiseed")
        for regime in ("ncc", "random")
        for arm in arms
        for seed in seeds
    ]


def preset_tenfold(seeds: List[int]) -> List[Job]:
    """10-fold CV, headline setting only."""
    return [
        Job(
            regime="ncc",
            arm=arm,
            seed=seed,
            experiment="tenfold",
            extra={"n_folds": "10"},
        )
        for arm in ["multitoken_dra", "standalone_mlp", "xgboost"]
        for seed in seeds
    ]


def preset_rnafiltered(seeds: List[int]) -> List[Job]:
    """The 592-cell-line subset that carries the 0.801 headline number."""
    arms = QFORMER_ARMS + BASELINE_ARMS + COMPARATOR_ARMS
    return [
        Job(
            regime="ncc",
            arm=arm,
            seed=seed,
            experiment="rnafiltered",
            extra={"dataset": "rna_filtered"},
        )
        for arm in arms
        for seed in seeds
    ]


def preset_rnafiltered_random(seeds: List[int]) -> List[Job]:
    """The 592-subset random split -- the one submitted figure with no revised counterpart.

    `preset_rnafiltered` covers regime=ncc only, so Figure 4A's 0.837 was still a single
    unseeded pre-revision number with a fold-pooled error bar, in a paper whose every other
    figure now reports across-seed SD. Same dataset flag and same arm as the NCC run, so the
    two are directly comparable; only the regime differs.
    """
    return [
        Job(
            regime="random", arm="multitoken_dra", seed=seed,
            experiment="rnafiltered_random",
            extra={"dataset": "rna_filtered"},
        )
        for seed in seeds
    ]


def preset_leakfree_ncc(seeds: List[int]) -> List[Job]:
    """Inductive arm: DR-A re-pretrained per fold on training cell lines only.

    The long pole of the campaign -- one pretraining per fold per seed.
    """
    return [
        Job(
            regime="strict_ncc", arm="multitoken_dra", seed=seed,
            experiment="leakfree_strict_ncc", runner="run_cv_leakfree.py",
            # Retain per-fold pretraining. run_cv_leakfree deletes each fold's
            # checkpoint once that fold is evaluated, which makes the cache useless
            # for resuming a part-finished stage: a pause after fold 3 left only
            # fold 3 on disk and would have re-pretrained folds 0-2 at ~3.2 h each.
            # A checkpoint is 285 MB against 18 TB free, so keeping them is the
            # obviously better trade.
            flags=["keep_checkpoints"],
        )
        for seed in seeds
    ]


def preset_smlc(seeds: List[int]) -> List[Job]:
    """Experiment A (pre-registered) -- size-matched leaky control.

    The inductive arm differs from the transductive one in two ways at once: no
    label exposure *and* a pretraining corpus of 799 of 998 cell lines. Its
    -0.0205 deficit against random init therefore cannot be attributed to either.
    This arm holds the corpus at 799 while putting every test cell line back in,
    so S - I isolates label exposure with folds, epochs, fine-tuning and the
    pretraining procedure all held fixed.

    Run with --out_root reports/final_experiments so no existing aggregate moves.
    """
    return [
        Job(
            regime="strict_ncc", arm="multitoken_dra", seed=seed,
            experiment="smlc_strict_ncc", runner="run_cv_leakfree.py",
            extra={"pretrain_exclude": "size_matched"},
            flags=["keep_checkpoints"],
        )
        for seed in seeds
    ]


def preset_leakfree_ncc_ckpt(seeds: List[int]) -> List[Job]:
    """Experiment B stage 1 -- inductive pretraining, checkpoints retained.

    `preset_leakfree_ncc` already passes --keep_checkpoints, but the completed
    2026-08 run predates that flag being wired, so
    reports/revision/leakfree_strict_ncc/checkpoints/ holds zero weight files and
    the leak-free transfer cannot be built from it. Re-runs under a distinct
    experiment key so the existing inductive result (0.7396) is not touched.
    """
    return [
        Job(
            regime="strict_ncc", arm="multitoken_dra", seed=seed,
            experiment="leakfree_strict_ncc_ckpt", runner="run_cv_leakfree.py",
            flags=["keep_checkpoints"],
        )
        for seed in seeds
    ]


def preset_leakfree_ncd(seeds: List[int]) -> List[Job]:
    """The NCD number all three of R1/R2/R3 are watching, run leak-free."""
    return [
        Job(
            regime="ncd", arm="multitoken_dra", seed=seed,
            experiment="leakfree_ncd", runner="run_cv_leakfree.py",
            flags=["keep_checkpoints"],  # see preset_leakfree_ncc
        )
        for seed in seeds
    ]


def preset_ncd_baselines(seeds: List[int]) -> List[Job]:
    """Comparators under NCD, so the drug-generalisation ceiling can be shown to be
    a property of the setting rather than of MoSQ."""
    return [
        Job(regime="ncd", arm=arm, seed=seed, experiment="ncd_baselines")
        for arm in ["standalone_mlp", "xgboost", "tcnn_style", "qformer_randinit"]
        for seed in seeds
    ]


DRUG_ENCODERS = {
    "chemberta": "data/drug_embeddings_chemberta.csv",
    "morgan": "data/drug_embeddings_morgan.csv",
    "descriptors": "data/drug_embeddings_descriptors.csv",
    "chemberta_mtr": "data/drug_embeddings_chemberta_mtr.csv",
}


def preset_encoder_sweep(seeds: List[int]) -> List[Job]:
    """Does a different chemical encoder rescue novel-drug generalisation?

    Uses the random-init Q-Former rather than the DR-A checkpoint. That checkpoint
    was pretrained on every drug, so using it under NCD would leak; random init
    makes the comparison leakage-free by construction and isolates the encoder,
    which is the variable under test.
    """
    return [
        Job(
            regime=regime, arm="qformer_randinit", seed=seed,
            experiment=f"encoder_{name}",
            extra={"drug_csv": str(ROOT / path)},
        )
        for name, path in DRUG_ENCODERS.items()
        for regime in ("ncd", "scaffold")
        for seed in seeds
    ]


def preset_encoder_partitions(seeds: List[int]) -> List[Job]:
    """Supplementary: is the encoder ranking stable across drug partitions?

    The main sweep replicates over seeds, which for ``ncd`` varies weight
    initialisation only -- its partition is fixed regardless of seed, so all three
    "seeds" scored one identical split of the 215 compounds. Any *comparative*
    claim between encoders therefore rests on a single draw. This preset holds the
    seed fixed and resamples the partition instead, which is the replication that
    matches the claim.

    ``seeds`` is reinterpreted as the list of ``fold_seed`` values. Results land in
    separate ``__fs<N>`` files, so nothing here overwrites the main sweep.
    """
    return [
        Job(
            regime=regime, arm="qformer_randinit", seed=0,
            experiment=f"encoder_{name}",
            extra={"drug_csv": str(ROOT / path), "fold_seed": str(fs)},
        )
        for name, path in DRUG_ENCODERS.items()
        for regime in ("ncd", "scaffold")
        for fs in seeds
    ]


def preset_style_common(seeds: List[int]) -> List[Job]:
    """Close the gap between the two levels of the comparator comparison.

    Level 1 (architectural analogues on our shared 1080-d vector) runs on the full
    996-cell-line dataset; Level 2 (faithful reimplementations on their own native
    inputs) necessarily runs on the 544-cell-line intersection where every
    method's inputs exist. That leaves "analogue vs faithful" confounded with
    "which cell lines", and the two levels cannot be bridged: tCNN-style scores
    0.7596 and faithful tCNNS 0.7373, but on different subsets.

    Running the analogues on the same 544 lines removes the confound, so the cost
    of the analogue is measurable rather than assumed. Fine-tune only -- no
    pretraining -- so this is cheap.
    """
    return [
        Job(
            regime="ncc", arm=arm, seed=seed, experiment="style_common",
            extra={"cellline_subset": "comparator_common"},
        )
        for arm in COMPARATOR_ARMS
        for seed in seeds
    ]


def preset_dra_noclrna(seeds: List[int]) -> List[Job]:
    """DR-A trained WITHOUT the CLRNA (image) warm start.

    Identical to the headline DR-A arm in every respect except the initialisation
    of the Phase 2 pretraining, so the difference isolates what the image stream
    contributes. Compare against experiment 'multiseed', arm 'multitoken_dra'.
    """
    return [
        Job(
            regime="ncc", arm="multitoken_dra", seed=seed,
            experiment="dra_noclrna",
            extra={"dra_ckpt": str(ROOT / "saved_checkpoints/dra_noclrna/pretrained_dra.pt")},
        )
        for seed in seeds
    ]


def preset_dra_contrastive_only(seeds: List[int]) -> List[Job]:
    """DR-A with the cross-modal reconstruction term switched off.

    This is the arm that decides whether the modality-slot design earns its place.
    DR-A has two objectives and they differ in how much architecture they require:

    * The supervised contrastive term runs on the pooled ``fused`` [B, D] vector
      against quintile bins. Nothing in it references modality structure, so it is
      expressible on any encoder -- including the 0.7 M-parameter MLP.
    * The cross-modal reconstruction term masks one modality's token and
      reconstructs it from the others. That needs separately addressable tokens, a
      learned mask embedding and a variable-length encoder, none of which a
      concatenating MLP provides.

    So if the reconstruction term contributes, the slots are justified by
    measurement rather than by assertion; if it does not, our headline reduces to an
    objective that needs no special architecture, and that is worth establishing
    directly.

    Completes a 2x2 with arms already on disk: full DR-A (multiseed), recon-only
    (transfer), and neither (qformer_randinit).
    """
    return [
        Job(
            regime="ncc", arm="multitoken_dra", seed=seed,
            experiment="dra_contrastive_only",
            extra={"dra_ckpt": str(ROOT / "saved_checkpoints/contrastive_only/pretrained_dra.pt")},
        )
        for seed in seeds
    ]


def preset_dra_recon_only(seeds: List[int]) -> List[Job]:
    """The fourth cell of the DR-A objective 2x2, in comparable units.

    A recon-only checkpoint has existed since 2026-08-19, but only ever as a
    *feature-transfer* arm (frozen queries -> XGBoost/MLP, 0.6204/0.6642 under
    experiment 'transfer'). Those are not comparable with the fine-tuned NCC
    numbers the other three cells report, so the 2x2 could not actually be read.

    This fine-tunes that same checkpoint through the identical NCC path. No
    pretraining -- the checkpoint is reused -- so it costs three fine-tunes.
    """
    ckpt = ROOT / "saved_checkpoints/recon_only/pretrained_dra.pt"
    if not ckpt.exists():
        logger.warning(f"recon_only: {ckpt} missing, skipping")
        return []
    return [
        Job(
            regime="ncc", arm="multitoken_dra", seed=seed,
            experiment="dra_recon_only", extra={"dra_ckpt": str(ckpt)},
        )
        for seed in seeds
    ]


# Reconstruction-weight sweep. The published value is 0.5, at which the term is
# inert: its loss *rises* 26% over training (0.394 -> 0.498) and it contributes 4.6%
# of the total, against a contrastive loss ~11x larger. So the contrastive-only
# ablation could not distinguish "reconstruction does not help" from "reconstruction
# was never trained". These weights bracket the point where the two terms are
# comparable in magnitude: 3.0 puts recon at roughly a fifth of the total, 11.0 at
# roughly half.
RECON_WEIGHTS = ["3.0", "11.0"]


def recon_tag(weight: str) -> str:
    return "reconw" + str(weight).replace(".", "p")


def preset_dra_reconweight(seeds: List[int]) -> List[Job]:
    """Does a *properly weighted* cross-modal reconstruction term contribute?

    Completes the objective sweep with recon_weight in {0 (contrastive-only),
    0.5 (published), 3.0, 11.0} and contrastive_weight=0 (recon-only) as the far
    endpoint. Skips any weight whose checkpoint the pretraining step did not produce,
    so a failed pretrain costs one arm rather than the whole stage.
    """
    jobs: List[Job] = []
    for weight in RECON_WEIGHTS:
        tag = recon_tag(weight)
        ckpt = ROOT / f"saved_checkpoints/{tag}/pretrained_dra.pt"
        if not ckpt.exists():
            logger.warning(f"{tag}: {ckpt} missing, skipping")
            continue
        jobs += [
            Job(regime="ncc", arm="multitoken_dra", seed=seed,
                experiment=f"dra_{tag}", extra={"dra_ckpt": str(ckpt)})
            for seed in seeds
        ]
    return jobs


BIN_MODES = ["global", "per_drug_rank", "per_drug_z", "random"]


def preset_binmode(seeds: List[int]) -> List[Job]:
    """Fine-tune each bin-mode checkpoint through the identical NCC path.

    Scheduled through the matrix rather than run serially so both GPUs are used;
    the pretraining that produces these checkpoints is inherently serial, the
    fine-tuning is not.
    """
    jobs = []
    for mode in BIN_MODES:
        ckpt = (
            ROOT / "saved_checkpoints/pretrained_dra.pt" if mode == "global"
            else ROOT / f"saved_checkpoints/binmode_{mode}/pretrained_dra.pt"
        )
        if not ckpt.exists():
            logger.warning(f"binmode {mode}: {ckpt} missing, skipping")
            continue
        jobs += [
            Job(regime="ncc", arm="multitoken_dra", seed=seed,
                experiment=f"binmode_{mode}", extra={"dra_ckpt": str(ckpt)})
            for seed in seeds
        ]
    return jobs


def preset_capacity(seeds: List[int]) -> List[Job]:
    """Capacity ladder: is the parameter count justified?"""
    return [
        Job(
            regime="ncc", arm="qformer_randinit", seed=seed,
            experiment=f"capacity_q{q}_l{layers}",
            extra={"num_query_tokens": str(q), "qformer_layers": str(layers)},
        )
        for q, layers in [(8, 2), (16, 4), (32, 6)]
        for seed in seeds
    ]


PRESETS = {
    "multiseed": preset_multiseed,
    "tenfold": preset_tenfold,
    "rnafiltered": preset_rnafiltered,
    "leakfree_ncc": preset_leakfree_ncc,
    "leakfree_ncd": preset_leakfree_ncd,
    "smlc": preset_smlc,
    "rnafiltered_random": preset_rnafiltered_random,
    "leakfree_ncc_ckpt": preset_leakfree_ncc_ckpt,
    "ncd_baselines": preset_ncd_baselines,
    "encoder_sweep": preset_encoder_sweep,
    "encoder_partitions": preset_encoder_partitions,
    "style_common": preset_style_common,
    "dra_noclrna": preset_dra_noclrna,
    "dra_contrastive_only": preset_dra_contrastive_only,
    "dra_recon_only": preset_dra_recon_only,
    "dra_reconweight": preset_dra_reconweight,
    "capacity": preset_capacity,
    "binmode_ft": preset_binmode,
}


def worker(
    slot_id: int,
    device: str,
    job_queue: "queue.Queue[Job]",
    args,
    results: List[dict],
    lock: threading.Lock,
) -> None:
    log_dir = Path(args.out_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            return

        cmd = job.command(device, args.out_root, args.skip_existing)
        log_path = log_dir / f"{job.experiment}__{job.regime}__{job.arm}__seed{job.seed}.log"
        started = time.time()
        logger.info(f"[slot {slot_id} {device}] START {job.tag}")

        # Unbuffered, otherwise a multi-hour job shows nothing until it exits.
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(args.omp_threads)}
        with open(log_path, "w") as log_file:
            proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

        elapsed = time.time() - started
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        logger.info(f"[slot {slot_id} {device}] {status} {job.tag} in {elapsed / 60:.1f} min")

        with lock:
            results.append(
                {
                    "tag": job.tag,
                    "device": device,
                    "returncode": proc.returncode,
                    "minutes": round(elapsed / 60, 2),
                    "log": str(log_path),
                }
            )
        job_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=list(PRESETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--slots", default="cuda:0,cuda:0,cuda:1,cuda:1")
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--omp_threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = PRESETS[args.preset](args.seeds)
    slots = [s.strip() for s in args.slots.split(",") if s.strip()]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "jobs.jsonl", "w") as handle:
        for job in jobs:
            handle.write(json.dumps({"tag": job.tag, **job.__dict__}) + "\n")

    logger.info(f"{len(jobs)} jobs across {len(slots)} slots ({args.preset})")
    if args.dry_run:
        for job in jobs:
            print("  " + " ".join(job.command(slots[0], args.out_root, args.skip_existing)))
        return

    job_queue: "queue.Queue[Job]" = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    results: List[dict] = []
    lock = threading.Lock()
    started = time.time()

    threads = [
        threading.Thread(target=worker, args=(i, device, job_queue, args, results, lock), daemon=True)
        for i, device in enumerate(slots)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failures = [r for r in results if r["returncode"] != 0]
    summary_path = out_root / f"matrix_{args.preset}_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(
            {
                "preset": args.preset,
                "seeds": args.seeds,
                "n_jobs": len(jobs),
                "n_failed": len(failures),
                "wall_clock_min": round((time.time() - started) / 60, 1),
                "results": results,
            },
            handle,
            indent=2,
        )

    logger.info(
        f"Done: {len(results) - len(failures)}/{len(jobs)} succeeded in "
        f"{(time.time() - started) / 60:.1f} min -> {summary_path}"
    )
    if failures:
        logger.warning(f"{len(failures)} jobs FAILED:")
        for failure in failures:
            logger.warning(f"  {failure['tag']} -> {failure['log']}")


if __name__ == "__main__":
    main()
