"""Shared plumbing for the reconstruction-objective investigation.

Importing this has no side effects beyond defining constants.

ISOLATION. This investigation must not perturb the main results artifacts.
`git status` CANNOT enforce that: `reports/revision/` and every ablation checkpoint
directory are untracked, so damage to them would not show up as a modification. The
guard here is therefore a SHA-256 manifest, checked at entry and exit of every script,
plus `write_table()`, which refuses to write anywhere outside this investigation's
own tree.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "reports" / "recon_investigation"
CKPT_ROOT = ROOT / "saved_checkpoints"

logger = logging.getLogger(__name__)

# nn.Linear default init is kaiming_uniform_(a=sqrt(5)), i.e. U(-b, b) with
# b = 1/sqrt(fan_in). For the 768->768 recon decoder that is:
INIT_BOUND_768 = 768 ** -0.5  # 0.0360844
UNIFORM_KURTOSIS = 1.8  # Pearson (non-Fisher) kurtosis of any uniform distribution

# (name, checkpoint subdir or None for top-level, recon_weight, contrastive_weight).
# `binmode_global` is deliberately absent: it is a byte-identical alias of the
# published arm, not an independent run.
CHECKPOINTS = [
    ("published_w0.5", None, 0.5, 1.0),
    ("contrastive_only_w0", "contrastive_only", 0.0, 1.0),
    ("reconw3p0", "reconw3p0", 3.0, 1.0),
    ("reconw11p0", "reconw11p0", 11.0, 1.0),
    ("recon_only", "recon_only", 1.0, 0.0),
    ("dra_noclrna", "dra_noclrna", 0.5, 1.0),
    ("binmode_random", "binmode_random", 0.5, 1.0),
    ("binmode_per_drug_z", "binmode_per_drug_z", 0.5, 1.0),
    ("binmode_per_drug_rank", "binmode_per_drug_rank", 0.5, 1.0),
]

# Artifacts this investigation must leave untouched.
GUARDED = [
    ROOT / "reports" / "revision" / "tables",
    CKPT_ROOT,
]


def checkpoint_path(subdir: str | None) -> Path:
    return CKPT_ROOT / "pretrained_dra.pt" if subdir is None else CKPT_ROOT / subdir / "pretrained_dra.pt"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _manifest(paths: Iterable[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                # Hashing every 298 MB checkpoint on each call is too slow; size+mtime
                # is a sufficient tripwire for accidental overwrite, and the .pt files
                # are the ones we would only ever damage by writing, not by editing.
                if f.is_file():
                    st = f.stat()
                    out[str(f.relative_to(ROOT))] = f"{st.st_size}:{int(st.st_mtime)}"
        elif p.is_file():
            out[str(p.relative_to(ROOT))] = sha256(p)
    return out


def guard_artifacts(phase: str) -> None:
    """Fail loudly if anything outside this investigation has changed.

    Call at entry (`phase="entry"`) and exit (`phase="exit"`) of every script.
    """
    mdir = OUT / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    ref = mdir / "guard.json"
    current = _manifest(GUARDED)
    if not ref.exists():
        ref.write_text(json.dumps(current, indent=2, sort_keys=True))
        logger.info(f"guard: baseline written ({len(current)} entries)")
        return
    baseline = json.loads(ref.read_text())
    changed = [k for k in set(baseline) & set(current) if baseline[k] != current[k]]
    removed = sorted(set(baseline) - set(current))
    if changed or removed:
        raise RuntimeError(
            f"ISOLATION VIOLATION at {phase}: revision artifacts changed.\n"
            f"  modified: {changed[:10]}\n  removed: {removed[:10]}\n"
            "This investigation must not touch reports/revision/ or saved_checkpoints/."
        )
    logger.info(f"guard ok at {phase} ({len(current)} entries unchanged)")


def write_table(df: pd.DataFrame, name: str) -> Path:
    """Single choke point for output. Refuses anything outside OUT."""
    path = (OUT / "tables" / name).resolve()
    if not str(path).startswith(str((OUT / "tables").resolve())):
        raise ValueError(f"refusing to write outside the investigation tree: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"wrote {path.relative_to(ROOT)} ({len(df)} rows)")
    return path


def write_text(text: str, name: str) -> Path:
    path = (OUT / "tables" / name).resolve()
    if not str(path).startswith(str((OUT / "tables").resolve())):
        raise ValueError(f"refusing to write outside the investigation tree: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    logger.info(f"wrote {path.relative_to(ROOT)}")
    return path


def load_phase3():
    """Import scripts/pretrain_phase3.py by path.

    `scripts/` is not a package, and a plain `import` silently failed for a whole
    campaign here (see the comment at scripts/analysis/make_figures.py:229-232).
    Mirrors the idiom at scripts/analysis/attention_slots.py:83-89. Safe: the module
    has an `if __name__ == '__main__'` guard, so this does not start training.
    """
    path = ROOT / "scripts" / "pretrain_phase3.py"
    spec = importlib.util.spec_from_file_location("pretrain_phase3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
