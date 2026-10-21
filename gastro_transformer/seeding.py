"""Reproducible seeding for the revision experiment campaign.

Prior to this module nothing in the repo called ``torch.manual_seed`` or
``random.seed``; only the CV splits were reproducible (via local
``np.random.default_rng(42)`` instances). Every published number therefore came
from an unseeded weight-initialisation and dropout draw.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU and all CUDA devices).

    ``deterministic`` additionally requests deterministic kernels. It is off by
    default: ``nn.MultiheadAttention`` under ``use_deterministic_algorithms``
    either raises or falls back to markedly slower kernels, and the campaign
    needs seeded runs whose variance is reported, not bitwise reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUBLAS needs this set before the first CUDA context is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - signature fixed by torch
    """``worker_init_fn`` for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int) -> torch.Generator:
    """Generator for ``DataLoader(generator=...)`` so shuffling is reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
