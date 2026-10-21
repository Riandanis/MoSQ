#!/usr/bin/env python3
"""Count parameters for every architecture the paper quotes.

Whether the parameter count is justified should not rest on a hand-typed number,
so the counts are measured here: the deployed checkpoint is counted from its own
``model_state_dict``, and each capacity-ladder rung is counted by constructing
the model at that configuration.

Two counts are reported per architecture, because they answer different
questions:

  ``total``     every parameter in the module, including the image branch and
                the pretraining-only auxiliary heads.
  ``inference`` the subnetwork actually evaluated when predicting IC50. The
                image tower and the reconstruction decoder are inert downstream,
                so quoting ``total`` overstates deployment cost.

Output: reports/revision/tables/param_counts.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Substrings identifying parameters that the IC50 forward pass never touches:
# the image-modality projector left over from Phase 1, the reconstruction decoder
# and its mask tokens, and the auxiliary classification heads that exist only to
# supervise pretraining. Anything matching is excluded from the inference count.
NON_INFERENCE = (
    "image", "recon", "mask_token", "clrna", "proj_image",
    "cancer_head", "tissue_head", "drug_class_head",
)


def _load_by_path(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _split_counts(named_params) -> tuple[int, int]:
    total = 0
    inference = 0
    for name, tensor in named_params:
        n = tensor.numel()
        total += n
        if not any(tag in name.lower() for tag in NON_INFERENCE):
            inference += n
    return total, inference


def count_checkpoint(path: Path) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    return _split_counts(state.items())


def count_config(num_query_tokens: int, qformer_layers: int) -> tuple[int, int, int]:
    """Build the model at one capacity rung and count it.

    Returns (total, inference, qformer_only). The Q-Former subtotal is the
    quantity the capacity ladder is actually varying, so it is reported
    separately from the encoders and head that stay fixed across rungs.
    """
    _abl = _load_by_path("_abl_pc", "scripts/benchmark_5fold_ncc_ablation.py")
    from gastro_transformer.model import ModalitySlotQFormer

    config = _abl.make_qformer_config(
        torch.device("cpu"),
        num_query_tokens=num_query_tokens,
        qformer_layers=qformer_layers,
        use_multitoken_cellline=True,
        use_qformer=True,
        use_qformer_for_ic50=True,
        use_ic50_attn_pool=True,
        num_cell_lines=1000,
    )
    model = ModalitySlotQFormer(config)
    total, inference = _split_counts(model.named_parameters())
    qformer = sum(
        t.numel() for n, t in model.named_parameters() if n.lower().startswith("qformer")
    )
    return total, inference, qformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    args = parser.parse_args()

    rows = []

    for label, rel in [
        ("deployed_dra_checkpoint", "saved_checkpoints/pretrained_dra.pt"),
        ("clrna_checkpoint", "saved_checkpoints/pretrained_clrna.pt"),
    ]:
        path = ROOT / rel
        if not path.exists():
            logger.warning(f"missing {path}")
            continue
        total, inference = count_checkpoint(path)
        rows.append(
            dict(arm=label, num_query_tokens=32, qformer_layers=6, source="checkpoint",
                 params_total=total, params_inference=inference, params_qformer="")
        )
        logger.info(f"{label}: total={total:,} inference={inference:,}")

    for q, layers in [(8, 2), (16, 4), (32, 6)]:
        total, inference, qformer = count_config(q, layers)
        rows.append(
            dict(arm=f"capacity_q{q}_l{layers}", num_query_tokens=q, qformer_layers=layers,
                 source="constructed", params_total=total, params_inference=inference,
                 params_qformer=qformer)
        )
        logger.info(f"q{q}_l{layers}: total={total:,} inference={inference:,} qformer={qformer:,}")

    # The standalone MLP is the floor the ladder is read against.
    _lit = _load_by_path("_lit_pc", "scripts/benchmark_literature_models.py")
    mlp = _lit.StandaloneMLP(input_dim=1080)
    total, inference = _split_counts(mlp.named_parameters())
    rows.append(
        dict(arm="standalone_mlp", num_query_tokens="", qformer_layers="", source="constructed",
             params_total=total, params_inference=inference, params_qformer="")
    )
    logger.info(f"standalone_mlp: total={total:,}")

    out_dir = Path(args.out_root) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "param_counts.csv", index=False)

    summary = {r["arm"]: {"total": int(r["params_total"]),
                          "inference": int(r["params_inference"])} for r in rows}
    (out_dir / "param_counts.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote {out_dir / 'param_counts.csv'}")


if __name__ == "__main__":
    main()
