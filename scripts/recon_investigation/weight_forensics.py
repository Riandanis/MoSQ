#!/usr/bin/env python
"""Did the reconstruction decoder train? Answer it from the weights alone. CPU only.

THE CLAIM UNDER TEST
--------------------
An earlier reading of the training curves held that the reconstruction loss "rises at
every weight where the contrastive term is active, and rises further the harder we
weight it", i.e. that the reconstruction objective "never trained".

THREE INDEPENDENT REFUTATIONS, ALL FREE
---------------------------------------
1. WEIGHT SUPPORT. nn.Linear initialises uniform on [-1/sqrt(fan_in), +1/sqrt(fan_in)]
   = [-0.036084, +0.036084] for the 768->768 decoder. **No weight can leave that
   support without a gradient update.** So `frac_outside_init_support` is exactly 0
   for an untrained head and strictly positive for a trained one. No target model, no
   forward pass, no baseline, no assumptions. Pearson kurtosis corroborates: exactly
   1.8 for a uniform, displaced toward 3 (Gaussian) once SGD has acted.

2. DOSE-RESPONSE. The final reconstruction loss FALLS monotonically with recon weight
   (0.4975 at w=0.5, 0.4672 at w=3, 0.4264 at w=11). The quoted percentages rise only
   because the epoch-1 denominator falls faster than the numerator -- and epoch 1
   falls with weight precisely because a more heavily weighted head learns faster
   within that first epoch. The "+26/+42/+52%" reading is an artifact of a moving
   denominator.

3. FROZEN-HEAD REFERENCE. `contrastive_only` runs at recon_weight=0, so its decoder
   gets zero gradient, yet pretrain_phase3.py:576-578 still computes both losses and
   :590 still logs the reconstruction one. Its loss rises +113% against the trained
   head's +26% on a common target trajectory.

WHAT THIS SCRIPT DOES NOT SHOW
------------------------------
That the term is USEFUL. Refutation 3 in particular is against a *random* MLP, whose
squared error is strictly worse than a constant predictor -- so beating it 4.35x is
entirely compatible with having learned nothing but the global mean. The bar that
matters is mean-prediction, i.e. normalised R2 = 1 - MSE/Var(target) > 0, which needs
eval-mode forward passes and lives in recon_r2.py. Nothing here may be quoted as
evidence the reconstruction objective works.

The removal decision is untouched either way: it rests on the paired downstream result
(-0.0004, CI [-0.0030, +0.0022] at 10 seeds), which no part of this investigation
disturbs.

OUTPUTS (reports/recon_investigation/tables/)
    decoder_weight_forensics.csv   per checkpoint x tensor
    mask_token_forensics.csv       mask tokens vs the zero-gradient null
    history_traces.csv             per-arm per-epoch losses
    weight_dose_response.csv       Finding 2, both framings side by side
    weight_forensics.md            the narrative summary
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHECKPOINTS, INIT_BOUND_768, ROOT, UNIFORM_KURTOSIS, checkpoint_path,
    guard_artifacts, write_table, write_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def pearson_kurtosis(x: np.ndarray) -> float:
    """Non-Fisher kurtosis: 1.8 for uniform, 3.0 for Gaussian."""
    xc = x - x.mean()
    m2 = (xc ** 2).mean()
    return float((xc ** 4).mean() / m2 ** 2) if m2 > 0 else float("nan")


def load_history(subdir: str | None) -> dict | None:
    """The published checkpoint has no sibling phase3_history.json.

    Its history is embedded under the 'history' key of the .pt instead.
    """
    if subdir is not None:
        p = checkpoint_path(subdir).parent / "phase3_history.json"
        return json.loads(p.read_text()) if p.exists() else None
    p = checkpoint_path(None)
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False).get("history")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    guard_artifacts("entry")

    wrows, mrows, hrows = [], [], []

    for name, subdir, rw, cw in CHECKPOINTS:
        ck = checkpoint_path(subdir)
        if not ck.exists():
            logger.warning(f"{name}: checkpoint missing, skipping")
            continue
        state = torch.load(ck, map_location="cpu", weights_only=False)

        # --- 1. decoder weight forensics -------------------------------------
        dec = state.get("recon_decoder_state_dict")
        if dec:
            for key, tensor in dec.items():
                w = tensor.detach().float().numpy().ravel()
                is_weight = key.endswith("weight")
                bound = INIT_BOUND_768 if is_weight else INIT_BOUND_768
                wrows.append(dict(
                    checkpoint=name, recon_weight=rw, contrastive_weight=cw,
                    head_trained=rw > 0, tensor=key, n_params=w.size,
                    std=float(w.std()), max_abs=float(np.abs(w).max()),
                    kurtosis=pearson_kurtosis(w),
                    init_bound=bound,
                    max_over_init_bound=float(np.abs(w).max() / bound),
                    # THE decisive column: exactly 0 without a gradient update.
                    frac_outside_init_support=float((np.abs(w) > bound).mean()),
                ))

        # --- 2. mask tokens ---------------------------------------------------
        dm, cm = state.get("drug_mask_token"), state.get("cellline_mask_token")
        if dm is not None and cm is not None:
            a, b = dm.flatten().float(), cm.flatten().float()
            mrows.append(dict(
                checkpoint=name, recon_weight=rw, head_trained=rw > 0,
                drug_mask_std=float(a.std()), drug_mask_norm=float(a.norm()),
                cl_mask_std=float(b.std()), cl_mask_norm=float(b.norm()),
                pairwise_cosine=float(torch.dot(a, b) / (a.norm() * b.norm())),
                drug_max_over_init=float(a.abs().max() / 0.02 / 3.0),
            ))

        # --- 3. history traces -------------------------------------------------
        hist = load_history(subdir)
        if hist and hist.get("recon_loss"):
            recon = hist["recon_loss"]
            contr = hist.get("contrastive_loss") or [np.nan] * len(recon)
            cos = hist.get("mean_cosine_sim") or [np.nan] * len(recon)
            for ep, (r, c, s) in enumerate(zip(recon, contr, cos), start=1):
                hrows.append(dict(checkpoint=name, recon_weight=rw,
                                  contrastive_weight=cw, epoch=ep, recon_loss=r,
                                  contrastive_loss=c, mean_cosine_sim=s))

    wdf, mdf, hdf = pd.DataFrame(wrows), pd.DataFrame(mrows), pd.DataFrame(hrows)
    write_table(wdf, "decoder_weight_forensics.csv")
    write_table(mdf, "mask_token_forensics.csv")
    write_table(hdf, "history_traces.csv")

    # --- 4. dose-response, both framings ------------------------------------
    dose = []
    for name, g in hdf.groupby("checkpoint", sort=False):
        g = g.sort_values("epoch")
        r = g.recon_loss.to_numpy()
        dose.append(dict(
            checkpoint=name, recon_weight=g.recon_weight.iloc[0],
            contrastive_weight=g.contrastive_weight.iloc[0],
            recon_first=r[0], recon_last=r[-1], recon_min=r.min(),
            pct_change=(r[-1] / r[0] - 1) * 100,
            abs_change=r[-1] - r[0],
            final_cos_sim=g.mean_cosine_sim.iloc[-1],
        ))
    ddf = pd.DataFrame(dose)
    write_table(ddf, "weight_dose_response.csv")

    # --- narrative ----------------------------------------------------------
    trained = wdf[wdf.head_trained]
    untrained = wdf[~wdf.head_trained]
    L = ["# Did the reconstruction decoder train?", "",
         f"`nn.Linear(768,768)` initialises uniform on ±{INIT_BOUND_768:.6f}. "
         "**No weight can leave that support without a gradient update**, and the "
         f"Pearson kurtosis of a uniform is exactly {UNIFORM_KURTOSIS}.", "",
         "## Decoder weight forensics", "",
         wdf[["checkpoint", "recon_weight", "head_trained", "tensor", "std", "max_abs",
              "kurtosis", "max_over_init_bound", "frac_outside_init_support"]]
         .to_markdown(index=False, floatfmt=".5f"), ""]
    if len(untrained):
        L += [f"The zero-gradient arm (`contrastive_only_w0`) has "
              f"`frac_outside_init_support` = {untrained.frac_outside_init_support.max():.5f} "
              f"and kurtosis {untrained['kurtosis'].mean():.3f} — i.e. still uniform, "
              f"still inside the init support, shrunk only by AdamW weight decay "
              f"(max/bound = {untrained.max_over_init_bound.max():.5f}).", ""]
    if len(trained):
        L += [f"Trained arms reach `frac_outside_init_support` up to "
              f"{trained.frac_outside_init_support.max():.5f} and kurtosis up to "
              f"{trained['kurtosis'].max():.3f}, with weights up to "
              f"{trained.max_over_init_bound.max():.2f}x the init bound. "
              "**The head demonstrably received and applied gradient.**", ""]
    L += ["## Dose-response: the final loss FALLS as the weight rises", "",
          ddf.to_markdown(index=False, floatfmt=".4f"), "",
          "Read `recon_last`, not `pct_change`. The percentages rise with weight only "
          "because the epoch-1 denominator falls faster than the numerator — and "
          "epoch 1 falls with weight because a more heavily weighted head learns "
          "faster within that first epoch. The published reading of this table is an "
          "artifact of a moving denominator.", "",
          "## Mask tokens", "",
          mdf.to_markdown(index=False, floatfmt=".4f"), "",
          "## What this does NOT establish", "",
          "That the objective is useful. The frozen-head comparison is against a "
          "*random* MLP, whose squared error is strictly worse than a constant "
          "predictor, so beating it is compatible with having learned only the global "
          "mean. The bar that matters is mean-prediction (normalised R² > 0) and it is "
          "measured in `recon_r2.py`. **The removal decision is unaffected either "
          "way** — it rests on the paired downstream result (−0.0004, CI [−0.0030, "
          "+0.0022], 10 seeds).", ""]
    write_text("\n".join(L), "weight_forensics.md")

    guard_artifacts("exit")
    print("\n".join(L))


if __name__ == "__main__":
    main()
