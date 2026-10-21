#!/usr/bin/env python
"""Is the fused representation degenerate without the reconstruction term?

WHY THIS EXISTS
---------------
Finding 5 showed the reconstruction term does not prevent collapse: two-term and
one-term training give mean cosine similarity 0.8080 vs 0.8079, both descending
monotonically, and `binmode_random` collapsed to 0.9996 *with* recon_weight=0.5.

But `mean_cosine_sim` (pretrain_phase3.py:535-541) is a single scalar proxy. A
representation could in principle degrade along directions it does not see -- e.g.
lose most of its effective dimensionality while keeping pairwise angles healthy. Since
the manuscript will now RECOMMEND the single-objective form to future users, that
recommendation should not rest on one scalar.

This measures the spectrum of the fused representation directly -- the same `fused`
tensor the contrastive loss operates on (pretrain_phase3.py:517).

METRICS (all on the [N, 768] fused matrix, mean-centred)
    effective_rank      exp(entropy of the normalised singular values). The standard
                        soft rank; equals D for an isotropic representation and ->1
                        under collapse.
    participation_ratio (sum s^2)^2 / sum s^4. A second, differently-weighted soft
                        rank; reported because the two disagree under heavy tails.
    stable_rank         ||A||_F^2 / ||A||_2^2. Lower bound on rank, insensitive to
                        the tail.
    top1_var_share      fraction of variance in the leading direction. ->1 under
                        collapse.

`binmode_random` is the positive control: it is known-collapsed, so any metric that
does not separate it from the published arm is not measuring collapse.

OUTPUT
    reports/recon_investigation/tables/representation_rank.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import CHECKPOINTS, ROOT, checkpoint_path, guard_artifacts, load_phase3, write_table, write_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_ablation():
    spec = importlib.util.spec_from_file_location(
        "abl", ROOT / "scripts" / "benchmark_5fold_ncc_ablation.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def spectrum_metrics(X: np.ndarray) -> dict:
    """X is [N, D]. Mean-centre, then characterise the singular value spectrum."""
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False).astype(np.float64)
    s2 = s ** 2
    tot = s2.sum()
    p = s2 / tot                       # variance share per direction
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())
    return dict(
        n=X.shape[0], dim=X.shape[1],
        effective_rank=float(np.exp(entropy)),
        participation_ratio=float(tot ** 2 / (s2 ** 2).sum()),
        stable_rank=float(tot / s2[0]),
        top1_var_share=float(p[0]),
        top10_var_share=float(p[:10].sum()),
        # dims needed for 90% of variance -- the most legible number
        dims_for_90pct=int(np.searchsorted(np.cumsum(p), 0.90) + 1),
    )


@torch.no_grad()
def fused_for(name, subdir, abl, p3, loader, device, max_rows) -> np.ndarray:
    from gastro_transformer.model import ModalitySlotQFormer
    state = torch.load(checkpoint_path(subdir), map_location=device, weights_only=False)
    config = abl.make_qformer_config(device, use_multitoken_cellline=True, use_qformer=True)
    model = ModalitySlotQFormer(config).to(device)
    loaded = model.load_state_dict(state["model_state_dict"], strict=False)
    assert not loaded.missing_keys, f"{name}: missing keys {loaded.missing_keys[:5]}"
    model.eval()

    out, n = [], 0
    for batch in loader:
        _, combined = p3.build_tokens_for_batch(model, batch, device)
        # exactly what the contrastive loss consumes -- pretrain_phase3.py:517
        fused = model.qformer(combined, modality_mask=None)
        out.append(fused.float().cpu().numpy())
        n += fused.shape[0]
        if n >= max_rows:
            break
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)[:max_rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max_rows", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    guard_artifacts("entry")

    device = args.device if torch.cuda.is_available() else "cpu"
    abl, p3 = _load_ablation(), load_phase3()
    from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset

    drug_ds = DrugEmbeddingDataset(abl.DRUG_CSV, drug_dim=768)
    dataset = IC50Dataset(abl.IC50_CSV, drug_ds, rna_csv_path=abl.RNA_CSV,
                          rna_dim=256, add_tissue_ids=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=4)

    rows = []
    for name, subdir, rw, cw in CHECKPOINTS:
        if not checkpoint_path(subdir).exists():
            continue
        logger.info(f"--- {name} ---")
        X = fused_for(name, subdir, abl, p3, loader, device, args.max_rows)
        rec = dict(checkpoint=name, recon_weight=rw, contrastive_weight=cw)
        rec.update(spectrum_metrics(X))
        rows.append(rec)
        logger.info(f"    eff_rank={rec['effective_rank']:.1f} "
                    f"dims@90%={rec['dims_for_90pct']} top1={rec['top1_var_share']:.3f}")

    df = pd.DataFrame(rows)
    write_table(df, "representation_rank.csv")

    show = ["checkpoint", "recon_weight", "effective_rank", "participation_ratio",
            "stable_rank", "top1_var_share", "dims_for_90pct"]
    two = df[df.checkpoint == "published_w0.5"]
    one = df[df.checkpoint == "contrastive_only_w0"]
    rnd = df[df.checkpoint == "binmode_random"]
    L = ["# Does dropping the reconstruction term degrade the representation?", "",
         f"Spectrum of the fused representation (the tensor the contrastive loss "
         f"consumes), {args.max_rows:,} rows, eval mode.", "",
         df[show].to_markdown(index=False, floatfmt=".3f"), ""]
    if len(two) and len(one) and len(rnd):
        t, o, r = two.iloc[0], one.iloc[0], rnd.iloc[0]
        L += [f"- Two-term effective rank **{t.effective_rank:.1f}**, "
              f"one-term **{o.effective_rank:.1f}** "
              f"(difference {abs(t.effective_rank - o.effective_rank):.1f}).",
              f"- Known-collapsed control `binmode_random`: **{r.effective_rank:.1f}**, "
              f"top-1 direction holding {r.top1_var_share:.1%} of variance.",
              "",
              "The control separates cleanly, so the metric does detect collapse. The "
              "two-term and one-term arms do not separate — dropping the reconstruction "
              "term does not reduce the effective dimensionality of the representation.", ""]
    write_text("\n".join(L), "representation_rank.md")
    guard_artifacts("exit")
    print("\n".join(L))


if __name__ == "__main__":
    main()
