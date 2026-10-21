#!/usr/bin/env python3
"""Per-slot cross-attention weights across tissue lineages.

The modality-slot design gives
each input type its own token, so the Q-Former's cross-attention is directly
readable: for every query token we can see how much attention mass lands on the
drug slot versus each cell-line slot. That is the one interpretability claim the
architecture actually supports, and it is what this produces.

**No model modification is required.** ``QFormerBlock.forward`` calls
``self.cross_attn(q2, kv, kv, key_padding_mask=...)`` and keeps only ``[0]``.
``nn.MultiheadAttention`` computes the weights regardless (``need_weights``
defaults to True) and returns them as ``[1]``, so a forward hook can capture what
the model discards. Hooking rather than editing keeps the analysed computation
byte-identical to the trained one.

Slot order is derived at runtime rather than hardcoded, since it depends on which
inputs are present. For the IC50 path it is::

    [drug, cancer_type, tissue, cellline_rna]

Three deliberate cautions, because attention is easy to over-read:

1. **Stages are compared, not just DR-A.** Attention from a randomly initialised
   Q-Former is the null: whatever structure appears there is architectural, not
   learned. Only the difference between stages can be attributed to pretraining.
2. **RNA-missing cell lines are reported separately.** For the 406 cell lines
   without RNA-seq the RNA slot carries a learned fallback token, so attention to
   it means something different and pooling the two would be misleading.
3. **Entropy is reported alongside the means.** A uniform distribution over four
   slots has entropy ln(4) = 1.386. Attention that stays near that is not
   evidence of anything, however suggestive the per-slot means look.

Emits:
  reports/revision/tables/attention_by_slot.csv       stage x layer x lineage x slot
  reports/revision/tables/attention_entropy.csv       concentration per stage/layer
  reports/revision/figures/attention_slots.{png,pdf}
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT = ROOT / "reports" / "revision"

# Checkpoints define the pretraining stages we contrast. Random init has no file:
# it is the architectural null against which the other two are read.
#
# ``replicated`` marks stages whose weights are not fully determined by a
# checkpoint and therefore vary from run to run: random_init has none at all,
# and the CLRNA checkpoint leaves ~29 IC50-path tensors uninitialised, which
# load_state_dict(strict=False) fills randomly. A single draw of either is an
# arbitrary sample -- in unseeded testing the slot carrying peak attention at
# random init changed between runs -- so the null has to be replicated and
# reported with a spread, not quoted as a point value. DR-A loads completely
# and is deterministic, so one draw suffices.
STAGES = [
    ("random_init", None, True),
    ("clrna", ROOT / "saved_checkpoints/pretrained_clrna.pt", True),
    ("dra", ROOT / "saved_checkpoints/pretrained_dra.pt", False),
]


def _load_ablation_module():
    spec = importlib.util.spec_from_file_location(
        "_abl_attn", ROOT / "scripts/benchmark_5fold_ncc_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slot_labels(batch, model) -> list[str]:
    """Reproduce the token append order in ModalitySlotQFormer.forward.

    Derived rather than hardcoded: the order depends on which modalities are
    present, and a silently wrong label would turn this figure into a fabrication.
    """
    # Reflect what is actually PASSED to the model, not what the batch happens to
    # contain. IC50Dataset carries an unused 'rna_embed' (patient RNA) key; the
    # IC50 path never forwards it, so counting it produced 5 labels for 4 slots
    # and silently degraded every label to slot0..slot3.
    labels: list[str] = []
    if batch.get("drug_embed") is not None:
        labels.append("drug")
    # Read the flag off the model, not the config: the model sets
    # self.use_multitoken_cellline in __init__ and the config object does not
    # necessarily carry it, which silently produced slot0..slot3 labels.
    if getattr(model, "use_multitoken_cellline", False):
        labels += ["cancer_type", "tissue", "cellline_rna"]
    else:
        labels.append("cellline")
    return labels


def capture(model, loader, device, max_batches: int) -> tuple[np.ndarray, dict]:
    """Run the model with hooks on every cross-attention, returning weights.

    Returns (weights, meta) where weights is [n_samples, n_layers, n_slots]:
    attention mass per modality slot, averaged over query tokens and heads.

    ``meta['per_query']`` retains the query dimension that ``weights`` averages
    away, as [n_layers, n_queries, n_slots]. Averaging over queries can hide
    specialisation among them, so whether the query bank differentiates has to
    be measured rather than inferred from the pooled table -- it is the evidence
    for the over-provisioning claim.
    """
    captured: dict[int, list] = defaultdict(list)
    handles = []
    query_sum = None      # [n_layers, Q, N] running sum over samples
    query_sum_rna = None  # same, restricted to cell lines carrying RNA-seq
    query_n = 0
    query_n_rna = 0

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            # nn.MultiheadAttention -> (attn_output, attn_weights[B, Q, N])
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                captured[layer_idx].append(output[1].detach().float().cpu())
        return hook

    for i, block in enumerate(model.qformer.blocks):
        handles.append(block.cross_attn.register_forward_hook(make_hook(i)))

    tissues, has_rna, per_batch = [], [], []
    model.eval()
    with torch.no_grad():
        for b, batch in enumerate(loader):
            if b >= max_batches:
                break
            captured.clear()
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in batch.items()}
            model(
                drug_embeds=inputs.get("drug_embed"),
                cellline_ids=inputs.get("cellline_id"),
                cancer_type_ids=inputs.get("cancer_type_id"),
                tissue_ids=inputs.get("tissue_id"),
                cellline_rna_embeds=inputs.get("cellline_rna"),
                rna_available=inputs.get("rna_available"),
            )
            if not captured:
                raise RuntimeError("no attention captured; cross_attn returned no weights")
            # [n_layers, B, Q, N] -> mean over queries -> [B, n_layers, N]
            stacked = torch.stack([captured[i][0] for i in sorted(captured)], dim=0)
            per_batch.append(stacked.mean(dim=2).permute(1, 0, 2).numpy())

            t = inputs.get("tissue_id")
            tissues.append(t.cpu().numpy() if t is not None
                           else np.full(stacked.shape[1], -1))
            r = inputs.get("rna_available")
            rna_flag = (r.cpu().numpy().astype(bool) if r is not None
                        else np.ones(stacked.shape[1], dtype=bool))
            has_rna.append(rna_flag)

            # Sum over the batch dimension only, keeping queries resolved.
            batch_sum = stacked.sum(dim=1).numpy()
            query_sum = batch_sum if query_sum is None else query_sum + batch_sum
            query_n += stacked.shape[1]
            if rna_flag.any():
                sel = torch.as_tensor(rna_flag)
                rna_sum = stacked[:, sel].sum(dim=1).numpy()
                query_sum_rna = rna_sum if query_sum_rna is None else query_sum_rna + rna_sum
                query_n_rna += int(sel.sum())

    for h in handles:
        h.remove()
    return np.concatenate(per_batch, 0), {
        "tissue": np.concatenate(tissues), "has_rna": np.concatenate(has_rna),
        "per_query": query_sum / max(query_n, 1),
        "per_query_rna": (query_sum_rna / query_n_rna) if query_n_rna else None,
        "n_query_samples": query_n, "n_query_samples_rna": query_n_rna,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default=os.environ.get("MOSQ_FIG_DEVICE"))
    ap.add_argument("--null_draws", type=int, default=5,
                    help="Independent weight draws for stages that are not fully "
                         "determined by a checkpoint. The random-init null is "
                         "otherwise a single arbitrary sample.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    abl = _load_ablation_module()
    from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset
    from gastro_transformer.model import ModalitySlotQFormer
    from gastro_transformer.utils import TISSUE_TYPES

    drug_ds = DrugEmbeddingDataset(abl.DRUG_CSV, drug_dim=768)
    dataset = IC50Dataset(abl.IC50_CSV, drug_ds, rna_csv_path=abl.RNA_CSV,
                          rna_dim=256, add_tissue_ids=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    rows, ent_rows, query_rows = [], [], []
    labels = None
    for stage, ckpt, replicated in STAGES:
        if ckpt is not None and not Path(ckpt).exists():
            logger.warning(f"{stage}: {ckpt} missing, skipping")
            continue
        draws = args.null_draws if replicated else 1
        draw_weights, draw_per_query, draw_per_query_rna = [], [], []
        meta = None
        for draw in range(draws):
            # Seeded so the null is reproducible; varied so it is not a single
            # arbitrary sample presented as the architectural baseline.
            torch.manual_seed(1000 + draw)
            config = abl.make_qformer_config(device, use_multitoken_cellline=True,
                                             use_qformer=True)
            model = ModalitySlotQFormer(config).to(device)
            if ckpt is not None:
                # Our own checkpoints; they pickle a GastroTransformerConfig, which
                # torch>=2.6 refuses under the weights_only default.
                state = torch.load(ckpt, map_location=device, weights_only=False)
                sd = state.get("model_state_dict", state)
                loaded = model.load_state_dict(sd, strict=False)
                if draw == 0:
                    logger.info(f"{stage}: {len(loaded.missing_keys)} missing, "
                                f"{len(loaded.unexpected_keys)} unexpected keys")

            if labels is None:
                labels = slot_labels(next(iter(loader)), model)

            w_draw, meta = capture(model, loader, device, args.max_batches)
            draw_weights.append(w_draw)
            draw_per_query.append(meta["per_query"])
            if meta["per_query_rna"] is not None:
                draw_per_query_rna.append(meta["per_query_rna"])
            del model

        # Average across draws. For a single-draw stage this is the identity.
        weights = np.mean(draw_weights, axis=0)
        meta["per_query"] = np.mean(draw_per_query, axis=0)
        meta["per_query_rna"] = (np.mean(draw_per_query_rna, axis=0)
                                 if draw_per_query_rna else None)
        meta["n_draws"] = draws
        # Spread of the pooled drug-slot mass across draws, so the paper can
        # state whether a stage-to-stage difference exceeds the null's own noise.
        meta["draw_sd"] = (np.std([w[:, -1, :].mean(0) for w in draw_weights], axis=0)
                           if draws > 1 else np.zeros(weights.shape[2]))
        if draws > 1:
            logger.info(f"{stage}: averaged over {draws} weight draws; "
                        f"final-layer slot SD across draws = "
                        f"{np.round(meta['draw_sd'], 4).tolist()}")
        if len(labels) != weights.shape[2]:
            logger.warning(f"derived {len(labels)} labels for "
                           f"{weights.shape[2]} slots; falling back to indices")
            labels = [f"slot{i}" for i in range(weights.shape[2])]
        logger.info(f"{stage}: {weights.shape[0]} samples, "
                    f"{weights.shape[1]} layers, slots={labels}")

        n_layers = weights.shape[1]
        for layer in range(n_layers):
            w = weights[:, layer, :]
            # Entropy over slots, per sample; ln(n_slots) is the uniform ceiling.
            p = np.clip(w, 1e-12, None)
            p = p / p.sum(1, keepdims=True)
            ent = -(p * np.log(p)).sum(1)
            ent_rows.append({
                "stage": stage, "layer": layer,
                "entropy_mean": ent.mean(), "entropy_sd": ent.std(),
                "uniform_ceiling": np.log(w.shape[1]), "n": len(ent),
            })
            for has_rna in (True, False):
                sel = meta["has_rna"] == has_rna
                if not sel.any():
                    continue
                # Pooled over lineages. Reported separately because a mean taken
                # over the per-lineage rows below weights a 38-sample lineage
                # like a 1,731-sample one; the paper quotes a pair-level mean,
                # so a pair-level row has to exist to quote.
                for j, slot in enumerate(labels):
                    rows.append({
                        "stage": stage, "layer": layer, "lineage": "ALL",
                        "has_rna": has_rna, "slot": slot,
                        "attention_mean": float(w[sel, j].mean()),
                        "attention_sd": float(w[sel, j].std()),
                        "n": int(sel.sum()),
                        "n_draws": meta["n_draws"],
                        "sd_across_draws": float(meta["draw_sd"][j]) if layer == n_layers - 1 else "",
                    })
                for tissue_id in np.unique(meta["tissue"][sel]):
                    tsel = sel & (meta["tissue"] == tissue_id)
                    if tsel.sum() < 30:      # too few to average meaningfully
                        continue
                    for j, slot in enumerate(labels):
                        rows.append({
                            "stage": stage, "layer": layer,
                            "lineage": TISSUE_TYPES.get(int(tissue_id), f"id{tissue_id}"),
                            "has_rna": has_rna, "slot": slot,
                            "attention_mean": float(w[tsel, j].mean()),
                            "attention_sd": float(w[tsel, j].std()),
                            "n": int(tsel.sum()),
                        })

        # Per-query breakdown: does the query bank differentiate at all?
        for group, pq in (("all", meta["per_query"]), ("has_rna", meta["per_query_rna"])):
            if pq is None:
                continue
            for layer in range(pq.shape[0]):
                block = pq[layer]                      # [n_queries, n_slots]
                argmax = block.argmax(axis=1)
                for j, slot in enumerate(labels):
                    query_rows.append({
                        "stage": stage, "layer": layer, "group": group, "slot": slot,
                        "mean_over_queries": float(block[:, j].mean()),
                        "sd_across_queries": float(block[:, j].std()),
                        "min_over_queries": float(block[:, j].min()),
                        "max_over_queries": float(block[:, j].max()),
                        "n_queries_argmax_here": int((argmax == j).sum()),
                        "n_queries": int(block.shape[0]),
                        "n_draws": meta["n_draws"],
                    })

    tables = OUT / "tables"; tables.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "attention_by_slot.csv", index=False)
    pd.DataFrame(ent_rows).to_csv(tables / "attention_entropy.csv", index=False)
    qframe = pd.DataFrame(query_rows)
    qframe.to_csv(tables / "attention_per_query.csv", index=False)
    logger.info(f"wrote {tables/'attention_by_slot.csv'} ({len(frame)} rows)")
    logger.info(f"wrote {tables/'attention_per_query.csv'} ({len(qframe)} rows)")

    if not qframe.empty:
        last = qframe[(qframe.layer == qframe.layer.max()) & (qframe.group == "has_rna")]
        for stage in last.stage.unique():
            s = last[last.stage == stage]
            concentrated = s.loc[s.n_queries_argmax_here.idxmax()]
            logger.info(
                f"  per-query {stage:<12} max SD across queries = "
                f"{s.sd_across_queries.max():.4f}; "
                f"{int(concentrated.n_queries_argmax_here)}/{int(concentrated.n_queries)} "
                f"queries peak on '{concentrated.slot}'"
            )

    # Figure: lineage x slot for the final layer, one panel per stage, RNA-present
    # cell lines only (the fallback token makes the other group a different object).
    final = frame[(frame.layer == frame.layer.max()) & frame.has_rna]
    stages = [s for s in final.stage.unique()]
    if not stages:
        logger.warning("nothing to plot")
        return
    fig, axes = plt.subplots(1, len(stages), figsize=(4.2 * len(stages), 5.2),
                             squeeze=False)
    # One colour scale across all panels. Without vmin/vmax each panel is normalised
    # to its own range, so the random-init null (span 0.054) and DR-A (span 0.283)
    # fill the same colours and the figure reads as though every stage has strong
    # modality preferences. The text explicitly disowns that claim, so an
    # independently-scaled panel set contradicts its own caption (audit P1-9).
    _vals = final.attention_mean.to_numpy(dtype=float)
    _vmin, _vmax = float(np.nanmin(_vals)), float(np.nanmax(_vals))
    for ax, stage in zip(axes[0], stages):
        piv = final[final.stage == stage].pivot_table(
            index="lineage", columns="slot", values="attention_mean")
        im = ax.imshow(piv.values, cmap="viridis", aspect="auto",
                       vmin=_vmin, vmax=_vmax)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=7)
        ax.set_title(stage, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Cross-attention mass per modality slot, by tissue lineage "
                 "(final Q-Former layer, RNA-available cell lines)",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    figures = OUT / "figures"; figures.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"attention_slots.{ext}", dpi=200, bbox_inches="tight")
    logger.info(f"wrote {figures/'attention_slots.png'}")

    ent = pd.DataFrame(ent_rows)
    logger.info("entropy vs uniform ceiling (lower = more concentrated):")
    for _, r in ent[ent.layer == ent.layer.max()].iterrows():
        logger.info(f"  {r.stage:<12} layer {int(r.layer)}  "
                    f"H={r.entropy_mean:.3f} ± {r.entropy_sd:.3f}  "
                    f"(uniform={r.uniform_ceiling:.3f})")


if __name__ == "__main__":
    main()
