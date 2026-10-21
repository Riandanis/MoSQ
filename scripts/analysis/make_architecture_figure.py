#!/usr/bin/env python3
"""Figure 1: the MoSQ architecture and its training phases.

Requirements: an explicit arrow per token type into cross-attention, **Phase 1 and
Phase 2 in structurally separate panels**, and an unambiguous depiction of whether
the pathology stream feeds the IC50 pipeline.

Phase 1 is retained in the figure as a documented negative result (its
contribution is bounded at +-0.007 R2), because a reader cannot check a structural
separation they cannot see.

Three panels, therefore: (a) Phase 1, the image-RNA contrastive stage, drawn muted
and labelled as an explored initialisation that did not pay off; (b) Phase 2, DR-A
pretraining; (c) IC50 fine-tuning. The arrows between panels carry the only things
that actually transfer, and the Phase 1 -> Phase 2 arrow is dashed because that
path was measured to contribute nothing.

Output: reports/revision/figures/architecture.{png,pdf}
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT = ROOT / "reports" / "revision" / "figures"

# Colour-blind-safe, and distinguishable in greyscale by lightness ordering.
# One hue per modality slot, reused wherever that slot appears, so a reader can
# follow a single token type across both panels.
SLOT = {
    "drug": "#4053d3",
    "cancer_type": "#ddb310",
    "tissue": "#00b25d",
    "cellline_rna": "#b51d14",
}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
BLOCK = "#eef1f6"
BLOCK_EDGE = "#9aa5b8"


def box(ax, x, y, w, h, label, *, fc=BLOCK, ec=BLOCK_EDGE, fontsize=8,
        weight="normal", tc=INK, radius=0.02):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.0, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=tc, weight=weight, zorder=3, linespacing=1.35)


def arrow(ax, xy_from, xy_to, *, color=MUTED, lw=1.1, style="-|>", ls="-",
          rad=0.0, zorder=1):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=9, linewidth=lw,
        color=color, linestyle=ls, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))


def draw_clrna_panel(ax) -> None:
    """Phase 1: image-RNA contrastive pretraining of the cell-line encoder.

    Muted throughout, because the panel's job is to show that this path was
    explored and did not pay off. Structurally distinct from Phase 2 -- no drug
    token, no Q-Former, no IC50 -- and the figure makes that separation visible.
    """
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    pale, pale_edge = "#f2f2f2", "#b9b9b9"

    box(ax, 0.30, 8.20, 4.30, 1.10, "H&E tiles\n(patient tumour)",
        fc=pale, ec=pale_edge, fontsize=5.4, tc=MUTED)
    box(ax, 5.40, 8.20, 4.30, 1.10, "Cell-line RNA\n(BulkRNABERT)",
        fc="white", ec=SLOT["cellline_rna"], fontsize=5.4)

    box(ax, 0.30, 6.15, 4.30, 0.95, "Image encoder", fc=pale, ec=pale_edge,
        fontsize=5.6, tc=MUTED)
    box(ax, 5.40, 6.15, 4.30, 0.95, "Cell-line encoder", fc="white",
        ec=BLOCK_EDGE, fontsize=5.6)
    arrow(ax, (2.45, 8.20), (2.45, 7.10), color=pale_edge)
    arrow(ax, (7.55, 8.20), (7.55, 7.10), color=SLOT["cellline_rna"])

    box(ax, 2.70, 4.20, 4.60, 1.00, "Image–RNA\ncontrastive loss",
        fc=MUTED, ec=MUTED, fontsize=5.6, tc="white", weight="bold")
    arrow(ax, (2.45, 6.15), (4.20, 5.20), color=pale_edge, rad=0.10)
    arrow(ax, (7.55, 6.15), (5.80, 5.20), color=pale_edge, rad=-0.10)

    ax.text(5.00, 3.70, "No IC50 label is seen at this phase.",
            ha="center", va="top", fontsize=4.8, color=MUTED)

    ax.add_patch(FancyBboxPatch(
        (0.30, 1.45), 9.40, 1.55, boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=0.9, facecolor="#fdf3f3", edgecolor="#c98c8c", zorder=2))
    ax.text(5.00, 2.23,
            "Evaluated, did not contribute:\n"
            "+0.0005 R²  (95% CI −0.0061 to +0.0072).\n"
            "Reported as a negative result; not claimed.",
            ha="center", va="center", fontsize=4.9, color="#8c3a3a",
            zorder=3, linespacing=1.5)


def draw_panel(ax, *, stage: str = "pretrain") -> None:
    """Phase 2: DR-A pretraining, with the IC50 fine-tuning head as a branch.

    Fine-tuning is not a third panel: it reuses the entire trunk and differs only
    in the head, so drawing the trunk twice would double the figure's width for no
    information. Modality typing is shown by box colour plus one arrow per token
    type into cross-attention, so the typing is unambiguous.
    """
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    tokens = [("drug", "Drug  (ChemBERTa-2)"),
              ("cancer_type", "Cancer type"),
              ("tissue", "Tissue"),
              ("cellline_rna", "Cell-line RNA  (BulkRNABERT)")]

    y0, dy, bh = 8.55, 0.98, 0.74
    for i, (key, label) in enumerate(tokens):
        y = y0 - i * dy
        box(ax, 0.15, y, 4.35, bh, label, fc="white", ec=SLOT[key], fontsize=5.0)
        # One explicit arrow per token type into cross-attention.
        arrow(ax, (4.55, y + bh / 2), (5.75, 6.05), color=SLOT[key], rad=0.14, lw=0.85)

    ax.text(2.32, y0 + bh + 0.22, "Modality inputs — one typed token each",
            ha="center", fontsize=4.8, style="italic", color=MUTED)

    box(ax, 5.75, 7.35, 4.10, 0.62, "32 learned query tokens", fc="white",
        ec=BLOCK_EDGE, fontsize=5.0)
    arrow(ax, (7.80, 7.35), (7.80, 6.90), color=MUTED, lw=0.85)

    box(ax, 5.75, 3.95, 4.10, 2.95, "", fc=BLOCK, ec=BLOCK_EDGE, radius=0.04)
    ax.text(7.80, 6.58, "Modality-Slot Q-Former", ha="center", fontsize=5.4,
            weight="bold", color=INK)
    ax.text(7.80, 6.28, "6 blocks", ha="center", fontsize=4.6, color=MUTED)
    for j, sub in enumerate(["self-attention",
                             "cross-attention over modality slots",
                             "feed-forward"]):
        yy = 5.72 - j * 0.60
        box(ax, 5.95, yy - 0.23, 3.70, 0.46, sub, fc="white", ec=BLOCK_EDGE,
            fontsize=4.5)

    arrow(ax, (7.80, 3.95), (7.80, 3.62), color=MUTED, lw=0.85)
    box(ax, 5.75, 2.92, 4.10, 0.70, "Fused representation  (32 × 768)",
        fc="white", ec=BLOCK_EDGE, fontsize=5.0)

    # Two heads on one trunk: the Phase 2 objective, and the fine-tuning head.
    arrow(ax, (6.80, 2.92), (5.95, 2.30), color=MUTED, rad=0.12, lw=0.85)
    arrow(ax, (8.60, 2.92), (8.85, 2.30), color=INK, rad=-0.12, lw=1.0)
    box(ax, 4.30, 1.45, 3.25, 0.85, "Supervised\ncontrastive loss",
        fc="#2a2a2a", ec="#2a2a2a", fontsize=4.7, tc="white", weight="bold")
    box(ax, 7.85, 1.45, 2.00, 0.85, "IC50 head\n→ LN(IC50)", fc="white",
        ec=INK, fontsize=4.7, weight="bold")
    ax.text(5.00, 1.02,
            "Positives are pairs sharing an IC50 quintile bin; no IC50 value is regressed "
            "at this phase.\nFine-tuning then trains the whole trunk with the IC50 head, per fold.",
            ha="center", va="top", fontsize=4.4, color=MUTED, linespacing=1.45)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(5.75, 3.45))
    draw_clrna_panel(axes[0])
    draw_panel(axes[1], stage="pretrain")

    titles = [
        ("(a)  Phase 1 — image–RNA contrastive",
         "Explored initialisation. Contributes nothing (+0.0005 R²)."),
        ("(b)  Phase 2 — Drug Response-Aware (DR-A), then IC50 fine-tuning",
         "Positives share an IC50 quintile bin. Fine-tuning reuses the whole trunk."),
    ]
    for ax, (title, sub) in zip(axes, titles):
        ax.text(0.0, 1.055, title, transform=ax.transAxes, fontsize=6.6,
                weight="bold", color=INK, ha="left")
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=5.0,
                color=MUTED, ha="left")

    # What crosses between panels, drawn explicitly, because the question is what
    # actually transfers between phases. (a)->(b) is dashed and
    # annotated because that path was measured to contribute nothing; (b)->(c) is
    # solid because it carries every reported result.
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.13, wspace=0.20)

    def connect(i, j, label, sub, *, dashed):
        x0 = axes[i].get_position().x1
        x1 = axes[j].get_position().x0
        y = axes[i].get_position().y0 + 0.435 * axes[i].get_position().height
        fig.add_artist(FancyArrowPatch(
            (x0 + 0.003, y), (x1 - 0.003, y), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=14, linewidth=1.8,
            color=MUTED if dashed else INK,
            linestyle=(0, (4, 2.5)) if dashed else "-", zorder=5))
        fig.text((x0 + x1) / 2, y + 0.020, label, ha="center", va="bottom",
                 fontsize=4.6, color=MUTED if dashed else INK, weight="bold",
                 linespacing=1.2)
        fig.text((x0 + x1) / 2, y - 0.022, sub, ha="center", va="top",
                 fontsize=4.0, color=MUTED, linespacing=1.2)

    connect(0, 1, "cell-line\nencoder", "(init only;\nno benefit)", dashed=True)

    fig.text(0.5, 0.018,
             "The histopathology stream in (a) does not feed the IC50 pipeline: it ends at "
             "the Phase 1 loss, and no image is seen in (b). Every reported result comes "
             "from (b).",
             ha="center", fontsize=4.6, color=MUTED, style="italic", wrap=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"architecture.{ext}", dpi=args.dpi, bbox_inches="tight",
                    facecolor="white")
    logger.info(f"wrote {OUT / 'architecture.png'}")


if __name__ == "__main__":
    main()
