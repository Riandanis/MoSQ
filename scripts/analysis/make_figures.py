#!/usr/bin/env python3
"""Result figures.

  ablation_ladder   random init -> CLRNA -> DR-A, with the per-drug-mean and
                    XGBoost reference lines, so Phase 1's stagnation is shown
                    rather than described
  scatter_<regime>  predicted vs true LN(IC50) with density contours,
                    MoSQ against each comparator, per protocol
  generalisation    the random -> NCC -> strict NCC -> NCD hierarchy
  clustering        MDS beside t-SNE, plus k-means silhouette scores

All figures are built from the prediction dumps, so they never require a rerun.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer.metrics import compute_regression_metrics  # noqa: E402
from gastro_transformer.predlog import load_all_predictions  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

# Neutral, colour-blind-safe, and consistent across every figure.
INK = "#1c1c1c"
MUTED = "#8a8a8a"
SERIES = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

ARM_LABELS = {
    "multitoken_dra": "MoSQ (DR-A)",
    "qformer_clrna": "Q-Former + CLRNA",
    "qformer_randinit": "Q-Former (rand init)",
    "standalone_mlp": "Standalone MLP",
    "xgboost": "XGBoost",
    "tcnn_style": "tCNN-style",
    "deepcdr_style": "DeepCDR-style",
    "drugcell_style": "DrugCell-style",
    "drug_mean": "Per-drug mean",
}


def _seed_sd(row) -> float:
    """Across-seed SD for a metrics_summary row, falling back to fold-pooled.

    ``r2_std`` pools fold and seed variation; ``r2_seed_std`` is the spread across
    per-seed means and is what every number in the paper quotes.
    Single-seed arms have no seed SD, so those fall back rather than vanish.
    """
    value = row.r2_seed_std.iloc[0] if "r2_seed_std" in row else float("nan")
    if value is None or not np.isfinite(value):
        return float(row.r2_std.iloc[0])
    return float(value)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {path}")


def fig_ablation_ladder(summary: pd.DataFrame, baselines: pd.DataFrame, out: Path) -> None:
    """The pretraining ladder, with Phase 1 shown honestly."""
    ladder = ["qformer_randinit", "qformer_clrna", "multitoken_dra"]
    ncc = summary[(summary.regime == "ncc") & (summary.experiment == "multiseed")]
    rows = [ncc[ncc.arm == a] for a in ladder]
    if any(len(r) == 0 for r in rows):
        logger.warning("ablation ladder: incomplete data, skipping")
        return

    means = [float(r.r2_mean.iloc[0]) for r in rows]
    # Across-seed SD, not the fold-pooled r2_std. The text quotes seed SD throughout,
    # and the two differ by 3.5-7.7x depending on the arm, so plotting r2_std beside
    # seed-SD prose drew error bars several times too wide (audit P1-10).
    errs = [_seed_sd(r) for r in rows]
    labels = ["Random init\n(architecture only)", "+ Phase 1\n(CLRNA)", "+ Phase 2\n(DR-A)"]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(labels, means, yerr=errs, capsize=4, width=0.6,
                  color=[MUTED, MUTED, SERIES[0]], edgecolor=INK, linewidth=0.6)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.012, f"{mean:.3f}",
                ha="center", fontsize=8.5)

    xgb = ncc[ncc.arm == "xgboost"]
    if len(xgb):
        ax.axhline(float(xgb.r2_mean.iloc[0]), ls="--", lw=1, color=SERIES[3])
        ax.text(2.42, float(xgb.r2_mean.iloc[0]), "XGBoost", va="bottom", ha="right",
                fontsize=8, color=SERIES[3])
    if baselines is not None:
        dm = baselines[baselines.arm == "drug_mean"].r2.mean()
        ax.axhline(dm, ls=":", lw=1.2, color=INK)
        ax.text(2.42, dm, "Per-drug mean baseline", va="bottom", ha="right",
                fontsize=8, color=INK)

    # Phase 1 adds essentially nothing; the axis should make that visible rather
    # than exaggerate it, so start just below the weakest bar.
    ax.set_ylim(min(means + [0.70]) - 0.06, max(means) + 0.05)
    ax.set_ylabel("R² (5-fold NCC, 998 cell lines)")
    ax.set_title("Pretraining ablation", loc="left", fontsize=10)
    _save(fig, out / "ablation_ladder.png")


def fig_scatter(preds: pd.DataFrame, regime: str, out: Path) -> None:
    """Predicted vs true with density contours."""
    arms = [a for a in ["multitoken_dra", "standalone_mlp", "xgboost", "tcnn_style"]
            if a in set(preds.arm)]
    if not arms:
        return

    sub = preds[(preds.regime == regime) & (preds.seed == preds.seed.min())]
    if not len(sub):
        return

    fig, axes = plt.subplots(1, len(arms), figsize=(3.1 * len(arms), 3.3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, arm in zip(axes, arms):
        data = sub[sub.arm == arm]
        if not len(data):
            continue
        y_true, y_pred = data.y_true.values, data.y_pred.values
        metrics = compute_regression_metrics(y_true, y_pred)

        ax.hexbin(y_true, y_pred, gridsize=55, cmap="Blues", bins="log",
                  mincnt=1, linewidths=0)
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.9, ls="--")
        ax.set_title(
            f"{ARM_LABELS.get(arm, arm)}\nR²={metrics['r2']:.3f}  "
            f"RMSE={metrics['rmse']:.2f}",
            fontsize=9, loc="left",
        )
        ax.set_xlabel("True LN(IC50)")
    axes[0].set_ylabel("Predicted LN(IC50)")
    fig.suptitle(f"Predicted vs true, {regime} protocol", fontsize=10, x=0.02, ha="left")
    _save(fig, out / f"scatter_{regime}.png")


def _reported_r2(summary, regime, arm, y_true, y_pred) -> str:
    """Panel label quoting the value the manuscript reports for this cell.

    Falls back to the pooled statistic only when the summary has no matching row,
    and marks it so the two are never silently conflated.
    """
    if summary is not None:
        row = summary[(summary.regime == regime) & (summary.arm == arm)]
        if len(row):
            row = row.sort_values("n_seeds").iloc[-1]
            value = row.r2_seed_mean if pd.notna(row.r2_seed_mean) else row.r2_mean
            sd = row.r2_seed_std if pd.notna(row.r2_seed_std) else float("nan")
            n = int(row.n_seeds) if pd.notna(row.n_seeds) else 1
            tail = f" ± {sd:.4f}" if np.isfinite(sd) else ""
            return f"R² = {value:.4f}{tail}  ({n} seed{'s' if n > 1 else ''})"
    m = compute_regression_metrics(y_true, y_pred)
    return f"R² = {m['r2']:.3f} (pooled, 1 seed)"


def fig_scatter_combined(preds: pd.DataFrame, out: Path,
                         summary: pd.DataFrame = None) -> None:
    """Supplementary S1: all four protocols and four models in one grid.

    Rows are protocols in decreasing order of information available to the model,
    columns are models. The strict-NCC row carries only MoSQ, and that is not a gap
    in the experiment: the other three models have **no pretraining stage at all**,
    so excluding test cell lines from pretraining is a no-op for them and their
    inductive result is their transductive result by construction. The row is
    annotated to say so rather than left looking unfinished.
    """
    REGIMES = [("random", "Random split"),
               ("ncc", "NCC — cell-line-blind, transductive"),
               ("strict_ncc", "Strict NCC — cell-line-blind, inductive"),
               ("ncd", "NCD — drug-blind")]
    ARMS = ["multitoken_dra", "standalone_mlp", "xgboost", "tcnn_style"]

    present = {(r, a) for r, a in zip(preds.regime, preds.arm)}
    if not present:
        logger.warning("combined scatter: no predictions, skipping")
        return

    fig, axes = plt.subplots(len(REGIMES), len(ARMS),
                             figsize=(2.85 * len(ARMS), 2.95 * len(REGIMES)),
                             sharex=True, sharey=True)

    for i, (regime, reg_label) in enumerate(REGIMES):
        sub = preds[(preds.regime == regime) & (preds.seed == preds.seed.min())]
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            data = sub[sub.arm == arm] if len(sub) else sub
            if not len(data):
                # No dump for this cell. For the inductive row this is expected.
                ax.set_xticks([]); ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                if regime == "strict_ncc":
                    ax.text(0.5, 0.5,
                            "no pretraining stage\n→ identical to the\nNCC panel above",
                            ha="center", va="center", fontsize=7.5, color=MUTED,
                            style="italic", linespacing=1.5)
                continue

            y_true, y_pred = data.y_true.values, data.y_pred.values
            ax.hexbin(y_true, y_pred, gridsize=48, cmap="Blues", bins="log",
                      mincnt=1, linewidths=0)
            lo = float(min(y_true.min(), y_pred.min()))
            hi = float(max(y_true.max(), y_pred.max()))
            ax.plot([lo, hi], [lo, hi], color=INK, lw=0.9, ls="--")
            ax.set_title(_reported_r2(summary, regime, arm, y_true, y_pred),
                         fontsize=8, loc="left", pad=3)

        axes[i, 0].set_ylabel(f"{reg_label}\n\nPredicted LN(IC50)", fontsize=8.5)

    for j, arm in enumerate(ARMS):
        axes[0, j].annotate(ARM_LABELS.get(arm, arm),
                            xy=(0.5, 1.18), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=10, weight="bold")
    for j in range(len(ARMS)):
        axes[-1, j].set_xlabel("True LN(IC50)", fontsize=8.5)

    fig.suptitle("Predicted versus true LN(IC50), by protocol and model "
                 "(hexbin, log-scaled density; dashed line is identity)",
                 fontsize=9.5, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out / "scatter_combined.png")


def fig_generalisation(summary: pd.DataFrame, baselines: pd.DataFrame, out: Path) -> None:
    """The protocol hierarchy, which is the paper's structural claim."""
    wanted = [
        ("random", "multiseed", "Random split"),
        ("ncc", "multiseed", "NCC (transductive)"),
        ("strict_ncc", "leakfree_strict_ncc", "Strict NCC (inductive)"),
        ("ncd", "leakfree_ncd", "NCD (novel drugs)"),
    ]
    labels, means, errs = [], [], []
    for regime, experiment, label in wanted:
        row = summary[
            (summary.regime == regime)
            & (summary.experiment == experiment)
            & (summary.arm == "multitoken_dra")
        ]
        if not len(row):
            continue
        labels.append(label)
        means.append(float(row.r2_mean.iloc[0]))
        errs.append(_seed_sd(row))  # across-seed SD; see the ablation ladder above
    if not labels:
        logger.warning("generalisation figure: no data, skipping")
        return

    fig, ax = plt.subplots(figsize=(5.6, 0.55 * len(labels) + 1.6))
    ypos = np.arange(len(labels))
    ax.barh(ypos, means, xerr=errs, height=0.55, color=SERIES[0],
            edgecolor=INK, linewidth=0.6, capsize=3)
    for y, mean in zip(ypos, means):
        ax.text(mean + 0.015, y, f"{mean:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()

    if baselines is not None:
        dm = baselines[baselines.arm == "drug_mean"].r2.mean()
        ax.axvline(dm, ls=":", lw=1.2, color=INK)
        ax.text(dm, -0.75, "per-drug mean", fontsize=8, ha="center", color=INK)

    ax.set_xlabel("R²")
    ax.set_xlim(min(0, min(means) - 0.1), 1.0)
    ax.set_title("Generalisation hierarchy, MoSQ", loc="left", fontsize=10)
    _save(fig, out / "generalisation_hierarchy.png")


def fig_clustering(out: Path, max_samples: int = 4000, seed: int = 42) -> None:
    """MDS beside t-SNE, with k-means silhouette scores.

    Embeddings come from the shipped checkpoints. Points are coloured by IC50
    quintile, which is what the DR-A objective is supposed to organise.
    """
    import torch
    from sklearn.cluster import KMeans
    from sklearn.manifold import MDS, TSNE
    from sklearn.metrics import silhouette_score

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_abl_fig", ROOT / "scripts/benchmark_5fold_ncc_ablation.py"
    )
    abl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abl)

    from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset
    from gastro_transformer.model import ModalitySlotQFormer

    device = os.environ.get("MOSQ_FIG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    drug_ds = DrugEmbeddingDataset(abl.DRUG_CSV, drug_dim=768)
    dataset = IC50Dataset(abl.IC50_CSV, drug_ds, rna_csv_path=abl.RNA_CSV,
                          rna_dim=256, add_tissue_ids=True)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(max_samples, len(dataset)), replace=False)

    stages = [("Random init", None), ("CLRNA", ROOT / "saved_checkpoints/pretrained_clrna.pt"),
              ("DR-A", ROOT / "saved_checkpoints/pretrained_dra.pt")]
    available = [(n, p) for n, p in stages if p is None or Path(p).exists()]

    fig, axes = plt.subplots(2, len(available), figsize=(3.3 * len(available), 6.4))
    axes = np.atleast_2d(axes)
    scores = []

    # scripts/ is not an importable package, so load by path like the rest of
    # the repo does. The plain import silently failed for the whole campaign and
    # was swallowed by the try/except around this figure, so the panel was
    # silently never produced.
    _tspec = importlib.util.spec_from_file_location(
        "_transfer_fig", ROOT / "scripts/benchmark_transfer.py"
    )
    _transfer = importlib.util.module_from_spec(_tspec)
    _tspec.loader.exec_module(_transfer)
    extract_queries = _transfer.extract_queries


    for col, (name, ckpt) in enumerate(available):
        config = abl.make_qformer_config(
            device, use_multitoken_cellline=True, use_qformer=True,
            use_qformer_for_ic50=True, use_ic50_attn_pool=True,
            num_cell_lines=max(dataset.num_celllines, 1000),
        )
        model = ModalitySlotQFormer(config).to(device)
        if ckpt is not None:
            model = abl.load_pretrained_weights(model, str(ckpt), device)

        features, targets = extract_queries(model, dataset, idx.tolist(), device)
        bins = np.searchsorted(np.percentile(targets, [20, 40, 60, 80]), targets)

        for row, (label, reducer) in enumerate(
            [("t-SNE", TSNE(n_components=2, perplexity=30, random_state=seed, init="pca")),
             ("MDS", MDS(n_components=2, random_state=seed, n_init=1, normalized_stress="auto"))]
        ):
            # MDS is O(n^2); subsample further so it stays tractable.
            take = slice(None) if label == "t-SNE" else slice(0, 1200)
            projected = reducer.fit_transform(features[take])
            ax = axes[row, col]
            ax.scatter(projected[:, 0], projected[:, 1], c=bins[take], cmap="viridis",
                       s=3, alpha=0.6, linewidths=0)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if row == 0:
                ax.set_title(name, fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)

        km = KMeans(n_clusters=5, random_state=seed, n_init=10).fit(features)
        scores.append({
            "stage": name,
            "silhouette_vs_kmeans": float(silhouette_score(features, km.labels_, sample_size=2000,
                                                           random_state=seed)),
            "silhouette_vs_ic50_quintile": float(silhouette_score(features, bins, sample_size=2000,
                                                                  random_state=seed)),
        })

    fig.suptitle("Representation structure by pretraining stage (colour = IC50 quintile)",
                 fontsize=10, x=0.02, ha="left")
    _save(fig, out / "clustering.png")

    frame = pd.DataFrame(scores)
    frame.to_csv(out.parent / "tables" / "clustering_silhouette.csv", index=False)
    logger.info("\n" + frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    parser.add_argument("--skip_clustering", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    figures = out_root / "figures"
    tables = out_root / "tables"

    summary_path = tables / "metrics_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else None
    baseline_path = tables / "mean_baselines_ncc.csv"
    baselines = pd.read_csv(baseline_path) if baseline_path.exists() else None

    if summary is not None:
        fig_ablation_ladder(summary, baselines, figures)
        fig_generalisation(summary, baselines, figures)
    else:
        logger.warning("no metrics_summary.csv yet; run aggregate_metrics.py first")

    try:
        frames = []
        for experiment in ("multiseed", "leakfree_strict_ncc", "leakfree_ncd",
                           "ncd_baselines"):
            try:
                frames.append(load_all_predictions(out_root, experiment=experiment))
            except FileNotFoundError:
                logger.warning(f"no prediction dumps for {experiment} yet")
        if not frames:
            raise FileNotFoundError("no prediction dumps at all")
        preds = pd.concat(frames, ignore_index=True)
        for regime in sorted(preds.regime.unique()):
            fig_scatter(preds, regime, figures)
        fig_scatter_combined(preds, figures, summary)  # supplementary S1
    except FileNotFoundError:
        logger.warning("no prediction dumps yet")

    if not args.skip_clustering:
        try:
            fig_clustering(figures)
        except Exception as exc:  # noqa: BLE001 - figures must not break the pipeline
            logger.warning(f"clustering figure failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
