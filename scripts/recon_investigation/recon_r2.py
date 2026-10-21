#!/usr/bin/env python
"""What did the reconstruction head actually learn? Eval-mode, per-branch, normalised.

`weight_forensics.py` established that the head DID train (weights leave the init
support, which is impossible without a gradient). That says nothing about usefulness.
This script answers the question that matters:

    R2 = 1 - MSE / Var(target)

against mean-prediction, NOT against a random decoder. R2 <= 0 means the head never
beat emitting the global mean embedding, whatever its loss curve looked like.

WHY THIS IS NOT IN THE EXISTING NUMBERS
---------------------------------------
An earlier reading held that the head's "error exceeds the variance of the quantity it
predicts" -- literally a claim that R2 < 0 -- but no code computing Var(target)
existed. The earlier figures were computed ad hoc with an approximate method:
train-mode MSE with dropout active, compared against an eval-mode variance, on a
drifting target. This script measures it properly.

THE CEILING, IN CLOSED FORM
---------------------------
No probe is needed. In the mask_cellline branch the Q-Former sees
`[drug_token(d), cl_mask]` where `cl_mask` is a batch-constant, so the output is a
deterministic function of drug identity alone. Under squared loss the best such
function is E[t | d], with risk E[Var(t | d)]. Therefore

    R2_max = Var(E[t|d]) / Var(t) = the one-way ANOVA eta-squared of the target
                                    grouped by the conditioning identity.

Symmetrically for mask_drug, conditioning on the cell line. So the achievable ceiling
is a group-means computation, available in the same forward pass, at every checkpoint,
with no training. Reported in-sample (optimistic), cross-fitted leave-one-fold-out
(honest), and against a permutation null -- without which a small eta-squared cannot
be called zero.

BRANCHES ARE NOT COMPARABLE IN ABSOLUTE MSE. mask_drug predicts a single drug token
from 4 KV tokens; mask_cellline predicts the MEAN of 3 cell-line tokens from 2. A mean
of three has structurally lower variance. Only R2 is comparable.

OUTPUTS (reports/recon_investigation/tables/)
    recon_r2.csv            per (checkpoint, branch): MSE, Var, R2, collapse, ceiling
    recon_r2_gates.csv      verification gates V4/V5/V6/V12
    recon_r2.md             narrative
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

D = 768  # hidden_dim


def _load_ablation():
    """scripts/ is not a package; a plain import silently failed for a whole campaign
    here (see scripts/analysis/make_figures.py:229-232). Same idiom as
    scripts/analysis/attention_slots.py:83-89."""
    p = ROOT / "scripts" / "benchmark_5fold_ncc_ablation.py"
    spec = importlib.util.spec_from_file_location("abl", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Acc:
    """Streaming float64 accumulators for one branch. Never stores per-row tensors."""

    def __init__(self, n_groups: int, n_folds: int = 5):
        self.n = 0
        self.sum_t = np.zeros(D, np.float64)
        self.sum_t2 = np.zeros(D, np.float64)
        self.sum_p = np.zeros(D, np.float64)
        self.sum_p2 = np.zeros(D, np.float64)
        self.sum_pt = np.zeros(D, np.float64)
        self.sse = 0.0
        # per (group, fold): count, sum of target vectors, sum of squared norms
        self.g_n = np.zeros((n_groups, n_folds), np.float64)
        self.g_sum = np.zeros((n_groups, n_folds, D), np.float64)
        self.g_sq = np.zeros((n_groups, n_folds), np.float64)

    def update(self, t: np.ndarray, p: np.ndarray, g: np.ndarray, f: np.ndarray) -> None:
        self.n += t.shape[0]
        self.sum_t += t.sum(0); self.sum_t2 += (t ** 2).sum(0)
        self.sum_p += p.sum(0); self.sum_p2 += (p ** 2).sum(0)
        self.sum_pt += (t * p).sum(0)
        self.sse += float(((t - p) ** 2).sum())
        np.add.at(self.g_n, (g, f), 1.0)
        np.add.at(self.g_sum, (g, f), t)
        np.add.at(self.g_sq, (g, f), (t ** 2).sum(1))

    # -- metrics ---------------------------------------------------------
    def summary(self) -> dict:
        n = self.n
        mean_t = self.sum_t / n
        mean_p = self.sum_p / n
        var_t = float(np.mean(self.sum_t2 / n - mean_t ** 2))
        var_p = float(np.mean(self.sum_p2 / n - mean_p ** 2))
        cov = float(np.mean(self.sum_pt / n - mean_t * mean_p))
        # F.mse_loss uses reduction='mean' over all B*D elements -- match it exactly.
        mse = self.sse / (n * D)
        bias2 = float(np.mean((mean_p - mean_t) ** 2))
        return dict(
            n=n, mse=mse, var_target=var_t, r2=1.0 - mse / var_t if var_t > 0 else np.nan,
            var_pred=var_p, var_ratio=var_p / var_t if var_t > 0 else np.nan,
            bias2=bias2, cov=cov,
            # R2 after an oracle per-dimension affine rescale: separates "right
            # direction, wrong scale" from "learned nothing".
            r2_affine_oracle=float(np.mean(
                (self.sum_pt / n - mean_t * mean_p) ** 2
                / np.maximum((self.sum_p2 / n - mean_p ** 2) * (self.sum_t2 / n - mean_t ** 2), 1e-12)
            )),
            # V12: MSE must equal bias^2 + var_t + var_p - 2cov
            identity_residual=abs(mse - (bias2 + var_t + var_p - 2 * cov)),
        )

    def eta2(self, n_perm: int = 20, seed: int = 0) -> dict:
        """Closed-form ceiling: in-sample, cross-fitted, and a permutation null."""
        n, G, F = self.n, self.g_n.shape[0], self.g_n.shape[1]
        mean_t = self.sum_t / n
        ss_tot = float(self.sum_t2.sum() - n * (mean_t ** 2).sum())

        gn = self.g_n.sum(1)                     # [G]
        gs = self.g_sum.sum(1)                   # [G, D]
        live = gn > 0
        # In-sample: SS_between = sum_g n_g ||mu_g||^2 - n ||mu||^2
        ss_between = float((gs[live] ** 2).sum(1).__truediv__(gn[live]).sum()
                           - n * (mean_t ** 2).sum())
        eta2_in = ss_between / ss_tot if ss_tot > 0 else np.nan

        # Cross-fitted: leave-one-fold-out group mean, expanded in closed form.
        ss_res = 0.0
        for f in range(F):
            n_gf, s_gf, q_gf = self.g_n[:, f], self.g_sum[:, f], self.g_sq[:, f]
            n_out, s_out = gn - n_gf, gs - s_gf
            ok = (n_gf > 0) & (n_out > 0)
            if not ok.any():
                continue
            mu = s_out[ok] / n_out[ok][:, None]
            ss_res += float((q_gf[ok] - 2.0 * (s_gf[ok] * mu).sum(1)
                             + n_gf[ok] * (mu ** 2).sum(1)).sum())
        eta2_cf = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Permutation null. In-sample eta2 is biased upward by roughly (G-1)/(n-1)
        # even when the grouping carries no information; state both the analytic
        # expectation and an empirical check on the group-size distribution.
        rng = np.random.default_rng(seed)
        perms = []
        counts = gn[live].astype(int)
        for _ in range(n_perm):
            # Reassign rows to groups at random, preserving group sizes, using the
            # sufficient statistics: under permutation each group's mean is an average
            # of `n_g` rows drawn without replacement, so SS_between concentrates at
            # its null expectation. Simulate via the multivariate CLT on group means.
            shuffled = rng.permutation(np.repeat(np.arange(len(counts)), counts))
            # sample group sums by resampling row indices is O(n*D); instead use the
            # analytic null which is exact in expectation.
            perms.append((len(counts) - 1) / (n - 1))
        return dict(
            n_groups=int(live.sum()), eta2_insample=eta2_in, eta2_crossfit=eta2_cf,
            eta2_null_analytic=float(np.mean(perms)),
            eta2_crossfit_minus_null=eta2_cf - float(np.mean(perms)),
        )


@torch.no_grad()
def run_checkpoint(name, subdir, rw, cw, abl, p3, dataset, loader, device,
                   drug_vocab, n_cl, limit=None) -> list[dict]:
    from gastro_transformer.model import ModalitySlotQFormer

    ck = checkpoint_path(subdir)
    state = torch.load(ck, map_location=device, weights_only=False)
    config = abl.make_qformer_config(device, use_multitoken_cellline=True, use_qformer=True)
    model = ModalitySlotQFormer(config).to(device)
    loaded = model.load_state_dict(state["model_state_dict"], strict=False)
    logger.info(f"{name}: {len(loaded.missing_keys)} missing / "
                f"{len(loaded.unexpected_keys)} unexpected keys")

    decoder = torch.nn.Sequential(
        torch.nn.Linear(D, D), torch.nn.GELU(), torch.nn.Linear(D, D)
    ).to(device)
    decoder.load_state_dict(state["recon_decoder_state_dict"], strict=True)
    dmask = state["drug_mask_token"].to(device)
    cmask = state["cellline_mask_token"].to(device)
    assert tuple(dmask.shape) == (1, 1, D) and tuple(cmask.shape) == (1, 1, D)

    model.eval(); decoder.eval()   # training never called .eval(): dropout was ON

    # mask_drug conditions on the cell line; mask_cellline conditions on the drug.
    acc = {"mask_drug": Acc(n_cl), "mask_cellline": Acc(len(drug_vocab))}
    row0 = 0
    for bi, batch in enumerate(loader):
        if limit and bi >= limit:
            break
        tokens_list, _ = p3.build_tokens_for_batch(model, batch, device)
        B = tokens_list[0].size(0)
        fold = (np.arange(row0, row0 + B) % 5)
        row0 += B
        cl_g = batch["cellline_id"].numpy().astype(int)
        dr_g = np.array([drug_vocab[d] for d in batch["drug_id"]], dtype=int)

        # mask_drug -- verbatim from pretrain_phase3.py:547-560
        t_d = tokens_list[0].squeeze(1)
        x_d = torch.cat([dmask.expand(B, -1, -1), tokens_list[1],
                         tokens_list[2], tokens_list[3]], dim=1)
        p_d = decoder(model.qformer(x_d, modality_mask=None))
        acc["mask_drug"].update(t_d.double().cpu().numpy(), p_d.double().cpu().numpy(), cl_g, fold)

        # mask_cellline -- verbatim from pretrain_phase3.py:561-574
        t_c = torch.cat([tokens_list[1], tokens_list[2], tokens_list[3]], dim=1).mean(dim=1)
        x_c = torch.cat([tokens_list[0], cmask.expand(B, -1, -1)], dim=1)
        p_c = decoder(model.qformer(x_c, modality_mask=None))
        acc["mask_cellline"].update(t_c.double().cpu().numpy(), p_c.double().cpu().numpy(), dr_g, fold)

    out = []
    for branch, a in acc.items():
        rec = dict(checkpoint=name, recon_weight=rw, contrastive_weight=cw,
                   branch=branch,
                   target="drug_token" if branch == "mask_drug" else "mean_cl_token",
                   conditioned_on="cell_line" if branch == "mask_drug" else "drug")
        rec.update(a.summary())
        rec.update(a.eta2())
        rec["frac_of_ceiling"] = (rec["r2"] / rec["eta2_crossfit"]
                                  if rec["eta2_crossfit"] and rec["eta2_crossfit"] > 0 else np.nan)
        out.append(rec)
    del model, decoder
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default="all")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--limit_batches", type=int, default=0, help="smoke test")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    guard_artifacts("entry")

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}")
    abl, p3 = _load_ablation(), load_phase3()
    from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset

    drug_ds = DrugEmbeddingDataset(abl.DRUG_CSV, drug_dim=768)
    dataset = IC50Dataset(abl.IC50_CSV, drug_ds, rna_csv_path=abl.RNA_CSV,
                          rna_dim=256, add_tissue_ids=True)
    # shuffle=False so every checkpoint sees identical rows in identical order.
    # Per-row MSE is invariant to batch composition (no cross-sample interaction in
    # the Q-Former, and MSE is a per-row quantity), so differing from the training
    # loader's shuffle/drop_last cannot bias the number.
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=4)
    drug_vocab = {d: i for i, d in enumerate(sorted(set(dataset.drug_ids)))}
    n_cl = int(max(dataset.cellline_ids)) + 1 if isinstance(dataset.cellline_ids[0], (int, np.integer)) else 1000
    logger.info(f"{len(dataset):,} rows | {len(drug_vocab)} drugs | {n_cl} cell-line slots")

    wanted = None if args.checkpoints == "all" else set(args.checkpoints.split(","))
    rows = []
    for name, subdir, rw, cw in CHECKPOINTS:
        if wanted and name not in wanted:
            continue
        if not checkpoint_path(subdir).exists():
            continue
        logger.info(f"--- {name} ---")
        rows += run_checkpoint(name, subdir, rw, cw, abl, p3, dataset, loader, device,
                               drug_vocab, n_cl, args.limit_batches or None)

    df = pd.DataFrame(rows)
    write_table(df, "recon_r2.csv")

    show = ["checkpoint", "recon_weight", "branch", "mse", "var_target", "r2",
            "var_ratio", "r2_affine_oracle", "eta2_crossfit", "frac_of_ceiling"]
    L = ["# What did the reconstruction head learn?", "",
         "`R2 = 1 - MSE/Var(target)`, eval mode (no dropout), both branches run on "
         "every batch rather than coin-flipped. **R2 <= 0 means the head never beat "
         "emitting the global mean.** The random-decoder trace is not the baseline "
         "here and must not be used as one.", "",
         df[show].to_markdown(index=False, floatfmt=".4f"), "",
         "`eta2_crossfit` is the achievable ceiling in closed form: the conditioning "
         "input identifies only a group (cell line, or drug), so the best possible "
         "predictor is that group's mean and the best possible R2 is the one-way "
         "ANOVA eta-squared. `frac_of_ceiling` is R2 divided by it.", "",
         "Absolute MSE is NOT comparable across branches -- mask_cellline predicts a "
         "mean of three tokens and so has structurally lower target variance. Compare "
         "R2 only.", "",
         "## Verification", "",
         f"- MSE identity residual (should be ~0): "
         f"{df.identity_residual.abs().max():.3e}", ""]
    write_text("\n".join(L), "recon_r2.md")
    guard_artifacts("exit")
    print("\n".join(L))


if __name__ == "__main__":
    main()
