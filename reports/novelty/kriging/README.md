# Panel-completion kriging — exploratory, NOT pre-registered (2026-10-01)

Copied verbatim from the original scratch output directory (byte-identical, verified with
`cmp`; checksums in `SHA256SUMS`). CPU only. All runs read the existing prediction dumps in
`reports/revision/multiseed/preds/` (`ncc__{qformer_randinit,multitoken_dra}__seed{0-9}`) and
nothing else, so no model was retrained.

## Method

For each held-out line, a fixed random half of its drugs is the **evaluation half**.
k drugs from the other half are **revealed** (k ∈ {5, 10, 20, all}; `all` = the entire
non-evaluation half, ~88). The draws are identical to `scripts/novelty/panel_completion_gate.py`
(rng `10000*seed + draw`; 10 seeds × 3 draws), and the offset arms reproduce that script's
summary exactly.

- **offset**: base prediction + mean of (y − base) over the line's revealed drugs.
- **kriging (lowrank)**: base prediction + Σ_ER (Σ_RR + a·s²·I)⁻¹ r_R. Here Σ is the drug×drug
  covariance from the fold's **training lines only** (soft-impute SVD, rank r, plus noise s²).
  r and a are chosen per (seed, fold, k) by simulated reveals on an inner 80/20 split of the
  training lines. The evaluation half is never used for selection.
- **Σ "shared"**: built from drug-centred training-line residuals (the same Σ for every base).
  **Σ "own"**: built from that base model's residuals on training lines, taken from the other
  folds' test dumps. This carries a mild indirect exposure for random init, because those fold
  models were trained on this fold's lines. Transductive DR-A is leaky in every arm.
- **placebo**: the selected Σ with the drug labels permuted (a fixed permutation per fold).
- **Ranking**: per-line Spearman ρ and top-10 recall on the evaluation half, over lines with
  ≥ 20 drugs. The definitions match `scripts/analysis/per_cellline_ranking.py`.

## Files

| Path | Produced by | Contents |
|---|---|---|
| `rows/mf_gate_rows_s{0-9}.csv` | `code_as_run/mf_gate.py <seed>` | bases drugmean / qformer (random init) × {offset, lowrank(sel), lowrank(placebo), full (r, a) grid}; R², dcR² by stratum (all / rna / no_rna) |
| `rows/mf_gate_selection_s{0-9}.csv` | same | selected (rank, a) per fold and k; inner-split SSE ratio vs offset |
| `rows/dra_rows_s{0-9}.csv` | `code_as_run/mf_gate_dra.py <seed>` | as above, plus base `dra_transductive` (shared Σ) |
| `rows/dra_selection_s{0-9}.csv` | same | selection (identical procedure) |
| `rows/fu_rows_s{0-9}.csv` | `code_as_run/krig_followup.py <seed>` | bases drugmean / randinit / dra × {offset, krig_shared, krig_own}; R², dcR², per-line ρ, top-10 recall by stratum |
| `logs/{log,dlog,flog}_s{0-9}.txt` | stdout of the three scripts | completion markers only |

Rows are per (seed, draw). Aggregate by averaging the draws within each seed, then take the
mean ± across-seed SD. Paired CIs are taken over the 10 seeds.

## Headline numbers (R², all lines, mean of 10 seeds)

| | k=5 | k=10 | k=20 | all |
|---|---|---|---|---|
| random init + offset | 0.7712 | 0.7906 | 0.8003 | 0.8079 |
| kriging alone (drugmean base) | 0.7930 | 0.8165 | 0.8401 | 0.8838 |
| random init + kriging (own Σ) | 0.8054 | 0.8230 | 0.8422 | 0.8798 |
| transductive DR-A + kriging (own Σ) | 0.8123 | 0.8254 | 0.8413 | 0.8759 |

## Caveats

- **These runs are exploratory.** They were run before any pre-registration, so cite them as a
  baseline or diagnostic, not as a confirmatory test.
- **The scripts in `code_as_run/` are the as-run snapshots, kept for provenance.** They
  hardcode a scratch directory for the import path and the outputs. The runnable,
  repo-relative versions are in `scripts/novelty/kriging_exploratory/` (identical logic;
  outputs go to `reports/novelty/kriging/rows/`).
- `mf_gate.py` is in the state that produced `mf_gate_rows_*` (rank grid up to 100, placebo
  arm). An earlier seed-0 run, made before those edits, was discarded.
