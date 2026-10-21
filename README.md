# MoSQ — Modality-Slot Q-Former

**Drug Response-Aware (DR-A) pretraining with a Modality-Slot Q-Former for cell-line IC50 prediction, evaluated under transductive and inductive cell-line-blind protocols.**

MoSQ gives each input modality (drug, cell-line RNA, cancer subtype, tissue type) its own typed token slot, fused by a Q-Former through cross-attention. The model is pretrained with IC50-aware supervised contrastive learning (**DR-A**) and fine-tuned with MSE regression.

This repository accompanies the revised paper *"Drug Response-Aware Pretraining with a Modality-Slot Q-Former: Revisiting Pretraining Gains in Cell-Line-Blind IC50 Prediction"*. It contains the code for every experiment in the paper and the aggregated result tables behind every reported number.

## Results

All numbers below come from 5-fold cross-validation on 998 GDSC cell lines (173,942 drug–cell-line pairs). **±** is the across-seed SD, and *n* is the number of seeds.

**Transductive vs inductive.** Under *transductive* NCC (no common cell line), one DR-A checkpoint is pretrained on all cell lines. Pretraining therefore sees the test lines' IC50 labels in binned form. Under *inductive* NCC, pretraining is repeated in every fold on that fold's training cell lines only.

| Model / protocol | R² | n |
|---|---|---|
| Per-drug mean (no cell-line information) | 0.7257 | 1 |
| XGBoost | 0.7572 ± 0.0015 | 10 |
| MoSQ, no pretraining | 0.7593 ± 0.0011 | 10 |
| Per-drug ridge on cell-line features | 0.7677 ± 0.0006 | 10 |
| **MoSQ (DR-A), transductive NCC** | **0.7867 ± 0.0028** | 10 |
| MoSQ (DR-A), size-matched control (test lines kept, 799-line corpus) | 0.7733 ± 0.0013 | 3 |
| **MoSQ (DR-A), inductive NCC** | **0.7396 ± 0.0026** | 3 |
| MoSQ (DR-A), random split | 0.8033 ± 0.0013 | 5 |
| MoSQ (DR-A), leak-free novel-drug (NCD) | 0.1457 ± 0.0050 | 3 |

What the paper concludes:
- **DR-A's gain depends on label exposure.**
  - Transductive DR-A improves on the identical unpretrained architecture by +0.0274.
  - Inductive DR-A lands 0.0205 *below* it.
  - The transductive/inductive gap is 0.0481 ± 0.0035 on paired seeds. Of that, 0.0337 is label exposure and 0.0144 is corpus size.
- **Frozen DR-A features help a downstream XGBoost only when the encoder has seen the test cell lines.** XGBoost scores 0.8074 on those features against 0.7585 on raw features. With leak-free per-fold encoders the features score 0.7463, below raw.
- **Architecture contributes little on shared features.** Five architectures fall within about 0.009 R² of each other, and a per-drug ridge baseline beats every deep model without exposure.

The protocols and controls are described in the paper; the tables behind each number are listed under [Result tables](#result-tables).

---

## Architecture

```
Drug embedding (768d)           ─┐
Cell-line RNA embedding (256d)  ─┤  Modality Projectors
Cancer subtype ID               ─┤  → typed token slots
Tissue type ID                  ─┘  → Q-Former (32 queries, 6 layers, 12 heads, d_k = 64)
                                     → attention pooling → IC50 Head → LN(IC50)
```

Each Q-Former layer is pre-norm: self-attention over the queries, then cross-attention from the queries to the modality tokens, then a feed-forward block (768 → 3072 → 768, GELU). Every sublayer has a residual connection. Attention is standard multi-head attention (`torch.nn.MultiheadAttention`), scaled per head by 1/√d_k with d_k = D/H = 64.

**Pretraining stages included:**
- **CLRNA**: image + RNA contrastive learning (Phase 1). It was kept as an initialisation, but its measured contribution is negligible.
- **DR-A**: drug–cell-line supervised contrastive learning on IC50 quintile bins (global, per-drug rank, per-drug z-score, or random), plus cross-modal reconstruction.

Fine-tuning uses MSE regression with a differential learning rate for the Q-Former. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

---

## Pretrained Checkpoints

| Checkpoint | Description | NCC R² (998 CL, transductive) |
|------------|-------------|-------------------------------|
| `saved_checkpoints/pretrained_dra.pt` | DR-A, pretrained on all 998 cell lines | 0.7867 ± 0.0028 |
| `saved_checkpoints/pretrained_clrna.pt` | CLRNA (Phase 1) | 0.7606 ± 0.0011 |

`pretrained_dra.pt` was pretrained on every cell line. Evaluating it with cell-line-blind folds on the same panel is therefore the **transductive** setting. For held-out cell lines, use the inductive pipeline (`scripts/run_cv_leakfree.py`), which pretrains per fold.

The `.pt` weight files are too large for git and are gitignored. Download them and place them in `saved_checkpoints/`:

```bash
# Download from:
# https://drive.google.com/drive/folders/129FU49n569OiQjhV9WJC1mhqkOou-lTa?usp=sharing
# then place at:
saved_checkpoints/pretrained_dra.pt
saved_checkpoints/pretrained_clrna.pt
```

`GastroTransformerConfig()` defaults match the shipped checkpoints: 32 query tokens, 6 layers and 12 heads. Loading uses the configuration stored inside the checkpoint.

---

## Installation

```bash
git clone https://github.com/Riandanis/MoSQ.git
cd MoSQ

# Install PyTorch for your CUDA version first:
# https://pytorch.org/get-started/locally/

pip install -e .
```

---

## Data

**Included** under `data/`:

| File | Description |
|------|-------------|
| `data/ic50_data.csv` | GDSC drug–cell-line IC50 pairs (LN_IC50), GDSC1 lineage, release 7.0 (v17.3) |
| `data/drug_embeddings.csv` | ChemBERTa-2 drug embeddings, `[n_drugs, 768]` |
| `data/ccle_rna_for_ic50.csv` | BulkRNABERT cell-line RNA embeddings, `[n_celllines, 256]` (592 of 998 lines) |
| `data/drug_smiles.csv` | SMILES for every drug, used by the alternative-encoder sweep |

<a id="external-data"></a>**External data (not redistributed).** Download these into the listed paths. Only the analyses named alongside each one need it.

| Path | Source | Used by |
|------|--------|---------|
| `data/external/v17.3_fitted_dose_response.xlsx` | GDSC release 7.0 (v17.3) fitted dose response, https://www.cancerrxgene.org/ | censoring strata, EVAPA by stratum, dataset provenance |
| `data/external/gdsc/screened_compounds_rel_8.5.csv` | https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/ | mechanism-of-action downstream case |
| `data/external/depmap/{Model.csv, OmicsExpressionProteinCodingGenesTPMLogp1.csv}` | DepMap 24Q4 Public, https://doi.org/10.25452/figshare.plus.27993248 | full-transcriptome ridge control |
| `data/comparators/` | the DeepCDR (https://github.com/kimmo1019/DeepCDR) and DrugCell (https://github.com/idekerlab/DrugCell) repositories | faithful comparator pipelines; `scripts/build_matched_inputs.py` builds `data/matched/` from them |

---

## Quick Demo (no real data needed)

```bash
python scripts/main.py --mode demo
```

Runs a full pretrain + fine-tune cycle on synthetic data on CPU. Useful for checking the installation.

---

## Reproducing the paper

Every experiment runs through one cross-validation harness. All arms of a seed share identical folds, so comparisons are fold- and seed-paired. Per-sample predictions are written to `reports/revision/<experiment>/preds/` and per-run metrics to JSON.

```bash
# One arm, one seed, transductive NCC (fine-tunes the shipped DR-A checkpoint)
python scripts/run_cv.py --regime ncc --arm multitoken_dra --seed 0 --device cuda:0

# The same architecture without pretraining, and the comparators
python scripts/run_cv.py --regime ncc --arm qformer_randinit --seed 0
python scripts/run_cv.py --regime ncc --arm xgboost --seed 0

# Inductive NCC: DR-A re-pretrained in every fold on that fold's training lines only
python scripts/run_cv_leakfree.py --regime strict_ncc --seed 0 --keep_checkpoints

# Size-matched leaky control (same corpus size, test lines restored)
python scripts/run_cv_leakfree.py --regime strict_ncc --seed 0 --pretrain_exclude size_matched
```

- **Novel-drug (NCD) folds** do not vary with `--seed`. Use `--fold_seed` to resample the compound partition.
- **The full experiment matrix** (all stages, seeds and arms) is defined in `scripts/run_matrix.py` and driven by `scripts/run_campaign.py`:

```bash
python scripts/run_campaign.py                  # all stages (multi-day on 2 GPUs)
python scripts/run_campaign.py --only analysis  # rebuild every table and figure from the prediction dumps
```

**Analyses** (CPU, from the prediction dumps) are in `scripts/analysis/`: metrics, significance, per-cell-line ranking, mean baselines, EVAPA, censoring strata, attention, parameter counts and figures.

**Experiments added during revision** are in `scripts/novelty/`. Several of them were pre-registered: the baseline sweep, the inductive bin-mode and auxiliary-loss tests, and the downstream mechanism-of-action case. Others are labelled exploratory or post hoc in their docstrings, such as kriging panel completion and the ridge controls.

**Number registry.** `scripts/numbers/build_numbers.py` recomputes every manuscript-facing value from `reports/` and writes `reports/numbers/numbers.{json,tex}`, one entry per value with its source file and computation. It needs the prediction dumps.

### Legacy single-seed scripts

`scripts/benchmark_5fold_ncc_ablation.py`, `benchmark_random_split_rnafiltered.py`, `benchmark_literature_models.py` and `sample_efficiency.py` are the submitted-version pipelines, kept for continuity. The revised results use `run_cv.py` / `run_cv_leakfree.py` above.

---

<a id="result-tables"></a>
## Result tables

| Path | Contents |
|------|----------|
| `reports/revision/tables/` | metrics per run and summarised (`metrics_full.csv`, `metrics_summary.csv`), significance, per-cell-line ranking, RNA subsets, EVAPA, censoring strata, attention, parameter counts |
| `reports/revision/figures/` | figures generated from the prediction dumps |
| `reports/final_experiments/tables/` | frozen-feature transfer, including the leak-free arm |
| `reports/novelty/tables/`, `reports/novelty/sweep/tables/` | revision experiments: inductive tests, baseline sweep, ridge controls, panel completion |
| `reports/novelty/kriging/` | exploratory panel-completion kriging (rows, method README) |
| `reports/novelty/downstream/` | mechanism-of-action downstream case |
| `reports/recon_investigation/tables/` | reconstruction-objective diagnostics |
| `reports/numbers/numbers.{json,tex}` | the number registry |

Per-sample prediction dumps (about 12 GB) and per-fold checkpoints are not in git.

---

## Fine-tuning from a Pretrained Checkpoint (CLI)

```bash
python scripts/main.py --mode finetune \
  --checkpoint          saved_checkpoints/pretrained_dra.pt \
  --drug_embeddings_csv data/drug_embeddings.csv \
  --ic50_csv            data/ic50_data.csv \
  --cellline_rna_csv    data/ccle_rna_for_ic50.csv \
  --finetune_epochs 10 \
  --qformer_finetune_lr_ratio 0.2 \
  --use_multitoken_cellline \
  --device cuda:0
```

> **Important:** Do not enable `--use_huber_loss`, `--use_ema`, or `--use_rdrop`. These regularization tricks fail badly at this data scale (R² drops to near zero).

---

## Key Design Invariants

1. **Cell-line-blind splitting.** No cell line appears in both train and test (`gastro_transformer/splits.py`). `ncc` and `strict_ncc` partitions are identical. They differ only in what pretraining may see.
2. **Shared RNA projector.** Patient RNA and cell-line RNA use the same projection weights.
3. **The IC50 head input is `[B, D]`**, not `[B, 3×D]`. Fusion happens inside the Q-Former via cross-attention.
4. **Differential LR.** The Q-Former uses a reduced learning rate during fine-tuning to preserve pretrained representations.
5. **Missing RNA.** For the 406 cell lines without RNA-seq, MoSQ uses a learned missing-modality token, while feature-based baselines receive a zero block. Comparisons restricted to the 592 RNA-complete lines control for this asymmetry.

---

## Tests

```bash
pytest tests/
```

---

## Repository Structure

```
MoSQ/
├── gastro_transformer/          # Core Python package
│   ├── model.py                 # ModalitySlotQFormer architecture
│   ├── config.py                # GastroTransformerConfig (defaults = shipped checkpoint)
│   ├── data.py                  # IC50Dataset
│   ├── splits.py                # canonical fold construction (random / ncc / strict_ncc / ncd / scaffold)
│   ├── metrics.py               # RMSE, MAE, MALE, EVAPA, ranking metrics
│   ├── predlog.py               # per-sample prediction dumps
│   ├── seeding.py               # deterministic seeding
│   ├── comparators/             # faithful DeepCDR / DrugCell / tCNNS ports and their data loaders
│   ├── train.py, losses.py, utils.py
├── scripts/
│   ├── run_cv.py                # cross-validation harness (all arms)
│   ├── run_cv_leakfree.py       # inductive / size-matched per-fold pretraining
│   ├── run_matrix.py            # experiment matrix (presets)
│   ├── run_campaign.py          # stage driver
│   ├── pretrain_phase3.py       # DR-A pretraining
│   ├── benchmark_transfer.py    # frozen-feature transfer
│   ├── benchmark_comparators.py # faithful published pipelines
│   ├── build_drug_encoders.py   # alternative chemical encoders (NCD sweep)
│   ├── build_matched_inputs.py  # matched cell-line inputs for comparators
│   ├── analysis/                # tables and figures from prediction dumps
│   ├── novelty/                 # revision experiments (pre-registered / exploratory)
│   ├── recon_investigation/     # reconstruction-objective diagnostics
│   ├── numbers/                 # number registry
│   └── main.py, inference.py    # CLI (demo / pretrain / finetune / evaluate)
├── tests/
├── docs/
├── reports/                     # aggregated result tables (see above)
├── saved_checkpoints/           # place .pt files here after download (gitignored)
├── data/
├── requirements.txt
└── setup.py
```

---

## Citation

If you use this code, please cite:

```bibtex
@software{mosq2026,
  title   = {MoSQ: Drug Response-Aware Pretraining with a Modality-Slot Q-Former},
  author  = {Riandanis},
  year    = {2026},
  url     = {https://github.com/Riandanis/MoSQ}
}
```
