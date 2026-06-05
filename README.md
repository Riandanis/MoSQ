# MoSQ — Modality-Slot Q-Former

**Multi-modal foundation model for gastric cancer drug response (IC50) prediction.**

MoSQ introduces the **Modality-Slot Q-Former** architecture: each biological modality (drug, cell-line RNA, cancer subtype, tissue type) occupies a typed token slot that feeds into a shared Q-Former cross-attention pool. The model is pretrained with IC50-aware supervised contrastive learning (**DR-A**) and fine-tuned with simple MSE regression.

## Results

| Benchmark | Model | R² |
|-----------|-------|----|
| **Random split (592 CL, RNA-filtered)** | **DR-A** | **0.838** |
| 5-fold NCC (592 CL, RNA-filtered) | DR-A | 0.801 |
| Random split (998 CL) | DR-A | 0.805 |
| 5-fold NCC (998 CL) | DR-A | 0.791 |
| CLRNA baseline (998 CL) | Q-Former + CLRNA | 0.759 |
| NCD Leak-Free | DR-A | 0.208 |

Evaluation uses **NCC (No Common Cell-Line)** cross-validation with `seed=42` — cell lines in the test fold never appear during training.
See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for full ablation tables and literature comparisons.

---

## Architecture

```
Drug embedding (768d)           ─┐
Cell-line RNA embedding (256d)  ─┤  Modality Projectors
Cancer subtype ID               ─┤  → typed token slots
Tissue type ID                  ─┘  → Q-Former (48 queries, 8 layers, 12 heads)
                                     → IC50 Head → LN(IC50)
```

**Two pretraining strategies included:**
- **CLRNA** — Stage 1: image + RNA contrastive learning (biological foundation)
- **DR-A** — IC50-aware: Drug–CellLine SupCon on IC50 quintile bins + cross-modal reconstruction (+3.2% R² over CLRNA)

Fine-tuning uses MSE regression with differential LR (Q-Former at 0.1× base LR).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full details.

---

## Pretrained Checkpoints

| Checkpoint | Description | R² (5-fold NCC, 998 CL) |
|------------|-------------|--------------------------|
| `saved_checkpoints/pretrained_dra.pt` | DR-A pretrained — **recommended** | 0.791 |
| `saved_checkpoints/pretrained_clrna.pt` | CLRNA baseline | 0.759 |

The `saved_checkpoints/` directory is tracked in the repo (the `.pt` weight files are too large for git and are gitignored). Download the weights and place them there:

```bash
# Download from:
# https://drive.google.com/drive/folders/129FU49n569OiQjhV9WJC1mhqkOou-lTa?usp=sharing
# then place at:
saved_checkpoints/pretrained_dra.pt
saved_checkpoints/pretrained_clrna.pt
```

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

The three preprocessed embedding files required for experiment reproduction are **included in this repository** under `data/`:

| File | Description | Shape |
|------|-------------|-------|
| `data/drug_embeddings.csv` | ChemBERTa-2 drug embeddings | `[n_drugs, 768]` |
| `data/ic50_data.csv` | GDSC drug–cell-line IC50 pairs (LN_IC50) | `DRUG_ID, CELL_LINE_NAME, LN_IC50, ...` |
| `data/ccle_rna_for_ic50.csv` | BulkRNABERT cell-line RNA embeddings | `[n_celllines, 256]` |

Raw source data (not included):
- GDSC IC50 measurements: https://www.cancerrxgene.org/
- CCLE RNA-seq: https://depmap.org/portal/
- BulkRNABERT: pre-trained RNA language model used to produce the 256-d cell-line embeddings
- ChemBERTa-2: pre-trained chemical language model used to produce the 768-d drug embeddings

---

## Quick Demo (no real data needed)

```bash
python scripts/main.py --mode demo
```

Runs a full pretrain + fine-tune cycle on synthetic data using CPU. Useful for verifying the installation.

---

## Reproducing the Best Result (R² = 0.801, 5-fold NCC)

### Step 1 — DR-A Pretraining  *(or skip — use the included checkpoint)*

```bash
python scripts/pretrain_phase3.py \
  --drug_embeddings_csv data/drug_embeddings.csv \
  --ic50_csv            data/ic50_data.csv \
  --cellline_rna_csv    data/ccle_rna_for_ic50.csv \
  --pretrain_epochs 100 \
  --device cuda:0 \
  --checkpoint_dir saved_checkpoints/phase3
```

### Step 2 — Fine-tune and Evaluate (5-fold NCC)

```bash
python scripts/benchmark_5fold_ncc_ablation.py \
  --checkpoint          saved_checkpoints/pretrained_dra.pt \
  --drug_embeddings_csv data/drug_embeddings.csv \
  --ic50_csv            data/ic50_data.csv \
  --cellline_rna_csv    data/ccle_rna_for_ic50.csv \
  --finetune_epochs 10 \
  --device cuda:0
```

Expected output: **R² ≈ 0.791 ± 0.012** (5-fold NCC, 998 CL) or **R² ≈ 0.801 ± 0.003** (RNA-filtered 592 CL).

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

> **Important:** Do not enable `--use_huber_loss`, `--use_ema`, or `--use_rdrop` — these regularization tricks catastrophically fail at this data scale (R² drops to near-zero).

---

## Other Benchmarks

| Experiment | Script |
|------------|--------|
| Full 5-fold NCC ablation (all baselines, 998 CL) | `scripts/benchmark_5fold_ncc_ablation.py` |
| Random split RNA-filtered (592 CL) | `scripts/benchmark_random_split_rnafiltered.py` |
| Literature model comparison | `scripts/benchmark_literature_models.py` |
| Sample efficiency curves | `scripts/sample_efficiency.py` |

---

## Key Design Invariants

1. **NCC splitting** with `seed=42` — no cell line appears in both train and test
2. **Shared RNA projector** — patient RNA and cell-line RNA use the same projection weights
3. **IC50 head input is `[B, D]`**, not `[B, 3×D]` — fusion happens inside Q-Former via cross-attention
4. **Differential LR** — Q-Former uses `0.1×` base LR during fine-tuning to preserve pretrained representations
5. **RNA-filtering** — removing zero-filled cell lines (998 → 592) before evaluation eliminates a zero-fill confound and improves R² by ~1%

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
│   ├── data.py                  # IC50Dataset, NCC splitting
│   ├── train.py                 # GastroTransformerTrainer (differential LR)
│   ├── losses.py                # Contrastive losses (SupCon, cross-modal)
│   ├── config.py                # GastroTransformerConfig
│   └── utils.py                 # Checkpoint loading, tissue type maps
├── scripts/
│   ├── main.py                  # CLI entry point (demo/pretrain/finetune/evaluate)
│   ├── pretrain_phase3.py       # DR-A pretraining (IC50-aware SupCon)
│   ├── benchmark_5fold_ncc_ablation.py   # Full ablation table
│   ├── benchmark_random_split_rnafiltered.py
│   ├── benchmark_literature_models.py
│   ├── sample_efficiency.py
│   └── inference.py
├── tests/
│   └── test_model.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PERFORMANCE.md
│   ├── KEY_FINDINGS.md
│   └── PROTOCOLS.md
├── saved_checkpoints/           # Place .pt files here after download (gitignored)
│   ├── pretrained_dra.pt        # DR-A (recommended)  R²=0.791 NCC
│   └── pretrained_clrna.pt      # CLRNA baseline       R²=0.759 NCC
├── data/                        # Place your CSV embeddings here (not tracked)
├── requirements.txt
└── setup.py
```

---

## Citation

If you use this code, please cite:

```bibtex
@software{mosq2026,
  title   = {MoSQ: Modality-Slot Q-Former for Gastric Cancer Drug Response Prediction},
  author  = {Riandanis},
  year    = {2026},
  url     = {https://github.com/Riandanis/MoSQ}
}
```
