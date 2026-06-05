# MoSQ — Modality-Slot Q-Former

**Multi-modal foundation model for gastric cancer drug response (IC50) prediction.**

MoSQ introduces the **Modality-Slot Q-Former** architecture: each biological modality (drug, cell-line RNA, cancer subtype, tissue type, ssGSEA pathways) occupies a typed token slot that feeds into a shared Q-Former cross-attention pool. The model is pretrained with IC50-aware supervised contrastive learning (DR-A) and fine-tuned with simple MSE regression.

## Results

| Benchmark | Model | R² |
|-----------|-------|----|
| **5-fold NCC + ssGSEA (592 CL, RNA-filtered)** | **DR-A + ssGSEA** | **0.852** |
| Random split (592 CL, RNA-filtered) | DR-A | 0.838 |
| 5-fold NCC + ssGSEA (998 CL) | DR-A + ssGSEA | 0.814 |
| Random split (998 CL) | DR-A | 0.805 |
| 5-fold NCC (998 CL) | DR-A | 0.791 |
| NCD Leak-Free | DR-A | 0.208 |

Evaluation uses **NCC (No Common Cell-Line)** cross-validation with `seed=42` — cell lines in the test set never appear in training.
See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for full ablation tables and literature comparisons.

---

## Architecture

```
Drug embedding (768d)           ─┐
Cell-line RNA embedding (256d)  ─┤  Modality Projectors
Cancer subtype ID               ─┤  → typed token slots
Tissue type ID                  ─┤  → Q-Former (48 queries, 8 layers, 12 heads)
ssGSEA pathway scores (768d)    ─┘  → IC50 Head → LN(IC50)
```

- **Pretraining (DR-A):** Drug–CellLine SupCon loss on IC50 quintile bins + cross-modal reconstruction
- **Fine-tuning:** MSE regression, differential LR (Q-Former uses 0.1× base LR)

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full details.

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

## Data Preparation

The model expects pre-computed embeddings in CSV format. Place files in `data/`:

| File | Description | Shape |
|------|-------------|-------|
| `drug_embeddings.csv` | ChemBERTa drug embeddings | `[n_drugs, 768]` |
| `ic50_data.csv` | Drug–cell-line IC50 pairs | `DRUG_ID, CELL_LINE_NAME, LN_IC50, ...` |
| `ccle_rna_for_ic50.csv` | BulkRNABERT cell-line RNA | `[n_celllines, 256]` |
| `ssgsea_scores.tsv` | ssGSEA pathway enrichment | `[n_celllines, 768]` |

GDSC IC50 data: https://www.cancerrxgene.org/
CCLE RNA: https://depmap.org/portal/
Embeddings (BulkRNABERT, ChemBERTa) are computed externally and stored as CSVs.

---

## Reproducing the Best Result (R² = 0.852)

### Step 1 — DR-A Pretraining

```bash
python scripts/pretrain_with_ssgsea.py \
  --drug_embeddings_csv data/drug_embeddings.csv \
  --ic50_csv            data/ic50_data.csv \
  --cellline_rna_csv    data/ccle_rna_for_ic50.csv \
  --ssgsea_tsv          data/ssgsea_scores.tsv \
  --pretrain_epochs 100 \
  --device cuda:0 \
  --checkpoint_dir checkpoints/phase3_ssgsea
```

### Step 2 — Fine-tune and Evaluate (5-fold NCC)

```bash
python scripts/benchmark_ncc_rnafiltered_ssgsea.py \
  --checkpoint      checkpoints/phase3_ssgsea/pretrained_phase3_ssgsea.pt \
  --drug_embeddings_csv data/drug_embeddings.csv \
  --ic50_csv            data/ic50_data.csv \
  --cellline_rna_csv    data/ccle_rna_for_ic50.csv \
  --ssgsea_tsv          data/ssgsea_scores.tsv \
  --finetune_epochs 10 \
  --device cuda:0
```

Expected output: **R² ≈ 0.852 ± 0.004** across 5 NCC folds.

---

## Quick Demo (no real data needed)

```bash
python scripts/main.py --mode demo
```

Runs a full pretrain + fine-tune cycle on synthetic data using CPU. Useful for verifying the installation.

---

## Reproducing Other Benchmarks

| Experiment | Script |
|------------|--------|
| Full 5-fold NCC ablation (998 CL) | `scripts/benchmark_5fold_ncc_ablation.py` |
| Random split RNA-filtered (592 CL) | `scripts/benchmark_random_split_rnafiltered.py` |
| ssGSEA vs RNA-BERT comparison | `scripts/benchmark_with_ssgsea.py` |
| Literature model comparison | `scripts/benchmark_literature_models.py` |
| Sample efficiency curves | `scripts/sample_efficiency.py` |

All benchmark scripts accept `--checkpoint`, `--drug_embeddings_csv`, `--ic50_csv`, `--cellline_rna_csv`, and `--device` arguments.

---

## Fine-tuning from a Pretrained Checkpoint

```bash
python scripts/main.py --mode finetune \
  --checkpoint checkpoints/phase3_ssgsea/pretrained_phase3_ssgsea.pt \
  --drug_embeddings_csv data/drug_embeddings.csv \
  --ic50_csv            data/ic50_data.csv \
  --cellline_rna_csv    data/ccle_rna_for_ic50.csv \
  --finetune_epochs 10 \
  --qformer_finetune_lr_ratio 0.2 \
  --device cuda:0
```

> **Note:** Do not enable `--use_huber_loss`, `--use_ema`, or `--use_rdrop` — these v3 regularization tricks catastrophically fail at this data scale (R² → near-zero).

---

## Key Design Invariants

1. **NCC splitting** with `seed=42` — no cell line appears in both train and test
2. **Shared RNA projector** — patient RNA and cell-line RNA use the same projection weights
3. **IC50 head input is `[B, D]`**, not `[B, 3×D]` — fusion happens inside Q-Former
4. **Differential LR** — Q-Former uses `0.1×` base LR during fine-tuning
5. **RNA-filtering** — remove zero-filled cell lines (998 → 592) before evaluation

---

## Tests

```bash
pytest tests/
```

---

## Repository Structure

```
MoSQ/
├── gastro_transformer/       # Core Python package
│   ├── model.py              # ModalitySlotQFormer architecture
│   ├── model_with_ssgsea.py  # + ssGSEA pathway tokens (best model)
│   ├── data.py               # IC50Dataset, NCC splitting
│   ├── train.py              # GastroTransformerTrainer (differential LR)
│   ├── losses.py             # Contrastive losses (SupCon, cross-modal)
│   ├── config.py             # GastroTransformerConfig
│   └── utils.py              # Checkpoint loading, tissue maps
├── scripts/
│   ├── main.py               # CLI entry point (demo/pretrain/finetune/evaluate)
│   ├── pretrain_with_ssgsea.py   # DR-A + ssGSEA pretraining
│   ├── pretrain_phase3.py        # DR-A pretraining (RNA-only)
│   ├── benchmark_ncc_rnafiltered_ssgsea.py  # Best benchmark
│   └── ...                   # Other benchmark and analysis scripts
├── tests/
│   └── test_model.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PERFORMANCE.md
│   ├── KEY_FINDINGS.md
│   └── PROTOCOLS.md
├── data/                     # Place your CSV/TSV embeddings here (not tracked)
├── checkpoints/              # Saved checkpoints (not tracked)
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
