# Architecture

## Overview

**Gastro-Transformer v2** is a multi-modal foundation model for gastric cancer drug response (IC50) prediction. The key innovation is the **Modality-Slot Q-Former** architecture, using typed token embeddings for clean multi-modal fusion.

## Key Differences from v4

1. **No Boolean Flag Sprawl**: Removed all architecture-controlling boolean flags (`ic50_use_qformer`, `ic50_use_tissue_bridge`, `ic50_use_cellline_rna`, `ic50_use_cross_attn`)
2. **Typed Modality Tokens**: Each modality (image, RNA, drug, cell-line) receives a type embedding that tells the Q-Former "what type of token this is"
3. **Clean IC50 Path**: IC50 head takes `[B, D]` fused output from Q-Former, NOT concatenated `[B, 3*D]` inputs. The fusion happens inside Q-Former via cross-attention.
4. **Differential Learning Rates**: During fine-tuning, Q-Former uses lower LR (0.1× base LR) to preserve pretrained knowledge
5. **Shared RNA Projector**: Cell-line RNA and patient RNA both go through the same `projectors['rna']` for shared embedding space

## Architecture Diagram

```
Pretraining:  Image + RNA → Projectors → Type Embeddings → Q-Former → Classification Heads

IC50 (ID-based):    Drug + CellLine(ID+RNA) → 2 typed tokens → Q-Former → IC50 Head
IC50 (Feature):     Drug + CellLine(RNA+cancer+tissue fused) → 2 tokens → Q-Former → IC50 Head
IC50 (MultiToken):  Drug + [Cancer, Tissue, RNA] → 4 typed tokens → Q-Former → IC50 Head  ← BEST
```

## Cell-Line Encoder Options

| Option | Config Flag | Tokens | Cold-Start? | R² (5-fold NCC, RNA-filtered) |
|--------|-------------|--------|:-----------:|-----|
| ID-based (v4) | `use_cellline_embeddings=True` | 1 (fused) | No | 0.740 |
| Feature-based | `use_feature_cellline_encoder=True` | 1 (fused) | Yes | 0.757 |
| **MultiToken** | **`use_multitoken_cellline=True`** | **3 (cancer+tissue+rna)** | **Yes** | **0.801** |

## Pretraining Pipeline

```
Stage 1 (CLRNA):   Image + RNA → Contrastive Learning → pretrained.pt
DR-A (IC50):    Drug + CellLine → IC50-Aware SupCon + Cross-Modal Recon → pretrained_phase3.pt  ← NEW BEST
Fine-tuning:       pretrained_phase3.pt → MSE IC50 regression → best model
```

## Code Architecture

```
gastro_transformer/
├── config.py              # Simplified config (no boolean flags)
├── model.py               # ModalitySlotQFormer (core)
├── losses.py              # Contrastive and task losses
├── data.py                # Data loading and preprocessing
├── utils.py               # Utility functions
├── train.py               # Differential LR training
├── visualization.py       # Visualization tools
├── model_with_ssgsea.py   # ssGSEA extension (5 KV tokens)
└── data_with_ssgsea.py   # ssGSEA dataset extension

scripts/
├── main.py                # Simplified CLI (pretrain, finetune, evaluate)
├── pretrain_phase3.py     # DR-A: IC50-aware pretraining (SupCon + reconstruction)
├── pretrain_with_ssgsea.py    # DR-A + ssGSEA (5 KV tokens, +2.3% R²)
├── benchmark_phase3.py    # DR-A vs CLRNA benchmark (3-fold CV)
├── benchmark_with_ssgsea.py    # 5-fold NCC: MultiToken vs MultiToken + ssGSEA
├── benchmark_5fold_ncc_ablation.py  # 5-fold NCC ablation (998 CL, all baselines)
└── benchmark_random_split.py  # Random split benchmark
```

## Key Invariants

1. Data format compatible with v4
2. NCC (No Common Cell Line) splitting with seed=42
3. Shared RNA projector for patient + cell-line RNA
4. IC50 head takes `[B, D]` NOT `[B, 3*D]`
5. Differential LR: Q-Former uses 0.1× LR during fine-tuning

## Configuration

Key config options:
- `qformer_finetune_lr_ratio: float = 0.2` — Q-Former LR multiplier during fine-tuning
- `use_multitoken_cellline: bool = False` — Decompose cell-line into 3 separate Q-Former tokens (best architecture)
- `use_feature_cellline_encoder: bool = False` — Feature-based cell encoder instead of ID embeddings
- `use_ic50_attn_pool: bool = False` — Attention pooling + gated residual fusion for IC50
- `freeze_qformer_in_finetune: bool = False` — Freeze Q-Former (ablation)
- `freeze_projectors_in_finetune: bool = False` — Freeze projectors (ablation)

## Data Requirements

Uses pre-computed embeddings in CSV/TSV format:
- `paired_image_csv`, `paired_rna_csv` — Paired patient data
- `unpaired_image_csv`, `unpaired_rna_csv` — Unpaired data
- `drug_embeddings_csv` — Drug embeddings (ChemBERT)
- `ic50_csv` — Drug-cell line IC50 pairs
- `cellline_rna_csv` — Cell-line RNA embeddings
- `ssgsea_tsv` — ssGSEA pathway enrichment TSV (768d, KEGG/Reactome/Hallmarks)
