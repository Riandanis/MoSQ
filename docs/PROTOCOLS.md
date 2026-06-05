# Protocols

## Security & Data Handling

### Endoscopy Image Processing

**CRITICAL — Endoscopy images must remain locally contained and secured at all times.**

1. **No external tool transmission**: Endoscopy images must NEVER be sent to external APIs (including vision understanding tools, cloud services, or any remote inference endpoints). All processing must occur locally.
2. **Local-only analysis**: Use only local file operations (Read, Glob, Bash with local tools). Do not use `mcp__MiniMax__understand_image` or similar external vision APIs.
3. **No screenshot sharing**: Endoscopy images should not be shared via screenshot tools that route to external services.
4. **Embedding extraction**: Ensure the encoder model runs locally (e.g., via `torch.load` of a locally-hosted ViT checkpoint).
5. **Classified/sensitive data**: If endoscopy data is classified, proprietary, or contains patient-identifiable information, it must be processed only on air-gapped systems.

## GPU Selection

Before running any CUDA command, check `nvidia-smi` to see which GPU has more free memory.

Currently available GPUs:
- GPU 0: NVIDIA RTX A6000 (~17GB free)
- GPU 1: NVIDIA RTX A6000 (~5GB free)

**Use cuda:0 for most training (more free memory available).**

## Quick Start Commands

```bash
# Check GPU availability
nvidia-smi

# Demo with synthetic data
python scripts/main.py --mode demo

# Full training
python scripts/main.py --mode train \
    --paired_image_csv data/processed/paired_image_ms-bcpp.csv \
    --paired_rna_csv data/processed/paired_rna_ms-bcpp.csv \
    --unpaired_image_csv data/processed/unpaired_image.csv \
    --unpaired_rna_csv data/processed/unpaired_rna.csv \
    --drug_embeddings_csv data/processed/drug_embeddings_20260224.csv \
    --ic50_csv data/processed/ic50_data_20260224.csv \
    --cellline_rna_csv data/processed/ccle_rna_for_ic50.csv

# Run tests
pytest tests/test_model.py -v
```

Check every 3 minutes if running background commands.

## Running DR-A Pretraining

```bash
# DR-A: IC50-aware SupCon + cross-modal reconstruction (15 epochs, ~70 min on A6000)
python scripts/pretrain_phase3.py --device cuda:0 --epochs 15

# DR-A + ssGSEA (best model, 5 KV tokens)
python scripts/pretrain_with_ssgsea.py --device cuda:0 --epochs 15
```

## Running Fine-tuning

```bash
# Default fine-tuning — DR-A checkpoint + simple MSE (BEST: R²=0.838 on RNA-filtered random split)
python scripts/main.py --mode finetune \
  --checkpoint checkpoints_save/checkpoints_phase3/pretrained_phase3.pt \
  --finetune_epochs 10 --device cuda:0 \
  --checkpoint_dir checkpoints_save/checkpoints_finetuned \
  --drug_embeddings_csv data/processed/drug_embeddings_20260224.csv \
  --ic50_csv data/processed/ic50_data_20260224.csv \
  --cellline_rna_csv data/processed/ccle_rna_for_ic50.csv \
  --qformer_finetune_lr_ratio 0.2
```

## Running Inference

```bash
python scripts/inference.py \
  --checkpoint checkpoints_save/checkpoints_phase3/pretrained_phase3.pt \
  --device cuda:0 \
  --cellline_rna_csv data/processed/ccle_rna_for_ic50.csv \
  --output_dir reports/inference_drna
```

## Running Benchmarks

```bash
# DR-A vs CLRNA benchmark (3-fold CV)
python scripts/benchmark_phase3.py --device cuda:0 --output_dir reports/ablations_v5

# 5-fold NCC with ssGSEA comparison
python scripts/benchmark_with_ssgsea.py --device cuda:0 --output_dir reports/benchmark_with_ssgsea

# Random split benchmark
python scripts/benchmark_random_split.py --device cuda:0

# NCD evaluation (unseen drugs)
python scripts/main.py --mode evaluate_ncd \
    --checkpoint checkpoints_save/checkpoints_CLRNA/pretrained.pt \
    --ic50_csv data/processed/ic50_data_20260224.csv \
    --drug_embeddings_csv data/processed/drug_embeddings_20260224.csv \
    --cellline_rna_csv data/processed/ccle_rna_for_ic50.csv \
    --device cuda:0 --n_folds 5 --finetune_epochs 10
```

## Note on v3 Tricks

v3 tricks (Huber, EMA, R-Drop, tissue multitask) are **data-hungry** and catastrophically fail below 50% training data (R²=0.005 at 10%). Even at 100% data they provide negligible benefit over simple MSE. **Use simple MSE fine-tuning as the default.**
