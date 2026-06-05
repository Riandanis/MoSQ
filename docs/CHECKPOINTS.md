# Checkpoint Files

## Available Checkpoints

| Model | Checkpoint Path | Size | Use |
|-------|----------------|------|-----|
| **DR-A + ssGSEA** | **`checkpoints_save/checkpoints_phase3_ssgsea/pretrained_phase3_ssgsea.pt`** | ~289 MB | **Best pretrained model (recommended)** |
| DR-A (RNA only) | `checkpoints_save/checkpoints_phase3/pretrained_phase3.pt` | ~251 MB | DR-A without ssGSEA |
| CLRNA Pretrained (Stage 1) | `checkpoints_save/checkpoints_CLRNA/pretrained.pt` | 251 MB | Stage 1 CLRNA pretrained |
| DCL 15ep (Control) | `checkpoints_save/checkpoints_dcl_15ep/pretrained_dcl.pt` | — | DCL 25ep control experiment |
| Q-Former v3 (improved) | `checkpoints_save/checkpoints_improved_v3/best_ic50_model.pt` | 950 MB | Best v3 fine-tuned model |
| Concat Baseline v3 | `checkpoints_save/checkpoints_concat_baseline_v3/best_ic50_model.pt` | — | Concat baseline |

## Checkpoint Usage

### Recommended: DR-A + ssGSEA (Best)

```bash
python scripts/main.py --mode finetune \
  --checkpoint checkpoints_save/checkpoints_phase3_ssgsea/pretrained_phase3_ssgsea.pt \
  --finetune_epochs 10 --device cuda:0 \
  --checkpoint_dir checkpoints_save/checkpoints_finetuned \
  --drug_embeddings_csv data/processed/drug_embeddings_20260224.csv \
  --ic50_csv data/processed/ic50_data_20260224.csv \
  --cellline_rna_csv data/processed/ccle_rna_for_ic50.csv \
  --qformer_finetune_lr_ratio 0.2
```

### Alternative: DR-A (RNA only)

```bash
python scripts/main.py --mode finetune \
  --checkpoint checkpoints_save/checkpoints_phase3/pretrained_phase3.pt \
  --finetune_epochs 10 --device cuda:0 \
  --checkpoint_dir checkpoints_save/checkpoints_finetuned \
  --drug_embeddings_csv data/processed/drug_embeddings_20260224.csv \
  --ic50_csv data/processed/ic50_data_20260224.csv \
  --cellline_rna_csv data/processed/ccle_rna_for_ic50.csv \
  --qformer_finetune_lr_ratio 0.2
```

### Inference

```bash
python scripts/inference.py \
  --checkpoint checkpoints_save/checkpoints_phase3/pretrained_phase3.pt \
  --device cuda:0 \
  --cellline_rna_csv data/processed/ccle_rna_for_ic50.csv \
  --output_dir reports/inference_drna
```

## Training History

DR-A training history: `checkpoints_save/checkpoints_phase3/phase3_history.json`
