# Performance Results

## v5 Benchmark: RNA-Filtered NCC + ssGSEA (592 Cell-Lines) — NEW BEST

**RNA-filtered NCC + ssGSEA achieves R²=0.852 — the best result across all benchmarks.**

RNA-filtering removes zero-filled cell-lines from the 998 CL dataset, leaving 592 cell-lines with actual RNA data. This removes the zero-fill confound and significantly improves generalization.

| Metric | Value |
|--------|-------|
| **Mean R²** | **0.8518 ± 0.0042** |
| Mean Pearson R | 0.9241 |
| Mean Spearman R | 0.9124 |
| Mean RMSE | 1.065 |
| Mean MAE | 0.777 |

Per-fold R²: 0.845, 0.851, 0.856, 0.851, 0.857 (very consistent)

## v5 Benchmark: 5-Fold NCC (No Common Cell-Line) Ablation — FULL 998 Cell-Lines

**ssGSEA integration adds +2.3% R² (0.791 → 0.814).**

All models use MultiToken cell-line encoding + attn_pool + MSE fine-tuning, no tricks, qformer_lr_ratio=0.2.

| Model | R² | Pearson R | Spearman R | RMSE | MAE |
|-------|-----|-----------|------------|------|-----|
| **MultiToken + DR-A + ssGSEA (5 KV tokens)** | **0.8139 ± 0.0075** | **0.9027** | **0.8850** | — | — |
| MultiToken + DR-A | 0.7909 ± 0.0123 | 0.8907 ± 0.0058 | 0.8646 ± 0.0072 | 1.267 | 0.935 |
| Q-Former + CLRNA | 0.7587 ± 0.0129 | 0.8730 ± 0.0069 | 0.8469 ± 0.0074 | 1.361 | 1.012 |
| Detached MLP | 0.7509 ± 0.0127 | 0.8680 ± 0.0065 | 0.8374 ± 0.0081 | 1.384 | 1.034 |
| Simple MLP | 0.7501 ± 0.0129 | 0.8677 ± 0.0067 | 0.8370 ± 0.0089 | 1.386 | 1.035 |
| Standalone MLP | 0.7546 ± 0.0093 | 0.8691 ± 0.0056 | 0.8392 ± 0.0071 | 1.373 | 1.027 |
| XGBoost | 0.7550 ± 0.0101 | 0.8693 ± 0.0056 | 0.8389 ± 0.0068 | 1.372 | 1.028 |

ssGSEA per-fold: 0.8101, 0.8196, 0.8019, 0.8224, 0.8153 (all consistent)

## v5 Benchmark: Random Split (5-fold, full 998 cell-lines)

| Model | R² | Pearson R | Spearman R |
|-------|-----|-----------|------------|
| **DR-A (IC50-Aware Pretrain)** | **0.8050 ± 0.0032** | **0.8980 ± 0.0020** | **0.8775 ± 0.0024** |
| CLRNA baseline | 0.7786 ± 0.0021 | 0.8830 ± 0.0012 | 0.8556 ± 0.0010 |

**Comparison with Garai et al. (Random Split):** DeepCDR/DrugCell: R²=0.77, **GT DR-A: R²=0.805 (+3.5%)**

## v5 Benchmark: RNA-Filtered Splits (592 Cell-Lines, matching Garai protocol)

RNA-filtered evaluation (592/998 cell-lines) removes zero-fill confound.

| Split | Model | R² | Pearson R | Spearman R | RMSE | MAE |
|-------|-------|-----|-----------|------------|------|-----|
| **NCC + ssGSEA** | **DR-A + ssGSEA** | **0.852 ± 0.004** | **0.924** | **0.912** | 1.065 | 0.777 |
| **Random** | **DR-A** | **0.838 ± 0.003** | **0.916** | **0.899** | 1.114 | 0.816 |
| Random | CLRNA baseline | 0.781 ± 0.005 | 0.885 | 0.859 | 1.294 | 0.960 |
| **NCC** | **DR-A** | **0.801 ± 0.003** | **0.898** | **0.877** | 1.234 | 0.906 |
| NCC | MLP baseline | 0.760 ± 0.009 | 0.872 | 0.844 | 1.356 | 1.014 |

## Previous 3-Fold CV Results (superseded by 5-fold NCC ablation)

| Model | R² | Pearson R | Spearman R | RMSE | MAE |
|-------|-----|-----------|------------|------|-----|
| **DR-A (IC50-Aware Pretrain)** | **0.7803 ± 0.0021** | **0.8850 ± 0.0019** | **0.8616 ± 0.0034** | **1.3002** | **0.9582** |
| XGBoost (fair features) | 0.7583 ± 0.0031 | 0.8711 ± 0.0018 | 0.8414 ± 0.0017 | 1.3635 | 1.0209 |
| MultiToken CellLine (CLRNA) | 0.7584 ± 0.0059 | 0.8727 ± 0.0032 | 0.8431 ± 0.0040 | 1.3632 | 1.0121 |
| Feature CellLine (1 fused token) | 0.7568 ± 0.0064 | 0.8710 ± 0.0031 | 0.8418 ± 0.0039 | 1.3678 | 1.0105 |
| MultiToken + DCL pretrain | 0.7548 ± 0.0062 | 0.8691 ± 0.0035 | 0.8400 ± 0.0038 | 1.3734 | 1.0228 |
| Stage 2b pretrain | 0.7413 ± 0.0034 | 0.8624 ± 0.0018 | 0.8308 ± 0.0020 | 1.4107 | 1.0401 |
| ID-based CellLine (v4 baseline) | 0.7395 ± 0.0041 | 0.8625 ± 0.0026 | 0.8322 ± 0.0027 | 1.4156 | 1.0413 |

## Previous v4 Results (3-Fold CV, ID-based cell-line embeddings)

| Model | R² | Pearson R | Spearman R | RMSE | MAE |
|-------|-----|-----------|------------|------|-----|
| Concat v3 | 0.7428 ± 0.0050 | 0.8621 ± 0.0029 | 0.8317 ± 0.0033 | 1.4065 | 1.0469 |
| Q-Former v4 (attn pool+gate) | 0.7425 ± 0.0034 | 0.8622 ± 0.0017 | 0.8316 ± 0.0013 | 1.4075 | 1.0431 |
| Q-Former v3 (mean pool) | 0.7420 ± 0.0053 | 0.8621 ± 0.0027 | 0.8312 ± 0.0025 | 1.4088 | 1.0438 |
| XGBoost (CL one-hot) | 0.7411 ± 0.0025 | 0.8614 ± 0.0016 | 0.8299 ± 0.0013 | 1.4113 | 1.0610 |
| Detached MLP (pretrained) | 0.7350 ± 0.0092 | 0.8586 ± 0.0048 | 0.8261 ± 0.0042 | 1.4276 | 1.0668 |
| Simple MLP (from scratch) | 0.6761 ± 0.0036 | 0.8295 ± 0.0016 | 0.7900 ± 0.0059 | 1.5784 | 1.1855 |

## Literature Model Comparison (RNA-Filtered, 592 Cell-Lines, NCC 5-fold CV)

| Model | R² | Pearson R | RMSE | MAE |
|-------|-----|-----------|------|-----|
| **DR-A Q-Former (pretrained)** | **0.801 ± 0.003** | **0.897** | **1.23** | **0.91** |
| Q-Former (random init) | 0.765 ± 0.010 | 0.876 | 1.34 | 1.00 |
| Standalone MLP | 0.760 ± 0.009 | 0.872 | 1.36 | 1.01 |
| tCNN-style | 0.759 ± 0.008 | 0.872 | 1.36 | 1.01 |
| DrugCell-style | 0.754 ± 0.010 | 0.869 | 1.37 | 1.03 |
| DeepCDR-style | 0.746 ± 0.009 | 0.868 | 1.39 | 1.04 |

## Sample Efficiency (3-fold CV, 10 epochs, simple MSE)

| Fraction | Standalone MLP | Foundation (rand. init) | Pretrained (no tricks) | Pretrained (+ v3 tricks) |
|:---:|:---:|:---:|:---:|:---:|
| **10%** | 0.701 ± 0.003 | 0.715 ± 0.001 | **0.721 ± 0.003** | 0.005 ± 0.033 |
| **25%** | 0.736 ± 0.005 | 0.737 ± 0.003 | **0.743 ± 0.004** | 0.383 ± 0.008 |
| **50%** | 0.747 ± 0.003 | 0.750 ± 0.005 | **0.753 ± 0.005** | 0.720 ± 0.004 |
| **100%** | 0.753 ± 0.004 | 0.749 ± 0.006 | **0.759 ± 0.005** | 0.758 ± 0.006 |

## Architecture Freeze Ablations (single split, ID-based)

| Model | R² | Pearson R | Spearman R | RMSE | MAE |
|-------|-----|-----------|------------|------|-----|
| **Baseline Q-Former (CLRNA)** | **0.7184** | **0.8511** | **0.8198** | **1.496** | **1.116** |
| Frozen Q-Former | 0.7203 | 0.8487 | 0.8170 | 1.491 | 1.119 |
| Full LR (no differential) | 0.7196 | 0.8506 | 0.8183 | 1.493 | 1.106 |
| Frozen Type Embeds | 0.7163 | 0.8488 | 0.8156 | 1.502 | 1.123 |
| Frozen Projectors | FAILED (NaN) | — | — | — | — |

## NCD (No Common Drug) Evaluation Results

### 5-Fold Leak-Free NCD (DR-A)

| Metric | Value |
|--------|-------|
| R² | 0.208 |
| Pearson R | 0.544 |
| Spearman R | 0.468 |
| RMSE | 2.465 |
| MAE | 1.979 |

### 3-Fold NCD Results (CLRNA baseline)

| Fold | R² | Pearson R | Spearman R |
|------|------|-----------|------------|
| 1 | 0.044 | 0.416 | 0.325 |
| 2 | 0.232 | 0.509 | 0.433 |
| 3 | 0.109 | 0.402 | 0.405 |
| **Mean** | **0.128 ± 0.078** | **0.442 ± 0.047** | **0.388 ± 0.046** |

## Q-Former Feature Transferability

| Model | R² | Pearson R | Spearman R |
|-------|-----|-----------|------------|
| **XGBoost + DR-A Q-Former (768d)** | **0.7814 ± 0.0080** | **0.8869 ± 0.0049** | **0.8798 ± 0.0055** |
| XGBoost + Raw (1080d) | 0.7319 ± 0.0092 | 0.8585 ± 0.0058 | 0.8386 ± 0.0072 |
| XGBoost + CLRNA Q-Former (768d) | 0.6146 ± 0.0114 | 0.8034 ± 0.0086 | 0.7594 ± 0.0116 |

## NCC Leak-Free Control Results

| Experiment | Pretrain | Evaluate On | R² |
|-----------|----------|------------|-----|
| Single checkpoint | 100% CLs | All 998 CLs | **0.791** |
| Control (full 5-fold) | 80% CLs | Same 80% CLs | **0.908 ± 0.005** |
| Leak-free NCC | 80% CLs | Held-out 20% CLs | **0.700 ± 0.007** |

Full 5-fold control results: R²=0.908 ± 0.005, Pearson R=0.953 ± 0.003, Spearman R=0.939 ± 0.003

## Reports Index

- **RNA-filtered NCC + ssGSEA (592 CL)**: `reports/ncc_rnafiltered_ssgsea/benchmark_results.json` — **DR-A + ssGSEA R²=0.852** (NEW BEST)
- **ssGSEA 5-fold NCC (998 CL)**: `reports/benchmark_with_ssgsea/` — DR-A + ssGSEA R²=0.814
- **5-fold NCC ablation**: `reports/5fold_ncc_ablation/ablation_results.json` — DR-A R²=0.791
- **Random Split 5-fold**: `reports/random_split_5fold/random_split_results.json` — DR-A R²=0.805
- **RNA-filtered splits**: `reports/random_split_5fold_rnafiltered/random_split_rnafiltered_results.json` — DR-A R²=0.838 (random) / 0.801 (NCC)
- **NCD leak-free**: `reports/ncd_leakfree_singlefold/ncd_results.json` — R²=0.208
- **Literature comparison**: `reports/literature_comparison/literature_results.json` — DR-A Q-Former R²=0.801
- **Q-Former transferability**: `reports/qformer_query_clrna_vs_dra/clrna_vs_dra_qformer_results.json`
- **Leak-free DR-A transfer**: `reports/qformer_query_leakfree_ncc/leakfree_ncc_qformer_results.json` — R²=0.739
- **NCC Leak-Free 5-fold**: `reports/ncc_leakfree_5fold/ncc_leakfree_results.json` — R²=0.700
- **NCC Leak-Free Control**: `reports/ncc_leakfree_5fold_control/ncc_leakfree_control_results.json` — R²=0.908
