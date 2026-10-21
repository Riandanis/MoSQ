# Did the reconstruction decoder train?

`nn.Linear(768,768)` initialises uniform on ±0.036084. **No weight can leave that support without a gradient update**, and the Pearson kurtosis of a uniform is exactly 1.8.

## Decoder weight forensics

| checkpoint            |   recon_weight | head_trained   | tensor   |     std |   max_abs |   kurtosis |   max_over_init_bound |   frac_outside_init_support |
|:----------------------|---------------:|:---------------|:---------|--------:|----------:|-----------:|----------------------:|----------------------------:|
| published_w0.5        |        0.50000 | True           | 0.weight | 0.02071 |   0.06409 |    1.96436 |               1.77612 |                     0.04094 |
| published_w0.5        |        0.50000 | True           | 0.bias   | 0.02160 |   0.04383 |    1.82293 |               1.21466 |                     0.07031 |
| published_w0.5        |        0.50000 | True           | 2.weight | 0.02195 |   0.09075 |    2.25651 |               2.51488 |                     0.08347 |
| published_w0.5        |        0.50000 | True           | 2.bias   | 0.02303 |   0.05907 |    2.23144 |               1.63711 |                     0.10807 |
| contrastive_only_w0   |        0.00000 | False          | 0.weight | 0.02072 |   0.03590 |    1.80083 |               0.99487 |                     0.00000 |
| contrastive_only_w0   |        0.00000 | False          | 0.bias   | 0.02044 |   0.03587 |    1.81862 |               0.99399 |                     0.00000 |
| contrastive_only_w0   |        0.00000 | False          | 2.weight | 0.02073 |   0.03590 |    1.79930 |               0.99487 |                     0.00000 |
| contrastive_only_w0   |        0.00000 | False          | 2.bias   | 0.02027 |   0.03588 |    1.85484 |               0.99424 |                     0.00000 |
| reconw3p0             |        3.00000 | True           | 0.weight | 0.02066 |   0.06225 |    1.99712 |               1.72501 |                     0.04478 |
| reconw3p0             |        3.00000 | True           | 0.bias   | 0.02059 |   0.04853 |    1.88272 |               1.34504 |                     0.05990 |
| reconw3p0             |        3.00000 | True           | 2.weight | 0.02229 |   0.08878 |    2.32529 |               2.46032 |                     0.09382 |
| reconw3p0             |        3.00000 | True           | 2.bias   | 0.02360 |   0.06160 |    2.32341 |               1.70722 |                     0.12109 |
| reconw11p0            |       11.00000 | True           | 0.weight | 0.02078 |   0.06893 |    2.02098 |               1.91015 |                     0.04958 |
| reconw11p0            |       11.00000 | True           | 0.bias   | 0.02059 |   0.04556 |    1.88201 |               1.26262 |                     0.06510 |
| reconw11p0            |       11.00000 | True           | 2.weight | 0.02233 |   0.08330 |    2.31796 |               2.30853 |                     0.09421 |
| reconw11p0            |       11.00000 | True           | 2.bias   | 0.02353 |   0.05883 |    2.37903 |               1.63037 |                     0.13021 |
| recon_only            |        1.00000 | True           | 0.weight | 0.02032 |   0.04842 |    1.84321 |               1.34178 |                     0.01502 |
| recon_only            |        1.00000 | True           | 0.bias   | 0.02038 |   0.04113 |    1.83912 |               1.13983 |                     0.02865 |
| recon_only            |        1.00000 | True           | 2.weight | 0.02056 |   0.04821 |    1.84458 |               1.33596 |                     0.02092 |
| recon_only            |        1.00000 | True           | 2.bias   | 0.02028 |   0.04005 |    1.90161 |               1.10995 |                     0.02865 |
| dra_noclrna           |        0.50000 | True           | 0.weight | 0.02074 |   0.07238 |    1.97237 |               2.00583 |                     0.04292 |
| dra_noclrna           |        0.50000 | True           | 0.bias   | 0.02076 |   0.04799 |    1.90500 |               1.32981 |                     0.06641 |
| dra_noclrna           |        0.50000 | True           | 2.weight | 0.02190 |   0.08533 |    2.24365 |               2.36466 |                     0.08228 |
| dra_noclrna           |        0.50000 | True           | 2.bias   | 0.02266 |   0.05726 |    2.33154 |               1.58695 |                     0.11328 |
| binmode_random        |        0.50000 | True           | 0.weight | 0.02047 |   0.04972 |    1.83799 |               1.37790 |                     0.01673 |
| binmode_random        |        0.50000 | True           | 0.bias   | 0.02038 |   0.04228 |    1.84917 |               1.17162 |                     0.03125 |
| binmode_random        |        0.50000 | True           | 2.weight | 0.02051 |   0.04818 |    1.83920 |               1.33518 |                     0.01860 |
| binmode_random        |        0.50000 | True           | 2.bias   | 0.02026 |   0.04146 |    1.88595 |               1.14905 |                     0.02474 |
| binmode_per_drug_z    |        0.50000 | True           | 0.weight | 0.02063 |   0.06582 |    1.95565 |               1.82420 |                     0.03911 |
| binmode_per_drug_z    |        0.50000 | True           | 0.bias   | 0.02044 |   0.04437 |    1.87277 |               1.22953 |                     0.05208 |
| binmode_per_drug_z    |        0.50000 | True           | 2.weight | 0.02172 |   0.07449 |    2.17975 |               2.06422 |                     0.07845 |
| binmode_per_drug_z    |        0.50000 | True           | 2.bias   | 0.02192 |   0.05521 |    2.23810 |               1.53015 |                     0.08594 |
| binmode_per_drug_rank |        0.50000 | True           | 0.weight | 0.02060 |   0.06256 |    1.95003 |               1.73373 |                     0.03781 |
| binmode_per_drug_rank |        0.50000 | True           | 0.bias   | 0.02035 |   0.04482 |    1.88439 |               1.24199 |                     0.04948 |
| binmode_per_drug_rank |        0.50000 | True           | 2.weight | 0.02171 |   0.07507 |    2.17231 |               2.08037 |                     0.07756 |
| binmode_per_drug_rank |        0.50000 | True           | 2.bias   | 0.02208 |   0.05767 |    2.16738 |               1.59827 |                     0.08854 |

The zero-gradient arm (`contrastive_only_w0`) has `frac_outside_init_support` = 0.00000 and kurtosis 1.818 — i.e. still uniform, still inside the init support, shrunk only by AdamW weight decay (max/bound = 0.99487).

Trained arms reach `frac_outside_init_support` up to 0.13021 and kurtosis up to 2.379, with weights up to 2.51x the init bound. **The head demonstrably received and applied gradient.**

## Dose-response: the final loss FALLS as the weight rises

| checkpoint            |   recon_weight |   contrastive_weight |   recon_first |   recon_last |   recon_min |   pct_change |   abs_change |   final_cos_sim |
|:----------------------|---------------:|---------------------:|--------------:|-------------:|------------:|-------------:|-------------:|----------------:|
| published_w0.5        |         0.5000 |               1.0000 |        0.3943 |       0.4975 |      0.3913 |      26.1819 |       0.1032 |          0.8080 |
| contrastive_only_w0   |         0.0000 |               1.0000 |        1.0187 |       2.1660 |      1.0187 |     112.6229 |       1.1473 |          0.8079 |
| reconw3p0             |         3.0000 |               1.0000 |        0.3279 |       0.4672 |      0.3279 |      42.4574 |       0.1392 |          0.8084 |
| reconw11p0            |        11.0000 |               1.0000 |        0.2798 |       0.4264 |      0.2740 |      52.3903 |       0.1466 |          0.8134 |
| recon_only            |         1.0000 |               0.0000 |        0.2478 |       0.1531 |      0.1112 |     -38.2191 |      -0.0947 |          0.7914 |
| dra_noclrna           |         0.5000 |               1.0000 |        0.3850 |       0.4895 |      0.3850 |      27.1229 |       0.1044 |          0.8049 |
| binmode_random        |         0.5000 |               1.0000 |        0.2636 |       0.1043 |      0.0920 |     -60.4396 |      -0.1593 |          0.9996 |
| binmode_per_drug_z    |         0.5000 |               1.0000 |        0.2642 |       0.4456 |      0.1333 |      68.6214 |       0.1813 |          0.9442 |
| binmode_per_drug_rank |         0.5000 |               1.0000 |        0.2641 |       0.4574 |      0.1243 |      73.2035 |       0.1933 |          0.9464 |

Read `recon_last`, not `pct_change`. The percentages rise with weight only because the epoch-1 denominator falls faster than the numerator — and epoch 1 falls with weight because a more heavily weighted head learns faster within that first epoch. The published reading of this table is an artifact of a moving denominator.

## Mask tokens

| checkpoint            |   recon_weight | head_trained   |   drug_mask_std |   drug_mask_norm |   cl_mask_std |   cl_mask_norm |   pairwise_cosine |   drug_max_over_init |
|:----------------------|---------------:|:---------------|----------------:|-----------------:|--------------:|---------------:|------------------:|---------------------:|
| published_w0.5        |         0.5000 | True           |          0.0200 |           0.5548 |        0.0208 |         0.5765 |            0.1715 |               1.0907 |
| contrastive_only_w0   |         0.0000 | False          |          0.0190 |           0.5278 |        0.0198 |         0.5477 |           -0.0019 |               1.1024 |
| reconw3p0             |         3.0000 | True           |          0.0194 |           0.5362 |        0.0199 |         0.5507 |            0.0699 |               1.1345 |
| reconw11p0            |        11.0000 | True           |          0.0193 |           0.5354 |        0.0198 |         0.5500 |            0.0704 |               1.1351 |
| recon_only            |         1.0000 | True           |          0.0190 |           0.5271 |        0.0198 |         0.5481 |           -0.0058 |               1.1447 |
| dra_noclrna           |         0.5000 | True           |          0.0194 |           0.5378 |        0.0202 |         0.5590 |            0.1317 |               1.0900 |
| binmode_random        |         0.5000 | True           |          0.0191 |           0.5293 |        0.0198 |         0.5480 |            0.0153 |               1.1629 |
| binmode_per_drug_z    |         0.5000 | True           |          0.0192 |           0.5327 |        0.0198 |         0.5488 |            0.0622 |               1.1858 |
| binmode_per_drug_rank |         0.5000 | True           |          0.0192 |           0.5325 |        0.0198 |         0.5491 |            0.0543 |               1.1817 |

## What this does NOT establish

That the objective is useful. The frozen-head comparison is against a *random* MLP, whose squared error is strictly worse than a constant predictor, so beating it is compatible with having learned only the global mean. The bar that matters is mean-prediction (normalised R² > 0) and it is measured in `recon_r2.py`. **The removal decision is unaffected either way** — it rests on the paired downstream result (−0.0004, CI [−0.0030, +0.0022], 10 seeds).
