# What did the reconstruction head learn?

`R2 = 1 - MSE/Var(target)`, eval mode (no dropout), both branches run on every batch rather than coin-flipped. **R2 <= 0 means the head never beat emitting the global mean.** The random-decoder trace is not the baseline here and must not be used as one.

| checkpoint            |   recon_weight | branch        |    mse |   var_target |      r2 |   var_ratio |   r2_affine_oracle |   eta2_crossfit |   frac_of_ceiling |
|:----------------------|---------------:|:--------------|-------:|-------------:|--------:|------------:|-------------------:|----------------:|------------------:|
| published_w0.5        |         0.5000 | mask_drug     | 0.7450 |       0.7440 | -0.0013 |      0.0002 |             0.0000 |         -0.0114 |          nan      |
| published_w0.5        |         0.5000 | mask_cellline | 0.2231 |       0.2230 | -0.0003 |      0.0002 |             0.0001 |          0.0016 |           -0.1609 |
| contrastive_only_w0   |         0.0000 | mask_drug     | 2.2015 |       0.7824 | -1.8137 |      0.2213 |             0.0000 |         -0.0116 |          nan      |
| contrastive_only_w0   |         0.0000 | mask_cellline | 2.2532 |       0.2222 | -9.1390 |      2.0912 |             0.0000 |          0.0019 |        -4720.4699 |
| reconw3p0             |         3.0000 | mask_drug     | 0.7248 |       0.7239 | -0.0012 |      0.0003 |             0.0000 |         -0.0112 |          nan      |
| reconw3p0             |         3.0000 | mask_cellline | 0.2204 |       0.2204 | -0.0001 |      0.0001 |             0.0001 |          0.0021 |           -0.0514 |
| reconw11p0            |        11.0000 | mask_drug     | 0.6396 |       0.6389 | -0.0010 |      0.0002 |             0.0001 |         -0.0113 |          nan      |
| reconw11p0            |        11.0000 | mask_cellline | 0.2182 |       0.2182 | -0.0000 |      0.0001 |             0.0004 |          0.0023 |           -0.0054 |
| recon_only            |         1.0000 | mask_drug     | 0.0808 |       0.0808 |  0.0005 |      0.0006 |             0.0010 |         -0.0049 |          nan      |
| recon_only            |         1.0000 | mask_cellline | 0.1933 |       0.1934 |  0.0006 |      0.0005 |             0.0008 |          0.0016 |            0.3657 |
| dra_noclrna           |         0.5000 | mask_drug     | 0.7695 |       0.7683 | -0.0014 |      0.0003 |             0.0000 |         -0.0115 |          nan      |
| dra_noclrna           |         0.5000 | mask_cellline | 0.2243 |       0.2242 | -0.0003 |      0.0002 |             0.0001 |          0.0019 |           -0.1465 |
| binmode_random        |         0.5000 | mask_drug     | 0.0012 |       0.0012 | -0.0504 |      0.0045 |             0.0000 |         -0.0116 |          nan      |
| binmode_random        |         0.5000 | mask_cellline | 0.1917 |       0.1917 | -0.0000 |      0.0000 |             0.0001 |          0.0014 |           -0.0238 |
| binmode_per_drug_z    |         0.5000 | mask_drug     | 0.6724 |       0.6714 | -0.0015 |      0.0003 |             0.0001 |         -0.0092 |          nan      |
| binmode_per_drug_z    |         0.5000 | mask_cellline | 0.2235 |       0.2235 | -0.0001 |      0.0003 |             0.0004 |          0.0020 |           -0.0742 |
| binmode_per_drug_rank |         0.5000 | mask_drug     | 0.6999 |       0.6988 | -0.0016 |      0.0003 |             0.0002 |         -0.0087 |          nan      |
| binmode_per_drug_rank |         0.5000 | mask_cellline | 0.2225 |       0.2225 | -0.0002 |      0.0003 |             0.0004 |          0.0020 |           -0.0796 |

`eta2_crossfit` is the achievable ceiling in closed form: the conditioning input identifies only a group (cell line, or drug), so the best possible predictor is that group's mean and the best possible R2 is the one-way ANOVA eta-squared. `frac_of_ceiling` is R2 divided by it.

Absolute MSE is NOT comparable across branches -- mask_cellline predicts a mean of three tokens and so has structurally lower target variance. Compare R2 only.

## Verification

- MSE identity residual (should be ~0): 1.110e-14
