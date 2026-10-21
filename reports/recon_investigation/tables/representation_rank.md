# Does dropping the reconstruction term degrade the representation?

Spectrum of the fused representation (the tensor the contrastive loss consumes), 20,000 rows, eval mode.

| checkpoint            |   recon_weight |   effective_rank |   participation_ratio |   stable_rank |   top1_var_share |   dims_for_90pct |
|:----------------------|---------------:|-----------------:|----------------------:|--------------:|-----------------:|-----------------:|
| published_w0.5        |          0.500 |            3.147 |                 1.597 |         1.275 |            0.784 |                3 |
| contrastive_only_w0   |          0.000 |            3.118 |                 1.589 |         1.271 |            0.787 |                3 |
| reconw3p0             |          3.000 |            3.272 |                 1.585 |         1.267 |            0.789 |                4 |
| reconw11p0            |         11.000 |            3.824 |                 1.706 |         1.316 |            0.760 |                7 |
| recon_only            |          1.000 |           10.100 |                 5.563 |         2.779 |            0.360 |               11 |
| dra_noclrna           |          0.500 |            3.528 |                 1.669 |         1.304 |            0.767 |                4 |
| binmode_random        |          0.500 |            2.543 |                 1.364 |         1.169 |            0.856 |                5 |
| binmode_per_drug_z    |          0.500 |           12.298 |                 3.691 |         2.017 |            0.496 |               31 |
| binmode_per_drug_rank |          0.500 |           12.386 |                 3.698 |         2.016 |            0.496 |               31 |

- Two-term effective rank **3.1**, one-term **3.1** (difference 0.0).
- Known-collapsed control `binmode_random`: **2.5**, top-1 direction holding 85.6% of variance.

The control separates cleanly, so the metric does detect collapse. The two-term and one-term arms do not separate — dropping the reconstruction term does not reduce the effective dimensionality of the representation.
