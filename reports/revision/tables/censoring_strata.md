# Performance by censoring stratum

Experiment `multiseed`, regime `ncc`. 62.2% of evaluated pairs are extrapolations above the maximum tested concentration.

R2 is normalised by within-stratum variance, so compare RMSE across strata and R2 only within one.

| arm | stratum | R2 | RMSE | MAE | Var(y) | n/fold |
|---|---|---|---|---|---|---|
| deepcdr_style | extrapolated | 0.5118 | 1.2291 | 0.8963 | 3.097 | 21,637 |
| deepcdr_style | measured | 0.6164 | 1.6089 | 1.2516 | 6.752 | 13,135 |
| drugcell_style | extrapolated | 0.5099 | 1.2315 | 0.9043 | 3.097 | 21,637 |
| drugcell_style | measured | 0.6359 | 1.5673 | 1.2163 | 6.752 | 13,135 |
| multitoken_dra | extrapolated | 0.5674 | 1.1568 | 0.8356 | 3.096 | 21,637 |
| multitoken_dra | measured | 0.6839 | 1.4603 | 1.1200 | 6.751 | 13,135 |
| qformer_clrna | extrapolated | 0.5155 | 1.2244 | 0.8915 | 3.097 | 21,637 |
| qformer_clrna | measured | 0.6442 | 1.5494 | 1.2020 | 6.752 | 13,135 |
| qformer_randinit | extrapolated | 0.5141 | 1.2263 | 0.8914 | 3.096 | 21,637 |
| qformer_randinit | measured | 0.6414 | 1.5555 | 1.2063 | 6.751 | 13,135 |
| standalone_mlp | extrapolated | 0.5085 | 1.2332 | 0.9061 | 3.096 | 21,637 |
| standalone_mlp | measured | 0.6340 | 1.5714 | 1.2202 | 6.751 | 13,135 |
| tcnn_style | extrapolated | 0.5094 | 1.2320 | 0.9056 | 3.097 | 21,637 |
| tcnn_style | measured | 0.6352 | 1.5688 | 1.2183 | 6.752 | 13,135 |
| xgboost | extrapolated | 0.5109 | 1.2303 | 0.9064 | 3.096 | 21,637 |
| xgboost | measured | 0.6378 | 1.5635 | 1.2179 | 6.751 | 13,135 |