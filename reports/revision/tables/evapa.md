# EVAPA (Garai et al. 2026 definition)

Share of predictions within a factor of 7.129 of the experimental IC50, i.e. |Δln(IC50)| ≤ 1.9641.

`measured` excludes values above the maximum tested concentration and is the column comparable with Garai, who filtered those out.

| experiment | regime | arm | EVAPA all | SD | EVAPA measured | EVAPA extrapolated |
|---|---|---|---|---|---|---|
| mean_baselines | ncc | cellline_mean | 53.93 | 0.84 | 48.21 | 57.46 |
| mean_baselines | ncc | drug_mean | 84.58 | 0.69 | 78.75 | 88.14 |
| mean_baselines | ncc | global_mean | 53.93 | 0.84 | 48.21 | 57.46 |
| mean_baselines | ncd | cellline_mean | 55.06 | 7.06 | 50.10 | 58.42 |
| mean_baselines | ncd | drug_mean | 53.73 | 6.45 | 48.36 | 57.27 |
| mean_baselines | ncd | global_mean | 53.73 | 6.45 | 48.36 | 57.27 |
| multiseed | ncc | deepcdr_style | 86.41 | 0.51 | 80.07 | 90.26 |
| multiseed | ncc | drugcell_style | 86.71 | 0.52 | 81.39 | 89.95 |
| multiseed | ncc | multitoken_dra | 88.84 | 0.60 | 84.18 | 91.68 |
| multiseed | ncc | qformer_clrna | 87.00 | 0.55 | 81.67 | 90.25 |
| multiseed | ncc | qformer_randinit | 86.95 | 0.53 | 81.53 | 90.25 |
| multiseed | ncc | standalone_mlp | 86.56 | 0.48 | 81.38 | 89.71 |
| multiseed | ncc | tcnn_style | 86.67 | 0.57 | 81.42 | 89.86 |
| multiseed | ncc | xgboost | 86.60 | 0.49 | 81.33 | 89.80 |
| multiseed | random | deepcdr_style | 87.58 | 0.20 | 81.45 | 91.30 |
| multiseed | random | drugcell_style | 87.80 | 0.20 | 82.75 | 90.87 |
| multiseed | random | multitoken_dra | 89.93 | 0.34 | 85.60 | 92.57 |
| multiseed | random | qformer_clrna | 88.20 | 0.17 | 83.21 | 91.23 |
| multiseed | random | qformer_randinit | 88.27 | 0.16 | 83.33 | 91.27 |
| multiseed | random | standalone_mlp | 87.04 | 0.16 | 81.84 | 90.20 |
| multiseed | random | tcnn_style | 87.58 | 0.19 | 82.45 | 90.70 |
| multiseed | random | xgboost | 88.81 | 0.16 | 84.11 | 91.67 |
| transfer | ncc | clrna__standalone_mlp | 84.22 | 0.70 | 77.81 | 88.13 |
| transfer | ncc | clrna__xgboost | 78.11 | 0.65 | 69.98 | 83.07 |
| transfer | ncc | dra__standalone_mlp | 89.86 | 0.43 | 84.97 | 92.84 |
| transfer | ncc | dra__xgboost | 90.18 | 0.37 | 85.57 | 92.99 |
| transfer | ncc | randinit__standalone_mlp | 85.30 | 0.71 | 79.48 | 88.84 |
| transfer | ncc | randinit__xgboost | 79.73 | 0.88 | 72.16 | 84.34 |
| transfer | ncc | raw__deepcdr_style | 85.56 | 0.60 | 78.58 | 89.80 |
| transfer | ncc | raw__drugcell_style | 86.05 | 0.59 | 79.73 | 89.88 |
| transfer | ncc | raw__standalone_mlp | 86.14 | 0.57 | 80.33 | 89.67 |
| transfer | ncc | raw__tcnn_style | 85.97 | 0.39 | 79.84 | 89.70 |
| transfer | ncc | raw__xgboost | 86.68 | 0.51 | 81.49 | 89.84 |
| transfer | ncc | recon_only__standalone_mlp | 80.39 | 1.12 | 72.64 | 85.11 |
| transfer | ncc | recon_only__xgboost | 77.31 | 0.50 | 68.73 | 82.56 |