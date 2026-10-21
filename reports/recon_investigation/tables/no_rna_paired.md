# Does the reconstruction term help where RNA is missing?

Row-paired over **10 seeds**; 173,942 unique (cell line, drug) pairs per seed; 71,155 no-RNA rows over 406 cell lines vs 592 with RNA. `y_true` identical between arms (max diff 0).

## Per-seed R2 by stratum

|                              |   mean |    std |
|:-----------------------------|-------:|-------:|
| ('all', 'one_term_w0')       | 0.7872 | 0.0027 |
| ('all', 'two_term_w0.5')     | 0.7867 | 0.0028 |
| ('has_rna', 'one_term_w0')   | 0.8107 | 0.0039 |
| ('has_rna', 'two_term_w0.5') | 0.8102 | 0.0043 |
| ('no_rna', 'one_term_w0')    | 0.7534 | 0.0018 |
| ('no_rna', 'two_term_w0.5')  | 0.7532 | 0.0015 |

## Paired tests

| comparison                                        |   n_seeds |   mean_diff |   sd_diff |   ci_low |   ci_high | ci_excludes_zero   |   wilcoxon_p |   wilcoxon_floor |   n_positive |
|:--------------------------------------------------|----------:|------------:|----------:|---------:|----------:|:-------------------|-------------:|-----------------:|-------------:|
| PRIMARY DiD: (two-one)|no_rna - (two-one)|has_rna |        10 |      0.0004 |    0.0054 |  -0.0029 |    0.0034 | False              |       0.7695 |           0.0020 |            5 |
| two_term - one_term (all)                         |        10 |     -0.0004 |    0.0036 |  -0.0025 |    0.0018 | False              |       0.5566 |           0.0020 |            4 |
| two_term - one_term (has_rna)                     |        10 |     -0.0006 |    0.0058 |  -0.0039 |    0.0029 | False              |       0.6953 |           0.0020 |            4 |
| two_term - one_term (no_rna)                      |        10 |     -0.0002 |    0.0010 |  -0.0008 |    0.0003 | False              |       0.5566 |           0.0020 |            5 |

## Reading

**The primary test is the difference-in-differences**, first row: it asks whether the two-term objective helps *more* where RNA is absent than where it is present. The stratum-wise rows are descriptive.

R2 levels must not be compared across strata -- the no-RNA rows are a different subpopulation with different target variance. Only within-stratum paired deltas and the DiD are interpretable.

**Verification gate:** the `(all)` row should reproduce the previously computed 10-seed paired result (delta = -0.0004, CI [-0.0030, +0.0022]). Measured here: -0.0004, CI [-0.0025, +0.0018].
