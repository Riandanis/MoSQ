# Drug-encoder sweep: seed replication vs partition replication

`primary` = one fixed compound split, 3 weight seeds (spread = seed noise).
`partition` = 3 resampled compound splits, seed 0 (spread = partition noise).

## ncd

| encoder | design | R2 | SD | n |
|---|---|---|---|---|
| descriptors | partition | +0.1205 | 0.0117 | 3 |
| morgan | partition | +0.0462 | 0.0082 | 3 |
| chemberta | partition | +0.0311 | 0.0280 | 3 |
| chemberta_mtr | partition | -0.0331 | 0.0636 | 3 |
| descriptors | primary | +0.0870 | 0.0008 | 3 |
| chemberta | primary | +0.0652 | 0.0422 | 3 |
| morgan | primary | +0.0462 | 0.0124 | 3 |
| chemberta_mtr | primary | -0.0259 | 0.0051 | 3 |

## scaffold

| encoder | design | R2 | SD | n |
|---|---|---|---|---|
| descriptors | partition | +0.0948 | 0.0466 | 3 |
| morgan | partition | +0.0332 | 0.0372 | 3 |
| chemberta | partition | +0.0164 | 0.0494 | 3 |
| chemberta_mtr | partition | -0.0484 | 0.0247 | 3 |
| descriptors | primary | +0.0889 | 0.0329 | 3 |
| chemberta | primary | +0.0731 | 0.0436 | 3 |
| morgan | primary | +0.0268 | 0.0196 | 3 |
| chemberta_mtr | primary | -0.0446 | 0.0201 | 3 |
