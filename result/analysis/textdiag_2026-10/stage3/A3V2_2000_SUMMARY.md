# A3 v2 readings (S = mean over axes of log(lift_own / lift_other); bootstrap 95 % CI over images; lexical pair rule on the run's reference captions)

Exploratory checkpoints (Gumbel ON) unless stated; validation rows only; descriptive, no test.

## codeword level, rule = lexical

| dataset | arm | n | S per seed | S mean | CI95 per seed (lo..hi) | axes R>0 per seed |
|---|---|---:|---|---:|---|---|
| flickr25k | td3_TD | 2 | -0.021 / +0.139 | +0.059 | [-0.07,+0.03] / [+0.09,+0.19] | 2 / 4 |
| flickr25k | td3_TDv2 | 1 | +0.065 | +0.065 | [+0.02,+0.12] | 3 |

## codeword level, rule = lexical_axis_exclusive

| dataset | arm | n | S per seed | S mean | CI95 per seed (lo..hi) | axes R>0 per seed |
|---|---|---:|---|---:|---|---|
| flickr25k | td3_TD | 2 | -0.027 / +0.174 | +0.073 | [-0.10,+0.04] / [+0.10,+0.25] | 2 / 4 |
| flickr25k | td3_TDv2 | 1 | +0.086 | +0.086 | [+0.01,+0.16] | 3 |

## codeword level, rule = caption_cos_top2pct

| dataset | arm | n | S per seed | S mean | CI95 per seed (lo..hi) | axes R>0 per seed |
|---|---|---:|---|---:|---|---|
| flickr25k | td3_TD | 2 | -0.058 / +0.017 | -0.020 | [-0.11,-0.01] / [-0.03,+0.06] | 2 / 2 |
| flickr25k | td3_TDv2 | 1 | +0.132 | +0.132 | [+0.08,+0.19] | 2 |

## codon level, rule = lexical

| dataset | arm | n | S per seed | S mean | CI95 per seed (lo..hi) | axes R>0 per seed |
|---|---|---:|---|---:|---|---|
| flickr25k | td3_TD | 2 | -0.091 / -0.046 | -0.068 | [-0.14,-0.04] / [-0.09,-0.01] | 2 / 3 |
| flickr25k | td3_TDv2 | 1 | +0.131 | +0.131 | [+0.08,+0.17] | 3 |

## codon level, rule = lexical_axis_exclusive

| dataset | arm | n | S per seed | S mean | CI95 per seed (lo..hi) | axes R>0 per seed |
|---|---|---:|---|---:|---|---|
| flickr25k | td3_TD | 2 | -0.128 / -0.054 | -0.091 | [-0.18,-0.07] / [-0.11,-0.00] | 2 / 3 |
| flickr25k | td3_TDv2 | 1 | +0.155 | +0.155 | [+0.10,+0.21] | 3 |

## Calibration (codeword, lexical): S when a fraction f of rows carries the oracle code (3-seed mean; CI of seed 1)

| dataset | arm | f=0.0 | f=0.1 | f=0.2 | f=0.3 | f=0.4 | f=0.5 | f=0.7 | f=1.0 |
|---|---|---|---|---|---|---|---|---|---|
| flickr25k | td3_TD | +0.06 [-0.07,+0.03] | — | — | +0.14 [+0.01,+0.13] | — | — | — | +0.92 [+0.86,+0.98] |
| flickr25k | td3_TDv2 | +0.06 [+0.02,+0.12] | — | — | +0.16 [+0.10,+0.22] | — | — | — | +0.92 [+0.86,+0.98] |
