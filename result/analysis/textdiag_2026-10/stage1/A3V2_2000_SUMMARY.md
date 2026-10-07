# A3 v2 readings (S = mean over axes of log(lift_own / lift_other); bootstrap 95 % CI over images; lexical pair rule on the run's reference captions)

Exploratory checkpoints (Gumbel ON) unless stated; validation rows only; descriptive, no test.

## codeword level, rule = lexical

| dataset | arm | n | S per seed | S mean | CI95 per seed (lo..hi) | axes R>0 per seed |
|---|---|---:|---|---:|---|---|
| flickr25k | td1_flickr_B0 | 3 | +0.109 / -0.022 / -0.051 | +0.012 | [+0.06,+0.15] / [-0.06,+0.03] / [-0.10,-0.00] | 2 / 2 / 3 |
| flickr25k | td1_flickr_H1 | 1 | +0.109 | +0.109 | [+0.06,+0.15] | 2 |
| flickr25k | td1_flickr_H2 | 3 | +0.083 / -0.005 / -0.021 | +0.019 | [+0.04,+0.13] / [-0.06,+0.05] / [-0.07,+0.03] | 3 / 3 / 3 |
| flickr25k | td1_flickr_OFF | 3 | +0.005 / +0.010 / -0.010 | +0.002 | [-0.03,+0.04] / [-0.03,+0.05] / [-0.04,+0.03] | 2 / 3 / 2 |

## codeword level, rule = lexical_axis_exclusive

| dataset | arm | n | S per seed | S mean | CI95 per seed (lo..hi) | axes R>0 per seed |
|---|---|---:|---|---:|---|---|
| flickr25k | td1_flickr_B0 | 3 | +0.141 / -0.030 / -0.076 | +0.012 | [+0.08,+0.20] / [-0.09,+0.04] / [-0.15,-0.00] | 3 / 2 / 3 |
| flickr25k | td1_flickr_H1 | 1 | +0.141 | +0.141 | [+0.08,+0.20] | 3 |
| flickr25k | td1_flickr_H2 | 3 | +0.110 / -0.009 / +0.002 | +0.034 | [+0.05,+0.18] / [-0.09,+0.06] / [-0.08,+0.08] | 3 / 2 / 3 |
| flickr25k | td1_flickr_OFF | 3 | -0.007 / +0.018 / -0.008 | +0.001 | [-0.06,+0.05] / [-0.05,+0.07] / [-0.06,+0.05] | 2 / 4 / 2 |

## codeword level, rule = caption_cos_top2pct

| dataset | arm | n | S per seed | S mean | CI95 per seed (lo..hi) | axes R>0 per seed |
|---|---|---:|---|---:|---|---|
| flickr25k | td1_flickr_B0 | 3 | +0.066 / -0.002 / -0.055 | +0.003 | [+0.02,+0.11] / [-0.04,+0.04] / [-0.10,-0.01] | 2 / 2 / 3 |
| flickr25k | td1_flickr_H1 | 1 | +0.066 | +0.066 | [+0.02,+0.11] | 2 |
| flickr25k | td1_flickr_H2 | 3 | -0.006 / +0.011 / +0.059 | +0.021 | [-0.05,+0.04] / [-0.05,+0.06] / [+0.00,+0.12] | 2 / 3 / 3 |
| flickr25k | td1_flickr_OFF | 3 | -0.010 / +0.022 / -0.031 | -0.006 | [-0.04,+0.02] / [-0.01,+0.06] / [-0.07,+0.00] | 1 / 2 / 1 |

## codon level, rule = lexical

| dataset | arm | n | S per seed | S mean | CI95 per seed (lo..hi) | axes R>0 per seed |
|---|---|---:|---|---:|---|---|
| flickr25k | td1_flickr_B0 | 3 | +0.096 / +0.005 / -0.067 | +0.011 | [+0.05,+0.14] / [-0.04,+0.06] / [-0.11,-0.02] | 3 / 3 / 3 |
| flickr25k | td1_flickr_H1 | 1 | +0.096 | +0.096 | [+0.05,+0.14] | 3 |
| flickr25k | td1_flickr_H2 | 3 | +0.161 / +0.196 / +0.109 | +0.155 | [+0.11,+0.21] / [+0.15,+0.24] / [+0.06,+0.16] | 4 / 4 / 3 |
| flickr25k | td1_flickr_OFF | 3 | +0.016 / +0.009 / -0.057 | -0.011 | [-0.02,+0.05] / [-0.04,+0.06] / [-0.10,-0.01] | 2 / 2 / 2 |

## codon level, rule = lexical_axis_exclusive

| dataset | arm | n | S per seed | S mean | CI95 per seed (lo..hi) | axes R>0 per seed |
|---|---|---:|---|---:|---|---|
| flickr25k | td1_flickr_B0 | 3 | +0.122 / -0.004 / -0.113 | +0.001 | [+0.06,+0.18] / [-0.07,+0.06] / [-0.17,-0.06] | 3 / 2 / 1 |
| flickr25k | td1_flickr_H1 | 1 | +0.122 | +0.122 | [+0.06,+0.18] | 3 |
| flickr25k | td1_flickr_H2 | 3 | +0.220 / +0.250 / +0.161 | +0.210 | [+0.15,+0.28] / [+0.19,+0.31] / [+0.09,+0.23] | 4 / 4 / 3 |
| flickr25k | td1_flickr_OFF | 3 | -0.005 / +0.028 / -0.090 | -0.023 | [-0.07,+0.05] / [-0.04,+0.09] / [-0.15,-0.03] | 2 / 2 / 1 |

## Calibration (codeword, lexical): S when a fraction f of rows carries the oracle code (3-seed mean; CI of seed 1)

| dataset | arm | f=0.0 | f=0.1 | f=0.2 | f=0.3 | f=0.4 | f=0.5 | f=0.7 | f=1.0 |
|---|---|---|---|---|---|---|---|---|---|
| flickr25k | td1_flickr_B0 | +0.01 [+0.06,+0.15] | +0.01 [+0.05,+0.15] | +0.06 [+0.09,+0.19] | +0.11 [+0.14,+0.25] | +0.24 [+0.24,+0.37] | +0.40 [+0.39,+0.52] | — | +0.92 [+0.86,+0.98] |
| flickr25k | td1_flickr_H1 | +0.11 [+0.06,+0.15] | +0.11 [+0.05,+0.15] | +0.14 [+0.09,+0.19] | +0.20 [+0.14,+0.25] | +0.30 [+0.24,+0.37] | +0.45 [+0.39,+0.52] | — | +0.92 [+0.86,+0.98] |
| flickr25k | td1_flickr_H2 | +0.02 [+0.04,+0.13] | +0.02 [+0.04,+0.13] | +0.05 [+0.06,+0.15] | +0.11 [+0.14,+0.26] | +0.25 [+0.25,+0.38] | +0.39 [+0.39,+0.53] | — | +0.92 [+0.86,+0.98] |
| flickr25k | td1_flickr_OFF | +0.00 [-0.03,+0.04] | +0.01 [-0.02,+0.06] | +0.04 [+0.00,+0.08] | +0.12 [+0.08,+0.18] | +0.23 [+0.17,+0.30] | +0.40 [+0.34,+0.48] | — | +0.92 [+0.86,+0.98] |
