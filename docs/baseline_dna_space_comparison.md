# Fair DNA-space comparison (base Hamming, E*-matched)

All methods evaluated in the SAME 18-base A/C/G/T DNA code space with base
Hamming (18-position mismatch). Baselines: 36-bit hash -> [N,18,2] -> base id
(BASE_TO_BITS 00=A,01=C,10=G,11=T, base=hi*2+lo), extracted at P0-selected E*.
Metric: mAP@R (CalcTopMap), Jaccard>0 relevance, CIFAR10@1000 else @5000.

## Table 1 - base mAP@R (the fair table)

| Dataset | CIBHash | CIMON | MLS3RDUH | **Ours** | Best baseline | **Margin (base)** |
|---|---|---|---|---|---|---|
| Flickr25k | 0.8052 | 0.8241 | 0.7774 | **0.8810** | cimon 0.8241 | **+0.0569** |
| MSCOCO | 0.7981 | 0.6679 | 0.6373 | **0.8134** | cibhash 0.7981 | **+0.0153** |
| NUSWIDE | 0.8050 | 0.7832 | 0.7719 | **0.8334** | cibhash 0.8050 | **+0.0284** |
| CIFAR10 | 0.8972 | 0.8316 | 0.5786 | **0.9046** | cibhash 0.8972 | **+0.0074** |

## Table 2 - margin: base-space (fair) vs bit-space (P0), Ours over best baseline

| Dataset | Ours | Best-baseline bit mAP@R | Best-baseline base mAP@R | Margin bit-space | Margin base-space | Change |
|---|---|---|---|---|---|---|
| Flickr25k | 0.8810 | cimon 0.8288 | cimon 0.8241 | +0.0522 | +0.0569 | +0.0047 |
| MSCOCO | 0.8134 | cibhash 0.8112 | cibhash 0.7981 | +0.0022 | +0.0153 | +0.0130 |
| NUSWIDE | 0.8334 | cibhash 0.8152 | cibhash 0.8050 | +0.0182 | +0.0284 | +0.0102 |
| CIFAR10 | 0.9046 | cibhash 0.9004 | cibhash 0.8972 | +0.0042 | +0.0074 | +0.0032 |

## Table 3 - per-baseline bit->base cost (E*-matched), and base DB-unique ratio

| Method | Dataset | E* | bit mAP@R (=P0 sanity) | base mAP@R | delta | base DB-unique ratio |
|---|---|---|---|---|---|---|
| cibhash | Flickr25k | 4 | 0.8233 | 0.8052 | -0.0181 | 0.9626 |
| cibhash | MSCOCO | 19 | 0.8112 | 0.7981 | -0.0130 | 0.7247 |
| cibhash | NUSWIDE | 4 | 0.8152 | 0.8050 | -0.0102 | 0.8126 |
| cibhash | CIFAR10 | 4 | 0.9004 | 0.8972 | -0.0032 | 0.5037 |
| cimon | Flickr25k | 49 | 0.8288 | 0.8241 | -0.0047 | 0.8169 |
| cimon | MSCOCO | 59 | 0.6716 | 0.6679 | -0.0037 | 0.4288 |
| cimon | NUSWIDE | 54 | 0.7860 | 0.7832 | -0.0028 | 0.4983 |
| cimon | CIFAR10 | 59 | 0.8367 | 0.8316 | -0.0051 | 0.2337 |
| mls3rduh | Flickr25k | 59 | 0.7811 | 0.7774 | -0.0037 | 0.5111 |
| mls3rduh | MSCOCO | 59 | 0.6423 | 0.6373 | -0.0050 | 0.4350 |
| mls3rduh | NUSWIDE | 59 | 0.7746 | 0.7719 | -0.0026 | 0.4601 |
| mls3rduh | CIFAR10 | 59 | 0.5793 | 0.5786 | -0.0006 | 0.0073 |

Sanity gate: bit mAP@R reproduced each P0 test_mAP_at_R exactly (max |delta| 6.2e-7). base DB-unique ratio == bit DB-unique ratio for every baseline because the 2-bit->base map is a bijection (lossless), so DNA conversion changes ranking but not code multiplicity.
