# Semantic-detail multi-dataset P0 aggregation

- Table ready: `true`
- Complete-valid: 12 / 12; incomplete: 0; missing: 0; invalid: 0
- Fixed contract: seed 42, K=128, 4 codons/codebook (24 bases), held-out-train E* selection, scratch full-train refit, mandatory bio-projection GC [0.416, 0.584].
- A is strict global-caption-free supervision; AB adds the own local minimal-pair foil; ABC additionally enables post-VQ DNA-bit CIBHash KL.

## Validated cells

| Dataset | A: bio mAP@R / DNA-unique | AB: bio mAP@R / DNA-unique | ABC: bio mAP@R / DNA-unique |
|---|---:|---:|---:|
| Flickr25k | 0.8807 / 0.4924 (E*=4) | 0.8792 / 0.5036 (E*=4) | 0.8793 / 0.4877 (E*=4) |
| MSCOCO | 0.8155 / 0.2185 (E*=39) | 0.8150 / 0.2112 (E*=34) | 0.8207 / 0.2230 (E*=24) |
| NUS-WIDE | 0.8341 / 0.2419 (E*=4) | 0.8349 / 0.2096 (E*=9) | 0.8366 / 0.2226 (E*=4) |
| CIFAR10 | 0.9073 / 0.2519 (E*=9) | 0.9039 / 0.2346 (E*=9) | 0.9056 / 0.2442 (E*=14) |

## Pairwise arm deltas

| Dataset | AB−A mAP@R | AB−A unique | ABC−AB mAP@R | ABC−AB unique | ABC−A mAP@R | ABC−A unique |
|---|---:|---:|---:|---:|---:|---:|
| Flickr25k | -0.0016 | +0.0112 | +0.0001 | -0.0159 | -0.0014 | -0.0047 |
| MSCOCO | -0.0005 | -0.0073 | +0.0056 | +0.0118 | +0.0052 | +0.0045 |
| NUS-WIDE | +0.0008 | -0.0323 | +0.0017 | +0.0130 | +0.0025 | -0.0193 |
| CIFAR10 | -0.0035 | -0.0174 | +0.0018 | +0.0096 | -0.0017 | -0.0077 |

## Requested arms versus archived K=128 · 24-base incumbent

| Dataset | Arm | Archived: bio mAP@R / DNA-unique | Arm: bio mAP@R / DNA-unique | Δ mAP@R | Δ unique |
|---|---|---:|---:|---:|---:|
| Flickr25k | A | 0.8742 / 0.4982 | 0.8807 / 0.4924 | +0.0065 | -0.0058 |
| Flickr25k | AB | 0.8742 / 0.4982 | 0.8792 / 0.5036 | +0.0049 | +0.0054 |
| Flickr25k | ABC | 0.8742 / 0.4982 | 0.8793 / 0.4877 | +0.0051 | -0.0105 |
| MSCOCO | A | 0.8257 / 0.2184 | 0.8155 / 0.2185 | -0.0102 | +0.0000 |
| MSCOCO | AB | 0.8257 / 0.2184 | 0.8150 / 0.2112 | -0.0106 | -0.0073 |
| MSCOCO | ABC | 0.8257 / 0.2184 | 0.8207 / 0.2230 | -0.0050 | +0.0045 |
| NUS-WIDE | A | 0.8328 / 0.2367 | 0.8341 / 0.2419 | +0.0013 | +0.0052 |
| NUS-WIDE | AB | 0.8328 / 0.2367 | 0.8349 / 0.2096 | +0.0021 | -0.0271 |
| NUS-WIDE | ABC | 0.8328 / 0.2367 | 0.8366 / 0.2226 | +0.0038 | -0.0141 |
| CIFAR10 | A | 0.9033 / 0.2572 | 0.9073 / 0.2519 | +0.0041 | -0.0052 |
| CIFAR10 | AB | 0.9033 / 0.2572 | 0.9039 / 0.2346 | +0.0006 | -0.0226 |
| CIFAR10 | ABC | 0.9033 / 0.2572 | 0.9056 / 0.2442 | +0.0024 | -0.0130 |
