# Semantic-detail multi-dataset P0 aggregation

- Table ready: `true`
- Complete-valid: 8 / 8; incomplete: 0; missing: 0; invalid: 0
- Fixed contract: seed 42, K=128, 4 codons/codebook (24 bases), held-out-train E* selection, scratch full-train refit, mandatory bio-projection GC [0.416, 0.584].
- A is strict global-caption-free supervision. ABC adds the own local minimal-pair foil and post-VQ DNA-bit CIBHash KL.

## Validated cells

| Dataset | A: bio mAP@R / DNA-unique | ABC: bio mAP@R / DNA-unique | ABC−A mAP@R | ABC−A unique |
|---|---:|---:|---:|---:|
| Flickr25k | 0.8807 / 0.4924 (E*=4) | 0.8793 / 0.4877 (E*=4) | -0.0014 | -0.0047 |
| MSCOCO | 0.8155 / 0.2185 (E*=39) | 0.8207 / 0.2230 (E*=24) | +0.0052 | +0.0045 |
| NUS-WIDE | 0.8341 / 0.2419 (E*=4) | 0.8366 / 0.2226 (E*=4) | +0.0025 | -0.0193 |
| CIFAR10 | 0.9073 / 0.2519 (E*=9) | 0.9056 / 0.2442 (E*=14) | -0.0017 | -0.0077 |

## ABC versus archived K=128 · 24-base incumbent

| Dataset | Archived incumbent: bio mAP@R / DNA-unique | ABC: bio mAP@R / DNA-unique | ABC−incumbent mAP@R | ABC−incumbent unique |
|---|---:|---:|---:|---:|
| Flickr25k | 0.8742 / 0.4982 | 0.8793 / 0.4877 | +0.0051 | -0.0105 |
| MSCOCO | 0.8257 / 0.2184 | 0.8207 / 0.2230 | -0.0050 | +0.0045 |
| NUS-WIDE | 0.8328 / 0.2367 | 0.8366 / 0.2226 | +0.0038 | -0.0141 |
| CIFAR10 | 0.9033 / 0.2572 | 0.9056 / 0.2442 | +0.0024 | -0.0130 |
