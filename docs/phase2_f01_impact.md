# Phase 2 — what the epoch-0 extraction bug (F01) cost

F01 only: the same checkpoint re-inferred with the training epoch restored, so the router runs at the annealed Sinkhorn epsilon instead of the initial one. Not an N selection.

- Ranking below is **post-bio diagnostic**; D1's selection metric is raw
  base-Hamming mAP@R, which these artefacts do not store.
- Extraction manifests were **backfilled** after the fact and are marked
  `backfilled: true`; inputs were verified byte-identical to the legacy source.
- Paired cells: **15 / 15**
- GC window (both sides): count [6, 9] at 15 bases (`gc-40-60-inclusive-v1`)

| cell | mAP@R legacy | mAP@R fixed | Δ | DNA-uniq legacy | DNA-uniq fixed | Δ | NMI legacy | NMI fixed | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10/N4 | 0.8823 | 0.8839 | +0.0016 | 0.0704 | 0.0868 | +0.0164 | 0.6766 | 0.6090 | -0.0676 |
| cifar10/N9 | 0.8789 | 0.8736 | -0.0053 | 0.0791 | 0.0994 | +0.0203 | 0.6870 | 0.5775 | -0.1095 |
| cifar10/N19 | 0.8775 | 0.8674 | -0.0102 | 0.0815 | 0.1129 | +0.0313 | 0.7037 | 0.6176 | -0.0860 |
| cifar10/N39 | 0.8751 | 0.8675 | -0.0077 | 0.0855 | 0.1019 | +0.0164 | 0.7195 | 0.6524 | -0.0670 |
| flickr25k/N4 | 0.8565 | 0.8558 | -0.0007 | 0.2824 | 0.3560 | +0.0736 | 0.6170 | 0.5396 | -0.0775 |
| flickr25k/N9 | 0.8507 | 0.8420 | -0.0088 | 0.3298 | 0.3753 | +0.0455 | 0.6358 | 0.4814 | -0.1544 |
| flickr25k/N19 | 0.8496 | 0.8360 | -0.0136 | 0.3217 | 0.3927 | +0.0710 | 0.6603 | 0.5580 | -0.1023 |
| nuswide/N4 | 0.8150 | 0.8161 | +0.0011 | 0.1416 | 0.1905 | +0.0488 | 0.6182 | 0.5448 | -0.0734 |
| nuswide/N9 | 0.8115 | 0.8115 | +0.0001 | 0.1327 | 0.1841 | +0.0514 | 0.6484 | 0.5831 | -0.0654 |
| nuswide/N19 | 0.8106 | 0.8090 | -0.0016 | 0.1299 | 0.1693 | +0.0393 | 0.6623 | 0.6129 | -0.0494 |
| nuswide/N39 | 0.8039 | 0.8062 | +0.0023 | 0.1168 | 0.1485 | +0.0317 | 0.6804 | 0.6445 | -0.0359 |
| mscoco/N4 | 0.7884 | 0.7801 | -0.0082 | 0.1281 | 0.1844 | +0.0563 | 0.7055 | 0.6114 | -0.0941 |
| mscoco/N9 | 0.8030 | 0.7977 | -0.0054 | 0.1252 | 0.1741 | +0.0490 | 0.7181 | 0.5833 | -0.1349 |
| mscoco/N19 | 0.8076 | 0.8099 | +0.0023 | 0.1342 | 0.1945 | +0.0603 | 0.7174 | 0.6326 | -0.0848 |
| mscoco/N39 | 0.8287 | 0.8233 | -0.0054 | 0.1204 | 0.1719 | +0.0515 | 0.7172 | 0.6613 | -0.0559 |

Mean Δ (fixed − legacy): mAP@R -0.0040, DNA-uniq +0.0442, NMI -0.0839.
