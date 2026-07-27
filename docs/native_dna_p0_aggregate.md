# Native-DNA common-P0 aggregation

- Expected train seeds: `42, 43, 44`
- Capacity: `18` DNA bases
- `U0-FD` and `S` are repository-local information-condition tags, not names claimed verbatim by every source paper.
- Supervision regimes: `U0-FD` = unsupervised with respect to the target benchmark: target ground-truth labels, taxonomy, and captions do not enter the encoder objective, and pseudo-pair targets come from frozen optimization-train feature distances; `S` = target ground-truth train labels enter the objective.
- As for the other common-P0 baselines, held-out labels are used only to score validation retrieval and select E*. Thus `U0-FD` describes the encoder objective, not a label-blind end-to-end evaluation protocol.
- PRIMO remains `U0-FD`: its frozen external hybridization-yield predictor is non-semantic and does not consume target labels.
- Every numeric cell requires matching checkpoint/evaluation SHA-256 and query/database post-compliance `1.0`.
- `†` is a complete diagnostic-only aggregate; it must not be copied to the strict paper main table.
- Records: expected 48, strict 0, diagnostic-only 48, missing 0, blocked 0, invalid 0, duplicate 0.

## Unsupervised (`U0-FD`; target-label-free encoder objective) feature-distance-pair direct predecessors — strict main

| Method | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |
|---|---:|---:|---:|---:|
| DNA24 (Stewart et al., 2018) | - | - | - | - |
| PRIMO-18 frozen-predictor length-transfer | - | - | - | - |

## Unsupervised (`U0-FD`; target-label-free encoder objective) feature-distance-pair direct predecessors — diagnostic

| Method | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |
|---|---:|---:|---:|---:|
| DNA24 (Stewart et al., 2018) | 0.7808 ± 0.0071† (raw 0.7917, u 0.8130, E* 42:59/43:19/44:4) | 0.6330 ± 0.0127† (raw 0.6391, u 0.4609, E* 42:54/43:24/44:39) | 0.7427 ± 0.0055† (raw 0.7514, u 0.4762, E* 42:4/43:9/44:44) | 0.7786 ± 0.0101† (raw 0.7858, u 0.3310, E* 42:4/43:4/44:64) |
| PRIMO-18 frozen-predictor length-transfer | 0.7882 ± 0.0209† (raw 0.8011, u 0.2142, E* 42:64/43:69/44:79) | 0.6251 ± 0.0129† (raw 0.6395, u 0.1329, E* 42:94/43:99/44:99) | 0.7320 ± 0.0109† (raw 0.7421, u 0.1845, E* 42:99/43:99/44:94) | 0.7344 ± 0.0123† (raw 0.7537, u 0.0816, E* 42:89/43:94/44:99) |

## Supervised (`S`) direct-prior baselines — strict main

| Method | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |
|---|---:|---:|---:|---:|
| Koike et al. (DATE/DAC 2024) | - | - | - | - |
| Koike et al. (TCBB 2026) | - | - | - | - |

## Supervised (`S`) direct-prior baselines — diagnostic

| Method | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |
|---|---:|---:|---:|---:|
| Koike et al. (DATE/DAC 2024) | 0.8353 ± 0.0047† (raw 0.8422, u 0.1178, E* 42:149/43:149/44:149) | 0.5795 ± 0.0151† (raw 0.5800, u 0.0109, E* 42:149/43:124/44:79) | 0.7453 ± 0.0419† (raw 0.7555, u 0.0546, E* 42:139/43:149/44:139) | 0.8850 ± 0.0090† (raw 0.8888, u 0.0366, E* 42:149/43:149/44:144) |
| Koike et al. (TCBB 2026) | 0.8916 ± 0.0037† (raw 0.8920, u 0.0362, E* 42:994/43:744/44:629) | 0.6189 ± 0.0045† (raw 0.6207, u 0.0221, E* 42:74/43:59/44:74) | 0.8156 ± 0.0009† (raw 0.8159, u 0.0253, E* 42:959/43:999/44:954) | 0.9304 ± 0.0054† (raw 0.9314, u 0.0368, E* 42:984/43:889/44:989) |

## Strict-main blockers

| Cell | Status | Blocker(s) |
|---|---|---|
| `feature_distance_pair/bee2018/CIFAR10/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2018/CIFAR10/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2018/CIFAR10/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2018/Flickr25k/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2018/Flickr25k/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2018/Flickr25k/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2018/MSCOCO/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2018/MSCOCO/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2018/MSCOCO/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2018/NUSWIDE/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2018/NUSWIDE/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2018/NUSWIDE/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `feature_distance_pair/bee2021/CIFAR10/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `feature_distance_pair/bee2021/CIFAR10/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `feature_distance_pair/bee2021/CIFAR10/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `feature_distance_pair/bee2021/Flickr25k/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `feature_distance_pair/bee2021/Flickr25k/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `feature_distance_pair/bee2021/Flickr25k/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `feature_distance_pair/bee2021/MSCOCO/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `feature_distance_pair/bee2021/MSCOCO/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `feature_distance_pair/bee2021/MSCOCO/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `feature_distance_pair/bee2021/NUSWIDE/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `feature_distance_pair/bee2021/NUSWIDE/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `feature_distance_pair/bee2021/NUSWIDE/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer |
| `supervised_direct_prior/koike2024/CIFAR10/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2024/CIFAR10/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2024/CIFAR10/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2024/Flickr25k/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2024/Flickr25k/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2024/Flickr25k/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2024/MSCOCO/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2024/MSCOCO/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2024/MSCOCO/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2024/NUSWIDE/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2024/NUSWIDE/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2024/NUSWIDE/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/CIFAR10/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/CIFAR10/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/CIFAR10/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/Flickr25k/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/Flickr25k/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/Flickr25k/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/MSCOCO/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/MSCOCO/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/MSCOCO/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/NUSWIDE/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/NUSWIDE/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
| `supervised_direct_prior/koike2026/NUSWIDE/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance |
