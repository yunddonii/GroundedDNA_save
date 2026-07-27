# Native-DNA 24-base common-P0 aggregation

- Expected train seeds: `42, 43, 44`
- Capacity: `24` DNA bases
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
| PRIMO-24 frozen-predictor length-transfer | - | - | - | - |

## Unsupervised (`U0-FD`; target-label-free encoder objective) feature-distance-pair direct predecessors — diagnostic

| Method | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |
|---|---:|---:|---:|---:|
| DNA24 (Stewart et al., 2018) | 0.7927 ± 0.0068† (raw 0.7968, u 0.9326, E* 42:19/43:29/44:64) | 0.6412 ± 0.0110† (raw 0.6447, u 0.6514, E* 42:64/43:14/44:4) | 0.7542 ± 0.0046† (raw 0.7577, u 0.7093, E* 42:14/43:9/44:29) | 0.7814 ± 0.0090† (raw 0.7840, u 0.5402, E* 42:44/43:59/44:64) |
| PRIMO-24 frozen-predictor length-transfer | 0.7962 ± 0.0125† (raw 0.7984, u 0.4773, E* 42:79/43:99/44:24) | 0.6326 ± 0.0049† (raw 0.6374, u 0.2858, E* 42:99/43:99/44:89) | 0.7522 ± 0.0033† (raw 0.7551, u 0.3757, E* 42:99/43:94/44:79) | 0.7569 ± 0.0085† (raw 0.7622, u 0.1899, E* 42:94/43:99/44:89) |

## Supervised (`S`) direct-prior baselines — strict main

| Method | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |
|---|---:|---:|---:|---:|
| Koike et al. (DATE/DAC 2024) | - | - | - | - |
| Koike et al. (TCBB 2026) | - | - | - | - |

## Supervised (`S`) direct-prior baselines — diagnostic

| Method | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |
|---|---:|---:|---:|---:|
| Koike et al. (DATE/DAC 2024) | 0.8167 ± 0.0071† (raw 0.8230, u 0.3508, E* 42:149/43:149/44:149) | 0.5788 ± 0.0203† (raw 0.5792, u 0.0214, E* 42:149/43:149/44:149) | 0.7598 ± 0.0113† (raw 0.7637, u 0.1009, E* 42:149/43:149/44:149) | 0.8812 ± 0.0049† (raw 0.8834, u 0.1075, E* 42:149/43:149/44:149) |
| Koike et al. (TCBB 2026) | 0.8939 ± 0.0043† (raw 0.8945, u 0.0791, E* 42:619/43:779/44:454) | 0.6359 ± 0.0128† (raw 0.6360, u 0.0404, E* 42:79/43:74/44:69) | 0.8285 ± 0.0016† (raw 0.8285, u 0.0765, E* 42:994/43:964/44:999) | 0.9295 ± 0.0014† (raw 0.9297, u 0.0576, E* 42:999/43:999/44:984) |

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
| `feature_distance_pair/bee2021/CIFAR10/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
| `feature_distance_pair/bee2021/CIFAR10/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
| `feature_distance_pair/bee2021/CIFAR10/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
| `feature_distance_pair/bee2021/Flickr25k/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
| `feature_distance_pair/bee2021/Flickr25k/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
| `feature_distance_pair/bee2021/Flickr25k/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
| `feature_distance_pair/bee2021/MSCOCO/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
| `feature_distance_pair/bee2021/MSCOCO/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
| `feature_distance_pair/bee2021/MSCOCO/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
| `feature_distance_pair/bee2021/NUSWIDE/seed42` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
| `feature_distance_pair/bee2021/NUSWIDE/seed43` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
| `feature_distance_pair/bee2021/NUSWIDE/seed44` | `complete_diagnostic_only` | legacy_cache_missing_strict_provenance; primo_frozen_predictor_length_transfer_80_to_24nt |
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
