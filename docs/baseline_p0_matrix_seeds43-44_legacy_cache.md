# Common-P0 baseline matrix aggregation

> **Eligibility warning:** `†` values completed the requested raw-code → DNA conversion and common biological post-processing, but are diagnostic-only when legacy cache provenance keeps `main_protocol_eligible=false`. They must not be copied into the paper main table as eligible results.

- Train seed(s): `43, 44`
- Validation split: ratio `0.1`, seed `42`; E* chosen by raw base-Hamming mAP@R
- Budgets: `36 bit → 18 bases`, `48 bit → 24 bases`
- Post-processing: exact common DNA projection on both query and database
- Complete: 70 / 156; missing: 86; invalid: 0; duplicate: 0; implementation-blocked: 0
- Noncanonical source-profile manifests excluded before duplicate handling: 6
- Implementation audit: `warning_known_non_scientific_performance_memo_transition`
- Strict paper-table admission: `False`; reasons: `['matrix_incomplete_or_invalid', 'not_all_records_paper_table_eligible', 'fewer_than_three_train_seeds']`
- Three-seed mean±std: pending (the current matrix is single-seed)

## U0 visual-only — 36 bits / 18 bases

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | - | - | - | - |
| CIMON | 0.8128 ± 0.0052† | 0.6687 ± 0.0014† | 0.7954 ± 0.0015† | 0.8509 ± 0.0054† |
| MLS³RDUH (paper-cache) | 0.7552 ± 0.0036† | 0.6342 ± 0.0054† | 0.7549 ± 0.0005† | 0.6059 ± 0.0208† |
| GreedyHash | 0.6701 ± 0.0096† | 0.5525 ± 0.0111† | 0.6414 ± 0.0059† | 0.1791 ± 0.0520† |
| Bi-half | 0.8189 ± 0.0131† | 0.7062 ± 0.0052† | 0.7575 ± 0.0068† | 0.7861 ± 0.0027† |
| SDC (paper) | 0.7279 ± 0.0013† | 0.8046 ± 0.0168† | 0.7534 ± 0.0164† | 0.8565 ± 0.0073† |
| OH | 0.8310 ± 0.0040† | 0.7690 ± 0.0118† | 0.8030 ± 0.0052† | 0.8671 ± 0.0042† |
| HHCH | 0.6245 ± 0.0152† | 0.4731 ± 0.0033† | 0.3874 ± 0.0036† | 0.2922 ± 0.0378† |
| CroVCA | 0.7706 ± 0.0063† | 0.8195 ± 0.0012† | 0.8005 ± 0.0036† | 0.8766 ± 0.0106† |

## U0 visual-only — 48 bits / 24 bases

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | - | - | - | - |
| CIMON | - | - | - | - |
| MLS³RDUH (paper-cache) | - | - | - | - |
| GreedyHash | - | - | - | - |
| Bi-half | - | - | - | - |
| SDC (paper) | - | - | - | - |
| OH | - | - | - | - |
| HHCH | - | - | - | - |
| CroVCA | - | - | - | - |

## U2 taxonomy-assisted — 36 bits / 18 bases

UMRCH consumes the exact target benchmark taxonomy and is not part of the strict U0 headline comparison.

| Method | Flickr25K | MSCOCO | NUS-WIDE |
|---|---:|---:|---:|
| UMRCH | 0.7913 ± 0.0002† | 0.8133 ± 0.0043† | 0.8265 ± 0.0017† |

## U2 taxonomy-assisted — 48 bits / 24 bases

UMRCH consumes the exact target benchmark taxonomy and is not part of the strict U0 headline comparison.

| Method | Flickr25K | MSCOCO | NUS-WIDE |
|---|---:|---:|---:|
| UMRCH | - | - | - |

## Implementation fingerprint audit

Old run manifests were not rewritten. Their exact recorded source fingerprints are compared across cells and against the current worktree; only unknown cross-cell drift blocks aggregation.

- Status: **warning_known_non_scientific_performance_memo_transition**
- Cross-cell comparison safe: `true`

| Source path | Recorded SHA-256 version(s) | Current SHA-256 | Classification |
|---|---|---|---|
| `scripts/run_modern_baseline_p0.py` | `2d7234a2a8f959f32a1f0fa5199cf00556289084f351359f7cb3cbcbdd4f5b16` | `1dec886eaed08b4f01cc04c8ebaec81b01952d1bc6913696cdcc7a75830461e0` | `non_scientific_for_reviewed_variants_only` |

- **Non-scientific implementation warning:** `scripts/run_modern_baseline_p0.py` — reviewed exact hashes cover the isolated CRH dispatch addition and the later CIBHash-only horizon correction; the latter is scientific for CIBHash and non-scientific only for the explicitly listed unaffected variants.

## Canonical source-profile exclusions

These manifests were rejected by exact source digest before duplicate resolution; no score or timestamp was consulted.

| Cell | Required profile | Actual source SHA-256 | Manifest |
|---|---|---|---|
| `u0/cibhash/Flickr25k/36b/seed43` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds43-44_legacy_cache/u0_cibhash_flickr25k_36b_seed43/attempt_001/260804/cibhash_flickr25k_36b_P0refit_seed43_pfe6e418cd661_e4_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/Flickr25k/36b/seed44` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds43-44_legacy_cache/u0_cibhash_flickr25k_36b_seed44/attempt_001/260804/cibhash_flickr25k_36b_P0refit_seed44_p25e9ef8206e4_e4_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/MSCOCO/36b/seed43` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds43-44_legacy_cache/u0_cibhash_mscoco_36b_seed43/attempt_001/260803/cibhash_mscoco_36b_P0refit_seed43_p62c5d8c1e8a0_e9_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/MSCOCO/36b/seed44` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds43-44_legacy_cache/u0_cibhash_mscoco_36b_seed44/attempt_001/260803/cibhash_mscoco_36b_P0refit_seed44_p173c85c3cfb0_e9_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/NUSWIDE/36b/seed43` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds43-44_legacy_cache/u0_cibhash_nuswide_36b_seed43/attempt_001/260804/cibhash_nuswide_36b_P0refit_seed43_p645b6d2d3384_e4_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/NUSWIDE/36b/seed44` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds43-44_legacy_cache/u0_cibhash_nuswide_36b_seed44/attempt_001/260804/cibhash_nuswide_36b_P0refit_seed44_p1ace8b374d96_e4_dnaeval/p0_run_manifest.json` |

## Exact DUH-EG

- Status: **blocked_not_aggregated**
- Reason: Exact DUH-EG requires the authors' ordered selected-WordNet noun bank or an unambiguous selection specification; neither is available.

## Audit legend

- `-`: no completed manifest found.
- `ERR`: artifact or protocol validation failed; no metric was admitted.
- `DUP`: multiple completed manifests exist for one cell; no automatic test-dependent choice was made.
- `SHA-DRIFT`: unknown cross-cell implementation drift; metric was not admitted.
- `PARTIAL`: only some requested seeds completed.
- `†`: completed but not main-table eligible (typically legacy cache provenance).

### Missing cells

- `u0/cibhash/Flickr25k/36b/seed43`
- `u0/cibhash/Flickr25k/36b/seed44`
- `u0/cibhash/Flickr25k/48b/seed43`
- `u0/cibhash/Flickr25k/48b/seed44`
- `u0/cibhash/MSCOCO/36b/seed43`
- `u0/cibhash/MSCOCO/36b/seed44`
- `u0/cibhash/MSCOCO/48b/seed43`
- `u0/cibhash/MSCOCO/48b/seed44`
- `u0/cibhash/NUSWIDE/36b/seed43`
- `u0/cibhash/NUSWIDE/36b/seed44`
- `u0/cibhash/NUSWIDE/48b/seed43`
- `u0/cibhash/NUSWIDE/48b/seed44`
- `u0/cibhash/CIFAR10/36b/seed43`
- `u0/cibhash/CIFAR10/36b/seed44`
- `u0/cibhash/CIFAR10/48b/seed43`
- `u0/cibhash/CIFAR10/48b/seed44`
- `u0/cimon/Flickr25k/48b/seed43`
- `u0/cimon/Flickr25k/48b/seed44`
- `u0/cimon/MSCOCO/48b/seed43`
- `u0/cimon/MSCOCO/48b/seed44`
- `u0/cimon/NUSWIDE/48b/seed43`
- `u0/cimon/NUSWIDE/48b/seed44`
- `u0/cimon/CIFAR10/48b/seed43`
- `u0/cimon/CIFAR10/48b/seed44`
- `u0/mls3rduh/Flickr25k/48b/seed43`
- `u0/mls3rduh/Flickr25k/48b/seed44`
- `u0/mls3rduh/MSCOCO/48b/seed43`
- `u0/mls3rduh/MSCOCO/48b/seed44`
- `u0/mls3rduh/NUSWIDE/48b/seed43`
- `u0/mls3rduh/NUSWIDE/48b/seed44`
- `u0/mls3rduh/CIFAR10/48b/seed43`
- `u0/mls3rduh/CIFAR10/48b/seed44`
- `u0/greedyhash/Flickr25k/48b/seed43`
- `u0/greedyhash/Flickr25k/48b/seed44`
- `u0/greedyhash/MSCOCO/48b/seed43`
- `u0/greedyhash/MSCOCO/48b/seed44`
- `u0/greedyhash/NUSWIDE/48b/seed43`
- `u0/greedyhash/NUSWIDE/48b/seed44`
- `u0/greedyhash/CIFAR10/48b/seed43`
- `u0/greedyhash/CIFAR10/48b/seed44`
- `u0/bihalf/Flickr25k/48b/seed43`
- `u0/bihalf/Flickr25k/48b/seed44`
- `u0/bihalf/MSCOCO/48b/seed43`
- `u0/bihalf/MSCOCO/48b/seed44`
- `u0/bihalf/NUSWIDE/48b/seed43`
- `u0/bihalf/NUSWIDE/48b/seed44`
- `u0/bihalf/CIFAR10/48b/seed43`
- `u0/bihalf/CIFAR10/48b/seed44`
- `u0/sdc-paper/Flickr25k/48b/seed43`
- `u0/sdc-paper/Flickr25k/48b/seed44`
- `u0/sdc-paper/MSCOCO/48b/seed43`
- `u0/sdc-paper/MSCOCO/48b/seed44`
- `u0/sdc-paper/NUSWIDE/48b/seed43`
- `u0/sdc-paper/NUSWIDE/48b/seed44`
- `u0/sdc-paper/CIFAR10/48b/seed43`
- `u0/sdc-paper/CIFAR10/48b/seed44`
- `u0/oh/Flickr25k/48b/seed43`
- `u0/oh/Flickr25k/48b/seed44`
- `u0/oh/MSCOCO/48b/seed43`
- `u0/oh/MSCOCO/48b/seed44`
- `u0/oh/NUSWIDE/48b/seed43`
- `u0/oh/NUSWIDE/48b/seed44`
- `u0/oh/CIFAR10/48b/seed43`
- `u0/oh/CIFAR10/48b/seed44`
- `u0/hhch/Flickr25k/48b/seed43`
- `u0/hhch/Flickr25k/48b/seed44`
- `u0/hhch/MSCOCO/48b/seed43`
- `u0/hhch/MSCOCO/48b/seed44`
- `u0/hhch/NUSWIDE/48b/seed43`
- `u0/hhch/NUSWIDE/48b/seed44`
- `u0/hhch/CIFAR10/48b/seed43`
- `u0/hhch/CIFAR10/48b/seed44`
- `u0/crovca/Flickr25k/48b/seed43`
- `u0/crovca/Flickr25k/48b/seed44`
- `u0/crovca/MSCOCO/48b/seed43`
- `u0/crovca/MSCOCO/48b/seed44`
- `u0/crovca/NUSWIDE/48b/seed43`
- `u0/crovca/NUSWIDE/48b/seed44`
- `u0/crovca/CIFAR10/48b/seed43`
- `u0/crovca/CIFAR10/48b/seed44`
- `u2/umrch/Flickr25k/48b/seed43`
- `u2/umrch/Flickr25k/48b/seed44`
- `u2/umrch/MSCOCO/48b/seed43`
- `u2/umrch/MSCOCO/48b/seed44`
- `u2/umrch/NUSWIDE/48b/seed43`
- `u2/umrch/NUSWIDE/48b/seed44`
