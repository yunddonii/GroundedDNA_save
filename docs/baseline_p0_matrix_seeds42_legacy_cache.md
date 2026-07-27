# Common-P0 baseline matrix aggregation

> **Eligibility warning:** `†` values completed the requested raw-code → DNA conversion and common biological post-processing, but are diagnostic-only when legacy cache provenance keeps `main_protocol_eligible=false`. They must not be copied into the paper main table as eligible results.

- Train seed(s): `42`
- Validation split: ratio `0.1`, seed `42`; E* chosen by raw base-Hamming mAP@R
- Budgets: `36 bit → 18 bases`, `48 bit → 24 bases`
- Post-processing: exact common DNA projection on both query and database
- Complete: 78 / 78; missing: 0; invalid: 0; duplicate: 0; implementation-blocked: 0
- Noncanonical source-profile manifests excluded before duplicate handling: 8
- Implementation audit: `warning_known_non_scientific_performance_memo_transition`
- Strict paper-table admission: `False`; reasons: `['not_all_records_paper_table_eligible', 'fewer_than_three_train_seeds']`
- Three-seed mean±std: pending (the current matrix is single-seed)

## U0 visual-only — 36 bits / 18 bases

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | 0.7824† | 0.7739† | 0.7829† | 0.8968† |
| CIMON | 0.8165† | 0.6751† | 0.7928† | 0.8478† |
| MLS³RDUH (paper-cache) | 0.7577† | 0.6311† | 0.7584† | 0.6409† |
| GreedyHash | 0.6077† | 0.5639† | 0.6511† | 0.1851† |
| Bi-half | 0.8161† | 0.7062† | 0.7489† | 0.7581† |
| SDC (paper) | 0.7230† | 0.8185† | 0.7520† | 0.8442† |
| OH | 0.8362† | 0.7587† | 0.8023† | 0.8737† |
| HHCH | 0.5867† | 0.4709† | 0.4329† | 0.2992† |
| CroVCA | 0.7682† | 0.8257† | 0.7944† | 0.8819† |

## U0 visual-only — 48 bits / 24 bases

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | 0.8077† | 0.7894† | 0.8100† | 0.9054† |
| CIMON | 0.8261† | 0.6877† | 0.8096† | 0.8576† |
| MLS³RDUH (paper-cache) | 0.7576† | 0.6414† | 0.7766† | 0.5780† |
| GreedyHash | 0.6234† | 0.5693† | 0.6731† | 0.2379† |
| Bi-half | 0.8207† | 0.7164† | 0.7571† | 0.7582† |
| SDC (paper) | 0.7427† | 0.8410† | 0.7854† | 0.8700† |
| OH | 0.8467† | 0.7740† | 0.8139† | 0.8797† |
| HHCH | 0.6209† | 0.5084† | 0.4735† | 0.3189† |
| CroVCA | 0.7634† | 0.8344† | 0.7967† | 0.8739† |

## U2 taxonomy-assisted — 36 bits / 18 bases

UMRCH consumes the exact target benchmark taxonomy and is not part of the strict U0 headline comparison.

| Method | Flickr25K | MSCOCO | NUS-WIDE |
|---|---:|---:|---:|
| UMRCH | 0.7994† | 0.8009† | 0.8224† |

## U2 taxonomy-assisted — 48 bits / 24 bases

UMRCH consumes the exact target benchmark taxonomy and is not part of the strict U0 headline comparison.

| Method | Flickr25K | MSCOCO | NUS-WIDE |
|---|---:|---:|---:|
| UMRCH | 0.7993† | 0.8237† | 0.8336† |

## Implementation fingerprint audit

Old run manifests were not rewritten. Their exact recorded source fingerprints are compared across cells and against the current worktree; only unknown cross-cell drift blocks aggregation.

- Status: **warning_known_non_scientific_performance_memo_transition**
- Cross-cell comparison safe: `true`

| Source path | Recorded SHA-256 version(s) | Current SHA-256 | Classification |
|---|---|---|---|
| `baseline/cache_provenance.py` | `4fe40c862a23e265ecf0559232d8b6ac5b84b282df720670c3aed58e3daf588d`, `ec761115371f09e6e3a00ab816888188b0df234805920b153ef06e08a598405c` | `4fe40c862a23e265ecf0559232d8b6ac5b84b282df720670c3aed58e3daf588d` | `non_scientific_performance_memo_only` |

- **Non-scientific implementation warning:** `baseline/cache_provenance.py` — reviewed exact-hash transition adds a stat-attested persistent SHA-256 performance memo; scientific inputs and outputs remain content-bound by the same artifact digests.

## Canonical source-profile exclusions

These manifests were rejected by exact source digest before duplicate resolution; no score or timestamp was consulted.

| Cell | Required profile | Actual source SHA-256 | Manifest |
|---|---|---|---|
| `u0/mls3rduh/CIFAR10/36b/seed42` | `ijcai2020-paper-cache-v1` | `e7859463e85f5164a16629e7da46d8055f46a33699913f0afc16fb5aa090d222` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_mls3rduh_cifar10_36b_seed42/attempt_001/260722/mls3rduh_cifar10_36b_P0refit_seed42_pc305818c6d78_e139_dnaeval/p0_run_manifest.json` |
| `u0/mls3rduh/CIFAR10/48b/seed42` | `ijcai2020-paper-cache-v1` | `e7859463e85f5164a16629e7da46d8055f46a33699913f0afc16fb5aa090d222` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_mls3rduh_cifar10_48b_seed42/attempt_001/260722/mls3rduh_cifar10_48b_P0refit_seed42_p1bfa8f37799c_e149_dnaeval/p0_run_manifest.json` |
| `u0/mls3rduh/Flickr25k/36b/seed42` | `ijcai2020-paper-cache-v1` | `e7859463e85f5164a16629e7da46d8055f46a33699913f0afc16fb5aa090d222` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_mls3rduh_flickr25k_36b_seed42/attempt_001/260722/mls3rduh_flickr25k_36b_P0refit_seed42_pf22cf9c12d39_e144_dnaeval/p0_run_manifest.json` |
| `u0/mls3rduh/Flickr25k/48b/seed42` | `ijcai2020-paper-cache-v1` | `e7859463e85f5164a16629e7da46d8055f46a33699913f0afc16fb5aa090d222` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_mls3rduh_flickr25k_48b_seed42/attempt_001/260722/mls3rduh_flickr25k_48b_P0refit_seed42_p4cebf460245d_e149_dnaeval/p0_run_manifest.json` |
| `u0/mls3rduh/MSCOCO/36b/seed42` | `ijcai2020-paper-cache-v1` | `e7859463e85f5164a16629e7da46d8055f46a33699913f0afc16fb5aa090d222` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_mls3rduh_mscoco_36b_seed42/attempt_001/260722/mls3rduh_mscoco_36b_P0refit_seed42_p86f8da1e76dc_e149_dnaeval/p0_run_manifest.json` |
| `u0/mls3rduh/MSCOCO/48b/seed42` | `ijcai2020-paper-cache-v1` | `e7859463e85f5164a16629e7da46d8055f46a33699913f0afc16fb5aa090d222` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_mls3rduh_mscoco_48b_seed42/attempt_001/260722/mls3rduh_mscoco_48b_P0refit_seed42_p422bee324cdc_e149_dnaeval/p0_run_manifest.json` |
| `u0/mls3rduh/NUSWIDE/36b/seed42` | `ijcai2020-paper-cache-v1` | `e7859463e85f5164a16629e7da46d8055f46a33699913f0afc16fb5aa090d222` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_mls3rduh_nuswide_36b_seed42/attempt_001/260722/mls3rduh_nuswide_36b_P0refit_seed42_p6263fbc386a8_e144_dnaeval/p0_run_manifest.json` |
| `u0/mls3rduh/NUSWIDE/48b/seed42` | `ijcai2020-paper-cache-v1` | `e7859463e85f5164a16629e7da46d8055f46a33699913f0afc16fb5aa090d222` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_mls3rduh_nuswide_48b_seed42/attempt_001/260722/mls3rduh_nuswide_48b_P0refit_seed42_p5b69f1af920f_e149_dnaeval/p0_run_manifest.json` |

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
