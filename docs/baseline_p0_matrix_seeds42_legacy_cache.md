# Common-P0 baseline matrix aggregation

> **Eligibility warning:** `†` values completed the requested raw-code → DNA conversion and common biological post-processing, but are diagnostic-only when legacy cache provenance keeps `main_protocol_eligible=false`. They must not be copied into the paper main table as eligible results.

- Train seed(s): `42`
- Validation split: ratio `0.1`, seed `42`; E* chosen by raw base-Hamming mAP@R
- Budgets: `36 bit → 18 bases`, `48 bit → 24 bases`
- Post-processing: exact common DNA projection on both query and database
- Complete: 81 / 117; missing: 36; invalid: 0; duplicate: 0; implementation-blocked: 81
- Noncanonical source-profile manifests excluded before duplicate handling: 16
- Implementation audit: `blocked_unverified_implementation_source`
- Strict paper-table admission: `False`; reasons: `['matrix_incomplete_or_invalid', 'not_all_records_paper_table_eligible', 'fewer_than_three_train_seeds']`
- Three-seed mean±std: pending (the current matrix is single-seed)

## U0 visual-only — 30 bits / 15 bases

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| CIMON | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| MLS³RDUH (paper-cache) | - | - | - | - |
| GreedyHash | - | - | - | - |
| Bi-half | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| SDC (paper) | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| OH | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| HHCH | - | - | - | - |
| CroVCA | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |

## U0 visual-only — 36 bits / 18 bases

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | - | - | - | - |
| CIMON | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| MLS³RDUH (paper-cache) | - | - | - | - |
| GreedyHash | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| Bi-half | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| SDC (paper) | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| OH | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| HHCH | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| CroVCA | - | - | - | - |

## U0 visual-only — 48 bits / 24 bases

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | - | - | - | - |
| CIMON | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| MLS³RDUH (paper-cache) | - | - | - | - |
| GreedyHash | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| Bi-half | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| SDC (paper) | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| OH | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| HHCH | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |
| CroVCA | - | - | - | - |

## U2 taxonomy-assisted — 30 bits / 15 bases

UMRCH consumes the exact target benchmark taxonomy and is not part of the strict U0 headline comparison.

| Method | Flickr25K | MSCOCO | NUS-WIDE |
|---|---:|---:|---:|
| UMRCH | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |

## U2 taxonomy-assisted — 36 bits / 18 bases

UMRCH consumes the exact target benchmark taxonomy and is not part of the strict U0 headline comparison.

| Method | Flickr25K | MSCOCO | NUS-WIDE |
|---|---:|---:|---:|
| UMRCH | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |

## U2 taxonomy-assisted — 48 bits / 24 bases

UMRCH consumes the exact target benchmark taxonomy and is not part of the strict U0 headline comparison.

| Method | Flickr25K | MSCOCO | NUS-WIDE |
|---|---:|---:|---:|
| UMRCH | SHA-DRIFT | SHA-DRIFT | SHA-DRIFT |

## Implementation fingerprint audit

Old run manifests were not rewritten. Their exact recorded source fingerprints are compared across cells and against the current worktree; only unknown cross-cell drift blocks aggregation.

- Status: **blocked_unverified_implementation_source**
- Cross-cell comparison safe: `false`

| Source path | Recorded SHA-256 version(s) | Current SHA-256 | Classification |
|---|---|---|---|
| `baseline/base_model.py` | `b9d7e1a0cfab613fd72ce3dc38fafeca82ca99f6121d20138bf22239ef34f3be`, `c9f39c05a27cca24ab0084bdd00462b634953499021a67c1cdc18dfc33837a47` | `b9d7e1a0cfab613fd72ce3dc38fafeca82ca99f6121d20138bf22239ef34f3be` | `non_scientific_dispatch_extension_only` |
| `scripts/baseline_val_select_p0.py` | `770156da4e0dea34e01144acf68d4256e14ce3263c464e53e095e727b60d48cd`, `cd273738899355e327381808b795956d9676a71faf898ac04e64e8e9e5ca38cc` | `770156da4e0dea34e01144acf68d4256e14ce3263c464e53e095e727b60d48cd` | `unknown_cross_cell_implementation_drift` |
| `scripts/extract_flat_baseline.py` | `02c805b0ce914daf8d54469f9a9369cf1cd96cacba55367ad8da1adff8284c4f`, `433be227bf28131a3266e298e37b7a7fcab5e11d41005b75629d6904f81d990e` | `433be227bf28131a3266e298e37b7a7fcab5e11d41005b75629d6904f81d990e` | `non_scientific_bit_budget_extension` |
| `scripts/run_modern_baseline_p0.py` | `4e9959b28fd7118ecdbd794555e22ede53064df6fe447c2ab85d30c1c8541a6c`, `9809a70fde66d473540fa11d10752a83453d60ac6bfc9b80d0490a1e7ce7eca5` | `4e9959b28fd7118ecdbd794555e22ede53064df6fe447c2ab85d30c1c8541a6c` | `non_scientific_for_reviewed_variants_only` |

- **Non-scientific implementation warning:** `baseline/base_model.py` — reviewed exact-hash transition adds only CRH lazy dispatch aliases and CLI help; existing U0/U2 model construction is unchanged.
- **Non-scientific implementation warning:** `scripts/extract_flat_baseline.py` — reviewed exact-hash transition widens the accepted checkpoint bit budget from (36, 48) to (30, 36, 48) and updates the error string; no change to packing, extraction, or evaluation.
- **Non-scientific implementation warning:** `scripts/run_modern_baseline_p0.py` — reviewed exact hashes cover the isolated CRH dispatch addition and the later CIBHash-only horizon correction; the latter is scientific for CIBHash and non-scientific only for the explicitly listed unaffected variants.

## Canonical source-profile exclusions

These manifests were rejected by exact source digest before duplicate resolution; no score or timestamp was consulted.

| Cell | Required profile | Actual source SHA-256 | Manifest |
|---|---|---|---|
| `u0/cibhash/CIFAR10/36b/seed42` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_cibhash_cifar10_36b_seed42/attempt_001/260722/cibhash_cifar10_36b_P0refit_seed42_p1a85fe43b576_e4_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/CIFAR10/48b/seed42` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_cibhash_cifar10_48b_seed42/attempt_001/260722/cibhash_cifar10_48b_P0refit_seed42_p299500f37c15_e4_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/Flickr25k/36b/seed42` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_cibhash_flickr25k_36b_seed42/attempt_001/260722/cibhash_flickr25k_36b_P0refit_seed42_p3f8f4d7f4fdd_e4_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/Flickr25k/48b/seed42` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_cibhash_flickr25k_48b_seed42/attempt_001/260722/cibhash_flickr25k_48b_P0refit_seed42_pfedc2f0d18f8_e4_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/MSCOCO/36b/seed42` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_cibhash_mscoco_36b_seed42/attempt_001/260722/cibhash_mscoco_36b_P0refit_seed42_p61251b540593_e19_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/MSCOCO/48b/seed42` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_cibhash_mscoco_48b_seed42/attempt_001/260722/cibhash_mscoco_48b_P0refit_seed42_p89fa8e659d94_e4_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/NUSWIDE/36b/seed42` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_cibhash_nuswide_36b_seed42/attempt_001/260722/cibhash_nuswide_36b_P0refit_seed42_p4ffe2154a14c_e4_dnaeval/p0_run_manifest.json` |
| `u0/cibhash/NUSWIDE/48b/seed42` | `cibhash-official-head-cache-adaptation-v2` | `1277bb94376e513f99aeb6f9d0c912c59502d035a4ba4626de9d8ca02ca696fc` | `/home/yschoi/GroundedDNA/result_baseline/p0_matrix_seeds42_legacy_cache/u0_cibhash_nuswide_48b_seed42/attempt_001/260722/cibhash_nuswide_48b_P0refit_seed42_p12b378894307_e4_dnaeval/p0_run_manifest.json` |
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

### Missing cells

- `u0/cibhash/Flickr25k/36b/seed42`
- `u0/cibhash/Flickr25k/48b/seed42`
- `u0/cibhash/MSCOCO/36b/seed42`
- `u0/cibhash/MSCOCO/48b/seed42`
- `u0/cibhash/NUSWIDE/36b/seed42`
- `u0/cibhash/NUSWIDE/48b/seed42`
- `u0/cibhash/CIFAR10/36b/seed42`
- `u0/cibhash/CIFAR10/48b/seed42`
- `u0/mls3rduh/Flickr25k/30b/seed42`
- `u0/mls3rduh/Flickr25k/36b/seed42`
- `u0/mls3rduh/Flickr25k/48b/seed42`
- `u0/mls3rduh/MSCOCO/30b/seed42`
- `u0/mls3rduh/MSCOCO/36b/seed42`
- `u0/mls3rduh/MSCOCO/48b/seed42`
- `u0/mls3rduh/NUSWIDE/30b/seed42`
- `u0/mls3rduh/NUSWIDE/36b/seed42`
- `u0/mls3rduh/NUSWIDE/48b/seed42`
- `u0/mls3rduh/CIFAR10/30b/seed42`
- `u0/mls3rduh/CIFAR10/36b/seed42`
- `u0/mls3rduh/CIFAR10/48b/seed42`
- `u0/greedyhash/Flickr25k/30b/seed42`
- `u0/greedyhash/MSCOCO/30b/seed42`
- `u0/greedyhash/NUSWIDE/30b/seed42`
- `u0/greedyhash/CIFAR10/30b/seed42`
- `u0/hhch/Flickr25k/30b/seed42`
- `u0/hhch/MSCOCO/30b/seed42`
- `u0/hhch/NUSWIDE/30b/seed42`
- `u0/hhch/CIFAR10/30b/seed42`
- `u0/crovca/Flickr25k/36b/seed42`
- `u0/crovca/Flickr25k/48b/seed42`
- `u0/crovca/MSCOCO/36b/seed42`
- `u0/crovca/MSCOCO/48b/seed42`
- `u0/crovca/NUSWIDE/36b/seed42`
- `u0/crovca/NUSWIDE/48b/seed42`
- `u0/crovca/CIFAR10/36b/seed42`
- `u0/crovca/CIFAR10/48b/seed42`
