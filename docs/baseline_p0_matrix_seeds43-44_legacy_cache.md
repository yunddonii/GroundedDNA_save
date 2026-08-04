# Common-P0 baseline matrix aggregation

> **Eligibility warning:** `†` values completed the requested raw-code → DNA conversion and common biological post-processing, but are diagnostic-only when legacy cache provenance keeps `main_protocol_eligible=false`. They must not be copied into the paper main table as eligible results.

- Train seed(s): `43, 44`
- Validation split: ratio `0.1`, seed `42`; E* chosen by raw base-Hamming mAP@R
- Budgets: `36 bit → 18 bases`, `48 bit → 24 bases`
- Post-processing: exact common DNA projection on both query and database
- Complete: 60 / 156; missing: 96; invalid: 0; duplicate: 0; implementation-blocked: 0
- Noncanonical source-profile manifests excluded before duplicate handling: 0
- Implementation audit: `consistent_with_current_source`
- Strict paper-table admission: `False`; reasons: `['matrix_incomplete_or_invalid', 'not_all_records_paper_table_eligible', 'fewer_than_three_train_seeds']`
- Three-seed mean±std: pending (the current matrix is single-seed)

## U0 visual-only — 36 bits / 18 bases

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | 0.7827 ± 0.0008† | 0.7681 ± 0.0019† | 0.7892 ± 0.0037† | - |
| CIMON | 0.8128 ± 0.0052† | 0.6687 ± 0.0014† | 0.7954 ± 0.0015† | - |
| MLS³RDUH (paper-cache) | 0.7552 ± 0.0036† | 0.6342 ± 0.0054† | 0.7549 ± 0.0005† | - |
| GreedyHash | 0.6701 ± 0.0096† | 0.5525 ± 0.0111† | 0.6414 ± 0.0059† | - |
| Bi-half | 0.8189 ± 0.0131† | 0.7062 ± 0.0052† | 0.7575 ± 0.0068† | - |
| SDC (paper) | 0.7279 ± 0.0013† | 0.8046 ± 0.0168† | 0.7534 ± 0.0164† | - |
| OH | 0.8310 ± 0.0040† | 0.7690 ± 0.0118† | 0.8030 ± 0.0052† | - |
| HHCH | 0.6245 ± 0.0152† | 0.4731 ± 0.0033† | 0.3874 ± 0.0036† | - |
| CroVCA | 0.7706 ± 0.0063† | 0.8195 ± 0.0012† | 0.8005 ± 0.0036† | - |

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

- Status: **consistent_with_current_source**
- Cross-cell comparison safe: `true`

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

- `u0/cibhash/Flickr25k/48b/seed43`
- `u0/cibhash/Flickr25k/48b/seed44`
- `u0/cibhash/MSCOCO/48b/seed43`
- `u0/cibhash/MSCOCO/48b/seed44`
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
- `u0/cimon/CIFAR10/36b/seed43`
- `u0/cimon/CIFAR10/36b/seed44`
- `u0/cimon/CIFAR10/48b/seed43`
- `u0/cimon/CIFAR10/48b/seed44`
- `u0/mls3rduh/Flickr25k/48b/seed43`
- `u0/mls3rduh/Flickr25k/48b/seed44`
- `u0/mls3rduh/MSCOCO/48b/seed43`
- `u0/mls3rduh/MSCOCO/48b/seed44`
- `u0/mls3rduh/NUSWIDE/48b/seed43`
- `u0/mls3rduh/NUSWIDE/48b/seed44`
- `u0/mls3rduh/CIFAR10/36b/seed43`
- `u0/mls3rduh/CIFAR10/36b/seed44`
- `u0/mls3rduh/CIFAR10/48b/seed43`
- `u0/mls3rduh/CIFAR10/48b/seed44`
- `u0/greedyhash/Flickr25k/48b/seed43`
- `u0/greedyhash/Flickr25k/48b/seed44`
- `u0/greedyhash/MSCOCO/48b/seed43`
- `u0/greedyhash/MSCOCO/48b/seed44`
- `u0/greedyhash/NUSWIDE/48b/seed43`
- `u0/greedyhash/NUSWIDE/48b/seed44`
- `u0/greedyhash/CIFAR10/36b/seed43`
- `u0/greedyhash/CIFAR10/36b/seed44`
- `u0/greedyhash/CIFAR10/48b/seed43`
- `u0/greedyhash/CIFAR10/48b/seed44`
- `u0/bihalf/Flickr25k/48b/seed43`
- `u0/bihalf/Flickr25k/48b/seed44`
- `u0/bihalf/MSCOCO/48b/seed43`
- `u0/bihalf/MSCOCO/48b/seed44`
- `u0/bihalf/NUSWIDE/48b/seed43`
- `u0/bihalf/NUSWIDE/48b/seed44`
- `u0/bihalf/CIFAR10/36b/seed43`
- `u0/bihalf/CIFAR10/36b/seed44`
- `u0/bihalf/CIFAR10/48b/seed43`
- `u0/bihalf/CIFAR10/48b/seed44`
- `u0/sdc-paper/Flickr25k/48b/seed43`
- `u0/sdc-paper/Flickr25k/48b/seed44`
- `u0/sdc-paper/MSCOCO/48b/seed43`
- `u0/sdc-paper/MSCOCO/48b/seed44`
- `u0/sdc-paper/NUSWIDE/48b/seed43`
- `u0/sdc-paper/NUSWIDE/48b/seed44`
- `u0/sdc-paper/CIFAR10/36b/seed43`
- `u0/sdc-paper/CIFAR10/36b/seed44`
- `u0/sdc-paper/CIFAR10/48b/seed43`
- `u0/sdc-paper/CIFAR10/48b/seed44`
- `u0/oh/Flickr25k/48b/seed43`
- `u0/oh/Flickr25k/48b/seed44`
- `u0/oh/MSCOCO/48b/seed43`
- `u0/oh/MSCOCO/48b/seed44`
- `u0/oh/NUSWIDE/48b/seed43`
- `u0/oh/NUSWIDE/48b/seed44`
- `u0/oh/CIFAR10/36b/seed43`
- `u0/oh/CIFAR10/36b/seed44`
- `u0/oh/CIFAR10/48b/seed43`
- `u0/oh/CIFAR10/48b/seed44`
- `u0/hhch/Flickr25k/48b/seed43`
- `u0/hhch/Flickr25k/48b/seed44`
- `u0/hhch/MSCOCO/48b/seed43`
- `u0/hhch/MSCOCO/48b/seed44`
- `u0/hhch/NUSWIDE/48b/seed43`
- `u0/hhch/NUSWIDE/48b/seed44`
- `u0/hhch/CIFAR10/36b/seed43`
- `u0/hhch/CIFAR10/36b/seed44`
- `u0/hhch/CIFAR10/48b/seed43`
- `u0/hhch/CIFAR10/48b/seed44`
- `u0/crovca/Flickr25k/48b/seed43`
- `u0/crovca/Flickr25k/48b/seed44`
- `u0/crovca/MSCOCO/48b/seed43`
- `u0/crovca/MSCOCO/48b/seed44`
- `u0/crovca/NUSWIDE/48b/seed43`
- `u0/crovca/NUSWIDE/48b/seed44`
- `u0/crovca/CIFAR10/36b/seed43`
- `u0/crovca/CIFAR10/36b/seed44`
- `u0/crovca/CIFAR10/48b/seed43`
- `u0/crovca/CIFAR10/48b/seed44`
- `u2/umrch/Flickr25k/48b/seed43`
- `u2/umrch/Flickr25k/48b/seed44`
- `u2/umrch/MSCOCO/48b/seed43`
- `u2/umrch/MSCOCO/48b/seed44`
- `u2/umrch/NUSWIDE/48b/seed43`
- `u2/umrch/NUSWIDE/48b/seed44`
