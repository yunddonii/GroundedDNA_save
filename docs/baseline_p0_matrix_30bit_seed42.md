# Common-P0 baseline matrix aggregation

> **Eligibility warning:** `†` values completed the requested raw-code → DNA conversion and common biological post-processing, but are diagnostic-only when legacy cache provenance keeps `main_protocol_eligible=false`. They must not be copied into the paper main table as eligible results.

- Train seed(s): `42`
- Validation split: ratio `0.1`, seed `42`; E* chosen by raw base-Hamming mAP@R
- Budgets: `36 bit → 18 bases`, `48 bit → 24 bases`
- Post-processing: exact common DNA projection on both query and database
- Complete: 0 / 117; missing: 90; invalid: 27; duplicate: 0; implementation-blocked: 0
- Noncanonical source-profile manifests excluded before duplicate handling: 0
- Implementation audit: `no_completed_cells_to_audit`
- Strict paper-table admission: `False`; reasons: `['matrix_incomplete_or_invalid', 'not_all_records_paper_table_eligible', 'fewer_than_three_train_seeds']`
- Three-seed mean±std: pending (the current matrix is single-seed)

## U0 visual-only — 30 bits / 15 bases

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | ERR | ERR | ERR | ERR |
| CIMON | ERR | ERR | ERR | ERR |
| MLS³RDUH (paper-cache) | - | - | - | - |
| GreedyHash | - | - | - | - |
| Bi-half | ERR | ERR | ERR | ERR |
| SDC (paper) | ERR | ERR | ERR | ERR |
| OH | ERR | ERR | ERR | ERR |
| HHCH | - | - | - | - |
| CroVCA | ERR | ERR | ERR | ERR |

## U0 visual-only — 36 bits / 18 bases

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

## U2 taxonomy-assisted — 30 bits / 15 bases

UMRCH consumes the exact target benchmark taxonomy and is not part of the strict U0 headline comparison.

| Method | Flickr25K | MSCOCO | NUS-WIDE |
|---|---:|---:|---:|
| UMRCH | ERR | ERR | ERR |

## U2 taxonomy-assisted — 36 bits / 18 bases

UMRCH consumes the exact target benchmark taxonomy and is not part of the strict U0 headline comparison.

| Method | Flickr25K | MSCOCO | NUS-WIDE |
|---|---:|---:|---:|
| UMRCH | - | - | - |

## U2 taxonomy-assisted — 48 bits / 24 bases

UMRCH consumes the exact target benchmark taxonomy and is not part of the strict U0 headline comparison.

| Method | Flickr25K | MSCOCO | NUS-WIDE |
|---|---:|---:|---:|
| UMRCH | - | - | - |

## Implementation fingerprint audit

Old run manifests were not rewritten. Their exact recorded source fingerprints are compared across cells and against the current worktree; only unknown cross-cell drift blocks aggregation.

- Status: **no_completed_cells_to_audit**
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

### Invalid cells

- `u0/cibhash/Flickr25k/30b/seed42`
- `u0/cibhash/MSCOCO/30b/seed42`
- `u0/cibhash/NUSWIDE/30b/seed42`
- `u0/cibhash/CIFAR10/30b/seed42`
- `u0/cimon/Flickr25k/30b/seed42`
- `u0/cimon/MSCOCO/30b/seed42`
- `u0/cimon/NUSWIDE/30b/seed42`
- `u0/cimon/CIFAR10/30b/seed42`
- `u0/bihalf/Flickr25k/30b/seed42`
- `u0/bihalf/MSCOCO/30b/seed42`
- `u0/bihalf/NUSWIDE/30b/seed42`
- `u0/bihalf/CIFAR10/30b/seed42`
- `u0/sdc-paper/Flickr25k/30b/seed42`
- `u0/sdc-paper/MSCOCO/30b/seed42`
- `u0/sdc-paper/NUSWIDE/30b/seed42`
- `u0/sdc-paper/CIFAR10/30b/seed42`
- `u0/oh/Flickr25k/30b/seed42`
- `u0/oh/MSCOCO/30b/seed42`
- `u0/oh/NUSWIDE/30b/seed42`
- `u0/oh/CIFAR10/30b/seed42`
- `u0/crovca/Flickr25k/30b/seed42`
- `u0/crovca/MSCOCO/30b/seed42`
- `u0/crovca/NUSWIDE/30b/seed42`
- `u0/crovca/CIFAR10/30b/seed42`
- `u2/umrch/Flickr25k/30b/seed42`
- `u2/umrch/MSCOCO/30b/seed42`
- `u2/umrch/NUSWIDE/30b/seed42`

### Missing cells

- `u0/cibhash/Flickr25k/36b/seed42`
- `u0/cibhash/Flickr25k/48b/seed42`
- `u0/cibhash/MSCOCO/36b/seed42`
- `u0/cibhash/MSCOCO/48b/seed42`
- `u0/cibhash/NUSWIDE/36b/seed42`
- `u0/cibhash/NUSWIDE/48b/seed42`
- `u0/cibhash/CIFAR10/36b/seed42`
- `u0/cibhash/CIFAR10/48b/seed42`
- `u0/cimon/Flickr25k/36b/seed42`
- `u0/cimon/Flickr25k/48b/seed42`
- `u0/cimon/MSCOCO/36b/seed42`
- `u0/cimon/MSCOCO/48b/seed42`
- `u0/cimon/NUSWIDE/36b/seed42`
- `u0/cimon/NUSWIDE/48b/seed42`
- `u0/cimon/CIFAR10/36b/seed42`
- `u0/cimon/CIFAR10/48b/seed42`
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
- `u0/greedyhash/Flickr25k/36b/seed42`
- `u0/greedyhash/Flickr25k/48b/seed42`
- `u0/greedyhash/MSCOCO/30b/seed42`
- `u0/greedyhash/MSCOCO/36b/seed42`
- `u0/greedyhash/MSCOCO/48b/seed42`
- `u0/greedyhash/NUSWIDE/30b/seed42`
- `u0/greedyhash/NUSWIDE/36b/seed42`
- `u0/greedyhash/NUSWIDE/48b/seed42`
- `u0/greedyhash/CIFAR10/30b/seed42`
- `u0/greedyhash/CIFAR10/36b/seed42`
- `u0/greedyhash/CIFAR10/48b/seed42`
- `u0/bihalf/Flickr25k/36b/seed42`
- `u0/bihalf/Flickr25k/48b/seed42`
- `u0/bihalf/MSCOCO/36b/seed42`
- `u0/bihalf/MSCOCO/48b/seed42`
- `u0/bihalf/NUSWIDE/36b/seed42`
- `u0/bihalf/NUSWIDE/48b/seed42`
- `u0/bihalf/CIFAR10/36b/seed42`
- `u0/bihalf/CIFAR10/48b/seed42`
- `u0/sdc-paper/Flickr25k/36b/seed42`
- `u0/sdc-paper/Flickr25k/48b/seed42`
- `u0/sdc-paper/MSCOCO/36b/seed42`
- `u0/sdc-paper/MSCOCO/48b/seed42`
- `u0/sdc-paper/NUSWIDE/36b/seed42`
- `u0/sdc-paper/NUSWIDE/48b/seed42`
- `u0/sdc-paper/CIFAR10/36b/seed42`
- `u0/sdc-paper/CIFAR10/48b/seed42`
- `u0/oh/Flickr25k/36b/seed42`
- `u0/oh/Flickr25k/48b/seed42`
- `u0/oh/MSCOCO/36b/seed42`
- `u0/oh/MSCOCO/48b/seed42`
- `u0/oh/NUSWIDE/36b/seed42`
- `u0/oh/NUSWIDE/48b/seed42`
- `u0/oh/CIFAR10/36b/seed42`
- `u0/oh/CIFAR10/48b/seed42`
- `u0/hhch/Flickr25k/30b/seed42`
- `u0/hhch/Flickr25k/36b/seed42`
- `u0/hhch/Flickr25k/48b/seed42`
- `u0/hhch/MSCOCO/30b/seed42`
- `u0/hhch/MSCOCO/36b/seed42`
- `u0/hhch/MSCOCO/48b/seed42`
- `u0/hhch/NUSWIDE/30b/seed42`
- `u0/hhch/NUSWIDE/36b/seed42`
- `u0/hhch/NUSWIDE/48b/seed42`
- `u0/hhch/CIFAR10/30b/seed42`
- `u0/hhch/CIFAR10/36b/seed42`
- `u0/hhch/CIFAR10/48b/seed42`
- `u0/crovca/Flickr25k/36b/seed42`
- `u0/crovca/Flickr25k/48b/seed42`
- `u0/crovca/MSCOCO/36b/seed42`
- `u0/crovca/MSCOCO/48b/seed42`
- `u0/crovca/NUSWIDE/36b/seed42`
- `u0/crovca/NUSWIDE/48b/seed42`
- `u0/crovca/CIFAR10/36b/seed42`
- `u0/crovca/CIFAR10/48b/seed42`
- `u2/umrch/Flickr25k/36b/seed42`
- `u2/umrch/Flickr25k/48b/seed42`
- `u2/umrch/MSCOCO/36b/seed42`
- `u2/umrch/MSCOCO/48b/seed42`
- `u2/umrch/NUSWIDE/36b/seed42`
- `u2/umrch/NUSWIDE/48b/seed42`
