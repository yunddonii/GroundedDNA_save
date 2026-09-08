# Common-P0 baseline matrix aggregation

> **Eligibility warning:** `†` values are diagnostic-only. Only `author_fixed_final` cells with a fully eligible cache and bio manifest may enter the paper main table.

- Train seed(s): `42, 43, 44`
- Protocol mode: `author_fixed_final`
- Checkpoint rule: full designated train for author horizon `H`; terminal `LAST = H-1`; no validation selection or refit
- Budgets: `30 bit → 15 bases (5x3)`
- Post-processing: exact common DNA projection on both query and database
- Complete: 105 / 108; missing: 3; invalid: 0; duplicate: 0; implementation-blocked: 0
- Noncanonical source-profile manifests excluded before duplicate handling: 0
- Implementation audit: `consistent_with_current_source`
- Strict paper-table admission: `False`; reasons: `['matrix_incomplete_or_invalid', 'not_all_records_paper_table_eligible']`
- Three-seed mean±std: candidate aggregate shown below

## U0 visual-only — 30 bits / 15 bases

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CIBHash | 0.7242 ± 0.0039 | 0.7692 ± 0.0076 | 0.7449 ± 0.0085 | 0.8164 ± 0.0056 |
| CIMON | 0.8141 ± 0.0034 | 0.6845 ± 0.0012 | 0.7980 ± 0.0046 | 0.8746 ± 0.0055 |
| MLS³RDUH (paper-cache) | 0.7507 ± 0.0081 | 0.6301 ± 0.0061 | 0.7590 ± 0.0026 | 0.6226 ± 0.0396 |
| GreedyHash | 0.6484 ± 0.0062 | 0.5621 ± 0.0060 | 0.6504 ± 0.0146 | 0.1059 ± 0.0000 |
| Bi-half | 0.8158 ± 0.0119 | 0.7191 ± 0.0037 | - | 0.7598 ± 0.0023 |
| SDC (paper) | 0.7267 ± 0.0034 | 0.8114 ± 0.0031 | 0.7692 ± 0.0031 | 0.7856 ± 0.0095 |
| OH | 0.8366 ± 0.0057 | 0.7653 ± 0.0103 | 0.8053 ± 0.0019 | 0.8665 ± 0.0096 |
| HHCH | 0.6144 ± 0.0218 | 0.4102 ± 0.0126 | 0.3844 ± 0.0217 | 0.2794 ± 0.0833 |
| CroVCA | 0.7715 ± 0.0017 | 0.8146 ± 0.0159 | 0.8002 ± 0.0041 | 0.8916 ± 0.0065 |

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

- `u0/bihalf/NUSWIDE/30b/seed42`
- `u0/bihalf/NUSWIDE/30b/seed43`
- `u0/bihalf/NUSWIDE/30b/seed44`
