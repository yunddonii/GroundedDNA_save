# Common-P0 baseline matrix aggregation

> **Eligibility warning:** `†` values completed the requested raw-code → DNA conversion and common biological post-processing, but are diagnostic-only when legacy cache provenance keeps `main_protocol_eligible=false`. They must not be copied into the paper main table as eligible results.

- Train seed(s): `42, 43, 44`
- Validation split: ratio `0.1`, seed `42`; E* chosen by raw base-Hamming mAP@R
- Budgets: `36 bit → 18 bases`, `48 bit → 24 bases`
- Post-processing: exact common DNA projection on both query and database
- Complete: 24 / 24; missing: 0; invalid: 0; duplicate: 0; implementation-blocked: 0
- Noncanonical source-profile manifests excluded before duplicate handling: 0
- Implementation audit: `consistent_with_current_source`
- Strict paper-table admission: `False`; reasons: `['not_all_records_paper_table_eligible']`
- Three-seed mean±std: candidate aggregate shown below

## Supervised baseline — 36 bits / 18 bases

CRH consumes benchmark training labels and the target class taxonomy. It is reported only as a supervised baseline and must not be merged into either the U0 or U2 ranking.

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CRH (supervised) | 0.8628 ± 0.0068† | 0.8435 ± 0.0013† | 0.8532 ± 0.0025† | 0.9346 ± 0.0019† |

## Supervised baseline — 48 bits / 24 bases

CRH consumes benchmark training labels and the target class taxonomy. It is reported only as a supervised baseline and must not be merged into either the U0 or U2 ranking.

| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |
|---|---:|---:|---:|---:|
| CRH (supervised) | 0.8816 ± 0.0068† | 0.8668 ± 0.0011† | 0.8634 ± 0.0037† | 0.9383 ± 0.0019† |

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
