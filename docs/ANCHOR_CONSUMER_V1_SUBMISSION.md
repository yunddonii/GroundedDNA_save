# Anchor consumer v1 — source generation, synthetic suite and mutation battery (audits 833–837, 840)

**Scope: preparation only.**
- No real input manifest is pinned (`inputs_sha256` is null), so every real run refuses.
- No live formatting ran.
- TODO2 and TODO8 are not complete.
- Successful recovery settlement and a separately audited input manifest are prerequisites (§836, §839).

## Source generation

| Item | Value |
|---|---|
| Branch / worktree | `arch-exp-2026-09-anchor-consumer`, `/home/yschoi/gdna_anchor_consumer` |
| Base | S `04e7fe871cf9fc19a3e9ed6b713038686ff31830` (r8 submission) |
| `f7e1e71` | five additions: `scripts/anchor_t_consumer.py`, `scripts/make_anchor_bioproj_table.py`, `scripts/make_anchor_nmi_table.py`, `tests/test_anchor_t_consumer.py`, `artifacts/anchor_consumer/consumer_manifest_v1.json` |
| `8532862` | test only (audit 837 CM1): the no-read-before-pin contract is asserted directly; the manifest's test digest is refreshed; the three source modules are unchanged |
| Tested commit | **`8532862`** |
| This document | added after testing, as an addition only; no source, test or manifest changed |

**Module digests at `8532862`:**
- `anchor_t_consumer.py` `2539da57…`
- `make_anchor_bioproj_table.py` `5373da69…`
- `make_anchor_nmi_table.py` `c29caaef…`
- test file `0dadd8f0…`

The consumer imports no producer module.

## Synthetic suite (inside the reviewed OS sandbox, runner `a684299c…`)

| Run | Result |
|---|---|
| `final_f7e1e71/suite` | 153 passed (clean `f7e1e71`) |
| **`final_8532862/suite`** | **153 passed** (clean `8532862`, `result.json` `8ad3a092…`); the three modules import from inside the tree at the digests above (`consumer_module_identities.json` `4ec842d5…`) |
| development history (kept) | 1 failed / 70 passed; 1 / 132; 133 passed; 73 failed / 80 passed (missing `final_checkpoint` in the fixture cell); 2 failed / 151 passed; 153 passed |

## Mutation battery (29 mutants, each disabling one whole condition)

- **Harness:** `tools/consumer_mutation_battery.py`, frozen at `acad7416…` before launch.
- **Declarations:** `declarations.json` `0163284a…`, with markers taken from the observation pass `battery_observe_f7e1e71`.
- **CM1:** its new marker was confirmed by `battery_observe_cm1_8532862` against the changed test.
- **Final run** (`battery_final_8532862/report.json` `5348cf62…`, commit `8532862`):
  - baseline 29/29 pass unmutated;
  - **29/29 detected as declared**;
  - every run's imported module hashes equal the sandbox copy's bytes;
  - the sandbox copy is restored and clean, and the tree is clean after.

| Classification | Count | Meaning |
|---|---:|---|
| acceptance | 21 | the formatter (or reader) accepted defective input; the test caught it (`assert 0 == 2`, `DID NOT RAISE`, `assert (0 == 2)`) |
| contract | 2 | CM1: a read happened before the pin was validated (direct read-count assertion). CM2: a repeated reference re-read the file |
| refusal-order only | 6 | CM15, CM19, CM21, CM25, CM26, CM29: another guard still refuses; only the diagnostic or refusal order changes (§837) |

**Not mutated:**
- the two-run / line-after-final count path, because its other guards make the mutant hard to isolate without a
  crash;
- publication's exclusive `mkdir`, because the mutant crashes with `FileExistsError` instead of a refusal.

## What this does not show

- No real stage-T record or metric was read.
- The synthetic world proves the admission rules, not any live file.
- The consumer's single-run recovery-ledger rule matches the r8 contract. A future successful retry
  (r9 exception chain) would add a second run to the recovery ledger, so the consumer contract must then
  follow that accepted lineage (§839).
