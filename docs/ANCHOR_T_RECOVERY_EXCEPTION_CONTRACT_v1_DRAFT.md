# Anchor model — r9 exception contract for a second recovery of NUS-WIDE seed 44 (design draft; audits 839–844)

**Status: design for review.**
- Nothing is implemented beyond the worker-dispatch repair (`f169b8b`) and its fixture fixes (`10af739`).
- Nothing runs.
- Executing it would also need:
  - the user's exception (requested by the audit in §839; pending, not asked again here);
  - a tested r9 generation and its exact request;
  - fresh peer GPU and disk coordination;
  - an exact audit approval.

## 1. What is bound (two histories, unchanged)

**The original stopped campaign:** `ancT9`. Its pinned lineage is `anchor_t_recovery_lineage_v1.json`
`b7676019…`, with:
- the settled parent ledger `31691514…`;
- the 11 carried cells;
- the interrupted target cell `nuswide|N=4|P=0.4,0.8|JD=0.05|stage=refit|seed=44|axis_center=anchors`.

**The failed recovery** (`ancT9r`, §838–§839). Every identity below is verified at admission, again just
before the claim, and again at publication:

| Item | Identity |
|---|---|
| request | `d34f505b…` (semantic), approval section 830 (window §838) |
| supervisor run | `20261007T061117Z-12e14ceb`: `exited`, rc 1, attempts [], charge 5.704555157572031 s |
| recovery ledger | `/home/yschoi/gdna_anchorRTrec_ops/device_budget_ledger.jsonl`. Its first two lines (start, final) are pinned **line by line**, because the file is append-only and grows (whole-file `593d327f…` at settlement) |
| claim (consumed) | `/home/yschoi/gdna_anchorRT_recovery_claims/dc4a8bc0….json` `7de24931…` |
| attempt | `ancT9r_attempt_ancR9_nuswide_A_v4_refit_N4_s44_AXanchors_P0408_JD005.json` `1add29fb…` |
| reservation / snapshot | `ancT9r_campaign_reservation.json` `6dffbab0…`; `ancT9r_snapshot_7205adea6c6238e4.json` `07708b36…` (bytes) |
| checked absence (lstat) | `ancT9r_entry_…`, `ancT9r_ancR9_…` (record), `ancT9r_recovery_complete.json`, and the target's 13 outputs |

**Rules for these identities:**
- They are pinned in a new JSON-only artifact, `anchor_t_recovery_exception_v1.json`. Its builder reads
  only the files above, and its digest is pinned in the stage source.
- **Absence is checked, not pinned.** It is re-observed by `lstat` at each boundary; an observation error or
  a dangling link refuses.
- The pre-failure payload access of the failed attempt (§839) is stated in the artifact. It is never
  described as "no access".

## 2. Stable exception claim

- **Key:** `json_digest({"failed_run_id", "failed_claim_sha256", "failed_attempt_sha256", "cell_id"})`. It
  is derived from the failed evidence alone, with no namespace, root, generation or caller in it.
- **File:** `RECOVERY_CLAIM_ROOT/<key>.json`, outside every worktree, created by O_EXCL and never removed.
- **Precondition:** the consumed claim `dc4a8bc0….json` must still exist at its bytes. The exception exists
  only because that claim was spent.
- **No second allowance:** an alternate namespace, record root, source generation or concurrent caller
  derives the same key, so it refuses after the first claim.
- **New namespace:** `ancT9s`. It must differ from both `ancT9` and `ancT9r`.

## 3. Accounting

**The append-only fixed recovery ledger** (§841 option 1):
- The supervisor stage stays `stage-T-recovery`, with the same ops root and the same 15,000 s stage budget.
- `prior_device_seconds` sums the failed run's final (5.704555157572031 s). Before its own allowance and
  headroom checks, the new run therefore has at most about **14,994.2954448424 s**.
- The parent binding is unchanged: 79,482.14623009507 s, ledger `31691514…`.
- The cumulative ceiling stays 94,482.14623009507 s.

**What is never done:**
- historical rows rewritten;
- a fresh ledger;
- double counting;
- the allowance refilled.

**Supervisor:** its source is unchanged. Its existing rules refuse an unfinished, unclean or unresolved
prior run.

## 4. Request form `anchor-terminal-test-recovery-request/2`

- **Carries everything from schema 1**, plus `exception_of` (the failed-recovery block of §1 and the
  exception-artifact pin) and `historical_generations` (r7 `2f24fc80…`, r8 `3537e297…`).
- **Claim fields:** `claim_key` is the exception key, and `claim_root` is unchanged.
- **The T entry:**
  - accepts schema 2 only together with mode `recovery`;
  - re-verifies the original lineage **and** the exception artifact;
  - reads the exception claim and requires it to name this request, approval line, namespace, record root
    and attempt.
- **Schema 1 requests:** refused by the r9 generation (the r8 recovery is spent).

## 5. Publication (`final_recovery_closure`)

- **Everything that holds today**, plus a re-check of the failed-recovery block: its pins, the ledger's
  first two lines, and the lstat absences.
- **The receipt** (`ancT9s_recovery_complete.json`) records:
  - `stopped` (ancT9);
  - `exception_of` (ancT9r, failed);
  - the 11 carried cells and the one executed `ancT9s` cell.

The failed attempt is never presented as an execution of the cell.

## 6. Source closure (as implemented; §841.5)

| Kind | Members |
|---|---|
| Changed vs S `04e7fe8` | `scripts/anchor_refit_stage.py` (worker dispatch + exception admission/claim/request/authority/closure), `scripts/anchor_confirm_manifest.py` (binds the exception artifact's pin), `docs/ANCHOR_T_RECOVERY_CONTRACT_v1.md` (the r9 section is appended; the designated contract path is unchanged, so the launcher's closure list is unchanged), `tests/test_anchor_t_recovery.py` |
| Unchanged | `scripts/anchor_terminal_test.py` (the entry takes the recovery schema from the stage constant, now `/2`), `scripts/phase3_selection_matrix.py` (the guard, the closure list), `scripts/anchor_confirm_supervisor.py` |
| Added outside the closure | `artifacts/anchor_confirmation/anchor_t_recovery_exception_v1.json` (`25320aa6…`, pinned in the stage source), `artifacts/anchor_confirmation/refit_v9/t_recovery_prep/build_exception.py`, these two design/proposal documents |

**Transition from r7.** The r7 → r9 closure transition keeps the reviewed sets: 7 changed, 2 added, 0
removed. Every r9 change falls inside members that already differ from r7.

**Tests** (`tests/test_anchor_t_recovery.py`, synthetic, OS sandbox):

| Area | Cases |
|---|---|
| Failed-attempt drift at admission | 13 forms: each pin, a ledger line, a missing ledger, an appended failed-run row, a present entry or record, a dangling receipt link, an lstat error, another cell, a non-consumed key |
| Namespaces | `ancT9`/`ancT9r` refused |
| Request | binds both histories; the exception key ignores namespace, root and generation |
| Rechecks | immediately before the claim (no claim, no attempt); in the T entry; during the cell (consumed claim, failed entry, dangling link, appended row) with no receipt |
| Schema | a schema-1 request is refused by the entry |
| Accounting | carry-forward in the append-only ledger (prefix kept, prior 5.70 s, cumulative); the carried charge pushes a tight budget into refusal |
| Concurrency | concurrent namespaces claim the exception once |
| Repeat | a second namespace after a completed exception is refused at admission |
| Real dispatch | the real-dispatch cases now run through the exception path |

## 7. Before any approved launch (not part of this preparation)

- **Fresh peer acknowledgment:** naming the physical GPU index and UUID, ownership, time window and release
  conditions, with shared-disk coordination.
- **Live checks:** re-check actual use and leases.
- **Approval:** the user's exception, then the audit's exact line.
