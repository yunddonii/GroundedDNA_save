# Anchor model — r9 repair of the recovery dispatch defect: source, regression and contract proposal (audit 839)

**Status: preparation only.**
- Nothing ran on real data.
- No claim, attempt, lease, GPU or official test was touched.
- The failed r8 tree (`/data/yschoi/gdna_anchor_refit_v9r8` at S `04e7fe8`) and every real record are
  untouched.

**Before any scientific execution:**
- the user's decision on the exception the audit requested in §839 (pending; not asked again here);
- a new generation manifest;
- an exact request;
- fresh peer GPU coordination;
- a separate audit approval.

## 1. The defect (r8, S `04e7fe8`)

- `_run_recovery` called `run_terminal_test_cell` directly on the main thread (`anchor_refit_stage.py:1295`).
- The launcher's managed-child launcher refuses the main thread (`phase3_selection_matrix.py:1364–1366`), so
  that a signal handler cannot race the registration of a new child.
- Stage T's normal path dispatches its cells from `_stream` worker threads (`anchor_refit_stage.py:705–731`).
- **On 2026-10-07** the one approved recovery consumed the lineage claim and reserved its attempt, then refused
  at the first managed child: `exited`, rc 1, 0 device-s, 5.704555157572031 s charged.
- **Why the r8 tests missed it:** the recovery campaign fixture replaced `_run_managed_process` with a
  stand-in (`tests/test_anchor_t_recovery.py:507–527`), so no composed test reached the guard.

## 2. The repair (branch `arch-exp-2026-09-anchor-refit-r9`, worktree `/home/yschoi/gdna_anchor_refit_r9`, from S)

| Commit | Change |
|---|---|
| `61f62e1` | test only: a failing-before regression through the **real** managed-child boundary |
| `f169b8b` | `_run_recovery` runs its one cell in one worker thread |

**How the worker is used:**
- It runs the whole cell: the reservation-owner check, generation recheck, space check, `run_terminal_test_cell`
  (claim → attempt → entry → post-chain), the second recheck, and `final_recovery_closure`.
- The main thread starts and joins it, then publishes the receipt only when the worker returned a closure.
- Any exception, or a worker ending without a closure, prints the failure and returns 1, with no receipt.

**Unchanged:**
- the main-thread guard;
- `Popen` with `start_new_session` and inherited lease descriptors;
- the session registry and drain;
- the claim-before-attempt order;
- the one-cell / one-claim semantics;
- the final closure and receipt contents.

`git diff --stat 04e7fe8 f169b8b`: `scripts/anchor_refit_stage.py` (+34/−19), `tests/test_anchor_t_recovery.py` (+98).

## 3. The regression (`tests/test_anchor_t_recovery.py`, three tests)

**Fixture `real_dispatch`:** the composed recovery campaign with the launcher's **real**
`M._run_managed_process`, captured before any fixture replaced it. That keeps the main-thread guard, `Popen`,
session registry and drain.
- **What is substituted:** only the scientific work. Each producer command becomes one tiny private synthetic
  child:
  - the entry child performs the real `anchor_terminal_test.claim_entry` and writes the synthetic outputs;
  - post-chain children only record their start.
- **What is recorded:** the dispatching thread of every launch.

| Test | Proves |
|---|---|
| `…launches_its_children_from_a_worker_thread` | every launch is off the main thread, all from one worker; the synthetic entry starts exactly once; the claim exists when the entry child starts (claim → attempt → entry order); the five children run in chain order; the combined receipt is published |
| `…a_failed_entry_child_publishes_no_receipt` | an entry child exiting 1 stops the chain (no post-chain child starts); no receipt; claim and attempt remain; dispatch stays off the main thread; the session registry is drained |
| `…an_exception_in_the_worker_publishes_no_receipt` | an exception in the worker (injected into the final closure) gives rc 1 with its message and no receipt |

**Evidence** (`/home/yschoi/anchor_rt_session_state/r9_prep/`, inside the reviewed OS sandbox runner
`a684299c…`):

| When | Run | Result |
|---|---|---|
| before the fix, uncommitted | `dev_before_fix_20261007T062153Z` | 3 failed: `managed campaign children must launch from a worker thread` |
| before the fix, at `61f62e1` | `before_fix_61f62e1` | 3 failed (7 occurrences of the guard message); tree clean at `61f62e1` |
| after the fix, uncommitted | first attempt | refused at the sandbox probe: a syntax error in my first edit (an f-string spanning lines), corrected before any test ran |
| after the fix, uncommitted | `dev_after_fix_20261007T062341Z` | **102 passed** (99 existing + 3 new) |
| at `f169b8b` | `suite_f169b8b` | all 28 files, `--fake-nvidia-smi`, `-ra`. Result: see §5 |

## 4. Contract proposal for any future attempt (not executable; §839.3)

1. **Bind both histories.** A new request must pin:
   - the original stopped campaign (ancT9, its lineage `b7676019…`, settlement `31691514…`);
   - the failed recovery: run `20261007T061117Z-12e14ceb`, recovery ledger `593d327f…`, claim
     `7de24931…` (key `dc4a8bc0…`), attempt `1add29fb…`, reservation, snapshot `07708b36…` (bytes);
   - its **no-entry / no-record / no-receipt** evidence, by lstat.

   Neither the spent claim nor the failed namespace is reused, renamed or removed.
2. **New namespace and claim lineage.** A new namespace (proposed `ancT9s`) and a new claim key derived from
   the failed recovery's own identities: its run ID, attempt digest and claim digest, plus the cell. The
   key is stored in a claim root that refuses the consumed key.
3. **Accounting.** The new attempt's ledger binds both earlier charges:
   - parent 79,482.14623009507 s;
   - the failed recovery's 5.704555157572031 s.

   Under the unchanged ceiling 94,482.14623009507 s, at most about 14,994.2954448424 s remain. There is no
   fresh 15,000 s.
4. **Generation.** An r9 manifest whose transition from r8 is exactly the changed
   `scripts/anchor_refit_stage.py` and `tests/test_anchor_t_recovery.py`. It is built and tested at its own
   commit, with the new full suite and a mutation check of the worker dispatch (e.g. moving the cell back to
   the main thread must fail `…from_a_worker_thread`).
5. **No unreviewed smoke.** The startup boundary is covered by the synthetic regression above. No
   official-test smoke is proposed (§839).
6. **Consumer.** Future consumer authority follows only the actually accepted successful lineage. The failed
   ancT9r settlement is never forced through its success predicates.

## 5. Full-suite results, and the fixture fix they required (audits 840–843)

| Run (all 28 files, `--fake-nvidia-smi`, `-ra`) | Result |
|---|---|
| `suite_f169b8b` (clean `f169b8b`) | **rc 1: 3 failed, 1556 passed, 11 skipped.** The three new cases refused with `campaign shutdown has begun; refusing a new child process` |
| `order_unfixed_f169b8b` (sparse copy of `f169b8b`) | `test_signal_cleanup_kills_managed_process_group_before_lease_release`, then the three cases: **1 passed, 3 failed** with the same shutdown refusal. This is the causal order: the signal test leaves the launcher's process-wide `_CHILD_LAUNCH_BLOCKED` true |
| `10af739` (test only) | the fixture starts from a fresh launcher child-registry state (`_CHILD_LAUNCH_BLOCKED`, `_CAMPAIGN_LEASE_FDS`, `_ACTIVE_CHILDREN`), restored by monkeypatch, and asserts its registry drained at teardown. Two negative controls: a shutdown set within the test still refuses (no child starts, no receipt), and the main thread is still refused by the real function |
| `dev_order_fixed_20261007T070239Z` | the same order on the fixed fixture: **the signal test itself failed once** (`ValueError` at `tests/test_phase3_selection_matrix.py:1504`: its synthetic child's PID file can be read empty, because the test signals as soon as the file exists; host load about 11). The 5 real-dispatch cases passed |
| `dev_order_fixed_rerun_20261007T070326Z` | the same order again: **6 passed**. The signal test's race is pre-existing and was not changed here; both results are kept |
| **`suite_10af739`** (clean `10af739`) | **rc 0: 1561 passed (1556 + 5 new), 11 skipped (the same opt-in / Phase-2 reasons), 2 subtests passed.** Stage `15432e3c…`, launcher `d8092436…`, supervisor `4732a99e…`, all imported from inside the tree |

Production source is unchanged since `f169b8b`. Relative to S, the changes are
`scripts/anchor_refit_stage.py` (the worker dispatch) and `tests/test_anchor_t_recovery.py`. This
completes the worker-dispatch repair only. The exception contract of §4 is designed separately in
`docs/ANCHOR_T_RECOVERY_EXCEPTION_CONTRACT_v1_DRAFT.md` and is not yet implemented.
