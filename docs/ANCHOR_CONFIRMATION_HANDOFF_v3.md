# Anchor Confirmation — Handoff, generation v3 (response to audit §702–§706)

**Status: preparation package for audit review. No scientific execution.**
- No training, smoke, probe, refit or terminal evaluation ran. There was no full seal verification,
  GPU lease, namespace reservation or run-directory claim, and no real binary was deserialised.
- Every process in the new tests is a private sleeper. Every lock is a private flock under the
  pytest temporary directory.
- The v2 package stays as history, unchanged: handoff `92c152de…`, manifest `a5f7b843…`,
  request `ee367b84…`.

## 0. Acknowledgment

- Read in full before this submission: §702–§706. The ledger SHA256 at the start of this work was
  `0ef69c45…` (54,248 lines); the ledger at submission is recorded in §7.
- **This revision.** Branch `arch-exp-2026-09-anchor-confirm`, source commit `0ea88af47a375a9bf58cade194fe41adca65fc7d`.
  - Generation manifest v3: `artifacts/anchor_confirmation/authority_manifest_v3.json`, SHA256
    `9e54bda303d6c34f080e8026492ff7eec7bfe432a6e53bd34a5a6056881a9b83`, 59 files.
  - Contract `docs/ANCHOR_CONFIRMATION_CONTRACT_v2.md`, unchanged at `26ebe2c1…`.
  - Operational addendum `docs/ANCHOR_CONFIRMATION_OPS_ADDENDUM_v1.md`, SHA256 `b36c84ba1c8f6c4b62deac74498225c35bbb238e39c5b108352fc40b55be5745`.
  - Stage-S request SHA256 `0c940824a6f598d733439486cd2eab9bf6966f156edf48589cab8801a17d781f`.
- **Unchanged.** The accepted S/D scientific design (audit §702.2), the approval-line format, the
  seals, the coordinates and the request's scientific content. The request digest changes only
  because the manifest digest inside it changed.

## 1. §705/§706 — the lifecycle repair (focused diff in `scripts/phase3_selection_matrix.py`)

**Before.** At `1badc49…`, `_run_managed_process` forgot a child as soon as its leader was reaped.
`_terminate_active_children` also built its TERM→KILL set from `Popen.poll()` on leaders. A
descendant therefore escaped in two ways: when it ignored TERM while its leader died, and when its
leader exited first. `_with_campaign_gpu_leases` then called `GpuLeaseSet.release`, whose
`LOCK_UN` unlocks the open file description that the escaped descendant still shared (audit §706).

**Now:**
1. **Ownership exact under PID reuse.**
   - A child is still started with `start_new_session=True`, so its PID is also its group and
     session ID. None of the four trainer shells calls `setsid`, and DataLoader workers fork
     inside the session.
   - The worker waits for the leader with `waitid(WEXITED | WNOWAIT)`, which does NOT reap it.
   - The kernel frees a PID only when no task uses it as a PID, group or session ID. While the
     leader is an unreaped zombie, no process outside its session can carry that number.
   - The leader is reaped, and the record forgotten, only after no live process remains in the
     session (`_owned_session_drained`: two `/proc` scans; a zombie counts as live only while one
     of its threads still runs).
   - A reaped record is never signalled.
2. **Escalation and completion based on live members, not on `poll()` or the registry.**
   - `_stop_owned_sessions` sends TERM to each owned session. It uses `killpg` on the pinned group,
     plus a pidfd-verified signal to any member that moved to another group of the same session.
   - It then waits a bounded grace of 5 s and re-sends SIGKILL for up to 30 s.
   - It returns the sessions that are still live. A timeout is reported, never taken as death.
   - `_terminate_active_children` raises if any session survives.
3. **The stream's own path.** When a leader exits and its session does not drain within 5 s, the
   stream stops the session itself (bounded) and refuses the cell.
4. **Leases.**
   - `_with_campaign_gpu_leases` releases only after cleanup returns AND a final check at the call
     site (`_owned_live_sessions()`) finds no live owned session.
   - Otherwise the leases stay held, launches stay blocked and the error propagates.
   - The interpreter-exit `release` that `acquire_gpu_leases` registers is unregistered, so an exit
     path cannot unlock the leases either.
   - Launches also stay blocked after a campaign ends, so a stream thread that outlives a signal
     cannot start a child once the leases are gone.
   - The launch-blocking race guard (main-thread refusal; registration under the lock) is
     unchanged.
5. **Bounds.** The worst-case cleanup is 5 s + 30 s plus `/proc` scans of about 20 ms each.

**Regression tests: `tests/test_anchor_confirm_lifecycle.py` (11).**
- Real private sleeper trees and an unrelated control sleeper in its own session, which survives
  every case.
- The three §705 cases are covered: the cooperative tree as the positive control, a TERM-resistant
  descendant, and a leader that exits first. Each asserts the descendant is dead when cleanup
  returns, and that termination is bounded.
- The stream's own stop and refusal.
- The PID pin is observed directly: the leader stays `Z` while the descendant lives.
- A reaped record receives no signal.
- The composed lease tests use the exact `_with_campaign_gpu_leases` and the real
  `acquire_gpu_leases`/`GpuLeaseSet` on a private lock root. They cover a real `SIGTERM` for both
  trees, a leader that exits first with the callback returning normally, and a survivor:
  - the private lock is held inside the callback;
  - at the moment `release` runs, no owned process is live (read independently from `/proc`);
  - the lock is free again afterwards;
  - signal handlers are restored.
- Bounded escalation and the survivor case use one stated stand-in. A process that survives
  SIGKILL cannot be made on purpose, so the scanner reports a phantom member for that one session.
  The survivor keeps the kernel lock held, the exit handler unregistered and the error propagating.
- The existing `test_signal_cleanup_kills_managed_process_group_before_lease_release` and every
  admission test pass unchanged.

**Differential check against the pre-repair launcher.** The same test file ran in a detached
worktree at `63654dd` (launcher `1badc49…`), with the result for each test declared in advance:
- the positive controls pass (cooperative tree, launch block, composed cooperative SIGTERM);
- the five escape cases fail exactly at their declared assertion: `assert not live(child)` three
  times, and `lease.at_release[0]["owned_live"] == []` twice.
- Result: **8/8 as declared**. The worktree was unchanged and no fixture process remained.
- The three tests that use the new API alone have no pre-repair counterpart.
- A first attempt failed during setup (pytest's `--basetemp` parent was missing) and counted zero;
  it is kept.

## 2. §703/§704 — the operational addendum

In `docs/ANCHOR_CONFIRMATION_OPS_ADDENDUM_v1.md` (versioned `anchor-confirm-ops/1`):
- the frozen environment and full-verification scope;
- the storage estimate and rules;
- the device-budget accounting and ledger;
- the wall-time limits and the stop mechanism;
- the exact commands;
- which component supplies each guarantee.

Two components:
- **Launcher, before every dispatch** (`anchor_dispatch_space_refusal`, anchor mode only). After
  full verification, a cell is refused unless the result filesystem keeps 10 GiB plus 0.75 GiB for
  every unfinished cell. Its stream then stops and no receipt is written.
- **External supervisor** (`scripts/anchor_confirm_supervisor.py`, a manifest member, standard
  library only).
  - It polls every second for live space, the budget and the wall time.
  - It keeps the S + D + probe ledger.
  - It stops the run with ONE SIGTERM to the launcher, so the repaired handler above does the
    cleanup.
  - It never signals anything else and never infers death from a timeout.
  - After the launcher exits, it checks for orphaned live attempts and held leases.

**Tests: `tests/test_anchor_confirm_supervisor.py` (12) and four launcher tests.**
- The composed cases run the REAL lease wrapper, managed child and signal handler in a synthetic
  launcher process, with a stand-in GPU inventory and a private lease root.
- The cases cover:
  - space refusal before start;
  - a space breach while a TERM-resistant child lives;
  - a budget breach while a child lives;
  - a failed, partial attempt carried into the next stage's budget;
  - a failed observation stopping dispatch;
  - a command slower than the stop bound, which is waited for and not declared dead;
  - an operator signal becoming one SIGTERM;
  - a launcher that leaves a live attempt, which becomes `unclean`, blocks the ledger and is not
    signalled;
  - an unfinished prior run and a second supervisor, both refused;
  - the generation self-check;
  - the CLI's GPU count;
  - the rule constants shared with the launcher.
- The unrelated control survives every case.
- Launcher tests: no dispatch below the floor; a mid-sweep breach stops later dispatch; the first
  dispatch reserves space for all twelve unfinished cells; the rule's arithmetic.

## 3. Evidence (CPU only)

All at source commit `0ea88af`, CPU only (`CUDA_VISIBLE_DEVICES=` empty,
`GDNA_NUM_SEMANTIC_PARTS=5`, `python -m pytest -q -p no:cacheprovider <files>`). One run of all 14
files: **800 passed, 1 skipped, 333 s, rc 0**, on a clean tree
(`artifacts/anchor_confirmation/mutation_v3/suite_0ea88af.log`).

| Suite | Tests |
|---|---:|
| `tests/test_anchor_confirm_port.py` | 16 |
| `tests/test_anchor_confirm_recipe.py` | 65 |
| `tests/test_anchor_confirm_launcher.py` | 164 (4 new storage-rule tests; the one opt-in real-artifact test is skipped) |
| `tests/test_anchor_confirm_reducer.py` | 169 |
| `tests/test_anchor_confirm_lifecycle.py` (new) | 11 |
| `tests/test_anchor_confirm_supervisor.py` (new) | 12 |
| legacy: `test_phase3_selection_matrix`, `test_phase3_select_n`, `test_result_identity`, `test_seal_phase3_inputs`, `test_phase3_clip_snapshot` | 341 |
| related: `test_extraction_manifest_integration`, `test_phase3_launcher_e2e`, `test_siglip2_criterion_runtime` | 23 |

**Differential check against the pre-repair launcher.** Result: 8/8 as declared (§1; report
`e5957dd0…`, harness `differential_705.py`). The first attempt, with the `--basetemp` parent
missing, failed during setup and counted zero; it is kept as `differential_705_attempt1_setup_failure/`.

**Mutation battery v5.**
- **Setup.**
  - Sandbox: a detached worktree at `0ea88af`, with its tracked bytes checked equal to the anchor
    worktree's.
  - All 20 declared tests passed unmutated first.
  - Each mutant is one exact replacement that disables a whole condition or call.
  - Each mutant runs only its declared tests, each alone. It counts only on pytest rc 1 with the
    marker declared in advance.
  - The anchor worktree's tracked bytes and `git status` were the same before and after, and no
    fixture process remained.
- **Mutants.**
  - Lifecycle:
    - ML1: the record is forgotten when the leader exits;
    - ML2: a session counts as drained once its leader exited;
    - ML3: no SIGKILL escalation;
    - ML4: a reaped record is signalled;
    - ML5: survivors are not reported;
    - ML6: the lease is released despite survivors;
    - ML7: the interpreter-exit release stays registered;
    - ML8: the stream does not stop a leftover.
  - Storage rule: ML9, no refusal; ML10, only this cell is counted.
  - Supervisor: MS1–MS11 (listed in the addendum §7).
- **Result: 20 of 21 detected as declared in the first run.**
  - The exception is ML7. It failed at the intended assertion: `exit_release_pending()` still
    listed the lease's `release`. But the marker I had declared left out the assertion's
    `lease.held and` prefix, so the automatic check did not count it.
  - v5b reran ML7 alone with the corrected marker: **detected 1/1**.
  - Reports: v5 `366631ae…`, v5b `b0c86f21…`. Harness digests: v5 `76d7e4b3…`, v5b `bb73ae50…`.
- All outputs are in `artifacts/anchor_confirmation/mutation_v3/`.

**Stated coverage limits:**
- The interpreter-exit property is tested by recording `atexit.register`/`unregister` in the test
  process; a real interpreter exit is not executed.
- A process that survives SIGKILL is a phantom stand-in in the scanner.
- `setsid` escapes are out of scope (§1).
- The supervisor's lease observation is tested on the real private lock under a private root, not
  the host-global root.
- The real trainers, GPUs, seals and a real full verification were never run.

## 4. Which prior evidence remains applicable

The diff from `491c3ce` touches:
- `scripts/phase3_selection_matrix.py`: the lifecycle block, the storage rule and its call in the
  anchor stream, and the closure list;
- `scripts/anchor_confirm_manifest.py`: generation v3 and its predecessor;
- tests: two new files and four launcher tests;
- `scripts/anchor_confirm_supervisor.py`: new.

The trainer, `config.py`, `model_siglip2.py`, `dna_utils/*`, the recipe module, the wrappers, the
reducer, the probe and the contract are byte-identical to the v2 generation.

Still applicable:
- the recipe, override-policy, admission, request/approval, reducer, probe-envelope and
  input-consistency evidence, with the audit's §695/§700/§701/§702 checks, for those unchanged
  files;
- the v3/v4/v4b mutation results for guards in unchanged files, and for launcher guards outside
  the changed block. Their source text is identical; only the file digest moved.

Not carried over:
- anything about process cleanup or lease release. The earlier cooperative test and audit §704's
  six local checks covered only what they covered (audit §705.2);
- the v2 manifest and request digests, which this generation replaces.

## 5. Revision of the v2 claim on the probe's late boundaries (audit §702.3)

The v2 handoff said the probe's post-load runtime-row check is "reachable only with real data" and
treated its pre-publication re-verification the same way. That was too broad.
- My external harness did not cover these two caller boundaries, but private caller tests with
  stand-ins can reach them. Audit §702.3 exercised eight such cases: drift before publication and
  a row-checker refusal propagated as `NotReducible`.
- What needs real data is only the real row verifier and a complete probe measurement.
- No source change is made for this, as §702.3 allows.

## 6. Observation disclosed: the exploratory runs that motivated the anchors did not use the approved recipe

While assembling a comparison for the user, I read the `args.txt` of every September 2026 run
directory under `gdna_archexp_result/` and `gdna_wt_mscoco/result/`:
- 214 of them, covering stages 6–12, both arms (`axis_center` = `none`/`anchors`/`readout`) and the
  architecture sweeps, record `use_gumbel_softmax=True`;
- the 257 approved `p3exec` directories record `False`.

The stage-6 cell commands carry no Gumbel option; their options are sorted as that builder emits
them. The cause is in the off-protocol command builder `scripts/build_offprotocol_cmd.py`:
- its option map keeps the first action for a destination, `--use_gumbel_softmax` (store_true,
  default True, `config.py:209-214`);
- a recorded `False` therefore emits nothing, and the default True applies.

The same defect class was recorded for the D5 20-base runs (audit §606/§609).

**Consequences:**
- The confirmation campaign is unaffected. It renders the approved recipe: `S5_FLAGS` carries
  `--no_gumbel_softmax`, and the typed admission requires `use_gumbel_softmax=False`.
- The exploratory anchor-versus-base gaps were measured with Gumbel ON in both arms. They say
  nothing direct about the approved recipe. That is one more reason the confirmation must decide.
- No scientific source or design change is proposed here.

## 7. Requested decision and scope

- Review the v3 generation (the §705/§706 repair) together with the operational addendum, the
  manifest digest and the exact stage-S request digest above.
- A stage-S approval, if given, is a ledger line naming the manifest `9e54bda303d6c34f080e8026492ff7eec7bfe432a6e53bd34a5a6056881a9b83` and the request
  `0c940824a6f598d733439486cd2eab9bf6966f156edf48589cab8801a17d781f`. Choosing a smoke first, a carried authority or another GPU count would be a
  different request.
- Full verification, leases, smoke, training and every downstream step stay closed until then.
- Ledger at submission: SHA256 `0ef69c4547b8d2fc4a45131e3dd23dfd2ddfc1bf1a26796ba8639cdcc9779936`, 54248 lines, last section §706 — unchanged since the start of this work.

## 8. Real files touched in this revision (nothing deserialised, nothing written outside the worktree and the scratchpad)

- **Package commands at `0ea88af`.**
  - `anchor_confirm_manifest.py inventory` read the two pinned JSON authorities, the `p3lamA`
    receipt bytes and the 59 closure files, and wrote the manifest once.
  - The stage-S `--plan` and the request preview (15:55:24–15:55:41 UTC, issue times) do the same
    as in v2 §4:
    - read the manifest and the closure;
    - stat the three whitening files;
    - render the three pinned wrappers 24 times under bash, with a capture interpreter, in
      temporary directories;
    - for the preview, read and hash the three stage-1 seal JSON files.
  - No result root, record, reservation or lease was created: the result root is absent, and the
    newest file in the host lease directory is from 2026-09-21.
- **Storage estimate.** `du -sb` on three historical `p3gE` N39 seed-42 run directories (stat
  only; no file opened) and `df` of `/data`.
- **Comparison and Gumbel check, text reads only:**
  - `args.txt` of 471 September run directories;
  - the stage-6 `cells_all.txt` and its tmux log;
  - `config.py`;
  - `scripts/build_offprotocol_cmd.py`;
  - the pinned `p3rfB` aggregate JSON;
  - the baseline matrix JSON;
  - the D5 tables;
  - the stage-6/12 summary JSONs;
  - the last row of three `log.csv` files.
- **Tests and batteries.** Only private sleeper processes, private flocks under temporary
  directories, and detached sandbox worktrees, which were removed afterwards.
