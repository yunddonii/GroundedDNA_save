# Anchor Confirmation — Handoff, generation v4 (response to audit §707–§708)

**Status: preparation package for audit review. No scientific execution.**
- No training, smoke, probe or refit ran. There was no full seal verification, GPU lease,
  namespace reservation or run-directory claim, and no real binary was deserialised.
- Every process in the new tests is a private synthetic process, and every lock is a private flock
  in a temporary directory.
- The v3 package stays as history, unchanged: handoff `edc3f32d…`, addendum `b36c84ba…`, manifest
  `9e54bda3…`, request `0c940824…`.

## 0. Acknowledgment

- Read in full before this revision: §707 and §708. The ledger was SHA256 `7469faac…`, 54,487
  lines. §707.3's delivery note is acknowledged and not asked for again.
- **This revision.** Branch `arch-exp-2026-09-anchor-confirm`, source commit `a153c1577073edc0f6a97a41617b9bb6c69e69e5`.
  - Generation manifest v4: `artifacts/anchor_confirmation/authority_manifest_v4.json`, SHA256
    `5a4481f4898fda3df27250e0214e204495022a91546d7d78e2e1c2749e17647b`, 59 files.
  - Operational addendum `docs/ANCHOR_CONFIRMATION_OPS_ADDENDUM_v2.md` (`anchor-confirm-ops/2`),
    SHA256 `6bdd2460735ae709f9ac2cc6aa97bb7fab53e6fda3397f10d40b9e101a0fa5d3`.
  - Stage-S request SHA256 `033b697945067d587473c81551f05e145d124ea9f69914549fd7199d333eec54`.
- **Unchanged.**
  - The launcher `253deb4f…`, including the lifecycle repair that §707.2 accepted in scope.
  - The S/D scientific design, the contract (`26ebe2c1…`), the approval format, the seals and the
    request's content. The request differs from v3's only in the manifest field (verified by diff,
    §3).
- **What changed.** Only `scripts/anchor_confirm_supervisor.py`, its test file and the manifest
  builder (generation label and predecessor).

## 1. §707.1 — the budget accounting

**The defect as found.**
- The v3 allowance was planned cells × the nominal poll interval. Observation work and scheduling
  are not bounded by that interval, so an attempt could start, run and be reaped while one
  observation was delayed.
- The final record still called the charge an upper bound, and the charge became the next stage's
  prior.
- My own regression reproduces the audit's counterexample on the v3 supervisor `f634fc55…`: the
  child's own boot-clock lifetime was 1.502 s, and v3 charged 0.1 s (§3).

**The repair (supervisor only):**
1. **Observation windows.**
   - A window runs from the START of one scan to the END of the next, on the boot clock. Every
     stat, scan, ledger write or scheduling delay is therefore inside it.
   - An attempt that no scan saw lived inside one window.
   - The window between the last scan and the moment the exit is seen is measured too.
2. **Allowance = planned cells × the longest window observed.** It is recomputed at every poll, so
   the live budget projection uses it, not only the final record.
   - Its premise is that the pinned launcher runs one managed child per stage-1 cell and never
     retries.
   - If more attempts are observed than cells were planned, the premise is void and the run
     becomes unresolved.
3. **Watchdog.**
   - A separate thread stops the command when no observation has completed for 10 s.
   - Reaping and signalling share one lock, and only one SIGTERM is ever sent.
   - The budget headroom is GPUs × (10-s watchdog + 120-s stop bound).
4. **Continuity.** The run is stopped and its final record is `unresolved` when any of these
   happens:
   - a window longer than the watchdog bound;
   - a watchdog stop;
   - a failed observation or ledger write;
   - more attempts than cells.

   The record keeps the observed charge and adds `pessimistic_bound_seconds` (GPUs × wall time)
   for reconciliation. `prior_device_seconds` refuses every later stage until the audit reconciles
   it, as it already did for `unclean` runs and runs without a `final` record.

**Regression tests (`tests/test_anchor_confirm_supervisor.py`, 19 tests; 7 added or changed).**
- **An entire attempt between two observations.** A `statvfs` stand-in releases a synthetic
  launcher right after the first scan and then blocks for 2.5 s. The launcher runs one session-
  leader child (1.5 s), reaps it and exits during the block.
  - Independent evidence: the child stamps its own boot-clock start and end.
  - Result: no scan saw the attempt, the charge is ≥ the child's own lifetime, and the longest
    window is ≥ 2.5 s.
  - The same test with a 1-s watchdog: the command is stopped during the stall, the run is
    `unresolved`, and the next stage is refused.
- **Normal-observation control.** The same child, not blocked: observed as one attempt, charged at
  least its own lifetime, windows under 1 s.
- **The watchdog signals during the stall.** The command stamps the moment SIGTERM arrives. The
  stamp lies after the 1-s bound and before the blocked observation returns.
- **More attempts than cells:** `unresolved`.
- **Live budget:** one 2.5-s window makes the live projection cross the budget, and the run is
  stopped.
- **Headroom:** just below GPUs × (watchdog + stop bound) the start is refused; above it, it runs.
- **Updated test:** a failed observation now makes the run `unresolved` and refuses the next
  stage.

**Differential check against the v3 supervisor.** The same test file ran in a detached worktree at
`0ea88af`, with results declared in advance: the positive control passes; the between-observations
test fails at `charged_seconds >= lived` (v3 charged 0.1 against 1.502); the headroom test fails at
its refusal assertion. Result: **3/3 as declared**, with the worktree unchanged and no fixture
process left. The five tests that use the new behaviour have no v3 counterpart.

**Stated limits:**
- The allowance premise (one managed child per cell, no retries) is a property of the pinned
  launcher. Observed excess voids the settlement; an unobserved excess cannot be seen.
- A supervisor process that is frozen as a whole (all threads stopped) cannot stop anything while
  frozen; its long window is recorded on resume and voids the settlement.
- The window check duplicates the watchdog for that frozen case. It is not separately mutated,
  because no test can freeze the supervisor without also freezing the watchdog.

## 2. §708 — the Gumbel inventory, labels and a wording correction

- **Correction.** Handoff v3 §6 named two roots for the 214 exploratory records. They span four:
  `gdna_archexp_result` 99, `gdna_wt_mscoco/result` 70, `gdna_wt_arms/result` 31 and
  `gdna_wt_arms2/result` 14 (§708.2). The v3 PROJECT_LOG entry now carries a dated correction.
- **Labels (§708.4).** The main-tree PROJECT_LOG has:
  - a 2026-09-27 label entry;
  - a one-line recipe label under each of the 19 affected entries (18 branch arch-exp-2026-09
    entries dated 09-19 to 09-25, plus the 09-20 direct-launch entry).

  Archived args, results and summaries are untouched. The old anchor gains and the CIFAR negative
  finding are presented as Gumbel-ON exploratory results, never as confirmatory ones.
- **Builder.** `scripts/build_offprotocol_cmd.py` is not on the confirmation path. It is recorded
  as needing the paired-Boolean repair and a requested-to-parsed round-trip test before any new use.
  Neither is done here, because §708.4 makes it a precondition of new use, not of stage S.
- **Two observations from args.txt text** (no deserialisation):
  - (a) args.txt pads every line to 100 characters with dashes, so a negative value's minus sign
    merges into the padding. `codebook_freeze_after_epoch` therefore reads `1` in the exploratory
    records but is `-1`: it is the parser default, and the exact stage-6 command parsed with that
    worktree's own parser gives `-1`. Any args.txt comparison must allow for this; my first diff
    did not.
  - (b) The 09-20 gate runs (direct launches of the `p3lamA` LBU0 cell rebuilt with the builder,
    logged as "cause not identified") differ from that incumbent cell in `use_gumbel_softmax`
    (True vs False) and in `lambda_bu` (0.02 vs 0.0), besides bookkeeping fields. The current
    builder does emit `--lambda_bu 0.0` for that args.txt, so the 09-20 loss of `lambda_bu` came
    from how those cells were built, which is not reconstructed here. Neither difference was
    tested as the cause of the epoch-1 divergence.
  - Among fields both record, the stage-6 base arm and the approved `p3gE` stage-1 cells differ
    only in Gumbel. Every other difference is an option added after the approved code, at its
    recorded value; those values were not verified to be inert.

## 3. Evidence (CPU only)

All at source commit `a153c15`, CPU only (`python -m pytest -q -p no:cacheprovider <files>`,
`CUDA_VISIBLE_DEVICES=` empty, `GDNA_NUM_SEMANTIC_PARTS=5`). One run of the same 14 files as v3:
**807 passed, 1 skipped, 344 s, rc 0**, on a clean tree
(`artifacts/anchor_confirmation/mutation_v4/suite_a153c15.log`). The supervisor file now has 19
tests; every other file is unchanged.

- **Differential check against the v3 supervisor.** `differential_707.py`, report `0a87f75b…`:
  3/3 as declared (§1).
- **Mutation battery v6.** Harness `782db05d…`, report `0495984d…`.
  - Sandbox: a detached worktree at `a153c15`, with its tracked bytes checked equal.
  - All seven declared tests passed unmutated first.
  - Each mutant is one exact replacement and runs only its declared tests. It counts only on
    pytest rc 1 with the marker declared in advance.
  - The anchor worktree's tracked bytes and `git status` were the same before and after, and no
    fixture process remained.
  - **9/9 detected as declared**, MV1–MV9: the list is in the addendum §7.
  - Not mutated, and stated in §1: the window-beyond-watchdog flag, which duplicates the watchdog.
- **Request.** The v4 request (`request_preview_ancS2_v4.txt`) was recomputed from its printed
  canonical JSON (`033b6979…`). It differs from v3's only in `manifest`.
- **Plan.** The v4 plan differs from v3's only in the manifest commit, the three changed file
  digests and the manifest path. All 24 cell renders are identical.

## 4. Which prior evidence remains applicable

- **Launcher.** Its bytes are unchanged, so the following still apply to this generation:
  - the lifecycle repair and its tests;
  - §707.2's scoped acceptance;
  - the storage rule;
  - the v5/v5b mutation results for ML1–ML10.
- **Supervisor.**
  - v3's accounting evidence is superseded by §1.
  - Its other behaviours are re-tested in the same 19-test file: space refusal, the stop through
    the launcher's handler, the operator signal, `unclean` after orphans or held leases, ledger
    refusals, the self pin and the CLI.
  - v5's MS1–MS11 mutants were run against `f634fc55…`. The guards they disabled still exist in
    the same form, but only the v6 mutants were run against these bytes.
- **Everything else** (recipe, trainer, reducer, probe, contract, wrappers) is byte-identical. The
  earlier evidence and the audit's §695–§708 checks apply unchanged.

## 5. Requested decision and scope

- Review the v4 generation, addendum v2 and the stage-S request. A stage-S approval, if given, is a
  ledger line naming manifest `5a4481f4898fda3df27250e0214e204495022a91546d7d78e2e1c2749e17647b` and request `033b697945067d587473c81551f05e145d124ea9f69914549fd7199d333eec54`.
- Full verification, leases, smoke, training and every downstream step stay closed until then.
- Ledger at submission: SHA256 `7469faac15795664c1c0ee0747d3d12147f8a2ca6239d1ea22ad68d9b151a927`, 54487 lines, last section §708.

## 6. Real files touched in this revision (nothing deserialised)

- Package commands at the source commit:
  - the manifest inventory: the two pinned JSON authorities, the `p3lamA` receipt bytes and the 59
    closure files;
  - the stage-S `--plan` and the request preview, as in v3 §8.
- The Gumbel and recipe checks read text only:
  - `args.txt` of the stage-6 runs, the 09-20 gate run and its incumbent;
  - the stage-6 cell file and runner;
  - `config.py`, `train_siglip2.py` and `model_siglip2.py` of the exploratory worktree, and their
    git history;
  - the parse of one stage-6 command with that worktree's parser (argument parsing only, no model);
  - a dry run of `build_offprotocol_cmd.py` that printed a command and trained nothing.
- One mistake in my own test fixture:
  - A synthetic launcher's SIGTERM handler deadlocked in `Popen.wait()`. The supervisor under test
    correctly kept waiting, so the test run hung.
  - I stopped only that fixture, its pytest process and one orphaned control sleeper, by PID after
    reading their command lines.
  - The fixture now reaps by PID and carries a 60-s self-destruct alarm.
