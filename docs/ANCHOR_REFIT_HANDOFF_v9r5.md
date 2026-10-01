# Anchor model — generation v9 revision 5: the coherent stage-R/T successor (audits §743–§756)

**PREPARATION ONLY. Nothing has been executed.**
- No refit, smoke, test access or GPU work.
- No real model, config or checkpoint was loaded.
- No seal payload or scientific array was read.

This package supersedes the submitted v9 r2 (`f51e89c`, manifest `b7e9264a…`), which audit §753 did
not accept. Revisions r3 (`a97671e2…`) and r4 (`0deec033…`) were never submitted. They are kept as
preparation history (§3), with every earlier log.

## 1. What is submitted

| Item | Value |
|---|---|
| Worktree, branch | `/data/yschoi/gdna_anchor_refit_v9`, `arch-exp-2026-09-anchor-refit` (from v8 `3dd1c02`; pushed) |
| Source commit | `ffdf006`; manifest commit `586187d` |
| Manifest v9 r5 | `artifacts/anchor_confirmation/authority_manifest_v9r5.json` `d0de9fb30ab58bfd178b4cab91be933e05e8ed2baa53c0cea8f165dc2935f2da` (69 files) |
| Contract | `docs/ANCHOR_REFIT_CONTRACT_v1.md` `f736edf6…` (closure member; updated for §746–§756) |
| R smoke request (full admission) | `7c62f2db19f9f408999c16bf9f0e6623c6c486dfdc7a51a1a2addf9464d7e4f2` |
| R run request (full admission; preview) | `206532395c74716c3b52509c85d38aafe84ab5e3f6d2b2d89fd179c1c9253c0e` |
| F authority | `ancF_candidate_v1.json` `5165f5dc…` + §744 exact text `13ef776b…` (both bound in every request) |
| Evidence | `artifacts/anchor_confirmation/refit_v9/r5_evidence/` (this revision; r3/r4 under `interim_r3/`, `interim_r4/`) |

Source and tests outside `artifacts/` changed by 3,893 lines added and 160 removed relative to v8 `3dd1c02`, across 16 files.

**r5 against r4 (`git diff ecacf90 586187d`):** two changes, neither in code that R or T runs.
- **Test:** the train-extraction fixture's stand-in legacy resume now sets the arguments as the
  real resume does (§2, §756 item).
- **Manifest generator:** `scripts/anchor_confirm_manifest.py` changes only its `revision`, its
  `predecessor` entry and its default output name.

The battery and suite-pair scripts point at the r5 manifest. Every other source file is
byte-identical to r4 `ecacf90`.

## 2. Each finding, its source change and its evidence

Each item below gives the source change (file and function), then the composed tests
(`tests/test_anchor_refit_stage.py`) and the battery mutants that must kill it.

**§746.1 / §748.2-1 — T child replay**
- **Source:** `scripts/anchor_terminal_test.py` `claim_entry`. After admission, before any config, model
  or test access, it creates `<ns>_entry_<R tag>.json` exclusively (`O_EXCL`, 0444, fsync). The file
  is never removed.
- **Tests:**
  - `the_same_attempt_never_enters_twice_after_a_before_output_failure` (the real entry twice, same attempt);
  - `concurrent_entries_with_the_same_attempt_reach_the_test_once` (4 threads → one test access);
  - every refusal before the claim leaves no claim.
- **Mutant:** RX25.

**§746.2 / §754.2 / §748.2-2 — saved config and terminal identity**
- **Source:** `anchor_terminal_test.admit`, `verified_configuration`, `check_config`.
  - Admission binds checkpoint bytes, runtime witness at the epoch the request allows (N, or the smoke's
    last), trainer campaign evidence (bytes, cell, sealed recipe, the witness's own) and config bytes.
  - **config.pt is read once.** The bytes are hashed, deserialized from that buffer and checked against
    every typed field of the cell's sealed recipe (from the R snapshot the receipt binds) and the
    cell's campaign binding.
  - The arguments are built from **that object** by the new shared `extraction_siglip2._apply_saved_config`
    (the resume helper's own flat step). The effective scientific fields are re-checked, then
    anchors, refit mode, cell and run directory.
- **Tests:**
  - config-byte replacement before admission;
  - a change after admission but before the read (refuses before `torch.load`);
  - typed drift in lambda and in a non-summary field (`codebook_size`);
  - a cross-cell binding; wrong epoch and checkpoint; evidence bytes;
  - a change after the read, restored → the verified 0.15 reaches the test, one load;
  - a change after the read, left changed → refused before the test;
  - effective-argument drift (axis, lambda, run directory).
- **Mutants:** RX26–RX29, RX37.

**§746.3 / §748.2-3 — exact F acceptance**
- **Source:** `anchor_refit_stage.anchor_freeze_acceptance`.
  - Ledger §744 from its heading to the next heading, stripped, must hash to `13ef776b…` (the §745
    digest) and name the record's path and digest.
  - `acceptance_sha256` is bound in the R and T requests.
  - The F bytes are unchanged and the mutable ledger is not pinned.
- **Tests:** `f_record_without_its_acceptance_refuses` [changed-language, token-only, no-acceptance,
  no-section]; a corrupt F record.
- **Mutant:** RX8, which removes only the digest test so the substring checks remain.

**§747.1 / §748.2-4 — T accounting**
- **Source:** `anchor_confirm_supervisor.CHILDREN_PER_CELL` (T = 5, tied to `T_CHAIN`).
  - The attempt ceiling and missed-attempt allowance use cells × children.
  - `--planned-cells` stays the logical count (membership, storage).
  - The ledger start record states `planned_cells`, `children_per_cell` and `planned_attempts`.
- **Tests:**
  - 1 and 12 T cells (5/60 producers) exit cleanly; a sixth producer is excess;
  - each failing producer ends the command; an R control allows 1;
  - **under stage T, with the real lease wrapper and managed child:** space-breach stop with
    cleanup, leases released and no orphan (TERM-resistant child included); a producer living between
    two observations is charged (allowance ≥ 5 × window); a watchdog stall is unresolved.
- **Mutants:** RX22, RX30.

**§747.2 / §748.2-5 — rechecks between producers**
- **Source:** `anchor_refit_stage.terminal_test_boundary` builds the T snapshot.
  - It takes the stage-R sources, inputs and seals, recomputed and required unchanged since R, with
    T's own environment and leased GPUs.
  - The launcher's unchanged `verify_snapshot` runs before the entry, at the entry-to-train
    transition (`run_terminal_test_cell`) and after each post-chain producer (`_run_refit_postprocess`
    with the snapshot).
- **Tests:**
  - drift at each of 6 positions → the next producer never starts;
  - the real `verify_snapshot` with environment drift after train extraction stops BIO;
  - sources changed since R refuse.
- **Mutants:** RX31–RX33.

**§756 — the per-cell R artifacts**
- **Source:** `anchor_refit_stage.check_cell_inputs` runs at every T boundary (before the entry, at
  the transition, after each producer via the new optional `boundary_check`). Its scope is the
  consumed cell.
  - **The producers receive the three pins** (`cell_input_env`).
  - `scripts/extract_train_split.py` (the producer that loads the config and model) verifies the
    pins **before any deserialization**: config read once into `_apply_saved_config`, weights loaded
    from the verified checkpoint bytes, the witness pinned. It re-checks all three before it writes.
    Without pins it is unchanged.
  - The T entry re-checks the files immediately before and after the test.
- **Tests:**
  - config / checkpoint / witness changed after the entry, after train extraction, after BIO and
    after NMI → the next producer never starts (12 cases);
  - a changed artifact before the entry → no attempt;
  - the producers receive the pins;
  - train extraction through its `main()`: verified bytes and one load; refusal before
    deserializing for each artifact; no write after drift during encoding; partial pins;
  - the legacy path without pins (r5): completes with one resume, no byte load and the checkpoint
    loaded by its path.
- **Mutants:** RX34–RX36, RX38.

**§755 / Current Handoff — shared-resume regressions**
- **Source:** `extraction_siglip2._resume_args_flat_or_legacy` now calls `_apply_saved_config` for its
  flat step.
- **Tests:**
  - `the_flat_resume_is_the_v8_resume`: the current function and v8's (`git show 3dd1c02`) give
    identical arguments on the same `config.pt` and CLI (no flags, `--inference_epoch`,
    `--selection_mode`);
  - the AST of the new step equals v8's flat-branch statements, and the nested branch is unchanged.

Earlier repairs stay as accepted at their scopes: provisional/bound recipe admission (§748, §752),
the legacy predicate (§748), the attempt reservation and R/T scopes, and the raw-evaluation producer.

## 3. Evidence at the submitted commit `586187d`

| Check | Result |
|---|---|
| 19-file suite, unguarded (`refit_v9/suite_pair_v3.sh`) | **1302 passed, 4 skipped**, rc 0 (579.9 s) |
| 19-file suite under the open() guard | **1302 passed, 4 skipped**, rc 0, **0 refused opens**; tree clean before and after (`r5_evidence/suitepair/`) |
| Mutation battery v13 revision d (38 mutants; `refit_v9/mutation_v13.py`) | **38/38 detected as declared**; every declared test failed at its declared assertion, 0 refused opens; inventory and checkout unchanged; harness refused 0 (`r5_evidence/battery13e/`) |
| Manifest generation under the allow-list guard | exit 0, 0 refused; 19 named JSON/source files + v9 source (`prep_manifest_v9r5/`) |
| R request renders under the allow-list guard | exit 0, 0 refused; 10 named JSON files, 245 source opens; real wrappers rendered in temporary directories (`r5_evidence/renders/`) |

**Interim attempts (kept, never relabelled as successor results):**
- **r1 `bdaba58`:** suite pair 1231/4; battery 22/24 (RX13 always-refusing, RX16 wrong marker).
- **r2 `9dc2229`:** suite pair 1232/4; battery 24/24. This is the submitted package that was not accepted.
- **r3 `0116eed`:** suite pair 1271/4 in both runs, 0 refused; battery 31/33.
  - RX25 failed at an earlier claim assertion.
  - RX27 was caught upstream by the admission's byte pin.
  - Both pairings were corrected in battery revision d.
- **r4 `b68c1f6`:** suite pair 1302/4 in both runs, 0 refused; battery 37/38.
  - **RX35 crash-killed** with an `AttributeError` (`'Config' object has no attribute
    'lambda_wasserstein'`). The train-extraction fixture's stand-in legacy resume set no arguments,
    so the mutated path stopped in the fixture instead of at the declared assertion.
  - The defect was in the test fixture, not in production source. r5 repairs it. Before
    committing, the RX35 mutation applied by hand fails at the declared assertion (`1 == 0`).

## 4. Limits stated plainly

- **T-path equivalence with the legacy refit is by construction, not by a real run.** The T entry
  calls the moved terminal function on arguments built from the verified saved config. The first
  real exercise is the T smoke.
- **The T entry's process environment** is the trainers' clean base. It is not re-attested against
  an R cell's expected child environment, because the GPU may differ. The T snapshot records the
  launcher's fingerprint of the leased GPUs.
- **The checkpoint as `extract_code` reads it.** Inside the T entry, `extract_code` loads the
  checkpoint by path. The entry verifies its bytes immediately before and after the official test;
  a swap that is restored between those checks is not excluded at that load.
- **The R run's carried-admission request** can be rendered only after the R smoke snapshot exists.
- **The budget** (80,000 device-seconds, own ledger) and the wall limits are proposals.

## 5. Proposed next step (each a separate approval)

1. **`stage-R-smoke`** for request `7c62f2db…`: manifest `d0de9fb3…`, freeze `5165f5dc…`.
   - One cell: Flickr25K N4 seed 42, one epoch.
   - Namespace `ancRsmk9`, tmux `ancRsmk9_v9`, one free GPU.
   - Supervisor: `--stage stage-R-smoke --planned-cells 1`.
   - Ledger `/home/yschoi/gdna_anchorRT_ops`; result root `/home/yschoi/gdna_anchorRT_result`.
   - It includes the full historical admission of the four refit seals.
   - **Command:** the one in `refit_v9/r5_evidence/renders/common_args.txt`, plus
     `--namespace ancRsmk9 --smoke --only flickr25k:4:anchors:42 --epochs 1 --gpus <GPU>
     --anchor-approval-section <N>`, under the supervisor with manifest r5.
2. **`stage-R-run`:** the carried request, rendered after the smoke.
3. **`stage-T-smoke`** on the R smoke checkpoint.
4. **`stage-T-run`** on the completed R run receipt.
