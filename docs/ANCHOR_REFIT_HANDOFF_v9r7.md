# Anchor model — generation v9 revision 7: the §758 R smoke result, the launcher repair, and stage T consuming what it verified (audits §758–§761)

**Two things to review.**
1. **The one approved r5 R smoke (§758) ran and was refused before training.** It used no device
   time. It is reported in §1 and was not retried.
2. **Revision 7, prepared in its own worktree:**
   - r6 is the §759–§760 repair: stage T consumes the weights and the runtime witness it verified.
   - r7 adds the launcher repair that the r5 refusal exposed.
   - r6 was never submitted; r7 supersedes it.

Nothing in r6 or r7 has run on real data, no real checkpoint, config or witness was loaded, and the
r5 tree is untouched (clean at `81ea828`).

## 1. The §758 R smoke (r5): refused at production admission, no device time

| Item | Value |
|---|---|
| Approval | §758 line, request `7c62f2db…`, manifest `d0de9fb3…`, freeze `5165f5dc…` |
| Run | supervisor run `20261005T110739Z-d14652fb`, tmux `ancRsmk9_v9`, GPU 5 (`GPU-bf4ed000-d77c-f059-2bb2-31ee39703e99`) |
| Command | exactly §758.2: `common_args.txt` + `--namespace ancRsmk9 --smoke --only flickr25k:4:anchors:42 --epochs 1 --gpus 5 --anchor-approval-section 758`, under `anchor_confirm_supervisor.py --stage stage-R-smoke --planned-cells 1 --ops-root /home/yschoi/gdna_anchorRT_ops --watch-path /home/yschoi` (start record `command_sha256 a586f6d0…`) |
| Times (supervisor records) | start 2026-10-05T11:07:39Z, final 12:36:30Z; wall 5330.2 s; pre-lease 5321.5 s |
| Result | launcher rc 2: `REFUSED: production admission requires the stdlib-only pre-import self-reexec handshake; invoke this file as its entrypoint` |
| Charge | device 0 s; charged 1.387 s (observation allowance); cumulative R/T 1.387 s of 80,000; lease GPU-seconds 5.37 |
| Settlement | final `status exited`; attempts []; leases held after exit []; orphaned attempts []; continuity lost []; tmux session ended; no live process |
| Left behind | no record, reservation, snapshot, run directory or result root; the namespace `ancRsmk9` was never reserved; free space at the final poll 375,539,560,448 bytes |

The settlement records are copied, never moved, to
`refit_v9/r5_smoke_ancRsmk9/` (with `SHA256SUMS`):
- the ledger;
- the command log;
- the tmux `cmd`, `log` and `status`.

**What failed.**
- Run as the entrypoint, `phase3_selection_matrix.py` is `__main__`. Only that instance runs the
  stdlib-only pre-import self-exec handshake.
- The new stage module `scripts/anchor_refit_stage.py` does `import scripts.phase3_selection_matrix
  as M`. That builds a **second** launcher instance, whose handshake flag is False.
- The refusal is the last production check in `_run_sweep`. It runs after the full historical seal
  admission, the lease and the plan snapshot, so the whole admission was spent first.
- The admission itself returned. A failing admission refuses earlier, with its own message. But it
  wrote no record, because the snapshot is published after this check. So it cannot be carried.

**Why nothing caught it.**
- Every in-process test imports the launcher by name and stubs `assert_production_source_authority`.
- The guarded `--plan` renders (`guarded_run.py --launcher-bundle`) also import the launcher by name
  first.
- `--plan` never reaches production admission.
- The same defect would have hit the T launcher. There, a second instance also skips
  `_assert_snapshot_gpu_leases`.

No retry and no repair was made under §758. A new attempt needs a new approval.

## 2. What is submitted

| Item | Value |
|---|---|
| Worktree, branch | `/data/yschoi/gdna_anchor_refit_v9r6`, `arch-exp-2026-09-anchor-refit-r6` (from r5 `81ea828`; pushed) |
| Source commit | `b582b7e`; manifest commit `c3afb78`; render-tool fix `446b461` |
| Manifest v9 r7 | `artifacts/anchor_confirmation/authority_manifest_v9r7.json` `2f24fc80bb83e2c73b17ec473c2731ade6d34a356e02414e6719fe9e9bec947c` (69 files) |
| Superseded, never submitted | r6 `9d6c1997…` (`b7c960a`; its evidence stays in `r6_evidence/`) |
| Contract | `docs/ANCHOR_REFIT_CONTRACT_v1.md` (closure member) |
| r7 R smoke request (full admission) | `9ba89a144c28459a2ea108ab772f6979e26abc3b4c2daae6052c3ac4550bb414` (`ancRsmk9r6`) |
| r7 R run request (full admission; preview) | `26665ca4e4a4711e36627807f76dabb0868ee1ecc84667d3ec82a57237f49870` (`ancR9`) |
| F authority | unchanged: `ancF_candidate_v1.json` `5165f5dc…` + §744 text `13ef776b…` |

The namespace label `ancRsmk9r6` comes from the contract text written at r6. It was never used, and
r7 keeps it.

**r7 against r5 (`git diff 81ea828 b582b7e`).** Eight source files change. Every other source file is
byte-identical to r5.

| File | Change | For |
|---|---|---|
| `dna_utils/runtime_state.py` | `VerifiedRuntime` / `verified_runtime`; optional `verified` on the resolver and `apply_inference_epoch` | §759–760 |
| `extraction_siglip2.py` | `extract_code(args, *, verified=None)` | §759 |
| `terminal_official_test.py` | passes `verified` through | §759 |
| `scripts/anchor_terminal_test.py` | step 8: bind the checkpoint and witness once; hand the binding to the test | §759 |
| `scripts/extract_train_split.py` | fourth pin (terminal epoch); the binding drives loading and epoch | §760 |
| `scripts/anchor_refit_stage.py` | terminal-epoch pin to producers; handshake check before any admission | §760, §1 |
| `scripts/phase3_selection_matrix.py` | `_bind_entrypoint_instance` before the R/T dispatch; `assert_preimport_handshake` | §1 |
| `scripts/anchor_confirm_manifest.py` | revision 7 | — |

## 3. The launcher repair (r7)

1. **Bind the instance.** Before it dispatches `--anchor-confirm refit|test`, the entrypoint binds the
   module name `scripts.phase3_selection_matrix` to itself (`__main__`). The stage module therefore
   holds the verified instance. It refuses if another instance, or a stage module holding one, was
   imported first. A plain import, as in the tests, is unchanged.
2. **Check the handshake first.** An executing `refit` or `test` command checks the handshake before
   the manifest, the seals or any admission. A misbound launcher now refuses in seconds, not after
   an 89-minute admission.

**Tests** (`tests/test_anchor_refit_stage.py`):
- **The real entrypoint**, run in a child exactly as the self-exec child runs: the launcher file as
  `__main__`, the bundle in the environment, PYTHONPATH unset, an open() guard in the child.
  - The stage holds `__main__` with the handshake verified.
  - The command refuses only at the next step (no manifest given).
- The same run with the launcher imported first refuses with "another instance".
- **In-process:** the handshake is checked before the manifest is read.
- **All three fail on the r6 source.**
  - The first two fail at their assertions; the stage holds `scripts.phase3_selection_matrix`.
  - The third initially failed through a crash in my stand-in loader. I made the stand-in refuse
    cleanly, and it now fails at its declared assertion.

## 4. Stage T consumes what it verified (r6, unchanged in r7)

**What §759–§760 showed.** Both consumers checked digests, but then opened the files again by path:
- the official extraction loaded the weights by path;
- both resolvers reopened the witness (and hashed the checkpoint).

**The repair.**
1. Read the checkpoint and the witness **once**.
2. `runtime_state.verified_runtime` binds them: both digests are the pins, and the witness, parsed
   from those bytes, names that checkpoint at the admitted terminal epoch.
3. The weights load from the verified bytes through the unchanged shared loader.
4. The resolver uses the verified witness object and opens neither file.
5. `apply_inference_epoch` requires the effective epoch and checkpoint to be the admitted ones
   before it returns, which is before any dataset access.
6. The final consistency checks stay.

**Where it applies.**
- The T entry passes this binding through `run_official_test` to `extract_code`.
- Train extraction gets a fourth pin (the terminal epoch) and builds the same binding.

**Legacy calls are unchanged.** The resolver's unbound branch is AST-equal to v8's; six resolver
cases and `apply_inference_epoch` equal v8's function; both consumers' legacy tests load by path.

**Tests: the real loader and resolver in both consumers.**
- Stable runs.
- Checkpoint swapped to weight 999, or a valid witness swapped to epoch 0, after verification:
  - restored before the check → the verified objects are consumed and the run completes;
  - left changed → consumed, then refused at the after-check.
- A file changed before the verified read → refused before any weight load.
- Epoch pin wrong, non-numeric or missing → refused.
- Verified-runtime predicates.
- 14 of the 18 selected consumer tests fail on the r5 source. The 4 that pass are the stable train
  control and r5's own pin checks.

## 5. Evidence

| Check | r7 at `c3afb78` | r6 at `b7c960a` (superseded) |
|---|---|---|
| 27-file suite, unguarded | **1456 passed / 11 skipped**, rc 0 | 1453 passed / 11 skipped |
| 27-file suite under the open() guard | **1456 / 11, 0 refused opens**; tree clean before and after (`r7_evidence/suitepair/`) | 1453 / 11, 0 refused |
| Mutation battery v13 | **52/52 detected as declared** (revision f; 56 baselines passed, 0 refused opens; inventory and checkout unchanged; `r7_evidence/battery13g/`) | 49/49 (revision e) |
| Manifest generation under the guard | exit 0, 0 refused, 19 named (`prep_manifest_v9r7/`) | exit 0, 0 refused |
| R request renders under the guard | exit 0, 0 refused; 10 named JSON files, 245 source opens, after the tool fix (`r7_evidence/renders/`); the two renders before the fix refused as designed (`refused_render_before_tool_fix/`) | 0 refused (before the r7 tool fix) |

**What the mutants disable:**
- **RX39–RX49:** the r6 binding (§4).
- **RX50–RX52:** the instance binding, the early handshake check, and the refusal of an instance
  imported first.

**The render-tool fix (`446b461`).**
- `guarded_run.py --launcher-bundle` computed the handshake bundle by importing the launcher by name.
  Under r7, that correctly refuses as "another instance imported first".
- The tool now computes the bundle from the file under a throwaway module name, as the real parent
  process does before its exec. This is an evidence tool outside the closure.

## 6. The transition (proposal; contract §7)

The rule that a T receipt admits only R work of its own manifest stays. The r5 smoke produced no
receipt; no old record is rewritten.

**Proposed order under r7**, each step under its own ledger line:
1. **`stage-R-smoke`** request `9ba89a144c28459a2ea108ab772f6979e26abc3b4c2daae6052c3ac4550bb414` (`ancRsmk9r6`): the same single cell, Flickr25K N4,
   seed 42, one epoch, one free GPU, with **full seal admission**. Nothing can be carried: the r5
   admission wrote no record. Expect about 5,300 s of CPU admission before about 3 minutes of
   training. The 3-hour wall limit of §758 holds.
2. **`stage-T-smoke`** (`ancTsmk9`) on that receipt. This exercises the repaired consumers on a real
   checkpoint before any full campaign.
3. **`stage-R-run`** (`ancR9`), carried from the r7 smoke's snapshot.
4. **`stage-T-run`** (`ancT9`).

**Budget:** one cumulative ledger (`/home/yschoi/gdna_anchorRT_ops`, §758's 80,000-s envelope, 1.387 s
used).

## 7. Limits stated plainly

- **Only on synthetic data.** The entrypoint test reproduces the child's composition. The real
  self-exec (`os.execve`), real seals and a real GPU lease have not run through r7. The r7 R smoke is
  the first real exercise.
- **The binding is proven on synthetic files.** It uses the real loader and resolver, but no real
  0.65 GB checkpoint has passed through it. Each T consumer holds that checkpoint in memory while it
  loads (r5's train path already did).
- **Other producers.** BIO, NMI and the cell seal never open the checkpoint or the witness (source
  inspection).
- **The early check covers the handshake only.** The full production source-authority check (clean
  sources, identical bundle digests) still runs at its place in `_run_sweep`, after admission.
