# Anchor model — generation v8, revision 2: the §734 boundary repairs (stage L, Flickr25K first)

**Status: repair preparation only (§734). No smoke, training, reduction, input rehash or GPU work
ran.**
- The guarded preparation commands of §6 read only the named JSON files. They read no binary payload,
  and the guard enforced this.
- This revision supersedes the package of `docs/ANCHOR_LAMBDA_HANDOFF_v8.md` (`ade95ec6…`, manifest
  `b7b5af9e…`, requests `142e710c…` / `962be4b7…` and their full variants). That package stays in
  the tree, unchanged, as superseded review evidence; it is not approved. §2 lists its corrections.
- The v7 tree and evidence are untouched. The closed S/D/probe charge stays at 19,264.613316638395 s,
  and the ancS6 full-admission authority is unchanged.

## 1. The four repairs

| §734.2 item | Repair | Tests (`tests/test_anchor_lambda_stage.py`) | Battery v11 mutant |
|---|---|---|---|
| 1. One campaign per decision | `one_lambda_campaign`, called in `reduce_lambda` (see below) | two individually valid complete campaigns mixed (candidates from another campaign; substituted control) refuse, while each campaign alone still reduces; a receipt cell beyond the six refuses; a request declaring a seventh cell refuses | LX22, LX23, LX24 |
| 2. Every historical read is tracked | `anchor_incumbent(read=…)` and `anchor_v7_history(…, read=…)`; the reducer passes its read-once reader (see below) | the v7 snapshot changed after reduction and before publication refuses and writes nothing; the output's read set lists all four v7 files | LX25, LX26 |
| 3. Synthetic tests and a real-load guard | the binding test uses a recording stand-in for `RunIdentity._artifact`; an independent `open()` guard (`artifacts/anchor_confirmation/lambda_v8/guarded_pytest.py`) refuses real-data and binary opens | the binding test now also asserts that the dispatched `EXTRA_ARGS` and the sealed argv carry the same one-lambda tokens | (LX2, retained) |
| 4. Consistent continuity statement | the manifest generator, contract, recipe comment and this handoff state the recipe-parser exception; manifest revision 2 supersedes `b7b5af9e` | the closure byte-equality test is unchanged | — |

**Item 1 in detail.** All six cells must share:
- one receipt (path and digest);
- one recorded approval and request;
- a receipt whose cells are exactly the six;
- a request whose executed and declared cells are exactly the six, in run mode;
- a request digest that is the approval's.

The audit's mixed-campaign reproduction now refuses.

**Item 2 in detail.**
- Every historical file is read once through the reader: the approved aggregate and selected-N
  record, the v7 N record, the D summary, the S receipt and its plan snapshot.
- The snapshot is bound twice: semantically by the pinned receipt, and by the byte digest of the
  actual read.
- `write_once` re-verifies all of them. The v7 pins are not weakened, and no v7 reducer runs.

**Item 3 in detail.** The guard covers this process's `open()`. Children are recorded but not
hooked (§7).

**Item 4: the exception, as accepted in §734.1.**
- `STAGE_L_REVIEWED_OVERRIDES` keeps the exact first literal, exactly two occurrences, at most one
  lambda destination per argv, and every existing repeat guard.
- The parser cannot tell a stage-L campaign from any other run. It is **not** an authorization
  boundary: the stage, dataset, candidate value, typed recipe and approval are checked by the
  launcher's stage-L admission, the sealed recipe the trainer is held to, and the reducer.

## 2. Corrections to the superseded handoff (`ade95ec6…`)

1. **"None read a binary payload" was false for the tests.**
   - From `3b960ee` on, the six parametrizations of
     `test_the_snapshot_binding_seals_each_cells_own_override` reached
     `RunIdentity._artifact → _digest_file`. A recording stand-in lists the real Flickr25K files that
     path hashes, and the guard refused the first of them:
     - `flickr25k_qwen3_v4_trainset.jsonl` (4.2 MB);
     - the feature cache's `meta.json`;
     - `text_whiten_optTrain_localOnly.npz` (1.05 MB).
   - They ran in the unguarded full suites at `3b960ee` and `8aad9ca`, in single-file runs, and in
     battery v10 (two declared runs of the `lambda_bu=0` case: baseline and LX2). The interrupted
     run at `30c4d6d` stopped near 39 %, before the new file.
   - These are content hashes of small metadata, caption and whitening files. No array, checkpoint
     or dataset row was read, and nothing was deserialised.
2. **The regression suite itself is not synthetic.** This predates v8 and is unchanged by it.
   - Under the guard, the two legacy files `tests/test_phase3_selection_matrix.py` and
     `tests/test_phase3_select_n.py` fail in 82 cases. Each fails at the first real open: a cache
     `meta.json`, CIFAR-10 in 82 refusals and Flickr25K in 3.
   - **What they read.** Their identity and plan-snapshot inventories (`plan_snapshot` inputs,
     `RunIdentity._artifact`) hash, per dataset:
     - the caption cache `.jsonl`;
     - the feature-cache and foil-cache `meta.json`;
     - the selection and refit whitening `.npz` files.

     A recording stand-in lists the identity paths for all four datasets
     (`r2_evidence/legacy_tests_identity_paths_recorded.json`).
   - **Who ran them.** Every unguarded run of the 16-file suite read these files: the v7 package's
     921/1 runs, this worktree's baseline at `0e81f2b`, and every v8 suite.
   - The other 15 files pass under the guard with **0 refused opens** (§5).
   - Making the two legacy files synthetic is outside stage L, and it is proposed, not done.
3. **The first package's `--plan` renders were not guarded at all.** The launcher re-executes itself
   (`os.execve`) at its pre-import boundary, so they ran as ordinary processes.
   - A first guarded attempt in this revision lost its audit hook at that exec. It was detected by a
     missing guard summary and kept in `r2_evidence/plan_attempt1_guard_lost_at_exec/`.
   - The guard now refuses `os.exec*`. It supplies the same pre-import bundle the re-executed child
     would receive, so the launcher verifies it in-process. The redone renders produced the same four
     request digests.
4. **Manifest `b7b5af9e` said the recipe file was byte-equal to v7.** It was not
   (`b1ed8e4f` → `82ef8ed3`, now `95ae7751`). The generator, contract and this handoff now state the
   exception.

## 3. The package (revision 2)

- **Worktree** `/data/yschoi/gdna_anchor_lambda_v8`, branch `arch-exp-2026-09-anchor-lambda`.
  - Source commit `69c70fd`; `1469669` adds evidence only.
  - The suites and battery ran at `1469669`.
  - The manifest was built at `44f90fb`, which adds evidence only; its closure bytes equal
    `69c70fd`'s.

| Item | Path | SHA256 |
|---|---|---|
| **Generation manifest v8 revision 2** (64 files) | `artifacts/anchor_confirmation/authority_manifest_v8r2.json` | **`58e69ae1a3bed5366da61a4d61a6cdd90c338d9ac7b0474fcf9b6e2e2f7f5bbc`** |
| Contract L v1 (revised) | `docs/ANCHOR_LAMBDA_CONTRACT_v1.md` | `34a62bfa…` |
| **Smoke request, carried** | `request_preview_ancLsmk8_v8r2_smoke_carried.txt` | **`6e7827936e751fdd60f6350e77f5b635a6fab6e283aa81358f2fd4ec543a4c78`** |
| Smoke request, full | `request_preview_ancLsmk8_v8r2_smoke_full.txt` | `21e2e48a666fad31990ec6cf712a34a079dc159993a61d587868410f3f83030f` |
| **Stage-L run request, carried** | `request_preview_ancL8_v8r2_run_carried.txt` | **`daf99128fe014dea65958ee620697bc44d9bba27bfbc1863901b2ff8114cd70a`** |
| Stage-L run request, full | `request_preview_ancL8_v8r2_run_full.txt` | `3505ec928cf2a70063d80b5a3d31bf9b936abf1196103d1935a81b041eba81b0` |

- The digests are canonical request SHA256s, each recomputed independently from the printed JSON.
- **Every request carries:**
  - manifest `58e69ae1…`, selection `5cda7adb…`, one GPU, the Flickr25K seal only;
  - the preregistered rule with incumbent 0.7641936888306327 and threshold 0.02819158958924184.
- The smoke runs `lambda_wasserstein=0.30` for one epoch, under `ancLsmk8`. The run declares and
  executes the six labelled cells, under `ancL8`.

**Source changes, `8aad9ca` → `69c70fd`:**

| File | Digest |
|---|---|
| launcher | `28c0f689` → `61dafdc8` |
| reducer | `187d4b30` → `ae247d4a` |
| manifest builder | `8adec43e` → `617eda4a` |
| recipe module | `82ef8ed3` → `95ae7751` (comment only) |
| contract | `a33ff7f5` → `34a62bfa` |
| new test file | `fe5981ba` → `2700082f` |

- Unchanged since r1: the supervisor `12119ff8` and the launcher tests `c7395d76`.
- Against v7, 56 of the 62 v7 members stay byte-equal, as the closure test asserts.

## 4. Evidence (CPU only; real payloads guarded where stated)

| Run | Result |
|---|---|
| New test file under the `open()` guard | **154 passed, 2 skipped (opt-in), 0 refused opens** |
| Guard positive control: the same guard on the r1 binding test | 6 failed, 15 refused opens (the feature-cache `meta.json` ×6, the rest the guard script's own self-reads, since allowed); kept in `guard_positive_control_old_binding_test/` |
| 17-file suite, unguarded, at `1469669` | **1084 passed, 3 skipped**, rc 0 |
| The same 17 files under the guard | 1002 passed, 82 failed: only the two legacy files, see §2 item 2 |
| **The 15 non-legacy files under the guard** | **816 passed, 3 skipped, 0 refused opens**, rc 0 |
| **Mutation battery v11 at `1469669`** (`mutation_v11_frozen.py` `ea538639`) | **26/26 detected as declared**: v10's 21 plus LX22–LX26; 0 crash kills; tree and status unchanged; no stray process |

## 5. Read footprints of the preparation commands (enforced)

Each command ran under `r2_evidence/prep_guard/guarded_run.py` (`c2062e65`). The guard is an
allow-list `open()` hook that also refuses `os.exec*` and records child processes. Logs are in
`r2_evidence/prep_guard/`.
- **Manifest inventory.** It read only these named files, each twice (the read and the publication
  recheck), plus the worktree's source:
  - the approved aggregate `b4f3b0df`, selected-N `2bf6133d` and the p3lamA receipt;
  - the historical verifier and its six sources;
  - the v7 N record, D summary and S receipt.

  No exec was refused.
- **Four `--plan` renders.** They read only these named files, plus the worktree's source:
  - the v8r2 manifest, the approved aggregate and selected-N;
  - the v7 N record, D summary, S receipt and S plan snapshot;
  - the Flickr25K seal JSON;
  - the ancS6 snapshot JSON, for the carried variants only.

  0 refused opens. **Children:** `git rev-parse`/`git show` (source authority) and `bash` on the
  pinned Flickr25K wrapper, seven times (one historical incumbent, six cells).
- **What the wrapper does before its capture interpreter**, checked by reading its body: it makes
  `logs/` in the temporary directory and tests that the whitening file exists. The launcher refuses
  a missing whitening file first, so the wrapper's build branch never runs.

## 6. Proposed execution (unchanged form; new digests; each after its own exact approval)

**Smoke** (carried `6e782793…`, one free GPU, never evidence): the handoff r1 §9 command with
manifest `58e69ae1a3bed5366da61a4d61a6cdd90c338d9ac7b0474fcf9b6e2e2f7f5bbc`:

```
/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh ancLsmk8_v8 \
  env -C /data/yschoi/gdna_anchor_lambda_v8 -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest artifacts/anchor_confirmation/authority_manifest_v8r2.json \
  --manifest-sha256 58e69ae1a3bed5366da61a4d61a6cdd90c338d9ac7b0474fcf9b6e2e2f7f5bbc \
  --stage stage-L-smoke --planned-cells 1 --watch-path /home/yschoi/gdna_anchor4_result -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py \
  --anchor-confirm lambda --namespace ancLsmk8 --anchor-arms anchors \
  --anchor-selection /data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/ancS7_selected_n.json \
  --anchor-selection-sha256 5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff \
  --smoke --only flickr25k:4:anchors:42:lambda_wasserstein=0.30 --epochs 1 --gpus <1 free GPU> \
  --result-root /home/yschoi/gdna_anchor4_result \
  --input-seal flickr25k:stage1=/data/yschoi/gdna_p3exec_seals/flickr25k.stage1.input-seal.json \
  --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v8r2.json \
  --anchor-manifest-sha256 58e69ae1a3bed5366da61a4d61a6cdd90c338d9ac7b0474fcf9b6e2e2f7f5bbc \
  --admission-authority /data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/ancS6_snapshot_acaa0374fa0780e2.json \
  --anchor-approval-section <section>
```

- **Stage L,** later and separately (carried `daf99128…`): `--stage stage-L-run --planned-cells 6`,
  namespace `ancL8`, tmux `ancL8_v8`, `--run --gpus <1>`.
- **The reduction (CPU),** after its own approval: `anchor_confirm_decision.py lambda`.
- **Approval lines:**
  - `scope=stage-L-smoke manifest=58e69ae1… selection=5cda7adb… request=6e782793…` (full:
    `21e2e48a…`);
  - later, `scope=stage-L-run` with `daf99128…` (full: `3505ec92…`).
- **Inputs, budget and storage:** as in handoff r1 §10.
  - Carried ancS6 reuse, or a full Flickr25K rehash; both are submitted for review and neither is
    assumed.
  - The 3,600-s L ledger at `/home/yschoi/gdna_anchorL_ops`, accepted in §734.1 as a planning cap,
    with no automatic retry.

## 7. Limits

- **The `open()` guards see this process only.** Child processes (`git`, the `bash` wrapper render,
  the wrapper's argv capture interpreter) are recorded but not hooked. Their reads come from reading
  their code (§5).
- **A passing guarded run is evidence of synthetic isolation for the guarded files only.** The two
  legacy files remain non-synthetic (§2 item 2).
- **The battery shows the declared checks are live.** It does not validate the science.

## 8. Decisions requested

1. Repairs 1–4 and the corrections of §2.
2. Manifest revision 2 `58e69ae1…` and the enforced footprints of §5.
3. The smoke (`6e782793…` carried, or `21e2e48a…` full) after the final source, dependency,
   environment and carried-input review §734.3 names.
4. Whether to make the two legacy test files synthetic as a separate change.
