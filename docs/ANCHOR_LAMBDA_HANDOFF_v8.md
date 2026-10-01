# Anchor model — generation v8 package for stage L (response to audit §733)

**Status: preparation only. Nothing here is executable until the audit approves an exact request.**
- No lambda training, smoke, real-input preflight, reduction or GPU work ran.
- The commands that did run are listed in §8 with their read footprints. They are the CPU test
  suites, a mutation battery in a sandbox copy, the manifest inventory and four `--plan` renders,
  and none read a binary payload.
- The v7 tree `/data/yschoi/gdna_anchor_confirm_v1` and all its evidence are untouched.
- **Scope (§733.1): the user decided "Flickr25K first"** on 2026-10-01, answering the question the
  audit put to them. It keeps the TODO and 2026-09-14 order. It is not an exemption from TODO
  13–15: another dataset joins only if Flickr25K's choice moves, then with its own checks, and a
  Flickr25K winner is never copied.

## 1. The package

- **Worktree** `/data/yschoi/gdna_anchor_lambda_v8`, branch `arch-exp-2026-09-anchor-lambda`.
  - It branches from the v7 head `0e81f2b`.
  - Source commit `8aad9ca`; package commit `be515e0` (records only; closure bytes equal to
    `8aad9ca`).

| Item | Path | SHA256 |
|---|---|---|
| **Generation manifest v8** (64 files) | `artifacts/anchor_confirmation/authority_manifest_v8.json` | **`b7b5af9ed13473ed21773ddf824c951c6064a8cdeeb18d61c3a8a8777ab3572d`** |
| Stage-L contract v1 (closure member) | `docs/ANCHOR_LAMBDA_CONTRACT_v1.md` | `a33ff7f540c9e117d1bd6d8c31e72adb9d99cf9d31333462fcc63ad025844cd7` |
| **Smoke request, carried** (proposed first) | `request_preview_ancLsmk8_v8_smoke_carried.txt` | **`142e710ca6ac5ea7c97dddfa041214578fad1cc17b7cce9511438a3101a9c77a`** |
| Smoke request, full | `request_preview_ancLsmk8_v8_smoke_full.txt` | `f95e844db0e22bf9c37f4c738250017aea5d03e69b22ce2571c9a0ec3bf7f0e3` |
| **Stage-L run request, carried** (after the smoke) | `request_preview_ancL8_v8_run_carried.txt` | **`962be4b7d29da447293fc216bb58fb4d628d3710d521cbd79ee594e052ccfb33`** |
| Stage-L run request, full | `request_preview_ancL8_v8_run_full.txt` | `d6182b7046c9aa634edba3349d706cff55a41ada06a2aec7db7e81f96a2ad259` |
| Mutation battery v10 harness and report | `artifacts/anchor_confirmation/lambda_v8/mutation_v10_frozen.py`, `…/mutation_v10/report.json` | `a304ec12…`, `aa7dc881…` |

- The digests are canonical request SHA256s, each recomputed independently from the printed JSON.
- **Every request carries:**
  - stage `lambda`, one GPU, the Flickr25K seal only;
  - selection `5cda7adb…` and manifest `b7b5af9e…`;
  - the preregistered rule with its v7 pins and unrounded numbers: incumbent seed-42 score
    0.7641936888306327, seeds [0.7641936888306327, 0.7436437784867651, 0.7360020992413908], range
    and threshold 0.02819158958924184.
- The run requests declare and execute the six cells: the `incumbent` continuity control and the
  five candidates. The smoke executes `lambda_wasserstein=0.30` with one epoch, under its own
  namespace `ancLsmk8`, so it never reserves the run's namespace `ancL8`.

## 2. Source changes (`git diff 0e81f2b 8aad9ca`)

| File | v7 → v8 | Change |
|---|---|---|
| `scripts/phase3_selection_matrix.py` | `2cb8343d` → `28c0f689` | anchor stage `lambda`: scope, candidate and v7 tables; `anchor_v7_history`, `anchor_lambda_admission`, `anchor_lambda_rule`, the control gate; request label and rule; scopes `stage-L-*`; one declared override admitted on an anchor cell in stage L only |
| `scripts/anchor_confirm_decision.py` | `79fbe131` → `187d4b30` | `lambda` reduction; `admit_metadata` / `campaign_approval` take a stage-L role (defaults keep S/D behaviour) |
| `scripts/anchor_confirm_supervisor.py` | `b5eecf58` → `12119ff8` | stages `stage-L-smoke` and `stage-L-run` (one GPU, 2-h limit), their own ledger root and a 3,600-s budget; crossed roots refuse |
| `scripts/anchor_confirm_manifest.py` | `2a3b8ca4` → `8adec43e` | generation v8, the v7 pins (read), the stage-L design, predecessor v7 |
| `dna_utils/scientific_recipe.py` | `b1ed8e4f` → `82ef8ed3` | **§3: a declared protocol-dependency change** |
| `tests/test_anchor_confirm_launcher.py` | `a542db5c` → `c7395d76` | two tests whose premise the stage-L override changes; the manifest fixtures carry the v8 fields |
| `tests/test_anchor_lambda_stage.py` | new, `fe5981ba` | 150 tests (2 opt-in) |
| `docs/ANCHOR_LAMBDA_CONTRACT_v1.md` | new | the stage-L contract |

**Byte-equal to the v7 manifest pins**, tested by `test_every_v7_member_but_the_declared_v8_sources_is_byte_equal`:
- every other v7 closure member: the trainer, model, dataloaders, configuration, the four wrappers,
  the runtime-environment module, the seal tooling, the probe, the contract v3 and every other
  test file.

## 3. Decision requested: the recipe module (a protocol dependency outside §733.3's byte-equal list)

- **The problem.** §733.3 item 2 asked for the recipe module to stay byte-equal, and with the
  pinned Flickr25K wrapper that is not possible.
  - The wrapper passes `--lambda_wasserstein 0.15` and `--lambda_text_hash_ntxent 0.05` literally
    (lines 118 and 121), and `--lambda_bu "$LBU"` (0.02).
  - A candidate appended through `EXTRA_ARGS` repeats its destination.
  - The v7 repeat rule refuses any repeat outside its seven reviewed destinations, both in the
    launcher's sealing and in the trainer's own check (`verify_trainer_recipe`, called by the
    unchanged `train_siglip2.py`). So no transport or text candidate could be sealed or started.
  - The opt-in real-wrapper test reproduces the repeat for all five candidates.
- **The change: one narrow, separate table.**
  - `STAGE_L_REVIEWED_OVERRIDES` covers `lambda_wasserstein` 0.15, `lambda_bu` 0.02 and
    `lambda_text_hash_ntxent` 0.05.
  - A destination in it may repeat only as exactly that wrapper literal followed by exactly one
    override, and **at most one such destination per argv**.
  - The seven reviewed overrides, their alternate, every other repeat refusal and every S/D argv
    are unchanged.
  - Tests cover one admitted repeat per axis; refusal of two lambda repeats, a third occurrence,
    another first literal, and another repeated option.
  - Battery mutants LX18 and LX19 confirm the two guards are live.
- **Alternatives.**
  - Editing the wrapper is excluded: its bytes are pinned.
  - Passing only the balance candidate through `LBU` would leave TODO 13 and 15 unrunnable.
- **The ask.** Accept this change, or name another route.

## 4. Defects found while porting (fixed, with tests)

1. **The stream passed no override in anchor mode.** `_run_sweep` passed overrides only for
   `--sweep lambda`.
2. **The plan snapshot sealed no override.** `expected_cell_binding` did not pass the cell's
   overrides into its sealed anchor recipe.

   **Together, 1 and 2 would have run the incumbent recipe under each candidate's label,
   unnoticed.** The sealed recipe and the trainer argv would both be the incumbent's, so the
   trainer's own check passes. Tests now hold each cell's override in both places (battery LX1,
   LX2).
3. **A shadowed launcher helper.** The first draft's stage-L `_finite_proportion(value, what)`
   replaced the refit aggregation's `_finite_proportion(payload, key, *, what)` of the same name.
   - Found by the full suite at `9fe990c`: 12 failed, 1054 passed, 3 skipped. Five were
     `test_phase3_select_n` failures; seven were the launcher tests' `copy_of_the_tree` manifest,
     which lacked the v8 fields.
   - Renamed to `_v7_seed_score`, with a new test that each changed module defines each top-level
     name once.
4. **Tests that could crash-kill.** Two refusal tests would have crashed rather than failed by
   assertion under their mutants. They were tightened at `8aad9ca`.

## 5. The stage-L design as built (contract L v1)

- **Cells.** One Flickr25K stream at N = 4, seed 42, anchors:
  - the continuity control first;
  - then transport 0.30 and 0.50, balance 0, and text 0.025 and 0.10.
- **Admission, before any lease.**
  - The incumbent is rendered with the v7 cell's admitted Flickr25K seal authority, and its typed
    fields must equal the v7 stage-S seed-42 cell's sealed fields exactly.
  - Each cell must carry the protocol values with exact types, including the three lambdas. It
    must differ from the incumbent in exactly its one lambda; the control differs in nothing.
  - The real `--plan` shows exactly this (see the previews).
- **The control gate (§733.2).**
  - It runs after the control completes: the terminal score must equal 0.7641936888306327 exactly,
    or the stream stops before any candidate and no receipt is written.
  - The reducer repeats the check. It also requires the control's sealed fields to equal the v7
    cell's.
  - The contract states the limitation before any score: a match does not prove unchanged training,
    and a mismatch is a diagnostic failure, not evidence about a recipe.
- **The reducer (`lambda`).**
  - Rule: `delta = score − incumbent_seed42`; a candidate qualifies iff `delta > threshold`,
    strictly and unrounded. The highest qualifier wins per axis. On an exact tie the nearer value
    wins (by exact decimal distance of the declared values), then the smaller.
  - Winners combine. A change states the reselection obligations of contract v3 §7.6 step 4 and
    the expansion rule.
  - **Distinct roles.**
    - The v7 frozen N record, D summary and S receipt are historical metadata, read at the accepted
      digests and never replayed.
    - Every L record must be this generation's stage-L campaign under a `stage-L-run` approval,
      whose request binds the same rule.
    - The S/D admission is unchanged and refuses an override; an L record is not S/D evidence, and
      an S/D record is not L evidence.

## 6. Historical handoff

| v7 authority | Path (v7 worktree) | SHA256 | Read as |
|---|---|---|---|
| manifest v7 | `authority_manifest_v7.json` | `c0612963…` | pinned in the v8 manifest and every request (not loaded) |
| frozen N record | `ancS7_selected_n.json` | `5cda7adb…` | JSON metadata: N |
| official D summary | `ancP7_decision.json` | `28b10a4c…` | JSON metadata: seed scores, range |
| stage-S receipt and its plan snapshot | `ancS7_sweep_complete.json` | `5915768767e3…` | JSON metadata: the seed-42 cell's sealed recipe and seal authority |

- These are launcher constants, required equal in the v8 manifest (`historical.anchor_v7`).
- The opt-in test (§7) checks them against the real files.
- No v8 code re-runs a v7 reduction or re-checks a v7 approval.

## 7. Evidence (CPU only; `CUDA_VISIBLE_DEVICES=` empty)

| Run | Result |
|---|---|
| Baseline, the v7 16-file suite at `0e81f2b` in this worktree | 921 passed, 1 skipped (= v7's own record) |
| 17 files (16 + the new file) at `9fe990c` | 12 failed, 1054 passed, 3 skipped (§4 item 3) |
| … at `3b960ee` | 1078 passed, 3 skipped |
| … at `30c4d6d` | **interrupted by me** at about 39 %: I stopped it to edit two tests. Its pytest ignored SIGTERM and was killed by PID; one orphaned fixture sleeper was then killed by its process group; no process remains. Not evidence. |
| **… at `8aad9ca` (final source)** | **1078 passed, 3 skipped**, rc 0, 524 s (the 3 skips are opt-in real-artifact tests) |
| Opt-in real tests (`GDNA_ALLOW_REAL_ARTIFACT_TESTS=1`) | 2 passed: the real pinned Flickr25K wrapper renders each candidate through the reviewed repeat; the real v7 JSON history gives N 4, 0.7641936888306327 and 0.02819158958924184, and the literal v7 pins equal the real manifest |
| **Mutation battery v10 at `8aad9ca`** | **21/21 detected as declared**; 22 declared tests passed unmutated first; every kill an assertion failure, none a crash; tree and status unchanged; no stray process |

## 8. Commands that ran, and their read footprints

- **The test suites and the battery:** private and synthetic. The battery used a detached sandbox
  that has since been removed.
- **The manifest inventory** (`anchor_confirm_manifest.py inventory`) read only these, all JSON,
  Python, Markdown or shell text:
  - the approved aggregate `b4f3b0df`, the selected-N record `2bf6133d` and the p3lamA receipt;
  - the historical verifier and its six sources;
  - the three v7 JSONs;
  - the 64 closure files.
- **Four `--plan` renders** read only these:
  - the v8 manifest and closure, the approved aggregate and the selected-N record;
  - the v7 frozen N record, D summary, S receipt and S plan snapshot (JSON);
  - the Flickr25K seal JSON, and the ancS6 snapshot JSON for the carried variant.

  Each also ran the pinned Flickr25K wrapper under bash with the capture interpreter in a
  temporary directory, statting the whitening file's existence.
- **Never read by any of them:** a cache, array, `config.pt`, checkpoint or dataset row. Nothing
  was deserialised and no GPU was used.

## 9. Proposed execution (each step after its own exact approval)

**Step 1 — smoke** (carried request `142e710c…`, one free GPU, never evidence; `ancLsmk8`):

```
/data/yschoi/gdna_p3exec_authority/bin/tmux_run.sh ancLsmk8_v8 \
  env -C /data/yschoi/gdna_anchor_lambda_v8 -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
  --manifest artifacts/anchor_confirmation/authority_manifest_v8.json \
  --manifest-sha256 b7b5af9ed13473ed21773ddf824c951c6064a8cdeeb18d61c3a8a8777ab3572d \
  --stage stage-L-smoke --planned-cells 1 --watch-path /home/yschoi/gdna_anchor4_result -- \
  /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py \
  --anchor-confirm lambda --namespace ancLsmk8 --anchor-arms anchors \
  --anchor-selection /data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/ancS7_selected_n.json \
  --anchor-selection-sha256 5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff \
  --smoke --only flickr25k:4:anchors:42:lambda_wasserstein=0.30 --epochs 1 --gpus <1 free GPU> \
  --result-root /home/yschoi/gdna_anchor4_result \
  --input-seal flickr25k:stage1=/data/yschoi/gdna_p3exec_seals/flickr25k.stage1.input-seal.json \
  --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v8.json \
  --anchor-manifest-sha256 b7b5af9ed13473ed21773ddf824c951c6064a8cdeeb18d61c3a8a8777ab3572d \
  --admission-authority /data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/ancS6_snapshot_acaa0374fa0780e2.json \
  --anchor-approval-section <section>
```

- **Why this smoke.** It exercises, on a real trainer, the stage-L path that tests cannot reach:
  - the trainer's own recipe check admits the reviewed `lambda_wasserstein` repeat;
  - `args.txt` reads back the three lambdas;
  - the record and receipt path completes.

**Step 2 — stage L**, later and separately (carried request `962be4b7…`): the same command with
`--stage stage-L-run --planned-cells 6`, namespace `ancL8`, tmux `ancL8_v8`, and
`--run --gpus <1 free GPU>` in place of the smoke flags.

**Step 3 — the reduction (CPU)**, after its own approval: `anchor_confirm_decision.py lambda
--sources <L sources> --sources-sha256 … --manifest …v8… --out ancL8_lambda_decision.json`.

**Approval lines needed:**
- `scope=stage-L-smoke manifest=b7b5af9e… selection=5cda7adb… request=142e710c…`, or `f95e844d…`
  for the full variant;
- later, `scope=stage-L-run` with `962be4b7…`, or `d6182b70…` for the full variant.

## 10. Inputs, budget and storage

- **Inputs.** The Flickr25K stage-1 seal `e44b363a…` only.
  - **Carried (proposed).** The original ancS6 full-admission snapshot `ef5a3e8d…` is reused
    through the unchanged v7 guard (`anchor_carried_admission_refusal`, byte-pinned in the request)
    and the stats-only recheck. The guard requires exactly the four seals the snapshot admitted.
    The request's admitted seal must then be the Flickr25K one.
  - **Full (alternative).** A historical-verifier rehash of the Flickr25K seal alone (about 33 GB,
    roughly 6 min at the v5 rate). It is not newly labelled anything; it is the same verifier.
  - The reuse is submitted for the review §733.3 item 6 asks for. It is not assumed.
- **Budget** (own ledger `/home/yschoi/gdna_anchorL_ops`, created at the first start; ceiling 3,600
  s). The S/D/probe ledger (19,264.613316638395 s) is reported, not reused.

  | Part | Planning charge |
  |---|---|
  | Smoke: one cell, one epoch (≤ a cold N4 cell, 274 s) + 1 × window allowance | ≤ 280 s |
  | Run: control + 5 candidates (expected 274 + 5 × 130 = 924 s; worst 6 × 274 = 1,646 s) + 6 × window | ≤ 1,655 s |
  | One failed attempt (reserve, about one expected run) | ≈ 925 s |
  | **Total planning** | **≈ 2,860 s, under 3,600 s** |

  - Headroom: 130 s per supervised command, reserved by the start rule.
  - Wall limits: 2 h each. Neither the ceiling nor a wall target is a completion bound.
- **Storage.** About 7 cells × 0.75 GiB plus the 10-GiB floor, on `/` (373.6 GB free, 2026-10-01).
  Records go in this worktree's `artifacts/anchor_confirmation/` on `/data`. Nothing is deleted
  to make room.

## 11. Decisions requested

1. The scope (Flickr25K first; the user's decision above) and contract L v1.
2. **The recipe-module change of §3**, or another route.
3. The v8 generation: manifest `b7b5af9e…`, the historical handoff (§6), and the evidence (§7).
4. The smoke: a `stage-L-smoke` line for `142e710c…` (carried) or `f95e844d…` (full), with the
   carried-admission reuse reviewed.
5. Stage L and its reduction, later and separately.
