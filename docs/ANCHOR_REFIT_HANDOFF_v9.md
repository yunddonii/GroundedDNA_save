# Anchor model — generation v9: the stage-R/T package (audits §743–§744)

**PREPARATION ONLY. Nothing has been executed.**
- No refit, test access or GPU work; no smoke.
- No real model, config or checkpoint was loaded.
- No scientific array, seal payload or cache was read or rehashed.

This package asks for review of the code and evidence, and then for one `stage-R-smoke` line (§9).

## 1. What is submitted

| Item | Value |
|---|---|
| Worktree, branch | `/data/yschoi/gdna_anchor_refit_v9`, `arch-exp-2026-09-anchor-refit` (from v8 `3dd1c02`; pushed) |
| Source commit | `acc4e01`; manifest commit `9dc2229` |
| Generation manifest v9 r2 | `artifacts/anchor_confirmation/authority_manifest_v9r2.json` `b7e9264a7a933e3efd30125628f638bc5c5304f32ccae2055b730ad382b215f6` (69 files) |
| Superseded, never submitted | v9 r1 `19acd584…` at `bdaba58` (§6) |
| Contract | `docs/ANCHOR_REFIT_CONTRACT_v1.md` `6d22f391…` (closure member) |
| F authority | `ancF_candidate_v1.json` `5165f5dc…` + ledger §744.1 |
| R smoke request (full admission) | `cd2c4164ecbf6310ea97215a8edc5104fc3a052b706c9a2d8920a99460b70a68` |
| R run request (full admission; preview) | `b393bc2dcf8d7369ca37bc0af15d343ca52980ceb85cabfab64b14424dd06cbb` |

**Changes relative to v8** (2626 lines added, 140 removed):

| File | Change |
|---|---|
| `p0_protocol.py` | adds `anchor_refit_withholds_official_test` and `campaign_cell_stage` (pure) |
| `train_siglip2.py` | start-up check; `_official_test_allowed`; the terminal block moved out |
| `terminal_official_test.py` (new) | `run_official_test`: the trainer's former terminal block, unchanged except one variable name |
| `scripts/anchor_refit_stage.py` (new) | the stage-R/T admission, requests, T campaign and attempt reservation |
| `scripts/anchor_terminal_test.py` (new) | the T entry |
| `scripts/phase3_selection_matrix.py` | stage dispatch; anchor refit allowed in `build_command`; arm in refit tags; R cells run no post-chain; `assert_official_test_withheld`; four approval scopes; closure; refit-contract pin |
| `scripts/anchor_confirm_supervisor.py` | the stage-R/T ledger |
| `scripts/anchor_confirm_manifest.py` | v9 r2 |
| tests | `tests/test_anchor_refit_stage.py` (new, 134 + 1 opt-in); three existing files adjusted (§7) |

## 2. The access boundary, against §743 and §744.2 item 1

**Fail-closed R identity.** One pure policy, `anchor_refit_withholds_official_test(axis_center,
p0_refit_active, cell_id, sealed_recipe)`, is applied twice in the trainer:
- **at start-up**, before the run directory exists;
- **at the terminal decision** (`train_siglip2._official_test_allowed`).

| Case | Result |
|---|---|
| Anchor model, P0 refit, cell `stage=refit`, sealed recipe | **withheld**: no test dataset, extraction or evaluation |
| Anchor model, P0 refit, no campaign binding / no cell id | **refuses at start-up** |
| Anchor model, P0 refit, cell stage not `refit` (marker altered) | **refuses at start-up** |
| Anchor stage-R cell without a sealed recipe | **refuses** (the existing recipe check, then the policy) |
| Anchor stage-R cell that is not a P0 refit | **refuses** |
| Stage-R anchor cell id running the pre-revision model | **refuses** |
| Any model without anchors | unchanged (`False`); a legacy cell id is not even parsed |
| Anchor stage-1 cells (S/D/L) | unchanged (`False`) |

- **Inputs come from the binding, not the caller.** The cell id and recipe digest come from the
  launcher's all-or-none campaign binding, which the trainer already verifies against its parsed
  identity. No caller variable can grant T, and no environment variable carries a T authority.
- **The parsed composition of an R cell** (from the guarded real-wrapper renders of §5). The
  wrapper body gives `--text_whiten_npz <trainOnly>` and, from `FINAL_EPOCH=1`, `--final_epoch_eval`;
  `VAL_RATIO=0.0` gives `--val_split_ratio 0.0`; `STOP_EP=N`. `EXTRA_ARGS` gives `-e N+1`, the seed,
  `--dna_distance_mode base`, `--selection_mode refit`, the seven reviewed overrides, top-p, joint,
  `--axis_center anchors` and the seal flags. The wrapper's last line adds `-ev -s`.
- **Where R ends.** The trainer saves the terminal checkpoint (epoch N), its runtime witness,
  `config.pt` and `args.txt`, then prints `[anchor-refit] ... WITHHELD`; training isolation
  (`final_epoch_eval` + `stop_after_epoch`) is unchanged. The launcher (`run_cell`) runs **no**
  post-chain for an anchor refit and requires the run directory to hold **none** of the 13
  `OFFICIAL_TEST_OUTPUTS`. The R record says `official_test: withheld`.
- **T happens in one explicit entry only:** `scripts/anchor_terminal_test.py`. No other T-capable
  path exists for the anchor model.

## 3. Stage T, against §744.2 items 2–4

- **Reserve before T access (item 2).** Per cell, `<ns>_attempt_<R tag>.json` is created
  exclusively (`O_EXCL`, 0444) before the entry starts. It is never removed. A second attempt —
  duplicate, concurrent or retry, after a failure at any step and even when no output exists —
  refuses without starting a process. Recovery needs a new authorization.
- **What the entry proves before it builds any model or dataset:**
  - the attempt bytes are the digest the launcher passed;
  - the attempt's request is its stated digest and names this cell and run directory;
  - the ledger carries a `stage-T-*` line for exactly that request, manifest and F record, and it
    is the line the attempt recorded;
  - the generation manifest still holds, imported modules included;
  - the checkpoint bytes and the runtime witness (terminal epoch) are pinned;
  - no test output exists;
  - then, on the saved configuration: anchors, refit mode, this cell id, this run directory.
  - Only then does it call `run_official_test`.
- **Raw evaluation is explicit (item 3).** `run_official_test` is the trainer's former terminal
  block: cache and whitening resolution, `extract_code` (query, db), then
  `evaluation(..., bio_project` left `False`), which writes `evaluation_siglip2_base.json`. The
  launcher then runs the unchanged post-chain in order: `extract_train_split`,
  `eval_cell_bioproj --require-train` (which validates the raw file before it writes the post-BIO
  one), `pairwise_nmi`, `seal_cell_analysis`. A composed test runs the real raw evaluator and the
  real BIO stage on a synthetic extraction; another shows that the BIO stage refuses without the
  raw file.
- **Fail-stop.** Every child failure stops the chain, keeps the attempt and publishes no record.
  Then `assert_refit_outputs`, a checkpoint re-hash, the T record, and the T receipt only when all
  cells complete.
- **The whole request is bound (item 4).**
  - **R request:** the anchor request (stage `refit`, mode, declared and executed cells, namespace,
    roots, epochs, input-seal file pins, admission pin, GPU count) plus `freeze` (F path, digest,
    acceptance section, per-dataset recipe digests). Stage R takes only the four approved refit
    seal files.
  - **T request:** manifest path and digest, F, the R receipt and its approval, and per cell the
    run directory, record, terminal checkpoint digest, runtime witness and epoch. It also lists the
    output names, the chain and the GPU count.
  - **Distinct meanings.** R receipts are `_sweep_complete.json` with stage `refit`; T receipts are
    `_test_complete.json`.
  - **Refused at admission:** a stage-1 seal standing in for a refit seal, a receipt of the other
    mode, and a receipt under another generation.
- **Refit-seal admission (proposal; nothing read in preparation).** The refit seals record the
  same six historical sources as the stage-1 seals, read from seal JSON metadata. The R smoke
  request declares the whole stage, so the smoke performs the full historical-verifier admission of
  all four refit seals. The R run would then carry that smoke snapshot (stats-only), as stage S
  carried ancS6. Its carried request can only be rendered after the smoke exists.

## 4. Recipe mapping (S/D → R), as implemented

`refit_admission` (before any lease, in every mode):
- every R render carries the refit protocol values (§3 of the contract, `-ev` included);
- the rendered `none` control differs in `axis_center` alone;
- R's typed recipe differs from the F record's validated recipe only in `REFIT_PROTOCOL_FIELDS`;
- with admitted seals, the four seal fields must be the refit seal's and the CLIP identity F's.

A `--plan` preview renders without seals, so it does not compare the input-authority fields; the
report states `input_authority_compared`.

The guarded preview of the real run differs from F only in the expected fields. For cifar10 seed
43, for example: `epoch`, `final_epoch_eval`, `keep_final_checkpoint`, `lr_schedule_horizon`,
`random_seed`, `selection_mode`, `sinkhorn_schedule_horizon`, `text_whiten_npz`, `val_split_ratio`,
plus the not-yet-admitted input fields.

## 5. Evidence

| Check | Result |
|---|---|
| New test file at `acc4e01` | 134 passed, 1 skipped (the opt-in real-wrapper test) |
| 19-file suite at `9dc2229`, unguarded | **1232 passed, 4 skipped** (the opt-in tests), rc 0 |
| 19-file suite under the open() guard | **1232 passed, 4 skipped**, rc 0, **0 refused opens** |
| Mutation battery v13b at `9dc2229` | **24/24 detected as declared**; 28 declared tests pass unmutated first; 0 refused opens; inventory (69 files) and the checkout unchanged; sparse sandbox without `artifacts/`; no stray process (`report.json` `09e36972…`) |
| Manifest generation under the allow-list guard | exit 0, 0 refused; only 37 named JSON/source files + v9 source opened |
| Request previews under the allow-list guard | exit 0, 0 refused; only 10 named JSON files opened (F record, v7 S snapshot, approved aggregate and selected-N, ledger, manifest, 4 refit seal JSONs); real wrappers rendered in temporary directories |

- **Suite runner.** The suite pair is `refit_v9/suite_pair_v3.sh`; it fails unless both runs pass
  with 0 refused opens.
- **Test kinds.** The tests are labelled `predicate`, `structural` or `composed`.
  - **Composed tests** go through the real entries: the trainer's `_resolve_save_path` and
    `_official_test_allowed`, the launcher `main()` and `run_cell`, `run_terminal_test_cell`, the
    T entry's `main()`, the real raw evaluator and BIO stage, and the supervisor's ledger choice.
    Only processes, leases and model encoding are stand-ins.
  - **Structural tests:** the AST wiring of the trainer's terminal call; the moved block compared
    line by line with v8's (`git show 3dd1c02:train_siglip2.py`); the closure delta.
- **Battery coverage (24 mutants).** Each disables one whole condition or step:
  - the R policy (withhold, falls-through, not-a-refit);
  - the trainer start-up and terminal wiring;
  - the R post-chain skip and the test-output check;
  - the F acceptance and summary checks;
  - the refit mapping, seal-authority and approved-seal checks;
  - the run/smoke scope;
  - the v7/v8 manifest refusal;
  - the T admission's `withheld` check;
  - attempt exclusivity and ordering;
  - the T post-chain;
  - the T entry's approval, checkpoint and configuration checks;
  - the R/T ledger;
  - raw → post-BIO;
  - the refit tag arm.

## 6. Failed and superseded attempts (kept)

Attempt 1 at `bdaba58` (evidence in `refit_v9/attempt1_bdaba58/`):
- **Suite pair:** 1231 passed / 4 skipped in both runs, 0 refused.
- **Battery v13: 22/24 as declared.**
  - **RX13** (a stage-S scope) refused every line, because a stage-S scope carries no F pin. It
    weakened nothing, so it could not test the scope. It is replaced by a run-scope line approving a
    smoke, with a new test.
  - **RX16** (non-exclusive attempt) was caught by its tests, but for another reason than declared:
    the retry re-ran the chain and failed at record publication. It now declares the case where a
    failed first attempt would let a silent retry succeed.
- **Superseded:** manifest r1 `19acd584…` and its request previews (`18945918…`, `c6f7e829…`).
  They were never submitted.
- **First suite run on the edited tree:** 80 failed, 25 errors. Causes:
  - the synthetic test manifest lacked the new contract pin;
  - the new test file did not exist yet;
  - the policy parsed a legacy fixture cell id (`fixture-cell`); **fixed**: models without anchors
    no longer parse cell ids;
  - three source-text tests pointed at moved code.

## 7. Changes to existing tests (disclosed)

| Test | Change |
|---|---|
| `test_anchor_confirm_launcher.py::test_the_refit_path_refuses_an_anchor_arm` | replaced: an anchor refit now renders (stage R's) but never with a lambda override |
| `test_anchor_confirm_launcher.py` manifest builders (2) | add the refit-contract pin |
| `test_phase3_selection_matrix.py` (2) | read the moved terminal block, and the R-aware completion literal |
| `test_anchor_lambda_stage.py` (2 v8-delta tests) | the v7→now delta adds `train_siglip2.py` and `p0_protocol.py`; the exact v8→v9 delta is pinned in the new file |

## 8. Limits stated plainly

- **Equivalence of T and the legacy path is by construction, not by a real run.** The T entry calls
  the same `run_official_test` on the configuration the trainer saved right before its terminal
  block (`config.pt`). `extract_code` already reloaded the saved checkpoint in the legacy path. The
  first real exercise would be the T smoke.
- **The T entry's process environment** is the trainers' clean base (passthrough set, start-up
  runtime variables, geometry, GPU UUID). It is not re-attested against an R cell's expected child
  environment, since the GPU may differ. The T snapshot records the launcher's environment
  fingerprint.
- **The R run's carried-admission request** can be rendered only after the R smoke snapshot exists.
- **The budget** (80,000 device-seconds, own ledger) and the wall limits are proposals.

## 9. Proposed next steps (each a separate approval)

1. **`stage-R-smoke`** for request `cd2c4164…`: manifest `b7e9264a…`, freeze `5165f5dc…`.
   - One cell, Flickr25K N4 seed 42, one epoch, namespace `ancRsmk9`, tmux `ancRsmk9_v9`, one free
     GPU, ledger `/home/yschoi/gdna_anchorRT_ops`, result root `/home/yschoi/gdna_anchorRT_result`.
   - It includes the full historical admission of the four refit seals.
   - Command, under the supervisor `--stage stage-R-smoke --planned-cells 1`:
     ```
     env -C /data/yschoi/gdna_anchor_refit_v9 -u PYTHONPATH GDNA_NUM_SEMANTIC_PARTS=5 \
       /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/anchor_confirm_supervisor.py \
       --manifest artifacts/anchor_confirmation/authority_manifest_v9r2.json \
       --manifest-sha256 b7e9264a7a933e3efd30125628f638bc5c5304f32ccae2055b730ad382b215f6 \
       --stage stage-R-smoke --planned-cells 1 --watch-path /home/yschoi/gdna_anchorRT_result -- \
       /home/yschoi/.conda/envs/dna_hashing/bin/python scripts/phase3_selection_matrix.py \
       --anchor-confirm refit --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v9r2.json \
       --anchor-manifest-sha256 b7e9264a7a933e3efd30125628f638bc5c5304f32ccae2055b730ad382b215f6 \
       --result-root /home/yschoi/gdna_anchorRT_result \
       --input-seal cifar10:refit=/data/yschoi/gdna_p3exec_seals/cifar10.refit.input-seal.json \
       --input-seal flickr25k:refit=/data/yschoi/gdna_p3exec_seals/flickr25k.refit.input-seal.json \
       --input-seal nuswide:refit=/data/yschoi/gdna_p3exec_seals/nuswide.refit.input-seal.json \
       --input-seal mscoco:refit=/data/yschoi/gdna_p3exec_seals/mscoco.refit.input-seal.json \
       --namespace ancRsmk9 --smoke --only flickr25k:4:anchors:42 --epochs 1 --gpus <GPU> \
       --anchor-approval-section <N>
     ```
     The request does not bind a GPU index, so the GPU is chosen at launch from `nvidia-smi`.
2. **`stage-R-run`:** the carried request, rendered after the smoke.
3. **`stage-T-smoke`** on the R smoke checkpoint.
4. **`stage-T-run`** on the completed R run receipt.
