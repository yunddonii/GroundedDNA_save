# Anchor model — stages R and T, contract v1 (generation v9)

**Status: NON-EXECUTABLE until the audit approves an exact request for a named scope.** Prepared
under audit §742–§744 (preparation of v9 code, synthetic tests, contract, manifest, request
rendering and bounded metadata preflight only). Stage R and stage T are separate approvals.

## 1. Authority and scope

- **The frozen model (F).** The record
  `/data/yschoi/gdna_anchor_lambda_v8/artifacts/anchor_confirmation/ancF_candidate_v1.json`, SHA256
  `5165f5dc9fcfb8334270bc16aa9816d09db67b03a04abae7ff846d5235bdca1d`, **together with** its
  acceptance in audit §744.1.
  - The launcher reads the record at that digest and requires ledger section 744 to name its path
    and digest as the accepted record.
  - Neither the record's filename nor its own "PREPARATION ONLY" text is approval.
  - Every stage-R/T approval line names the record as `freeze=`.
- **The model.** `axis_center=anchors` on all four datasets (user decision, audit §709), with each
  dataset's frozen N, top-p window, joint weight and lambdas, as F binds them.
- **What R and T are.**
  - **Stage R:** twelve scratch full-train refits of that model.
  - **Stage T:** the one official-test extraction and evaluation of each stage-R checkpoint.
- **What they are not.** R/T select nothing, change nothing in F, and reopen no selection. T is
  reported descriptively next to the approved incumbent (`p3rfB`, `b4f3b0df…`): not paired, not
  causal, no superiority claim.

## 2. Membership

| Dataset | Frozen N | Seeds | Cells |
|---|---:|---|---:|
| CIFAR-10 | 4 | 42, 43, 44 | 3 |
| Flickr25K | 4 | 42, 43, 44 | 3 |
| NUS-WIDE | 4 | 42, 43, 44 | 3 |
| MS-COCO | 39 | 42, 43, 44 | 3 |

- **Stage R** runs all twelve: four dataset streams on four GPUs, seeds in order within a stream.
  The S/D checkpoints are not these refits.
- **Stage T** evaluates exactly the twelve checkpoints of one completed stage-R run receipt.
- **Smokes** (diagnostic only, never evidence, never reported):
  - the stage-R smoke is one cell, `--only flickr25k:4:anchors:42 --epochs 1`;
  - the stage-T smoke is the T chain on that smoke checkpoint.

## 3. The recipe mapping (S/D → R)

A stage-R cell's typed recipe is the F record's validated recipe (the v7 stage-S seed-42 cell of
the dataset, 353 fields). It differs only in the contracted full-train fields below. Every value is
the approved incumbent refit's own (`p3rfB` `args.txt`, read for all four datasets).

| Field | S/D (validated) | R |
|---|---|---|
| `epoch` | 60 | N+1 |
| `lr_schedule_horizon` | 60 | unset (→ `epoch` = N+1) |
| `sinkhorn_schedule_horizon` | N+1 | unset (→ `epoch` = N+1) |
| `stop_after_epoch` | N | N |
| `val_split_ratio` | 0.1 | 0.0 (the full designated train) |
| `selection_mode` | `select` | `refit` |
| `keep_final_checkpoint` | True | False (`final_epoch_eval` keeps the terminal weights) |
| `final_epoch_eval` | False | True |
| `text_whiten_npz` | `…_optTrain_localOnly.npz` | `…_trainOnly_localOnly.npz` |
| `random_seed` | 42 (S) | 42 / 43 / 44 |
| the four `phase3_input_*` / split fields | the stage-1 seal | the refit seal (section 4) |

- **Unchanged.** `axis_center=anchors`, every lambda, top-p, joint, Gumbel off, `siglip_cos`, text
  supervision, geometry, K, and `-ev` (the wrapper's; set in S/D too).
- **Admission** (`refit_admission`, before any lease):
  - every rendered R recipe carries these values;
  - it differs from F's validated recipe in no other field (`REFIT_PROTOCOL_FIELDS`,
    `REFIT_SEAL_FIELDS`);
  - with the admitted seals, its input-authority fields are the refit seal's;
  - the rendered `none` control differs from it in `axis_center` alone.
- **Disclosed.** The LR horizon is 60 in S/D and N+1 in R, as in the approved protocol. The
  terminal checkpoint is epoch N in both.

## 4. Inputs

- **The approved refit input seals,** built in the historical tree and never edited:

  | Seal | SHA256 |
  |---|---|
  | `/data/yschoi/gdna_p3exec_seals/cifar10.refit.input-seal.json` | `943bb953…` |
  | `/data/yschoi/gdna_p3exec_seals/flickr25k.refit.input-seal.json` | `71506fd1…` |
  | `/data/yschoi/gdna_p3exec_seals/mscoco.refit.input-seal.json` | `a9d49e5d…` |
  | `/data/yschoi/gdna_p3exec_seals/nuswide.refit.input-seal.json` | `05b7d24b…` |

  They record the same six historical sources as the stage-1 seals (five foil producers and
  `val_split.py`).
- **Full admission.** The pinned historical verifier runs as an isolated child after those
  sources are checked in place, unchanged since generation v6.
  - Proposal: the **stage-R smoke** performs that full admission for all four refit seals (its
    request declares the whole stage).
  - The **stage-R run** then carries the smoke snapshot's admission (stats-only recheck), as stage S
    carried ancS6.
  - Nothing reads a seal payload during preparation.
- **Stage T** re-checks the stage-R campaign's admitted seals stats-only before and after every
  cell.

## 5. Stage R — execution and where it ends

The launcher (`--anchor-confirm refit`, `scripts/anchor_refit_stage.py`):
- admits the request;
- verifies the audit's `stage-R-*` line naming `manifest`, `freeze` and the request digest;
- verifies the seals;
- runs the ordinary anchor sweep, each cell sealed with its typed recipe.

**The access boundary**, fail-closed on three sides:
1. **Trainer start-up.** `p0_protocol.anchor_refit_withholds_official_test`: an anchor-model P0
   refit must be a sealed stage-R campaign cell (cell id `stage=refit`, sealed recipe). Otherwise
   the trainer refuses before its run directory exists. A missing, malformed or altered marker
   cannot fall through to the automatic test path.
2. **Trainer end.** The same policy withholds the official-test extraction and evaluation. No test
   dataset is constructed at any point of an R cell; training isolation (`final_epoch_eval` +
   `stop_after_epoch`) is unchanged.
3. **Launcher.** It runs no post-chain for an R cell and requires its run directory to hold none of
   the official-test outputs (`OFFICIAL_TEST_OUTPUTS`). The R record states
   `official_test: withheld`.

**R ends at:** the terminal checkpoint (epoch N), its runtime witness, `log.csv`, `args.txt`, the
trainer's campaign evidence and the typed `config.pt` check; then the receipt
`<ns>_sweep_complete.json`. An R receipt is never evidence of T.

## 6. Stage T — execution

The launcher (`--anchor-confirm test --anchor-refit-receipt … --anchor-refit-receipt-sha256 …`):
1. **Admits the stage-R receipt from JSON metadata.**
   - Generation, F record and stage-R approval, re-verified now.
   - Per cell: record, campaign binding, completion pins, trainer evidence and runtime witness at
     their digests (terminal epoch), the sealed recipe, `official_test: withheld`, and no test
     output present.
   - Exactly the twelve F coordinates (a smoke: one).
2. **Builds the request** (`anchor-terminal-test-request/1`). It binds the manifest (path and
   digest), F, the R receipt and approval, and per cell the run directory, record, terminal
   checkpoint SHA256, runtime witness and terminal epoch. It also binds the outputs (each created
   once), the chain and the GPU count.
3. **Verifies the audit's `stage-T-*` line** for that request digest, then leases the GPUs,
   reserves the namespace and publishes its snapshot.
4. **Per cell, fail-stop:**
   - re-checks the runtime witness, no test output and the seals;
   - **reserves the attempt** — `<ns>_attempt_<R tag>.json`, exclusive, durable, retained on
     failure — **before any model or test construction**. A second attempt (duplicate, concurrent or
     retry) refuses even if no output exists; recovering an attempted cell needs a new authorization;
   - runs the **T entry** `scripts/anchor_terminal_test.py`. Before the test, it re-verifies the
     attempt bytes and request, the standing stage-T line, the generation, the checkpoint bytes and
     runtime epoch, the absent outputs and the saved configuration (anchors, refit, cell id). Then
     it calls `terminal_official_test.run_official_test`, the trainer's own former terminal block,
     moved unchanged: the evaluation-cache and whitening resolution, `extract_code` (query, db) and
     the **raw** evaluation `evaluation_siglip2_base.json` (`bio_project=False`);
   - runs the unchanged refit post-chain, in order: `extract_train_split`,
     `eval_cell_bioproj --require-train` (which reads and validates the raw file before it writes the
     post-BIO one), `pairwise_nmi`, `seal_cell_analysis`;
   - checks every output with the legacy `assert_refit_outputs`, re-checks the checkpoint bytes, and
     publishes the T record.
5. **The T receipt** `<ns>_test_complete.json` is written only when every cell completed.

**Legacy behavior is unchanged.** A model without anchors keeps the fused refit (trainer terminal
block, then post-chain), now through the same function.

## 7. Approvals and order

| Scope | Pins | Prerequisite |
|---|---|---|
| `stage-R-smoke` | manifest, freeze, request | this generation; full seal admission |
| `stage-R-run` | manifest, freeze, request | the R smoke reviewed |
| `stage-T-smoke` | manifest, freeze, request | the R smoke receipt |
| `stage-T-run` | manifest, freeze, request | the completed R run receipt |

Each is a separate ledger line for one exact request. No line is inherited, and no smoke escalates
by itself.

## 8. Reporting

- Per cell and per dataset (mean and sample SD, n = 3): raw and post-BIO base-Hamming mAP@R on the
  official split (R = 1000 for CIFAR-10, 5000 for the others; 15 bases, GC count [6, 9], maximum
  run 3), with NMI as reported by the post-chain.
- Reported next to the incumbent's `b4f3b0df…` values. Descriptive only; no test, no switch.

## 9. Resources (proposal; the audit sets them)

- **Ledger.** Its own root `/home/yschoi/gdna_anchorRT_ops`, budget 80,000 device-seconds for both
  smokes, R and T together.
  - Planning reference: the twelve incumbent `p3rfB` cells, R and T fused, took 66,415 s wall in
    total. CIFAR-10 there trained to N19; here to N4.
  - Both settled ledgers (S/D/probe `986bdcd1…`, L `477e447c…`) are carried unchanged and are not an
    R/T allowance.
- **Wall limits.** R run 8 h on 4 GPUs; T run 8 h on 4 GPUs; each smoke 3 h on 1 GPU.
- **Storage.** Unchanged rule: 10 GiB floor + 0.75 GiB per unfinished cell (an incumbent refit
  directory holds 0.64–0.77 GiB).
- **Roots and namespaces.**
  - Result root: `/home/yschoi/gdna_anchorRT_result` (fresh).
  - Records: this tree's `artifacts/anchor_confirmation`.
  - Namespaces: `ancRsmk9`, `ancR9`, `ancTsmk9`, `ancT9`.
- **Failure rules.**
  - On a refusal or failure: settle the ledger and report.
  - No retry, no deletion of a run directory, attempt, reservation or record, and no repair of
    inputs or sources under the same approval.

## 10. What remains open

R/T complete only stage R/T. Every downstream TODO item (`docs/ANCHOR_MODEL_TODO_MIGRATION_v2.md`,
in the v7 tree) needs its own evidence and approval, and so does paper regeneration. The
master-audit F09/F12/F18 requirements and the open provenance limits stay as they are.
