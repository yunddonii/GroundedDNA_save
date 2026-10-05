# Anchor model — stages R and T, contract v1 (generation v9)

**Status: NON-EXECUTABLE until the audit approves an exact request for a named scope.** Prepared
under audit §742–§744 (preparation of v9 code, synthetic tests, contract, manifest, request
rendering and bounded metadata preflight only). Stage R and stage T are separate approvals.

## 1. Authority and scope

- **The frozen model (F).** The record
  `/data/yschoi/gdna_anchor_lambda_v8/artifacts/anchor_confirmation/ancF_candidate_v1.json`, SHA256
  `5165f5dc9fcfb8334270bc16aa9816d09db67b03a04abae7ff846d5235bdca1d`, **together with** its
  acceptance in audit §744.1.
  - The launcher reads the record at that digest. Ledger section 744 — from its heading line to the
    next heading, stripped — must hash to
    `13ef776bc1485b3917253e51ee4aa99879a8f1f30f5539e0d1e224d3faaa1c56`, the corrected text audit §745
    names, and must name the record's path and digest.
  - Free-standing tokens or changed wording are not the acceptance (audit §746.3). The mutable ledger
    as a whole is not pinned, and the F bytes are never edited.
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
4. **Per cell, fail-stop.** Every producer runs under a T boundary snapshot (audit §747.2). It holds
   the stage-R campaign's sources, inputs and admitted seals, recomputed at T start and required
   unchanged since stage R, together with THIS stage's environment and leased GPUs. The launcher's
   unchanged `verify_snapshot` re-checks it:
   - before the entry;
   - at the entry-to-train-extraction transition;
   - after each of the four post-chain producers.

   At each of those boundaries the launcher also re-checks the consumed cell's own stage-R inputs
   (config.pt, the terminal checkpoint, its runtime witness) against their pins (audit §756).
   The producers receive the same three pins and the cell's admitted terminal epoch.
   `extract_train_split.py`, the post-chain producer that loads the configuration and the model,
   verifies them itself before any deserialization: the config is read once (the verified-buffer
   rule), and the checkpoint and witness are read once and bound at the admitted terminal epoch
   (`dna_utils.runtime_state.verified_runtime`; audits §759–§760). The weights load from the verified
   bytes and the epoch resolves from the verified witness object; neither file is reopened by path
   before encoding, and the effective epoch and checkpoint must be the admitted ones before any
   dataset access. It re-checks all three files before it writes. Without the pins it is unchanged.

   A failed check starts no further producer. In order, per cell, the launcher then:
   - re-checks the runtime witness and that no test output exists;
   - **reserves the attempt** — `<ns>_attempt_<R tag>.json`, exclusive, durable, retained on
     failure — **before any model or test construction**. A second attempt (duplicate, concurrent or
     retry) refuses even if no output exists; recovering an attempted cell needs a new authorization;
   - runs the **T entry** `scripts/anchor_terminal_test.py`. Before the test, it re-verifies:
     - the attempt bytes and request, the standing stage-T line, and the generation;
     - the checkpoint bytes; the runtime witness at the terminal epoch the request allows (N, or the
       smoke's last epoch);
     - the trainer's campaign evidence (pinned bytes, this cell, this sealed recipe, the witness's
       own) and the config.pt bytes;
     - that no test output exists.

     It then **claims the entry exclusively and durably** (`<ns>_entry_<R tag>.json`, never removed;
     audit §746.1). It reads config.pt **once**: it hashes the bytes, deserializes them, and requires
     every typed field of the cell's sealed recipe (from the stage-R snapshot its receipt binds) and
     this cell's campaign binding. It builds the arguments from that same object through the shared
     resume helper's flat-layout step (no second read; audit §754.2) and re-checks the effective
     fields. It reads the terminal checkpoint and its runtime witness **once** and binds them: both
     digests are the pins, the witness (parsed from those bytes) names that checkpoint and is at the
     admitted terminal epoch (audits §759–§760). Only then does it call
     `terminal_official_test.run_official_test`, the trainer's own former terminal block, moved
     unchanged except that it passes this binding to `extract_code`: the evaluation-cache and
     whitening resolution, `extract_code` (query, db) — weights from the verified bytes, epoch from
     the verified witness, the effective epoch and checkpoint required to be the admitted ones before
     any dataset access — and the **raw** evaluation `evaluation_siglip2_base.json`
     (`bio_project=False`). It re-checks the three files immediately before and after the test;
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

**Revision transition r5 → r6 (audit §759.2; proposal, the audit decides).** A T receipt admits
only stage-R work of its own manifest, and that check stays. The r5 stage-R smoke (`ancRsmk9`,
§758) is therefore the r5 diagnostic of the R path on real inputs; its receipt is never consumed by
an r6 T, and it is not a full R campaign. Under r6, in this order, each with its own line:
1. `stage-R-smoke` (`ancRsmk9r6`): the same one cell. Its seal admission is either carried from the
   r5 smoke's snapshot (the same four seals and verifier, if the audit accepts that carry across
   revisions, as v7 carried v6's) or performed in full again.
2. `stage-T-smoke` (`ancTsmk9`) on that receipt: the repaired consumers on a real checkpoint before
   any full campaign.
3. `stage-R-run` (`ancR9`), carried from the r6 smoke snapshot; then `stage-T-run` (`ancT9`).

## 8. Reporting

- Per cell and per dataset (mean and sample SD, n = 3): raw and post-BIO base-Hamming mAP@R on the
  official split (R = 1000 for CIFAR-10, 5000 for the others; 15 bases, GC count [6, 9], maximum
  run 3), with NMI as reported by the post-chain.
- Reported next to the incumbent's `b4f3b0df…` values. Descriptive only; no test, no switch.

## 9. Resources (proposal; the audit sets them)

- **Ledger.** Its own root `/home/yschoi/gdna_anchorRT_ops`, budget 80,000 device-seconds for all
  smokes, R and T together, of both revisions (accepted by §758.2 as the maximum cumulative
  accounting envelope, not as permission to spend it on any stage).
  - Planning reference: the twelve incumbent `p3rfB` cells, R and T fused, took 66,415 s wall in
    total. CIFAR-10 there trained to N19; here to N4.
  - Both settled ledgers (S/D/probe `986bdcd1…`, L `477e447c…`) are carried unchanged and are not an
    R/T allowance.
- **Wall limits.** R run 8 h on 4 GPUs; T run 8 h on 4 GPUs; each smoke 3 h on 1 GPU.
- **Supervision (audit §747.1).** `--planned-cells` is the logical cell count (membership and
  storage). A T cell is five managed producers, so stage T's attempt ceiling and missed-attempt
  allowance are cells × 5 (`CHILDREN_PER_CELL`, tied by a test to the T chain). A sixth child of a
  T cell, or a second of any other cell, is still excess.
- **Storage.** Unchanged rule: 10 GiB floor + 0.75 GiB per unfinished cell (an incumbent refit
  directory holds 0.64–0.77 GiB).
- **Roots and namespaces.**
  - Result root: `/home/yschoi/gdna_anchorRT_result` (fresh).
  - Records: this tree's `artifacts/anchor_confirmation`.
  - Namespaces: `ancRsmk9` (r5, used by §758's smoke), then under r6 `ancRsmk9r6`, `ancTsmk9`,
    `ancR9`, `ancT9`.
- **Failure rules.**
  - On a refusal or failure: settle the ledger and report.
  - No retry, no deletion of a run directory, attempt, reservation or record, and no repair of
    inputs or sources under the same approval.

## 10. What remains open

R/T complete only stage R/T. Every downstream TODO item (`docs/ANCHOR_MODEL_TODO_MIGRATION_v2.md`,
in the v7 tree) needs its own evidence and approval, and so does paper regeneration. The
master-audit F09/F12/F18 requirements and the open provenance limits stay as they are.
