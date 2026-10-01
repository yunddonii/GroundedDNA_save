# Anchor model — candidate F record and the R/T proposal (audit §742.2, §743)

**PREPARATION ONLY.** The record below is a candidate for audit review. It is not the accepted
freeze. It authorizes no stage R or T, no official-test access, no source change or deployment,
and no downstream work. Nothing was trained, loaded, extracted or evaluated to prepare it.

## 1. What is submitted

| Item | Path (branch `arch-exp-2026-09-anchor-lambda`) | SHA256 |
|---|---|---|
| Candidate F record | `artifacts/anchor_confirmation/ancF_candidate_v1.json` (schema `anchor-freeze-candidate/1`, written once, mode 0444) | `5165f5dc9fcfb8334270bc16aa9816d09db67b03a04abae7ff846d5235bdca1d` |
| Builder and validator | `artifacts/anchor_confirmation/f_v1/build_f_candidate.py` | `746463a69f784674c9030c38d4b6b581c3076e73d52b629b4a133d49dbe7ad27` |
| Check output | `artifacts/anchor_confirmation/f_v1/check.txt` | `c6d1e52cced133c3fd013e6ff24a2868156cc6b84b3c2419421d5aa79b2b63cd` |
| Self-test output | `artifacts/anchor_confirmation/f_v1/self_test.txt` | `014cee1f53664302abd33e8f799997439c54e54be23c45f6fd791b25fc78cbcc` |

The builder is evidence tooling. It is not a manifest-closure member and not a production entry
point. The v7 and v8 closures are unchanged (§6).

## 2. The candidate F record

**Per dataset** (anchors fixed for all four datasets: the user's decision recorded in audit §709):

| Dataset | Final N | Transport / balance / text | Top-p min/max | Joint | Gumbel | Validated recipe (seed-42 S cell, 353 typed fields) | Lambda authority |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | 4 | 0.15 / 0.02 / 0.05 | 0.3 / 0.7 | 0.02 | off | in `ancS7_snapshot_23e07630…`, cell `cifar10\|N=4\|P=0.3,0.7\|JD=0.02\|stage=select\|seed=42\|axis_center=anchors` | its own approved values, kept under the Flickr-first scope |
| Flickr25K | 4 | 0.15 / 0.02 / 0.05 | 0.6 / 0.95 | 0.02 | off | same snapshot, Flickr seed-42 cell | the stage-L decision: no candidate qualified |
| NUS-WIDE | 4 | 0.15 / 0.02 / 0.05 | 0.4 / 0.8 | 0.05 | off | same snapshot, NUS seed-42 cell | its own approved values, kept under the Flickr-first scope |
| MS-COCO | 39 | 0.05 / 0.02 / 0.10 | 0.6 / 0.95 | 0.03 | off | same snapshot, COCO seed-42 cell | its own approved values, kept under the Flickr-first scope |

- **Each dataset entry binds:**
  - the complete typed recipe, by its sealed digest and the snapshot file digest;
  - the seed-42 S record and the seed-43/44 D records (record, receipt, cell id, recipe digest and
    terminal validation mAP@R);
  - the three probe envelopes, with hits, total and ratio;
  - the stage-D summary (retrieval and code-to-own-axis, mean and sample SD).
- **The lambda authority is stated per dataset.** The three non-Flickr datasets keep their own
  approved lambdas because the preregistered Flickr-first scope stops when Flickr's choice does
  not move. They were not tested with Flickr candidates. MS-COCO keeps transport 0.05 and text
  0.10; nothing was copied from Flickr.

**Validation lineage: generation v7 for all four datasets.**

| Authority | Path | SHA256 |
|---|---|---|
| Manifest v7 | `/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/authority_manifest_v7.json` | `c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128` |
| Contract v3 | `…/docs/ANCHOR_CONFIRMATION_CONTRACT_v3.md` | `e6c4978e6d30559e01c2ee34ae2b4d39c01717557ee16c9e96f85980aa8d3efc` |
| Stage S receipt / snapshot (approval §725, request `bf2d3f56…`) | `ancS7_sweep_complete.json` / `ancS7_snapshot_23e0763089474895.json` | `59157687…` / `6f2c0352…` |
| Frozen N record (reduction §726.3, accepted §727) and its sources | `ancS7_selected_n.json` / `ancS7_select_sources.json` | `5cda7adb…` / `0cfab951…` |
| Stage D receipt / snapshot (approval §727, request `6ef9d4ea…`) | `ancD7_sweep_complete.json` / `ancD7_snapshot_64f311500273a3ed.json` | `4272eeac…` / `e383ff68…` |
| Probe sources (approval §730, request `748851142a87…`) | `ancP7_probe_sources.json` | `39b304e9…` |
| Stage-D summary (reduction §731) and its sources | `ancP7_decision.json` / `ancP7_decide_sources.json` | `28b10a4c…` / `19678606…` |

**Lambda-check authority: generation v8, linked separately.** It is not merged into any S/D
reduction.

| Authority | SHA256 |
|---|---|
| Manifest v8 r2 | `58e69ae1…` |
| L contract `docs/ANCHOR_LAMBDA_CONTRACT_v1.md` / L proposal `ANCHOR_LAMBDA_PROPOSAL_L_v1.md` (v7 tree) | `34a62bfa…` / `fe375fdf…` |
| L receipt / snapshot (approval §739) | `0991f0a0…` / `d7c8874f…` |
| L sources / official L decision (reduction §741, verified §742) | `e0c32688…` / `22014e6b…` |

**Incumbent recipe authorities** (top-p and joint): `p3rfB_refit_aggregate.json` `b4f3b0df…` and
`selected_n.json` `2bf6133d…`.

All paths and full digests are in the JSON. The record also states what F does not claim: global
optimality, equivalence or significance; anchor superiority; R/T or TODO completion; any smoke,
old-model or Gumbel-ON result as evidence.

## 3. Validation evidence

**Reads.** The builder reads JSON/CSV metadata and the audit ledger's text only:
- 21 pinned files, 12 S/D records and 12 probe envelopes.
- The recipe digest helper is loaded by file path from the v7 tree's
  `dna_utils/scientific_recipe.py` (stdlib imports only), the code that sealed those recipes.
- It never opens `config.pt`, a checkpoint, a criterion payload, an extraction or a cached array,
  and it imports neither the launcher nor the reducer.

**Checks** (each tagged; all pass on the real inputs, 0 problems; `check.txt`):

| Tag | What is required |
|---|---|
| pins | each of the 21 inputs has its full pinned digest |
| membership | exactly 4 datasets × seeds 42/43/44 at the frozen N, anchors only; no record or probe reused; stage-R membership = the same 12 |
| generation | the N record and D summary name manifest v7 and contract v3; the D summary names the N record; one fixed-architecture authority (anchors, all four datasets) in the N record, D summary and L decision; S and D campaigns ran under manifest v7 |
| links | record bytes = sources = receipt cell; receipt → snapshot (semantic digest); the N record's seed-42 evidence = the S record; the D summary consumed each record and probe at these digests and reports exactly their per-seed values |
| N | record N = N record = D summary = audit table = terminal checkpoint epoch |
| recipe | the sealed recipe digest, recomputed, = binding = record anchor block; seeds 43/44 differ from seed 42 in `random_seed` only |
| table | the audit's cross-check fields plus axis `anchors`, Gumbel off, `siglip_cos`, text supervision on, 5 slots / 5 codebooks; top-p and joint = the approved aggregate |
| probe | each envelope names its coordinate, record and manifest v7; hits/total = ratio; 2000 decisions |
| approval | §725, §727, §730 and §739 lines still stand verbatim in their sections and name these requests (and, for D, the N record) |
| lambda | v8 Flickr-only decision; its v7 history pins = this lineage; `changed_axes` empty, all winners null, control equal; its recipe = Flickr's sealed lambdas; it consumed sources `e0c32688…` |

- **Positive control** (`--self-test`, `self_test.txt`). Ten in-memory mutants were each caught by
  their declared check, and the unbroken inputs pass: a duplicate seed, a Flickr lambda copied to
  MS-COCO, the D summary at another N, seed-43 recipe drift, an L decision that changes an axis, a
  probe of another record, changed record bytes, a missing approval line, a replaced pinned input,
  and a fixed architecture missing a dataset.
- **Ledger state.** The record was written at ledger digest
  `d8719a7100c3e805f186dab59a43b67ea49217a389ad15926305395195ebb83a` (through §743).

## 4. The R/T plan

### 4.1 Membership and inputs

- **Twelve NEW scratch full-train anchor refits:** CIFAR-10, Flickr25K and NUS-WIDE at N4, and
  MS-COCO at N39, each with seeds 42/43/44.
- **S/D checkpoints are not these refits.** The `p3rfB` incumbent refits remain the descriptive
  comparator only (contract v3 §7.4–§7.5).
- **Inputs.** The approved refit input seals, built in the historical tree:

  | Dataset | Seal | SHA256 |
  |---|---|---|
  | CIFAR-10 | `/data/yschoi/gdna_p3exec_seals/cifar10.refit.input-seal.json` | `943bb953f44a9d943e66944c15ec0d82bcef95d9769eea243c8b22e3993dec9f` |
  | Flickr25K | `/data/yschoi/gdna_p3exec_seals/flickr25k.refit.input-seal.json` | `71506fd19966216f2fd25f952db7e16e58d980eebc9eaa4d9a5488e63022c83c` |
  | MS-COCO | `/data/yschoi/gdna_p3exec_seals/mscoco.refit.input-seal.json` | `a9d49e5db9d4b2b934a9340b69bf986fae435c7d712783c379d28ed8f0f177e4` |
  | NUS-WIDE | `/data/yschoi/gdna_p3exec_seals/nuswide.refit.input-seal.json` | `05b7d24b7ede41c989ad3c6d510769949c5cc6d33ec843c34cc52151ee2992a4` |

  The seals cover the full designated train rows and the `trainOnly` whitening file. They are
  admitted through the pinned historical verifier, as in generation v6, and never edited.

### 4.2 The recipe mapping (S → R), from the parsed arguments

The approved recipe's own refit is the reference. Per dataset, I compared the parsed `args.txt`
of the anchor stage-S seed-42 cell (v7) with that of the approved incumbent refit `p3rfB` seed 42.
This is a text comparison only. In all four datasets the only training-setting difference is
`axis_center` (`anchors` vs `None`). Every lambda, top-p, joint, Gumbel, geometry and text-path
field is equal.

Stage R therefore keeps the F recipe and changes exactly the protocol fields the approved refit
changes:

| Field | S/D (validated) | R (as the approved refit) |
|---|---|---|
| `epoch` | 60 | N+1 (5; COCO 40) |
| `lr_schedule_horizon` | 60 | unset → `epoch` = N+1 |
| `sinkhorn_schedule_horizon` | N+1 | unset → `epoch` = N+1 (same value) |
| `stop_after_epoch` | N | N |
| `val_split_ratio` | 0.1 | 0.0 (full designated train) |
| `selection_mode` | `select` | `refit` |
| `keep_final_checkpoint` | True | False (`final_epoch_eval` keeps the terminal weights) |
| `final_epoch_eval` | False | True |
| `text_whiten_npz` | `…_optTrain_localOnly.npz` | `…_trainOnly_localOnly.npz` |
| input seal / split identity | `<ds>.stage1` | `<ds>.refit` |
| `random_seed` | 42 (S), 43/44 (D) | 42 / 43 / 44 |
| `axis_center` | `anchors` | **`anchors`** (the only change from the incumbent refit) |

- **Disclosed, not changed.** The LR horizon is 60 in S/D but N+1 in R, which is the approved D2
  protocol: the incumbent's stage-1 cells and refits differ in the same way. §743 requires the
  N+1 training/LR/Sinkhorn horizons to be preserved. The terminal checkpoint is epoch N in both
  stages.
- **Composition.**
  - `-ev -s` comes from the wrapper's last line, after `EXTRA_ARGS` (Flickr wrapper line 153), so
    `evaluation=True` cannot be overridden off by the launcher.
  - `FINAL_EPOCH=1` expands `${FINAL_EPOCH:+--final_epoch_eval}` (line 96).
  - `evaluation=True` is also a field of every sealed S/D recipe, so leaving it on changes nothing
    in the recipe.

### 4.3 The access boundary today (§743)

The legacy refit command runs R and T as one unit:
- The trainer is isolated during training. `final_epoch_eval` + `stop_after_epoch` → no test
  loader and no training-time evaluation.
- Its terminal block (`train_siglip2.py` ~1552) then calls `extract_code(args)`. That reloads the
  saved checkpoint and builds the official query/db datasets, because `-ev` is set and no
  validation split is active.
- The launcher post-chain (`_run_refit_postprocess`) then runs `extract_train_split`,
  `eval_cell_bioproj`, `pairwise_nmi` and `seal_cell_analysis`.

So an R-only permission cannot use the legacy command unchanged. Clearing or zeroing `FINAL_EPOCH`
is not a fix: §743 shows it reopens the test loader and training-time evaluation, and a nonempty
value still expands. Dropping `-ev` would change a typed recipe field. Both are rejected.

### 4.4 Proposed implementation: a new generation v9 (not started)

**Where.** A new worktree `/data/yschoi/gdna_anchor_refit_v9` on branch
`arch-exp-2026-09-anchor-refit`, branched from v8. The v7 and v8 closures, their seals and the
historical tree are not edited.

**Two execution scopes, each with its own approval line.**

1. **`stage-R-run` (training only).**
   - The launcher composes exactly the legacy refit command plus `axis_center=anchors`, sealing the
     R recipe = the F recipe with only the §4.2 fields changed. A test checks this field by field
     against the F record.
   - The campaign binding carries `anchor_stage: refit` and **no T authority**.
   - **Trainer gate.** When the sealed campaign binding names an anchor refit, the terminal
     official-test block additionally requires a T authority bound to:
     - the approval line digest;
     - the run identity;
     - the terminal checkpoint SHA256 and runtime epoch N.
   - Without that authority the trainer constructs no official-test dataset and saves the terminal
     checkpoint and runtime sidecar. It then writes `r_complete.json` (`official_test: withheld`),
     exits 0, and the launcher never calls the post-chain.
   - **R ends at:** terminal checkpoint (epoch N), sidecar, `log.csv`, `args.txt`, campaign
     evidence and `r_complete.json`.
   - Training isolation is unchanged: `final_epoch_eval` and `stop_after_epoch` keep blocking the
     test loader and training-time evaluation.
2. **`stage-T-run` (official test, once per checkpoint).** Approved separately, naming the R
   receipt and the 12 terminal checkpoint digests. Per cell:
   - verify the record, checkpoint SHA256, runtime epoch N and sealed recipe digest;
   - run the same `extract_code(args)` path in a separate process (it already reloads from disk);
   - run the unchanged post-chain in order, fail-stop: train extraction, raw and post-BIO
     evaluation (15 bases, GC [6, 9], max run 3), NMI, analysis seal;
   - write every output once, and refuse if any T output already exists, so a failure or resume
     cannot duplicate extraction or metric publication.
   - T results are reported descriptively against `b4f3b0df…` and select nothing.

**Legacy behavior is unchanged.** The gate triggers only on a sealed anchor-refit binding, so
legacy refits and historical `--recipe` replay behave as before.

**Tests before admission** (composed paths, through `main()`, synthetic inputs):
- an R cell without T authority stops before any test access (positive control: the same cell
  with authority reaches extraction);
- R isolation holds with full-train rows (`val_split_ratio 0`, `trainOnly` whitening);
- T admission binds the terminal checkpoint SHA256 and runtime epoch, and refuses a mismatch;
- a failing subprocess stops the chain;
- after a failure or resume there is no second extraction or metric file;
- the legacy refit path and historical replay are unchanged.

Then a mutation battery, the full suite under the guard, and one real R one-cell smoke on its own
approval: a new launcher path needs a real cell. The smoke is not evidence.

**Budget** (planning arithmetic only). The 12 incumbent `p3rfB` cells, R and T fused, took 66,415
s of wall time: CIFAR-10 at N19 16,228 s, Flickr25K 1,088 s, NUS-WIDE 24,962 s, MS-COCO 24,138 s.
Anchor CIFAR-10 trains to N4, not N19. A ceiling and ledger will be proposed with the v9 package.

## 5. TODO view

F is not R/T, and R/T is not downstream completion. The changed-model TODO matrix
(`docs/ANCHOR_MODEL_TODO_MIGRATION_v2.md` `32e7b88f…`, bound in the record) still governs:
- items 2–19, D4 and D6, and the paper each need their own dependency-complete evidence and
  approval on the new model;
- old-model analyses and unchanged baselines certify none of them;
- the master-audit F09/F12/F18 requirements, the D4 provenance limits, the D5/D6 metric holds and
  the missing human responses stay open;
- no human response is fabricated and nothing is completed by relabelling.

## 6. State at submission

- **v8 closure.** All 64 files still equal manifest `58e69ae1…`, and the bounded head-clean check
  passes. The new files are outside the closure.
- **v7 tree.** Untouched.
- **Both GPU ledgers unchanged:** L `477e447c…`, S/D/probe `986bdcd1…`. No GPU process.

## 7. Decisions requested

1. Accept, revise or reject the candidate F record `5165f5dc…`.
2. The R/T design of §4.4: two scopes, the trainer gate on sealed anchor-refit bindings, and the
   separate T entry. If accepted, I prepare the v9 package (code, tests, manifest, requests)
   without running anything.
