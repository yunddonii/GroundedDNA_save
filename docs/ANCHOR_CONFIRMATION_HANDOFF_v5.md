# Anchor Confirmation — Handoff, generation v5 (response to audit §709–§712)

**Status: stage-S readiness package for audit review. No scientific execution.**
- No training, smoke, probe or refit ran. There was no full seal verification, GPU lease,
  namespace reservation or run-directory claim, and no real binary was deserialised.
- The v4 package stays as history, unchanged: handoff `934cd51e…`, addendum `6bdd2460…`, manifest
  `5a4481f4…`, request `033b6979…`, contract v2 `26ebe2c1…`. Its request is superseded (§709.1) and
  now refuses by version (§1, item 1).

## 0. Acknowledgment and what this package is

- Read in full before this revision: §709, §710, §711 and §712.
  - §710 adds no requirement. Its finding that the parser refuses all five scopes (no approval
    line exists) is the expected state.
  - §711 verifies the user's decisions and the core of `b0b9d6e`, and requires the lambda-check
    correction answered in §2. Its caveat on the later comparison is now in contract v3 §1.
  - §712 finds that correction installed at `2f79cb0` and asks for the regenerated final
    package. This is that package. Its four-seal inventory (531.69 GB) and free-space figures
    agree with addendum v3. As §712.1 notes, an affected-dataset re-selection after a lambda
    change needs its own generation and review: this generation's launcher runs all four
    datasets.
- **The user's design decisions (2026-09-27), submitted before any campaign result exists.**
  No S or D cell of any generation has ever run.
  - `axis_center=anchors` is fixed for all four datasets (the §709 decision).
  - **Anchors only.** S = 16 cells (4 datasets × N ∈ {4, 9, 19, 39} × seed 42), D = 8 cells
    (seeds 43/44 at the frozen N), 12 probes. The control arm `none` is rendered by the admission
    at every coordinate, to prove each anchor cell is the approved recipe with `axis_center` alone
    changed, and is never run.
  - The anchor-versus-incumbent comparison comes after R/T, on the official test, against the
    approved `p3rfB` results. It is descriptive, not a paired S/D experiment and not a causal
    anchor effect (audit §711.1).
  - Run directories and the operations ledger are on the root filesystem (`/home/yschoi`).
  - The 12.5 GPU-hour ceiling is kept; no budget change is requested (§3).
  - One probe population for all four datasets: the first 500 validation rows (§4).
- **A superseded draft, disclosed** (also seen by §712.3). Before I read §711, I rendered a
  manifest (`16ba96c6…`), plan (`8cb9af87…`) and request (`c87fbad3…`) at `b0b9d6e`. They were
  never submitted or committed, and they approve nothing. This package, which carries the §711.3
  correction, supersedes them. The final request differs from that draft only in the `manifest`
  field.
- **This revision.** Branch `arch-exp-2026-09-anchor-confirm`, source commit `2f79cb05354e3c5a290d4e26616dd7242d8c10dd`.

  | Item | Path | SHA256 |
  |---|---|---|
  | Generation manifest v5 (59 files) | `artifacts/anchor_confirmation/authority_manifest_v5.json` | `84e94e2f1897e9910df2aa0352e4737f87dd485f018a644b5be23c1b4e036284` |
  | Contract v3 (manifest member) | `docs/ANCHOR_CONFIRMATION_CONTRACT_v3.md` | `bd99ab0d8810d1247cc1716246e736056592ce913f701393981b7408db626799` |
  | Operational addendum `anchor-confirm-ops/3` | `docs/ANCHOR_CONFIRMATION_OPS_ADDENDUM_v3.md` | `f270229e610d470d0ff457b4ba0b9ae270d9f2ab18aa742068ebd0562fed76f5` |
  | Changed-model TODO migration matrix v2 (§709.5, §711.3) | `docs/ANCHOR_MODEL_TODO_MIGRATION_v2.md` | `32e7b88feffa9da9e4f7d527e4819e3d617267f1220b6b1c1a7008910642833e` |
  | Launcher (manifest member) | `scripts/phase3_selection_matrix.py` | `abd902c650286d585b755196117becb9a345950674fa4e3b597bd58baf63e9e5` |
  | Reducer (manifest member) | `scripts/anchor_confirm_decision.py` | `79fbe1318a35e15ed501faec09ff9fafc9bd1fe4652938c850972e7351da3bf5` |
  | Supervisor (manifest member) | `scripts/anchor_confirm_supervisor.py` | `b5eecf58cbbc9d7d03463e4ea7a35ac107842aa6657a3b66e3770528cbdd1dd0` |
  | Stage-S plan | `artifacts/anchor_confirmation/plan_select_ancS5_v5.txt` | `f2cef91467c56fd134002e7ef32f4f56b2e1c499cc24c63e78c83c09acd48144` |
  | **Stage-S request** | `artifacts/anchor_confirmation/request_preview_ancS5_v5.txt` | **`bd3117a9108ee558dfb27f9b429806a62092d23736febd7c0c0f8c858e4882c7`** |

  Migration v1 (`056f79ad…`) stays unchanged as history.

## 1. §709.1 — the binding instruction, item by item

1. **A new generation, nothing repinned in place.**
   - New: contract v3, launcher version `anchor-confirm/2` (campaign kind `anchor_confirmation_v2`),
     request schema `anchor-confirm-request/2`, reducer `anchor-confirm-reducer/3`, probe schema
     `anchor-confirm-code-axis/3`, manifest v5, namespaces `ancS5`/`ancD5`/`ancSmk5`.
   - Kept as history, byte-unchanged: contract v2 and every v1–v4 manifest, plan, request, addendum
     and handoff.
   - A version-1 approval line, request or record refuses by version. Tests: the v4 package's own
     line (manifest `5a4481f4…`, request `033b6979…`) approves nothing; a v1 line with otherwise
     correct fields refuses; a v2-reducer N record refuses.
2. **`axis_center=anchors` for the final model of all four datasets.**
   - The captured rule `adopted = "anchors" if (retrieval_ok and axis_ok) else "none"` is removed.
     The reducer has no adoption rule, no threshold and no reuse path.
   - Stage D writes a descriptive summary per dataset (retrieval and code-to-axis per seed, mean,
     sample SD), the fixed-architecture declaration, and the stage-R membership: 12 scratch refits,
     4 datasets × seeds 42/43/44, `axis_center=anchors`. The record says that membership is
     provisional until the final recipe freeze (§2).
   - The replay of the frozen N record requires the fixed-architecture declaration. A record
     without it, or with three datasets, refuses even when stage D and the probes were approved
     for that very record.
   - Missing or invalid evidence still refuses, as before: every refusal test of the reducer and
     the probe is kept.
3. **Preserved.** Gumbel OFF (`--no_gumbel_softmax` in every cell; the protocol fields require
   `use_gumbel_softmax=False`); text-free deployment in the probe; split isolation (train-only
   validation 0.1/42, no official-test load); stage-specific schedules; typed recipe identity;
   immutable outputs; the historical digest handoff (`b4f3b0df…`, `2bf6133d…`, `p3lamA` receipt).
   - No unrelated arch-exp change is imported. `git diff 7d740a9..2f79cb05354e3c5a290d4e26616dd7242d8c10dd` touches only the files
     listed in §7.
   - N and the lambdas are settled before any freeze or official-test access (§2).
4. **Updated together, with the tests §709.1 names.**
   - **Contract / schema / version:** above.
   - **Request and cell enumeration:** `ANCHOR_DATASETS` has four datasets and
     `ANCHOR_RUN_ARMS = ("anchors",)`. A control arm, a per-dataset pair list or a plan without a
     dataset refuses. The request's GPU count must equal its dataset streams, with distinct GPUs.
   - **Trainer admission:** `train_siglip2.py` is unchanged. It admits the sealed recipe of its
     cell (the arm from the cell id) before claiming the run directory. The one recipe change is the
     reviewed alternate CIFAR-10 literal (below).
   - **Consumer admission:** the reducer and the probe accept only the anchor coordinates of the
     four datasets, only campaign receipts of this generation, and only the §4 population.
   - **Negative scores with anchors still fixed, all four datasets present:** a stage-D world whose
     every probe has 0 hits and whose retrieval scores are all poor reduces to anchors on all four
     datasets, 12 stage-R rows, and no `none` anywhere. A stage-D source without CIFAR-10 refuses.
   - **Stale three-dataset requests refused:** above, plus a changed request with GPUs `0,1,2` or
     `0,1,2,2`.
   - **Exact wrapper-to-effective-recipe round trips:** for each dataset × both arms, the ACTUAL
     wrapper composition (rendered with synthetic paths) repeats exactly the seven reviewed
     destinations; the first occurrences are the reviewed literals; the payload's protocol fields
     equal the contract's; the config round trip reproduces the sealed fields.
   - **CIFAR-10's actual wrapper and protocol.**
     - K = 64 and mAP@1000. One test holds `DATASETS["K"]`, `MAP_R_CUTOFF`, the evaluation's
       `MAP_AT_R_BY_DATASET`, the protocol's `codebook_size` and the train rows to one value per
       dataset.
     - The CIFAR-10 wrapper passes `--lambda_codeword_codon_sinkhorn "${CCS:-0.1}"` and the
       launcher sets `CCS=0.1`. The first literal is therefore 0.1, admitted as a reviewed
       alternate for that destination only. The 0.0 override is still required, and the protocol
       binds the effective 0.0.
     - All 61 approved CIFAR-10 run directories in `/data/yschoi/gdna_p3exec_result` record 0.0
       (`args.txt`, text only).
     - A test proves an effective 0.1 refuses.
   - **Other dataset-specific values** (contract v3 §3): the top-p window and joint weight come from
     `b4f3b0df…`/`2bf6133d…`, read by pinned digest (the plan prints them: CIFAR-10 N 19, top-p
     0.3–0.7, joint 0.02; Flickr25K N 4, 0.6–0.95, 0.02; NUS-WIDE N 4, 0.4–0.8, 0.05; MS-COCO
     N 39, 0.6–0.95, 0.03). The train/validation rows are the seals' recorded counts (§4).
5. **The stage-S readiness package and the migration matrix:** this handoff, contract v3,
   addendum v3, the plan and request, and migration matrix v2. The freeze order is fixed before S
   is admitted (§2).

## 2. §711.3 — the lambda checks are required; the recipe lineage

Contract v3 §7.6 and migration v2 §2 now say the same thing:
1. **Required, not optional.** TODO 13–15 are required train-only checks of the anchor model
   before the final recipe freeze, one factor at a time at N_S(d):
   - `lambda_wasserstein` 0.30/0.50 vs 0.15;
   - `lambda_bu` 0 vs 0.02;
   - `lambda_text_hash_ntxent` 0.025/0.10 vs 0.05.

   The anchor-only choice does not waive them, and the old-model `p3lamA` results do not transfer.
   They get their own preregistered proposal: dataset scope, seeds, decision rule (the `p3lamA`
   rule is the stated template) and budget are fixed before any L score. The design is bounded (at
   most five alternates per dataset, seed 42) and its budget is separate from the S/D allowance.
   Planning bound: ≤ 20 cells, about 1.2–9.3 GPU-h.
2. **No cycle.** The order is S → D/probes → L → F → R → T → downstream. L follows S/D and
   precedes F. Downstream analyses follow F and their actual R/T dependencies. S waits for none
   of L, R/T or downstream work.
3. **If a lambda changes for dataset d:**
   - N is reselected for d's new recipe on seed 42, in a new generation whose cells are the
     affected datasets only.
   - d's stage D and three probes are regenerated at that N, from records of that same generation.
   - d's old-lambda records stay historical, not final-model validation. Unchanged datasets keep
     the v5 records.
   - No reduction merges two generations; each reducer already refuses another generation's
     records.
   - The freeze record binds, per dataset, one generation's frozen N record and stage-D decision
     record. It is written before any R/T or official-test access.
   - The v5 decision record's stage-R membership is labelled provisional in the record itself.

## 3. §709.2 — cells, inputs and resources

- **Cells.** S 16, D 8, probes 12 (anchors only; §0). The plan renders and admits all 16 stage-S
  coordinates with the real wrappers, with the control rendered at each for the axis-alone check.
- **The CIFAR-10 stage-1 seal** `/data/yschoi/gdna_p3exec_seals/cifar10.stage1.input-seal.json`
  (file SHA256 `3e9bc7b902333663973e6b782590b018b84b7c67b11b118ece9111be6c7662fb`) joins the other three, whose digests are unchanged from v4. The request
  pins its path and file digest. The launcher's full verification and the trainer's input
  admission bind its source and runtime population exactly as for the other three. Its recorded
  split: 5000 designated train, 4500 optimization, 500 validation.
- **Four GPUs.** One stream per dataset. A request with three GPUs cannot be formed.
- **Full-verification I/O:** 531.69 GB over the four seals (addendum v3 §1). The elapsed time is
  not measured; the historical 82-min figure is an estimate.
- **Storage at every stage:** 22 GiB free at the first S dispatch, 16 GiB at the first D dispatch,
  13 GiB while running. `/` has 364.9 GiB free (2026-09-27). S writes about 9.8 GiB and D about
  4.9 GiB (0.61 GiB per cell, measured on historical cells of all three sizes).
- **Cross-stage charged time.** One append-only ledger for S, D and the probes, never reset. The
  ops/2 ledger and result root never existed, so no earlier charge is dropped.
- **Budget: feasible within the kept 12.5-GPU-h ceiling.** Planning arithmetic: S 3.50, D
  0.47–3.73, probes ≈ 0.4; total ≈ 4.4–7.6 GPU-h. No grid or seed is reduced to fit, and no budget
  change is requested. L and later stages have their own budgets.
- **Operations** (paths, GPU count, wall limits, as §711.2 asks): addendum v3.

## 4. §709.3 — the probe population

- **Policy:** the first 500 rows of each run's own train-only validation split, ascending dataset
  index, for all four datasets. 4 local slots × 500 = 2000 strict decisions.
- **Feasibility, from the seals** (`split_identity.counts.heldout_train_validation`, read from the
  JSON only): CIFAR-10 500, Flickr25K 500, NUS-WIDE 1050, MS-COCO 1000.
- **Bound throughout:**
  - contract v3 §8.2;
  - source: `PROBE_IMAGES = 500`, `PROBE_LOCAL_SLOTS = 4` and the declared `PROBE_POPULATION`;
  - the probe request (the population is a field of the canonical request the probe approval
    names);
  - row identity (`row_ids_sha256`, one per dataset across its seeds, with the split identity);
  - the envelope (`population`, `n_images`, `total`);
  - the reducer, which requires the exact declaration and the exact counts.
- **Tests.**
  - 500-row positives through the trainer's own split carve (`val_split.carve_val_indices`) with
    synthetic label arrays of each dataset's sealed size: exactly the first 500 ascending
    validation rows, none of them optimization rows.
  - 499 validation rows refuse, and a split other than 0.1/42 refuses.
  - Wrong counts refuse: 499 images, 1999 decisions, the v2 512/2048 counts, and hits above the
    total.
  - A wrong population declaration refuses: none, 512/2048, other rows.
  - A v2 (512-row) probe schema refuses.
  - An approval of the same records with another population does not cover the probes.
  - The strict tie rule and the matched-rows check are unchanged.
- It does not alter any exploratory measurement.

## 5. §709.4–§709.5

- **§709.4.** The v4 supervisor's accounting is unchanged in code. Only two constants changed: the
  default operations root and the per-stage GPU maximum (3 → 4). The accepted lifecycle, the
  disclosed one-attempt-per-cell / no-retry premise and the unresolved policy are kept
  (addendum v3).
- **§709.5.** Migration matrix v2:
  - the proposed spec amendment defining `axis_center` (kept distinct from the anchor-EMA loss and
    `codon_text_anchor`), not applied to the spec;
  - the freeze order and lineage rule of §2;
  - a per-item matrix (TODO 1–19, D4, D6, the paper) with the retained evidence and its scope, new
    work, dependencies, cells and seeds, planning budget, output roots and acceptance checks;
  - the items the architecture decision does not close.

  The historical statuses stay in `docs/TODO_reexperiments.md` (`a56f3e9d…`), unchanged.

## 6. Evidence (CPU only)

All CPU only (`CUDA_VISIBLE_DEVICES=` empty, `GDNA_NUM_SEMANTIC_PARTS=5`); details and report
digests in addendum v3 §7.
- **Full run at `2f79cb0`** (the manifest's commit), 14 files, tracked tree clean: **836 passed,
  1 skipped** (the opt-in real-artifact test), 344 s, rc 0. Each earlier commit of this generation
  also passed in full.
- **Mutation battery v7b** (15 declared mutants of the new semantics, detached sandbox): **15/15
  detected as declared**, at `b0b9d6e` and again at `2f79cb0`. Tree unchanged, and no fixture
  process remained.
- **The first attempt v7** (at `4caa76e`) is kept: 14/15. It showed one test that did not isolate
  its check (MX13); the test was fixed and given a positive control.
- **Request.** The printed canonical request recomputes to `bd3117a9108ee558dfb27f9b429806a62092d23736febd7c0c0f8c858e4882c7`. It has 16 cells (4 datasets ×
  N 4/9/19/39, seed 42, anchors only), 4 GPUs, the four seal pins (each file digest checked) and
  the result root `/home/yschoi/gdna_anchor4_result`.
- **Plan.** The plan admits all 16 coordinates: each anchor cell and its rendered control differ in
  `axis_center` alone, with the seven reviewed overrides. Its first section is byte-identical to
  the request preview's.
- **Manifest.** 59 files; every recorded digest equals the bytes at `2f79cb0`.

## 7. Files changed from generation v4

`git diff --stat 7d740a9 2f79cb05354e3c5a290d4e26616dd7242d8c10dd`: 13 files.
- Source: `scripts/phase3_selection_matrix.py`, `scripts/anchor_confirm_decision.py`,
  `scripts/anchor_confirm_code_axis.py`, `scripts/anchor_confirm_supervisor.py`,
  `scripts/anchor_confirm_manifest.py`, `dna_utils/scientific_recipe.py`.
- Tests: the launcher, recipe, reducer and supervisor files.
- Documents: contract v3, migration v1 and v2.

`train_siglip2.py`, the wrappers, the seals and every other closure file are byte-identical to v4.

## 8. Which prior evidence remains applicable

- **Launcher lifecycle, storage rule and supervisor accounting.** Their code is unchanged apart
  from the constants named in §5, so §707.2's and §709.4's scoped acceptances apply to those
  paths, as do the earlier ML/MS/MV mutation results. Only the v7 mutants were run against these
  bytes.
- **Everything the four-dataset change touches** (arm plan, cells, request, recipe alternate,
  reducer, probe) is re-tested in this generation (§6).

## 9. Requested decision and scope

- Review generation v5: contract v3, addendum v3, migration v2 and the stage-S request.
- A stage-S approval, if given, is a ledger line
  `ANCHOR-CONFIRM-APPROVAL version=anchor-confirm/2 scope=stage-S-run manifest=84e94e2f1897e9910df2aa0352e4737f87dd485f018a644b5be23c1b4e036284 request=bd3117a9108ee558dfb27f9b429806a62092d23736febd7c0c0f8c858e4882c7`.
- Full verification, leases, smoke, training and every later step stay closed until then. Stage L
  gets its own proposal after S/D.
- Ledger at submission: SHA256 `8c5c01e427b4364bbc12ef3e495655fd4ce9a2ff6f1b7de7ba4b0aaa0f3efa1b`, 54946 lines, last section §712.

## 10. Real files touched in this revision (nothing deserialised)

- The manifest inventory: the two pinned JSON authorities, the `p3lamA` receipt bytes and the
  closure files.
- The stage-S `--plan` and the request preview (the four seal JSON files by digest; no payload).
- Text and stat reads only:
  - the four stage-1 seal JSONs (split counts and inventory sizes);
  - `args.txt` of the 61 approved CIFAR-10 run directories;
  - the `p3lamA` decision JSON (its axes and rule);
  - `du -sb` of seven historical stage-1 cell directories;
  - `df` of `/` and `/data`.
