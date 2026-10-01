# Anchor model — stage L contract v1 (generation v8): the TODO 13–15 lambda checks, Flickr25K first

**Status: NON-EXECUTABLE until the audit approves this exact generation for a named stage.**
- Prepared under audit §731.3–§733: preparation only, no lambda training.
- It refines contract v3 §7.6 step 3 (`docs/ANCHOR_CONFIRMATION_CONTRACT_v3.md`). That contract
  stays the scope-bearing contract of the fixed anchor model, unchanged. This file binds stage L
  only.
- It supersedes the four-dataset campaign of the proposal `docs/ANCHOR_LAMBDA_PROPOSAL_L_v1.md`
  (`fe375fdf…`) where the two differ.

## 1. Scope: the user's decision (2026-10-01, answering audit §733.1)

- **Flickr25K first**, as the main TODO lambda section and the 2026-09-14 instruction order it.
- **No change on Flickr25K ends stage L.** All four datasets then keep their approved lambdas,
  and their v7 S/D/probe records stand for the freeze.
- **If Flickr25K's choice moves**, each other dataset needs its own checks before its recipe
  changes. That takes its own proposal, contract revision, generation, approval and budget. Every
  check is on that dataset's own incumbent; MS-COCO's starts at transport 0.05 and text 0.10, and
  the audit prefers proposal option A for it.
- **A Flickr winner is never copied** into another dataset's recipe.
- This is not an exemption from TODO 13–15. The final model stays the fixed four-dataset anchor
  model.

## 2. The incumbent (historical, bound by audit-accepted digests)

| Item | Value | Authority |
|---|---|---|
| Generation of the incumbent evidence | manifest v7 `c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128` | audit §722–§732 |
| Frozen N record | `ancS7_selected_n.json` `5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff`: Flickr25K N = 4 | §726–§727 |
| Official stage-D summary | `ancP7_decision.json` `28b10a4c850f508bacf4d9402cda8c458fe92d2bfc8b616e14a1a6ce9866cc32` | §731–§732 |
| Stage-S receipt (the seed-42 cell's sealed recipe) | `ancS7_sweep_complete.json` `5915768767e3fd12b28c3f09e1a071c26fd5cbddbbac15662c495b80c4876f6b` | §725–§726 |

- These files live in `/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/`.
- The reduction reads every one of them, the stage-S plan snapshot and the approved recipe
  authorities included, once through its read-once record. It re-verifies those bytes before it
  publishes (audit 734.2 item 2).
- Generation v8 reads them at these digests as **historical metadata** and never replays them:
  it neither re-runs the v7 reducer nor re-checks v7 approvals with v8 code. Their historical
  verification stays in the v7 tree.
- **Incumbent recipe:** Flickr25K, `axis_center=anchors`, N = 4, seed 42, with the approved lambdas
  `lambda_wasserstein` 0.15, `lambda_bu` 0.02 and `lambda_text_hash_ntxent` 0.05. The typed fields
  are those of the v7 stage-S seed-42 cell's sealed recipe.
- **Incumbent seed-42 score:** 0.7641936888306327 (raw base-Hamming mAP@R, R = 5000, train-only
  validation, terminal epoch 4).
- **Seed range:** 0.02819158958924184 over seeds 42/43/44, from the D summary.
- **Threshold:** T = max(0.002, range) = **0.02819158958924184**, unrounded.

## 3. Cells: one stream, Flickr25K, N = 4, seed 42, anchors

| Order | Cell | Change from the incumbent |
|---:|---|---|
| 1 | continuity control | none (zero scientific field differences) |
| 2 | transport 0.30 | `lambda_wasserstein` 0.15 → 0.30 |
| 3 | transport 0.50 | `lambda_wasserstein` 0.15 → 0.50 |
| 4 | balance 0 | `lambda_bu` 0.02 → 0 |
| 5 | text 0.025 | `lambda_text_hash_ntxent` 0.05 → 0.025 |
| 6 | text 0.10 | `lambda_text_hash_ntxent` 0.05 → 0.10 |

- **Unchanged in every cell:** N, seed 42, anchors, Gumbel off, the split (0.1, seed 42), the epoch
  and LR budget 60, Sinkhorn horizon N + 1, stop epoch N, `hash_target_mode siglip_cos`, the text
  path on, and every other field of contract v3 §6.
- **Rendered checks.** The admission renders the incumbent and each cell. A candidate must differ in
  its one lambda alone, at the declared typed value. The control must differ in nothing. The
  rendered incumbent, given the v7 cell's admitted Flickr25K seal authority, must equal the v7
  seed-42 cell's sealed fields exactly.
- **Not counted as a candidate:** the unchanged incumbent value.

## 4. The continuity control (a preregistered diagnostic gate, audit §733.2)

- It runs first. When it completes, its terminal score must **equal** the incumbent seed-42 score
  (0.7641936888306327) exactly.
- **A mismatch stops the stream before any candidate starts**, and the decision refuses. Nothing is
  repeated, the threshold is not relaxed, and the other run is never used as a more favourable
  reference.
- The control is not a lambda candidate and not a fourth seed. The comparator and the spread stay
  the v7 values above.
- **Limitation, stated before any L score.** A matching terminal scalar does not prove that training
  is unchanged. The old p3gD/p3gE Flickr coincidence proves no bitwise determinism across datasets,
  devices or a new source closure. Equality is a conservative fail-closed gate only. The evidence
  that training is unchanged is the source, input, environment and typed-recipe checks: the
  trainer closure is byte-equal to v7 (§7), and the recipes are rendered and checked (§3). A
  mismatch is a diagnostic failure to investigate, not evidence that a recipe or the architecture is
  worse.

## 5. The rule (unrounded; predeclared)

- **Score.** Raw base-Hamming mAP@R at the cell's terminal epoch 4, on the train-only validation
  split, from the pinned terminal `log.csv` row.
- **Qualifying.** A candidate c qualifies iff score(c) − 0.7641936888306327 > 0.02819158958924184,
  with a strict inequality on unrounded floats.
- **Per axis.** Among the qualifying candidates of an axis, the highest score wins. On an exact score
  tie, the value closer to the incumbent value wins, then the smaller value. An axis with no
  qualifier keeps its incumbent value.
- **Per dataset.** The new recipe is the incumbent with every axis winner applied.
- **If any axis changes:**
  - Flickr25K's N is reselected for the new recipe on seed 42 over {4, 9, 19, 39}, followed by D
    (seeds 43/44) and its three probes. That runs in a new generation with its own contract,
    request and approval.
  - Flickr25K's old-lambda records stay historical.
  - A combined two-axis recipe was not tested together in L; that fresh S/D/probe validation tests
    it.
  - §1's expansion obligation applies.
- **Not part of the rule:** probes, alignment, test data, significance tests and any other statistic.

## 6. Missing, failed or invalid evidence

The reduction refuses and writes nothing on any of these:
- a missing, extra or duplicated cell, or a failed or non-finite one;
- a seed other than 42 or an N other than 4;
- a dataset outside the scope;
- a recipe off its declared one-lambda difference;
- a record from another generation or campaign: the six cells of one decision must come from ONE
  approved stage-L campaign -- one receipt, one approval and request, a receipt that lists exactly
  these six cells, and a request that executes and declares exactly them (audit 734.2 item 1);
- a stale request;
- a mismatched or missing control;
- historical inputs at other digests.

A re-run needs a new namespace and approval.

## 7. Generation v8 and its source changes

- **Worktree:** `/data/yschoi/gdna_anchor_lambda_v8`, branch `arch-exp-2026-09-anchor-lambda`.
- **What changes:**
  - **The launcher.** Anchor stage `lambda`, with the scope and candidate tables and the historical
    pins. One declared lambda override is admitted on an anchor cell, and only in this stage.
    Requests and smoke cells carry the override label. The control gate sits in the stream. The new
    approval scopes are `stage-L-smoke` and `stage-L-run`.
  - **The reducer.** A `lambda` reduction. The existing S/D/probe paths keep their behaviour. A
    role parameter selects the L approval scope and cell identity, and it never admits S/D records
    into L or L records into S/D.
  - **The supervisor.** Stages `stage-L-smoke` and `stage-L-run` with the L budget, and their own
    ledger root.
  - **The manifest builder.** Generation v8, with v7 predecessor pins.
  - **The recipe module (`dna_utils/scientific_recipe.py`): a declared protocol-dependency change
    for audit decision.**
    - **Why.** The pinned Flickr25K wrapper passes `--lambda_wasserstein 0.15` and
      `--lambda_text_hash_ntxent 0.05` literally, and `--lambda_bu "$LBU"` (0.02). A candidate
      appended through `EXTRA_ARGS` therefore repeats its destination, and the v7 repeat rule refuses
      every repeat outside its seven reviewed destinations. It refuses both in the launcher's sealing
      and in the trainer's own check (`verify_trainer_recipe`). Without a change, no transport or text
      candidate can be sealed or started through the pinned wrapper.
    - **The change.** One new, separate table, `STAGE_L_REVIEWED_OVERRIDES`. Each of the three
      lambda destinations may repeat only as its exact Flickr25K wrapper literal followed by exactly
      one override, and **at most one such destination per argv**. That is one lambda per cell, at the
      recipe level, on both the launcher and the trainer side.
    - **Not an authorization boundary (audit 734.1).** The parser is shared and cannot tell a stage-L
      campaign from any other run, so the table authorizes nothing by itself. The stage, dataset,
      candidate value, typed recipe and approval are checked by the callers: the launcher's stage-L
      admission, the sealed recipe the trainer is held to, and the reducer.
    - **Accepted direction.** Audit §734.1 accepted this as a narrow exception to §733.3's
      byte-equality instruction, for this v8 generation.
    - **Unchanged:** the seven reviewed overrides, their alternates and every other repeat refusal.
      An S/D argv carries no lambda repeat, so its sealing and digest are unchanged.
    - **The alternative**, editing the wrapper, was not taken: wrapper bytes are pinned.
  - **This contract and a test file.**
- **Unchanged and tested byte-equal to the v7 manifest pins:** the trainer, the model, the
  dataloaders, the configuration, the four wrappers, the runtime-environment module, the seal
  tooling and the probe. Every other v7 closure member is byte-equal as well.

## 8. Inputs, budget and records

- **Inputs.** The Flickr25K stage-1 seal `e44b363a…` only. Proposed admission: the carried ancS6
  full admission (`ef5a3e8d…`), using the unchanged v7 guard and stats-only recheck, as in S/D.
  Alternative: a full historical-verifier admission of the Flickr25K seal alone (about 33 GB).
- **Budget.** The own ledger `/home/yschoi/gdna_anchorL_ops`. Ceiling **3,600 s (1.0 GPU-h)**
  covers the smoke, the control, five candidates, a failed attempt, the observation allowance and
  the stop headroom. The closed S/D/probe ledger (19,264.613316638395 s) is reported, not reused.
  - The figures are Flickr25K anchor cells' own wall times: 122.6–274.4 s per N = 4 cell.
  - Planned: 6 × 274.4 s = 1,646 s for the run, and ≤ 300 s for the smoke.
  - Allowance: planned cells × the longest window. Headroom: 130 s per supervised command.
  - Neither the ceiling nor a wall target is a completion bound.
- **Wall limits.** `stage-L-smoke` 2 h; `stage-L-run` 2 h.
- **Records.** The v8 worktree's `artifacts/anchor_confirmation/`. Run directories:
  `/home/yschoi/gdna_anchor4_result`, namespace `ancL8`.

## 9. Not covered

- Execution before approval.
- Other datasets.
- Probes of L cells.
- Changing N inside L.
- Moving two lambdas in one cell.
- Other coefficients.
- Official-test data, R/T and the freeze.
