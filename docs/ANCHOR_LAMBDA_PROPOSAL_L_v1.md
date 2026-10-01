# Anchor model — stage L proposal: the TODO 13–15 lambda checks (response to audit §731.3)

**Status: a preregistration proposal for audit review. Nothing here is executable.**
- No lambda training, smoke, GPU work or exploratory checkpoint use has happened.
- No source has changed. This file is a document in the v7 worktree, not a manifest member.
- Everything below is fixed **before any L score exists**: the dataset scope, candidates, seeds, rule,
  thresholds and budget. Contract v3 §7.6 step 3 requires this.

## 1. Starting point (bound inputs)

| Input | SHA256 | Use in L |
|---|---|---|
| Generation manifest v7 | `c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128` | the generation of the incumbent evidence |
| Frozen N record `ancS7_selected_n.json` | `5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff` | N_S(d) for every L cell |
| Official stage-D summary `ancP7_decision.json` (§731 reduction) | `28b10a4c850f508bacf4d9402cda8c458fe92d2bfc8b616e14a1a6ce9866cc32` | the incumbent's seed-42 score and seed spread |

**The incumbent of each dataset** is the fixed anchor model at its frozen N with its approved lambdas.
The values are read from the sealed recipes of the D cells (all three seeds agree), not from wrapper
text. Its seed-42 score is the stage-S cell, and the range is max − min over seeds 42/43/44 in the
D summary.

| Dataset | N_S | `lambda_wasserstein` | `lambda_bu` | `lambda_text_hash_ntxent` | Incumbent seed-42 mAP@R | Seed range |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 4 | 0.15 | 0.02 | 0.05 | 0.7823275443473103 | 0.0535532958307553 |
| Flickr25K | 4 | 0.15 | 0.02 | 0.05 | 0.7641936888306327 | 0.02819158958924184 |
| NUS-WIDE | 4 | 0.15 | 0.02 | 0.05 | 0.7225163591400635 | 0.00669749994409552 |
| MS-COCO | 39 | **0.05** | 0.02 | **0.10** | 0.6340656450539125 | 0.012385900327336441 |

**MS-COCO's approved recipe differs on two of the three axes.** Its wrapper default (`WASS=0.05`,
`THASH=0.10`, the approved F2 recipe) is what every approved MS-COCO cell records. Contract v3 §7.6
names its candidates against the Flickr values (0.15 / 0.02 / 0.05), so it does not fit MS-COCO as
written. §2 proposes how to apply it, as a decision for the audit.

## 2. Axes and candidates (one factor at a time, seed 42, at N_S(d), anchors on)

Every candidate is the incumbent recipe with **exactly one** lambda changed. Everything else stays
the same: the frozen N, the horizons, the split, top-p, `lambda_codon_joint`, `axis_center=anchors`,
`hash_target_mode siglip_cos`, no Gumbel, and every other field of the 353-field typed recipe. The
admission renders the incumbent and the candidate at each coordinate. It requires them to differ in
that one field alone, as the S/D admission does for `axis_center`.

**CIFAR-10, Flickr25K, NUS-WIDE** (contract v3 §7.6, as written): five candidates each.

| Axis (TODO) | Incumbent | Candidates |
|---|---:|---|
| transport `lambda_wasserstein` (13) | 0.15 | 0.30, 0.50 |
| codebook balance `lambda_bu` (14) | 0.02 | 0 |
| text-code contrast `lambda_text_hash_ntxent` (15) | 0.05 | 0.025, 0.10 |

**MS-COCO: decision requested.**
- **Recommended (option A): the contract's value set, nearest the incumbent.** It stays at five
  candidates.
  - transport: 0.15 and 0.30. These are the two contract values nearest the incumbent 0.05; 0.50
    is dropped.
  - balance: 0.
  - text: 0.025 and 0.05. These are the contract values other than the incumbent 0.10.
- **Option B: the contract's literal alternates.** That means transport 0.30 and 0.50, balance 0,
  and text 0.025. Text 0.10 would be the incumbent itself and is dropped, so B has four candidates.
  It jumps the transport weight 6–10× and skips 0.15, the value the other three datasets use.

Either way at most five alternates per dataset, no Cartesian grid, no other coefficient
(`lambda_xmodal_commit` and the rest stay at their approved values). That gives **20 candidate cells**
(option A) or 19 (option B).

**Continuity controls (recommended, a decision for the audit).**
- **What they are:** four extra cells, one per dataset — the incumbent recipe itself at seed 42 and
  N_S, rerun in the new generation (§5).
- **Gate:** the reducer requires each control's terminal score to equal the v7 stage-S seed-42 score
  exactly; otherwise that dataset's L decision refuses.
- **Why:**
  - L compares new-generation candidates against old-generation incumbent scores, and the controls
    prove that the generation change altered nothing in training.
  - A same-seed repeat reproduces the score to full precision on this trainer (p3gD/p3gE Flickr N4:
    0.7635532117270416 twice), so exact equality is the right test.
  - Without controls the same claim rests on byte identity of the trainer closure (§5), which is
    checked in any case.
- **Cost:** 2,273 s (0.63 GPU-h), most of it MS-COCO (§8).

## 3. The selection rule (preregistered; the p3lamA rule of ledger §536.1 applied to the anchor model)

- **Score.** Raw base-Hamming mAP@R at the cell's own terminal epoch N_S(d), on the train-only
  validation split. It is parsed from the pinned terminal `log.csv` row, exactly as in stage S
  (R = 1000 for CIFAR-10, 5000 otherwise).
- **Threshold.** T(d) = max(0.002, the incumbent's seed range from the D summary):

  | Dataset | T(d) |
  |---|---:|
  | CIFAR-10 | 0.0535532958307553 |
  | Flickr25K | 0.02819158958924184 |
  | NUS-WIDE | 0.00669749994409552 |
  | MS-COCO | 0.012385900327336441 |

- **Qualifying.** A candidate c of dataset d qualifies iff score(c) − incumbent_42(d) > T(d), with a
  strict inequality.
- **Per axis.** Among the qualifying candidates of one axis, the highest score wins. On an exact
  score tie, the value closer to the incumbent value wins, then the smaller value. An axis with no
  qualifying candidate keeps its incumbent value.
- **Per dataset.** The new recipe is the incumbent with every axis winner applied. If no axis changes,
  the dataset keeps the incumbent, and its v7 S/D/probe records stand for the freeze.
- **If any axis changes** (contract v3 §7.6 step 4): N is reselected for that dataset's new recipe on
  seed 42 over {4, 9, 19, 39}, followed by D (seeds 43/44) and its three probes. These run in a new
  generation with its own contract, request and approval, and they are the affected datasets only.
  The old-lambda records stay historical. A recipe that combines two winning axes was not tested
  together in L, and that fresh S/D/probe validation is what tests it.
- **What the rule is not.** It is not a test of significance, equivalence or superiority. It finds
  no global optimum and no interactions. It is not official-test evidence. The probes are not part
  of L: no L cell is probed, and alignment does not enter the rule. The anchors stay fixed for all
  four datasets whatever L shows.
- **Missing or invalid evidence.** The reducer refuses and writes nothing on any of these:
  - a missing, extra, duplicated, failed or non-finite cell;
  - a cell whose recipe differs from its rendered incumbent in anything but its one lambda;
  - another generation's record;
  - a mismatched continuity control, if controls are adopted.

  Nothing is imputed or re-run silently. A re-run needs a new namespace and approval.

## 4. Dependencies and order

1. The audit's acceptance of the stage-D result (§731.3).
2. This proposal's decisions (§10).
3. The v8 package (§5–§7) and its review.
4. A one-cell L smoke, under its own approval and never evidence. A new launcher path needs a real
   cell before a batch.
5. Stage L, under its own approval.
6. The L reduction, under its own approval.
7. Any reselection (§3), each under its own proposal and approval.
8. The freeze F; R and T after it, each separately approved.

Nothing in L reads the official test split.

## 5. Source generation and historical handoff

- **The v7 tree stays frozen.**
  - `/data/yschoi/gdna_anchor_confirm_v1` (branch `arch-exp-2026-09-anchor-confirm`, source
    `59477d8`, manifest `c0612963…`) keeps every manifest member at its pinned bytes.
  - Later commits there add records or documents only.
  - The v7 S/D/probe/decision artifacts are never moved or rewritten. Their historical verification
    keeps running in that tree with the v7 commands, and nothing in v8 needs a changed tree to
    verify them.
- **L is generation v8, in a new worktree** (`/data/yschoi/gdna_anchor_lambda_v8`, a new branch
  from the v7 head).
  - **Source changes are limited to:**
    - the launcher: a new anchor stage `lambda` rendering §2's cells, with each candidate's one-field
      difference checked against its rendered incumbent;
    - the reducer: a new `lambda` reduction implementing §3;
    - the supervisor: stages `stage-L-smoke` and `stage-L-run`, with the L budget and an L ledger
      root (§8);
    - the manifest builder: generation v8 with its v7 predecessor pins;
    - tests.
  - **Unchanged and checked byte-equal to the v7 manifest pins by a test:** the trainer, the model,
    the dataloaders, the configuration, the four wrappers, the recipe module, the runtime-environment
    module, the seal tooling and the probe.
- **How v8 consumes v7 evidence: by digest, not by replay.**
  - The L reducer binds the frozen N record (`5cda7adb…`) and the D summary (`28b10a4c…`) at their
    audit-accepted digests.
  - It checks their internal fields: the v7 generation, `n_selected`, and the per-seed retrieval
    values with their mean and range.
  - It does not re-run the v7 reducer in the v8 tree. This follows the lesson of the 2026-09-15
    lambda confirmation: a protocol-source edit breaks cross-tree replay, so a new campaign binds to
    a digest-pinned record.
- **Inputs.** The same four stage-1 seals. Admission is the carried ancS6 full admission
  (`ef5a3e8d…`, stats-only recheck) as in S/D, or a full admission if the audit prefers it.
- **Outputs.**
  - Run directories: `/home/yschoi/gdna_anchor4_result` (namespace `ancL8`).
  - Records: the v8 worktree's `artifacts/anchor_confirmation/`.
  - Operations ledger: a new root, `/home/yschoi/gdna_anchorL_ops`, so L's budget is its own (§8).
    The closed S/D/probe ledger (19,264.613316638395 s) is not reused or reset.

## 6. Required tests (for the v8 package)

- **Plan membership.** It is exactly §2's cells per dataset (and the controls, if adopted). Every
  other set refuses: an extra or missing candidate, a second axis moved, another N or seed, a control
  arm, an official-test split.
- **Admission.** Each candidate's typed recipe differs from the rendered incumbent in its one lambda
  only.
  - Positive cases cover each axis for each dataset.
  - Negative cases are two fields changed, a wrong value, and a changed non-lambda field.
- **Trainer closure.** It is byte-equal to the v7 manifest pins. Negative: one changed byte.
- **Reducer.**
  - It applies the threshold with a strict inequality, the highest-wins rule, the tie order and the
    multi-axis combination.
  - Refusals: missing, extra and duplicate cells, a non-finite score, a wrong generation, a D summary
    or N record at another digest, and a control mismatch.
  - All of this is driven through `main()` on production-shaped records.
- **Supervisor.** The L stages, their budget and wall limit, and the separate ledger root.
- **Battery and smoke.**
  - A mutation battery in a sandbox copy, with declared markers and a positive control that kills
    the whole condition.
  - The full suite of the 16 existing anchor test files at the v8 commit.
  - The one-cell smoke (§4) before stage L.

## 7. Commands (forms; exact digests come with the v8 package)

- **Plan:**
  `python scripts/phase3_selection_matrix.py --anchor-confirm lambda --namespace ancL8 --anchor-arms anchors --anchor-selection <N record> --anchor-selection-sha256 5cda7adb… --anchor-decision <D summary> --anchor-decision-sha256 28b10a4c… --plan`.
  With the execution arguments it prints the request digest.
- **Smoke:** the same command with `--smoke --only flickr25k:4:anchors:42:lambda_wasserstein=0.30
  --epochs 1 --gpus <1>`, under `anchor_confirm_supervisor.py --stage stage-L-smoke`. It is never
  evidence.
- **Stage L:** the same command with `--run --gpus <4 free>` and the four input seals, the manifest
  pins, `--admission-authority` (if carried) and `--anchor-approval-section`. It runs under the
  supervisor `--stage stage-L-run --planned-cells 20` (24 with controls) in tmux `ancL8_v8`, one
  dataset stream per GPU.
- **Reduction (CPU):**
  `python scripts/anchor_confirm_decision.py lambda --sources <L sources> --sources-sha256 … --selection … --decision 28b10a4c… --manifest <v8> … --out ancL8_lambda_decision.json`.
- **Approval lines.** The same form, with new scopes `stage-L-smoke` and `stage-L-run` naming the v8
  manifest, the selection, the D summary and the request.

## 8. Budget, wall time and storage (separate from the S/D/probe ceiling)

**Per-cell planning figures** are the anchor cells' own `wall_seconds` from S and D. Each stream's
first cell is counted at its slow first-in-stream S value, which includes the cold cache load:

| Dataset (N_S) | First cell | Later cells | 5 candidates | + control |
|---|---:|---:|---:|---:|
| CIFAR-10 (4) | 1188 s | 171 s | 1,872 s | 2,043 s |
| Flickr25K (4) | 274 s | 130 s | 794 s | 924 s |
| NUS-WIDE (4) | 1493 s | 246 s | 2,477 s | 2,723 s |
| MS-COCO (39) | ≈ 3,030 s (1,726 + a 1,300-s cold load) | 1,726 s | 9,934 s | 11,660 s |
| **Total** | | | **15,077 s (4.19 GPU-h)** | **17,350 s (4.82 GPU-h)** |

- **Proposed L ceiling: 21,600 s (6.0 GPU-h)** for the smoke plus stage L, in their own ledger, with
  the same accounting rules (process lifetime plus allowance, 520-s headroom on four GPUs). The
  contract's planning bound for L was 1.2–9.3 GPU-h.
- **Wall time.** The MS-COCO stream sets it: about 2.8 h, or 3.2 h with its control. The proposed
  `stage-L-run` limit is 5 h; the smoke's is 2 h.
- **Storage.** 24 cells × 0.75 GiB + the 10 GiB floor = 28 GiB at the first dispatch, on `/`, which
  had 373.6 GB free on 2026-10-01. Records are JSON in the v8 worktree on `/data`. Nothing is
  deleted to make room.

## 9. Not in this proposal

- Running anything.
- Probes of L cells.
- Changing N inside L.
- Moving two lambdas in one cell.
- Any other coefficient.
- Official-test data, R/T or the freeze.
- Reselection after a change: that is its own proposal (§3).

## 10. Decisions requested

1. The scope (all four datasets, seed 42, at N_S, anchors on), the rule and the thresholds of §3, and
   the candidates of §2 for CIFAR-10, Flickr25K and NUS-WIDE.
2. **MS-COCO's candidates**: option A (recommended: transport 0.15/0.30, balance 0, text 0.025/0.05)
   or option B (transport 0.30/0.50, balance 0, text 0.025).
3. **Continuity controls**: include the four as a reducer gate (recommended), or rely on the
   trainer-closure byte identity alone.
4. The generation plan of §5: a v8 worktree, and v7 evidence consumed by digest without cross-tree
   replay.
5. The L budget of §8 (21,600 s, own ledger, 5-h run limit).
6. Authority to build the v8 package (§5–§7) as non-executable preparation: source, tests, battery,
   manifest and the smoke and run requests. A smoke request would follow, and stage L after it.
