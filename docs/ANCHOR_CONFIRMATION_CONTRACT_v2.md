# Anchor Confirmation v1 — Contract, generation v2

**Status: NON-EXECUTABLE until the audit approves this exact generation for a named stage.**
Prepared under audit §659, §678 and §686 (PREPARATION_AUTHORIZED; EXECUTION_NOT_AUTHORIZED;
FINAL_RECIPE_NOT_APPROVED), revised for §665–§690. It supersedes the v1 draft
`docs/ANCHOR_CONFIRMATION_CONTRACT_v1.md` (`418091ee…`), which stays unchanged as history. The
generation this contract describes is identified by the versioned manifest
`artifacts/anchor_confirmation/authority_manifest_v2.json`; that manifest pins this file, so its own
digest is named in the handoff, not here. Gates are stage-specific (§7.6): approving stage S does
not require, and does not imply, any approval of stage D, R or T. Every executed operation needs
its own approval line in the audit ledger (§11.3); the code refuses until that line exists.

## 1. Scientific question

Does routing with **axis-centred anchors** (`--axis_center anchors`: before the transport cost is
computed, each local routing anchor loses the per-image mean of the active local anchors; no
parameter and no loss term are added) raise the within-image code-to-own-axis alignment of the
multi-label models without a retrieval loss larger than one control seed SD, **judged on
train-only validation data**? For the recipe frozen by that train-only decision, what is its
official-test retrieval relative to the approved incumbent (descriptive only)?

## 2. Recorded decisions this contract carries

| Decision | Recorded | Where |
|---|---|---|
| Anchors are proposed for the multi-label datasets only; CIFAR-10 keeps the incumbent and its negative result is reported | user, 2026-09-23 | `docs/ANCHOR_SCOPE_AND_ACCEPTANCE_2026-09-23.md` §2 (branch `arch-exp-2026-09`) |
| Official-test criterion: anchored mean within one incumbent sample SD, raw and post-BIO — **descriptive role only** | user, 2026-09-23; role per audit §664.1, §666.1 | same document §4 |
| The final recipe is chosen by a predeclared **train-only** rule; model, N and scope are frozen before any new official-test access | user, 2026-09-26 (audit §666) | §7 |
| Train-only rule = retrieval within one control validation SD **and** a strict code-to-own-axis gain | user, 2026-09-26 (audit §666) | §7.3 |
| N is re-selected for both arms | user, 2026-09-25 | §7.1 |

The earlier "anchors if the official test passes" path is superseded (audit §666.1) and is not kept
as a fallback. The official-test criterion is reported for the frozen recipe, whatever it shows; it
never switches the model, N, scope or checkpoint.

## 3. Dataset policy

- **In scope:** Flickr25K, NUS-WIDE, MS-COCO (multi-label).
- **CIFAR-10:** out of scope, keeps the approved incumbent. **No CIFAR cell is part of this study**
  (audit §668.2, §675.1). Its exploratory negative result is kept, labelled exploratory, with both
  definitions: −0.1186 mean mAP@R with anchors at the incumbent's N=19 and −0.0780 at the anchored
  recipe's own N=4 (stage-1 validation, raw base-Hamming, three seeds, branch comparator; not the
  official test, not post-BIO). A confirmatory CIFAR claim would need a separate proposal.
- No dataset is added or dropped after any score of this campaign is seen.

## 4. Arms

| Arm | `--axis_center` | Everything else |
|---|---|---|
| control | `none` (the historical routing path) | the approved incumbent recipe of the dataset (§6) |
| candidate | `anchors` | identical to the control at the same N, seed and stage |

The approved incumbent recipe is the approved P3 refit's (ledger §285; aggregate `b4f3b0df…`;
selected-N `2bf6133d…`; recipe authority `e61748d7…`): top-p (0.6, 0.95) / JD 0.02 for Flickr25K,
(0.4, 0.8) / 0.05 for NUS-WIDE, (0.6, 0.95) / 0.03 for MS-COCO, through the per-dataset wrapper
scripts whose bytes equal the historical execution tree's. **Excluded:** the weak gate (`ancsoft`,
rejected in stage 11), `readout`/`both` centring, the concept codebook, queues and every other
arch-exp change.

**Both arms are trained fresh in this generation (stages S and D).** The v1 proposal to reuse the
historical `p3gE` selection records and the `p3lamA` Flickr25K seeds as control evidence is
withdrawn. Reasons: a fresh control is rendered by the same generation as the candidate, so the two
arms differ in `axis_center` alone by construction and the comparison does not depend on proving
historical equivalence field by field; no historical `config.pt` has to be deserialised before
stages S and D; the added cost is small (§12). The reducer keeps a reuse path, but it accepts a reuse
admission only when its digest is listed in the launcher's `APPROVED_REUSE_ADMISSION_SHA256`,
which this generation leaves empty (§11).

## 5. Source generation

- Worktree `/data/yschoi/gdna_anchor_confirm_v1`, branch `arch-exp-2026-09-anchor-confirm`.
- Base commit `88c3a25b1b309550eafc276c2ce5be7575507173` (the approved P3 model/trainer bytes of
  `5304005` plus the lambda launcher extension). The historical tree `/data/yschoi/gdna_p3exec`, its
  seals, records, manifests and verifiers are not edited; historical `--recipe` replay and
  source-drift refusal stay intact (the anchor mode is a separate, versioned launcher mode).
- Version boundary (audit §678): this generation consumes the approved aggregate and selected-N
  authority by the full digests pinned in its source, never by replaying the historical campaigns.
- The generation manifest v2 lists every source, test and contract file of the closure, the pinned
  wrapper scripts, the historical pins and the environment. The launcher's `--smoke`/`--run`, the
  reducer and the probe refuse unless the tree matches the manifest they are given.
- **Stale file warning:** the worktree's committed `artifacts/phase3_selection/selected_n.json`
  (`f2218aa7…`) is older; the approved bytes (`2bf6133d…`) live only under `/data/yschoi/gdna_p3exec`.

## 6. Split, schedule and protocol values (stages S and D)

- Split: the designated train split with a fixed held-out validation part
  (`--val_split_ratio 0.1 --val_split_seed 42`), opt-train whitening, `--selection_mode select`,
  `--keep_final_checkpoint`, no `--final_epoch_eval`. **No official-test split is loaded.**
- Horizons: `-e 60 --lr_schedule_horizon 60 --sinkhorn_schedule_horizon N+1 --stop_after_epoch N`.
- **Protocol values, enforced with their exact types for both arms at every planned coordinate**
  before any lease, reservation or dispatch (`anchor_protocol_fields`): the horizons above; the seed;
  the split and selection flags; `routing_adaptive_topp` on and `no_routing_adaptive_topp` off; the
  approved top-p window and `lambda_codon_joint`; `lambda_codeword_codon_sinkhorn 0.0`;
  `post_eval_compositional` off; `dna_distance_mode base`; 5 slots, 5 codebooks, 3 bases per slot,
  codebook size 128; no Gumbel softmax; counterfactual weight 0; the two global-slot skips;
  visualisation off; **`hash_target_mode siglip_cos`** (the unsupervised invariant) and
  **`disable_text_supervision` off** (the text path is part of the method); `axis_center` = the arm.
  A smoke changes only the four horizon values.
- **Reviewed wrapper overrides (audit §669.3, §677.2).** The three wrapper bodies hardcode seven
  options that the launcher sets again through `EXTRA_ARGS`, which lands after the body; argparse
  keeps the last occurrence. A destination may occur twice only as exactly: the wrapper literal
  first, then one launcher override:

  | Destination | Wrapper literal (first) | Launcher override (last) |
  |---|---|---|
  | `epoch` | `-e 60` | `-e 60` |
  | `routing_adaptive_topp` | `--routing_adaptive_topp` | `--routing_adaptive_topp` |
  | `routing_adaptive_topp_min` | `--routing_adaptive_topp_min 0.3` | the approved minimum |
  | `routing_adaptive_topp_max` | `--routing_adaptive_topp_max 0.7` | the approved maximum |
  | `lambda_codeword_codon_sinkhorn` | `--lambda_codeword_codon_sinkhorn 0.0` | `0.0` |
  | `post_eval_compositional` | `--post_eval_compositional` | `--no-post_eval_compositional` |
  | `dna_distance_mode` | `--dna_distance_mode base` | `base` |

  A third occurrence, a different first occurrence (value, alias, boolean form or inline `=`), an
  override placed first, or a repeat of any other destination refuses; `axis_center` is never
  repeated. The override's value is bound by the sealed recipe and by the protocol values above.
  The rendered argv of every in-scope wrapper repeats exactly these seven destinations.
- Retrieval score: raw base-Hamming mAP@R on the held-out part at the cell's own terminal epoch,
  R = 5000 for all three datasets, parsed from the pinned terminal `log.csv` row.

## 7. Stages, rules and gates

### 7.1 Stage S — train-only N selection, per arm and dataset

Grid N ∈ {4, 9, 19, 39}, seed 42, both arms: **24 new cells** (3 datasets × 2 arms × 4 N). Rule:
per dataset and arm, argmax of the retrieval score; ties go to the smallest N. The reducer
(`anchor_confirm_decision.py select`) writes the frozen N record, which names the generation
manifest every stage-S campaign ran under. N is chosen from seed 42 only and is never re-optimised
on seeds 43/44 (audit §665.1).

### 7.2 Stage D — train-only decision cells

Membership is derived per dataset from the **replayed** frozen N record (never from its digest
alone): for each dataset *d* and arm *a*, seeds 43 and 44 at N_S(d, a): **12 new cells**. Seed 42 at
N_S(d, a) is the stage-S cell itself. The two arms of one dataset may therefore run at different N;
their recipes then differ in `axis_center` and in exactly the N-dependent protocol values
(`stop_after_epoch`, `sinkhorn_schedule_horizon`), and the admission still compares the arms at
matched N, seed and stage. Every stage-D coordinate (18: 3 datasets × 2 arms × 3 seeds) gets one
code-to-axis probe (§8.2).

### 7.3 Decision rule (train-only, per dataset)

For each in-scope dataset *d*, over seeds 42/43/44 of each arm at its own N_S(d, a):

- (i) retrieval: `mean(candidate) >= mean(control) − sd(control)`, raw base-Hamming validation
  mAP@R (§6); `sd` is the sample SD with ddof = 1; equality passes; a zero control SD is allowed
  (then (i) reads `mean(candidate) >= mean(control)`). Post-BIO is not computed in S/D (it belongs
  to stage T). The seeds are shared by design, but the rule compares the arms' means; no paired
  statistic is used.
- (ii) alignment: `mean(candidate) > mean(control)` of the unrounded code-to-own-axis ratios (§8.2);
  equality fails.

Adopt anchors for *d* iff (i) and (ii) hold; otherwise *d* keeps the control recipe. Datasets are
decided independently over the fixed three-dataset scope. The reducer
(`anchor_confirm_decision.py decide`) writes the frozen decision record: per dataset the adopted
arm, its N and the stage-R requirement of §7.4. This is an operational three-seed rule, not a
statistical equivalence or superiority finding.

### 7.4 Stage R — scratch full-train refit (separate authorization)

Stage-R membership follows from the frozen decision and from admitted reuse only (audit §668.1):

| Decision for dataset *d* | Stage R |
|---|---|
| anchors adopted | scratch refit of anchors at N_S(d, anchors), seeds 42/43/44 |
| control kept, N_S(d, none) = the approved N | the approved `p3rfB` refit stands in **only if** a separately authorized historical inspection shows an exact recipe/N/schedule/input/provenance match; otherwise a scratch control refit |
| control kept, N_S(d, none) ≠ the approved N | scratch control refit at N_S(d, none); the old refit cannot stand in |

A refit uses the full designated train (`--val_split_ratio 0`), `-e N+1 --stop_after_epoch N`,
seeds 42/43/44, as the approved refit protocol. This generation does not render stage-R commands
(the launcher refuses an anchor refit); stage R needs its own reviewed generation or extension and
approval after the decision exists. The historical `config.pt` inspection that the second row needs
is **not part of this generation** (the v1 tool was removed so that nothing can run it without an
approved scope); stage R proposes it with its own approval scope.

### 7.5 Stage T — terminal official-test evaluation (descriptive)

Once per stage-R checkpoint: raw and post-BIO mAP@R on the official split, exactly as the approved
refit post-processing, reported against the approved incumbent's means and sample SDs from
`b4f3b0df…` (raw / post-BIO SD: Flickr25K .006759 / .006190, NUS-WIDE .006384 / .006972, MS-COCO
.001268 / .000599) as "within / outside one incumbent SD". Reported whatever it shows; nothing is
switched by it.

### 7.6 Stage gates (audit §668.2)

| Stage | Prerequisites (all before any lease, reservation, dispatch or real load) | Approval line (§11.3) |
|---|---|---|
| S smoke (optional) | this generation (manifest match), the admission, input seals | `scope=stage-S-smoke manifest=… request=…` |
| S | this generation, the plan, this contract, the §7.3 rule, the protocol values, input seals, environment | `scope=stage-S-run manifest=… request=…` |
| D | the stage-S frozen N record, replayed (JSON, logs and pinned bytes; nothing deserialised) from the approved stage-S receipts of **the same generation**; the stage-D plan derived from it | `scope=stage-D-run manifest=… selection=<frozen N sha256> request=…` (smoke: `stage-D-smoke`) |
| probes | the approved stage-S/D receipts; the frozen N record; the stage-D records | `scope=probe manifest=… selection=<frozen N sha256> request=…` |
| R, T | the frozen decision record | separate, later |

No S or D prerequisite depends on an R/T approval. One generation carries S, the frozen N record,
D, the probes and the decision (the launcher, the reducer and the probe refuse a mix).

## 8. Endpoints

### 8.1 Decision retrieval score — §6.

### 8.2 Decision alignment score: within-image codeword-to-own-axis top-1 accuracy (strict)

Frozen definition (`scripts/anchor_confirm_code_axis.py`, schema `anchor-confirm-code-axis/2`):

- **Rows:** the run's own train-only validation split — the trainer's
  `val_split.carve_val_indices` with ratio 0.1 and seed 42 — in ascending dataset index, the first
  512. Fewer than 512 validation rows refuses; no smaller population is substituted. The rows' cache
  identities (`trainset._feat_cache_rows`, the admitted dataset-to-cache mapping) are hashed
  (`row_ids_sha256`), and the probe records the split identity its input admission verified; all
  probes of one dataset must report the same row digest and split identity (one population for
  both arms and all seeds).
- **Forward:** deployment — eval mode, no caption reaches the model (`cached_text_part_raw=None`);
  the four local slots' quantised codewords `quantized_tokens[:, 1:, :]`, in slot order. The model is
  built from the run's saved typed `config.pt` and loads the checkpoint bytes the record pins with
  no missing or unexpected key.
- **Target:** the same images' cached caption features for the four local axes, through the same
  checkpoint's text adapter (`_adapt_pooled_text_for_loss`), local axes in the same order. Captions
  are only the target.
- **Centring:** in float64, per slot, once over the whole selected set (not per loader batch), then
  L2 normalisation. A centred vector with norm ≤ 1e-12 refuses the measurement.
- **Hit:** for image *i* and slot *m*, the own-axis cosine is **strictly** greater than each of the
  other three axes'; a tie is a miss.
- **Output (the measurement envelope, schema `anchor-confirm-code-axis/2`):** the exact coordinate;
  the record, checkpoint and `config.pt` digests; the record's execution authority; the generation
  manifest and the probe's approval line; the split, the admitted split identity and the row digest;
  the caption target (features, adapter, the admitted input seal); integer hits, ties and total
  (= 4 × 512 = 2048) with `0 ≤ ties ≤ total − hits`; the unrounded float ratio hits/total; the
  split `{val_split_ratio: 0.1, val_split_seed: 42}` with exact types; the probe request digest; the
  producer's digest. Non-finite or mis-shaped inputs refuse; nothing is dropped.
- **Decision value:** the unrounded mean of the three seeds' ratios; rounding is for display only.
- **Limits:** a within-image, relative measure. It is not global codeword naming and not
  word-level codon decodability (scope contract §3). Its tie rule and row source differ from the
  exploratory `quant_gap_diag.py` probe, so the exploratory values (which rose on 12/12
  dataset-seed pairs) are not the same measurement.
- **Admission first:** the probe checks the generation manifest, replays the frozen N record
  (metadata only), builds its request from the sources (exactly the 18 stage-D records) and checks
  its approval line for that request, requires the coordinate to be one of its stage-D coordinates,
  admits the record at the JSON level and checks the `config.pt` and checkpoint bytes against their
  pins — all before its first deserialisation; then the trainer's own input admission
  (`_phase3_input_authority_from_args`: seal statistics and runtime paths) must reproduce the
  record's input authority before any model or dataset is built.

### 8.3 Reported, not decisive

Unique code ratio on the database split, dead-codeword share, routing effective slots per patch,
local codon NMI, per-slot label AP. They cannot change the decision.

## 9. Aggregation and statistics

Seeds 42/43/44, mean and sample SD (ddof = 1), n = 3 per arm and dataset. **No significance or
equivalence test is run or claimed.** The margins in §7.3 and §7.5 are predeclared decision and
description rules.

## 10. Missing, non-finite and failed cells

- A missing, duplicated, extra, reused, non-finite or out-of-[0, 1] score, a count that is not
  4 × 512, or a record the evidence authority does not admit refuses the reduction; the reducer
  writes nothing. Nothing is imputed, averaged over fewer seeds or re-run silently.
- A cell that exits non-zero, lacks its terminal checkpoint, or fails an identity, recipe, split or
  epoch check stops its dataset stream. A re-run needs a new namespace; the failure stays recorded.

## 11. Evidence authority

- A record counts for (dataset, arm, N, seed) only through a completed campaign receipt of this
  generation: the receipt lists the recomputed cell id, the record and its completion digests; the
  plan snapshot it names seals that cell's typed recipe and names the generation manifest; the
  trainer-owned evidence, the runtime sidecar and the saved `config.pt` agree with it, input fields
  included. Labels, split and terminal epochs are compared with exact types.
- Reuse: only through a reuse-admission file whose digest the launcher's
  `APPROVED_REUSE_ADMISSION_SHA256` lists in reviewed source, filled only from an audit decision. A
  caller's digest, or a file that names its own approver, admits nothing (audit §672.1, §678.2).
  This generation lists none.
- Campaign approval: every receipt-backed record must come from a campaign whose plan snapshot
  names the audit approval it ran under; the reducer re-verifies that exact line in the ledger now
  (stage-S records: `stage-S-run`; stage-D records: `stage-D-run` naming the frozen N record).
- The reducer never deserialises a binary: `config.pt` is checked by byte identity against the pin
  of the record's completed anchor check (the typed comparison with the sealed recipe that ran
  inside the approved campaign) or of an approved reuse admission.

### 11.3 Approval authority

Approval is an explicit line in the audit ledger
(`/home/yschoi/GroundedDNA/docs/PHASE1_PHASE2_REAUDIT_2026-08-14.md`), which the modification agent
cannot write and which lies outside the pinned tree (so no manifest names an approval and no hash
cycle arises). One numbered section (`## <n>. …`) must contain exactly one line for the scope:

```
ANCHOR-CONFIRM-APPROVAL version=anchor-confirm/1 scope=<scope> manifest=<sha256> [selection=<sha256>] request=<sha256>
```

| Scope | Fields besides version and scope | Checked by |
|---|---|---|
| `stage-S-smoke` | `manifest`, `request` | launcher `--smoke`, before inputs and leases |
| `stage-S-run` | `manifest`, `request` | launcher `--run`; the reducer, for every stage-S record |
| `stage-D-smoke` | `manifest`, `selection`, `request` | launcher `--smoke` |
| `stage-D-run` | `manifest`, `selection`, `request` | launcher `--run`; the reducer, for every stage-D record |
| `probe` | `manifest`, `selection`, `request` | the probe, before its first load; the reducer, for every probe |

**The request (audit §689.2).** An approval covers one exact operation, named by the SHA256 of its
canonical request (JSON with sorted keys and compact separators, schema `anchor-confirm-request/1`):

- launcher (stage S/D, run or smoke): `stage`, `mode`, `manifest`, `selection` (stage D), `namespace`,
  `record_dir`, `result_root`, `declared_cells` and executed `cells` as (dataset, arm, N, seed),
  `epochs` (smoke horizon, else null), `input_seals` (each declared seal by path and file digest),
  `admission_authority` (path and file digest, or null) and `gpu_count`;
- probe: `operation=probe`, `manifest`, `selection` and `records`, the 18 stage-D
  (dataset, arm, N, seed, record digest).

`--plan` given together with `--run` or `--smoke` and otherwise the same arguments prints the
request and its digest (it reads the declared seal and authority files to hash them); the audit
approves that digest. Any changed dimension — another namespace, root, cell subset, smoke horizon
or cell, seal, authority or GPU count — is a different request and refuses under the old line.
The plan snapshot and receipt of a campaign carry its request and approval; the reducer
re-verifies the line in the ledger with the request's digest and requires each record's
coordinate, the receipt's namespace and the run directory's root to be those of the request.

A missing, fictional or untyped section reference, no line or a second line for the scope, another
version or scope, a missing, extra or different field refuses. Nothing is approved by this
contract, the manifest, a commit or a test.

## 12. Budget (every branch)

Minutes per epoch from past runs of this code on one GPU: Flickr25K 0.40, NUS-WIDE 1.08, MS-COCO
0.75 (sum 2.23). Planning arithmetic, not a runtime promise.

| Stage | Cells | Epochs | GPU-hours |
|---|---|---|---|
| S, per arm (N ∈ {4, 9, 19, 39}: 75 epochs per dataset) | 12 | 225 | 2.7875 |
| S, both arms | 24 | 450 | 5.575 |
| D, per arm (2 seeds × (N+1) per dataset) | 6 | 30 – 240 | 0.3717 – 2.9733 |
| D, both arms | 12 | 60 – 480 | 0.7433 – 5.9467 |
| probes (18 coordinates, deployment forward of 512 rows) | — | — | ≈ 0.6 (≤ 2 min each) |
| **S + D + probes** | **36** | 510 – 930 | **≈ 6.9 – 12.1** |
| R, per dataset branch (3 seeds × (N+1), full train) | 3 | 15 – 120 | see below |
| R, all three datasets | ≤ 9 | 45 – 360 | 0.5575 – 4.46 (+ about 11 % for the full train's larger epoch) |

Stage R's count depends on the decision: zero scratch cells for a dataset whose control is kept at
the approved N and whose old refit is admitted by the inspection; otherwise three. Post-processing
(BIO, test extraction) follows the approved refit protocol and is costed with stage R/T. The launcher
runs one dataset stream per GPU (3 GPUs); the slowest stream (NUS-WIDE) sets the wall time: about
2.7 h for stage S and up to 2.9 h for stage D. **Requested ceiling for S + D + probes: 12.5
GPU-hours on 3 GPUs.** R and T are costed again when they are proposed.

## 13. Typed recipe, admission, launcher, reducer (implementation)

- **Recipe** (`dna_utils/scientific_recipe.py`, schema `groundeddna-scientific-recipe/2`): exactly
  `argv`, `fields` and `schema`; every parser destination except `tag`/`log_dir`, as type-sensitive
  canonical JSON; documented normalisations only (`setting`, the tokenizer JSON quote layer, the
  historical absence of `axis_center`); the reviewed override policy of §6.
- **Trainer** (`train_siglip2.py`): an anchor-confirmation cell (its campaign cell id carries the
  arm) must carry the sealed recipe; the payload rebuilt from the trainer's own argv must equal it
  (argv, typed fields, digest) and every field must be present on the post-processed arguments with
  the same canonical value. **Scope:** this runs in `_resolve_save_path`, before the run directory
  is claimed or created; the real entrypoint calls `Config.set_args()` earlier, which may create
  legacy directories, so the check is pre-claim, not pre-directory (audit §675.2).
- **Launcher** (`scripts/phase3_selection_matrix.py --anchor-confirm`): one admission for `--plan`,
  `--smoke` and `--run` before any lease, reservation or dispatch — cells from the approved
  incumbent, the replayed stage-S selection for `decide` (metadata only), both arms rendered at
  every coordinate, the override policy, the protocol values, arms differing in `axis_center`
  alone, and the generation manifest when given. `--smoke`/`--run` additionally require the
  manifest and the approval line for their exact scope (§11.3), and verify the input seals, all
  before the lease; after the lease the sweep re-checks the seals, stats-only. `--plan` renders the
  pinned wrappers under bash with a capture interpreter in temporary directories (their only
  writes, removed on exit); it reserves, records, leases and launches nothing and deserialises no
  binary. It is not a synthetic operation.
- **Reducer** (`scripts/anchor_confirm_decision.py`): §11 authority (receipt, snapshot, approval
  re-verified in the ledger, sealed recipe shape and protocol values, trainer evidence, sidecar,
  anchor check, config pin), exact membership, type-exact labels, one generation for the chain, the
  §7.1 and §7.3 rules, the §8.2 envelope; never deserialises; reads each file once and parses the
  hashed bytes; re-verifies before a single O_EXCL write.
- **Probe** (`scripts/anchor_confirm_code_axis.py`): §8.2.

## 14. Namespaces, roots and commands (only after approval; `GDNA_NUM_SEMANTIC_PARTS=5` exported)

- Records: `/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/`; run directories:
  `--result-root /data/yschoi/gdna_anchor_confirm_v1_result`.
- Namespaces: `ancS2` (stage S), `ancD2` (stage D); a smoke, if approved, `ancSmk2`, never evidence.
- Plan: `python scripts/phase3_selection_matrix.py --anchor-confirm select --namespace ancS2
  --anchor-arms none,anchors --plan`; add the stage-S execution arguments below together with
  `--plan` to print the exact request digest for approval.
- Stage S: the same with `--run --gpus A,B,C --input-seal DATASET:select=SEAL ... [--admission-authority PATH]
  --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v2.json --anchor-manifest-sha256 H
  --anchor-approval-section <ledger section> --result-root <root>`.
- N record: `python scripts/anchor_confirm_decision.py select --sources S --sources-sha256 H
  --manifest M --manifest-sha256 H --out N.json`.
- Stage D: `--anchor-confirm decide --namespace ancD2 --anchor-arms none,anchors --anchor-selection
  N.json --anchor-selection-sha256 H --anchor-manifest M --anchor-manifest-sha256 H
  --anchor-approval-section <ledger section> --run ...`.
- Probes: `python scripts/anchor_confirm_code_axis.py --sources S --sources-sha256 H --manifest M
  --manifest-sha256 H --selection N.json --selection-sha256 H --approval-section <ledger section>
  --coordinate DATASET:ARM:N:SEED --out P.json`.
- Decision: `python scripts/anchor_confirm_decision.py decide --sources S --sources-sha256 H
  --selection N.json --selection-sha256 H --manifest M --manifest-sha256 H --out D.json`.

## 15. Decisions requested from the audit

1. **Fresh controls** in stages S and D (§4), replacing the withdrawn v1 reuse proposal.
2. **The reviewed override policy** (§6 table) as the admitted repeat rule for the three wrappers,
   and **the approval-line format** of §11.3 as the way the audit authorizes each operation.
3. **The alignment endpoint** of §8.2 as a decision authority, with the probe and reducer bytes of
   the manifest.
4. **Input and environment admission for stage S** (separately scoped): which stage-1 input seals
   (`--input-seal`), whether a carried admission authority (`--admission-authority`) replaces a full
   rehash, and the GPUs/environment.
5. **Stage S execution** for this exact generation (manifest digest in the handoff), within the
   §12 ceiling; whether a one-cell smoke under `ancSmk2` precedes it.
6. Stage D, then R and T, each later and separately.
