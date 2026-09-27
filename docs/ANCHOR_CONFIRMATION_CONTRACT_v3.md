# Anchor Confirmation — Contract v3 (generation v5): the fixed four-dataset anchor model

**Status: NON-EXECUTABLE until the audit approves this exact generation for a named stage.**
Prepared under audit §709–§711 (PREPARATION_AUTHORIZED; EXECUTION_NOT_AUTHORIZED;
FINAL_RECIPE_NOT_APPROVED). §7.6 carries the §711.3 correction: the lambda checks are required.
- It supersedes contract v2 (`docs/ANCHOR_CONFIRMATION_CONTRACT_v2.md`, `26ebe2c1…`), which stays
  unchanged as history, together with its three-dataset scope, its CIFAR-incumbent exception and its
  conditional adoption rule.
- The generation is identified by `artifacts/anchor_confirmation/authority_manifest_v5.json`. That
  manifest pins this file, so its digest is named in the handoff, not here.
- Gates are stage-specific (§7.7). Every executed operation needs its own approval line in the audit
  ledger (§11.3); the code refuses until that line exists.

## 1. Scientific question

The architecture is decided, not tested here. By the user's decision recorded in audit §709,
**axis-centred routing anchors** (`--axis_center anchors`, §4.1) are part of the model for all four
datasets. This campaign asks, on **train-only validation data only**:
- which training length N the anchor model of each dataset should use; and
- how that model behaves across seeds 42/43/44 in retrieval and in within-image code-to-own-axis
  alignment.

The official-test result of the frozen model (stage T, later) is reported descriptively against the
approved incumbent. It never selects the architecture, N, scope or a checkpoint.

**What this campaign is not** (audit §711.1). S and D select N and validate the fixed model; they
are not a matched anchor-versus-none experiment. Rendering the control proves a configuration
difference, not an effect on performance. The later comparison against the approved `p3rfB`
official-test results is descriptive. It is never presented as a fresh paired S/D experiment or as
a causal effect of the anchors.

## 2. Recorded decisions this contract carries

| Decision | Recorded | Where |
|---|---|---|
| `axis_center=anchors` is fixed for CIFAR-10, Flickr25K, NUS-WIDE and MS-COCO; this is an architecture decision, not a claim of superiority | user; audit §709 | audit §709.1 |
| Stages S/D run the anchor arm only; the control (`none`) is rendered by the admission for the axis_center-alone comparison and never run | user, 2026-09-27 | §4, handoff v5 |
| The anchor-versus-incumbent comparison is made after stages R/T, on the official test, against the approved `p3rfB` results (descriptive; not paired, not causal) | user, 2026-09-27; audit §711.1 | §1, §7.5 |
| Run directories and the operations ledger live on the root filesystem (`/home/yschoi`), not on `/data` (99 % used) | user, 2026-09-27 | §14, addendum v3 |
| The 12.5 GPU-hour ceiling for S + D + probes is kept | user, 2026-09-27; audit §709.2 | §12 |
| One probe population for all four datasets: the first 500 validation rows | audit §709.3 (a feasible policy) | §8.2 |
| N is chosen from seed 42 by the train-only rule; nothing is re-optimised on seeds 43/44 | audit §665.1 | §7.1 |
| The TODO 13–15 lambda checks are **required** train-only work on the anchor model, after S/D and before the final recipe freeze; the anchor-only design does not waive them, and the old-model `p3lamA` evidence does not transfer | audit §711.3 | §7.6 |

**Superseded, kept as history.** Contract v2's conditional rule (anchors iff retrieval within one
control SD and a strict alignment gain), its CIFAR-incumbent policy, its three-dataset cell grid and
its paired control arm. Nothing in this generation can restore `axis_center=none`: the reducer has no
adoption rule, and a control arm, a missing dataset or a v1 request refuses (§13).

## 3. Datasets and dataset-specific values

All four datasets are in scope. None is added or dropped after any score of this campaign is seen.

| Dataset | Wrapper (pinned bytes) | K | mAP@R cutoff R | Top-p (min, max) | `lambda_codon_joint` | Designated train / validation rows |
|---|---|---:|---:|---|---:|---|
| CIFAR-10 | `scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh` | 64 | 1000 | (0.3, 0.7) | 0.02 | 5000 / 500 |
| Flickr25K | `scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh` | 128 | 5000 | (0.6, 0.95) | 0.02 | 5000 / 500 |
| NUS-WIDE | `scripts/train_nuswide_v185_sweep_clip.sh` | 128 | 5000 | (0.4, 0.8) | 0.05 | 10500 / 1050 |
| MS-COCO | `scripts/train_mscoco_F2_sweep_clip.sh` | 128 | 5000 | (0.6, 0.95) | 0.03 | 10000 / 1000 |

- K and R are the launcher's `DATASETS[...]["K"]` and `MAP_R_CUTOFF`, and the evaluation's
  `MAP_AT_R_BY_DATASET`. A test holds the three sources to one value per dataset.
- The top-p window and joint weight come from the approved aggregate `b4f3b0df…` and its selected-N
  authority `2bf6133d…` (ledger §285). The launcher reads both by their pinned digests and requires
  them to agree.
- The rows are the stage-1 seals' recorded populations (audit §709.3).
- CIFAR-10 is single-label. Its relevance rule is label equality, as in the approved protocol.

## 4. The model and the arm

### 4.1 Model revision: what `axis_center=anchors` is (and is not)

- **What it is.** Before the routing transport cost is computed, each LOCAL routing anchor
  (`route_centroids[:, n_global:, :]`) has the per-image mean of the active local anchors subtracted
  (`SigLIP2SemanticOTModel._axis_center_local`). The global prefix is untouched. **No parameter and
  no loss term are added.** It is a representation change inside routing, enabled by
  `--axis_center anchors` (config default `none`, which is the historical path).
- **What it is not:**
  - the codebook **anchor-EMA loss** `loss_anchor` (weight `lambda_anchor`), which stays at the
    approved 0.05;
  - the per-codon text-anchored prototype CE `codon_text_anchor`, which stays off (`False`; its
    weight 0.1 is inert).
- `MODEL_AND_PROTOCOL_SPEC.md` does not yet define `axis_center`. The proposed amendment text is in
  `docs/ANCHOR_MODEL_TODO_MIGRATION_v2.md` §1. The spec itself is edited only when this model is
  consolidated, never silently.

### 4.2 The arm

| Arm | `--axis_center` | Everything else | Runs |
|---|---|---|---|
| anchors | `anchors` | the approved recipe of the dataset (§3, §6) | yes |
| none (rendered comparator) | `none` | identical | **no**: rendered by the admission only |

- At every coordinate the admission renders both arms and requires them to differ in `axis_center`
  alone. This is how an anchor cell is proven to be the approved recipe with one change.
- **Excluded:** the weak gate, `readout`/`both` centring, the concept codebook, queues, every other
  arch-exp change, and Gumbel sampling (off, as approved).
- **No historical reuse:** every S/D record is a fresh anchor cell of this generation.

## 5. Source generation

- Worktree `/data/yschoi/gdna_anchor_confirm_v1`, branch `arch-exp-2026-09-anchor-confirm`, which
  continues from generation v4.
- The historical tree `/data/yschoi/gdna_p3exec`, its seals, records, manifests and verifiers are
  not edited. Historical `--recipe` replay and source-drift refusal stay intact: the anchor mode is
  a separate, versioned launcher mode.
- Version boundary (audit §678): the approved aggregate and selected-N authority are consumed by the
  full digests pinned in source, never by replaying the historical campaigns.
- The generation manifest v5 lists every source, test and contract file of the closure, the pinned
  wrapper scripts, the historical pins and the environment. The launcher's `--smoke`/`--run`, the
  reducer and the probe refuse unless the tree matches the manifest they are given.

## 6. Split, schedule and protocol values (stages S and D)

- **Split.** The designated train split with a fixed held-out validation part
  (`--val_split_ratio 0.1 --val_split_seed 42`), opt-train whitening, `--selection_mode select`,
  `--keep_final_checkpoint`, no `--final_epoch_eval`. **No official-test split is loaded.**
- **Horizons.** `-e 60 --lr_schedule_horizon 60 --sinkhorn_schedule_horizon N+1
  --stop_after_epoch N`.
- **Protocol values**, enforced with their exact types at every planned coordinate before any lease,
  reservation or dispatch (`anchor_protocol_fields`):
  - the horizons above, the seed, and the split and selection flags;
  - `routing_adaptive_topp` on and `no_routing_adaptive_topp` off;
  - the approved top-p window and `lambda_codon_joint` (§3);
  - `lambda_codeword_codon_sinkhorn 0.0`;
  - `post_eval_compositional` off and `dna_distance_mode base`;
  - 5 slots, 5 codebooks, 3 bases per slot, and the dataset's codebook size K;
  - **no Gumbel softmax**, counterfactual weight 0, the two global-slot skips, visualisation off;
  - **`hash_target_mode siglip_cos`** and **`disable_text_supervision` off**;
  - `axis_center` = the arm.

  A smoke changes only the four horizon values.
- **Reviewed wrapper overrides** (audit §669.3, §677.2). Each wrapper body hardcodes seven options
  that the launcher sets again through `EXTRA_ARGS`, which lands after the body; argparse keeps the
  last occurrence. A destination may occur twice only as exactly the wrapper literal first, then one
  launcher override:

  | Destination | Wrapper literal (first) | Launcher override (last) |
  |---|---|---|
  | `epoch` | `-e 60` | `-e 60` |
  | `routing_adaptive_topp` | `--routing_adaptive_topp` | `--routing_adaptive_topp` |
  | `routing_adaptive_topp_min` | `--routing_adaptive_topp_min 0.3` | the approved minimum |
  | `routing_adaptive_topp_max` | `--routing_adaptive_topp_max 0.7` | the approved maximum |
  | `lambda_codeword_codon_sinkhorn` | `--lambda_codeword_codon_sinkhorn 0.0`; **CIFAR-10: `0.1`** | `0.0` |
  | `post_eval_compositional` | `--post_eval_compositional` | `--no-post_eval_compositional` |
  | `dna_distance_mode` | `--dna_distance_mode base` | `base` |

  **New in v3 (audit §709.1 item 4).** The CIFAR-10 wrapper passes
  `--lambda_codeword_codon_sinkhorn "${CCS:-0.1}"`, and the launcher sets `CCS=0.1` for it. The
  reviewed alternate first literal is therefore `0.1`, and the required override is `0.0`: the value
  every approved CIFAR-10 stage-1 and refit args.txt records. The override is still required, and the
  protocol values bind the effective 0.0.

  A third occurrence, any other first occurrence, an override placed first, or a repeat of any other
  destination refuses. `axis_center` is never repeated.
- **Retrieval score.** Raw base-Hamming mAP@R on the held-out part at the cell's own terminal epoch,
  with R per dataset as in §3 (CIFAR-10: 1000; the other three: 5000), parsed from the pinned
  terminal `log.csv` row.

## 7. Stages, rules and gates

### 7.1 Stage S — train-only N selection

- Grid N ∈ {4, 9, 19, 39}, seed 42, anchor arm: **16 new cells** (4 datasets × 4 N).
- Rule: per dataset, argmax of the retrieval score; ties go to the smallest N.
- The reducer (`anchor_confirm_decision.py select`) writes the frozen N record, which names the
  generation manifest and the fixed architecture.

### 7.2 Stage D — seeds 43 and 44 at the frozen N

- Membership is derived per dataset from the **replayed** frozen N record, never from its digest
  alone: seeds 43 and 44 at N_S(d). That is **8 new cells**.
- Seed 42 at N_S(d) is the stage-S cell itself.
- Each of the **12 stage-D coordinates** (4 datasets × 3 seeds) gets one code-to-axis probe (§8.2).

### 7.3 Stage-D summary (descriptive; no adoption rule)

- Per dataset, the reducer (`anchor_confirm_decision.py decide`) reports over seeds 42/43/44 at
  N_S(d):
  - the retrieval scores, their mean and sample SD (ddof = 1);
  - the code-to-own-axis ratios, their hits/totals, mean and sample SD.
- **No score selects the architecture.** A low retrieval or alignment value is reported as it is.
  It never adopts `none`, keeps an old refit, drops a dataset or suppresses a negative result.
- Missing or invalid evidence still refuses the reduction (§10).
- The decision record also lists the stage-R membership of §7.4 at N_S(d). That membership is
  provisional: it stands for a dataset whose recipe the lambda checks (§7.6 step 3) leave
  unchanged, and the final recipe freeze decides it.

### 7.4 Stage R — scratch full-train anchor refits (separate authorization)

- **12 cells:** 4 datasets × seeds 42/43/44, anchor arm, at the frozen final N of each dataset
  (N_S(d) unless a lambda change reselected it, §7.6).
- Settings: the full designated train split (`--val_split_ratio 0`), `-e N+1 --stop_after_epoch N`,
  and the approved refit post-processing (three-split extraction, BIO projection, NMI).
- The twelve approved incumbent refits (`p3rfB`) are **not** the new model. They stay the
  descriptive comparator of §7.5.
- This generation does not render stage-R commands (the launcher refuses an anchor refit). Stage R
  needs its own reviewed generation or extension, and its own approval.

### 7.5 Stage T — terminal official-test evaluation (descriptive)

- Once per stage-R checkpoint: raw and post-BIO mAP@R (15 bases, GC count [6, 9], maximum run 3) on
  the official split, exactly as the approved refit post-processing.
- It is reported next to the approved incumbent's means and sample SDs from `b4f3b0df…`, whatever it
  shows. Nothing is switched by it.

### 7.6 Freeze order and recipe lineage (audit §709.1 item 3, §709.5, §711.3)

1. **Stage S** selects N_S(d) at the approved lambdas (§3), seed 42.
2. **Stage D** reports the seed behaviour at N_S(d); then the 12 probes and the stage-D summary.
3. **Stage L: the TODO 13–15 lambda checks (required).** Train-only checks of the anchor model at
   N_S(d), one factor at a time:
   - TODO 13, transport (`--lambda_wasserstein`): 0.30 and 0.50 against 0.15;
   - TODO 14, codebook balance (`--lambda_bu`): 0 against 0.02;
   - TODO 15, text-code contrastive (`--lambda_text_hash_ntxent`): 0.025 and 0.10 against 0.05.

   They follow S/D and precede the freeze. They are not waived by the anchor-only design, and the
   old-model `p3lamA` results (Flickr25K, ledger §536.1) are not anchor-model evidence.

   They get their own preregistered proposal: the dataset scope, seeds, decision rule and budget
   are fixed before any L score is seen. The natural template is the `p3lamA` rule: a candidate
   replaces the value on its axis iff its seed-42 score exceeds the incumbent's seed-42 score by
   more than max(0.002, the incumbent's seed spread), which stage D supplies for the anchor model;
   the highest such candidate wins. The budget is separate from the §12 S/D/probe ceiling. The
   design stays bounded (at most the five alternates above per dataset; no Cartesian grid) and
   uses no official-test data.
4. **If L changes a lambda of dataset d**:
   - N is selected again for d's new recipe on seed 42 over {4, 9, 19, 39}, in a new generation
     with its own contract, request and approval. Its cells are the affected datasets only.
   - Stage D (seeds 43/44) and the three probes of d are regenerated at that N, from records of
     that same generation.
   - d's old-lambda S/D/probe records stay historical. They are not validation of the final model.
   - Datasets whose lambdas L leaves unchanged keep this generation's records.
   - No reduction merges records of two generations (each reducer already refuses another
     generation's records). The freeze record (step 5) binds, per dataset, one generation's frozen
     N record and stage-D decision record, each already single-generation.
5. **Final recipe freeze (F)**, recorded before any R/T or official-test access. Per dataset:
   `axis_center=anchors`, the final N, the final lambdas, and the generation whose S/D/probe
   records validate exactly that recipe.
6. **Stage R** (§7.4), then **stage T** (§7.5), each separately approved.
7. **Downstream TODO items** on the frozen model after F and their actual R/T dependencies, in the
   order of `docs/ANCHOR_MODEL_TODO_MIGRATION_v2.md`, each with its own budget and approval.

S does not wait for L, R/T or downstream implementations or approvals. Their order is fixed here
before S is admitted.

### 7.7 Stage gates (audit §668.2)

| Stage | Prerequisites (all before any lease, reservation, dispatch or real load) | Approval line (§11.3) |
|---|---|---|
| S smoke (optional) | this generation (manifest match), the admission, input seals | `scope=stage-S-smoke manifest=… request=…` |
| S | this generation, the plan, this contract, the protocol values, the four input seals, the environment | `scope=stage-S-run manifest=… request=…` |
| D | the stage-S frozen N record, replayed from the approved stage-S receipts of **the same generation** (JSON, logs and pinned bytes; nothing deserialised); the stage-D plan derived from it | `scope=stage-D-run manifest=… selection=<frozen N sha256> request=…` (smoke: `stage-D-smoke`) |
| probes | the approved stage-S/D receipts, the frozen N record and the stage-D records | `scope=probe manifest=… selection=<frozen N sha256> request=…` |
| L (TODO 13–15) | the stage-D summary of this generation; its own preregistered proposal | separate, later |
| R, T | the final recipe freeze (§7.6 step 5) | separate, later |

## 8. Endpoints

### 8.1 Retrieval score (N selection)

As defined in §6.

### 8.2 Alignment score: within-image codeword-to-own-axis top-1 accuracy (strict)

Frozen definition (`scripts/anchor_confirm_code_axis.py`, schema **`anchor-confirm-code-axis/3`**):
- **Population (new in v3, audit §709.3).**
  - Rows: the run's own train-only validation split (the trainer's `val_split.carve_val_indices`,
    ratio 0.1, seed 42), in ascending dataset index, **the first 500**. That gives **4 local slots ×
    500 = 2000 strict decisions**.
  - CIFAR-10 and Flickr25K have exactly 500 validation rows, and one policy applies to all four
    datasets.
  - Fewer than 500 rows refuses. Exactly 500 are measured. There is no smaller, resampled,
    enlarged, optimization-train or official-test population.
  - The envelope carries the population declaration `{rows, n_images: 500, local_slots: 4,
    decisions: 2000}`, and the reducer requires it exactly. The probe request (§11.3) carries the
    same declaration, so a probe approval names the population it covers.
  - The rows' cache identities are hashed (`row_ids_sha256`). All three probes of one dataset must
    report the same row digest and split identity.
- **Forward:** deployment. Eval mode; no caption reaches the model. The four local slots' quantised
  codewords `quantized_tokens[:, 1:, :]`, in slot order. The model is built from the run's saved
  typed `config.pt` and loads the pinned checkpoint bytes with no missing or unexpected key.
- **Target:** the same images' cached caption features for the four local axes, through the same
  checkpoint's text adapter (`_adapt_pooled_text_for_loss`). Captions are only the target.
- **Centring:** in float64, per slot, once over the whole population, then L2 normalisation. A
  centred norm ≤ 1e-12 refuses the measurement.
- **Hit:** the own-axis cosine is **strictly** greater than each of the other three axes'. A tie is a
  miss.
- **Output (the measurement envelope):**
  - the exact coordinate `(dataset, "anchors", N, seed)`;
  - the record, checkpoint and `config.pt` digests, and the record's execution authority;
  - the generation manifest, the probe approval line and the probe request digest;
  - the split, the admitted split identity and the row digest;
  - the caption target (features, adapter, the admitted input seal);
  - the population;
  - integer hits, ties and total (= 2000), with `0 ≤ ties ≤ total − hits`, and the unrounded float
    ratio hits/total;
  - the producer's digest.
- **Limits:** a within-image, relative measure. It is not global codeword naming and not word-level
  decodability. The exploratory values (Gumbel ON, a different tie rule and a different row source)
  are not the same measurement.
- **Admission first.** Before its first deserialisation, the probe:
  1. checks the generation manifest;
  2. replays the frozen N record (metadata only);
  3. builds its request from the sources (exactly the 12 stage-D records) and checks its approval
     line;
  4. requires the coordinate to be a stage-D coordinate;
  5. admits the record at the JSON level;
  6. checks the `config.pt` and checkpoint bytes against their pins.

  Then the trainer's own input admission must reproduce the record's input authority before any
  model or dataset is built.

### 8.3 Reported, never decisive

Unique code ratio on the database split, dead-codeword share, routing effective slots per patch,
local codon NMI and per-slot label AP.

## 9. Aggregation and statistics

Seeds 42/43/44, mean and sample SD (ddof = 1), n = 3 per dataset. **No significance, equivalence or
superiority test is run or claimed.**

## 10. Missing, non-finite and failed cells

- The reduction refuses, and the reducer writes nothing, on any of these:
  - a missing, duplicated, extra, reused, non-finite or out-of-[0, 1] score;
  - a count that is not 4 × 500;
  - a population other than §8.2's;
  - a record the evidence authority does not admit.

  Nothing is imputed, averaged over fewer seeds or re-run silently.
- A cell that exits non-zero, lacks its terminal checkpoint, or fails an identity, recipe, split or
  epoch check stops its dataset stream. A re-run needs a new namespace, and the failure stays
  recorded.

## 11. Evidence authority

- A record counts for `(dataset, anchors, N, seed)` only through a completed campaign receipt of
  this generation.
  - The receipt lists the recomputed cell id, the record and its completion digests.
  - The plan snapshot it names seals that cell's typed recipe and names the generation manifest.
  - The trainer-owned evidence, the runtime sidecar and the saved `config.pt` agree with it, input
    fields included.
  - Labels, split and terminal epochs are compared with exact types.
- **No reuse.** Contract v3 has no reuse admission. A record whose source is not a campaign receipt
  refuses.
- **Campaign approval.** The reducer re-verifies the exact approval line the campaign's plan
  snapshot names, in the ledger now: `stage-S-run` for seed 42 and `stage-D-run` naming the frozen N
  record for seeds 43/44.
- **No deserialisation.** The reducer never deserialises a binary. `config.pt` is checked by byte
  identity against the pin of the record's completed anchor check.
- **Inputs** (audit §694, §696). The campaign's admitted seals must be exactly the approved
  request's seal pins. Each record's input authority must be the admitted seal of its dataset, and
  its launch and cell bindings must carry that seal's identities.

### 11.3 Approval authority

Approval is an explicit line in the audit ledger
(`/home/yschoi/GroundedDNA/docs/PHASE1_PHASE2_REAUDIT_2026-08-14.md`). One numbered section must
contain exactly one line for the scope:

```
ANCHOR-CONFIRM-APPROVAL version=anchor-confirm/2 scope=<scope> manifest=<sha256> [selection=<sha256>] request=<sha256>
```

| Scope | Fields besides version and scope | Checked by |
|---|---|---|
| `stage-S-smoke` | `manifest`, `request` | launcher `--smoke` |
| `stage-S-run` | `manifest`, `request` | launcher `--run`; the reducer, for every stage-S record |
| `stage-D-smoke` | `manifest`, `selection`, `request` | launcher `--smoke` |
| `stage-D-run` | `manifest`, `selection`, `request` | launcher `--run`; the reducer, for every stage-D record |
| `probe` | `manifest`, `selection`, `request` | the probe, before its first load; the reducer, for every probe |

**The request** is the SHA256 of canonical JSON, schema **`anchor-confirm-request/2`**, version
`anchor-confirm/2`. Its fields:
- `stage`, `mode`, `manifest`, `selection` (stage D), `namespace`, `record_dir`, `result_root`;
- `declared_cells` and executed `cells` as `(dataset, "anchors", N, seed)`;
- `epochs`, `input_seals` (path and file digest), `admission_authority` and `gpu_count`.

The GPU count must equal the number of dataset streams the request runs: 4 for a stage run, 1 for a
one-cell smoke.

**The probe request** (`anchor_confirm_decision.probe_request`, same schema and version) names the
generation manifest, the frozen N record, the 12 stage-D records `{coordinate: record digest}` and
the §8.2 population. One `probe` approval line covers the 12 probes of that request. **A version-1 (three-dataset) request, approval line or record refuses by version.**
A request with three GPUs, a missing dataset or a control arm cannot be formed. `--plan` together
with the execution arguments prints the request and its digest.

## 12. Budget

Minutes per epoch are the most conservative of the historical stage-1 timings (`p3gE` wall time per
epoch) and the v2 contract's figures: CIFAR-10 0.56, Flickr25K 0.40, NUS-WIDE 1.08, MS-COCO 0.76
(sum 2.80). This is planning arithmetic, not a runtime promise.

| Stage | Cells | Epochs | GPU-hours |
|---|---|---|---|
| S (N ∈ {4, 9, 19, 39}: 75 epochs per dataset) | 16 | 300 | 3.50 |
| D (2 seeds × (N+1) per dataset) | 8 | 40 – 320 | 0.47 – 3.73 |
| probes (12 coordinates, 500 rows each) | — | — | ≈ 0.4 (≤ 2 min each) |
| **S + D + probes** | **24** | 340 – 620 | **≈ 4.4 – 7.6** |
| R (12 cells, full train) | 12 | 60 – 480 | ≈ 0.7 – 5.6 (+ about 11 % for the full train); costed with its proposal |

- The ceiling for S + D + probes stays **12.5 GPU-hours**; the ledger charges are defined in
  addendum v3.
- Stage L (§7.6 step 3) and any N reselection it causes have their own budget in their own
  proposal. Planning bound for L: at most 5 alternates × 4 datasets = 20 seed-42 cells at N_S(d),
  about 1.2 GPU-h if every N_S is 4 and 9.3 GPU-h if every N_S is 39. They never draw on the S/D
  allowance.
- The launcher runs one dataset stream per GPU (**4 GPUs**). The slowest stage-S stream (NUS-WIDE)
  takes about 81 min after full input verification of the four stage-1 seals (about 532 GB; the
  historical four-dataset figure is 82 min, an estimate).
- Storage: 16 S cells need 22 GiB free at the first dispatch, and 8 D cells need 16 GiB. The result
  root is on the root filesystem (§14).

## 13. Typed recipe, admission, launcher, reducer (implementation)

- **Recipe** (`dna_utils/scientific_recipe.py`, schema `groundeddna-scientific-recipe/2`). Unchanged
  except the reviewed alternate CIFAR-10 literal (§6).
- **Trainer** (`train_siglip2.py`). Unchanged. An anchor cell must carry the sealed recipe, checked
  before the run directory is claimed.
- **Launcher** (`scripts/phase3_selection_matrix.py --anchor-confirm`).
  - Version `anchor-confirm/2`.
  - `ANCHOR_DATASETS` holds the four datasets, and `ANCHOR_RUN_ARMS = ("anchors",)`.
  - `--anchor-arms` must be `anchors`, and `--anchor-cells` refuses.
  - The arm plan must cover all four datasets.
  - The request's GPU count must equal its dataset streams.
  - The admission renders both arms at every coordinate. Everything else is as in v2.
- **Reducer** (`scripts/anchor_confirm_decision.py`, `anchor-confirm-reducer/3`).
  - It has no adoption rule and no reuse path.
  - It carries the fixed-architecture declaration in the N record and the decision record, and
    requires it in the replay.
  - It uses the §8.2 population and the §7.4 stage-R membership.
- **Probe** (`scripts/anchor_confirm_code_axis.py`): §8.2.
- **Generation continuity** (audit §697): unchanged from v2; every file of the v5 closure is
  re-verified at every boundary.

## 14. Namespaces, roots and commands (only after approval; `GDNA_NUM_SEMANTIC_PARTS=5` exported)

- Records: `/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/`.
- Run directories: `--result-root /home/yschoi/gdna_anchor4_result`.
- Operations ledger: `/home/yschoi/gdna_anchor4_ops`.
- Namespaces: `ancS5` (stage S), `ancD5` (stage D). A smoke, if approved, uses `ancSmk5` and is
  never evidence.
- **Plan:** `python scripts/phase3_selection_matrix.py --anchor-confirm select --namespace ancS5
  --anchor-arms anchors --plan`. Add the execution arguments to print the request digest.
- **Stage S:** the same command with `--run --gpus A,B,C,D --input-seal cifar10:stage1=SEAL
  --input-seal flickr25k:stage1=SEAL --input-seal nuswide:stage1=SEAL --input-seal
  mscoco:stage1=SEAL --anchor-manifest artifacts/anchor_confirmation/authority_manifest_v5.json
  --anchor-manifest-sha256 H --anchor-approval-section <ledger section> --result-root
  /home/yschoi/gdna_anchor4_result`, run under the supervisor of addendum v3.
- **N record:** `python scripts/anchor_confirm_decision.py select --sources S --sources-sha256 H
  --manifest M --manifest-sha256 H --out N.json`.
- **Stage D:** `--anchor-confirm decide --namespace ancD5 --anchor-arms anchors --anchor-selection
  N.json --anchor-selection-sha256 H ... --run`.
- **Probes:** `python scripts/anchor_confirm_code_axis.py ... --coordinate DATASET:anchors:N:SEED
  --out P.json`.
- **Stage-D summary:** `python scripts/anchor_confirm_decision.py decide --sources S
  --sources-sha256 H --selection N.json --selection-sha256 H --manifest M --manifest-sha256 H --out
  D.json`.

## 15. Decisions requested from the audit

1. This contract v3 as the scope-bearing replacement of v2, including the fixed-architecture
   reducer semantics (§7.3) and the anchor-only design with a rendered comparator (§4).
2. The reviewed alternate CIFAR-10 wrapper literal (§6).
3. The 500-row probe population (§8.2).
4. Input and environment admission for stage S (the four stage-1 seals, a full rehash, four GPUs)
   and stage-S execution for this exact generation, within the §12 ceiling, under addendum v3.
5. Stage D, the probes, then R and T, each later and separately.
6. The corrected freeze order and recipe lineage (§7.6, audit §711.3): stage L required after S/D
   and before the freeze, with its own later proposal; the consequences of a lambda change.
