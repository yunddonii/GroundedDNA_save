# Anchor Confirmation v1 — Contract

**Status: NON-EXECUTABLE DRAFT.** Prepared under audit §659 (PREPARATION_AUTHORIZED;
EXECUTION_NOT_AUTHORIZED; FINAL_RECIPE_NOT_APPROVED) with the clarifications of §660 and §664.
No GPU smoke, selection, decision, refit or official-test evaluation may start until the audit
approves the exact source, plan, rules, inputs and runtime scope named here. Every item listed in
§15 as unresolved blocks execution until it is decided.

## 1. Scientific question

Does routing with **axis-centred anchors** (`--axis_center anchors`: before the transport cost is
computed, each local routing anchor loses the per-image mean of the active local anchors; no
parameter and no loss term are added) raise the within-image code-to-axis alignment of the
multi-label models without a retrieval loss larger than one control seed SD, **judged on
train-only validation data**? For the recipe frozen by that train-only decision, what is its
official-test retrieval relative to the approved incumbent (descriptive verdict only)?

## 2. Recorded decisions this contract carries

| Decision | Recorded | Where |
|---|---|---|
| Anchors are proposed for the multi-label datasets only; CIFAR-10 keeps the incumbent and its negative result is reported | user, 2026-09-23 | `docs/ANCHOR_SCOPE_AND_ACCEPTANCE_2026-09-23.md` §2 (branch `arch-exp-2026-09`) |
| Official-test criterion: anchored mean within one incumbent sample SD, raw and post-BIO | user, 2026-09-23 | same document §4; **role clarified below** |
| The final recipe is chosen by a predeclared **train-only** rule, then model, scope and N are frozen before any new official-test access | user, 2026-09-26, after audit §664.1 | this contract §7 |
| Train-only rule = retrieval within one control validation SD **and** a code-to-axis gain | user, 2026-09-26 | this contract §7.3 |
| N is re-selected under the anchored recipe (not fixed) | user, 2026-09-25 | this contract §7.1 |

The earlier user choice "anchors if the official test passes, incumbent if it fails" was replaced
by the train-only path above after §664.1 confirmed it conflicts with train-only selection
(protocol audit §5.2). **The official-test 1-SD criterion therefore has a descriptive role only:**
it is reported for the frozen recipe, whatever it shows, and it never switches the main model,
N, scope or checkpoint.

## 3. Dataset policy

- **In scope:** Flickr25K, NUS-WIDE, MS-COCO (multi-label; each image carries several axes).
- **CIFAR-10:** out of scope, keeps the approved incumbent. Its exploratory negative result is
  retained with both definitions: −0.1186 mean mAP@R with anchors pinned to the incumbent's N=19,
  and −0.0780 at the anchored recipe's own N=4 (stage-1 validation, raw base-Hamming, three seeds,
  branch comparator; not official test, not post-BIO). Whether a protocol-level CIFAR measurement
  is required is left to the audit (§15, item 2).
- No dataset is added or dropped after any score of this campaign is seen.

## 4. Arms and the recipe they share

| Arm | `--axis_center` | Everything else |
|---|---|---|
| control | `none` (historical routing path) | the approved incumbent recipe of the dataset |
| candidate | `anchors` | identical to control |

The approved incumbent recipe is the one the approved P3 refit used (ledger §285; aggregate
`b4f3b0df…`; selected-N `2bf6133d…`; recipe authority `e61748d7…`): top-p (0.6, 0.95) / JD 0.02 for
Flickr25K, top-p (0.4, 0.8) / JD 0.05 for NUS-WIDE, top-p (0.6, 0.95) / JD 0.03 for MS-COCO, and the
per-dataset trainer script defaults. **Excluded:** the weak gate (`ancsoft`, rejected in stage 11),
`readout`/`both` centring, the concept codebook, queues and every other arch-exp change.

## 5. Source generation

- Worktree `/data/yschoi/gdna_anchor_confirm_v1`, branch `arch-exp-2026-09-anchor-confirm`.
- Base commit `88c3a25b1b309550eafc276c2ce5be7575507173`: the approved P3 model/trainer bytes of
  `5304005` (unchanged between the two commits) plus the lambda launcher extension. All 47 files of
  the launcher's executable closure were byte-identical to the historical execution tree
  `/data/yschoi/gdna_p3exec` when the worktree was created.
- Additions are listed commit by commit in the handoff manifest (§14). The historical tree, its
  seals, manifests and verifier are not edited.
- **Stale file warning:** the worktree's committed `artifacts/phase3_selection/selected_n.json`
  (`f2218aa7…`) is an older version. The approved bytes (`2bf6133d…`) live only at
  `/data/yschoi/gdna_p3exec/artifacts/phase3_selection/selected_n.json`. The committed copy is never
  read as authority.

## 6. Split, schedule and score (all stages S and D)

- Split: the designated train split with a fixed 10 % held-out validation part
  (`--val_split_ratio 0.1 --val_split_seed 42`), opt-train whitening, `--selection_mode select`.
  **No official test split is loaded** (protocol audit §5.2 items 1-2).
- Horizons: `-e 60 --lr_schedule_horizon 60 --sinkhorn_schedule_horizon N+1 --stop_after_epoch N`
  (protocol audit §5.3: each horizon recorded separately).
- Retrieval score: raw base-Hamming mAP@R on the held-out part at the cell's own terminal epoch,
  R = 5000 for all three datasets, parsed from the verified terminal `log.csv` row.

## 7. Stages and rules

### 7.1 Stage S — train-only N selection, per arm and dataset

Grid N ∈ {4, 9, 19, 39}, seed 42. Rule (as `scripts/phase3_select_n.py`): argmax of the retrieval
score; ties go to the smallest N. The candidate arm needs 12 new cells. For the control arm the
approved P3 selection evidence (namespace `p3gE`, N = 4 / 4 / 39) is **proposed for reuse** under
§664.2. Evidence for the proposal, all read-only: (a) the control executes the historical
model/trainer bytes (§5; the port only adds lines, and the only new call is gated on `anchors`);
(b) all 12 `p3gE` records pass the reducer's evidence rule of §13 for the coordinates
(dataset, `none`, N, 42) -- their saved typed configurations equal this generation's rendered
control recipes outside the tag and the sealed-input fields, and their scores verify from the pinned
logs -- and the rule reproduces N = 4 / 4 / 39. If the audit does not admit that reuse, 12 new
control cells run in this generation instead (§15, item 1).

### 7.2 Stage D — train-only decision cells

Each arm at its own selected N, seeds 43 and 44, same split and schedule (seed 42 comes from
stage S). Candidate: 6 new cells. Control: Flickr25K seeds 43/44 at N=4 exist in the approved
lambda campaign `p3lamA` (receipt `5a8901b7…`, ledger §536); both records pass the same evidence
rule for (flickr25k, `none`, 4, 43/44) and are proposed for reuse. NUS-WIDE and MS-COCO need 4 new
control cells (seeds 43/44 at the control N). Every checkpoint of stages S and D at the frozen N
also yields the code-to-axis probe of §8.2.

### 7.3 Decision rule, frozen now (train-only)

For each in-scope dataset *d*, with means and sample SDs (ddof = 1) over seeds 42/43/44 of stage S+D:

- (i) retrieval: `mean_val(candidate) >= mean_val(control) - sd_val(control)`;
- (ii) interpretability: `mean_code_axis(candidate) > mean_code_axis(control)`.

Adopt anchors for *d* if and only if (i) and (ii) both hold; otherwise *d* keeps the approved
incumbent. Equality passes (i) and fails (ii). The adopted recipe, its N (the adopted arm's
stage-S choice) and the scope are written to a frozen decision record before stage R.

### 7.4 Stage R — scratch full-train refit (separate authorization, never chained)

Only for datasets where anchors were adopted: full designated train (`--val_split_ratio 0`),
`-e N+1 --stop_after_epoch N` with both horizons N+1, seeds 42/43/44, as the approved refit
protocol. Where the incumbent is kept, the approved `p3rfB` refit is the result and nothing reruns.

### 7.5 Stage T — terminal official-test evaluation (descriptive)

Once per stage-R checkpoint: raw and post-BIO mAP@R on the official split, exactly as the approved
refit post-processing. Reported against the approved incumbent's means and sample SDs from
`b4f3b0df…` (raw / post-BIO SD: Flickr25K .006759 / .006190, NUS-WIDE .006384 / .006972,
MS-COCO .001268 / .000599) as "within / outside one incumbent SD". The full outcome is reported,
including failure; no recipe, N, scope or checkpoint substitution follows from it.

## 8. Endpoints

### 8.1 Decision retrieval score — §6.

### 8.2 Decision interpretability score: code-to-own-axis

Deployment forward of the terminal checkpoint (eval mode, no text reaches the model) on the first
`min(512, n_val)` rows of the run's own validation split: the trainer's `val_split.carve_val_indices`
with the run's typed `val_split_ratio`/`val_split_seed`, in ascending dataset index. The model is
built from the run's saved `config.pt` (typed; never `args.txt`) and loads the checkpoint bytes the
record pins with no missing or unexpected key (`scripts/anchor_confirm_code_axis.py`). For each
image and local slot *m*: the slot's quantised codeword is compared by cosine (after subtracting
the batch mean of codewords and of caption embeddings) with the same image's four axis-caption
embeddings; the slot scores 1 if its own axis caption is the nearest. The score is the mean over
images and the four local slots; chance is 0.25. Captions are only the target, never an input.
This is the metric that rose on 12/12 exploratory dataset-seed pairs; it is a within-image,
relative measure and is not evidence of globally nameable codewords (stage 10). The arithmetic is
the exploratory probe's; the row source, the model construction and the checkpoint loading are
not, so confirmatory values are not interchangeable with the exploratory ones.

### 8.3 Reported, not decisive

Unique code ratio on the database split, dead-codeword share, routing effective slots per patch,
local codon NMI, per-slot label AP. They cannot change the decision.

## 9. Aggregation and statistics

Seeds 42/43/44, mean and sample SD (ddof = 1). n = 3. **No significance or equivalence test is
run or claimed.** The margins in §7.3 and §7.5 are predeclared decision/description rules.

## 10. Missing, non-finite and failed cells

- Any score that is missing, duplicated, non-finite or outside [0, 1] refuses the decision for
  that dataset; nothing is imputed, averaged over fewer seeds or re-run silently.
- A cell that exits non-zero, lacks its terminal checkpoint, or fails an identity, recipe, split or
  epoch check stops its stage. A re-run needs a new namespace, and the failure stays in the record.
- A decision record is written only from an exact, complete, verified membership (§13).

## 11. Disclosures

- The official test split has been used before for the incumbent; its results are known. This is
  not a confirmation on an untouched benchmark.
- The held-out validation split was used repeatedly in exploration on `arch-exp-2026-09`
  (stages 6b-12). Stage 12's N choices are exploratory, not selection authority (§660.2).

## 12. Budget

Per-epoch wall time from past runs of this code (median minutes per epoch on one GPU): Flickr25K
0.40, NUS-WIDE 1.08, MS-COCO 0.75.

| Stage | Cells | GPU-hours (estimate) |
|---|---|---|
| S candidate | 12 | 2.8 |
| S control (only if reuse is refused) | 12 | 2.8 |
| D candidate (depends on chosen N) | 6 | 0.3 – 3.0 |
| D control (NUS-WIDE, MS-COCO) | 4 | 1.2 |
| code-to-axis probes | up to 18 | 0.3 |
| R refit (separately authorized) | up to 9 | up to 2.1 + post-processing |

Six GPUs; stages S and D run in about 1.5 – 3 hours of wall time with reuse admitted.

## 13. Typed scientific recipe, admission and reducer

Evidence already produced (read-only, no run): this generation's rendered control **refit**
recipe equals each approved refit's saved `config.pt` (Flickr25K `d4502d42…`, NUS-WIDE `27145906…`,
MS-COCO `abeade4a…`, reached from `b4f3b0df…` by record -> query extraction manifest -> config.pt)
in every field outside the 11 sealed-input fields `phase3_input_seal`, `phase3_input_seal_sha256`,
`phase3_input_aggregate_sha256`, `phase3_split_identity_sha256`, `phase3_hf_identity_sha256` and the
six `clip_snapshot_*` fields, which a campaign supplies from its own sealed inputs at launch; and at
all 12 stage-S coordinates the two arms differ in `axis_center` alone.


- **Recipe:** every destination of the trainer's own parser, parsed from the exact argv the
  per-dataset script would execute (rendered with a capture shim, never by running training), as
  typed canonical JSON (`groundeddna-scientific-recipe/1`), minus a short, declared list of
  non-scientific keys. The launcher seals its digest in the plan; the trainer recomputes it from
  its own argv **before** claiming a run or loading anything, refuses on mismatch, abbreviations or
  repeated options, and cross-checks the post-processed arguments; the digest travels in the
  campaign binding into the checkpoint metadata and runtime sidecar; the reducer recomputes it
  from the saved `config.pt` (same bytes it hashed). `args.txt` is display-only and cannot express
  a negative value.
- **Arm check:** the candidate's recipe must differ from the control's in `axis_center` alone
  (plus the tag), and the control's must equal the approved incumbent recipe except for the
  declared stage fields.
- **Reducer** (`scripts/anchor_confirm_decision.py`): a record counts for (dataset, arm, N, seed)
  only if it is a stage-1, train-only (`val_split_ratio` 0.1), non-smoke candidate whose terminal
  checkpoint and selection epoch are N, whose score parses from the log bytes it pins (finite, in
  [0, 1], equal to its own selection value), and whose saved typed configuration equals the recipe
  this generation renders for that coordinate outside the tag and the sealed-input fields. The same
  rule judges new and reused records. Membership is the contract's exactly (no missing, extra,
  duplicate or reused record); every input is read once and parsed from its hashed bytes,
  re-verified before the output is written once (O_EXCL). Probe outputs must name the checkpoint
  and config digests of their record.

## 14. Namespaces, output roots, commands and handoff

- Records: `/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/` (the launcher's
  anchor mode writes there, never to `artifacts/phase3_selection/`).
- Run directories: `--result-root /data/yschoi/gdna_anchor_confirm_v1_result`.
- Commands (to be run only after approval; `GDNA_NUM_SEMANTIC_PARTS=5` exported):
  - plan, side-effect free: `python scripts/phase3_selection_matrix.py --anchor-confirm select
    --namespace ancS1 --plan`
  - stage S: the same with `--run --gpus A,B,C --input-seal DATASET:select=SEAL ...
    --result-root <root>`; a smoke, if approved, `--smoke --only flickr25k:4:anchors:42` under
    `ancSmk1`
  - N record: `python scripts/anchor_confirm_decision.py select --sources S --sources-sha256 H
    --arms none,anchors --out N.json` (control coordinates point at the admitted `p3gE` records)
  - stage D: `--anchor-confirm decide --anchor-selection N.json --anchor-selection-sha256 H
    --anchor-arms anchors` (+ `none` for NUS-WIDE/MS-COCO only if the launcher is extended to
    per-dataset arms; see §15 item 6)
  - probes: `python scripts/anchor_confirm_code_axis.py --record R --record-sha256 H --out P.json`
  - decision: `python scripts/anchor_confirm_decision.py decide --sources S --sources-sha256 H
    --selection N.json --selection-sha256 H --out D.json`
- Namespaces: `ancS1` (stage S), `ancD1` (stage D), `ancR1` (stage R), `ancT1` (stage T); a smoke,
  if approved, uses `ancSmk1` and never counts as evidence.
- Handoff manifest `artifacts/anchor_confirmation/authority_manifest_v1.json`: old artifact and
  record pins with their approval sections, dataset/seed/N/recipe coordinates, new source and
  environment pins, and this contract's digest.

## 15. Unresolved decisions (execution blocked until each is resolved)

1. **Control reuse** (§664.2): admit the approved P3 selection evidence (N = 4/4/39, seed 42) and
   the `p3lamA` Flickr25K seeds 43/44 as control-arm evidence, or require new control cells.
2. **CIFAR-10:** whether the exploratory negative result suffices for the out-of-scope statement,
   or a protocol-level CIFAR measurement is required.
3. **Code-to-axis probe:** acceptance of the §8.2 definition and its pinned implementation as a
   decision endpoint.
4. **Validation SD source:** §7.3 uses the control's three-seed validation SD; with control reuse,
   seed 42 comes from the P3 selection matrix and seeds 43/44 from `p3lamA` or new cells — the audit
   must accept that these form one three-seed sample.
5. **Stage R/T authorization** after the frozen decision exists.
6. **Per-dataset control cells in stage D:** with reuse admitted, the control needs new seeds 43/44
   on NUS-WIDE and MS-COCO but not on Flickr25K. The launcher's `--anchor-arms` applies to all three
   datasets; running `none` everywhere would duplicate the admitted Flickr25K `p3lamA` seeds.
   Either accept that duplication (2 extra cells) or add a per-dataset arm list before execution.
7. **The probe and reducer as decision authorities:** their source bytes are in the handoff
   manifest; the audit must accept them before their outputs decide anything.
