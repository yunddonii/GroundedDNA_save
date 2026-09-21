# arch-exp-3 — making a slot own its axis without adding a loss term

Written 2026-09-20, **before any arm's numbers were read**. Branch `arch-exp-2026-09`, off-protocol
(no `GDNA_PHASE3_*` binding). Exploratory: nothing here may enter `docs/paper_draft/`.

## 0. Why this program looks different from arch-exp-2

arch-exp-2 tried four mechanisms and none moved M1. Its reading was that no active term ever compares
one axis against another *inside one image*. The obvious repair is a new contrastive loss — and that
is exactly what is ruled out here: the objective already carries twelve terms, and each further term
weakens the claim that the structure, not the loss budget, produces the codes. So this program is
ordered by *axis-role gain per added loss term*, and the first three arms add none.

## 1. Protocol, fixed (docs/MODEL_AND_PROTOCOL_SPEC.md)

Identical to arch-exp-2 so the two programs are directly comparable.

| item | value |
|---|---|
| dataset / split | Flickr25K, setting1, train 5,000 / val 10 % of train (`--val_split_ratio 0.1`) |
| official test | **never read** in stage 1 |
| backbone | frozen CLIP ViT-B/16, pinned snapshot sha256 as in the sealed recipe |
| geometry | `GDNA_NUM_SEMANTIC_PARTS=5`, `--num_semantic_parts 5 --num_codebooks 5 --num_codons_per_codebook 3` → 15 bases / 30 bits |
| supervision | `--hash_target_mode siglip_cos` (unsupervised invariant), Qwen v4 captions |
| quantiser | K=128, EMA .99, euclidean, `text_init_codebook none` |
| router | sinkhorn, UOT λa=λb=1.0, ε 1.0→0.1 over `--sinkhorn_schedule_horizon 5` |
| schedule | `--epoch 60 --stop_after_epoch 4` (stage-1 selection cell, ε aligned to the stopping epoch) |
| losses | the approved stage-1 λ set (`codon_joint .02`, `text_code_kl .05`, `xmodal_commit .05`, `text_hash_ntxent .05`, `wasserstein .15`, `cibhash 1.0`) |
| seeds | 42 / 43 / 44, every arm |
| launch mode | direct (non-campaign) for every arm **and** for the baseline it is compared with |

Baseline = the `base_s*` cells of `../arch_exp2_20260920/` (same tree, same launcher, same mode):
mAP@R .7453 ± .0133, M1 .0065 ± .0053, M2 .796, dead .241, unique .536.

## 2. Arms

Each arm is a **single delta** from the recipe above.

| arm | delta | code change | loss terms |
|---|---|---|---|
| **P1a** | `--routing_adaptive_topp_min 0.3 --routing_adaptive_topp_max 0.7` | none | 12 |
| **P1b** | `--routing_adaptive_topp_min 0.15 --routing_adaptive_topp_max 0.5` | none | 12 |
| **P2** | `--slot_center_readout`: local slot tokens get the per-image mean across local slots subtracted before the adapter/quantiser; slot 0 (global) untouched | flag, default off | 12 |
| **P3** | `--routing_slot_choice Q`: the plan is masked per **slot** (each slot keeps its top-Q share of patch mass) instead of per patch | flag, default off | 12 |
| **P4** | `--lambda_role`: `L_role` (§2 of today's PROJECT_LOG design entry) **with `--lambda_text_code_kl 0.0`** — a substitution, not an addition | flag, default off | 12 |

P5 (block-diagonal caption decoder replacing three text terms, 12 → 10) is written down but not
pre-registered: it is contingent on what P1–P4 show.

Motivation in one line each. P1: a patch currently reaches 2.97 of 4 slots and only 3.1 % of patches
reach exactly one, so the readouts are near-duplicates of the image mean. P2: whatever is common to
all slots is precisely what the global slot already carries. P3: expert-choice routing is how
mixture-of-experts obtained specialisation *and* balance while deleting its auxiliary loss. P4: the
one comparison the objective has never made, paid for by removing its weaker form.

## 3. Endpoints

Primary: **M1**, the own-slot-minus-other-slot decoding advantage for axis-distinctive caption words,
measured on held-out validation rows through the **deployment** forward (`cached_text_part_raw=None`),
`scripts/slot_role_probe.py`, unchanged from arch-exp-2.

Secondary, all pre-declared: mAP@R (val rows); M2 caption-free dataset-label decoding; M3 codebook
perplexity min/median; M4 per-image slot mass max/min and empty-slot rate; and for P1/P3 the routing
diagnostics `routing_mean_effective_k` and `routing_fraction_top1`.

## 4. Screening rule — **this is a screening threshold, not a significance test**

n = 3 seeds supports no test, and none is run. An arm is carried to stage 2 iff, over its 3 seeds:

1. mean M1 ≥ **0.020** (baseline mean .0065, baseline seed SD .0053), **and**
2. mean mAP@R ≥ **0.7353** (baseline mean − one baseline SD), **and**
3. mean dead-code ratio ≤ **0.30** and mean unique ≥ **0.45** (no starvation, no collapse).

Reported either way: the three means with their seed SDs and the per-seed values. An arm that fails
(1) but improves (2) or (3) is recorded as such and not carried.

Ordering rule agreed in advance: arms are run in the order P1 → P2 → P3 → P4; a later arm is run
regardless of whether an earlier one passes, because they act on different parts of the pipeline
(routing mask, readout, selection direction, objective). If two arms pass, the combination is run.

## 5. Stage 2, for any arm that passes

MS-COCO stage-1 selection cell, seeds 42/43/44, the MS-COCO λ set (`cibhash 1.5`, `codon_joint .03`,
`text_code_kl/xmodal/text_hash .10`, `wasserstein .05`, K=128, prompt v5b, `--epoch 40`). MS-COCO
is the confirmation set because it carries the dissection annotations, so a role claim there can be
checked against dense annotations rather than against captions alone, and because its codebook
collapse is mild — an improvement there is not merely a repair of Flickr25K's fragility.

## 6. Entry gate for every code-bearing arm (P2, P3, P4)

With the new flag at its default, the tree must reproduce the pre-change run in every logged value.
Same gate as arch-exp-2 (280/280 values). An arm whose gate fails is not run.

---

## Decision log (appended after the run it names; each entry is a departure from §2)

**2026-09-20, after P1 only.** P2 is split into two arms and the split is run *before* P3.
`--axis_center anchors` centres the router's text anchors (cost side); `--axis_center readout`
centres the slot tokens and the text tokens the losses see (readout side); `both` is held back.
Reason: P1 drove the routing mask to 67 % single-slot patches with no M1 gain, so the mask is not the
binding constraint and P3 is another mask-direction change. The cost is the remaining suspect, and
the anchors arm is its direct test. Endpoints, screening rule and baseline are unchanged. P3 is not
withdrawn — it is moved behind P2.

**Entry gate for the `--axis_center` code.** Cell `gate_ac`, the baseline command on seed 42 with the
new flag at its default, compared value-by-value against `base_s42`.

**2026-09-21, priority run.** Four additions, all outside the original §2 list, run in the order the
user set. (i) A metric-calibration control with no model and no training, `m1_calibration.py`, which
takes precedence over further mechanism search because every §4 verdict depends on M1's scale.
(ii) The MS-COCO transfer test of the .20/.60 window, which §5 already required of any carried arm;
it is run here even though the window failed §4 criterion 1, because its gain was on the secondary
endpoints, which §5 does not cover. (iii) `--quant_center_local` and `--quant_center_rescale`, new
arms motivated by the P5 result, screened against the same §4 rule with `p4drop` as their control
because they require `--lambda_text_code_kl 0.0`. (iv) `--lambda_text_code_kl 0.0` on MS-COCO, the
second-dataset check of the one objective simplification §P4 found. Endpoints and the screening rule
are unchanged. The vector endpoint remains post-hoc and decides nothing.

**Deviation recorded.** MS-COCO cells ran in a detached worktree at the same commit so the main tree
stayed editable; the four source files were verified byte-identical before launch. MS-COCO cells
were rebuilt from the Flickr cell plus per-dataset deltas read from `scripts/train_mscoco_F2_sweep_clip.sh`
and `scripts/phase3_selection_matrix.py`, not from a recorded MS-COCO `args.txt`, because none exists
on the current protocol. `--phase3_hf_identity_sha256` is carried over from the Flickr cell: it names
the frozen CLIP snapshot, whose own pins are identical in both commands, and it is not in either seal.
