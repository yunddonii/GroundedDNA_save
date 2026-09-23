# Axis-centred anchors — scope decision and confirmatory acceptance criteria (2026-09-23)

Branch `arch-exp-2026-09`. Written to satisfy gates 2 and 3 of the private audit review
`anchor_recipe_review_20260923/REVIEW.md`, **before** any confirmatory evaluation of this
candidate. Nothing here approves a production rerun; it fixes what the rerun would have to show.

## 1. What the candidate is

`--axis_center anchors`: before the routing cost is computed, each local routing anchor has the
per-image mean of the active local anchors subtracted (`model_siglip2.py:2741`, applied at
`model_siglip2.py:4236`). Any global prefix is untouched. **No parameter and no loss term is
added.** Training centres caption-derived anchors; deployment centres codebook-mean anchors and
sees no text.

The weak global gate (`--global_gate_init_logit -3.0`) is **not** part of this candidate. Stage 11
rejected it under its pre-registered rule (retrieval fails on 3 of 4 datasets). It is not to be
re-run in search of a passing outcome.

## 2. Scope decision (audit gate 2)

**The target is a justified multi-label-only scope, not a common four-dataset recipe.**

- Anchors apply to **Flickr25K, NUS-WIDE, MS-COCO**.
- **CIFAR-10 keeps the incumbent recipe and its negative result is preserved and reported**: anchors
  cost −0.118591 mean mAP@R there, down in all three seeds. This is stated as a finding with its
  reason (a single-positive label set does not present four co-occurring axes per image, the same
  ground on which §4.7c already excludes CIFAR-10 from the interpretability aggregate), not removed
  from the paper and not hidden.
- **No dataset-specific recipe switching beyond this one declared scope boundary**, and the boundary
  is declared here before the confirmatory evaluation rather than chosen after seeing its result.

## 3. The interpretability claim this candidate is allowed to make

Supportable and to be claimed:

- **routing sharpness** — mean effective slots per patch falls (Flickr 3.07 → 2.42);
- **within-image axis alignment** — the probe asks which of the *same image's* four axis captions a
  slot's codeword is nearest to; anchors raise it on **12/12 dataset-seed pairs** (Flickr +.056…+.073,
  CIFAR +.062…+.149, NUS +.073…+.094, MS-COCO +.030…+.102), measured on ≤512 validation images.

Explicitly **not** claimed, per stage 10 and the audit's interpretability boundary:

- globally meaningful or nameable codewords — the nearest-caption naming procedure is at chance
  (axis specificity .2517 for anchors vs .2438 base, chance .25; name precision AP .2723 vs a prior
  of .2837);
- consistent word-level roles, or any improvement in the caption-path or codon word readings;
- semantic grounding inferred from retrieval or from code diversity. Neither is evidence for it.

## 4. Confirmatory retrieval acceptance margin (audit gate 3)

Fixed now, from the approved P3 refit aggregate `p3rfB_refit_aggregate.json`
(raw SHA256 `b4f3b0df…`), 3 seeds, official test:

| dataset | incumbent mAP@R | 1×SD (raw) | incumbent post-BIO mAP@R | 1×SD (post-BIO) |
|---|---:|---:|---:|---:|
| Flickr25K | 0.853461 | 0.006759 | 0.849807 | 0.006190 |
| NUS-WIDE | 0.822507 | 0.006384 | 0.816052 | 0.006972 |
| MS-COCO | 0.832919 | 0.001268 | 0.827119 | 0.000599 |

**Rule.** For each in-scope dataset, the anchored recipe's 3-seed mean must not fall below the
incumbent's mean by more than **one incumbent sample SD of that dataset and that metric**, on both
raw and post-BIO mAP@R. Failing either metric on any in-scope dataset rejects the candidate for
that scope; the scope is not renegotiated afterwards to rescue it.

Consequences accepted in advance: MS-COCO's incumbent SD is small (0.000599 post-BIO), so MS-COCO
carries the tightest bound of the three, tighter than Stage 11's fixed 0.005. The stage-1
exploratory MS-COCO difference (−0.0017 raw, held-out-train validation) already exceeds 1×SD(raw),
so **MS-COCO is the dataset most likely to reject this candidate.** That is recorded here, before
the run, rather than discovered and reinterpreted afterwards.

A sample SD from n = 3 is itself noisy. This is a pre-registered decision rule, **not a statistical
test**, and no equivalence or significance is claimed from it.

## 5. What a confirmatory campaign must do (audit gates 3–5)

1. Freeze an exact **typed** recipe — not `args.txt`. `args.txt` cannot express a negative value:
   its `<key><dashes><value>` layout runs the dashes into the value, so `-3.0` reads back as `3.0`
   (verified against `scripts/regen_viz_routing.py:_parse_args_txt`). Trained values survive only
   because the checkpoint carries them (`global_gate_logits` loaded with 0 missing / 0 unexpected
   keys, init +3.0 → trained −2.70…−2.35). Bind the recipe, source bytes, data/cache/split
   identities, schedule and selection rule in a typed artifact with digests.
2. Follow `docs/EXPERIMENT_PROTOCOL_AUDIT_2026-08-13.md:167` — train-only selection, no official
   test in selection, scratch full-train refit at the selected N, same protocol across seeds,
   locked terminal official-test evaluation. An existing selection may be reused only if its
   protocol and provenance actually cover this changed architecture.
3. Recompute raw and post-BIO results and every affected downstream analysis under an explicitly
   admitted campaign. **Main, ablation and interpretability results of the incumbent cannot be
   inherited by the changed routing geometry.**
4. Repeated use of the current validation set is exploratory evidence, not independent confirmation.
