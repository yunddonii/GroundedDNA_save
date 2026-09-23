# Stage 11 pre-registration — anchors + weak global gate on the remaining three datasets

Written before any cell of this stage runs. Branch `arch-exp-2026-09`, off-protocol exploration;
not a paper campaign.

## Question

Stage 9 tested `ancsoft` = axis-centred anchors + a weak global gate
(`--axis_center anchors --global_gate_init_logit -3.0`, gate starts at .047) on **Flickr25K only**,
3 seeds. It kept retrieval (mAP@R .7512 vs anchors .7513) and raised code diversity
(unique .5397 -> .5783, 3/3 seeds; dead codewords .186 -> .148). This stage asks whether that
holds on CIFAR-10, NUS-WIDE and MS-COCO.

## Cells

9 = {cifar, nus, msc} x seeds {42, 43, 44}, `cells.txt`. Each is its dataset's `p2anc` cell from
`result/analysis/stage6_p2anc/cells_all.txt` with exactly two changes, verified by parsing both
command lines: `--tag` and `--global_gate_init_logit 4.595 -> -3.0`. Run in
`/data/yschoi/gdna_wt_mscoco` (8cae54d), the same tree and commit that produced the `base` and
`p2anc` comparators for these three datasets, so the comparison is paired on source bytes.

## Comparators (already on disk, same tree, same seeds)

`base` (no anchors, default gate) and `p2anc` (anchors, default gate), 3 seeds each per dataset.
Flickr25K's three arms come from stage 9 and are reported unchanged.

## Metrics and how they are produced

The stage-6b pipeline, unchanged: `eval_one.sh` then `anchor_analysis.py`.
Primary: mAP@R (dataset-specific R) and unique code ratio, both as the existing pipeline computes
them for every arm, so all arms share one denominator. Secondary: dead codeword share, effective
slots per patch, code->axis match, local codon NMI, per-axis supervised ceiling, caption/CLIP
readings, M1, label AP@k5.

## Decision rule, fixed now

Adopt `weak gate on top of anchors` for the final recipe only if both hold:

1. **Retrieval**: mean mAP@R of ancsoft is not below p2anc by more than 0.005 on any of the four
   datasets, and
2. **Diversity**: mean unique code ratio of ancsoft is above p2anc on at least 3 of the 4 datasets.

If 1 fails on any dataset, the weak gate is discarded and anchors alone stay.
If 1 holds and 2 fails, the gate is recorded as neutral and not adopted.

## What this stage does not claim

n = 3 per cell. **No statistical test is run**; means, sample SDs and per-seed deltas are reported
as description only. Nothing here is a paper number, and no dataset's interpretability verdict
changes on retrieval alone.
