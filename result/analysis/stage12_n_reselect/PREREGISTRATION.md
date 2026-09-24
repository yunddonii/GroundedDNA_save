# Stage 12 pre-registration — re-select N for the anchored recipe

Written before any cell of this stage runs. Branch `arch-exp-2026-09`, off-protocol exploration.

## Why

Every arm compared so far (base, anchors, anchors+gate) stopped at the **incumbent's** selected N
— CIFAR-10 19, Flickr25K 4, NUS-WIDE 4, MS-COCO 39 — chosen by the approved campaign for the
incumbent routing geometry. Anchors change that geometry, so the recorded anchored numbers are the
new recipe read at another recipe's stopping point. `docs/ANCHOR_SCOPE_AND_ACCEPTANCE_2026-09-23.md`
and audit gate 4 both require selection under the recipe being proposed.

## Cells

13 new cells = 4 datasets x candidate N {4, 9, 19, 39}, **seed 42 only** (the approved selection
uses seed 42 and reuses its N for 42/43/44), minus three anchored cells that already exist at the
incumbent N in the same tree (cifar10@19, nuswide@4, mscoco@39). Flickr25K@4 is re-run here so all
four of its candidates come from one tree.

Each cell is its dataset's existing anchored cell with only the schedule moved: `--tag`,
`--stop_after_epoch N`, `--sinkhorn_schedule_horizon N+1`. Verified by parsing both command lines;
nothing else differs, and the incumbent-N re-run differs in the tag alone. This reproduces the
approved stage-1 selection shape, which those cells already carry: `-e 60 --stop_after_epoch N
--lr_schedule_horizon 60 --sinkhorn_schedule_horizon N+1`, `--val_split_ratio 0.1
--val_split_seed 42 --selection_mode select --dna_distance_mode base`. Tree
`/data/yschoi/gdna_wt_mscoco` at 8cae54d, as in stage 11.

## Selection rule, fixed now

Identical to `scripts/phase3_select_n.py`: **argmax of raw base-Hamming mAP@R at the candidate's own
terminal epoch; ties go to the smallest N.** The value is the terminal `eval_mAP_at_R` row of that
cell's `log.csv`, on the 10 % held-out train split. **No official test is read at any point.**

## What is then reported

1. The anchored N per dataset, next to the incumbent N.
2. The anchors-vs-incumbent retrieval comparison **re-read at each recipe's own N** — the incumbent
   at its N (already on disk) against anchors at the N chosen here. This replaces the stage 6b/11
   comparison, which pinned anchors to the incumbent's N.
3. CIFAR-10 is included although it is out of the adopted scope: the scope decision rests on the
   CIFAR-10 cost, and that cost must be measured at CIFAR's own anchored N, not at the incumbent's.

## What this stage is not

Not a confirmatory campaign and not a protocol launcher run. `scripts/phase3_selection_matrix.py`
cannot express `--axis_center` without editing a protocol source, which would break `--recipe`
replay; admitting the anchored recipe into that launcher is a separate decision that belongs to the
confirmatory campaign, together with the scratch refit at the selected N and the locked official
test. These are exploratory selection cells on the same 90/10 split, n = 1 seed, and no statistical
test is run.
