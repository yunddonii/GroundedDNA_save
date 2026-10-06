# Stage 1 pre-registration — baseline, text-OFF and two clean-ups at the approved recipe (text-path line)

Written 2026-10-06 before the first cell. Branch `text-diag-2026-09`, worktree `/home/yschoi/gdna_textdiag`.
Off-protocol exploration: commands rebuilt from the approved ancS7 seed-42 `args.txt` with the seal /
authority arguments removed and `--no_gumbel_softmax` made explicit; never a paper number. Audited
checkpoints are not loaded. Idle GPUs only (another session's campaigns have priority).

## Cells (base + one delta; seeds 42/43/44; Flickr25K first, then NUS-WIDE and MS-COCO)

| cell | delta vs B0 | purpose |
|---|---|---|
| B0 | none (N = 4 / 4 / 39, `use_gumbel_softmax False`, `hash_target_mode siglip_cos`, 5 slots, `axis_center anchors`) | reference for every later arm |
| OFF | `--disable_text_supervision` | text-OFF reference (codebook-mean routing in training, text losses 0) |
| H1 | `--lambda_anchor 0 --lambda_cibhash_kl 0 --lambda_codon_text_anchor 0 --lambda_recon 0` (seed 42 only) | tier-0 loss clean-up: weights must be bit-identical to B0 |
| H2 | `--bidirectional_token_prune` removed (EOS-pooled caption anchor) | D6: noise-keyed pruning changes 21–32 % of codewords |
| W | H2 + whitening fitted on local slots 1–4 only (new npz) | isolate the effect of dropping the unused 5th slot from the fit |

## Entry gates
- One real seed-42 Flickr B0 cell before any batch: `args.txt` shows `use_gumbel_softmax False`,
  `hash_target_mode siglip_cos`, `num_semantic_parts 5`, no `phase3_input_seal`; last-epoch validation
  mAP@R within ±.01 of the campaign run (.7642). A larger gap is reported, not hidden.
- Distinct output directory per cell (tag carries dataset, cell, seed).

## Measurements (every cell)
- Retrieval: last-epoch validation mAP@R from `log.csv` (CalcTopMap, R = 5000).
- Codebook health: dead-codeword share, unique-code ratio, worst-local-codebook perplexity (from the run's
  final-eval JSON / `log.csv`).
- A3 v2 (`result/analysis/textdiag_2026-10/a3_v2.py`, all validation rows, lexical rule, boot 1000):
  S at codeword and codon level with CIs; calibration curve.
- D4 re-run on B0 (`d4d6_train_vs_deploy.py`): S of the caption-routed codes vs the deployed codes.

## Pre-declared readings and rules
- **Thresholds fixed by this stage:** T = max(2 × SD_seed[S(OFF)], smallest f whose calibration CI excludes 0
  converted to S); base values for P-RET (B0 mean, SD) and P-HEALTH (B0 dead / unique / perplexity).
- **H1 pass:** `torch.equal` on every parameter tensor vs B0 seed 42, and every shared `log.csv` column
  identical; `train_loss` lower by .05 × anchor up to float32 rounding. Pass → the 10-term recipe is the
  base for all later cells (recipe identity changes; documented).
- **H2 / W pass (each vs its own base, paired by seed):** mean val mAP@R ≥ base − .008 and no seed below
  its base seed by more than .02; dead ≤ max(.30, base + .05); unique ≥ base − .08; S not lower than base
  by more than T. Pass → H2 becomes B1 (and W replaces it if W passes on top).
- **Reported, not judged:** S(B0) vs S(OFF) at codon level (Stage 0 saw +.12 vs −.06 on exploratory
  checkpoints); D4 gap at the approved recipe.
- n = 3 seeds, descriptive; no significance test is claimed.

## Stop / escalation
- A B0 smoke outside ±.01 of the campaign run stops the batch until the cause is named.
- Any cell that fails P-HEALTH is terminal for that delta; no rescue tuning.
