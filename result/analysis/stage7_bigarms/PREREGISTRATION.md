# Stage 7 pre-registration (2026-09-22, written before any arm cell ran)

Branch `arch-exp-2026-09`, code `8e81a9b`, worktree `/data/yschoi/gdna_wt_arms`. Off-protocol screening,
Flickr25K stage-1 cell (selection recipe, stop after epoch 4), seeds 42/43/44, `hash_target_mode siglip_cos`.

## Arms (each = the `base_s*` command plus exactly these flags)

| arm | flags | what it tests |
|---|---|---|
| gate7 | none (seed 42 only) | the new code with every flag off must reproduce `base_s42` log.csv 280/280 |
| **bc** (B+C) | `--cibhash_local_target axis_soft --cibhash_local_target_tau 0.2` | local slots' NT-Xent target = axis-m caption neighbourhood (augmentation partner keeps ~.47 of the target mass at batch 64); global slot keeps the instance target |
| **ac** (A+C) | `--codebook_size 64 --concept_codebook_npz /data/yschoi/gdna_stage7/concepts_flickr25k.npz --lambda_concept 1.0 --concept_tau 0.5 --cibhash_local_target none --lambda_text_code_kl 0.0 --lambda_text_hash_ntxent 0.0 --codon_joint_slots 0` | local slots: named caption concept (k-means, 64 per axis, opt rows only) with a fixed bijective Hamming-aware codon layout, trained by concept cross-entropy; no instance NT-Xent on local slots; the concept CE replaces text_code_kl and text_hash_ntxent as the caption supervision of local slots; codon_joint kept on the global slot only (its uniform prior conflicts with unequal concept sizes) |
| k64 | `--codebook_size 64` | control: A also changes K 128 -> 64 for every slot |

The concept file is `/data/yschoi/gdna_stage7/concepts_flickr25k.npz`, sha256 `5c78cd0d11e49669…`
(`build_concepts.py`; its JSON sidecar has the full digest).

## Endpoints (held-out val rows of the stage-1 split, deployment forward, no text)

- R: mAP@R (`eval_mAP_at_R`, last row of log.csv)
- D: unique codes (`eval_unique_code_ratio`), dead codewords (`eval_dead_code_ratio_mean`)
- T: text-free reading of local codons, mean over the 4 axes, with the stage-2 reader
  (`zscr_pilot.py`: supervised ceiling, caption path, CLIP-only). For **ac** also the dictionary-free
  concept-name reading: each deployed local codon is read as its concept's caption-word posterior.
- secondary: per-axis values, codeword->own-axis (`quant_gap_diag.py`) where it applies

## Decision rule (thresholds on 3-seed means; descriptive, no significance test)

An arm is carried to NUS-WIDE and MS-COCO only if ALL hold:

1. **Interpretability.** Its best text-free reading T is at least base's supervised ceiling + .02
   (base .3469, so at least .367). It must also be at least base's ceiling on 3 of the 4 axes.
2. **Retrieval.** R is at least base − .03 (base .7453, so at least .715).
3. **Diversity.** Unique codes are at least .45.

If the **ac** arm passes but k64 alone moves R by more than half of ac's change, the retrieval
attribution is reported as confounded with K.

Nothing in this file is edited after the first arm cell starts. Deviations, if any, are appended
below with a timestamp.

## Deviation 1 (2026-09-22 12:06:15 +0900, after the first arms finished, before any A-prime cell ran)

The **ac** arm failed its own mechanism: training concept cross-entropy stayed at 4.35-4.63 against
ln 64 = 4.16, and the deployed concept equals the caption concept for 1.7-3.6% of val images (chance
1.6%). Cause, read from the code: the local codebooks are EMA-updated from NEAREST-codeword
assignments, so codeword k never becomes concept k. The cross-entropy pulls tokens toward a codeword
whose position is set by other tokens. This is an implementation failure of the intended mechanism,
not a test of the idea; **ac** is recorded as failed under the rule above.

**New arm ac2 (A-prime + C):** the **ac** flags plus `--concept_label_ema`. The EMA update of each
local codebook assigns every training token to its caption concept, so codeword k becomes the running
mean of concept-k slot tokens (a class prototype, as in prototypical networks). Output assignment is
unchanged (nearest codeword), so deployment still reads the image alone. Same seeds, same endpoints,
same decision rule. Entry gate **gate7b_s42**: all flags off must again reproduce base_s42 280/280.
