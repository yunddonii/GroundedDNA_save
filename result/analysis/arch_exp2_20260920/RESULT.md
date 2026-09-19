# Slot-role mechanisms a-1 / a-2 / b-1 / c-1 — Flickr25K, 3 seeds each, 2026-09-20

Off-protocol, branch `arch-exp-2026-09`, code commit `8925504`. Nothing here may enter
`docs/paper_draft/`. Design, arms, measurements and the decision rule were fixed in
`PREREGISTRATION.md` before any arm ran; its decision log records each step as it happened.

## Validity checks done first
- **Entry gate.** With every new flag off, the final code reproduces the pre-change run in all 280
  logged values (`base_s42` vs `gate1_incumbent_s42`; terminal val mAP@R 0.7588274741 both).
- **Same-mode baseline.** Campaign-mode cells are not comparable: the identical command launched
  directly diverges from epoch 1 (campaign .7635532117 vs direct .7588274741; both deterministic;
  not the code, not the GPU, not the best-checkpoint save; cause unidentified). The baseline is
  therefore the same recipe launched directly, seeds 42/43/44.
- **Mechanism unit checks** (default path bit-identical to the frozen router; column mass follows a
  target; a free column takes cost-driven mass; bypassed quantizer is the identity and leaves the
  codebook untouched) and a smoke run of each arm, all before the gate.
- **Probe controls.** M1 with validation codes shuffled across images gives D ≈ .227 against a real
  diagonal ≈ .359, so the probe measures code–caption association and not a constant.

## Results (mean ± SD over seeds 42/43/44; retrieval = terminal val mAP@R on held-out train)

| arm | mAP@R | dead | unique | M1 role adv. | own-slot argmax /4 | M2 label AP | M3 min/median pp | M4 mass max/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | .7453 ± .0133 | .241 | .536 | .0065 ± .0053 | 1.7 | .796 ± .019 | .58 ± .34 | 1.32 |
| a-1 mass targets | .7476 ± .0099 | .239 | .547 | .0062 ± .0093 | 1.7 | .795 ± .018 | .53 ± .21 | 1.79 |
| a-2 + free null | .7165 ± .0309 | .531 | .346 | −.0070 ± .0030 | 1.0 | .673 ± .079 | .69 ± .25 | 1.68 |
| b-1 entity completion | .7566 ± .0075 | .253 | .589 | .0021 ± .0053 | 1.3 | .811 ± .002 | .91 ± .12 | 1.20 |
| c-1 two-stage VQ | .7429 ± .0109 | .050 | .489 | .0011 ± .0034 | 0.7 | .824 ± .002 | .94 ± .04 | 1.08 |

Pre-registered rule (operational, not a significance test; n = 3): pass iff mAP@R ≥ .7320 AND
M1 advantage rises by more than .0053. **No arm passes.** a-1, b-1, c-1 keep retrieval; none raises
the role advantage; a-2 fails both halves. b-1' was not run (it was reserved for a b-1 retrieval
failure, which did not happen).

## What each mechanism did
- **a-1** did what it was built to do: per-image slot mass became content-dependent (max/min 1.32 →
  1.79) with no empty slot in any image. Nothing downstream changed. Moving mass between slots does
  not change what a slot's code means.
- **a-2**: the free null column absorbed 15 %, 11 % and 46 % of patch mass on average across the three
  seeds (0–86 % per image), dead codes doubled and label decoding fell by .12. The dustbin idea does
  not survive a free marginal either.
- **b-1** learned its task — within-batch completion accuracy rose 17 % → 53 % (chance 1.6 %) in
  every seed — and made the code better on every axis except the one it was meant for: retrieval
  +.011, unique codes +.05, label decoding +.015, and the one-codebook collapse gone (min/median
  .58 → .91). The completion target distinguishes IMAGES, so any slot that carries the image's content
  can solve it; nothing penalises slot m for carrying axis m' just as well.
- **c-1** removed the seed-dependent collapse outright (every codebook at perplexity 80–121 in all
  three seeds; dead codes .24 → .05) and gave the largest label-decoding gain (+.028), at a small cost
  in code uniqueness (−.047). It too leaves roles unchanged.

## Reading
Across four mechanism families — slot marginals, a background dustbin, per-slot entity completion,
two-stage quantisation — the role advantage never rose above the baseline's ~.007. The codes do carry
caption semantics (diagonal .36 against .23 shuffled), but each slot's code predicts the other axes'
words about as well as its own: the slots are redundant carriers of image-level content. That is the
same conclusion the 2026-07 cross-slot matrix reached, now robust to four interventions.

The b-1 result points at the missing ingredient. Every objective in this model — CIBHash per-codebook
instance discrimination (≈ 82 % of the loss budget), the text-hash InfoNCE, and b-1's completion — is
contrastive over IMAGES. None contrasts one axis against another WITHIN an image. A role-specific
objective would have to: e.g. slot m's code must match axis m's entities better than the entities of
axis m' of the same image (in-image cross-axis negatives). That design has not been tried in this
project (no log entry) and is the natural next step if slot roles are the goal.

Two by-products are worth recording regardless of roles: b-1 and c-1 both eliminate the
seed-dependent codebook collapse found on Flickr25K, by different routes.

## Records
Run dirs `result/260920+flickr25k_setting1_{base,arm_a1,arm_a2,arm_b1,arm_c1}_s4{2,3,4}+…` (stored
under `/data/yschoi/gdna_archexp_result/`, symlinked), probe JSONs `probe/*.json`, aggregate
`aggregate_base_arm_a1_arm_a2_arm_b1_arm_c1.json`, commands `cells_*.txt`, sources `SOURCE_MANIFEST.json`.
