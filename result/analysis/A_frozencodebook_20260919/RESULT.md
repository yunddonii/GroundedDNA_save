# (A) frozen text-initialised codebook — Flickr25k, 4 arms x 3 seeds, 2026-09-19

Off-protocol exploration on branch `arch-exp-2026-09`. Same base recipe as (B):
the 2026-09-15 `p3lamA` stage-1 selection cell (5 slots / 5 codebooks / K=128 /
3 codons = 15 bases, epsilon schedule aligned, `hash_target_mode=siglip_cos`),
`selection_mode=select` + `val_split_ratio=0.1` so the official test split is
never read. Twelve cells, single deltas verified by command diff and by reading
`config.pt` back.

## Hypothesis

No active loss pins a codeword's meaning: `text_code_kl` is exactly invariant to
relabelling codewords, `codon_joint` targets a uniform distribution, and the
codebook is an EMA buffer with no gradient. So codeword identity is set by
initialisation and optimisation path alone, which is why it changes with the
seed. If codeword k were instead a FIXED text-derived vector, the symmetry
would never exist. `--codebook_freeze_after_epoch` was added to test this.

## Arms

| arm | text init | freeze | question |
|---|---|---|---|
| A1 | kmeans | never | does text init alone do anything? |
| A2 | kmeans | epoch 0 | fixed text-derived codewords throughout |
| A3 | kmeans | epoch 2 | late consolidation, as the VQ literature applies it |
| A4 | **none** | epoch 0 | **control**: separates "text init" from "not moving" |

## Results

| arm | mAP@R mean | sd | dead | unique code | vs baseline |
|---|---:|---:|---:|---:|---|
| baseline (no init, never frozen) | .7483 | .0139 | .258 | .530 | — |
| A1 kmeans, never frozen | .7439 | .0063 | .313 | .484 | −.004, inside seed noise |
| A2 kmeans + frozen@0 | .6327 | .0136 | .542 | **.027** | **−.116, outside** |
| A3 kmeans + frozen@2 | .7357 | .0181 | **.808** | .130 | −.013, inside |
| A4 random + frozen@0 | .7161 | .0187 | .482 | .275 | **−.032, outside** |

Baseline seed SD is .0139; a difference must exceed that to count.

## Verdict: the hypothesis is refuted, and in the informative direction

**Text initialisation alone does nothing.** A1 differs from baseline by −.004,
well inside seed noise, on every axis. This matches the 2026-05 decision to
discard `--text_init_codebook` before measuring it — the option turns out to
have been worth discarding.

**Freezing is harmful in every form.** A2 loses .116 mAP and collapses unique
code ratio from .530 to .027, i.e. the codes stop distinguishing images at all.
A3 keeps retrieval (−.013, inside noise) but leaves **81% of codewords dead**.

**The control settles the attribution.** A4 (random init, frozen) beats A2
(text init, frozen) on both retrieval (.716 vs .633) and unique codes
(.275 vs .027). So the text-derived initialisation is not merely unhelpful when
frozen — it is *worse* than freezing a random codebook. Whatever k-means on
adapter-projected caption embeddings produces, it is a poorer fixed vocabulary
than Normal(0, 1/sqrt(768)) noise.

One likely reason is visible in the code: `--text_init_codebook` runs at
`train_siglip2.py:788`, before the optimiser is built, and projects the raw
caption embeddings through `text_adapter` while that adapter is still randomly
initialised. The "text-derived" codewords are therefore text passed through a
random projection, not fixed concept embeddings. The literature's frozen-
vocabulary precedent (V2L Tokenizer, arXiv 2403.07874) instead uses genuinely
pretrained token embeddings plus a *trained* projector.

## A warning this produces for (B)

The frozen arms separate the text anchors far better than the baseline —
A2 .0189 and A4 .0299 against .13 — and the cost-stage cosine drops to .215 and
.301 against .84. Those are the same directions (B) treated as improvement, yet
these are the worst arms on every other axis. With the codebook held still the
text adapter moves instead and over-separates the axes at the cost of the
representation. **Anchor separation on its own is not evidence of a better
model**, and the (B) reading should be qualified accordingly.

## What would still be worth trying

Not this, at least not this way. The one mechanism in the literature that
provably breaks codeword permutation symmetry is a prediction head from a
codeword to a *specific* caption word (LG-VQ's masked text prediction). Unlike
the InfoNCE formulation already rejected in this project, a cross-entropy over
a fixed vocabulary has no collapse attractor, because the target does not
depend on the batch. If the frozen-vocabulary route is retried at all, it should
use real pretrained token embeddings with a trained projector, and freeze late
rather than from step 0.

## Why freezing fails: the frozen codewords become unreachable

Per-codebook perplexity (K=128), measured on 2000 train images through the
text-routed forward pass. cb0 is the global slot, which bypasses the router.

| cell | cb0 | cb1 | cb2 | cb3 | cb4 |
|---|---:|---:|---:|---:|---:|
| A1_s42 kmeans, never frozen | 38.2 | 48.6 | 48.2 | 42.0 | 46.0 |
| A1_s43 | 41.8 | 52.0 | 44.7 | 47.9 | 50.3 |
| A1_s44 | 40.7 | 47.4 | 45.5 | 43.7 | 46.3 |
| A2_s42 kmeans, frozen@0 | 74.9 | **4.0** | **4.6** | **4.7** | **7.6** |
| A2_s43 | 79.2 | **6.0** | **4.3** | **4.2** | **5.3** |
| A2_s44 | 49.4 | **4.8** | **3.8** | **3.3** | **6.6** |
| A3_s42 kmeans, frozen@2 | **9.7** | 7.2 | 6.3 | 6.4 | 6.6 |
| A3_s43 | **9.9** | 6.1 | 5.9 | 6.3 | 5.9 |
| A3_s44 | **10.0** | 6.8 | 6.5 | 6.0 | 6.3 |
| A4_s42 random, frozen@0 | 89.8 | **5.2** | **4.5** | **5.4** | **4.9** |
| A4_s43 | 24.1 | **6.1** | **5.5** | **7.7** | **5.8** |
| A4_s44 | 73.3 | **5.0** | **6.8** | **5.2** | **6.4** |

Freezing does not hold the codebook in place in any useful sense: it makes most
codewords **unreachable**. Every frozen arm uses roughly 4-8 effective codewords
per local codebook out of 128, against 38-52 when the codebook is free to move.
The global codebook cb0 survives in A2/A4 precisely because it bypasses the
router, and it too collapses in A3 once the encoder has spent two epochs adapting
to a moving codebook and then has the target pulled out from under it.

This is the mechanism behind the headline numbers: unique code ratio .027 for A2
is not a subtle degradation, it is five codebooks addressing a handful of
codewords each.

Caveat on the measurement: these perplexities come from the TEXT-routed forward
pass, not the deployment path that the extraction npz files use. They are
comparable to each other because every cell was measured the same way, but they
are not comparable to the deployment-path numbers quoted elsewhere.
