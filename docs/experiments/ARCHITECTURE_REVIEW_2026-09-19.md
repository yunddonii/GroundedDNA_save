# What would have to change for a slot to mean the same thing twice

Written for: the GroundedDNA author, deciding where to spend the next round of
architecture work. Off-protocol: nothing here has been folded into the paper
model, and none of it belongs in `docs/paper_draft/`.

## 0. Read this before the rest: which claim is at stake

The project's own consolidated verdict (2026-07) already settled what the paper
rests on:

| claim | status |
|---|---|
| (1) each slot explains **its own** semantic part | refuted on Flickr25k |
| (2) similar meaning -> similar codeword **within** a slot | confirmed vs chance on 3/3 |
| (3) (2) survives quantisation to the codon | confirmed on 2/3 |

plus the held-out decoding advantage over flat hashes, recorded there as "the
strongest and most uniform result in the project".

Claims (2) and (3) are statements *inside* one trained model. Neither needs
axis `m` to mean the same thing in a differently-seeded run. So the instability
this document is about sits on claim (1) -- the one already withdrawn -- and
fixing it would not strengthen what the paper currently argues. It would buy a
*new* claim, and only a narrow one (see section 7). Spend the next round here
only if a seed-stable axis is something you want to claim; if the goal is to
shore up the existing argument, sections 6.2 and 6.3 are the relevant items and
the rest is optional.

## 1. The failure, stated precisely

The dissection probe asks, per semantic axis, which MS-COCO category that axis
localises best. Across three seeds of the same approved recipe:

| axis | s42 | s43 | s44 |
|---|---|---|---|
| primary_object | giraffe .135 | dog .026 | person .026 |
| secondary_object | laptop .033 | umbrella .022 | person .023 |
| activity_relation | cake .029 | person .006 | orange .008 |
| color_texture | refrigerator .017 | bench .016 | cake .095 |

No axis keeps its best category. Three-way top-5 overlap is 1/5, 0/5, 1/5, 0/5,
and the only category shared at all is `person`, the most frequent one in the
dataset -- a frequency artifact, not a semantic one. `activity_relation` at s43
scores *below* its own shuffled control (.0056 vs .0099).

This is not new. The 2026-07 consolidated verdict already recorded claim (1)
"each slot explains its own semantic part" as refuted on Flickr25k. What is new
is that the instability is now measured across seeds, and that the mechanism
has been located.

## 2. Where the failure is NOT

Four candidate causes were measured this session and each is excluded.

**Not the text anchors.** Mean pairwise cosine between the four local text
anchors is .13 after the per-slot adapter, down from .59 at the raw CLIP
embedding. The adapter separates the axes well. The probe's own criterion --
"if stage 2 is already near 1.0 the anchors are the bottleneck" -- is not met.

**Not the entropic term.** Replaying the Sinkhorn plan from the identical cost
matrix at eps 1.0, 0.5, 0.3, 0.1, 0.05, 0.02 moves the centered slot cosine
from -.3056 to -.3289 -- essentially flat. Raw cosine moves a lot (.994 to
.194), but raw cosine is dominated by the shared "which patches carry mass"
envelope. Epsilon is not a lever.

**Not the top-p mask.** Stages 4 and 5 of the pipeline agree to four decimals
(.9946 / .9946 pre- and post-mask). An earlier reading that the mask does all
the differentiating came from the 6-slot, epsilon-misaligned era and does not
describe the current model.

**Not routing starvation.** In the run where codebook 1 collapsed to perplexity
6.0 of 128, that slot still received 21.74% of transported patch mass against
21.73% in a healthy seed. The collapsed axis is fed normally and fails
downstream, inside VQ.

## 3. Where the failure is

**One axis per run loses its codebook, and which one is seed-dependent.**
Measured from deployment-path extractions, per-codebook perplexity (K=128):

| run | cb0 | cb1 | cb2 | cb3 | cb4 | min/median |
|---|---:|---:|---:|---:|---:|---:|
| flickr s42 | 78.7 | **6.0** | 77.3 | 85.3 | 79.4 | 0.08 |
| flickr s43 | 77.6 | 83.5 | 67.1 | **44.9** | 77.4 | 0.58 |
| flickr s44 | 61.1 | 82.0 | 78.7 | **51.9** | 76.0 | 0.68 |
| mscoco s42 | 117.4 | 93.5 | **82.4** | 107.2 | 95.4 | 0.86 |
| nuswide s43 | 86 | 85 | **76** | 77 | 76 | 0.99 |

Across fifteen 5-slot runs the weakest axis changes with the seed in every
dataset, and the two catastrophic cases (ratio .07, .08) are both Flickr25k --
5,000 training images. MS-COCO at 107k is uniformly healthy. The project log
already contains the same phenomenon under a different name: "the dead slot
MOVES rather than disappearing", and a Qwen2.5 -> Qwen3 caption swap that took
cb1 from 60.9% dead to 9.4% while cb3 rose 12.5% -> 29.7%. Total collapse is
conserved; only its location moves.

**No active loss can pin a codeword's meaning.** This is a property of the
objective, not a bug:

- `text_code_kl` builds K-way distributions over codewords for the visual and
  text views and minimises their KL. Permute the rows of codebook `m` and both
  `logits_v` and `logits_t` permute identically, so the loss is *exactly*
  unchanged. Its own docstring confirms why nothing resists: gradient flows
  through `z_visual` only, `z_text` is detached, and `C` is an EMA buffer that
  receives no gradient at all.
- `codon_joint` (active, lambda .02) pushes the codon distribution toward
  *uniform*. A uniform target is the most permutation-symmetric target there is.
- `codeword_codon_sinkhorn`, which would have forced a codeword-codon
  bijection, is 0.0 in the approved recipe.
- `_loss_codebook_ortho` acts on per-codebook batch-mean directions: it
  separates slots, and says nothing about codewords within a slot.

So the codebook is a closed loop -- the encoder picks a codeword, EMA drags
that codeword toward the encoder, the encoder follows -- with no external
anchor. Codeword identity is determined by initialisation and optimisation path
alone, which is exactly what "changes with the seed" means.

## 4. What the literature does and does not offer

**The guarantee you want does not exist yet.** Kori et al. (NeurIPS 2024,
arXiv 2406.07141) is the strongest identifiability result for slot attention,
and it certifies representations only *up to permutation and invertible affine
transformation*. The authors say plainly that slot 1 in run A may correspond to
slot 3 in run B, and that semantic meaning holds within a single trained model,
not across seeds. Requiring axis `m` to mean the same thing across seeds is
strictly stronger than the field's best theorem.

**Text-aligned codebooks do not close it either.** LG-VQ (arXiv 2405.14206)
transfers text semantics into the codebook through global alignment, masked
text prediction and a relationship loss -- but the relationship loss constrains
*pairwise* similarities between codes, which is permutation-invariant, and the
paper reports no seed-stability analysis. TA-VQ (arXiv 2503.01261, CVPR 2025)
aligns codes to word/phrase/sentence granularities and likewise never measures
whether an individual code keeps its meaning. CTRL-O (arXiv 2503.21747)
conditions slots on language queries but runs no query-swap control and reports
no seed variance. The one component in this literature that genuinely breaks
the symmetry is LG-VQ's masked text prediction: predicting a *specific* word
from a *specific* code is not invariant to relabelling codes.

**The one precedent that gets seed-independence is a frozen vocabulary.** The
V2L Tokenizer (arXiv 2403.07874) uses frozen LLaMA-2 token embeddings as the
codebook, with a trainable projector bridging language and vision and a
CLIP-based filter cutting 32,000 tokens to 11,908. Because the codewords never
move, the same visual content maps to the same token id in every run. It did
not cost quality: FID 3.41 against VQ-GAN's 5.48.

**And the warnings against it are specific.** Frozen pretrained embedding
spaces are anisotropic: semantics live in direction while VQ assigns by
Euclidean distance, so high-norm codewords can dominate assignment on magnitude
alone; imposing a unit-sphere prior fixes that but degrades reconstruction. A
frozen vision representation is also "not necessarily cluster-structured or
quantization-friendly". And in practice freezing is used as *late-stage
consolidation* to stop latent drift, not as an initial tokenisation assumption.
Two of these bite us less than they bite a generic VQ-VAE, because this model
already normalises and compares by cosine throughout; the third -- freeze late,
not early -- shaped the experiment design below.

## 5. What was tried here, and what it showed

**(B) loss-budget rebalancing** -- the hypothesis that the router and alignment
losses are too small a share of the budget. Raising `lambda_wasserstein` was
measured on the eight existing lambda cells and is *wrong-signed*: the text
anchors collapse toward each other (cosine .16 -> .26 -> .46 as lambda goes
.15 -> .30 -> .50) and dead codewords rise past seed noise (+.24 at .50).
Raising `lambda_text_hash_ntxent` is the opposite: anchors separate threefold
(.129 -> .055 -> .044 across 3 seeds each), retrieval is unchanged (means span
.0036 against a seed SD of .008-.014), dead stays inside noise. But the routing
plan does not follow -- centered slot cosine moves .294 -> .302, against a
metric structurally pinned near -1/(M-1) = -.333. **Rebalancing reaches the
text side and stops there.**

**(A) frozen text-initialised codebook** -- `--codebook_freeze_after_epoch`
suppresses the EMA write and dead-code revival from a chosen epoch, so a
codebook initialised by the existing (but never-measured, discarded in
2026-05) `--text_init_codebook` stays where it was put. Four arms x 3 seeds
separate the two explanations: A1 init-only, A2 frozen from the start, A3
frozen late (the literature's usage), A4 random init frozen -- the control
without which "text init helped" cannot be told from "not moving helped".

## 6. What I would and would not spend the next round on

**Worth doing.**

1. *A masked-caption-word prediction head off each codeword.* This is the only
   mechanism found in the literature that provably breaks codeword permutation
   symmetry, and it is a head, not an architecture change. The risk is known
   and documented in this project: an InfoNCE formulation of the same idea
   (`L_text_codeword_contrastive`) collapsed the codebook at every lambda
   tested, because with K=128 codewords and more concepts than that, the loss
   is *minimised* by collapsing to a few clean clusters. A cross-entropy over a
   fixed vocabulary does not have that attractor, because the target is fixed
   rather than defined by the batch.
2. *Report the seed-dependent collapse as a finding.* One axis per Flickr run
   loses its codebook and the position is arbitrary. That is a real, measured,
   reproducible property of small-dataset training with this objective, and it
   bounds what any slot-role claim can say.
3. *Principled loss balancing (GradNorm or uncertainty weighting).* Eleven
   active loss terms are currently balanced by hand-tuned constants that a
   selection campaign showed are indistinguishable within seed noise. This is
   not a fix for the identifiability problem, but it removes a confound from
   every future comparison.

**Not worth doing.**

4. *Learnable shared queries (BO-QSA style).* Learnable queries are still
   learned, so the permutation freedom survives; follow-up work says cross-run
   concept binding remains unsolved even with them. And our measurements say
   the slots already differentiate at the routing stage -- this would improve a
   stage that is not failing.
5. *More epsilon or top-p tuning.* Both were measured inert this session.
6. *Re-running `L_text_codeword_contrastive` at another lambda.* Already
   rejected at every lambda tested, with a written mechanism for why lower
   lambda collapses *more*.
7. *Orthogonality losses.* `--lambda_codebook_ortho` exists, acts on slot-level
   batch means, and cannot touch codeword identity; and the orthogonality
   requirement was itself withdrawn in 2026-07 as imported from a protocol that
   does not apply here.

## 7. The honest ceiling

Even a fully frozen text-derived codebook buys seed-stable *codeword* identity.
It does not buy the claim that axis `m` attends to the region a human would call
axis `m` -- the dissection result -- because that is about where visual mass is
routed, and routing is already uniform across axes. Those are two different
claims and only the first is within reach of the change being tested here.
