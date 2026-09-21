# arch-exp-3 results

Rules and endpoints: `PREREGISTRATION.md` (written before any number here was read).
Baseline throughout: the `base_s*` cells of `../arch_exp2_20260920/`, same tree, same launcher,
same launch mode.

## P1 — routing sharpness (no code change, no loss change)

`--routing_adaptive_topp_min/max` moved from .6/.95 to .3/.7 (P1a) and .15/.5 (P1b). Seeds 42/43/44,
`archexp3_p1w1`, rc=0, 137 s for all six cells.

| | base .6/.95 | P1a .3/.7 | P1b .15/.5 |
|---|---:|---:|---:|
| patches per slot (`val_routing_mean_effective_k`) | 3.068 ± .087 | 1.776 ± .049 | 1.328 ± .020 |
| patches reaching exactly one slot | .040 ± .016 | .274 ± .038 | **.672 ± .020** |
| **M1 role advantage (primary)** | **.0065 ± .0043** | **.0017 ± .0014** | **−.0002 ± .0062** |
| M1 per seed | .0087 / .0104 / .0004 | .0034 / .0014 / .0002 | .0070 / .0006 / −.0082 |
| axes whose own slot is argmax (of 4) | 3 / 2 / 0 | 0 / 1 / 2 | 1 / 2 / 1 |
| mAP@R | .7453 ± .011 | .7511 ± .003 | .7447 ± .012 |
| M2 label decoding | .796 ± .015 | .813 ± .002 | .813 ± .002 |
| M3 codebook min/median perplexity | .582 ± .277 | **.894 ± .091** | **.884 ± .060** |
| dead codes | .241 ± .029 | .251 ± .053 | .177 ± .047 |
| unique codes | .536 ± .014 | .575 ± .018 | .593 ± .016 |
| per-image slot mass max/min | 1.32 | 1.73 | 2.05 |
| images with an empty slot | 0 % | 0 % | 0 % |

**Screening verdict: both arms fail criterion 1 and are not carried.** M1 does not rise; it falls
monotonically with sharpness, and at P1b the mean is indistinguishable from zero with a seed SD
larger than the effect.

**What it does buy, at no cost.** Sharpening removes the seed-dependent codebook collapse outright
(min/median perplexity .58 → .89, and the .277 seed spread that made the baseline unpredictable
becomes .09), raises caption-free label decoding, raises unique codes, lowers dead codes at P1b, and
leaves retrieval where it was. It reaches the collapse repair that c-1 (two-stage k-means codebook,
arch-exp-2) reached, with **no code change and no extra loss** — one flag pair.

**What it settles.** The mask is not what keeps the slots redundant. At P1b two thirds of all patches
reach exactly one slot, so the partition is as hard as this router can make it, and the slots still
decode every axis equally. A hard partition along uninformative directions is still uninformative:
the remaining suspect is the transport cost itself, `1 − cos(patch, axis anchor)`, whose columns are
nearly parallel. P2-anchors is the direct test, and this result is why it was promoted ahead of the
pre-registered P3 (slot-choice masking), which is another mask-direction change.

## P2 — axis-deviation representation (code added, default off; entry gate 280/280 identical)

`--axis_center anchors` subtracts the per-image mean across the four axis anchors before the transport
cost; `--axis_center readout` subtracts it from the slot tokens and from the text tokens the losses
see. Seeds 42/43/44, `archexp3_p2w1`, rc=0, 137 s.

| | base | P2 anchors | P2 readout |
|---|---:|---:|---:|
| **M1 role advantage (primary)** | **+.0065 ± .0043** | **+.0066 ± .0085** | **−.0022 ± .0026** |
| M1 per seed | .0087 / .0104 / .0004 | −.0052 / .0149 / .0100 | .0012 / −.0026 / −.0052 |
| mAP@R | .7453 | **.7513** | **.7009** |
| M2 label decoding | .796 | .805 | **.647** |
| M3 codebook min/median | .582 | **.882** | .520 |
| dead codes | .241 | .186 | **.408** |
| unique codes | .536 | .540 | .638 |
| patches per slot | 3.07 | 2.42 | 2.77 |

**Screening verdict: both fail criterion 1; readout also fails criteria 2 and 3.**

Anchors behaves exactly as the algebra predicts and for the reason the algebra gives. Sinkhorn is
invariant to any per-row term in the cost, and with the four anchors nearly equidistant from their own
mean, centring them is close to a per-row shift plus a common rescaling — that is, an effective
epsilon change. It sharpens (3.07 → 2.42 patches per slot), repairs the collapse like P1 did, adds
.006 mAP, and leaves M1 where it was, with a seed spread twice the mean.

Readout is the informative failure. Removing the image-common component from the slot tokens costs
.044 mAP, doubles dead codes and drops caption-free label decoding by .15. The shared component the
slots carry is not a defect to be subtracted: it is most of what the retrieval code is made of, and
the global slot does not carry enough of it to compensate.

## P3 — removing the global codeword that every slot receives (no code change)

Before the codon heads, every local slot receives `q_local_m + sigmoid(alpha_m) * q_global` with a
learned gate of **.993**. Two arms: `--disable_global_gate` (the addition is skipped) and
`--global_gate_init_logit=-3.0` (the gate is learnable but starts at .047). Seeds 42/43/44,
`archexp3_p3gate`, rc=0, 134 s.

| | base (gate .993) | gate removed | gate started at .047 |
|---|---:|---:|---:|
| **M1 role advantage** | **+.0065 ± .0043** | **+.0062 ± .0042** | **+.0080 ± .0043** |
| mAP@R | .7453 | .7456 | .7385 |
| dead codes | .241 | .157 | .128 |
| unique codes | .536 | .587 | .594 |
| M3 codebook min/median | .582 | .643 | .557 |
| learned gate at the end | .993 | (skipped) | **.045 – .064** |

**Screening verdict: fails criterion 1.** This replicates the 2026-07-20 refutation in the present
5-slot unsupervised stage-1 recipe and on a different role metric. The soft arm adds the fact that
term was missing then: started near zero, the gate **stays** near zero (.045–.064) and nothing else
changes, so the .993 of the standard recipe is not something the optimiser is driving towards.

## P4 — the in-image cross-axis term, as a replacement for `text_code_kl`

`--lambda_role 0.05 --role_tau 0.07` with `--lambda_text_code_kl 0.0`: the objective keeps twelve
terms. Control arm `p4drop` removes `text_code_kl` and puts nothing in its place (eleven terms).
Entry gate for the new code: 280/280 identical. Seeds 42/43/44, `archexp3_p4w1`, rc=0, 130 s.

| | base | role replaces text_code_kl | text_code_kl simply removed |
|---|---:|---:|---:|
| **M1 role advantage** | **+.0065 ± .0043** | **+.0005 ± .0011** | **+.0078 ± .0009** |
| mAP@R | .7453 | .7113 | .7409 |
| dead / unique | .241 / .536 | .349 / .379 | .232 / .517 |
| M2 label decoding | .796 | .710 | **.803** |
| M3 codebook min/median | .582 | .358 | **.672** |

**Screening verdict: both fail criterion 1**, and the substitution fails 2 and 3 as well.

Two things worth keeping. First, `text_code_kl` is removable: dropping it costs .004 mAP (inside the
baseline's own .011 seed SD), improves the codebook on every diversity measure, and leaves the role
metric where it was — an objective of eleven terms that behaves like the twelve-term one. Second,
the new term does not get minimised at the codeword: `train_loss_role` ends at 3.13 against a chance
value of ln 4 = 1.386, so the model is confidently matching slot m to the *wrong* axis.

## P5 — where the axis signal is lost: before or at quantisation

Same term at `--lambda_role 1.0 --role_tau 0.2`, scored either on the codeword (`--role_source
quantized`) or on the slot token the quantiser receives (`--role_source pre_quant`). Seeds 42/43/44.
(The first launch lost three checkpoints to a full root filesystem and two more were written
truncated; all five cells were re-run after the run directories were relocated, and every checkpoint
was then verified to load, not merely to exist.)

`train_loss_role` per epoch, seed 42, chance = 1.386:

| arm | epochs 0 → 4 |
|---|---|
| on the codeword, λ = 0.05 | 2.21 → 1.78 → 1.47 → 1.25 → 1.05 |
| on the codeword, λ = 1.0 | 2.74 → 3.72 → 3.87 → 3.73 → **3.59** |
| on the slot token, λ = 1.0 | 1.19 → 0.93 → 0.64 → 0.47 → **0.33** |

| | base | on codeword λ=1.0 | on slot token λ=1.0 |
|---|---:|---:|---:|
| **M1 role advantage** | **+.0065 ± .0043** | **−.0000 ± .0057** | **+.0069 ± .0047** |
| mAP@R | .7453 | .6624 | .7001 |
| unique codes | .536 | **.045** | **.728** |
| M3 codebook min/median | .582 | .624 | .757 |

**Screening verdict: both fail criterion 1.** The continuous slot token can be made to carry the axis
(loss .33, i.e. the four-way match is solved); pressing the same objective onto the codeword instead
drives it far above chance and collapses the codebook to 4.5 % unique codes.

## A post-hoc endpoint, and why it matters

`quant_gap_diag.py` asks a narrower question than M1, on the same held-out rows and through the same
deployment forward (no text reaches the model): after removing each column's batch mean, is slot m's
**codeword** the nearest of the four to axis m's caption embedding **of that image**? Chance is .250.
This endpoint was **not pre-registered**; it is reported as exploratory and it does not change any
screening verdict above.

| arm | code → own axis | axis → own code | diag − off-diag cosine |
|---|---:|---:|---:|
| base | .307 ± .005 | .374 ± .002 | .020 ± .003 |
| P1a top-p .3/.7 | .335 ± .013 | .424 ± .022 | .035 ± .006 |
| P1b top-p .15/.5 | .300 ± .028 | .419 ± .042 | .038 ± .012 |
| **P2 anchors** | **.370 ± .011** | **.493 ± .006** | .043 ± .005 |
| P3 gate removed | .290 ± .012 | .327 ± .008 | .015 ± .004 |
| P4 text_code_kl removed | .302 ± .005 | .380 ± .027 | .018 ± .003 |
| P4 role at codeword λ=.05 | .292 ± .016 | .326 ± .046 | .039 ± .023 |
| P5 role at codeword λ=1.0 | **.227 ± .012** | .229 ± .041 | −.085 ± .051 |
| **P5 role at slot token λ=1.0** | **.454 ± .025** | **.464 ± .036** | **.120 ± .016** |

Read with care in both directions. The pre-quantisation arm optimises a training-mode version of
exactly this quantity, so its .454 is the objective working, not independent evidence. The anchors
arm is not related to the metric in any such way, and it moves .307 → .370 and .374 → .493 with seed
SDs of .011 and .006, on 3/3 seeds — for a change that adds no loss term and no parameter.

**The two endpoints disagree, and that is the finding to carry forward.** Over all 27 runs the
pre-registered M1 and this endpoint correlate at Pearson +.30 (Spearman +.30); M1 spans .023 from
worst to best while this one spans .27. No test was pre-specified for either, and none is claimed.
Either M1 is too blunt to register a change of this size, or the alignment lives in the codeword
*vectors* and does not survive into the codon indices that M1 decodes. Separating those two is what
the next pre-registration has to do, before any further mechanism is tried — a mechanism search
judged by an endpoint of unknown sensitivity is not worth running.

## Standing after this program

Ten mechanisms have now been measured against M1 on the same baseline, three seeds each: slot mass
prediction, a free-marginal null, masked entity completion, a two-stage k-means codebook (arch-exp-2),
then routing sharpness ×2, axis-deviation anchors, axis-deviation readout, the global gate ×2, and
the in-image cross-axis term ×3. **None passes.** Each moves its own proximal target, and several
repair the codebook collapse for free — P1 and P2-anchors both take the worst codebook's perplexity
from .58 to .88 with a flag and no extra loss, which is what the two-stage k-means start bought at
the cost of a training-schedule change.

Two candidate readings remain, and they call for different work:

1. **The code cannot hold a role.** The retrieval code needs the image-level component that every
   slot shares; removing it (P2 readout, and the .70 mAP of the pre-quantisation arm) costs retrieval
   every time. If the axis and the hash want the same capacity, the paper's interpretability claim
   should stay where §4.7 already puts it, at codon-level decoding, and not be extended to per-slot
   roles.
2. **M1 cannot see a role that is there.** The post-hoc endpoint moved by .15 in an arm where M1
   moved by .0004. Before anything else, M1's sensitivity has to be established — for example by
   constructing a model whose slots are axis-specific by construction and checking that M1 registers
   it.

Reading 2 is cheap to settle and blocks reading 1, so it goes first.

## P6 — the top-p window, swept (2026-09-21, no code change)

P1 showed the window is not a role lever. This sweep asks the other question it raised: how narrow
should the window be for the things it *does* move. Four more windows, seeds 42/43/44, `archexp3_p6topp`,
rc=0, 270 s for twelve cells. Nothing else changed.

| window | patches per slot | patches owned by one slot | mAP@R | dead | unique | codebook min/median | M1 | empty slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| .60/.95 (recipe) | 3.07 | 4.0 % | .7453 ± .011 | .241 | .536 | .582 ± .277 | +.0065 ± .0043 | 0 % |
| .45/.85 | 2.20 | 15.0 % | .7472 ± .002 | .174 | .588 | .842 ± .036 | −.0003 ± .0033 | 0 % |
| .30/.70 | 1.78 | 27.4 % | .7511 ± .003 | .251 | .575 | .894 ± .091 | +.0017 ± .0014 | 0 % |
| **.20/.60** | 1.55 | 44.5 % | **.7547 ± .005** | .208 | .580 | **.925 ± .041** | +.0005 ± .0066 | 0 % |
| .15/.50 | 1.33 | 67.2 % | .7447 ± .012 | .177 | .593 | .884 ± .060 | −.0002 ± .0062 | 0 % |
| .10/.40 | 1.05 | 94.8 % | .7478 ± .006 | .197 | .583 | .865 ± .080 | +.0007 ± .0038 | 0 % |
| .05/.30 | 1.00 | 100 % | .7509 ± .014 | .219 | .555 | .588 ± .079 | −.0007 ± .0055 | 0 % |

**.20/.60 is the best cell of this entire program on retrieval**, +.0094 over the recipe window with
half its seed spread, and it is simultaneously the healthiest codebook measured anywhere here:
worst-codebook perplexity .925 of median against .582, and the .277 seed spread that made the
baseline unpredictable becomes .041.

The optimum is interior in both directions. Fully hard routing at .05/.30 sends every patch to one
slot and the collapse comes back, .925 → .588 — so the gain is not "sharper is better" but "neither
end is good". No window produces empty slots on Flickr25K, and M1 is flat across the whole sweep,
which is P1's verdict re-confirmed over five more settings.

**Status: adopt-candidate, not adopted.** It is one dataset, at the 4-epoch stage-1 cell, on
validation rows. Before it can touch the recipe it needs the same sweep on a second dataset, because
this project has a recorded Flickr/MS-COCO asymmetry for related routing knobs (2026-06-18) and a
recorded CIFAR empty-slot failure for sharp windows (2026-08). That confirmation was not launched
here: the current-protocol MS-COCO selection cell could not be reconstructed from an existing
`args.txt`, and improvising one risks a wrong per-dataset delta or a read of the official test split.

## Priority 1 — what M1 reads when a role IS present (2026-09-21, no training)

Every verdict above rests on M1, and nothing had established its scale. `m1_calibration.py` builds
codes whose role structure is dialled in and runs the same M1 computation on them. Slot m's code is
the k-means label of `normalise((1-a) * global_caption_emb + a * axis_m_caption_emb)`, K=128, on the
same opt/val split and with the same vocabulary and decoder the probe uses. At `a = 0` four slots
quantise the same shared vector; at `a = 1` slot m quantises axis m alone.

| role fraction a | M1 (3 clustering seeds) | own slot is argmax | code → own axis |
|---:|---:|---:|---:|
| 0.00 | −.0020 ± .0032 | 1/4 | .247 – .253 |
| 0.05 | +.0051 | 1–2/4 | .267 – .275 |
| 0.10 | **+.0123 ± .0011** | 4/4 | .306 – .316 |
| 0.25 | +.0280 ± .0029 | 3–4/4 | .416 – .422 |
| 0.50 | +.0734 | 4/4 | .679 |
| 0.75 | +.1139 | 4/4 | .853 |
| 1.00 | +.1306 | 4/4 | .911 |

**M1 is not blunt.** Its null is −.002 ± .003, it separates a 10 % role from noise by a factor of
four, and it is monotone over the whole range. The pre-registered threshold of .020 corresponds to
a ≈ .18. **Every verdict in arch-exp-2 and arch-exp-3 stands.**

**What the real models are worth on this scale.** Base M1 +.0065 sits at a ≈ .05; the best arm
measured here, +.0096, at a ≈ .08. The trained models hold roughly a twentieth of the role structure
that quantising the axis caption directly would give.

**And it explains the endpoint disagreement.** On synthetic codes the two endpoints agree: a ≈ .10
gives M1 .012 and code → axis .31, exactly where the real baseline sits on both. They part company
only on the pre-quantisation arm, which reads a ≈ .05 by M1 and a ≈ .28 by the vector endpoint. That
arm optimises embedding alignment, so its codewords point at the right caption vectors without
predicting the axis-distinctive words any better. The two endpoints are both valid and measure
different things; M1 measures the one the paper's claim is about.

## Priority 3 — quantise the deviation, carry the shared part around the codebook

P5 localised the loss of axis structure at the nearest-codeword step. `--quant_center_local`
subtracts the per-image mean across local slots **only from the quantiser input** and adds it back to
the quantiser output, so the codeword INDEX is chosen by the deviation while every downstream
consumer, including the retrieval hash, still receives the shared content. `--quant_center_rescale`
additionally restores each centred token to its original norm, exactly invertibly. Both default off;
entry gates 280/280 identical. `--lambda_text_code_kl 0.0` in every cell, because that term compares
an uncentred token with a codebook that now lives in the centred space; its own control is `p4drop`.

| | base | p4drop control | centre | centre + rescale | + window .20/.60 | window .20/.60 alone |
|---|---:|---:|---:|---:|---:|---:|
| **M1** | +.0065 ± .0043 | +.0078 ± .0009 | −.0026 ± .0069 | **+.0096 ± .0094** | +.0015 ± .0016 | −.0009 ± .0003 |
| M1 per seed | — | — | — | −.003 / .011 / **.020** | — | — |
| mAP@R | .7453 | .7409 | .7499 | **.7588** | **.6790** | .7491 |
| dead codes | .241 | .232 | .680 | .586 | .707 | .223 |
| unique codes | .536 | .517 | .771 | .732 | .316 | .564 |
| M2 label decoding | .796 | .803 | .560 | .607 | .537 | .815 |
| codebook min/median | .582 | .672 | .511 | **.125** | .495 | .865 |

**Screening verdict: all three fail.** Centring alone fails on M1 and on dead codes. Centring with
the norm restored posts the **highest mAP@R of the entire program, .7588**, and the highest M1 mean
of any trained arm, but its seed spread is as large as its mean and its codebook is wrecked: .586
dead, worst-codebook perplexity .125 of median. The combination with the narrow window, which was
the obvious repair for a dead codebook, is the worst cell measured anywhere: .6790 mAP and .707 dead.

The failure mode is informative and consistent across all three. Quantising the deviation gives the
codebook a much smaller target to cover; the EMA concentrates on a few codewords, and the codes that
survive are diverse but few. Restoring the norm recovers retrieval and part of the role signal but
not the balance, and the window, which balances the *routing*, cannot balance a codebook whose input
distribution has shrunk. A codebook-side remedy — fewer codewords, or a re-initialisation matched to
the centred scale — is what this points to, and it was not tried here.

## Priority 2 — does the .20/.60 window transfer to MS-COCO? (2026-09-21)

Run in a detached worktree at the same commit, `/data/yschoi/gdna_wt_mscoco`, so the main tree stayed
free; its four trainer/loss/model/config files were verified byte-identical to the main tree first.
MS-COCO stage-1 selection cell rebuilt from the Flickr cell with the per-dataset deltas read from
`scripts/train_mscoco_F2_sweep_clip.sh` and `scripts/phase3_selection_matrix.py`: `--dataset MSCOCO`,
mscoco caches and v5b captions, `cibhash 1.5`, `wasserstein .05`, `xmodal / text_hash / text_code_kl
.10`, `codon_joint .03`, prune ratios 1.0/1.0, `--sinkhorn_schedule_horizon 40 --stop_after_epoch 39`,
K=128, and the four MS-COCO provenance pins taken from `mscoco.stage1.input-seal.json`. Split is
train 10,000 / opt 9,000 / val 1,000. Seeds 42/43/44, `archexp3_mscoco_win`, rc=0, 2151 s.

| window | patches per slot | one-slot patches | mAP@R | dead | unique | codebook min/median | M1 | empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| .60/.95 recipe | 3.05 | 2.0 % | .6396 ± .003 | .004 | .358 | .853 | +.0036 ± .0039 | 0 % |
| .20/.60 candidate | 1.70 | 30.2 % | .6369 ± .012 | .005 | .363 | **.931** | +.0011 ± .0026 | 0 % |

per-seed mAP@R: recipe .6393 / .6359 / .6436, candidate .6389 / .6218 / .6500.

**The retrieval gain does not transfer.** On Flickr25K the narrow window was worth +.0094 mAP; on
MS-COCO it is −.0027, well inside the candidate's own .012 seed spread, and the candidate is the
worse of the two on two seeds of three. The mechanism still fires — patches per slot 3.05 → 1.70 —
and the codebook still improves, .853 → .931, but MS-COCO's codebook was not sick: .004 dead codes
against Flickr's .241, and .853 of median against Flickr's .582. There was nothing there to repair,
so nothing was gained.

**Verdict: the adopt-candidate is withdrawn as a universal recipe change.** What remains is a
Flickr25K-specific observation, and the protocol already carries per-dataset deltas, so it could be
proposed as one — but only through the refit/test protocol, not from a branch screening cell. This
is the Flickr/MS-COCO asymmetry the 2026-06-18 record predicted for a related routing knob, now
measured for this one.

## The post-hoc endpoint over every arm measured in this program

Chance .250. Not pre-registered; reported for completeness and for the disagreement it exposed.

| arm | code → own axis | axis → own code |
|---|---:|---:|
| base | .307 ± .005 | .374 ± .002 |
| P1 window .45/.85 | .320 ± .014 | .392 ± .022 |
| P1 window .30/.70 | .335 ± .013 | .424 ± .022 |
| P1 window .20/.60 | .298 ± .018 | .435 ± .043 |
| P1 window .15/.50 | .300 ± .028 | .419 ± .042 |
| **P2 anchors** | **.370 ± .011** | **.493 ± .006** |
| P3 gate removed | .290 ± .012 | .327 ± .008 |
| P4 text_code_kl removed | .302 ± .005 | .380 ± .027 |
| P4 role at codeword | .292 ± .016 | .326 ± .046 |
| P5 role at codeword λ=1 | .227 ± .012 | .229 ± .041 |
| **P5 role at slot token λ=1** | **.454 ± .025** | **.464 ± .036** |
| P7 quantiser centred | .261 ± .005 | .283 ± .016 |
| P7b centred + rescaled | .278 ± .006 | .346 ± .007 |
| P8 window .20/.60 + tckl 0 | .295 ± .014 | .408 ± .016 |
| P8 combination | .248 ± .006 | .261 ± .009 |

Every quantiser-side arm sits **below** the baseline on this endpoint, P7b included. So P7b's higher
M1 mean, which came with a seed spread as large as itself, is not corroborated: on the endpoint that
separates arms most sharply it is .278 against the baseline's .307. Read together, the two endpoints
agree that the quantiser intervention did not create a role.
