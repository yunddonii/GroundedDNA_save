# GroundedDNA — Project Log

Living record of *important* design decisions, experiment results, and
infrastructure issues for the GroundedDNA deep-hashing project. Add a
new section whenever a non-trivial change lands (new architectural variant,
new ablation, dataset/cache change, baseline result, etc.). Keep entries
short and dated; if a decision was later reverted, mark it `[reverted YYYY-MM-DD]`
rather than deleting — the reasoning is the value.

Format conventions:
- **Date** = YYYY-MM-DD (absolute; never "yesterday").
- **Status** at top of each section: 🟢 active / 🟡 superseded / 🔴 reverted.
- Tables are preferred over prose when reporting numbers.
- "v6" etc. without dataset prefix = the Flickr25k run unless noted.

**Layout policy** (agreed 2026-05-15, policy A):
1. Document header (this preamble).
2. **`## Current state` snapshot pinned at top** — always edited in
   place to reflect the current best result + active ablation set.
3. All other dated sections in **reverse chronological order** (newest
   first). Within the same date, sort by **largest version number
   descending** (e.g. v25 above v24 above v23); sections without a
   version token (infra notes, baseline summaries) fall to the bottom
   of that day's block in their original insertion order.
4. `## Infrastructure & repo hygiene` is static and pinned at the
   bottom (just above the HTML comment).
5. Re-run `scripts/reorder_project_log.py` after appending or editing
   entries to re-enforce the policy (idempotent).

---

## Current state (as of 2026-05-18)

- **Best supervised Flickr25k**: **v18** (HashNet-style logistic on
  continuous DNA code) -- mAP **0.7883**. Above every binary baseline
  incl. HashNet's own 0.7800.
- **Best supervised + diversity-balanced Flickr25k**: **v24b** -- mAP
  **0.7742**, unique 0.324.
- **Best unsupervised Flickr25k (ours)**: **v30a** (v29 NtXent +
  half-size MLP adapter, hidden=768) -- mAP **0.6646**, unique
  0.4299, dead 0.003. Beats CIBHash 0.6543 by +0.010.
- **Unsupervised leaderboard (Flickr25k setting1, 36-bit, frozen SigLIP2, 60 epoch)**:

  | Run | Adapter / variant | mAP | unique (test) | per-cb-unique | dead |
  |-----|-------------------|----:|--------------:|--------------:|-----:|
  | **v30a** ★ | MLP h=768 (half v29) | **0.6646** | 0.4299 | — | 0.003 |
  | **v30c** | Linear d=384 | 0.6628 | 0.5171 | — | 0.000 |
  | **v33b** | per-codebook + routing top-k=2 | 0.6594 | 0.6497 | 0.0118 | 0.000 |
  | **v31b** | per-codebook NtXent | 0.6590 | 0.6321 | 0.0099 | 0.000 |
  | v33a | per-codebook + sinkhorn eps anneal | 0.6584 | 0.7515 | 0.0118 | 0.076 ⚠ |
  | **v29**  | global NtXent baseline | 0.6580 | 0.5257 | — | 0.000 |
  | CIBHash (external) | flat Linear(768,36) + NtXent | 0.6543 | 0.997 | — | — |
  | CIMON (external) | spectral-PL + NtXent | 0.6456 | 0.881 | — | — |
  | MLS3RDUH (external) | kNN graph + LogCosh | 0.5947 | 0.184 | — | — |
  | v32 | + train-only text inject α=0.2 | 0.6422 | 0.3422 | 0.0040 | 0.000 |
  | v28b | + FeatureDecoder (recon) | 0.5648 | 0.4057 | — | 0.003 |
  | v27b | SigLIP2 cos top-k 20% | 0.5639 | 0.302 | — | 0.000 |
  | v28a | + PixelDecoder (recon) | 0.5514 | 0.4425 | — | 0.146 |
  | v30b | Linear d=768 (collapse) 🔴 | 0.5399 | 0.0005 | — | 0.628 |

- **Active loss set in v30a / v29 / v31b**: 6 terms -- `loss_ntxent`
  (★ primary), `loss_vq`, `loss_quant`, `loss_anchor`, `loss_dna`,
  `loss_bu`. `loss_hash` / `loss_hash_hard` / `loss_recon` /
  `loss_wasserstein` all disabled (λ=0).
- **Adapter capacity sweet spot**: MLP hidden=768 (v30a, 1.18M params)
  or Linear at d=384 (v30c, 0.30M). v29's hidden=1536 (2.36M) overfits
  Flickr25k 5K train images; Linear at full d=768 (v30b) is too
  underparameterised and collapses.
- **Negative results catalogued (2026-05-15)**:
  - v27a (continuous siglip_cos target) → collapse.
  - v28a/b (recon decoder) → flat / mild regression.
  - v30b (Linear d=768) → collapse.
  - v32 (train-only text inject α=0.2 detach) → mAP −0.016 vs v29
    (train-test distribution shift dominated; smaller α or anneal
    might still help).
- **Standard supervised baseline**: **v6** = `sinkhorn router +
  c_global_source=siglip2_global + K=32, 6 codebooks × 3 codons × 2 bits
  = 36-bit DNA hash`. SigLIP2 frozen; only adapter + codebooks +
  codon head are trained. mAP 0.7556 with MSE-Jaccard loss_hash.
- **R-series archive**: R1/R2/R3/R4 + v15 codebook-orthogonality result
  dirs are in `backup/results_R_series/` with per-experiment summary in
  the README there. The corresponding loss / config / output-dict code
  paths were removed from the main tree on 2026-05-13.
- **MSCOCO**: v6 = mAP **0.5243**; v6+V2 prompt = **0.5385**. v18 / v29 /
  v30a form not yet tested on MSCOCO.
- **CIFAR10**: v6 = mAP **0.5335**, top-1 vs all four binary baselines.
- **NUS-WIDE**: cache build paused.
- **Result directory naming convention**: `<date>+<dataset>_<setting>_<user_tag>+bs+<bs>+e+<epoch>+proj_lr+<lr>/`.
  Enforced automatically by `train_siglip2._resolve_save_path` since
  2026-05-15.
- **Layout policy**: `## Current state` pinned at top, dated sections
  reverse-chronological (newest first), `## Infrastructure` pinned at
  bottom. Re-enforced via `python scripts/reorder_project_log.py`
  (idempotent).

---

---

---

---

---

---

---

## 2026-05-18 — v33a / v33b: harden Sinkhorn routing (eps anneal · top-k mask)

🟢 v33b kept as a useful diversity-improving variant on top of v31b.
🟡 v33a documented but not adopted (introduces dead codes).

Question: routing in v29-family is *soft* (Sinkhorn OT plan gives each
patch a smooth weight across all 5 local parts). Does *hardening* the
routing increase per-codebook discrimination (per-cb-unique) and/or
total compositional diversity without losing retrieval mAP?

Common setting: **v30a + v31b merged** — MLP hidden=768 adapter
(v30a, 1.18M params/branch) + per-codebook NtXent loss (v31b).
Two variants:

- **v33a**: epsilon annealing. `eps(t) = 0.01 + (0.1 - 0.01) * (1 + cos(π·t/T))/2`,
  smoothly shrinks from 0.1 (smooth routing) at epoch 0 to 0.01
  (near-hard) at epoch 59. Sinkhorn balance preserved (marginals still
  enforced) but the OT plan becomes peakier as training progresses.
- **v33b**: top-k mask per patch. After Sinkhorn, keep only the top-k
  largest part-weights per patch (k=2) and renormalize each row to its
  original marginal. Each patch now contributes to exactly 2 parts
  instead of 5 (or fewer with concentrated mass).

Final results (Flickr25k setting1, 60 epoch, cached aug):

| Run        | mAP    | unique (db) | per-cb-unique | dead   | Notes |
|------------|-------:|------------:|--------------:|-------:|-------|
| v31b (baseline) | 0.6590 | 0.3651 | 0.0009 | 0.0000 | reference |
| **v33a** (eps anneal) | 0.6584 | 0.3725 | 0.0010 | **0.0755** | ⚠ dead codes appear |
| **v33b** (top-k=2)    | **0.6594** | **0.3947** | 0.0010 | 0.0000 | strict Pareto improvement |

Mid-eval trajectory (test split, ep 9-59):

```
v33a:  mAP 0.667 → 0.658 → 0.659 → 0.657 → 0.659 → 0.657
       uniq 0.668 → 0.672 → 0.693 → 0.686 → 0.719 → 0.752
v33b:  mAP 0.659 → 0.663 → 0.658 → 0.657 → 0.654 → 0.657
       uniq 0.720 → 0.668 → 0.675 → 0.666 → 0.659 → 0.650
```

v33a peaks early on mAP then drifts; unique grows monotonically.
v33b is steady on both axes from epoch 9 onwards.

Findings:
1. **Routing-hardening helps diversity, not mAP.** Both variants give
   roughly v31b mAP (0.6584–0.6594, Δ within ±0.001) but lift database
   unique by +0.03 to +0.07.
2. **v33b is a strict Pareto improvement over v31b** (slightly higher
   mAP AND higher unique). Adopted as the new compositional-diversity
   variant.
3. **v33a develops dead codebook entries** (7.5% per codebook by ep 59).
   Annealing eps to 0.01 makes the Sinkhorn plan so peaky that some
   codewords are never selected for an entire batch → EMA revival
   can't reach them. Useful warning for any future eps schedules.
4. **Neither variant beats v30a on mAP** (0.6646). Routing-hardening is
   orthogonal to adapter-capacity reduction; combining v30a's smaller
   adapter with v33b's top-k=2 router could give the next Pareto point.

Code:
- `models.semantic_router.SemanticSinkhornRouter.forward` gains
  `epsilon_override` and `topk_per_patch` kwargs.
- `model_siglip2.SigLIP2SemanticOTModel`: `set_current_epoch()` setter
  + `_current_sinkhorn_epsilon()` cosine schedule helper. Router call
  injects the per-epoch eps and the top-k flag.
- `train_siglip2.py`: pushes the current epoch to the model at the
  start of each epoch (immediately after the gumbel-tau anneal).
- `config.py`: `--sinkhorn_epsilon_init`, `--sinkhorn_epsilon_final`,
  `--routing_topk` flags.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

## 2026-05-15 — v32: train-only text injection into quantizer (regression)

🔴 reverted — `z' = z_v + 0.2 * sg(t)` at train time, fall back to z_v
at eval. The codeword embeddings were supposed to absorb text-semantic
structure during training while keeping the inference path text-free,
but the train-test distribution shift dominated and mAP dropped by
~0.02 vs v29.

| Epoch | mAP | unique (test) | per-cb-unique | dead |
|------:|----:|--------------:|--------------:|-----:|
|   9   | 0.6378 | 0.1850 | 0.0032 | 0.063 |
|  19   | 0.6399 | 0.3120 | 0.0039 | 0.013 |
|  29   | 0.6341 | 0.3130 | 0.0040 | 0.000 |
|  39   | 0.6351 | 0.3327 | 0.0040 | 0.003 |
|  49   | 0.6387 | 0.3453 | 0.0039 | 0.000 |
|  59   | 0.6387 | 0.3422 | 0.0040 | 0.000 |
| **final eval** | **0.6422** | 0.1063 | 0.0003 | 0.000 |

Compared to v29 (0.6580 final) the text-inject path cost −0.016 mAP and
also dropped diversity. The hypothesis (text-bias the codeword centroids
during training, then look up unbiased visual at test) failed because
the codeword centroids that the codebook learned were systematically
shifted by `0.2·t` away from the test-time visual feature distribution
— at lookup time many samples picked a "wrong" closest codeword.

Why this is a useful negative result:
1. **Train-test distribution shift dominates** even at α=0.2 with the
   text gradient detached. The codebook commit loss `MSE(q, sg(z))` is
   tight enough that the shift is preserved through training rather than
   being smoothed out.
2. **SigLIP2 text-pooled vectors are near-uniform across slots**
   (cos sim ~0.88 cross-slot, recorded in the v22 thread). Adding a
   near-uniform shift to all six visual slots collapses inter-codebook
   diversity rather than enriching it.
3. **per-codebook unique = 0.0040 — same as v31b** while total unique
   collapsed to 0.34 (vs v31b's 0.65). So text injection both fails to
   help per-codebook independence AND removes total compositional
   diversity.

Variants worth considering before declaring the entire direction dead:
- (b') smaller α (e.g. 0.05) — minimum train-test shift.
- (c) anneal α → 0 over training so the codebook's late-stage learning
  matches inference distribution.
- separate per-codebook text projections (v22a-style) so the injected
  signal differs per slot and breaks the cross-slot uniformity.

Code: `--text_inject_train_only {none, add}`,
`--text_inject_alpha`, `--text_inject_detach` (default True) flags
in `config.py`; logic in `model_siglip2.SigLIP2SemanticOTModel.forward`
inserted between routing and quantization.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

## 2026-05-15 — v31b: per-codebook NtXent (compositional-independence ablation)

🟢 active — replaces v29's single global NtXent on the whole 18-base
DNA code with 6 separate NtXents on each codebook's 3-codon block,
averaged. Question asked: "does forcing each codebook to discriminate
independently raise per-codebook unique_rate without hurting mAP?"

| Epoch | mAP | total unique (test) | per-cb-unique | dead |
|------:|----:|--------------------:|--------------:|-----:|
|   9   | 0.6583 | 0.6033 | 0.0085 | 0.008 |
|  19   | 0.6569 | 0.6255 | 0.0098 | 0.003 |
|  29   | 0.6598 | 0.6512 | 0.0102 | 0.000 |
|  39   | 0.6588 | 0.6230 | 0.0100 | 0.000 |
|  49   | 0.6589 | 0.6285 | 0.0099 | 0.000 |
|  59   | 0.6562 | 0.6321 | 0.0099 | 0.000 |
| **final eval** | **0.6590** | 0.3651 (db) | 0.0009 (db) | 0.000 |

Comparison vs v29 (global NtXent):

|        | v29 final | v31b final | Δ |
|--------|----------:|-----------:|---|
| mAP    | 0.6580 | 0.6590 | +0.001 (tied) |
| unique (test ep59) | 0.5257 | 0.6321 | +0.106 |
| per-cb-unique | (no metric) | 0.0099 | — |
| dead   | 0.000  | 0.000  | — |

Hypothesis vs reality:
- **Hypothesis**: per-codebook NtXent forces each codebook to identify
  samples on its own → per-cb-unique rises to 0.4-0.55.
- **Reality**: per-cb-unique only ~0.01 (each codebook still uses
  ~20 distinct 3-base codes out of 2000 queries) — single codebook
  alone is FAR from being a 1-of-2000 discriminator.
- **What actually happened**: total unique (18-base) jumped from 0.53
  to 0.65 even though each codebook stays heavy-collapsed. The 6
  codebooks landed on DIFFERENT collapse patterns, so the combined
  18-base code is more diverse than under v29's global loss.

Reframed interpretation: per-codebook NtXent does not produce
*independent discriminators* (each codebook can't separate 2000
samples into 64 codes by itself). It produces *orthogonal collapse
patterns* — 6 weak hashers whose Cartesian product is much more
diverse than 6 redundant ones. mAP is preserved because the joint
code remains discriminative, and total compositional diversity
improves "for free".

Code: `--ntxent_mode per_codebook` toggles the variant. Implemented
in `loss_siglip2._loss_ntxent_dna_per_codebook`. The companion
`per_codebook_unique_ratio` metric (added in `evaluation_siglip2`
for this run) lets us measure the per-codebook collapse degree
directly.

v31a (additive: keep global + add per-codebook regularizer at λ=0.2)
not yet launched. Likely to land between v29 and v31b on both axes.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

## 2026-05-15 — v30 final results (adapter-capacity ablation under v29 loss)

🟢 v30a / v30c kept as available variants · 🔴 v30b reverted (collapses).

Question: does the 2-layer MLP adapter (~2.36M params per branch) actually
contribute under v29's paired-aug NtXent loss, or is a flat
LayerNorm+Linear (~0.59M, CIBHash-style) enough?

| Run | Adapter | params / branch | final mAP | Δ vs v29 | unique (test ep59) | dead | Status |
|-----|---------|---:|---:|---:|---:|---:|---|
| v29  (baseline) | MLP hidden=1536, d=768 | 2.36M | 0.6580 |  —    | 0.5257 | 0.000 | reference |
| **v30a** | MLP hidden=768, d=768 | 1.18M | **0.6646** | **+0.007** | 0.4299 | 0.003 | 🟢 best mAP |
| **v30b** | Linear (LayerNorm + single Linear), d=768 | 0.59M | 0.5399 | −0.118 | 0.0005 | 0.628 | 🔴 collapsed |
| **v30c** | Linear (LayerNorm + single Linear), d=384 | 0.30M | 0.6628 | +0.005 | 0.5171 | 0.000 | 🟢 lightweight option |

Findings:
1. **Smaller MLP wins** — v30a (hidden 768, half of v29's 1536) gives
   the best mAP (0.6646) among ALL our runs to date. Confirms the
   hypothesis that v29's adapter is overparameterised for Flickr25k's
   5K train images and the paired-aug NtXent signal.
2. **Linear at full d=768 collapses immediately** — v30b stuck at
   mAP=0.5393 / unique=0.0005 from epoch 9 onwards. A bare
   LayerNorm+Linear at the SigLIP2 projection dim cannot keep the
   routing geometry from degenerating under NtXent + frozen backbone.
   Same collapse signature as v27a (siglip_cos continuous target).
3. **Linear at d=384 works fine** — v30c (LayerNorm + single Linear,
   d=384) reaches mAP=0.6628, only −0.002 below v30a despite having
   ~4x fewer adapter params. The dim reduction from 768→384 gives the
   single Linear enough relative capacity to find a stable mapping.
4. **Adapter capacity sweet spot**: MLP hidden=768 (v30a, 1.18M params)
   or Linear at d=384 (v30c, 0.30M). Anything smaller (Linear at d=768)
   is too underparameterised.

Decision: **adopt v30a as the new SOTA-equivalent baseline** (best mAP,
similar wall-clock to v29 since cached-features path dominates). v30c
kept as the lightweight reference for downstream comparison. v30b
documented as a clear failure mode.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

## 2026-05-15 — Cached paired-aug visual_tokens (v29+ training speedup)

🟢 infrastructure — paired-aug NtXent training previously ran the
(frozen) SigLIP2 backbone live on `img_tr1` / `img_tr2` every step,
limiting throughput to ~1 it/s. We now precompute K augmented
visual_tokens + visual_global views per image and cache them as
`visual_{tokens,global}_aug{i}.f16.npy`, restoring ~5-8 it/s.

Code changes:
- `extract_siglip2_features.py --save_aug_views K` now allocates BOTH
  `visual_global_aug{i}.f16.npy` (existing) and
  `visual_tokens_aug{i}.f16.npy` (new). Each step runs `vision_model`
  once (so the token-axis output is captured) plus `get_image_features`
  for the projection head.
- `dataloaders._SigLIP2FeatureCache` scans the cache dir for
  `visual_{tokens,global}_aug{i}` pairs and exposes them as
  `cached_visual_{tokens,global}_aug{i}` per row.
- `train_siglip2.py`: when `--use_paired_aug_ntxent` is on AND the
  cache exposes the aug pair, view-1 forward consumes
  `cached_*_aug0` and view-2 forward consumes `cached_*_aug1` --
  both via the cached-features path inside the model. PIL decode is
  also disabled in that case. When the cache is absent, falls back
  to the original live-backbone path. A single startup line prints
  which path is in use.

Disk impact: each aug view is `[N=27000, 196, 768]` fp16 ≈ 8.1 GB on
Flickr25k; two views = 16.2 GB additional cache. Acceptable.

The Flickr25k cache build (`/home/yschoi/GroundedDNA/cache/flickr25k_siglip2/`)
was re-run on 2026-05-15 with `--save_aug_views 2 --aug_only` so the
existing deterministic visual_tokens / visual_global cache is preserved.

---

---

---

---

---

---

---

## 2026-05-15 — v29: paired-aug NtXent on DNA codes (clear new SOTA among our runs)

🟢 active — replaces v27b's batch top-k cosine pseudo-positive with a
CIBHash-style instance-discrimination contrast on the DNA code itself
(STE one-hot per position). Same image's two augmented views are positive;
all other batch samples are negative.

| Epoch | mAP | unique (test) | dead |
|------:|----:|--------------:|----:|
|   9   | 0.6627 | 0.4138 | 0.047 |
|  19   | 0.6596 | 0.5272 | 0.008 |
|  29   | 0.6603 | 0.5227 | 0.000 |
|  39   | 0.6517 | 0.5262 | 0.003 |
|  49   | 0.6510 | 0.5383 | 0.003 |

vs v27b (cos top-k) and CIBHash (best unsupervised baseline):

| Method | mAP | unique (test) |
|--------|----:|--------------:|
| v27b (ours, cos top-k 20%)        | 0.5639 | 0.302 |
| CIBHash (NtXent on Linear(768, 36)) | 0.6543 | 0.997 |
| **v29 (ours, NtXent on DNA code)** | **0.6627 (peak ep 9)** | 0.41-0.54 |

Key observations:
1. **Loss matters more than capacity.** Same backbone + same compositional
   architecture as v27b, only the retrieval signal changed. mAP jumped
   from 0.5639 to 0.6627 in 9 epochs and exceeded CIBHash by +0.008.
2. **mAP peaked early, drifts down with longer training.** Epoch 9 was
   the maximum; by epoch 49 mAP had drifted to 0.65. Suggests either
   LR-schedule-driven late overfitting or insufficient regularisation
   at the codebook-utilisation level.
3. **unique-code count rises monotonically (0.41 → 0.54)** while mAP is
   slowly drifting down. Codes get more diverse but not more
   retrieval-discriminative -- evidence the architecture is *not*
   collapsing, just slowly drifting in code space.
4. **CIBHash unique 0.997 vs ours 0.54** -- CIBHash's contrast on a flat
   36-bit binary code naturally hits the per-image unique limit because
   the loss directly supervises 36 independent bits with K=2 each.
   Our 6×3-codon code with K=64 codewords per slot has a discrete
   bottleneck that caps unique below the per-image guarantee at small B.
5. **Why CIBHash leads on unique despite lower-than-ours mAP:** the
   CIBHash design is simply better tuned to "frozen feature + tiny
   projection head" -- discussed in detail in the
   `CIBHash가 sota를 기록하고 있는 원인 분석` thread on 2026-05-15.

Code path: `--use_paired_aug_ntxent --lambda_ntxent 1.0
--ntxent_temperature 0.3 --hash_target_mode siglip_cos_topk
--lambda_hash 0.0 --lambda_hash_hard 0.0`. `lambda_hash_*` zeroed so
NtXent is the only retrieval signal; hash_target_mode kept on
siglip_cos_topk just to satisfy the criterion's "S must come from
somewhere" assertion when `multi_hot_labels=None`.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

---

---

---

---

---

---

## 2026-05-15 — Architecture diagram for v29 (forward + losses)

🟢 reference — `docs/architecture_v29.png` rendered by
`scripts/draw_architecture.py`. Solid arrows = data flow, dashed red
arrows = loss targets. Detail boxes for the Sinkhorn-OT router
(6 sub-steps), per-slot VQ codebook (6 × K=64 grid drawn), 6 codon
heads, and all 7 active loss terms (`L_ntxent`, `L_vq`, `L_quant`,
`L_anchor`, `L_dna`, `L_bu`, `L_wass`). Disabled losses (`L_hash`,
`L_hash_hard`, `L_recon`) listed in a side panel.

---

---

---

---

---

---

---

## 2026-05-15 — v28a / v28b: reconstruction-decoder ablation on top of v27b

🟡 superseded by v29 (NtXent gives much stronger retrieval signal).

Tested whether attaching a decoder that maps the 6 selected codewords
back to either the raw image (v28a) or the cached SigLIP2
visual_global (v28b) helps fix v27b's noisy batch-top-k positive signal.

Final results (Flickr25k setting1, 60 epoch):

| Run     | Decoder           | mAP    | unique | dead   |
|---------|-------------------|-------:|-------:|-------:|
| v27b    | none              | 0.5639 | 0.302  | 0.000  |
| **v28a** | PixelDecoder (~18M params) | 0.5514 | 0.240 | 0.065 |
| **v28b** | FeatureDecoder (~16M params) | **0.5648** | 0.210 | 0.000 |

Findings:
1. **Reconstruction did not move mAP.** Both v28a and v28b stay within
   ±0.01 of v27b. The pixel target slightly *hurts* (−0.013) because
   it forces the codebook to encode low-level texture that's irrelevant
   for retrieval; the SigLIP2-feature target is neutral (slightly +).
2. **Codebook utilisation got better, retrieval did not.** v28b's
   codebook ended at dead=0.0 with healthy per-slot histograms, but
   the discriminative quality of those codes did not improve.
3. **Decoder is dropped going forward (v29).** The reconstruction
   signal optimises a different objective (inversion) than retrieval,
   and on the small Flickr25k train set (5K images) the extra
   parameter budget is not paying for itself.

Both modules (`PixelDecoder`, `FeatureDecoder`) and the loss term
(`lambda_recon`) are kept in the codebase behind `--use_decoder` for
future use on larger datasets where the inversion signal might pay off.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

---

---

---

---

---

---

## 2026-05-15 — Unsupervised baseline comparison: v27b vs CIBHash / CIMON / MLS3RDUH

🔴 honest finding — under a fully unsupervised supervision regime,
GroundedDNA (v27b) underperforms two of the three unsupervised baselines
we managed to run on the same Flickr25k setting1, same backbone (frozen
SigLIP2-base), same 36-bit storage, and same train / query / database
splits.

Result table (Flickr25k setting1, 60 epoch, bs=64, frozen SigLIP2):

| Method        | Supervision           | mAP    | P@1    | P@100  | P@1000 |
|---------------|-----------------------|-------:|-------:|-------:|-------:|
| CIBHash       | NtXent (paired aug)   | **0.6543** | 0.8350 | 0.8111 | 0.7648 |
| CIMON         | NtXent + spectral PL  | 0.6456 | 0.8355 | 0.7943 | 0.7454 |
| MLS3RDUH      | kNN graph + LogCosh   | 0.5947 | 0.6780 | 0.6796 | 0.6645 |
| **v27b (ours)** | SigLIP2 cos top-k 20% | 0.5639 | 0.6665 | 0.6531 | 0.6174 |
| SPQ           | (not runnable; `baseline/SPQ.py` is incomplete — `_get_fixed_config_dict` syntax error at line 176, `_train_model` never implemented). | — | — | — | — |

Gap interpretation:

1. **CIBHash / CIMON beat v27b by ~0.08–0.09 mAP.** Both use paired
   augmentations of the *same* image as positive contrastive pairs. That
   is a very sharp positive signal (near-identity), much stronger than
   v27b's "top-20% batch-cosine pairs are positive" heuristic. The
   advantage is essentially the supervision-strength gap between
   augmentation-based contrastive (per-image positive guarantee) and
   batch-similarity pseudo-labels (noisy, variable per batch).

2. **MLS3RDUH beats v27b by ~0.03 mAP.** It computes a kNN affinity
   graph over training-set features (not just within-batch), so its
   positive set is *globally* informed instead of per-batch. v27b's
   top-k pseudo-positives drift batch-to-batch.

3. **The compositional claim is not validated by mAP alone.** Our
   38-LoC pseudo-positive scheme, applied to the rest of the
   GroundedDNA stack (codebooks, OT routing, codon heads), produces a
   weaker retrieval signal than the simpler CIBHash architecture
   (single Linear head + NtXent on aug pairs).

Implications for the paper narrative:
- "GroundedDNA is unsupervised" cannot honestly mean "beats supervised
  baselines without labels". It means "operates without labels and
  produces a structured / interpretable code".
- The retrieval-mAP comparison in an unsupervised setting must include
  CIBHash / CIMON as lower bounds we currently miss.
- A natural next experiment: feed CIBHash's NtXent paired-aug positive
  signal (instead of batch top-k cos) into the GroundedDNA loss. That
  isolates "supervision quality" from "architectural overhead" and may
  recover some of the gap.

Infrastructure work landed for this comparison (committed `5075b00`):
- `extract_siglip2_features.py --save_aug_views K --aug_only` produces
  `visual_global_aug{0..K-1}.f16.npy` per image so paired-aug baselines
  can use the frozen-backbone cache.
- `baseline/base_model.py` `CachedFeatureDataset` now honors
  `paired_aug` and `return_index`. `_build_method` registers
  `cibhash`, `cimon`, `mls3rduh`. `init_experiment` propagates the
  per-baseline `_get_fixed_config_dict()` flags into `load_dataset`
  (was previously hardcoded False — silent bug).
- `baseline/MLS3RDUH.py` adds `SigLIP2` to the `dim_feature` mapping.
- SPQ deliberately not registered: source file is incomplete.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

---

---

---

---

---

---

## 2026-05-15 — Result-directory naming convention (auto-prefix dataset+setting)

🟢 infrastructure — `train_siglip2._resolve_save_path` now auto-prepends
`<dataset.lower()>_<setting>_` to the user `--tag` when the user tag does
not already start with that prefix.

Schema for all *future* runs:
`result/<date>+<dataset>_<setting>_<user_tag>+bs+<bs>+e+<epoch>+proj_lr+<lr>/`

Example: `--tag v31a_per_codebook_ntxent` on Flickr25k setting1
becomes `result/260515+flickr25k_setting1_v31a_per_codebook_ntxent+bs+64+e+60+proj_lr+0.001/`.

Background: v28a/b/c, v29, v30a/b/c launches dropped the
`flickr25k_setting1_` prefix (we passed bare `--tag v28a_unsup_recon_pixel`
etc.), breaking the existing `260513+flickr25k_setting1_v18_hashnetloss`
convention. Folders for completed v28a/v28b were renamed in-place to add
the missing prefix; v29 / v30a / v30b / v30c were left untouched while
training was active (open file handles) and will be renamed after the
runs finish. The `_resolve_save_path` change covers everything launched
from now on without requiring discipline at the CLI.

---

---

---

---

---

---

---

## 2026-05-14 — v27b: unsupervised hash target (SigLIP2 cos top-k pseudo-positives) [completed]

🟢 follow-up to v27a — replaces the continuous cosine target with a
binary {0, 1} top-k pseudo-label scheme to fix v27a's collapse. The top
`siglip_cos_pos_rate=0.2` fraction of off-diagonal pairs (ranked by
SigLIP2 visual_global cosine within each batch) get S=1, the rest S=0.

Why v27a failed (diagnosed below) and v27b should fix it:
- SigLIP2 image embeddings on Flickr25k have cosine sim distributed in
  roughly `[0.5, 0.9]` (natural-image features sit on a narrow cone). The
  rescale `(cos+1)/2` yields S ≈ 0.85 for nearly every pair.
- HashNet logistic with `S_target ≈ 0.85` everywhere computes gradient
  `sigmoid(score) − 0.85`, i.e. **monotonically positive** for almost all
  pairs: every pair pushes its `score → +∞`. All sample codes collapse
  to a single positive cluster.

v27b mechanism:
- Binarize via `tau = quantile(cos_offdiag, 0.8)` within each batch, so
  exactly 20% of off-diagonal pairs are positive and 80% are negative.
  Mirrors CIBHash (NtXent positive/negative split), CIMON (binarized
  spectral pseudo-labels), MLS3RDUH (binary kNN graph).
- The HashNet logistic now sees a mix of S=1 and S=0 targets, restoring
  the contrastive pressure that spreads codes across the codebook.

Implementation:
- New helper `build_siglip_cos_topk_similarity(visual_global, pos_rate)`
  in `loss_siglip2.py`.
- New `--hash_target_mode siglip_cos_topk` and `--siglip_cos_pos_rate 0.2`
  CLI flags in `config.py`. The continuous `siglip_cos` mode is retained
  for the record but expected to remain useless on natural-image data.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

---

---

---

---

---

---

## 2026-05-14 — v27a: unsupervised hash target (frozen SigLIP2 cosine) [collapsed → killed]

🔴 reverted — continuous SigLIP2 cosine target collapses on natural-image
data (Flickr25k cos distribution is narrow around 0.7–0.9 → S target is
near-uniform 0.85 → HashNet logistic pushes every pair into the same
positive code cluster).

| run                              | epoch |   mAP  | unique | dead  |
|----------------------------------|------:|-------:|-------:|------:|
| v24b (supervised, SOTA balanced) |  60   | 0.7742 | 0.3242 | 0.000 |
| v27a (siglip_cos)                |   9   | 0.5393 | 0.0005 | 0.875 |
| v27a (siglip_cos)                |  14   | 0.5393 | 0.0005 | 0.844 |
| v27a (siglip_cos)                |  19   | 0.5393 | 0.0005 | 0.794 |

mAP perfectly flat across three checkpoints — fully degenerate state.
Killed at epoch 19; result directory deleted to keep `result/` clean. The
`siglip_cos` mode remains in the codebase (gated by `--hash_target_mode`)
as a documented negative finding and the starting point for the v27b fix.

Original motivation (kept for context):
- The v18-family loss treats Jaccard of `multi_hot_labels` as the
  pairwise target — that is a label-supervised signal, structurally
  identical to HashNet / DPSH / CSQ / OrthoHash. Calling our method
  "compositional unsupervised" while the hash supervision itself is fully
  supervised is misleading.
- CIBHash / CIMON / SPQ / MLS3RDUH sidestep labels by deriving the
  pairwise target from feature-space proximity (NtXent positives, spectral
  pseudo-labels, kNN graphs). v27a was the simplest version of that — and
  it failed because the cos distribution on natural images is too narrow.

---

---

---

---

---

---

---

## 2026-05-14 — v26a / v26b: V3 prompt × current SOTA setups → both regress

🟡 superseded — V3 prompt helps v6 (MSE form) but not v18+ (hashnet form).

Observation that triggered the experiment: the standalone v6+V3-prompt
run from 2026-05-13 (`260513+flickr25k_setting1_v6_promptv3`) showed an
unusually healthy hash-utilisation profile — unique 0.4553 (vs V1's
0.3958) with mAP 0.7543 (vs V1's 0.7556, basically tied). User noted
that signal and asked to retest V3 prompt under the current best setups.

Two parallel runs (Flickr25k, 60 epochs):

- **v26a** = v20-K64 (K=64, hashnet binary, with global gate) + V3 prompt.
- **v26b** = v23b (K=64, hashnet binary, no global gate) + V3 prompt.

Result table comparing each V3 variant to its V1 sibling:

| run                            |   mAP  | Δ mAP   | unique | Δ unique | dup    | gap |
|--------------------------------|-------:|--------:|-------:|---------:|-------:|----:|
| v6  V1 prompt (MSE)            | 0.7556 |   —     | 0.3958 |    —     | 0.6042 | 2.86 |
| v6  V3 prompt (MSE)            | 0.7543 | −0.001  | 0.4553 |  +0.06   | 0.5447 | 2.77 |
| v20-K64 V1 (hashnet, gate)     | 0.7892 |   —     | 0.0982 |    —     | 0.9018 | 5.20 |
| **v26a V3 (hashnet, gate)**    | 0.7765 | **−0.013** | 0.1402 |  +0.042  | 0.8598 | 4.41 |
| v23b V1 (hashnet, no gate)     | 0.7877 |   —     | 0.1107 |    —     | 0.8893 | 5.05 |
| **v26b V3 (hashnet, no gate)** | 0.7664 | **−0.021** | 0.1102 |   ≈ 0    | 0.8898 | 4.37 |

Findings:

1. **V3-prompt effect is loss-form dependent.** Under MSE-Jaccard (v6) it
   gives a strict diversity win (unique +0.06) at essentially zero mAP
   cost. Under the hashnet binary form (v20-K64 / v23b) the cost flips:
   mAP regresses 0.013–0.021 and the diversity gain is at best modest.

2. **The "no gate" + V3 combo is the worst.** v26b loses both axes: mAP
   −0.021 vs v23b AND unique barely changes (0.110 vs 0.111). The two
   diversity mechanisms (gate-removal and richer prompt) are not
   complementary — they collapse to the same operating point in the
   logistic-loss regime.

3. **Why MSE responds to V3 but hashnet does not.** MSE-Jaccard makes the
   per-pair target a sharp function of label overlap, so any extra
   per-sample text variation flows through to per-sample code variation.
   The hashnet binary logistic only cares about sign-of-score, so all
   positives are pushed into the same `score > 0` half-plane regardless
   of how nuanced the underlying text was — the V3 prompt's extra
   information has nowhere to land.

Decision: keep V3 prompt available (cache stays at
`./cache/flickr25k_qwen_v3.jsonl` and `./cache/flickr25k_siglip2_v3/`)
for any future MSE-form ablations or for cross-dataset narratives, but
do not adopt it as the default on the current hashnet-based runs. The
Pareto frontier is unchanged: v20-K64 (max mAP), v23b (mAP-tied at
slightly higher unique), v24b (balanced — high unique).

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

---

---

---

---

---

---

## 2026-05-14 — v25a / v25b: `loss_hash_hard` form ablation

🔴 both reverted — confirms that the **mixed loss form** in v23b is intentional.

User question: should the STE hard-code retrieval loss (`loss_hash_hard`,
λ=0.5) follow the same form as `loss_hash`? It is currently always MSE-
Jaccard even when `lambda_hash_type=hashnet`. Two ablations launched in
parallel from the v23b setup (K=64, no global gate):

- **v25a**: route the STE hard code through `_loss_hash_hashnet` whenever
  `lambda_hash_type=hashnet` so soft and hard paths share the logistic form.
- **v25b**: `--lambda_hash_hard 0.0` — disable the hard-path loss entirely.

Result (Flickr25k):

| run                                |   mAP  | Δ vs v23b | unique | dup    | gap |
|------------------------------------|-------:|----------:|-------:|-------:|----:|
| v23b (mixed form)                  | 0.7877 |    —      | 0.1107 | 0.8893 | 5.05 |
| **v25a (hashnet form on hard)**    | 0.7608 | **−0.027** | 0.0444 | 0.9556 | 4.63 |
| **v25b (lambda_hash_hard=0)**      | 0.7631 | **−0.025** | 0.0512 | 0.9488 | 5.00 |

Both regress on **both axes** — mAP AND unique simultaneously. The
unique-code count drops to 0.044 / 0.051 (half of v23b) which contradicts
the prior hypothesis that "hard MSE-Jaccard was the collision source".

Reinterpretation: the soft-and-hard combination is doing complementary
work:

- `loss_hash` (hashnet binary) gives a **margin signal** — push positives
  above 0, negatives below 0 — but lets all positives pile onto one
  cluster.
- `loss_hash_hard` (MSE-Jaccard on STE-quantised codes) supplies a
  **graduated target** per pair (Jaccard ∈ [0,1] instead of {0,1}). For
  partial-overlap positives it asks for `sim_dna ≈ 0.33` not `sim_dna ≈ 1`,
  which gives the model room to spread same-cluster samples across nearby
  codes instead of merging them.

Remove or homogenise that signal and the model collapses onto fewer
hash codes (unique 0.11 → 0.04) AND the hard path loses its train↔test
bridge (mAP −0.025).

The forward branch was reverted to "soft → hashnet if selected; hard →
always MSE-Jaccard". The comment in the code documents the rationale so
future passes don't try the same thing again. `--lambda_hash_hard` stays
in the CLI as a tunable knob (with the v25b result as warning).

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

---

---

---

---

---

---

## 2026-05-14 — v24a (Wasserstein restored + no gate), v24b (Jaccard + no gate)

🟢 v24b is a **new Pareto point** (mAP > v6, unique 3x v23b). v24a is
inactive (no net benefit over v23b).

Goal: revisit two previously-killed loss/target choices now that v23b has
opened the (mAP ≈ 0.788, unique ≈ 0.11) corner of the frontier without a
C_0 gate.

### v24a — Wasserstein restored, K=64, no gate

R4's `lambda_wasserstein` (v11 sweet spot 0.05) was removed during the
2026-05-13 cleanup. The Sinkhorn router still computes `ot_cost`; we
re-added the loss term plus its config / train-types entry. Smoke test
confirmed gradient flow.

| run                            |   mAP  | unique | dup    | dead  | pos→neg gap |
|--------------------------------|-------:|-------:|-------:|------:|------------:|
| v23b (no gate, binary S)       | 0.7877 | 0.1107 | 0.8893 | 0.005 | 5.05 |
| **v24a (v23b + Wasserstein 0.05)** | 0.7881 | 0.0734 | 0.9266 | 0.000 | **5.38** |

mAP is statistically tied with v23b (+0.0004). The Wasserstein term
widened the pos→neg Hamming gap from 5.05 to 5.38 (best of any variant
so far), but it collapsed the per-image unique count: 0.111 → 0.073, the
LOWEST in the v18-derived line. Net result: the extra OT alignment
trades unique codes for margin without buying any mAP. Not Pareto-
improving on v23b. `--lambda_wasserstein` stays in the tree but
defaults to 0.

### v24b — Jaccard target + no gate, K=64

The other never-combined ablation: replace binary `S = any-shared` with
fractional `S = Jaccard`, on top of v23b (no gate). Hypothesis: with the
gate gone, the model has the structural room to honour Jaccard's
graduated target probabilities instead of collapsing all positive pairs
onto a single cluster.

| run                            |   mAP  | unique | dup    | dead  | pos→neg gap |
|--------------------------------|-------:|-------:|-------:|------:|------------:|
| v19a (Jaccard + gate, K=32)    | 0.7728 | 0.2240 | 0.7760 | 0.000 | 3.90 |
| v23b (binary + no gate, K=64)  | 0.7877 | 0.1107 | 0.8893 | 0.005 | 5.05 |
| **v24b (Jaccard + no gate, K=64)** | 0.7742 | **0.3237** | 0.6763 | 0.000 | 3.81 |
| v6 (MSE-Jaccard, K=32)         | 0.7556 | 0.3958 | 0.6042 | 0.000 | 2.86 |

v24b is now the strongest "balanced" variant on the (mAP, unique) plane:
- vs v6 baseline (the prior balanced anchor): mAP +0.0186, unique only
  drops 0.07 (0.396 → 0.324).
- vs v19a (previous Pareto balanced point): mAP +0.0014, unique
  **+0.0997** (0.224 → 0.324) — strict Pareto-dominate.
- vs v23b (current default operating point): mAP −0.014, unique **+0.213**
  (0.111 → 0.324) — different trade-off, not a strict dominance.

Mid-eval trajectory was characteristic: starts low (ep10 mAP 0.7343 with
unique 0.435) because Jaccard granular targets need more time to settle,
then climbs to 0.76 range while unique stays 0.6+. The final hash code
fans out into 7,448 distinct hashes for 23K DB images vs v18's 2,243.

Updated Pareto frontier:
```
mAP
0.79  v20-K64 ⭐ (max)   v18   v24a   v23b
0.78
0.77                                          ⭐ v24b ⭐ (max-balance)
0.76                            v19a
0.75  v6
      +-----------+-----------+-----------+-----------> unique
      0.10        0.20        0.30        0.40
```

### Recommendation

Paper-wise, **report both v20-K64 and v24b** — same dataset, same backbone,
same training budget, just different `lambda_hash_type` and target form.
The pair gives a clean (high-mAP) ↔ (high-unique-compositional) trade-off
story instead of forcing one number to do everything.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

---

---

---

---

---

---

## 2026-05-14 — v23a (mean-pool text) reverted, v23b (no global gate) → Pareto step

🔴 v23a reverted · 🟢 v23b kept as available ablation.

**v23a** — replace SigLIP2's `pooler_output` with the mean of valid token
features as the per-slot pooled text embedding. Hypothesis: pooler is
last-token pool and lands on near-identical vectors per image (cos 0.88).
Token mean is empirically more diverse (cos 0.78), so this should propagate
to more discriminative routing centroids.

Result: stopped mid-training around epoch 30 — mid-eval mAP tracked
v20-K64 within 0.005 but unique stayed in the same 0.10 band. Effect was
not large enough to be worth a full 60-epoch run, given v23b was already
queued. The (small) `--text_pool_mode mean` code path was removed from
the source tree on revert.

**v23b** — disable the gated C_0 → C_1..5 codeword addition before the
codon heads. Each local codon head now sees its own quantised codeword in
isolation (no global blending). New CLI flag `--disable_global_gate`.

Setup: v20-K64 base (K=64, hashnet binary, siglip2_global) + only the
new flag.

Result (Flickr25k, single seed):

| run                                    |   mAP   | unique  | dup    | dead  | pos→neg gap |
|----------------------------------------|--------:|--------:|-------:|------:|------------:|
| v20-K64 (with C_0 gate)                | 0.7892  | 0.0982  | 0.9018 | 0.005 | 5.20 |
| **v23b (no C_0 gate)**                 | **0.7877** | **0.1107** | 0.8893 | 0.005 | 5.05 |
| Δ                                      | −0.0015 | **+0.013** | −0.013 |  0    | −0.15 |

- mAP regression is within noise (0.0015 / 0.78 ≈ 0.2%).
- Unique codes: **2,256 → 2,556** for the same 23K DB → +300 distinct
  hashes (+13.3%) at essentially zero retrieval cost.
- Per-codebook perplexity unchanged (mean 52.2 → 51.8), so the gain isn't
  from reviving dead codewords — it's purely from undoing the redundancy
  the C_0 addition was injecting into every local codon head.

This is the first ablation that genuinely moves the Pareto frontier on
the (mAP, unique) plane near our best operating point. The codebook
geometry diagnostic earlier in the log warned about exactly this: adding
the C_0 codeword vector (whose codewords are nearly orthogonal, low
effective rank 2.5) to each local codeword (clustered around its own
centroid, low effective rank 3.4) was contaminating the local hashes
with a shared additive offset and reducing the count of distinct hash
patterns the model could produce.

Decision: keep `--disable_global_gate` in the tree as a recommended
toggle for "compositional-code-prioritising" runs. v20-K64 (with gate)
keeps the absolute mAP record by 0.0015; v23b is the better choice when
unique-code count matters for the paper claim.

Pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

---

---

---

---

---

---

## 2026-05-14 — MSCOCO baseline comparison in 4-base DNA space (fair-axis ablation)

🟢 reference

Motivation: the existing 2026-05-12 cross-dataset table compares baselines
in their native **36-bit binary** space against our model in **18-base
DNA** space. The two spaces use different distance metrics (signed binary
Hamming vs base-level Hamming) and different "unique code" semantics
(36-bit string vs 18-base sequence), which makes the mAP / unique trade-off
not directly comparable. This ablation forces every method onto the same
DNA axis.

Procedure:
1. Re-train OrthoHash / HashNet / DPSH on MSCOCO setting1 (60 epochs,
   36-bit head on cached SigLIP2, identical optimiser to the 2026-05-12
   run). CSQ skipped — its 260512 run had failed to produce a result.
   Trial names: `<method>_mscoco_dnacompare` under
   `result_baseline/260514/`, checkpoints at `params/260514/`.
2. Re-extract sign-binary codes `{-1,+1}^36` on query (5,000) and DB
   (107,218).
3. Convert to 18-base DNA: reshape `[N, 36] → [N, 18, 2]`, map
   `{00=A, 01=C, 10=G, 11=T}` (same `BASE_TO_BITS` convention as
   `dna_utils/dna_code_utils.py`).
4. Run `evaluation_siglip2.evaluate_retrieval(..., distance_mode='base')`
   — base-level Hamming, multi-hot Jaccard relevance. `unique_code_ratio`
   computed over the full 18-base sequences.
5. Compare to Ours v6_promptv3, which already produces base codes
   natively (`result/260514+mscoco_setting1_v6_promptv3.../`).

Conversion + eval script: `/tmp/convert_baseline_to_dna_eval.py`. Summary
dump: `cache/baseline_mscoco_dna/summary.json`.

| Method               | mAP (base) | Δ vs native binary mAP | Unique DNA codes / 107,218 | Unique % |
|----------------------|-----------:|-----------------------:|---------------------------:|---------:|
| **OrthoHash**        | **0.6025** | −0.022 (was 0.6243)    | 88,774                     | 82.80%   |
| **HashNet**          |   0.6015   | −0.006 (was 0.6076)    | 31,489                     | 29.37%   |
| **Ours v6_promptv3** |   0.5339   | (DNA-native, no Δ)     | 21,688                     | **20.23%** |
| **DPSH**             |   0.4637   | +0.007 (was 0.4567)    |    138                     |  0.13%   |

Findings:

1. **mAP gap survives the axis change.** OrthoHash / HashNet stay ahead of
   v6_promptv3 by +0.07 even after collapsing their 36-bit space into 18
   base pairs. The base-mode handicap (1-bit-different and 2-bits-different
   pairs both count as one mismatch) costs them at most −0.022 mAP, not
   enough to close the gap.

2. **Compression axis is where our model wins.** v6_promptv3 produces 4×
   fewer unique DB codes than OrthoHash (21.7k vs 88.8k) and ~1.5× fewer
   than HashNet. That's by design — the codebook-VQ + DNA construction
   actively shares codes across semantically similar images, whereas
   OrthoHash's 36 bits are effectively unconstrained random per image.

3. **DPSH is degenerate.** 138 unique codes across 107K images = 99.87%
   collapse. Its mAP 0.4637 is therefore not a meaningful retrieval signal
   — it ranks within a tiny number of equivalence classes. Should be
   excluded from any "compactness vs retrieval" discussion.

4. **No apples-to-apples Pareto neighbour.** None of the binary baselines
   live in our compression regime (20% unique). Reporting this as
   "Pareto-incomparable" is more honest than reporting only mAP.

Practical implication for the paper: when claiming retrieval mAP we should
report the **native distance** for each method (OrthoHash 0.6243 vs Ours
0.5339), not the base-mode rewrite — base mode unfairly handicaps the
binary baselines. The DNA-axis numbers above are the right context for
the **compression** claim (unique-code ratio), not the retrieval claim.

CSQ MSCOCO still missing — its 260512 trial folder contains only
`config.json`. Re-add to a future re-run if a 4-method baseline panel is
needed.

---

---

---

---

---

---

---

## 2026-05-14 — Repository placed under git version control

🟢 active — pushed to `github.com:yunddonii/GroundedDNA_save` (`main`).

---

---

---

---

---

---

---

## 2026-05-13 — v22b: visual-attention pooled text tokens (Option B) — regression

🔴 reverted — net regression of −0.022 vs v20-K64. With v22a (−0.046) this
closes the "fix the text encoder collapse" line of work: pushing the six
slot embeddings apart, by ANY mechanism explored so far, hurts retrieval.

Setup: v20-K64 base + cross-attention block that takes the (re-extracted)
token-level SigLIP2 text features and lets visual patches cross-attend
over them per slot. New infrastructure:

- `extract_siglip2_features.py --save_text_tokens`: additionally saves
  `text_tokens.f16.npy` ([N, 6, T, D_proj]) and `text_token_mask.bool.npy`
  ([N, 6, T]). Falls back to `last_hidden_state` directly if the model
  has no separate `text_projection` layer (SigLIP2 does not).
- `dataloaders.py::_SigLIP2FeatureCache.__init__/get`: loads the token
  cache if present and exposes `cached_text_tokens` / `cached_text_token_mask`
  in the batch.
- `model_siglip2.py`: new `nn.MultiheadAttention` block + LN. Per slot m,
  cross-attend `visual_tokens` (Q) over `text_tokens[:, m, :, :]` (K, V)
  with the per-slot attention mask, then mean-pool over the visual axis to
  get a [B, D] visually-conditioned slot embedding. Stacks to [B, 6, D]
  for the Sinkhorn router.
- `config.py`: new CLI `--use_text_token_attention --text_attn_num_heads`.

Diversity check at init (untrained random attention block): cross-slot
mean cos sim ≈ 0.07 (vs 0.88 of the pooled cache). Architecturally the
slots ARE distinguishable.

Token-level cache verification on Flickr25k:
- Token mean-pooled cross-slot cos: ~0.78 (vs 0.88 for pooler_output)
  → token-level is inherently more diverse than pooled, confirming
  the cache contains additional signal.

Result (Flickr25k, K=64, hashnet binary, single seed):

| run                                |   mAP  | Δ vs v20-K64 | unique | dup    | dead  | pos→neg gap |
|------------------------------------|-------:|-------------:|-------:|-------:|------:|------------:|
| v20-K64 (pooled text, shared MLP)  | 0.7892 |       —      | 0.0982 | 0.9018 | 0.005 | 5.20 |
| **v22b (token-level + visual-attn pool)** | **0.7673** | **−0.022** | 0.0660 | 0.9340 | 0.003 | 5.09 |

Mid-eval trajectory was volatile:
```
ep 10  0.7408
ep 20  0.7271   ← deepest dip
ep 30  0.7396
ep 40  0.7570
ep 50  0.7549
ep 60  0.7543 / final 0.7673
```

Compared to v22a (per-slot MLP) which final-ed at 0.7433, v22b is
markedly better (visual conditioning is informative), but still
substantially worse than v20-K64. The bouncing trajectory matches
v22a's pattern: introducing a mechanism that decorrelates the slot
embeddings destabilises the codebook learning step.

Mechanism (post-hoc, same as v22a's lesson): the 0.88 cos similarity is
NOT a defect of the SigLIP2 text encoder — it is **an image-level
consistency anchor**. With it, the six routing centroids per image lie
in a small neighbourhood, and the Sinkhorn coupling is driven primarily
by visual content rather than by which-centroid-is-closer. Both v22a
(per-slot MLP, random orthogonality) and v22b (visual-attention, content-
aware orthogonality) break that anchor. v22b breaks it less aggressively
(content-aware), hence smaller regression (−0.022 vs v22a's −0.046).

Decision: code path remains in the tree (cache extraction, dataloader,
model branch) but `--use_text_token_attention` defaults to off. The
cache at `cache/flickr25k_siglip2_tokens/` is retained for future
"text + something else" experiments.

The conclusion across v22a/v22b is paper-relevant: trying to "fix" the
text-encoder collapse from the OUTPUT side (after the encoder) does not
help retrieval. Any future revisiting should attack at the source —
e.g., richer per-slot prompts, fine-tuned text encoder, or replacement
with Qwen2.5-VL hidden states.

---

---

---

---

---

---

---

## 2026-05-13 — v22a: per-slot `text_adapter` (Option A) — regression

🔴 reverted — per-slot text adapter hurts mAP by −0.046 on Flickr25k.

Trigger: diagnostic on SigLIP2 cached text embeddings revealed that the
six per-slot pooled vectors are **near-identical** across an image
(pairwise cos ~0.88, per-slot effective rank 2.7/D=768). The shared
`TextAdapter` MLP applied to all 6 slots preserves that collapse, so the
Sinkhorn router has essentially identical centroids per image -> codebook
supervision is uninformative -> high cross-codebook correlation
(observed v18/v20-K64 codeword pairwise cos 0.40, codebook effective
rank 3 / K).

User proposal — **Option A (cheap)**: replace the shared MLP with an
`nn.ModuleList` of six independent MLPs (one per codebook slot). The
slot-specific weights have the structural freedom to push the six
near-identical inputs apart.

Architectural change in `model_siglip2.py`:
```python
if self.per_slot_text_adapter:
    self.text_adapter = nn.ModuleList([TextAdapter(...) for _ in range(6)])
# forward
per_slot = [self.text_adapter[m](raw[:, m, :]) for m in range(6)]
text_part_tokens = torch.stack(per_slot, dim=1)         # [B, 6, D]
```
New CLI: `--per_slot_text_adapter`.

Init smoke test confirmed the architectural goal works *at init*: six
independent random MLPs produce cross-slot cos ~0.01 (essentially
orthogonal) from cos 0.88 inputs.

Result (Flickr25k, K=64, hashnet binary, single seed):

| run                                  |   mAP  | Δ vs sib | unique | dup    | dead  | pos→neg gap | base normH |
|--------------------------------------|-------:|---------:|-------:|-------:|------:|------------:|-----------:|
| v20-K64 (shared text_adapter)        | 0.7892 |    —     | 0.0982 | 0.9018 | 0.005 | 5.20 | 0.7628 |
| **v22a (per-slot text_adapter)**     | 0.7433 | −0.0459 | 0.0640 | 0.9360 | 0.060 | 3.82 | 0.6809 |

Mid-eval trajectory was the diagnostic warning:
```
ep 10  0.7475
ep 20  0.7525   ← peak
ep 30  0.7408
ep 40  0.7295   ← bouncing down
ep 50  0.7353
ep 60  0.7293 / final 0.7433
```
Per-codebook perplexity (K=64) shows where the damage lands:
```
                 C_0   C_1   C_2   C_3   C_4   C_5
v20-K64:        60.4  54.0  54.6  49.8  46.3  47.9   (mean 52.2)
v22a:           60.8  23.5  22.5  33.4  22.5  31.0   (mean 32.3)
                       ↓↓    ↓↓          ↓↓
```
C_0 is intact (it's fed by `visual_global`, not by text routing) but the
five local codebooks lose ~half their effective codeword count.

Mechanism (post-hoc): the 0.88 cross-slot cos sim under the shared
`TextAdapter` was NOT a bug — it carried "all six descriptions refer to
the *same image*" as a consistency anchor. The Sinkhorn router used that
anchor to route patches of one image to a coherent set of part-specific
centroids that all lie in the same image-conditional neighbourhood.
Six random orthogonal centroids (the per-slot init state) destroy that
anchor: patches now fall onto whichever centroid they happen to lie
closest to, with no image-level consistency. Routing becomes
content-blind → codebook EMA centres see inconsistent inputs → local
codebooks (C_1..C_5) collapse to a smaller effective codeword count.

The "cos 0.88 is bad" hypothesis was wrong; the right description is
"semantic orthogonality across slots is missing", which is a different
problem and the per-slot MLP can't solve it (it has no signal to
discover semantic axes from near-identical inputs).

Decision: revert `--per_slot_text_adapter` to default off. Code path
remains but inert. If we still want to spread the slot embeddings, the
fix has to inject extra information — Option B (visual-attention-pooled
TEXT TOKENS, currently un-cached) or Option C (use Qwen hidden states
directly) — not just structural slot-specific weights with no extra
signal.

---

---

---

---

---

---

---

## 2026-05-13 — v21a / v21b: `c_global_source=mean_pool` ablation under hashnet form

🟡 superseded — both regress vs their `siglip2_global` counterparts.

User question: under the new hashnet logistic form (v18/v20-K64), does the
legacy `c_global_source=mean_pool` work better than `siglip2_global`?
The original v5→v6 jump was switching mean_pool → siglip2_global, but
that win predates the loss-form switch.

Setup: identical to v18 / v20-K64 except `--c_global_source mean_pool`.

Results (Flickr25k, 36-bit, single seed):

| run                                |   mAP  | Δ vs sib | unique | dup    | dead  | pos→neg gap | base normH |
|------------------------------------|-------:|---------:|-------:|-------:|------:|------------:|-----------:|
| v18  K=32, siglip2_global          | 0.7883 |    —     | 0.0975 | 0.9025 | 0.000 | 5.10 | 0.7454 |
| **v21a K=32, mean_pool**           | 0.7821 |  −0.0062 | **0.0423** | 0.9577 | 0.000 | 4.47 | 0.7151 |
| v20-K64 K=64, siglip2_global       | 0.7892 |    —     | 0.0982 | 0.9018 | 0.005 | 5.20 | 0.7628 |
| **v21b K=64, mean_pool**           | 0.7834 |  −0.0058 | 0.0683 | 0.9317 | 0.021 | 4.84 | 0.7443 |

Findings:

1. **Both mean_pool variants regress.** mAP drops by 0.006 at both
   K=32 and K=64; magnitude is similar so the effect is purely from the
   global slot, not from K.

2. **Unique code count gets WORSE under mean_pool.** v21a hits 0.042 —
   a new minimum across all v6-derivative runs. Of 23,000 DB images, only
   ~970 distinct hashes are produced.

3. **Mechanism**: `mean_pool` averages the *post-adapter* visual_tokens —
   that's the same pool the 5-part router feeds from. C_0 ends up sampling
   the same feature distribution as C_1..C_5, so the 6 codebooks become
   more correlated. v18/v20-K64's `siglip2_global` feeds the MAP-pooled
   pre-projection class-aware embedding into C_0 through `global_adapter`,
   giving C_0 an information channel the local codebooks don't share.

4. The original v5→v6 win (+0.007 on Flickr) was therefore the SigLIP2 MAP
   feature being a strong, codebook-decorrelated input — a property the
   hashnet form preserves and amplifies. `siglip2_global` is firmly the
   default going forward.

Pareto frontier (unchanged top):
```
v20-K64 (unique 0.098, mAP 0.7892)   ← mAP SOTA
v18     (unique 0.098, mAP 0.7883)
v20a    (unique 0.097, mAP 0.7840)
v21b    (unique 0.068, mAP 0.7834)
v19b    (unique 0.086, mAP 0.7827)
v21a    (unique 0.042, mAP 0.7821)   ← worst unique
v19a    (unique 0.224, mAP 0.7728)
v6      (unique 0.396, mAP 0.7556)
```

---

---

---

---

---

---

---

## 2026-05-13 — v20a / v20-K64: S-cap and capacity ablations (new mAP SOTA, unique unchanged)

🟢 reference — v20-K64 establishes a new Flickr25k mAP best (0.7892); unique
code utilisation remains at the v18 floor (~0.10) across all hashnet
variants, regardless of S-cap or K.

User goal restatement: maximise unique_code_ratio simultaneously with mAP,
since "compositional code" is a paper contribution. v18 collision (90%)
undermines that claim. Two single-variable ablations off v18:

- **v20a**: cap `S_target` at 0.9 inside the hashnet logistic. Same-powerset
  positives no longer aim for `sigmoid(score)=1` exactly → residual degree
  of freedom inside each cluster should yield distinct codes.
  New CLI flag: `--hashnet_S_cap`.
- **v20-K64**: same v18 form but `--codebook_size 64` (vs default 32).
  Doubles per-codebook codeword count → larger candidate hash space.

Results (Flickr25k, 36-bit, otherwise v6 architecture, single seed):

| run                                |   mAP  | Δ vs v6 | unique | dup    | dead  | pos→neg gap | base normH |
|------------------------------------|-------:|--------:|-------:|-------:|------:|------------:|-----------:|
| v6   baseline (MSE-Jaccard)        | 0.7556 |    —    | 0.3958 | 0.6042 | 0.000 | 2.86 | 0.9516 |
| v18  logistic + binary S, K=32     | 0.7883 | +0.033  | 0.0975 | 0.9025 | 0.000 | 5.10 | 0.7454 |
| v19a logistic + Jaccard S, K=32    | 0.7728 | +0.017  | 0.2240 | 0.7760 | 0.000 | 3.90 | 0.9662 |
| v19b logistic + binary, η=5, K=32  | 0.7827 | +0.027  | 0.0858 | 0.9142 | 0.000 | 4.85 | 0.7373 |
| **v20a logistic + binary, S_cap=0.9, K=32** | 0.7840 | +0.028 | 0.0965 | 0.9035 | 0.000 | 4.60 | 0.7876 |
| **v20-K64 logistic + binary, K=64** | **0.7892** | **+0.034** | 0.0982 | 0.9018 | 0.005 | **5.20** | 0.7628 |

Findings:

1. **`K=64` is the new mAP best on Flickr25k (0.7892)**, marginally above
   v18's 0.7883. The bigger codebook produces slightly better
   discriminability (pos→neg gap 5.20 vs 5.10) at the cost of 5 of 64
   codewords going dead per codebook. mAP ceiling for the v18 form
   appears to be ~0.79; doubling K does not break through it.

2. **S-cap=0.9 (v20a) does NOT recover unique code count.** Diversity stays
   at 0.097 — *identical* to v18. The cap successfully raised base
   normalised entropy 0.745 → 0.788 (per-position usage slightly more
   uniform), but the cluster-level collapse (same-powerset → one hash)
   was unaffected. The capped target still has all positives aiming at
   the same `sigmoid(score)=0.9`, so they still pile up on the same code.

3. **K does not increase unique either.** v20-K64 hash-unique-ratio is
   0.098 — almost the same as K=32 (v18: 0.0975). Capacity is not the
   bottleneck for collision; the *form* of the loss is.

4. **The unique-code-ratio floor ≈0.10 is structural** across every
   hashnet-form variant tried (v18, v19b, v20a, v20-K64). Flickr25k has
   ~3.7 labels per image → ~60–70% of random pairs share at least one
   label → all those positives converge onto a small number of cluster
   centres under the binary-S logistic. Single-knob fixes (cap, η, K)
   shift base-entropy and mAP but leave the cluster count unchanged.

Pareto frontier of the variants explored so far (Flickr25k):
```
v20-K64 (unique 0.098, mAP 0.7892)   ← new mAP SOTA
v18     (unique 0.098, mAP 0.7883)
v20a    (unique 0.097, mAP 0.7840)
v19b    (unique 0.086, mAP 0.7827)
v19a    (unique 0.224, mAP 0.7728)   ← Pareto-dominates v6
v6      (unique 0.396, mAP 0.7556)
```
v19a stays Pareto-optimal for the "balanced" axis (best mAP/unique trade-off);
v20-K64 wins pure-mAP.

Implication: to get both high mAP AND high unique simultaneously, the
hashnet form alone is insufficient. Candidates for the next iteration:
- Explicit anti-collision loss (e.g. soft cap on max cluster size)
- Hadamard / class-centre target codes (CSQ-style) so distinct
  label-powersets target deterministically distinct hashes
- Combine v19a's Jaccard target with v20-K64's larger K

---

---

---

---

---

---

---

## 2026-05-13 — v19a / v19b: trade-off ablations off v18

🟢 reference — v18 remains the highest-mAP variant; v19a is the
diversity-recovery balance point.

After v18 won mAP-wise (+0.033 over v6) but pushed collision to 90% and
per-position base normalised entropy to 0.745 (vs v6's 0.952), two
single-variable ablations were launched in parallel from the v18 starting
point:

- **v19a**: same logistic likelihood but plug FRACTIONAL Jaccard S into the
  target instead of binary "any-shared". Hypothesis: per-label-combo
  granularity in the target probability restores per-image distinctness.
  New CLI flag: `--hashnet_use_jaccard`.
- **v19b**: v18's binary S form unchanged, but raise the base-balance
  coefficient inside `loss_dna`: `eta_base_balance 1.0 → 5.0`. Hypothesis:
  per-position 25% A/C/G/T uniformity is what the model is failing.
  New CLI flag: `--eta_base_balance`.

Results (Flickr25k, 36-bit, otherwise v6 architecture, single seed):

| run                                              |   mAP  | Δ vs v6 | unique | dup    | pos→neg gap | base normH |
|--------------------------------------------------|-------:|--------:|-------:|-------:|------------:|-----------:|
| v6   baseline (MSE-Jaccard)                      | 0.7556 |    —    | 0.3958 | 0.6042 | 2.86 | 0.9516 |
| **v18  logistic + binary S, η=1**                | **0.7883** | **+0.033** | 0.0975 | 0.9025 | **5.10** | 0.7454 |
| **v19a logistic + Jaccard S, η=1**               | 0.7728 | +0.017 | **0.2240** | 0.7760 | 3.90 | **0.9662** |
| **v19b logistic + binary S, η=5**                | 0.7827 | +0.027 | 0.0858 | 0.9142 | 4.85 | 0.7373 |

Findings:

1. **Both single-variable changes leave v18 the mAP winner.** v19a drops by
   −0.0155, v19b by −0.0056. v18's binary-S logistic remains the
   strongest mAP configuration.

2. **v19a successfully restores diversity** with a non-trivial mAP cost.
   `unique_code_ratio` 0.098 → 0.224 (×2.3), `mean_base_normalised_entropy`
   0.745 → 0.966 (essentially uniform A/C/G/T). The Jaccard target form
   gives each label-powerset combination its own target probability, so the
   model doesn't collapse all positive pairs onto a single cluster.

3. **v19b does NOT recover base balance.** `mean_base_normalised_entropy`
   actually drops 0.7454 → 0.7373. Raising `eta_base_balance` 5× had no
   visible effect, because (a) the term is on the batch-mean base
   distribution rather than per-position, and (b) the entropy term in
   `loss_dna` (one-hot pressure) competes with it. A proper fix would
   require a per-position uniformity term, not just a bigger η.

4. **Per-codebook perplexity** is highest in v19a (mean 28.3) — v19a is
   the most "compositional" code in terms of codebook diversity, even
   though v18 still gives the best retrieval. Useful for the paper's
   compositionality claim.

Pareto frontier (diversity vs mAP) for the four runs:
```
v18  (unique 0.098, mAP 0.7883)   ← max mAP
v19b (unique 0.086, mAP 0.7827)
v19a (unique 0.224, mAP 0.7728)   ← balance point, Pareto-dominates v6
v6   (unique 0.396, mAP 0.7556)   ← max diversity (legacy MSE)
```
v19a strictly Pareto-dominates v6 on both axes (better mAP AND lower
duplicate rate). v18 trades almost all the diversity to win the last
0.0155 of mAP.

Recommendation:
- Keep `--lambda_hash_type hashnet` (v18 form) as the **primary** running
  configuration.
- Keep `--hashnet_use_jaccard` available as the **balanced** variant
  documented in the paper's compositionality argument.
- Drop `--eta_base_balance` strengthening for now; address base imbalance
  via a per-position term in a future iteration (not committed).

---

---

---

---

---

---

---

## 2026-05-13 — v18: HashNet-style logistic `loss_hash` (Option A) — new best

🟢 active — adopted as the recommended hash loss form.

Trigger: diagnostic showed that the MSE-on-Jaccard form of `loss_hash`
structurally pushes full-label-overlap pairs to *identical* continuous
codes, producing the 60% duplicate-rate that has been the headline
problem since v6. The user asked to follow the HashNet (Cao et al.,
ICCV 2017) line of prior work and replace the MSE form with a sigmoid /
logistic-likelihood form.

Implementation: see `_loss_hash_hashnet` in `loss_siglip2.py`. Adapts
HashNet to our continuous-code shape `[B, 18, 4]`:

```text
sim_dna  = mean_r <u[i,r,:], u[j,r,:]>            # ∈ [0, 1] -- existing DNA sim
score    = α · (2 · sim_dna − 1)                  # ∈ [-α, α] -- centred + scaled
S_bin    = (label_sim > 0).float()                # binary: any-shared = positive
nll      = softplus(-score) + (1 − S_bin) · score # HashNet logistic likelihood
loss_hash = weighted-mean-of-nll over off-diagonal pairs
            with class-balance weight (HashNet's S/S₀ and S/S₁).
```

CLI: `--lambda_hash_type {mse, hashnet}` (default `mse` preserves v6),
plus `--hashnet_alpha` (default 1.0).

Result (Flickr25k, 60 epochs):

| run                                |   mAP  | unique | dup    | dead | pos→neg gap |
|------------------------------------|-------:|-------:|-------:|-----:|------------:|
| v6  (MSE-Jaccard `loss_hash`)      | 0.7556 | 0.3958 | 0.6042 | 0.00 | 2.86 |
| **v18** (HashNet-style, α=1.0)     | **0.7883** | 0.0975 | 0.9025 | 0.00 | **5.10** |
| Δ                                  | **+0.033** | −0.30 | +0.30 | 0 | **+2.24** |

Standing vs the four binary baselines on Flickr25k (36-bit, frozen SigLIP2):

| Method                | mAP    | P@1   | P@10  | P@100 | P@1000 |
|-----------------------|-------:|------:|------:|------:|-------:|
| **v18 (Ours)**        | **0.7883** | 0.880 | 0.862 | 0.860 | 0.849 |
| HashNet               | 0.7800 | 0.872 | 0.871 | 0.864 | 0.853 |
| CSQ                   | 0.7596 | 0.889 | 0.879 | 0.875 | 0.856 |
| OrthoHash             | 0.7565 | 0.895 | 0.888 | 0.882 | 0.860 |
| v6 (Ours, MSE form)   | 0.7556 | —     | —     | —     | —      |

→ v18 leads on mAP by +0.008 over HashNet itself, even though baselines
still edge us out marginally on the top-K precision metrics. Codebook
utilisation stays healthy (per-codebook perplexity 23.0–30.0 on average
26.7, vs v6's 27.8 — small drop).

Diagnostic surprises:
1. **Collision goes *up*, mAP goes *up* together.** Duplicate-rate
   60% → 90%, yet mAP improves by +0.033. The collisions land on the
   "right" same-label clusters; the binary-S logistic form actively
   encourages this in multi-label data, while pos→neg distance gap
   widens from 2.86 → 5.10 (the real driver of mAP).
2. Earlier hypothesis that "high collision necessarily harms mAP" is
   wrong for multi-label benchmarks like Flickr25k. Collision quality
   matters, not collision quantity.
3. The MSE-on-Jaccard form was over-constraining same-label pairs to
   identical codes WITHOUT widening their margin from negatives;
   HashNet's logistic form makes that margin a first-class objective
   ("score > 0 for positives, < 0 for negatives") and our model
   exploits the freed capacity to push it further (gap 5.10 ≫ 2.86).

Code state: `--lambda_hash_type` defaults to `mse` so existing scripts
reproduce v6 exactly; the new default for fresh runs going forward is
`hashnet`.

---

---

---

---

---

---

---

## 2026-05-13 — v17: re-wired `loss_global` to use STE (option (a)) → net regression

🔴 reverted — properly activating `loss_global` HURTS Flickr25k mAP.

Follow-up to the "no-op under EMA" entry below. Two ways to make
`loss_global` actually train something were proposed: (a) feed it the
STE-wrapped `quantized_tokens` so the gradient reaches `global_adapter`,
(b) drop the term entirely so the inventory matches reality. The user chose
to try (a) first.

Implementation (one-line patch in `loss_siglip2.py::forward`):
```diff
- loss_global = self._loss_global(q[:, 0, :], S, mask)        # q = quantized_tokens_raw (no grad)
+ q_ste = outputs.get("quantized_tokens")
+ loss_global = self._loss_global(q_ste[:, 0, :], S, mask)    # STE-wrapped (grad path to global_adapter)
```
Smoke test confirmed `loss_global.grad_fn` now exists and a 4421-magnitude
gradient flows into `global_adapter` (with `visual_adapter` / `text_adapter`
correctly receiving zero, since C_0 routes only through `global_adapter`).

Result (Flickr25k, 60 epochs, otherwise v6 config):

| Run                                   |   mAP  | unique | dup    | dead  | pos→neg gap |
|---------------------------------------|-------:|-------:|-------:|------:|------------:|
| v6  baseline (`loss_global` no-op)    | 0.7556 | 0.3958 | 0.6042 | 0.000 | 2.86 |
| **v17 `loss_global` rewired (STE)**   | **0.7439** | 0.3247 | 0.6753 | 0.010 | 2.78 |
| Δ                                     | **−0.012** | −0.071 | +0.071 | +0.010 | −0.08 |

Per-codebook perplexity collapsed asymmetrically on the local slots:
```
            C_0   C_1   C_2   C_3   C_4   C_5
v6:        31.3  26.8  29.0  28.7  26.4  24.3   (mean 27.8)
v17:       31.0  15.3  27.1  26.7  14.6  25.0   (mean 23.3)
                  ↓↓                ↓↓
```
C_0 itself looks normal (perp 31). C_1 and C_4 dropped to ~half their v6
utilisation -- their codeword count effectively halved.

Proposed mechanism: when `loss_global` actually trains `global_adapter`,
the C_0 codeword pairwise similarity gets pulled directly toward the
label-similarity matrix S. `loss_hash` (which already supervises the full
36-bit code against S) then receives a much weaker gradient on C_0 because
C_0 is already in agreement -- so `loss_hash`'s gradient concentrates on the
five local slots. C_1 and C_4 absorb the brunt and partially collapse.

Same "alignment-style intervention → codebook utilisation degradation →
mAP regression" pattern as R1–R4 and v15.

Conclusion for the immediate decision: **option (a) does not help**. The
two remaining real options are:
- keep `loss_global` as no-op cosmetic (current v6, since `lambda_global > 0`
  contributes only to the CSV log; no harm done), or
- option (b) — remove `lambda_global` / `_loss_global` from the codebase to
  clean up the inventory. Mathematically identical to current v6.

Result directory of v17 retained under `result/` for the record; it is NOT
moved to backup because the "STE rewire" itself is documented here.

---

---

---

---

---

---

---

## 2026-05-13 — v15 codebook orthogonality (per-image selected-codeword)

🔴 reverted — net regression on Flickr25k.

Motivation: post-hoc diagnostic on Flickr v6 (see "deep-dive analysis" entry
below) revealed mean off-diagonal codebook MI = 1.36 and 60% hash duplicate
rate -- 5 of 6 codebooks carry ~40% redundant information. Standard
multi-codebook VQ fix: enforce that the M=6 codewords selected for each
image span mutually orthogonal directions in R^D.

Implementation:
- New loss term in `loss_siglip2.py::_loss_codebook_ortho`:
  `mean over (m != m') of <q_m, q_m'>^2` where `q_m` is the L2-normalised,
  STE-wrapped, selected codeword for codebook m.
- Applied to `outputs["quantized_tokens"]` (STE) so the loss value reflects
  the actual selected codewords while the gradient straight-throughs into
  the encoder / router (EMA codebook itself is a non-grad buffer).
- New config arg `--lambda_codebook_ortho`. v15 used λ=0.01.
- Note: an earlier prototype applied the penalty to `semantic_visual_tokens`
  (pre-quantization routed tokens) instead. User clarified the intent was
  the **selected codewords**, and the prototype was replaced.

Result (Flickr25k):

| Metric         |   v6   |   v15  |    Δ     |
|----------------|-------:|-------:|---------:|
| mAP            | 0.7556 | 0.7219 | −0.034 ❌ |
| unique_codes   | 0.3958 | 0.3549 | −0.041   |
| duplicate_rate | 0.6042 | 0.6451 | +0.041   |
| dead_codes     | 0.0000 | 0.1771 | +0.177 ⚠️|
| pos→neg gap    | 2.8618 | 2.5043 | −0.36    |

Per-codebook perplexity collapsed on the 5 local slots:
```
           C_0   C_1   C_2   C_3   C_4   C_5
v6:        31.3  26.8  29.0  28.7  26.4  24.3   (mean 27.8)
v15:       30.1  10.0  13.3   7.3  13.6   6.3   (mean 13.4)
```

Mechanism — orthogonality penalty pulled all images toward a common set of
~13 "mutually-orthogonal anchor codewords" per local codebook. Within-image
orthogonality was achieved at the cost of between-image diversity. Same
"representation specialisation hurts retrieval" pattern as R1-R3.

Code path remains in the tree (`lambda_codebook_ortho` config + loss method)
but defaults to 0; superseded for the current line of work.

---

---

---

---

---

---

---

## 2026-05-13 — v6 Flickr25k deep-dive diagnostic

🟢 reference (still-useful audit of where v6 wins / loses)

Standalone diagnostic on `result/.../v6_globadp.../extract_db.npz` +
`extract_query.npz`, no model needed. Output of `/tmp/analyze_v6_flickr.py`.

1. **Per-codebook drop-ablation** (set one codebook's 6 hash bits to constant,
   re-measure mAP):

| Codebook |   mAP  |  Δ mAP  |
|----------|-------:|--------:|
| C_0      | 0.7488 | +0.0000 |
| C_1      | 0.7343 | −0.0144 |
| C_2      | 0.7455 | −0.0033 |
| C_3      | 0.7389 | −0.0099 |
| C_4      | 0.7351 | −0.0137 |
| C_5      | 0.7460 | −0.0027 |

C_0 (global) contributes **zero** to retrieval distance ranking. The win
from `c_global_source=siglip2_global` was therefore a TRAIN-time effect
(stable anchor for the other codebooks) not a retrieval-time contribution.

2. **Pairwise codebook mutual information** (db, 23000 samples): mean
   off-diagonal = 1.36, mean diagonal = 3.32 → ratio 0.41. C_0 is roughly
   independent of locals (off-diag 0.5-0.6) but C_1–C_5 are heavily
   correlated with each other (1.4–2.1).

3. **Pos / neg Hamming distance**: pos mean 15.56 ± 4.94, neg mean 18.84 ±
   3.80. 22.75% of positive pairs land beyond the median negative — direct
   long-tail mAP loss.

4. **Hash collision**: 9,104 unique hashes for 23,000 samples (duplicate
   rate 60.4%). Worst single-hash cluster: 180 samples. Per-codebook
   codeword usage is full (32/32 used in every codebook); the collisions
   come from the high codebook-pair MI, not under-trained codebooks.

5. **Per-class P@10**: best classes (c_0, c_22, c_5) hit ~0.78; worst
   (c_7, c_20, c_2) drop to 0.04–0.17 — rare-class retrieval is essentially
   random. mean per-class P@10 = 0.4175, vs raw sample-mean P@10 ≈ 0.69 —
   the mAP we report is dominated by common classes.

Headline lesson: the diagnosed bottleneck for v6 on Flickr is
**codebook-pair MI 1.36 → high hash collision → rare-class P@10 collapse**,
not codebook underutilisation.

---

---

---

---

---

---

---

## 2026-05-13 — Prompt V3 (Option D, scene-aware **caption-style sentences**) [running]

🟡 in-progress

Motivation: inspecting V2 outputs confirmed that most slots produce bare
noun phrases or short labels (`C_primary_object: "boat"`,
`C_activity_or_relation: "fishing"`, `C_scene_type: "lake at dusk"`).
SigLIP2's text encoder is contrastively trained on natural captions, not
bare nouns, so sparse-label slots feed it the kind of input it under-utilises
at inference -- reducing how discriminative each slot's embedding is.

V3 (Option D) **keeps V2's six axes unchanged** (so PART_ORDER and the
auto-detect schema in `extract_codebook_texts` are unaffected) but rewrites
the per-axis instructions to require a short caption-style sentence
(~15-25 words) per slot, with an explicit "do not paraphrase C_global across
the other slots" rule. The expectation is **denser, captionish embeddings →
higher per-slot discriminability**, while still preserving V2's
multi-object friendliness.

Implementation:
- New constant `_PROMPT_V3` in `dna_utils/vlm_qwen25_descriptions.py`
  (V3 reuses `CODEBOOK_KEYS_V2` -- same keys, different instructions).
- `preprocess_qwen_codebook_texts.py --prompt_version v3` added; auto-detect
  schema in the extractor already handles V3 because the key tuple matches V2.
- Cache files separated: `*_qwen_v3.jsonl` + `*_siglip2_v3/`.
- Generic launcher at `/tmp/run_prompt_pipeline.sh` parameterises the prompt
  version so future variants (v4, …) reuse the same harness.

Background pipelines launched 2026-05-13 (GPU 0: Flickr 5K, GPU 2: MSCOCO 10K).
Results table to be filled in once both runs complete.

| Dataset    | V1 prompt | V2 prompt | V3 prompt |
|------------|----------:|----------:|----------:|
| Flickr25k  |    0.7556 |    0.7457 |       *(running)* |
| MSCOCO     |    0.5243 |    0.5385 |       *(running)* |

---

---

---

---

---

---

---

## 2026-05-13 — Prompt V2 (Option C, scene-aware) ablation

🟢 active

Hypothesis: V1's head/body/limb decomposition is biologically biased and
breaks down on multi-object scenes (MSCOCO has dining/sports/urban scenes
with no clear primary object). Forcing the VLM to fill all 6 fields under
V1 produces near-paraphrases across C_global, C_head, C_body, C_limb
(confirmed by inspecting sample MSCOCO Qwen outputs).

V2 (Option C) keys, same positional order so PART_ORDER stays compatible:

| Slot | V1 (legacy)              | V2 (scene-aware)        |
|------|--------------------------|-------------------------|
|  0   | C_global                 | C_global                |
|  1   | C_head_or_main_part      | C_primary_object        |
|  2   | C_body_or_secondary_part | C_secondary_object      |
|  3   | C_limb_or_detail_part    | C_activity_or_relation  |
|  4   | C_color_texture          | C_color_texture         |
|  5   | C_background_null        | C_scene_type            |

Implementation:
- New prompt constant `_PROMPT_V2` + key tuple `CODEBOOK_KEYS_V2` in `dna_utils/vlm_qwen25_descriptions.py`.
- `preprocess_qwen_codebook_texts.py --prompt_version {v1,v2}` flag.
- Cache files separated: `*_qwen_v2.jsonl` + `*_siglip2_v2/`.
- `text_description_processor.extract_codebook_texts` auto-detects schema from which keys are present.

Results (mAP, v6 architecture, 60 epochs):

| Dataset    | V1 prompt | V2 prompt |    Δ     |
|------------|----------:|----------:|---------:|
| Flickr25k  |    0.7556 |    0.7457 | −0.0099 ❌ |
| MSCOCO     |    0.5243 |    0.5385 | +0.0142 ✅ |

Interpretation: prompt design is **dataset-dependent**.
- MSCOCO (multi-object scenes) benefits from scene-aware decomposition.
- Flickr25k (single-object photos: flowers, animals, landscapes) suffers
  because head/body/limb is the more natural decomposition for that
  content distribution. unique_code_ratio also marginally drops there.

No single prompt is universally optimal. For a paper claim, this is a
"prompt-as-hyperparameter" story rather than "V2 strictly dominates".

---

---

---

---

---

---

---

## 2026-05-13 — Codebase cleanup: removed deprecated / no-op loss terms

🟢 active — main tree now reflects the verified-active loss set only.

After v17 confirmed that activating `loss_global` regresses Flickr25k mAP
by 0.012, the user requested option (b): drop the deprecated term entirely
and also strip the four R1–R4 / v15 inactive loss terms that were still
emitting empty / no-op columns into `log.csv`. All trial-and-error history
for the removed terms is preserved in (a) this document and (b)
`backup/results_R_series/README.md`; nothing was removed before checking
documentation coverage.

Files touched (single 2026-05-13 cleanup commit conceptually):

- `loss_siglip2.py`:
  - Removed methods `_loss_global`, `_loss_align`, `_loss_siglip_align`,
    `_loss_codebook_ortho`.
  - Removed `lambda_global`, `lambda_align`, `align_temperature`,
    `lambda_siglip_align`, `siglip_align_temperature`,
    `lambda_wasserstein`, `lambda_codebook_ortho` from `__init__`.
  - Removed corresponding entries from the forward `total` sum and the
    returned dict.
  - Top-line docstring updated to enumerate only the active components.
- `config.py`:
  - Removed CLI args `--lambda_global`, `--lambda_align`,
    `--align_temperature`, `--lambda_siglip_align`,
    `--siglip_align_temperature`, `--lambda_wasserstein`,
    `--lambda_codebook_ortho`.
- `train_siglip2.py`:
  - Trimmed `loss_types` list so `log.csv` no longer carries
    `train_loss_global / val_loss_global / train_loss_align / ... /
     train_loss_codebook_ortho / ...` columns.

Active loss set after cleanup (CSV-tracked):

| Term            | λ     | Role |
|-----------------|------:|------|
| `loss_hash`     | 1.00  | continuous code pairwise sim ↔ label sim |
| `loss_hash_hard`| 0.50  | STE-quantised hard code, same supervision |
| `loss_vq`       | 0.25  | codebook commitment (z ↔ q) |
| `loss_quant`    | 0.05  | codon-side commitment |
| `loss_anchor`   | 0.05  | local codebooks track EMA text anchor |
| `loss_dna`      | 0.05  | codon entropy + base balance (aggregate) |
| `loss_bu`       | 0.02  | codebook usage balance + uncorrelated (aggregate) |

→ **7 top-level active terms**, 11 individually-tracked CSV columns
(`loss_dna` and `loss_bu` each expand into 2 inner components).

Smoke verified: `python -c "from loss_siglip2 import DNACodonHashLoss; ..."`
returns exactly the 12 expected keys and `loss.backward()` succeeds.

The R-series code (attention router class, `ot_cost` propagation) is left
untouched in `models/semantic_router.py` and `model_siglip2.py` since
those don't pollute `log.csv` and removing them is more invasive (router
class + config branch). Future cleanup possible but low-priority.

---

---

---

---

---

---

---

## 2026-05-13 — `loss_global` is a no-op under EMA codebook mode (cosmetic only)

🟡 inventory correction — `loss_global` was listed as λ=0.1 active in the
inventory below but a code trace + empirical bit-perfect parity confirms it
contributes ZERO gradient to any trainable parameter in EMA mode.

Investigation trigger: the user asked whether `loss_global` is necessary.
Launched **v16** = v6 with `--lambda_global 0.0` on Flickr25k to ablate. At
the first mid-eval checkpoint v16 produced **bit-perfect identical** metrics
to v6 (mAP=0.7383 / unique=0.5625 / dead=0.3698, all four decimal places
matching). That should not happen if `loss_global` contributed any gradient.

Root cause:
- `loss_siglip2.py` calls `self._loss_global(q[:, 0, :], S, mask)` where
  `q = outputs["quantized_tokens_raw"]`.
- `quantized_tokens_raw = self.codebooks[m_idx, indices]` (`model_siglip2.py`).
- In EMA mode `self.codebooks` is a `register_buffer`, so the indexed result
  has `requires_grad=False` and `grad_fn=None`.
- The loss value is correctly computed for logging, but `.backward()` on
  `loss_global` alone fails with "element 0 of tensors does not require grad",
  i.e. **no gradient flows to any trainable parameter**.

Verified by isolated smoke test: `loss_global.requires_grad == False`,
`loss_global.grad_fn is None`.

Implication for the loss inventory:
- Declared active terms (λ > 0):   8
- **Effectively active (carry gradient):  7**
- The 8th declared term (`loss_global`) is cosmetic only under EMA mode.
- If we ever switch the codebook to gradient mode (`--codebook_update gradient`)
  OR route `loss_global` through `quantized_tokens` (STE-wrapped) instead of
  `quantized_tokens_raw`, it would re-acquire a real gradient path. Both are
  one-line changes; not done here because the immediate question was "is
  `loss_global` necessary in its current form" — answer: no.

v16 result directory was deleted (training would have been mathematically
identical to v6; no new data to keep).

Recommended follow-ups (not done yet, recorded for future reference):
- Either re-wire `loss_global` to `quantized_tokens` so it actually trains C_0;
  or drop the `lambda_global` argument entirely so the codebase isn't
  misleading.

---

---

---

---

---

---

---

## 2026-05-13 — Active loss-term inventory (post-cleanup)

🟢 reference — supersedes the earlier "as of v15" inventory.

Final active set in `loss_siglip2.DNACodonHashLoss` after the codebase
cleanup. All previously-listed inactive / no-op terms (R1–R4 alignment
family, v15 `loss_codebook_ortho`, deprecated `loss_global`) have been
**removed from the source tree**, not just λ=0-gated. The archived
history of each removed term is in `docs/PROJECT_LOG.md` and
`backup/results_R_series/README.md`.

| # | Term            | λ    | Role |
|---|-----------------|-----:|------|
| 1 | `loss_hash`     | 1.00 | continuous code pairwise sim ↔ label sim (MSE) |
| 2 | `loss_hash_hard`| 0.50 | STE-quantised hard code, same supervision |
| 3 | `loss_vq`       | 0.25 | semantic_visual_tokens ↔ quantized_tokens commitment |
| 4 | `loss_quant`    | 0.05 | continuous code ↔ DNA hard code commitment |
| 5 | `loss_anchor`   | 0.05 | local codebook centroids track EMA text anchor |
| 6 | `loss_dna`      | 0.05 | aggregate: `loss_entropy + η · loss_base_balance` |
| 7 | `loss_bu`       | 0.02 | aggregate: `loss_cb_balance + ρ · loss_cb_uncorr` |

→ **7 top-level active terms.**

Internal expansion (CSV-tracked individual components):
```
loss = 1.0  · loss_hash
     + 0.5  · loss_hash_hard
     + 0.25 · loss_vq
     + 0.05 · loss_quant
     + 0.05 · loss_anchor
     + 0.05 · (loss_entropy + 1.0 · loss_base_balance)
     + 0.02 · (loss_cb_balance + 0.1 · loss_cb_uncorr)
```
→ 11 CSV columns (7 top-level − 2 aggregates + 4 inner).

Removed from the source tree (and from `log.csv`):
`loss_global`, `loss_align`, `loss_siglip_align`, `loss_wasserstein`,
`loss_codebook_ortho`.

---

---

---

---

---

---

---

## 2026-05-12 — MSCOCO cache corruption discovered & fixed

🟢 fixed

Symptom: MSCOCO v6 baseline (and v14) both showed identical mAP 0.36, unique_code_ratio 0.0002, dead_code 0.97. Initially attributed to R4 Wasserstein loss, but reproduced WITHOUT the loss.

Root cause: `dataset/MSCOCO/images` symlink was never created (`download.sh`'s `ln -s` step had been skipped). The SigLIP2 feature extractor processed 122,218 catalog entries but **could not decode 114,141 of them** → those rows in `visual_global.f16.npy` were left as zeros. The dataloader read zero features → no learning signal → permanent codebook collapse.

Fix:
1. Created the missing symlink: `dataset/MSCOCO/images -> /data/.../MSCOCO/images`
2. Moved corrupted cache aside as `cache/mscoco_siglip2.broken.<timestamp>/`
3. Re-extracted SigLIP2 features (~40 min, 122K images)
4. Re-trained v6 baseline → mAP **0.5243** (was 0.3528 with broken cache)

Side finding: `val2014` partition shows transient I/O errors on a subset of files. Doesn't fully break extraction but means a small slice of val images is unreliable. Not fixed; flagged for later disk health check.

**Practical implication**: any R-series MSCOCO conclusion before 2026-05-12 is unreliable because the underlying baseline was broken. Fresh MSCOCO results are still being collected.

---

---

---

---

---

---

---

## 2026-05-12 — Cross-dataset baseline comparison

🟢 reference

All baselines: 36-bit hash, SigLIP2-frozen backbone, identical 60-epoch training, batch 64, Adam lr=1e-4.

**Flickr25k setting1:**

| Method      |     mAP | Rank |
|-------------|--------:|:----:|
| HashNet     |  0.7800 | 🥇 |
| CSQ         |  0.7596 | 🥈 |
| OrthoHash   |  0.7565 |  3 |
| **Ours v6** |  0.7556 |  4 |
| DPSH        |  0.7092 |  5 |

**MSCOCO setting1** (final-epoch eval, freshly extracted cache):

| Method      |     mAP | P@1   | P@10  | P@100 |
|-------------|--------:|------:|------:|------:|
| OrthoHash   |  0.6243 | 0.853 | 0.848 | 0.840 |
| HashNet     |  0.6076 | 0.453 | 0.691 | 0.704 |
| **Ours v6** |  0.5243 | 0.703 | 0.690 | 0.672 |
| DPSH        |  0.4567 | 0.445 | 0.455 | 0.520 |

**CIFAR10 setting1** (single-label):

| Method      |     mAP | Rank |
|-------------|--------:|:----:|
| **Ours v6** |  0.5335 | 🥇 |
| OrthoHash   |  0.4901 | 🥈 |
| CSQ         |  0.4589 |  3 |
| HashNet     |  0.4358 |  4 |
| DPSH        |  0.1111 |  5 |

Observations:
- **Single-label CIFAR10**: ours is the top method, beating the next baseline (OrthoHash) by +4.3 mAP.
- **Multi-label Flickr25k**: competitive but behind HashNet by 2.4 mAP. v11 (Wasserstein) closes most of that gap.
- **Multi-label MSCOCO**: we are clearly behind. Long-tail ranking is the weak spot — our P@1 (0.703) is the second-best, but P@1000 drops to 0.632 while OrthoHash sustains 0.797. duplicate_rate 0.888 is the direct diagnostic.

Code path to add MSCOCO/NUSWIDE to baseline driver: `baseline/base_model.py::NUM_CLASS / MULTI_LABEL / DEFAULT_CACHE_DIR` patched 2026-05-12.

---

---

---

---

---

---

---

## 2026-05-12 — Per-codeword interpretability tooling

🟢 active

Script: `interpret_compositional_code.py`. Produces, per codebook m in {0..5}:
- **V1**: image grid (`codeword_grid_C{m}_<name>.png`) — rows = codewords, cols = top-N example DB images per codeword. Visual evidence for "codeword X of C_head is always faces"-style claims.
- **V3**: per-codeword Qwen-text word histogram (`codeword_qwen_words_C{m}_<name>.{json,txt}`) — auto-generated semantic label per codeword.

Output goes under `<run_dir>/interpret/`. Already generated for:
- Flickr v6: `result/260510+flickr25k_setting1_v6_globadp.../interpret/`
- Flickr v11: `result/260511+flickr25k_setting1_v11_wass005.../interpret/`

Comparison v6 vs v11 (codebook interpretability): v6's per-codeword Qwen word lists are more semantically clean (e.g., C_4 codeword 1 → "white, texture, smooth, fur" — clear furry-animal cluster). v11 has higher per-image P@1 but its codewords have more mixed semantics. Confirms the alignment-vs-retrieval trade-off.

---

---

---

---

---

---

---

## 2026-05-12 (background, ongoing) — NUS-WIDE cache build

🟡 in-progress

Background pipeline kicked off to build a permanent NUS-WIDE cache:
- SigLIP2 features for full union (195,834 images): **done 2026-05-12 16:07** (~41 min)
- Qwen V1 text for full train.txt (193,734 images): ~50 hours, **~10% complete** as of 2026-05-13 morning. GPU 1.

When the Qwen cache completes, NUS-WIDE will be ready for direct training
without further preprocessing.

---

---

---

---

---

---

---

## 2026-05-10 to 2026-05-12 — R-series alignment ablations (v7 → v14)

🟡 superseded — all archived to `backup/results_R_series/` with a per-version README.

Hypothesis tested: more semantically aligned routing → better retrieval.

| Tag | Modification (one line)                            | Flickr25k mAP | Δ vs v6 |
|-----|----------------------------------------------------|--------------:|--------:|
| v7  | **R1**: Sinkhorn → Attention router               |        0.6975 |  −0.058 ❌ |
| v8  | **R2**: per-image part-contrastive CE loss (λ=0.1) |       0.7189 |  −0.037 ❌ |
| v9  | **R3**: post-adapter SigLIP-style InfoNCE (λ=0.1)  |       0.7073 |  −0.048 ❌ |
| v10 | **R4**: Wasserstein OT cost (λ=0.10)               |       0.7494 |  −0.006 ❌ |
| v11 | R4 λ=0.05                                          |    **0.7624** | **+0.007** ✅ |
| v12 | R4 λ=0.03                                          |       0.7583 |  +0.003 ⚪ |
| v13 | R4 λ=0.07                                          |       0.7586 |  +0.003 ⚪ |
| v14 | R4 λ=0.05 on MSCOCO                                |       0.3528 |  see note |

Per-modification notes:
- **R1 (v7)**: attention router removed Sinkhorn's marginal-balance constraint. 5 local codebooks all collapsed (47–56% dead codes). Confirmed Sinkhorn balance is what was protecting codebook diversity.
- **R2 (v8)**: per-image part-vs-part contrastive CE between routed visual tokens and text part embeddings. Forced part specialisation conflicts with codebook utilisation; net regression.
- **R3 (v9)**: image-level InfoNCE between pooled visual and text features. Initial draft used pre-adapter cached features → **zero gradient** → bit-identical to v6. Fixed to post-adapter; still net regression.
- **R4 (v10–v13)**: added the entropic-OT cost `<π, cost>` from the router (already computed) as a loss term. λ=0.05 is the unique net-positive setting; inverted-U around it. Improves pos→neg distance gap (2.86 → 3.43) but increases full-hash duplicate rate.
- **v14 MSCOCO**: looked like catastrophic collapse (mAP 0.35) but the root cause was a corrupted MSCOCO feature cache, *not* the Wasserstein loss (see next entry). Comparison vs R4 baseline is therefore inconclusive.

Code paths for R1–R4 remain in the main tree but default to `lambda_*=0` / `router_type=sinkhorn`, so they are inert unless explicitly re-enabled. See `backup/results_R_series/README.md` for the full per-experiment context.

Aggregate lesson: **"more interpretable routing" did NOT translate to "better retrieval mAP"** for this architecture on these datasets. The four interventions were all retrieval-neutral-to-negative despite producing visibly cleaner routing / higher unique-code-ratio.

---

---

---

---

---

---

---

## 2026-05-08 — Baseline establishment (v1 → v6)

🟢 active baseline

Goal: pick a stable SigLIP2-based DNA hashing baseline before exploring
alignment-style improvements.

Important changes that mattered (only the inflection points; intermediate
hyper-parameter sweeps elided):

| Tag | Key change                                | CIFAR10 mAP | Flickr25k mAP |
|-----|-------------------------------------------|------------:|--------------:|
| v1  | initial SigLIP2 skeleton                  |      0.3719 |        0.7482 |
| v3  | CIFAR setting1 stabilised                 |      **0.6217** |       — |
| v4  | K=48 codebook size                        |      0.5789 |       — |
| v5  | K=48 + global adapter (`globadp`)         |      0.5929 |       — |
| **v6** | **K=32 + `c_global = SigLIP visual_global`** | 0.5335 | **0.7556** |

Key takeaway: pulling SigLIP2's pooled visual feature directly into
C_global (bypassing the routing path for the global slot) was the single
biggest gain on Flickr25k. CIFAR10 peaked at v3 — we accept the small
CIFAR regression at v6 because Flickr is the multi-label benchmark we're
prioritising.

v6 architecture details:
- 6 codebooks: C_0 = global (fed from `visual_global`), C_1..C_5 = routed local parts.
- Qwen2.5-VL produces 6 per-codebook text descriptions per image (V1 prompt: head/body/limb decomposition).
- Sinkhorn OT router with marginal balance: each part receives ~equal patch mass even when the image has no obvious content for that part.
- Codebook update: EMA with revive-on-dead.
- Final hash: 6 codebook indices → 18 DNA codons (3/codebook) → 36 bits.

---

---

---

---

---

---

---

## Infrastructure & repo hygiene

🟢 active

- All R-series result directories moved to `backup/results_R_series/` with
  per-experiment summary in `backup/results_R_series/README.md`.
- Empty / smoke result dirs left under `result/` (low priority cleanup).
- Slide-ready visualisation snapshots are mirrored under `slides_assets/`
  with `/`-replaced-by-`_` filenames so they can be uploaded into a
  presentation generator without context loss.

---

---

---

---

---

---

---

<!--
HOW TO UPDATE THIS DOCUMENT
- For every non-trivial change (new variant, new ablation, new dataset,
  new cache, new finding, reverted decision), add a `## YYYY-MM-DD — ...`
  section with the absolute date it lands. Do not edit older sections
  except to mark them 🟡 superseded / 🔴 reverted.
- Layout policy is reverse chronological (newest first) with "Current
  state" pinned at the top — see the preamble for the full rule.
  After adding/editing entries, run:
        python scripts/reorder_project_log.py
  to re-sort sections deterministically (idempotent; no-op if already
  in order).
- Keep the "Current state" snapshot in sync with the latest results.
- Numbers belong in tables; rationale belongs in 1–2 sentences below.
- Code paths or commit refs (when applicable) belong in fenced spans.
-->
