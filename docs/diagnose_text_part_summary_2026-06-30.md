# Diagnostic: text_part cos sim across slots (CUB champion)

## Source
- model: wass 0.15 + best-ckpt champion (mAP 0.1618)
- cache: cub200_clip_v6bplus (whole-image, 196 tokens)
- N=5994 train captions (filtered has_text)

## Off-diag local mean cos sim (m=1..5)

| Stage | mean | max | min |
|---|---|---|---|
| RAW (CLIP text encoder) | 0.7070 | 0.7343 | 0.6717 |
| WHITENED (partial γ=0.25 ZCA) | 0.4826 | 0.5227 | 0.4353 |
| **ADAPTED (post per-slot text_adapter) ★ ROUTING TARGET** | **0.1336** | **0.1792** | **0.1071** |

## Verdict on earlier "cos 0.88 bottleneck" hypothesis

REFUTED. The per-slot text adapter substantially orthogonalizes text_part: from 0.71 → 0.13 mean cos sim. The routing target IS slot-distinct in feature space.

## Implication for routingText λ=0.10 failure

The failure CANNOT be blamed on text similarity bottleneck. New candidate explanations:

1. **Cross-modal subspace disconnect**: visual_adapter and text_adapter trained separately; no per-patch cross-modal alignment loss. cos(text_part[m], visual_token[p]) may not carry meaningful signal even with orthogonal text.

2. **KL direction**: forward KL(target || routing) may be too strict if target is very sharp.

3. **Loss dynamics conflict**: CIBHash NtXent (λ=1.0) pulls routing toward class-discriminative; routingText pulls toward text-anchored. Competing.

## Next diagnostic candidates

- (a) Distribution of cos(text_part_adapted[m], visual_token[p]) across slots
- (b) Direct comparison of L_routing_text target vs actual OT routing on same image
- (c) Reverse KL: KL(routing || target) instead of forward

---

# Diagnostic (a): cross-modal subspace disconnect

## Source
Same champion run. Forced `model.train()` (no_grad) to compute text path which is skipped at eval. N=224 images, M=6 slots, P=196 patches (whole-image cache).

## Per-(image, slot) sim distribution over patches

| metric | cos(text[m], visual[p]) | cos(codebook[m, k_m*], visual[p]) |
|---|---|---|
| mean | 0.161 | 0.331 |
| std | 0.142 | 0.187 |
| max | 0.433 | 0.664 |
| min | -0.218 | -0.152 |

## Cross-slot differential per (image, patch)

| metric | text-target | codebook-target |
|---|---|---|
| cross-slot std (over m) at fixed (n, p) | 0.142 | 0.174 |
| cross-slot Pearson corr (over m-pairs, ~100 imgs) | **0.278** | 0.542 |
| per-image distinct top-1 patches across M=6 | **5.70 / 6** | 4.82 / 6 |

## Verdict

REFUTED — cross-modal subspace is NOT disconnected. text-vs-visual signal is actually MORE slot-discriminative than codebook-vs-visual:
- text-target produces near-distinct top-1 patches across slots (5.7 / 6 max)
- text-target has lower cross-slot correlation (0.28 → healthy zone)

## True bottleneck candidate (new)

Signal MAGNITUDE: text max sim = 0.43 (vs codebook 0.66). softmax(sim/τ=0.1) target is too flat for text:
- text peak-vs-mean ratio after softmax ≈ exp((0.43 − 0.16) / 0.1) ≈ 14
- codebook peak-vs-mean ratio ≈ exp((0.66 − 0.33) / 0.1) ≈ 27

Flat target → weak supervision → routing under-driven → codebooks collapse (NMI ↑).

## Next experiment candidates

(C1) routingText λ=0.10 + τ=0.05 (sharpen target) — fastest test of magnitude hypothesis
(C2) sim centering before softmax: sim_c = sim − sim.mean(-1)
(C3) per-image normalization: scaled = sim / sim.std(-1)
(C4) hybrid text+codebook target
