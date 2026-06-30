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
