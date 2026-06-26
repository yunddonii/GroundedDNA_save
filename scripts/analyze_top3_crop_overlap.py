"""Measure top-K=3 FAIRrank crop overlap via actual CLIP ranking."""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, '/home/yschoi/GroundedDNA')


def gen_crop_params(W, H, scale_min, scale_max, rng):
    scale = rng.uniform(scale_min, scale_max)
    h = max(1, int(round(np.sqrt(scale) * H)))
    w = max(1, int(round(np.sqrt(scale) * W)))
    top = rng.integers(0, max(1, H - h + 1))
    left = rng.integers(0, max(1, W - w + 1))
    return int(top), int(left), int(h), int(w)


def box_iou(box_a, box_b):
    t1, l1, h1, w1 = box_a; b1, r1 = t1 + h1, l1 + w1
    t2, l2, h2, w2 = box_b; b2, r2 = t2 + h2, l2 + w2
    inter_t = max(t1, t2); inter_l = max(l1, l2)
    inter_b = min(b1, b2); inter_r = min(r1, r2)
    if inter_b <= inter_t or inter_r <= inter_l: return 0.0
    inter = (inter_b - inter_t) * (inter_r - inter_l)
    union = h1*w1 + h2*w2 - inter
    return inter / union if union > 0 else 0.0


def union_area(boxes, W, H):
    mask = np.zeros((H, W), dtype=bool)
    for t, l, h, w in boxes:
        mask[t:t+h, l:l+w] = True
    return mask.sum() / (W * H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor_dir", required=True)
    ap.add_argument("--pathlist_root", required=True)
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--scale_min", type=float, default=0.25)
    ap.add_argument("--scale_max", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--image_size", type=int, default=224)
    args = ap.parse_args()

    image_ids = json.load(open(os.path.join(args.donor_dir, "image_ids.json")))
    vg = np.load(os.path.join(args.donor_dir, "visual_global.f16.npy"), mmap_mode="r")
    N = len(image_ids)
    view_seed = int(args.seed) * 1009
    rng_global = np.random.default_rng(view_seed)
    per_image_seeds = rng_global.integers(0, 2**31 - 1, size=N)

    # CLIP setup
    from transformers import CLIPVisionModelWithProjection
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").to(args.device).eval()
    tfm = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])

    rng_sample = np.random.default_rng(0)
    sampled_idx = rng_sample.choice(N, size=args.num_samples, replace=False)

    ious_top3, ious_all8, cov_top3, cov_all8 = [], [], [], []
    per_crop_areas = []
    overlap_top3_with_each = []  # pairwise IoU among top-3
    
    with torch.no_grad():
        for i in sampled_idx:
            relpath = image_ids[i]
            img_path = None
            for prefix in ["", "images/"]:
                cand = os.path.join(args.pathlist_root, prefix, relpath)
                if os.path.exists(cand):
                    img_path = cand; break
            if img_path is None: continue
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            W, H = pil.size
            img_rng = np.random.default_rng(int(per_image_seeds[i]))
            crops_param = [gen_crop_params(W, H, args.scale_min, args.scale_max, img_rng) for _ in range(args.L)]

            # Render each crop and forward through CLIP
            crop_tensors = []
            for t, l, h, w in crops_param:
                c = pil.crop((l, t, l+w, t+h))
                crop_tensors.append(tfm(c))
            batch = torch.stack(crop_tensors).to(args.device)
            out = model(pixel_values=batch).image_embeds  # [L, D=512]
            crop_emb = F.normalize(out, dim=-1)
            anchor = torch.from_numpy(np.asarray(vg[i], dtype=np.float32)).to(args.device)
            anchor = F.normalize(anchor, dim=-1)
            sim = (crop_emb * anchor.unsqueeze(0)).sum(-1)  # [L]
            topk_idx = sim.topk(args.K).indices.cpu().tolist()
            top3_boxes = [crops_param[j] for j in topk_idx]

            # Compute IoU among top-3
            pair_ious = [box_iou(top3_boxes[a], top3_boxes[b]) for a in range(args.K) for b in range(a+1, args.K)]
            ious_top3.append(np.mean(pair_ious))
            overlap_top3_with_each.extend(pair_ious)
            cov_top3.append(union_area(top3_boxes, W, H))

            # All-L IoU
            all_iou_pairs = [box_iou(crops_param[a], crops_param[b])
                             for a in range(args.L) for b in range(a+1, args.L)]
            ious_all8.append(np.mean(all_iou_pairs))
            cov_all8.append(union_area(crops_param, W, H))
            per_crop_areas.append(np.mean([h*w/(W*H) for _, _, h, w in top3_boxes]))

    print(f"\n=== FAIRrank top-K={args.K} crop overlap analysis ({len(ious_top3)} CUB samples) ===")
    print(f"crop scale: ({args.scale_min}, {args.scale_max}) -> top-3 mean per-crop area = {np.mean(per_crop_areas):.3f} of image")
    print(f"\nAmong top-K={args.K} selected crops (image_global anchor ranking):")
    print(f"  Mean pairwise IoU:    {np.mean(ious_top3):.3f}  (median {np.median(ious_top3):.3f})")
    print(f"  Mean pairwise IoU distribution: min={np.min(overlap_top3_with_each):.3f}, max={np.max(overlap_top3_with_each):.3f}, std={np.std(overlap_top3_with_each):.3f}")
    print(f"  Mean union coverage:  {np.mean(cov_top3):.3f} of image area")
    print(f"  Median union coverage: {np.median(cov_top3):.3f}")
    print(f"\nComparison with all L={args.L} random crops (no selection):")
    print(f"  Mean pairwise IoU:    {np.mean(ious_all8):.3f}")
    print(f"  Mean union coverage:  {np.mean(cov_all8):.3f}")
    print(f"\nInterpretation:")
    print(f"  Top-3 IoU {np.mean(ious_top3):.3f} vs random-8 IoU {np.mean(ious_all8):.3f}: ", end='')
    if np.mean(ious_top3) > np.mean(ious_all8): print(f"top-3 are MORE overlapped (+{np.mean(ious_top3)-np.mean(ious_all8):.3f})")
    else: print(f"top-3 are LESS overlapped (−{np.mean(ious_all8)-np.mean(ious_top3):.3f})")
    print(f"  Top-3 covers {np.mean(cov_top3)*100:.0f}% of image area")
    avg_top3_area = np.mean(per_crop_areas)
    if_no_overlap = min(1.0, 3 * avg_top3_area)
    print(f"  If non-overlapping: 3 × {avg_top3_area:.3f} = {if_no_overlap:.3f}")
    print(f"  Actual overlap fraction = {1.0 - np.mean(cov_top3)/if_no_overlap:.3f}")


if __name__ == "__main__":
    main()
