#!/usr/bin/env python
"""Network Dissection for semantic slots, against MS-COCO instance masks.

WHY THIS AND NOT ANOTHER PROBE. Every interpretability number we currently
report is measured against text we generated ourselves: the codon-decoding
dictionary is built from Qwen captions, and three training losses explicitly pull
the code toward those captions. A reviewer can call that circular, and the axis
names (`primary_object`, `activity_relation`) are simply the words we put in the
prompt -- nothing so far shows a slot attends to the thing the word denotes.
Network Dissection (Bau, Zhou, Khosla, Oliva, Torralba, CVPR 2017) exists for
exactly this: it scores a unit against DENSE HUMAN ANNOTATION that the model
never saw, so the label cannot have leaked from our own pipeline.

PROTOCOL, following the paper.

1. Upsample slot m's routing column from the 14x14 grid to image resolution.
2. Binarise it at the top-quantile threshold T_m determined over the whole
   probe set, with P(A_m > T_m) = 0.005 in the paper. We expose --quantile
   because our maps are transport mass over 5 slots, not ReLU activations over
   hundreds of channels: at 0.005 a slot that spreads its mass evenly would be
   thresholded into a few scattered patches and score ~0 for reasons that have
   nothing to do with grounding. The paper's value is reported alongside.
3. IoU(m, c) = |M_m AND L_c| / |M_m OR L_c| summed over the probe set, where
   L_c is the union of instance masks of COCO category c. The paper calls a
   unit a detector of c when IoU > 0.04; we report the full ranking as well,
   because with 5 slots and 80 categories the interesting question is WHICH
   category each slot ranks first, not how many clear the bar.

WHAT A NULL RESULT MEANS. If every slot's best category is the same one, or if
no slot clears the threshold, the axes are not grounded in the human sense --
which is a finding, not a failure of the script. The scores are also bounded
above by how well the frozen CLIP patch grid separates anything at 14x14, so a
random-routing control is computed for comparison rather than assuming 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

SLOTS = ["primary_object", "secondary_object", "activity_relation", "color_texture"]


def _load_coco(ann_dir: str):
    """-> {basename: (W, H, [(cat_id, [polys])])}, {cat_id: name}"""
    import itertools
    imgs, cats, anns = {}, {}, defaultdict(list)
    for split in ("train2014", "val2014"):
        p = os.path.join(ann_dir, f"instances_{split}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for c in d["categories"]:
            cats[c["id"]] = c["name"]
        idmap = {}
        for im in d["images"]:
            idmap[im["id"]] = im["file_name"]
            imgs[im["file_name"]] = (im["width"], im["height"])
        for a in d["annotations"]:
            if a.get("iscrowd"):
                continue
            fn = idmap.get(a["image_id"])
            if fn is not None:
                anns[fn].append((a["category_id"], a.get("segmentation"), a.get("bbox")))
    return imgs, cats, anns


def _mask_from_bbox(shape, bbox):
    """Rasterise one bbox. Used when pycocotools is unavailable -- coarser than
    the polygon mask, and the script says so rather than silently degrading."""
    H, W = shape
    m = np.zeros((H, W), dtype=bool)
    x, y, w, h = bbox
    x0, y0 = int(max(0, round(x))), int(max(0, round(y)))
    x1, y1 = int(min(W, round(x + w))), int(min(H, round(y + h)))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--ann_dir", default="dataset/MSCOCO/annotations")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epoch", type=int, required=True,
                    help="epoch the checkpoint was TRAINED to; sets the annealed "
                         "Sinkhorn epsilon. Passing the nominal total puts the "
                         "model in a regime it never trained in.")
    ap.add_argument("--quantile", type=float, default=0.005,
                    help="P(A > T) as in the paper; 0.005 = top 0.5 percent")
    ap.add_argument("--iou_threshold", type=float, default=0.04)
    ap.add_argument("--side", type=int, default=112,
                    help="resolution the masks and maps are compared at")
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    try:
        from pycocotools import mask as coco_mask
        HAVE_COCO_TOOLS = True
    except Exception:
        coco_mask = None
        HAVE_COCO_TOOLS = False

    from regen_viz_routing import _parse_args_txt
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.visualization import _forward_text_routed

    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    model.load_state_dict(torch.load(os.path.join(a.result_dir, "model_state_dict.pth"),
                                     map_location="cpu", weights_only=False), strict=False)
    model.set_current_epoch(a.epoch)

    _, cats, anns = _load_coco(a.ann_dir)
    if not anns:
        raise SystemExit(f"no instances_*.json under {a.ann_dir}")
    print(f"COCO annotations: {len(anns)} images, {len(cats)} categories, "
          f"polygon masks={'yes' if HAVE_COCO_TOOLS else 'NO -- bbox fallback'}")

    tr, te, _ = load_dataset(
        getattr(args, "dataset_dir", "dataset"), args.dataset,
        setting=getattr(args, "setting", "setting1"),
        train_transform=None, test_transform=None,
        load_train=True, load_database=False, load_test=False, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))
    rng = np.random.default_rng(a.sample_seed)
    order = rng.permutation(len(tr))[:min(a.n, len(tr))].tolist()

    S = a.side
    # pass 1 -- collect routing maps and the matching category masks
    maps, masks, kept = [], [], []
    with torch.no_grad():
        for i0 in range(0, len(order), 32):
            sel = order[i0:i0 + 32]
            samples = [tr[i] for i in sel]
            batch = {}
            for k in samples[0]:
                v = [s[k] for s in samples]
                if torch.is_tensor(v[0]):
                    batch[k] = torch.stack(v).to(a.device)
                elif isinstance(v[0], np.ndarray):
                    batch[k] = torch.from_numpy(np.stack(v)).to(a.device)
                elif isinstance(v[0], (bool, np.bool_, int, float)):
                    batch[k] = torch.as_tensor(v).to(a.device)
                else:
                    batch[k] = v
            P = _forward_text_routed(model, batch, a.device)["local_routing_matrix"].float()
            B, N, M = P.shape
            side = int(round(N ** 0.5))
            up = F.interpolate(P.permute(0, 2, 1).reshape(B, M, side, side),
                               size=(S, S), mode="bilinear", align_corners=False)
            for b, r in enumerate(sel):
                path = tr.img_paths[r] if hasattr(tr, "img_paths") else tr.images[r]
                fn = os.path.basename(str(path))
                if fn not in anns:
                    continue
                per_cat = {}
                for cid, seg, bbox in anns[fn]:
                    if HAVE_COCO_TOOLS and seg is not None and not isinstance(seg, dict):
                        # polygons are in ORIGINAL image coords; rasterise there
                        # then resize, so the mask and the map share a frame.
                        from PIL import Image
                        with Image.open(str(path)) as im:
                            W0, H0 = im.size
                        rle = coco_mask.frPyObjects(seg, H0, W0)
                        mm = coco_mask.decode(coco_mask.merge(rle)).astype(bool)
                        mm = np.array(
                            Image.fromarray(mm.astype(np.uint8) * 255).resize((S, S)),
                            dtype=np.uint8) > 127
                    elif bbox is not None:
                        from PIL import Image
                        with Image.open(str(path)) as im:
                            W0, H0 = im.size
                        mm = _mask_from_bbox((H0, W0), bbox)
                        mm = np.array(
                            Image.fromarray(mm.astype(np.uint8) * 255).resize((S, S)),
                            dtype=np.uint8) > 127
                    else:
                        continue
                    per_cat[cid] = per_cat.get(cid, np.zeros((S, S), bool)) | mm
                if not per_cat:
                    continue
                maps.append(up[b].cpu().numpy())          # [M, S, S]
                masks.append(per_cat)
                kept.append(fn)

    if not maps:
        raise SystemExit("no probe image matched the COCO annotations")
    A = np.stack(maps)                                     # [P, M, S, S]
    Pn, M = A.shape[0], A.shape[1]
    print(f"probe set: {Pn} images with annotations, {M} local slots, {S}x{S}\n")

    # pass 2 -- per-slot top-quantile threshold over the WHOLE probe set
    T = np.quantile(A.reshape(Pn, M, -1).transpose(1, 0, 2).reshape(M, -1),
                    1.0 - a.quantile, axis=1)              # [M]
    Mk = A >= T[None, :, None, None]                       # [P, M, S, S]

    # F12: fix the category universe BEFORE accumulating, and charge every
    # probe image. Iterating `per_cat.items()` skipped images where a category
    # is absent, so their activation area -- pure false positive -- never
    # entered that category's union and the IoU was systematically inflated.
    from dna_utils.dissection_iou import accumulate_iou, category_universe
    universe = category_universe(masks)
    iou = accumulate_iou(Mk, masks, universe)

    # control -- the same threshold applied to a shuffled routing assignment,
    # which keeps each map's shape but destroys which slot it belongs to.
    perm = np.stack([rng.permutation(M) for _ in range(Pn)])
    Ar = np.take_along_axis(A, perm[:, :, None, None], axis=1)
    Tr = np.quantile(Ar.reshape(Pn, M, -1).transpose(1, 0, 2).reshape(M, -1),
                     1.0 - a.quantile, axis=1)
    Mr = Ar >= Tr[None, :, None, None]
    # Same universe and same accumulation as the real branch, or the control
    # cannot bound the real score: the old code inflated both in the same
    # direction, which is why the control never exposed the defect.
    iou_r = accumulate_iou(Mr, masks, universe)

    print(f"top-quantile = {a.quantile}  (paper: 0.005)   "
          f"detector threshold IoU > {a.iou_threshold} (paper: 0.04)\n")
    print(f"  {'slot':20s}{'best category':>18s}{'IoU':>9s}{'shuffled':>10s}"
          f"{'#cat > thr':>12s}   runner-up")
    rows = {}
    for m in range(M):
        rank = sorted(((float(iou[c][m]), c) for c in iou), reverse=True)
        best_v, best_c = rank[0]
        second = f"{cats.get(rank[1][1], rank[1][1])} {rank[1][0]:.4f}" if len(rank) > 1 else "-"
        n_det = sum(1 for c in iou if iou[c][m] > a.iou_threshold)
        name = SLOTS[m] if m < len(SLOTS) else str(m)
        print(f"  {name:20s}{cats.get(best_c, best_c):>18s}{best_v:9.4f}"
              f"{float(iou_r[best_c][m]):10.4f}{n_det:12d}   {second}")
        rows[name] = {"best_category": cats.get(best_c, str(best_c)),
                      "best_iou": round(best_v, 5),
                      "shuffled_iou_same_category": round(float(iou_r[best_c][m]), 5),
                      "n_categories_above_threshold": int(n_det),
                      "top5": [(cats.get(c, str(c)), round(v, 5)) for v, c in rank[:5]]}

    distinct = len({rows[k]["best_category"] for k in rows})
    print(f"\n  distinct best categories across slots: {distinct} / {M}")
    print("  If this is 1, the slots all detect the same thing and the axis names")
    print("  are not grounded in the human-annotation sense.")

    out = {"dataset": args.dataset, "dir": a.result_dir, "epoch": a.epoch,
           "probe_images": Pn, "side": S, "quantile": a.quantile,
           "iou_threshold": a.iou_threshold,
           "polygon_masks": bool(HAVE_COCO_TOOLS),
           "distinct_best_categories": distinct, "slots": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
