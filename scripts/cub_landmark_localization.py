#!/usr/bin/env python
"""Landmark localisation error on CUB-200 -- the grounding metric we lack.

WHAT IT ANSWERS. Codon decoding says a slot's code carries concept information;
the caption-swap counterfactual says a slot's attention depends causally on its
own caption. Neither says the attention lands in the RIGHT PLACE. CUB-200 is the
only dataset here with human part annotations, so it is the only place we can
check that directly -- and it is the metric ConceptHash (CVPRW'24) reports, so
the numbers are comparable to the closest prior work.

PROTOCOL (following TASN / ConceptHash Table 3).

1. Each slot's routing column over the 14x14 patch grid is read as a spatial
   distribution and reduced to one point, its mass-weighted centroid:

       (x_m, y_m) = sum_n  P[n, m] / sum_k P[k, m]  *  (x_n, y_n)

   in normalised [0, 1] image coordinates. The centroid, not the argmax, keeps
   the estimate stable on a coarse grid.

2. Discovered slots have no a-priori correspondence to human parts, so the
   mapping is FITTED, not assumed: a linear regressor goes from the 2M slot
   coordinates to each landmark's (x, y), fitted on train only.

3. Error is the L2 distance on the held-out split in normalised coordinates,
   reported in % of image size. Lower is better.

Because the test transform is a plain Resize((224, 224)) with no crop, dividing
the annotated pixel coordinates by the original width/height gives exactly the
same normalised frame as the patch grid -- no crop bookkeeping is needed.

WHAT IT DOES NOT SHOW. A fitted linear map means the metric scores the slot
ENSEMBLE, not "slot m is the beak". A model whose slots collectively span the
bird scores well even if no single slot is a nameable part. That is the same
caveat the prior work carries, and it is why ConceptHash still calls for manual
inspection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

SLOTS = ["global", "primary_object", "secondary_object",
         "activity_relation", "color_texture", "scene_type"]


def _load_parts(root: str):
    idx_to_rel, sizes = {}, {}
    with open(os.path.join(root, "images.txt")) as fh:
        for line in fh:
            i, rel = line.strip().split(" ", 1)
            idx_to_rel[int(i)] = rel
    n_parts = 15
    n_img = max(idx_to_rel)
    xy = np.zeros((n_img + 1, n_parts + 1, 2), dtype=np.float32)
    vis = np.zeros((n_img + 1, n_parts + 1), dtype=bool)
    with open(os.path.join(root, "parts", "part_locs.txt")) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 5:
                continue
            im, pt, x, y, v = int(p[0]), int(p[1]), float(p[2]), float(p[3]), int(p[4])
            xy[im, pt] = (x, y)
            vis[im, pt] = bool(v)
    return idx_to_rel, xy, vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--cub_root", default="dataset/CUB_200")
    ap.add_argument("--parts", type=int, nargs="*", default=[2, 9, 14],
                    help="part ids to report; default beak/left-wing/tail, the "
                         "three-landmark style ConceptHash reports")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from PIL import Image
    from regen_viz_routing import _parse_args_txt
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.visualization import _forward_text_routed

    args = _parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    model.load_state_dict(torch.load(os.path.join(a.result_dir, "model_state_dict.pth"),
                                     map_location="cpu", weights_only=False), strict=False)
    if a.epoch is not None:
        model.set_current_epoch(a.epoch)

    idx_to_rel, xy, vis = _load_parts(a.cub_root)
    rel_to_idx = {rel: i for i, rel in idx_to_rel.items()}
    img_root = os.path.join(a.cub_root, "images")

    tr, te, _ = load_dataset(
        getattr(args, "dataset_dir", "dataset"), args.dataset,
        setting=getattr(args, "setting", "setting1"),
        train_transform=None, test_transform=None,
        load_train=True, load_database=False, load_test=True, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None))

    def centroids(ds, n_take):
        """-> [N, M, 2] slot centroids and the dataset row order used."""
        out, rows = [], []
        with torch.no_grad():
            for i0 in range(0, min(n_take, len(ds)), 32):
                sel = list(range(i0, min(i0 + 32, min(n_take, len(ds)))))
                samples = [ds[i] for i in sel]
                batch = {}
                for k in samples[0]:
                    v = [s[k] for s in samples]
                    if torch.is_tensor(v[0]):
                        batch[k] = torch.stack(v)
                    elif isinstance(v[0], np.ndarray):
                        batch[k] = torch.from_numpy(np.stack(v))
                    else:
                        batch[k] = v
                P = _forward_text_routed(model, batch, a.device)["routing_matrix"].float()
                B, N, M = P.shape
                side = int(round(N ** 0.5))
                if side * side != N:
                    raise SystemExit(f"patch count {N} is not square; cannot map to a grid")
                # patch centres in normalised [0, 1] coordinates
                gy, gx = torch.meshgrid(
                    (torch.arange(side, device=P.device) + 0.5) / side,
                    (torch.arange(side, device=P.device) + 0.5) / side,
                    indexing="ij")
                grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)   # [N, 2]
                w = P / P.sum(dim=1, keepdim=True).clamp_min(1e-12)        # [B, N, M]
                c = torch.einsum("bnm,nd->bmd", w, grid)                   # [B, M, 2]
                out.append(c.cpu().numpy()); rows.extend(sel)
        return np.concatenate(out), rows

    def targets(ds, rows):
        """Normalised landmark coords + visibility for the same rows."""
        Y, V = [], []
        for r in rows:
            path = ds.img_paths[r] if hasattr(ds, "img_paths") else ds.images[r]
            rel = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
            im_id = rel_to_idx.get(rel)
            if im_id is None:
                Y.append(None); V.append(None); continue
            with Image.open(os.path.join(img_root, rel)) as img:
                W, H = img.size
            Y.append(np.stack([xy[im_id, p] / np.array([W, H], np.float32) for p in a.parts]))
            V.append(np.array([vis[im_id, p] for p in a.parts]))
        return Y, V

    C_tr, rows_tr = centroids(tr, a.n_train)
    C_te, rows_te = centroids(te, a.n_test)
    Y_tr, V_tr = targets(tr, rows_tr)
    Y_te, V_te = targets(te, rows_te)

    M = C_tr.shape[1]
    Xtr = np.concatenate([C_tr.reshape(len(C_tr), -1),
                          np.ones((len(C_tr), 1), np.float32)], 1)
    Xte = np.concatenate([C_te.reshape(len(C_te), -1),
                          np.ones((len(C_te), 1), np.float32)], 1)

    print(f"CUB-200 landmark localisation   dir={os.path.basename(a.result_dir)}")
    print(f"  slots={M}  train={len(C_tr)}  test={len(C_te)}  parts={a.parts}")
    print(f"  {'part':16s}{'n_test':>8s}{'err %':>9s}{'centroid-mean %':>17s}")
    part_names = {}
    with open(os.path.join(a.cub_root, "parts", "parts.txt")) as fh:
        for line in fh:
            i, nm = line.strip().split(" ", 1)
            part_names[int(i)] = nm
    rows_out = {}
    for j, pid in enumerate(a.parts):
        mtr = np.array([v is not None and V_tr[k][j] for k, v in enumerate(V_tr)])
        mte = np.array([v is not None and V_te[k][j] for k, v in enumerate(V_te)])
        if mtr.sum() < 50 or mte.sum() < 50:
            print(f"  {part_names.get(pid, pid):16s}  too few visible, skipped")
            continue
        ytr = np.stack([Y_tr[k][j] for k in range(len(Y_tr)) if mtr[k]])
        yte = np.stack([Y_te[k][j] for k in range(len(Y_te)) if mte[k]])
        W_, *_ = np.linalg.lstsq(Xtr[mtr], ytr, rcond=None)
        pred = Xte[mte] @ W_
        err = float(np.mean(np.linalg.norm(pred - yte, axis=1)) * 100)
        # trivial control: predict the training mean position for every image
        base = float(np.mean(np.linalg.norm(ytr.mean(0)[None, :] - yte, axis=1)) * 100)
        print(f"  {part_names.get(pid, pid):16s}{int(mte.sum()):8d}{err:9.2f}{base:17.2f}")
        rows_out[part_names.get(pid, str(pid))] = {
            "n_test_visible": int(mte.sum()),
            "localisation_error_pct": round(err, 4),
            "constant_mean_baseline_pct": round(base, 4),
        }
    print("\n  centroid-mean % = predict the train-mean location for every image;")
    print("  the slot centroids must beat it or they carry no spatial information.")

    out = {"dataset": "CUB_200", "dir": a.result_dir, "slots": int(M),
           "n_train": len(C_tr), "n_test": len(C_te), "parts": a.parts,
           "per_part": rows_out}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
