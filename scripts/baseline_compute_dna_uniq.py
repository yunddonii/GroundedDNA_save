"""Compute DNA-uniq for baseline checkpoints on CUB/MSCOCO/Flickr DB split.

For each baseline (CIBHash, CIMON, MLS3RDUH), loads the saved encoder checkpoint
+ the corresponding cached visual_global features, runs encoder, sign-binarizes
to 36-bit hash codes, and reports DNA-uniq (= unique 36-bit codes / N).

Output: prints summary table.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/yschoi/GroundedDNA")
from baseline.base_model import BackboneWithEncoder


CONFIGS = [
    # (method, dataset, ckpt_path, cache_dir, db_setting_txt)
    ("cibhash", "CUB_200", "params_baseline/260618/cibhash_cub200_unsup60/epoch_059.pth",
     "cache/cub200_clip", "dataset/CUB_200/setting1/database.txt"),
    ("cimon", "CUB_200", "params_baseline/260618/cimon_cub200_unsup60/epoch_059.pth",
     "cache/cub200_clip", "dataset/CUB_200/setting1/database.txt"),
    ("mls3rduh", "CUB_200", "params_baseline/260618/mls3rduh_cub200_unsup60/epoch_059.pth",
     "cache/cub200_clip", "dataset/CUB_200/setting1/database.txt"),
    ("cibhash", "MSCOCO", "params_baseline/260529/cibhash_mscoco_clip_unsup60/epoch_059.pth",
     "cache/mscoco_clip_v5b", "dataset/MSCOCO/setting1/database.txt"),
    ("cimon", "MSCOCO", "params_baseline/260529/cimon_mscoco_clip_unsup60/epoch_059.pth",
     "cache/mscoco_clip_v5b", "dataset/MSCOCO/setting1/database.txt"),
    ("mls3rduh", "MSCOCO", "params_baseline/260529/mls3rduh_mscoco_clip_unsup60/epoch_059.pth",
     "cache/mscoco_clip_v5b", "dataset/MSCOCO/setting1/database.txt"),
]


def load_db_indices(cache_dir: str, db_txt: str) -> np.ndarray:
    """Read database.txt -> match to image_ids.json -> return row indices in cache."""
    image_ids = json.load(open(os.path.join(cache_dir, "image_ids.json")))
    id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    db_rel = []
    with open(db_txt) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # database.txt may have "relpath label" or just "relpath"
            parts = line.split()
            db_rel.append(parts[0])
    # match - try suffix match if direct doesn't work
    indices = []
    for r in db_rel:
        if r in id_to_idx:
            indices.append(id_to_idx[r])
        else:
            # try suffix match
            matched = None
            for iid in image_ids:
                if iid.endswith(r) or r.endswith(iid):
                    matched = iid; break
            if matched is not None:
                indices.append(id_to_idx[matched])
            else:
                indices.append(-1)
    valid = sum(1 for i in indices if i >= 0)
    print(f"  matched {valid}/{len(db_rel)} DB entries")
    return np.array([i for i in indices if i >= 0])


def compute_dna_uniq(method: str, dataset: str, ckpt_path: str,
                     cache_dir: str, db_txt: str, bit: int = 36,
                     device: str = "cuda:0", batch_size: int = 512) -> dict:
    print(f"\n=== {method} on {dataset} ===")
    if not os.path.exists(ckpt_path):
        print(f"  ❌ checkpoint not found: {ckpt_path}")
        return None
    # Load cached visual_global
    vg = np.load(os.path.join(cache_dir, "visual_global.f16.npy"), mmap_mode="r")
    print(f"  visual_global: {vg.shape}")
    d_in = int(vg.shape[1])

    # Build model + load weights
    model = BackboneWithEncoder(d_in=d_in, bit=bit, hidden_nodes=None, batch_norm=False)
    info = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(info, dict) and "encoder_layers" in info:
        model.encoder_layers.load_state_dict(info["encoder_layers"])
    else:
        # fallback - try direct
        model.load_state_dict(info, strict=False)
    model.eval().to(device)

    # Filter to DB indices
    db_idx = load_db_indices(cache_dir, db_txt)
    print(f"  DB size: {len(db_idx)}")

    # Forward + sign-binarize
    bins = []
    with torch.no_grad():
        for i in range(0, len(db_idx), batch_size):
            batch_idx = db_idx[i:i+batch_size]
            feat = torch.from_numpy(np.asarray(vg[batch_idx], dtype=np.float32)).to(device)
            cont = model.encoder_layers(feat)
            b = (cont.sign() > 0).cpu().numpy().astype(np.uint8)
            bins.append(b)
    bits = np.concatenate(bins, axis=0)
    print(f"  hash codes: {bits.shape}")

    # Compute DNA-uniq directly on 36-bit codes (treat as 36-element tuple)
    n_total = bits.shape[0]
    n_unique = len(set(tuple(r) for r in bits))
    dna_uniq = n_unique / n_total

    # Also: 6-codebook split (6 x 6-bit) for cb-tuple comparison
    M, b_per = 6, 6
    cb_idx = np.zeros((n_total, M), dtype=np.int64)
    for m in range(M):
        for j in range(b_per):
            cb_idx[:, m] = (cb_idx[:, m] << 1) | bits[:, m*b_per + j].astype(np.int64)
    cb_tuple = len(set(tuple(r) for r in cb_idx)) / n_total
    n_dead = int((cb_idx.std(axis=0) == 0).sum())

    print(f"  DNA-uniq (36-bit): {dna_uniq:.4f}  ({n_unique}/{n_total})")
    print(f"  cb-tuple uniq (6x6-bit): {cb_tuple:.4f}")
    print(f"  dead codebooks: {n_dead}/6")

    return {
        "method": method, "dataset": dataset,
        "N": n_total, "DNA_uniq": float(dna_uniq), "cb_tuple_uniq": float(cb_tuple),
        "dead_cb": n_dead,
    }


def main():
    results = []
    for cfg in CONFIGS:
        r = compute_dna_uniq(*cfg)
        if r: results.append(r)
    # Pretty print summary
    print("\n\n" + "=" * 80)
    print(f"{'Method':>15s} | {'Dataset':>10s} | {'N':>8s} | {'DNA-uniq':>9s} | {'cb-tuple':>9s} | {'dead':>5s}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:>15s} | {r['dataset']:>10s} | {r['N']:>8d} | {r['DNA_uniq']:>9.4f} | {r['cb_tuple_uniq']:>9.4f} | {r['dead_cb']:>5d}")

    # Save
    out_path = "docs/baseline_dna_uniq.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
