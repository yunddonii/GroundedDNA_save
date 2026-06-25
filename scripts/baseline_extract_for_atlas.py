"""Extract a flat-hash baseline's 36-bit codes for the codeword_concept_atlas tool.

Loads a baseline checkpoint (CIBHash/CIMON/MLS3RDUH), runs encoder on the CUB
DB split, splits the 36-bit binary code into 6 x 6-bit "imaginary codebooks"
(codebook_indices [N, 6] in [0, 64)), and writes extract_db.npz with the same
schema as our v170a runs so we can hand the result to tools/codeword_concept_atlas.py.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
from types import SimpleNamespace

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to baseline epoch_059.pth")
    ap.add_argument("--method", required=True, choices=["cibhash","cimon","mls3rduh"])
    ap.add_argument("--dataset", default="CUB_200")
    ap.add_argument("--setting", default="setting1")
    ap.add_argument("--cache_dir", default="cache/cub200_clip")
    ap.add_argument("--qwen_jsonl", default="cache/cub200_qwen_v6b_trainset.jsonl")
    ap.add_argument("--out_dir", required=True, help="Output dir for extract_db.npz")
    ap.add_argument("--bit", type=int, default=36)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build dataset via dataloaders
    from dataloaders import ImgRtvCUB2011, ImgRtvDataset
    if args.dataset == "CUB_200":
        ds = ImgRtvCUB2011(
            root=os.path.join("dataset", "CUB_200"),
            mode="train",
            setting_name=args.setting,
            qwen_text_cache_path=args.qwen_jsonl,
            siglip2_feature_cache_dir=args.cache_dir,
        )
    else:
        ds = ImgRtvDataset(
            root=os.path.join("dataset", args.dataset),
            mode="train",
            setting_name=args.setting,
            qwen_text_cache_path=args.qwen_jsonl,
            siglip2_feature_cache_dir=args.cache_dir,
        )

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Build baseline model (uses cached visual_global)
    from baseline.base_model import _build_baseline_model
    # We need to provide minimal args
    bargs = SimpleNamespace(
        method=args.method, bit=args.bit,
        batch_size=args.batch_size, device=args.device,
        cache_dir=args.cache_dir,
    )
    model = _build_baseline_model(bargs).to(args.device)
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Extract codes
    all_bin = []
    all_paths = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            vg = batch.get("cached_visual_global", None)
            if vg is None: vg = batch.get("visual_global", None)
            if vg is None:
                # fallback: try image_paths to load CLIP - skip for now
                continue
            vg = vg.to(args.device).float()
            out = model.encode(vg) if hasattr(model, "encode") else model(vg)
            if isinstance(out, tuple): cont = out[0]
            elif isinstance(out, dict): cont = out.get("z") or out.get("hash") or list(out.values())[0]
            else: cont = out
            bin_codes = torch.sign(cont).clamp(min=-1, max=1).cpu().numpy().astype(np.int8)
            all_bin.append(bin_codes)
            paths = batch.get("image_path", None)
            if paths is not None:
                all_paths.extend(paths)
            lbl = batch.get("multi_hot_label", None)
            if lbl is not None:
                all_labels.append(lbl.numpy())
    bins = np.concatenate(all_bin, axis=0)              # [N, 36] in {-1, +1}
    print(f"[extract] N={bins.shape[0]} bits={bins.shape[1]}")

    # Convert -1/+1 -> 0/1
    bits = (bins > 0).astype(np.uint8)                  # [N, 36]
    # Split into 6 x 6-bit -> codebook_indices [N, 6]
    M, b_per_cb = 6, args.bit // 6
    cb_idx = np.zeros((bits.shape[0], M), dtype=np.int64)
    for m in range(M):
        chunk = bits[:, m*b_per_cb:(m+1)*b_per_cb]
        # binary to int [0, 2^b_per_cb)
        for j in range(b_per_cb):
            cb_idx[:, m] = (cb_idx[:, m] << 1) | chunk[:, j].astype(np.int64)
    # base_indices: each codeword's two bits within codon-style (not meaningful for baseline; reuse bits)
    base = np.zeros((bits.shape[0], M*3), dtype=np.int64)
    for j in range(b_per_cb):
        base[:, j::b_per_cb] = bits[:, j::b_per_cb]
    paths = np.array(all_paths, dtype=object) if all_paths else np.array([], dtype=object)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((bits.shape[0], 1), dtype=np.float32)

    np.savez(
        os.path.join(args.out_dir, "extract_db.npz"),
        base_indices=base,
        hash_2bit=bits,
        codebook_indices=cb_idx,
        multi_hot_labels=labels.astype(np.float32),
        image_paths=paths,
        allow_pickle=True,
    )
    print(f"[extract] wrote {args.out_dir}/extract_db.npz")
    print(f"  codebook_indices unique per cb: {[len(np.unique(cb_idx[:,m])) for m in range(M)]}")

if __name__ == "__main__":
    main()
