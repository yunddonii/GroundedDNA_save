"""Run inference (extract_db + extract_query + mAP eval) with a TRAINED checkpoint
on a DIFFERENT cache than was used at training.

Use case: trained on FAIRrank L8K3 multi-view cache (588 tokens) → evaluate on
whole-image cache (196 tokens) to test if training's codebook learning generalizes
to standard whole-image inference.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from types import SimpleNamespace

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)


def _parse_args_txt(path: str) -> SimpleNamespace:
    """Read args.txt produced by train_siglip2.py at run start."""
    ns = SimpleNamespace()
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            for k_end in range(len(line)):
                if line[k_end] == "-": break
            else: continue
            v_start = k_end
            while v_start < len(line) and line[v_start] == "-": v_start += 1
            key = line[:k_end]
            val = line[v_start:].strip()
            if not key: continue
            if val == "None": val = None
            elif val == "True": val = True
            elif val == "False": val = False
            else:
                try: val = int(val)
                except ValueError:
                    try: val = float(val)
                    except ValueError:
                        if val.startswith("[") and val.endswith("]"):
                            inner = val[1:-1].strip()
                            if inner.startswith("'") and inner.endswith("'"):
                                val = [inner[1:-1]]
            setattr(ns, key, val)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", required=True,
                    help="Trained run dir (with model_state_dict.pth + args.txt).")
    ap.add_argument("--out_dir", required=True,
                    help="Target dir to save extract_db.npz + evaluation_siglip2_base.json.")
    ap.add_argument("--inference_cache", required=True,
                    help="Path to cache to use for inference (e.g. cub200_clip_v6bplus).")
    ap.add_argument("--device", default="cuda:0")
    args_cli = ap.parse_args()

    # Load args from training run, then override cache path
    src_args_path = os.path.join(args_cli.source_dir, "args.txt")
    args = _parse_args_txt(src_args_path)
    args.siglip2_feature_cache_dir = args_cli.inference_cache
    args.text_whiten_npz = os.path.join(args_cli.inference_cache, "text_whiten.npz")
    args.device = args_cli.device
    print(f"[INFER] source: {args_cli.source_dir}")
    print(f"[INFER] cache override: {args.siglip2_feature_cache_dir}")
    print(f"[INFER] text_whiten override: {args.text_whiten_npz}")

    # Build model + load weights
    from model_siglip2 import SigLIP2SemanticOTModel
    model = SigLIP2SemanticOTModel(args).to(args.device)
    ckpt = os.path.join(args_cli.source_dir, "model_state_dict.pth")
    sd = torch.load(ckpt, map_location=args.device, weights_only=False)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[INFER] loaded checkpoint: missing={len(miss)} unexpected={len(unexp)}")
    model.eval()

    # Build datasets
    from dataloaders import ImgRtvCUB2011, ImgRtvDataset
    dataset_name = getattr(args, "dataset", None)
    setting = getattr(args, "setting", "setting1")
    dataset_dir = getattr(args, "dataset_dir", "dataset")
    qwen_path = getattr(args, "qwen_text_cache_path", None)
    cache_dir = args.siglip2_feature_cache_dir
    if dataset_name == "CUB_200":
        db_set = ImgRtvCUB2011(root=os.path.join(dataset_dir, "CUB_200"),
                                mode="database", setting_name=setting,
                                qwen_text_cache_path=qwen_path,
                                siglip2_feature_cache_dir=cache_dir)
        qy_set = ImgRtvCUB2011(root=os.path.join(dataset_dir, "CUB_200"),
                                mode="test", setting_name=setting,
                                qwen_text_cache_path=qwen_path,
                                siglip2_feature_cache_dir=cache_dir)
    else:
        ds_root = os.path.join(dataset_dir, dataset_name)
        db_set = ImgRtvDataset(root=ds_root, mode="database", setting_name=setting,
                                qwen_text_cache_path=qwen_path,
                                siglip2_feature_cache_dir=cache_dir)
        qy_set = ImgRtvDataset(root=ds_root, mode="test", setting_name=setting,
                                qwen_text_cache_path=qwen_path,
                                siglip2_feature_cache_dir=cache_dir)

    db_loader = torch.utils.data.DataLoader(db_set, batch_size=64, shuffle=False, num_workers=2)
    qy_loader = torch.utils.data.DataLoader(qy_set, batch_size=64, shuffle=False, num_workers=2)
    print(f"[INFER] DB={len(db_set)}  Query={len(qy_set)}")

    # Run extraction using existing extraction_siglip2.encode_split
    os.makedirs(args_cli.out_dir, exist_ok=True)
    from extraction_siglip2 import encode_split
    t0 = time.time()
    db_out = encode_split(model=model, loader=db_loader, device=args.device, split_name="db")
    t1 = time.time(); print(f"[INFER] DB extracted in {t1-t0:.1f}s")
    qy_out = encode_split(model=model, loader=qy_loader, device=args.device, split_name="query")
    t2 = time.time(); print(f"[INFER] Query extracted in {t2-t1:.1f}s")

    np.savez(os.path.join(args_cli.out_dir, "extract_db.npz"),    **db_out)
    np.savez(os.path.join(args_cli.out_dir, "extract_query.npz"), **qy_out)
    print(f"[INFER] saved extract_db.npz + extract_query.npz")

    # Evaluate
    from evaluation_siglip2 import evaluation
    res = evaluation(
        path=args_cli.out_dir,
        distance_mode=getattr(args, "dna_distance_mode", "base"),
        codebook_size=getattr(args, "codebook_size", 64),
    )
    # Save eval results to json
    eval_out = os.path.join(args_cli.out_dir, "evaluation_siglip2_base.json")
    with open(eval_out, "w") as f:
        # convert numpy types
        def _conv(o):
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o
        json.dump(res, f, indent=2, default=_conv)
    print(f"[INFER] eval done: mAP = {res.get('mAP', '--')}")
    print(f"[INFER] saved {eval_out}")


if __name__ == "__main__":
    main()
