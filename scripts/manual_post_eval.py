"""Manual post-eval helper for the v122a (L=4, K=256) run.

The originating v122a process imported the OLD extraction_siglip2.py with
the (N, 18) buffer hardcoded; mid-eval skipped via try/except and the
final extraction also failed silently before the evaluation JSON files
could be written. This helper:
    1. Loads model + checkpoint from the v122a result dir
    2. Re-runs encode_split on test (= query) and DB loaders using the
       FIXED extraction_siglip2.py (M*L buffer derived from the model)
    3. Computes retrieval, collapse, compositional, NMI metrics
    4. Writes the same JSON files the original training run would have
       written

Usage:
    python scripts/manual_post_eval.py <result_dir>

The result_dir must contain args.txt + config.pt + model_state_dict.pth.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=str)
    args_cli = ap.parse_args()

    rd = Path(args_cli.result_dir)
    if not rd.is_dir():
        raise SystemExit(f"[post-eval] not a directory: {rd}")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import Config
    from model_siglip2 import SigLIP2SemanticOTModel
    from extraction_siglip2 import encode_split
    from evaluation_siglip2 import evaluate_retrieval, evaluate_code_collapse
    from dataloaders import load_dataset
    from dna_utils import get_transform

    cfg_path = rd / "config.pt"
    if not cfg_path.exists():
        raise SystemExit(f"[post-eval] missing config.pt in {rd}")
    cfg_state = torch.load(cfg_path, map_location="cpu", weights_only=False)
    if isinstance(cfg_state, dict):
        # config.pt is a dict (per Config.save_arg()); turn it into a
        # Namespace so the model's `getattr(args, ...)` works.
        from argparse import Namespace
        args = Namespace(**cfg_state)
    else:
        args = cfg_state
    print(f"[post-eval] num_codons_per_codebook = {getattr(args, 'num_codons_per_codebook', '?')} "
          f"codebook_size = {getattr(args, 'codebook_size', '?')} "
          f"backbone_type = {getattr(args, 'backbone_type', '?')}")

    # ---- model ----------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[post-eval] device={device}")
    model = SigLIP2SemanticOTModel(args).to(device)
    sd = torch.load(rd / "model_state_dict.pth", map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[post-eval] state_dict load: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    print(f"[post-eval] model.num_codons_per_codebook = "
          f"{getattr(model, 'num_codons_per_codebook', '?')}")

    # ---- data loaders ---------------------------------------------------
    transform      = get_transform("train")
    test_transform = get_transform("test")
    qwen_text_cache_path = getattr(args, "qwen_text_cache_path", None)
    feature_cache_dir    = getattr(args, "siglip2_feature_cache_dir", None)
    use_pixel_input = False  # we have cached features so pixel decode is unnecessary

    _train_set, queryset, dbset = load_dataset(
        args.dataset_dir, args.dataset, setting="setting1",
        train_transform=transform, test_transform=test_transform,
        load_train=False, load_test=True, load_database=True,
        return_index=True, return_paired_aug_img=False,
        qwen_text_cache_path=qwen_text_cache_path,
        siglip2_feature_cache_dir=feature_cache_dir,
        force_pixel_decode=use_pixel_input,
    )

    bs = int(getattr(args, "extract_batch_size", 256))
    db_loader = torch.utils.data.DataLoader(
        dbset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=False,
    )
    qy_loader = torch.utils.data.DataLoader(
        queryset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=False,
    )

    print("[post-eval] encoding db ...")
    db_ext = encode_split(model, db_loader, device, split_name="db")
    print(f"[post-eval] db base_indices: {db_ext['base_indices'].shape}")
    print("[post-eval] encoding query ...")
    qy_ext = encode_split(model, qy_loader, device, split_name="query")
    print(f"[post-eval] query base_indices: {qy_ext['base_indices'].shape}")

    # Save extractions
    np.savez(rd / "extract_db.npz",    **{k: v for k, v in db_ext.items() if isinstance(v, np.ndarray)})
    np.savez(rd / "extract_query.npz", **{k: v for k, v in qy_ext.items() if isinstance(v, np.ndarray)})
    print(f"[post-eval] saved extract_{{db,query}}.npz")

    # ---- evaluation -----------------------------------------------------
    distance_mode = str(getattr(args, "dna_distance_mode", "base"))
    print(f"[post-eval] distance_mode = {distance_mode}")
    codebook_size = int(getattr(args, "codebook_size", 64))

    retrieval = evaluate_retrieval(
        qy_ext, db_ext,
        distance_mode=distance_mode,
        precision_at_k_list=(1, 5, 10, 50, 100),
        remove_self_match=False,
    )
    collapse = evaluate_code_collapse(db_ext, codebook_size=codebook_size)

    eval_json = {
        "mAP":              float(retrieval["mAP"]),
        "precision_at_k":   {str(k): float(v) for k, v in retrieval["precision_at_k"].items()},
        "recall_at_k":      {str(k): float(v) for k, v in retrieval.get("recall_at_k", {}).items()},
        "pr_curve":         retrieval.get("pr_curve", []),
        "mean_positive_distance": float(retrieval["mean_positive_distance"]),
        "mean_negative_distance": float(retrieval["mean_negative_distance"]),
        "codebook_entropy":  collapse.get("codebook_entropy"),
        "codebook_normalized_entropy": collapse.get("codebook_normalized_entropy"),
        "codebook_perplexity": collapse.get("codebook_perplexity"),
        "dead_code_ratio":   collapse.get("dead_code_ratio"),
        "codeword_usage_counts": collapse.get("codeword_usage_counts"),
        "base_entropy":      collapse.get("base_entropy"),
        "base_normalized_entropy": collapse.get("base_normalized_entropy"),
        "mean_base_entropy": collapse.get("mean_base_entropy"),
        "mean_base_normalized_entropy": collapse.get("mean_base_normalized_entropy"),
        "unique_code_ratio": float(collapse["unique_code_ratio"]),
        "duplicate_rate":    float(collapse["duplicate_rate"]),
        "per_codebook_unique_count":   collapse.get("per_codebook_unique_count"),
        "per_codebook_unique_ratio":   collapse.get("per_codebook_unique_ratio"),
        "mean_per_codebook_unique_ratio": float(collapse.get("mean_per_codebook_unique_ratio", 0.0)),
        "distance_mode":     distance_mode,
        "codebook_size":     codebook_size,
        "n_query":           int(qy_ext["base_indices"].shape[0]),
        "n_db":              int(db_ext["base_indices"].shape[0]),
    }
    # numpy → python types for JSON
    def _conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        return o
    eval_json = json.loads(json.dumps(eval_json, default=_conv))

    out_path = rd / "evaluation_siglip2_base.json"
    with open(out_path, "w") as f:
        json.dump(eval_json, f, indent=2)
    print(f"[post-eval] wrote {out_path}")
    print(f"[post-eval] SUMMARY: mAP={eval_json['mAP']:.4f}  "
          f"P@1={eval_json['precision_at_k']['1']:.4f}  "
          f"DNA={eval_json['unique_code_ratio']:.4f}  "
          f"dead={float(np.mean(eval_json['dead_code_ratio'])):.4f}")


if __name__ == "__main__":
    main()
