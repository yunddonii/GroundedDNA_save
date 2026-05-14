"""Extraction pipeline for ``SigLIP2SemanticOTModel``.

Replaces the legacy `extraction.py` (which uses the old `model.Model`).
Loads a checkpoint, runs the model in eval mode (codebook_mean routing,
no text input), and saves both the per-base index code and the 2-bit
DB-compatible code for db / query splits.

Output (per split, saved as .npz):
    base_indices       [N, 18]
    hash_2bit          [N, 36]   (uint8, DB-compatible)
    codebook_indices   [N, 6]
    labels             [N]                     (if single-label)
    multi_hot_labels   [N, num_classes]        (if multi-hot)
    image_paths        [N]                     (if available)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from config import Config, set_random_seed
from dataloaders import load_dataset
from model_siglip2 import SigLIP2SemanticOTModel
# `get_transform` is now mirrored in dna_utils so we don't have to import the
# legacy `utils.py` (which depends on DWKM / sinkhornbarycenters).
from dna_utils import base_indices_to_2bit_flat, get_transform


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _get_pixel_values(batch: Dict[str, Any]) -> torch.Tensor:
    pv = batch.get("pixel_values", batch.get("img"))
    if pv is None:
        raise RuntimeError(
            "[extraction_siglip2] batch must contain 'pixel_values' or 'img' key."
        )
    return pv


def _get_labels(batch: Dict[str, Any]):
    """Resolve labels with auto-detection of multi-hot vs single-class.

    The legacy `ImgRtvDataset` (Flickr25k / MSCOCO / NUSWIDE / ImageNet100)
    puts a multi-hot vector ``[num_classes]`` in ``batch['label']`` --
    the same key that single-class datasets use for the int class id. This
    helper routes the tensor to the correct slot so the evaluation Jaccard
    relevance matrix uses multi-hot semantics for multi-label datasets.
    """
    labels = batch.get("labels", batch.get("label", batch.get("target", None)))
    multi_hot_labels = batch.get(
        "multi_hot_labels", batch.get("multi_label", None)
    )
    # Auto-promote: if `labels` is [B, C] with C > 1, treat as multi-hot.
    if (labels is not None and hasattr(labels, "ndim")
            and labels.ndim == 2 and labels.shape[1] > 1):
        if multi_hot_labels is None:
            multi_hot_labels = labels
        labels = None
    return labels, multi_hot_labels


def encode_split(
    model: SigLIP2SemanticOTModel,
    loader: torch.utils.data.DataLoader,
    device,
    split_name: str = "",
) -> Dict[str, np.ndarray]:
    """Run the model on a split and accumulate extraction tensors."""
    n_samples = len(loader.dataset)
    base_idx_all = np.zeros((n_samples, 18), dtype=np.int64)
    hash2bit_all = np.zeros((n_samples, 36), dtype=np.uint8)
    cb_idx_all   = np.zeros((n_samples, 6),  dtype=np.int64)
    labels_list:    List[np.ndarray] = []
    mh_list:        List[np.ndarray] = []
    paths_list:     List[str]        = []

    write_idx = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"extract[{split_name}]"):
            cached_vt = batch.get("cached_visual_tokens_raw", None)
            if cached_vt is not None:
                cached_vt = cached_vt.to(device)
                cached_vg = batch.get("cached_visual_global", None)
                cached_vg = cached_vg.to(device) if cached_vg is not None else None
                B = cached_vt.shape[0]
                outputs = model(
                    pixel_values=None,
                    part_input_ids=None,
                    part_attention_mask=None,
                    return_routing=True,
                    cached_visual_tokens_raw=cached_vt,
                    cached_visual_global=cached_vg,
                    cached_text_part_raw=None,    # not needed at eval
                    cached_has_text=None,
                )
            else:
                pixel_values = _get_pixel_values(batch).to(device)
                B = pixel_values.shape[0]
                outputs = model(
                    pixel_values=pixel_values,
                    part_input_ids=None,
                    part_attention_mask=None,
                    return_routing=True,
                )
            assert outputs["routing_mode"] == "codebook_mean", (
                f"[extraction] expected codebook_mean routing at eval, got "
                f"{outputs['routing_mode']!r}"
            )
            base_indices = outputs["base_indices"]               # [B, 18]
            cb_indices   = outputs["codebook_indices"]           # [B, 6]
            hash_2bit    = base_indices_to_2bit_flat(base_indices)  # [B, 36]

            base_idx_all[write_idx:write_idx + B] = _to_numpy(base_indices)
            cb_idx_all  [write_idx:write_idx + B] = _to_numpy(cb_indices)
            hash2bit_all[write_idx:write_idx + B] = _to_numpy(hash_2bit).astype(np.uint8)

            labels, multi_hot_labels = _get_labels(batch)
            if labels is not None:
                labels_list.append(_to_numpy(labels))
            if multi_hot_labels is not None:
                mh_list.append(_to_numpy(multi_hot_labels))
            if "image_path" in batch:
                # default collate makes this a list of strings already
                paths_list.extend(list(batch["image_path"]))

            write_idx += B

    out: Dict[str, np.ndarray] = {
        "base_indices":     base_idx_all[:write_idx],
        "hash_2bit":        hash2bit_all[:write_idx],
        "codebook_indices": cb_idx_all  [:write_idx],
    }
    if labels_list:
        labels_arr = np.concatenate(labels_list, axis=0)
        # Existing dataloaders return labels as [N, C] one-hot/multi-hot.
        # Detect single-label by max-1-on-row and emit both forms.
        if labels_arr.ndim == 2:
            out["multi_hot_labels"] = labels_arr.astype(np.int64)
            row_sums = labels_arr.sum(axis=1)
            if (row_sums == 1).all():
                out["labels"] = labels_arr.argmax(axis=1).astype(np.int64)
        else:
            out["labels"] = labels_arr.astype(np.int64)
    if mh_list:
        out["multi_hot_labels"] = np.concatenate(mh_list, axis=0).astype(np.int64)
    if paths_list:
        out["image_paths"] = np.array(paths_list, dtype=object)
    return out


def _find_model_checkpoint(save_model_state_path: str) -> str:
    """Locate the trained model checkpoint, preferring the new flat name.

    Preference order:
        1. ``<dir>/model_state_dict.pth``     (current SigLIP2 layout)
        2. ``<dir>/model_siglip2.pt``         (legacy SigLIP2 name)
    """
    candidates = [
        os.path.join(save_model_state_path, "model_state_dict.pth"),
        os.path.join(save_model_state_path, "model_siglip2.pt"),  # legacy
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # return preferred path even if missing (for warn msg)


def extract_code(args: Config) -> None:
    model = SigLIP2SemanticOTModel(args).to(args.device)

    ckpt_path = _find_model_checkpoint(args.save_model_state_path)
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=args.device)
        # `strict=False` because the checkpoint may predate the new gumbel/anchor keys.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[extraction] load_state_dict: "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        print(f"[extraction] loaded checkpoint from {ckpt_path}")
    else:
        print(f"[extraction] WARNING: no checkpoint at {ckpt_path}; "
              f"using fresh model.")

    transform = get_transform("test")
    qwen_text_cache_path = getattr(args, "qwen_text_cache_path", None)
    feature_cache_dir    = getattr(args, "siglip2_feature_cache_dir", None)

    _, queryset, dbset = load_dataset(
        args.dataset_dir, args.dataset, setting=args.setting,
        train_transform=transform, test_transform=transform,
        load_train=False, load_database=True, load_test=True,
        return_index=True,
        qwen_text_cache_path=qwen_text_cache_path,
        siglip2_feature_cache_dir=feature_cache_dir,
    )
    if feature_cache_dir is not None:
        print(f"[extraction] using cached SigLIP2 features from {feature_cache_dir}; "
              f"encoder pass will be skipped.")

    bs = int(getattr(args, "extract_batch_size", 256))
    db_loader = torch.utils.data.DataLoader(
        dbset, batch_size=bs, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )
    query_loader = torch.utils.data.DataLoader(
        queryset, batch_size=bs, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )

    print("[extraction] encoding db ...")
    db_out = encode_split(model, db_loader, args.device, split_name="db")
    print("[extraction] encoding query ...")
    qy_out = encode_split(model, query_loader, args.device, split_name="query")

    print(f"[extraction] db    -> base_indices {db_out['base_indices'].shape}")
    print(f"[extraction] query -> base_indices {qy_out['base_indices'].shape}")

    out_dir = args.save_result_path
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "extract_db.npz"),    **db_out)
    np.savez(os.path.join(out_dir, "extract_query.npz"), **qy_out)
    print(f"[extraction] saved to {out_dir}")


def _resume_args_flat_or_legacy(args: Config) -> None:
    """Populate ``args`` from a saved ``config.pt``, supporting both layouts.

    Accepts:
      - Flat (new SigLIP2 layout): ``<config_path>/config.pt``
      - Nested (legacy):           ``<config_path>/model_state/config.pt``

    Mirrors what ``Config.load_args`` does, but additionally checks the flat
    location used by ``train_siglip2._resolve_save_path``.
    """
    cli = Config.get_config()
    config_path = cli.config_path
    if not config_path:
        raise SystemExit("[extraction] --config_path is required for standalone run.")

    flat_pt   = os.path.join(config_path, "config.pt")
    nested_pt = os.path.join(config_path, "model_state", "config.pt")

    if os.path.exists(flat_pt):
        sd = torch.load(flat_pt, map_location="cpu")
        for k, v in sd.items():
            setattr(args, k, v)
        args.save_result_path      = config_path
        args.save_log_path         = os.path.join(config_path, "")
        args.save_model_state_path = os.path.join(config_path, "")
        nd = int(getattr(args, "num_devices", 0) or 0)
        args.device = torch.device(
            f"cuda:{nd}" if torch.cuda.is_available() else "cpu"
        )
        print(f"[extraction] loaded flat config.pt from {flat_pt}")
    elif os.path.exists(nested_pt):
        # legacy nested layout
        args.load_args()
        print(f"[extraction] loaded legacy nested config.pt from {nested_pt}")
    else:
        raise FileNotFoundError(
            f"[extraction] no config.pt found at\n  {flat_pt}\nor\n  {nested_pt}"
        )


if __name__ == "__main__":
    set_random_seed(42)
    args = Config()
    _resume_args_flat_or_legacy(args)
    args.print_info()
    extract_code(args)
