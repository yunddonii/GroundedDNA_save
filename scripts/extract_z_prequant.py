#!/usr/bin/env python
"""Capture the pre-quantisation routed features z = quant_input [N, 6, D].

`scripts/codebook_semantic_alignment.py` showed the MSCOCO CODEBOOK geometry is
degenerate (rho 0.13 vs 0.58/0.49) and refuted four codebook-side causes. The
one untested mechanism is upstream: the routed features z that the codebook
quantises. If z is already high-rank / semantically ungraded on MSCOCO, the
codebook is faithfully mirroring its input and the fix belongs in the
encoder/router, not in a codebook-side loss.

z is the tensor passed INTO `self.quantizer(quant_input)` (model_siglip2.py
~L4335). It is not in the model outputs dict, so we grab it with a forward
pre-hook on the quantizer -- no model edit. Emits `extract_z.npz` alongside the
run's existing extractions:

    z              [N, 6, D]  float16   pre-quantisation routed features
    codebook_indices [N, 6]   int64     (re-emitted, for pairing with z)
    multi_hot_labels / labels, image_paths   (copied schema)

    python scripts/extract_z_prequant.py --config_path <result_dir> [--split db|query|train]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, set_random_seed
from dataloaders import load_dataset
from dna_utils import get_transform
from extraction_siglip2 import _find_model_checkpoint, _resume_args_flat_or_legacy, _get_labels
from model_siglip2 import SigLIP2SemanticOTModel


def main() -> None:
    set_random_seed(42)
    args = Config()
    _resume_args_flat_or_legacy(args)
    split = str(getattr(args, "z_split", None) or os.environ.get("Z_SPLIT", "db"))

    model = SigLIP2SemanticOTModel(args).to(args.device)
    ckpt = _find_model_checkpoint(args.save_model_state_path)
    model.load_state_dict(torch.load(ckpt, map_location=args.device), strict=False)
    model.eval()
    print(f"[z] loaded {ckpt}; split={split}")

    # forward pre-hook on the quantizer captures its input = quant_input = z
    captured = {}
    def hook(_module, inputs):
        captured["z"] = inputs[0].detach()
    handle = model.quantizer.register_forward_pre_hook(hook)

    transform = get_transform("test")
    load_kw = dict(load_train=False, load_database=False, load_test=False)
    load_kw[{"db": "load_database", "query": "load_test", "train": "load_train"}[split]] = True
    tr, qy, dbs = load_dataset(
        args.dataset_dir, args.dataset, setting=args.setting,
        train_transform=transform, test_transform=transform, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None),
        **load_kw)
    dataset = {"db": dbs, "query": qy, "train": tr}[split]

    # When capping (Z_MAX): sample EVENLY SPACED CONTIGUOUS BLOCKS.
    # Scattered indices make the memory-mapped feature cache read randomly and
    # the extraction becomes I/O bound (~90x slower, GPU at 0%). Contiguous
    # blocks keep reads sequential; spreading the blocks over the whole manifest
    # avoids head bias, which is material on NUS-WIDE (head-25K label profile
    # deviates 0.27 of 2.09 labels/img from the full DB; MSCOCO only 0.06/2.93).
    z_max_env = int(os.environ.get("Z_MAX", "0") or 0)
    if z_max_env and z_max_env < len(dataset):
        n_blocks = int(os.environ.get("Z_BLOCKS", "40"))
        n = len(dataset)
        per = max(1, z_max_env // n_blocks)
        starts = np.linspace(0, n - per, n_blocks).astype(int)
        idx = np.unique(np.concatenate([np.arange(s, s + per) for s in starts]))
        dataset = torch.utils.data.Subset(dataset, idx.tolist())
        print(f"[z] sampling {len(idx)} rows as {n_blocks} contiguous blocks of "
              f"~{per} spread over {n}")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(getattr(args, "extract_batch_size", 256)),
        shuffle=False, num_workers=args.num_workers, drop_last=False)

    # A few thousand samples suffice to estimate z geometry (eff-rank, cosine
    # spread, per-codeword label distributions with support >= 20); the full
    # 107K/193K DB forward pass is unnecessary. Cap via Z_MAX (0 = all).
    z_max = int(os.environ.get("Z_MAX", "0") or 0)
    seen = 0
    z_list, cb_list, mh_list, lab_list, paths = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"z[{split}]"):
            if z_max and seen >= z_max:
                break
            cvt = batch.get("cached_visual_tokens_raw")
            if cvt is not None:
                cvg = batch.get("cached_visual_global")
                out = model(pixel_values=None, part_input_ids=None,
                            part_attention_mask=None, return_routing=True,
                            cached_visual_tokens_raw=cvt.to(args.device),
                            cached_visual_global=cvg.to(args.device) if cvg is not None else None,
                            cached_text_part_raw=None, cached_has_text=None)
            else:
                from extraction_siglip2 import _get_pixel_values
                out = model(pixel_values=_get_pixel_values(batch).to(args.device),
                            part_input_ids=None, part_attention_mask=None,
                            return_routing=True)
            z_list.append(captured["z"].to(torch.float16).cpu().numpy())
            cb_list.append(out["codebook_indices"].cpu().numpy())
            lab, mh = _get_labels(batch)
            if mh is not None:
                mh_list.append(mh.cpu().numpy() if torch.is_tensor(mh) else np.asarray(mh))
            elif lab is not None:
                lab_list.append(lab.cpu().numpy() if torch.is_tensor(lab) else np.asarray(lab))
            if "image_path" in batch:
                paths.extend(list(batch["image_path"]))
            seen += cb_list[-1].shape[0]
    handle.remove()

    payload = {
        "z": np.concatenate(z_list).astype(np.float16),
        "codebook_indices": np.concatenate(cb_list).astype(np.int64),
    }
    if mh_list:
        payload["multi_hot_labels"] = np.concatenate(mh_list).astype(np.int64)
    if lab_list:
        payload["labels"] = np.concatenate(lab_list).astype(np.int64)
    if paths:
        payload["image_paths"] = np.array(paths, dtype=object)

    dest = os.path.join(args.save_result_path, f"extract_z_{split}.npz")
    np.savez(dest, **payload)
    print(f"[z] {payload['z'].shape} -> {dest}")


if __name__ == "__main__":
    main()
