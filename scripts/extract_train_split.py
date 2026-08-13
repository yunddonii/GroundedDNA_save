#!/usr/bin/env python
"""Additively emit `extract_train.npz` for a finished run (REQUIRED_EXPERIMENTS §2.4).

`extraction_siglip2.py` only encodes db + query. Held-out codon decoding also
needs the TRAIN rows, because the `(slot, code) -> concept` dictionary is built
on train and evaluated on test.

For Flickr25k and NUS-WIDE `train` is a subset of the DB extraction, so the
train codes can simply be sliced out by image basename and this script is not
needed. MSCOCO's train split is disjoint from its DB, so its train codes have
to be encoded explicitly -- that is what this is for.

The db/query schema is untouched; this only adds a file.

    python scripts/extract_train_split.py --config_path <result_dir>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, set_random_seed
from dataloaders import load_dataset
from dna_utils import get_transform
from extraction_siglip2 import (
    encode_split, _find_model_checkpoint, _resume_args_flat_or_legacy,
)
from model_siglip2 import SigLIP2SemanticOTModel


def main() -> None:
    set_random_seed(42)
    args = Config()
    _resume_args_flat_or_legacy(args)

    model = SigLIP2SemanticOTModel(args).to(args.device)
    ckpt = _find_model_checkpoint(args.save_model_state_path)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no checkpoint at {ckpt}")
    missing, unexpected = model.load_state_dict(
        torch.load(ckpt, map_location=args.device), strict=False)
    print(f"[extract-train] loaded {ckpt} "
          f"(missing={len(missing)} unexpected={len(unexpected)})")
    # F01: same fail-closed epoch restore as extraction_siglip2, via the SAME
    # helper. Duplicating the resolution logic is how the two paths drifted into
    # producing train and query/DB codes at different operating points.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dna_utils.runtime_state import apply_inference_epoch
    _resolved = apply_inference_epoch(model, ckpt, args)
    print(f"[extract-train] inference epoch={_resolved.epoch} "
          f"(source={_resolved.source}) "
          f"effective_sinkhorn_epsilon={_resolved.effective_sinkhorn_epsilon}")

    transform = get_transform("test")           # no augmentation for extraction
    trainset, _, _ = load_dataset(
        args.dataset_dir, args.dataset, setting=args.setting,
        train_transform=transform, test_transform=transform,
        load_train=True, load_database=False, load_test=False,
        return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=getattr(args, "siglip2_feature_cache_dir", None),
    )

    loader = torch.utils.data.DataLoader(
        trainset, batch_size=int(getattr(args, "extract_batch_size", 256)),
        shuffle=False, num_workers=args.num_workers, drop_last=False,
    )
    out = encode_split(model, loader, args.device, split_name="train")

    dest = os.path.join(args.save_result_path, "extract_train.npz")
    np.savez(dest, **out)
    print(f"[extract-train] {out['base_indices'].shape[0]} rows -> {dest}")


if __name__ == "__main__":
    main()
