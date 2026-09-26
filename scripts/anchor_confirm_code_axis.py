#!/usr/bin/env python
"""Code-to-own-axis probe for one verified checkpoint (docs/ANCHOR_CONFIRMATION_CONTRACT_v1.md 8.2).

For the first `min(512, n_val)` images of the run's own train-only validation split (the trainer's
`val_split.carve_val_indices` with the run's typed ratio and seed, ascending dataset index), a
deployment forward (eval mode, no text reaches the model) gives each local slot's quantised
codeword. Each codeword is compared, after batch-centring, with the same image's four axis-caption
embeddings (the cached caption features through the model's own text adapter); the slot scores 1
when its own axis caption is the nearest. Captions are only the target. The arithmetic of
`match_acc` is the exploratory probe's (result/analysis/arch_exp3_20260920/quant_gap_diag.py on
branch arch-exp-2026-09), unchanged.

What differs from the exploratory probe, on purpose:
  * the model is built from the run's saved typed configuration (config.pt), not from the
    display-only args.txt, which cannot express a negative value;
  * the checkpoint is the bytes the run record pins, loaded with no missing or unexpected key;
  * the rows come from the trainer's own split function, not from a cache-row file.
Every input is read once and parsed from the hashed bytes; the output is written once.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.anchor_confirm_decision import (  # noqa: E402
    Consumed, NotReducible, PROBE_KIND, need, proportion)

PROBE_VERSION = "anchor-confirm-code-axis/1"
DEFAULT_IMAGES = 512


def match_acc(q, t) -> dict:
    """[B, M, D] codewords and [B, M, D] caption embeddings of the same images and slots."""
    import torch
    import torch.nn.functional as F
    q = q.float()
    t = t.float()
    q = F.normalize(q - q.mean(0, keepdim=True), dim=-1)
    t = F.normalize(t - t.mean(0, keepdim=True), dim=-1)
    S = torch.einsum("bmd,bad->bma", q, t)                    # [B, code, axis]
    M = S.shape[1]
    tgt = torch.arange(M, device=S.device).expand(S.shape[0], M)
    return {"code_picks_own_axis": (S.argmax(dim=2) == tgt).float().mean().item(),
            "axis_picks_own_code": (S.argmax(dim=1) == tgt).float().mean().item(),
            "chance": 1.0 / M}


def held_out_indices(trainset, args, n_images: int) -> list:
    from val_split import carve_val_indices, get_train_labels
    ratio, seed = float(args.val_split_ratio), int(args.val_split_seed)
    need(0.0 < ratio < 0.5, f"the run has no train-only validation split (ratio {ratio})")
    _, val_idx, _ = carve_val_indices(get_train_labels(trainset), ratio=ratio, seed=seed)
    ordered = sorted(int(i) for i in val_idx)
    need(len(ordered) > 0, "the validation split is empty")
    return ordered[:min(n_images, len(ordered))]


def probe(record_path, record_sha256: str, *, device: str, n_images: int) -> dict:
    import torch
    from types import SimpleNamespace
    consumed = Consumed()
    record = consumed.json(record_path, record_sha256)
    completion = record["completion"]
    run_dir = Path(record["run_dir"])
    checkpoint = run_dir / completion["final_checkpoint"]
    ckpt_raw = consumed.read(checkpoint, completion["final_checkpoint_sha256"])
    consumed.read(Path(str(checkpoint) + ".runtime.json"), completion["checkpoint_runtime_sha256"])
    config_raw = consumed.read(run_dir / "config.pt")
    anchor = record.get("anchor_confirmation")
    if anchor is not None:
        need(anchor.get("config_pt_sha256") == hashlib.sha256(config_raw).hexdigest(),
             f"{run_dir}: config.pt is not the one the record's anchor evidence pins")
    config = torch.load(io.BytesIO(config_raw), map_location="cpu", weights_only=False)
    args = SimpleNamespace(**{k: v for k, v in config.items() if not k.startswith("__")})
    need(os.environ.get("GDNA_NUM_SEMANTIC_PARTS") == "5",
         "export GDNA_NUM_SEMANTIC_PARTS=5 before python starts")

    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch
    model = SigLIP2SemanticOTModel(args).to(device).eval()
    state = torch.load(io.BytesIO(ckpt_raw), map_location="cpu", weights_only=False)
    result = model.load_state_dict(state, strict=False)
    need(not result.missing_keys and not result.unexpected_keys,
         f"checkpoint keys do not match the model: missing {result.missing_keys[:4]}, "
         f"unexpected {result.unexpected_keys[:4]}")
    apply_inference_epoch(model, str(checkpoint), args)
    cache = args.siglip2_feature_cache_dir
    trainset, _, _ = load_dataset(
        getattr(args, "dataset_dir", "dataset"), args.dataset, setting=args.setting,
        train_transform=None, test_transform=None, load_train=True, load_database=False,
        load_test=False, return_index=True,
        qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
        siglip2_feature_cache_dir=cache)
    rows = held_out_indices(trainset, args, n_images)
    codes, captions = [], []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            samples = [trainset[i] for i in rows[start:start + 64]]
            out = model(pixel_values=None, part_input_ids=None, part_attention_mask=None,
                        return_routing=True,
                        cached_visual_tokens_raw=torch.stack(
                            [s["cached_visual_tokens_raw"] for s in samples]).to(device),
                        cached_visual_global=torch.stack(
                            [s["cached_visual_global"] for s in samples]).to(device),
                        cached_text_part_raw=None, cached_has_text=None)   # deployment forward
            raw_text = torch.stack([s["cached_text_part_raw"] for s in samples]).to(device)
            codes.append(out["quantized_tokens"][:, 1:, :].cpu())
            captions.append(model._adapt_pooled_text_for_loss(raw_text)[:, 1:, :].cpu())
    scores = match_acc(torch.cat(codes), torch.cat(captions))
    proportion(scores["code_picks_own_axis"], "code_picks_own_axis")
    return {"artifact_kind": PROBE_KIND, "version": PROBE_VERSION,
            "record_sha256": record_sha256,
            "checkpoint_sha256": completion["final_checkpoint_sha256"],
            "config_pt_sha256": hashlib.sha256(config_raw).hexdigest(),
            "n_images": len(rows), "first_rows": rows[:8], **scores,
            "note": "deployment forward; captions are only the target", "_consumed": consumed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", required=True)
    parser.add_argument("--record-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--images", type=int, default=DEFAULT_IMAGES)
    args = parser.parse_args(argv)
    try:
        need(args.images == DEFAULT_IMAGES, "the contract fixes 512 images")
        payload = probe(args.record, args.record_sha256, device=args.device,
                        n_images=args.images)
        from scripts.anchor_confirm_decision import write_once
        digest = write_once(Path(args.out), payload)
    except (NotReducible, KeyError, FileExistsError) as error:
        print(f"[anchor-probe] REFUSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"[anchor-probe] wrote {args.out} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
