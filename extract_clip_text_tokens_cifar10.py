"""Extract CLIP TOKEN-LEVEL text features for CIFAR10 (add-on to cache/cifar10_clip).

Reuses image_ids.json from an existing cifar10_clip cache, reads Qwen JSONL,
remaps CIFAR10 keys to V3V4 slot positions, and saves:
    text_tokens.f16.npy      [N, M=6, T, D_proj]
    text_token_mask.bool.npy [N, M=6, T]

The output files are written INTO the same cache dir so training scripts
can point to it directly (no need for a separate donor-symlink cache).

Usage:
    python extract_clip_text_tokens_cifar10.py \\
        --cache_dir ./cache/cifar10_clip \\
        --qwen_cache_path ./cache/cifar10_qwen.jsonl \\
        --text_max_length 32
"""
from __future__ import annotations
import argparse, json, os, sys, time
from typing import List
import numpy as np
import torch
from tqdm import tqdm

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path: sys.path.insert(0, _REPO)

from models.pretrained_backbone_clip import CLIPBackbone, DEFAULT_CLIP_BACKBONE
from dna_utils import build_clip_text_tokenizer, DEFAULT_CLIP_TOKENIZER_NAME

SLOT_KEYS_V3V4 = (
    "C_global", "C_primary_object", "C_secondary_object",
    "C_activity_or_relation", "C_color_texture", "C_scene_type",
)
CIFAR10_KEY_TO_V3V4 = {
    "C_global": "C_global",
    "C_head_or_main_part": "C_primary_object",
    "C_body_or_secondary_part": "C_secondary_object",
    "C_limb_or_detail_part": "C_activity_or_relation",
    "C_color_texture": "C_color_texture",
    "C_background_null": "C_scene_type",
}


def _load_qwen_cache(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = r.get("image_id")
            cb = r.get("codebook_texts", {})
            if iid and cb:
                remapped = {CIFAR10_KEY_TO_V3V4.get(k, k): v for k, v in cb.items()}
                out[iid] = remapped
    return out


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--qwen_cache_path", required=True)
    ap.add_argument("--clip_backbone", default=DEFAULT_CLIP_BACKBONE)
    ap.add_argument("--tokenizer_name", default=DEFAULT_CLIP_TOKENIZER_NAME)
    ap.add_argument("--text_max_length", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    image_ids: List[str] = json.load(open(os.path.join(args.cache_dir, "image_ids.json")))
    N = len(image_ids)
    M = len(SLOT_KEYS_V3V4)
    T = args.text_max_length
    print(f"[cifar10-text-tok] N={N}, M={M}, T={T}")

    print(f"[cifar10-text-tok] loading {args.clip_backbone}")
    backbone = CLIPBackbone(args.clip_backbone).to(args.device).eval()
    for p in backbone.parameters(): p.requires_grad = False
    D = backbone.projection_dim
    tok = build_clip_text_tokenizer(args.tokenizer_name)

    print(f"[cifar10-text-tok] loading qwen cache {args.qwen_cache_path}")
    qwen = _load_qwen_cache(args.qwen_cache_path)
    coverage = sum(1 for iid in image_ids if iid in qwen)
    print(f"[cifar10-text-tok] coverage: {coverage}/{N}")

    text_tok_p = os.path.join(args.cache_dir, "text_tokens.f16.npy")
    mask_p     = os.path.join(args.cache_dir, "text_token_mask.bool.npy")
    tokens_mm = np.lib.format.open_memmap(text_tok_p, mode="w+", dtype=np.float16,
                                          shape=(N, M, T, D))
    mask_mm   = np.lib.format.open_memmap(mask_p,     mode="w+", dtype=np.bool_,
                                          shape=(N, M, T))
    print(f"[cifar10-text-tok] preallocated {text_tok_p} = {tokens_mm.nbytes/1e9:.2f} GB")

    captions: List[str] = []
    for iid in image_ids:
        sents = qwen.get(iid)
        if sents is None:
            captions.extend(["", "", "", "", "", ""])
        else:
            captions.extend([str(sents.get(k, "") or "") for k in SLOT_KEYS_V3V4])
    assert len(captions) == N * M

    bs = args.batch_size
    t0 = time.time()
    for i in tqdm(range(0, len(captions), bs), desc="text-tokens"):
        chunk = captions[i:i+bs]
        enc = tok(chunk, padding="max_length", truncation=True,
                  max_length=T, return_tensors="pt",
                  return_attention_mask=True).to(args.device)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask")
        if attn is None:
            pad_id = getattr(tok, "pad_token_id", None)
            attn = (input_ids != pad_id).long() if pad_id is not None \
                    else torch.ones_like(input_ids)
        out = backbone.model.text_model(input_ids=input_ids, attention_mask=attn)
        tokens_h = out.last_hidden_state                                     # [b, T, H_t]
        tokens_d = backbone.model.text_projection(tokens_h)                  # [b, T, D]
        tokens_d = tokens_d.float().cpu().numpy().astype(np.float16)
        mask_np  = attn.bool().cpu().numpy()
        for k in range(tokens_d.shape[0]):
            row = (i + k) // M
            slot = (i + k) % M
            tokens_mm[row, slot] = tokens_d[k]
            mask_mm[row, slot] = mask_np[k]

    tokens_mm.flush(); del tokens_mm
    mask_mm.flush();   del mask_mm
    print(f"[cifar10-text-tok] done in {time.time()-t0:.1f}s -> {args.cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
