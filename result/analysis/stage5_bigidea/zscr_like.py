"""Shared helpers for stage 5: the stage-2 posterior reader and the deployment-path codes."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage2_zscr"))
from zscr_pilot import read_scores, codon_of  # noqa: E402,F401  identical scoring to stage 2


def deployed_codes(result_dir, tr, indices, device):
    """Codons [n, 5] and bases [n, 15] from the deployment forward (no text, eval mode)."""
    from slot_role_probe import parse_args_txt
    from model_siglip2 import SigLIP2SemanticOTModel
    from dna_utils.runtime_state import apply_inference_epoch
    args = parse_args_txt(os.path.join(result_dir, "args.txt"))
    args.lambda_mec = 0.0
    ckpt = os.path.join(result_dir, "model_state_dict.pth")
    model = SigLIP2SemanticOTModel(args).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    apply_inference_epoch(model, ckpt, args)
    codons, bases = [], []
    with torch.no_grad():
        for s0 in range(0, len(indices), 128):
            smp = [tr[i] for i in indices[s0:s0 + 128]]
            vt = torch.stack([s["cached_visual_tokens_raw"] for s in smp]).to(device)
            vg = torch.stack([s["cached_visual_global"] for s in smp]).to(device)
            o = model(pixel_values=None, part_input_ids=None, part_attention_mask=None,
                      return_routing=True, cached_visual_tokens_raw=vt, cached_visual_global=vg,
                      cached_text_part_raw=None, cached_has_text=None)
            b = o["base_indices_per_codebook"]                       # [b, 5, 3]
            codons.append(codon_of(b).cpu())
            bases.append(b.reshape(b.shape[0], -1).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(codons).numpy(), torch.cat(bases).numpy()
