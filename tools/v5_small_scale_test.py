"""V5 prompt small-scale test: 50 Flickr25k images.

Generates V5 captions via Qwen3-VL, encodes via CLIP, measures cross-slot
cosine. Compares against V4 (already cached). Decides GO/NO-GO for full V5
re-extraction.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)


V5_PROMPT = """\
You are a vision-language parser. Given the image, output a single JSON
object with six captions. Each caption uses a COMPLETELY DIFFERENT
vocabulary type. Reuse of nouns between captions is FORBIDDEN.

Rules:
- Output ONLY a single JSON object. No prose, no markdown fences.
- All string fields are natural English sentences, 15-20 words each.
- Every slot must be visually grounded in the image.
- CRITICAL: Do NOT reuse any noun across captions. If caption [1] mentions
  a person, captions [2-6] must use pronouns (they, it, this) instead.
- "codebook_texts" MUST contain EXACTLY the six listed keys.

Schema:
{
  "codebook_texts": {
    "C_subject_identity": "",
    "C_action_state": "",
    "C_visual_aesthetics": "",
    "C_spatial_composition": "",
    "C_mood_atmosphere": "",
    "C_temporal_context": ""
  }
}

Disjoint vocabulary axes:
- C_subject_identity: WHO or WHAT is in the scene. Use primarily nouns
  (people, animals, vehicles, named objects). 15-20 words. NO verbs of
  action, NO color words, NO mood words.
- C_action_state: ONLY what is happening or being done. Use verbs,
  adverbs, body-position words. Refer to subjects with PRONOUNS only
  (they, it, this). NO object nouns. 15-20 words.
- C_visual_aesthetics: ONLY visual qualities: colors, textures, patterns,
  materials, contrast. NO object nouns, NO actions. 15-20 words.
- C_spatial_composition: The image as a photographic composition:
  framing, depth, focus, perspective, foreground vs background, scale.
  NO subjects, NO colors. 15-20 words.
- C_mood_atmosphere: ONLY the emotional/atmospheric feeling. Use feeling
  words, ambient descriptors, energy levels. NO objects, NO actions,
  NO colors. 15-20 words.
- C_temporal_context: ONLY time-of-day, season, weather, lighting
  conditions, or temporal cues. NO subjects, NO actions, NO emotions.
  15-20 words.

Example output:
{
  "codebook_texts": {
    "C_subject_identity": "A young adult female with bold ink decorations enjoys casual outdoor leisure activity",
    "C_action_state": "Standing confidently while exhaling, posing in relaxed assertive bearing throughout the moment",
    "C_visual_aesthetics": "Vivid crimson contrasts deep emerald, accented by intricate multicolored ink patterns",
    "C_spatial_composition": "Centered foreground subject occupies mid-frame, soft background blur creating depth",
    "C_mood_atmosphere": "Casual confidence radiates with warm relaxation and quiet contemplative summery energy",
    "C_temporal_context": "Late afternoon golden sunlight bathes the summertime informal gathering with warmth"
  }
}
"""

V5_KEYS = (
    "C_subject_identity",
    "C_action_state",
    "C_visual_aesthetics",
    "C_spatial_composition",
    "C_mood_atmosphere",
    "C_temporal_context",
)


def _load_qwen(device):
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    mdl = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    return mdl, proc


def _strip(raw):
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip()
        nl = s.find("\n")
        if nl != -1 and s[:nl].strip().lower() in ("json", ""):
            s = s[nl + 1:]
        s = s.rstrip("`").strip()
    return s


@torch.no_grad()
def gen_v5(img_path, mdl, proc):
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": V5_PROMPT},
    ]}]
    chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[chat], images=[img], return_tensors="pt", padding=True).to(mdl.device)
    out = mdl.generate(**inp, max_new_tokens=768, do_sample=False)
    gen = out[:, inp.input_ids.shape[1]:]
    raw = proc.batch_decode(gen, skip_special_tokens=True)[0]
    try:
        return json.loads(_strip(raw))
    except Exception as e:
        return {"_error": str(e), "_raw": raw}


def main():
    out_jsonl = Path("/home/yschoi/GroundedDNA/cache/flickr25k_qwen_v5_smoke.jsonl")
    n_images = 50
    seed = 42

    # Pick image paths
    v4_path = Path("/home/yschoi/GroundedDNA/cache/flickr25k_qwen_v4.jsonl")
    v4_records = []
    with open(v4_path) as f:
        for line in f:
            v4_records.append(json.loads(line))

    rng = np.random.default_rng(seed)
    idx_pick = rng.choice(len(v4_records), n_images, replace=False)
    picked = [v4_records[i] for i in idx_pick]
    print(f"picked {len(picked)} images, ids: {[r['image_id'] for r in picked[:3]]}...")

    # Load Qwen
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES set externally to use GPU 1
    print(f"loading Qwen on {device}...")
    t0 = time.time()
    mdl, proc = _load_qwen(device)
    print(f"  Qwen loaded in {time.time()-t0:.0f}s")

    # Generate V5 captions
    print(f"generating V5 captions for {n_images} images...")
    v5_records = []
    n_parse_fail = 0
    t0 = time.time()
    for i, rec in enumerate(picked):
        img_path = rec["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join("/home/yschoi/GroundedDNA/dataset/Flickr25k", img_path)
        d = gen_v5(img_path, mdl, proc)
        new_rec = {
            "image_id": rec["image_id"],
            "image_path": rec["image_path"],
            "v4_codebook_texts": rec.get("codebook_texts", {}),
        }
        if "_error" in d:
            n_parse_fail += 1
            new_rec["v5_codebook_texts"] = {}
            new_rec["v5_parse_error"] = d["_error"]
            new_rec["v5_raw"] = d["_raw"][:500]
        else:
            new_rec["v5_codebook_texts"] = d.get("codebook_texts", {})
        v5_records.append(new_rec)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_images - i - 1)
            print(f"  [{i+1}/{n_images}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, parse_fail={n_parse_fail}")

    # Save
    with open(out_jsonl, "w") as f:
        for r in v5_records:
            f.write(json.dumps(r) + "\n")
    print(f"saved -> {out_jsonl}")

    # Free Qwen, load CLIP
    del mdl, proc
    torch.cuda.empty_cache()
    print("\nloading CLIP for encoding...")
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    clip = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch16"
    ).to(device).eval()

    @torch.no_grad()
    def encode(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = clip(**enc).text_embeds.cpu().numpy()
        n = np.linalg.norm(out, axis=-1, keepdims=True)
        return out / np.maximum(n, 1e-9)

    # Cross-slot cos for V5 (skip parse-failed records)
    valid = [r for r in v5_records if r["v5_codebook_texts"] and all(k in r["v5_codebook_texts"] for k in V5_KEYS)]
    print(f"\nvalid V5 records: {len(valid)}/{n_images} ({n_parse_fail} parse failures)")
    if len(valid) < 20:
        print("ERROR: too few valid records, V5 prompt may need debugging")
        return

    v5_texts = []
    for r in valid:
        for k in V5_KEYS:
            v5_texts.append(r["v5_codebook_texts"][k])
    v5_emb = encode(v5_texts).reshape(len(valid), 6, -1)
    G_v5 = np.einsum("nmd,nkd->nmk", v5_emb, v5_emb).mean(0)
    od_v5 = G_v5[~np.eye(6, dtype=bool)]

    # V4 baseline for same images
    v4_slots = ("C_global", "C_primary_object", "C_secondary_object",
                "C_activity_or_relation", "C_color_texture", "C_scene_type")
    v4_texts = []
    for r in valid:
        for k in v4_slots:
            v4_texts.append(r["v4_codebook_texts"][k])
    v4_emb = encode(v4_texts).reshape(len(valid), 6, -1)
    G_v4 = np.einsum("nmd,nkd->nmk", v4_emb, v4_emb).mean(0)
    od_v4 = G_v4[~np.eye(6, dtype=bool)]

    print("\n=== Same-image cross-slot mean cos (N={}) ===".format(len(valid)))
    print(f"  V4 (matched 50 images):  {od_v4.mean():.3f}")
    print(f"  V5 (same 50 images):     {od_v5.mean():.3f}")
    print(f"  Reduction:               {od_v4.mean() - od_v5.mean():+.3f}")
    print(f"\n=== Reference points ===")
    print(f"  V4 full Flickr25k:       0.665")
    print(f"  CLIP intrinsic floor:    0.500 (theoretical lower bound)")

    # Per-slot pair cos
    print(f"\n=== V5 cross-slot cos matrix ===")
    for r in G_v5:
        print("  " + " ".join(f"{v:6.3f}" for v in r))
    print(f"  min: {od_v5.min():.3f}, max: {od_v5.max():.3f}, range: {od_v5.max()-od_v5.min():.3f}")

    # Decision
    print(f"\n=== Decision (V5 same-image cos) ===")
    cs = od_v5.mean()
    if cs <= 0.52:
        verdict = "GREEN: full 25K re-extraction (8-12h)"
    elif cs <= 0.58:
        verdict = "TUNE: prompt revision, then re-test"
    elif cs <= 0.64:
        verdict = "MARGINAL: small gain; weigh cost vs benefit"
    else:
        verdict = "NO-GO: no meaningful improvement over V4"
    print(f"  cos={cs:.3f} -> {verdict}")

    # Save summary
    summary_path = Path("/home/yschoi/GroundedDNA/docs/v5_small_scale_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_images": n_images,
            "n_valid": len(valid),
            "n_parse_fail": n_parse_fail,
            "v4_same_50_cos": float(od_v4.mean()),
            "v5_same_50_cos": float(od_v5.mean()),
            "v5_cos_matrix": G_v5.tolist(),
            "v5_cos_min": float(od_v5.min()),
            "v5_cos_max": float(od_v5.max()),
            "verdict": verdict,
        }, f, indent=2)
    print(f"\nsaved summary -> {summary_path}")


if __name__ == "__main__":
    main()
