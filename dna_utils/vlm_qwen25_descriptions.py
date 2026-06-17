"""Qwen-VL utility for OFFLINE structured description generation.

Default model switched from Qwen2.5-VL-7B-Instruct → Qwen3-VL-8B-Instruct
(2026-06-01). Qwen3-VL gives ~4× faster per-image inference (≈6-7 s vs
≈25-30 s on a 48 GB GPU at bfloat16) and slightly more compliant V4-prompt
JSON output. Function names retain the legacy "qwen25" suffix for
backward compatibility — internal model class is Qwen3VLForConditionalGeneration.

Use this as a PREPROCESSING step. DO NOT call from inside the training loop —
it loads an 8B VLM and is far too slow per step. The recommended workflow is:

    1) for each image in the dataset, run `generate_object_centric_scene_graph`
    2) save the resulting JSON to disk (one .json per image, or a single
       parquet/jsonl indexed by image path)
    3) at training time, load the cached JSON and call `extract_codebook_texts`
       to get the six fixed-order codebook strings — those are then tokenized
       with the CLIP/SigLIP2 tokenizer and fed to the model as `part_input_ids`.

This file deliberately does NOT touch the dataloader or the training loop.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image


DEFAULT_VLM = "Qwen/Qwen3-VL-8B-Instruct"

# Order MUST match `model_siglip2.PART_ORDER`.
# --- V1 (legacy, anatomy-decomposition prompt) -----------------------------
# Forces a head/body/limb decomposition that maps cleanly onto single-object
# imagery (Flickr25k flowers/animals/landmarks) but causes redundancy when
# applied to scene-heavy multi-object datasets (MSCOCO) -- C_head, C_body,
# C_limb all describe the same primary subject.
CODEBOOK_KEYS: Tuple[str, ...] = (
    "C_global",
    "C_head_or_main_part",
    "C_body_or_secondary_part",
    "C_limb_or_detail_part",
    "C_color_texture",
    "C_background_null",
)
# Alias requested by the project spec.
CODEBOOK_TEXT_KEYS: Tuple[str, ...] = CODEBOOK_KEYS

# --- V2 (Option C, scene-aware, multi-object friendly) --------------------
# Designed to give the six codebooks more orthogonal information axes so they
# do not collapse into paraphrases of one another on multi-object datasets.
# Position order is preserved so PART_ORDER / routing stay compatible; only
# the semantic interpretation per slot changes.
CODEBOOK_KEYS_V2: Tuple[str, ...] = (
    "C_global",
    "C_primary_object",
    "C_secondary_object",
    "C_activity_or_relation",
    "C_color_texture",
    "C_scene_type",
)


# ----------------------------------------------------------------- prompt
# Simplified: only the six per-codebook sentences. We deliberately drop the
# full object-centric scene-graph schema (objects/parts/relations/background)
# because the only thing this stage needs is one short sentence per codebook
# to drive the SigLIP2 text encoder.
_PROMPT = """\
You are a vision-language parser. Given the image, output a single JSON object
containing six short text descriptions, one for each semantic codebook.

Rules:
- Output ONLY a single JSON object. No prose, no markdown fences.
- All string fields must be plain English.
- Each string field should be one short sentence, at most ~30 words.
- If a field does not apply, use "none" or "not visible".
- "codebook_texts" MUST contain EXACTLY the six listed keys, no more, no less.

Schema:
{
  "codebook_texts": {
    "C_global": "",
    "C_head_or_main_part": "",
    "C_body_or_secondary_part": "",
    "C_limb_or_detail_part": "",
    "C_color_texture": "",
    "C_background_null": ""
  }
}

Meaning of each key:
- C_global: summarize the whole image, main objects, and scene.
- C_head_or_main_part: describe the head, front, or most discriminative main part of the primary object.
- C_body_or_secondary_part: describe the body, central region, or secondary object/region.
- C_limb_or_detail_part: describe limbs, wheels, wings, handles, small parts, or other fine details.
- C_color_texture: describe color, texture, material, and visual pattern.
- C_background_null: describe background, distractors, or state "none" if not useful.
"""


# V2 prompt (Option C, scene-aware -- BARE NOUN PHRASES). Six axes chosen so
# that they are as orthogonal as possible across multi-object scenes:
#   1. C_global              — full scene summary
#   2. C_primary_object      — dominant object CATEGORY (single noun)
#   3. C_secondary_object    — second most prominent object CATEGORY
#   4. C_activity_or_relation— action / interaction / spatial relation
#   5. C_color_texture       — colors, materials, textures (cross-cutting)
#   6. C_scene_type          — scene category (kitchen/sports field/etc.)
#
# Each field is constrained so the VLM is less tempted to paraphrase C_global
# across the other slots -- the per-axis instructions explicitly distinguish
# "object category" from "what is happening" from "where it takes place".
#
# KNOWN LIMITATION (-> see V3 below): the constraint to output bare noun
# phrases / short labels (e.g. "boat", "fishing", "indoor retail space") gives
# SigLIP2's text encoder very sparse input. SigLIP2 was trained on natural
# captions, not bare nouns, so its embedding of these short labels carries
# less discriminative info per slot than a sentence would.
_PROMPT_V2 = """\
You are a vision-language parser. Given the image, output a single JSON
object with six short, MUTUALLY-DISTINCT text descriptions. The six axes
are chosen so each captures a DIFFERENT aspect of the scene -- do NOT
paraphrase the global summary across the other fields.

Rules:
- Output ONLY a single JSON object. No prose, no markdown fences.
- All string fields must be plain English.
- Each string field should be at most ~20 words (shorter is fine).
- If a field does not apply (e.g. only one object in the image), use "none".
- "codebook_texts" MUST contain EXACTLY the six listed keys, no more, no less.

Schema:
{
  "codebook_texts": {
    "C_global": "",
    "C_primary_object": "",
    "C_secondary_object": "",
    "C_activity_or_relation": "",
    "C_color_texture": "",
    "C_scene_type": ""
  }
}

Meaning of each key (treat them as INDEPENDENT axes):
- C_global: one-sentence summary of the whole image (subject + setting + action). Up to 25 words.
- C_primary_object: the SINGLE most prominent object category as a short noun
  phrase, e.g. "golden retriever", "red sedan", "espresso cup". DO NOT describe
  attributes -- just the category. Use "none" if no clear primary object.
- C_secondary_object: the SECOND most prominent object category, same noun-phrase
  format as above. Use "none" if there is only one notable object.
- C_activity_or_relation: what is happening or the spatial relation between objects,
  e.g. "person riding bicycle", "cat next to laptop", "static product on shelf".
  Use "none" if it is a still scene with no clear action.
- C_color_texture: dominant colors, materials, and textures, e.g. "matte black
  metal with brushed silver accents". Do NOT name objects here.
- C_scene_type: scene category as a short label, e.g. "kitchen interior",
  "outdoor sports field", "urban street at night", "studio product shot",
  "macro photograph". Use "none" if completely abstract.
"""


# V3 prompt (Option D, scene-aware + DESCRIPTIVE SENTENCES). Same six axes
# as V2 but each is a short caption-style sentence (~15-25 words) rather
# than a bare noun / label, so the SigLIP2 text encoder gets the kind of
# input it was contrastively trained on. Each axis must focus on a
# DIFFERENT aspect to avoid the V1 "paraphrase C_global six times" failure.
_PROMPT_V3 = """\
You are a vision-language parser. Given the image, output a single JSON
object with six short, MUTUALLY-DISTINCT caption-style sentences. Each
sentence should sound like a natural image caption (not a bare label).

Rules:
- Output ONLY a single JSON object. No prose, no markdown fences.
- All string fields are full English sentences, 15-25 words each.
- Each sentence MUST focus on a DIFFERENT aspect of the image. Do NOT
  rephrase the global summary across the other slots.
- If an axis genuinely does not apply (e.g. only one notable object in
  the scene), use exactly "none".
- "codebook_texts" MUST contain EXACTLY the six listed keys.

Schema:
{
  "codebook_texts": {
    "C_global": "",
    "C_primary_object": "",
    "C_secondary_object": "",
    "C_activity_or_relation": "",
    "C_color_texture": "",
    "C_scene_type": ""
  }
}

Per-axis instructions (treat them as INDEPENDENT aspects):
- C_global: one-sentence summary of the WHOLE image: subject + setting +
  what is happening, as a natural caption.
- C_primary_object: one sentence that DESCRIBES the most prominent object
  in detail (its appearance, distinctive attributes, pose, condition).
  Mention the object only here; do not also describe it under C_secondary.
- C_secondary_object: one sentence describing the SECOND most prominent
  object with the same level of detail. Use exactly "none" if only one
  notable object exists.
- C_activity_or_relation: one sentence describing the action, interaction,
  or spatial relation between objects (e.g. "A person rides a red bicycle
  past parked cars on a wet street"). Use "none" for a static scene with
  no clear action.
- C_color_texture: one sentence describing dominant colors, materials, and
  textures of the visible surfaces. Do NOT name objects in this sentence.
- C_scene_type: one sentence describing the SETTING / environment / scene
  category (e.g. "An indoor commercial kitchen with stainless-steel
  countertops and overhead fluorescent lighting").
"""


# V4 prompt (Option E, retrieval-evidence axes, user A2 2026-05-19). Same
# six keys as V2/V3 but reframed as INDEPENDENT VISUAL EVIDENCE TYPES for
# retrieval. Key changes vs V3:
#   - explicit "every slot must be visually grounded" + "never output none"
#     pressures (V3 emits "none" for ~half of secondary-object slot, which
#     collapses cb2 into a single dominant codeword -- v43b extract showed
#     cb2 81% dead with 14216/23000 samples on one codeword).
#   - secondary slot falls back to other salient region / background /
#     texture / spatial cue instead of "none".
#   - axes reframed as evidence types to push Qwen toward genuine semantic
#     diversity rather than topic restatement.
_PROMPT_V4 = """\
You are a vision-language parser. Given the image, output a single JSON
object with six mutually distinct caption-style sentences. The six
sentences should describe different visual evidence types for retrieval.

Rules:
- Output ONLY a single JSON object. No prose, no markdown fences.
- All string fields are natural English sentences, 15-25 words each.
- Every slot must be visually grounded in the image.
- Every slot must focus on a different evidence type.
- Do not copy or paraphrase the same sentence across slots.
- Do not output "none" unless the image is blank or unrecognizable.
- If an object-specific slot is not applicable, describe a salient region,
  background element, texture, lighting cue, or spatial cue instead.
- "codebook_texts" MUST contain EXACTLY the six listed keys.

Schema:
{
  "codebook_texts": {
    "C_global": "",
    "C_primary_object": "",
    "C_secondary_object": "",
    "C_activity_or_relation": "",
    "C_color_texture": "",
    "C_scene_type": ""
  }
}

Independent evidence axes:
- C_global: A whole-image caption covering the main subject, setting, and
  overall visual event.
- C_primary_object: The main subject's identity, shape, pose, structure,
  or distinctive appearance.
- C_secondary_object: A secondary visual cue: another object, background
  item, salient region, or non-primary detail.
- C_activity_or_relation: The action, relation, spatial layout, viewpoint,
  or arrangement among visible elements.
- C_color_texture: Colors, materials, textures, patterns, and surface
  qualities only; avoid object names.
- C_scene_type: The scene category, environment, background, lighting,
  weather, or indoor/outdoor context.
"""


# v5b -- sentence-style fine-grained captions with STRICT disjoint
# vocabulary per axis. Motivated by 2026-06-17 finding that v4 MSCOCO
# captions have local↔local cosine 0.666 (Flickr 0.592), driving
# codebook redundancy NMI 0.726 and DNA-uniq 0.119 (Flickr v160b 0.625
# / 0.423). Length stays similar to v4 (10-15 words) but each axis is
# constrained to its own vocabulary domain with explicit "FORBIDDEN"
# lists. Goal: drop local↔local cosine to 0.50-0.55 without losing
# fine-grained semantic detail.
_PROMPT_V5b = """\
You are a vision-language parser. Output a single JSON with six
fine-grained sentences describing STRICTLY DISJOINT visual axes.

Length & form:
- Each sentence is 10-15 words. Use vivid sensory detail WITHIN your
  axis. Do NOT pad with generic phrases.
- Output ONLY a single JSON object. No prose, no markdown fences.
- Output "none" if the slot's evidence is genuinely absent (rare).

Disjoint vocabulary rule (CRITICAL):
Each axis has its OWN vocabulary domain. Words from another axis's
domain are FORBIDDEN except in C_global (the summary slot).

| Axis                   | Allowed vocabulary                              |
| ---------------------- | ----------------------------------------------- |
| C_primary_object       | object noun, body part, shape, size, structure  |
| C_secondary_object     | object noun, role, position (left/right/near)   |
| C_activity_or_relation | verb, motion, spatial relation, viewpoint, gesture |
| C_color_texture        | color hue, saturation, material, pattern, surface |
| C_scene_type           | scene category, location, environment, lighting, weather |
| C_global               | any (multi-domain summary)                      |

Hard forbidden cross-axis words in NON-global axes:
- C_primary_object / C_secondary_object: NO action verbs, NO color
  words, NO scene words.
- C_activity_or_relation: NO object nouns beyond pronouns (he/she/it/
  they), NO color words, NO scene words.
- C_color_texture: NO object nouns, NO action verbs, NO scene words.
- C_scene_type: NO object nouns, NO action verbs, NO color words.

Schema:
{
  "codebook_texts": {
    "C_global": "",
    "C_primary_object": "",
    "C_secondary_object": "",
    "C_activity_or_relation": "",
    "C_color_texture": "",
    "C_scene_type": ""
  }
}

Per-axis encouraged details (fine-grained guidance):
- C_global (10-15 words): summary of who/what/where/doing.
- C_primary_object (10-15 words): identity, body, build, posture,
  pose, attire silhouette, distinctive shape. NO color, NO action.
- C_secondary_object (10-15 words): identity, size, position relative
  to the primary, role in composition. NO color, NO action.
- C_activity_or_relation (10-15 words): dynamic verbs, spatial layout,
  interaction direction, viewpoint, gestures. NO object nouns beyond
  pronouns, NO color, NO scene.
- C_color_texture (10-15 words): dominant hues, contrast, saturation,
  material (matte / glossy / woven / synthetic), surface texture,
  lighting interaction. NO object nouns, NO action, NO scene.
- C_scene_type (10-15 words): indoor/outdoor category, specific
  location type, time of day, ambient lighting, weather, atmospheric
  quality. NO object nouns, NO action, NO color.

Example (soccer match):
{
  "codebook_texts": {
    "C_global": "Intense soccer moment with two opposing players competing closely on a turf surface.",
    "C_primary_object": "Tall lean athlete mid-stride, balanced posture, focused expression, jersey loose at the shoulders.",
    "C_secondary_object": "Small round ball spinning along the surface, positioned between both athletes' feet.",
    "C_activity_or_relation": "Dribbling forward as another approaches diagonally from the left to intercept the path.",
    "C_color_texture": "Vivid saturated yellow against deep blue, smooth synthetic surface, crisp white boundary lines.",
    "C_scene_type": "Enclosed indoor arena, bright artificial overhead floodlights, uniform shadows, evening atmosphere."
  }
}
"""


# ----------------------------------------------------------------- builder
def build_qwen25_vl_generator(
    model_name: str = DEFAULT_VLM,
    device: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    """Lazily import and return a (model, processor) tuple.

    Function name retained for backward compatibility; default model
    is now Qwen3-VL-8B-Instruct (transformers ≥ 5.x).

    Returns:
        (model, processor)  -- pass back to `generate_object_centric_scene_graph`
                               to avoid reloading the 8B weights every call.
    """
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as e:
        raise ImportError(
            "Qwen3-VL requires `transformers>=5.0`. Install with: "
            "`pip install 'transformers>=5.0' qwen-vl-utils accelerate`"
        ) from e

    processor = AutoProcessor.from_pretrained(model_name)
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, processor


# ----------------------------------------------------------------- helpers
def _load_image(image_path_or_pil: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image_path_or_pil, Image.Image):
        return image_path_or_pil.convert("RGB")
    return Image.open(image_path_or_pil).convert("RGB")


def _strip_code_fences(raw: str) -> str:
    """Remove leading/trailing ``` fences if the model emits them anyway."""
    s = raw.strip()
    if not s.startswith("```"):
        return s
    s = s.strip("`").lstrip()
    nl = s.find("\n")
    if nl != -1 and s[:nl].strip().lower() in ("json", ""):
        s = s[nl + 1:]
    return s.rstrip("`").strip()


# ----------------------------------------------------------------- generator
@torch.no_grad()
def generate_object_centric_scene_graph(
    image_path_or_pil: Union[str, Image.Image],
    model=None,
    processor=None,
    prompt: str = _PROMPT,
    max_new_tokens: int = 768,
    model_name: str = DEFAULT_VLM,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Run Qwen3-VL on one image and return the parsed scene-graph JSON.

    For batch / large-scale use, build (model, processor) ONCE with
    `build_qwen25_vl_generator` and pass them in.

    NOTE: this is OFFLINE preprocessing. DO NOT call inside a training step.
    """
    if model is None or processor is None:
        model, processor = build_qwen25_vl_generator(
            model_name=model_name, device=device,
        )

    image = _load_image(image_path_or_pil)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    cleaned = _strip_code_fences(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Qwen3-VL did not return valid JSON.\n"
            "---- raw output ----\n"
            f"{raw}\n"
            "--------------------\n"
            f"{e}"
        )


# ----------------------------------------------------------------- extractor
def extract_codebook_texts(description_json: Dict[str, Any]) -> List[str]:
    """Return the six codebook texts in fixed order.

    The order matches the part axis (M=6) consumed by
    `SigLIP2SemanticOTModel`:

        [0] C_global
        [1] C_head_or_main_part
        [2] C_body_or_secondary_part
        [3] C_limb_or_detail_part
        [4] C_color_texture
        [5] C_background_null

    Missing or non-string values are coerced to "".
    """
    cb = description_json.get("codebook_texts", {}) or {}
    out: List[str] = []
    for k in CODEBOOK_KEYS:
        v = cb.get(k, "")
        if v is None:
            v = ""
        if not isinstance(v, str):
            v = str(v)
        out.append(v)
    if len(out) != 6:
        raise ValueError(
            f"expected 6 codebook texts, got {len(out)}: {out}"
        )
    return out
