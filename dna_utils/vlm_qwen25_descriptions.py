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


# --- V6 (CUB-200 anatomical, single-object fine-grained) ------------------
# CUB-200 is a single-bird fine-grained dataset, so the V2/V4/V5b "scene"
# axes (primary/secondary object, activity, scene type) do not apply.
# Instead the six slots map onto disjoint BIRD ANATOMICAL regions. The schema
# below uses CUB-specific key names (C_head_bill etc.). The downstream feature
# extractors (extract_clip_text_features.py / _token_features.py) and the
# dataloader are POSITION-based on the V3/V4 key tuple, so the caption
# generator (tools/qwen3_v6_cub_trainset.py) re-maps V6 keys onto the V3V4
# key positions before caching:
#     C_global          -> C_global
#     C_head_bill       -> C_primary_object
#     C_upperparts_wing -> C_secondary_object
#     C_underparts      -> C_activity_or_relation
#     C_tail_appendages -> C_color_texture
#     C_background       -> C_scene_type
# The original V6 keys are preserved in a side field for interpretability.
CODEBOOK_KEYS_V6_CUB: Tuple[str, ...] = (
    "C_global",
    "C_head_bill",
    "C_upperparts_wing",
    "C_underparts",
    "C_tail_appendages",
    "C_background",
)

# V6b CUB keys -- replaces C_background with C_pattern_markings (the discrete
# field marks slot). Same positional mapping onto V3V4 downstream keys, except
# slot 5 carries pattern_markings instead of background.
CODEBOOK_KEYS_V6B_CUB: Tuple[str, ...] = (
    "C_global",
    "C_head_bill",
    "C_upperparts_wing",
    "C_underparts",
    "C_tail_appendages",
    "C_pattern_markings",
)

_PROMPT_V6_CUB = """\
You are a fine-grained bird morphology parser. Given the image, output
a single JSON object containing six short, MUTUALLY-DISTINCT
caption-style sentences.

These sentences will be embedded by a frozen CLIP text encoder and used
as independent semantic anchors for six different visual codebooks.
Each local sentence must therefore describe ONLY its assigned
anatomical or environmental region.

Length and form:
- Output ONLY a single valid JSON object. No prose, no markdown fences.
- Every field must contain exactly one natural English sentence.
- Each sentence must contain 10-15 words.
- Use precise, visible, fine-grained visual evidence without generic padding.
- NEVER name or infer the bird species, genus, family, or dataset class.
- NEVER output "none".
- If a region is partly or fully occluded, describe its visibility,
  visible contour, or occlusion without inventing hidden attributes.
- "codebook_texts" MUST contain EXACTLY the six listed keys.

Cross-slot distinctness rules (CRITICAL):
- Every non-global sentence must focus on a DIFFERENT anatomical or
  environmental region.
- Do NOT paraphrase the global summary in any local slot.
- Do NOT describe the same visual cue in multiple local slots.
- Do NOT repeat content nouns or adjective phrases across local slots,
  unless repetition is unavoidable for factual correctness.
- Never invent a different color, shape, or pattern merely to avoid repetition.
- If the same color covers several regions, mention it only in the most
  diagnostic region; describe shape, pattern, contrast, or visibility elsewhere.
- Local slots must not use generic nouns such as "bird", "animal",
  "subject", "image", "photo", or "plumage".
- Each local sentence must contain at least two anchor terms belonging
  to its own anatomical vocabulary.
- Anatomical anchor terms assigned to another slot are forbidden.
- Function words may repeat, but distinctive visual phrases should not.

Disjoint anatomical vocabulary domains:

| Axis                    | Required focus and allowed anchor vocabulary                  |
| ----------------------- | ------------------------------------------------------------- |
| C_global                | silhouette, stature, proportions, posture, viewpoint, profile |
| C_head_bill             | crown, forehead, eye, eye-ring, face, bill, beak, crest        |
| C_upperparts_wing       | nape, mantle, back, shoulder, wing, covert, feather, wing-bar  |
| C_underparts            | throat, breast, chest, belly, flank, streak, spot, barring     |
| C_tail_appendages       | tail, undertail, band, fork, leg, foot, claw, rear contour     |
| C_background            | branch, foliage, sky, water, ground, perch, blur, backdrop     |

Hard forbidden cross-axis terms in NON-global slots:
- C_head_bill:
  NO wing, back, mantle, breast, belly, flank, tail, leg, foot,
  branch, foliage, sky, water, or ground terms.
- C_upperparts_wing:
  NO crown, eye, bill, beak, throat, breast, belly, flank,
  tail, leg, foot, or background terms.
- C_underparts:
  NO crown, eye, bill, beak, wing, back, mantle, tail,
  leg, foot, or background terms.
- C_tail_appendages:
  NO crown, eye, bill, beak, wing, back, mantle, throat,
  breast, belly, flank, or background terms.
- C_background:
  NO bird anatomy, body-part, feather-pattern, posture,
  species, or taxonomic terms.

Schema:
{
  "codebook_texts": {
    "C_global": "",
    "C_head_bill": "",
    "C_upperparts_wing": "",
    "C_underparts": "",
    "C_tail_appendages": "",
    "C_background": ""
  }
}

Per-axis fine-grained instructions:

- C_global (10-15 words):
  Describe the WHOLE visible form using silhouette, relative size,
  body proportions, posture, orientation, and viewpoint.
  You may mention a broad overall palette, but do not list detailed
  head, wing, breast, tail, leg, or background markings.

- C_head_bill (10-15 words):
  Describe ONLY the crown, forehead, eyes, eye-ring, facial markings,
  crest, bill, or beak.
  Focus on bill length, thickness, curvature, tip shape, head contour,
  eye markings, local color, and facial contrast.
  Begin with a head-region anatomical noun phrase.

- C_upperparts_wing (10-15 words):
  Describe ONLY the nape, mantle, back, shoulders, and visible wings.
  Focus on wing shape, folded or spread structure, feather layering,
  bars, spots, streaks, patches, edging, and local contrast.
  Begin with an upperparts or wing anatomical noun phrase.

- C_underparts (10-15 words):
  Describe ONLY the throat, breast, chest, belly, and visible flanks.
  Focus on local hue, gradients, streaking, spotting, barring,
  patches, contrast boundaries, and underpart shape.
  Begin with an underpart anatomical noun phrase.

- C_tail_appendages (10-15 words):
  Describe the tail first, including length, width, shape, spread,
  fork, banding, edging, and contrast.
  Add visible legs, feet, claws, or grip only when reliable.
  If hidden, describe the visible rear contour or degree of occlusion.
  Begin with a tail or rear-appendage anatomical noun phrase.

- C_background (10-15 words):
  Describe ONLY non-bird visual evidence such as branches, foliage,
  sky, water, ground, perch surfaces, depth blur, clutter,
  illumination, and environmental backdrop.
  Do not mention any bird anatomy, feathering, markings, posture,
  or possible species.

Example:
{
  "codebook_texts": {
    "C_global": "A compact upright form presents balanced proportions, side profile, and muted overall tones.",
    "C_head_bill": "Dark crown, pale eye-ring, and slender pointed bill define the rounded face.",
    "C_upperparts_wing": "Layered brown mantle and folded wings display two narrow contrasting feather bars.",
    "C_underparts": "Cream throat blends into a lightly streaked breast and smooth pale belly.",
    "C_tail_appendages": "Narrow tail carries faint bands above thin legs with delicate gripping feet.",
    "C_background": "Soft foliage, diagonal branches, and shallow green blur form a quiet woodland backdrop."
  }
}
"""


# CUB-200 V6b -- anatomical morphology + dedicated diagnostic markings.
#
# C_background is removed from the final semantic codebooks because
# background appearance is often label-irrelevant and may increase DNA
# code collision.
#
# C_pattern_markings exclusively owns discrete field marks such as
# eye-rings, facial stripes, wing bars, spots, streaks, barring, and
# tail bands.
#
# CRITICAL:
# The anatomical slots describe shape, structure, proportions, visibility,
# and broad base color only. They must NOT repeat discrete markings owned
# by C_pattern_markings.
#
# Recommended architecture:
# Keep a separate routing-only background/dustbin anchor, but do not
# quantize it and do not include it in the final DNA code.

_PROMPT_V6B_CUB = """\
You are a fine-grained bird morphology parser. Given the image, output
a single JSON object containing six short, MUTUALLY-DISTINCT
caption-style sentences.

These sentences will be embedded by a frozen CLIP text encoder and used
as independent semantic anchors for six different visual codebooks.
Each sentence must therefore encode a different and non-redundant type
of visual evidence.

Length and form:
- Output ONLY a single valid JSON object. No prose, no markdown fences.
- Every field must contain exactly one natural English sentence.
- Each sentence must contain 10-16 words.
- Describe only clearly visible evidence; never invent hidden attributes.
- NEVER name or infer the bird species, genus, family, or dataset class.
- NEVER output "none".
- If a body region is occluded, describe its visible contour, partial
  structure, or degree of occlusion without inventing details.
- "codebook_texts" MUST contain EXACTLY the six listed keys.

Cross-slot distinctness rules (CRITICAL):
- Every sentence must describe a DIFFERENT visual evidence type.
- Do NOT paraphrase the global summary in any local slot.
- Do NOT repeat the same visual phrase across multiple local slots.
- Do NOT move a marking into an anatomical slot merely to fill space.
- Local anatomical slots describe morphology, proportions, visibility,
  feather structure, and broad base tone only.
- C_pattern_markings exclusively describes discrete markings, boundaries,
  stripes, bars, spots, streaks, patches, rings, bands, and edging.
- If the same base color appears across several regions, mention it only
  where it is most visually informative.
- Local slots must not use generic words such as "bird", "animal",
  "subject", "image", "photo", or "plumage".
- Function words may repeat, but discriminative content phrases should not.

Disjoint semantic domains:

| Axis                    | Allowed semantic focus                                      |
| ----------------------- | ----------------------------------------------------------- |
| C_global                | silhouette, stature, proportions, posture, orientation      |
| C_head_bill             | head contour, bill geometry, eye placement, crest, base tone |
| C_upperparts_wing       | wing geometry, mantle structure, feather layering, base tone |
| C_underparts            | throat/chest/belly shape, volume, base hue, tonal gradient   |
| C_tail_appendages       | tail geometry, leg/foot structure, rear contour, base tone   |
| C_pattern_markings      | localized field marks, stripes, rings, bars, spots, bands    |

Pattern vocabulary reserved ONLY for C_pattern_markings:
- eye-ring
- eye-line
- supercilium
- eyebrow stripe
- crown stripe
- facial stripe
- malar stripe
- throat patch or bib
- wing bar
- wing patch
- spotting
- streaking
- barring
- mottling
- scalloping
- feather edging
- tail band
- terminal band
- contrasting tip
- sharp color boundary

Hard forbidden cross-axis terms:

- C_global:
  NO detailed eye-rings, facial stripes, wing bars, spots, streaks,
  barring, patches, tail bands, or other localized field marks.

- C_head_bill:
  NO wing, back, breast, belly, tail, leg, or background descriptions.
  NO eye-ring, eye-line, supercilium, crown stripe, facial stripe,
  malar stripe, spotting, streaking, patch, or band descriptions.
  Describe only head and bill morphology, proportions, visibility,
  and broad base tone.

- C_upperparts_wing:
  NO head, bill, throat, breast, belly, tail, leg, or background descriptions.
  NO wing bars, wing patches, spotting, streaking, barring, mottling,
  scalloping, contrasting edging, or discrete pattern descriptions.
  Describe only wing/back structure, feather layering, proportions,
  visibility, and broad base tone.

- C_underparts:
  NO head, bill, wing, back, tail, leg, or background descriptions.
  NO throat patch, bib, spots, streaks, barring, mottling, scalloping,
  or sharply bounded pattern descriptions.
  Describe only throat, breast, and belly shape, volume, visibility,
  broad base hue, and smooth tonal gradients.

- C_tail_appendages:
  NO head, bill, wing, back, throat, breast, belly, or background descriptions.
  NO tail bands, terminal bands, contrasting tips, spots, bars,
  or feather-edge pattern descriptions.
  Describe only tail geometry, spread, length, visible legs or feet,
  rear contour, visibility, and broad base tone.

- C_pattern_markings:
  May name anatomical locations ONLY to bind a visible marking to its
  correct region.
  NO whole-body silhouette, posture, size, viewpoint, habitat,
  background, lighting, or general scene descriptions.
  NO broad color inventory without an associated marking or contrast.

Schema:
{
  "codebook_texts": {
    "C_global": "",
    "C_head_bill": "",
    "C_upperparts_wing": "",
    "C_underparts": "",
    "C_tail_appendages": "",
    "C_pattern_markings": ""
  }
}

Per-axis fine-grained instructions:

- C_global (10-16 words):
  Describe the whole visible form using silhouette, relative size,
  body proportions, posture, orientation, and viewpoint.
  A broad overall palette may be mentioned, but do not list any
  localized markings or detailed anatomical features.

- C_head_bill (10-16 words):
  Describe only the shape and structure of the crown, forehead, face,
  eyes, crest, bill, or beak.
  Focus on bill length, thickness, curvature, tip shape, head contour,
  eye placement, visibility, and broad base tone.
  Do not describe any discrete facial or crown marking.

- C_upperparts_wing (10-16 words):
  Describe only the structural appearance of the nape, mantle, back,
  shoulders, and visible wings.
  Focus on wing length, folded or spread shape, feather layering,
  proportions, visibility, and broad base tone.
  Do not describe bars, spots, streaks, patches, or feather-edge patterns.

- C_underparts (10-16 words):
  Describe only the structural appearance of the throat, breast, chest,
  belly, and visible flanks.
  Focus on volume, contour, visibility, broad base hue, and smooth
  tonal transitions.
  Do not describe streaks, spots, barring, bibs, or localized patches.

- C_tail_appendages (10-16 words):
  Describe the tail's length, width, shape, spread, fork, and visible
  rear contour.
  Add reliable leg, foot, claw, or grip structure when visible.
  Do not describe bands, contrasting tips, edging, or localized markings.

- C_pattern_markings (10-16 words):
  Describe the one to three most diagnostic visible field marks and
  bind each marking to its anatomical location.
  Prioritize eye-rings, facial lines, crown stripes, throat patches,
  wing bars, wing patches, breast streaks, spots, barring, and tail bands.
  If no strong discrete marking is visible, describe clearly observed
  uniformity, an absent marking, or a sharp tonal boundary.
  State that a marking is absent only when the relevant region is
  clearly visible; otherwise describe only visible evidence.

Example:
{
  "codebook_texts": {
    "C_global": "A compact upright form shows balanced proportions, side orientation, and muted overall tones.",
    "C_head_bill": "A rounded head supports a slender pointed bill with a gently tapered tip.",
    "C_upperparts_wing": "Folded wings show compact proportions, layered flight feathers, and a dark brown base tone.",
    "C_underparts": "The softly rounded chest transitions into a pale belly with smooth tonal shading.",
    "C_tail_appendages": "A narrow medium-length tail extends above thin legs with delicate gripping feet.",
    "C_pattern_markings": "A pale eye-ring, two wing bars, and fine breast streaks create distinct field marks."
  }
}
"""


# =============================================================================
# V7 CUB — 6 anatomy-only slots. Removes C_pattern_markings (20.7% hedged in v6b)
# and splits C_head_bill into C_head_face_eye + C_bill. Markings redistributed
# into their natural anatomy slots. Absolute prohibition on hedging language and
# on the "bird" family (including C_global — v6b had 55% "bird" occurrence there).
# =============================================================================

CODEBOOK_KEYS_V7_CUB: Tuple[str, ...] = (
    "C_global",
    "C_head_face_eye",
    "C_bill",
    "C_wing_back",
    "C_underparts",
    "C_tail_legs",
)

_PROMPT_V7_CUB = """\
You are a fine-grained morphology parser. Given the image, output a
single JSON object containing six short, MUTUALLY-DISTINCT
caption-style sentences.

These sentences will be embedded by a frozen CLIP text encoder and used
as independent semantic anchors for six different visual codebooks.
Each sentence must therefore encode a different and non-redundant type
of visual evidence.

Length and form:
- Output ONLY a single valid JSON object. No prose, no markdown fences.
- Every field must contain exactly one natural English sentence.
- Each sentence must contain 10-16 words.
- Describe only clearly visible evidence; never invent hidden attributes.
- NEVER name or infer the species, genus, family, or dataset class.
- "codebook_texts" MUST contain EXACTLY the six listed keys.

===============================================================
ABSOLUTELY FORBIDDEN VOCABULARY (across ALL slots, no exceptions)
===============================================================

Generic-subject words — banned in every slot including C_global:
- bird, birds, avian, avifauna, aviform
- animal, animals, creature, creatures
- specimen, specimens, subject, subjects
- individual, individuals, entity, entities
- feathered, feathery, plumage

If you would write "the bird", rewrite as "the visible form",
"the visible body", or a direct anatomical noun.

===============================================================
ABSOLUTELY FORBIDDEN LANGUAGE PATTERNS (across ALL slots)
===============================================================

No absence/uncertainty/hedging phrasing under any circumstance:
- "no distinct ..."         (banned)
- "no clear(ly) ..."        (banned)
- "no visible ..."          (banned)
- "not (clearly) visible"   (banned)
- "not discernible"         (banned)
- "not determinable"        (banned)
- "cannot be seen"          (banned)
- "cannot be determined"    (banned)
- "unclear ..."             (banned)
- "indeterminate ..."       (banned)
- "indiscernible ..."       (banned)
- "none visible"            (banned)
- "no (visible|apparent) markings/features/patterns"  (banned)

Positive substitution rule (MANDATORY):
If a feature is uncertain, occluded, or absent, describe the
VISIBLE CONTOUR, PARTIAL STRUCTURE, or SURFACE TEXTURE instead of
absence. Never write what is NOT there.

Examples:
  wrong: "No distinct wing bars visible."
  right: "Smooth uniform wing surface with layered dark feathers."
  wrong: "No visible markings on the belly."
  right: "Belly surface appears uniform with a soft tonal gradient."
  wrong: "Tail not clearly visible."
  right: "A short tail contour extends behind the body."

===============================================================
Cross-slot distinctness rules (CRITICAL)
===============================================================
- Every sentence must describe a DIFFERENT visual evidence type.
- Do NOT paraphrase the global summary in any local slot.
- Do NOT repeat the same visual phrase across multiple local slots.
- Discriminative content phrases should not repeat.
- If the same base color appears across regions, mention it only
  where it is most visually informative.
- Function words may repeat.

===============================================================
Disjoint semantic domains — anatomy only, no dedicated pattern slot
===============================================================

| Slot              | Allowed semantic focus                                                     |
| ----------------- | -------------------------------------------------------------------------- |
| C_global          | silhouette, stature, proportions, posture, orientation, overall base tone   |
| C_head_face_eye   | crown, forehead, face contour, eye placement + orbital ring/stripe markings |
| C_bill            | bill length, thickness, curvature, tip shape, base tone                     |
| C_wing_back       | mantle, back, wing geometry + wing bars, wing patches, feather edging       |
| C_underparts      | throat, breast, belly shape + throat patch, breast streaks, belly barring   |
| C_tail_legs       | tail geometry + tail bands, terminal band, visible legs and feet structure  |

Note: markings (bars, patches, stripes, rings, spots, streaks, bands)
are ALLOWED but ONLY where they naturally belong anatomically. Head
markings (eye-ring, eye-line, supercilium, crown stripe) go in
C_head_face_eye. Wing markings go in C_wing_back. Belly/throat markings
go in C_underparts. Tail bands go in C_tail_legs. C_bill and C_global
do NOT describe discrete markings.

Hard forbidden cross-slot topics:

- C_global:
  NO detailed field marks, eye-rings, facial stripes, wing bars,
  spots, streaks, barring, patches, tail bands.
  NO bill morphology details (C_bill's domain).
  Broad overall base palette may be mentioned.

- C_head_face_eye:
  NO wing, back, breast, belly, tail, leg, or background descriptions.
  NO bill morphology (that belongs in C_bill).
  DO describe: head contour, crown shape, forehead, face structure,
  eye placement, eye-ring, eye-line, supercilium, crown stripe,
  malar stripe, facial stripe.

- C_bill:
  NO head, face, eye, wing, back, breast, belly, tail, leg, or
  background descriptions.
  NO markings (bill's field is morphology, not markings).
  DO describe: bill length, thickness, curvature, tip shape,
  base color/tone.

- C_wing_back:
  NO head, bill, throat, breast, belly, tail, leg, or background.
  DO describe: wing geometry, back/mantle structure, feather
  layering, wing bars, wing patches, feather edging, contrasting
  wing pattern, base tone.

- C_underparts:
  NO head, bill, wing, back, tail, leg, or background.
  DO describe: throat, breast, belly shape/volume, throat patch,
  bib, breast streaks, belly barring, belly spotting, tonal
  gradient, base hue.

- C_tail_legs:
  NO head, bill, wing, back, throat, breast, belly, or background.
  DO describe: tail geometry, spread, length, tail bands, terminal
  band, contrasting tips, visible legs, feet, leg color, rear contour.

Schema:
{
  "codebook_texts": {
    "C_global": "",
    "C_head_face_eye": "",
    "C_bill": "",
    "C_wing_back": "",
    "C_underparts": "",
    "C_tail_legs": ""
  }
}

Per-slot fine-grained instructions:

- C_global (10-16 words):
  Whole visible form via silhouette, relative size, body
  proportions, posture, orientation, viewpoint, and overall
  base palette. No localized markings. No mention of "bird",
  "specimen", or any banned generic word.

- C_head_face_eye (10-16 words):
  Head morphology plus head-region markings. Describe crown
  shape, forehead slope, face contour, eye placement, and any
  eye-ring, eye-line, supercilium, crown stripe, malar stripe,
  or facial stripe. Do NOT describe the bill.

- C_bill (10-16 words):
  Bill morphology only: length relative to head, thickness at
  the base, curvature, tip shape (pointed/hooked/blunt),
  color base tone, cutting-edge contour. No markings.

- C_wing_back (10-16 words):
  Wing and upper-body structure plus their markings. Describe
  wing geometry (folded/spread, primary length), mantle base
  tone, feather layering, and wing bars, wing patches,
  contrasting feather edging when present.

- C_underparts (10-16 words):
  Ventral morphology plus ventral markings. Describe throat,
  breast, and belly shape, volume, tonal gradient, and any
  throat patch, bib, breast streaks, belly barring, belly
  spotting when present.

- C_tail_legs (10-16 words):
  Rear morphology plus tail markings and visible legs.
  Describe tail shape/length/spread, tail bands, terminal
  band, contrasting tips, and visible legs, foot color,
  or perching posture.

Correct output example:
{
  "codebook_texts": {
    "C_global": "A compact rounded form perches upright on a horizontal branch under soft daylight.",
    "C_head_face_eye": "Round head with pale eye-ring circling a small dark eye and faint supercilium stripe.",
    "C_bill": "Short conical bill with sharp pointed tip and pale grey base tone.",
    "C_wing_back": "Folded wings display two thin whitish wing bars over layered olive-brown feathers.",
    "C_underparts": "Pale yellow throat blends into a lightly streaked whitish breast and belly.",
    "C_tail_legs": "Short square tail shows two faint dark bands above slender pinkish perching legs."
  }
}
"""


# =============================================================================
# V7.1 CUB — refinement of V7 to eliminate residual hedging in anatomy slots.
# V7 sample 24 diagnosis showed:
#   - Forbidden vocab: 0 occurrences   ✅
#   - Pattern slot removed (v6b 20.7% hedging eliminated)   ✅
#   - BUT anatomy slots redistributed hedging: wing_back 37.5%, tail_legs 83.3%
#     because Qwen wrote "with no visible bars/bands/streaks" as factual
#     observations when markings were absent.
# V7.1 fix:
#   (a) Anatomy slot descriptions removed ALL "when present" markings mentions.
#   (b) Anatomy slots now describe MORPHOLOGY + SURFACE PROPERTIES only.
#   (c) Explicit ban on "with no ..." construction.
#   (d) Positive descriptor fallback vocabulary listed:
#       smooth, uniform, monochromatic, unbroken, plain, even, homogeneous.
# =============================================================================

CODEBOOK_KEYS_V7_1_CUB: Tuple[str, ...] = CODEBOOK_KEYS_V7_CUB

_PROMPT_V7_1_CUB = """\
You are a fine-grained morphology parser. Given the image, output a
single JSON object containing six short, MUTUALLY-DISTINCT
caption-style sentences.

These sentences will be embedded by a frozen CLIP text encoder and used
as independent semantic anchors for six different visual codebooks.
Each sentence must therefore encode a different and non-redundant type
of visual evidence.

Length and form:
- Output ONLY a single valid JSON object. No prose, no markdown fences.
- Every field must contain exactly one natural English sentence.
- Each sentence must contain 10-16 words.
- Describe only clearly visible evidence; never invent hidden attributes.
- NEVER name or infer the species, genus, family, or dataset class.
- "codebook_texts" MUST contain EXACTLY the six listed keys.

===============================================================
ABSOLUTELY FORBIDDEN VOCABULARY (across ALL slots, no exceptions)
===============================================================

Generic-subject words — banned in every slot including C_global:
- bird, birds, avian, avifauna, aviform
- animal, animals, creature, creatures
- specimen, specimens, subject, subjects
- individual, individuals, entity, entities
- feathered, feathery, plumage

If you would write "the bird", rewrite as "the visible form",
"the visible body", or a direct anatomical noun.

===============================================================
ABSOLUTELY FORBIDDEN LANGUAGE PATTERNS (across ALL slots)
===============================================================

No absence/uncertainty/hedging under any circumstance. All of the
following are banned:
- "no distinct ..."         (banned)
- "no clear(ly) ..."        (banned)
- "no visible ..."          (banned)
- "with no ..."             (banned — critical, this was the v7 leak)
- "without visible ..."     (banned)
- "not visible"             (banned)
- "not clearly ..."         (banned)
- "not discernible"         (banned)
- "not determinable"        (banned)
- "cannot be seen"          (banned)
- "cannot be determined"    (banned)
- "unclear ..."             (banned)
- "indeterminate ..."       (banned)
- "indiscernible ..."       (banned)
- "none visible"            (banned)
- "none apparent"           (banned)
- "no (visible|apparent) markings/features/patterns/bars/bands/spots/streaks/patches"  (banned)

CRITICAL RULE — do not negate. Never write what is NOT there. Write
only what IS there. If a feature would tempt an absence sentence,
substitute a POSITIVE descriptor of the visible surface instead.

Positive fallback vocabulary for uniform / unmarked surfaces:
- smooth, uniform, unbroken, plain, homogeneous, monochromatic
- even, consistent, level, seamless
- solid tone, single hue, continuous shade
- soft gradient, gentle transition, subtle blend

Examples of correct positive rewriting:
  wrong: "No distinct wing bars visible."
  right: "Smooth layered wing feathers with uniform dark base tone."

  wrong: "Belly appears smooth with no visible streaks."
  right: "Belly appears smooth with an even soft brownish tone."

  wrong: "Tail extends behind body with no visible bands."
  right: "Tail extends behind the body as a solid dark contour."

  wrong: "Uniform mantle with no visible feather edging."
  right: "Uniform mantle in a solid slate-grey tone."

  wrong: "Short tail contour extends behind body with no visible bands or leg structures."
  right: "Short tail contour extends behind body as a compact solid dark shape."

===============================================================
Cross-slot distinctness rules (CRITICAL)
===============================================================
- Every sentence must describe a DIFFERENT visual evidence type.
- Do NOT paraphrase the global summary in any local slot.
- Do NOT repeat the same visual phrase across multiple local slots.
- If the same base color appears across regions, mention it only
  where it is most visually informative.
- Function words may repeat; discriminative content phrases should not.

===============================================================
Disjoint semantic domains — anatomy only, no dedicated pattern slot
===============================================================

Each local slot describes MORPHOLOGY + SURFACE PROPERTIES. Markings
(bars, patches, stripes, rings, spots, streaks, bands) are ALLOWED
only when they are CLEARLY VISIBLE and PROMINENT. If markings are
absent or ambiguous, describe the surface as smooth / uniform /
monochromatic / etc. and do NOT mention the absence.

| Slot              | Allowed morphology + surface focus                              |
| ----------------- | --------------------------------------------------------------- |
| C_global          | silhouette, stature, proportions, posture, orientation, base tone |
| C_head_face_eye   | crown, forehead, face contour, eye placement, base surface tone   |
| C_bill            | bill length, thickness, curvature, tip shape, base tone           |
| C_wing_back       | mantle, back, wing geometry, feather layering, base tone          |
| C_underparts      | throat, breast, belly shape, volume, gradient, base hue           |
| C_tail_legs       | tail geometry + visible legs and feet, tail contour               |

Hard forbidden cross-slot topics:

- C_global:
  NO detailed field marks, eye-rings, facial stripes, wing bars,
  spots, streaks, barring, patches, tail bands.
  NO bill morphology details (C_bill's domain).
  Broad overall base palette may be mentioned.

- C_head_face_eye:
  NO wing, back, breast, belly, tail, leg, or background descriptions.
  NO bill morphology (that belongs in C_bill).
  DO describe head contour, crown shape, forehead, face structure,
  eye placement, and — if visibly present — eye-ring, eye-line,
  supercilium, crown stripe, malar stripe, facial stripe.

- C_bill:
  NO head, face, eye, wing, back, breast, belly, tail, leg, or
  background descriptions.
  DO describe bill length, thickness, curvature, tip shape,
  base color/tone.

- C_wing_back:
  NO head, bill, throat, breast, belly, tail, leg, or background.
  DO describe wing geometry, back/mantle structure, feather layering,
  base tone. If wing bars, wing patches, or contrasting edging are
  clearly visible and prominent, they may be described; otherwise
  describe the surface positively (smooth, uniform, layered).

- C_underparts:
  NO head, bill, wing, back, tail, leg, or background.
  DO describe throat/breast/belly shape/volume, tonal gradient,
  base hue. If a throat patch, bib, streaks, or barring are clearly
  visible and prominent, they may be described; otherwise describe
  the surface positively.

- C_tail_legs:
  NO head, bill, wing, back, throat, breast, belly, or background.
  DO describe tail geometry, spread, length, and visible legs
  or feet. If tail bands, terminal band, or contrasting tips are
  clearly visible and prominent, they may be described; otherwise
  describe the tail as a solid contour or single-tone shape.

Schema:
{
  "codebook_texts": {
    "C_global": "",
    "C_head_face_eye": "",
    "C_bill": "",
    "C_wing_back": "",
    "C_underparts": "",
    "C_tail_legs": ""
  }
}

Per-slot fine-grained instructions:

- C_global (10-16 words):
  Whole visible form via silhouette, relative size, body
  proportions, posture, orientation, viewpoint, and overall
  base palette. No localized markings. Never use "bird" or
  any banned generic word.

- C_head_face_eye (10-16 words):
  Head morphology. Describe crown shape, forehead slope, face
  contour, eye placement, and any clearly visible head markings.
  Do NOT describe the bill.

- C_bill (10-16 words):
  Bill morphology only: length relative to head, thickness at
  the base, curvature, tip shape (pointed/hooked/blunt),
  color base tone, cutting-edge contour.

- C_wing_back (10-16 words):
  Wing and upper-body structure. Describe wing geometry
  (folded/spread, primary length), mantle base tone, feather
  layering. Add clearly visible wing bars/patches only when
  prominent; otherwise describe the surface as smooth or uniform.

- C_underparts (10-16 words):
  Ventral morphology. Describe throat, breast, and belly
  shape, volume, and tonal gradient. Add clearly visible
  streaks/patches only when prominent; otherwise describe
  the surface positively.

- C_tail_legs (10-16 words):
  Rear morphology + visible legs. Describe tail shape, length,
  spread, and any visible legs/feet or perching contour.
  Add tail markings only when clearly visible; otherwise
  describe the tail as a solid dark or pale contour.

Correct output example (well-marked bird):
{
  "codebook_texts": {
    "C_global": "A compact rounded form perches upright on a horizontal branch under soft daylight.",
    "C_head_face_eye": "Round head with pale eye-ring circling a small dark eye and faint supercilium stripe.",
    "C_bill": "Short conical bill with sharp pointed tip and pale grey base tone.",
    "C_wing_back": "Folded wings display two thin whitish wing bars over layered olive-brown feathers.",
    "C_underparts": "Pale yellow throat blends into a lightly streaked whitish breast and belly.",
    "C_tail_legs": "Short square tail shows two faint dark bands above slender pinkish perching legs."
  }
}

Correct output example (uniform / unmarked bird):
{
  "codebook_texts": {
    "C_global": "A streamlined form glides low over water with broad wings extended horizontally.",
    "C_head_face_eye": "Smooth rounded crown slopes to a small dark eye set in a pale facial area.",
    "C_bill": "Long slender bill curves gently downward with a sharp pointed tip and dark tone.",
    "C_wing_back": "Layered wing feathers show a uniform dark brown base with smooth surface tone.",
    "C_underparts": "Uniform dark brown surface across throat, breast, and belly with soft even shading.",
    "C_tail_legs": "Short tail extends behind the body as a compact solid dark contour."
  }
}
"""


# --- V8 (scope-stratified, generic natural-photo datasets) ----------------
# The V2/V4/V5b axis set mixes three incompatible carving principles:
#   salience rank (primary vs secondary object), predicate (activity), and
#   attribute (colour) -- and the last two are LOGICALLY ARGUMENTS of the
#   first. A relation cannot be stated without naming its participants and a
#   colour cannot be stated without naming what is coloured, so V5b had to
#   forbid object nouns inside C_activity_or_relation / C_color_texture to
#   keep them disjoint, which strips those axes of their referents. The
#   measured cross-slot anchor cosine (.5742-.6693) is the residue of that
#   trade, not a prompt-tuning failure.
#
# V8 replaces the CATEGORY partition with a SCOPE partition of the frame:
#   subject / periphery / ground / backdrop. Three consequences:
#   1. No axis is an argument of another, so no forbidden-vocabulary rule is
#      needed -- disjointness comes from the referents not overlapping.
#   2. Attributes and actions live in the axis that OWNS them; there is no
#      separate colour axis and no separate action axis.
#   3. Every axis is always instantiable, because a region of the frame
#      cannot be absent the way a category can. This is what lets V8 forbid
#      "none" outright: a reserved ABSENT codeword would manufacture exactly
#      the codeword collisions the DNA-unique metric is already worst at.
# Slot 5 is intentionally unused -- the 5-slot config truncates the cached
# text slots to positions 0-4, so no sixth sentence is generated.
CODEBOOK_KEYS_V8: Tuple[str, ...] = (
    "C_global",
    "C_subject",
    "C_periphery",
    "C_ground",
    "C_backdrop",
)

_PROMPT_V8 = """\
You are a vision-language parser. Output a single JSON with five sentences.
Four of them describe four NON-OVERLAPPING PARTS of the frame. Every visible
pixel belongs to exactly one of C_subject, C_periphery, C_ground, C_backdrop.

Parts:
- C_subject   : the largest, sharpest, most prominent content in the frame --
                whatever the photograph is primarily showing.
- C_periphery : content beside or around the subject, at the frame edges or
                at a similar distance -- companions, accessories, nearby items.
- C_ground    : the lower region and the surface things rest on -- ground,
                floor, road, water, table, sand, snow.
- C_backdrop  : the upper and far region behind everything -- sky, wall,
                distant scenery, out-of-focus background.
- C_global    : the image as a whole -- scene category, place, time of day,
                weather, ambient light, mood, viewpoint.

Rules:
- Describe ONLY your own part of the frame. Do not mention content that
  belongs to another part.
- Within your part, describe everything about it: what it is, its shape,
  size, COLOR, MATERIAL, TEXTURE, and what it is DOING. Attributes and
  actions belong to the part that owns them -- there is no separate colour
  slot and no separate action slot.
- Every part always exists, because it is a region of the frame, not a
  category of object. If your region holds no distinct object, describe the
  surface, material, colour or blur that occupies it. NEVER output "none",
  and NEVER borrow content from another part.
- Each sentence is 10-15 words, concrete and specific. No generic filler.
- Output ONLY a single JSON object. No prose, no markdown fences.

Schema:
{
  "codebook_texts": {
    "C_global": "",
    "C_subject": "",
    "C_periphery": "",
    "C_ground": "",
    "C_backdrop": ""
  }
}

Example (a cyclist on a coastal road at dusk):
{
  "codebook_texts": {
    "C_global": "Dusk coastal ride, low side-on view, cool breezy autumn evening light.",
    "C_subject": "Lone cyclist crouched low, red windbreaker taut, legs driving a black road bike.",
    "C_periphery": "A weathered timber guardrail and two dark leaning signposts crowd the right edge.",
    "C_ground": "Damp grey asphalt streaked with faded white paint and shallow reflective puddles.",
    "C_backdrop": "Hazy violet sky over a soft blurred sea, thin band of amber cloud."
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
