"""Stage-2 caption pipeline prompts (text-path line, design record 2026-10-07 in docs/MODI_PROJECT_LOG.md).

The prompt texts below are fixed verbatim; every output row of the Stage-2 tools records
``sha256_of(<prompt>)`` so a caption file can be tied to the exact wording that produced it.
``tests/test_stage2_tools.py`` pins the digests.  Do not edit the texts in place -- a new wording is a
new constant (and a new prompt_version).

Step 2a history (all 2026-10-07):
  * ``PROMPT_CONCEPTS_UPDATE`` (cumulative list restated every batch) -- UNUSED: the restated list
    overran max_new_tokens on NUS-WIDE (reply cut mid-string) and oscillated on Flickr25k (69 -> 46 -> 81).
  * ``PROMPT_CONCEPTS_BATCH_V1`` / ``PROMPT_CONCEPTS_CONSOLIDATE_V1`` ({name, examples} objects) --
    UNUSED: Flickr returned one "concept" per image (summaries, not concepts; union 633 after 18
    batches), NUS-WIDE fell into a greedy repetition loop (one 26k-character line), MS-COCO produced
    malformed JSON mid-object that an ending-only repair cannot fix.
  * ``PROMPT_CONCEPTS_BATCH`` / ``PROMPT_CONCEPTS_CONSOLIDATE`` (current): flat string lists, names
    only, with the clarification sentence "A concept names something that can recur ...".

Placeholders: ``<PHRASES>``, ``<CONCEPTS>``, ``<G1>``..``<G3>`` (text steps) and ``<A1>``..``<A4>``,
``<D1>``..``<D4>`` (caption step; attribute key names and definitions from attributes.json).
"""
from __future__ import annotations

import hashlib

PROMPT_VERSION = "v10"

PROMPT_SURVEY = """You are describing one image from a photo collection. List what you can
actually see in it, as short phrases of one to four words. Include anything
a person might use to find this image again among the other images of the
same collection. Each phrase must point to a specific thing, part, or region
you can see. Phrases, not sentences; no "there is" or "image of". As many
phrases as the image supports, at most 12; fewer is fine.
Output ONLY a JSON object: {"phrases": ["...", "..."]}.
Do not guess what is outside the image."""

# UNUSED since 2026-10-07 (kept for the record; see module docstring).
PROMPT_CONCEPTS_UPDATE = """Below are phrases describing images from one photo collection, one line per
image, followed by the concepts found so far. Update the concept list: add
concepts that are useful for telling the images in this collection apart,
and keep the list free of concepts that are not visual, that mean the same
thing as another concept (when two concepts differ only in wording, keep one
name and list the other as an example), or that apply to nearly every image
in the collection or to almost none. Output ONLY a JSON object:
{"concepts": [{"name": "...", "examples": ["...", "..."]}]}.

Phrases:
<PHRASES>

Concepts found so far:
<CONCEPTS>"""

# UNUSED since 2026-10-07 (kept for the record; see module docstring).
PROMPT_CONCEPTS_BATCH_V1 = """Below are phrases describing images from one photo collection, one line per image. From these, list the visual concepts that are useful for telling the images in this collection apart. Keep concepts that are visible in the image itself; drop concepts that are not visual, that mean the same thing as another concept (when two concepts differ only in wording, keep one name and list the other as an example), or that apply to nearly every image in the collection or to almost none. Output ONLY a JSON object: {"concepts": [{"name": "...", "examples": ["...", "..."]}]}.

Phrases:
<PHRASES>"""

# UNUSED since 2026-10-07 (kept for the record; see module docstring).
PROMPT_CONCEPTS_CONSOLIDATE_V1 = """Below is a list of visual concepts collected from one photo collection in several passes; it contains duplicates and noise. Consolidate it: merge concepts that mean the same thing (keep one name, list the others as examples), drop concepts that are not visual, and drop concepts that apply to nearly every image in the collection or to almost none. Output ONLY a JSON object: {"concepts": [{"name": "...", "examples": ["...", "..."]}]}.

Concepts:
<CONCEPTS>"""

PROMPT_CONCEPTS_BATCH = """Below are phrases describing images from one photo collection, one line per image. A concept names something that can recur across many images in the collection, not a description of one image; keep each concept to one to three words. From these, list the visual concepts that are useful for telling the images in this collection apart. Keep concepts that are visible in the image itself; drop concepts that are not visual, that mean the same thing as another concept (when two concepts differ only in wording, keep one name and list the other as an example), or that apply to nearly every image in the collection or to almost none. Output ONLY a JSON object: {"concepts": ["...", "..."]}.

Phrases:
<PHRASES>"""

PROMPT_CONCEPTS_CONSOLIDATE = """Below is a list of visual concepts collected from one photo collection in several passes; it contains duplicates and noise. A concept names something that can recur across many images in the collection, not a description of one image; keep each concept to one to three words. Consolidate it: merge concepts that mean the same thing (keep one name, list the others as examples), drop concepts that are not visual, and drop concepts that apply to nearly every image in the collection or to almost none. Output ONLY a JSON object: {"concepts": ["...", "..."]}.

Concepts:
<CONCEPTS>"""

PROMPT_VISUAL_CHECK = """For each concept below, answer whether it can be seen in a photo without
outside knowledge. Output ONLY a JSON object:
{"concepts": [{"name": "...", "visible": true}]}.

Concepts:
<CONCEPTS>"""

PROMPT_ATTRIBUTES = """Below is a list of visual concepts found in one photo collection. Group them
into exactly 4 attributes. Each attribute should be a kind of visual
information that can be described for most images in the collection and
whose description differs from image to image; the 4 attributes should
overlap as little as possible, and each attribute should be describable
without naming what another attribute describes. If fewer than 4 clearly
distinct attributes exist, say so in a "note" field. For each attribute give
a short name, a one-sentence definition, and the concepts that belong to it.
Output ONLY a JSON object:
{"attributes": [{"name": "...", "definition": "...", "concepts": ["..."]}],
 "note": ""}.

Concepts:
<CONCEPTS>"""

PROMPT_ATTRIBUTES_MERGE = """Below are three groupings of the same visual concepts into 4 attributes,
produced independently. Merge them into one grouping of exactly 4 attributes
that best agrees with all three, keeping the same requirements: each
attribute can be described for most images in the collection, its description
differs from image to image, the attributes overlap as little as possible, and
each is describable without naming what another describes. Output ONLY a JSON
object: {"attributes": [{"name": "...", "definition": "...", "concepts": ["..."]}], "note": ""}.

Grouping 1:
<G1>

Grouping 2:
<G2>

Grouping 3:
<G3>"""

PROMPT_CAPTIONS_TEMPLATE = """You are a vision-language parser. Given the image, output a single JSON
object with one short caption for each of the four attributes defined below,
followed by one whole-image caption.

Rules:
- Output ONLY a single JSON object. No prose, no markdown fences.
- Each attribute field: about 5 to 10 words, describing only that attribute
  as it appears in this image, and keep it to what is actually in the image.
- "C_global": one sentence about the whole image, 10-15 words, most
  important content first.
- Every field must name something that is visible in the image. If the
  attribute is not obvious in this image, describe the nearest visible thing
  of that kind; never state that something is absent or unclear.
- Do not repeat the same content in two fields.
- Do not begin a field with the attribute name or a label.

Schema:
{"codebook_texts": {"<A1>": "", "<A2>": "", "<A3>": "", "<A4>": "", "C_global": ""}}

Attributes:
- <A1>: <D1>
- <A2>: <D2>
- <A3>: <D3>
- <A4>: <D4>"""

ALL_PROMPTS = {
    "survey": PROMPT_SURVEY,
    "concepts_update": PROMPT_CONCEPTS_UPDATE,
    "concepts_batch_v1": PROMPT_CONCEPTS_BATCH_V1,
    "concepts_consolidate_v1": PROMPT_CONCEPTS_CONSOLIDATE_V1,
    "concepts_batch": PROMPT_CONCEPTS_BATCH,
    "concepts_consolidate": PROMPT_CONCEPTS_CONSOLIDATE,
    "visual_check": PROMPT_VISUAL_CHECK,
    "attributes": PROMPT_ATTRIBUTES,
    "attributes_merge": PROMPT_ATTRIBUTES_MERGE,
    "captions_template": PROMPT_CAPTIONS_TEMPLATE,
}


def sha256_of(prompt: str) -> str:
    """Hex sha256 of the exact prompt text (utf-8)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_caption_prompt(attributes: list) -> str:
    """Fill PROMPT_CAPTIONS_TEMPLATE from the ordered 4-attribute list of attributes.json.

    ``attributes[i]`` must carry ``key`` (JSON-key form, used verbatim as the output key) and
    ``definition``.  The filled prompt is what the VLM sees; its sha256 is what caption rows record.
    """
    if len(attributes) != 4:
        raise ValueError(f"caption prompt needs exactly 4 attributes, got {len(attributes)}")
    text = PROMPT_CAPTIONS_TEMPLATE
    for i, a in enumerate(attributes, start=1):
        key, definition = str(a["key"]), str(a["definition"]).strip()
        if not key or not definition:
            raise ValueError(f"attribute {i} has an empty key or definition: {a!r}")
        text = text.replace(f"<A{i}>", key).replace(f"<D{i}>", definition)
    return text


if __name__ == "__main__":
    for name, p in ALL_PROMPTS.items():
        print(f"{name:26s} {sha256_of(p)}")
