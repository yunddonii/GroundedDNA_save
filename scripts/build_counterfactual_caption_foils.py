#!/usr/bin/env python3
"""Build deterministic, local-slot minimal-pair caption foils.

This is an offline data-preparation utility.  It does not call a VLM and does
not alter the source caption cache.  For every local caption slot it tries to
replace exactly one supported semantic atom with a contrasting atom from the
same lexical family.  Slot 0 (``C_global``) is always invalid.

The generator is intentionally conservative:

* a foil changes one and only one character span;
* the replacement belongs to a role-compatible contrast set;
* the replacement atom must not already occur anywhere in the image's six
  factual captions (reduces obvious false negatives);
* captions without a supported edit are marked invalid rather than rewritten.

The output JSONL is consumed by ``extract_counterfactual_foil_features.py``.
It contains one independently changed caption per local slot, not a single
six-caption world in which all five slots were changed at once.

Example:

    python scripts/build_counterfactual_caption_foils.py \
        --input cache/flickr25k_qwen3_v4_trainset.jsonl \
        --output cache/flickr25k_qwen3_v4_trainset.foils.jsonl \
        --dry-run --max-rows 100 --show 3

Remove ``--dry-run`` after inspecting the reported slot coverage and examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "groundeddna-local-caption-foil-v1"

SLOT_KEYS_SCENE = (
    "C_global",
    "C_primary_object",
    "C_secondary_object",
    "C_activity_or_relation",
    "C_color_texture",
    "C_scene_type",
)

SLOT_KEYS_LEGACY = (
    "C_global",
    "C_head_or_main_part",
    "C_body_or_secondary_part",
    "C_limb_or_detail_part",
    "C_color_texture",
    "C_background_null",
)


@dataclass(frozen=True)
class AtomGroup:
    """A set of mutually contrasting surface forms."""

    name: str
    family: str
    terms: Tuple[str, ...]


# These are contrast sets, not synonym sets.  Inflected forms live in separate
# groups so a one-span substitution remains grammatical.  The vocabulary is
# deliberately finite: unsupported captions become invalid instead of being
# heuristically paraphrased.
ATOM_GROUPS: Tuple[AtomGroup, ...] = (
    # Object identity.
    AtomGroup("human_adult", "object", ("man", "woman")),
    AtomGroup("human_child", "object", ("boy", "girl")),
    AtomGroup("human_age", "object", ("adult", "child")),
    AtomGroup("pet", "object", ("cat", "dog")),
    AtomGroup("farm_animal", "object", ("horse", "cow", "sheep", "goat")),
    AtomGroup("road_vehicle", "object", ("car", "truck", "bus", "van")),
    AtomGroup("cycle", "object", ("bicycle", "motorcycle", "scooter")),
    AtomGroup("watercraft", "object", ("boat", "kayak", "canoe")),
    AtomGroup("furniture", "object", ("chair", "table", "bench")),
    AtomGroup("container", "object", ("cup", "bottle", "bowl")),
    AtomGroup("hand_object", "object", ("book", "laptop", "phone")),
    AtomGroup("instrument", "object", ("guitar", "violin", "drum")),
    AtomGroup("play_object", "object", ("ball", "frisbee", "kite")),
    AtomGroup("humans_plural", "object", ("men", "women", "boys", "girls")),
    AtomGroup("pets_plural", "object", ("cats", "dogs")),
    AtomGroup("vehicles_plural", "object", ("cars", "trucks", "buses", "vans")),
    # Size, morphology, and pose-compatible properties.
    AtomGroup("scale", "size", ("small", "large")),
    AtomGroup("length", "size", ("long", "short")),
    AtomGroup("width", "shape", ("broad", "narrow")),
    AtomGroup("thickness", "shape", ("thick", "thin")),
    AtomGroup("curvature", "shape", ("curved", "straight")),
    AtomGroup("tip_shape", "shape", ("pointed", "rounded")),
    AtomGroup("outline", "shape", ("round", "square", "angular")),
    AtomGroup("proportion", "shape", ("compact", "elongated")),
    AtomGroup("orientation", "relation", ("horizontal", "vertical", "diagonal")),
    AtomGroup("orientation_adverb", "relation", ("horizontally", "vertically", "diagonally")),
    AtomGroup("state", "shape", ("open", "closed")),
    AtomGroup("posture", "shape", ("upright", "bent", "reclined")),
    # Action and relation.  Tense/aspect is kept within each group.
    AtomGroup("posture_ing", "action", ("sitting", "standing", "kneeling")),
    AtomGroup("posture_present", "action", ("sits", "stands", "kneels")),
    AtomGroup("locomotion_ing", "action", ("walking", "running", "jumping")),
    AtomGroup("locomotion_present", "action", ("walks", "runs", "jumps")),
    AtomGroup("air_action_ing", "action", ("flying", "perching", "landing")),
    AtomGroup("water_action_ing", "action", ("swimming", "floating", "diving")),
    AtomGroup("vehicle_state", "action", ("parked", "moving")),
    AtomGroup("vertical_direction", "relation", ("upward", "downward")),
    AtomGroup("travel_direction", "relation", ("forward", "backward")),
    AtomGroup("side", "relation", ("left", "right")),
    AtomGroup("depth", "relation", ("foreground", "background")),
    AtomGroup("containment", "relation", ("inside", "outside")),
    AtomGroup("vertical_relation", "relation", ("above", "below")),
    # Appearance.
    AtomGroup(
        "color",
        "color",
        (
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "black",
            "white",
            "brown",
            "gray",
            "silver",
            "gold",
            "beige",
        ),
    ),
    AtomGroup("tone", "color", ("dark", "light", "pale")),
    AtomGroup("saturation", "color", ("muted", "vivid")),
    AtomGroup("surface", "texture", ("smooth", "rough", "glossy", "matte")),
    AtomGroup("marking", "texture", ("striped", "spotted", "mottled", "plain", "banded")),
    AtomGroup("density", "texture", ("dense", "sparse")),
    AtomGroup("wetness", "texture", ("wet", "dry")),
    AtomGroup("covering", "texture", ("furry", "scaly", "hairless")),
    AtomGroup(
        "material",
        "material",
        ("wooden", "metallic", "plastic", "glass", "concrete", "brick", "leather"),
    ),
    # Environment.  These are grammatical noun-for-noun/adjective-for-
    # adjective edits; the target-absence check prevents many collisions.
    AtomGroup("indoor_outdoor", "scene", ("indoor", "outdoor")),
    AtomGroup("time_of_day", "lighting", ("daytime", "nighttime")),
    AtomGroup("twilight", "lighting", ("dawn", "dusk")),
    AtomGroup("illumination", "lighting", ("dim", "bright")),
    AtomGroup("weather", "weather", ("sunny", "cloudy", "rainy", "snowy", "foggy")),
    AtomGroup("ground_adjective", "ground", ("grassy", "sandy", "snowy", "muddy", "rocky")),
    AtomGroup("ground_noun", "ground", ("grass", "sand", "snow", "mud", "gravel")),
    AtomGroup("water_setting", "scene", ("lake", "river", "ocean")),
    AtomGroup("open_setting", "scene", ("field", "beach", "forest", "desert")),
    AtomGroup("built_setting", "scene", ("street", "kitchen", "bedroom", "office", "stadium")),
    AtomGroup("settlement", "scene", ("urban", "rural")),
)


ROLE_FAMILY_PRIORITY: Mapping[str, Tuple[str, ...]] = {
    "primary": ("object", "size", "shape", "action", "color", "texture"),
    "secondary": ("object", "relation", "size", "shape", "color", "texture"),
    "activity": ("action", "relation"),
    "appearance": ("color", "material", "texture", "shape", "size"),
    "scene": ("ground", "scene", "lighting", "weather", "relation"),
    "anatomy": ("shape", "size", "color", "texture", "relation"),
}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _stable_index(parts: Iterable[str], size: int) -> int:
    payload = "\0".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % size


def _term_pattern(term: str) -> re.Pattern:
    # Unlike ``\\b``, this also handles a multiword phrase predictably.
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


def _contains_term(text: str, term: str) -> bool:
    return _term_pattern(term).search(text) is not None


def _preserve_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source.istitle():
        return target.title()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def _article_compatible(sentence: str, start: int, target: str) -> bool:
    """Avoid producing obvious ``a orange`` / ``an blue`` errors."""

    prefix = sentence[:start]
    match = re.search(r"(?i)(?<!\w)(a|an)\s+$", prefix)
    if match is None:
        return True
    target_starts_vowel = target[:1].lower() in "aeiou"
    return (match.group(1).lower() == "an") == target_starts_vowel


def _local_context_compatible(
    sentence: str, start: int, end: int, source: str, target: str
) -> bool:
    """Reject a few high-confidence adjective contradictions.

    This is deliberately narrow.  It is safer to lose a candidate than to
    silently turn ``sharp pointed tip`` into ``sharp rounded tip``.
    """

    before = sentence[max(0, start - 20) : start].lower()
    after = sentence[end : min(len(sentence), end + 20)].lower()
    target_lower = target.lower()
    if target_lower == "rounded" and re.search(r"\bsharp\s*$", before):
        return False
    if target_lower == "pointed" and re.search(r"\bblunt\s*$", before):
        return False
    if target_lower == "closed" and re.match(r"^\s+(?:air|space|water)\b", after):
        return False
    if target_lower == "open" and re.match(r"^\s+(?:door|lid|mouth|wings?)\b", after):
        # ``open door`` is valid; this branch documents that no blanket
        # open/closed rejection is intended.
        return True
    return True


def _role_for_slot(key: str, slot_index: int, prompt_version: str) -> str:
    if slot_index == 0:
        return "global"
    if "cub" in prompt_version.lower():
        return "anatomy"
    if key in ("C_primary_object", "C_head_or_main_part"):
        return "primary"
    if key in ("C_secondary_object", "C_body_or_secondary_part"):
        return "secondary"
    if key in ("C_activity_or_relation", "C_limb_or_detail_part"):
        return "activity"
    if key == "C_color_texture":
        return "appearance"
    if key in ("C_scene_type", "C_background_null"):
        return "scene"
    return "anatomy"


def _resolve_slot_keys(
    codebook_texts: Mapping[str, object], allow_unknown_schema: bool
) -> Tuple[str, ...]:
    keys = set(codebook_texts)
    if set(SLOT_KEYS_SCENE).issubset(keys):
        return SLOT_KEYS_SCENE
    if set(SLOT_KEYS_LEGACY).issubset(keys):
        return SLOT_KEYS_LEGACY
    if allow_unknown_schema and len(codebook_texts) == 6 and "C_global" in keys:
        ordered = tuple(codebook_texts.keys())
        if ordered[0] != "C_global":
            ordered = ("C_global",) + tuple(k for k in ordered if k != "C_global")
        return ordered
    raise ValueError(
        "unrecognized codebook_texts schema; expected the current scene schema "
        "or the legacy six-slot schema"
    )


def _candidate_edits(
    sentence: str,
    all_factual_text: str,
    role: str,
    image_id: str,
    slot_key: str,
    seed: int,
) -> List[dict]:
    priorities = ROLE_FAMILY_PRIORITY[role]
    candidates: List[dict] = []
    for family_rank, family in enumerate(priorities):
        for group in ATOM_GROUPS:
            if group.family != family:
                continue
            for source_term in group.terms:
                for match in _term_pattern(source_term).finditer(sentence):
                    possible_targets = [
                        target
                        for target in group.terms
                        if target.lower() != source_term.lower()
                        and not _contains_term(all_factual_text, target)
                        and _article_compatible(sentence, match.start(), target)
                        and _local_context_compatible(
                            sentence,
                            match.start(),
                            match.end(),
                            source_term,
                            target,
                        )
                    ]
                    if not possible_targets:
                        continue
                    target = possible_targets[
                        _stable_index(
                            (
                                str(seed),
                                image_id,
                                slot_key,
                                group.name,
                                source_term,
                                str(match.start()),
                            ),
                            len(possible_targets),
                        )
                    ]
                    candidates.append(
                        {
                            "family_rank": family_rank,
                            "family": family,
                            "group": group.name,
                            "source_term": source_term,
                            "source_surface": match.group(0),
                            "target_term": target,
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )
    return candidates


def _make_foil(
    sentence: str,
    all_factual_text: str,
    role: str,
    image_id: str,
    slot_key: str,
    seed: int,
) -> Optional[Tuple[str, dict]]:
    candidates = _candidate_edits(
        sentence, all_factual_text, role, image_id, slot_key, seed
    )
    if not candidates:
        return None

    # Prefer the role's highest-priority family.  Hash selection within that
    # family avoids always editing the first adjective in a sentence.
    best_rank = min(item["family_rank"] for item in candidates)
    candidates = [item for item in candidates if item["family_rank"] == best_rank]
    candidates.sort(key=lambda x: (x["start"], -(x["end"] - x["start"]), x["group"]))
    chosen = candidates[
        _stable_index((str(seed), image_id, slot_key, "candidate"), len(candidates))
    ]

    replacement = _preserve_case(chosen["source_surface"], chosen["target_term"])
    start, end = chosen["start"], chosen["end"]
    foil = sentence[:start] + replacement + sentence[end:]
    if foil == sentence:
        return None
    if foil[:start] != sentence[:start] or foil[start + len(replacement) :] != sentence[end:]:
        raise AssertionError("foil edit touched more than the selected character span")

    metadata = {
        "valid": True,
        "operation": "single_span_contrast_substitution",
        "role": role,
        "family": chosen["family"],
        "contrast_set": chosen["group"],
        "source_atom": chosen["source_surface"],
        "target_atom": replacement,
        "source_span": [start, end],
        "target_span": [start, start + len(replacement)],
        "target_absent_from_all_factual_captions": True,
    }
    return foil, metadata


def _build_output_row(
    row: Mapping[str, object],
    seed: int,
    allow_unknown_schema: bool,
) -> Tuple[dict, Counter]:
    image_id = row.get("image_id")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("row has no non-empty string image_id")
    codebook_texts = row.get("codebook_texts")
    if not isinstance(codebook_texts, dict):
        raise ValueError(f"{image_id}: codebook_texts is not an object")
    keys = _resolve_slot_keys(codebook_texts, allow_unknown_schema)
    prompt_version = str(row.get("prompt_version", "") or "")

    factual: Dict[str, str] = {}
    for key in keys:
        value = codebook_texts.get(key, "")
        factual[key] = str(value or "").strip()
    all_factual_text = "\n".join(factual.values())

    foil_texts: Dict[str, str] = {}
    foil_valid: Dict[str, bool] = {}
    foil_edits: Dict[str, dict] = {}
    stats: Counter = Counter()

    for slot_index, key in enumerate(keys):
        sentence = factual[key]
        if slot_index == 0:
            foil_texts[key] = ""
            foil_valid[key] = False
            foil_edits[key] = {
                "valid": False,
                "reason": "global_slot_excluded",
                "role": "global",
            }
            stats["invalid:global_slot_excluded"] += 1
            continue
        role = _role_for_slot(key, slot_index, prompt_version)
        if not sentence or sentence.lower() == "none":
            foil_texts[key] = ""
            foil_valid[key] = False
            foil_edits[key] = {
                "valid": False,
                "reason": "source_caption_missing",
                "role": role,
            }
            stats[f"invalid:{key}:source_caption_missing"] += 1
            continue
        result = _make_foil(
            sentence, all_factual_text, role, image_id, key, seed
        )
        if result is None:
            foil_texts[key] = ""
            foil_valid[key] = False
            foil_edits[key] = {
                "valid": False,
                "reason": "no_supported_atomic_substitution",
                "role": role,
            }
            stats[f"invalid:{key}:no_supported_atomic_substitution"] += 1
            continue
        foil, edit = result
        foil_texts[key] = foil
        foil_valid[key] = True
        foil_edits[key] = edit
        stats[f"valid:{key}"] += 1
        stats[f"family:{edit['family']}"] += 1
        stats[f"contrast_set:{edit['contrast_set']}"] += 1

    output = {
        "image_id": image_id,
        "source_prompt_version": prompt_version,
        "foil_schema_version": SCHEMA_VERSION,
        "slot_keys": list(keys),
        "foil_codebook_texts": foil_texts,
        "foil_valid": foil_valid,
        "foil_edits": foil_edits,
    }
    for metadata_key in ("split", "index"):
        if metadata_key in row:
            output[metadata_key] = row[metadata_key]
    return output, stats


def _print_summary(stats: Counter, rows: int, keys_seen: Sequence[str]) -> None:
    print(f"[foil] rows={rows}")
    for key in keys_seen:
        if key == "C_global":
            print(f"[foil] {key}: valid=0/{rows} (excluded by contract)")
            continue
        valid = stats[f"valid:{key}"]
        print(f"[foil] {key}: valid={valid}/{rows} ({valid / max(rows, 1):.1%})")
    family_counts = sorted(
        (
            (name.split(":", 1)[1], count)
            for name, count in stats.items()
            if name.startswith("family:")
        ),
        key=lambda x: (-x[1], x[0]),
    )
    print("[foil] valid edit families: " + ", ".join(f"{k}={v}" for k, v in family_counts))


def _atomic_write_json(path: str, payload: Mapping[str, object]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".foil-meta-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source Qwen caption JSONL.")
    parser.add_argument("--output", required=True, help="Output foil JSONL.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="Process at most this many source rows; intended for audits.",
    )
    parser.add_argument(
        "--show", type=int, default=0, help="Print this many generated rows."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize without writing output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output and manifest.",
    )
    parser.add_argument(
        "--allow-unknown-schema",
        action="store_true",
        help="Accept an insertion-ordered six-key schema with C_global first.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"[foil] input not found: {args.input}")
    output_abs = os.path.abspath(args.output)
    input_abs = os.path.abspath(args.input)
    if output_abs == input_abs:
        raise SystemExit("[foil] --output must differ from --input")
    manifest_path = args.output + ".meta.json"
    if not args.dry_run and not args.overwrite:
        existing = [p for p in (args.output, manifest_path) if os.path.exists(p)]
        if existing:
            raise SystemExit(
                "[foil] refusing to overwrite existing output: "
                + ", ".join(existing)
                + " (pass --overwrite explicitly)"
            )

    source_sha256 = _sha256_file(args.input)
    output_file = None
    temp_output = None
    if not args.dry_run:
        output_dir = os.path.dirname(output_abs) or "."
        os.makedirs(output_dir, exist_ok=True)
        fd, temp_output = tempfile.mkstemp(
            prefix=".caption-foils-", suffix=".jsonl", dir=output_dir
        )
        output_file = os.fdopen(fd, "w", encoding="utf-8")

    stats: Counter = Counter()
    rows = 0
    seen_ids = set()
    keys_seen: Tuple[str, ...] = ()
    examples: List[dict] = []
    try:
        with open(args.input, "r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if args.max_rows > 0 and rows >= args.max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    output_row, row_stats = _build_output_row(
                        row, args.seed, args.allow_unknown_schema
                    )
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{args.input}:{line_number}: invalid source row: {exc}"
                    ) from exc
                image_id = output_row["image_id"]
                if image_id in seen_ids:
                    raise ValueError(
                        f"{args.input}:{line_number}: duplicate image_id {image_id!r}"
                    )
                seen_ids.add(image_id)
                row_keys = tuple(output_row["slot_keys"])
                if keys_seen and row_keys != keys_seen:
                    raise ValueError(
                        f"{args.input}:{line_number}: mixed slot schemas "
                        f"{keys_seen!r} and {row_keys!r}"
                    )
                keys_seen = row_keys
                rows += 1
                stats.update(row_stats)
                if len(examples) < args.show:
                    examples.append(output_row)
                if output_file is not None:
                    output_file.write(
                        json.dumps(output_row, ensure_ascii=False, sort_keys=False)
                        + "\n"
                    )
        if rows == 0:
            raise ValueError("[foil] input produced zero rows")
        if output_file is not None:
            output_file.flush()
            os.fsync(output_file.fileno())
            output_file.close()
            output_file = None
            os.replace(temp_output, output_abs)
            temp_output = None
    finally:
        if output_file is not None:
            output_file.close()
        if temp_output is not None and os.path.exists(temp_output):
            os.remove(temp_output)

    _print_summary(stats, rows, keys_seen)
    for index, example in enumerate(examples, start=1):
        print(f"[foil] example {index}:")
        print(json.dumps(example, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("[foil] dry-run: no files written")
        return 0

    output_sha256 = _sha256_file(output_abs)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_jsonl": input_abs,
        "source_sha256": source_sha256,
        "output_jsonl": output_abs,
        "output_sha256": output_sha256,
        "rows": rows,
        "slot_keys": list(keys_seen),
        "seed": args.seed,
        "max_rows": args.max_rows,
        "valid_per_slot": {
            key: (0 if key == "C_global" else stats[f"valid:{key}"])
            for key in keys_seen
        },
        "edit_family_counts": {
            name.split(":", 1)[1]: count
            for name, count in sorted(stats.items())
            if name.startswith("family:")
        },
        "contract": {
            "global_slot_valid": False,
            "edits_per_foil_caption": 1,
            "replacement_absent_from_all_factual_captions": True,
            "generation_uses_external_model_or_api": False,
        },
    }
    _atomic_write_json(manifest_path, manifest)
    print(f"[foil] wrote {output_abs}")
    print(f"[foil] wrote {os.path.abspath(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
