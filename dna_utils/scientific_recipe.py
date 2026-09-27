"""Typed scientific recipe: the trainer's complete parsed configuration, bound by one digest.

Anchor confirmation v1 (audit sections 532.1, 659.3, 669, 670). ``RunIdentity`` (schema 5) binds
about forty fields while the trainer parses 355, so runs that differ only in an unbound field --
``axis_center``, ``lambda_wasserstein``, the global gate -- share one identity digest. This module
binds the whole configuration, from the one thing every stage sees identically: the exact argv
the trainer receives, parsed by the trainer's own parser.

Payload (schema ``groundeddna-scientific-recipe/2``), exactly three keys:
  ``argv``   -- the raw token list, bound as is;
  ``fields`` -- every parser destination except NON_SCIENTIFIC, as canonical JSON values, with the
                documented normalisations below applied;
  ``schema``.

Comparisons are TYPE-SENSITIVE: two values are equal only if their canonical JSON is equal, so
``True``, ``1`` and ``1.0`` differ. The trainer requires the digest of the payload it rebuilds to
equal the sealed digest exactly, before any directory or claim exists.

Documented normalisations (the only places an effective value is not the parsed string):
  * ``setting``: Config.set_args turns ``1`` into ``"setting1"`` on the post-processed args; the
    fields keep the parsed value and the comparison undoes the prefix.
  * ``clip_snapshot_tokenizers_sha256_json``: the launcher shell-quotes the compact JSON and the
    dataset wrappers expand EXTRA_ARGS without a second parse, so the trainer receives one literal
    quote layer, which ``canonical_tokenizer_json`` strips before re-serialising. The fields hold
    that canonical form; the raw argv keeps the quoted token.
  * ``axis_center`` absent from a historical config: read as ``none`` only where the caller
    declares the historical boundary.

Repeated options (audit 669.3, 677.2): the pinned dataset scripts hardcode some options in their
bodies and the launcher's EXTRA_ARGS, which lands after the body, sets them again; argparse keeps the
last occurrence. A destination may be given more than once only as the exact sequence in
REVIEWED_OVERRIDES: FIRST the wrapper's literal tokens listed there, THEN exactly one launcher
override. The override's value is not decided here: the sealed fields bind it, and the launcher's
admission checks it against the protocol. A third occurrence, a first occurrence that is not the
reviewed literal (another value, alias or boolean form, or the override placed first), or a repeat
of any other destination -- through the same option, an alias (``-bs``/``--batch_size``) or a
boolean pair (``--x``/``--no-x``) -- refuses. ``axis_center`` is never repeated, and appears exactly
once (equal to the planned arm) in an anchor-confirmation cell.

``args.txt`` is display-only: its ``<key><dashes><value>`` layout cannot tell ``-3.0`` from ``3.0``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, Iterable, Mapping, Optional

#: the bytes this module was imported from (audit 697)
with open(__file__, "rb") as _source:
    _IMPORTED_SOURCE_SHA256 = hashlib.sha256(_source.read()).hexdigest()
RECIPE_SCHEMA = "groundeddna-scientific-recipe/2"
PAYLOAD_KEYS = ("argv", "fields", "schema")
EXPECTED_RECIPE_DIGEST_ENV = "GDNA_PHASE3_EXPECTED_RECIPE_DIGEST"
EXPECTED_RECIPE_JSON_ENV = "GDNA_PHASE3_EXPECTED_RECIPE_JSON"
#: The campaign cell id of an anchor-confirmation cell ends with this marker and the arm.
ANCHOR_CELL_MARKER = "|axis_center="

NON_SCIENTIFIC = {
    "tag": "the run label; bound by the argv and by the campaign binding's expected_tag",
    "log_dir": "the legacy output folder; Config.set_args deletes it before training",
}
ADDED_SINCE_HISTORICAL = {"axis_center": "none"}
#: destination -> the exact tokens the three in-scope wrappers (Flickr25K, NUS-WIDE, MS-COCO; bytes
#: pinned in the launcher's DATASET_SCRIPT_SHA256) put in their bodies. Every wrapper carries each of
#: these; the launcher overrides all seven (S5/stage-1/quiet flags and the approved top-p window).
REVIEWED_OVERRIDES = {
    "epoch": ("-e", "60"),
    "routing_adaptive_topp": ("--routing_adaptive_topp",),
    "routing_adaptive_topp_min": ("--routing_adaptive_topp_min", "0.3"),
    "routing_adaptive_topp_max": ("--routing_adaptive_topp_max", "0.7"),
    "lambda_codeword_codon_sinkhorn": ("--lambda_codeword_codon_sinkhorn", "0.0"),
    "post_eval_compositional": ("--post_eval_compositional",),
    "dna_distance_mode": ("--dna_distance_mode", "base"),
}
#: Other reviewed first occurrences, per destination (contract v3 section 6, audit 709.1 item 4).
#: The CIFAR-10 wrapper's body passes ``--lambda_codeword_codon_sinkhorn "${CCS:-0.1}"`` and the
#: launcher sets CCS=0.1 for it; the launcher's override that must follow is 0.0, the value every
#: approved CIFAR-10 stage-1 and refit args.txt records. The override is still required, and the
#: protocol values bind the effective 0.0.
REVIEWED_OVERRIDE_ALTERNATES = {
    "lambda_codeword_codon_sinkhorn": (("--lambda_codeword_codon_sinkhorn", "0.1"),),
}

_NEGATIVE_NUMBER = re.compile(r"^-\d+$|^-\d*\.\d+$|^-\d+(\.\d*)?[eE][-+]?\d+$")


class RecipeMismatch(RuntimeError):
    """The configuration a process would run is not the one that was sealed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RecipeMismatch(message)


def canonical_tokenizer_json(value) -> str:
    """Undo the one literal quote layer left by unquoted EXTRA_ARGS (moved unchanged from
    train_siglip2._canonical_phase3_tokenizer_json, which now delegates here)."""
    raw = str(value or "")
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        raw = raw[1:-1]
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("tokenizer SHA authority must be a JSON object")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _normalise(dest: str, value: Any) -> Any:
    if dest == "clip_snapshot_tokenizers_sha256_json" and value not in (None, ""):
        try:
            return canonical_tokenizer_json(value)
        except ValueError as error:
            raise RecipeMismatch(f"{dest}: {error}") from None
    return value


def canonical(value: Any) -> str:
    """Type-sensitive canonical JSON of one value (True, 1 and 1.0 all differ)."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RecipeMismatch(f"value {value!r} has no canonical JSON form: {error}") from None


def _typed(dest: str, value: Any) -> Any:
    """A JSON-exact form of one parsed value; refuses anything else."""
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_typed(dest, item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        _need(math.isfinite(value), f"{dest}: non-finite value {value!r} cannot be a recipe value")
        return float(value)
    raise RecipeMismatch(f"{dest}: value of type {type(value).__name__} is not a recipe value")


def destinations(parser) -> list:
    return sorted({action.dest for action in parser._actions if action.dest != "help"})


def _is_option(token: str) -> bool:
    return token.startswith("-") and token != "-" and not _NEGATIVE_NUMBER.match(token)


def option_occurrences(parser, argv: Iterable[str]) -> dict:
    """Destination -> the token spans that set it, in order: the option token plus the value token
    it takes (none for a flag, or when the value is inline as ``--x=v``). Refuses abbreviated or
    unknown options, which argparse would otherwise resolve silently."""
    known = parser._option_string_actions
    argv = list(argv)
    seen: dict = {}
    for i, token in enumerate(argv):
        if not _is_option(token):
            continue
        option = token.split("=", 1)[0]
        _need(option in known, f"option {option!r} is not an exact option of the trainer parser "
                               "(abbreviated or unknown)")
        action = known[option]
        span = (token,)
        if "=" not in token and action.nargs in (None, "?") and i + 1 < len(argv) \
                and not _is_option(argv[i + 1]):
            span = (token, argv[i + 1])
        seen.setdefault(action.dest, []).append(span)
    return seen


def admitted_overrides(parser, argv: Iterable[str]) -> dict:
    """The reviewed wrapper overrides this argv uses, as destination -> [wrapper literal, override];
    refuses every other repeat (see the module docstring)."""
    occurrences = option_occurrences(parser, argv)
    _need(len(occurrences.get("axis_center", [])) <= 1, "axis_center is given more than once")
    admitted = {}
    for dest, spans in sorted(occurrences.items()):
        if len(spans) == 1:
            continue
        _need(dest in REVIEWED_OVERRIDES,
              f"repeated options outside the reviewed wrapper overrides: {dest} {spans}")
        _need(len(spans) == 2, f"{dest} is given {len(spans)} times; a reviewed override is the "
                               "wrapper literal followed by exactly one launcher override")
        _need(spans[0] == REVIEWED_OVERRIDES[dest]
              or spans[0] in REVIEWED_OVERRIDE_ALTERNATES.get(dest, ()),
              f"{dest}: the first occurrence {list(spans[0])} is not a reviewed wrapper literal "
              f"{[list(REVIEWED_OVERRIDES[dest]), *map(list, REVIEWED_OVERRIDE_ALTERNATES.get(dest, ()))]}")
        admitted[dest] = [list(span) for span in spans]
    return admitted


def fields_from_values(values: Mapping[str, Any], dests: Iterable[str], *,
                       allow_historical_absence: bool = False) -> dict:
    fields = {}
    for dest in dests:
        if dest in NON_SCIENTIFIC:
            continue
        if dest not in values:
            _need(allow_historical_absence and dest in ADDED_SINCE_HISTORICAL,
                  f"{dest}: missing from the configuration")
            fields[dest] = ADDED_SINCE_HISTORICAL[dest]
            continue
        fields[dest] = _typed(dest, _normalise(dest, values[dest]))
    return fields


def build_payload(parser, argv: Iterable[str], *, planned_arm: Optional[str] = None) -> dict:
    """The sealed object. `planned_arm` marks an anchor-confirmation cell: its argv must set
    axis_center exactly once, to that arm."""
    argv = [str(token) for token in argv]
    admitted_overrides(parser, argv)                   # axis repeat first: a named refusal
    axis_count = len(option_occurrences(parser, argv).get("axis_center", []))
    try:
        namespace = parser.parse_args(argv)
    except SystemExit as error:
        raise RecipeMismatch(f"the trainer parser rejected the argv (exit {error.code})") from None
    fields = fields_from_values(vars(namespace), destinations(parser))
    if planned_arm is not None:
        _need(axis_count == 1 and fields["axis_center"] == planned_arm,
              f"an anchor-confirmation cell must set axis_center exactly once to {planned_arm!r}")
    return {"schema": RECIPE_SCHEMA, "argv": argv, "fields": fields}


def canonical_bytes(payload: Mapping) -> bytes:
    return canonical(payload).encode("ascii")


def digest(payload: Mapping) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def check_payload_shape(payload) -> None:
    _need(isinstance(payload, dict) and tuple(sorted(payload)) == PAYLOAD_KEYS,
          f"a recipe payload has exactly the keys {PAYLOAD_KEYS}")
    _need(payload["schema"] == RECIPE_SCHEMA, f"unsupported recipe schema {payload['schema']!r}")
    _need(isinstance(payload["argv"], list) and all(isinstance(t, str) for t in payload["argv"]),
          "recipe argv must be a list of strings")
    _need(isinstance(payload["fields"], dict), "recipe fields must be an object")


def field_differences(a: Mapping, b: Mapping) -> list:
    """Names whose presence or canonical (type-sensitive) value differs."""
    return sorted(k for k in set(a) | set(b)
                  if (k in a) != (k in b) or canonical(a.get(k)) != canonical(b.get(k)))


def fields_from_config(config: Mapping[str, Any], parser, *, historical: bool = False) -> dict:
    """Typed fields from a saved config.pt dict (Config.save_arg), undoing only the documented
    `setting` rewrite. `historical=True` admits the declared ADDED_SINCE_HISTORICAL absences."""
    values = dict(config)
    setting = values.get("setting")
    if isinstance(setting, str) and setting.startswith("setting"):
        action = next(a for a in parser._actions if a.dest == "setting")
        raw = setting[len("setting"):]
        values["setting"] = action.type(raw) if callable(action.type) else raw
    return fields_from_values(values, destinations(parser), allow_historical_absence=historical)


def anchor_arm_of(cell_id: Optional[str]) -> Optional[str]:
    """The arm an anchor-confirmation cell id names, or None for any other cell."""
    if not cell_id or ANCHOR_CELL_MARKER not in cell_id:
        return None
    return cell_id.rsplit(ANCHOR_CELL_MARKER, 1)[1]


def verify_trainer_recipe(parser, argv: Iterable[str], post_processed,
                          campaign_binding: Optional[Mapping] = None) -> Optional[dict]:
    """The trainer-side check, before the run directory is claimed.

    Without both environment variables and outside an anchor-confirmation cell this is an ordinary
    run and returns None. An anchor-confirmation cell (its campaign cell id carries the arm) must
    carry the sealed recipe; the payload rebuilt from this process's argv must have exactly the
    sealed digest; and every field must be present on the post-processed args with the same
    canonical value, after only the documented normalisations."""
    sealed_digest = os.environ.get(EXPECTED_RECIPE_DIGEST_ENV, "")
    sealed_json = os.environ.get(EXPECTED_RECIPE_JSON_ENV, "")
    planned_arm = anchor_arm_of((campaign_binding or {}).get("cell_id"))
    if not sealed_digest and not sealed_json:
        _need(planned_arm is None,
              "an anchor-confirmation cell must carry its sealed scientific recipe")
        _need(campaign_binding is None or getattr(post_processed, "axis_center", "none") == "none",
              "axis_center=anchors inside a campaign requires an anchor-confirmation cell")
        return None
    _need(bool(sealed_digest) and bool(sealed_json),
          "incomplete scientific-recipe binding: both the digest and the JSON are required")
    _need(campaign_binding is not None and planned_arm is not None,
          "a sealed scientific recipe is only valid inside an anchor-confirmation campaign cell")
    try:
        sealed = json.loads(sealed_json)
    except ValueError:
        raise RecipeMismatch("the sealed scientific recipe is malformed JSON") from None
    check_payload_shape(sealed)
    _need(digest(sealed) == sealed_digest, "the sealed scientific recipe does not match its digest")
    actual = build_payload(parser, argv, planned_arm=planned_arm)
    if actual["argv"] != sealed["argv"]:
        raise RecipeMismatch("the trainer argv differs from the sealed argv "
                             f"({len(actual['argv'])} vs {len(sealed['argv'])} tokens)")
    differing = field_differences(actual["fields"], sealed["fields"])
    _need(not differing, f"parsed recipe differs from the sealed recipe in {differing[:8]}")
    _need(digest(actual) == sealed_digest, "the rebuilt recipe digest is not the sealed digest")
    for dest, value in actual["fields"].items():
        _need(hasattr(post_processed, dest), f"{dest}: absent from the post-processed arguments")
        post = getattr(post_processed, dest)
        if dest == "setting" and post == f"setting{value}":
            post = value
        _need(canonical(_typed(dest, _normalise(dest, post))) == canonical(value),
              f"{dest}: the post-processed value {post!r} differs from the parsed {value!r}")
    return {"scientific_recipe_schema": RECIPE_SCHEMA, "scientific_recipe_sha256": sealed_digest}
