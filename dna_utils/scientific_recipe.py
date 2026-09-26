"""Typed scientific recipe: the trainer's complete parsed configuration, bound by one digest.

Anchor confirmation v1 (audit sections 532.1, 659.3). ``RunIdentity`` (schema 5) binds about forty
fields, while the trainer parses 355. Runs that differ only in an unbound field -- ``axis_center``,
``lambda_wasserstein``, the global gate -- therefore share one identity digest. This module binds
the whole configuration instead, and it does so from the one thing every stage can see identically:
the exact argv the trainer receives, parsed by the trainer's own parser.

* The launcher renders that argv before launch (the dataset script run with a capture shim in
  place of the interpreter), builds the payload below and seals its digest in the plan.
* The trainer rebuilds the payload from ``sys.argv`` before it claims a run directory or loads a
  model or data, refuses any difference, and cross-checks the post-processed arguments.
* The reducer rebuilds the typed fields from the saved ``config.pt`` (the same bytes it hashed)
  and compares them with the sealed ones.

The payload binds the exact token list as well as the typed values. The approved commands repeat
some options on purpose (a script default followed by the launcher's override), so a repeat is not
itself refused; it has to be the repeat the plan rendered. An abbreviated or unknown option -- which
argparse would otherwise resolve silently -- is refused outright.

``args.txt`` is display-only: its ``<key><dashes><value>`` layout cannot tell ``-3.0`` from ``3.0``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, Iterable, Mapping, Optional

RECIPE_SCHEMA = "groundeddna-scientific-recipe/1"
EXPECTED_RECIPE_DIGEST_ENV = "GDNA_PHASE3_EXPECTED_RECIPE_DIGEST"
EXPECTED_RECIPE_JSON_ENV = "GDNA_PHASE3_EXPECTED_RECIPE_JSON"

#: Parsed destinations left out of ``fields``, each with its reason. Everything else is bound. The
#: exact argv (which does carry the tag) is bound separately in the payload.
NON_SCIENTIFIC = {
    "tag": "the run label; bound by the argv and by the campaign binding's expected_tag",
    "log_dir": "the legacy output folder; Config.set_args deletes it before training",
}

#: Destinations this source generation added on top of the historical one, with the value that
#: reproduces the historical behaviour. Only these may be absent from a historical config.pt.
ADDED_SINCE_HISTORICAL = {"axis_center": "none"}

_NEGATIVE_NUMBER = re.compile(r"^-\d+$|^-\d*\.\d+$|^-\d+(\.\d*)?[eE][-+]?\d+$")


class RecipeMismatch(RuntimeError):
    """The configuration a process would run is not the one that was sealed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RecipeMismatch(message)


def _typed(dest: str, value: Any) -> Any:
    """A canonical, JSON-exact form of one parsed value; refuses anything else."""
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
    """Every destination of ``parser`` except help, in a fixed order."""
    return sorted({action.dest for action in parser._actions if action.dest != "help"})


def check_option_tokens(parser, argv: Iterable[str]) -> None:
    """Refuse abbreviated or unknown options. argparse resolves ``--lambda_codon_j`` to
    ``--lambda_codon_joint`` without a word; a sealed command may only use exact option strings."""
    known = parser._option_string_actions
    for token in argv:
        if not token.startswith("-") or token == "-" or _NEGATIVE_NUMBER.match(token):
            continue
        option = token.split("=", 1)[0]
        _need(option in known, f"option {option!r} is not an exact option of the trainer parser "
                               "(abbreviated or unknown)")


def fields_from_values(values: Mapping[str, Any], dests: Iterable[str], *,
                       allow_historical_absence: bool = False) -> dict:
    """The typed recipe fields from a mapping of destination -> value."""
    fields = {}
    for dest in dests:
        if dest in NON_SCIENTIFIC:
            continue
        if dest not in values:
            _need(allow_historical_absence and dest in ADDED_SINCE_HISTORICAL,
                  f"{dest}: missing from the configuration")
            fields[dest] = ADDED_SINCE_HISTORICAL[dest]
            continue
        fields[dest] = _typed(dest, values[dest])
    return fields


def build_payload(parser, argv: Iterable[str]) -> dict:
    """The sealed object: schema, exact argv and the typed fields the parser produces from it."""
    argv = [str(token) for token in argv]
    check_option_tokens(parser, argv)
    try:
        namespace = parser.parse_args(argv)
    except SystemExit as error:
        raise RecipeMismatch(f"the trainer parser rejected the argv (exit {error.code})") from None
    return {"schema": RECIPE_SCHEMA, "argv": argv,
            "fields": fields_from_values(vars(namespace), destinations(parser))}


def canonical_bytes(payload: Mapping) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def digest(payload: Mapping) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def field_differences(a: Mapping, b: Mapping) -> list:
    """Names of the fields whose presence or value differs between two ``fields`` mappings."""
    return sorted(k for k in set(a) | set(b) if (k in a) != (k in b) or a.get(k) != b.get(k))


def fields_from_config(config: Mapping[str, Any], parser, *, historical: bool = False) -> dict:
    """Typed fields from a saved ``config.pt`` dict (Config.save_arg), undoing the one value
    Config.set_args rewrites (``setting`` 1 -> ``"setting1"``). ``historical=True`` admits the
    declared ADDED_SINCE_HISTORICAL destinations as absent."""
    values = dict(config)
    setting = values.get("setting")
    if isinstance(setting, str) and setting.startswith("setting"):
        action = next(a for a in parser._actions if a.dest == "setting")
        raw = setting[len("setting"):]
        values["setting"] = action.type(raw) if callable(action.type) else raw
    return fields_from_values(values, destinations(parser), allow_historical_absence=historical)


def verify_trainer_recipe(parser, argv: Iterable[str], post_processed) -> Optional[dict]:
    """The trainer-side check, run before the run directory is claimed.

    All-or-none: without both environment variables this is an ordinary run and returns None.
    With them, the payload rebuilt from this process's argv must equal the sealed one, and every
    field must also equal the post-processed attribute the trainer will actually use."""
    sealed_digest = os.environ.get(EXPECTED_RECIPE_DIGEST_ENV, "")
    sealed_json = os.environ.get(EXPECTED_RECIPE_JSON_ENV, "")
    if not sealed_digest and not sealed_json:
        return None
    _need(bool(sealed_digest) and bool(sealed_json),
          "incomplete scientific-recipe binding: both the digest and the JSON are required")
    try:
        sealed = json.loads(sealed_json)
    except ValueError:
        raise RecipeMismatch("the sealed scientific recipe is malformed JSON") from None
    _need(isinstance(sealed, dict) and digest(sealed) == sealed_digest,
          "the sealed scientific recipe does not match its digest")
    _need(sealed.get("schema") == RECIPE_SCHEMA, f"unsupported recipe schema {sealed.get('schema')!r}")
    actual = build_payload(parser, argv)
    if actual["argv"] != sealed["argv"]:
        raise RecipeMismatch("the trainer argv differs from the sealed argv "
                             f"({len(actual['argv'])} vs {len(sealed['argv'])} tokens)")
    differing = field_differences(actual["fields"], sealed["fields"])
    _need(not differing, f"parsed recipe differs from the sealed recipe in {differing[:8]}")
    for dest, value in actual["fields"].items():
        post = getattr(post_processed, dest, None)
        if dest == "setting":
            post = sealed["fields"]["setting"] if post == f"setting{value}" else post
        _need(_typed(dest, post) == value,
              f"{dest}: the post-processed value {post!r} differs from the parsed {value!r}")
    return {"scientific_recipe_schema": RECIPE_SCHEMA,
            "scientific_recipe_sha256": digest(actual)}
