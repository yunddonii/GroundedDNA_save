#!/usr/bin/env python
"""Run `diagnose_slot_heatmap_collapse.py` on a run dir whose `args.txt` holds a
DASH-FREE line, which the shared parser skips entirely.

`train_siglip2` pads `<key>` and `<value>` with hyphens to a fixed width; when
the value is long enough (the tokenizer-digest JSON is ~370 chars) no hyphen is
emitted at all, and `_parse_args_txt`'s `for/else: continue` drops the line.
`model_siglip2.SigLIP2SemanticOTModel.__init__` then fails on a bare `getattr`.

Neither the probe nor the parser is modified on disk: the parser is patched in
memory for this call only, and the missing value is recovered VERBATIM from the
same file rather than reconstructed.
"""
from __future__ import annotations

import os
import re
import sys

_REPO = "/home/yschoi/GroundedDNA"
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import regen_viz_routing as _rvr

_orig_parse = _rvr._parse_args_txt
_IDENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(.*)$")


def _parse_args_txt_with_dashless(path: str):
    ns = _orig_parse(path)
    known = set(vars(ns))
    recovered = []
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or "-" in line:
                continue                      # the parser already handled it
            m = _IDENT.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if key and key not in known:
                setattr(ns, key, val)
                recovered.append(key)
    if recovered:
        print(f"[dashless-shim] recovered from args.txt: {recovered}", file=sys.stderr)
    return ns


_rvr._parse_args_txt = _parse_args_txt_with_dashless

import diagnose_slot_heatmap_collapse as _probe   # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_probe.main())
