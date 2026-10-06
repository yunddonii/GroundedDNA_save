#!/usr/bin/env python
"""Rebuild a train_siglip2.py command from an approved run's args.txt, for OFF-PROTOCOL exploration
cells of the text-path line. Copy of scripts/build_offprotocol_cmd.py with three fixes:
  * imports the parser from THIS worktree, not /home/yschoi/GroundedDNA;
  * drops every Phase-3 seal / HF-authority key (phase3_*, clip_snapshot_*), so the trainer does not
    run the seal verification that a changed caption/whitening/cache would fail;
  * a dest with a store_true AND a store_false option (use_gumbel_softmax / --no_gumbel_softmax) emits
    the store_false switch when the recorded value is False and the parser default is True.
The rendered command is printed; nothing is launched here.
"""
from __future__ import annotations
import argparse, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, WORKTREE)
os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")

SKIP = {"save_result_path", "save_model_state_path", "save_log_path", "date", "tag", "config_path", "device",
        "num_classes", "_phase3_campaign_binding", "_phase3_input_authority"}
SKIP_PREFIX = ("phase3_", "clip_snapshot_")
RECORDED_TO_CLI = {"setting": {"setting1": "1", "setting2": "2"}}
_LINE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)-{3,}(.*)$")


def read_args_txt(path):
    out = {}
    for line in open(path):
        m = _LINE.match(line.rstrip("\n"))
        if m:
            out[m.group(1)] = m.group(2).strip()
        else:
            m2 = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(\{.*)$", line.rstrip("\n"))
            if m2:
                out[m2.group(1)] = m2.group(2).strip()
    return out


def build(args_txt, overrides, drop=()):
    from config import Config
    parser = Config.build_parser()
    defaults = vars(parser.parse_args([]))
    form = {}                                      # dest -> {"true": opt, "false": opt, "value": opt, "bool_optional": [opts]}
    for a in parser._actions:
        if not a.option_strings:
            continue
        cls = a.__class__.__name__
        f = form.setdefault(a.dest, {})
        if cls == "BooleanOptionalAction":
            f["bool_optional"] = list(a.option_strings)
        elif cls == "_StoreTrueAction":
            f["true"] = max(a.option_strings, key=len)
        elif cls == "_StoreFalseAction":
            f["false"] = max(a.option_strings, key=len)
        else:
            f["value"] = max(a.option_strings, key=len)
    recorded = read_args_txt(args_txt)
    for k in drop:
        recorded.pop(k, None)
    recorded.update({k: str(v) for k, v in overrides.items()})
    cmd = []
    for key in sorted(recorded):
        if key in SKIP or key.startswith(SKIP_PREFIX) or key not in form or key not in defaults:
            continue
        val = RECORDED_TO_CLI.get(key, {}).get(recorded[key], recorded[key])
        if val == str(defaults[key]) and key not in overrides:
            continue
        f = form[key]
        if "bool_optional" in f:
            on = [o for o in f["bool_optional"] if not o.startswith("--no")][0]
            off = [o for o in f["bool_optional"] if o.startswith("--no")]
            if val == "True" and defaults[key] is not True:
                cmd.append(on)
            elif val == "False" and defaults[key] is not False and off:
                cmd.append(off[0])
            continue
        if "true" in f or "false" in f:
            if val == "True" and "true" in f and defaults[key] is not True:
                cmd.append(f["true"])
            elif val == "False" and "false" in f and defaults[key] is not False:
                cmd.append(f["false"])
            elif val == "True" and "true" in f and key in overrides:
                cmd.append(f["true"])
            elif val == "False" and "false" in f and key in overrides:
                cmd.append(f["false"])
            continue
        if val == "None":
            continue
        name = f["value"]
        if val.startswith("-") and val[1:2].isdigit():
            cmd.append(f"{name}={val}")
        else:
            cmd += [name, val]
    return cmd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_args", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="k=v")
    ap.add_argument("--drop", action="append", default=[], metavar="key", help="remove a recorded key (falls back to the parser default)")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    ov = dict(s.split("=", 1) for s in a.set)
    cmd = build(a.from_args, ov, a.drop)
    print(" ".join(["python", "train_siglip2.py", "--tag", a.tag] + cmd))
