#!/usr/bin/env python
"""Rebuild a train_siglip2.py command line from an existing run's args.txt.

Reading the recipe off a real args.txt (rather than retyping 68 flags) keeps an
off-protocol cell a genuine single-delta from the recorded run. Flag *form* is
taken from the live parser, so store_true flags are emitted bare and valued
flags with their value; a value the parser already defaults to is dropped.
"""
from __future__ import annotations
import argparse, os, re, sys

sys.path.insert(0, "/home/yschoi/GroundedDNA")
os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")

# args.txt keys that are outputs, campaign bookkeeping, or not parser flags.
SKIP = {"save_result_path", "save_model_state_path", "save_log_path", "date",
        "tag", "config_path", "device", "num_classes",
        "_phase3_campaign_binding", "_phase3_input_authority"}

#: args.txt records the value the trainer *ran with*, which for a few keys is a
#: normalised form the parser itself would reject.  `--setting` is declared
#: ``type=int, choices=[1, 2]`` but train_siglip2 hardcodes ``'setting1'`` when
#: it calls the loader, and that string is what lands in args.txt.
RECORDED_TO_CLI = {"setting": {"setting1": "1", "setting2": "2"}}
_LINE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)-{3,}(.*)$")


def read_args_txt(path):
    out = {}
    for line in open(path):
        m = _LINE.match(line.rstrip("\n"))
        if m:
            out[m.group(1)] = m.group(2).strip()
        else:                       # dash-free line (long JSON value)
            m2 = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(\{.*)$", line.rstrip("\n"))
            if m2:
                out[m2.group(1)] = m2.group(2).strip()
    return out


def build(args_txt: str, overrides: dict) -> list:
    from config import Config
    parser = Config.build_parser()
    defaults = vars(parser.parse_args([]))
    # dest -> (option string, is_store_true)
    form = {}
    for a in parser._actions:
        if not a.option_strings:
            continue
        cls = a.__class__.__name__
        if cls == "BooleanOptionalAction":
            form[a.dest] = (list(a.option_strings), "bool_optional")
        elif cls == "_StoreTrueAction":
            form.setdefault(a.dest, ([max(a.option_strings, key=len)], "store_true"))
        elif cls == "_StoreFalseAction":
            # a store_false sibling only matters if the dest defaults True
            form.setdefault(a.dest, ([max(a.option_strings, key=len)], "store_false"))
        else:
            form[a.dest] = ([max(a.option_strings, key=len)], "value")
    recorded = read_args_txt(args_txt)
    recorded.update({k: str(v) for k, v in overrides.items()})
    cmd = []
    for key in sorted(recorded):
        if key in SKIP or key not in form or key not in defaults:
            continue
        val = recorded[key]
        val = RECORDED_TO_CLI.get(key, {}).get(val, val)
        if val == str(defaults[key]) and key not in overrides:
            continue                                   # already the default
        opt, kind = form[key]
        if kind == "bool_optional":
            # argparse.BooleanOptionalAction takes NO value: --x / --no-x
            on = [o for o in opt if not o.startswith("--no")][0]
            off = [o for o in opt if o.startswith("--no")]
            if val == "True" and defaults[key] is not True:
                cmd.append(on)
            elif val == "False" and defaults[key] is not False and off:
                cmd.append(off[0])
            continue
        name = opt[0] if isinstance(opt, list) else opt
        if kind == "store_true":
            if val == "True":
                cmd.append(name)                       # bare switch
            continue
        if kind == "store_false":
            if val == "False":
                cmd.append(name)
            continue
        if val == "None":
            continue
        cmd += [name, val]
    return cmd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_args", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="k=v")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    ov = dict(s.split("=", 1) for s in a.set)
    cmd = build(a.from_args, ov)
    print(" ".join(["python", "train_siglip2.py", "--tag", a.tag] + cmd))
