"""Emit one ablation cell's effective flags, from the declared spec (F09).

The chain hand-wrote its cells in shell strings, and the strings were wrong in a
way the tags hid: the shared `S5` prefix carries `--no_gumbel_softmax`
unconditionally, so `A5none` and `A5nogumbel` composed to the SAME command, as
did `A5joint` and `A5both`. Four tags, two configurations -- and a test that
checked the four tags existed said the factorial was there.

`dna_utils/ablation_spec.py` already declares the cells correctly: A4 at matched
total capacity, a distinct `--selection_mode` per cell so two cells cannot share
a result directory, and A5 as the real 2x2. This prints them, and refuses to
print a cell whose composed flags do not survive the real parser.

Usage:
    python scripts/_ablation_cells.py --list A2 A4 A5 --dataset CIFAR10
    python scripts/_ablation_cells.py --flags A5_both --dataset CIFAR10 \
        --base "--num_semantic_parts 5 ..."
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dna_utils.ablation_spec import (  # noqa: E402
    ABLATIONS, AblationInvalid, preflight_ablation)

#: Flags a cell sets for itself and which therefore must NOT arrive from the
#: shared base recipe. `--no_gumbel_softmax` is the one that collapsed A5.
_CELL_OWNED = ("--no_gumbel_softmax", "--lambda_codon_joint",
               "--share_codebook", "--codebook_size", "--selection_mode",
               "--disable_text_supervision")


def _strip_owned(base: list) -> list:
    """Drop from the base anything a cell decides for itself.

    A base that already says `--no_gumbel_softmax` makes "with" and "without"
    the same command, whatever the cell adds.
    """
    out, i = [], 0
    takes_value = {"--lambda_codon_joint", "--codebook_size",
                   "--selection_mode"}
    while i < len(base):
        token = base[i]
        if token in _CELL_OWNED:
            i += 2 if token in takes_value else 1
            continue
        out.append(token)
        i += 1
    return out


def cells(names, dataset: str):
    from dna_utils.ablation_spec import _BUILDERS
    for name in names:
        if name not in _BUILDERS:
            raise AblationInvalid(f"unknown ablation {name!r}")
        for cell in _BUILDERS[name](dataset):
            yield cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", nargs="+", default=None)
    parser.add_argument("--flags", default=None, help="one cell name")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base", default="", help="shared recipe flags")
    args = parser.parse_args()

    base = _strip_owned(shlex.split(args.base))

    if args.list:
        for cell in cells(args.list, args.dataset):
            print(cell.name)
        return 0

    if not args.flags:
        parser.error("pass --list or --flags")
    for cell in cells(["A2", "A4", "A5"], args.dataset):
        if cell.name != args.flags:
            continue
        composed = base + list(cell.flags)
        preflight_ablation(composed, dataset=args.dataset)
        print(" ".join(shlex.quote(f) for f in composed))
        return 0
    print(f"[ablation-cells] unknown cell {args.flags!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
