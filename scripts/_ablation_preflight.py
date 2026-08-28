"""Parse an ablation's flags before any GPU time is spent on it (F09).

Two of the three ablation commands could not have run at all:
`--router_type mean` is not one of argparse's choices, and `--num_codebooks 1`
against M=5 inputs raises inside the quantizer's first forward. Both would have
died at the first cell, hours into a chain that had already trained the ones
before it.

Usage:
    python scripts/_ablation_preflight.py -- <flags...>
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main() -> int:
    flags = sys.argv[1:]
    if flags and flags[0] == "--":
        flags = flags[1:]
    sys.argv = ["train_siglip2.py"] + flags
    try:
        from config import Config
        args = Config.get_config()
    except SystemExit as error:            # argparse rejected a flag
        print(f"[ablation-preflight] refused: {error}", file=sys.stderr)
        return 1
    except Exception as error:             # noqa: BLE001 - report, do not mask
        print(f"[ablation-preflight] refused: {error}", file=sys.stderr)
        return 1

    # The A4 failure is not an argparse error: M is compared against the
    # quantizer's codebook count on the first forward, so it has to be checked
    # here rather than left to the trainer.
    slots = int(getattr(args, "num_semantic_parts", 0) or 0)
    codebooks = int(getattr(args, "num_codebooks", 0) or 0)
    if codebooks and slots and codebooks != slots:
        print(f"[ablation-preflight] refused: num_codebooks={codebooks} but "
              f"num_semantic_parts={slots}; the quantizer raises on the first "
              f"forward. Use --share_codebook to tie the slots to one "
              f"codebook.", file=sys.stderr)
        return 1
    print(f"[ablation-preflight] ok (M={slots}, codebooks={codebooks})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
