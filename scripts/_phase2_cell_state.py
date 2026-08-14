"""What still has to be done for one Phase 2 cell (§18.5).

The launcher used to decide "already complete" from the extraction validator
alone, so a cell whose NPZs were valid but whose bio-eval and NMI had never run
-- or had run and failed -- was skipped outright and silently produced no
numbers at all. That is exactly how the 15 Phase 2 cells ended up bound to
their inputs but with no admissible metrics.

Completion is therefore two conditions, not one: the extraction has to validate,
AND the analysis marker has to exist and still be bound to those same inputs.
This prints one word on stdout so the shell can branch on it, and the reason on
stderr for a human:

    complete   extraction valid, analysis marker valid and bound -> skip
    analysis   extraction valid, analysis missing/stale -> re-run metrics only
    invalid    extraction itself does not validate -> refuse or re-infer
    absent     nothing there yet -> full run

It lives in a file rather than a heredoc because the heredoc ran with whatever
cwd the caller happened to have, and `from dna_utils...` then failed with
ModuleNotFoundError; under `set -e` that aborted the launcher with an import
traceback instead of a decision.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import (  # noqa: E402
    ExtractionInvalid,
    describe_failure,
    read_analysis_marker,
)


def cell_state(cell: str, *, allow_backfilled: bool = False) -> tuple[str, str]:
    path = Path(cell)
    if not path.is_dir():
        return "absent", f"{cell}: no such directory"
    if not any(path.iterdir()):
        return "absent", f"{cell}: empty"

    reason = describe_failure(str(path), allow_backfilled=allow_backfilled)
    if reason:
        return "invalid", reason
    try:
        read_analysis_marker(str(path), allow_backfilled=allow_backfilled)
    except ExtractionInvalid as error:
        return "analysis", str(error)
    return "complete", ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell")
    parser.add_argument("--allow-backfilled", action="store_true")
    args = parser.parse_args()

    state, reason = cell_state(args.cell,
                               allow_backfilled=args.allow_backfilled)
    print(state)
    if reason:
        print(reason, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
