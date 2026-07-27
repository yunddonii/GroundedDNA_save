#!/usr/bin/env python3
"""Versioned 24-base entry point for the native-DNA trainer.

The canonical trainer predates the 24-base comparison and labels every
non-18-base terminal evaluation as ``original_length_only_adaptation``.  That
label is not correct for a matched 24-base comparison because the historical
methods used different native sequence lengths.  This entry point delegates
all training, extraction, and metric computation to the canonical trainer and
then atomically replaces only that terminal protocol label.

The wrapper is intentionally strict:

* ``--length`` must be exactly 24;
* a stage-1 selection run, which has no terminal evaluation, is untouched;
* a terminal evaluation must carry the exact legacy label and length before it
  can be relabelled; and
* no metric, extraction, checkpoint, or model configuration is modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import train_native_dna_baseline as canonical


LENGTH_BASES = 24
PROTOCOL_LABEL = "matched_24nt_adaptation"
LEGACY_NON18_LABEL = "original_length_only_adaptation"
GC_COUNT_RANGE = (10, 14)


def _invocation(argv: list[str]) -> tuple[int, Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--out", required=True)
    args, _ = parser.parse_known_args(argv)
    if args.length != LENGTH_BASES:
        raise ValueError(
            f"24-base trainer requires --length {LENGTH_BASES}, "
            f"found {args.length}"
        )
    return args.length, Path(args.out).expanduser().resolve()


def _normalized_evaluation(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    if result.get("length") != LENGTH_BASES:
        raise ValueError(
            "terminal evaluation length mismatch: expected "
            f"{LENGTH_BASES}, found {result.get('length')!r}"
        )
    if result.get("protocol") != LEGACY_NON18_LABEL:
        raise ValueError(
            "refusing to rewrite an unexpected terminal protocol label: "
            f"{result.get('protocol')!r}"
        )
    result["protocol"] = PROTOCOL_LABEL
    result["matched_capacity_bases"] = LENGTH_BASES
    result["common_dp_gc_count_range"] = list(GC_COUNT_RANGE)
    return result


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _normalize_terminal_metadata(output: Path) -> bool:
    evaluation = output / "evaluation_native_dna.json"
    if not evaluation.exists():
        return False
    with evaluation.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"terminal evaluation must be a JSON object: {evaluation}")
    _atomic_json(evaluation, _normalized_evaluation(loaded))
    return True


def main(argv: list[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    _, output = _invocation(forwarded)
    prior_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *forwarded]
        returncode = int(canonical.main())
    finally:
        sys.argv = prior_argv
    if returncode != 0:
        return returncode
    _normalize_terminal_metadata(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
