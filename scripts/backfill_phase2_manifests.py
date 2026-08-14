"""Write the missing extraction manifests for the Phase 2 diagnostic cells.

The 15 Phase 2 cells were produced before `_write_split_manifest` existed, so
the root holds 15 DB and 15 query NPZs and 0 manifests. The launcher skipped a
cell whenever `extract_db.npz` existed, so simply re-running it backfills
nothing, and the aggregator accepted the manifest-free root as
`15 paired, 0 unpaired`.

Backfill rather than re-inference was chosen deliberately: Phase 2 is a
diagnostic of F01's impact and is never promoted to a paper result, so the
question is whether the artefacts can be *bound* to their inputs, not whether
they can be reproduced bit-for-bit on new hardware. This tool answers the
binding question and refuses whenever it cannot.

What it verifies before writing anything, per cell:

  * the copied checkpoint and config are byte-identical to the canonical legacy
    source they were taken from, so the cell's inputs are provable;
  * the recorded inference epoch in `extract.log` is the cell's own N and the
    source is `explicit_flag`, i.e. the F01 fix was actually in force;
  * the NPZ geometry matches the cell's configuration.

Every manifest it writes is marked `backfilled: true` with the reason, so no
reader can mistake it for a manifest emitted by the extraction itself.

Usage:
    python scripts/backfill_phase2_manifests.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from dna_utils.runtime_state import (  # noqa: E402
    ResolvedEpoch,
    sha256_file,
    write_extraction_manifest,
)
from scripts.aggregate_phase2_f01 import LEGACY  # noqa: E402

PHASE2_ROOT = REPO / "result_diagnostic" / "phase2_F01_only"
_SPLITS = (("db", "extract_db.npz"), ("query", "extract_query.npz"))
_EPOCH_LINE = re.compile(
    r"inference epoch=(\d+) \(source=(\w+)\) "
    r"effective_sinkhorn_epsilon=([0-9.eE+-]+)")


class BackfillRefused(RuntimeError):
    """A cell could not be bound to its inputs, so no manifest was written."""


def _arg(args_txt: Path, field: str) -> str:
    pattern = re.compile(rf"^{re.escape(field)}-+(.*)$")
    for line in args_txt.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    raise BackfillRefused(f"{args_txt} has no field {field!r}")


def _runtime_from_log(cell: Path) -> tuple[int, str, float]:
    log = cell / "extract.log"
    if not log.is_file():
        raise BackfillRefused(f"{cell.name}: no extract.log to read the "
                              f"resolved epoch from")
    text = log.read_text(encoding="utf-8", errors="ignore")
    matches = _EPOCH_LINE.findall(text)
    if not matches:
        raise BackfillRefused(
            f"{cell.name}: extract.log records no resolved inference epoch")
    epochs = {(int(e), src) for e, src, _ in matches}
    if len(epochs) != 1:
        raise BackfillRefused(
            f"{cell.name}: extract.log records several runtime states {epochs}")
    epoch, source = next(iter(epochs))
    return epoch, source, float(matches[0][2])


def backfill_cell(cell: Path, *, dry_run: bool = False) -> list[str]:
    dataset, n = cell.name.rsplit("_N", 1)
    legacy_rel = LEGACY.get((dataset, int(n)))
    if legacy_rel is None:
        raise BackfillRefused(f"{cell.name}: no canonical legacy source")
    legacy = REPO / legacy_rel

    # 1. the copied inputs must be byte-identical to the legacy source
    for name in ("model_state_dict.pth", "config.pt"):
        here, there = cell / name, legacy / name
        if not here.is_file() or not there.is_file():
            raise BackfillRefused(f"{cell.name}: missing {name}")
        if sha256_file(str(here)) != sha256_file(str(there)):
            raise BackfillRefused(
                f"{cell.name}: {name} differs from its legacy source; the cell "
                f"cannot be bound to the run it claims to come from")

    # 2. the run must have used the F01 fix, at this cell's own epoch
    epoch, source, epsilon = _runtime_from_log(cell)
    if int(epoch) != int(n):
        raise BackfillRefused(
            f"{cell.name}: log says inference epoch {epoch}, expected {n}")
    if source != "explicit_flag":
        raise BackfillRefused(
            f"{cell.name}: epoch source is {source!r}, expected explicit_flag")

    args_txt = cell / "args.txt"
    if not args_txt.is_file():
        args_txt = legacy / "args.txt"
    slots = int(_arg(args_txt, "num_semantic_parts"))
    per_slot = int(_arg(args_txt, "num_codons_per_codebook"))

    resolved = ResolvedEpoch(
        epoch=int(epoch), source=source,
        effective_sinkhorn_epsilon=epsilon,
        sinkhorn_schedule_horizon=int(_arg(args_txt, "epoch")),
        checkpoint_sha256=sha256_file(str(cell / "model_state_dict.pth")))

    written = []
    for split, npz_name in _SPLITS:
        npz = cell / npz_name
        if not npz.is_file():
            raise BackfillRefused(f"{cell.name}: missing {npz_name}")
        with np.load(npz, allow_pickle=False) as stored:
            codes = np.asarray(stored["base_indices"])
        if codes.ndim != 2 or codes.shape[1] != slots * per_slot:
            raise BackfillRefused(
                f"{cell.name}/{split}: base_indices {codes.shape} does not "
                f"match {slots}x{per_slot}")
        out = cell / f"extraction_manifest_{split}.json"
        if dry_run:
            written.append(f"would write {out}")
            continue
        write_extraction_manifest(
            str(out), checkpoint_path=str(cell / "model_state_dict.pth"),
            resolved=resolved, num_slots=slots, bases_per_slot=per_slot,
            split=split, n_rows=int(codes.shape[0]),
            extra={
                "npz_path": str(npz.resolve()),
                "npz_sha256": sha256_file(str(npz)),
                "config_path": str((cell / "config.pt").resolve()),
                "config_sha256": sha256_file(str(cell / "config.pt")),
                "dataset": dataset,
                "random_seed": int(_arg(args_txt, "random_seed")),
                "codebook_size": int(_arg(args_txt, "codebook_size")),
                # Never let a reader mistake this for a manifest the extraction
                # itself emitted.
                "backfilled": True,
                "backfill_reason": (
                    "written after the fact from the preserved checkpoint, "
                    "config, args and extraction log; the cell's inputs were "
                    "verified byte-identical to its canonical legacy source"),
                "legacy_source_dir": str(legacy),
            })
        written.append(str(out))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PHASE2_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    cells = sorted(p for p in root.iterdir() if p.is_dir())
    ok, refused = [], []
    for cell in cells:
        try:
            written = backfill_cell(cell, dry_run=args.dry_run)
        except BackfillRefused as error:
            refused.append(str(error))
            continue
        ok.append((cell.name, len(written)))

    for name, count in ok:
        print(f"  {name:<16} {count} manifest(s)")
    if refused:
        print("\nREFUSED:")
        for reason in refused:
            print(f"  - {reason}")
    print(f"\n{len(ok)} of {len(cells)} cells bound"
          f"{' (dry run)' if args.dry_run else ''}")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
