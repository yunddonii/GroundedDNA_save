"""Bind the LEGACY side of the Phase 2 delta so one evaluator can score both.

The F01 impact is a difference between two numbers. The fixed side is recomputed
from a committed evaluator; the legacy side was whatever ran in August, under an
older bio-projection and an older NMI. A difference between two evaluators is
not a measurement of F01 (§19.4).

Scoring the legacy extraction with today's evaluator needs the legacy NPZs to be
bound to their inputs the way the fixed side is. This builds that binding in a
SEPARATE root and never writes into the legacy run directory: each cell holds
symlinks to the legacy NPZs, checkpoint and config, a copy of `args.txt`, and
manifests naming exactly those bytes.

What cannot be proved, and is therefore stated rather than implied: the legacy
runs have no `extract.log`, so their inference epoch is not recorded anywhere.
It is 0 because that is what the code did -- `_current_epoch` is a plain int,
absent from the state dict, so a fresh model loaded raw weights and ran the
router at the INITIAL epsilon. That is defect F01 itself. The manifests record
`inference_epoch: 0` with source `f01_unrestored`, and `backfilled: true` with
the reason, so no reader can mistake this for an extraction that recorded its
own epoch.

Usage:
    python scripts/bind_legacy_phase2.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from dna_utils.extraction_validation import (  # noqa: E402
    ExtractionInvalid,
    validate_code_arrays,
)
from dna_utils.runtime_state import (  # noqa: E402
    ResolvedEpoch,
    annealed_epsilon,
    sha256_file,
    write_extraction_manifest,
)
from scripts.aggregate_phase2_f01 import LEGACY  # noqa: E402

OUT_ROOT = REPO / "result_diagnostic" / "phase2_legacy_bound"
_SPLITS = (("db", "extract_db.npz"), ("query", "extract_query.npz"))
_LINKED = ("extract_db.npz", "extract_query.npz", "model_state_dict.pth",
           "config.pt")

#: The epoch the legacy extraction ran at, and why it is not read from an
#: artefact. F01 IS this value: nothing restored the epoch, so it stayed 0.
_LEGACY_EPOCH = 0
_LEGACY_SOURCE = "f01_unrestored"


class BindRefused(RuntimeError):
    """A legacy cell cannot be bound, so nothing was written for it."""


def _arg(args_txt: Path, field: str) -> str:
    pattern = re.compile(rf"^{re.escape(field)}-+(.*)$")
    for line in args_txt.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    raise BindRefused(f"{args_txt} has no field {field!r}")


def _atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def bind_cell(dataset: str, n: int, *, dry_run: bool = False) -> list[str]:
    legacy = REPO / LEGACY[(dataset, n)]
    if not legacy.is_dir():
        raise BindRefused(f"{dataset}/N{n}: no legacy run at {legacy}")
    for name in _LINKED:
        if not (legacy / name).is_file():
            raise BindRefused(f"{dataset}/N{n}: legacy run has no {name}")

    args_txt = legacy / "args.txt"
    if not args_txt.is_file():
        raise BindRefused(f"{dataset}/N{n}: legacy run has no args.txt")
    slots = int(_arg(args_txt, "num_semantic_parts"))
    per_slot = int(_arg(args_txt, "num_codons_per_codebook"))
    codebook_size = int(_arg(args_txt, "codebook_size"))
    budget = int(_arg(args_txt, "epoch"))
    stop = int(_arg(args_txt, "stop_after_epoch"))
    if stop != n:
        raise BindRefused(
            f"{dataset}/N{n}: args.txt stop_after_epoch={stop}, not {n}; this "
            f"is not the run the cell claims to be")

    eps_init = float(_arg(args_txt, "sinkhorn_epsilon_init"))
    eps_final = float(_arg(args_txt, "sinkhorn_epsilon_final"))
    # At epoch 0 the cosine schedule is exactly the initial epsilon. That is the
    # operating point the codes were produced at, and the whole point of F01.
    epsilon = annealed_epsilon(_LEGACY_EPOCH, budget, eps_init, eps_final)

    # Validate every split before writing anything for the cell.
    checked = {}
    for split, npz_name in _SPLITS:
        npz = legacy / npz_name
        with np.load(npz, allow_pickle=False) as stored:
            for name in ("base_indices", "hash_2bit", "codebook_indices"):
                if name not in stored:
                    raise BindRefused(f"{dataset}/N{n}/{split}: no {name}")
            codes = np.asarray(stored["base_indices"])
            hashed = np.asarray(stored["hash_2bit"])
            codebook = np.asarray(stored["codebook_indices"])
        if codes.ndim != 2:
            raise BindRefused(
                f"{dataset}/N{n}/{split}: base_indices has rank {codes.ndim}")
        rows = int(codes.shape[0])
        try:
            validate_code_arrays(
                codes, hashed, codebook, rows=rows, slots=slots,
                per_slot=per_slot, codebook_size=codebook_size,
                split=f"{dataset}/N{n}/{split}")
        except ExtractionInvalid as error:
            raise BindRefused(str(error)) from None
        checked[split] = rows

    cell = OUT_ROOT / f"{dataset}_N{n}"
    if dry_run:
        return [f"would bind {cell} -> {legacy}"]

    cell.mkdir(parents=True, exist_ok=True)
    # Invalidate before writing: a stale marker beside half-written manifests
    # reads as a complete cell to every consumer.
    marker_path = cell / "extraction_complete.json"
    if marker_path.exists():
        marker_path.unlink()

    # Symlinks, not copies: the evaluator loads `extract_db.npz` from the cell
    # directory, and duplicating tens of gigabytes to say so would be absurd.
    # The legacy run itself is never written to.
    for name in _LINKED:
        link = cell / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(legacy / name)
    (cell / "args.txt").write_text(args_txt.read_text(encoding="utf-8"),
                                   encoding="utf-8")

    resolved = ResolvedEpoch(
        epoch=_LEGACY_EPOCH, source=_LEGACY_SOURCE,
        effective_sinkhorn_epsilon=epsilon,
        sinkhorn_schedule_horizon=budget,
        checkpoint_sha256=sha256_file(str(legacy / "model_state_dict.pth")),
        sinkhorn_annealing_enabled=True)

    written = []
    for split, rows in sorted(checked.items()):
        out = cell / f"extraction_manifest_{split}.json"
        written.append(write_extraction_manifest(
            str(out), checkpoint_path=str(cell / "model_state_dict.pth"),
            resolved=resolved, num_slots=slots, bases_per_slot=per_slot,
            split=split, n_rows=rows,
            lr_schedule_horizon=budget, training_epoch_budget=budget,
            training_stop_epoch=stop,
            extra={
                "npz_path": str((cell / dict(_SPLITS)[split]).resolve()),
                "npz_sha256": sha256_file(str(legacy / dict(_SPLITS)[split])),
                "config_path": str((cell / "config.pt").resolve()),
                "config_sha256": sha256_file(str(legacy / "config.pt")),
                "dataset": dataset,
                "random_seed": int(_arg(args_txt, "random_seed")),
                "codebook_size": codebook_size,
                "backfilled": True,
                "backfill_reason": (
                    "the legacy run wrote no extract.log, so its inference "
                    "epoch is recorded nowhere. It is 0 because nothing "
                    "restored it -- defect F01 itself -- and the router "
                    "therefore ran at the initial epsilon. Every other field "
                    "is read from the preserved artefacts."),
                "legacy_source_dir": str(legacy),
                "args_sha256": sha256_file(str(args_txt)),
                "bind_tool_sha256": sha256_file(__file__),
            }))

    _atomic_write(marker_path, {
        "schema_version": 1,
        "splits": sorted(checked),
        "manifest_sha256": {
            split: sha256_file(str(cell / f"extraction_manifest_{split}.json"))
            for split in sorted(checked)},
        "backfilled": True,
    })
    written.append(str(marker_path))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ok, refused = [], []
    for (dataset, n) in sorted(LEGACY):
        try:
            written = bind_cell(dataset, n, dry_run=args.dry_run)
        except BindRefused as error:
            refused.append(str(error))
            continue
        ok.append((f"{dataset}_N{n}", len(written)))

    for name, count in ok:
        print(f"  {name:<16} {count} file(s)")
    if refused:
        print("\nREFUSED:")
        for reason in refused:
            print(f"  - {reason}")
    print(f"\n{len(ok)} of {len(LEGACY)} legacy cells bound"
          f"{' (dry run)' if args.dry_run else ''}")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
