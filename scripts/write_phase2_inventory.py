"""A tracked inventory so a fresh clone can re-verify the Phase 2 snapshot (§17.11).

The 30 manifests and ~37 GiB of NPZs live under untracked `result_diagnostic/`,
and `docs/phase2_f01_impact.json` carries only the checkpoint SHA, epoch,
epsilon and the `backfilled` flag. A fresh clone therefore cannot check the
"15/15 bound" claim at all -- it has the conclusion and none of the evidence.

Large arrays do not belong in git, but their digests do. This writes a compact
tracked file holding, per cell and split, the digest of every artefact, the row
counts and geometry, the runtime identity, the canonical legacy source it was
bound to, and the version and result of the strict validator that admitted it.
Someone with the data can then re-run the same checks; someone without it can at
least see exactly what was claimed and about which bytes.

It covers both halves of a cell (§19.7). Recording only the extraction side left
the question the audit actually asks -- which numbers came out, under which
evaluator and which protocol -- unanswerable from a fresh clone, since the
metric JSONs are as untracked as the NPZs. So each entry also carries the
analysis marker (its metrics, protocol constants, evaluator digests and the
input binding it sealed) and the digests of the metric files beside it, plus the
execution context: when it ran, by which command, at which commit, and whether
the tree was dirty at the time. A digest taken from a dirty tree is not
reproducible and has to say so.

The output is written atomically. A half-written inventory that still parses is
worse than none, because it reads as a complete record of a smaller run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import (  # noqa: E402
    ANALYSIS_MARKER_NAME,
    ExtractionInvalid,
    read_analysis_marker,
    validate_extraction_run,
)
from dna_utils.runtime_state import sha256_file  # noqa: E402
from scripts.aggregate_phase2_f01 import LEGACY  # noqa: E402

PHASE2_ROOT = REPO / "result_diagnostic" / "phase2_F01_only"
OUT = REPO / "docs" / "phase2_extraction_inventory.json"
_VALIDATOR = REPO / "dna_utils" / "extraction_validation.py"


_METRIC_FILES = (
    "cell_result.json",
    "evaluation_siglip2_base_bioproj.json",
    "pairwise_nmi.json",
)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True,
            text=True, timeout=30, check=True).stdout.strip()
    except Exception:                       # noqa: BLE001 - inventory is best-effort here
        return None


def _git_head() -> str | None:
    return _git("rev-parse", "HEAD")


def _execution_context(argv: list[str]) -> dict:
    """When, by what, at which commit -- and whether that commit describes it.

    A digest set produced from a dirty tree cannot be reproduced from the
    recorded commit, so the record says so rather than implying otherwise.
    """
    status = _git("status", "--porcelain")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "command": " ".join(argv),
        "git_head": _git_head(),
        "git_tree_dirty": None if status is None else bool(status),
        "reproducible_from_git_head": (
            None if status is None else not status),
        "host_repo": str(REPO),
    }


def _analysis_entry(cell: Path) -> dict:
    """The metric half of the cell, or the reason there isn't one."""
    entry = {
        "marker_present": (cell / ANALYSIS_MARKER_NAME).is_file(),
        "marker_sha256": None,
        "admitted": False,
        "refusal": None,
        "metrics": None,
        "protocol": None,
        "analysis_sources": None,
        "input_binding": None,
        "metric_files": {},
    }
    for name in _METRIC_FILES:
        path = cell / name
        entry["metric_files"][name] = (
            sha256_file(str(path)) if path.is_file() else None)

    if not entry["marker_present"]:
        entry["refusal"] = "no analysis marker; this cell carries no admissible numbers"
        return entry
    entry["marker_sha256"] = sha256_file(str(cell / ANALYSIS_MARKER_NAME))
    try:
        payload = read_analysis_marker(str(cell), allow_backfilled=True)
    except ExtractionInvalid as error:
        entry["refusal"] = str(error)
        return entry
    entry.update(
        admitted=True,
        metrics=dict(payload["metrics"]),
        protocol=dict(payload["protocol"]),
        analysis_sources=dict(payload["analysis_sources"]),
        input_binding=dict(payload["input_binding"]),
    )
    return entry


def _cell_entry(cell: Path) -> dict:
    run = validate_extraction_run(str(cell), allow_backfilled=True)
    dataset, n = cell.name.rsplit("_N", 1)
    legacy = REPO / LEGACY[(dataset, int(n))]

    splits = {}
    for split, manifest in sorted(run.splits.items()):
        splits[split] = {
            "npz_path_relative": str(
                Path(manifest["npz_path"]).relative_to(REPO)),
            "npz_sha256": manifest["npz_sha256"],
            "manifest_sha256": sha256_file(
                str(cell / f"extraction_manifest_{split}.json")),
            "n_rows": manifest["n_rows"],
        }

    entry = {
        "cell": cell.name,
        "dataset": dataset,
        "stop_epoch_zero_based": int(n),
        "splits": splits,
        "runtime_identity": dict(run.common),
        "checkpoint_sha256": run.common["checkpoint_sha256"],
        "config_sha256": run.common["config_sha256"],
        "canonical_legacy_source": {
            "dir": str(legacy.relative_to(REPO)),
            "model_state_dict_sha256": sha256_file(
                str(legacy / "model_state_dict.pth")),
            "config_sha256": sha256_file(str(legacy / "config.pt")),
        },
        "completion_marker_sha256": sha256_file(
            str(cell / "extraction_complete.json")),
        "backfilled": run.backfilled,
        "eligibility": "diagnostic_only",
        "analysis": _analysis_entry(cell),
    }
    for name in ("args.txt", "extract.log"):
        path = cell / name
        entry[f"{name.replace('.', '_')}_sha256"] = (
            sha256_file(str(path)) if path.is_file() else None)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PHASE2_ROOT))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    root = Path(args.root)
    cells, refused = [], []
    for cell in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            cells.append(_cell_entry(cell))
        except ExtractionInvalid as error:
            refused.append({"cell": cell.name, "reason": str(error)})

    with_metrics = [c for c in cells if c["analysis"]["admitted"]]
    payload = {
        "schema_version": 2,
        "what_this_is": (
            "Digest-level inventory of the Phase 2 F01 diagnostic snapshot, "
            "covering both the extraction and the metric half of every cell. "
            "The arrays and metric JSONs themselves are untracked; this records "
            "exactly which bytes were validated, under which contract and which "
            "evaluator, so the claim can be re-checked rather than taken on "
            "trust."),
        "eligibility": "diagnostic_only_never_promoted_to_paper_main",
        "external_root": str(root),
        "validator": {
            "module": "dna_utils/extraction_validation.py",
            "sha256": sha256_file(str(_VALIDATOR)),
            "allow_backfilled": True,
            "required_splits": ["db", "query"],
        },
        "generator": {
            "script": "scripts/write_phase2_inventory.py",
            "sha256": sha256_file(__file__),
            "git_head": _git_head(),
        },
        "execution": _execution_context([sys.argv[0], *sys.argv[1:]]),
        "cells_validated": len(cells),
        "cells_with_admissible_metrics": len(with_metrics),
        "cells_refused": refused,
        "cells": cells,
    }

    # Atomic: a truncated inventory that still parses reads as a complete
    # record of a smaller run.
    out = Path(args.out)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out)

    print(f"wrote {args.out}: {len(cells)} validated, "
          f"{len(with_metrics)} with admissible metrics, "
          f"{len(refused)} refused")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
