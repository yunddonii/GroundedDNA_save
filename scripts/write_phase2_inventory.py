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
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import (  # noqa: E402
    ExtractionInvalid,
    validate_extraction_run,
)
from dna_utils.runtime_state import sha256_file  # noqa: E402
from scripts.aggregate_phase2_f01 import LEGACY  # noqa: E402

PHASE2_ROOT = REPO / "result_diagnostic" / "phase2_F01_only"
OUT = REPO / "docs" / "phase2_extraction_inventory.json"
_VALIDATOR = REPO / "dna_utils" / "extraction_validation.py"


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True, timeout=30, check=True).stdout.strip()
    except Exception:                       # noqa: BLE001 - inventory is best-effort here
        return None


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

    payload = {
        "schema_version": 1,
        "what_this_is": (
            "Digest-level inventory of the Phase 2 F01 diagnostic snapshot. The "
            "arrays themselves are untracked; this records exactly which bytes "
            "were validated and under which contract, so the claim can be "
            "re-checked rather than taken on trust."),
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
        "cells_validated": len(cells),
        "cells_refused": refused,
        "cells": cells,
    }
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(cells)} validated, {len(refused)} refused")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
