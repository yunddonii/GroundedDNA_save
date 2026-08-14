"""Phase 2: what the epoch-0 extraction bug (F01) actually cost.

Pairs each seed-42 N candidate's LEGACY artefacts -- produced by an extraction
that ran the router at the initial Sinkhorn epsilon because `_current_epoch` is
a plain int absent from the state dict -- against a re-inference of the SAME
checkpoint with the epoch restored.

Only cells whose two sides used the same GC window are compared. The legacy P0
cell runner passed `--gc_min 0.416 --gc_max 0.584`, which is count [7, 8] at 15
bases against the central policy's [6, 9]; a pair straddling those windows would
measure the window change, not F01, so it is reported as unpaired rather than
quietly differenced.

This is a diagnostic of the defect's impact. It does NOT select N -- that is
F02/F03's train-only protocol in Phase 3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.gc_policy import resolve_gc_policy  # noqa: E402

#: dataset/N -> legacy run directory, as audited against args.txt before launch.
LEGACY = {
    ("cifar10", 4): "result/260811+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e4+bs+64+e+5+proj_lr+0.001",
    ("cifar10", 9): "result/260811+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e9+bs+64+e+10+proj_lr+0.001",
    ("cifar10", 19): "result/260811+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e19+bs+64+e+20+proj_lr+0.001",
    ("cifar10", 39): "result/260812+cifar10_setting1_promptAblA_cifar_A_v4_P0refit_e39+bs+64+e+40+proj_lr+0.001",
    ("flickr25k", 4): "result/260811+flickr25k_setting1_promptAblA_flickr_A_v4_P0refit_e4+bs+64+e+5+proj_lr+0.001",
    ("flickr25k", 9): "result/260811+flickr25k_setting1_promptAblA_flickr_A_v4_P0refit_e9+bs+64+e+10+proj_lr+0.001",
    ("flickr25k", 19): "result/260812+flickr25k_setting1_promptAblA_flickr_A_v4_P0refit_e19+bs+64+e+20+proj_lr+0.001",
    ("nuswide", 4): "result/260811+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e4+bs+64+e+5+proj_lr+0.001",
    ("nuswide", 9): "result/260811+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e9+bs+64+e+10+proj_lr+0.001",
    ("nuswide", 19): "result/260811+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e19+bs+64+e+20+proj_lr+0.001",
    ("nuswide", 39): "result/260812+nuswide_setting1_promptAblA_nuswide_A_v4_P0refit_e39+bs+64+e+40+proj_lr+0.001",
    ("mscoco", 4): "result/260811+mscoco_setting1_promptAblA_mscoco_A_v5b_P0refit_e4+bs+64+e+5+proj_lr+0.001",
    ("mscoco", 9): "result/260811+mscoco_setting1_promptAblA_mscoco_A_v5b_P0refit_e9+bs+64+e+10+proj_lr+0.001",
    ("mscoco", 19): "result/260811+mscoco_setting1_promptAblA_mscoco_A_v5b_P0refit_e19+bs+64+e+20+proj_lr+0.001",
    ("mscoco", 39): "result/260812+mscoco_setting1_promptAblA_mscoco_A_v5b_s42_P0refit_e39+bs+64+e+40+proj_lr+0.001",
}
ORDER = ("cifar10", "flickr25k", "nuswide", "mscoco")
TOTAL_BASES = 15


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _nmi(run_dir: Path):
    payload = _json(run_dir / "pairwise_nmi.json")
    if not isinstance(payload, dict):
        return None
    if "mean_off_diag_nmi" in payload:
        return float(payload["mean_off_diag_nmi"])
    # The combined layout keys by result directory.
    for value in payload.values():
        if isinstance(value, dict) and "mean_off_diag_nmi" in value:
            return float(value["mean_off_diag_nmi"])
    return None


def _require_manifests(run_dir: Path) -> dict:
    """Admit a cell only if the shared strict validator admits it.

    The first version checked that two JSON files existed and that four fields
    agreed. A probe with no NPZ, no checkpoint, no config, `schema_version=999`
    and `dataset=WRONG` was accepted, so `15 paired / 0 unpaired` said nothing
    about provenance. The validator opens every file the manifests name,
    recomputes every digest, and checks the code arrays themselves.

    `allow_backfilled=True` because Phase 2 IS the retrospectively bound
    diagnostic; a paper path must not pass it.
    """
    from dna_utils.extraction_validation import (
        ExtractionInvalid, validate_extraction_run)
    try:
        validated = validate_extraction_run(
            str(run_dir), required_splits=("db", "query"),
            allow_backfilled=True)
    except ExtractionInvalid as error:
        raise ManifestMissing(str(error)) from None
    return validated.splits


class ManifestMissing(RuntimeError):
    """A cell has metrics but no admissible provenance."""


def _side(run_dir: Path, fallback: Path | None = None):
    """Metrics plus the GC window they were projected under.

    Two legacy cells never had every artefact written (cifar10/N19 has no
    bio-projection JSON, mscoco/N39 no NMI). `fallback` points at a directory
    holding those quantities RECOMPUTED FROM THE LEGACY EXTRACTION -- the same
    epoch-0 codes the other legacy cells were scored from, so the comparison
    stays like-for-like. Nothing is written back into the legacy run.
    """
    bioproj = _json(run_dir / "evaluation_siglip2_base_bioproj.json")
    recomputed = []
    if not isinstance(bioproj, dict) and fallback is not None:
        bioproj = _json(fallback / "evaluation_siglip2_base_bioproj.json")
        if isinstance(bioproj, dict):
            recomputed.append("bio_projection")
    if not isinstance(bioproj, dict):
        return None
    nmi = _nmi(run_dir)
    if nmi is None and fallback is not None:
        nmi = _nmi(fallback)
        if nmi is not None:
            recomputed.append("pairwise_nmi")
    return {
        "dir": str(run_dir),
        "recomputed_from_legacy_extraction": recomputed,
        "map_at_R_bioproj": bioproj.get("mAP_at_R"),
        "full_map_bioproj": bioproj.get("mAP"),
        "dna_unique_db": bioproj.get("unique_code_ratio"),
        "nmi": nmi,
        "gc_min_frac": bioproj.get("bio_gc_min_frac"),
        "gc_max_frac": bioproj.get("bio_gc_max_frac"),
    }


def _delta(new, old):
    if new is None or old is None:
        return None
    return float(new) - float(old)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase2-root",
        default=str(REPO / "result_diagnostic" / "phase2_F01_only"))
    parser.add_argument(
        "--legacy-recomputed",
        default=str(REPO / "result_diagnostic" / "phase2_legacy_recomputed"),
        help=("Quantities recomputed from the LEGACY extraction for cells whose "
              "original run never wrote them. Never written back into the "
              "legacy run directory."))
    parser.add_argument("--out-json", default=str(
        REPO / "docs" / "phase2_f01_impact.json"))
    parser.add_argument("--out-md", default=str(
        REPO / "docs" / "phase2_f01_impact.md"))
    args = parser.parse_args()

    policy = resolve_gc_policy(TOTAL_BASES)
    root = Path(args.phase2_root)
    cells, unpaired = [], []
    for dataset in ORDER:
        for (ds, n), legacy_rel in sorted(LEGACY.items()):
            if ds != dataset:
                continue
            cell_dir = root / f"{ds}_N{n}"
            try:
                manifests = _require_manifests(cell_dir)
            except ManifestMissing as error:
                unpaired.append({"cell": f"{ds}/N{n}", "reason": str(error)})
                continue
            new = _side(cell_dir)
            old = _side(REPO / legacy_rel,
                        fallback=Path(args.legacy_recomputed) / f"{ds}_N{n}")
            if new is None or old is None:
                unpaired.append({
                    "cell": f"{ds}/N{n}",
                    "reason": ("phase2 side missing" if new is None
                               else "legacy side missing"),
                })
                continue
            if (new["gc_min_frac"], new["gc_max_frac"]) != (
                    old["gc_min_frac"], old["gc_max_frac"]):
                unpaired.append({
                    "cell": f"{ds}/N{n}",
                    "reason": "GC windows differ; the difference would measure "
                              "the window change, not F01",
                    "phase2_gc": [new["gc_min_frac"], new["gc_max_frac"]],
                    "legacy_gc": [old["gc_min_frac"], old["gc_max_frac"]],
                })
                continue
            cells.append({
                "cell": f"{ds}/N{n}", "dataset": ds, "N": n,
                "legacy": old, "phase2": new,
                "legacy_recomputed": old["recomputed_from_legacy_extraction"],
                "provenance": {
                    "checkpoint_sha256": manifests["db"]["checkpoint_sha256"],
                    "inference_epoch": manifests["db"]["inference_epoch"],
                    "inference_epoch_source":
                        manifests["db"]["inference_epoch_source"],
                    "effective_sinkhorn_epsilon":
                        manifests["db"]["effective_sinkhorn_epsilon"],
                    "backfilled": bool(manifests["db"].get("backfilled", False)),
                },
                "delta": {
                    key: _delta(new[key], old[key])
                    for key in ("map_at_R_bioproj", "full_map_bioproj",
                                "dna_unique_db", "nmi")
                },
            })

    def _mean(key):
        values = [c["delta"][key] for c in cells if c["delta"][key] is not None]
        return statistics.fmean(values) if values else None

    payload = {
        "what_this_measures": (
            "F01 only: the same checkpoint re-inferred with the training epoch "
            "restored, so the router runs at the annealed Sinkhorn epsilon "
            "instead of the initial one. Not an N selection."),
        "total_bases": TOTAL_BASES,
        "gc_policy": policy.as_manifest_record(),
        "paired_cells": len(cells),
        "unpaired": unpaired,
        "mean_delta": {
            key: _mean(key)
            for key in ("map_at_R_bioproj", "full_map_bioproj",
                        "dna_unique_db", "nmi")
        },
        "cells": cells,
    }
    Path(args.out_json).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def fmt(value, places=4):
        return "-" if value is None else f"{float(value):+.{places}f}"

    def raw(value, places=4):
        return "-" if value is None else f"{float(value):.{places}f}"

    lines = [
        "# Phase 2 — what the epoch-0 extraction bug (F01) cost",
        "",
        payload["what_this_measures"],
        "",
        f"- Paired cells: **{len(cells)} / {len(LEGACY)}**",
        f"- GC window (both sides): count "
        f"[{policy.gc_min_count}, {policy.gc_max_count}] at {TOTAL_BASES} bases "
        f"(`{policy.policy_version}`)",
        "",
        "| cell | mAP@R legacy | mAP@R fixed | Δ | DNA-uniq legacy | DNA-uniq fixed | Δ | NMI legacy | NMI fixed | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        lines.append(
            f"| {c['cell']} "
            f"| {raw(c['legacy']['map_at_R_bioproj'])} "
            f"| {raw(c['phase2']['map_at_R_bioproj'])} "
            f"| {fmt(c['delta']['map_at_R_bioproj'])} "
            f"| {raw(c['legacy']['dna_unique_db'])} "
            f"| {raw(c['phase2']['dna_unique_db'])} "
            f"| {fmt(c['delta']['dna_unique_db'])} "
            f"| {raw(c['legacy']['nmi'])} "
            f"| {raw(c['phase2']['nmi'])} "
            f"| {fmt(c['delta']['nmi'])} |")
    lines += [
        "",
        "Mean Δ (fixed − legacy): "
        f"mAP@R {fmt(payload['mean_delta']['map_at_R_bioproj'])}, "
        f"DNA-uniq {fmt(payload['mean_delta']['dna_unique_db'])}, "
        f"NMI {fmt(payload['mean_delta']['nmi'])}.",
    ]
    if unpaired:
        lines += ["", "## Unpaired", ""]
        lines += [f"- `{u['cell']}`: {u['reason']}" for u in unpaired]
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}: "
          f"{len(cells)} paired, {len(unpaired)} unpaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
