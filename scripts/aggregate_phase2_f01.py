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


#: What each dataset's cells must declare. The aggregator knows these from the
#: protocol, so it says so rather than accepting whatever the artefact claims:
#: fifteen copies of one CIFAR cell under fifteen names previously produced a
#: `complete: true` report of 15/15 (§22.4).
_DATASET_TOKEN = {"cifar10": "CIFAR10", "flickr25k": "Flickr25k",
                  "nuswide": "NUSWIDE", "mscoco": "MSCOCO"}
_CODEBOOK_SIZE = {"cifar10": 64, "flickr25k": 128, "nuswide": 128,
                  "mscoco": 128}
_MAP_R_CUTOFF = {"cifar10": 1000, "flickr25k": 5000, "nuswide": 5000,
                 "mscoco": 5000}
SEED = 42
SLOTS, BASES_PER_SLOT = 5, 3


def expected_identity(dataset: str, n: int, *, epoch: int,
                      epoch_source: str) -> "ExpectedIdentity":
    from dna_utils.extraction_validation import ExpectedIdentity
    return ExpectedIdentity(
        dataset=dataset, random_seed=SEED, inference_epoch=epoch,
        inference_epoch_source=epoch_source, num_slots=SLOTS,
        bases_per_slot=BASES_PER_SLOT,
        codebook_size=_CODEBOOK_SIZE[dataset])


def expected_protocol(dataset: str, policy) -> dict:
    return {
        "dataset": _DATASET_TOKEN[dataset],
        "codebook_size": _CODEBOOK_SIZE[dataset],
        "total_bases": TOTAL_BASES,
        "map_r_cutoff": _MAP_R_CUTOFF[dataset],
        "gc_policy_version": policy.policy_version,
        "gc_count_min_inclusive": policy.gc_min_count,
        "gc_count_max_inclusive": policy.gc_max_count,
    }


def _require_manifests(run_dir: Path, *, identity=None,
                       protocol: dict | None = None) -> dict:
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
            allow_backfilled=True, expected=identity)
    except ExtractionInvalid as error:
        raise ManifestMissing(str(error)) from None

    # §17.3, second half: the aggregator reads metric JSONs that live beside the
    # NPZs but were never tied to them. A stale metric next to a valid
    # extraction was admitted, so the reported numbers had no end-to-end
    # binding. Each metric file must now name the inputs it was computed from.
    from dna_utils.extraction_validation import read_analysis_marker
    try:
        marker = read_analysis_marker(str(run_dir), allow_backfilled=True,
                                      expected_protocol=protocol)
    except ExtractionInvalid as error:
        raise ManifestMissing(str(error)) from None
    # Return the payload the validator just admitted. Re-opening the file
    # afterwards is the §19.3 defect in miniature: the bytes that were checked
    # and the bytes that are read should be the same object.
    return validated.splits, marker


class ManifestMissing(RuntimeError):
    """A cell has metrics but no admissible provenance."""


def _side(run_dir: Path, sealed: dict):
    """Metrics plus the GC window they were projected under.

    BOTH sides come from a sealed marker now. The legacy side used to be read
    unsigned, from whatever `evaluation_siglip2_base_bioproj.json` happened to
    sit in the August run directory -- computed by an older bio-projection and
    an older NMI. A difference between two evaluators is not a measurement of
    F01 (§19.4), so the legacy extraction is re-scored by today's evaluator via
    `scripts/bind_legacy_phase2.py`, which binds the legacy NPZs in a separate
    root and never writes into the legacy run.
    """
    metrics = sealed["metrics"]
    protocol = sealed["protocol"]
    return {
        "dir": str(run_dir),
        "sealed": True,
        "map_at_R_bioproj": metrics["map_at_R_bioproj"],
        "full_map_bioproj": metrics["full_map_bioproj"],
        "dna_unique_db": metrics["dna_unique_db"],
        "nmi": metrics["mean_off_diag_nmi"],
        "gc_min_frac": protocol["gc_min_frac"],
        "gc_max_frac": protocol["gc_max_frac"],
        "analysis_sources": sealed["analysis_sources"],
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
        "--legacy-root",
        default=str(REPO / "result_diagnostic" / "phase2_legacy_bound"),
        help=("The LEGACY extraction re-scored by the same evaluator as the "
              "fixed side, bound by scripts/bind_legacy_phase2.py. Never "
              "written back into the legacy run directory."))
    parser.add_argument("--out-json", default=str(
        REPO / "docs" / "phase2_f01_impact.json"))
    parser.add_argument("--out-md", default=str(
        REPO / "docs" / "phase2_f01_impact.md"))
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help=("Write a `.partial` report beside the official paths when not "
              "every cell is admissible. The official files are never written "
              "from an incomplete run, and the exit status stays nonzero."))
    args = parser.parse_args()

    policy = resolve_gc_policy(TOTAL_BASES)
    root = Path(args.phase2_root)
    cells, unpaired = [], []
    for dataset in ORDER:
        for (ds, n), _legacy_rel in sorted(LEGACY.items()):
            if ds != dataset:
                continue
            cell_dir = root / f"{ds}_N{n}"
            legacy_dir = Path(args.legacy_root) / f"{ds}_N{n}"
            protocol = expected_protocol(ds, policy)
            try:
                # The fixed side was re-inferred AT this cell's own N; the
                # legacy side is the F01 defect, which never restored the epoch.
                manifests, marker = _require_manifests(
                    cell_dir, protocol=protocol,
                    identity=expected_identity(
                        ds, n, epoch=n, epoch_source="explicit_flag"))
                _, legacy_marker = _require_manifests(
                    legacy_dir, protocol=protocol,
                    identity=expected_identity(
                        ds, n, epoch=0, epoch_source="f01_unrestored"))
            except ManifestMissing as error:
                unpaired.append({"cell": f"{ds}/N{n}", "reason": str(error)})
                continue
            new = _side(cell_dir, marker)
            old = _side(legacy_dir, legacy_marker)

            # Both sides must have been scored by the SAME evaluator, or the
            # difference measures the evaluator change as much as F01.
            if new["analysis_sources"] != old["analysis_sources"]:
                unpaired.append({
                    "cell": f"{ds}/N{n}",
                    "reason": ("the two sides were scored by different "
                               "analysis sources; the delta would measure the "
                               "evaluator change, not F01"),
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
                "fixed_side_sealed": bool(new["sealed"]),
                "legacy_side_sealed": bool(old["sealed"]),
                "same_evaluator_both_sides": True,
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
        "expected_cells": len(LEGACY),
        "complete": len(cells) == len(LEGACY),
    }

    # A report covering 1 of 15 cells is not a smaller version of the answer,
    # it is a different claim -- and it used to be written straight over the
    # official file with rc0 (§20.8). The official paths are only ever written
    # by a complete run.
    complete = len(cells) == len(LEGACY)
    out_json, out_md = Path(args.out_json), Path(args.out_md)
    if not complete:
        if not args.allow_incomplete:
            print(f"REFUSED: {len(cells)} of {len(LEGACY)} cells admissible; "
                  f"{len(unpaired)} unpaired. Not writing {out_json} or "
                  f"{out_md}. Pass --allow-incomplete to write a .partial "
                  f"report instead.")
            for entry in unpaired:
                print(f"  - {entry['cell']}: {entry['reason']}")
            return 1
        out_json = out_json.with_suffix(f".partial{out_json.suffix}")
        out_md = out_md.with_suffix(f".partial{out_md.suffix}")

    out_json.write_text(
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
        "- Ranking below is **post-bio diagnostic**. D1's selection metric is "
        "raw base-Hamming mAP@R, which these artefacts do not store, so this "
        "does not show what the train-only selection would have chosen.",
        "- Both sides are scored by the SAME committed evaluator, and the "
        "aggregation refuses any pair whose recorded analysis sources differ.",
        "- Extraction manifests were **backfilled** after the fact and carry "
        "`backfilled: true`; the cells' inputs were verified byte-identical to "
        "their canonical legacy sources, but the manifests are not evidence "
        "that the extraction recorded itself.",
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
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json} and {out_md}: "
          f"{len(cells)} paired, {len(unpaired)} unpaired")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
