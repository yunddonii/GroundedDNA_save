"""Publish a cell's `analysis_complete.json` once BOTH halves are in (§20.4).

A cell's analysis is two computations with two different sources:

    scripts/eval_cell_bioproj.py  ->  cell_result.json   (retrieval, DNA unique)
    scripts/pairwise_nmi.py       ->  pairwise_nmi.json  (codebook redundancy)

The bio-evaluation used to seal `analysis_complete.json` itself, immediately
after its own half. The sealed metric set therefore had no `mean_off_diag_nmi`
at all, the NMI stage never re-sealed anything, and the aggregator read the
missing key with `.get()` and still counted the cell as paired -- so a report of
"15 paired" with a blank NMI column throughout was reachable. A file named
`analysis_complete` must be written by something that can see the whole
analysis, which is what this is.

It refuses unless, for the cell it is given:

  * both partial files exist, parse, and each declares an input binding;
  * both bindings are identical to the extraction's binding right now, so the
    two halves describe the same forward pass and not two different ones;
  * the metric set is exactly the five the contract requires, each a finite
    proportion;
  * the protocol block is complete, including the NMI averaging method and the
    sklearn version that produced it, copied from the stage that used them;
  * every analysis source file still hashes to what it hashed during the run.

Usage:
    python scripts/seal_cell_analysis.py --dir <cell> [--allow-backfilled]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.extraction_validation import (  # noqa: E402
    ANALYSIS_MARKER_NAME,
    ExtractionInvalid,
    analysis_source_digests,
    check_metric_input_binding,
    metric_input_binding,
    write_analysis_marker,
)


class SealRefused(RuntimeError):
    """The two halves do not together make one admissible analysis."""


def _read(cell: Path, name: str) -> dict:
    path = cell / name
    if not path.is_file():
        raise SealRefused(
            f"{cell.name}: {name} is missing, so the analysis is not complete")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SealRefused(f"{cell.name}/{name}: {error}") from None
    if not isinstance(payload, dict):
        raise SealRefused(f"{cell.name}/{name} is not a JSON object")
    return payload


def _number(payload: dict, key: str, *, where: str) -> float:
    value = payload.get(key)
    if value is None:
        raise SealRefused(f"{where} has no {key}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SealRefused(f"{where}.{key} is {value!r}, not a number")
    return float(value)


def seal(cell: Path, *, allow_backfilled: bool) -> str:
    evaluation = _read(cell, "cell_result.json")
    nmi = _read(cell, "pairwise_nmi.json")

    # Both halves must name the SAME inputs, and those inputs must be this
    # cell's, checked against the extraction as it stands now rather than
    # against each other alone.
    for name, payload in (("cell_result.json", evaluation),
                          ("pairwise_nmi.json", nmi)):
        try:
            check_metric_input_binding(
                str(cell), payload, what=f"{cell.name}/{name}",
                allow_backfilled=allow_backfilled)
        except ExtractionInvalid as error:
            raise SealRefused(str(error)) from None
    if evaluation["input_binding"] != nmi["input_binding"]:
        raise SealRefused(
            f"{cell.name}: the two analysis halves declare different inputs; "
            f"they describe different forward passes")

    binding = metric_input_binding(str(cell), allow_backfilled=allow_backfilled)

    metrics = {
        "map_at_R_bioproj": _number(evaluation, "mAP_at_R_bioproj",
                                    where=f"{cell.name}/cell_result.json"),
        "full_map_bioproj": _number(evaluation, "full_mAP_bioproj",
                                    where=f"{cell.name}/cell_result.json"),
        "full_map_pre_projection": _number(
            evaluation, "full_mAP_pre_projection",
            where=f"{cell.name}/cell_result.json"),
        "dna_unique_db": _number(evaluation, "DNA_unique_DB",
                                 where=f"{cell.name}/cell_result.json"),
        "mean_off_diag_nmi": _number(nmi, "mean_off_diag_nmi",
                                     where=f"{cell.name}/pairwise_nmi.json"),
    }

    method = nmi.get("nmi_average_method")
    version = nmi.get("sklearn_version")
    if not method or not version:
        raise SealRefused(
            f"{cell.name}/pairwise_nmi.json predates the recorded averaging "
            f"method and sklearn version; re-run the NMI so the number can be "
            f"interpreted")

    protocol = {
        "dataset": evaluation.get("dataset"),
        "codebook_size": evaluation.get("K"),
        "total_bases": evaluation.get("total_bases"),
        "map_r_cutoff": evaluation.get("map_r_cutoff"),
        "gc_policy_version": evaluation.get("gc_policy_version"),
        "gc_count_min_inclusive": evaluation.get("gc_count_min_inclusive"),
        "gc_count_max_inclusive": evaluation.get("gc_count_max_inclusive"),
        "gc_min_frac": evaluation.get("gc_min_frac"),
        "gc_max_frac": evaluation.get("gc_max_frac"),
        "nmi_average_method": method,
        "sklearn_version": version,
    }
    unset = sorted(k for k, v in protocol.items() if v is None)
    if unset:
        raise SealRefused(
            f"{cell.name}: the analysis protocol is incomplete, no "
            f"{', '.join(unset)}")

    try:
        return write_analysis_marker(
            str(cell), metrics=metrics, binding=binding, protocol=protocol,
            sources=analysis_source_digests())
    except ExtractionInvalid as error:
        raise SealRefused(str(error)) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, nargs="+",
                        help="cell directories to seal")
    parser.add_argument("--allow-backfilled", action="store_true",
                        help=("Admit retrospectively bound inputs. Off by "
                              "default so a paper path cannot pick them up."))
    args = parser.parse_args()

    refused = []
    for raw in args.dir:
        cell = Path(raw)
        # A stale seal must not survive a refusal: it would keep describing
        # numbers that this run has just shown to be unpublishable.
        marker = cell / ANALYSIS_MARKER_NAME
        try:
            path = seal(cell, allow_backfilled=args.allow_backfilled)
        except SealRefused as error:
            if marker.exists():
                marker.unlink()
            refused.append(str(error))
            print(f"[seal] REFUSED {cell.name}: {error}")
            continue
        print(f"[seal] {cell.name} -> {path}")

    if refused:
        print(f"\n{len(refused)} of {len(args.dir)} cell(s) refused")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
