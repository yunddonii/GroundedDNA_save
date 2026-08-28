"""Choose (N) per dataset from the completed 16-cell matrix (D1).

The rule is short and the enforcement is the whole point:

    argmax over the candidate's raw base-Hamming mAP@R at ITS OWN terminal
    epoch; ties go to the SMALLEST N; the chosen N is then reused unchanged for
    seeds 42, 43 and 44.

What this refuses, because each was reachable:

  * fewer than sixteen records, or more, or two for one cell -- a partial
    matrix picks N from whichever cells happened to finish;
  * records from more than one namespace or protocol -- a matrix assembled from
    two different launchers compares numbers that were not produced the same way;
  * smoke records -- they are short runs with the horizons collapsed, and the
    launcher marks them `is_candidate_cell: false`;
  * a record whose selection did not come from its own terminal epoch.

The output names every record digest it read and the protocol sources those
records were produced under, so the choice can be re-derived rather than
believed.

Usage:
    python scripts/phase3_select_n.py
    python scripts/phase3_select_n.py --records artifacts/phase3_selection
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.phase3_selection_matrix import (  # noqa: E402
    BASES_PER_SLOT,
    CANDIDATE_N,
    DATASETS,
    LR_HORIZON,
    RECORD_SCHEMA,
    SEED,
    SLOTS,
    TOTAL_BASES,
    TOTAL_BITS,
    VAL_RATIO,
    VAL_SEED,
    cell_keys,
    protocol_digests,
)

DEFAULT_RECORDS = REPO / "artifacts" / "phase3_selection"
DEFAULT_OUT = DEFAULT_RECORDS / "selected_n.json"


class SelectionRefused(RuntimeError):
    """The matrix cannot be reduced to a choice."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_record(payload: dict, path: Path) -> tuple:
    if payload.get("schema_version") != RECORD_SCHEMA:
        raise SelectionRefused(
            f"{path.name}: unknown record schema_version "
            f"{payload.get('schema_version')!r}")
    if not payload.get("is_candidate_cell"):
        raise SelectionRefused(
            f"{path.name}: marked as not a candidate cell (a smoke run has "
            f"its horizons collapsed and cannot be compared)")

    dataset, n = payload.get("dataset"), payload.get("N")
    if (dataset, n) not in cell_keys():
        raise SelectionRefused(
            f"{path.name}: ({dataset}, {n}) is not one of the "
            f"{len(cell_keys())} cells")

    expected = {
        "seed": SEED, "val_split_ratio": VAL_RATIO, "val_split_seed": VAL_SEED,
        "lr_schedule_horizon": LR_HORIZON,
        "sinkhorn_schedule_horizon": n + 1,
    }
    wrong = {k: (payload.get(k), v) for k, v in expected.items()
             if payload.get(k) != v}
    if wrong:
        raise SelectionRefused(f"{path.name}: protocol differs {wrong}")

    geometry = payload.get("geometry") or {}
    geo_wrong = {k: (geometry.get(k), v) for k, v in (
        ("num_semantic_parts", SLOTS), ("num_codebooks", SLOTS),
        ("num_codons_per_codebook", BASES_PER_SLOT)) if geometry.get(k) != v}
    if geo_wrong:
        raise SelectionRefused(f"{path.name}: geometry differs {geo_wrong}")

    selection = payload.get("selection") or {}
    if selection.get("selection_metric") != "eval_mAP_at_R":
        raise SelectionRefused(
            f"{path.name}: selection metric is "
            f"{selection.get('selection_metric')!r}, not raw base-Hamming "
            f"mAP@R")
    if selection.get("selection_epoch_zero_based") != n:
        raise SelectionRefused(
            f"{path.name}: the value comes from epoch "
            f"{selection.get('selection_epoch_zero_based')}, not the "
            f"candidate's own terminal epoch {n}")
    value = selection.get("selection_value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not 0.0 <= float(value) <= 1.0:
        raise SelectionRefused(
            f"{path.name}: selection value {value!r} is not a proportion")
    return (dataset, n), float(value)


def load_matrix(records_dir: Path) -> dict:
    """The exact sixteen, from one namespace and one protocol."""
    paths = sorted(p for p in records_dir.glob("*.json")
                   if p.name != DEFAULT_OUT.name)
    if not paths:
        raise SelectionRefused(f"no records under {records_dir}")

    cells, namespaces, protocols, digests = {}, set(), set(), {}
    problems = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            problems.append(f"{path.name}: {error}")
            continue
        try:
            key, value = _check_record(payload, path)
        except SelectionRefused as error:
            problems.append(str(error))
            continue
        if key in cells:
            problems.append(
                f"{path.name}: {key[0]}/N{key[1]} is recorded twice; a cell "
                f"cannot have two answers")
            continue
        cells[key] = {"value": value, "record": path.name,
                      "run_dir": payload.get("run_dir"),
                      "tag": payload.get("tag")}
        namespaces.add(payload.get("namespace"))
        protocols.add(json.dumps(payload.get("protocol_sources"),
                                 sort_keys=True))
        digests[path.name] = _sha(path)

    if problems:
        raise SelectionRefused(
            "records refused:\n  " + "\n  ".join(problems))
    missing = [f"{d}/N{n}" for d, n in cell_keys() if (d, n) not in cells]
    if missing:
        raise SelectionRefused(
            f"{len(cells)} of {len(cell_keys())} cells present; missing "
            f"{missing}. A partial matrix chooses N from whichever cells "
            f"happened to finish.")
    if len(namespaces) != 1:
        raise SelectionRefused(
            f"records come from {len(namespaces)} namespaces {sorted(namespaces)}; "
            f"one matrix, one namespace")
    if len(protocols) != 1:
        raise SelectionRefused(
            "records were produced under different protocol sources; the "
            "numbers were not made the same way")
    return {"cells": cells, "namespace": namespaces.pop(),
            "protocol_sources": json.loads(protocols.pop()),
            "record_sha256": digests}


def choose(cells: dict) -> dict:
    """argmax on the metric; ties to the smallest N."""
    chosen = {}
    for dataset in sorted(DATASETS):
        # Sorting by (-value, N) makes the tie-break part of the ordering
        # rather than a branch someone can forget.
        ranked = sorted(
            ((cells[(dataset, n)]["value"], n) for n in CANDIDATE_N),
            key=lambda pair: (-pair[0], pair[1]))
        best_value, best_n = ranked[0]
        tied = [n for value, n in ranked if value == best_value]
        chosen[dataset] = {
            "selected_N": best_n,
            "selection_value": best_value,
            "tied_with": sorted(tied),
            "tie_broken_by_smallest_N": len(tied) > 1,
            "all_candidates": {str(n): cells[(dataset, n)]["value"]
                               for n in CANDIDATE_N},
        }
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    try:
        matrix = load_matrix(Path(args.records))
    except SelectionRefused as error:
        print(f"[phase3-select] REFUSED: {error}", file=sys.stderr)
        return 1

    chosen = choose(matrix["cells"])
    payload = {
        "schema_version": 1,
        "what_this_is": (
            "D1's (N) choice per dataset: argmax of the raw base-Hamming mAP@R "
            "at each candidate's own terminal epoch, ties to the smallest N. "
            "The chosen N is reused unchanged for seeds 42, 43 and 44."),
        "selection_metric": "eval_mAP_at_R",
        "selection_distance": "base_hamming",
        "candidate_n": list(CANDIDATE_N),
        "seed": SEED,
        "val_split_ratio": VAL_RATIO,
        "val_split_seed": VAL_SEED,
        "geometry": {"num_slots": SLOTS, "bases_per_slot": BASES_PER_SLOT,
                     "total_bases": TOTAL_BASES, "total_bits": TOTAL_BITS},
        "namespace": matrix["namespace"],
        "protocol_sources": matrix["protocol_sources"],
        "aggregator_sha256": _sha(Path(__file__)),
        "current_protocol_sources": protocol_digests(),
        "record_sha256": matrix["record_sha256"],
        "selected": chosen,
    }

    out = Path(args.out)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, out)

    for dataset in sorted(chosen):
        entry = chosen[dataset]
        note = "  (tie -> smallest)" if entry["tie_broken_by_smallest_N"] else ""
        print(f"  {dataset:<10} N={entry['selected_N']:<3} "
              f"mAP@R={entry['selection_value']:.4f}{note}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
