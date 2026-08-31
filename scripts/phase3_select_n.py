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

from dna_utils.run_identity import load_run_manifest  # noqa: E402
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


def _reverify_run(payload: dict, path: Path) -> None:
    """Open the run the record describes and re-derive its claims.

    The record is a report, and every field in it was written by the same
    process that is now being trusted. A safe probe built sixteen records whose
    `run_dir` did not exist and whose completion digests were repeated
    characters, and they were accepted -- the checks were shape and truthiness.
    So the directory is opened, the manifest is re-read, the checkpoint and
    log.csv are re-hashed, and the selected value is re-read from the CSV.
    """
    run_dir = Path(payload["run_dir"])
    if not run_dir.is_dir():
        raise SelectionRefused(
            f"{path.name}: run_dir {run_dir} does not exist, so nothing in "
            f"this record can be re-derived")

    identity = load_run_manifest(str(run_dir))
    if identity is None:
        raise SelectionRefused(
            f"{path.name}: {run_dir} has no readable run manifest under the "
            f"current identity schema")
    if identity.digest != payload.get("identity_digest"):
        raise SelectionRefused(
            f"{path.name}: the run's identity is {identity.digest[:12]}..., "
            f"the record says {str(payload.get('identity_digest'))[:12]}...")

    completion = payload["completion"]
    checkpoint = run_dir / str(completion.get("final_checkpoint") or
                               "model_state_dict.pth")
    if not checkpoint.is_file():
        raise SelectionRefused(
            f"{path.name}: {checkpoint.name} is not in {run_dir}")
    if _sha(checkpoint) != completion.get("final_checkpoint_sha256"):
        raise SelectionRefused(
            f"{path.name}: {checkpoint.name} does not hash to the digest the "
            f"record claims; the weights changed after the cell finished")

    csv_path = run_dir / "log.csv"
    if not csv_path.is_file():
        raise SelectionRefused(f"{path.name}: {run_dir} has no log.csv")
    selection = payload["selection"]
    if _sha(csv_path) != selection.get("log_csv_sha256"):
        raise SelectionRefused(
            f"{path.name}: log.csv does not hash to the digest the record "
            f"claims; the metric was read from different bytes")

    # And the number itself, re-read from those bytes rather than copied.
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = {int(r["epoch"]): r for r in _csv.DictReader(handle)
                if (r.get("epoch") or "").strip().isdigit()}
    epoch = selection["selection_epoch_zero_based"]
    if epoch not in rows:
        raise SelectionRefused(
            f"{path.name}: log.csv has no row for epoch {epoch}")
    raw = (rows[epoch].get("eval_mAP_at_R") or "").strip()
    try:
        actual = float(raw)
    except ValueError:
        raise SelectionRefused(
            f"{path.name}: log.csv epoch {epoch} eval_mAP_at_R={raw!r} is not "
            f"a number") from None
    if abs(actual - float(selection["selection_value"])) > 1e-12:
        raise SelectionRefused(
            f"{path.name}: the record says {selection['selection_value']}, "
            f"log.csv epoch {epoch} says {actual}")


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

    # A record must carry the evidence a cell produces, not merely the shape a
    # reader looks for. Sixteen hand-written JSONs with none of this -- and
    # `protocol_sources: null` throughout -- passed the first version.
    required = ("completion", "inputs", "matrix", "stage", "epoch_budget",
                "run_dir", "tag", "identity_digest", "protocol_sources")
    absent = [k for k in required if payload.get(k) in (None, "", {}, [])]
    if absent:
        raise SelectionRefused(
            f"{path.name}: no {', '.join(absent)} -- this is not a record a "
            f"completed cell wrote")
    if payload.get("stage") != "select":
        raise SelectionRefused(
            f"{path.name}: stage is {payload.get('stage')!r}, not a selection "
            f"cell")
    completion = payload["completion"]
    for field in ("final_checkpoint_sha256", "log_csv_sha256"):
        digest = completion.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise SelectionRefused(
                f"{path.name}: completion.{field} is {digest!r}, not a digest")
    if not (payload.get("selection") or {}).get("map_r_cutoff"):
        raise SelectionRefused(
            f"{path.name}: the metric records no mAP@R cutoff")

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
    _reverify_run(payload, path)
    return (dataset, n), float(value)


def load_matrix(records_dir: Path) -> dict:
    """The exact sixteen, from one namespace and one protocol."""
    paths = sorted(p for p in records_dir.glob("*.json")
                   if p.name != DEFAULT_OUT.name)
    if not paths:
        raise SelectionRefused(f"no records under {records_dir}")

    cells, namespaces, protocols, digests = {}, set(), set(), {}
    trainers = {}
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
        sources = payload.get("protocol_sources") or {}
        current_keys = set(protocol_digests())
        # The trainer key is per-dataset, so it is compared on its own; the
        # rest has to be identical across every record in the matrix.
        trainers[path.name] = {k: v for k, v in sources.items()
                               if k not in current_keys}
        protocols.add(json.dumps({k: v for k, v in sources.items()
                                  if k in current_keys}, sort_keys=True))
        digests[path.name] = _sha(path)

    if problems:
        raise SelectionRefused(
            "records refused:\n  " + "\n  ".join(problems))
    protocols = {p for p in protocols}
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
    # One protocol is not enough -- sixteen records agreeing on `null` are also
    # "one". It has to be THIS protocol.
    #
    # Records now also name the TRAINER SHELL they ran, and that differs by
    # dataset, so the shared part and the per-dataset part are compared
    # separately: every record must agree on the common sources, and each
    # record's own trainer must match the tree reading it.
    stored = json.loads(next(iter(protocols))) if protocols else None
    current = protocol_digests()
    shared = dict(stored or {})
    if shared != current:
        differing = sorted(
            k for k in set(current) | set(shared)
            if shared.get(k) != current.get(k))
        raise SelectionRefused(
            f"records were produced under a different protocol than the one "
            f"reading them: {differing}. Re-run the affected cells or read "
            f"them from the commit that produced them.")
    trainer_bad = sorted(
        f"{rel} in {name}" for name, rel_map in trainers.items()
        for rel, digest in rel_map.items() if _sha(REPO / rel) != digest)
    if trainer_bad:
        raise SelectionRefused(
            f"the trainer shell a cell ran is not the one in this tree: "
            f"{trainer_bad}")
    return {"cells": cells, "namespace": namespaces.pop(),
            "protocol_sources": json.loads(protocols.pop()),
            "record_sha256": digests}


#: The recipe reduction, declared BEFORE the cells run so the rule cannot be
#: picked after the numbers are seen.
#:
#:   metric      train-only validation raw base-Hamming mAP@R at the cell's own
#:               terminal epoch -- the same number D1 uses for N. Codon decoding
#:               is NOT in it: the held-out decoder reads the official query
#:               split, so using it would make the choice test-informed.
#:   winner      argmax of that metric within a dataset
#:   tie         the INCUMBENT coordinate first, then the smaller one. Ties go
#:               to not-moving, so a coin-flip cannot rewrite the recipe.
#:   fail        any missing, duplicated or non-finite cell refuses the WHOLE
#:               dataset. A recipe chosen from whichever cells finished is the
#:               partial-matrix defect under another name.
RECIPE_REDUCTION = {
    "metric": "eval_mAP_at_R",
    "distance": "base_hamming",
    "split": "train_only_validation",
    "tie_break": "incumbent_first_then_smaller",
    "fail_rule": "any_missing_or_nonfinite_refuses_the_dataset",
}


def choose_recipe(cells: dict, *, axis: str, grid: list, incumbent) -> dict:
    """Pick one coordinate per dataset on `RECIPE_REDUCTION`.

    `cells` maps (dataset, coordinate) -> value. `grid` is the exact set of
    coordinates every dataset must have run, so a dataset that is one cell
    short is refused rather than reduced.
    """
    datasets = sorted({d for d, _ in cells})
    if not datasets:
        raise SelectionRefused("no cells to reduce")
    chosen = {}
    for dataset in datasets:
        missing = [c for c in grid if (dataset, c) not in cells]
        if missing:
            raise SelectionRefused(
                f"{dataset}: {len(missing)} of {len(grid)} {axis} cells are "
                f"absent {missing}. A recipe chosen from whichever cells "
                f"finished is a partial matrix.")
        extra = sorted(c for d, c in cells if d == dataset and c not in grid)
        if extra:
            raise SelectionRefused(
                f"{dataset}: {extra} are not in the declared {axis} grid "
                f"{grid}")
        values = {}
        for c in grid:
            v = cells[(dataset, c)]
            if not isinstance(v, (int, float)) or isinstance(v, bool) \
                    or v != v or not 0.0 <= float(v) <= 1.0:
                raise SelectionRefused(
                    f"{dataset}/{c}: value {v!r} is not a finite proportion")
            values[c] = float(v)
        best = max(values.values())
        tied = [c for c in grid if values[c] == best]
        # Incumbent first, then the smaller coordinate. Both are positional in
        # `grid`, so the rule is part of the ordering rather than a branch.
        if incumbent in tied:
            winner = incumbent
        else:
            winner = sorted(tied, key=lambda c: grid.index(c))[0]
        chosen[dataset] = {
            "selected": list(winner) if isinstance(winner, tuple) else winner,
            "selection_value": values[winner],
            "tied_with": [list(c) if isinstance(c, tuple) else c for c in tied],
            "tie_broken": len(tied) > 1,
            "all_candidates": {
                str(c): values[c] for c in grid},
            # The incumbent is only ON the grid for the top-p sweep. The
            # lambda grid is the five positive values, and its zero control
            # comes from the top-p stage, so `values[incumbent]` raised
            # KeyError('0.0') and the whole lambda reduction was unrunnable.
            "delta_vs_incumbent": (
                round(values[winner] - values[incumbent], 6)
                if incumbent in values else None),
            "incumbent_on_grid": incumbent in values,
        }
    return chosen


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


def _recipe_matrix(records_dir: Path, *, axis: str,
                   namespace: str | None = None) -> dict:
    """The same per-record authentication, keyed by the swept coordinate."""
    from scripts.phase3_selection_matrix import (
        JOINT_GRID, JOINT_INCUMBENT, TOPP_GRID, TOPP_INCUMBENT)

    grid = ([tuple(p) for p in TOPP_GRID] if axis == "topp"
            else list(JOINT_GRID))
    incumbent = (tuple(TOPP_INCUMBENT) if axis == "topp" else JOINT_INCUMBENT)

    cells, digests, problems, namespaces, snapshots = {}, {}, [], set(), set()
    for path in sorted(records_dir.glob("*.json")):
        if "_snapshot" in path.name or "_sweep_complete" in path.name \
                or path.name == DEFAULT_OUT.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("is_candidate_cell"):
            continue
        # A recipe sweep and the N matrix share one directory, so the sweep
        # must be selected by its NAMESPACE rather than by "everything that
        # looks like a candidate". Reading all of them made the sixteen
        # N-selection records -- written under an older identity schema -- into
        # sixteen fatal problems, so the reducer could not run at all.
        if namespace is not None and payload.get("namespace") != namespace:
            continue
        recipe = payload.get("recipe") or {}
        coord = ((str(recipe.get("routing_adaptive_topp_min")),
                  str(recipe.get("routing_adaptive_topp_max")))
                 if axis == "topp" else str(recipe.get("lambda_codon_joint")))
        try:
            # Everything except the (dataset, N) membership test, which is
            # about the N matrix rather than a recipe sweep.
            _reverify_run(payload, path)
        except SelectionRefused as error:
            problems.append(str(error))
            continue
        key = (payload["dataset"], coord)
        if key in cells:
            problems.append(f"{path.name}: {key} is recorded twice")
            continue
        cells[key] = float(payload["selection"]["selection_value"])
        digests[path.name] = _sha(path)
        namespaces.add(payload.get("namespace"))
        snapshots.add(payload.get("plan_snapshot_sha256"))
    if problems:
        raise SelectionRefused("records refused:\n  " + "\n  ".join(problems))
    if not cells:
        raise SelectionRefused(
            f"no candidate {axis} records in {records_dir}"
            + (f" for namespace {namespace!r}" if namespace else ""))
    if len(namespaces) > 1:
        raise SelectionRefused(
            f"records come from {len(namespaces)} namespaces "
            f"{sorted(namespaces)}; one sweep, one namespace")
    # Every cell must have run against the SAME plan. Nine `--only` processes
    # each took their own snapshot, so nine cells carried up to nine different
    # digests and "these came from one plan" was not a statement anyone could
    # make about them.
    if len(snapshots) != 1 or None in snapshots:
        raise SelectionRefused(
            f"the cells carry {len(snapshots)} different plan snapshots "
            f"{sorted(str(x)[:12] for x in snapshots)}; a sweep reduced across "
            f"several plans is not one experiment")
    return {"cells": cells, "grid": grid, "incumbent": incumbent,
            "record_sha256": digests,
            "plan_snapshot_sha256": snapshots.pop(),
            "namespace": namespaces.pop() if namespaces else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--namespace", default=None,
                        help=("which sweep to reduce; the N matrix and a recipe "
                              "sweep share one records directory"))
    parser.add_argument("--axis", choices=("n", "topp", "joint"), default="n",
                        help="which coordinate this reduction chooses")
    args = parser.parse_args()

    if args.axis != "n":
        try:
            matrix = _recipe_matrix(Path(args.records), axis=args.axis,
                                    namespace=args.namespace)
            chosen = choose_recipe(matrix["cells"], axis=args.axis,
                                   grid=matrix["grid"],
                                   incumbent=matrix["incumbent"])
        except SelectionRefused as error:
            print(f"[phase3-select] REFUSED: {error}", file=sys.stderr)
            return 1
        payload = {
            "schema_version": 1,
            "axis": args.axis,
            "reduction": RECIPE_REDUCTION,
            "grid": [list(c) if isinstance(c, tuple) else c
                     for c in matrix["grid"]],
            "incumbent": (list(matrix["incumbent"])
                          if isinstance(matrix["incumbent"], tuple)
                          else matrix["incumbent"]),
            "namespace": matrix["namespace"],
            "plan_snapshot_sha256": matrix["plan_snapshot_sha256"],
            "aggregator_sha256": _sha(Path(__file__)),
            "protocol_sources": protocol_digests(),
            "record_sha256": matrix["record_sha256"],
            "selected": chosen,
            # What the next stage must hold fixed, and what must be re-checked
            # once N is chosen: the two axes interact, so a winner here is
            # provisional until the confirmation pass agrees.
            "stability": {
                "confirmed": False,
                "note": ("provisional: re-check this axis at the finally "
                         "chosen N and lambda; if it moves, N is chosen again"),
            },
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, out)
        for dataset in sorted(chosen):
            e = chosen[dataset]
            note = "  (tie -> incumbent)" if e["tie_broken"] else ""
            print(f"  {dataset:<10} {args.axis}={e['selected']} "
                  f"mAP@R={e['selection_value']:.4f} "
                  f"delta={e['delta_vs_incumbent']:+.4f}{note}")
        print(f"wrote {out}")
        return 0

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
