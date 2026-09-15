#!/usr/bin/env python
"""Reduce one lambda confirmation campaign (`--sweep lambda`) to a decision.

Pre-declared BEFORE the campaign ran (contract
docs/LAMBDA_RESWEEP_CONTRACT_2026-09-14.md). The rule, in words:

  * The incumbent recipe was trained at three seeds. Their spread
    (max - min of the terminal-epoch train-only `eval_mAP_at_R`) is the noise
    floor, never smaller than `--min-delta` (0.002).
  * A candidate (one coefficient moved, seed 42) replaces the incumbent value
    on its axis only if its score exceeds the incumbent's SEED-42 score by more
    than that floor. Several candidates above the floor: the highest wins.
    Anything else keeps the incumbent.
  * Every number is a train-only validation score. Nothing here touches the
    official test split, and a decision here does not change the paper's
    recipe by itself: it says which value the 3-seed refit should be run at.

Every record named by the receipt is reopened and its digest checked; the
score is re-read from the run's own `log.csv` (digest-bound by the record)
rather than trusted from the record's copy. The decision file is written
exclusively (never overwritten).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DECISION_SCHEMA = 1


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DecisionRefused(RuntimeError):
    pass


def _require(cond, msg):
    if not cond:
        raise DecisionRefused(msg)


def _terminal_score(run_dir: Path, *, terminal_epoch: int,
                    expected_log_sha256: str) -> float:
    log = run_dir / "log.csv"
    _require(log.is_file(), f"{run_dir}: no log.csv")
    _require(_sha(log) == expected_log_sha256,
             f"{log}: bytes differ from the record's log_csv_sha256")
    with open(log, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    hits = [r for r in rows if int(float(r.get("epoch", -1))) == terminal_epoch
            and r.get("eval_mAP_at_R") not in (None, "")]
    _require(len(hits) == 1,
             f"{log}: {len(hits)} scored rows at terminal epoch {terminal_epoch}")
    value = float(hits[0]["eval_mAP_at_R"])
    _require(math.isfinite(value), f"{log}: terminal eval_mAP_at_R is not finite")
    return value


def reduce_campaign(receipt_path: Path, *, records_dir: Path,
                    min_delta: float) -> dict:
    from scripts.phase3_selection_matrix import (
        LAMBDA_AXES, LAMBDA_INCUMBENT, LAMBDA_INCUMBENT_SEEDS,
        LAMBDA_CAMPAIGN_KIND, SEED)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("campaign_kind") == LAMBDA_CAMPAIGN_KIND
             and receipt.get("axis") == "lambda"
             and receipt.get("fixed_point_authority") is False,
             "not a lambda confirmation receipt")
    _require(receipt.get("cell_count") == receipt.get("expected_cells")
             == receipt.get("declared_cells") == len(receipt.get("cells") or {}),
             "receipt is not a complete campaign")
    cells = []
    for key, entry in sorted(receipt["cells"].items()):
        record_path = records_dir / entry["record"]
        _require(record_path.is_file() and _sha(record_path) == entry["record_sha256"],
                 f"{record_path}: missing or not the bytes the receipt names")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        _require(record["tag"] == entry["tag"] and record["run_dir"] == entry["run_dir"],
                 f"{key}: record/receipt disagree on tag or run_dir")
        _require(record.get("smoke") is False and record.get("is_candidate_cell") is True
                 and record.get("stage") == "select", f"{key}: not a production selection cell")
        sel = record["selection"]
        _require(sel.get("selection_metric") == "eval_mAP_at_R",
                 f"{key}: selection metric is {sel.get('selection_metric')!r}")
        dataset = record["dataset"]
        overrides = dict((record.get("recipe") or {}).get("lambda_overrides") or {})
        effective = dict((record.get("recipe") or {}).get("lambda_effective") or {})
        expected = {**LAMBDA_INCUMBENT[dataset], **overrides}
        _require(set(effective) == set(LAMBDA_AXES)
                 and all(float(effective[f]) == float(expected[f]) for f in LAMBDA_AXES),
                 f"{key}: args.txt coefficients {effective} != expected {expected}")
        _require(len(overrides) <= 1, f"{key}: more than one coefficient moved")
        terminal = int(sel["selection_epoch_zero_based"])
        score = _terminal_score(Path(record["run_dir"]), terminal_epoch=terminal,
                                expected_log_sha256=record["completion"]["log_csv_sha256"])
        _require(abs(score - float(sel["selection_value"])) < 1e-12,
                 f"{key}: log.csv terminal score {score} != record {sel['selection_value']}")
        cells.append({"key": key, "dataset": dataset, "seed": int(record["seed"]),
                      "overrides": overrides, "score": score, "tag": record["tag"],
                      "record_sha256": entry["record_sha256"],
                      "identity_digest": entry["identity_digest"]})

    decisions = {}
    for dataset in sorted({c["dataset"] for c in cells}):
        inc = {c["seed"]: c for c in cells if c["dataset"] == dataset and not c["overrides"]}
        _require(sorted(inc) == sorted(LAMBDA_INCUMBENT_SEEDS),
                 f"{dataset}: incumbent seeds {sorted(inc)} != {list(LAMBDA_INCUMBENT_SEEDS)}")
        scores = [inc[s]["score"] for s in LAMBDA_INCUMBENT_SEEDS]
        floor = max(float(min_delta), max(scores) - min(scores))
        base = inc[SEED]["score"]
        axes = {}
        for flag, spec in LAMBDA_AXES.items():
            cands = [c for c in cells if c["dataset"] == dataset
                     and list(c["overrides"]) == [flag]]
            _require(all(c["seed"] == SEED for c in cands),
                     f"{dataset}/{flag}: a candidate is not at seed {SEED}")
            grid_seen = {c["overrides"][flag] for c in cands} | {LAMBDA_INCUMBENT[dataset][flag]}
            _require(grid_seen == set(spec["grid"]),
                     f"{dataset}/{flag}: grid {sorted(grid_seen)} != declared {spec['grid']}")
            above = [c for c in cands if c["score"] - base > floor]
            winner = max(above, key=lambda c: c["score"]) if above else None
            axes[flag] = {
                "incumbent_value": LAMBDA_INCUMBENT[dataset][flag],
                "incumbent_seed42_score": base,
                "candidates": {c["overrides"][flag]: {"score": c["score"],
                                                      "delta_vs_incumbent_seed42": c["score"] - base,
                                                      "above_floor": c["score"] - base > floor,
                                                      "tag": c["tag"]}
                               for c in sorted(cands, key=lambda c: float(c["overrides"][flag]))},
                "selected_value": winner["overrides"][flag] if winner else LAMBDA_INCUMBENT[dataset][flag],
                "changed": winner is not None,
            }
        decisions[dataset] = {
            "incumbent_scores_by_seed": {str(s): inc[s]["score"] for s in LAMBDA_INCUMBENT_SEEDS},
            "incumbent_spread": max(scores) - min(scores),
            "noise_floor": floor,
            "axes": axes,
            "any_change": any(a["changed"] for a in axes.values()),
        }
    return {
        "schema_version": DECISION_SCHEMA,
        "artifact_kind": "phase3_lambda_decision",
        "rule": ("candidate replaces the incumbent value on its axis iff "
                 "score(candidate, seed 42) - score(incumbent, seed 42) > "
                 "max(min_delta, max-min over incumbent seeds); highest wins; "
                 "otherwise the incumbent stays"),
        "min_delta": float(min_delta),
        "metric": "eval_mAP_at_R at the cell's own terminal epoch, train-only "
                  "validation split (VAL_RATIO 0.1, VAL_SEED 42); official test never read",
        "reducer_sha256": _sha(Path(__file__)),
        "receipt": {"path": str(receipt_path), "sha256": _sha(receipt_path),
                    "namespace": receipt.get("namespace"),
                    "campaign_nonce": receipt.get("campaign_nonce")},
        "cells": cells,
        "decisions": decisions,
        "paper_result_eligible": False,
        "official_test_used": False,
        "next_step_if_changed": "3-seed refit at the selected value(s) through the "
                                "refit path; nothing else is re-run until then",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--records-dir", default=str(REPO / "artifacts" / "phase3_selection"))
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--out", default=None, help="decision JSON; written exclusively")
    a = ap.parse_args(argv)
    try:
        out = reduce_campaign(Path(a.receipt).resolve(), records_dir=Path(a.records_dir).resolve(),
                              min_delta=a.min_delta)
    except DecisionRefused as e:
        print(f"[lambda-decision] REFUSED: {e}", file=sys.stderr)
        return 2
    for dataset, d in out["decisions"].items():
        print(f"{dataset}: incumbent by seed {d['incumbent_scores_by_seed']} spread {d['incumbent_spread']:.4f} "
              f"floor {d['noise_floor']:.4f}")
        for flag, ax in d["axes"].items():
            row = ", ".join(f"{v}: {c['score']:.4f} ({c['delta_vs_incumbent_seed42']:+.4f})"
                            for v, c in ax["candidates"].items())
            print(f"  {flag:26s} incumbent {ax['incumbent_value']} = {ax['incumbent_seed42_score']:.4f} | "
                  f"{row} -> {'CHANGE to ' + ax['selected_value'] if ax['changed'] else 'keep'}")
    if a.out:
        fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print(f"[lambda-decision] wrote {a.out} sha256 {_sha(Path(a.out))[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
