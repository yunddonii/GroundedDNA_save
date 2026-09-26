#!/usr/bin/env python
"""Anchor confirmation v1 reducer (docs/ANCHOR_CONFIRMATION_CONTRACT_v1.md sections 7, 8, 13).

select  -- evidence for every (dataset, arm, N in the grid, seed 42) -> each arm's N per dataset:
           argmax of the raw base-Hamming mAP@R at the cell's own terminal epoch, ties to the
           smallest N. Writes the frozen N record that `--anchor-confirm decide` consumes.
decide  -- the frozen N record + evidence for every (dataset, arm, frozen N, seed 42/43/44) +
           one code-to-axis probe per checkpoint -> the train-only rule of section 7.3 per
           dataset. Writes the frozen decision record.

Evidence is named in a sources file (pinned by digest on the command line): per coordinate, a
run record and, for `decide`, a probe output. A record counts for its coordinate only if
  * it is a stage-1, train-only, non-smoke candidate cell whose terminal epoch is N,
  * its terminal score is parsed from the log.csv bytes the record pins, finite and in [0, 1],
    and equal to the record's own selection value,
  * the typed configuration its run saved (config.pt) equals the recipe this source generation
    renders for that coordinate, outside the tag and the sealed-input fields
    (scripts.phase3_selection_matrix.INPUT_AUTHORITY_DESTS).
The last condition is what lets a historical approved control record and a new record be judged
by one rule; admitting such reuse is still the audit's decision (contract section 15, item 1).

Every file is read once and parsed from the bytes that were hashed (the pattern accepted in audit
section 500), re-verified before the output is written, and the output is written once (O_EXCL).
Membership is recomputed from the contract, never taken from a count inside an input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.phase3_selection_matrix as M  # noqa: E402

REDUCER_VERSION = "anchor-confirm-reducer/1"
N_SELECTION_KIND = "anchor_confirmation_n_selection"
DECISION_KIND = "anchor_confirmation_decision"
PROBE_KIND = "anchor_confirmation_code_axis"


class NotReducible(RuntimeError):
    """The evidence cannot support the requested output."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise NotReducible(message)


class Consumed:
    """Every file read, at the digest of the bytes that were read (audit section 500 pattern)."""

    def __init__(self) -> None:
        self.digests: dict = {}

    def read(self, path, want: str | None = None) -> bytes:
        name = str(Path(path))
        try:
            data = Path(path).read_bytes()
        except OSError as error:
            raise NotReducible(f"{name} is unreadable: {error}") from None
        digest = hashlib.sha256(data).hexdigest()
        need(self.digests.get(name, digest) == digest, f"{name} changed between two reads")
        if want is not None:
            need(digest == want, f"{name} is not the pinned bytes {str(want)[:12]}...")
        self.digests[name] = digest
        return data

    def json(self, path, want: str | None = None):
        return json.loads(self.read(path, want).decode("utf-8"))

    def reverify(self) -> list:
        return [name for name, want in sorted(self.digests.items())
                if not Path(name).is_file()
                or hashlib.sha256(Path(name).read_bytes()).hexdigest() != want]


def proportion(value, what: str) -> float:
    """A finite real number in [0, 1]; bool, NaN and +-Inf are refused explicitly."""
    need(not isinstance(value, bool) and isinstance(value, (int, float)), f"{what} is {value!r}")
    number = float(value)
    need(math.isfinite(number), f"{what} is {number!r}, not finite")
    need(0.0 <= number <= 1.0, f"{what} is {number!r}, outside [0, 1]")
    return number


def expected_coordinates(stage: str, *, arms, frozen=None) -> list:
    """The exact membership the contract declares: (dataset, arm, N, seed)."""
    need(set(arms) <= set(M.ANCHOR_ARMS) and len(set(arms)) == len(arms) and arms,
         f"arms {arms} are not a subset of {M.ANCHOR_ARMS}")
    if stage == "select":
        return [(ds, arm, n, M.SEED) for ds in M.ANCHOR_DATASETS for arm in arms
                for n in M.CANDIDATE_N]
    return [(ds, arm, frozen[ds][arm], seed) for ds in M.ANCHOR_DATASETS for arm in arms
            for seed in (M.SEED, *M.ANCHOR_DECIDE_SEEDS)]


def rendered_fields(coordinate, incumbent) -> dict:
    ds, arm, n, seed = coordinate
    return M.anchor_scientific_recipe(
        ds, n, namespace="ancReducerCheck", stage="select", seed=seed,
        topp=incumbent[ds]["topp"], joint=incumbent[ds]["joint"], anchor_arm=arm)["fields"]


def verify_record(consumed: Consumed, coordinate, entry: dict, incumbent: dict) -> dict:
    """One coordinate's run record, checked as the module docstring states."""
    import torch
    from config import Config
    from dna_utils.scientific_recipe import field_differences, fields_from_config
    ds, arm, n, seed = coordinate
    record = consumed.json(entry["record"], entry["record_sha256"])
    need(record.get("dataset") == ds and int(record.get("N", -1)) == n
         and int(record.get("seed", -1)) == seed, f"{entry['record']}: not the record of {coordinate}")
    need(record.get("stage") == "select" and record.get("selection_mode") == "select",
         f"{entry['record']}: not a stage-1 selection record (refit/test records are refused)")
    need(record.get("smoke") is False and record.get("is_candidate_cell") is True,
         f"{entry['record']}: a smoke or non-candidate record is not evidence")
    need(float(record.get("val_split_ratio", -1)) == M.VAL_RATIO,
         f"{entry['record']}: not the 90/10 train-only split")
    completion, selection = record["completion"], record["selection"]
    need(int(completion["final_checkpoint_epoch_zero_based"]) == n
         and int(selection["selection_epoch_zero_based"]) == n,
         f"{entry['record']}: terminal checkpoint/selection epoch is not N={n}")
    need(selection.get("selection_metric") == "eval_mAP_at_R"
         and selection.get("distance_mode") == "base",
         f"{entry['record']}: not the raw base-Hamming mAP@R")
    run_dir = Path(record["run_dir"])
    rows = list(csv.DictReader(io.StringIO(consumed.read(
        run_dir / "log.csv", completion["log_csv_sha256"]).decode("utf-8"))))
    need(bool(rows) and rows[-1].get("epoch", "").strip().isdigit()
         and int(rows[-1]["epoch"]) == n, f"{run_dir}: the terminal log row is not epoch {n}")
    try:
        raw_score = float(rows[-1]["eval_mAP_at_R"])
    except (KeyError, ValueError):
        raise NotReducible(f"{run_dir}: the terminal row has no mAP@R") from None
    score = proportion(raw_score, f"{run_dir} terminal mAP@R")
    need(score == proportion(selection["selection_value"], "record selection value"),
         f"{entry['record']}: the record's selection value is not the pinned log's")
    config_raw = consumed.read(run_dir / "config.pt")
    anchor = record.get("anchor_confirmation")
    if anchor is not None:
        need(anchor.get("config_pt_sha256") == hashlib.sha256(config_raw).hexdigest()
             and anchor.get("arm") == arm, f"{entry['record']}: anchor evidence disagrees")
    saved = fields_from_config(torch.load(io.BytesIO(config_raw), map_location="cpu",
                                          weights_only=False), Config.build_parser(),
                               historical=anchor is None)
    differing = [k for k in field_differences(saved, rendered_fields(coordinate, incumbent))
                 if k not in M.INPUT_AUTHORITY_DESTS]
    need(not differing, f"{entry['record']}: its saved configuration is not the recipe of "
                        f"{coordinate}: {differing[:8]}")
    return {"record": str(entry["record"]), "record_sha256": entry["record_sha256"],
            "checkpoint_sha256": completion["final_checkpoint_sha256"],
            "config_pt_sha256": hashlib.sha256(config_raw).hexdigest(),
            "score": score, "historical": anchor is None}


def verify_probe(consumed: Consumed, coordinate, entry: dict, verified: dict) -> float:
    probe = consumed.json(entry["probe"], entry["probe_sha256"])
    need(probe.get("artifact_kind") == PROBE_KIND, f"{entry['probe']}: not a code-to-axis probe")
    need(probe.get("checkpoint_sha256") == verified["checkpoint_sha256"]
         and probe.get("config_pt_sha256") == verified["config_pt_sha256"],
         f"{entry['probe']}: measured another checkpoint or configuration than {coordinate}")
    need(int(probe.get("n_images", 0)) > 0, f"{entry['probe']}: no images")
    return proportion(probe.get("code_picks_own_axis"), f"{entry['probe']} code->axis")


def load_sources(consumed: Consumed, path, want: str, coordinates, *, with_probe: bool) -> dict:
    sources = consumed.json(path, want)
    need(sources.get("version") == M.ANCHOR_CONFIRM_VERSION, f"{path}: wrong sources version")
    entries = sources.get("coordinates")
    need(isinstance(entries, list), f"{path}: no coordinate list")
    keyed = {}
    for entry in entries:
        key = (entry["dataset"], entry["arm"], int(entry["N"]), int(entry["seed"]))
        need(key not in keyed, f"duplicate evidence for {key}")
        keyed[key] = entry
    need(set(keyed) == set(coordinates),
         f"evidence membership differs from the contract: missing "
         f"{sorted(set(coordinates) - set(keyed))[:4]}, extra {sorted(set(keyed) - set(coordinates))[:4]}")
    for key in ("record_sha256",) + (("probe_sha256",) if with_probe else ()):
        digests = [entry[key] for entry in entries]
        need(len(set(digests)) == len(digests), f"one {key.split('_')[0]} is reused across coordinates")
    return keyed


def select_n(scores: dict) -> int:
    """argmax of the score, ties to the smallest N."""
    best = max(scores.values())
    return min(n for n, value in scores.items() if value == best)


def reduce_select(args) -> dict:
    consumed = Consumed()
    incumbent = M.anchor_incumbent()
    arms = tuple(a for a in args.arms.split(",") if a)
    coords = expected_coordinates("select", arms=arms)
    keyed = load_sources(consumed, args.sources, args.sources_sha256, coords, with_probe=False)
    verified = {c: verify_record(consumed, c, keyed[c], incumbent) for c in coords}
    n_selected, scores = {}, {}
    for ds in M.ANCHOR_DATASETS:
        for arm in arms:
            per_n = {c[2]: verified[c]["score"] for c in coords if c[0] == ds and c[1] == arm}
            n_selected.setdefault(ds, {})[arm] = select_n(per_n)
            scores.setdefault(ds, {})[arm] = {str(n): v for n, v in sorted(per_n.items())}
    return {"artifact_kind": N_SELECTION_KIND, "version": M.ANCHOR_CONFIRM_VERSION,
            "reducer": REDUCER_VERSION, "rule": "argmax raw base-Hamming mAP@R at own terminal "
            "epoch, ties to the smallest N, seed 42", "arms": list(arms),
            "n_selected": n_selected, "scores": scores,
            "evidence": {"|".join(map(str, c)): v for c, v in verified.items()},
            "_consumed": consumed}


def reduce_decide(args) -> dict:
    consumed = Consumed()
    incumbent = M.anchor_incumbent()
    frozen = consumed.json(args.selection, args.selection_sha256)
    need(frozen.get("artifact_kind") == N_SELECTION_KIND
         and frozen.get("version") == M.ANCHOR_CONFIRM_VERSION, "not a frozen N record")
    arms = M.ANCHOR_ARMS
    n_frozen = {ds: {arm: int(frozen["n_selected"][ds][arm]) for arm in arms}
                for ds in M.ANCHOR_DATASETS}
    coords = expected_coordinates("decide", arms=arms, frozen=n_frozen)
    keyed = load_sources(consumed, args.sources, args.sources_sha256, coords, with_probe=True)
    retrieval, code_axis = {}, {}
    for c in coords:
        verified = verify_record(consumed, c, keyed[c], incumbent)
        retrieval[c] = verified["score"]
        code_axis[c] = verify_probe(consumed, c, keyed[c], verified)
    decisions = {}
    for ds in M.ANCHOR_DATASETS:
        val = {arm: [retrieval[(ds, arm, n_frozen[ds][arm], s)]
                     for s in (M.SEED, *M.ANCHOR_DECIDE_SEEDS)] for arm in arms}
        axis = {arm: [code_axis[(ds, arm, n_frozen[ds][arm], s)]
                      for s in (M.SEED, *M.ANCHOR_DECIDE_SEEDS)] for arm in arms}
        mean = {arm: statistics.fmean(val[arm]) for arm in arms}
        sd_control = statistics.stdev(val["none"])
        axis_mean = {arm: statistics.fmean(axis[arm]) for arm in arms}
        retrieval_ok = mean["anchors"] >= mean["none"] - sd_control       # equality passes
        axis_ok = axis_mean["anchors"] > axis_mean["none"]                # equality fails
        adopt = retrieval_ok and axis_ok
        decisions[ds] = {
            "adopted_axis_center": "anchors" if adopt else "none",
            "frozen_N": n_frozen[ds]["anchors" if adopt else "none"],
            "retrieval": {"per_seed": val, "mean": mean, "control_sample_sd": sd_control,
                          "rule": "mean(anchors) >= mean(none) - sd(none)", "passes": retrieval_ok},
            "code_to_axis": {"per_seed": axis, "mean": axis_mean,
                             "rule": "mean(anchors) > mean(none)", "passes": axis_ok}}
    return {"artifact_kind": DECISION_KIND, "version": M.ANCHOR_CONFIRM_VERSION,
            "reducer": REDUCER_VERSION, "decisions": decisions,
            "note": "train-only; n = 3; descriptive rule, no significance or equivalence test",
            "_consumed": consumed}


def write_once(path: Path, payload: dict) -> str:
    consumed = payload.pop("_consumed")
    changed = consumed.reverify()
    need(not changed, f"inputs changed before the output was written: {changed[:4]}")
    payload["consumed_sha256"] = dict(sorted(consumed.digests.items()))
    blob = (json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(blob)
    return hashlib.sha256(blob).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("select", "decide"))
    parser.add_argument("--sources", required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--selection", help="decide: the frozen N record")
    parser.add_argument("--selection-sha256")
    parser.add_argument("--arms", default="none,anchors", help="select: arms with evidence")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.stage == "decide":
            need(bool(args.selection) and bool(args.selection_sha256),
                 "decide needs --selection and --selection-sha256")
        payload = reduce_select(args) if args.stage == "select" else reduce_decide(args)
        digest = write_once(Path(args.out), payload)
    except (NotReducible, M.CellRefused, KeyError, ValueError, FileExistsError) as error:
        print(f"[anchor-reducer] REFUSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"[anchor-reducer] wrote {args.out} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
