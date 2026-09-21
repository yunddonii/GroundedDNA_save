"""Stage 1: where does the s4.7 codon-decoding advantage come from -- captions or codebooks?

Applies the paper's held-out codon decoding (pinned probe, approved decoder settings and bio
projection policy of the approved F10 aggregate) to the F09 ablation cells at seed 42:
  A5_both  -- the full recipe (lambda_codon_joint + no Gumbel), the reference arm
  A2_no_text -- the same recipe with every caption input removed
  A4_shared_codebook -- the same recipe with one codebook shared by all slots
The train rows must be exactly the approved panel's (their basename digest is compared), so the
arms are scored on the same rows as the paper's ours/CIBHash numbers. Nothing is trained.
"""
import argparse
import glob
import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import scripts.interp_control_a_partitions as drv  # noqa: E402  verified probe loading + admission

ARMS = ("A5_both", "A2_no_text", "A4_shared_codebook")
DATASETS = {"flickr25k": ("Flickr25k", "flickr_A_v4", "dataset/Flickr25k/setting1/train.txt"),
            "nuswide": ("NUSWIDE", "nuswide_A_v4", "dataset/NUSWIDE/setting1/train.txt"),
            "mscoco": ("MSCOCO", "mscoco_A_v5b", None)}


def run_dir(slug, tag, arm):
    hits = sorted(glob.glob(str(REPO / f"result/*_setting1_promptAblA_{tag}_{arm}_P0refit_*")))
    assert len(hits) == 1, (slug, arm, hits)
    return hits[0]


def coco_rows(H, d, arm):
    """MS-COCO: train is disjoint from the DB. Train codes come from mscoco_train_codes.py (encoder
    validated against this run's own extract_db.npz), bound to the same checkpoint as the query;
    the query is read at the digest this run's extraction manifest declares."""
    import os
    manifest = json.load(open(os.path.join(d, "extraction_manifest_query.json")))
    qy = H.load_verified_npz(os.path.join(d, "extract_query.npz"), manifest["npz_sha256"],
                             what=f"{arm} query")
    codes = REPO / "result/analysis/stage1_f09_decoding/mscoco_train_codes"
    meta = json.load(open(codes / f"{arm}.npz.json"))
    raw = (codes / f"{arm}.npz").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == meta["npz_sha256"], f"{arm}: train codes changed"
    assert meta["checkpoint_sha256"] == manifest["checkpoint_sha256"], \
        f"{arm}: train codes and query come from different checkpoints"
    import io
    tr = np.load(io.BytesIO(raw), allow_pickle=True)
    base = lambda paths: np.array([os.path.basename(str(p)) for p in paths])  # noqa: E731
    train_names, query_names = base(tr["image_paths"]), base(qy["image_paths"])
    assert not set(train_names) & set(query_names), "query/train leak"
    return {"train_src": tr, "train_idx": np.arange(len(train_names)), "train_basenames": train_names,
            "qy": qy, "query_ids": query_names,
            "train_labels": np.asarray(tr["multi_hot_labels"], np.int64),
            "test_labels": np.asarray(qy["multi_hot_labels"], np.int64)}


def paired_delta(ap_a, ap_b, n_boot=1000, seed=42):
    a, b = np.asarray(ap_a, float), np.asarray(ap_b, float)
    assert len(a) == len(b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    with np.errstate(invalid="ignore"):
        for i in range(n_boot):
            take = rng.integers(0, len(a), len(a))
            deltas[i] = np.nanmean(a[take]) - np.nanmean(b[take])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": float(np.nanmean(a) - np.nanmean(b)), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["flickr25k", "nuswide"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    H, importer = drv.load_probe()
    project = drv.make_projector()
    out = {"probe_sha256": drv.PROBE_SHA, "driver_sha256": drv.DRIVER_SHA256_AT_START, "cells": {}}
    for slug in args.datasets:
        dataset, tag, manifest = DATASETS[slug]
        record, aggregate, record_sha = drv.admitted_record(f"{slug}_seed42")
        settings, policy = aggregate["decoder_settings"], record["input_admission"]["applied_analysis_policy"]
        approved_rows = record["config"]["baseline_cells"]["cibhash"]["alignment_ids_sha256"]
        H._set_n_slots(5)
        per_arm, approved_train_set = {}, None

        def approved_train_names(slug_, record_=record):
            """The approved panel's train rows, read through the probe from its own run."""
            src = record_["input_admission"].get("expected_identity_source") or {}
            expect = (H.expected_identity_from_record(src["approved_aggregate"], src["sha256"],
                                                      dataset, int(src["seed"]))
                      if src.get("approved_aggregate") else None)
            ref = H.load_split(record_["config"]["ours_dir"], None, dataset=dataset, expect=expect)
            names = list(map(str, ref["train_basenames"]))
            query = list(map(str, ref["query_ids"]))
            for key, value in (("train", names), ("query", query)):
                digest = hashlib.sha256("\n".join(value).encode()).hexdigest()
                assert digest == approved_rows[key], f"{slug_}: approved {key} digest mismatch"
            return set(names), query
        for arm in ARMS:
            d = run_dir(slug, tag, arm)
            rows = (H.load_split(d, str(REPO / manifest), dataset=dataset, expect=None) if manifest
                    else coco_rows(H, d, arm))
            # The query must match in order (paired comparisons); the train rows only as a set --
            # the dictionary is order-free, and slicing train out of the DB keeps DB order where
            # the approved panel's extract_train.npz kept its own.
            if approved_train_set is None:
                approved_train_set, approved_query = approved_train_names(slug)
            assert list(map(str, rows["query_ids"])) == approved_query, f"{slug}/{arm}: query differs"
            assert set(map(str, rows["train_basenames"])) == approved_train_set, \
                f"{slug}/{arm}: train rows differ from the approved panel's as a set"
            tr = project(rows["train_src"]["base_indices"][rows["train_idx"]], policy)
            te = project(rows["qy"]["base_indices"], policy)
            res = H.decode_all_slots(H.codon_ids(tr, 3), rows["train_labels"],
                                     H.codon_ids(te, 3), rows["test_labels"], 4 ** 3,
                                     float(settings["alpha"]), int(settings["min_support"]))
            per_arm[arm] = {"run_dir": d, "concept_mAP": res["slot_mean"]["concept_mAP"],
                            "per_slot": [s.get("concept_mAP") for s in res["per_slot"]],
                            "n_train": int(len(rows["train_idx"])), "n_test": int(len(te)),
                            "_ap": res["_ap_per_sample"]}
        ref = per_arm["A5_both"]["_ap"]
        cell = {"dataset": dataset, "record_sha256": record_sha,
                "approved_ours_codon_P3": record["results"]["ours_codon"]["slot_mean"]["concept_mAP"],
                "approved_cibhash_chunk": record["results"]["cibhash_chunk"]["slot_mean"]["concept_mAP"],
                "arms": {a: {k: v for k, v in r.items() if k != "_ap"} for a, r in per_arm.items()},
                "vs_A5_both": {a: paired_delta(per_arm[a]["_ap"], ref) for a in ARMS if a != "A5_both"}}
        out["cells"][slug] = cell
        print(f"\n{dataset}  (approved P3 ours {cell['approved_ours_codon_P3']:.4f}, "
              f"CIBHash chunk {cell['approved_cibhash_chunk']:.4f})")
        for a in ARMS:
            extra = "" if a == "A5_both" else \
                f"   Δ vs A5 {cell['vs_A5_both'][a]['delta']:+.4f} CI [{cell['vs_A5_both'][a]['ci95'][0]:+.4f}, {cell['vs_A5_both'][a]['ci95'][1]:+.4f}]"
            print(f"  {a:20s} {per_arm[a]['concept_mAP']:.4f}  per-slot "
                  f"{[round(x, 3) if x is not None else None for x in per_arm[a]['per_slot']]}{extra}")
    out["executed_dependencies"] = dict(sorted(importer.executed.items()))
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
