"""Stage 3: does meaning ACCUMULATE across codon positions? (claim 3, compositional contribution)

On the approved s4.7 panel rows, for ours (5 codons) and the CIBHash control (its 15 projected bases cut
contiguously into five 3-base units, exactly the approved `cibhash_chunk` arm), labels are decoded
from a SUBSET S of positions by combining the per-position supervised dictionaries naive-Bayes style:
    score(label) = sum_{m in S} logit p_m(label | unit_m) - (|S|-1) * logit p(label)
(per-position posteriors from the probe's own `build_dictionary`, alpha and min_support from the
approved aggregate; unsupported units fall back to the prior). AP over the label ranking, averaged
over all subsets of each size. A code whose positions carry complementary meaning gains with |S|;
one whose positions repeat the same information stays flat.

Rows, bytes and settings are admitted exactly as by the repaired Reinforcement-A driver.
"""
import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import scripts.interp_control_a_partitions as drv  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def subset_curve(H, tr_units, tr_lab, te_units, te_lab, alpha, min_support, n_units=64):
    prior = (tr_lab.sum(0) + alpha) / (len(tr_lab) + 2 * alpha)
    per_pos = []
    for m in range(tr_units.shape[1]):
        p, sup = H.build_dictionary(tr_units[:, m], tr_lab, n_units, alpha)
        post = np.where((sup[te_units[:, m]] >= min_support)[:, None], p[te_units[:, m]], prior[None, :])
        per_pos.append(logit(post))
    base = logit(prior)[None, :]
    curve = {}
    for k in range(1, tr_units.shape[1] + 1):
        aps = []
        for S in itertools.combinations(range(tr_units.shape[1]), k):
            score = sum(per_pos[m] for m in S) - (k - 1) * base
            aps.append(float(np.nanmean(H.label_ranking_ap(score, te_lab))))
        curve[k] = {"mean": float(np.mean(aps)), "min": float(np.min(aps)), "max": float(np.max(aps))}
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=[f"{s}_seed{x}" for s in ("flickr25k", "nuswide", "mscoco")
                                                   for x in (42, 43, 44)])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    H, importer = drv.load_probe()
    project = drv.make_projector()
    out = {"driver_sha256": drv.DRIVER_SHA256_AT_START, "cells": {}}
    for name in args.cells:
        record, aggregate, record_sha = drv.admitted_record(name)
        settings = aggregate["decoder_settings"]
        policy = record["input_admission"]["applied_analysis_policy"]
        dataset = drv.DATASETS[name.rsplit("_", 1)[0]]
        seed = int(name.rsplit("seed", 1)[1])
        H._set_n_slots(5)
        source = record["input_admission"].get("expected_identity_source") or {}
        expect = (H.expected_identity_from_record(source["approved_aggregate"], source["sha256"],
                                                  dataset, int(source["seed"]))
                  if source.get("approved_aggregate") else None)
        rows = H.load_split(record["config"]["ours_dir"], None, dataset=dataset, expect=expect)
        ours_tr = H.codon_ids(project(rows["train_src"]["base_indices"][rows["train_idx"]], policy), 3)
        ours_te = H.codon_ids(project(rows["qy"]["base_indices"], policy), 3)
        tnpz, qnpz, trr, qrr, _ = drv.paired_control(H, record["config"], dataset, seed, rows)
        cib_tr = H.codon_ids(project(drv.hash_to_base(tnpz["hash_2bit"][trr]), policy), 3)
        cib_te = H.codon_ids(project(drv.hash_to_base(qnpz["hash_2bit"][qrr]), policy), 3)
        a, ms = float(settings["alpha"]), int(settings["min_support"])
        cell = {"ours": subset_curve(H, ours_tr, rows["train_labels"], ours_te, rows["test_labels"], a, ms),
                "cibhash_chunk": subset_curve(H, cib_tr, rows["train_labels"], cib_te, rows["test_labels"], a, ms),
                "record_sha256": record_sha}
        out["cells"][name] = cell
        o, c = cell["ours"], cell["cibhash_chunk"]
        print(f"{name:18s} ours  " + " ".join(f"{o[k]['mean']:.4f}" for k in range(1, 6)) +
              f"   gain1->5 {o[5]['mean'] - o[1]['mean']:+.4f}")
        print(f"{'':18s} cib   " + " ".join(f"{c[k]['mean']:.4f}" for k in range(1, 6)) +
              f"   gain1->5 {c[5]['mean'] - c[1]['mean']:+.4f}")
    out["executed_dependencies"] = dict(sorted(importer.executed.items()))
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
