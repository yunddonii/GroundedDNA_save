#!/usr/bin/env python
"""Held-out codeword/codon decoding (REQUIRED_EXPERIMENTS_GROUNDEDDNA_PAPER.md §2).

Question: does a `(slot, code) -> concept` dictionary built ONLY on train
predict the concepts of unseen test images from the code alone?

Protocol
    dictionary split : train rows  (Flickr/NUS: train is a subset of the DB
                       extraction, matched by image basename; MSCOCO ships its
                       own train extraction rows the same way)
    evaluation split : official test (extract_query.npz) -- disjoint from db
                       and from train, verified at load time
    target           : dataset multi-hot labels.  These are INDEPENDENT of the
                       Qwen captions used as the training teacher, so this is
                       not a circular evaluation (§2.2).
    decoder input    : the slot's integer code id and nothing else.  No test
                       caption, no test image feature.

Code units compared, all with the identical dictionary/decode procedure (§2.9):
    ours-codeword : codebook_indices[:, m]                      K per slot
    ours-codon    : 16*b0 + 4*b1 + b2 from base_indices[3m:3m+3] 64 per slot
    <base>-chunk  : contiguous 6-bit chunks of the 36-bit hash   64 per slot
    majority      : train label frequency, ignores the code
    shuffled      : train unit assignment permuted before the dictionary is
                    built, so the code carries no real information

Smoothing and min-support are fixed a priori (see DEFAULTS) and are NOT tuned
on test.  Codes with support below the threshold are `unknown` and fall back to
the prior; coverage is always reported alongside (§2.2, §2.7).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import tempfile
from typing import Dict, List, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SLOT_NAMES = [
    "global", "primary_object", "secondary_object",
    "activity_relation", "color_texture", "scene_type",
]
# Slot count is INFERRED from the extraction at run time (see main), because a
# run may carry fewer than six: dropping `scene_type` -- the only slot whose
# removal IMPROVES decoding on 4/4 datasets -- leaves 5 slots / 15 bases.
# Left as a module global so the helpers below keep their current signatures;
# `_set_n_slots` is the single place that changes it.
N_SLOTS = 6


def _set_n_slots(n: int) -> None:
    """Point the module at an n-slot code. Must be called before decoding."""
    global N_SLOTS, SLOT_NAMES
    n = int(n)
    if not (2 <= n <= 6):
        raise ValueError(f"n_slots must be in [2, 6], got {n}")
    N_SLOTS = n
    SLOT_NAMES = SLOT_NAMES[:n]

# Pre-registered, never tuned on test.
DEFAULT_ALPHA = 1.0        # Beta(alpha, alpha) smoothing per label
DEFAULT_MIN_SUPPORT = 10   # train samples needed before a code is decodable
DEFAULT_N_SHUFFLE = 5
DEFAULT_N_BOOT = 1000


# --------------------------------------------------------------------------
# code extraction
# --------------------------------------------------------------------------
def codon_ids(base_indices: np.ndarray, n_bases: int = 3) -> np.ndarray:
    """[N, 6*n_bases] A/C/G/T indices -> [N, 6] codon ids in base-4."""
    n, total = base_indices.shape
    if total != N_SLOTS * n_bases:
        raise ValueError(f"expected {N_SLOTS * n_bases} bases, got {total}")
    out = np.zeros((n, N_SLOTS), dtype=np.int64)
    for m in range(N_SLOTS):
        seg = base_indices[:, m * n_bases:(m + 1) * n_bases]
        val = np.zeros(n, dtype=np.int64)
        for r in range(n_bases):
            val = val * 4 + seg[:, r].astype(np.int64)
        out[:, m] = val
    return out


def bit_chunk_ids(hash_2bit: np.ndarray, n_chunks: int | None = None) -> np.ndarray:
    # Default resolved at CALL time, not def time: `_set_n_slots` may have
    # lowered N_SLOTS after this function object was created.
    """36-bit flat hash -> `n_chunks` contiguous chunk ids (§2.9)."""
    if n_chunks is None:
        n_chunks = N_SLOTS
    n, nbits = hash_2bit.shape
    if nbits % n_chunks:
        # `31 // 5 == 6` silently dropped the trailing bit, so a 31-bit code
        # scored identically to its 30-bit prefix -- the audit reproduced this
        # by flipping only bit 31. A budget that does not divide into the chunk
        # count is not a geometry this control can express.
        raise ValueError(
            f"hash_2bit has {nbits} bits, which is not divisible by "
            f"{n_chunks} chunks; the remainder would be discarded")
    width = nbits // n_chunks
    bits = (hash_2bit > 0).astype(np.int64)
    out = np.zeros((n, n_chunks), dtype=np.int64)
    for j in range(n_chunks):
        seg = bits[:, j * width:(j + 1) * width]
        val = np.zeros(n, dtype=np.int64)
        for r in range(width):
            val = val * 2 + seg[:, r]
        out[:, j] = val
    return out


# --------------------------------------------------------------------------
# dictionary + decoding
# --------------------------------------------------------------------------
def build_dictionary(units: np.ndarray, labels: np.ndarray, n_units: int,
                     alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """-> p[n_units, C] smoothed P(label=1 | unit), support[n_units]."""
    n_lab = labels.shape[1]
    cnt = np.zeros((n_units, n_lab), dtype=np.float64)
    sup = np.zeros(n_units, dtype=np.float64)
    np.add.at(cnt, units, labels.astype(np.float64))
    np.add.at(sup, units, 1.0)
    p = (cnt + alpha) / (sup[:, None] + 2.0 * alpha)
    return p, sup


def label_ranking_ap(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-sample AP over the label ranking. Samples with no positive -> nan."""
    n, c = scores.shape
    order = np.argsort(-scores, axis=1, kind="stable")
    hits = np.take_along_axis(truth, order, axis=1).astype(np.float64)
    csum = np.cumsum(hits, axis=1)
    ranks = np.arange(1, c + 1, dtype=np.float64)[None, :]
    prec = csum / ranks
    npos = hits.sum(axis=1)
    ap = np.where(npos > 0, (prec * hits).sum(axis=1) / np.maximum(npos, 1e-12), np.nan)
    return ap


def conditional_entropy(p: np.ndarray, sup: np.ndarray) -> float:
    """Support-weighted mean per-label Bernoulli entropy H(concept | code), nats."""
    w = sup / max(sup.sum(), 1e-12)
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    h = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))   # [n_units, C]
    return float((w[:, None] * h).sum(axis=0).mean())


def marginal_entropy(prior: np.ndarray) -> float:
    pc = np.clip(prior, 1e-12, 1 - 1e-12)
    return float((-(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))).mean())


def decode_slot(train_units: np.ndarray, train_labels: np.ndarray,
                test_units: np.ndarray, test_labels: np.ndarray,
                n_units: int, alpha: float, min_support: int) -> dict:
    p, sup = build_dictionary(train_units, train_labels, n_units, alpha)
    prior = (train_labels.sum(axis=0) + alpha) / (len(train_labels) + 2.0 * alpha)

    covered = sup[test_units] >= min_support
    scores = np.where(covered[:, None], p[test_units], prior[None, :])

    ap = label_ranking_ap(scores, test_labels)
    top1_idx = np.argmax(scores, axis=1)
    top1 = test_labels[np.arange(len(test_labels)), top1_idx].astype(np.float64)

    valid = ~np.isnan(ap)
    both = valid & covered
    active = sup >= min_support
    return {
        "concept_mAP": float(np.nanmean(ap)),
        "concept_mAP_covered_only": _defined_scalar(float(np.nanmean(ap[both])) if both.any() else float("nan")),
        "top1_acc": float(top1[valid].mean()),
        # `coverage` is over ALL test queries; the AP mean is over the queries
        # that HAVE a positive label. Reporting one denominator invited the
        # other to be assumed.
        "coverage": float(covered.mean()),
        "n_queries": int(len(ap)),
        "n_valid_queries": int(valid.sum()),
        "n_active_units": int(active.sum()),
        "n_units": int(n_units),
        # None, not NaN: undefined has to survive JSON, and `0.0` would claim
        # a conditional entropy that no active unit supports.
        "H_cond": (conditional_entropy(p[active], sup[active])
                   if active.any() else None),
        "H_marg": marginal_entropy(prior),
        "_ap": ap,          # kept for bootstrap, stripped before serialising
    }


def decode_all_slots(train_units, train_labels, test_units, test_labels,
                     n_units, alpha, min_support) -> dict:
    per_slot = [
        decode_slot(train_units[:, m], train_labels, test_units[:, m],
                    test_labels, n_units, alpha, min_support)
        for m in range(N_SLOTS)
    ]
    ap_stack = np.stack([s["_ap"] for s in per_slot])        # [6, N]
    mean_ap = np.nanmean(ap_stack, axis=0)
    return {
        "per_slot": [{k: v for k, v in s.items() if k != "_ap"} for s in per_slot],
        "slot_mean": {
            "concept_mAP": _defined_mean(mean_ap),
            "n_queries": int(per_slot[0]["n_queries"]),
            # Identical across slots: a query has a positive label or it does
            # not, and that does not depend on which slot decodes it.
            "n_valid_queries": int(per_slot[0]["n_valid_queries"]),
            "top1_acc": float(np.mean([s["top1_acc"] for s in per_slot])),
            "coverage": float(np.mean([s["coverage"] for s in per_slot])),
            # `nanmean` over all-NaN warns and returns NaN, which json.dump
            # then wrote as a bare `NaN` -- not valid JSON, and a reader could
            # not tell it apart from a number. Undefined is reported as null,
            # with the count of slots that actually defined it. It is NOT
            # rewritten as 0: no unit reaching min_support is a real state.
            "H_cond": _defined_mean(
                [np.nan if s["H_cond"] is None else s["H_cond"]
                 for s in per_slot]),
            "H_cond_defined_slots": int(sum(
                1 for s in per_slot if s["H_cond"] is not None)),
            "slots": len(per_slot),
        },
        "_ap_per_sample": mean_ap,
    }


def _defined_scalar(value):
    """A float, or None when it is undefined.

    Written out as JSON null. `0.0` would assert a measurement that no covered
    row supports, and NaN made the artifact unreadable by a strict parser.
    """
    value = float(value)
    return None if (np.isnan(value) or np.isinf(value)) else value


def _defined_mean(values):
    """Mean of the defined entries, or None when every entry is undefined."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


def majority_control(train_labels, test_labels, alpha) -> dict:
    prior = (train_labels.sum(axis=0) + alpha) / (len(train_labels) + 2.0 * alpha)
    scores = np.repeat(prior[None, :], len(test_labels), axis=0)
    ap = label_ranking_ap(scores, test_labels)
    top1 = test_labels[np.arange(len(test_labels)), np.argmax(scores, axis=1)]
    valid = ~np.isnan(ap)
    return {
        "concept_mAP": float(np.nanmean(ap)),
        "top1_acc": float(top1[valid].mean()),
        "H_marg": marginal_entropy(prior),
        "_ap_per_sample": ap,
    }


def shuffled_control(train_units, train_labels, test_units, test_labels,
                     n_units, alpha, min_support, n_rep, seed) -> dict:
    rng = np.random.default_rng(seed)
    maps, aps = [], []
    for _ in range(n_rep):
        perm = rng.permutation(len(train_units))
        res = decode_all_slots(train_units[perm], train_labels, test_units,
                               test_labels, n_units, alpha, min_support)
        maps.append(res["slot_mean"]["concept_mAP"])
        aps.append(res["_ap_per_sample"])
    return {
        "concept_mAP": float(np.mean(maps)),
        "concept_mAP_std": float(np.std(maps)),
        "n_rep": n_rep,
        "_ap_per_sample": np.nanmean(np.stack(aps), axis=0),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    """95% CI of mean(a) - mean(b), paired over test samples (§2.11)."""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "delta": float(a.mean() - b.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def codeword_codon_jsd(train_cw: np.ndarray, train_cd: np.ndarray,
                       train_labels: np.ndarray, k: int, alpha: float,
                       min_support: int) -> dict:
    """JSD between concept distributions of codewords merged into one codon (§2.8)."""
    p_cw, sup_cw = build_dictionary(train_cw, train_labels, k, alpha)
    jsds, n_pairs = [], 0
    for codon in np.unique(train_cd):
        members = np.unique(train_cw[train_cd == codon])
        members = [c for c in members if sup_cw[c] >= min_support]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                p, q = p_cw[members[i]], p_cw[members[j]]
                m = 0.5 * (p + q)
                # per-label Bernoulli JSD, averaged over labels
                def kl(x, y):
                    x = np.clip(x, 1e-12, 1 - 1e-12); y = np.clip(y, 1e-12, 1 - 1e-12)
                    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))
                jsds.append(float((0.5 * kl(p, m) + 0.5 * kl(q, m)).mean()))
                n_pairs += 1
    return {
        "mean_jsd": _defined_scalar(float(np.mean(jsds)) if jsds else float("nan")),
        "max_jsd": _defined_scalar(float(np.max(jsds)) if jsds else float("nan")),
        "n_merged_pairs": n_pairs,
    }


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
from dna_utils.extraction_validation import (  # noqa: E402
    ExpectedIdentity, ExtractionInvalid, canonical_dataset_name,
    validate_extraction_run,
)


def _basenames(paths: Sequence) -> np.ndarray:
    return np.array([os.path.basename(str(p)) for p in paths])


def expected_identity_from_record(aggregate_path: str, aggregate_sha256: str,
                                  dataset: str, seed: int) -> dict:
    """Build the expected identity FROM the approved aggregate and its record.

    A JSON the caller happens to pass is not an approved identity: an empty
    `{}` satisfied "a file was given" and published a paper-path analysis. So
    the expectation is derived here from bytes that are pinned externally --
    the aggregate must hash to `aggregate_sha256`, and each per-seed record
    must hash to what that aggregate declares for it.
    """
    path = pathlib.Path(aggregate_path)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != aggregate_sha256:
        raise ExtractionInvalid(
            f"{aggregate_path} hashes to {actual}, pinned {aggregate_sha256}")
    aggregate = json.loads(payload.decode("utf-8"))
    key = canonical_dataset_name(dataset, what="dataset").lower()
    blob = (aggregate.get("datasets") or {}).get(key)
    if not isinstance(blob, Mapping):
        raise ExtractionInvalid(
            f"approved aggregate has no dataset {key!r}")
    if seed not in list(blob.get("seeds") or ()):
        raise ExtractionInvalid(
            f"approved aggregate lists seeds {blob.get('seeds')} for {key}, "
            f"not {seed}")
    digest = (blob.get("record_sha256") or {}).get(str(seed))
    if not isinstance(digest, str):
        raise ExtractionInvalid(f"no record digest for {key} seed {seed}")
    # The aggregate NAMES its records; use that map rather than hashing every
    # neighbouring file and taking the first collision.
    named = [name for name, value in (aggregate.get("record_sha256") or {}).items()
             if value == digest]
    if len(named) != 1:
        raise ExtractionInvalid(
            f"{key} seed {seed}: the aggregate's record map has {len(named)} "
            f"filenames for digest {digest}, expected exactly one")
    record_path = path.parent / named[0]
    if not record_path.is_file():
        raise ExtractionInvalid(f"declared record is missing: {record_path}")
    body = record_path.read_bytes()
    actual_record = hashlib.sha256(body).hexdigest()
    if actual_record != digest:
        raise ExtractionInvalid(
            f"{named[0]} hashes to {actual_record}, aggregate declares {digest}")
    record = json.loads(body.decode("utf-8"))
    # The record must SAY it is this cell. Copying the CLI's dataset/seed into
    # the expectation let an aggregate claiming Flickr/42 be satisfied by a
    # record whose own body said NUSWIDE/44.
    wanted_dataset = canonical_dataset_name(dataset, what="dataset")
    record_dataset = record.get("dataset")
    if record_dataset is None or \
            canonical_dataset_name(record_dataset,
                                   what="record dataset") != wanted_dataset:
        raise ExtractionInvalid(
            f"{named[0]} declares dataset {record_dataset!r}, the aggregate "
            f"selected {wanted_dataset}")
    record_seed = record.get("seed", record.get("random_seed"))
    if record_seed != seed:
        raise ExtractionInvalid(
            f"{named[0]} declares seed {record_seed!r}, the aggregate selected "
            f"{seed}")
    completion = record.get("completion") or {}
    identity = {
        "dataset": canonical_dataset_name(dataset, what="dataset"),
        "random_seed": seed,
        "checkpoint_sha256": completion.get("final_checkpoint_sha256"),
        "inference_epoch": completion.get("final_checkpoint_epoch_zero_based"),
    }
    # A None reaches `ExpectedIdentity` as "do not check", so a missing field
    # would silently become a satisfied expectation.
    missing = sorted(k for k, v in identity.items() if v is None)
    if missing:
        raise ExtractionInvalid(
            f"{named[0]}: required identity fields are absent: {missing}; a "
            f"null expectation is not a check")
    protocol = completion.get("analysis_protocol")
    if not isinstance(protocol, Mapping):
        raise ExtractionInvalid(
            f"{named[0]}: completion.analysis_protocol is absent; the GC and "
            f"homopolymer policy this cell was projected under is not optional")
    identity["approved_analysis_protocol"] = {
        key: protocol.get(key) for key in
        ("gc_count_min_inclusive", "gc_count_max_inclusive",
         "bio_max_homopolymer_run", "gc_policy_version", "total_bases")}
    # The layout the cell was TRAINED under, from the record itself. The panel
    # constant says what the paper reports; it cannot say what this run did.
    geometry = record.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ExtractionInvalid(
            f"{named[0]}: record.geometry is absent; the slot and codon layout "
            f"this cell was trained under is not optional")
    declared = {key: geometry.get(key) for key in
                ("num_semantic_parts", "num_codebooks",
                 "num_codons_per_codebook")}
    absent = sorted(key for key, value in declared.items() if value is None)
    if absent:
        raise ExtractionInvalid(
            f"{named[0]}: record.geometry declares no {absent}; a null is not "
            f"a satisfied expectation")
    # Per-dataset and deliberately NOT pinned by the panel constant (CIFAR
    # trains K=64, Flickr K=128), so this is the one geometry field the panel
    # check cannot bind -- and a K=64 run passed off as the approved K=128 cell
    # is exactly the substitution that would otherwise go unnoticed.
    codebook_size = record.get("codebook_size")
    if codebook_size is None:
        raise ExtractionInvalid(
            f"{named[0]}: record.codebook_size is absent; the panel constant "
            f"does not pin it, so nothing else would")
    declared["codebook_size"] = codebook_size
    identity["approved_geometry"] = declared
    recipe = record.get("recipe")
    if not isinstance(recipe, Mapping) or not recipe:
        raise ExtractionInvalid(
            f"{named[0]}: record.recipe is absent; the selected hyperparameters "
            f"are what makes this cell the approved one")
    identity["approved_recipe"] = dict(recipe)
    # The same knobs as the geometry block recorded them, so the record can be
    # checked against itself. `recipe` stores strings and `geometry` floats, so
    # they are compared numerically, never as text.
    identity["approved_recipe_as_trained"] = {
        key: geometry.get(key) for key in recipe}
    identity["approved_N"] = record.get("N")
    run_dir = record.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir:
        raise ExtractionInvalid(
            f"{named[0]}: record.run_dir is absent; without it the approved "
            f"record does not name which run produced these codes")
    identity["approved_run_dir"] = run_dir
    # The run's OWN identity manifest. The record describing itself coherently
    # is not the run having been that run: `geometry` and `recipe` agreeing
    # says only that one file is self-consistent.
    run_identity_sha = completion.get("run_identity_sha256")
    if not isinstance(run_identity_sha, str) or not HEX64.match(run_identity_sha):
        raise ExtractionInvalid(
            f"{named[0]}: completion.run_identity_sha256 is absent or is not a "
            f"digest; without it the run's identity manifest is never checked")
    record_digest = record.get("identity_digest")
    geometry_digest = geometry.get("identity_digest")
    if not isinstance(record_digest, str) or not HEX64.match(record_digest):
        raise ExtractionInvalid(
            f"{named[0]}: record.identity_digest is absent or is not a digest")
    if geometry_digest != record_digest:
        raise ExtractionInvalid(
            f"{named[0]}: record.identity_digest {record_digest} and "
            f"geometry.identity_digest {geometry_digest} disagree; the record "
            f"names two different runs")
    schedule = {}
    for field in ("N", "stop_after_epoch", "epoch_budget",
                  "lr_schedule_horizon", "sinkhorn_schedule_horizon",
                  "selection_mode"):
        value = record.get(field)
        if value is None:
            raise ExtractionInvalid(
                f"{named[0]}: record.{field} is absent; a null is not a "
                f"satisfied expectation")
        schedule[field] = value
    aggregate_n = blob.get("N")
    if aggregate_n != schedule["N"]:
        raise ExtractionInvalid(
            f"{named[0]} declares N={schedule['N']!r} but the approved "
            f"aggregate selected N={aggregate_n!r} for {key}")
    # The aggregate carries the selected recipe as well as N. Comparing only N
    # left the selection half-checked: the aggregate could name one recipe
    # while the record, its geometry and the run identity agreed on another.
    aggregate_recipe = blob.get("recipe")
    if not isinstance(aggregate_recipe, Mapping) or not aggregate_recipe:
        raise ExtractionInvalid(
            f"the approved aggregate declares no recipe for {key}; N alone "
            f"does not identify the selected cell")
    for field, selected in aggregate_recipe.items():
        claimed = identity["approved_recipe"].get(field)
        if claimed is None:
            raise ExtractionInvalid(
                f"the approved aggregate selects {field}={selected!r} for "
                f"{key}, which {named[0]} does not declare")
        try:
            same = float(selected) == float(claimed)
        except (TypeError, ValueError):
            same = str(selected) == str(claimed)
        if not same:
            raise ExtractionInvalid(
                f"the approved aggregate selects {field}={selected!r} for "
                f"{key} but {named[0]} declares {field}={claimed!r}")
    identity["approved_run_identity"] = dict(
        schedule, sha256=run_identity_sha, digest=record_digest,
        aggregate_N=aggregate_n,
        inference_epoch=completion.get("final_checkpoint_epoch_zero_based"))
    for field in ("npz_sha256", "extraction_manifest_sha256",
                  "extraction_complete_sha256", "required_splits"):
        value = completion.get(field)
        if value is None:
            raise ExtractionInvalid(
                f"{named[0]}: completion.{field} is absent; a null is not a "
                f"satisfied expectation")
        identity.setdefault("approved_extraction", {})[field] = value
    identity["source"] = {
        "aggregate": str(path), "aggregate_sha256": aggregate_sha256,
        "record_file": named[0], "record_sha256": digest,
        "record_path": str(record_path),
    }
    return identity


def admit_run(result_dir: str, *, dataset: str, required_splits: Sequence[str],
              expect: "Mapping | None" = None):
    """Admit an extraction run before reading a single array from it.

    The previous contract was three `np.load` calls and a basename overlap: it
    never read the completion marker or the manifests, so it could not say
    which checkpoint produced these codes or at which epoch, and it would
    decode a run some other cell wrote.

    `validate_extraction_run` closes the INTERNAL question -- the artefacts
    exist, their digests match, and the splits agree with each other. It does
    NOT establish that this run is the current main's cell: the extraction
    manifests carry no recipe or campaign identity at all, and a self-consistent
    run from any campaign passes. That is what `expect` is for, and it must come
    from the approved aggregate or record, not from anything inside this run.
    """
    wanted = canonical_dataset_name(dataset, what="--dataset")
    if expect and "dataset" in expect:
        # Populating `fields["dataset"]` from the CLI made this key look
        # checked while the declared expectation was never compared: an
        # `expect` of NUSWIDE passed on a CIFAR10 run.
        declared = canonical_dataset_name(expect["dataset"],
                                          what="expected dataset")
        if declared != wanted:
            raise ExtractionInvalid(
                f"expected dataset {declared} != --dataset {wanted}")
    fields = {"dataset": wanted}
    for key in ("random_seed", "inference_epoch", "inference_epoch_source",
                "num_slots", "bases_per_slot", "codebook_size",
                "checkpoint_sha256", "config_sha256"):
        if expect and key in expect:
            fields[key] = expect[key]
    admitted = validate_extraction_run(
        result_dir, required_splits=tuple(required_splits), verify_npz=True,
        expected=ExpectedIdentity(**fields))
    approved = (expect or {}).get("approved_extraction")
    if approved:
        declared_splits = tuple(approved.get("required_splits") or ())
        if set(declared_splits) != set(required_splits):
            raise ExtractionInvalid(
                f"{result_dir}: the approved record requires splits "
                f"{sorted(declared_splits)}, this run has "
                f"{sorted(required_splits)}")
        for split, digest in (approved.get("npz_sha256") or {}).items():
            got = admitted.splits.get(split, {}).get("npz_sha256")
            if got != digest:
                raise ExtractionInvalid(
                    f"{result_dir}: {split} NPZ is {got}, the approved record "
                    f"declares {digest}; a self-consistent run is not this "
                    f"record's extraction")
        for split, digest in (approved.get("extraction_manifest_sha256") or {}).items():
            got = admitted.manifest_sha256.get(split)
            if got != digest:
                raise ExtractionInvalid(
                    f"{result_dir}: {split} manifest is {got}, the approved "
                    f"record declares {digest}")
        marker = approved.get("extraction_complete_sha256")
        if marker and admitted.completion_marker_sha256 != marker:
            raise ExtractionInvalid(
                f"{result_dir}: completion marker is "
                f"{admitted.completion_marker_sha256}, the approved record "
                f"declares {marker}")

    if expect:
        # Anything the validator has no field for is compared here rather than
        # quietly dropped, so an unmatched expectation cannot look satisfied.
        unchecked = sorted(set(expect) - set(fields) - _EXPECT_META)
        if unchecked:
            raise ExtractionInvalid(
                f"{result_dir}: expected identity carries {unchecked}, which "
                f"this admission does not verify; supply them through a "
                f"checked field or remove them rather than implying a check")
    return admitted


#: Derivations this consumer knows how to check. A `kind` outside this set is
#: refused rather than accepted as a label, because the audit passed
#: `kind="invented-source"` with a well-formed digest and it went through.
SUPPORTED_ID_DERIVATIONS = frozenset({"sealed_cifar_batches+frozen_sampler"})

#: The approved Phase-3 aggregate. Pinned here, not accepted from the CLI: a
#: caller who supplies both an aggregate and the digest that blesses it has
#: asserted both halves, which is not an approval.
APPROVED_AGGREGATE_SHA256 = \
    "b4f3b0dff467f7c1fd4ca134edba66085115939a4097e6bb201bd13ca83452a5"

#: Keys of `expect` that describe the expectation itself rather than a field
#: the run must match.
_EXPECT_META = frozenset(
    {"row_identity_sha256", "source", "approved_extraction",
     "approved_analysis_protocol",
     # Checked by check_record_binding against the admitted run and the paper
     # panel, not by the identity comparison this set guards.
     "approved_geometry", "approved_recipe", "approved_recipe_as_trained",
     "approved_N", "approved_run_dir", "approved_run_identity"})


def _distinct_ids(values, split: str, what: str) -> np.ndarray:
    """Identities usable as a set: present, non-empty and pairwise distinct.

    `_basenames` maps a path to its last component, so two different source
    paths can collapse onto one name. A duplicate silently shrinks the sets the
    leakage guard intersects, which makes an overlap look smaller than it is --
    the guard would then pass on a split it should reject.
    """
    ids = np.array([str(v) for v in values])
    if any(not v.strip() for v in ids):
        raise ExtractionInvalid(f"{split}: {what} contains an empty identity")
    unique = set(ids.tolist())
    if len(unique) != len(ids):
        raise ExtractionInvalid(
            f"{split}: {what} has {len(ids) - len(unique)} duplicate "
            f"identities; the leakage guard compares sets, so duplicates would "
            f"understate an overlap")
    return ids


def derive_cifar_row_ids(sealed, split: str) -> np.ndarray:
    """Derive CIFAR row identities the way the sealed extraction did.

    Runs the FROZEN functions -- `get_idx_for_uniform_sampling` for the order
    and `_cifar10_image_id` for the identity -- rather than trusting a written
    list. Hashing a declared `source_path` only shows that file is unchanged;
    it says nothing about whether the sampler ran, which is why an unrelated
    `{"unrelated": true}` file passed an earlier version with arbitrary ids.

    The split construction is `dataloaders.py:1131-1143`, and it is NOT one
    pool sampled three ways -- a first draft assumed that and produced a query
    set contained in the database:

      train     trainset, 500/class                       ->  5,000
      query     testset,  100/class                       ->  1,000
      database  ALL of trainset, then testset 900/class    -> 59,000
                at offset 100, shifted by len(trainset)
    """
    from dataloaders import _cifar10_image_id, get_idx_for_uniform_sampling

    class _Frozen:
        pass

    needed = {"train_images", "train_targets", "test_images", "test_targets"}
    # Accepts either an NpzFile or the plain mapping `cifar_pools_from_sealed`
    # returns, so the seal-rooted path and a synthetic NPZ share one deriver.
    present = set(getattr(sealed, "files", None) or sealed.keys())
    if not needed <= present:
        raise ExtractionInvalid(
            f"{split}: sealed source must carry {sorted(needed)}; found "
            f"{sorted(present)}")
    tr_img, tr_tgt = sealed["train_images"], sealed["train_targets"]
    te_img, te_tgt = sealed["test_images"], sealed["test_targets"]

    def _pool(targets):
        holder = _Frozen()
        holder.targets = list(targets)
        return holder

    if split == "train":
        idx = get_idx_for_uniform_sampling(_pool(tr_tgt), 10, 500)
        images = [tr_img[i] for i in idx]
        labels = [int(tr_tgt[i]) for i in idx]
    elif split == "query":
        idx = get_idx_for_uniform_sampling(_pool(te_tgt), 10, 100)
        images = [te_img[i] for i in idx]
        labels = [int(te_tgt[i]) for i in idx]
    elif split == "db":
        extra = get_idx_for_uniform_sampling(_pool(te_tgt), 10, 900, offset=100)
        images = [tr_img[i] for i in range(len(tr_img))] + \
                 [te_img[i] for i in extra]
        labels = [int(t) for t in tr_tgt] + [int(te_tgt[i]) for i in extra]
    else:
        raise ExtractionInvalid(f"{split}: no CIFAR sampling plan")
    ids = np.array([_cifar10_image_id(np.asarray(im)) for im in images])
    # Labels come back with the ids because they must be in the SAME sampled
    # order. Deriving only the ids left the extraction's labels unchecked, and
    # a fixture whose labels were `row_index % 10` -- disagreeing on 4,500 of
    # 5,000 train rows -- was admitted.
    return ids, np.asarray(labels, dtype=np.int64)


#: Exactly these, no more and no fewer. A seal missing the class metadata
#: passed when only the batches were looked for.
REQUIRED_CIFAR_ORIGINALS = frozenset(
    [f"cifar_train_batch_{i}" for i in range(1, 6)]
    + ["cifar_test_batch", "cifar_class_metadata"])


def sealed_cifar_source_from_record(record: "Mapping") -> dict:
    """Walk approved record -> pinned input seal -> the original CIFAR files.

    Replaces "a source NPZ the caller made and hashed". Every hop is checked
    against the declaration one level up, so the originals are reached by the
    same authority that approved the cell rather than by assertion:

      record.input_authority.seal_path + seal_file_sha256
        -> seal.inputs.dataset_rows.files[*].path + content.sha256
        -> the CIFAR batch pickles themselves
    """
    authority = record.get("input_authority")
    if not isinstance(authority, Mapping):
        raise ExtractionInvalid("record declares no input_authority")
    seal_path = authority.get("seal_path")
    seal_sha = authority.get("seal_file_sha256")
    if not isinstance(seal_path, str) or not isinstance(seal_sha, str):
        raise ExtractionInvalid(
            "input_authority declares no seal_path/seal_file_sha256")
    seal_file = pathlib.Path(seal_path)
    if not seal_file.is_file():
        raise ExtractionInvalid(f"pinned input seal is missing: {seal_file}")
    body = seal_file.read_bytes()
    actual = hashlib.sha256(body).hexdigest()
    if actual != seal_sha:
        raise ExtractionInvalid(
            f"input seal {seal_path} hashes to {actual}, record declares "
            f"{seal_sha}")
    seal = json.loads(body.decode("utf-8"))
    files = ((seal.get("inputs") or {}).get("dataset_rows") or {}).get("files")
    if not isinstance(files, list) or not files:
        raise ExtractionInvalid("input seal declares no dataset_rows.files")

    resolved: dict[str, bytes] = {}
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ExtractionInvalid("a dataset_rows file entry is not an object")
        name, path = entry.get("logical_name"), entry.get("path")
        declared = (entry.get("content") or {}).get("sha256")
        if not isinstance(name, str) or not isinstance(path, str) \
                or not isinstance(declared, str):
            raise ExtractionInvalid(
                f"dataset_rows entry {name!r} lacks path/content.sha256")
        if name in resolved:
            raise ExtractionInvalid(
                f"input seal declares {name!r} twice; the later entry would "
                f"silently replace the earlier one")
        target = pathlib.Path(path)
        if not target.is_file():
            raise ExtractionInvalid(f"sealed original is missing: {target}")
        # Return the VERIFIED BYTES, not the path. Handing back a path and
        # re-opening it later consumed a file that was swapped in between --
        # the hash comparison happened, just not on what was used.
        payload = target.read_bytes()
        got = hashlib.sha256(payload).hexdigest()
        if got != declared:
            raise ExtractionInvalid(
                f"sealed original {name} hashes to {got}, seal declares "
                f"{declared}")
        resolved[name] = payload
    if set(resolved) != REQUIRED_CIFAR_ORIGINALS:
        raise ExtractionInvalid(
            f"input seal declares {sorted(resolved)}, expected exactly "
            f"{sorted(REQUIRED_CIFAR_ORIGINALS)}")
    return resolved


def _read_cifar_batch(payload_bytes: bytes):
    """One CIFAR-10 python batch, unpickled from the VERIFIED bytes."""
    import io
    import pickle
    payload = pickle.load(io.BytesIO(payload_bytes), encoding="bytes")
    data = payload[b"data"] if b"data" in payload else payload["data"]
    labels = payload[b"labels"] if b"labels" in payload else payload["labels"]
    images = np.asarray(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.asarray(labels, dtype=np.int64)


def cifar_pools_from_sealed(resolved: "Mapping") -> dict:
    """The two original pools the loader uses, from the verified batch files."""
    train_names = [f"cifar_train_batch_{i}" for i in range(1, 6)]
    missing = [n for n in train_names + ["cifar_test_batch"] if n not in resolved]
    if missing:
        raise ExtractionInvalid(f"input seal is missing originals: {missing}")
    chunks = [_read_cifar_batch(resolved[n]) for n in train_names]
    tr_img = np.concatenate([c[0] for c in chunks])
    tr_tgt = np.concatenate([c[1] for c in chunks])
    te_img, te_tgt = _read_cifar_batch(resolved["cifar_test_batch"])
    return {"train_images": tr_img, "train_targets": tr_tgt,
            "test_images": te_img, "test_targets": te_tgt}


#: The derivations this consumer can actually PERFORM, not merely name.
_DERIVERS = {"sealed_cifar_batches+frozen_sampler": derive_cifar_row_ids}


#: Never write an analysis into these. `--overwrite` does not lift it: the
#: writer must not be able to open a producer, sealed or repository-source path
#: at all, whatever flags are passed.
PROTECTED_ROOTS = (
    "/data/yschoi/gdna_p3exec", "/data/yschoi/gdna_p3exec_result",
    "/data/yschoi/gdna_p3exec_seals", "/data/yschoi/gdna_p3exec_authority",
    "/data/yschoi/gdna_p4baseline", "/home/yschoi/GroundedDNA/scripts",
    "/home/yschoi/GroundedDNA/dna_utils", "/home/yschoi/GroundedDNA/tests",
    "/home/yschoi/GroundedDNA/dataset",
)

#: The one place an F10 derivative analysis may be written. Not caller-supplied:
#: an output root a caller chooses is a boundary the caller drew for itself.
def _fd_state(fd: int) -> tuple:
    """Where a descriptor IS right now, plus which inode it holds.

    /proc/self/fd/<n> tracks the open inode's CURRENT path, so moving that
    directory shows up here even though the fd and its st_ino do not change.
    st_dev is included because an inode number is only unique per device.
    Ported from the accepted combined writer, which draws the same boundary."""
    stat = os.fstat(fd)
    return (os.readlink(f"/proc/self/fd/{fd}"), stat.st_dev, stat.st_ino)


def _unlink_at(name: str, dir_fd: int) -> None:
    try:
        os.unlink(name, dir_fd=dir_fd)
    except FileNotFoundError:
        pass


DERIVATIVE_OUTPUT_ROOT = "/data/yschoi/gdna_f10_derivative"

#: A lowercase hex sha256, so a placeholder like "c"*64 or a null cannot pass
#: as a digest.
HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

#: The approved Phase-4 baseline aggregate, pinned here for the same reason as
#: the Phase-3 one: a caller who brings both the file and the digest has
#: asserted both halves.
APPROVED_BASELINE_AGGREGATE_SHA256 = \
    "63b597e0513c629f4f2908394c2ab87d0cd6d8f9a2501b717a5f1908135bd8d0"


#: The MSCOCO train-only derivatives this consumer may pair against, pinned by
#: their manifest digest. MSCOCO's train split is disjoint from its database,
#: so the control's train rows cannot be selected out of the DB the way the
#: other three datasets allow; one derivative per approved cell closes that.
#:
#: Empty: none has been produced. `scripts/f10_mscoco_train_preflight.py`
#: checks the parents and the planned outputs; producing and approving the
#: artefacts is a separate, separately reviewed step, and until it happens the
#: pairing stays refused. A path on the command line is not an approval.
APPROVED_MSCOCO_TRAIN_DERIVATIVES: "Mapping[str, Mapping[str, object]]" = {
    # u0/cibhash/MSCOCO/30b/seed42
    "b79c61d91bdf44fe9b58073cb388e9481fd2be71d9543601cc9a0cb28896f0ed": {
        "parent_key": "u0/cibhash/MSCOCO/30b/seed42", "seed": 42,
        "dataset": "MSCOCO", "variant": "cibhash"},
    # u0/cibhash/MSCOCO/30b/seed43
    "9f63278157051717aa0aa9a442d452d46059993a3ecb35b3aea9881c5d62995a": {
        "parent_key": "u0/cibhash/MSCOCO/30b/seed43", "seed": 43,
        "dataset": "MSCOCO", "variant": "cibhash"},
    # u0/cibhash/MSCOCO/30b/seed44
    "fc6e5747b915301daa33eaaef3a3c830cc3ce8039ec0370effc22c2f3fdb4d86": {
        "parent_key": "u0/cibhash/MSCOCO/30b/seed44", "seed": 44,
        "dataset": "MSCOCO", "variant": "cibhash"},
}

#: What a derivative manifest must say it is.
TRAIN_DERIVATIVE_KIND = "f10_mscoco_train_derivative"


def load_approved_train_derivative(cell: "Mapping", manifest_path: str, *,
                                   variant: str, dataset: str, seed: int):
    """One approved train derivative, bound to the cell it was produced from.

    Checked in the same order everything else here is: the manifest's own bytes
    against the approved registry, the manifest's claims against the parent
    cell, and then the array against the digest the manifest declares. A
    derivative that is merely present, or merely well-formed, is not a control.
    """
    path = pathlib.Path(manifest_path)
    if not path.is_file():
        raise ExtractionInvalid(
            f"baseline {variant}: no train derivative manifest at {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    approved = APPROVED_MSCOCO_TRAIN_DERIVATIVES.get(digest)
    if approved is None:
        raise ExtractionInvalid(
            f"baseline {variant}: {path} hashes to {digest}, which is not an "
            f"approved train derivative; the pairing is refused for the same "
            f"reason an arbitrary extract_train.npz is")
    manifest = json.loads(payload.decode("utf-8"))
    if manifest.get("artifact_kind") != TRAIN_DERIVATIVE_KIND:
        raise ExtractionInvalid(
            f"{path} declares artifact_kind "
            f"{manifest.get('artifact_kind')!r}, not {TRAIN_DERIVATIVE_KIND}")
    if manifest.get("status") != "complete":
        raise ExtractionInvalid(
            f"{path} declares status {manifest.get('status')!r}; a derivative "
            f"is a control only once its inference AND its verification "
            f"finished")
    # The registry says which cell each approved digest belongs to, so a real
    # derivative cannot be paired with the wrong seed's control.
    for field, want in (("parent_key", cell.get("key")),
                        ("seed", seed),
                        ("dataset", canonical_dataset_name(
                            dataset, what="--dataset")),
                        ("variant", variant)):
        for source, claimed in (("manifest", manifest.get(field)),
                                ("registry", approved.get(field, want))):
            if claimed != want:
                raise ExtractionInvalid(
                    f"{path}: the {source} declares {field}={claimed!r}, this "
                    f"control is {field}={want!r}")
    for field, want in (
            ("parent_checkpoint_sha256", cell.get("final_checkpoint_sha256")),
            ("parent_epoch_zero_based", cell.get("final_epoch_zero_based")),
            ("parent_manifest_sha256", cell.get("manifest_sha256")),
            ("parent_protocol_manifest_sha256",
             cell.get("protocol_manifest_sha256"))):
        if manifest.get(field) != want:
            raise ExtractionInvalid(
                f"{path}: {field}={manifest.get(field)!r} but the approved "
                f"cell {cell.get('key')} declares {want!r}; a derivative names "
                f"the checkpoint it was produced from")
    for field, want in (("bit_length", PAPER_GEOMETRY["total_bits"]),
                        ("base_length", PAPER_GEOMETRY["total_bases"])):
        if manifest.get(field) != want:
            raise ExtractionInvalid(
                f"{path}: {field}={manifest.get(field)!r}, this panel is "
                f"{want}")
    artifact_sha = manifest.get("artifact_sha256")
    artifact_name = manifest.get("artifact") or "extract_train.npz"
    if not isinstance(artifact_sha, str) or not HEX64.match(artifact_sha):
        raise ExtractionInvalid(
            f"{path} declares no digest for {artifact_name}")
    npz = load_verified_npz(str(path.parent / artifact_name), artifact_sha,
                            what=f"train derivative {cell.get('key')}")
    rows = int(np.asarray(npz["hash_2bit"]).shape[0])
    if rows != manifest.get("n_rows"):
        raise ExtractionInvalid(
            f"{path}: the array carries {rows} rows, the manifest declares "
            f"{manifest.get('n_rows')!r}")
    labels = int(np.asarray(npz["multi_hot_labels"]).shape[1])
    if labels != manifest.get("n_labels"):
        raise ExtractionInvalid(
            f"{path}: the array carries {labels} label columns, the manifest "
            f"declares {manifest.get('n_labels')!r}")
    if "image_paths" not in npz.files:
        raise ExtractionInvalid(
            f"{path}: the derivative carries no image_paths, so its rows "
            f"cannot be paired with ours; a positional match is not a "
            f"correspondence")
    return npz, {"manifest": str(path), "manifest_sha256": digest,
                 "artifact": str(path.parent / artifact_name),
                 "artifact_sha256": artifact_sha, "n_rows": rows,
                 "n_labels": labels,
                 "parent_key": manifest.get("parent_key"),
                 "parent_checkpoint_sha256":
                     manifest.get("parent_checkpoint_sha256"),
                 "split_list_sha256": manifest.get("split_list_sha256")}


def approved_baseline_cell(aggregate_path: str, aggregate_sha256: str, *,
                           variant: str, dataset: str, seed: int, bit: int) -> dict:
    """Select ONE Phase-4 cell and read the files its manifest declares.

    Consuming a baseline directory because it sits next to the right name is
    what let a 30-bit extraction with no checkpoint, manifest or completion
    evidence be paired against the paper model. The pairing has to come from
    the approved aggregate: it names the cell, the cell names its manifest, and
    the manifest names the artefacts and their digests.
    """
    path = pathlib.Path(aggregate_path)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != aggregate_sha256:
        raise ExtractionInvalid(
            f"{aggregate_path} hashes to {actual}, pinned {aggregate_sha256}")
    aggregate = json.loads(payload.decode("utf-8"))
    wanted = canonical_dataset_name(dataset, what="baseline dataset")
    hits = [r for r in aggregate.get("records", [])
            if isinstance(r, Mapping) and r.get("variant") == variant
            and canonical_dataset_name(str(r.get("dataset")),
                                       what="record dataset") == wanted
            and r.get("seed") == seed and r.get("bit") == bit]
    if len(hits) != 1:
        raise ExtractionInvalid(
            f"{len(hits)} Phase-4 cells match variant={variant} "
            f"dataset={wanted} seed={seed} bit={bit}, expected exactly one")
    record = hits[0]
    if record.get("paper_table_eligible") is not True:
        raise ExtractionInvalid(
            f"{record.get('key')} is not paper_table_eligible")

    manifest_path = record.get("manifest")
    declared = record.get("manifest_sha256")
    if not isinstance(manifest_path, str) or not isinstance(declared, str):
        raise ExtractionInvalid(f"{record.get('key')} declares no manifest")
    body = pathlib.Path(manifest_path).read_bytes()
    got = hashlib.sha256(body).hexdigest()
    if got != declared:
        raise ExtractionInvalid(
            f"{manifest_path} hashes to {got}, the aggregate declares {declared}")
    manifest = json.loads(body.decode("utf-8"))

    for field, want in (("dataset", wanted), ("seed", seed),
                        ("bit_length", bit), ("method", variant)):
        value = manifest.get(field)
        if field == "dataset":
            value = canonical_dataset_name(str(value), what="manifest dataset")
        if value != want:
            raise ExtractionInvalid(
                f"{manifest_path} declares {field}={manifest.get(field)!r}, the "
                f"selected cell is {want!r}")
    # The parent files, not just their declared digests. Deleting or editing
    # the checkpoint or the protocol manifest left the query NPZ consumable,
    # because nothing here opened them.
    record_ckpt = record.get("final_checkpoint_sha256")
    if not isinstance(record_ckpt, str) or not HEX64.match(record_ckpt):
        raise ExtractionInvalid(
            f"{record.get('key')} declares final_checkpoint_sha256="
            f"{record_ckpt!r}; the approved author-fixed panel carries one, and "
            f"an absent expectation is not a satisfied one")
    manifest_ckpt = manifest.get("final_checkpoint_sha256")
    if not isinstance(manifest_ckpt, str) or not HEX64.match(manifest_ckpt):
        raise ExtractionInvalid(
            f"{manifest_path} declares final_checkpoint_sha256="
            f"{manifest_ckpt!r}, which is not a sha256")
    if record_ckpt != manifest_ckpt:
        raise ExtractionInvalid(
            f"{record.get('key')} declares checkpoint {record_ckpt}, its "
            f"manifest declares {manifest_ckpt}")
    epoch = manifest.get("final_epoch_zero_based")
    if not isinstance(epoch, int):
        raise ExtractionInvalid(
            f"{manifest_path} declares final_epoch_zero_based={epoch!r}; a "
            f"null terminal epoch is not a check")
    # An integer is not the APPROVED integer. A record whose terminal epoch is
    # 59 paired with a manifest saying 58 passed, because only the type was
    # checked. Each method keeps its own approved terminal epoch -- this does
    # not require ours and the baseline to share one.
    # No best_epoch fallback: this panel is author-fixed terminal, and a
    # missing declaration must not resolve to the local manifest's own value.
    record_epoch = record.get("final_epoch_zero_based")
    if not isinstance(record_epoch, int):
        raise ExtractionInvalid(
            f"{record.get('key')} declares final_epoch_zero_based="
            f"{record_epoch!r}; the approved record must state its terminal "
            f"epoch rather than inherit the manifest's")
    if record_epoch != epoch:
        raise ExtractionInvalid(
            f"{record.get('key')} declares terminal epoch {record_epoch}, its "
            f"manifest declares {epoch}")
    for label, path_field, digest in (
            ("checkpoint", "final_checkpoint", manifest_ckpt),
            ("protocol manifest", "protocol_manifest",
             manifest.get("protocol_manifest_sha256"))):
        target = manifest.get(path_field)
        if not isinstance(target, str) or not isinstance(digest, str) \
                or not HEX64.match(digest):
            raise ExtractionInvalid(
                f"{manifest_path} declares no usable {label} path/sha256")
        file_path = pathlib.Path(target)
        if not file_path.is_file():
            raise ExtractionInvalid(f"{label} is missing: {file_path}")
        got_file = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if got_file != digest:
            raise ExtractionInvalid(
                f"{label} {file_path} hashes to {got_file}, the manifest "
                f"declares {digest}")

    artifacts = manifest.get("extraction_artifact_sha256")
    extraction_dir = manifest.get("extraction_dir")
    if not isinstance(artifacts, Mapping) or not isinstance(extraction_dir, str):
        raise ExtractionInvalid(
            f"{manifest_path} declares no extraction_dir/extraction_artifact_sha256")
    for name in ("extract_db.npz", "extract_query.npz"):
        if name not in artifacts:
            raise ExtractionInvalid(
                f"{manifest_path} declares no digest for {name}")
    return {
        "key": record.get("key"), "extraction_dir": extraction_dir,
        "artifact_sha256": dict(artifacts),
        "final_checkpoint_sha256": manifest.get("final_checkpoint_sha256"),
        "final_epoch_zero_based": manifest.get("final_epoch_zero_based"),
        "protocol_manifest_sha256": manifest.get("protocol_manifest_sha256"),
        "manifest": manifest_path, "manifest_sha256": declared,
    }


def load_admitted_baseline(cell: "Mapping", split: str):
    """One baseline split, from the bytes its manifest vouches for."""
    name = f"extract_{split}.npz"
    digest = cell["artifact_sha256"].get(name)
    if not digest:
        raise ExtractionInvalid(f"{cell['key']}: no declared digest for {name}")
    return load_verified_npz(os.path.join(cell["extraction_dir"], name), digest,
                             what=f"baseline {cell['key']} {split}")


#: The panel this control is for: 5 slots x 3 bases = 15 bases = 30 bits.
PAPER_GEOMETRY = {"num_slots": 5, "bases_per_slot": 3,
                  "total_bases": 15, "total_bits": 30}

#: The approved aggregate is the refit aggregate: N is selected on train only
#: and the reported cell is retrained at it. A `select` cell carries the same
#: record and identity fields and is a different experiment, so record and
#: identity agreeing with each other does not make one a paper row.
PAPER_SELECTION_MODE = "refit"


def check_bio_policy(args, expect) -> dict:
    """The projection policy actually applied must be the approved one.

    `--gc_min_frac 0.4444 --gc_max_frac 0.5556` (the old [7,8] window) and
    `--max_run 4` both produced `paper_eligible` output, and nothing in it said
    which policy had been used -- a wrong override and the default run were
    indistinguishable afterwards. Compared on the INTEGER feasible set, so a
    different fraction spelling of the same window is not a different
    experiment.
    """
    from dna_utils.gc_policy import resolve_gc_policy
    from dna_utils.bio_constraints import (_resolve_gc_count_range,
                                           _validate_constraint_parameters)

    total_bases = PAPER_GEOMETRY["total_bases"]
    central = resolve_gc_policy(total_bases)
    gmin = args.gc_min_frac if args.gc_min_frac is not None else central.gc_min_frac
    gmax = args.gc_max_frac if args.gc_max_frac is not None else central.gc_max_frac
    mrun = args.max_run if args.max_run is not None else central.max_run
    # Resolve through the SAME central functions `is_valid_batch` calls, on the
    # same arguments. Recomputing the window here with a hand-written epsilon
    # was a second implementation of the policy, and the two disagreed:
    # 0.400000000001/0.599999999999 recorded [6,9] while the projector applied
    # (7,8), so the artifact named a feasible set the codes had never been
    # projected onto. This also inherits the central domain/finiteness checks.
    try:
        gmin, gmax, mrun = _validate_constraint_parameters(gmin, gmax, mrun)
        lo, hi = _resolve_gc_count_range(total_bases, gmin, gmax)
    except (ValueError, TypeError) as exc:
        raise ExtractionInvalid(
            f"the requested projection policy is not usable: {exc}") from None
    applied = {"gc_count_min_inclusive": int(lo), "gc_count_max_inclusive": int(hi),
               "bio_max_homopolymer_run": int(mrun),
               "total_bases": total_bases,
               # The version the consuming code implements, never the version a
               # record claims: echoing the record made `unknown-policy-v99`
               # print as though it were the policy actually applied.
               "gc_policy_version": central.policy_version,
               "projection": "bio" if args.bio_project else "raw"}
    approved = (expect or {}).get("approved_analysis_protocol")
    if approved:
        claimed = approved.get("gc_policy_version")
        if claimed != central.policy_version:
            raise ExtractionInvalid(
                f"the approved cell declares gc_policy_version={claimed!r} but "
                f"this tree implements {central.policy_version!r}; the code that "
                f"projects is the authority on which policy is in force")
        for key in ("gc_count_min_inclusive", "gc_count_max_inclusive",
                    "bio_max_homopolymer_run", "total_bases"):
            want = approved.get(key)
            if want is None:
                raise ExtractionInvalid(
                    f"the approved analysis protocol declares no {key}")
            if applied[key] != want:
                raise ExtractionInvalid(
                    f"applied {key}={applied[key]} but the approved cell was "
                    f"projected under {key}={want}; a different feasible set is "
                    f"a different experiment")
    return applied


def load_verified_run_identity(run_dir: str, expected_sha256: str):
    """Read the run's identity manifest ONCE and parse the verified bytes.

    `load_run_manifest` re-opens the path and answers every failure with the
    same `None` -- missing, unparseable, wrong schema, forged digest. A
    consumer cannot tell those apart, and hashing a path and then re-opening it
    is the gap this module closes everywhere else.
    """
    import dataclasses
    from dna_utils.run_identity import (RunIdentity, MANIFEST_NAME,
                                        _SCHEMA_VERSION)

    path = pathlib.Path(run_dir) / MANIFEST_NAME
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ExtractionInvalid(
            f"the approved record declares run_identity_sha256 "
            f"{expected_sha256} but {path} cannot be read ({exc})") from None
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ExtractionInvalid(
            f"{path} hashes to {actual}, the approved record declares "
            f"{expected_sha256}")
    try:
        blob = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractionInvalid(f"{path} is not readable JSON: {exc}") from None
    if not isinstance(blob, Mapping):
        raise ExtractionInvalid(f"{path} is not a JSON object")
    fields = dict(blob)
    stored = fields.pop("digest", None)
    if fields.get("schema_version") != _SCHEMA_VERSION:
        raise ExtractionInvalid(
            f"{path} is schema {fields.get('schema_version')!r}, this consumer "
            f"reads schema {_SCHEMA_VERSION}")
    known = {f.name for f in dataclasses.fields(RunIdentity)}
    unknown = sorted(set(fields) - known)
    missing = sorted(known - set(fields))
    if unknown or missing:
        raise ExtractionInvalid(
            f"{path} does not describe a schema {_SCHEMA_VERSION} run "
            f"(unexpected {unknown}, absent {missing})")
    identity = RunIdentity(**fields)
    if stored != identity.digest:
        raise ExtractionInvalid(
            f"{path} stores digest {stored!r} but its own fields hash to "
            f"{identity.digest}; the manifest has been edited")
    return identity, str(path), actual


def check_run_identity_binding(admitted, args, expect) -> dict:
    """Bind the approved record to the identity the run actually recorded.

    Everything checked so far lives inside the record: its geometry against its
    recipe, its digests against each other. A run trained under a different
    lambda, stopped at a different epoch, or annealed over a different horizon
    satisfies all of it, because nothing had read the run's own manifest.
    """
    approved = (expect or {}).get("approved_run_identity")
    if not approved:
        return {}
    identity, path, sha = load_verified_run_identity(
        (expect or {})["approved_run_dir"], approved["sha256"])
    if identity.digest != approved["digest"]:
        raise ExtractionInvalid(
            f"{path} is run {identity.digest}, the approved record names run "
            f"{approved['digest']}; the digest is over the identity fields, so "
            f"these are different runs")
    if canonical_dataset_name(identity.dataset, what="run identity dataset") != \
            (expect or {}).get("dataset"):
        raise ExtractionInvalid(
            f"{path} was run on {identity.dataset!r}, the approved record "
            f"selected {(expect or {}).get('dataset')!r}")
    if identity.seed != (expect or {}).get("random_seed"):
        raise ExtractionInvalid(
            f"{path} was run at seed {identity.seed!r}, the approved record "
            f"selected seed {(expect or {}).get('random_seed')!r}")
    geometry = (expect or {}).get("approved_geometry") or {}
    for attr, want, where in (
            ("num_slots", geometry.get("num_semantic_parts"), "record geometry"),
            ("bases_per_slot", geometry.get("num_codons_per_codebook"),
             "record geometry"),
            ("codebook_size", geometry.get("codebook_size"), "record geometry"),
            ("total_bases", PAPER_GEOMETRY["total_bases"], "the paper panel"),
            ("total_bits", PAPER_GEOMETRY["total_bits"], "the paper panel")):
        got = getattr(identity, attr)
        if got != want:
            raise ExtractionInvalid(
                f"{path} recorded {attr}={got!r}, {where} declares {want!r}")
    # N is the SELECTED epoch, so it is the epoch trained to, the epoch stopped
    # at and the epoch whose checkpoint was extracted; the budget and both
    # anneal horizons span N+1 epochs. A cell whose schedules outrun its stop
    # was annealed on a curve it never reached the end of.
    n = approved["N"]
    for label, got in (("record.stop_after_epoch", approved["stop_after_epoch"]),
                       ("run identity stop_after_epoch",
                        identity.stop_after_epoch),
                       ("record completion inference_epoch",
                        approved.get("inference_epoch")),
                       ("extraction inference_epoch",
                        admitted.common.get("inference_epoch")),
                       ("extraction training_stop_epoch",
                        admitted.common.get("training_stop_epoch"))):
        if got != n:
            raise ExtractionInvalid(
                f"the approved cell selected N={n} but {label} is {got!r}; the "
                f"selected epoch is the epoch this analysis decodes")
    for label, got in (("record.epoch_budget", approved["epoch_budget"]),
                       ("record.lr_schedule_horizon",
                        approved["lr_schedule_horizon"]),
                       ("record.sinkhorn_schedule_horizon",
                        approved["sinkhorn_schedule_horizon"]),
                       ("run identity epoch_budget", identity.epoch_budget),
                       ("run identity lr_schedule_horizon",
                        identity.lr_schedule_horizon),
                       ("run identity sinkhorn_schedule_horizon",
                        identity.sinkhorn_schedule_horizon),
                       # The extraction manifests declare these too, and the
                       # shared validator only checks they agree ACROSS splits.
                       # Three manifests can agree with each other on a horizon
                       # the approved cell never used.
                       ("extraction training_epoch_budget",
                        admitted.common.get("training_epoch_budget")),
                       ("extraction lr_schedule_horizon",
                        admitted.common.get("lr_schedule_horizon")),
                       ("extraction sinkhorn_schedule_horizon",
                        admitted.common.get("sinkhorn_schedule_horizon"))):
        if got != n + 1:
            raise ExtractionInvalid(
                f"the approved cell selected N={n}, so the budget and both "
                f"anneal horizons are {n + 1}; {label} is {got!r}")
    if identity.selection_mode != approved["selection_mode"]:
        raise ExtractionInvalid(
            f"{path} recorded selection_mode={identity.selection_mode!r}, the "
            f"approved record declares {approved['selection_mode']!r}")
    if identity.selection_mode != PAPER_SELECTION_MODE:
        raise ExtractionInvalid(
            f"{path} recorded selection_mode={identity.selection_mode!r}; the "
            f"approved panel is the {PAPER_SELECTION_MODE}, and a record and "
            f"an identity that agree with each other on some other mode still "
            f"describe a different experiment")
    recipe = (expect or {}).get("approved_recipe") or {}
    for key, selected in recipe.items():
        if not hasattr(identity, key):
            raise ExtractionInvalid(
                f"the approved recipe selects {key}, which the run identity "
                f"does not record; it cannot be confirmed against the run")
        trained = getattr(identity, key)
        try:
            same = float(selected) == float(trained)
        except (TypeError, ValueError):
            same = str(selected) == str(trained)
        if not same:
            raise ExtractionInvalid(
                f"the approved recipe selects {key}={selected!r} but {path} "
                f"recorded the run using {key}={trained!r}")
    return {"run_identity_path": path, "run_identity_sha256": sha,
            "digest": identity.digest, "N": n,
            "epoch_budget": identity.epoch_budget,
            "lr_schedule_horizon": identity.lr_schedule_horizon,
            "sinkhorn_schedule_horizon": identity.sinkhorn_schedule_horizon,
            "selection_mode": identity.selection_mode,
            "verified_recipe": {key: getattr(identity, key) for key in recipe}}


def check_record_binding(admitted, args, expect) -> dict:
    """The admitted run must be the run the approved record describes.

    `check_paper_geometry` compares the extraction against the paper panel
    constant, which every 5 x 3 run satisfies. It cannot tell the approved cell
    from another cell of the same shape trained under a different recipe, and
    the record's own `geometry`/`recipe`/`run_dir` were never read at all -- so
    an approved record could be paired with a different run directory.
    """
    approved = (expect or {}).get("approved_geometry")
    if not approved:
        return {}
    slots = approved["num_semantic_parts"]
    per_slot = approved["num_codons_per_codebook"]
    if approved["num_codebooks"] != slots:
        raise ExtractionInvalid(
            f"the approved record declares num_codebooks="
            f"{approved['num_codebooks']} and num_semantic_parts={slots}; one "
            f"codebook per slot is what makes a slot's codons a codebook")
    if slots != PAPER_GEOMETRY["num_slots"] or \
            per_slot != PAPER_GEOMETRY["bases_per_slot"]:
        raise ExtractionInvalid(
            f"the approved record was trained at {slots} x {per_slot}, the "
            f"paper panel is {PAPER_GEOMETRY['num_slots']} x "
            f"{PAPER_GEOMETRY['bases_per_slot']}")
    if slots * per_slot != PAPER_GEOMETRY["total_bases"]:
        raise ExtractionInvalid(
            f"{slots} x {per_slot} is {slots * per_slot} bases, the panel is "
            f"{PAPER_GEOMETRY['total_bases']}")
    common = admitted.common
    for record_key, run_key in (("num_semantic_parts", "num_slots"),
                                ("num_codons_per_codebook", "bases_per_slot"),
                                ("codebook_size", "codebook_size")):
        got = common.get(run_key)
        if got != approved[record_key]:
            raise ExtractionInvalid(
                f"the extraction reports {run_key}={got!r} but the approved "
                f"record was trained with {record_key}={approved[record_key]!r}")
    # Compared numerically: the recipe stores "0.02" where the geometry stores
    # 0.02, and a string comparison would call every cell a mismatch.
    recipe = (expect or {}).get("approved_recipe") or {}
    as_trained = (expect or {}).get("approved_recipe_as_trained") or {}
    for key, selected in recipe.items():
        trained = as_trained.get(key)
        if trained is None:
            raise ExtractionInvalid(
                f"the approved recipe selects {key}={selected!r} but the "
                f"record's geometry does not say what was trained")
        try:
            same = float(selected) == float(trained)
        except (TypeError, ValueError):
            same = str(selected) == str(trained)
        if not same:
            raise ExtractionInvalid(
                f"the approved recipe selects {key}={selected!r} but the cell "
                f"was trained with {key}={trained!r}")
    approved_dir = pathlib.Path((expect or {})["approved_run_dir"])
    given = pathlib.Path(args.ours_dir)
    if given.resolve() != approved_dir.resolve():
        raise ExtractionInvalid(
            f"--ours_dir {given} is not the run the approved record names "
            f"({approved_dir}); an approved record does not approve whichever "
            f"directory is passed with it")
    return {"num_semantic_parts": slots, "num_codons_per_codebook": per_slot,
            "num_codebooks": approved["num_codebooks"],
            "codebook_size": approved["codebook_size"],
            "recipe": dict(recipe), "N": (expect or {}).get("approved_N"),
            "run_dir": str(approved_dir)}


def check_paper_geometry(admitted, probe) -> None:
    """The declared and the actual geometry must both be the paper panel.

    `bit_chunk_ids` refusing an indivisible width is not enough: 35- and 40-bit
    codes divide by 5 too, so they produced 7- and 8-bit chunks and passed as a
    control for a 30-bit / 5 x 6-bit model. Being expressible is not being
    comparable.
    """
    common = admitted.common
    for field, want in PAPER_GEOMETRY.items():
        got = common.get(field)
        if got != want:
            raise ExtractionInvalid(
                f"admitted geometry {field}={got!r}, this control is for "
                f"{field}={want}; a different budget is not the same panel")
    bases = np.asarray(probe["base_indices"])
    bits = np.asarray(probe["hash_2bit"])
    if bases.shape[1] != PAPER_GEOMETRY["total_bases"] or \
            bits.shape[1] != PAPER_GEOMETRY["total_bits"]:
        raise ExtractionInvalid(
            f"arrays carry {bases.shape[1]} bases / {bits.shape[1]} bits, the "
            f"declared panel is {PAPER_GEOMETRY['total_bases']} / "
            f"{PAPER_GEOMETRY['total_bits']}")


def load_verified_npz(path: str, expected_sha256: str, *, what: str):
    """Load arrays from the bytes that were VERIFIED, not from the path again.

    `admit_run` hashes each split's NPZ, and the consumer then re-opened it by
    name. A file swapped between those two moments was consumed unchecked --
    the same shape as the source-chain gap, one layer down.
    """
    import io
    payload = pathlib.Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and actual != expected_sha256:
        raise ExtractionInvalid(
            f"{what}: {path} hashes to {actual}, the admitted manifest "
            f"declares {expected_sha256}")
    return np.load(io.BytesIO(payload), allow_pickle=True)


#: Which route actually produced each split's identities, so the output can
#: state it instead of guessing from whether a CLI flag was passed.
_ROW_IDENTITY_SOURCE: dict[str, str] = {}


def _index_by_identity(npz, what: str) -> dict:
    """row identity -> row index, refusing anything a dict would swallow."""
    if "image_paths" not in npz.files:
        raise ExtractionInvalid(
            f"{what} carries no image_paths, so its rows cannot be paired with "
            f"ours; a positional match is not a correspondence")
    ids = _distinct_ids(_basenames(npz["image_paths"]), what, "image_paths")
    return {name: i for i, name in enumerate(ids)}


def _paired_rows(index: "Mapping", wanted, what: str, *,
                 exact: bool = False) -> np.ndarray:
    """Our rows, in our order, located in the control -- or a refusal.

    `exact` is for the query: it is the same evaluation set, so a control with
    EXTRA query rows is not that set. The database is different -- it legitimately
    holds pool rows beyond the train selection -- so this is not applied there.
    """
    missing = [w for w in wanted if w not in index]
    if missing:
        raise ExtractionInvalid(
            f"{what} is missing {len(missing)} of our {len(wanted)} rows "
            f"(first: {missing[0]!r}); a control that does not cover our rows "
            f"is not paired with them")
    if exact and len(index) != len(wanted):
        extra = sorted(set(index) - set(wanted))
        raise ExtractionInvalid(
            f"{what} has {len(index)} rows for our {len(wanted)}; "
            f"{len(extra)} extra (first: {extra[0]!r}). The query is the same "
            f"evaluation set on both sides, not a superset")
    return np.array([index[w] for w in wanted])


def _check_paired_labels(npz, rows, ours_labels, what: str) -> None:
    """The paired rows must carry the same labels we scored ours against."""
    theirs = np.asarray(npz["multi_hot_labels"])[rows]
    ours = np.asarray(ours_labels)
    if theirs.shape != ours.shape or not np.array_equal(theirs, ours):
        wrong = (int((theirs != ours).any(axis=1).sum())
                 if theirs.shape == ours.shape else len(ours))
        raise ExtractionInvalid(
            f"{what}: {wrong} paired rows carry different labels than ours; "
            f"same identities with different labels are not the same rows")


def _check_labels(npz, split: str, derived_labels) -> None:
    """The consumed labels must be in the derived sampling order.

    Compares the exact one-hot matrix, not argmax plus a max-of-sums: the
    earlier form checked that the LARGEST row sum was 1, which an all-zero row
    does not change, and `argmax` of an all-zero vector is 0 -- so a blanked
    class-0 row read as a correct class-0 row in all three splits.

    Scoped to the CIFAR derivation; Flickr's legitimately zero-positive
    multi-label rows are not affected.
    """
    stored = np.asarray(npz["multi_hot_labels"])
    if stored.ndim != 2 or stored.shape[0] != len(derived_labels):
        raise ExtractionInvalid(
            f"{split}: multi_hot_labels has shape {stored.shape}, derivation "
            f"produced {len(derived_labels)} rows")
    expected_onehot = np.eye(stored.shape[1], dtype=stored.dtype)[derived_labels]
    if not np.array_equal(stored, expected_onehot):
        wrong = int((stored != expected_onehot).any(axis=1).sum())
        raise ExtractionInvalid(
            f"{split}: {wrong} of {len(derived_labels)} label rows disagree "
            f"with the derived sampling order; matching ids do not make the "
            f"labels the same rows")


def _row_ids(npz, split: str, sidecar: "Mapping | None", npz_sha256: str,
             *, pools: "Mapping | None" = None):
    """Row identities for one split, or a refusal that names what is missing.

    CIFAR-10's sealed extractions carry `base_indices`, `hash_2bit`,
    `codebook_indices` and `multi_hot_labels` -- and no `image_paths`. The old
    reader indexed `npz["image_paths"]` unconditionally, so those runs raise
    `KeyError` before any leak check happens.

    A sidecar is NOT an answer on its own. Matching an npz digest proves the
    sidecar names these bytes; it proves nothing about whether its Nth entry is
    this NPZ's Nth row. A file could carry the right digest and the list
    `[0, 1, 2, ...]` and pass -- which is the row-number substitution this is
    supposed to refuse. So the sidecar must additionally declare where the
    identities were DERIVED from, and the caller must pin that derivation
    externally (`--row-identity-sha256`), the way §294 rebuilt CIFAR ids from
    the sealed batches and the frozen sampler rather than from the NPZ.
    """
    if "image_paths" in npz.files:
        _ROW_IDENTITY_SOURCE[split] = "npz image_paths"
        return _distinct_ids(_basenames(npz["image_paths"]), split, "image_paths")
    if pools is not None:
        _ROW_IDENTITY_SOURCE[split] = "sealed originals via approved record"
        # Reached through the approved record's pinned seal, so nothing the
        # caller wrote is involved.
        derived, derived_labels = derive_cifar_row_ids(pools, split)
        _check_labels(npz, split, derived_labels)
        return _distinct_ids(derived, split, "seal-derived ids")
    _ROW_IDENTITY_SOURCE[split] = "--row-identity sidecar"
    if sidecar is None:
        raise ExtractionInvalid(
            f"{split}: this extraction has no `image_paths` (keys: "
            f"{sorted(npz.files)}), so row identity cannot be derived from it. "
            f"Supply --row-identity plus --row-identity-sha256 naming a "
            f"derivation bound to the sealed source; row numbers or synthesised "
            f"names would satisfy the leakage guard without evidence.")
    entry = sidecar.get(split)
    if not isinstance(entry, Mapping):
        raise ExtractionInvalid(f"{split}: --row-identity declares no entry")
    if entry.get("npz_sha256") != npz_sha256:
        raise ExtractionInvalid(
            f"{split}: --row-identity is bound to npz_sha256="
            f"{entry.get('npz_sha256')!r}, but the admitted extraction is "
            f"{npz_sha256!r}")
    source = entry.get("derived_from")
    if not isinstance(source, Mapping):
        raise ExtractionInvalid(
            f"{split}: --row-identity entry declares no `derived_from`; an id "
            f"list that vouches only for itself is not evidence of row order")
    kind = source.get("kind")
    if kind not in SUPPORTED_ID_DERIVATIONS:
        raise ExtractionInvalid(
            f"{split}: derived_from.kind {kind!r} is not one of "
            f"{sorted(SUPPORTED_ID_DERIVATIONS)}; naming an unsupported or "
            f"invented source does not make the ids derived from anything")
    source_path = source.get("source_path")
    if not isinstance(source_path, str) or not os.path.isfile(source_path):
        raise ExtractionInvalid(
            f"{split}: derived_from names no readable `source_path`; a digest "
            f"with no file behind it cannot be checked")
    source_payload = pathlib.Path(source_path).read_bytes()
    actual = hashlib.sha256(source_payload).hexdigest()
    if actual != source.get("source_sha256"):
        raise ExtractionInvalid(
            f"{split}: derived_from source {source_path} hashes to {actual}, "
            f"declared {source.get('source_sha256')!r}")
    # RUN the derivation. Matching a digest proves the source file is intact;
    # only executing the sampler proves these ids came out of it.
    # Same rule for the fallback sidecar source: the payload was hashed above,
    # so restore the arrays from it rather than re-opening the path.
    import io as _io
    with np.load(_io.BytesIO(source_payload), allow_pickle=False) as sealed:
        derived, derived_labels = _DERIVERS[kind](sealed, split)
    declared_ids = entry.get("ids")
    if not isinstance(declared_ids, list) or \
            [str(v) for v in declared_ids] != derived.tolist():
        raise ExtractionInvalid(
            f"{split}: --row-identity ids are not what {kind} produces from "
            f"the declared source; the list was written down, not derived")
    _check_labels(npz, split, derived_labels)
    ids = entry.get("ids")
    if not isinstance(ids, list) or len(ids) != len(npz["base_indices"]):
        raise ExtractionInvalid(
            f"{split}: --row-identity lists {len(ids) if isinstance(ids, list) else '?'} "
            f"ids for {len(npz['base_indices'])} rows")
    return _distinct_ids(ids, split, "--row-identity ids")


def load_row_identity(path: str, expected_sha256: str) -> dict:
    """Read a row-identity sidecar and pin it to an EXTERNAL digest.

    The digest is supplied by the caller, not read from the file, so a sidecar
    cannot certify itself. Without this the whole construct reduces to "trust
    whatever list is on disk".
    """
    payload = pathlib.Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ExtractionInvalid(
            f"--row-identity {path} hashes to {actual}, but "
            f"--row-identity-sha256 pins {expected_sha256}")
    return json.loads(payload.decode("utf-8"))


def sealed_pools_for(expect: "Mapping | None"):
    """CIFAR pools reached through the approved record, if one is available.

    This is the path §319.2 asked for: the originals are found by walking the
    record's pinned seal, so no caller-supplied source file is involved. When
    it is available the sidecar is not needed at all -- the ids and labels are
    derived here.
    """
    source = (expect or {}).get("source") or {}
    record_path = source.get("record_path")
    if not record_path:
        return None
    payload = pathlib.Path(record_path).read_bytes()
    declared = source.get("record_sha256")
    # Re-reading the record without re-checking its digest let a record whose
    # input_authority had been swapped for a different valid seal be consumed,
    # even though the expectation already knew what that record must hash to.
    if declared:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != declared:
            raise ExtractionInvalid(
                f"{record_path} hashes to {actual}, the approved expectation "
                f"declares {declared}")
    record = json.loads(payload.decode("utf-8"))
    return cifar_pools_from_sealed(sealed_cifar_source_from_record(record))


def load_split(result_dir: str, train_manifest: str | None, *,
               dataset: str, row_identity: "Mapping | None" = None,
               expect: "Mapping | None" = None) -> dict:
    """Return the train-side and test-side code/label arrays.

    Two ways to get the train rows:
      * `extract_train.npz` present -> use it directly (MSCOCO, whose train
        split is disjoint from its DB);
      * otherwise slice them out of the DB extraction by image basename using
        `train_manifest` (Flickr25k, NUS-WIDE, where train is a DB subset).
    """
    tr_path = os.path.join(result_dir, "extract_train.npz")
    has_train = os.path.exists(tr_path)
    admitted = admit_run(
        result_dir, dataset=dataset, expect=expect,
        required_splits=("db", "query", "train") if has_train else ("db", "query"))

    def _sha(split):
        return admitted.splits[split].get("npz_sha256", "")

    db = load_verified_npz(os.path.join(result_dir, "extract_db.npz"),
                           _sha("db"), what="db")
    qy = load_verified_npz(os.path.join(result_dir, "extract_query.npz"),
                           _sha("query"), what="query")
    tr_npz = (load_verified_npz(tr_path, _sha("train"), what="train")
              if has_train else None)

    # Only CIFAR takes the seal derivation. Calling it unconditionally made
    # Flickr/MS-COCO/NUS-WIDE fail with "missing originals: cifar_train_batch_1",
    # since their seals name different files entirely.
    _ROW_IDENTITY_SOURCE.clear()
    pools = (sealed_pools_for(expect)
             if canonical_dataset_name(dataset, what="--dataset") == "CIFAR10"
             else None)
    lab_key = "multi_hot_labels" if "multi_hot_labels" in db.files else "labels"
    qy_b = _row_ids(qy, "query", row_identity, _sha("query"), pools=pools)

    if tr_npz is not None:
        tr_b = _row_ids(tr_npz, "train", row_identity, _sha("train"), pools=pools)
        src, tr_idx = tr_npz, np.arange(len(tr_b))
        missing, mode = 0, "extract_train.npz"
    else:
        if not train_manifest:
            raise SystemExit("need --train_manifest when extract_train.npz is absent")
        db_b = _row_ids(db, "db", row_identity, _sha("db"), pools=pools)
        want = {os.path.basename(l.split()[0]) for l in open(train_manifest) if l.strip()}
        pos = {b: i for i, b in enumerate(db_b)}
        tr_idx = np.array([pos[b] for b in db_b if b in want and b in pos])
        src, tr_b = db, db_b[tr_idx]
        missing, mode = len(want) - len(tr_idx), "sliced from extract_db.npz"

    # leakage guards -- fail loudly rather than silently producing a nice number
    overlap_qt = len(set(qy_b) & set(tr_b))
    overlap_qd = len(set(qy_b) & set(_row_ids(db, "db", row_identity, _sha("db"), pools=pools)))
    if overlap_qt or overlap_qd:
        raise RuntimeError(f"split leak: query∩train={overlap_qt}, query∩db={overlap_qd}")

    def _as_multihot(arr, n_cls=None):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            n_cls = n_cls or int(arr.max()) + 1
            return np.eye(n_cls, dtype=np.int64)[arr], n_cls
        return arr.astype(np.int64), arr.shape[1]

    tr_lab, n_cls = _as_multihot(src[lab_key][tr_idx])
    te_lab, _ = _as_multihot(qy[lab_key], n_cls)

    return {
        "train_labels": tr_lab, "test_labels": te_lab,
        "train_idx": tr_idx, "train_src": src, "train_basenames": tr_b,
        "db": db, "qy": qy,
        "n_train_missing": int(missing), "train_mode": mode,
        "admitted": admitted, "row_identity_source": dict(_ROW_IDENTITY_SOURCE),
        "query_ids": qy_b,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir", required=True)
    ap.add_argument("--baseline_dirs", nargs="*", default=[],
                    help="dirs with extract_{db,query}.npz for flat-hash chunk controls")
    ap.add_argument("--baseline_names", nargs="*", default=[])
    ap.add_argument("--train_manifest", default=None,
                    help="only needed when the run has no extract_train.npz")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--min_support", type=int, default=DEFAULT_MIN_SUPPORT)
    ap.add_argument("--n_shuffle", type=int, default=DEFAULT_N_SHUFFLE)
    ap.add_argument("--n_boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=42,
                    help="Shuffle/bootstrap RNG only. NOT the training seed -- "
                         "that comes from the admitted extraction manifests.")
    ap.add_argument("--train-derivative", dest="train_derivative",
                    action="append", default=None,
                    help="NAME=PATH of an approved MSCOCO train-derivative "
                         "manifest; the digest must be in the approved "
                         "registry, so a path alone grants nothing")
    ap.add_argument("--row-identity", dest="row_identity", default=None,
                    help="JSON declaring per-split row ids for sealed "
                         "extractions that carry no image_paths (CIFAR-10). "
                         "Each entry must name its `derived_from` source; ids "
                         "are never synthesised here.")
    ap.add_argument("--row-identity-sha256", dest="row_identity_sha256",
                    default=None,
                    help="External digest of the --row-identity file. Required "
                         "with it: the sidecar cannot certify itself.")
    ap.add_argument("--approved-aggregate", dest="approved_aggregate",
                    default=None,
                    help="Approved Phase-3 aggregate to take the expected "
                         "identity from. The paper path.")
    ap.add_argument("--approved-baseline-aggregate",
                    dest="approved_baseline_aggregate", default=None,
                    help="Approved Phase-4 aggregate the baseline controls are "
                         "selected from. Required with --baseline_dirs on a "
                         "paper path.")
    ap.add_argument("--approved-aggregate-sha256",
                    dest="approved_aggregate_sha256", default=None)
    ap.add_argument("--expect-seed", dest="expect_seed", type=int, default=None,
                    help="Which seed's cell this run must be.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace an existing --out. Off by default.")
    ap.add_argument("--internal-consistency-only", action="store_true",
                    dest="internal_consistency_only",
                    help="Skip the approved-record expectation. For diagnostics "
                         "only; the result is not a paper path.")
    ap.add_argument("--expect-identity", dest="expect_identity", default=None,
                    help="JSON of the identity this cell must have, taken from "
                         "the approved aggregate/record. The extraction "
                         "manifests carry no recipe or campaign, so internal "
                         "consistency alone cannot show this is the main cell.")
    ap.add_argument("--bio_project", action="store_true",
                    help="project every 18-base DNA code to bio-valid (GC window "
                         "by code length, homopolymer <= 3) before codon "
                         "extraction, matching the deployed valid-DNA codes. "
                         "Baseline chunk control becomes the projected per-slot "
                         "codon of the baseline's own DNA code.")
    # F07: default None -> resolved from the actual code length by the
    # central policy. The old 0.4444/0.5556 gave L=15 the window [7,8]
    # while the main path projected onto [6,9].
    ap.add_argument("--gc_min_frac", type=float, default=None)
    ap.add_argument("--gc_max_frac", type=float, default=None)
    ap.add_argument("--max_run", type=int, default=3)
    args = ap.parse_args()

    if len(args.baseline_dirs) != len(args.baseline_names):
        raise SystemExit(
            f"--baseline_dirs has {len(args.baseline_dirs)} entries and "
            f"--baseline_names {len(args.baseline_names)}; zip() would drop "
            f"the excess silently and publish a table missing a control")
    if len(set(args.baseline_names)) != len(args.baseline_names):
        raise SystemExit(
            f"--baseline_names repeats: {args.baseline_names}; a later control "
            f"would overwrite an earlier one in the results dict")
    out_resolved = pathlib.Path(args.out).resolve()
    # Block-list first, so a producer/sealed path is refused with that reason
    # even when the allow-list would also catch it.
    for protected in PROTECTED_ROOTS:
        root = pathlib.Path(protected)
        try:
            root = root.resolve()
        except OSError:
            continue
        # No `exists()` skip: a root that is absent right now is not a licence
        # to write where it will be.
        if out_resolved == root or root in out_resolved.parents:
            raise SystemExit(
                f"--out {out_resolved} is inside {root}; a derivative analysis "
                f"is never written into a producer or sealed tree, and "
                f"--overwrite does not lift this")
    allowed = pathlib.Path(DERIVATIVE_OUTPUT_ROOT).resolve()
    if allowed not in out_resolved.parents:
        raise SystemExit(
            f"--out {out_resolved} is outside the derivative output root "
            f"{allowed}. An allow-list is the boundary: a block-list of trees "
            f"someone remembered cannot cover the ones they did not")
    if out_resolved.suffix != ".json":
        raise SystemExit(f"--out must be a .json analysis, got {out_resolved.name}")
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(
            f"{args.out} exists; pass --overwrite to replace it. A run that "
            f"silently replaces a published analysis leaves no trace of what "
            f"it removed")
    # Pin WHICH directory was approved, by inode and by current path. The
    # publication below re-opens it and refuses if either has changed: a
    # decoding run takes minutes, and `os.replace(temp, args.out)` walks that
    # path again at the end, so a directory moved aside and replaced by a
    # symlink after this point used to redirect the write out of the allow-list
    # entirely -- storing atomically, but somewhere else.
    _out_parent = out_resolved.parent
    # Only ever inside the allow-listed root, which was checked above.
    os.makedirs(str(_out_parent), exist_ok=True)
    # The boundary is fixed HERE, once, and every later step is judged against
    # this value. Re-resolving DERIVATIVE_OUTPUT_ROOT at publication time let a
    # root that had been moved aside and aliased answer the question about
    # itself: the new target resolved to the new root, so it was "inside".
    canonical_root = allowed
    try:
        _pfd = os.open(str(_out_parent), os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise SystemExit(f"--out parent {_out_parent} cannot be opened ({exc})")
    try:
        approved_out_dir = _fd_state(_pfd)
    finally:
        os.close(_pfd)
    # Opening a name is itself a fresh lookup, so the descriptor is not yet
    # known to be the directory that was validated. If the name became a
    # symlink between the allow-list check and this open, the descriptor points
    # elsewhere -- and approving it would make the target its own boundary,
    # with every later comparison agreeing on the wrong directory.
    _opened = pathlib.Path(approved_out_dir[0])
    if _opened != _out_parent:
        raise SystemExit(
            f"--out parent {_out_parent} opened as {_opened}; it is no longer "
            f"the approved directory and publication would leave the allow-list")
    # A backstop, and unreachable while the check above stands: the allow-list
    # already put `_out_parent` inside `canonical_root`, so an opened directory
    # equal to it is inside too. Kept because it is what makes the boundary
    # explicit at the point the descriptor is approved -- but it is NOT the
    # guard that stops an alias, and no test can distinguish it from the one
    # above. The live boundary comparison is the one at publication.
    if canonical_root != _opened and canonical_root not in _opened.parents:
        raise SystemExit(
            f"the opened output directory {_opened} is outside the derivative "
            f"output root {canonical_root}")

    expect = None
    if args.approved_aggregate:
        if not args.approved_aggregate_sha256 or args.expect_seed is None:
            raise SystemExit("--approved-aggregate needs "
                             "--approved-aggregate-sha256 and --expect-seed")
        if args.approved_aggregate_sha256 != APPROVED_AGGREGATE_SHA256:
            raise SystemExit(
                f"--approved-aggregate-sha256 {args.approved_aggregate_sha256} "
                f"is not the approved aggregate {APPROVED_AGGREGATE_SHA256}; a "
                f"digest the caller chose does not make a paper path")
        expect = expected_identity_from_record(
            args.approved_aggregate, args.approved_aggregate_sha256,
            args.dataset, args.expect_seed)
    elif args.expect_identity:
        with open(args.expect_identity) as handle:
            expect = json.load(handle)
        if not isinstance(expect, dict) or not expect:
            raise SystemExit(
                f"--expect-identity {args.expect_identity} is empty; a file "
                f"that exists is not an approved identity")
        if not args.internal_consistency_only:
            raise SystemExit(
                "--expect-identity is a diagnostic input. For a paper path use "
                "--approved-aggregate/--approved-aggregate-sha256/--expect-seed "
                "so the expectation is built from approved bytes.")
    elif not args.internal_consistency_only:
        raise SystemExit(
            "--expect-identity is required: the extraction manifests carry no "
            "recipe or campaign, so a self-consistent run from ANY campaign "
            "passes admission. Pass the identity taken from the approved "
            "record, or --internal-consistency-only for a non-paper check.")

    row_identity = None
    if args.row_identity:
        # The sidecar's digest is owned by the approved expectation, not by a
        # free CLI flag: `--row-identity-sha256` alone lets the same caller
        # supply both the file and the digest that blesses it.
        pinned = (expect or {}).get("row_identity_sha256")
        if pinned is None:
            if not args.internal_consistency_only:
                raise SystemExit(
                    "--row-identity needs `row_identity_sha256` inside "
                    "--expect-identity; a digest passed beside the file it "
                    "certifies proves nothing about where the ids came from")
            pinned = args.row_identity_sha256
        elif args.row_identity_sha256 and args.row_identity_sha256 != pinned:
            raise SystemExit(
                f"--row-identity-sha256 {args.row_identity_sha256} disagrees "
                f"with the approved expectation {pinned}")
        if not pinned:
            raise SystemExit("--row-identity has no digest to pin it")
        row_identity = load_row_identity(args.row_identity, pinned)
    d = load_split(args.ours_dir, args.train_manifest, dataset=args.dataset,
                   row_identity=row_identity, expect=expect)

    # Slot count from the ADMITTED arrays, before anything reads N_SLOTS.
    # `codebook_indices` is [N, M], one entry per slot, so it is authoritative;
    # the base count alone is ambiguous (15 bases could be 5x3 or 3x5).
    # It used to be a separate `np.load` of extract_query.npz, which reopened a
    # file the admission had already verified.
    admitted = d["admitted"]
    paper_path = bool(args.approved_aggregate) and not args.internal_consistency_only
    if args.baseline_dirs and paper_path and not args.approved_baseline_aggregate:
        raise SystemExit(
            "--baseline_dirs on a paper path needs --approved-baseline-aggregate: "
            "a directory named on the command line is not an approved control")
    baseline_cells: dict = {}
    _probe = d["qy"]
    if "codebook_indices" in _probe:
        _set_n_slots(int(_probe["codebook_indices"].shape[1]))
        print(f"[{args.dataset}] slots inferred from extraction: {N_SLOTS}")
    check_paper_geometry(d["admitted"], _probe)
    approved_binding = check_record_binding(d["admitted"], args, expect)
    run_identity_binding = check_run_identity_binding(d["admitted"], args, expect)
    applied_policy = check_bio_policy(args, expect)
    tr_lab, te_lab = d["train_labels"], d["test_labels"]
    tr_idx, src, db, qy = d["train_idx"], d["train_src"], d["db"], d["qy"]
    n_bases = db["base_indices"].shape[1] // N_SLOTS

    print(f"[{args.dataset}] train={len(tr_lab)} ({d['train_mode']}) "
          f"test={len(te_lab)} labels={tr_lab.shape[1]} bases/slot={n_bases} "
          f"(train rows not found: {d['n_train_missing']})")

    # bio-projection (2026-07-21 invariant): project the 18-base DNA sequence to
    # bio-valid before codon extraction, so decoding uses the deployed valid-DNA
    # codes. codebook_indices are NOT touched, so ours_codeword is invariant.
    def _bioproj(base_arr):
        if not args.bio_project:
            return base_arr
        from dna_utils.bio_constraints import project_to_valid, is_valid_batch
        from dna_utils.gc_policy import resolve_gc_policy
        arr = np.ascontiguousarray(base_arr).astype(np.int8)
        # F07: resolve the window from the ACTUAL code length. The old default
        # 0.4444/0.5556 projected a 15-base code onto [7,8] while the main path
        # used [6,9], so decoding was scored on a different feasible set than
        # the retrieval numbers it was compared against.
        _pol = resolve_gc_policy(int(arr.shape[1]))
        _gmin = args.gc_min_frac if args.gc_min_frac is not None else _pol.gc_min_frac
        _gmax = args.gc_max_frac if args.gc_max_frac is not None else _pol.gc_max_frac
        _mrun = args.max_run if args.max_run is not None else _pol.max_run
        uniq, inv = np.unique(arr, axis=0, return_inverse=True)
        valid = is_valid_batch(uniq, _gmin, _gmax, _mrun)
        out = uniq.copy()
        for i in np.where(~valid)[0]:
            out[i], _ = project_to_valid(uniq[i], _gmin, _gmax, _mrun)
        return out[inv].astype(np.int64)

    tag_bp = " [bio-projected]" if args.bio_project else ""
    tr_base = _bioproj(src["base_indices"][tr_idx])
    qy_base = _bioproj(qy["base_indices"])

    # The DECLARED cardinality, not `max(observed) + 1`. An unobserved high
    # codeword does not shrink the codebook: with admitted codebook_size 64 and
    # a small fixture the inferred K came out as 1, and every dictionary was
    # built over a single unit.
    declared_k = admitted.common.get("codebook_size")
    if not isinstance(declared_k, int) or declared_k <= 0:
        raise ExtractionInvalid(
            f"admitted run declares codebook_size {declared_k!r}; the codeword "
            f"cardinality cannot be inferred from the rows that happen to appear")
    K = declared_k
    observed = int(max(src["codebook_indices"].max(), qy["codebook_indices"].max()))
    if observed >= K:
        raise ExtractionInvalid(
            f"codeword id {observed} is outside the declared codebook of size {K}")
    tr_cw = src["codebook_indices"][tr_idx]
    units = {
        "ours_codeword": (tr_cw, qy["codebook_indices"], K),
        "ours_codon": (codon_ids(tr_base, n_bases),
                       codon_ids(qy_base, n_bases), 4 ** n_bases),
    }
    print(f"[{args.dataset}]{tag_bp} decoding units built")

    results: Dict[str, dict] = {}
    for name, (tu, qu, n_u) in units.items():
        results[name] = decode_all_slots(tu, tr_lab, qu, te_lab, n_u,
                                         args.alpha, args.min_support)
        print(f"  {name:16s} mAP={results[name]['slot_mean']['concept_mAP']:.4f} "
              f"cov={results[name]['slot_mean']['coverage']:.3f}")

    train_derivatives = {}
    for entry in (args.train_derivative or ()):
        name, _, where = str(entry).partition("=")
        if not name or not where:
            raise SystemExit(
                f"--train-derivative {entry!r} must be NAME=PATH")
        if name in train_derivatives:
            raise SystemExit(
                f"--train-derivative names {name} twice")
        train_derivatives[name] = where
    unknown = sorted(set(train_derivatives) - set(args.baseline_names or ()))
    if unknown:
        raise SystemExit(
            f"--train-derivative names {unknown}, which are not baselines of "
            f"this run ({list(args.baseline_names or ())})")

    # flat-hash chunk controls
    for bdir, bname in zip(args.baseline_dirs, args.baseline_names):
        if paper_path:
            # The pairing comes from the approved Phase-4 aggregate, not from a
            # directory that happens to be named on the command line.
            cell = approved_baseline_cell(
                args.approved_baseline_aggregate,
                APPROVED_BASELINE_AGGREGATE_SHA256, variant=bname,
                dataset=args.dataset,
                seed=int(admitted.common.get("random_seed")),
                bit=PAPER_GEOMETRY["total_bits"])
            baseline_cells[bname] = cell
            bdb = load_admitted_baseline(cell, "db")
            bqy = load_admitted_baseline(cell, "query")
            if canonical_dataset_name(args.dataset, what="--dataset") == "MSCOCO":
                # MSCOCO's train split is disjoint from its DB, so train rows
                # cannot be selected out of the DB the way the other three do.
                # The only thing that closes it is a derivative produced from
                # THIS cell's checkpoint and approved as such; reading an
                # arbitrary extract_train.npz would substitute an unapproved
                # artefact for one.
                declared = train_derivatives.get(bname)
                if not declared:
                    raise ExtractionInvalid(
                        f"baseline {bname}: MSCOCO needs an approved train "
                        f"derivative bound to {cell['key']}; none is declared, "
                        f"and its DB does not contain the train rows")
                bdb, provenance = load_approved_train_derivative(
                    cell, declared, variant=bname, dataset=args.dataset,
                    seed=int(admitted.common.get("random_seed")))
                # `bdb` is the DB position only in the sense that our train
                # rows are looked up in it; for MSCOCO those rows live here.
                baseline_cells[bname]["train_derivative"] = provenance
        else:
            # Diagnostic, directory-only. Never a paper path.
            if canonical_dataset_name(args.dataset, what="--dataset") == "MSCOCO":
                # The same substitution, one route over: a file appearing in a
                # directory is not an approved derivative, and a diagnostic
                # number computed against one reads exactly like a paper one.
                raise ExtractionInvalid(
                    f"baseline {bname}: MSCOCO train rows come only from an "
                    f"approved derivative bound to a Phase-4 cell; a "
                    f"directory-only input cannot supply them, whatever it "
                    f"contains")
            btr_path = os.path.join(bdir, "extract_train.npz")
            bsrc_path = btr_path if os.path.exists(btr_path) \
                else os.path.join(bdir, "extract_db.npz")
            bdb = np.load(bsrc_path, allow_pickle=True)
            bqy = np.load(os.path.join(bdir, "extract_query.npz"),
                          allow_pickle=True)
        # The control must share the panel. `bit_chunk_ids` only refuses widths
        # that do not divide by the slot count, so 35- and 40-bit baselines
        # produced 7- and 8-bit chunks against a 30-bit / 5 x 6-bit model.
        for name, arr in (("db", bdb), ("query", bqy)):
            width = int(np.asarray(arr["hash_2bit"]).shape[1])
            if width != PAPER_GEOMETRY["total_bits"]:
                raise ExtractionInvalid(
                    f"baseline {bname} {name} is {width}-bit; this control is "
                    f"for {PAPER_GEOMETRY['total_bits']}-bit and a different "
                    f"budget is not a comparison")
        # Realign to OUR rows by identity. `d["query_ids"]` is what load_split
        # actually used -- for CIFAR that is the seal-derived order, so the old
        # `qy["image_paths"]` assumption crashed there. Duplicates are refused
        # before a dict can collapse them into a shorter, wrong correspondence.
        bpos = _index_by_identity(bdb, f"baseline {bname} db")
        qpos = _index_by_identity(bqy, f"baseline {bname} query")
        tr_b = _paired_rows(bpos, d["train_basenames"], f"baseline {bname} train")
        te_b = _paired_rows(qpos, d["query_ids"], f"baseline {bname} query",
                            exact=True)
        _check_paired_labels(bdb, tr_b, tr_lab, f"baseline {bname} train")
        _check_paired_labels(bqy, te_b, te_lab, f"baseline {bname} query")
        if bname in baseline_cells:
            baseline_cells[bname]["alignment_ids_sha256"] = {
                "train": hashlib.sha256(
                    "\n".join(map(str, d["train_basenames"])).encode()).hexdigest(),
                "query": hashlib.sha256(
                    "\n".join(map(str, d["query_ids"])).encode()).hexdigest()}
        key = f"{bname}_chunk"
        if args.bio_project:
            # DNA-space-consistent control: the baseline's own sign code -> one
            # base per 2 bits -> project to bio-valid -> per-slot 3-base codon,
            # decoded exactly like ours. (Same 64 values/slot as the bit chunk.)
            #
            # The base count comes from the baseline's own width, NOT a constant:
            # it was 18 while the budget was 36 bits, and is 15 at 30 bits. A
            # baseline at a different budget from ours cannot be chunked into
            # N_SLOTS codons at all, so say which pair mismatched rather than
            # letting reshape fail with a bare size error.
            _n_base = bdb["hash_2bit"].shape[1] // 2
            if _n_base != N_SLOTS * n_bases:
                raise SystemExit(
                    f"baseline '{bname}' is {bdb['hash_2bit'].shape[1]}-bit "
                    f"({_n_base} bases) but ours is {N_SLOTS} slots x {n_bases} "
                    f"bases = {N_SLOTS * n_bases}; pass control dirs at the "
                    f"matching budget")

            def _hash_to_base(h2, _nb=_n_base):
                b = (h2 > 0).astype(np.int64).reshape(len(h2), _nb, 2)
                return b[:, :, 0] * 2 + b[:, :, 1]
            btr_codon = codon_ids(_bioproj(_hash_to_base(bdb["hash_2bit"][tr_b])), n_bases)
            bte_codon = codon_ids(_bioproj(_hash_to_base(bqy["hash_2bit"][te_b])), n_bases)
            results[key] = decode_all_slots(btr_codon, tr_lab, bte_codon, te_lab,
                                            4 ** n_bases, args.alpha, args.min_support)
        else:
            results[key] = decode_all_slots(
                bit_chunk_ids(bdb["hash_2bit"][tr_b]), tr_lab,
                bit_chunk_ids(bqy["hash_2bit"][te_b]), te_lab,
                2 ** (bdb["hash_2bit"].shape[1] // N_SLOTS), args.alpha, args.min_support)
        print(f"  {key:16s} mAP={results[key]['slot_mean']['concept_mAP']:.4f} "
              f"cov={results[key]['slot_mean']['coverage']:.3f}")

    results["majority"] = majority_control(tr_lab, te_lab, args.alpha)
    print(f"  {'majority':16s} mAP={results['majority']['concept_mAP']:.4f}")

    results["shuffled"] = shuffled_control(
        units["ours_codon"][0], tr_lab, units["ours_codon"][1], te_lab,
        units["ours_codon"][2], args.alpha, args.min_support,
        args.n_shuffle, args.seed)
    print(f"  {'shuffled':16s} mAP={results['shuffled']['concept_mAP']:.4f} "
          f"+-{results['shuffled']['concept_mAP_std']:.4f}")

    # §2.8 collision accounting
    results["collision"] = {
        "decoding_loss_codeword_minus_codon":
            results["ours_codeword"]["slot_mean"]["concept_mAP"]
            - results["ours_codon"]["slot_mean"]["concept_mAP"],
        "entropy_increase_codon_minus_codeword":
            (results["ours_codon"]["slot_mean"]["H_cond"]
             - results["ours_codeword"]["slot_mean"]["H_cond"])
            if None not in (results["ours_codon"]["slot_mean"]["H_cond"],
                            results["ours_codeword"]["slot_mean"]["H_cond"])
            else None,
        "per_slot_jsd": [
            codeword_codon_jsd(tr_cw[:, m],
                               units["ours_codon"][0][:, m], tr_lab, K,
                               args.alpha, args.min_support)
            for m in range(N_SLOTS)
        ],
    }

    # §2.11 paired bootstrap of ours-codon against every control
    ref = results["ours_codon"]["_ap_per_sample"]
    results["bootstrap_vs_ours_codon"] = {
        name: paired_bootstrap(ref, results[name]["_ap_per_sample"],
                               args.n_boot, args.seed)
        for name in results
        if name != "ours_codon" and "_ap_per_sample" in results[name]
    }

    payload = {
        "dataset": args.dataset,
        # Stated because the name does not fix the definition: two different
        # quantities are called "concept mAP" in this literature.
        "metric_definitions": {
            "concept_mAP": (
                "mean over test QUERIES of the average precision of that "
                "query's ranking OVER LABEL COLUMNS (stable tie order). It is "
                "not the mean over classes of an image-ranking AP."),
            "concept_mAP_query_exclusion": (
                "a query with no positive label has undefined AP and is "
                "excluded from the mean; n_valid_queries counts those kept"),
            "coverage_denominator": (
                "all test queries, including the ones excluded from the AP "
                "mean; it is not the same denominator as concept_mAP"),
            "H_cond": "support-weighted mean per-label Bernoulli entropy, nats",
        },
        # What produced these codes. Without this the analysis names a
        # directory and nothing else -- and a directory is not an identity.
        "input_admission": {
            "run_dir": admitted.run_dir,
            "completion_marker_sha256": admitted.completion_marker_sha256,
            "manifest_sha256": dict(admitted.manifest_sha256),
            "npz_sha256": {k: v.get("npz_sha256")
                           for k, v in admitted.splits.items()},
            "n_rows": {k: v.get("n_rows") for k, v in admitted.splits.items()},
            "checkpoint_sha256": admitted.common.get("checkpoint_sha256"),
            "config_sha256": admitted.common.get("config_sha256"),
            "inference_epoch": admitted.common.get("inference_epoch"),
            "inference_epoch_source": admitted.common.get("inference_epoch_source"),
            "training_seed": admitted.common.get("random_seed"),
            "training_stop_epoch": admitted.common.get("training_stop_epoch"),
            "geometry": {k: admitted.common.get(k) for k in
                         ("num_slots", "bases_per_slot", "total_bases",
                          "total_bits", "codebook_size")},
            "backfilled": admitted.backfilled,
            # What was actually used, reported by load_split -- not inferred
            # from whether a flag happened to be passed. A CIFAR run whose ids
            # came from the sealed originals said "npz image_paths".
            "row_identity_source": d.get("row_identity_source"),
            "applied_analysis_policy": applied_policy,
            "approved_record_binding": approved_binding or None,
            "approved_run_identity": run_identity_binding or None,
            "approved_analysis_protocol": (
                (expect or {}).get("approved_analysis_protocol")),
            "row_identity_sha256": args.row_identity_sha256,
            "expected_identity_source": (
                {"approved_aggregate": args.approved_aggregate,
                 "sha256": args.approved_aggregate_sha256,
                 "seed": args.expect_seed}
                if args.approved_aggregate else args.expect_identity),
            # A diagnostic run is never a paper path, even when the approved
            # aggregate is also supplied.
            "paper_eligible": bool(args.approved_aggregate)
                              and not args.internal_consistency_only,
            "diagnostic_only": bool(args.internal_consistency_only),
        },
        "config": {
            "alpha": args.alpha, "min_support": args.min_support,
            "n_shuffle": args.n_shuffle, "n_boot": args.n_boot,
            # Named so it cannot be read as provenance: the training seed is in
            # `input_admission.training_seed`.
            "shuffle_bootstrap_seed": args.seed,
            "n_train": int(len(tr_lab)), "n_test": int(len(te_lab)),
            "n_labels": int(tr_lab.shape[1]), "K": K, "n_bases_per_slot": n_bases,
            "ours_dir": args.ours_dir,
            # What was ASKED for. On a paper path the approved manifest's
            # extraction_dir is what is actually read, so recording the CLI
            # value as the source described a directory nobody opened.
            "baselines_requested": dict(zip(args.baseline_names,
                                            args.baseline_dirs)),
            "baseline_cells": {
                k: {"key": v["key"],
                    "consumed_extraction_dir": v["extraction_dir"],
                    "manifest": v["manifest"],
                    "manifest_sha256": v["manifest_sha256"],
                    "protocol_manifest_sha256": v["protocol_manifest_sha256"],
                    "final_checkpoint_sha256": v["final_checkpoint_sha256"],
                    "final_epoch_zero_based": v["final_epoch_zero_based"],
                    "consumed_npz_sha256": {
                        name: v["artifact_sha256"].get(name)
                        for name in ("extract_db.npz", "extract_query.npz")},
                    "alignment_ids_sha256": v.get("alignment_ids_sha256"),
                    # Present only for MSCOCO, where the control's train rows
                    # come from an approved derivative rather than its DB.
                    "train_derivative": v.get("train_derivative")}
                for k, v in baseline_cells.items()},
            "baseline_aggregate": (
                {"path": args.approved_baseline_aggregate,
                 "sha256": APPROVED_BASELINE_AGGREGATE_SHA256}
                if baseline_cells else None),
        },
        "slot_names": SLOT_NAMES,
        "results": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                    for k, v in results.items()},
    }
    # Serialise FULLY first. `open(out, "w")` truncates before json.dump runs,
    # so a strict-serialisation failure used to leave an empty or half-written
    # file where a published analysis had been.
    text = json.dumps(payload, indent=2, allow_nan=False)
    # Re-open the parent and require it to be the directory that was approved.
    # Verifying a path and then writing to that path are two lookups; the write
    # is bound to the descriptor that was checked, so nothing between them can
    # move the target.
    try:
        dir_fd = os.open(str(_out_parent), os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise SystemExit(f"--out parent {_out_parent} disappeared during the "
                         f"run ({exc}); nothing was written")
    try:
        if _fd_state(dir_fd) != approved_out_dir:
            raise SystemExit(
                f"the output directory {_out_parent} moved or was replaced "
                f"during the run; publication refused")
        # The boundary fixed at the start of the run, never a fresh resolution.
        opened = pathlib.Path(_fd_state(dir_fd)[0])
        if canonical_root != opened and canonical_root not in opened.parents:
            raise SystemExit(
                f"the output directory now resolves to {opened}, outside the "
                f"derivative output root {canonical_root}; publication refused")
        final_name = os.path.basename(args.out)
        temp_name = f".f10-{os.getpid()}-{final_name}.tmp"
        handle = os.open(temp_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644,
                         dir_fd=dir_fd)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as f:
                f.write(text)
            if args.overwrite:
                os.replace(temp_name, final_name,
                           src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            else:
                try:
                    os.link(temp_name, final_name,
                            src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
                except FileExistsError:
                    raise SystemExit(
                        f"{args.out} was created during the run; pass "
                        f"--overwrite to replace it") from None
                finally:
                    _unlink_at(temp_name, dir_fd)
        except BaseException:
            _unlink_at(temp_name, dir_fd)
            raise
    finally:
        os.close(dir_fd)
    print(f"[{args.dataset}] wrote {args.out}")


if __name__ == "__main__":
    main()
