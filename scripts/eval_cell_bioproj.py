#!/usr/bin/env python
"""Post-hoc bio-projected mAP@R for one main-table cell.

The training scripts run their internal `-ev` final eval WITHOUT bio-projection
(evaluation() defaults bio_project=False when called from train_siglip2), so the
in-run JSON is the pre-projection number. The reported paper number is the
MANDATORY post-projection mAP@R (2026-07-21 invariant). This helper recomputes
it from the cell's extract_db.npz / extract_query.npz, applying the length-
dependent GC window:

F07/Phase 2: the window is NOT a free parameter. It comes from
`dna_utils.gc_policy` (40-60% inclusive), keyed by the code length read from the
extraction itself:

  15 bases -> count [6, 9]     18 bases -> count [8, 10]
  20 bases -> count [8, 12]    24 bases -> count [10, 14]

`--gc_min/--gc_max` are now optional overrides that are CHECKED against that
policy rather than trusted. The historical `--gc_min 0.416 --gc_max 0.584`,
carried over from the 24-base panel, yields count [7, 8] at 15 bases -- a
different, narrower feasible set than the policy's [6, 9], so results projected
under it are not comparable with results projected under the policy.

Emits a one-line [CELL-RESULT] marker (grep-able) and writes
evaluation_siglip2_base_bioproj.json into the result dir (evaluation() does the
write).
"""
import argparse
import hashlib
import json
import math
import os
import stat
import sys
from collections.abc import Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from dna_utils.bio_constraints import _resolve_gc_count_range  # noqa: E402
from dna_utils.gc_policy import resolve_gc_policy  # noqa: E402
from evaluation_siglip2 import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    _load_npz_bound,
    evaluation,
    resolve_map_at_r,
)


def _read_json_artifact(path: str) -> tuple[dict, dict]:
    """Read one canonical JSON inode and return its exact byte identity."""
    absolute = os.path.abspath(path)
    flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NOFOLLOW", 0))
    try:
        fd = os.open(absolute, flags)
    except OSError as error:
        raise RuntimeError(
            f"[eval_cell_bioproj] cannot open required JSON {absolute}: "
            f"{error}") from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(
                f"[eval_cell_bioproj] required JSON is not regular: {absolute}")
        chunks = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        current = os.stat(absolute, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(
            f"[eval_cell_bioproj] JSON changed while read: {absolute}: "
            f"{error}") from None
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns",
              "st_ctime_ns")
    if stat.S_ISLNK(current.st_mode) or any(
            getattr(before, key) != getattr(after, key)
            or getattr(after, key) != getattr(current, key)
            for key in fields):
        raise RuntimeError(
            f"[eval_cell_bioproj] JSON changed or was replaced while read: "
            f"{absolute}")
    raw = b"".join(chunks)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"[eval_cell_bioproj] invalid JSON {absolute}: {error}") from None
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"[eval_cell_bioproj] JSON is not an object: {absolute}")
    return payload, {
        "path": absolute,
        "resolved_path": os.path.realpath(absolute),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": int(after.st_size),
        "st_dev": int(after.st_dev),
        "st_ino": int(after.st_ino),
        "st_mtime_ns": int(after.st_mtime_ns),
        "st_ctime_ns": int(after.st_ctime_ns),
    }


def _validated_npz_artifacts(result_dir: str, *, required_splits,
                             binding: Mapping) -> tuple[int, dict]:
    """Reopen canonical validated NPZ bytes and prove their actual width."""
    root = os.path.abspath(result_dir)
    artifacts, widths = {}, {}
    for split in required_splits:
        canonical = os.path.join(root, f"extract_{split}.npz")
        arrays, artifact = _load_npz_bound(canonical)
        if artifact.get("path") != canonical \
                or artifact.get("resolved_path") != canonical:
            raise RuntimeError(
                f"[eval_cell_bioproj] {split} evaluator input is not the "
                f"canonical adjacent NPZ {canonical}")
        if artifact.get("sha256") != (binding.get("npz_sha256") or {}).get(split):
            raise RuntimeError(
                f"[eval_cell_bioproj] {split} NPZ bytes differ from the "
                "validated extraction binding")
        bases = np.asarray(arrays.get("base_indices"))
        if bases.ndim != 2 or bases.shape[0] <= 0 or bases.shape[1] <= 0:
            raise RuntimeError(
                f"[eval_cell_bioproj] {split} base_indices must be non-empty "
                f"rank-2, found {bases.shape}")
        widths[split] = int(bases.shape[1])
        artifacts[split] = artifact
    if len(set(widths.values())) != 1:
        raise RuntimeError(
            f"[eval_cell_bioproj] extraction splits disagree on actual "
            f"base_indices width: {widths}")
    return next(iter(widths.values())), artifacts


def _finite_proportion(payload: Mapping, key: str, *, where: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) \
            or not 0.0 <= float(value) <= 1.0:
        raise RuntimeError(
            f"[eval_cell_bioproj] {where}.{key} is not a finite proportion")
    return float(value)


def _check_evaluation_payload(payload: Mapping, *, where: str, dataset: str,
                              codebook_size: int, cutoff: int,
                              expected_inputs: Mapping,
                              bio_project: bool) -> None:
    exact = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "dataset": dataset,
        "distance_mode": "base",
        "codebook_size": codebook_size,
        "bio_project": bio_project,
        "mAP_R_cutoff": cutoff,
        "input_artifacts": dict(expected_inputs),
    }
    wrong = {key: (payload.get(key), want) for key, want in exact.items()
             if payload.get(key) != want}
    if wrong:
        raise RuntimeError(
            f"[eval_cell_bioproj] {where} protocol/input identity differs: "
            f"{wrong}")
    _finite_proportion(payload, "mAP", where=where)
    _finite_proportion(payload, "mAP_at_R", where=where)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="result dir with extract_{db,query}.npz")
    ap.add_argument("--dataset", required=True,
                    help="canonical name: Flickr25k / MSCOCO / NUSWIDE / CIFAR10")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--gc_min", type=float, default=None,
                    help="Optional; checked against the central GC policy.")
    ap.add_argument("--gc_max", type=float, default=None,
                    help="Optional; checked against the central GC policy.")
    ap.add_argument("--allow-backfilled", action="store_true",
                    help=("Admit retrospectively bound inputs. Off by default "
                          "so a paper path cannot pick them up silently."))
    ap.add_argument(
        "--require-train", action="store_true",
        help=("Require one completion transaction over db/query/train. This is "
              "mandatory for MAIN refits; two-split diagnostics leave it off."))
    ap.add_argument("--bases", type=int, default=None,
                    help="Code length; read from extract_db.npz when omitted.")
    args = ap.parse_args()
    required_splits = ("db", "query", "train") if args.require_train else (
        "db", "query")

    from dna_utils.extraction_validation import metric_input_binding
    binding_before = metric_input_binding(
        args.dir, required_splits=required_splits,
        allow_backfilled=args.allow_backfilled)
    if binding_before.get("dataset") != args.dataset \
            or binding_before.get("codebook_size") != args.K:
        raise SystemExit(
            "[eval_cell_bioproj] --dataset/--K do not equal the validated "
            f"extraction identity: requested {(args.dataset, args.K)!r}, "
            "extraction records "
            f"{(binding_before.get('dataset'), binding_before.get('codebook_size'))!r}")

    actual_bases, npz_artifacts_before = _validated_npz_artifacts(
        args.dir, required_splits=required_splits, binding=binding_before)
    if args.bases is not None and args.bases != actual_bases:
        raise SystemExit(
            f"[eval_cell_bioproj] --bases={args.bases} disagrees with the "
            f"actual extraction width {actual_bases}; GC policy cannot be "
            "selected from caller metadata")
    total_bases = actual_bases
    policy = resolve_gc_policy(total_bases)
    gc_min = 0.40 if args.gc_min is None else float(args.gc_min)
    gc_max = 0.60 if args.gc_max is None else float(args.gc_max)
    got = _resolve_gc_count_range(total_bases, gc_min, gc_max)
    if got != (policy.gc_min_count, policy.gc_max_count):
        raise SystemExit(
            f"[eval_cell_bioproj] gc=[{gc_min}, {gc_max}] gives count {got} at "
            f"{total_bases} bases, but the central policy "
            f"({policy.policy_version}) is "
            f"[{policy.gc_min_count}, {policy.gc_max_count}]. Projecting onto a "
            f"window nobody reviewed is how four conventions ended up in the "
            f"codebase; fix the caller rather than this check.")
    args.gc_min, args.gc_max = gc_min, gc_max

    R = resolve_map_at_r(args.dataset)
    if R is None:
        raise SystemExit(
            f"[eval_cell_bioproj] no canonical mAP@R cutoff for "
            f"{args.dataset!r}")
    evaluator_inputs_before = {
        split: npz_artifacts_before[split] for split in ("db", "query")}
    raw_path = os.path.join(
        os.path.abspath(args.dir), "evaluation_siglip2_base.json")
    raw_before, raw_artifact_before = _read_json_artifact(raw_path)
    _check_evaluation_payload(
        raw_before, where="evaluation_siglip2_base.json",
        dataset=args.dataset, codebook_size=args.K, cutoff=R,
        expected_inputs=evaluator_inputs_before, bio_project=False)

    # evaluation() atomically replaces the post-BIO JSON. Any prior completion
    # marker is stale from that instant onward, including if a later check
    # refuses this run, so remove it before asking the evaluator to write.
    from dna_utils.extraction_validation import ANALYSIS_MARKER_NAME
    stale = os.path.join(args.dir, ANALYSIS_MARKER_NAME)
    if os.path.exists(stale):
        os.unlink(stale)

    res = evaluation(
        args.dir,
        distance_mode="base",
        codebook_size=args.K,
        bio_project=True,
        bio_gc_min_frac=args.gc_min,
        bio_gc_max_frac=args.gc_max,
        bio_max_homopolymer_run=policy.max_run,
        map_at_r=R,
        dataset_name=args.dataset,
    )
    binding_after = metric_input_binding(
        args.dir, required_splits=required_splits,
        allow_backfilled=args.allow_backfilled)
    actual_bases_after, npz_artifacts_after = _validated_npz_artifacts(
        args.dir, required_splits=required_splits, binding=binding_after)
    if binding_after != binding_before \
            or actual_bases_after != actual_bases \
            or npz_artifacts_after != npz_artifacts_before:
        raise RuntimeError(
            "[eval_cell_bioproj] extraction changed while evaluation ran")
    evaluator_inputs = {
        split: npz_artifacts_after[split] for split in ("db", "query")}
    if res.get("input_artifacts") != evaluator_inputs:
        raise RuntimeError(
            "[eval_cell_bioproj] evaluator input_artifacts do not exactly "
            "match the validated canonical extraction")

    post_path = os.path.join(
        os.path.abspath(args.dir),
        "evaluation_siglip2_base_bioproj.json")
    post, post_artifact = _read_json_artifact(post_path)
    returned_json = json.loads(json.dumps(res, allow_nan=False))
    if post != returned_json:
        raise RuntimeError(
            "[eval_cell_bioproj] evaluator return value differs from the "
            "post-BIO JSON it published")
    _check_evaluation_payload(
        post, where="evaluation_siglip2_base_bioproj.json",
        dataset=args.dataset, codebook_size=args.K, cutoff=R,
        expected_inputs=evaluator_inputs, bio_project=True)
    raw, raw_artifact = _read_json_artifact(raw_path)
    if raw != raw_before or raw_artifact != raw_artifact_before:
        raise RuntimeError(
            "[eval_cell_bioproj] raw evaluation changed while post-BIO "
            "evaluation ran")

    bio_stats = post.get("bio_stats")
    if not isinstance(bio_stats, Mapping):
        raise RuntimeError(
            "[eval_cell_bioproj] post-BIO evaluation has no bio_stats object")
    exact_pre = {
        "mAP_pre_projection": raw.get("mAP"),
        "mAP_at_R_pre_projection": raw.get("mAP_at_R"),
        "mAP_R_cutoff_pre_projection": raw.get("mAP_R_cutoff"),
    }
    wrong_pre = {key: (bio_stats.get(key), want)
                 for key, want in exact_pre.items()
                 if bio_stats.get(key) != want}
    if wrong_pre:
        raise RuntimeError(
            "[eval_cell_bioproj] BIO statistics are not tied to the raw "
            f"evaluation: {wrong_pre}")
    for prefix in ("db", "qy"):
        if bio_stats.get(f"{prefix}_num_proj_failed") != 0 \
                or bio_stats.get(f"{prefix}_post_compliance") != 1.0:
            raise RuntimeError(
                f"[eval_cell_bioproj] {prefix} BIO projection was not fully "
                "successful")

    binding = binding_after
    map_r = post.get("mAP_at_R")
    pre = bio_stats.get("mAP_pre_projection")
    print(
        f"[CELL-RESULT] dataset={args.dataset} K={args.K} "
        f"dir={os.path.basename(os.path.normpath(args.dir))} "
        f"mAP@R_bioproj={map_r if map_r is None else round(float(map_r), 4)} "
        f"full_mAP_bioproj={round(float(post['mAP']), 4)} "
        f"full_mAP_pre={None if pre is None else round(float(pre), 4)} "
        f"DNAunique_DB={round(float(post.get('unique_code_ratio', -1)), 4)} "
        f"gc=[{args.gc_min},{args.gc_max}]"
    )

    # This stage produces HALF the analysis: the retrieval and DNA numbers, not
    # the pairwise NMI. It therefore writes its own partial, bound output and
    # seals nothing. It used to write `analysis_complete.json` right here, so a
    # marker named complete existed with `mean_off_diag_nmi` simply absent, and
    # the aggregator read the missing key with `.get()` and still counted the
    # cell as paired (§20.4). `scripts/seal_cell_analysis.py` publishes the
    # marker once both halves exist and agree on their inputs.
    payload = {
        "input_binding": binding,
        "evaluation_artifacts": {
            "raw": raw_artifact,
            "bio_projected": post_artifact,
        },
        "evaluator_input_artifacts": evaluator_inputs,
        "bio_stats": dict(bio_stats),
        "dataset": args.dataset, "K": args.K,
        "mAP_at_R_bioproj": map_r,
        "full_mAP_bioproj": post["mAP"],
        "full_mAP_pre_projection": pre,
        "DNA_unique_DB": post.get("unique_code_ratio"),
        "gc_min_frac": args.gc_min, "gc_max_frac": args.gc_max,
        "total_bases": total_bases,
        "gc_count_min_inclusive": policy.gc_min_count,
        "gc_count_max_inclusive": policy.gc_max_count,
        "bio_max_homopolymer_run": policy.max_run,
        "gc_policy_version": policy.policy_version,
        "map_r_cutoff": post.get("mAP_R_cutoff"),
    }
    # Atomic: a half-written cell_result that still parses is read as a result.
    out = os.path.join(args.dir, "cell_result.json")
    tmp = f"{out}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)

    if not os.path.exists(stale):
        print(f"[CELL-RESULT] no analysis seal is current; run "
              f"scripts/seal_cell_analysis.py once the NMI is done")


if __name__ == "__main__":
    main()
