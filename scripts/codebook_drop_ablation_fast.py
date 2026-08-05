"""Vectorised slot-axis drop diagnostic.

Dropping a slot means removing those coordinates from Hamming distance (the
implementation sets the same constant on query and DB).  The transformed rows
are therefore *not* emitted DNA strands.  With ``--bio_project``, the input
metric space is first projected to valid deployed DNA, which is mandatory for
paper-facing attribution; without it, output is an explicitly raw diagnostic.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ap_from_sorted_relevance(rel_sorted: np.ndarray) -> np.ndarray:
    """Vectorised AP over a [Nq, Ndb] sorted-relevance matrix."""
    rel = rel_sorted.astype(np.float32)
    n_rel = rel.sum(axis=1)
    cs = np.cumsum(rel, axis=1)
    pos_at_k = cs / np.arange(1, rel.shape[1] + 1, dtype=np.float32)[None, :]
    ap = (pos_at_k * rel).sum(axis=1) / np.maximum(n_rel, 1.0)
    ap[n_rel == 0] = 0.0
    return ap


def _base_dist(qbi: np.ndarray, dbbi: np.ndarray) -> np.ndarray:
    """Hamming distance over base indices [Nq, R] vs [Ndb, R] -> [Nq, Ndb]."""
    Nq, R = qbi.shape
    Ndb = dbbi.shape[0]
    out = np.zeros((Nq, Ndb), dtype=np.int32)
    for r in range(R):
        out += (qbi[:, r:r+1] != dbbi[None, :, r]).astype(np.int32)
    return out


def codebook_slices(num_bases: int, num_codebooks: int):
    """Return equal contiguous base-index slices for semantic codebooks."""
    if num_codebooks <= 0:
        raise ValueError(f"num_codebooks must be positive, got {num_codebooks}")
    if num_bases % num_codebooks != 0:
        raise ValueError(
            f"base-index width R={num_bases} is not divisible by "
            f"num_codebooks={num_codebooks}"
        )
    codons_per_codebook = num_bases // num_codebooks
    return [
        slice(m * codons_per_codebook, (m + 1) * codons_per_codebook)
        for m in range(num_codebooks)
    ]


def project_base_indices(
    codes: np.ndarray,
    gc_min_frac: float,
    gc_max_frac: float,
    max_run: int,
) -> np.ndarray:
    """Memoized, fail-closed projection of a base-index matrix."""
    from dna_utils.bio_constraints import is_valid_batch, project_to_valid

    arr = np.ascontiguousarray(codes).astype(np.int8)
    unique, inverse = np.unique(arr, axis=0, return_inverse=True)
    valid = is_valid_batch(unique, gc_min_frac, gc_max_frac, max_run)
    projected = unique.copy()
    for index in np.where(~valid)[0]:
        projected[index], cost = project_to_valid(
            unique[index], gc_min_frac, gc_max_frac, max_run,
        )
        if cost < 0:
            raise RuntimeError("no bio-valid projection exists for an input code")
    out = projected[inverse].astype(np.int64)
    if not bool(is_valid_batch(
        out, gc_min_frac, gc_max_frac, max_run,
    ).all()):
        raise RuntimeError("refusing to evaluate non-compliant projected DNA")
    return out


def compute_mAP_pk(q_bi, db_bi, q_lbl, db_lbl, p_at_k=(1, 10, 100, 1000), batch=200):
    Nq = q_bi.shape[0]
    aps = np.zeros(Nq, dtype=np.float32)
    pk_acc = {k: 0.0 for k in p_at_k}
    db_lbl_f = db_lbl.astype(np.int32)
    for s in range(0, Nq, batch):
        e = min(s + batch, Nq)
        qb = q_bi[s:e]
        d = _base_dist(qb, db_bi)                         # [b, Ndb]
        order = np.argsort(d, axis=1, kind="stable")      # [b, Ndb]
        rel = (db_lbl_f @ q_lbl[s:e].T).T > 0             # [b, Ndb]
        rel_sorted = np.take_along_axis(rel, order, axis=1)
        aps[s:e] = _ap_from_sorted_relevance(rel_sorted)
        for k in p_at_k:
            pk_acc[k] += rel_sorted[:, :k].mean(axis=1).sum()
    out = {"mAP": float(aps.mean())}
    for k in p_at_k:
        out[f"P@{k}"] = float(pk_acc[k] / Nq)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--subset_queries", type=int, default=0,
                    help="If >0, randomly subsample queries for speed.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--num_codebooks",
        type=int,
        default=6,
        help="Number of semantic codebooks. Each codebook drops R/M bases.",
    )
    ap.add_argument("--bio_project", action="store_true")
    ap.add_argument("--gc_min_frac", type=float, default=None)
    ap.add_argument("--gc_max_frac", type=float, default=None)
    ap.add_argument("--max_run", type=int, default=3)
    args = ap.parse_args()
    rd = args.result_dir

    q = np.load(os.path.join(rd, "extract_query.npz"), allow_pickle=True)
    db = np.load(os.path.join(rd, "extract_db.npz"), allow_pickle=True)
    q_bi = np.asarray(q["base_indices"]).copy()
    db_bi = np.asarray(db["base_indices"]).copy()
    gc_min_frac = args.gc_min_frac
    gc_max_frac = args.gc_max_frac
    if args.bio_project:
        if (gc_min_frac is None) != (gc_max_frac is None):
            raise ValueError(
                "--gc_min_frac and --gc_max_frac must be supplied together"
            )
        if gc_min_frac is None:
            if q_bi.shape[1] != 18:
                raise ValueError(
                    "non-18-base projection requires explicit --gc_min_frac "
                    "and --gc_max_frac"
                )
            gc_min_frac, gc_max_frac = 0.4444, 0.5556
        q_bi = project_base_indices(
            q_bi, gc_min_frac, gc_max_frac, args.max_run,
        )
        db_bi = project_base_indices(
            db_bi, gc_min_frac, gc_max_frac, args.max_run,
        )
    q_lbl = np.asarray(q["multi_hot_labels"]).astype(np.int32)
    db_lbl = np.asarray(db["multi_hot_labels"]).astype(np.int32)
    if args.subset_queries > 0 and args.subset_queries < q_bi.shape[0]:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(q_bi.shape[0], args.subset_queries, replace=False)
        q_bi = q_bi[idx]; q_lbl = q_lbl[idx]
        print(f"[fast_drop] subsampled queries -> {len(q_bi)}")
    Nq, R = q_bi.shape
    M = int(args.num_codebooks)
    slices = codebook_slices(R, M)
    L = R // M
    print(
        f"[fast_drop] R={R}, M={M}, L={L}, Nq={Nq}, Ndb={db_bi.shape[0]}",
        flush=True,
    )

    t0 = time.time()
    base = compute_mAP_pk(q_bi, db_bi, q_lbl, db_lbl)
    print(f"[fast_drop] baseline: mAP={base['mAP']:.4f}  P@1={base['P@1']:.4f}  "
          f"P@10={base['P@10']:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    results = {
        "baseline": base,
        "drops": {},
        "subset_queries": int(args.subset_queries),
        "num_codebooks": M,
        "codons_per_codebook": L,
        "input_code_space": (
            "bio_projected" if args.bio_project else "raw_unprojected"
        ),
        "drop_operator": "hamming_distance_axis_mask",
        "transformed_rows_are_emittable_dna": False,
        "bio_projection_applied": bool(args.bio_project),
        "paper_projection_compliant": False,
        "paper_result_eligible": False,
        "paper_eligibility_blockers": [
            "projection_protocol_manifest_not_bound"
        ],
        "bio_constraints": (
            {
                "gc_min_frac": gc_min_frac,
                "gc_max_frac": gc_max_frac,
                "max_run": args.max_run,
            }
            if args.bio_project else None
        ),
    }
    for m, codebook_slice in enumerate(slices):
        t1 = time.time()
        qd = q_bi.copy(); ddb = db_bi.copy()
        qd[:, codebook_slice] = 0
        ddb[:, codebook_slice] = 0
        r = compute_mAP_pk(qd, ddb, q_lbl, db_lbl)
        delta = r["mAP"] - base["mAP"]
        results["drops"][m] = {**r, "delta_mAP": delta}
        print(f"[fast_drop] drop cb{m}: mAP={r['mAP']:.4f}  delta={delta:+.4f}  "
              f"P@1={r['P@1']:.4f}  ({time.time()-t1:.1f}s)", flush=True)

    suffix = f"_subset{args.subset_queries}" if args.subset_queries > 0 else ""
    if args.bio_project:
        suffix += "_bioproj"
    out_p = os.path.join(rd, f"codebook_drop_ablation{suffix}.json")
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[fast_drop] wrote {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
