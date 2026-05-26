"""Compositional faithfulness evaluation for a trained GroundedDNA model.

Metrics produced (writes alongside the result dir):

B. codeword-caption semantic concentration
   For each (codebook m, codeword k), gather the DB samples that selected
   codeword k in codebook m, and measure how semantically coherent their
   per-part Qwen captions are. We use the cached SigLIP2 text_part feature
   for the same part m as a proxy for the caption: intra-cluster mean
   cosine similarity tells us whether a codeword corresponds to a
   consistent semantic concept. A random baseline (shuffled codeword
   assignment) is also reported so the absolute number is interpretable.

   Output keys:
     codeword_intra_text_sim[m][k]    -- per-codeword score
     codebook_mean_intra_text_sim[m]  -- per-codebook average (active codewords)
     random_baseline_per_codebook[m]  -- shuffled-assignment baseline
     compositional_lift_per_codebook[m] = above - baseline   -- "concentration" amount

C. top-K image visualization
   For each codebook m, take the top `viz_top_codewords` populated
   codewords; for each such (m, k), grab `viz_per_grid` sample images
   and tile them as a 3x3 grid PNG. Useful for human-reviewer evidence
   that the codeword captures a coherent visual concept.

Usage:
    python compositional_eval.py \
        --result_dir result/260518+flickr25k_setting1_v34_*/ \
        --cache_dir  cache/flickr25k_siglip2 \
        --dataset_root dataset/Flickr25k \
        --viz_top_codewords 5 --viz_per_grid 9
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------- helpers -----------------------

def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def _intra_cluster_cos_sim(feats: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity of L2-normalised rows.
    Returns 1.0 if N<2 (single-sample cluster is trivially "coherent")."""
    if feats.shape[0] < 2:
        return 1.0
    f = _l2_normalize(feats.astype(np.float32), axis=-1)
    sim = f @ f.T
    # mean of upper triangle (off diag)
    N = sim.shape[0]
    iu = np.triu_indices(N, k=1)
    return float(sim[iu].mean())


def _build_path_to_cache_row(image_paths: np.ndarray, cache_image_ids: List[str],
                              dataset_root: str) -> np.ndarray:
    """For each DB sample (by full image path), find its cache row id.

    Cache keys are repository-relative (e.g. 'images/im15316.jpg').
    Returns int64 array of length N (with -1 for misses).
    """
    id_to_row = {iid: i for i, iid in enumerate(cache_image_ids)}
    rows = np.full(len(image_paths), -1, dtype=np.int64)
    miss = 0
    for i, p in enumerate(image_paths):
        # try relpath against dataset_root, fall back to basename heuristics
        relp = os.path.relpath(str(p), dataset_root)
        if relp in id_to_row:
            rows[i] = id_to_row[relp]
            continue
        # fall back: try with leading './' or just 'images/<base>'
        bn = "images/" + os.path.basename(str(p))
        if bn in id_to_row:
            rows[i] = id_to_row[bn]
            continue
        miss += 1
    if miss > 0:
        print(f"[compositional] WARN: {miss}/{len(image_paths)} db rows unmatched in cache.")
    return rows


def _save_image_grid(image_paths: List[str], out_path: str,
                     cols: int = 3, thumb: int = 224) -> None:
    """Tile up to len(image_paths) images into a square-ish grid PNG."""
    from PIL import Image
    n = len(image_paths)
    rows = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), color=(245, 245, 245))
    for i, p in enumerate(image_paths):
        try:
            im = Image.open(p).convert("RGB").resize((thumb, thumb))
        except Exception:
            continue
        r, c = i // cols, i % cols
        canvas.paste(im, (c * thumb, r * thumb))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


# ----------------------- main metrics -----------------------

def _per_codebook_concentration(
    cb_idx_v: np.ndarray,           # [Nv, M]
    feats_per_slot: np.ndarray,     # [Nv, M, D]  for text_part; or [Nv, D] for global
    min_cluster: int,
    rng_seed: int,
    use_per_slot_feature: bool,
    centered: bool,
) -> Tuple[List[List[Optional[float]]], List[List[int]], List[float], List[float], List[float]]:
    """Generic core: per-codebook intra-cluster cosine of `feats_per_slot`.
    If use_per_slot_feature: feats[:, m, :] is the feature seen by codebook m.
    Else: feats[:] is global and shared across codebooks.
    If centered: subtract per-slot mean from features first.
    """
    Nv, M = cb_idx_v.shape
    K = int(cb_idx_v.max()) + 1
    rng = np.random.default_rng(rng_seed)

    out_per_codeword: List[List[Optional[float]]] = [[None] * K for _ in range(M)]
    out_per_codeword_pop: List[List[int]] = [[0] * K for _ in range(M)]
    cb_means: List[float] = []
    cb_baselines: List[float] = []

    for m in range(M):
        if use_per_slot_feature:
            ev_all = feats_per_slot[:, m, :]            # [Nv, D]
        else:
            ev_all = feats_per_slot                     # [Nv, D]
        if centered:
            ev_all = ev_all - ev_all.mean(axis=0, keepdims=True)

        intra_vals, weights = [], []
        for k in range(K):
            idxs = np.where(cb_idx_v[:, m] == k)[0]
            out_per_codeword_pop[m][k] = int(idxs.size)
            if idxs.size < min_cluster:
                continue
            s = _intra_cluster_cos_sim(ev_all[idxs])
            out_per_codeword[m][k] = s
            intra_vals.append(s)
            weights.append(idxs.size)
        cb_means.append(
            float(np.average(intra_vals, weights=weights)) if intra_vals else float("nan")
        )

        shuffled = rng.permutation(cb_idx_v[:, m])
        intra_vals_b, weights_b = [], []
        for k in range(K):
            idxs = np.where(shuffled == k)[0]
            if idxs.size < min_cluster:
                continue
            intra_vals_b.append(_intra_cluster_cos_sim(ev_all[idxs]))
            weights_b.append(idxs.size)
        cb_baselines.append(
            float(np.average(intra_vals_b, weights=weights_b)) if intra_vals_b else float("nan")
        )

    lifts = [
        (m - b) if (not np.isnan(m) and not np.isnan(b)) else float("nan")
        for m, b in zip(cb_means, cb_baselines)
    ]
    return out_per_codeword, out_per_codeword_pop, cb_means, cb_baselines, lifts


def metric_b_text_concentration(
    codebook_indices: np.ndarray,        # [N, M]
    text_part: np.ndarray,               # [N_cache, M, D_proj]
    visual_global: Optional[np.ndarray], # [N_cache, D_proj] or None
    cache_rows: np.ndarray,              # [N] -> cache row (-1 = miss)
    has_text: np.ndarray,                # [N_cache] bool
    min_cluster: int = 5,
    rng_seed: int = 0,
) -> Dict[str, object]:
    """Three variants of intra-cluster semantic concentration:

    B0: raw text_part per-slot (legacy; SigLIP2 text encoder has a tight
        baseline ~0.88 so lift is near-zero noise)
    B1: text_part per-slot, centered by per-slot mean (removes the slot
        baseline so any non-zero lift reflects true within-slot concept clusters)
    B2: visual_global, SHARED across codebooks (uses *all* DB samples,
        not just the 5K with captions; baseline is meaningful)

    For each variant we report the per-codebook clustered cosine vs a
    shuffled-assignment baseline; the "lift" is real - baseline.
    """
    N, M = codebook_indices.shape
    # B0/B1 use text features -> only samples with cached real captions
    valid_text = (cache_rows >= 0) & has_text[np.clip(cache_rows, 0, len(has_text) - 1)]
    valid_text_idx = np.where(valid_text)[0]
    print(f"[B0/B1] text-based: using {valid_text_idx.size}/{N} db samples (has_text=True).")
    out = {
        "n_text_samples": int(valid_text_idx.size),
        "min_cluster":    int(min_cluster),
    }
    if valid_text_idx.size > 0:
        cb_idx_t = codebook_indices[valid_text_idx]
        text_feats = np.asarray(text_part[cache_rows[valid_text_idx]], dtype=np.float32)  # [Nv, M, D]
        for tag, centered in [("b0_raw_text", False), ("b1_centered_text", True)]:
            cw, pop, means, baselines, lifts = _per_codebook_concentration(
                cb_idx_t, text_feats, min_cluster, rng_seed,
                use_per_slot_feature=True, centered=centered,
            )
            out[tag] = {
                "codeword_intra_sim":           cw,
                "codeword_population":          pop,
                "codebook_mean_intra_sim":      means,
                "random_baseline_per_codebook": baselines,
                "compositional_lift_per_codebook": lifts,
                "mean_compositional_lift":      float(np.nanmean(lifts)),
            }
    else:
        print("[B0/B1] skipped: no db rows have cached captions (text path unavailable for this cache).")

    # B2: visual_global on ALL DB samples
    if visual_global is not None:
        valid_v = cache_rows >= 0
        valid_v_idx = np.where(valid_v)[0]
        print(f"[B2] visual_global-based: using {valid_v_idx.size}/{N} db samples.")
        cb_idx_v = codebook_indices[valid_v_idx]
        vg_feats = np.asarray(visual_global[cache_rows[valid_v_idx]], dtype=np.float32)  # [Nv, D]
        cw, pop, means, baselines, lifts = _per_codebook_concentration(
            cb_idx_v, vg_feats, min_cluster, rng_seed,
            use_per_slot_feature=False, centered=False,
        )
        out["b2_visual_global"] = {
            "n_samples":                    int(valid_v_idx.size),
            "codeword_intra_sim":           cw,
            "codeword_population":          pop,
            "codebook_mean_intra_sim":      means,
            "random_baseline_per_codebook": baselines,
            "compositional_lift_per_codebook": lifts,
            "mean_compositional_lift":      float(np.nanmean(lifts)),
        }
    return out


def metric_c_grids(
    codebook_indices: np.ndarray,        # [N, M]
    image_paths: np.ndarray,             # [N]
    out_root: str,
    viz_top_codewords: int = 5,
    viz_per_grid: int = 9,
    cols: int = 3,
) -> Dict[str, object]:
    """For each codebook, pick top-`viz_top_codewords` populated codewords
    and tile their top-`viz_per_grid` sample images as a 3x3 grid."""
    N, M = codebook_indices.shape
    out: List[Dict] = []
    for m in range(M):
        K = int(codebook_indices[:, m].max()) + 1
        counts = np.bincount(codebook_indices[:, m].astype(np.int64), minlength=K)
        order = np.argsort(-counts)        # descending populated
        chosen: List[Dict] = []
        for k in order[:viz_top_codewords]:
            idxs = np.where(codebook_indices[:, m] == k)[0]
            sample_paths = [str(image_paths[i]) for i in idxs[:viz_per_grid]]
            grid_path = os.path.join(
                out_root, f"codebook_grids/cb{m}_cw{int(k):03d}.png"
            )
            _save_image_grid(sample_paths, grid_path, cols=cols)
            chosen.append({
                "codeword":      int(k),
                "population":    int(counts[k]),
                "grid_path":     os.path.relpath(grid_path, out_root),
                "sample_paths":  sample_paths,
            })
        out.append({"codebook": m, "top_codewords": chosen})
    return {
        "viz_top_codewords": int(viz_top_codewords),
        "viz_per_grid":      int(viz_per_grid),
        "per_codebook":      out,
    }


# ----------------------- driver -----------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True,
        help="Result directory with extract_db.npz")
    ap.add_argument("--cache_dir",   default="cache/flickr25k_siglip2")
    ap.add_argument("--dataset_root", default="dataset/Flickr25k")
    ap.add_argument("--min_cluster", type=int, default=5)
    ap.add_argument("--viz_top_codewords", type=int, default=5)
    ap.add_argument("--viz_per_grid",      type=int, default=9)
    ap.add_argument("--skip_grids", action="store_true", default=False,
        help="Skip metric C (image grids). Just compute metric B.")
    args = ap.parse_args()

    extract_p = os.path.join(args.result_dir, "extract_db.npz")
    if not os.path.exists(extract_p):
        raise FileNotFoundError(f"missing {extract_p}")
    print(f"[compositional] loading {extract_p}")
    d = np.load(extract_p, allow_pickle=True)
    cb_idx       = np.asarray(d["codebook_indices"], dtype=np.int64)   # [N, M]
    image_paths  = np.asarray(d["image_paths"])                         # [N]
    N, M = cb_idx.shape
    print(f"[compositional] N={N}, M={M}, K~{int(cb_idx.max())+1}")

    # cache
    ids_p   = os.path.join(args.cache_dir, "image_ids.json")
    tp_p    = os.path.join(args.cache_dir, "text_part.f16.npy")
    vg_p    = os.path.join(args.cache_dir, "visual_global.f16.npy")
    ht_p    = os.path.join(args.cache_dir, "has_text.bool.npy")
    cache_image_ids = json.load(open(ids_p))
    text_part = np.load(tp_p, mmap_mode="r")               # [N_cache, 6, D_proj]
    visual_global = np.load(vg_p, mmap_mode="r") if os.path.exists(vg_p) else None
    has_text  = np.asarray(np.load(ht_p, mmap_mode="r"))   # [N_cache]
    cache_rows = _build_path_to_cache_row(
        image_paths, cache_image_ids, args.dataset_root,
    )

    # --- metric B (three variants: B0 raw text, B1 centered text, B2 visual_global)
    print("[compositional] computing metric B (text+visual concentration) ...")
    metric_b = metric_b_text_concentration(
        cb_idx, text_part, visual_global, cache_rows, has_text,
        min_cluster=args.min_cluster,
    )
    for tag in ("b0_raw_text", "b1_centered_text", "b2_visual_global"):
        if tag not in metric_b:
            continue
        s = metric_b[tag]
        print(f"[{tag}] mean lift = {s['mean_compositional_lift']:.4f}  "
              f"means={[f'{x:.3f}' for x in s['codebook_mean_intra_sim']]}  "
              f"base ={[f'{x:.3f}' for x in s['random_baseline_per_codebook']]}")

    # --- metric C
    metric_c: Dict[str, object] = {}
    if not args.skip_grids:
        print("[compositional] computing metric C (image grids) ...")
        metric_c = metric_c_grids(
            cb_idx, image_paths,
            out_root=args.result_dir,
            viz_top_codewords=args.viz_top_codewords,
            viz_per_grid=args.viz_per_grid,
        )
        print(f"[C] saved {sum(len(cb['top_codewords']) for cb in metric_c['per_codebook'])} grids.")

    out = {
        "result_dir":     args.result_dir,
        "cache_dir":      args.cache_dir,
        "dataset_root":   args.dataset_root,
        "metric_b":       metric_b,
        "metric_c":       metric_c if metric_c else None,
    }
    out_p = os.path.join(args.result_dir, "compositional_eval.json")
    with open(out_p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[compositional] saved -> {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
