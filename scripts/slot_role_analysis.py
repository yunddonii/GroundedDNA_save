#!/usr/bin/env python
"""Slot role specialisation (A) + within-slot graded consistency (B).

Motivation
    Held-out codon decoding (scripts/heldout_codon_decoding.py) showed that a
    slot's code carries decodable semantics, but its target was a single
    image-level label vector shared by all six slots.  It therefore could not
    distinguish "the slot explains something" from "the slot explains ITS OWN
    assigned semantic part", and it only tested `same code -> same concept`,
    not the graded claim `similar meaning -> similar codeword`.

    Orthogonality between slots is NOT required for either claim and is not
    measured here.

(A) Cross-slot decoding matrix
    D[m, m'] = decode slot m''s concept vocabulary from slot m's code.
    Role specialisation shows up as a COLUMN-WISE diagonal advantage: for a
    fixed target slot m', its own code should decode it better than the other
    five slots' codes do.  Off-diagonal cells are expected to be well above
    chance (slots share information) -- that is redundancy, not failure.

    !! CIRCULARITY !!  Per-slot targets come from the Qwen captions that also
    supervised training, and REQUIRED_EXPERIMENTS §2.2 forbids using test
    captions as the answer key.  So (A) is a RELATIVE diagnostic only: the
    diagonal-vs-off-diagonal contrast is meaningful because the circularity
    applies equally to every cell, but the absolute numbers are NOT evidence of
    grounding.  Independent per-slot targets (CUB attributes) are still needed
    for that.

(B) Within-slot graded consistency
    Over test image pairs, correlate code-space distance with semantic distance
    inside a single slot:
        codeword distance : cosine between the assigned codeword embeddings
        codon distance    : base-Hamming between the 3-base codons
        flat chunk (ctrl) : bit-Hamming inside the baseline's 6-bit chunk
    against two semantic distances:
        label     : 1 - Jaccard of image labels        (independent of Qwen)
        slot_text : 1 - cosine of slot-m text features (Qwen-derived, circular)
    Spearman rho, with a shuffled-assignment control.  A positive rho well
    above the shuffled control is the graded claim; comparing codeword vs codon
    rho shows how much of it survives quantisation to 3 bases.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

SLOT_NAMES = ["global", "primary_object", "secondary_object",
              "activity_relation", "color_texture", "scene_type"]
# order of the Qwen caption dict keys, mapped onto our slot order
CAPTION_KEYS = ["C_global", "C_primary_object", "C_secondary_object",
                "C_activity_or_relation", "C_color_texture", "C_scene_type"]
N_SLOTS = 6

STOP = set("""a an the and or but if then than that this these those there here
of in on at to for with from by as is are was were be been being it its it's
into over under above below near about across through during before after
some any all both each few more most other such no nor not only own same so
too very can will just should now which who whom what when where why how
one two three several around against between within without also while
image images photo picture scene shows showing showing appears appear seen
visible frame foreground background left right center centre top bottom side
""".split())

TOKEN_RE = re.compile(r"[a-z]+")


# --------------------------------------------------------------------------
def _basenames(paths: Sequence) -> np.ndarray:
    return np.array([os.path.basename(str(p)) for p in paths])


def codon_ids(base_indices: np.ndarray, n_bases: int) -> np.ndarray:
    n = len(base_indices)
    out = np.zeros((n, N_SLOTS), dtype=np.int64)
    for m in range(N_SLOTS):
        seg = base_indices[:, m * n_bases:(m + 1) * n_bases]
        v = np.zeros(n, dtype=np.int64)
        for r in range(n_bases):
            v = v * 4 + seg[:, r].astype(np.int64)
        out[:, m] = v
    return out


def load_slot_captions(jsonl: str) -> dict[str, list[str]]:
    """basename -> [6 slot caption strings]."""
    out = {}
    with open(jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ct = r.get("codebook_texts", {})
            out[os.path.basename(r["image_id"])] = [ct.get(k, "") for k in CAPTION_KEYS]
    return out


def _slot_df(caps: list[list[str]], slot: int) -> Counter:
    df = Counter()
    for c in caps:
        df.update(set(w for w in TOKEN_RE.findall(c[slot].lower())
                      if w not in STOP and len(w) > 2))
    return df


def build_slot_vocab(caps: list[list[str]], slot: int, vocab_size: int,
                     min_df: int, max_df_frac: float,
                     distinctive_ratio: float = 0.0,
                     all_df: list[Counter] | None = None) -> list[str]:
    """Vocabulary for one slot's caption.

    With `distinctive_ratio > 0`, keep only words that are at least that many
    times more frequent in THIS slot than their mean frequency in the other
    five.  Slot captions share a lot of generic vocabulary ('white', 'dark',
    'person'), and a shared vocabulary makes cross-slot decoding trivially easy
    -- which would mask any real role specialisation.  This filter isolates the
    slot-specific content so the diagonal-vs-off-diagonal contrast is a fair
    test.
    """
    df = all_df[slot] if all_df else _slot_df(caps, slot)
    n = len(caps)
    cand = [(w, d) for w, d in df.items() if d >= min_df and d <= max_df_frac * n]
    if distinctive_ratio > 0.0 and all_df is not None:
        kept = []
        for w, d in cand:
            other = np.mean([all_df[s].get(w, 0) for s in range(N_SLOTS) if s != slot])
            if d >= distinctive_ratio * max(other, 1.0):
                kept.append((w, d))
        cand = kept
    cand.sort(key=lambda x: -x[1])
    return [w for w, _ in cand[:vocab_size]]


def caption_multihot(caps: list[list[str]], slot: int, vocab: list[str]) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    y = np.zeros((len(caps), len(vocab)), dtype=np.int64)
    for i, c in enumerate(caps):
        for w in set(TOKEN_RE.findall(c[slot].lower())):
            j = idx.get(w)
            if j is not None:
                y[i, j] = 1
    return y


# --------------------------------------------------------------------------
def build_dictionary(units, labels, n_units, alpha):
    cnt = np.zeros((n_units, labels.shape[1])); sup = np.zeros(n_units)
    np.add.at(cnt, units, labels.astype(np.float64))
    np.add.at(sup, units, 1.0)
    return (cnt + alpha) / (sup[:, None] + 2.0 * alpha), sup


def label_ranking_ap(scores, truth):
    c = scores.shape[1]
    order = np.argsort(-scores, axis=1, kind="stable")
    hits = np.take_along_axis(truth, order, axis=1).astype(np.float64)
    csum = np.cumsum(hits, axis=1)
    prec = csum / np.arange(1, c + 1, dtype=np.float64)[None, :]
    npos = hits.sum(axis=1)
    return np.where(npos > 0, (prec * hits).sum(axis=1) / np.maximum(npos, 1e-12), np.nan)


def decode(train_u, train_y, test_u, test_y, n_units, alpha, min_support):
    p, sup = build_dictionary(train_u, train_y, n_units, alpha)
    prior = (train_y.sum(0) + alpha) / (len(train_y) + 2.0 * alpha)
    covered = sup[test_u] >= min_support
    scores = np.where(covered[:, None], p[test_u], prior[None, :])
    return float(np.nanmean(label_ranking_ap(scores, test_y)))


# --------------------------------------------------------------------------
def graded_consistency(code_dist, sem_dist, rng, n_shuffle=5):
    rho = spearmanr(code_dist, sem_dist).statistic
    sh = []
    for _ in range(n_shuffle):
        sh.append(spearmanr(rng.permutation(code_dist), sem_dist).statistic)
    return float(rho), float(np.mean(sh)), float(np.std(sh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir", required=True)
    ap.add_argument("--caption_jsonl", default=None,
                    help="Qwen jsonl covering BOTH train and test images. If "
                         "absent, (A) is skipped and (B) runs on the "
                         "label-distance target only (MSCOCO/NUS-WIDE have no "
                         "captions for their test split).")
    ap.add_argument("--train_manifest", default=None)
    ap.add_argument("--ckpt", default=None, help="defaults to <ours_dir>/model_state_dict.pth")
    ap.add_argument("--baseline_dirs", nargs="*", default=[])
    ap.add_argument("--baseline_names", nargs="*", default=[])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab_size", type=int, default=100)
    ap.add_argument("--distinctive_ratio", type=float, default=0.0,
                    help="keep only words >= this many times more frequent in "
                         "the slot than in the other five (0 = no filter)")
    ap.add_argument("--min_df", type=int, default=20)
    ap.add_argument("--max_df_frac", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--min_support", type=int, default=10)
    ap.add_argument("--n_pairs", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    db = np.load(os.path.join(args.ours_dir, "extract_db.npz"), allow_pickle=True)
    qy = np.load(os.path.join(args.ours_dir, "extract_query.npz"), allow_pickle=True)
    tr_path = os.path.join(args.ours_dir, "extract_train.npz")
    tr = np.load(tr_path, allow_pickle=True) if os.path.exists(tr_path) else None

    n_bases = db["base_indices"].shape[1] // N_SLOTS
    caps_by_name = load_slot_captions(args.caption_jsonl) if args.caption_jsonl else {}

    # ---- train rows
    if tr is not None:
        tr_b = _basenames(tr["image_paths"])
        tr_codon = codon_ids(tr["base_indices"], n_bases)
        tr_cw = tr["codebook_indices"]
    else:
        if not args.train_manifest:
            raise SystemExit("need --train_manifest when extract_train.npz is absent")
        want = {os.path.basename(l.split()[0]) for l in open(args.train_manifest) if l.strip()}
        db_b = _basenames(db["image_paths"])
        keep = np.array([i for i, b in enumerate(db_b) if b in want])
        tr_b = db_b[keep]
        tr_codon = codon_ids(db["base_indices"][keep], n_bases)
        tr_cw = db["codebook_indices"][keep]

    te_b = _basenames(qy["image_paths"])
    te_codon = codon_ids(qy["base_indices"], n_bases)
    te_cw = qy["codebook_indices"]

    have_caps = bool(caps_by_name)
    if have_caps:
        miss = [b for b in list(tr_b) + list(te_b) if b not in caps_by_name]
        if miss:
            raise SystemExit(
                f"{len(miss)}/{len(tr_b) + len(te_b)} images have no caption "
                f"(e.g. {miss[:3]}). --caption_jsonl must cover train AND test; "
                f"drop the flag to run (B) only.")
        tr_caps = [caps_by_name[b] for b in tr_b]
        te_caps = [caps_by_name[b] for b in te_b]
    K = int(max(tr_cw.max(), te_cw.max())) + 1

    print(f"[{args.dataset}] train={len(tr_b)} test={len(te_b)} K={K} bases/slot={n_bases}")

    # ================= (A) cross-slot decoding matrix =====================
    mats: dict = {}
    diag_adv: dict = {}
    vocabs: list = []
    te_y: list = []
    prior_ref = np.full(N_SLOTS, np.nan)
    shuf_ref = np.full(N_SLOTS, np.nan)

    if not have_caps:
        print("  [skip] (A) cross-slot matrix -- no --caption_jsonl. "
              "MSCOCO/NUS-WIDE have Qwen captions for train only, so their "
              "test split has no per-slot target and (A) cannot be run there.")
    else:
        all_df = [_slot_df(tr_caps, m) for m in range(N_SLOTS)]
        tr_y = []
        for m in range(N_SLOTS):
            v = build_slot_vocab(tr_caps, m, args.vocab_size, args.min_df,
                                 args.max_df_frac, args.distinctive_ratio, all_df)
            vocabs.append(v)
            tr_y.append(caption_multihot(tr_caps, m, v))
            te_y.append(caption_multihot(te_caps, m, v))
            print(f"  vocab[{SLOT_NAMES[m]:18s}] {len(v):3d} words, e.g. {v[:6]}")

        for level, tru, teu, n_u in [("codon", tr_codon, te_codon, 4 ** n_bases),
                                     ("codeword", tr_cw, te_cw, K)]:
            M = np.zeros((N_SLOTS, N_SLOTS))
            for src in range(N_SLOTS):
                for tgt in range(N_SLOTS):
                    M[src, tgt] = decode(tru[:, src], tr_y[tgt], teu[:, src],
                                         te_y[tgt], n_u, args.alpha, args.min_support)
            mats[level] = M

        for tgt in range(N_SLOTS):
            prior = (tr_y[tgt].sum(0) + args.alpha) / (len(tr_y[tgt]) + 2 * args.alpha)
            prior_ref[tgt] = float(np.nanmean(label_ranking_ap(
                np.repeat(prior[None, :], len(te_y[tgt]), 0), te_y[tgt])))
            perm = rng.permutation(len(tr_codon))
            shuf_ref[tgt] = decode(tr_codon[perm, tgt], tr_y[tgt], te_codon[:, tgt],
                                   te_y[tgt], 4 ** n_bases, args.alpha, args.min_support)

        print("\n(A) cross-slot decoding, CODON level "
              "(rows = source slot, cols = target slot)")
        print("src \\ tgt".ljust(18) + "".join(f"{s[:9]:>10s}" for s in SLOT_NAMES))
        for src in range(N_SLOTS):
            print(f"{SLOT_NAMES[src]:18s}" + "".join(
                f"{mats['codon'][src, t]:9.4f}" + ("*" if t == src else " ")
                for t in range(N_SLOTS)))
        print(f"{'prior':18s}" + "".join(f"{prior_ref[t]:10.4f}" for t in range(N_SLOTS)))
        print(f"{'shuffled':18s}" + "".join(f"{shuf_ref[t]:10.4f}" for t in range(N_SLOTS)))

        for level, M in mats.items():
            adv = []
            for tgt in range(N_SLOTS):
                off = np.array([M[s, tgt] for s in range(N_SLOTS) if s != tgt])
                adv.append({
                    "slot": SLOT_NAMES[tgt],
                    "diagonal": float(M[tgt, tgt]),
                    "offdiag_mean": float(off.mean()),
                    "offdiag_max": float(off.max()),
                    "advantage_vs_mean": float(M[tgt, tgt] - off.mean()),
                    "advantage_vs_max": float(M[tgt, tgt] - off.max()),
                    "is_column_argmax": bool(np.argmax(M[:, tgt]) == tgt),
                })
            diag_adv[level] = adv

        print("\n    column-wise diagonal advantage (codon):")
        for a in diag_adv["codon"]:
            print(f"      {a['slot']:18s} diag={a['diagonal']:.4f} "
                  f"offdiag_mean={a['offdiag_mean']:.4f} "
                  f"adv={a['advantage_vs_mean']:+.4f} "
                  f"vs_best_other={a['advantage_vs_max']:+.4f} "
                  f"argmax={'OWN' if a['is_column_argmax'] else 'OTHER'}")

    # ================= (B) within-slot graded consistency ==================
    ckpt = args.ckpt or os.path.join(args.ours_dir, "model_state_dict.pth")
    import torch
    sd = torch.load(ckpt, map_location="cpu")
    codebooks = sd["quantizer.codebooks"].float().numpy()          # [6, K, D]
    cb = codebooks / np.maximum(np.linalg.norm(codebooks, axis=2, keepdims=True), 1e-12)

    lab_key = "multi_hot_labels" if "multi_hot_labels" in qy.files else "labels"
    te_lab = np.asarray(qy[lab_key])
    if te_lab.ndim == 1:
        te_lab = np.eye(int(te_lab.max()) + 1, dtype=np.int64)[te_lab]

    n = len(te_b)
    ii = rng.integers(0, n, args.n_pairs)
    jj = rng.integers(0, n, args.n_pairs)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]

    inter = (te_lab[ii] & te_lab[jj]).sum(1)
    union = (te_lab[ii] | te_lab[jj]).sum(1)
    sem_label = 1.0 - inter / np.maximum(union, 1)

    # caption-word Jaccard per slot (Qwen-derived; reported separately)
    graded = []
    for m in range(N_SLOTS):
        if have_caps:
            yv = te_y[m]
            i2 = (yv[ii] & yv[jj]).sum(1); u2 = (yv[ii] | yv[jj]).sum(1)
            sem_slot = 1.0 - i2 / np.maximum(u2, 1)

        e_i, e_j = cb[m][te_cw[ii, m]], cb[m][te_cw[jj, m]]
        d_cw = 1.0 - (e_i * e_j).sum(1)

        seg = slice(m * n_bases, (m + 1) * n_bases)
        d_cd = (qy["base_indices"][ii, seg] != qy["base_indices"][jj, seg]).sum(1).astype(float)

        row = {"slot": SLOT_NAMES[m]}
        sem_targets = [("label", sem_label)]
        if have_caps:
            sem_targets.append(("slot_caption", sem_slot))
        for dname, dvec in [("codeword", d_cw), ("codon", d_cd)]:
            for sname, svec in sem_targets:
                r, s, sd_ = graded_consistency(dvec, svec, rng)
                row[f"{dname}_vs_{sname}"] = {
                    "spearman": r, "shuffled_mean": s, "shuffled_std": sd_,
                    "lift": r - s}
        graded.append(row)

    # flat-hash chunk control
    flat_graded = []
    for bdir, bname in zip(args.baseline_dirs, args.baseline_names):
        bqy = np.load(os.path.join(bdir, "extract_query.npz"), allow_pickle=True)
        qpos = {b: i for i, b in enumerate(_basenames(bqy["image_paths"]))}
        bh = (bqy["hash_2bit"][[qpos[b] for b in te_b]] > 0).astype(np.int64)
        w = bh.shape[1] // N_SLOTS
        for m in range(N_SLOTS):
            seg = slice(m * w, (m + 1) * w)
            d = (bh[ii, seg] != bh[jj, seg]).sum(1).astype(float)
            r, s, sd_ = graded_consistency(d, sem_label, rng)
            flat_graded.append({"method": bname, "slot": SLOT_NAMES[m],
                                "chunk_vs_label": {"spearman": r, "shuffled_mean": s,
                                                   "lift": r - s}})

    print("\n(B) within-slot graded consistency (Spearman rho, pairs="
          f"{len(ii)})")
    cols = ["codeword_vs_label", "codon_vs_label"] + (
        ["codeword_vs_slot_caption", "codon_vs_slot_caption"] if have_caps else [])
    print(f"{'slot':18s}" + "".join(f"{c.replace('_vs_', '~'):>26s}" for c in cols))
    for row in graded:
        print(f"{row['slot']:18s}" + "".join(
            f"{row[c]['spearman']:>14.4f} (lift {row[c]['lift']:+.3f})"[-26:]
            for c in cols))
    if flat_graded:
        print("    flat-hash chunk vs label (control):")
        for bname in args.baseline_names:
            v = [f["chunk_vs_label"]["spearman"] for f in flat_graded if f["method"] == bname]
            print(f"      {bname:12s} mean rho={np.mean(v):.4f}  per-slot="
                  f"{' '.join(f'{x:.3f}' for x in v)}")

    payload = {
        "dataset": args.dataset,
        "config": {k: v for k, v in vars(args).items()},
        "slot_names": SLOT_NAMES,
        "caveat": ("(A) targets are Qwen captions that also supervised training; "
                   "REQUIRED_EXPERIMENTS §2.2 forbids these as a held-out answer "
                   "key. (A) is therefore a RELATIVE role-specialisation "
                   "diagnostic (diagonal vs off-diagonal), NOT grounding "
                   "evidence. (B) 'label' targets are independent of Qwen; "
                   "'slot_caption' targets are not."),
        "A_ran": have_caps,
        "A_cross_slot_matrix": {k: v.tolist() for k, v in mats.items()},
        "A_prior_reference": prior_ref.tolist(),
        "A_shuffled_reference": shuf_ref.tolist(),
        "A_diagonal_advantage": diag_adv,
        "A_vocabularies": ({SLOT_NAMES[m]: vocabs[m] for m in range(N_SLOTS)}
                           if have_caps else None),
        "B_graded_consistency": graded,
        "B_flat_chunk_control": flat_graded,
        "n_pairs_used": int(len(ii)),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[{args.dataset}] wrote {args.out}")


if __name__ == "__main__":
    main()
