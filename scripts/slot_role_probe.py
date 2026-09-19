#!/usr/bin/env python
"""Slot-role probe for the off-protocol architecture experiments (branch arch-exp-2026-09).

One Flickr25k stage-1 selection-cell run directory in, one JSON out. Runs the
DEPLOYMENT forward (image only, no captions, epoch restored from the checkpoint
sidecar) and scores the HELD-OUT validation rows, which no arm ever trained on.
The dictionary side always uses the optimisation rows.

 M1  role (primary). Cross-slot caption decoding. For each local axis a, a
     vocabulary of axis-DISTINCTIVE words: document frequency >= 20 on the
     optimisation rows and >= 2x the largest frequency on any other axis, top 100
     by frequency (the 2026-07 cross-slot design). D[m, a] = mean label-ranking AP
     of predicting axis a's words from local slot m's codeword. Role advantage of
     axis a = D[a, a] - mean_{m != a} D[m, a]. Reported: the mean advantage over
     the 4 axes and the number of axes whose own slot is the column argmax.
     Control: the same score with the validation codes shuffled across images.
     Caveat: captions supervise every arm, and b-1 trains on caption words, so
     M1 can favour b-1. M2 does not depend on the captions.
 M2  semantics. Per-slot decoding of the dataset's own multi-hot labels (not the
     captions), mean over the 4 local slots.
 M3  codebook health on the deployment path: per-codebook perplexity (K=128),
     min/median ratio, never-used fraction.
 M4  mechanism: per-image routing mass per local slot (spread across slots).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

REPO = "/home/yschoi/GroundedDNA"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

AXES = ("C_primary_object", "C_secondary_object", "C_activity_or_relation", "C_color_texture")
WORD = re.compile(r"[A-Za-z]+")


def parse_args_txt(path):
    import regen_viz_routing as r
    ns = r._parse_args_txt(path)
    known = set(vars(ns))
    for line in open(path):                                 # dash-free lines (long JSON values)
        line = line.rstrip("\n")
        if not line or "-" in line:
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(.*)$", line)
        if m and m.group(1) not in known:
            setattr(ns, m.group(1), m.group(2).strip())
    return ns


def distinctive_vocab(caps, ids, top_k=100, min_df=20, ratio=2.0):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from collections import Counter
    df = {ax: Counter() for ax in AXES}
    for i in ids:
        for ax in AXES:
            df[ax].update({w.lower() for w in WORD.findall(caps[i][ax])
                           if len(w) >= 3 and w.lower() not in ENGLISH_STOP_WORDS})
    vocab = {}
    for ax in AXES:
        others = [df[o] for o in AXES if o != ax]
        keep = [(w, c) for w, c in df[ax].items()
                if c >= min_df and c >= ratio * max(o.get(w, 0) for o in others)]
        vocab[ax] = [w for w, _ in sorted(keep, key=lambda x: -x[1])[:top_k]]
    return vocab


def multi_hot(caps, ids, ax, vocab):
    idx = {w: j for j, w in enumerate(vocab)}
    Y = np.zeros((len(ids), len(vocab)), dtype=np.int64)
    for r, i in enumerate(ids):
        for w in WORD.findall(caps[i][ax]):
            j = idx.get(w.lower())
            if j is not None:
                Y[r, j] = 1
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qwen_jsonl", default="/data/yschoi/dataset/deephashing/cache/flickr25k_qwen3_v4_trainset.jsonl")
    ap.add_argument("--shuffle_seed", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    os.environ.setdefault("GDNA_NUM_SEMANTIC_PARTS", "5")

    from heldout_codon_decoding import decode_slot, DEFAULT_ALPHA, DEFAULT_MIN_SUPPORT
    from model_siglip2 import SigLIP2SemanticOTModel
    from dataloaders import load_dataset
    from dna_utils.runtime_state import apply_inference_epoch

    args = parse_args_txt(os.path.join(a.result_dir, "args.txt"))
    args.lambda_mec = 0.0                        # the MEC head never runs at deployment
    ckpt = os.path.join(a.result_dir, "model_state_dict.pth")
    model = SigLIP2SemanticOTModel(args).to(a.device).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not k.startswith(("backbone.", "visual_encoder.", "text_encoder."))]
    bad_unexp = [k for k in unexpected if not k.startswith("mec.")]
    if bad_missing or bad_unexp:
        raise SystemExit(f"checkpoint/model mismatch: missing={bad_missing[:6]} unexpected={bad_unexp[:6]}")
    resolved = apply_inference_epoch(model, ckpt, args)

    cache = args.siglip2_feature_cache_dir
    fc_ids = json.load(open(os.path.join(cache, "image_ids.json")))
    opt_rows = set(int(r) for r in np.load(os.path.join(cache, "opt_train_rows.npy")))
    all_rows = set(int(r) for r in np.load(os.path.join(cache, "train_all_rows.npy")))
    val_rows = all_rows - opt_rows

    tr, _, _ = load_dataset(getattr(args, "dataset_dir", "dataset"), args.dataset,
                            setting=getattr(args, "setting", "setting1"), train_transform=None,
                            test_transform=None, load_train=True, load_database=False, load_test=False,
                            return_index=True, qwen_text_cache_path=getattr(args, "qwen_text_cache_path", None),
                            siglip2_feature_cache_dir=cache)
    rows_of = np.asarray(tr._feat_cache_rows)
    idx_opt = [i for i in range(len(tr)) if int(rows_of[i]) in opt_rows]
    idx_val = [i for i in range(len(tr)) if int(rows_of[i]) in val_rows]

    caps = {}
    for line in open(a.qwen_jsonl):
        d = json.loads(line)
        caps[str(d["image_id"])] = {k: str(d["codebook_texts"].get(k, "") or "") for k in AXES}

    def run(indices):
        codes, labels, ids, masses = [], [], [], []
        with torch.no_grad():
            for s0 in range(0, len(indices), 128):
                samples = [tr[i] for i in indices[s0:s0 + 128]]
                vt = torch.stack([s["cached_visual_tokens_raw"] for s in samples]).to(a.device)
                vg = torch.stack([s["cached_visual_global"] for s in samples]).to(a.device)
                o = model(pixel_values=None, part_input_ids=None, part_attention_mask=None,
                          return_routing=True, cached_visual_tokens_raw=vt, cached_visual_global=vg,
                          cached_text_part_raw=None, cached_has_text=None)
                codes.append(o["codebook_indices"].cpu().numpy())
                labels.append(np.stack([np.asarray(s["label"]) for s in samples]))
                ids += ["images/" + os.path.basename(str(s["image_path"])) for s in samples]
                rm = o.get("local_routing_matrix")
                if rm is not None:
                    col = rm.float().sum(dim=1)
                    masses.append((col / col.sum(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy())
        return (np.concatenate(codes), np.concatenate(labels).astype(np.int64), ids,
                (np.concatenate(masses) if masses else None))

    c_opt, y_opt, id_opt, _ = run(idx_opt)
    c_val, y_val, id_val, mass_val = run(idx_val)
    K = int(getattr(args, "codebook_size", 128))
    M = c_val.shape[1]
    loc = list(range(1, M))                                      # local slots

    vocab = distinctive_vocab(caps, id_opt)
    T_opt = {ax: multi_hot(caps, id_opt, ax, vocab[ax]) for ax in AXES}
    T_val = {ax: multi_hot(caps, id_val, ax, vocab[ax]) for ax in AXES}
    rng = np.random.default_rng(a.shuffle_seed)
    perm = rng.permutation(len(id_val))
    Dm = np.zeros((len(loc), len(AXES)))
    Dsh = np.zeros_like(Dm)
    for i, m in enumerate(loc):
        for j, ax in enumerate(AXES):
            r = decode_slot(c_opt[:, m], T_opt[ax], c_val[:, m], T_val[ax], K, DEFAULT_ALPHA, DEFAULT_MIN_SUPPORT)
            Dm[i, j] = r["concept_mAP"]
            rs = decode_slot(c_opt[:, m], T_opt[ax], c_val[perm, m], T_val[ax], K, DEFAULT_ALPHA, DEFAULT_MIN_SUPPORT)
            Dsh[i, j] = rs["concept_mAP"]
    adv = [float(Dm[j, j] - np.mean([Dm[i, j] for i in range(len(loc)) if i != j])) for j in range(len(AXES))]
    own_argmax = int(sum(int(np.argmax(Dm[:, j]) == j) for j in range(len(AXES))))

    lab = [decode_slot(c_opt[:, m], y_opt, c_val[:, m], y_val, K, DEFAULT_ALPHA, DEFAULT_MIN_SUPPORT)["concept_mAP"]
           for m in loc]

    allc = np.concatenate([c_opt, c_val])
    pp, unused = [], []
    for m in range(M):
        cnt = np.bincount(allc[:, m], minlength=K).astype(float)
        p = cnt / cnt.sum()
        pp.append(float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())))
        unused.append(float((cnt == 0).mean()))

    out = {
        "result_dir": a.result_dir, "tag": args.tag, "random_seed": getattr(args, "random_seed", None),
        "inference_epoch": int(resolved.epoch), "n_opt": len(idx_opt), "n_val": len(idx_val),
        "M1_role": {"axes": list(AXES), "D": Dm.round(5).tolist(), "D_shuffled_control": Dsh.round(5).tolist(),
                    "advantage_per_axis": [round(x, 5) for x in adv], "advantage_mean": round(float(np.mean(adv)), 5),
                    "own_slot_is_argmax": own_argmax,
                    "vocab_sizes": {ax: len(vocab[ax]) for ax in AXES}},
        "M2_label_decoding": {"per_local_slot": [round(x, 5) for x in lab], "mean": round(float(np.mean(lab)), 5)},
        "M3_codebook": {"perplexity": [round(x, 2) for x in pp],
                        "min_over_median": round(min(pp) / float(np.median(pp)), 3),
                        "never_used_fraction": [round(x, 4) for x in unused]},
        "M4_mass": (None if mass_val is None else {
            "mean_per_slot": mass_val.mean(0).round(4).tolist(),
            "per_image_max_over_min_median": round(float(np.median(mass_val.max(1) / np.clip(mass_val.min(1), 1e-12, None))), 3),
            "fraction_images_with_empty_slot": round(float((mass_val.min(1) < 1e-6).mean()), 4)}),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("tag", "inference_epoch", "n_opt", "n_val")}))
    print("M1 adv/axis", out["M1_role"]["advantage_per_axis"], "mean", out["M1_role"]["advantage_mean"],
          "own-argmax", own_argmax, "| M2", out["M2_label_decoding"]["mean"], "| M3", out["M3_codebook"]["perplexity"])


if __name__ == "__main__":
    main()
