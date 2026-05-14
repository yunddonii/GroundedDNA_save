"""V1 + V3 compositional-code interpretation report.

Reads a finished SigLIP2 run's `extract_db.npz` + the Qwen-text JSONL cache
and produces, per codebook m in {0..5}:

    V1  codeword_grid_C{m}.png
        Grid of representative DB images for every codeword k used by C_m.
        Rows = codeword id (0..K-1), columns = top-N images that mapped to
        that codeword (closest cluster center under the same EMA codebook
        the model trained with). Visual evidence for "codeword 7 of C_head
        is always faces"-style claims.

    V3  codeword_qwen_words_C{m}.{json,txt}
        For each codeword k, the top-N most-frequent content words drawn
        from the Qwen-generated part text aligned with codebook m (e.g.
        C_head -> C_head_or_main_part). Automatic semantic label per code.

The script is dataset-agnostic in spirit but the part-name mapping below
assumes the Qwen-text schema we use for Flickr25k / NUS-WIDE / MSCOCO.

Usage
-----
    python interpret_compositional_code.py \\
        --run_dir result/260510+flickr25k_setting1_v6_globadp+bs+64+e+60+proj_lr+0.001 \\
        --qwen_jsonl ./cache/flickr25k_qwen.jsonl \\
        --top_n_images 8 --top_n_words 12
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# --- the 6 codebook -> Qwen part-text key mapping (must match dataloader) ---
CODEBOOK_NAMES = [
    "C_global",
    "C_head_or_main_part",
    "C_body_or_secondary_part",
    "C_limb_or_detail_part",
    "C_color_texture",
    "C_background_null",
]
CODEBOOK_SHORT = ["global", "head", "body", "limb", "color", "background"]

# minimal English stop-word list; we don't pull in nltk just for this
STOP = set("""
a an and are as at be by for from has have he her here him his i in into is it
its of on or our she so such that the their them then there these they this to
was we were what when where which who will with you your none null n a image
photo picture scene shot showing show shows containing contain contains
between among also very many various several some most much one two three
""".split())
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _load_qwen_jsonl(path: str) -> Dict[str, dict]:
    """Map basename(image_path) -> codebook_texts dict."""
    out: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            # `image_id` in jsonl is like "images/im15316.jpg"; we key by
            # basename so it matches whatever path style extract_db saved.
            key = os.path.basename(o.get("image_path") or o["image_id"])
            out[key] = o.get("codebook_texts", {})
    return out


def _top_words(texts: List[str], top_n: int) -> List[tuple]:
    counter: Counter = Counter()
    for t in texts:
        if not isinstance(t, str):
            continue
        for w in WORD_RE.findall(t.lower()):
            if w in STOP or len(w) < 3:
                continue
            counter[w] += 1
    return counter.most_common(top_n)


def _sample_indices_for_codeword(
    code_indices: np.ndarray,    # [N, 6]
    m: int,
    k: int,
    top_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return at most `top_n` row-indices where code_indices[row, m] == k.

    No model distances are available in `extract_db.npz`, so within-cluster
    ranking falls back to a deterministic random sample for the visual grid.
    """
    matches = np.where(code_indices[:, m] == k)[0]
    if len(matches) == 0:
        return matches
    if len(matches) <= top_n:
        return matches
    return rng.choice(matches, size=top_n, replace=False)


def _draw_codeword_grid(
    code_indices: np.ndarray,
    image_paths: np.ndarray,
    m: int,
    out_path: str,
    top_n_images: int,
    cell_size: int,
    K_max: int | None = None,
):
    """Save a per-codebook grid PNG (rows = used codewords, cols = sampled imgs)."""
    rng = np.random.default_rng(seed=0)
    used = np.unique(code_indices[:, m])
    if K_max is not None and len(used) > K_max:
        # Show the K_max most populous codewords.
        counts = np.bincount(code_indices[:, m].astype(np.int64))
        ordered = np.argsort(-counts)
        used = np.array([k for k in ordered if k in set(used.tolist())][:K_max])
    used = sorted(used.tolist())
    n_rows, n_cols = len(used), top_n_images

    px = cell_size
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(0.5 * n_cols * px / 50, 0.5 * n_rows * px / 50),
        squeeze=False,
    )
    for r, k in enumerate(used):
        idxs = _sample_indices_for_codeword(code_indices, m, int(k), top_n_images, rng)
        count = int((code_indices[:, m] == k).sum())
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(idxs):
                try:
                    img = Image.open(image_paths[idxs[c]]).convert("RGB").resize((px, px))
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "(load fail)", ha="center", va="center",
                            transform=ax.transAxes)
            else:
                ax.set_facecolor("0.95")
            if c == 0:
                ax.set_ylabel(f"k={k}\n(n={count})", rotation=0,
                              labelpad=24, fontsize=8, va="center")
        for spine in axes[r][0].spines.values():
            spine.set_visible(False)
    fig.suptitle(f"Codebook C_{m} ({CODEBOOK_SHORT[m]})  --  per-codeword example images",
                 fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _build_qwen_word_table(
    code_indices: np.ndarray,
    image_paths: np.ndarray,
    qwen: Dict[str, dict],
    m: int,
    top_n_words: int,
) -> Dict[int, List[tuple]]:
    """For each codeword k in codebook m, top-N content words from Qwen text."""
    qwen_key = CODEBOOK_NAMES[m]
    by_codeword: Dict[int, List[str]] = {}
    for i, k in enumerate(code_indices[:, m]):
        bn = os.path.basename(image_paths[i])
        entry = qwen.get(bn)
        if entry is None:
            continue
        txt = entry.get(qwen_key, "")
        by_codeword.setdefault(int(k), []).append(txt)
    return {
        k: _top_words(texts, top_n_words)
        for k, texts in sorted(by_codeword.items())
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",       type=str, required=True,
                    help="path to a result/<run> directory containing extract_db.npz")
    ap.add_argument("--qwen_jsonl",    type=str, default="./cache/flickr25k_qwen.jsonl",
                    help="JSONL with codebook_texts per image (V3 input)")
    ap.add_argument("--top_n_images",  type=int, default=8,
                    help="number of example images per codeword in V1 grids")
    ap.add_argument("--top_n_words",   type=int, default=12,
                    help="number of frequent words per codeword in V3 tables")
    ap.add_argument("--cell_size",     type=int, default=96,
                    help="pixel size of each image cell in V1 grids")
    ap.add_argument("--k_max_per_grid", type=int, default=None,
                    help="if set, cap rows per V1 grid to this many most-populous codewords")
    ap.add_argument("--out_subdir",    type=str, default="interpret",
                    help="output subdirectory inside the run dir")
    args = ap.parse_args()

    db_path = os.path.join(args.run_dir, "extract_db.npz")
    d = np.load(db_path, allow_pickle=True)
    code_indices = d["codebook_indices"]   # [N, 6]
    image_paths  = d["image_paths"]        # object array of strings
    print(f"[interpret] loaded {db_path}")
    print(f"[interpret] N={code_indices.shape[0]} samples, K_per_codebook={code_indices.max() + 1}")

    qwen = _load_qwen_jsonl(args.qwen_jsonl)
    print(f"[interpret] loaded {len(qwen)} Qwen text entries from {args.qwen_jsonl}")

    out_dir = os.path.join(args.run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    summary: Dict[str, dict] = {}
    for m in range(6):
        name = f"C{m}_{CODEBOOK_SHORT[m]}"
        # ---- V1
        grid_path = os.path.join(out_dir, f"codeword_grid_C{m}_{CODEBOOK_SHORT[m]}.png")
        _draw_codeword_grid(
            code_indices, image_paths, m, grid_path,
            top_n_images=args.top_n_images,
            cell_size=args.cell_size,
            K_max=args.k_max_per_grid,
        )
        print(f"[V1] saved {grid_path}")

        # ---- V3
        table = _build_qwen_word_table(
            code_indices, image_paths, qwen, m, args.top_n_words,
        )
        json_path = os.path.join(out_dir, f"codeword_qwen_words_C{m}_{CODEBOOK_SHORT[m]}.json")
        with open(json_path, "w") as f:
            json.dump(
                {str(k): [{"word": w, "count": int(c)} for (w, c) in entries]
                 for k, entries in table.items()},
                f, indent=2,
            )
        # ---- a human-readable .txt mirror for quick eyeballing
        txt_path = json_path.replace(".json", ".txt")
        with open(txt_path, "w") as f:
            f.write(f"# Codebook C_{m} ({CODEBOOK_NAMES[m]})\n")
            f.write(f"# Top {args.top_n_words} content words per codeword\n\n")
            for k, entries in table.items():
                n_samples = int((code_indices[:, m] == k).sum())
                joined = ", ".join(f"{w}({c})" for w, c in entries)
                f.write(f"k={k:>3d}  n={n_samples:>5d}  {joined}\n")
        print(f"[V3] saved {json_path} (+ .txt mirror)")

        summary[name] = {
            "n_codewords_used": int(len(np.unique(code_indices[:, m]))),
            "max_codeword_count": int(np.bincount(code_indices[:, m]).max()),
            "min_codeword_count_nonzero": int(np.bincount(code_indices[:, m])[
                np.bincount(code_indices[:, m]) > 0
            ].min()),
        }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[interpret] summary -> {out_dir}/summary.json")
    print(f"[interpret] DONE. All outputs under {out_dir}")


if __name__ == "__main__":
    main()
