"""Build a qualitative codeword concept atlas for a finished GroundedDNA run.

The tool is intentionally post-hoc: it only reads a result directory's
`extract_db.npz` and an offline Qwen JSONL cache.  It produces a compact report
that helps answer whether each codebook/codeword forms a human-inspectable
concept cluster.

Outputs under `<run_dir>/<out_subdir>/`:
  - `atlas.json`: machine-readable per-codebook/per-codeword summary.
  - `report.md`: human-readable interpretation notes and top concepts.
  - `C{m}_atlas.png`: per-codebook representative-image grid.

Expected arrays in `extract_db.npz`:
  - codebook_indices: [N, M] int codeword ids per image and codebook.
  - image_paths:      [N] image file paths.
  - multi_hot_labels: [N, C] optional multi-label targets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import textwrap
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CODEBOOK_KEYS: Tuple[str, ...] = (
    "C_global",
    "C_head_or_main_part",
    "C_body_or_secondary_part",
    "C_limb_or_detail_part",
    "C_color_texture",
    "C_background_null",
)
CODEBOOK_NAMES: Tuple[str, ...] = (
    "global",
    "main_part",
    "secondary_part",
    "detail_part",
    "color_texture",
    "background",
)
CODEBOOK_KEYS_V4: Tuple[str, ...] = (
    "C_global",
    "C_primary_object",
    "C_secondary_object",
    "C_activity_or_relation",
    "C_color_texture",
    "C_scene_type",
)
CODEBOOK_NAMES_V4: Tuple[str, ...] = (
    "global",
    "primary_object",
    "secondary_object",
    "activity_relation",
    "color_texture",
    "scene_type",
)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
STOPWORDS = set(
    """
    a an and are as at be been being by for from has have having he her here him
    his i in into is it its of on or our she so such that the their them then
    there these they this to was we were what when where which who will with you
    your image photo picture scene shot showing show shows contains containing
    contain visible none null not various several some many much very one two
    three main primary secondary background foreground object objects area
    areas part parts
    """.split()
)


def _load_qwen_jsonl(path: str) -> Dict[str, dict]:
    """Return basename(image_path) -> codebook_texts mapping."""
    out: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = os.path.basename(obj.get("image_path") or obj["image_id"])
            out[key] = obj.get("codebook_texts", {})
    return out


def _infer_codebook_schema(qwen: Dict[str, dict]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Infer the six text-slot keys used by the Qwen cache.

    The project has two cached schemas:
      - legacy anatomy-style keys, used by early Flickr experiments.
      - v4 scene-aware keys, used by qwen3/qwen-v4 descriptions.
    """
    for texts in qwen.values():
        keys = set(texts.keys())
        if set(CODEBOOK_KEYS_V4).issubset(keys):
            return CODEBOOK_KEYS_V4, CODEBOOK_NAMES_V4
        if set(CODEBOOK_KEYS).issubset(keys):
            return CODEBOOK_KEYS, CODEBOOK_NAMES
    return CODEBOOK_KEYS, CODEBOOK_NAMES


def _content_words(texts: Iterable[str]) -> List[str]:
    words: List[str] = []
    for text in texts:
        if not isinstance(text, str):
            continue
        for word in WORD_RE.findall(text.lower()):
            if len(word) < 3 or word in STOPWORDS:
                continue
            words.append(word)
    return words


def _normalized_entropy(counts: np.ndarray) -> Tuple[float, float]:
    """Return normalized entropy and perplexity for a non-negative count vector."""
    total = float(counts.sum())
    if total <= 0:
        return 0.0, 0.0
    p = counts.astype(np.float64) / total
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    return float(entropy / max_entropy), float(math.exp(entropy))


def _safe_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _make_thumb(path: str, size: int) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), (244, 244, 244))
        x = (size - img.width) // 2
        y = (size - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas
    except Exception:
        canvas = Image.new("RGB", (size, size), (230, 230, 230))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, size // 2 - 8), "load fail", fill=(80, 80, 80), font=_safe_font(12))
        return canvas


def _write_codebook_grid(
    out_path: str,
    rows: Sequence[dict],
    image_paths: np.ndarray,
    thumb: int,
    samples_per_codeword: int,
    seed: int,
) -> None:
    """Write one codebook atlas image.

    Each row is one codeword.  The left cell contains the codeword summary and
    the right cells contain [samples_per_codeword] representative images.
    """
    rng = np.random.default_rng(seed)
    label_w = 320
    row_h = thumb + 18
    width = label_w + samples_per_codeword * thumb
    height = max(1, len(rows)) * row_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _safe_font(13)
    small_font = _safe_font(11)

    for r, row in enumerate(rows):
        y = r * row_h
        draw.rectangle((0, y, width, y + row_h - 1), outline=(220, 220, 220))
        title = f"k={row['codeword']}  n={row['count']}  labels={row['top_labels_short']}"
        words = ", ".join(row["top_words_short"])
        wrapped = textwrap.wrap(title, width=42)[:2] + textwrap.wrap(words, width=42)[:3]
        for j, line in enumerate(wrapped):
            draw.text((8, y + 6 + j * 17), line, fill=(20, 20, 20), font=font if j == 0 else small_font)

        idxs = np.asarray(row["sample_indices"], dtype=np.int64)
        if len(idxs) > samples_per_codeword:
            idxs = rng.choice(idxs, size=samples_per_codeword, replace=False)
        for c in range(samples_per_codeword):
            x = label_w + c * thumb
            if c < len(idxs):
                canvas.paste(_make_thumb(str(image_paths[idxs[c]]), thumb), (x, y))
            else:
                draw.rectangle((x, y, x + thumb - 1, y + thumb - 1), fill=(245, 245, 245))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def _summarize_codeword(
    code_indices: np.ndarray,
    labels: np.ndarray | None,
    image_paths: np.ndarray,
    qwen: Dict[str, dict],
    codebook_keys: Sequence[str],
    m: int,
    k: int,
    samples_per_codeword: int,
    seed: int,
) -> dict:
    """Summarize one (codebook, codeword) cluster.

    Shapes:
      code_indices: [N, M]
      labels:       [N, C] or None
      image_paths:  [N]
    """
    assert code_indices.ndim == 2, f"code_indices must be [N, M], got {code_indices.shape}"
    assert image_paths.ndim == 1 and image_paths.shape[0] == code_indices.shape[0]
    if labels is not None:
        assert labels.ndim == 2 and labels.shape[0] == code_indices.shape[0]

    idxs = np.where(code_indices[:, m] == k)[0]
    texts: List[str] = []
    for i in idxs:
        q = qwen.get(os.path.basename(str(image_paths[i])), {})
        texts.append(q.get(codebook_keys[m], ""))
    words = _content_words(texts)
    word_counter = Counter(words)
    n_words = sum(word_counter.values())
    top_words = word_counter.most_common(12)
    word_top1 = float(top_words[0][1] / n_words) if n_words and top_words else 0.0
    word_top5 = float(sum(c for _, c in top_words[:5]) / n_words) if n_words else 0.0

    top_labels: List[Tuple[int, int, float]] = []
    label_top1 = 0.0
    label_top3_sum = 0.0
    if labels is not None and len(idxs) > 0:
        label_counts = labels[idxs].sum(axis=0).astype(np.int64)
        order = np.argsort(-label_counts)
        for label_id in order[:6]:
            count = int(label_counts[label_id])
            if count <= 0:
                continue
            top_labels.append((int(label_id), count, float(count / len(idxs))))
        if top_labels:
            label_top1 = top_labels[0][2]
            label_top3_sum = float(sum(x[2] for x in top_labels[:3]))

    rng = np.random.default_rng(seed + 1009 * m + k)
    if len(idxs) > samples_per_codeword:
        sample_indices = rng.choice(idxs, size=samples_per_codeword, replace=False)
    else:
        sample_indices = idxs

    return {
        "codeword": int(k),
        "count": int(len(idxs)),
        "frequency": float(len(idxs) / code_indices.shape[0]),
        "top_words": [{"word": w, "count": int(c)} for w, c in top_words],
        "top_words_short": [w for w, _ in top_words[:6]],
        "word_top1_focus": word_top1,
        "word_top5_focus": word_top5,
        "top_labels": [
            {"label_id": label_id, "count": count, "prevalence": prevalence}
            for label_id, count, prevalence in top_labels
        ],
        "top_labels_short": ",".join(str(label_id) for label_id, _, _ in top_labels[:3]),
        "label_top1_prevalence": label_top1,
        "label_top3_prevalence_sum": label_top3_sum,
        "sample_indices": [int(i) for i in sample_indices.tolist()],
        "sample_images": [str(image_paths[i]) for i in sample_indices.tolist()],
    }


def build_atlas(args: argparse.Namespace) -> dict:
    db_path = os.path.join(args.run_dir, "extract_db.npz")
    data = np.load(db_path, allow_pickle=True)
    code_indices = np.asarray(data["codebook_indices"])  # [N, M]
    image_paths = np.asarray(data["image_paths"])        # [N]
    labels = np.asarray(data["multi_hot_labels"]) if "multi_hot_labels" in data.files else None

    assert code_indices.ndim == 2, f"expected codebook_indices [N, M], got {code_indices.shape}"
    assert image_paths.ndim == 1 and image_paths.shape[0] == code_indices.shape[0]
    if labels is not None:
        assert labels.ndim == 2 and labels.shape[0] == code_indices.shape[0]

    qwen = _load_qwen_jsonl(args.qwen_jsonl)
    codebook_keys, codebook_names = _infer_codebook_schema(qwen)
    out_dir = os.path.join(args.run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    N, M = code_indices.shape
    K = int(code_indices.max()) + 1
    report = {
        "run_dir": args.run_dir,
        "qwen_jsonl": args.qwen_jsonl,
        "num_samples": int(N),
        "num_codebooks": int(M),
        "max_codewords": int(K),
        "codebooks": [],
    }

    md_lines: List[str] = [
        "# Codeword Concept Atlas",
        "",
        f"- run_dir: `{args.run_dir}`",
        f"- qwen_jsonl: `{args.qwen_jsonl}`",
        f"- samples: `{N}`",
        f"- codebook_indices shape: `{tuple(code_indices.shape)}`",
        "",
    ]

    for m in range(M):
        counts = np.bincount(code_indices[:, m].astype(np.int64), minlength=K)
        entropy, perplexity = _normalized_entropy(counts)
        active = np.where(counts > 0)[0]
        dead_ratio = float((counts == 0).sum() / len(counts))
        ordered = active[np.argsort(-counts[active])]
        selected = ordered[: args.top_codewords]

        codewords = [
            _summarize_codeword(
                code_indices, labels, image_paths, qwen, codebook_keys, m, int(k),
                args.samples_per_codeword, args.seed,
            )
            for k in selected
        ]

        weighted_count = max(1, sum(row["count"] for row in codewords))
        mean_word_top1 = sum(row["word_top1_focus"] * row["count"] for row in codewords) / weighted_count
        mean_word_top5 = sum(row["word_top5_focus"] * row["count"] for row in codewords) / weighted_count
        mean_label_top1 = sum(row["label_top1_prevalence"] * row["count"] for row in codewords) / weighted_count

        name = codebook_names[m] if m < len(codebook_names) else f"C{m}"
        slot_key = codebook_keys[m] if m < len(codebook_keys) else None
        grid_path = os.path.join(out_dir, f"C{m}_{name}_atlas.png")
        _write_codebook_grid(
            grid_path,
            codewords,
            image_paths,
            thumb=args.thumb,
            samples_per_codeword=args.samples_per_codeword,
            seed=args.seed,
        )

        cb_summary = {
            "codebook": int(m),
            "name": name,
            "slot_text_key": slot_key,
            "active_codewords": int(len(active)),
            "dead_ratio": dead_ratio,
            "normalized_entropy": entropy,
            "perplexity": perplexity,
            "top_codewords_by_population": codewords,
            "selected_weighted_word_top1_focus": float(mean_word_top1),
            "selected_weighted_word_top5_focus": float(mean_word_top5),
            "selected_weighted_label_top1_prevalence": float(mean_label_top1),
            "grid_path": grid_path,
        }
        report["codebooks"].append(cb_summary)

        md_lines.extend([
            f"## C{m} {cb_summary['name']}",
            "",
            f"- active/dead: `{len(active)}/{len(counts) - len(active)}`",
            f"- normalized entropy: `{entropy:.4f}`",
            f"- perplexity: `{perplexity:.2f}`",
            f"- selected weighted word top-1/top-5 focus: `{mean_word_top1:.3f}` / `{mean_word_top5:.3f}`",
            f"- selected weighted label top-1 prevalence: `{mean_label_top1:.3f}`",
            f"- grid: `{os.path.relpath(grid_path, args.run_dir)}`",
            "",
            "| codeword | n | top words | top labels | word top5 | label top1 |",
            "|---:|---:|---|---|---:|---:|",
        ])
        for row in codewords:
            words = ", ".join(row["top_words_short"])
            labels_txt = ", ".join(
                f"{x['label_id']}:{x['prevalence']:.2f}" for x in row["top_labels"][:4]
            )
            md_lines.append(
                f"| {row['codeword']} | {row['count']} | {words} | {labels_txt} | "
                f"{row['word_top5_focus']:.3f} | {row['label_top1_prevalence']:.3f} |"
            )
        md_lines.append("")

    json_path = os.path.join(out_dir, "atlas.json")
    md_path = os.path.join(out_dir, "report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--qwen_jsonl", default="cache/flickr25k_qwen.jsonl")
    parser.add_argument("--out_subdir", default="codeword_concept_atlas")
    parser.add_argument("--top_codewords", type=int, default=10)
    parser.add_argument("--samples_per_codeword", type=int, default=8)
    parser.add_argument("--thumb", type=int, default=112)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = build_atlas(args)
    out_dir = os.path.join(args.run_dir, args.out_subdir)
    print(f"[atlas] wrote {os.path.join(out_dir, 'atlas.json')}")
    print(f"[atlas] wrote {os.path.join(out_dir, 'report.md')}")
    for cb in report["codebooks"]:
        print(
            f"[atlas] C{cb['codebook']} {cb['name']}: "
            f"active={cb['active_codewords']} "
            f"H={cb['normalized_entropy']:.3f} "
            f"ppl={cb['perplexity']:.1f} "
            f"word_top5={cb['selected_weighted_word_top5_focus']:.3f} "
            f"label_top1={cb['selected_weighted_label_top1_prevalence']:.3f}"
        )


if __name__ == "__main__":
    main()
