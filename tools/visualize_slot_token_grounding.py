"""Visualize whether each Qwen text slot is grounded in relevant image patches.

The tool uses the whole-image CLIP cache so the N=196 visual tokens map to a
14x14 spatial grid. It compares three pre-routing signals per text slot:

1. pooled cosine: cosine(CLIP-projected patch, pooled slot caption embedding)
2. token max: max cosine over valid token embeddings in the slot caption
3. mutual attention: sum_t sqrt(softmax_t(S) * softmax_n(S))

It deliberately selects difficult cases from a random candidate pool instead
of only rendering attractive examples: high cross-slot overlap, weak peak
cosine, and low pooled-vs-mutual agreement. Outputs a Markdown report, JSON
metrics, and three summary PNGs under ``out_dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dna_utils import build_clip_text_tokenizer
from models.pretrained_backbone_clip import CLIPBackbone, DEFAULT_CLIP_BACKBONE


SLOT_KEYS: Tuple[str, ...] = (
    "C_global",
    "C_primary_object",
    "C_secondary_object",
    "C_activity_or_relation",
    "C_color_texture",
    "C_scene_type",
)
SLOT_NAMES: Tuple[str, ...] = (
    "C0 global",
    "C1 primary",
    "C2 secondary",
    "C3 activity",
    "C4 color",
    "C5 scene",
)


@dataclass
class SampleScores:
    cache_index: int
    image_id: str
    image_path: str
    captions: List[str]
    pooled: np.ndarray             # [6, N]
    token_max: np.ndarray          # [6, N]
    mutual: np.ndarray             # [6, N]
    token_similarity: np.ndarray   # [6, N, T]
    token_mask: np.ndarray         # [6, T]
    token_strings: List[List[str]]
    peak_cosine_mean: float
    mutual_slot_overlap: float
    pooled_mutual_agreement: float
    mutual_entropy_mean: float


def _load_qwen(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id") or row.get("image_path") or "")
            if image_id:
                out[image_id] = row.get("codebook_texts", {}) or {}
                out[os.path.basename(image_id)] = out[image_id]
    return out


def _caption_for(image_id: str, qwen: Dict[str, Dict[str, str]], key: str) -> str:
    row = qwen.get(image_id) or qwen.get(os.path.basename(image_id)) or {}
    return str(row.get(key, "") or "")


def _token_strings(tokenizer, caption: str, max_length: int) -> List[str]:
    encoded = tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    ids = encoded["input_ids"][0].tolist()
    return tokenizer.convert_ids_to_tokens(ids)


def _normalized_entropy(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = x / x.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=dim)
    return entropy / math.log(max(2, x.shape[dim]))


def _pairwise_slot_overlap(maps: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine among local slot heatmaps. maps: [B, 5, N]."""
    assert maps.dim() == 3 and maps.shape[1] == 5
    maps_n = F.normalize(maps, dim=-1)
    sim = torch.matmul(maps_n, maps_n.transpose(1, 2))             # [B, 5, 5]
    tri = torch.triu_indices(5, 5, offset=1, device=maps.device)
    return sim[:, tri[0], tri[1]].mean(dim=-1)                    # [B]


def _map_agreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean local-slot Pearson correlation. Inputs: [B, 5, N]."""
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1).mean(dim=-1)


def _score_batch(
    visual_tokens: torch.Tensor,       # [B, N, H_v]
    pooled_text: torch.Tensor,         # [B, 6, D]
    text_tokens: torch.Tensor,         # [B, 6, T, D]
    text_mask: torch.Tensor,           # [B, 6, T]
    visual_projection: torch.nn.Module,
    logit_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    assert visual_tokens.dim() == 3
    assert pooled_text.dim() == 3 and pooled_text.shape[1] == 6
    assert text_tokens.dim() == 4 and text_tokens.shape[:2] == pooled_text.shape[:2]
    assert text_mask.shape == text_tokens.shape[:3]

    visual_shared = visual_projection(visual_tokens.float())       # [B, N, D]
    visual_n = F.normalize(visual_shared.float(), dim=-1)
    pooled_n = F.normalize(pooled_text.float(), dim=-1)
    token_n = F.normalize(text_tokens.float(), dim=-1)

    pooled = torch.einsum("bnd,bmd->bmn", visual_n, pooled_n)      # [B, 6, N]
    token_similarity = torch.einsum(
        "bnd,bmtd->bmnt", visual_n, token_n,
    )                                                               # [B, 6, N, T]
    pair_valid = text_mask.to(torch.bool).unsqueeze(2)              # [B, 6, 1, T]
    masked_similarity = token_similarity.masked_fill(~pair_valid, -1e4)
    token_max = masked_similarity.max(dim=-1).values                # [B, 6, N]

    scaled = masked_similarity * logit_scale.to(masked_similarity.dtype)
    visual_to_text = scaled.softmax(dim=-1)                         # [B, 6, N, T]
    text_to_visual = scaled.softmax(dim=-2)                         # [B, 6, N, T]
    mutual_pairs = torch.sqrt((visual_to_text * text_to_visual).clamp_min(0.0))
    mutual_pairs = mutual_pairs * pair_valid.to(mutual_pairs.dtype)
    mutual = mutual_pairs.sum(dim=-1)                               # [B, 6, N]

    local_pooled = pooled[:, 1:, :]                                 # [B, 5, N]
    local_mutual = mutual[:, 1:, :]                                 # [B, 5, N]
    pooled_peak_patch = local_pooled.argmax(dim=-1)                  # [B, 5]
    mutual_peak_patch = local_mutual.argmax(dim=-1)                  # [B, 5]

    # Number of distinct peak patches selected by the five local slots.
    # A value near 1 means that semantically different captions all point to
    # the same visual evidence; 5 means every slot has a distinct peak.
    pooled_unique = torch.tensor(
        [row.unique().numel() for row in pooled_peak_patch],
        dtype=local_pooled.dtype, device=local_pooled.device,
    )                                                               # [B]
    mutual_unique = torch.tensor(
        [row.unique().numel() for row in mutual_peak_patch],
        dtype=local_mutual.dtype, device=local_mutual.device,
    )                                                               # [B]
    return {
        "pooled": pooled,
        "token_max": token_max,
        "mutual": mutual,
        "token_similarity": token_similarity,
        "peak_cosine_mean": local_pooled.max(dim=-1).values.mean(dim=-1),
        "mutual_slot_overlap": _pairwise_slot_overlap(local_mutual),
        "pooled_mutual_agreement": _map_agreement(local_pooled, local_mutual),
        "mutual_entropy_mean": _normalized_entropy(local_mutual).mean(dim=-1),
        "pooled_unique_peak_patches": pooled_unique,
        "mutual_unique_peak_patches": mutual_unique,
    }


def _normalize_map(heatmap: np.ndarray) -> np.ndarray:
    x = np.asarray(heatmap, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _overlay(image: Image.Image, heatmap: np.ndarray, grid: int) -> np.ndarray:
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    hm = Image.fromarray(np.uint8(_normalize_map(heatmap).reshape(grid, grid) * 255.0))
    hm = hm.resize(image.size, resample=Image.Resampling.BICUBIC)
    hm_np = np.asarray(hm, dtype=np.float32) / 255.0
    color = plt.get_cmap("turbo")(hm_np)[..., :3].astype(np.float32)
    alpha = (0.15 + 0.55 * hm_np)[..., None]
    return np.clip(image_rgb * (1.0 - alpha) + color * alpha, 0.0, 1.0)


def _render_summary(
    samples: Sequence[SampleScores],
    metric: str,
    out_path: str,
    image_size: int,
) -> None:
    n_rows = len(samples)
    fig, axes = plt.subplots(
        n_rows, 7,
        figsize=(18, max(3.0, 2.55 * n_rows)),
        squeeze=False,
    )
    for row, sample in enumerate(samples):
        image = Image.open(sample.image_path).convert("RGB")
        # The whole-image CLIP cache uses Resize((224, 224)) without a crop.
        # Match that geometry exactly so patch coordinates remain aligned.
        canvas = image.resize(
            (image_size, image_size), resample=Image.Resampling.BICUBIC,
        )
        axes[row, 0].imshow(canvas)
        axes[row, 0].set_title(
            f"{os.path.basename(sample.image_id)}\n"
            f"peak={sample.peak_cosine_mean:.3f} overlap={sample.mutual_slot_overlap:.3f}",
            fontsize=7,
        )
        maps = getattr(sample, metric)                                       # [6, N]
        grid = int(round(math.sqrt(maps.shape[-1])))
        assert grid * grid == maps.shape[-1], (
            f"visual token count {maps.shape[-1]} is not a square grid"
        )
        for slot in range(6):
            axes[row, slot + 1].imshow(_overlay(canvas, maps[slot], grid))
            if row == 0:
                axes[row, slot + 1].set_title(SLOT_NAMES[slot], fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(
        {
            "pooled": "Pooled caption-to-patch cosine",
            "token_max": "Maximum text-token-to-patch cosine",
            "mutual": "Mutual dual-softmax patch importance",
        }[metric],
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _select_indices(metrics: Dict[str, np.ndarray], per_group: int, seed: int) -> List[Tuple[int, str]]:
    selected: List[Tuple[int, str]] = []
    used = set()
    groups = (
        (np.argsort(-metrics["mutual_slot_overlap"]), "high_slot_overlap"),
        (np.argsort(metrics["peak_cosine_mean"]), "weak_peak_cosine"),
        (np.argsort(metrics["pooled_mutual_agreement"]), "low_method_agreement"),
    )
    for order, reason in groups:
        count = 0
        for idx in order.tolist():
            if idx in used:
                continue
            selected.append((idx, reason))
            used.add(idx)
            count += 1
            if count >= per_group:
                break
    rng = np.random.default_rng(seed + 17)
    for idx in rng.permutation(len(metrics["peak_cosine_mean"])).tolist():
        if idx in used:
            continue
        selected.append((idx, "random_control"))
        used.add(idx)
        if sum(reason == "random_control" for _, reason in selected) >= per_group:
            break
    return selected


def _distribution(values: np.ndarray) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p10": float(np.percentile(x, 10)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--qwen_jsonl", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--candidate_pool", type=int, default=512)
    parser.add_argument("--per_group", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=180)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(args.cache_dir, "meta.json"), encoding="utf-8"))
    backbone_name = str(meta.get("backbone") or DEFAULT_CLIP_BACKBONE)
    tokenizer_name = str(meta.get("tokenizer") or backbone_name)
    max_length = int(meta.get("text_token_max_length", 32))
    image_ids: List[str] = json.load(
        open(os.path.join(args.cache_dir, "image_ids.json"), encoding="utf-8")
    )
    qwen = _load_qwen(args.qwen_jsonl)

    visual_mm = np.load(os.path.join(args.cache_dir, "visual_tokens.f16.npy"), mmap_mode="r")
    pooled_mm = np.load(os.path.join(args.cache_dir, "text_part.f16.npy"), mmap_mode="r")
    token_mm = np.load(os.path.join(args.cache_dir, "text_tokens.f16.npy"), mmap_mode="r")
    mask_mm = np.load(os.path.join(args.cache_dir, "text_token_mask.bool.npy"), mmap_mode="r")
    N = len(image_ids)
    assert visual_mm.shape[0] == pooled_mm.shape[0] == token_mm.shape[0] == mask_mm.shape[0] == N
    assert visual_mm.shape[1] == 196, (
        "Use the whole-image cache with 196 tokens; FAIRrank concatenated crops "
        f"cannot be overlaid on one source image, got {visual_mm.shape[1]} tokens."
    )

    valid = np.asarray([
        i for i, image_id in enumerate(image_ids)
        if os.path.exists(os.path.join(args.dataset_root, image_id))
        and (image_id in qwen or os.path.basename(image_id) in qwen)
    ], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    candidate_count = min(int(args.candidate_pool), len(valid))
    candidates = rng.choice(valid, size=candidate_count, replace=False)

    print(f"[grounding] loading {backbone_name} on {args.device}")
    backbone = CLIPBackbone(backbone_name).to(args.device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    tokenizer = build_clip_text_tokenizer(tokenizer_name)
    logit_scale = backbone.model.logit_scale.exp().detach().clamp(max=100.0)

    chunks: Dict[str, List[np.ndarray]] = {
        "pooled": [], "token_max": [], "mutual": [], "token_similarity": [],
        "peak_cosine_mean": [], "mutual_slot_overlap": [],
        "pooled_mutual_agreement": [], "mutual_entropy_mean": [],
        "pooled_unique_peak_patches": [], "mutual_unique_peak_patches": [],
    }
    with torch.no_grad():
        for start in range(0, candidate_count, int(args.batch_size)):
            idx = candidates[start:start + int(args.batch_size)]
            visual = torch.from_numpy(np.asarray(visual_mm[idx]).copy()).to(args.device)
            pooled = torch.from_numpy(np.asarray(pooled_mm[idx]).copy()).to(args.device)
            tokens = torch.from_numpy(np.asarray(token_mm[idx]).copy()).to(args.device)
            masks = torch.from_numpy(np.asarray(mask_mm[idx]).copy()).to(args.device)
            scored = _score_batch(
                visual, pooled, tokens, masks,
                backbone.model.visual_projection,
                logit_scale,
            )
            for key in chunks:
                chunks[key].append(scored[key].float().cpu().numpy())
    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    aggregate = {
        key: _distribution(arrays[key])
        for key in (
            "peak_cosine_mean",
            "mutual_slot_overlap",
            "pooled_mutual_agreement",
            "mutual_entropy_mean",
            "pooled_unique_peak_patches",
            "mutual_unique_peak_patches",
        )
    }
    aggregate["rates"] = {
        "peak_cosine_mean_le_0": float(np.mean(arrays["peak_cosine_mean"] <= 0.0)),
        "peak_cosine_mean_lt_0p05": float(np.mean(arrays["peak_cosine_mean"] < 0.05)),
        "mutual_slot_overlap_ge_0p90": float(
            np.mean(arrays["mutual_slot_overlap"] >= 0.90)
        ),
        "pooled_unique_peak_patches_le_2": float(
            np.mean(arrays["pooled_unique_peak_patches"] <= 2.0)
        ),
        "mutual_unique_peak_patches_le_2": float(
            np.mean(arrays["mutual_unique_peak_patches"] <= 2.0)
        ),
    }
    selected = _select_indices(arrays, per_group=int(args.per_group), seed=int(args.seed))

    samples: List[SampleScores] = []
    selection_rows: List[dict] = []
    for candidate_pos, reason in selected:
        cache_index = int(candidates[candidate_pos])
        image_id = image_ids[cache_index]
        captions = [_caption_for(image_id, qwen, key) for key in SLOT_KEYS]
        token_strings = [
            _token_strings(tokenizer, caption, max_length=max_length)
            for caption in captions
        ]
        sample = SampleScores(
            cache_index=cache_index,
            image_id=image_id,
            image_path=os.path.join(args.dataset_root, image_id),
            captions=captions,
            pooled=arrays["pooled"][candidate_pos],
            token_max=arrays["token_max"][candidate_pos],
            mutual=arrays["mutual"][candidate_pos],
            token_similarity=arrays["token_similarity"][candidate_pos],
            token_mask=np.asarray(mask_mm[cache_index], dtype=np.bool_),
            token_strings=token_strings,
            peak_cosine_mean=float(arrays["peak_cosine_mean"][candidate_pos]),
            mutual_slot_overlap=float(arrays["mutual_slot_overlap"][candidate_pos]),
            pooled_mutual_agreement=float(arrays["pooled_mutual_agreement"][candidate_pos]),
            mutual_entropy_mean=float(arrays["mutual_entropy_mean"][candidate_pos]),
        )
        samples.append(sample)
        selection_rows.append({
            "reason": reason,
            "cache_index": cache_index,
            "image_id": image_id,
            "peak_cosine_mean": sample.peak_cosine_mean,
            "mutual_slot_overlap": sample.mutual_slot_overlap,
            "pooled_mutual_agreement": sample.pooled_mutual_agreement,
            "mutual_entropy_mean": sample.mutual_entropy_mean,
        })

    for metric in ("pooled", "token_max", "mutual"):
        _render_summary(
            samples,
            metric=metric,
            out_path=os.path.join(args.out_dir, f"summary_{metric}.png"),
            image_size=int(args.image_size),
        )

    report_lines = [
        "# Slot-Token Grounding Visualization",
        "",
        f"- cache: `{args.cache_dir}`",
        f"- candidate pool: `{candidate_count}`",
        f"- selected samples: `{len(samples)}`",
        "- selection groups: high slot overlap / weak peak cosine / low method agreement / random control",
        "- image geometry: direct square resize, matching the CLIP cache transform",
        "- note: each heatmap is min-max normalized independently; use the numeric cosine values to compare strength",
        "",
        "## Scope",
        "",
        "- These maps diagnose the frozen CLIP shared space immediately before bidirectional pruning; the tool does not load the trained checkpoint.",
        "- The pooled and token-max maps are direct cosine similarities used to inspect the available grounding evidence.",
        "- The mutual map is the corrected dual-softmax diagnostic. It is not v185's legacy softmax-sum importance, whose normalized-axis sum is constant and therefore cannot provide a meaningful attention heatmap.",
        "",
        "## Candidate-Pool Statistics",
        "",
        "The following values summarize all candidates, not only the deliberately difficult visualized subset.",
        "",
        "| metric | mean | p10 | median | p90 |",
        "|---|---:|---:|---:|---:|",
        "",
    ]
    for metric in (
        "peak_cosine_mean",
        "mutual_slot_overlap",
        "pooled_mutual_agreement",
        "mutual_entropy_mean",
        "pooled_unique_peak_patches",
        "mutual_unique_peak_patches",
    ):
        stats = aggregate[metric]
        report_lines.append(
            f"| {metric} | {stats['mean']:.4f} | {stats['p10']:.4f} | "
            f"{stats['median']:.4f} | {stats['p90']:.4f} |"
        )
    rates = aggregate["rates"]
    report_lines.extend([
        "",
        f"- peak cosine mean below 0.05: `{rates['peak_cosine_mean_lt_0p05']:.1%}`",
        f"- mutual slot overlap at least 0.90: `{rates['mutual_slot_overlap_ge_0p90']:.1%}`",
        f"- at most two distinct mutual peak patches across five local slots: "
        f"`{rates['mutual_unique_peak_patches_le_2']:.1%}`",
        "",
        "## Summary Images",
        "",
        "- [Pooled caption cosine](summary_pooled.png)",
        "- [Maximum token cosine](summary_token_max.png)",
        "- [Mutual dual-softmax attention](summary_mutual.png)",
        "",
    ])
    json_samples: List[dict] = []
    for sample, selection in zip(samples, selection_rows):
        report_lines.extend([
            f"## {os.path.basename(sample.image_id)} — {selection['reason']}",
            "",
            f"- peak cosine mean: `{sample.peak_cosine_mean:.4f}`",
            f"- mutual slot overlap: `{sample.mutual_slot_overlap:.4f}`",
            f"- pooled-mutual agreement: `{sample.pooled_mutual_agreement:.4f}`",
            f"- mutual entropy: `{sample.mutual_entropy_mean:.4f}`",
            "",
            "| slot | caption | peak pooled cos | top matched token | token cos |",
            "|---|---|---:|---|---:|",
        ])
        slot_rows: List[dict] = []
        for slot in range(6):
            sim = sample.token_similarity[slot].copy()                      # [N, T]
            sim[:, ~sample.token_mask[slot]] = -np.inf
            flat_index = int(np.argmax(sim))
            patch_index, token_index = np.unravel_index(flat_index, sim.shape)
            token = (
                sample.token_strings[slot][token_index]
                if token_index < len(sample.token_strings[slot]) else f"token_{token_index}"
            )
            caption_md = sample.captions[slot].replace("|", "\\|")
            report_lines.append(
                f"| {SLOT_NAMES[slot]} | {caption_md} | "
                f"{float(sample.pooled[slot].max()):.4f} | `{token}` (t={token_index}, p={patch_index}) | "
                f"{float(sim[patch_index, token_index]):.4f} |"
            )
            slot_rows.append({
                "slot": SLOT_NAMES[slot],
                "caption": sample.captions[slot],
                "peak_pooled_cosine": float(sample.pooled[slot].max()),
                "top_token": token,
                "top_token_index": int(token_index),
                "top_patch_index": int(patch_index),
                "top_token_cosine": float(sim[patch_index, token_index]),
            })
        report_lines.append("")
        json_samples.append({**selection, "slots": slot_rows})

    with open(os.path.join(args.out_dir, "grounding_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "cache_dir": args.cache_dir,
            "candidate_pool": candidate_count,
            "candidate_pool_statistics": aggregate,
            "samples": json_samples,
        }, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[grounding] wrote {args.out_dir}/report.md")
    print(f"[grounding] wrote {args.out_dir}/grounding_metrics.json")
    for metric in ("pooled", "token_max", "mutual"):
        print(f"[grounding] wrote {args.out_dir}/summary_{metric}.png")


if __name__ == "__main__":
    main()
