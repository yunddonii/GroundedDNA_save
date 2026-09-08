"""V1 + V3 compositional-code interpretation report.

Reads one manifest-bound extraction split and produces one report per
manifest-declared codebook. DB and train outputs are deliberately separated:

    V1  codeword_grid_C{m}.png
        Grid of sampled images from the selected split for every codeword k.
        Rows = codeword id (0..K-1), columns = a deterministic sample of
        images mapped to that codeword. Extraction stores no pre-quantization
        vector or center distance, so these are not described as center-nearest.

    V3  codeword_qwen_words_C{m}.{json,txt}  (``--split train`` only)
        For each codeword k, the top-N most-frequent content words drawn
        from the Qwen-generated part text aligned with codebook m under the
        current V4/V5b positional schema. The configured Qwen cache is a
        trainset cache (and MSCOCO train is disjoint from DB), so these tables
        must never be joined to ``extract_db.npz``.

The script is dataset-agnostic in spirit but the part-name mapping below
assumes the Qwen-text schema we use for Flickr25k / NUS-WIDE / MSCOCO.

Usage
-----
    python interpret_compositional_code.py \\
        --run_dir result/260510+flickr25k_setting1_v6_globadp+bs+64+e+60+proj_lr+0.001 \\
        --split train \\
        --qwen_jsonl ./cache/flickr25k_qwen3_v4_trainset.jsonl \\
        --dataset_root ./dataset/Flickr25k \\
        --top_n_images 8 --top_n_words 12
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from compositional_eval import (
    CompositionalInputError,
    _assert_file_evidence_current,
    _atomic_json,
    _canonical_ids_from_image_paths,
    _load_phase5_extraction,
    _resolve_config_path,
    _sha256_file,
    _stable_file_evidence,
    _strict_canonical_image_id,
    _validate_dataset_root_binding,
)


# Exact scene-aware positional schema consumed by current V4/V5b caches.
QWEN_SLOT_KEYS = [
    "C_global",
    "C_primary_object",
    "C_secondary_object",
    "C_activity_or_relation",
    "C_color_texture",
    "C_scene_type",
]
CODEBOOK_SHORT = [
    "global", "primary", "secondary", "activity", "color", "scene"
]

# minimal English stop-word list; we don't pull in nltk just for this
STOP = set("""
a an and are as at be by for from has have he her here him his i in into is it
its of on or our she so such that the their them then there these they this to
was we were what when where which who will with you your none null n a image
photo picture scene shot showing show shows containing contain contains
between among also very many various several some most much one two three
""".split())
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _canonical_qwen_id(value: Any) -> str:
    return _strict_canonical_image_id(value, what="Qwen image_id")


def _ordered_id_digest(image_ids: Sequence[str]) -> str:
    wire = json.dumps(
        list(image_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _load_qwen_jsonl(
    path: str, *, dataset_root: str
) -> tuple[Dict[str, dict], dict[str, Any]]:
    """Load an exact V4/V5b cache keyed by canonical image_id."""
    evidence = _stable_file_evidence(path)
    out: Dict[str, dict] = {}
    versions: set[str] = set()
    vlms: set[str] = set()
    # Preserve the manifest-configured lexical root. Canonical datasets keep
    # `images/` as a symlink to bulk storage; resolving it would make a valid
    # row appear to escape the dataset metadata directory.
    root = os.path.abspath(dataset_root)
    if not os.path.isdir(root):
        raise CompositionalInputError(f"dataset_root is not a directory: {root}")
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError as error:
        raise CompositionalInputError(f"cannot open Qwen JSONL {path}: {error}") from error
    with handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                raise CompositionalInputError(
                    f"Qwen JSONL has a blank row at line {line_number}"
                )
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise CompositionalInputError(
                    f"Qwen JSONL line {line_number} is invalid JSON: {error}"
                ) from error
            if not isinstance(row, Mapping):
                raise CompositionalInputError(
                    f"Qwen JSONL line {line_number} is not an object"
                )
            image_id = _canonical_qwen_id(row.get("image_id"))
            if image_id in out:
                raise CompositionalInputError(
                    f"Qwen JSONL repeats image_id {image_id!r}"
                )
            texts = row.get("codebook_texts")
            if not isinstance(texts, Mapping) or set(texts) != set(QWEN_SLOT_KEYS):
                raise CompositionalInputError(
                    f"Qwen JSONL line {line_number} does not have the exact "
                    f"V4/V5b keys {QWEN_SLOT_KEYS}"
                )
            if any(
                not isinstance(texts[key], str) or not texts[key].strip()
                for key in QWEN_SLOT_KEYS
            ):
                raise CompositionalInputError(
                    f"Qwen JSONL line {line_number} has blank/non-string slot text"
                )
            version = row.get("prompt_version")
            if version not in {"v4", "v5b"}:
                raise CompositionalInputError(
                    f"Qwen JSONL line {line_number} prompt_version={version!r}; "
                    "expected v4 or v5b"
                )
            versions.add(str(version))
            image_path = row.get("image_path")
            if not isinstance(image_path, str) or not image_path:
                raise CompositionalInputError(
                    f"Qwen JSONL line {line_number} has invalid/missing image_path"
                )
            expected_path = os.path.abspath(os.path.join(root, image_id))
            if os.path.commonpath((root, expected_path)) != root or (
                os.path.abspath(os.path.normpath(image_path)) != expected_path
            ):
                raise CompositionalInputError(
                    f"Qwen image_path/image_id disagreement at line {line_number}"
                )
            vlm = row.get("vlm")
            if not isinstance(vlm, str) or not vlm.strip():
                raise CompositionalInputError(
                    f"Qwen JSONL line {line_number} has invalid/missing vlm"
                )
            vlms.add(vlm)
            out[image_id] = dict(texts)
    if not out:
        raise CompositionalInputError("Qwen JSONL is empty")
    if len(versions) != 1:
        raise CompositionalInputError(
            f"Qwen JSONL mixes prompt versions: {sorted(versions)}"
        )
    if len(vlms) != 1:
        raise CompositionalInputError(
            f"Qwen JSONL mixes VLM identities: {sorted(vlms)}"
        )
    if _sha256_file(path) != evidence["sha256"]:
        raise CompositionalInputError(
            "Qwen JSONL changed between byte sealing and semantic parsing"
        )
    _assert_file_evidence_current(evidence)
    ordered_ids = list(out)
    return out, {
        "schema": "scene-aware-v4-v5b",
        "prompt_version": next(iter(versions)),
        "vlm": next(iter(vlms)),
        "slot_keys": list(QWEN_SLOT_KEYS),
        "row_count": len(out),
        "ordered_image_ids_sha256": _ordered_id_digest(ordered_ids),
        "file": evidence,
    }


def _bind_qwen_to_train_extraction(
    qwen: Mapping[str, Mapping[str, str]], image_ids: Sequence[str]
) -> dict[str, Any]:
    """Require exact train IDs and seal the Qwen-to-extraction permutation.

    Official sharded Flickr/MSCOCO JSONLs contain the exact train ID set but
    not dataset order. Dictionary lookup is exact; recording the unique
    Qwen-row permutation into extraction order makes that join auditable
    without imposing a false raw-order equality requirement.
    """
    extraction_ids = list(image_ids)
    qwen_ids = list(qwen)
    missing = sorted(set(extraction_ids) - set(qwen_ids))
    extra = sorted(set(qwen_ids) - set(extraction_ids))
    if missing or extra or len(extraction_ids) != len(qwen_ids):
        raise CompositionalInputError(
            "train Qwen/extraction image-ID sets differ: "
            f"missing={len(missing)} first={missing[:3]}, "
            f"extra={len(extra)} first={extra[:3]}"
        )
    qwen_row = {image_id: index for index, image_id in enumerate(qwen_ids)}
    permutation = np.asarray(
        [qwen_row[image_id] for image_id in extraction_ids], dtype="<i8"
    )
    if np.unique(permutation).size != len(extraction_ids):
        raise CompositionalInputError("Qwen/extraction order mapping is not one-to-one")
    return {
        "policy": "exact_train_id_set_with_explicit_order_permutation",
        "row_count": len(extraction_ids),
        "exact_set_equal": True,
        "raw_order_equal": qwen_ids == extraction_ids,
        "extraction_ordered_image_ids_sha256": _ordered_id_digest(extraction_ids),
        "qwen_ordered_image_ids_sha256": _ordered_id_digest(qwen_ids),
        "qwen_row_for_extraction_order_sha256": hashlib.sha256(
            permutation.tobytes(order="C")
        ).hexdigest(),
    }


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
    code_indices: np.ndarray,    # [N, M]
    image_ids: Sequence[str],
    m: int,
    k: int,
    top_n: int,
) -> np.ndarray:
    """Return at most `top_n` row-indices where code_indices[row, m] == k.

    No model distances are available in the extraction NPZ. Members are ordered
    by SHA-256 of their canonical image ID, which is deterministic and honestly
    described as sampling rather than as a center-distance ranking.
    """
    matches = np.where(code_indices[:, m] == k)[0]
    ordered = sorted(
        matches.tolist(),
        key=lambda index: hashlib.sha256(
            image_ids[index].encode("utf-8")
        ).hexdigest(),
    )
    return np.asarray(ordered[:top_n], dtype=np.int64)


def _draw_codeword_grid(
    code_indices: np.ndarray,
    image_paths: np.ndarray,
    image_ids: Sequence[str],
    m: int,
    out_path: str,
    top_n_images: int,
    cell_size: int,
    codebook_short: Sequence[str],
    K_max: int | None = None,
):
    """Save a per-codebook grid of deterministically sampled cluster members."""
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
        idxs = _sample_indices_for_codeword(
            code_indices, image_ids, m, int(k), top_n_images
        )
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
    fig.suptitle(
        f"Codebook C_{m} ({codebook_short[m]}) -- deterministic sampled members "
        "(not center-nearest)",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _build_qwen_word_table(
    code_indices: np.ndarray,
    image_ids: Sequence[str],
    qwen: Dict[str, dict],
    slot_keys: Sequence[str],
    m: int,
    top_n_words: int,
) -> Dict[int, List[tuple]]:
    """For each codeword k in codebook m, top-N content words from Qwen text."""
    qwen_key = slot_keys[m]
    by_codeword: Dict[int, List[str]] = {}
    for i, k in enumerate(code_indices[:, m]):
        entry = qwen.get(image_ids[i])
        if entry is None:
            continue
        txt = entry.get(qwen_key, "")
        by_codeword.setdefault(int(k), []).append(txt)
    return {
        k: _top_words(texts, top_n_words)
        for k, texts in sorted(by_codeword.items())
    }


def _atomic_text(path: str, value: str) -> None:
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",       type=str, required=True,
                    help="path to a result/<run> directory with sealed extractions")
    ap.add_argument(
        "--split", choices=("train", "db"), required=True,
        help=(
            "train: exact Qwen semantic tables plus train grids; "
            "db: DB image grids only"
        ),
    )
    ap.add_argument(
        "--qwen_jsonl", type=str, default=None,
        help="manifest-configured V4/V5b trainset JSONL; required for --split train",
    )
    ap.add_argument(
        "--dataset_root", type=str, required=True,
        help="dataset directory used to derive exact canonical extraction image IDs",
    )
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
    ap.add_argument(
        "--allow_backfilled", action="store_true", default=False,
        help=(
            "admit retrospectively bound extraction for diagnostics; output is "
            "marked paper-ineligible"
        ),
    )
    args = ap.parse_args()

    for name, value in (
        ("top_n_images", args.top_n_images),
        ("top_n_words", args.top_n_words),
        ("cell_size", args.cell_size),
    ):
        if value <= 0:
            raise CompositionalInputError(f"{name} must be positive")
    if args.k_max_per_grid is not None and args.k_max_per_grid <= 0:
        raise CompositionalInputError("k_max_per_grid must be positive")
    if args.split == "train" and not args.qwen_jsonl:
        raise CompositionalInputError("--split train requires --qwen_jsonl")
    if args.split == "db" and args.qwen_jsonl:
        raise CompositionalInputError(
            "--split db does not consume the trainset Qwen cache; run a separate "
            "--split train report for semantic word tables"
        )

    extraction = _load_phase5_extraction(
        args.run_dir, split=args.split, allow_backfilled=args.allow_backfilled
    )
    dataset_root = _validate_dataset_root_binding(
        args.dataset_root, extraction=extraction
    )
    code_indices = np.asarray(extraction["codebook_indices"], dtype=np.int64)
    image_paths = np.asarray(extraction["image_paths"])
    M = int(extraction["geometry"]["M"])
    L = int(extraction["geometry"]["L"])
    K = int(extraction["geometry"]["K"])
    if M > len(QWEN_SLOT_KEYS):
        raise CompositionalInputError(
            f"manifest declares M={M}, but V4/V5b schema has only "
            f"{len(QWEN_SLOT_KEYS)} positional slots"
        )
    image_ids = _canonical_ids_from_image_paths(image_paths, dataset_root)

    qwen: Dict[str, dict] | None = None
    qwen_evidence: dict[str, Any] | None = None
    qwen_join: dict[str, Any] | None = None
    if args.split == "train":
        configured_qwen = extraction["config"].get("qwen_text_cache_path")
        if not isinstance(configured_qwen, str) or not configured_qwen:
            raise CompositionalInputError(
                "manifest-bound config names no qwen_text_cache_path"
            )
        assert args.qwen_jsonl is not None
        if _resolve_config_path(configured_qwen) != os.path.realpath(args.qwen_jsonl):
            raise CompositionalInputError(
                "explicit qwen_jsonl differs from the manifest-bound config"
            )
        qwen, qwen_evidence = _load_qwen_jsonl(
            args.qwen_jsonl, dataset_root=dataset_root
        )
        qwen_join = _bind_qwen_to_train_extraction(qwen, image_ids)
        print(
            f"[interpret] admitted split=train N={len(image_ids)}, M={M}, "
            f"L={L}, K={K}; exact Qwen set={len(qwen)}"
        )
    else:
        print(
            f"[interpret] admitted split=db N={len(image_ids)}, M={M}, "
            f"L={L}, K={K}; Qwen semantic tables disabled"
        )

    out_root = os.path.realpath(os.path.join(args.run_dir, args.out_subdir))
    run_root = os.path.realpath(args.run_dir)
    if os.path.commonpath((run_root, out_root)) != run_root or out_root == run_root:
        raise CompositionalInputError("out_subdir must be a child of run_dir")
    out_dir = os.path.join(out_root, args.split)
    os.makedirs(out_dir, exist_ok=True)

    summary: Dict[str, dict] = {}
    slot_keys = QWEN_SLOT_KEYS[:M]
    codebook_short = CODEBOOK_SHORT[:M]
    for m in range(M):
        name = f"C{m}_{CODEBOOK_SHORT[m]}"
        # ---- V1
        grid_path = os.path.join(out_dir, f"codeword_grid_C{m}_{CODEBOOK_SHORT[m]}.png")
        _draw_codeword_grid(
            code_indices, image_paths, image_ids, m, grid_path,
            top_n_images=args.top_n_images,
            cell_size=args.cell_size,
            codebook_short=codebook_short,
            K_max=args.k_max_per_grid,
        )
        print(f"[V1] saved {grid_path}")

        # ---- V3: train extraction only. Never mix trainset Qwen with DB codes.
        if qwen is not None:
            table = _build_qwen_word_table(
                code_indices, image_ids, qwen, slot_keys, m, args.top_n_words,
            )
            json_path = os.path.join(
                out_dir, f"codeword_qwen_words_C{m}_{CODEBOOK_SHORT[m]}.json"
            )
            _atomic_json(
                json_path,
                {str(k): [{"word": w, "count": int(c)} for (w, c) in entries]
                 for k, entries in table.items()},
            )
            # ---- a human-readable .txt mirror for quick eyeballing
            txt_path = json_path.replace(".json", ".txt")
            lines = [
                f"# Codebook C_{m} ({slot_keys[m]})\n",
                f"# Top {args.top_n_words} content words per codeword\n\n",
            ]
            for k, entries in table.items():
                n_samples = int((code_indices[:, m] == k).sum())
                joined = ", ".join(f"{w}({c})" for w, c in entries)
                lines.append(f"k={k:>3d}  n={n_samples:>5d}  {joined}\n")
            _atomic_text(txt_path, "".join(lines))
            print(f"[V3] saved {json_path} (+ .txt mirror)")

        summary[name] = {
            "n_codewords_used": int(len(np.unique(code_indices[:, m]))),
            "max_codeword_count": int(np.bincount(code_indices[:, m]).max()),
            "min_codeword_count_nonzero": int(np.bincount(code_indices[:, m])[
                np.bincount(code_indices[:, m]) > 0
            ].min()),
        }

    summary["_evidence"] = {
        "schema_version": 2,
        "analysis_split": args.split,
        "input_binding": extraction["input_binding"],
        "extraction_file_evidence": extraction["transaction_evidence"],
        "geometry": extraction["geometry"],
        "qwen_binding": qwen_evidence,
        "qwen_extraction_join": qwen_join,
        "qwen_rows_mapped_to_extraction": len(qwen) if qwen is not None else 0,
        "qwen_coverage_of_extraction": (
            1.0 if qwen is not None else None
        ),
        "semantic_word_tables_produced": qwen is not None,
        "slot_keys_used": list(slot_keys),
        "representative_selection": {
            "method": "deterministic_sha256_image_id_sample_within_codeword",
            "center_distances_available": False,
            "center_nearest_claim": False,
        },
        "paper_eligibility": {
            "eligible": bool(extraction["paper_eligible"]),
            "reason": (
                (
                    "fresh train extraction/checkpoint manifests and exact-set "
                    "Qwen/image-ID order binding"
                    if args.split == "train"
                    else "fresh DB extraction/checkpoint manifests; DB grids contain "
                    "no trainset-Qwen join"
                )
                if extraction["paper_eligible"]
                else "diagnostic only: extraction manifests were backfilled"
            ),
        },
        "producer": {
            "path": os.path.abspath(__file__),
            "sha256": _sha256_file(__file__),
        },
    }
    for evidence in extraction["transaction_evidence"].values():
        _assert_file_evidence_current(evidence)
    if qwen_evidence is not None:
        _assert_file_evidence_current(qwen_evidence["file"])
    _atomic_json(os.path.join(out_dir, "summary.json"), summary)
    print(f"[interpret] summary -> {out_dir}/summary.json")
    print(f"[interpret] DONE. All outputs under {out_dir}")


if __name__ == "__main__":
    main()
