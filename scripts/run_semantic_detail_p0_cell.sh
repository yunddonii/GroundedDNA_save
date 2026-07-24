#!/usr/bin/env bash
# Paper-protocol A/ABC semantic-detail cell.
#
# Usage:
#   bash scripts/run_semantic_detail_p0_cell.sh <GPU> <DATASET> <ARM>
#
#   DATASET in {flickr,mscoco,nuswide,cifar10}
#   ARM     in {A,ABC}
#
# Fixed comparison contract:
#   * K=128, L=4 (24 DNA bases), seed=42
#   * A   = strict global-caption-free supervision
#   * ABC = A + own local minimal-pair foils + post-VQ DNA-bit CIBHash KL
#   * Stage 1 fits local-only whitening on optTrain, selects E* on the held-out
#     10% train split, and is forbidden from touching the official test split.
#   * Stage 2 starts from scratch, fits/uses full-train local-only whitening,
#     stops at E*, evaluates the official test split once, then applies the
#     mandatory 24-base biological projection.
#
# Prepare a dataset before launching its cells:
#   bash scripts/prepare_semantic_detail_cache.sh <DATASET> [GPU]
#
# Optional env:
#   PREFLIGHT_ONLY=1  validate every cache/protocol/no-overwrite invariant
#                     without creating a manifest or launching training.
#   BASE_CACHE=...    override the canonical factual cache.
#   FOIL_CACHE=...    override the canonical foil overlay cache.
#   EVAL_CACHE=...    override the canonical whole-image evaluation cache.
#   QWEN=...          override the canonical Qwen caption JSONL.
#   ARTIFACT_DIR=...  manifest root (default below).
#   PY=...            Python interpreter.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$#" -ne 3 ]]; then
    printf 'Usage: %s <GPU> <flickr|mscoco|nuswide|cifar10> <A|ABC>\n' \
        "$0" >&2
    exit 2
fi

GPU="$1"
DS="$2"
ARM="${3^^}"
PY="${PY:-/home/yschoi/.conda/envs/dna_hashing/bin/python}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/semantic_detail_multidataset/p0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
    printf '[semantic-p0] GPU must be one non-negative integer, got %q\n' \
        "$GPU" >&2
    exit 2
fi
if [[ "$ARM" != "A" && "$ARM" != "ABC" ]]; then
    printf '[semantic-p0] ARM must be A or ABC, got %q\n' "$ARM" >&2
    exit 2
fi
if [[ "$PREFLIGHT_ONLY" != "0" && "$PREFLIGHT_ONLY" != "1" ]]; then
    printf '[semantic-p0] PREFLIGHT_ONLY must be 0 or 1, got %q\n' \
        "$PREFLIGHT_ONLY" >&2
    exit 2
fi

declare -a DATASET_ENV
case "$DS" in
    flickr)
        CANON="Flickr25k"
        RESULT_SLUG="flickr25k"
        LAUNCHER="scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh"
        DEFAULT_BASE_CACHE="./cache/flickr25k_clip_v4plus_qwen3_tokens"
        DEFAULT_QWEN="./cache/flickr25k_qwen3_v4_trainset.jsonl"
        DEFAULT_EVAL_CACHE="./cache/flickr25k_clip_v4plus_qwen3_tokens"
        DATASET_ENV=(
            CIBNT=1.0 LBU=0.02 SLA=1.0
            WHITEN_GAMMA=0.25 BI_V=0.5 BI_T=0.5 BIDIR_MODE=legacy
        )
        ;;
    mscoco)
        CANON="MSCOCO"
        RESULT_SLUG="mscoco"
        LAUNCHER="scripts/train_mscoco_F2_sweep_clip.sh"
        DEFAULT_BASE_CACHE="./cache/mscoco_clip_v5b_tokens"
        DEFAULT_QWEN="./cache/mscoco_qwen3_v5b_trainset.jsonl"
        DEFAULT_EVAL_CACHE="./cache/mscoco_clip_v5b"
        DATASET_ENV=(
            WASS=0.05 XMODAL=0.10 THASH=0.10 TCKL=0.10
            CIBNT=1.5 CCS=0.0 SLA=1.0 CELL=semantic_detail_p0
        )
        ;;
    nuswide)
        CANON="NUSWIDE"
        RESULT_SLUG="nuswide"
        LAUNCHER="scripts/train_nuswide_v185_sweep_clip.sh"
        DEFAULT_BASE_CACHE="./cache/nuswide_clip_tokens"
        DEFAULT_QWEN="./cache/nuswide_qwen3_v4_trainset.jsonl"
        DEFAULT_EVAL_CACHE="./cache/nuswide_clip_tokens"
        DATASET_ENV=(
            WASS=0.15 XMODAL=0.05 THASH=0.05 TCKL=0.05
            CIBNT=1.5 CCS=0.0 GATE=4.595
            WHITEN_GAMMA=0.25 BI_V=0.5 BI_T=0.5
            CELL=semantic_detail_p0
        )
        ;;
    cifar10)
        CANON="CIFAR10"
        RESULT_SLUG="cifar10"
        LAUNCHER="scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh"
        DEFAULT_BASE_CACHE="./cache/cifar10_clip"
        DEFAULT_QWEN="./cache/cifar10_qwen.jsonl"
        DEFAULT_EVAL_CACHE="./cache/cifar10_clip"
        DATASET_ENV=(
            CIBNT=1.0 CCS=0.1
            WHITEN_GAMMA=0.25 BI_V=0.5 BI_T=0.5
        )
        ;;
    *)
        printf '[semantic-p0] unknown DATASET=%q\n' "$DS" >&2
        exit 2
        ;;
esac

BASE_CACHE="${BASE_CACHE:-$DEFAULT_BASE_CACHE}"
FOIL_CACHE="${FOIL_CACHE:-${BASE_CACHE}_foils}"
EVAL_CACHE="${EVAL_CACHE:-$DEFAULT_EVAL_CACHE}"
QWEN="${QWEN:-$DEFAULT_QWEN}"
WHITEN_OPT="${FOIL_CACHE}/text_whiten_optTrain_localOnly.npz"
WHITEN_TRAIN="${FOIL_CACHE}/text_whiten_trainOnly_localOnly.npz"
TRAIN_CACHE="$BASE_CACHE"
if [[ "$ARM" == "ABC" ]]; then
    TRAIN_CACHE="$FOIL_CACHE"
fi

BASE_TAG="${DS}_semantic_detail_${ARM}_K128L4"
S1TAG="${BASE_TAG}_P0val_s42"
MANIFEST="${ARTIFACT_DIR}/${DS}_${ARM}_K128L4_p0.json"
COMMON_ARGS=(
    --no-post_eval_compositional
    --no_visualize
    --xmodal_commit_skip_global
    --cibhash_dynamic_tau_skip_global
)
if [[ "$ARM" == "ABC" ]]; then
    COMMON_ARGS+=(
        --text_hash_counterfactual_weight 0.10
        --text_hash_counterfactual_margin 0.02
        --text_hash_counterfactual_warmup_epochs 1
        --cibhash_visual_token_bit_kl
    )
fi
printf -v EXTRA_ARGS_VALUE '%q ' "${COMMON_ARGS[@]}"
# The launchers intentionally expand EXTRA_ARGS as shell words.
EXTRA_ARGS_VALUE="${EXTRA_ARGS_VALUE% }"

# Cache/protocol/no-overwrite preflight. Large feature arrays are checked by
# shape, dtype, byte size, and donor realpath rather than re-hashed in full.
"$PY" - \
    "$DS" "$ARM" "$CANON" "$RESULT_SLUG" "$GPU" \
    "$BASE_CACHE" "$FOIL_CACHE" "$TRAIN_CACHE" "$EVAL_CACHE" "$QWEN" \
    "$WHITEN_OPT" "$WHITEN_TRAIN" "$LAUNCHER" "$S1TAG" "$MANIFEST" \
    "$EXTRA_ARGS_VALUE" "$PREFLIGHT_ONLY" <<'PY'
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

(
    ds,
    arm,
    canon,
    result_slug,
    gpu,
    base_cache,
    foil_cache,
    train_cache,
    eval_cache,
    qwen,
    whiten_opt,
    whiten_train,
    launcher,
    stage1_tag,
    manifest_path,
    extra_args,
    preflight_only,
) = sys.argv[1:]


def fail(message: str) -> None:
    raise SystemExit(f"[semantic-p0] preflight failed: {message}")


def require_file(path: str) -> None:
    if not os.path.isfile(path):
        fail(f"missing file: {path}")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_info(path: str) -> dict:
    require_file(path)
    arr = np.load(path, mmap_mode="r")
    return {
        "path": os.path.realpath(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "bytes": int(arr.nbytes),
    }


def load_ids(path: str) -> list:
    require_file(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        fail(f"expected a JSON list: {path}")
    return data


def validate_whitening(path: str, expected_rows: str) -> dict:
    require_file(path)
    require_file(path + ".meta.json")
    require_file(expected_rows)
    with np.load(path) as bundle:
        if set(bundle.files) != {"mu", "U", "S"}:
            fail(f"unexpected whitening keys in {path}: {bundle.files}")
        mu = np.asarray(bundle["mu"])
        U = np.asarray(bundle["U"])
        S = np.asarray(bundle["S"])
    if mu.ndim != 1 or U.shape != (mu.size, mu.size) or S.shape != mu.shape:
        fail(
            f"invalid whitening shapes in {path}: "
            f"mu={mu.shape}, U={U.shape}, S={S.shape}"
        )
    if not (np.isfinite(mu).all() and np.isfinite(U).all() and np.isfinite(S).all()):
        fail(f"non-finite whitening values: {path}")
    with open(path + ".meta.json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not bool(meta.get("local_slots_only")):
        fail(f"whitening is not local-slots-only: {path}")
    if not bool(meta.get("leakage_free_fit")):
        fail(f"whitening is not row-restricted/leakage-free: {path}")
    if bool(meta.get("residualize_first")):
        fail(f"local-only whitening unexpectedly residualizes global text: {path}")
    recorded_rows = meta.get("row_index_npy")
    if not recorded_rows or os.path.realpath(recorded_rows) != os.path.realpath(expected_rows):
        fail(
            f"whitening row-index mismatch for {path}: "
            f"{recorded_rows!r} != {expected_rows!r}"
        )
    row_idx = np.unique(
        np.asarray(np.load(expected_rows), dtype=np.int64).reshape(-1)
    )
    has_text_path = os.path.join(base_cache, "has_text.bool.npy")
    has_text = np.asarray(np.load(has_text_path), dtype=np.bool_)
    if row_idx.size == 0 or row_idx.min() < 0 or row_idx.max() >= has_text.shape[0]:
        fail(f"invalid row indices in {expected_rows}")
    expected_kept = int(has_text[row_idx].sum())
    if int(meta.get("N_kept", -1)) != expected_kept:
        fail(
            f"N_kept mismatch in {path}: {meta.get('N_kept')} != {expected_kept}"
        )
    # There are exactly five local semantic slots (indices 1..5).
    if int(meta.get("rows_used", -1)) != expected_kept * 5:
        fail(
            f"rows_used mismatch in {path}: "
            f"{meta.get('rows_used')} != {expected_kept * 5}"
        )
    return {
        "path": os.path.realpath(path),
        "sha256": sha256(path),
        "meta_path": os.path.realpath(path + ".meta.json"),
        "meta_sha256": sha256(path + ".meta.json"),
        "N_kept": expected_kept,
        "vectors_used": expected_kept * 5,
        "row_index": os.path.realpath(expected_rows),
    }


for required in (
    launcher,
    qwen,
    os.path.join(foil_cache, "semantic_detail_cache_manifest.json"),
    os.path.join(base_cache, "image_ids.json"),
    os.path.join(base_cache, "has_text.bool.npy"),
    os.path.join(base_cache, "text_part.f16.npy"),
    os.path.join(base_cache, "text_tokens.f16.npy"),
    os.path.join(base_cache, "text_token_mask.bool.npy"),
    os.path.join(base_cache, "visual_tokens_aug0.f16.npy"),
    os.path.join(base_cache, "visual_tokens_aug1.f16.npy"),
    os.path.join(base_cache, "opt_train_rows.npy"),
    os.path.join(base_cache, "train_all_rows.npy"),
):
    require_file(required)
if not os.path.isdir(eval_cache):
    fail(f"evaluation cache directory is missing: {eval_cache}")
if os.path.realpath(whiten_opt) == os.path.realpath(whiten_train):
    fail("optTrain and trainOnly whitening resolve to the same file")

cache_manifest_path = os.path.join(
    foil_cache, "semantic_detail_cache_manifest.json"
)
with open(cache_manifest_path, "r", encoding="utf-8") as handle:
    cache_manifest = json.load(handle)
cache_manifest_expected = {
    "schema_version": "groundeddna-semantic-detail-cache-overlay-v1",
    "dataset": ds,
    "base_cache": os.path.realpath(base_cache),
    "overlay": os.path.realpath(foil_cache),
    "qwen_jsonl": os.path.realpath(qwen),
}
for key, expected in cache_manifest_expected.items():
    if cache_manifest.get(key) != expected:
        fail(
            f"cache-preparation manifest mismatch for {key}: "
            f"{cache_manifest.get(key)!r} != {expected!r}"
        )
if cache_manifest.get("qwen_jsonl_sha256") != sha256(qwen):
    fail("Qwen JSONL digest no longer matches the cache-preparation manifest")

base_ids = load_ids(os.path.join(base_cache, "image_ids.json"))
if cache_manifest.get("image_ids_sha256") != sha256(
    os.path.join(base_cache, "image_ids.json")
):
    fail("image_ids digest no longer matches the cache-preparation manifest")
has_text = np.load(os.path.join(base_cache, "has_text.bool.npy"), mmap_mode="r")
text_part = np.load(os.path.join(base_cache, "text_part.f16.npy"), mmap_mode="r")
text_tokens = np.load(
    os.path.join(base_cache, "text_tokens.f16.npy"), mmap_mode="r"
)
text_token_mask = np.load(
    os.path.join(base_cache, "text_token_mask.bool.npy"), mmap_mode="r"
)
if has_text.shape != (len(base_ids),):
    fail(f"has_text/image_ids mismatch: {has_text.shape} vs {len(base_ids)}")
if text_part.ndim != 3 or text_part.shape[:2] != (len(base_ids), 6):
    fail(f"expected factual text_part [N,6,D], got {text_part.shape}")
if text_tokens.ndim != 4 or text_tokens.shape[:2] != (len(base_ids), 6):
    fail(f"expected factual text_tokens [N,6,T,D], got {text_tokens.shape}")
if text_token_mask.shape != text_tokens.shape[:-1]:
    fail(
        f"factual token/mask mismatch: "
        f"{text_tokens.shape} vs {text_token_mask.shape}"
    )

whitening = {
    "stage1_optTrain_localOnly": validate_whitening(
        whiten_opt, os.path.join(base_cache, "opt_train_rows.npy")
    ),
    "stage2_trainOnly_localOnly": validate_whitening(
        whiten_train, os.path.join(base_cache, "train_all_rows.npy")
    ),
}
for filename, path in (
    ("text_whiten_optTrain_localOnly.npz", whiten_opt),
    ("text_whiten_trainOnly_localOnly.npz", whiten_train),
):
    cache_whitening = cache_manifest.get("whitening", {}).get(filename, {})
    if cache_whitening.get("sha256") != sha256(path):
        fail(f"cache manifest has a stale whitening digest for {filename}")
    expected_rows = (
        os.path.join(base_cache, "opt_train_rows.npy")
        if filename.startswith("text_whiten_optTrain")
        else os.path.join(base_cache, "train_all_rows.npy")
    )
    if cache_whitening.get("row_index_sha256") != sha256(expected_rows):
        fail(f"cache manifest has a stale row-index digest for {filename}")

foil_artifacts = {}
foil_names = (
    "text_foil_part.f16.npy",
    "text_foil_valid.bool.npy",
    "text_foil_tokens.f16.npy",
    "text_foil_token_mask.bool.npy",
    "text_foil_image_ids.json",
    "text_foil_token_image_ids.json",
)
if arm == "A":
    leaked = [
        name for name in foil_names
        if os.path.exists(os.path.join(base_cache, name))
    ]
    if leaked:
        fail(
            "A must consume the factual cache without foil sidecars; found "
            + ", ".join(leaked)
        )
    if os.path.realpath(train_cache) != os.path.realpath(base_cache):
        fail("A training cache is not the factual base cache")
else:
    if os.path.realpath(train_cache) != os.path.realpath(foil_cache):
        fail("ABC training cache is not the foil overlay")
    for donor_name in (
        "image_ids.json",
        "has_text.bool.npy",
        "text_part.f16.npy",
        "text_tokens.f16.npy",
        "text_token_mask.bool.npy",
        "visual_tokens_aug0.f16.npy",
        "visual_tokens_aug1.f16.npy",
    ):
        donor = os.path.join(base_cache, donor_name)
        overlay = os.path.join(foil_cache, donor_name)
        require_file(overlay)
        if os.path.realpath(donor) != os.path.realpath(overlay):
            fail(f"foil overlay donor mismatch: {donor_name}")
    foil_part_path = os.path.join(foil_cache, "text_foil_part.f16.npy")
    foil_valid_path = os.path.join(foil_cache, "text_foil_valid.bool.npy")
    foil_tokens_path = os.path.join(foil_cache, "text_foil_tokens.f16.npy")
    foil_mask_path = os.path.join(
        foil_cache, "text_foil_token_mask.bool.npy"
    )
    for path in (
        foil_part_path,
        foil_valid_path,
        foil_tokens_path,
        foil_mask_path,
    ):
        require_file(path)
    foil_part = np.load(foil_part_path, mmap_mode="r")
    foil_valid = np.load(foil_valid_path, mmap_mode="r")
    foil_tokens = np.load(foil_tokens_path, mmap_mode="r")
    foil_mask = np.load(foil_mask_path, mmap_mode="r")
    if foil_part.shape != text_part.shape:
        fail(f"factual/foil pooled shape mismatch: {text_part.shape}/{foil_part.shape}")
    if foil_valid.shape != text_part.shape[:2]:
        fail(f"foil-valid shape mismatch: {foil_valid.shape}")
    if bool(np.asarray(foil_valid[:, 0]).any()):
        fail("global slot is marked valid in the foil-valid mask")
    if not bool(np.asarray(foil_valid[:, 1:]).any()):
        fail("foil-valid mask has no valid local counterfactual")
    valid_per_slot = [
        int(value)
        for value in np.asarray(foil_valid).sum(axis=0).tolist()
    ]
    if cache_manifest.get("valid_per_slot") != valid_per_slot:
        fail(
            "foil-valid counts no longer match the cache-preparation manifest: "
            f"{valid_per_slot} != {cache_manifest.get('valid_per_slot')}"
        )
    if foil_tokens.shape != text_tokens.shape:
        fail(
            f"factual/foil token shape mismatch: "
            f"{text_tokens.shape}/{foil_tokens.shape}"
        )
    if foil_mask.shape != text_token_mask.shape:
        fail(
            f"factual/foil token-mask mismatch: "
            f"{text_token_mask.shape}/{foil_mask.shape}"
        )
    for id_name in ("text_foil_image_ids.json", "text_foil_token_image_ids.json"):
        if load_ids(os.path.join(foil_cache, id_name)) != base_ids:
            fail(f"foil ID ordering mismatch: {id_name}")
    foil_artifacts = {
        name: (
            {"path": os.path.realpath(os.path.join(foil_cache, name)),
             "sha256": sha256(os.path.join(foil_cache, name))}
            if name.endswith(".json")
            else array_info(os.path.join(foil_cache, name))
        )
        for name in foil_names
    }

stage1_log = os.path.join("logs", stage1_tag + ".log")
result_pattern = os.path.join(
    "result", f"*+{result_slug}_setting1_{stage1_tag}+*"
)
refit_pattern = os.path.join(
    "result", f"*+{result_slug}_setting1_{ds}_semantic_detail_{arm}_"
    "K128L4_P0refit_e*_s42+*"
)
refit_log_pattern = os.path.join(
    "logs",
    f"{ds}_semantic_detail_{arm}_K128L4_P0refit_e*_s42.log",
)
conflicts = []
if os.path.exists(manifest_path):
    conflicts.append(manifest_path)
if os.path.exists(stage1_log):
    conflicts.append(stage1_log)
conflicts.extend(glob.glob(result_pattern))
conflicts.extend(glob.glob(refit_pattern))
conflicts.extend(glob.glob(refit_log_pattern))
if conflicts:
    fail("refusing to overwrite existing run artifacts: " + ", ".join(conflicts))

source_files = (
    "config.py",
    "dataloaders.py",
    "evaluation_siglip2.py",
    "extraction_siglip2.py",
    "loss_siglip2.py",
    "model_siglip2.py",
    "p0_protocol.py",
    "train_siglip2.py",
    "val_split.py",
    "scripts/build_text_whiten_matrix.py",
    "scripts/eval_cell_bioproj.py",
    "scripts/prepare_semantic_detail_cache.py",
    "scripts/prepare_semantic_detail_cache.sh",
    "scripts/run_semantic_detail_multidataset_p0.sh",
    "scripts/run_semantic_detail_p0_cell.sh",
    launcher,
) + tuple(sorted(glob.glob("dna_utils/*.py"))) + tuple(
    sorted(glob.glob("models/*.py"))
)
source_sha256 = {}
for path in source_files:
    require_file(path)
    source_sha256[path] = sha256(path)

manifest = {
    "schema_version": 1,
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "preflight_passed",
    "protocol": {
        "name": "semantic-detail cross-dataset P0",
        "dataset": ds,
        "canonical_dataset": canon,
        "arm": arm,
        "seed": 42,
        "K": 128,
        "num_codons_per_codebook": 4,
        "dna_bases": 24,
        "stage1": "90% opt-train; 10% held-out train validation; select E*",
        "stage2": "scratch refit on 100% train through E*",
        "official_test_evaluations": 1,
        "bio_projection": {
            "mandatory": True,
            "gc_min_frac": 0.416,
            "gc_max_frac": 0.584,
        },
    },
    "gpu": int(gpu),
    "launcher": launcher,
    "stage1_tag": stage1_tag,
    "stage2_tag": None,
    "extra_args": extra_args,
    "cache": {
        "base": os.path.realpath(base_cache),
        "foil_overlay": os.path.realpath(foil_cache),
        "training": os.path.realpath(train_cache),
        "evaluation": os.path.realpath(eval_cache),
        "qwen_jsonl": os.path.realpath(qwen),
        "preparation_manifest": {
            "path": os.path.realpath(cache_manifest_path),
            "sha256": sha256(cache_manifest_path),
            "payload": cache_manifest,
        },
    },
    "factual_arrays": {
        name: array_info(os.path.join(base_cache, name))
        for name in (
            "has_text.bool.npy",
            "text_part.f16.npy",
            "text_tokens.f16.npy",
            "text_token_mask.bool.npy",
            "visual_tokens_aug0.f16.npy",
            "visual_tokens_aug1.f16.npy",
        )
    },
    "whitening": whitening,
    "foil_artifacts": foil_artifacts,
    "source_sha256": source_sha256,
}

if preflight_only == "1":
    print(
        f"[semantic-p0] preflight-only passed: "
        f"dataset={ds} arm={arm} K=128 L=4"
    )
else:
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[semantic-p0] wrote manifest: {manifest_path}")
PY

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    exit 0
fi

resolve_result_dir() {
    local tag="$1"
    local matches=()
    shopt -s nullglob
    matches=(result/*+"${RESULT_SLUG}_setting1_${tag}"+*)
    shopt -u nullglob
    if [[ "${#matches[@]}" -ne 1 ]]; then
        printf '[semantic-p0] expected one result for %s, found %d\n' \
            "$tag" "${#matches[@]}" >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

# Strip launcher controls that could silently contaminate the fixed protocol.
declare -a CLEAN_ENV=(
    -u FINAL_EPOCH -u STOP_EP -u VAL_RATIO -u VAL_SEED
    -u DISABLE_GATE -u DISABLE_TEXT -u SHARE_CB
    -u SEQRES -u SEQRES_G -u TAG -u EXTRA_ARGS
)

printf '[semantic-p0] stage1 dataset=%s arm=%s gpu=%s tag=%s\n' \
    "$DS" "$ARM" "$GPU" "$S1TAG"
env "${CLEAN_ENV[@]}" \
    CACHE="$TRAIN_CACHE" \
    EVAL_CACHE="$EVAL_CACHE" \
    QWEN="$QWEN" \
    WHITEN_NPZ="$WHITEN_OPT" \
    K=128 \
    NUM_CODONS=4 \
    VAL_RATIO=0.1 \
    VAL_SEED=42 \
    TAG="$S1TAG" \
    EXTRA_ARGS="$EXTRA_ARGS_VALUE" \
    "${DATASET_ENV[@]}" \
    bash "$LAUNCHER" "$GPU"

S1LOG="logs/${S1TAG}.log"
if [[ ! -f "$S1LOG" ]]; then
    printf '[semantic-p0] missing stage1 log: %s\n' "$S1LOG" >&2
    exit 3
fi
if ! grep -Fq "Final checkpoint saved" "$S1LOG"; then
    printf '[semantic-p0] stage1 did not reach its final checkpoint\n' >&2
    exit 3
fi
if ! grep -Fq "[p0-stage1] SKIP official-test extraction/evaluation" "$S1LOG"; then
    printf '[semantic-p0] stage1 official-test guard marker is missing\n' >&2
    exit 3
fi
ESTAR="$(
    grep -oE 'new best mid-eval mAP=[0-9.]+ at epoch [0-9]+' "$S1LOG" \
        | grep -oE 'epoch [0-9]+$' \
        | grep -oE '[0-9]+' \
        | tail -1 \
        || true
)"
if [[ -z "$ESTAR" || ! "$ESTAR" =~ ^[0-9]+$ || "$ESTAR" -gt 59 ]]; then
    printf '[semantic-p0] invalid/unparseable E*=%q from %s\n' \
        "${ESTAR:-}" "$S1LOG" >&2
    exit 3
fi
S1RD="$(resolve_result_dir "$S1TAG")"
for required in args.txt log.csv model_state_dict.pth model_state_dict_best.pth; do
    if [[ ! -f "${S1RD}/${required}" ]]; then
        printf '[semantic-p0] incomplete stage1 result: %s\n' \
            "${S1RD}/${required}" >&2
        exit 3
    fi
done
for forbidden in extract_db.npz extract_query.npz evaluation_siglip2_base.json; do
    if [[ -e "${S1RD}/${forbidden}" ]]; then
        printf '[semantic-p0] protocol breach: stage1 produced %s\n' \
            "${S1RD}/${forbidden}" >&2
        exit 3
    fi
done

S2TAG="${BASE_TAG}_P0refit_e${ESTAR}_s42"
S2LOG="logs/${S2TAG}.log"
if [[ -e "$S2LOG" ]]; then
    printf '[semantic-p0] refusing to overwrite %s\n' "$S2LOG" >&2
    exit 3
fi
if shopt -s nullglob; _existing=(result/*+"${RESULT_SLUG}_setting1_${S2TAG}"+*); \
        shopt -u nullglob; [[ "${#_existing[@]}" -ne 0 ]]; then
    printf '[semantic-p0] refusing to overwrite stage2 result: %s\n' \
        "${_existing[*]}" >&2
    exit 3
fi

"$PY" - "$MANIFEST" "$ESTAR" "$S1RD" "$S2TAG" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys

path, estar, stage1_result, stage2_tag = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)


def sha256(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


for filename, expected in data["source_sha256"].items():
    actual = sha256(filename)
    if actual != expected:
        raise SystemExit(
            f"[semantic-p0] source changed during stage1: {filename}"
        )
for entry in data["whitening"].values():
    if sha256(entry["path"]) != entry["sha256"]:
        raise SystemExit(
            f"[semantic-p0] whitening changed during stage1: {entry['path']}"
        )
cache_manifest = data["cache"]["preparation_manifest"]
if sha256(cache_manifest["path"]) != cache_manifest["sha256"]:
    raise SystemExit("[semantic-p0] cache-preparation manifest changed during stage1")

data.update({
    "status": "stage1_complete",
    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "selected_epoch": int(estar),
    "stage1_result_dir": os.path.realpath(stage1_result),
    "stage2_tag": stage2_tag,
})
tmp = path + ".tmp"
with open(tmp, "x", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
PY

printf '[semantic-p0] stage2 dataset=%s arm=%s gpu=%s E*=%s tag=%s\n' \
    "$DS" "$ARM" "$GPU" "$ESTAR" "$S2TAG"
env "${CLEAN_ENV[@]}" \
    CACHE="$TRAIN_CACHE" \
    EVAL_CACHE="$EVAL_CACHE" \
    QWEN="$QWEN" \
    WHITEN_NPZ="$WHITEN_TRAIN" \
    K=128 \
    NUM_CODONS=4 \
    FINAL_EPOCH=1 \
    STOP_EP="$ESTAR" \
    TAG="$S2TAG" \
    EXTRA_ARGS="$EXTRA_ARGS_VALUE" \
    "${DATASET_ENV[@]}" \
    bash "$LAUNCHER" "$GPU"

if [[ ! -f "$S2LOG" ]] \
        || ! grep -Fq "Final checkpoint saved" "$S2LOG" \
        || ! grep -Fq "[refit-stop] stopping after epoch ${ESTAR}" "$S2LOG"; then
    printf '[semantic-p0] stage2 did not complete the E* scratch refit\n' >&2
    exit 4
fi
if ! grep -Fq "[p0-stage2] OFFICIAL TEST ISOLATED" "$S2LOG"; then
    printf '[semantic-p0] stage2 official-test isolation marker is missing\n' >&2
    exit 4
fi
if ! grep -Fq \
        "[final-eval] PRESERVE training text_whiten across cache override" \
        "$S2LOG"; then
    printf '[semantic-p0] final extraction did not preserve trainOnly local whitening\n' >&2
    exit 4
fi
S2RD="$(resolve_result_dir "$S2TAG")"
for required in \
    args.txt log.csv model_state_dict.pth extract_db.npz extract_query.npz \
    evaluation_siglip2_base.json; do
    if [[ ! -f "${S2RD}/${required}" ]]; then
        printf '[semantic-p0] incomplete stage2 result: %s\n' \
            "${S2RD}/${required}" >&2
        exit 4
    fi
done
for protected in cell_result.json evaluation_siglip2_base_bioproj.json; do
    if [[ -e "${S2RD}/${protected}" ]]; then
        printf '[semantic-p0] refusing to overwrite bio result: %s\n' \
            "${S2RD}/${protected}" >&2
        exit 4
    fi
done

"$PY" scripts/eval_cell_bioproj.py \
    --dir "$S2RD" \
    --dataset "$CANON" \
    --K 128 \
    --gc_min 0.416 \
    --gc_max 0.584 \
    | tee -a "$S2LOG"

for required in cell_result.json evaluation_siglip2_base_bioproj.json; do
    if [[ ! -f "${S2RD}/${required}" ]]; then
        printf '[semantic-p0] missing mandatory bio result: %s\n' \
            "${S2RD}/${required}" >&2
        exit 5
    fi
done

"$PY" - "$MANIFEST" "$S2RD" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys

path, stage2_result = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)


def sha256(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


for filename, expected in data["source_sha256"].items():
    if sha256(filename) != expected:
        raise SystemExit(
            f"[semantic-p0] source changed during stage2: {filename}"
        )
for entry in data["whitening"].values():
    if sha256(entry["path"]) != entry["sha256"]:
        raise SystemExit(
            f"[semantic-p0] whitening changed during stage2: {entry['path']}"
        )
cache_manifest = data["cache"]["preparation_manifest"]
if sha256(cache_manifest["path"]) != cache_manifest["sha256"]:
    raise SystemExit("[semantic-p0] cache-preparation manifest changed during stage2")

with open(
    os.path.join(stage2_result, "cell_result.json"),
    "r",
    encoding="utf-8",
) as handle:
    cell_result = json.load(handle)
data.update({
    "status": "complete",
    "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "stage2_result_dir": os.path.realpath(stage2_result),
    "cell_result": cell_result,
})
tmp = path + ".tmp"
with open(tmp, "x", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp, path)
PY

printf '[semantic-p0] DONE dataset=%s arm=%s E*=%s result=%s\n' \
    "$DS" "$ARM" "$ESTAR" "$S2RD"
