#!/usr/bin/env bash
# Four-arm fixed-E*=4 screening for fine-grained semantic/codeword learning.
#
# Arms are single-delta relative to a fresh v185 Flickr25k K128/L4 control:
#   A: remove every C_global-caption dependency active in the MAIN recipe;
#   B: add a token-domain-matched own-foil negative to local text-DNA InfoNCE;
#   C: restore CIBHash Bernoulli bit-KL beside visual-token NT-Xent.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="/home/yschoi/.conda/envs/dna_hashing/bin/python"
LAUNCHER="scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh"
BASE_CACHE="./cache/flickr25k_clip_v4plus_qwen3_tokens"
FOIL_CACHE="./cache/flickr25k_clip_v4plus_qwen3_tokens_foils"
BASE_WHITEN="${BASE_CACHE}/text_whiten_trainOnly.npz"
LOCAL_WHITEN="${BASE_CACHE}/text_whiten_trainOnly_localOnly.npz"
FOIL_WHITEN="${FOIL_CACHE}/text_whiten_trainOnly.npz"
ARTIFACT_DIR="artifacts/semantic_detail_ablation"
mkdir -p "$ARTIFACT_DIR"

"$PY" - "$BASE_CACHE" "$FOIL_CACHE" "$LOCAL_WHITEN" "$ARTIFACT_DIR" <<'PY'
import datetime
import hashlib
import json
import os
import sys

base_cache, foil_cache, local_whiten, artifact_dir = sys.argv[1:]

def require(path):
    if not os.path.exists(path):
        raise SystemExit(f"[semantic-detail] missing required artifact: {path}")

for name in (
    "text_foil_part.f16.npy",
    "text_foil_valid.bool.npy",
    "text_foil_tokens.f16.npy",
    "text_foil_token_mask.bool.npy",
    "text_foil_token_image_ids.json",
):
    require(os.path.join(foil_cache, name))
    if os.path.exists(os.path.join(base_cache, name)):
        raise SystemExit(
            f"[semantic-detail] shared control cache must not contain {name}"
        )

require(local_whiten)
with open(local_whiten + ".meta.json", "r", encoding="utf-8") as handle:
    whiten_meta = json.load(handle)
if not whiten_meta.get("local_slots_only"):
    raise SystemExit("[semantic-detail] A whitening is not local_slots_only")
if not whiten_meta.get("leakage_free_fit"):
    raise SystemExit("[semantic-detail] A whitening is not train-only")
if int(whiten_meta.get("rows_used", -1)) != 25_000:
    raise SystemExit("[semantic-detail] A whitening expected 5000*5 rows")

shared = (
    "image_ids.json",
    "text_part.f16.npy",
    "text_tokens.f16.npy",
    "visual_tokens_aug0.f16.npy",
    "visual_tokens_aug1.f16.npy",
)
for name in shared:
    left = os.path.realpath(os.path.join(base_cache, name))
    right = os.path.realpath(os.path.join(foil_cache, name))
    if left != right:
        raise SystemExit(
            f"[semantic-detail] B donor mismatch for {name}: {left} != {right}"
        )

source_files = (
    "config.py",
    "dataloaders.py",
    "model_siglip2.py",
    "loss_siglip2.py",
    "train_siglip2.py",
    "scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh",
)
hashes = {}
for path in source_files:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    hashes[path] = digest.hexdigest()

manifest = {
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "protocol": "Flickr25k fixed-E*=4 refit screen; seed=42; K=128; L=4",
    "base_cache": os.path.realpath(base_cache),
    "foil_cache": os.path.realpath(foil_cache),
    "a_local_only_whiten": os.path.realpath(local_whiten),
    "source_sha256": hashes,
    "arms": {
        "control": {"gpu": 3, "extra_args": ["--no_visualize"]},
        "A_strict_global_free": {
            "gpu": 0,
            "extra_args": [
                "--no_visualize",
                "--xmodal_commit_skip_global",
                "--cibhash_dynamic_tau_skip_global",
            ],
        },
        "B_own_foil": {
            "gpu": 1,
            "rho": 0.10,
            "margin": 0.02,
            "warmup_epochs": 1,
        },
        "C_bit_kl": {
            "gpu": 2,
            "lambda_cibhash_kl": 0.001,
            "extra_args": ["--no_visualize", "--cibhash_visual_token_bit_kl"],
        },
    },
}
with open(
    os.path.join(artifact_dir, "screen_manifest.json"),
    "w",
    encoding="utf-8",
) as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
print("[semantic-detail] artifact and isolation preflight passed")
PY

run_arm() {
    local arm="$1"
    local gpu="$2"
    local cache="$3"
    local whiten="$4"
    local tag="$5"
    local extra_args="$6"
    local driver_log="${ARTIFACT_DIR}/${arm}.driver.log"

    (
        set -o pipefail
        env \
            CACHE="$cache" \
            WHITEN_NPZ="$whiten" \
            K=128 \
            NUM_CODONS=4 \
            CIBNT=1.0 \
            BIDIR_MODE=legacy \
            FINAL_EPOCH=1 \
            STOP_EP=4 \
            TAG="$tag" \
            EXTRA_ARGS="$extra_args" \
            bash -o pipefail "$LAUNCHER" "$gpu"
    ) >"$driver_log" 2>&1
}

declare -a arms=()
declare -a pids=()

run_arm \
    "control" 3 "$BASE_CACHE" "$BASE_WHITEN" \
    "semantic_detail_e4_control_K128L4" \
    "--no_visualize" &
arms+=("control"); pids+=("$!")

run_arm \
    "A_strict_global_free" 0 "$BASE_CACHE" "$LOCAL_WHITEN" \
    "semantic_detail_e4_A_strictGlobalFree_K128L4" \
    "--no_visualize --xmodal_commit_skip_global --cibhash_dynamic_tau_skip_global" &
arms+=("A_strict_global_free"); pids+=("$!")

run_arm \
    "B_own_foil" 1 "$FOIL_CACHE" "$FOIL_WHITEN" \
    "semantic_detail_e4_B_ownFoil_rho0p10_K128L4" \
    "--no_visualize --text_hash_counterfactual_weight 0.10 --text_hash_counterfactual_margin 0.02 --text_hash_counterfactual_warmup_epochs 1" &
arms+=("B_own_foil"); pids+=("$!")

run_arm \
    "C_bit_kl" 2 "$BASE_CACHE" "$BASE_WHITEN" \
    "semantic_detail_e4_C_bitKL_K128L4" \
    "--no_visualize --cibhash_visual_token_bit_kl" &
arms+=("C_bit_kl"); pids+=("$!")

for index in "${!arms[@]}"; do
    printf '[semantic-detail] started %-24s pid=%s\n' \
        "${arms[$index]}" "${pids[$index]}"
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf '[semantic-detail] completed %-22s\n' "${arms[$index]}"
    else
        printf '[semantic-detail] FAILED %-25s log=%s/%s.driver.log\n' \
            "${arms[$index]}" "$ARTIFACT_DIR" "${arms[$index]}" >&2
        status=1
    fi
done
exit "$status"
