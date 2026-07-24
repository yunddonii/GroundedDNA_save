#!/usr/bin/env bash
# Complete the missing cells of the A/B/C semantic-detail 2^3 factorial.
#
# Existing cells (do not rerun): control, A, B, C.
# This launcher runs only:
#   AB  strict global-caption-free + own-foil local text-DNA InfoNCE
#   AC  strict global-caption-free + CIBHash post-VQ bit-KL
#   BC  own-foil local text-DNA InfoNCE + CIBHash post-VQ bit-KL
#   ABC all three changes
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
FOIL_LOCAL_WHITEN="${FOIL_CACHE}/text_whiten_trainOnly_localOnly.npz"
ARTIFACT_DIR="artifacts/semantic_detail_factorial"

GPU_AB="${GPU_AB:-0}"
GPU_AC="${GPU_AC:-1}"
GPU_BC="${GPU_BC:-2}"
GPU_ABC="${GPU_ABC:-3}"

TAG_AB="semantic_detail_e4_AB_strictGlobalFree_ownFoil_rho0p10_K128L4"
TAG_AC="semantic_detail_e4_AC_strictGlobalFree_bitKL_K128L4"
TAG_BC="semantic_detail_e4_BC_ownFoil_rho0p10_bitKL_K128L4"
TAG_ABC="semantic_detail_e4_ABC_strictGlobalFree_ownFoil_rho0p10_bitKL_K128L4"

mkdir -p "$ARTIFACT_DIR"

"$PY" - \
    "$BASE_CACHE" "$FOIL_CACHE" "$LOCAL_WHITEN" "$FOIL_LOCAL_WHITEN" \
    "$ARTIFACT_DIR" \
    "$GPU_AB" "$GPU_AC" "$GPU_BC" "$GPU_ABC" \
    "$TAG_AB" "$TAG_AC" "$TAG_BC" "$TAG_ABC" <<'PY'
import datetime
import glob
import hashlib
import json
import os
import sys

(
    base_cache,
    foil_cache,
    local_whiten,
    foil_local_whiten,
    artifact_dir,
    gpu_ab,
    gpu_ac,
    gpu_bc,
    gpu_abc,
    tag_ab,
    tag_ac,
    tag_bc,
    tag_abc,
) = sys.argv[1:]


def require(path):
    if not os.path.exists(path):
        raise SystemExit(f"[semantic-factorial] missing artifact: {path}")


for path in (
    os.path.join(base_cache, "text_whiten_trainOnly.npz"),
    os.path.join(foil_cache, "text_whiten_trainOnly.npz"),
    local_whiten,
    foil_local_whiten,
):
    require(path)

if os.path.realpath(local_whiten) != os.path.realpath(foil_local_whiten):
    raise SystemExit(
        "[semantic-factorial] base and foil local-only whitening differ"
    )

with open(local_whiten + ".meta.json", "r", encoding="utf-8") as handle:
    whiten_meta = json.load(handle)
if not (
    bool(whiten_meta.get("local_slots_only"))
    and bool(whiten_meta.get("leakage_free_fit"))
    and int(whiten_meta.get("rows_used", -1)) == 25_000
):
    raise SystemExit(
        "[semantic-factorial] invalid local-only train whitening metadata"
    )

foil_names = (
    "text_foil_part.f16.npy",
    "text_foil_valid.bool.npy",
    "text_foil_tokens.f16.npy",
    "text_foil_token_mask.bool.npy",
    "text_foil_token_image_ids.json",
)
for name in foil_names:
    require(os.path.join(foil_cache, name))
    if os.path.exists(os.path.join(base_cache, name)):
        raise SystemExit(
            f"[semantic-factorial] base cache unexpectedly contains {name}"
        )

for name in (
    "image_ids.json",
    "text_part.f16.npy",
    "text_tokens.f16.npy",
    "visual_tokens_aug0.f16.npy",
    "visual_tokens_aug1.f16.npy",
):
    base_real = os.path.realpath(os.path.join(base_cache, name))
    foil_real = os.path.realpath(os.path.join(foil_cache, name))
    if base_real != foil_real:
        raise SystemExit(
            f"[semantic-factorial] donor mismatch for {name}: "
            f"{base_real} != {foil_real}"
        )

tags = (tag_ab, tag_ac, tag_bc, tag_abc)
gpus = (gpu_ab, gpu_ac, gpu_bc, gpu_abc)
if len(set(gpus)) != len(gpus):
    raise SystemExit(
        "[semantic-factorial] AB/AC/BC/ABC must use four distinct GPUs: "
        + ", ".join(gpus)
    )
for tag in tags:
    matches = glob.glob(os.path.join("result", f"*+flickr25k_setting1_{tag}+*"))
    if matches:
        raise SystemExit(
            f"[semantic-factorial] refusing to overwrite result for {tag}: "
            + ", ".join(matches)
        )
    log_path = os.path.join("logs", f"{tag}.log")
    if os.path.exists(log_path):
        raise SystemExit(
            f"[semantic-factorial] refusing to overwrite log: {log_path}"
        )

for arm in ("AB", "AC", "BC", "ABC"):
    driver_log = os.path.join(artifact_dir, f"{arm}.driver.log")
    if os.path.exists(driver_log):
        raise SystemExit(
            f"[semantic-factorial] refusing to overwrite driver log: {driver_log}"
        )

source_files = (
    "config.py",
    "dataloaders.py",
    "model_siglip2.py",
    "loss_siglip2.py",
    "train_siglip2.py",
    "scripts/codebook_drop_ablation_fast.py",
    "scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh",
    "scripts/run_semantic_detail_factorial.sh",
)
source_sha256 = {}
for path in source_files:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    source_sha256[path] = digest.hexdigest()

singleton_manifest = os.path.join(
    "artifacts", "semantic_detail_ablation", "screen_manifest.json",
)
require(singleton_manifest)
with open(singleton_manifest, "r", encoding="utf-8") as handle:
    singleton_data = json.load(handle)
digest = hashlib.sha256()
with open(singleton_manifest, "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)

common_a = [
    "--xmodal_commit_skip_global",
    "--cibhash_dynamic_tau_skip_global",
]
common_b = [
    "--text_hash_counterfactual_weight", "0.10",
    "--text_hash_counterfactual_margin", "0.02",
    "--text_hash_counterfactual_warmup_epochs", "1",
]
common_c = ["--cibhash_visual_token_bit_kl"]
manifest = {
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "protocol": (
        "Flickr25k missing 2^3 factorial cells; fixed-E*=4 refit screen; "
        "seed=42; K=128; L=4"
    ),
    "singleton_manifest": os.path.realpath(singleton_manifest),
    "singleton_manifest_sha256": digest.hexdigest(),
    "source_delta_from_singleton": {
        path: {
            "singleton": singleton_data.get("source_sha256", {}).get(path),
            "factorial": current_hash,
        }
        for path, current_hash in source_sha256.items()
        if singleton_data.get("source_sha256", {}).get(path) != current_hash
    },
    "base_cache": os.path.realpath(base_cache),
    "foil_cache": os.path.realpath(foil_cache),
    "local_only_whiten": os.path.realpath(local_whiten),
    "source_sha256": source_sha256,
    "arms": {
        "AB": {
            "gpu": int(gpu_ab),
            "tag": tag_ab,
            "cache": os.path.realpath(foil_cache),
            "flags": common_a + common_b,
        },
        "AC": {
            "gpu": int(gpu_ac),
            "tag": tag_ac,
            "cache": os.path.realpath(base_cache),
            "flags": common_a + common_c,
        },
        "BC": {
            "gpu": int(gpu_bc),
            "tag": tag_bc,
            "cache": os.path.realpath(foil_cache),
            "flags": common_b + common_c,
        },
        "ABC": {
            "gpu": int(gpu_abc),
            "tag": tag_abc,
            "cache": os.path.realpath(foil_cache),
            "flags": common_a + common_b + common_c,
        },
    },
}
with open(
    os.path.join(artifact_dir, "factorial_manifest.json"),
    "w",
    encoding="utf-8",
) as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
print("[semantic-factorial] artifact, result, and cache preflight passed")
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    printf '[semantic-factorial] preflight-only mode complete\n'
    exit 0
fi

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
    "AB" "$GPU_AB" "$FOIL_CACHE" "$FOIL_LOCAL_WHITEN" "$TAG_AB" \
    "--no_visualize --xmodal_commit_skip_global --cibhash_dynamic_tau_skip_global --text_hash_counterfactual_weight 0.10 --text_hash_counterfactual_margin 0.02 --text_hash_counterfactual_warmup_epochs 1" &
arms+=("AB"); pids+=("$!")

run_arm \
    "AC" "$GPU_AC" "$BASE_CACHE" "$LOCAL_WHITEN" "$TAG_AC" \
    "--no_visualize --xmodal_commit_skip_global --cibhash_dynamic_tau_skip_global --cibhash_visual_token_bit_kl" &
arms+=("AC"); pids+=("$!")

run_arm \
    "BC" "$GPU_BC" "$FOIL_CACHE" "$FOIL_WHITEN" "$TAG_BC" \
    "--no_visualize --text_hash_counterfactual_weight 0.10 --text_hash_counterfactual_margin 0.02 --text_hash_counterfactual_warmup_epochs 1 --cibhash_visual_token_bit_kl" &
arms+=("BC"); pids+=("$!")

run_arm \
    "ABC" "$GPU_ABC" "$FOIL_CACHE" "$FOIL_LOCAL_WHITEN" "$TAG_ABC" \
    "--no_visualize --xmodal_commit_skip_global --cibhash_dynamic_tau_skip_global --text_hash_counterfactual_weight 0.10 --text_hash_counterfactual_margin 0.02 --text_hash_counterfactual_warmup_epochs 1 --cibhash_visual_token_bit_kl" &
arms+=("ABC"); pids+=("$!")

for index in "${!arms[@]}"; do
    printf '[semantic-factorial] started %-3s gpu=%s pid=%s\n' \
        "${arms[$index]}" \
        "$(
            case "${arms[$index]}" in
                AB) printf '%s' "$GPU_AB" ;;
                AC) printf '%s' "$GPU_AC" ;;
                BC) printf '%s' "$GPU_BC" ;;
                ABC) printf '%s' "$GPU_ABC" ;;
            esac
        )" \
        "${pids[$index]}"
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf '[semantic-factorial] completed %-3s\n' "${arms[$index]}"
    else
        printf '[semantic-factorial] FAILED %-3s log=%s/%s.driver.log\n' \
            "${arms[$index]}" "$ARTIFACT_DIR" "${arms[$index]}" >&2
        status=1
    fi
done

if [[ "$status" == "0" ]]; then
    "$PY" - "$TAG_AB" "$TAG_AC" "$TAG_BC" "$TAG_ABC" <<'PY'
import glob
import os
import sys

required = (
    "args.txt",
    "compositional_eval.json",
    "evaluation_siglip2_base.json",
    "extract_db.npz",
    "extract_query.npz",
    "log.csv",
    "model_state_dict.pth",
)
for tag in sys.argv[1:]:
    matches = glob.glob(os.path.join("result", f"*+flickr25k_setting1_{tag}+*"))
    if len(matches) != 1:
        raise SystemExit(
            f"[semantic-factorial] expected exactly one result for {tag}, "
            f"found {len(matches)}"
        )
    missing = [
        name for name in required
        if not os.path.isfile(os.path.join(matches[0], name))
    ]
    if missing:
        raise SystemExit(
            f"[semantic-factorial] incomplete result for {tag}: "
            + ", ".join(missing)
        )
    print(f"[semantic-factorial] verified {tag}: {matches[0]}")
PY
fi
exit "$status"
