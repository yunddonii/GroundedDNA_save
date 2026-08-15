#!/usr/bin/env bash
# Rebuild one dataset's CLIP cache WITH strict provenance, under /data.
#
# The four caches in use predate `hf_provenance` and `canonical_transform`, and
# two CLIP snapshots sit in the HF hub dir, so nothing proves which one built
# them. `paper_table_eligible=True` requires that proof, hence D-A: regenerate
# and re-run. `extract_clip_features.py` already emits both fields and fails
# closed on a non-immutable commit hash, so no builder change is needed.
#
# Output goes to /data (909 GB free) rather than the home partition (61 GB),
# so nothing has to be deleted and every existing artefact stays intact.
#
# Usage: scripts/build_clip_cache_v6prov.sh <gpu> <dataset>
set -euo pipefail

PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
REPO=/home/yschoi/GroundedDNA
ROOT=/data/yschoi/groundeddna_cache_v6prov
cd "$REPO"

GPU="${1:?usage: build_clip_cache_v6prov.sh <gpu> <dataset>}"
DS="${2:?dataset: Flickr25k|MSCOCO|NUSWIDE|CIFAR10}"

case "$DS" in
Flickr25k) SLUG=flickr25k; QWEN="$REPO/cache/flickr25k_qwen3_v4_trainset.jsonl" ;;
MSCOCO)    SLUG=mscoco;    QWEN="$REPO/cache/mscoco_qwen3_v5b_trainset.jsonl" ;;
NUSWIDE)   SLUG=nuswide;   QWEN="$REPO/cache/nuswide_qwen3_v4_trainset.jsonl" ;;
CIFAR10)   SLUG=cifar10;   QWEN="$REPO/cache/cifar10_qwen_v4.jsonl" ;;
*) echo "unknown dataset $DS" >&2; exit 2 ;;
esac

POOLED="$ROOT/${SLUG}_clip"
TOKENS="$ROOT/${SLUG}_clip_tokens"
[[ -f "$QWEN" ]] || { echo "missing caption cache: $QWEN" >&2; exit 2; }

echo "[cache-v6prov] $DS on gpu $GPU"
echo "  captions: $QWEN ($(sha256sum "$QWEN" | cut -c1-12)...)"
echo "  pooled  : $POOLED"
echo "  tokens  : $TOKENS"

# CIFAR-10 is stored as torchvision batches, not as a setting1 path list, so it
# has its own extractor. Both emit the same meta contract, including
# `hf_provenance` and `canonical_transform`; only the enumeration differs.
if [[ ! -f "$POOLED/text_part.f16.npy" ]]; then
    if [[ "$DS" == "CIFAR10" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" extract_clip_features_cifar10.py \
            --cifar10_root "$REPO/dataset/CIFAR10" \
            --qwen_cache_path "$QWEN" \
            --cache_dir "$POOLED" \
            --save_aug_views 2 \
            --batch_size 128 \
            --image_size 224
    else
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" extract_clip_features.py \
            --mode pathlist \
            --pathlist_root "$REPO/dataset/$DS" \
            --pathlist_setting setting1 \
            --qwen_cache_path "$QWEN" \
            --cache_dir "$POOLED" \
            --save_aug_views 2 \
            --batch_size 128 \
            --image_size 224
    fi
fi

if [[ ! -f "$TOKENS/text_tokens.f16.npy" ]]; then
    mkdir -p "$TOKENS"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" extract_clip_text_token_features.py \
        --qwen_cache "$QWEN" --donor_dir "$POOLED" --out_dir "$TOKENS"
fi

# The whitening matrix is fitted BEFORE training, so it must see the
# optimization-train rows only. Fitting it on every cache row lets it observe
# the validation rows that pick the epoch and the query/DB rows that are the
# reported number -- a transductive leak. The first version of this script
# omitted the restriction entirely, so all four rebuilt caches were written with
# `leakage_free_fit: false`.
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-42}"

if [[ ! -f "$TOKENS/opt_train_rows.npy" ]]; then
    "$PY" scripts/build_opt_train_rows.py \
        --dataset "$DS" --dataset_dir "$REPO/dataset" --cache_dir "$TOKENS" \
        --val_split_ratio "$VAL_RATIO" --val_split_seed "$VAL_SEED" \
        --out "$TOKENS/opt_train_rows.npy"
fi
if [[ ! -f "$TOKENS/train_all_rows.npy" ]]; then
    "$PY" scripts/build_opt_train_rows.py \
        --dataset "$DS" --dataset_dir "$REPO/dataset" --cache_dir "$TOKENS" \
        --all_train --out "$TOKENS/train_all_rows.npy"
fi

# Rebuild whenever the existing matrix is absent OR was fitted without the
# restriction -- "the file exists" is not the property that matters.
WHITEN_OK=0
if [[ -f "$TOKENS/text_whiten.npz.meta.json" ]]; then
    WHITEN_OK=$("$PY" -c "import json,sys; print(1 if json.load(open(sys.argv[1])).get('leakage_free_fit') else 0)" \
        "$TOKENS/text_whiten.npz.meta.json")
fi
if [[ ! -f "$TOKENS/text_whiten.npz" || "$WHITEN_OK" != "1" ]]; then
    "$PY" scripts/build_text_whiten_matrix.py \
        --cache_dir "$TOKENS" --out "$TOKENS/text_whiten.npz" \
        --row_index_npy "$TOKENS/opt_train_rows.npy"
fi

# Fail loudly here rather than at training time: the whole point of the rebuild
# is that these two fields exist.
"$PY" - "$POOLED" "$TOKENS" <<'PYEOF'
import json, sys
for d in sys.argv[1:3]:
    meta = json.load(open(f"{d}/meta.json"))
    missing = [k for k in ("canonical_transform", "hf_provenance")
               if not isinstance(meta.get(k), dict)]
    if missing:
        raise SystemExit(f"{d}/meta.json still lacks {missing}")
    rev = meta["hf_provenance"].get("model_revision")
    print(f"  OK {d}: revision={rev}")

# The whitening fit is as much a part of eligibility as the backbone snapshot.
whiten_meta = f"{sys.argv[2]}/text_whiten.npz.meta.json"
wm = json.load(open(whiten_meta))
if not wm.get("leakage_free_fit"):
    raise SystemExit(f"{whiten_meta}: leakage_free_fit is false; the whitening "
                     f"matrix saw rows outside the optimization-train split")
print(f"  OK {sys.argv[2]}/text_whiten.npz: fitted on {wm['rows_used']} "
      f"optimization-train rows")
PYEOF
echo "[cache-v6prov] $DS DONE"
