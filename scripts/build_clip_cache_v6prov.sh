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

if [[ ! -f "$POOLED/text_part.f16.npy" ]]; then
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

if [[ ! -f "$TOKENS/text_tokens.f16.npy" ]]; then
    mkdir -p "$TOKENS"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" extract_clip_text_token_features.py \
        --qwen_cache "$QWEN" --donor_dir "$POOLED" --out_dir "$TOKENS"
fi

if [[ ! -f "$TOKENS/text_whiten.npz" ]]; then
    "$PY" scripts/build_text_whiten_matrix.py \
        --cache_dir "$TOKENS" --out "$TOKENS/text_whiten.npz"
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
PYEOF
echo "[cache-v6prov] $DS DONE"
