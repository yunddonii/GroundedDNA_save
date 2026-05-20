#!/bin/bash
# MSCOCO V4 autopilot. Runs in background; logs to logs/mscoco_autopilot.log
#
# Pipeline:
#   1. Poll MSCOCO V4 caption extraction until both parts reach 5000
#   2. Merge V4 parts into mscoco_qwen_v4.jsonl
#   3. Build mscoco_siglip2_v4plus (extract SigLIP2 visual+text+aug views
#      on full MSCOCO union — train+test+database = ~122K images)
#   4. Launch v63a (v57-setup, K=64) on MSCOCO + v63b (v57-setup, K=128)
#      in parallel
#   5. Wait for both trainings to complete
#   6. Run compositional eval + per-cb breakdown on each
#   7. Write analysis doc into docs/ANALYSIS_mscoco_v63.md
#
# Designed to run as: nohup bash scripts/mscoco_autopilot.sh > logs/mscoco_autopilot.log 2>&1 &

set -u
cd /home/yschoi/GroundedDNA
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
LOG_PREFIX=$(date +%Y%m%d_%H%M%S)
exec > >(stdbuf -oL tee -a /home/yschoi/GroundedDNA/logs/mscoco_autopilot.log) 2>&1

echo "=========================================="
echo "MSCOCO autopilot start: $(date)"
echo "=========================================="

# -------- Stage 1: wait for V4 caption extraction --------
echo "[stage1] waiting for MSCOCO V4 caption extraction (both parts to reach 5000)..."
while true; do
  n0=$(wc -l < /home/yschoi/GroundedDNA/cache/mscoco_qwen_v4_part0.jsonl 2>/dev/null || echo 0)
  n1=$(wc -l < /home/yschoi/GroundedDNA/cache/mscoco_qwen_v4_part1.jsonl 2>/dev/null || echo 0)
  echo "[stage1] $(date '+%H:%M:%S') part0=$n0/5000 part1=$n1/5000"
  if [ "$n0" -ge 5000 ] && [ "$n1" -ge 5000 ]; then
    break
  fi
  # safety check: ensure procs are still running
  alive=$(pgrep -f "mscoco_qwen_v4_part" | wc -l)
  if [ "$alive" -eq 0 ] && [ "$n0" -lt 5000 ]; then
    echo "[stage1] ERROR: V4 extraction processes died with incomplete cache. Abort."
    exit 1
  fi
  sleep 600
done
echo "[stage1] V4 extraction complete."

# -------- Stage 2: merge V4 parts --------
echo "[stage2] merging V4 parts into mscoco_qwen_v4.jsonl ..."
$PY <<'PY'
import json, os
sources = [
    "/home/yschoi/GroundedDNA/cache/mscoco_qwen_v4_part0.jsonl",
    "/home/yschoi/GroundedDNA/cache/mscoco_qwen_v4_part1.jsonl",
]
out = {}
for s in sources:
    if not os.path.exists(s):
        print(f"skip missing {s}")
        continue
    n_in = 0; n_added = 0
    with open(s) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            iid = d.get("image_id")
            cb = d.get("codebook_texts", {}) or {}
            err = d.get("error")
            n_in += 1
            # only keep entries with valid (non-empty, non-error) captions
            if err or not iid:
                continue
            if not any(v and v.strip().lower() != "none" for v in cb.values()):
                continue
            if iid not in out:
                out[iid] = d
                n_added += 1
    print(f"  {os.path.basename(s)}: read {n_in}, valid added {n_added}")
print(f"Total merged unique valid: {len(out)}")
merged_p = "/home/yschoi/GroundedDNA/cache/mscoco_qwen_v4.jsonl"
with open(merged_p, "w") as f:
    for iid, d in out.items():
        f.write(json.dumps(d) + "\n")
print(f"Wrote {merged_p}  ({len(out)} entries)")
PY

# -------- Stage 3: build mscoco_siglip2_v4plus --------
# Strategy: re-run full extract_siglip2_features for MSCOCO with V4 cache + aug
# views. This is needed because the existing mscoco_siglip2/ has V1 text and no
# aug views (paired-aug NtXent needs aug views).
echo "[stage3] extracting MSCOCO SigLIP2 cache (visual + text V4 + aug views)..."
echo "[stage3] this covers all of train+test+database = ~122K images, ~1-2h"

CUDA_VISIBLE_DEVICES=0 $PY /home/yschoi/GroundedDNA/extract_siglip2_features.py \
  --mode pathlist \
  --pathlist_root ./dataset/MSCOCO \
  --pathlist_setting setting1 \
  --qwen_cache_path ./cache/mscoco_qwen_v4.jsonl \
  --cache_dir ./cache/mscoco_siglip2_v4plus \
  --batch_size 128 \
  --save_aug_views 2 \
  --dtype float16 \
  --num_workers 8
echo "[stage3] MSCOCO SigLIP2 v4plus cache built."

# Sanity check
$PY <<'PY'
import numpy as np, os, json
d = "/home/yschoi/GroundedDNA/cache/mscoco_siglip2_v4plus"
for f in ["text_part.f16.npy", "visual_tokens.f16.npy", "visual_global.f16.npy",
          "visual_tokens_aug0.f16.npy", "visual_tokens_aug1.f16.npy",
          "has_text.bool.npy", "image_ids.json", "meta.json"]:
    p = os.path.join(d, f)
    if os.path.exists(p):
        size_mb = os.path.getsize(p) / 1e6
        print(f"  {f}: OK ({size_mb:.1f} MB)")
    else:
        print(f"  {f}: MISSING")
ids = json.load(open(os.path.join(d, "image_ids.json")))
ht = np.load(os.path.join(d, "has_text.bool.npy"))
print(f"  image_ids: N={len(ids)}, has_text True: {int(ht.sum())}")
PY

# -------- Stage 4: launch v63a (K=64) + v63b (K=128) in parallel --------
echo "[stage4] launching v63a (K=64) + v63b (K=128) on MSCOCO ..."
TS=$(date +%H%M%S)

launch_mscoco() {
  local TAG=$1; local K=$2; local GPU=$3
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY /home/yschoi/GroundedDNA/train_siglip2.py \
    -nd 0 \
    --dataset MSCOCO --dataset_dir /home/yschoi/GroundedDNA/dataset \
    --qwen_text_cache_path /home/yschoi/GroundedDNA/cache/mscoco_qwen_v4.jsonl \
    --siglip2_feature_cache_dir /home/yschoi/GroundedDNA/cache/mscoco_siglip2_v4plus \
    --tag $TAG \
    -e 60 -bs 64 --proj_lr 0.001 \
    --num_codebooks 6 --codebook_size $K \
    --c_global_source siglip2_global --disable_global_gate \
    --use_gumbel_softmax --gumbel_tau_init 2.0 --gumbel_tau_final 0.3 --gumbel_tau_anneal \
    --codebook_update ema --codebook_ema_decay 0.99 --codebook_revive \
    --use_paired_aug_ntxent --ntxent_mode per_codebook --ntxent_temperature 0.5 \
    --ntxent_dynamic_tau --ntxent_dynamic_tau_alpha 0.3 \
    --lambda_wasserstein 0.05 --lambda_ntxent 1.0 \
    --lambda_hash 0 --lambda_hash_hard 0 \
    --hash_target_mode siglip_cos_topk --siglip_cos_pos_rate 0.2 \
    --lr_scheduler cosine --lr_eta_min 1e-5 \
    --sinkhorn_epsilon_init 1.0 --sinkhorn_epsilon_final 0.1 \
    --routing_topp 0.7 \
    --eval_every 10 --extract_batch_size 256 -s --eval \
    > /home/yschoi/GroundedDNA/logs/${TAG}_${TS}.log 2>&1 &
  disown
  echo "[stage4] $TAG (K=$K, GPU=$GPU) PID=$!"
}
launch_mscoco v63a_mscoco_v57setup_K64  64  0
launch_mscoco v63b_mscoco_v57setup_K128 128 1

echo "[stage4] both v63a + v63b launched. Waiting for completion..."

# -------- Stage 5: wait for both trainings to complete --------
while true; do
  alive_a=$(pgrep -f "v63a_mscoco_v57setup_K64" | wc -l)
  alive_b=$(pgrep -f "v63b_mscoco_v57setup_K128" | wc -l)
  if [ "$alive_a" -eq 0 ] && [ "$alive_b" -eq 0 ]; then
    echo "[stage5] both v63a + v63b processes ended."
    break
  fi
  echo "[stage5] $(date '+%H:%M:%S') v63a alive=$alive_a, v63b alive=$alive_b"
  sleep 600
done

# Wait extra 60s for extract+eval cleanup
sleep 60

# -------- Stage 6: run compositional eval on both --------
echo "[stage6] running compositional_eval.py on v63a + v63b ..."
for TAG in v63a_mscoco_v57setup_K64 v63b_mscoco_v57setup_K128; do
  RESULT=$(ls -d /home/yschoi/GroundedDNA/result/*${TAG}* 2>/dev/null | head -1)
  if [ -z "$RESULT" ]; then echo "[stage6] $TAG: result dir not found, skip"; continue; fi
  echo "[stage6] $TAG: running compositional_eval.py ..."
  CUDA_VISIBLE_DEVICES=0 $PY /home/yschoi/GroundedDNA/compositional_eval.py \
    --result_dir $RESULT \
    --cache_dir /home/yschoi/GroundedDNA/cache/mscoco_siglip2_v4plus \
    --dataset_root /home/yschoi/GroundedDNA/dataset/MSCOCO \
    --min_cluster 5 \
    --skip_grids 2>&1 | tail -20
done

# -------- Stage 7: emit final summary marker --------
echo "[stage7] all stages complete: $(date)"
echo "[stage7] check logs/v63*_${TS}.log for training logs"
echo "[stage7] result dirs: ls -d /home/yschoi/GroundedDNA/result/*v63*"
touch /home/yschoi/GroundedDNA/logs/MSCOCO_AUTOPILOT_DONE
