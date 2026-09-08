#!/usr/bin/env bash
# Post-audit recovery pipeline (docs/CRITICAL_PYTHON_AUDIT_AND_RERUN_PLAN_2026-08-04.md §5).
#
# Runs, in the dependency order the audit prescribes, everything that needs a GPU:
#
#   1. wait for the in-flight bio-constraint queue (normal code path, unaffected by
#      the audit -- §7 says it does not need re-running)
#   2. CIBHash corrected matrix, 24 cells = 4 datasets x {36,48} bits x seeds {42,43,44}
#      (F3: the old head was 18,468 params vs 562,212 corrected, plus horizon 100->60
#      and a fixed-Adam schedule, so every historical CIBHash number is retired)
#   3. A2 / A4 re-runs, 8 cells, under NEW tags `*_auditfix_20260804`
#      (F1: disable_text_supervision leaked cached text into adapter/quantizer/codon
#       head; F2: share_codebook looked up bank 0 but EMA-updated per-slot banks.
#       Both corrupted E* selection, so old E* must not be reused and old tags must
#       not be overwritten.)
#   4. aggregate the CIBHash matrix
#
# GPUs 0-3 belong to another project (NSMT); only 4/5 are ours. Every stage waits on
# the previous PROCESS rather than polling for "free" GPUs, which would steal a slot
# from a running job (bug hit 2026-08-03).
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python

wait_for() {  # $1 = pgrep pattern, $2 = label
  echo "[fix] waiting for $2 @ $(date '+%F %T')"
  while pgrep -f "$1" > /dev/null 2>&1; do sleep 120; done
  echo "[fix] $2 finished @ $(date '+%F %T')"
}

wait_for "queue_bioconstraint_after_matrix.sh" "bio-constraint queue"
wait_for "sweep_joint_cell.sh [45] "            "its last cell"

# ---- 2) CIBHash corrected matrix, 24 cells ---------------------------------
echo "[fix] CIBHash corrected matrix (24 cells) @ $(date '+%F %T')"
$PY scripts/run_baseline_p0_matrix.py \
  --gpus 4 5 --panel u0 --variants cibhash \
  --datasets Flickr25k MSCOCO NUSWIDE CIFAR10 --bits 36 48 --seeds 42 43 44 \
  --model-root params_baseline/p0_cibhash_sourcefix_20260804 \
  --result-root result_baseline/p0_cibhash_sourcefix_20260804 \
  --compress-root compress_baseline/p0_cibhash_sourcefix_20260804 \
  --log-root logs/p0_cibhash_sourcefix_20260804

echo "[fix] aggregating CIBHash @ $(date '+%F %T')"
$PY scripts/aggregate_baseline_p0_matrix.py \
  --result-root result_baseline/p0_cibhash_sourcefix_20260804 \
  --seeds 42 43 44 --panels u0 \
  --out-json docs/baseline_cibhash_sourcefix_20260804.json \
  --out-markdown docs/baseline_cibhash_sourcefix_20260804.md || true

# ---- 3) A2 / A4 with NEW tags ----------------------------------------------
T=auditfix_20260804
NB="--no_gumbel_softmax"
CB="--no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0"

run_pair() {  # $1,$2 = "exp|tag|K|aux" specs dispatched to GPU 4 and 5
  local i=0
  for spec in "$@"; do
    IFS='|' read -r exp tag K aux <<<"$spec"
    local g=$(( i == 0 ? 4 : 5 ))
    echo "[fix] GPU$g <- $exp $tag ${K:+K=$K} @ $(date '+%F %T')"
    setsid nohup env ${K:+K=$K} bash scripts/sweep_joint_cell.sh "$g" "$exp" "$tag" "$aux" \
      > "logs/SWEEP_${exp}_${tag}.out" 2>&1 < /dev/null &
    eval "PID$g=\$!"
    i=$((i+1)); sleep 60
  done
  for g in 4 5; do v="PID$g"; [ -n "${!v:-}" ] && wait "${!v}" 2>/dev/null || true; done
}

run_pair "mscoco_A_v5b|abA2_$T||--lambda_codon_joint 0.03 $NB --disable_text_supervision" \
         "nuswide_A_v4|abA2_$T||--lambda_codon_joint 0.05 $NB --disable_text_supervision"
run_pair "flickr_A_v4|abA2_$T||--lambda_codon_joint 0.02 $NB --disable_text_supervision" \
         "cifar_A_v4|abA2_$T||--lambda_codon_joint 0.03 $CB --disable_text_supervision"
run_pair "mscoco_A_v5b|abA4_$T|768|--lambda_codon_joint 0.03 $NB --share_codebook" \
         "nuswide_A_v4|abA4_$T|768|--lambda_codon_joint 0.05 $NB --share_codebook"
run_pair "flickr_A_v4|abA4_$T|768|--lambda_codon_joint 0.02 $NB --share_codebook" \
         "cifar_A_v4|abA4_$T|768|--lambda_codon_joint 0.03 $CB --share_codebook"

echo "[fix] PIPELINE DONE @ $(date '+%F %T')"
