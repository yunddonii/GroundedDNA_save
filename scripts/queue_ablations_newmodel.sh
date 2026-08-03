#!/usr/bin/env bash
# Re-run the draft's causal ablations on the NEW unified-recipe model.
#
# Everything in DRAFT_GROUNDEDDNA_PAPER_KO.md section 4.8 was measured on
# pre-P0 / pre-bio checkpoints of an older recipe, and the draft itself marks
# those rows "main causal table에 사용하지 않는다". This queue reproduces them
# under the current protocol (P0 2-stage, bio-projected, unified recipe) so the
# causal table can finally be filled with admissible numbers.
#
#   A2  --disable_text_supervision   "no text supervision"
#   A4  --share_codebook K=768       "six slots share one bank"
#
# Base recipe per dataset = the 2026-08-03/04 sweep optimum:
#   MSCOCO   lam 0.03 + noGumbel
#   NUSWIDE  lam 0.05 + noGumbel
#   Flickr   lam 0.02 + noGumbel
#   CIFAR    lam 0.03 + noGumbel + bijection OFF
#
# GPUs 4/5 belong to the baseline multi-seed matrix; this queue must not touch
# them (they read as free between matrix cells -- observed 2026-08-03).
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs docs/sweep_rows

ALLOWED="${ALLOWED_GPUS:-0 1 2 3}"
NB="--no_gumbel_softmax"
CB="--no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0"

# exp : tag : aux
QUEUE=(
  "mscoco_A_v5b:abA2:--lambda_codon_joint 0.03 $NB --disable_text_supervision"
  "nuswide_A_v4:abA2:--lambda_codon_joint 0.05 $NB --disable_text_supervision"
  "flickr_A_v4:abA2:--lambda_codon_joint 0.02 $NB --disable_text_supervision"
  "cifar_A_v4:abA2:--lambda_codon_joint 0.03 $CB --disable_text_supervision"
  "mscoco_A_v5b:abA4:--lambda_codon_joint 0.03 $NB --share_codebook"
  "nuswide_A_v4:abA4:--lambda_codon_joint 0.05 $NB --share_codebook"
  "flickr_A_v4:abA4:--lambda_codon_joint 0.02 $NB --share_codebook"
  "cifar_A_v4:abA4:--lambda_codon_joint 0.03 $CB --share_codebook"
)
# A4 needs the matched-capacity shared bank; K is an env override in the runner.
declare -A KOVR=( ["abA4"]=768 )

echo "[abl] ${#QUEUE[@]} cells @ $(date '+%F %T')  allowed GPUs: $ALLOWED"

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '($2+0) < 2000 && ($3+0) < 15 {print $1}' \
    | grep -xE "$(echo $ALLOWED | tr ' ' '|')" | sort -n
}

declare -A RUNNING
i=0
while [ "$i" -lt "${#QUEUE[@]}" ]; do
  for g in "${!RUNNING[@]}"; do
    kill -0 "${RUNNING[$g]}" 2>/dev/null || unset "RUNNING[$g]"
  done
  g1=$(free_gpus); sleep 30; g2=$(free_gpus)
  for g in $(comm -12 <(echo "$g1") <(echo "$g2")); do
    [ -n "${RUNNING[$g]:-}" ] && continue
    [ "$i" -ge "${#QUEUE[@]}" ] && break
    IFS=: read -r exp tag aux <<<"${QUEUE[$i]}"
    K="${KOVR[$tag]:-}"
    echo "[abl] GPU$g <- $exp $tag ${K:+K=$K} @ $(date '+%F %T')"
    setsid nohup env ${K:+K=$K} bash scripts/sweep_joint_cell.sh "$g" "$exp" "$tag" "$aux" \
      > "logs/SWEEP_${exp}_${tag}.out" 2>&1 < /dev/null &
    RUNNING[$g]=$!
    i=$((i+1)); sleep 60
  done
  sleep 30
done
echo "[abl] all ${#QUEUE[@]} cells dispatched @ $(date '+%F %T')"
