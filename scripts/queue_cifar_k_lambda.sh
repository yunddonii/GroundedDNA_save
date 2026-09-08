#!/usr/bin/env bash
# Separate the two candidate causes of the CIFAR-10 topp69 drop.
#
# (i)  K=64 leaves no headroom. CIFAR baseline already uses 64/64 codewords in
#      every slot, so when a wider nucleus makes pooled features less
#      distinguishable the quantizer cannot hold 64 distinct assignments and
#      codeword usage collapses (color_texture 64 -> 49), taking the codon
#      vocabulary with it (45 -> 37). NUS-WIDE at K=128 keeps 128/128 and its
#      codons stay pinned at the 4^3 = 64 ceiling (ratio exactly 2.00), which
#      is why removing top-p costs it almost nothing (mAP -0.0066).
#      -> t69K128 gives CIFAR the same 2x headroom.
#
# (ii) lambda_codon_joint = 0.03 was tuned under the NARROW nucleus. A wider
#      nucleus makes pooled features more similar, so the joint-diversity
#      pressure needed to hold codon spread should be higher.
#      -> t69jd006 / t69jd010 raise it at fixed K=64.
#
# Reading: if t69K128 recovers and the lambda cells do not, (i) is the cause;
# if both recover, they are separable contributions; if neither, the loss is
# intrinsic to the wide nucleus.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs
QUEUE=(
  "cifar_A_v4|t69K128|128|--lambda_codon_joint 0.03 --no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0 --routing_adaptive_topp_min 0.6 --routing_adaptive_topp_max 0.95"
  "cifar_A_v4|t69jd006|64|--lambda_codon_joint 0.06 --no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0 --routing_adaptive_topp_min 0.6 --routing_adaptive_topp_max 0.95"
  "cifar_A_v4|t69jd010|64|--lambda_codon_joint 0.10 --no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0 --routing_adaptive_topp_min 0.6 --routing_adaptive_topp_max 0.95"
)
ALLOWED=(2 5 0 1 3 4)
declare -A PID
i=0
while [ "$i" -lt "${#QUEUE[@]}" ]; do
  for g in "${ALLOWED[@]}"; do
    if [ -n "${PID[$g]:-}" ] && kill -0 "${PID[$g]}" 2>/dev/null; then continue; fi
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$u" -gt 500 ] && continue
    [ "$i" -ge "${#QUEUE[@]}" ] && break
    IFS='|' read -r exp tag K aux <<<"${QUEUE[$i]}"
    echo "[kl] GPU$g <- $exp $tag K=$K @ $(date '+%F %T')"
    setsid nohup env K=$K HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      bash scripts/sweep_joint_cell.sh "$g" "$exp" "$tag" "$aux" \
      > "logs/SWEEP_${exp}_${tag}.out" 2>&1 < /dev/null &
    PID[$g]=$!
    i=$((i+1)); sleep 45
  done
  sleep 60
done
echo "[kl] all dispatched @ $(date '+%F %T')"
