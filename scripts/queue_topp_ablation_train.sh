#!/usr/bin/env bash
# The per-patch adaptive top-p nucleus mask is the PROXIMATE cause of the dead
# slots: on the same checkpoint, disabling it at inference lifts CIFAR
# secondary_object 0.90 -> 13.23 % and scene_type 1.12 -> 13.46 %, and every
# dead slot in every dataset/seed revives the same way. The underlying Sinkhorn
# plan has no dead slots at all (all local slots land at 13.2-20.4 %).
#
# Mechanism: keep_sorted[...,0]=True guarantees rank-1 survival, so a slot that
# is never any patch's argmax depends entirely on the cumulative nucleus
# tau = 0.3 + 0.4*(1-p_max); for a confident patch tau=0.34 keeps ONE slot.
# top1 % predicts death exactly: CIFAR s43 (survives) has 7.7/7.6 %, s42 and
# s44 (both dead) have 0.5/1.0 % and 0.4/0.3 %.
#
# top-p exists to sharpen routing, so removing it may cost retrieval. That
# trade-off is the result these cells are for -- do not assume it is free.
#
# Waits for the OT cells to release GPUs 0-2 rather than polling for "free"
# GPUs, which previously stole a slot from a running matrix between its cells.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs
CB="--no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0"

for t in cifar_A_v4_otLB05 cifar_A_v4_otLB20 cifar_A_v4_otNOANN; do
  while ! grep -q "\[sweep .*\] \(DONE\|ERROR\)" "logs/SWEEP_${t}.out" 2>/dev/null; do sleep 60; done
  echo "[topp-train] $t released @ $(date '+%F %T')"
done

QUEUE=(
  "0|cifar_A_v4|noTOPP|--lambda_codon_joint 0.03 $CB --no_routing_adaptive_topp"
  "1|cifar_A_v4|topp69|--lambda_codon_joint 0.03 $CB --routing_adaptive_topp_min 0.6 --routing_adaptive_topp_max 0.95"
  "2|nuswide_A_v4|noTOPP|--lambda_codon_joint 0.05 --no_gumbel_softmax --no_routing_adaptive_topp"
)
for spec in "${QUEUE[@]}"; do
  IFS='|' read -r g exp tag aux <<<"$spec"
  echo "[topp-train] GPU$g <- $exp $tag @ $(date '+%F %T')"
  setsid nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    bash scripts/sweep_joint_cell.sh "$g" "$exp" "$tag" "$aux" \
    > "logs/SWEEP_${exp}_${tag}.out" 2>&1 < /dev/null &
  sleep 40
done
echo "[topp-train] dispatched @ $(date '+%F %T')"
