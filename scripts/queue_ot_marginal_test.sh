#!/usr/bin/env bash
# Test the hypothesis that the dead CIFAR slots come from a WEAK slot-marginal
# constraint in the unbalanced-OT router.
#
# models/semantic_router.py:78   tau_b = lambda_b / (lambda_b + eps)
#
# tau_b=1 is a hard (balanced) column constraint -- every slot must receive its
# prescribed mass; tau_b->0 leaves the slot marginal free. eps sits in the
# DENOMINATOR, so the entropic annealing schedule silently sets the marginal
# enforcement:
#
#   epoch  eps    tau_b (lambda_b=1)
#     0    1.00   0.500      <- half-strength for the whole early phase
#    19    0.79   0.559      <- CIFAR s42 stops here, collapsed
#    39    0.33   0.751
#    59    0.10   0.909
#
# So for every epoch that matters the router is free to abandon a slot whose
# text anchor loses to all patches. CIFAR-10 32x32 has no real "secondary
# object" and no "scene", those two anchors lose, their codebooks stop getting
# gradient, and the collapse locks in. MSCOCO gives every axis real content and
# never collapses despite training 39 epochs.
#
# Two independent ways to raise tau_b:
#   otLB05 / otLB20 : raise lambda_b        -> tau_b 0.833 / 0.952 at eps=1.0
#   otNOANN         : drop the eps annealing (eps=0.1 flat) -> tau_b 0.909
# mscoco_otLB20 is the CONTROL: the fix must not damage an already-healthy set.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs
CB="--no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0"
NB="--no_gumbel_softmax"
QUEUE=(
  "0|cifar_A_v4|otLB05|--lambda_codon_joint 0.03 $CB --sinkhorn_lambda_b 5.0"
  "1|cifar_A_v4|otLB20|--lambda_codon_joint 0.03 $CB --sinkhorn_lambda_b 20.0"
  "2|cifar_A_v4|otNOANN|--lambda_codon_joint 0.03 $CB --sinkhorn_epsilon_init 0.1"
  "3|mscoco_A_v5b|otLB20|--lambda_codon_joint 0.03 $NB --sinkhorn_lambda_b 20.0"
)
echo "[ot] ${#QUEUE[@]} cells on GPUs 0-3 @ $(date '+%F %T')"
for spec in "${QUEUE[@]}"; do
  IFS='|' read -r g exp tag aux <<<"$spec"
  echo "[ot] GPU$g <- $exp $tag"
  setsid nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    bash scripts/sweep_joint_cell.sh "$g" "$exp" "$tag" "$aux" \
    > "logs/SWEEP_${exp}_${tag}.out" 2>&1 < /dev/null &
  sleep 40
done
echo "[ot] dispatched @ $(date '+%F %T')"
