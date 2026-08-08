#!/usr/bin/env bash
# Re-run the NUS-WIDE ablations under the adopted recipe.
#
# scripts/train_nuswide_v185_sweep_clip.sh now carries
# --sinkhorn_epsilon_init 0.5 (adopted 2026-08-06), so these cells pick it up
# automatically; only the ablation's own flags are passed here. The old rows
# (abA2 .8099, abA4 .8252, bioON .8288) were all trained at eps_init 1.0 and
# are therefore no longer paired with the .8246 headline number.
#
# A2  --disable_text_supervision
# A4  --share_codebook at K=768 (6x128 collapsed into one bank)
# bio --lambda_bio_constraint 10.0  (the `+ L_bio` row of the main table)
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs
NB="--no_gumbel_softmax"
BIO="--lambda_bio_constraint 10.0 --bio_constraint_max_run 3 --bio_constraint_gc_weight 1.0"
QUEUE=(
  "0|nuswide_A_v4|abA2_eps05||--lambda_codon_joint 0.05 $NB --disable_text_supervision"
  "1|nuswide_A_v4|abA4_eps05|768|--lambda_codon_joint 0.05 $NB --share_codebook"
  "2|nuswide_A_v4|bioON_eps05||--lambda_codon_joint 0.05 $NB $BIO"
)
for spec in "${QUEUE[@]}"; do
  IFS='|' read -r g exp tag K aux <<<"$spec"
  echo "[nusabl] GPU$g <- $exp $tag ${K:+K=$K} @ $(date '+%F %T')"
  setsid nohup env ${K:+K=$K} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    bash scripts/sweep_joint_cell.sh "$g" "$exp" "$tag" "$aux" \
    > "logs/SWEEP_${exp}_${tag}.out" 2>&1 < /dev/null &
  sleep 40
done
echo "[nusabl] dispatched @ $(date '+%F %T')"
