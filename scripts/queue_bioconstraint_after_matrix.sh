#!/usr/bin/env bash
# Run the bio-constraint-ON cells AFTER the CIFAR baseline matrix releases GPUs 4/5.
#
# GPUs 0-3 belong to another project (NSMT gpu_pool2.sh) as of 2026-08-04, and the
# baseline matrix owns 4/5 while it runs. Polling for "free GPUs" would steal a slot
# from the matrix between its cells (that bug was hit on 2026-08-03), so this simply
# WAITS for the matrix process to exit and then dispatches two cells at a time.
#
# Cells = unified recipe + `--lambda_bio_constraint 10.0`. lambda 10 is the value
# validated on Flickr on 2026-07-29: pre-projection validity 53.1% -> 90.1% at
# +0.0007 mAP@R, projection cost -0.0093 -> -0.0014. Known cost: DNA-unique -26%.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs

echo "[bioq] waiting for the baseline matrix to finish @ $(date '+%F %T')"
while pgrep -f "run_baseline_p0_matrix.py" > /dev/null 2>&1; do sleep 120; done
echo "[bioq] matrix done @ $(date '+%F %T')"

NB="--no_gumbel_softmax"
CB="--no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0"
BIO="--lambda_bio_constraint 10.0"

# exp : tag : aux   (per-dataset optimum lambda from the 2026-08-03/04 sweep)
QUEUE=(
  "flickr_A_v4:bioON:--lambda_codon_joint 0.02 $NB $BIO"
  "mscoco_A_v5b:bioON:--lambda_codon_joint 0.03 $NB $BIO"
  "nuswide_A_v4:bioON:--lambda_codon_joint 0.05 $NB $BIO"
  "cifar_A_v4:bioON:--lambda_codon_joint 0.03 $CB $BIO"
)

i=0
while [ "$i" -lt "${#QUEUE[@]}" ]; do
  for g in 4 5; do
    [ "$i" -ge "${#QUEUE[@]}" ] && break
    IFS=: read -r exp tag aux <<<"${QUEUE[$i]}"
    echo "[bioq] GPU$g <- $exp $tag @ $(date '+%F %T')"
    setsid nohup bash scripts/sweep_joint_cell.sh "$g" "$exp" "$tag" "$aux" \
      > "logs/SWEEP_${exp}_${tag}.out" 2>&1 < /dev/null &
    eval "P$g=\$!"
    i=$((i+1)); sleep 60
  done
  # wait for this pair before dispatching the next
  for g in 4 5; do
    v="P$g"; [ -n "${!v:-}" ] && wait "${!v}" 2>/dev/null || true
  done
done
echo "[bioq] all ${#QUEUE[@]} cells done @ $(date '+%F %T')"
