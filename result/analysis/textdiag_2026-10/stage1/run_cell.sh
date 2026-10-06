#!/usr/bin/env bash
# Run one off-protocol training cell of the text-path line. Usage: run_cell.sh <gpu> <cmdfile>
# The cmd file holds one rendered "python train_siglip2.py ..." line (from build_cmd.py).
set -u
GPU=$1; CMDFILE=$2
cd /home/yschoi/gdna_textdiag || exit 90
CMD=$(cat "$CMDFILE")
echo "GPU=$GPU"; echo "CMD=$CMD"; echo "HEAD=$(git rev-parse --short HEAD)"; date
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$GPU" GDNA_NUM_SEMANTIC_PARTS=5 \
  /home/yschoi/.conda/envs/dna_hashing/bin/${CMD}
rc=$?
echo "rc=$rc"; date
exit "$rc"
