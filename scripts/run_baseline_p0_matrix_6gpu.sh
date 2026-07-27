#!/usr/bin/env bash
# Convenience wrapper for the labelled P0 baseline matrix (SEEDS defaults to 42).
# No method-specific horizon, batch size, optimizer, or schedule is overridden.
# Environment filters: METHOD_GROUP, BITS, DATASETS, SEEDS.
# Example excluding an independently running CroVCA queue:
#   METHOD_GROUP=modern_no_crovca BITS='36 48' DATASETS='Flickr25k MSCOCO' \
#     bash scripts/run_baseline_p0_matrix_6gpu.sh 0 1 2 3 4 5
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${GROUNDEDDNA_PYTHON:-/home/yschoi/.conda/envs/dna_hashing/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 2
fi

if [[ "$#" -eq 0 ]]; then
  set -- 0 1 2 3 4 5
fi

exec "$PYTHON_BIN" scripts/run_baseline_p0_matrix.py --gpus "$@"
