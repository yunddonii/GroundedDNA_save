#!/usr/bin/env bash
# The twelve code-to-own-axis probes of contract v3 section 8.2 (audit 729), one supervised probe at a
# time (the supervisor's ledger lock admits one), in a fixed order, stopping at the first nonzero exit
# and exiting with it. NON-EXECUTABLE until the audit ledger holds a `probe` approval line naming
# request 748851142a8734703c076bbccd52af8c383dba14ee105d29235c55c92c476c44.
# Usage: probe_chain_ancP7_v7.sh <one free GPU index> <approving ledger section>
set -u
if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 <gpu index> <approving ledger section>" >&2
  exit 2
fi
GPU=$1
SECTION=$2
if [[ "${GDNA_NUM_SEMANTIC_PARTS:-}" != 5 || -n "${PYTHONPATH+set}" || -n "${CUDA_VISIBLE_DEVICES+set}" ]]; then
  echo "environment: GDNA_NUM_SEMANTIC_PARTS=5 with PYTHONPATH and CUDA_VISIBLE_DEVICES unset" >&2
  exit 2
fi
cd /data/yschoi/gdna_anchor_confirm_v1 || exit 2
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
REC=/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation
OUT=$REC/ancP7_probes
MANIFEST=$REC/authority_manifest_v7.json
MANIFEST_SHA256=c061296309fe41203205dff65de3a1afb843c950e93c0614dfd27a8216a90128
SOURCES=$REC/ancP7_probe_sources.json
SOURCES_SHA256=39b304e92a94f879a71bdbfdc8b2cf9bce3209e81e09bdf460d047a4ca2b8702
SELECTION=$REC/ancS7_selected_n.json
SELECTION_SHA256=5cda7adb055ed126efb0a8ec198e06d1176ccdc57e7d38fe35ff74b74a4a92ff
COORDINATES=(
  flickr25k:anchors:4:42 flickr25k:anchors:4:43 flickr25k:anchors:4:44
  cifar10:anchors:4:42 cifar10:anchors:4:43 cifar10:anchors:4:44
  nuswide:anchors:4:42 nuswide:anchors:4:43 nuswide:anchors:4:44
  mscoco:anchors:39:42 mscoco:anchors:39:43 mscoco:anchors:39:44
)
# a fresh output directory: an earlier attempt's outputs are never resumed or overwritten
mkdir "$OUT" || { echo "$OUT exists or cannot be made; a re-run needs a new namespace" >&2; exit 2; }
for coordinate in "${COORDINATES[@]}"; do
  IFS=: read -r dataset arm n seed <<< "$coordinate"
  echo "[probe-chain] $(date -u +%FT%TZ) start $coordinate on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/anchor_confirm_supervisor.py \
    --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA256" \
    --stage probe --planned-cells 0 --label "ancP7:$coordinate" --watch-path "$REC" -- \
    "$PY" scripts/anchor_confirm_code_axis.py \
    --sources "$SOURCES" --sources-sha256 "$SOURCES_SHA256" \
    --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA256" \
    --selection "$SELECTION" --selection-sha256 "$SELECTION_SHA256" \
    --approval-section "$SECTION" --coordinate "$coordinate" \
    --out "$OUT/${dataset}_${arm}_N${n}_s${seed}.json" --device cuda:0
  rc=$?
  echo "[probe-chain] $(date -u +%FT%TZ) end $coordinate rc=$rc"
  if [[ $rc -ne 0 ]]; then
    echo "[probe-chain] stopped at $coordinate; the remaining probes did not start" >&2
    exit "$rc"
  fi
done
echo "[probe-chain] $(date -u +%FT%TZ) 12 of 12 probes written to $OUT"
exit 0
