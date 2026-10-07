#!/usr/bin/env bash
# Stage-2 caption pipeline for ONE dataset on TWO GPUs.
#   tools/stage2_run_dataset.sh <Flickr25k|NUS-WIDE|MS-COCO> <gpuA> <gpuB>
# Steps (each appends "step=<name> rc=<rc> start=<utc> end=<utc> seconds=<s>" to <out>/pipeline.status;
# a step with rc=0 on record is skipped on rerun; the script exits on the first failing step):
#   survey (4 shards, 2 per GPU) -> survey_extra (4 shards) -> concepts (GPU A) -> captions_pilot (4 shards)
#   -> validate_pilot (gates; rc 3 = a gate failed, pipeline stops for a user decision)
#   -> captions_full (4 shards, all train rows) -> concat <out>/<short>_qwen3_v10_trainset.jsonl -> validate_full (report only)
# Run it inside tmux (gdna_p3exec_authority/bin/tmux_run.sh) so the chain survives a session swap.
set -u
if [ $# -ne 3 ]; then echo "usage: $0 <dataset> <gpuA> <gpuB>"; exit 64; fi
DS=$1; GA=$2; GB=$3
WT=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
SHORT=$(env -u PYTHONPATH "$PY" -c "import sys; sys.path.insert(0,'$WT/tools'); from stage2_common import DatasetSpec; print(DatasetSpec('$DS').short)") || exit 65
OUT=$WT/cache_eval/stage2/$SHORT
LOG=$OUT/logs
STATUS=$OUT/pipeline.status
mkdir -p "$LOG"
cd "$WT" || exit 66
echo "dataset=$DS short=$SHORT gpus=$GA,$GB out=$OUT" | tee -a "$LOG/run.log"

run() {                                   # run <step> <function>
  local name=$1; shift
  if grep -q "^step=$name rc=0 " "$STATUS" 2>/dev/null; then echo "skip $name (rc=0 on record)"; return 0; fi
  local start t0 rc
  start=$(date -u +%FT%TZ); t0=$(date +%s)
  "$@"; rc=$?
  echo "step=$name rc=$rc start=$start end=$(date -u +%FT%TZ) seconds=$(( $(date +%s) - t0 ))" >> "$STATUS"
  if [ "$rc" -ne 0 ]; then echo "FAILED step=$name rc=$rc (see $LOG)"; exit "$rc"; fi
}

par4() {                                  # par4 <logtag> <cmd...>   shards 0..3, even on GPU A, odd on GPU B, 30 s apart
  local tag=$1; shift
  local pids=() k gpu rc=0
  for k in 0 1 2 3; do
    gpu=$GA; [ $((k % 2)) -eq 1 ] && gpu=$GB
    ( env -u PYTHONPATH CUDA_VISIBLE_DEVICES=$gpu GDNA_NUM_SEMANTIC_PARTS=5 "$@" --shard_id "$k" --n_shards 4 ) \
        > "$LOG/$tag.shard$k.log" 2>&1 &
    pids+=($!)
    [ "$k" -lt 3 ] && sleep 30
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

one() {                                   # one <logtag> <gpu> <cmd...>
  local tag=$1 gpu=$2; shift 2
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=$gpu GDNA_NUM_SEMANTIC_PARTS=5 "$@" > "$LOG/$tag.log" 2>&1
}

cpu() {                                   # cpu <logtag> <cmd...>  (CLIP text encoder may still use GPU A)
  local tag=$1; shift
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=$GA GDNA_NUM_SEMANTIC_PARTS=5 "$@" 2>&1 | tee "$LOG/$tag.log"
  return "${PIPESTATUS[0]}"
}

step_survey() {
  par4 survey "$PY" tools/stage2_survey.py --dataset "$DS" --n 1000 --seed 20261007 --out_dir "$OUT" || return 1
  env -u PYTHONPATH "$PY" tools/stage2_common.py concat --out "$OUT/survey.jsonl" --expect "$OUT/survey_sample.json" \
      --max_missing_frac 0.02 "$OUT"/survey.shard*.jsonl
}
step_survey_extra() {
  par4 survey_extra "$PY" tools/stage2_survey.py --dataset "$DS" --n 1000 --seed 20261007 \
      --extra_n 500 --extra_seed 20261008 --out_dir "$OUT" || return 1
  env -u PYTHONPATH "$PY" tools/stage2_common.py concat --out "$OUT/survey_extra.jsonl" \
      --expect "$OUT/survey_extra_sample.json" --max_missing_frac 0.02 "$OUT"/survey_extra.shard*.jsonl
}
step_concepts() {
  one concepts "$GA" "$PY" tools/stage2_concepts.py --dataset "$DS" --in_dir "$OUT" --out_dir "$OUT" \
      --batch_images 80 --shuffles 3 --max_new_tokens 4096 || return 1
  [ -s "$OUT/attributes.json" ] || { echo "attributes.json missing"; return 1; }
  env -u PYTHONPATH "$PY" -c "import json; a=json.load(open('$OUT/attributes.json')); print('ATTRIBUTES', json.dumps([(x['key'], x['definition']) for x in a['attributes']])); print('key_map', a['key_map'])" | tee -a "$LOG/run.log"
}
step_captions_pilot() {
  par4 captions_pilot "$PY" tools/stage2_captions.py --dataset "$DS" --tag pilot --attributes "$OUT/attributes.json" \
      --ids all_opt --limit 500 --seed 20261009 --out_dir "$OUT" || return 1
  env -u PYTHONPATH "$PY" tools/stage2_common.py concat --out "$OUT/captions_pilot.jsonl" \
      --expect "$OUT/captions_pilot_sample.json" --max_missing_frac 0.05 "$OUT"/captions_pilot.shard*.jsonl
}
step_validate_pilot() {
  cpu validate_pilot "$PY" tools/stage2_validate.py --dataset "$DS" --captions "$OUT"/captions_pilot.shard*.jsonl \
      --survey "$OUT"/survey.shard*.jsonl "$OUT"/survey_extra.shard*.jsonl --attributes "$OUT/attributes.json" \
      --out_json "$OUT/pilot_validation.json" --out_md "$OUT/pilot_validation.md" --title "Stage-2 pilot $DS" --exit_on_fail
}
step_captions_full() {
  par4 captions_full "$PY" tools/stage2_captions.py --dataset "$DS" --tag full --attributes "$OUT/attributes.json" \
      --ids all_train --out_dir "$OUT" || return 1
  env -u PYTHONPATH "$PY" tools/stage2_common.py concat --out "$OUT/${SHORT}_qwen3_v10_trainset.jsonl" \
      --expect "$OUT/captions_full_sample.json" --max_missing_frac 0.01 "$OUT"/captions_full.shard*.jsonl
}
step_validate_full() {
  cpu validate_full "$PY" tools/stage2_validate.py --dataset "$DS" --captions "$OUT"/captions_full.shard*.jsonl \
      --survey "$OUT"/survey.shard*.jsonl "$OUT"/survey_extra.shard*.jsonl --attributes "$OUT/attributes.json" \
      --out_json "$OUT/full_validation.json" --out_md "$OUT/full_validation.md" --title "Stage-2 full $DS (report only)"
}

run survey          step_survey
run survey_extra    step_survey_extra
run concepts        step_concepts
run captions_pilot  step_captions_pilot
run validate_pilot  step_validate_pilot
run captions_full   step_captions_full
run validate_full   step_validate_full
echo "ALL DONE: $OUT/${SHORT}_qwen3_v10_trainset.jsonl" | tee -a "$LOG/run.log"
exit 0
