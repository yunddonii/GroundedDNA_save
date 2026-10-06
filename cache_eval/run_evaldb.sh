#!/usr/bin/env bash
# Two shards on two idle GPUs, then concat. Usage: run_evaldb.sh <gpuA> <gpuB>
set -u; cd /home/yschoi/gdna_textdiag || exit 90
GA=$1; GB=$2; OUT=/home/yschoi/gdna_textdiag/cache_eval; P=/home/yschoi/.conda/envs/dna_hashing/bin/python
# sample list first (single process, deterministic), then the two shards in parallel
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= $P - <<PY
import sys; sys.argv=["x"]; sys.path.insert(0, "tools")
import qwen3_v4_flickr25k_evaldb as m; print(len(m.sample_paths(1500, 20261007, "$OUT")))
PY
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=$GA $P tools/qwen3_v4_flickr25k_evaldb.py --shard_id 0 --n_shards 2 --out_dir $OUT > $OUT/shard0.log 2>&1 &
p0=$!
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=$GB $P tools/qwen3_v4_flickr25k_evaldb.py --shard_id 1 --n_shards 2 --out_dir $OUT > $OUT/shard1.log 2>&1 &
p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
echo "shard rc $r0 $r1"
[ $r0 -eq 0 ] && [ $r1 -eq 0 ] || exit 1
cat $OUT/flickr25k_qwen3_v4_evaldb1500.shard0.jsonl $OUT/flickr25k_qwen3_v4_evaldb1500.shard1.jsonl > $OUT/flickr25k_qwen3_v4_evaldb1500.jsonl
wc -l $OUT/flickr25k_qwen3_v4_evaldb1500.jsonl
