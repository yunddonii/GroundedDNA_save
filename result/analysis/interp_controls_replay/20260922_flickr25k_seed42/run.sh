#!/usr/bin/env bash
# Dry-run replay of one Reinforcement-A cell through the installed driver; nothing is written by the driver.
set -uo pipefail
cd /home/yschoi/GroundedDNA
D=result/analysis/interp_controls_replay/20260922_flickr25k_seed42
start=$(date '+%F %T %z')
/home/yschoi/.conda/envs/dna_hashing/bin/python scripts/interp_control_a_partitions.py \
  --out "$D/never_created" --cells flickr25k_seed42 --allow-partial --dry-run > "$D/stdout.txt" 2> "$D/stderr.txt"
rc=$?
end=$(date '+%F %T %z')
/home/yschoi/.conda/envs/dna_hashing/bin/python - "$rc" "$start" "$end" <<'PY'
import hashlib, json, pathlib, sys
D = pathlib.Path("result/analysis/interp_controls_replay/20260922_flickr25k_seed42")
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
json.dump({"command": "python scripts/interp_control_a_partitions.py --out <D>/never_created --cells flickr25k_seed42 --allow-partial --dry-run",
           "rc": int(sys.argv[1]), "start": sys.argv[2], "end": sys.argv[3],
           "driver_sha256": sha("scripts/interp_control_a_partitions.py"),
           "probe_sha256": sha("scripts/heldout_codon_decoding.py"),
           "stdout_sha256": sha(D / "stdout.txt"), "stderr_sha256": sha(D / "stderr.txt"),
           "out_created": (D / "never_created").exists(),
           "stored_record": "result/analysis/interp_controls/A_partitions_20260917/flickr25k_seed42.json",
           "stored_record_sha256": sha("result/analysis/interp_controls/A_partitions_20260917/flickr25k_seed42.json")},
          open(D / "replay_record.json", "w"), indent=1)
PY
exit "$rc"
