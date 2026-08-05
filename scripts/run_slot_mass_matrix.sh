#!/usr/bin/env bash
# Dead-slot verification matrix: 4 datasets x 3 seeds x {train,test}.
#
# (b) does the SAME slot die in every seed?  seed-dependent => training
#     instability; seed-independent => the fixed 6-axis schema does not match
#     the dataset (CIFAR-10 32x32 single-object has no "secondary object").
# (c) does the pattern hold on the official test split, or is it an artifact of
#     the training rows the router was fitted on?
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs docs/newmodel_analysis/slotmass
N=${N:-512}

# ds|seed|dir   (s42 dirs taken from the measured JSONs; the bare-tag glob is
# ambiguous because e.g. "uni003" also appears in a flickr dir name)
RUNS=(
"flickr|42|result/260803+flickr25k_setting1_promptAblA_flickr_A_v4_uni002_P0refit_e4+bs+64+e+60+proj_lr+0.001"
"flickr|43|result/260803+flickr25k_setting1_promptAblA_flickr_A_v4_uni002_s43_P0refit_e4+bs+64+e+60+proj_lr+0.001"
"flickr|44|result/260803+flickr25k_setting1_promptAblA_flickr_A_v4_uni002_s44_P0refit_e4+bs+64+e+60+proj_lr+0.001"
"mscoco|42|result/260803+mscoco_setting1_promptAblA_mscoco_A_v5b_uni003_P0refit_e39+bs+64+e+60+proj_lr+0.001"
"mscoco|43|result/260803+mscoco_setting1_promptAblA_mscoco_A_v5b_uni003_s43_P0refit_e9+bs+64+e+60+proj_lr+0.001"
"mscoco|44|result/260803+mscoco_setting1_promptAblA_mscoco_A_v5b_uni003_s44_P0refit_e24+bs+64+e+60+proj_lr+0.001"
"nuswide|42|result/260803+nuswide_setting1_promptAblA_nuswide_A_v4_uni005_P0refit_e4+bs+64+e+60+proj_lr+0.001"
"nuswide|43|result/260803+nuswide_setting1_promptAblA_nuswide_A_v4_uni005_s43_P0refit_e4+bs+64+e+60+proj_lr+0.001"
"nuswide|44|result/260803+nuswide_setting1_promptAblA_nuswide_A_v4_uni005_s44_P0refit_e4+bs+64+e+60+proj_lr+0.001"
"cifar|42|result/260803+cifar10_setting1_promptAblA_cifar_A_v4_uniB003_P0refit_e19+bs+64+e+60+proj_lr+0.001"
"cifar|43|result/260804+cifar10_setting1_promptAblA_cifar_A_v4_uniB003_s43_P0refit_e4+bs+64+e+60+proj_lr+0.001"
"cifar|44|result/260804+cifar10_setting1_promptAblA_cifar_A_v4_uniB003_s44_P0refit_e14+bs+64+e+60+proj_lr+0.001"
)

JOBS=()
for r in "${RUNS[@]}"; do
  IFS='|' read -r ds sd dir <<<"$r"
  if [ ! -f "$dir/model_state_dict.pth" ]; then
    echo "[slotmass] MISSING checkpoint, skipping: $ds s$sd $dir" >&2; continue
  fi
  for sp in train test; do JOBS+=("$ds|$sd|$sp|$dir"); done
done
echo "[slotmass] ${#JOBS[@]} jobs, N=$N @ $(date '+%F %T')"

GPUS=(0 1 2 3 4)   # GPU5 = nuswide abA4 auditfix retry
declare -A PID
i=0
while [ "$i" -lt "${#JOBS[@]}" ]; do
  for g in "${GPUS[@]}"; do
    if [ -n "${PID[$g]:-}" ] && kill -0 "${PID[$g]}" 2>/dev/null; then continue; fi
    [ "$i" -ge "${#JOBS[@]}" ] && break
    IFS='|' read -r ds sd sp dir <<<"${JOBS[$i]}"
    o="docs/newmodel_analysis/slotmass/${ds}_s${sd}_${sp}.json"
    echo "[slotmass] GPU$g <- $ds s$sd $sp"
    setsid env CUDA_VISIBLE_DEVICES="$g" nohup \
      /home/yschoi/.conda/envs/dna_hashing/bin/python \
      scripts/diagnose_slot_routing_mass.py --result_dir "$dir" \
      --split "$sp" --n "$N" --device "cuda:0" --out "$o" \
      > "logs/SLOTMASS_${ds}_s${sd}_${sp}.log" 2>&1 < /dev/null &
    PID[$g]=$!
    i=$((i+1)); sleep 2
  done
  sleep 10
done
for g in "${GPUS[@]}"; do [ -n "${PID[$g]:-}" ] && wait "${PID[$g]}" 2>/dev/null; done
echo "[slotmass] done @ $(date '+%F %T')"
