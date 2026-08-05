#!/usr/bin/env bash
# Is the per-patch adaptive top-p nucleus mask what kills the slots?
# Same checkpoint, same images, mask ON vs OFF, evaluated at the training E*.
set -u
cd /home/yschoi/GroundedDNA
mkdir -p logs docs/newmodel_analysis/toppabl
RUNS=(
"flickr|42|4|result/260803+flickr25k_setting1_promptAblA_flickr_A_v4_uni002_P0refit_e4+bs+64+e+60+proj_lr+0.001"
"mscoco|42|39|result/260803+mscoco_setting1_promptAblA_mscoco_A_v5b_uni003_P0refit_e39+bs+64+e+60+proj_lr+0.001"
"nuswide|42|4|result/260803+nuswide_setting1_promptAblA_nuswide_A_v4_uni005_P0refit_e4+bs+64+e+60+proj_lr+0.001"
"cifar|42|19|result/260803+cifar10_setting1_promptAblA_cifar_A_v4_uniB003_P0refit_e19+bs+64+e+60+proj_lr+0.001"
"cifar|43|4|result/260804+cifar10_setting1_promptAblA_cifar_A_v4_uniB003_s43_P0refit_e4+bs+64+e+60+proj_lr+0.001"
"cifar|44|14|result/260804+cifar10_setting1_promptAblA_cifar_A_v4_uniB003_s44_P0refit_e14+bs+64+e+60+proj_lr+0.001"
"nuswide|44|4|result/260803+nuswide_setting1_promptAblA_nuswide_A_v4_uni005_s44_P0refit_e4+bs+64+e+60+proj_lr+0.001"
)
for r in "${RUNS[@]}"; do
  IFS='|' read -r ds sd es dir <<<"$r"
  for m in on off; do
    flag=""; [ "$m" = off ] && flag="--disable_adaptive_topp"
    o="docs/newmodel_analysis/toppabl/${ds}_s${sd}_topp${m}.json"
    CUDA_VISIBLE_DEVICES=4 /home/yschoi/.conda/envs/dna_hashing/bin/python \
      scripts/diagnose_slot_routing_mass.py --result_dir "$dir" --split train \
      --n 384 --epoch "$es" --device cuda:0 --out "$o" $flag \
      > "logs/TOPPABL_${ds}_s${sd}_${m}.log" 2>&1
    echo "[topp] $ds s$sd E*=$es $m done"
  done
done
echo "[topp] ALL DONE @ $(date '+%F %T')"
