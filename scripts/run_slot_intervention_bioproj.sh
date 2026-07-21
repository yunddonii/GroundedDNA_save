#!/usr/bin/env bash
# Slot intervention under bio-projection (2026-07-21), 3 datasets.
set -u
cd /home/yschoi/GroundedDNA
P=/home/yschoi/.conda/envs/dna_hashing/bin/python

run() {  # $1=dataset $2=ours_dir $3=train_manifest(optional)
    local ds="$1" dir="$2" mf="${3:-}"
    local extra=""
    [ -n "$mf" ] && extra="--train_manifest $mf"
    CUDA_VISIBLE_DEVICES=0 $P scripts/slot_intervention_eval.py \
        --dataset "$ds" --ours_dir "$dir" $extra \
        --n_query 400 --db_subsample 12000 --k 100 --n_boot 1000 --bio_project \
        --out "docs/slot_intervention_$(echo "$ds" | tr 'A-Z' 'a-z')_bioproj.json"
}

run Flickr25k "result/260717+flickr25k_setting1_flickr_P0refit_e4+bs+64+e+60+proj_lr+0.001" dataset/Flickr25k/setting1/train.txt
run MSCOCO    "result/260717+mscoco_setting1_mscoco_P0refit_e49+bs+64+e+60+proj_lr+0.001"
run NUSWIDE   "result/260717+nuswide_setting1_nuswide_P0refit_e4+bs+64+e+60+proj_lr+0.001" dataset/NUSWIDE/setting1/train.txt
echo "SLOT_ALLDONE $(date '+%T')"
