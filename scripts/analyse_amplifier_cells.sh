#!/usr/bin/env bash
# Post-process the amplifier-intervention cells: extract z, then measure the
# three quantities the hypothesis makes predictions about --
#   rho_z        (does the amplifier build gradedness in the routed features?)
#   rho_codebook (does it propagate to the codebook?)
#   mAP@R        (what does changing it cost retrieval?)
set -u
cd /home/yschoi/GroundedDNA
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python

DIRS=(); LABELS=()
add() { [ -d "$1" ] && { DIRS+=("$1"); LABELS+=("$2"); } || echo "[skip] missing $2"; }

# champions (already measured; re-included so every number in the final table
# comes from one identical pass)
add "result/260717+flickr25k_setting1_flickr_P0refit_e4+bs+64+e+60+proj_lr+0.001"  "FLK_cb1.0_champ"
add "result/260717+mscoco_setting1_mscoco_P0refit_e49+bs+64+e+60+proj_lr+0.001"    "COCO_cb1.5_champ"
for d in result/*flickr_AMPL_cb0.0*/; do add "${d%/}" "FLK_cb0.0"; done
for d in result/*flickr_AMPL_cb0.5*/; do add "${d%/}" "FLK_cb0.5"; done
for d in result/*mscoco_AMPL_cb0.0*/;  do add "${d%/}" "COCO_cb0.0"; done
for d in result/*mscoco_AMPL_cb0.5*/;  do add "${d%/}" "COCO_cb0.5"; done

echo "### 1. extract z (skipped where extract_z_db.npz already exists)"
gpu=0
for i in "${!DIRS[@]}"; do
    d="${DIRS[$i]}"
    if [ -f "$d/extract_z_db.npz" ]; then echo "  [have] ${LABELS[$i]}"; continue; fi
    echo "  [run ] ${LABELS[$i]} on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu Z_SPLIT=db Z_MAX=20000 Z_BLOCKS=40 \
        $PY scripts/extract_z_prequant.py --config_path "$d" \
        > "logs/ampl_z_${LABELS[$i]}.log" 2>&1
    gpu=$(( (gpu + 1) % 4 ))
done

echo; echo "### 2. z geometry (rho_z, rho_codebook, eff_rank)"
$PY scripts/z_geometry_analysis.py --out docs/amplifier_z_geometry.json \
    --dirs "${DIRS[@]}" --labels "${LABELS[@]}"

echo; echo "### 3. codebook semantic alignment (independent check)"
$PY scripts/codebook_semantic_alignment.py --split db \
    --out docs/amplifier_codebook_alignment.json \
    --dirs "${DIRS[@]}" --labels "${LABELS[@]}"

echo; echo "### 4. retrieval cost (mAP@R from each cell's final eval)"
$PY - "${DIRS[@]}" <<'EOF'
import json, os, sys
print(f"{'cell':22s}{'mAP@R':>9s}{'full mAP':>10s}{'P@1':>8s}{'DNA-uniq':>10s}")
for d in sys.argv[1:]:
    p = os.path.join(d, "evaluation_siglip2_base.json")
    if not os.path.exists(p):
        print(f"{os.path.basename(d)[:22]:22s}    (no eval json yet)"); continue
    r = json.load(open(p))
    print(f"{os.path.basename(d)[:22]:22s}{r.get('mAP_at_R', float('nan')):>9.4f}"
          f"{r.get('mAP', float('nan')):>10.4f}"
          f"{r.get('precision_at_k', {}).get('1', float('nan')):>8.4f}"
          f"{r.get('unique_code_ratio', float('nan')):>10.4f}")
EOF
