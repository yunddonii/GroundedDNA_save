#!/usr/bin/env bash
# DISABLED 2026-08-13 by the protocol audit.
# docs/EXPERIMENT_PROTOCOL_AUDIT_2026-08-13.md operational decisions 2/5: do not
# enter the post-BU05 stages or launch any 20-base work before F01/F05-F09 are
# fixed. The launcher is kept for reference; it must be rewritten against the
# decisions in docs/PROTOCOL_DECISIONS_2026-08-13.md, not re-enabled as is.
echo "[disabled] see docs/EXPERIMENT_PROTOCOL_AUDIT_2026-08-13.md" >&2
exit 91
# After the pending runs finish, work through the analysis backlog unattended.
#
# Standing instruction (2026-08-12): once the queued experiments end, run
#   lambda_bu 0.05 check  ->  4.7  ->  4.8  ->  4.10.3  ->  4.10c
# in that order, without waiting for a further go-ahead.
#
# Ordering is not arbitrary. 4.7 (codon decoding) and 4.10.3 (slot intervention)
# read the extraction and cost nothing, so they go before 4.8, which is 12 GPU
# cells. 4.10c (qualitative figures) goes last because the user cherry-picks
# from them and that is easier once the numbers are in.
#
# Every step takes only GPUs measured as idle. This machine runs other projects.
# `set -u` alone let a failed python step print "step done" and continue, and a
# background launch's exit code was never collected at all. `-E` carries the
# trap into functions and subshells; `pipefail` matters because every step here
# ends in a pipe.
set -Eeuo pipefail
trap 'rc=$?; printf "[auto-chain] FAILED at line %s (rc %s): %s\n" \
        "$LINENO" "$rc" "$BASH_COMMAND" >&2; exit "$rc"' ERR
cd /home/yschoi/GroundedDNA
PY=/home/yschoi/.conda/envs/dna_hashing/bin/python
NJSON=docs/newmodel_analysis/fixed_N.json
IDLE_MIB="${IDLE_MIB:-200}"
LOG=logs/AUTO_CHAIN.log
mkdir -p logs docs/newmodel_analysis
say(){ echo "[chain $(date '+%F %T')] $*" | tee -a "$LOG"; }
free_gpus(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
             | awk -F', ' -v t="$IDLE_MIB" '$2+0 < t {printf "%s ", $1}'; }
wait_idle(){ while :; do G=$(free_gpus); [ -n "${G// /}" ] && return 0; sleep 120; done; }
# Match the PYTHON process, not the pattern text. `pgrep -f` also matches any
# shell whose command line merely CONTAINS these words, so a monitoring command
# typed while the chain waits made it block on itself indefinitely.
busy(){ pgrep -af "train_siglip2.py|eval_cell_bioproj.py" | grep -qE "python[0-9.]* +[^ ]*(train_siglip2|eval_cell_bioproj)\.py"; }

say "waiting for the pending runs"
while busy; do sleep 120; done
say "pending runs finished"

DS_LIST="cifar_A_v4:CIFAR10:cifar10 flickr_A_v4:Flickr25k:flickr25k nuswide_A_v4:NUSWIDE:nuswide mscoco_A_v5b:MSCOCO:mscoco"
getN(){ "$PY" -c "import json;print(json.load(open('$NJSON'))['$1']['N'])"; }
getJD(){ "$PY" -c "import json;print(json.load(open('$NJSON'))['$1']['lambda_codon_joint'])"; }
# Must match ONLY the reference run. Without the `_v4`/`_v5b` boundary this
# also matches the _BU05 and _A2/_A4/_A5 variants, which differ from the
# reference by exactly the thing being ablated.
# `ls -dt ... | head -1` returned the NEWEST match, so a concurrently running
# ablation could be evaluated as the reference run, and "no match" came back as
# the empty string. Exactly one, or refuse.
rundir(){
    local matches=() keep=() d
    shopt -s nullglob
    matches=(result/2608*"$1"*_P0refit_e"$2"+bs+64+e+"$3"+*)
    shopt -u nullglob
    for d in "${matches[@]}"; do
        [[ "$d" =~ _(BU05|A2|A4|A5[a-z]*|s4[34])_P0refit ]] || keep+=("$d")
    done
    if [[ "${#keep[@]}" -ne 1 ]]; then
        printf "[auto-chain] %d reference runs match %s e%s; refusing to guess:\n" \
            "${#keep[@]}" "$1" "$2" >&2
        printf "    %s\n" "${keep[@]:-<none>}" >&2
        return 1
    fi
    printf "%s\n" "${keep[0]}"
}

S5="--num_semantic_parts 5 --num_codebooks 5 --no_gumbel_softmax --lambda_codeword_codon_sinkhorn 0.0"
T69="--routing_adaptive_topp_min 0.6 --routing_adaptive_topp_max 0.95"

# ---- 0) lambda_codebook_balance 0.02 -> 0.05 --------------------------------
# Its gradient norm measured 6.2e-06 at the decided operating point, i.e. the
# term is inert there. Raising it is the cheapest way to learn whether that is
# because the codebook is already balanced or because the weight is too small.
say "step 0: lambda_bu 0.05"
for E in $DS_LIST; do
    IFS=: read -r EXP CANON SLUG <<<"$E"
    N=$(getN "$EXP"); JD=$(getJD "$EXP")
    wait_idle; g=${G%% *}
    say "  bu05 $EXP N=$N -> GPU $g"
    GDNA_NUM_SEMANTIC_PARTS=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    FIXED_N="$N" EVERY=1 TAG_SUFFIX="_BU05" \
    AUX_ARGS="$S5 --lambda_codon_joint $JD $T69 -e $((N+1)) --lambda_bu 0.05" \
    setsid nohup bash scripts/prompt_ablation_A_cell_fixedN.sh "$g" "$EXP" \
        > "logs/BU05_${EXP}.out" 2>&1 < /dev/null &
    sleep 150
done
while busy; do sleep 120; done
say "step 0 done"

# ---- 0b) 4.6 bio-projection, re-measured at 15 bases ------------------------
# The published table is 18-base: the GC fraction [0.40, 0.60] is unchanged but
# the integer window it implies moves from [8,10] to [6,9], so every projection
# cost in that table was computed under a different constraint set.
say "step 0b: 4.6 bio-projection"
for E in $DS_LIST; do
    IFS=: read -r EXP CANON SLUG <<<"$E"
    N=$(getN "$EXP"); RD=$(rundir "$SLUG" "$N" "$((N+1))")
    [ -n "$RD" ] || { say "  SKIP $SLUG"; continue; }
    K=128; [ "$SLUG" = cifar10 ] && K=64
    say "  bioproj $SLUG (K=$K)"
    "$PY" scripts/eval_cell_bioproj.py --dir "$RD" --dataset "$CANON" --K "$K" \
        --gc_min 0.40 --gc_max 0.60 >> "$LOG" 2>&1
done
say "step 0b done"

# ---- 1) 4.7 held-out codon decoding, 30-bit controls -----------------------
say "step 1: 4.7 codon decoding"
B=result_baseline
for E in $DS_LIST; do
    IFS=: read -r EXP CANON SLUG <<<"$E"
    N=$(getN "$EXP"); RD=$(rundir "$SLUG" "$N" "$((N+1))")
    [ -n "$RD" ] || { say "  SKIP $SLUG: no run dir"; continue; }
    OURS="$RD"
    if [ "$SLUG" = cifar10 ]; then
        [ -f "$RD/withids/extract_query.npz" ] || "$PY" scripts/cifar_inject_image_ids.py --result_dir "$RD" >/dev/null 2>&1
        OURS="$RD/withids"; MAN=dataset/CIFAR10/setting1/train_image_ids.txt
    elif [ "$SLUG" = flickr25k ]; then MAN=dataset/Flickr25k/setting1/train.txt
    elif [ "$SLUG" = nuswide ];  then MAN=dataset/NUSWIDE/setting1/train_10500.txt
    else MAN=""; fi
    dirs=(); for m in cibhash cimon sdc-paper oh crovca; do
        # One baseline evaluation per method, or name the ambiguous method
        # rather than taking whichever sorted first.
        mapfile -t _cand < <(ls -d $B/*/u0_${m}_${SLUG}_30b_seed42/attempt_*/*/*_dnaeval 2>/dev/null || true)
        if [[ "${#_cand[@]}" -ne 1 ]]; then
            say "  baseline $m/$SLUG: ${#_cand[@]} candidate dirs; skipping"
            continue
        fi
        dirs+=("${_cand[0]}"); done
    say "  decode $SLUG"
    "$PY" scripts/heldout_codon_decoding.py --ours_dir "$OURS" --dataset "$CANON" \
        --bio_project ${MAN:+--train_manifest "$MAN"} \
        --baseline_dirs "${dirs[@]}" --baseline_names CIBHash CIMON SDC OH CroVCA \
        --out "docs/heldout_decoding_${SLUG}_fixedN.json" >> "$LOG" 2>&1
done
say "step 1 done"

declare -a ABL_PIDS=()

# ---- 2) 4.8 causal ablation (A2 / A4 / A5) ---------------------------------
#
# Each of the three was wrong, and two could not have run at all:
#
#   A2  zeroed three text losses and left `--disable_text_supervision` off, so
#       the caption-derived routing and token pruning stayed active. The cell
#       claimed "no text path" while text still shaped the codes.
#   A4  passed `--num_codebooks 1` against M=5 inputs, which raises
#       "semantic_visual_tokens has M=5 but quantizer was built with
#       num_codebooks=1" before the first step. `--share_codebook` is the flag
#       that ties the five slots to one codebook.
#   A5  passed `--router_type mean`, which argparse rejects outright -- the
#       choices are sinkhorn/attention/slot/cluster_attn/cross_attn. And the
#       paper's A5 is not a router ablation at all: it is the 2x2 factorial of
#       `L_joint` and `noGumbel` (draft 4.8, A5), whose whole point is that
#       noGumbel alone is harmful on some datasets and only helps in
#       combination. Collapsing that to one cell would have hidden exactly the
#       interaction it exists to show.
# Every ablation's flags are parsed before the first GPU second. Two of the
# three commands could not have run at all, and would have died at the first
# cell -- hours into a chain that had already trained the ones before it.
say "step 2: preflight -- parsing every ablation command"
for E in $DS_LIST; do
    IFS=: read -r EXP CANON SLUG <<<"$E"
    N=$(getN "$EXP"); JD=$(getJD "$EXP")
    for AB in "A2:--disable_text_supervision --lambda_text_code_kl 0.0 --lambda_text_hash_ntxent 0.0 --lambda_xmodal_commit 0.0" \
              "A4:--share_codebook" \
              "A5none:--lambda_codon_joint 0.0" \
              "A5joint:--lambda_codon_joint $JD" \
              "A5nogumbel:--lambda_codon_joint 0.0 --no_gumbel_softmax" \
              "A5both:--lambda_codon_joint $JD --no_gumbel_softmax"; do
        TAG=${AB%%:*}; FLAGS=${AB#*:}
        GDNA_NUM_SEMANTIC_PARTS=5 "$PY" scripts/_ablation_preflight.py -- \
            $S5 $T69 $FLAGS >/dev/null || {
            say "  PREFLIGHT FAILED: $TAG $EXP"; exit 1; }
    done
done
say "step 2: preflight ok"

say "step 2: 4.8 ablation"
for E in $DS_LIST; do
    IFS=: read -r EXP CANON SLUG <<<"$E"
    N=$(getN "$EXP"); JD=$(getJD "$EXP")
    # A5 is four cells, not one: neither / joint only / noGumbel only / both.
    for AB in "A2:--disable_text_supervision --lambda_text_code_kl 0.0 --lambda_text_hash_ntxent 0.0 --lambda_xmodal_commit 0.0" \
              "A4:--share_codebook" \
              "A5none:--lambda_codon_joint 0.0" \
              "A5joint:--lambda_codon_joint $JD" \
              "A5nogumbel:--lambda_codon_joint 0.0 --no_gumbel_softmax" \
              "A5both:--lambda_codon_joint $JD --no_gumbel_softmax"; do
        TAG=${AB%%:*}; FLAGS=${AB#*:}
        wait_idle; g=${G%% *}
        say "  $TAG $EXP -> GPU $g"
        # A5 sets `--lambda_codon_joint` itself, so it must not also come from
        # the shared prefix; A2/A4 keep the champion value.
        case "$TAG" in
            A5*) BASE_FLAGS="$S5 $T69 -e $((N+1))" ;;
            *)   BASE_FLAGS="$S5 --lambda_codon_joint $JD $T69 -e $((N+1))" ;;
        esac
        GDNA_NUM_SEMANTIC_PARTS=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        FIXED_N="$N" EVERY=1 TAG_SUFFIX="_$TAG" \
        AUX_ARGS="$BASE_FLAGS $FLAGS" \
        setsid nohup bash scripts/prompt_ablation_A_cell_fixedN.sh "$g" "$EXP" \
            > "logs/ABL_${TAG}_${EXP}.out" 2>&1 < /dev/null &
        ABL_PIDS+=("$!:${TAG}_${EXP}")
        sleep 150
    done
done
while busy; do sleep 120; done
# `while busy` watches process names and GPU memory, which is not the same as
# every child having succeeded. Collect the exit codes.
ABL_FAILED=0
for entry in "${ABL_PIDS[@]:-}"; do
    [[ -n "$entry" ]] || continue
    pid=${entry%%:*}; name=${entry#*:}
    if wait "$pid" 2>/dev/null; then
        say "  $name ok"
    else
        say "  $name FAILED (see logs/ABL_${name}.out)"
        ABL_FAILED=1
    fi
done
[[ "$ABL_FAILED" == "0" ]] || { say "step 2 INCOMPLETE"; exit 1; }
say "step 2 done"

# ---- 3) 4.10.3 slot intervention -------------------------------------------
say "step 3: 4.10.3 slot intervention"
for E in $DS_LIST; do
    IFS=: read -r EXP CANON SLUG <<<"$E"
    N=$(getN "$EXP"); RD=$(rundir "$SLUG" "$N" "$((N+1))")
    [ -n "$RD" ] || continue
    say "  intervention $SLUG"
    "$PY" scripts/slot_concept_specificity.py --result_dir "$RD" --dataset "$CANON" \
        --out "docs/newmodel_analysis/slot_concept_specificity_${SLUG}_fixedN.json" >> "$LOG" 2>&1
done
say "step 3 done"

# ---- 3b) 4.10b per-image empty-slot diagnostic -----------------------------
# Needed before any decision about re-tuning the confidence-adaptive top-p mask:
# a slot receiving zero tokens on an image emits a codon with no content.
say "step 3b: 4.10b empty slots"
for E in $DS_LIST; do
    IFS=: read -r EXP CANON SLUG <<<"$E"
    N=$(getN "$EXP"); RD=$(rundir "$SLUG" "$N" "$((N+1))")
    [ -n "$RD" ] || continue
    wait_idle; g=${G%% *}
    say "  empty-slot $SLUG epoch=$N -> GPU $g"
    GDNA_NUM_SEMANTIC_PARTS=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PY" scripts/diagnose_per_image_empty_slots.py --result_dir "$RD" \
        --n 512 --epoch "$N" --device "cuda:$g" \
        --out "docs/newmodel_analysis/empty_slots_${SLUG}_fixedN.json" >> "$LOG" 2>&1
done
say "step 3b done"

# ---- 4) 4.10c qualitative figures, ALL datasets -----------------------------
# --epoch is mandatory: _current_epoch defaults to 0 and is not in the state
# dict, so a freshly loaded checkpoint would draw at the INITIAL epsilon.
say "step 4: 4.10c figures"
for E in $DS_LIST; do
    IFS=: read -r EXP CANON SLUG <<<"$E"
    N=$(getN "$EXP"); RD=$(rundir "$SLUG" "$N" "$((N+1))")
    [ -n "$RD" ] || continue
    wait_idle; g=${G%% *}
    say "  figures $SLUG epoch=$N -> GPU $g"
    GDNA_NUM_SEMANTIC_PARTS=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PY" scripts/regen_viz_routing.py --result_dir "$RD" --num_samples 16 \
        --epoch "$N" --tsne --tsne_samples 3000 \
        --save_name viz_routing_heatmap_final.png \
        --tsne_save_name viz_codebook_tsne_final.png --device "cuda:$g" \
        >> "$LOG" 2>&1
    cp "$RD/viz_routing_heatmap_final.png"  "docs/figures/qualitative_s5/${SLUG}_routing_final.png"  2>/dev/null
    cp "$RD/viz_codebook_tsne_final.png"    "docs/figures/qualitative_s5/${SLUG}_codebook_tsne_final.png" 2>/dev/null
done
say "step 4 done -- 16 images per dataset, ready for cherry picking"
say "chain complete"
