#!/usr/bin/env bash
# Self-test of probe_chain_ancP7_v7.sh on a sandbox copy whose only differences are three lines
# (working directory, interpreter, records root), with a stub interpreter that records its argv and
# exits as told. Nothing real runs: no supervisor, probe, python or GPU. Then three mutants of the
# chain, each of which one scenario must catch.
# Usage: chain_selftest.sh <empty scratch directory>
set -u
CHAIN=/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation/probe_chain_ancP7_v7.sh
BOX=${1:?scratch directory}
[[ -d "$BOX" && -z "$(ls -A "$BOX")" ]] || { echo "need an empty scratch directory" >&2; exit 2; }
mkdir -p "$BOX/tree/rec"
cat > "$BOX/stub" <<'EOF'
#!/usr/bin/env bash
n=$(( $(cat "$STUB_DIR/count" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$STUB_DIR/count"
printf 'CUDA_VISIBLE_DEVICES=%s %s\n' "${CUDA_VISIBLE_DEVICES-unset}" "$*" >> "$STUB_DIR/calls"
[[ "$n" == "${STUB_FAIL_AT:-0}" ]] && exit "${STUB_RC:-1}"
exit 0
EOF
chmod +x "$BOX/stub"

render() {   # $1: source chain, $2: output copy
  sed -e "s|^cd /data/yschoi/gdna_anchor_confirm_v1 |cd $BOX/tree |" \
      -e "s|^PY=/home/yschoi/.conda/envs/dna_hashing/bin/python$|PY=$BOX/stub|" \
      -e "s|^REC=/data/yschoi/gdna_anchor_confirm_v1/artifacts/anchor_confirmation$|REC=$BOX/tree/rec|" \
      "$1" > "$2"
}
render "$CHAIN" "$BOX/chain.sh"
changed=$(diff "$CHAIN" "$BOX/chain.sh" | grep -c '^>')
echo "sandbox copy differs from the chain in $changed lines (want 3)"
[[ "$changed" == 3 ]] || exit 1

FAILS=0
run() {   # $1 label, $2 chain copy, $3 want rc, $4 want calls, $5 env setup, rest: chain args
  local label=$1 copy=$2 want_rc=$3 want_calls=$4 setup=$5; shift 5
  rm -rf "$BOX/tree/rec/ancP7_probes" "$BOX/count" "$BOX/calls"
  eval "$setup"
  env -u PYTHONPATH -u CUDA_VISIBLE_DEVICES GDNA_NUM_SEMANTIC_PARTS=5 STUB_DIR="$BOX" \
      ${EXTRA_ENV:-} bash "$copy" "$@" > "$BOX/out" 2>&1
  local rc=$? calls; calls=$(cat "$BOX/count" 2>/dev/null || echo 0)
  local verdict=pass
  [[ "$rc" == "$want_rc" && "$calls" == "$want_calls" ]] || verdict=FAIL
  [[ $verdict == FAIL ]] && FAILS=$((FAILS + 1))
  printf '%-44s rc=%s (want %s) calls=%s (want %s) %s\n' "$label" "$rc" "$want_rc" "$calls" "$want_calls" "$verdict"
  unset STUB_FAIL_AT STUB_RC EXTRA_ENV
}

echo "== scenarios on the faithful copy"
run "all twelve succeed" "$BOX/chain.sh" 0 12 ":" 3 731
order=$(sed -n 's/.*--coordinate \([^ ]*\) .*/\1/p' "$BOX/calls" | tr '\n' ' ')
echo "  order: $order"
bad=$(grep -vc -- '^CUDA_VISIBLE_DEVICES=3 scripts/anchor_confirm_supervisor.py .*--stage probe --planned-cells 0 .* -- .*scripts/anchor_confirm_code_axis.py .*--approval-section 731 .*--device cuda:0$' "$BOX/calls")
outs=$(sed -n 's/.*--out \([^ ]*\) .*/\1/p' "$BOX/calls" | sort -u | wc -l)
echo "  calls not of the approved shape: $bad (want 0); distinct --out paths: $outs (want 12)"
[[ "$bad" == 0 && "$outs" == 12 ]] || FAILS=$((FAILS + 1))
run "third probe exits 5: stop, exit 5" "$BOX/chain.sh" 5 3 "export STUB_FAIL_AT=3 STUB_RC=5" 3 731
run "first probe exits 1: stop, exit 1" "$BOX/chain.sh" 1 1 "export STUB_FAIL_AT=1 STUB_RC=1" 3 731
run "last probe exits 4: exit 4" "$BOX/chain.sh" 4 12 "export STUB_FAIL_AT=12 STUB_RC=4" 3 731
run "output directory already exists" "$BOX/chain.sh" 2 0 "mkdir -p $BOX/tree/rec/ancP7_probes" 3 731
run "non-integer GPU" "$BOX/chain.sh" 2 0 ":" x 731
run "section 0" "$BOX/chain.sh" 2 0 ":" 3 0
run "one argument" "$BOX/chain.sh" 2 0 ":" 3
run "CUDA_VISIBLE_DEVICES set" "$BOX/chain.sh" 2 0 "EXTRA_ENV=CUDA_VISIBLE_DEVICES=0" 3 731
run "PYTHONPATH set" "$BOX/chain.sh" 2 0 "EXTRA_ENV=PYTHONPATH=/x" 3 731
rm -rf "$BOX/tree/rec/ancP7_probes" "$BOX/count"
env -u PYTHONPATH -u CUDA_VISIBLE_DEVICES -u GDNA_NUM_SEMANTIC_PARTS STUB_DIR="$BOX" bash "$BOX/chain.sh" 3 731 > /dev/null 2>&1
rc=$?; calls=$(cat "$BOX/count" 2>/dev/null || echo 0)
printf '%-44s rc=%s (want 2) calls=%s (want 0) %s\n' "GDNA_NUM_SEMANTIC_PARTS unset" "$rc" "$calls" \
  "$([[ $rc == 2 && $calls == 0 ]] && echo pass || { FAILS=$((FAILS + 1)); echo FAIL; })"

echo "== mutants (each must be caught by the named scenario)"
mutant() {   # $1 name, $2 sed expression, then the scenario: want rc, want calls, setup
  local name=$1 expr=$2; shift 2
  sed -e "$expr" "$CHAIN" > "$BOX/mutant_src.sh"
  if cmp -s "$CHAIN" "$BOX/mutant_src.sh"; then echo "$name: the edit did not apply"; FAILS=$((FAILS + 1)); return; fi
  render "$BOX/mutant_src.sh" "$BOX/mutant.sh"
  local before=$FAILS
  run "  $name" "$BOX/mutant.sh" "$@" 3 731 > "$BOX/mutant_line"
  if (( FAILS > before )); then FAILS=$before; echo "$name: caught ($(cut -c45- "$BOX/mutant_line"))"
  else echo "$name: SURVIVED"; FAILS=$((FAILS + 1)); fi
}
mutant "exit 0 instead of the child status" 's/exit "\$rc"/exit 0/' 5 3 "export STUB_FAIL_AT=3 STUB_RC=5"
mutant "no stop on failure" '/if \[\[ \$rc -ne 0 \]\]; then/,/^  fi$/d' 5 3 "export STUB_FAIL_AT=3 STUB_RC=5"
mutant "mkdir -p (resumes an old directory)" 's/^mkdir "\$OUT"/mkdir -p "$OUT"/' 2 0 "mkdir -p $BOX/tree/rec/ancP7_probes"
echo "failures: $FAILS"
exit $(( FAILS > 0 ))
