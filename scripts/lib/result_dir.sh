# Resolve the result directory of a run by tag -- exactly one, or refuse.
#
# Source this, then call `resolve_one_result_dir <tag>`; it prints the directory
# on stdout and returns non-zero if the answer is not unique.
#
# WHY. Eleven runners located their child's output with
# `ls -dt result/*TAG* | head -1`, which answers "the newest directory whose
# name contains this substring". Two cells running concurrently on this machine
# share `result/`, so the newest match can belong to the OTHER run -- the
# evaluation then scores someone else's checkpoint and reports it under this
# cell's name. `head -1` also turns "no match" into the empty string and
# "several matches" into an arbitrary one, so neither failure announces itself.
#
# The paper's P0 cell runner already required exactly one match and proved the
# discipline works; this is that rule, shared.
#
# The manifest is READ, not merely mentioned. An earlier version of this note
# said a manifest would be "reported" while no code opened one, so a sole
# substring match carrying a wrong-dataset manifest -- or none at all -- came
# back rc0. `resolve_one_claimed_result_dir` requires a loadable
# `run_identity.json`, optionally for a named dataset.

resolve_one_result_dir() {
    local tag="$1"
    local root="${2:-result}"
    local matches=()

    if [[ -z "$tag" ]]; then
        printf '[result-dir] refusing to resolve an empty tag\n' >&2
        return 2
    fi

    shopt -s nullglob
    matches=("$root"/*"$tag"*)
    shopt -u nullglob

    if [[ "${#matches[@]}" -eq 0 ]]; then
        printf '[result-dir] no run matches %q under %s/\n' "$tag" "$root" >&2
        return 1
    fi
    if [[ "${#matches[@]}" -gt 1 ]]; then
        printf '[result-dir] %d runs match %q; refusing to guess:\n' \
            "${#matches[@]}" "$tag" >&2
        printf '  %s\n' "${matches[@]}" >&2
        printf '[result-dir] narrow the tag, or remove the runs that do not belong.\n' >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

# Exactly one match, which must also carry a run manifest naming this run.
# Use this wherever the answer feeds an evaluation: a directory that cannot say
# which run produced it is not an answer, and the stale August result dirs are
# precisely unmanifested.
# Extra expectations are passed through RESULT_DIR_EXPECT, an array the caller
# fills with `--expect-...` flags. Naming only the dataset let a same-dataset
# run with seed 99, M=6 and K=999 resolve as this run's.
resolve_one_claimed_result_dir() {
    local tag="$1"
    local required="$2"
    local dataset="${3:-}"
    local root="${4:-result}"
    local repo dir py
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    py="${PY:-/home/yschoi/.conda/envs/dna_hashing/bin/python}"

    if [[ -n "$required" ]]; then
        dir="$(resolve_one_result_dir_with "$tag" "$required" "$root")" || return 1
    else
        dir="$(resolve_one_result_dir "$tag" "$root")" || return 1
    fi
    if ! "$py" "$repo/scripts/_run_manifest_check.py" "$dir" --require-manifest \
            ${dataset:+--expect-dataset "$dataset"} \
            "${RESULT_DIR_EXPECT[@]:-}" >/dev/null; then
        printf '[result-dir] %s does not carry a run manifest for this run;\n' "$dir" >&2
        printf '             refusing to evaluate a directory that cannot name its run.\n' >&2
        return 1
    fi
    printf '%s\n' "$dir"
}

# Same, plus a required file the directory must actually contain. Existing
# callers checked for `extract_db.npz` after the fact and reported a confusing
# "RD=" when the glob had produced nothing at all.
resolve_one_result_dir_with() {
    local tag="$1"
    local required="$2"
    local root="${3:-result}"
    local dir
    dir="$(resolve_one_result_dir "$tag" "$root")" || return 1
    if [[ ! -f "$dir/$required" ]]; then
        printf '[result-dir] %s has no %s; the run did not finish\n' \
            "$dir" "$required" >&2
        return 1
    fi
    printf '%s\n' "$dir"
}
