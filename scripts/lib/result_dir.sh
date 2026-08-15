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
# When the directory carries a run manifest (F08, `dna_utils/run_identity.py`),
# it is reported, so a caller can tell an identity-claimed directory from one
# that merely matched a glob.

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
