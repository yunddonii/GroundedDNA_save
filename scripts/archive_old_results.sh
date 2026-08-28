#!/usr/bin/env bash
# Move old result directories to /data and leave a symlink behind.
#
# `/` is at 95% and Phase 3 needs roughly 20 GB of new run directories. The
# archive is a MOVE plus a symlink at the original path, not a delete, because
# absolute paths into `result/` are load-bearing: the Phase 2 legacy binding
# records `npz_path` as `/home/yschoi/GroundedDNA/result/.../extract_db.npz`
# and `phase2_legacy_bound/*/extract_db.npz` symlinks to the same. A symlink at
# the old path keeps every one of those resolving to the same bytes.
#
# Usage:
#   scripts/archive_old_results.sh <prefix-glob> [--apply]
#   scripts/archive_old_results.sh '2605*'            # dry run
#   scripts/archive_old_results.sh '2605*' --apply
set -uo pipefail

REPO=/home/yschoi/GroundedDNA
DEST=/data/yschoi/GroundedDNA/result_archive
cd "$REPO"

GLOB="${1:?usage: archive_old_results.sh <prefix-glob> [--apply]}"
APPLY="${2:-}"
mkdir -p "$DEST"

shopt -s nullglob
CANDIDATES=(result/$GLOB/)
shopt -u nullglob
if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
    echo "[archive] nothing matches result/$GLOB" >&2; exit 1
fi

moved=0; failed=0; skipped=0
for d in "${CANDIDATES[@]}"; do
    d="${d%/}"
    name="$(basename "$d")"
    if [[ -L "$d" ]]; then
        skipped=$((skipped + 1)); continue          # already archived
    fi
    if [[ "$APPLY" != "--apply" ]]; then
        echo "  would archive $name"
        moved=$((moved + 1)); continue
    fi
    if [[ -e "$DEST/$name" ]]; then
        echo "[archive] $DEST/$name already exists; skipping $name" >&2
        failed=$((failed + 1)); continue
    fi
    # Copy first, verify, then remove: a mv interrupted across filesystems can
    # leave a partial destination and no source.
    if ! cp -a "$d" "$DEST/$name"; then
        echo "[archive] copy failed for $name" >&2
        rm -rf "$DEST/$name"; failed=$((failed + 1)); continue
    fi
    src_n=$(find "$d" -type f | wc -l)
    dst_n=$(find "$DEST/$name" -type f | wc -l)
    src_b=$(du -sb "$d" | cut -f1)
    dst_b=$(du -sb "$DEST/$name" | cut -f1)
    if [[ "$src_n" != "$dst_n" || "$src_b" != "$dst_b" ]]; then
        echo "[archive] $name copy differs (files $src_n/$dst_n bytes $src_b/$dst_b)" >&2
        rm -rf "$DEST/$name"; failed=$((failed + 1)); continue
    fi
    rm -rf "$d"
    ln -s "$DEST/$name" "$d"
    if [[ ! -d "$d" ]]; then
        echo "[archive] symlink for $name does not resolve" >&2
        failed=$((failed + 1)); continue
    fi
    moved=$((moved + 1))
done

echo "[archive] $moved moved, $skipped already archived, $failed failed"
[[ "$failed" -eq 0 ]]
