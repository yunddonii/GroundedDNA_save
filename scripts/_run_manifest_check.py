"""Report whether a result directory carries a usable run manifest (F08).

`scripts/lib/result_dir.sh` said a manifest would be "reported" and no code
read one, so a sole substring match with a wrong-dataset manifest, or with no
manifest at all, resolved rc0 -- which is how a stale old-cache directory can
be evaluated as if it were the run just launched.

Prints one word on stdout:

    manifested   a valid run_identity.json is present (digest on stderr)
    unmanifested no run_identity.json -- the run that wrote this is unnamed
    unreadable   present but not loadable

Exits 0 always unless --require-manifest is passed, so callers that only want
the label are not forced to branch on the status.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dna_utils.run_identity import MANIFEST_NAME, load_run_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument("--expect-dataset", default=None)
    # Naming the dataset alone lets a same-dataset run with the wrong seed,
    # geometry or codebook size through -- a probe with seed 99, M=6 and K=999
    # was admitted. Every axis the caller knows should be stated.
    parser.add_argument("--expect-seed", type=int, default=None)
    parser.add_argument("--expect-slots", type=int, default=None)
    parser.add_argument("--expect-bases-per-slot", type=int, default=None)
    parser.add_argument("--expect-codebook-size", type=int, default=None)
    parser.add_argument("--expect-stop", type=int, default=None)
    parser.add_argument("--expect-mode", default=None)
    args = parser.parse_args()

    path = os.path.join(args.run_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        print("unmanifested")
        print(f"{args.run_dir} has no {MANIFEST_NAME}; the run that wrote it "
              f"cannot be identified", file=sys.stderr)
        return 1 if args.require_manifest else 0

    identity = load_run_manifest(args.run_dir)
    if identity is None:
        print("unreadable")
        print(f"{path} is present but not loadable", file=sys.stderr)
        return 1 if args.require_manifest else 0

    wrong = {
        name: (got, want) for name, got, want in (
            ("dataset", identity.dataset, args.expect_dataset),
            ("seed", identity.seed, args.expect_seed),
            ("num_slots", identity.num_slots, args.expect_slots),
            ("bases_per_slot", identity.bases_per_slot,
             args.expect_bases_per_slot),
            ("codebook_size", identity.codebook_size,
             args.expect_codebook_size),
            ("stop_after_epoch", identity.stop_after_epoch, args.expect_stop),
            ("selection_mode", identity.selection_mode, args.expect_mode),
        ) if want is not None and got != want}
    if wrong:
        print("unreadable")
        print(f"{path} disagrees with what the caller expects: {wrong}",
              file=sys.stderr)
        return 1 if args.require_manifest else 0

    print("manifested")
    print(f"{identity.digest[:12]} dataset={identity.dataset} "
          f"seed={identity.seed} stop={identity.stop_after_epoch} "
          f"K={identity.codebook_size}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
