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

    if args.expect_dataset and identity.dataset != args.expect_dataset:
        print("unreadable")
        print(f"{path} names dataset {identity.dataset!r}, not "
              f"{args.expect_dataset!r}", file=sys.stderr)
        return 1 if args.require_manifest else 0

    print("manifested")
    print(f"{identity.digest[:12]} dataset={identity.dataset} "
          f"seed={identity.seed} stop={identity.stop_after_epoch} "
          f"K={identity.codebook_size}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
