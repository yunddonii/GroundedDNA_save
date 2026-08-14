"""Recompute the native-DNA method-protocol locks from the CURRENT source.

The lock is a digest over the *stable* part of a cell's protocol identity:
source equations, training stages, bio contract, implementation revisions and
the frozen PRIMO model. `scripts/run_native_dna_p0.py` is itself listed in
`IMPLEMENTATION_PATHS`, so editing that file -- as F18 did -- moves the digest
and invalidates every registered lock. Any new 15-base cell then fails with

    protocol_identity.method_protocol_lock_sha256: stable source/stage/bio/
    implementation/execution contract mismatch

even though the run is correct. The fix is to review the source diff and
regenerate the locks, not to disable the check.

This script builds the identity the driver would build, strips the
path-dependent fields exactly as the aggregator does, and prints the digest per
method. It never writes to the registry: the digests are pasted in deliberately,
after the diff has been read.

Usage:
    python scripts/recompute_native_method_locks.py --length 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dna_utils.native_protocol import resolve_native_protocol  # noqa: E402
import scripts.aggregate_native_dna_p0 as agg  # noqa: E402
import scripts.run_native_dna_p0 as driver  # noqa: E402

#: The execution contract the registered locks were computed under. Recorded in
#: the aggregator's comment block; repeated here so the recomputation is
#: reproducible rather than folklore.
EXECUTION = {"num_workers": 4, "extract_batch_size": 512, "query_chunk": 64}

#: The lock is path-independent, so any dataset/seed yields the same digest.
#: These are the values the original registration used.
PROBE = {"dataset": "CIFAR10", "setting": "setting1", "seed": 42}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=15)
    parser.add_argument("--dataset-root", default=str(REPO / "dataset"))
    parser.add_argument(
        "--primo-predictor-npz",
        default="/data/yschoi/groundeddna_native_p0/artifacts/"
                "primo_yield_predictor.npz")
    parser.add_argument("--json", action="store_true",
                        help="Emit only the digest map, for diffing.")
    args = parser.parse_args()

    protocol = resolve_native_protocol(args.length)
    predictor = Path(args.primo_predictor_npz)
    registered = agg._METHOD_PROTOCOL_LOCK_BY_LENGTH.get(protocol.length_bases,
                                                         {})

    # The cache audit enters the FULL protocol digest but not the method lock,
    # so a placeholder keeps this offline. Verified by comparing an unchanged
    # method's digest against its registered value.
    cache_audit = {"cache_dir": "<lock-independent>", "blockers": []}

    digests, report = {}, []
    for method in driver.METHODS:
        identity, _ = driver._protocol_identity(
            method=method,
            dataset=PROBE["dataset"],
            setting=PROBE["setting"],
            seed=PROBE["seed"],
            cache_audit=cache_audit,
            dataset_root=Path(args.dataset_root),
            predictor=predictor if method == "bee2021" else None,
            device="cuda:0",
            **EXECUTION,
        )
        digest = agg._method_protocol_lock_digest(identity)
        digests[method] = digest
        was = registered.get(method)
        report.append((method, digest, was,
                       "UNCHANGED" if digest == was else "MOVED"))

    if args.json:
        print(json.dumps(digests, indent=2, sort_keys=True))
        return 0

    print(f"method-protocol locks at {protocol.length_bases} bases "
          f"({protocol.protocol_label})")
    print(f"execution contract: {EXECUTION}")
    print()
    for method, digest, was, state in report:
        print(f"  {method:<11} {state}")
        print(f"    current    {digest}")
        print(f"    registered {was}")
    moved = [m for m, _, _, s in report if s == "MOVED"]
    print()
    print(f"{len(moved)} of {len(report)} moved: {moved}" if moved
          else "all locks already match the current source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
