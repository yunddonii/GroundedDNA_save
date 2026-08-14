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

DEFAULT_PREDICTOR = ("/data/yschoi/groundeddna_native_p0/artifacts/"
                     "primo_yield_predictor.npz")


def locks_for(length: int, *, dataset_root: Path | None = None,
              predictor: Path | None = None) -> dict:
    """The method locks a real run at `length` produces.

    Computed in the SAME context the runner uses. 24 bases execute through
    `run_native_dna_p0_24`, which patches implementation paths and a pipeline
    variant into the identity, so computing 24 with the canonical driver under
    an environment override yields digests no real run ever emits -- which is
    how four wrong 24-base digests were registered. Everything else runs the
    canonical driver at that length.
    """
    root = dataset_root or (REPO / "dataset")
    pred = predictor or Path(DEFAULT_PREDICTOR)

    def _compute() -> dict:
        out = {}
        for method in driver.METHODS:
            identity, _ = driver._protocol_identity(
                method=method,
                dataset=PROBE["dataset"],
                setting=PROBE["setting"],
                seed=PROBE["seed"],
                cache_audit={"cache_dir": "<lock-independent>", "blockers": []},
                dataset_root=root,
                predictor=pred if method == "bee2021" else None,
                device="cuda:0",
                **EXECUTION,
            )
            out[method] = agg._method_protocol_lock_digest(identity)
        return out

    if int(length) == 24:
        from scripts.run_native_dna_p0_24 import configured_canonical_driver
        with configured_canonical_driver():
            return _compute()

    # The canonical driver reads its length from the module constant, so the
    # length has to be in force while the identity is built.
    previous_length = driver.MATCHED_LENGTH
    previous_protocol = driver.DEFAULT_PROTOCOL
    protocol = resolve_native_protocol(int(length))
    driver.MATCHED_LENGTH = protocol.length_bases
    driver.DEFAULT_PROTOCOL = protocol
    try:
        return _compute()
    finally:
        driver.MATCHED_LENGTH = previous_length
        driver.DEFAULT_PROTOCOL = previous_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=15,
                        choices=(15, 18, 20, 24))
    parser.add_argument("--dataset-root", default=str(REPO / "dataset"))
    parser.add_argument("--primo-predictor-npz", default=DEFAULT_PREDICTOR)
    parser.add_argument("--json", action="store_true",
                        help="Emit only the digest map, for diffing.")
    args = parser.parse_args()

    protocol = resolve_native_protocol(args.length)
    digests = locks_for(args.length,
                        dataset_root=Path(args.dataset_root),
                        predictor=Path(args.primo_predictor_npz))
    if args.json:
        print(json.dumps(digests, indent=2, sort_keys=True))
        return 0

    try:
        registered = agg.reviewed_method_locks(protocol)
    except SystemExit:
        registered = {}
    print(f"method-protocol locks at {protocol.length_bases} bases "
          f"({protocol.protocol_label})")
    print(f"context: {'run_native_dna_p0_24 wrapper' if args.length == 24 else 'canonical driver'}")
    print(f"execution contract: {EXECUTION}")
    print()
    moved = []
    for method, digest in digests.items():
        known = digest in registered.get(method, frozenset())
        print(f"  {method:<11} {'REGISTERED' if known else 'UNREGISTERED'}")
        print(f"    current  {digest}")
        print(f"    reviewed {sorted(registered.get(method, []))}")
        if not known:
            moved.append(method)
    print()
    print(f"{len(moved)} unregistered: {moved}" if moved
          else "every lock is already reviewed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
