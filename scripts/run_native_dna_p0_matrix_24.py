#!/usr/bin/env python3
"""Launch the sealed 4 x 4 x 3 native-DNA matrix at 24 bases.

This versioned launcher uses a distinct 24-base cell driver and rejects a
matrix root that already contains 18-base cells.  The default matrix remains
four methods, four datasets, and train seeds 42/43/44.  As in the canonical
launcher, ``--data-root`` is mandatory, must be below ``/data``, and owns
separate ``runs/`` and ``logs/`` directories.
"""

from __future__ import annotations


from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterator, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_native_dna_p0_matrix as canonical  # noqa: E402
from scripts.aggregate_native_dna_p0_24 import (  # noqa: E402
    Key as AggregateKey,
    _validate_manifest,
)
from scripts.run_native_dna_p0_24 import (  # noqa: E402
    MATCHED_LENGTH,
    configured_canonical_driver,
)


RUNNER = REPO / "scripts" / "run_native_dna_p0_24.py"
DEFAULT_METHODS = canonical.DEFAULT_METHODS
DEFAULT_DATASETS = canonical.DEFAULT_DATASETS
DEFAULT_SEEDS = canonical.DEFAULT_SEEDS
_ORIGINAL_VALIDATE_DATA_ROOT = canonical._validate_data_root


def _validate_data_root_24(path: str | Path) -> Path:
    resolved = _ORIGINAL_VALIDATE_DATA_ROOT(path)
    runs = resolved / "runs"
    if runs.is_dir() and any(
        "_18nt_P0_" in child.name for child in runs.iterdir()
    ):
        raise ValueError(
            "24-base execution requires a separate data root; "
            f"18-base cells already exist below {runs}"
        )
    return resolved


# Declares this process as the 24-base wrapper so the canonical
# entrypoint guard stands down. A module attribute, not an env var:
# an env var would also disable the guard in every child process.
_PATCHED_FIELDS = {
    "WRAPPER_DISPATCH": True,
    "MATCHED_LENGTH": MATCHED_LENGTH,
    "RUNNER": RUNNER,
    "AggregateKey": AggregateKey,
    "_validate_manifest": _validate_manifest,
    "_validate_data_root": _validate_data_root_24,
}


@contextmanager
def configured_canonical_matrix() -> Iterator[None]:
    previous = {
        name: getattr(canonical, name) for name in _PATCHED_FIELDS
    }
    try:
        for name, value in _PATCHED_FIELDS.items():
            setattr(canonical, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(canonical, name, value)


def build_parser():
    with configured_canonical_driver(), configured_canonical_matrix():
        return canonical.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    with configured_canonical_driver(), configured_canonical_matrix():
        return canonical.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
