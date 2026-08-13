"""Immutable run identity and fail-closed lookup (F08).

`train_siglip2._resolve_save_path` derives the result directory from date, tag,
batch, epoch and LR and then calls `os.makedirs(..., exist_ok=True)`. The seed,
the slot count, the bases per slot and the selection/refit mode never reach the
path, so two runs differing only in those share a directory and the second
overwrites the first.

That happened. On 2026-08-12 eight 3-seed cells were launched with a variable
name the launcher does not read, and every seed for a given (dataset, N) wrote
to one path. What survived was worse than nothing: `args.txt` came from whichever
process started last and the evaluation JSON from whichever finished last, so a
directory labelled seed 44 contained seed 43's numbers. It was caught only
because two seeds agreed to full float precision. The wreckage is kept under
`result_quarantine_collided_20260812/`.

Lookup has the same defect in the other direction. `ls -dt <glob> | head -1`
returns the newest match, so an ablation or a lambda variant sharing the prefix
can be picked as the reference run and evaluated as if it were.

The contract here:

    identity   digest over everything that distinguishes one run from another
    claim      refuse a directory already claimed by a DIFFERENT identity, and
               refuse a non-empty directory with no manifest at all
    resolve    match on identity, never on mtime; ambiguity and absence both
               raise instead of guessing

Quarantined collision roots are excluded from resolution: they are preserved as
evidence, and silently reading from them is exactly what the quarantine exists
to prevent.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, List, Optional

MANIFEST_NAME = "run_identity.json"
_SCHEMA_VERSION = 1

#: Directory-name fragments that must never be resolved as a live run.
_EXCLUDED_FRAGMENTS = ("quarantine", "_QUARANTINE", "smoke")


class RunCollision(RuntimeError):
    """A directory is already claimed by a different run, or two claim one."""


class RunNotFound(RuntimeError):
    """No directory carries this identity. Raised instead of falling back."""


@dataclass(frozen=True)
class RunIdentity:
    schema_version: int
    dataset: str
    setting: str
    seed: int
    num_slots: int
    bases_per_slot: int
    total_bases: int
    total_bits: int
    stop_after_epoch: Optional[int]
    epoch_budget: int
    lr_schedule_horizon: int
    sinkhorn_schedule_horizon: int
    selection_mode: str

    @property
    def digest(self) -> str:
        payload = {f.name: getattr(self, f.name) for f in fields(self)
                   if f.name != "schema_version"}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @classmethod
    def from_args(cls, args: Any) -> "RunIdentity":
        m = int(getattr(args, "num_semantic_parts", 0) or 0)
        l = int(getattr(args, "num_codons_per_codebook", 3) or 3)
        budget = int(getattr(args, "epoch", 60) or 60)
        stop = getattr(args, "stop_after_epoch", None)
        lr_h = getattr(args, "lr_schedule_horizon", None)
        sk_h = getattr(args, "sinkhorn_schedule_horizon", None)
        return cls(
            schema_version=_SCHEMA_VERSION,
            dataset=str(getattr(args, "dataset", "")),
            setting=str(getattr(args, "setting", "")),
            seed=int(getattr(args, "random_seed", 42) or 42),
            num_slots=m, bases_per_slot=l,
            total_bases=m * l, total_bits=2 * m * l,
            stop_after_epoch=None if stop is None else int(stop),
            epoch_budget=budget,
            lr_schedule_horizon=int(lr_h) if lr_h else budget,
            sinkhorn_schedule_horizon=int(sk_h) if sk_h else budget,
            selection_mode=str(getattr(args, "selection_mode", "refit")),
        )

    def differing_fields(self, other: "RunIdentity") -> List[str]:
        out = []
        for f in fields(self):
            if f.name == "schema_version":
                continue
            if getattr(self, f.name) != getattr(other, f.name):
                out.append(f.name)
        return out

    def as_record(self) -> dict:
        d = asdict(self)
        d["digest"] = self.digest
        return d


def _manifest_path(run_dir: str) -> str:
    return os.path.join(run_dir, MANIFEST_NAME)


def load_run_manifest(run_dir: str) -> Optional[RunIdentity]:
    p = _manifest_path(run_dir)
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None
    d.pop("digest", None)
    if d.get("schema_version") != _SCHEMA_VERSION:
        return None
    try:
        return RunIdentity(**d)
    except TypeError:
        return None


def write_run_manifest(run_dir: str, identity: RunIdentity) -> str:
    os.makedirs(run_dir, exist_ok=True)
    out = _manifest_path(run_dir)
    tmp = out + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(identity.as_record(), fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    return out


def _looks_occupied(run_dir: str) -> bool:
    """Artefacts that mean a previous run wrote here."""
    if not os.path.isdir(run_dir):
        return False
    markers = ("model_state_dict.pth", "extract_db.npz", "extract_query.npz",
               "args.txt", "log.csv")
    return any(os.path.exists(os.path.join(run_dir, m)) for m in markers)


def claim_run_dir(run_dir: str, identity: RunIdentity) -> str:
    """Take ownership of `run_dir`, or refuse.

    Re-claiming with the SAME identity is a resume and is allowed. A different
    identity, or artefacts with no manifest at all, raise -- merging those is
    what produced a directory holding one seed's args and another's metrics.
    """
    existing = load_run_manifest(run_dir)
    if existing is not None:
        if existing.digest != identity.digest:
            diff = identity.differing_fields(existing)
            raise RunCollision(
                f"{run_dir} is already claimed by a different run; they differ "
                f"in {diff}. Writing here would merge two runs' artefacts -- "
                f"give this one its own directory instead.")
        return run_dir
    if _looks_occupied(run_dir):
        raise RunCollision(
            f"{run_dir} holds run artefacts but no {MANIFEST_NAME}, so the run "
            f"that produced them cannot be identified. Move it aside rather "
            f"than writing into it.")
    write_run_manifest(run_dir, identity)
    return run_dir


def resolve_run_dir(search_root: str, identity: RunIdentity) -> str:
    """Find the ONE directory carrying this identity.

    Matching is on the manifest, never on mtime, so a newer ablation cannot be
    returned as the reference run. Absence and ambiguity both raise: falling
    back to the newest match is the defect this replaces.
    """
    hits: List[str] = []
    for dirpath, dirnames, _ in os.walk(search_root):
        dirnames[:] = [d for d in dirnames
                       if not any(x in d for x in _EXCLUDED_FRAGMENTS)]
        if any(x in dirpath for x in _EXCLUDED_FRAGMENTS):
            continue
        md = load_run_manifest(dirpath)
        if md is not None and md.digest == identity.digest:
            hits.append(dirpath)
    if not hits:
        raise RunNotFound(
            f"no run under {search_root} carries identity {identity.digest[:12]} "
            f"(dataset={identity.dataset} seed={identity.seed} "
            f"M={identity.num_slots} L={identity.bases_per_slot} "
            f"stop={identity.stop_after_epoch} mode={identity.selection_mode}). "
            f"Run it, or point at the right root -- do not substitute a "
            f"similar directory.")
    if len(hits) > 1:
        raise RunCollision(
            f"{len(hits)} directories claim identity {identity.digest[:12]}: "
            f"{sorted(hits)}. Resolve which is authoritative; picking the "
            f"newest is how an ablation got evaluated as the reference run.")
    return hits[0]
