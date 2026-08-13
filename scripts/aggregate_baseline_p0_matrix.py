#!/usr/bin/env python3
"""Validate and aggregate completed common-P0 baseline cells.

Only a final ``p0_run_manifest.json`` whose referenced
``bio_projection.json`` passes identity, SHA-256, protocol, and projection
checks contributes a number.  Duplicate completed cells are intentionally not
resolved by timestamp or score: doing so could introduce test-dependent
selection, so they are reported as ambiguous until the operator identifies the
intended run explicitly.

The default input is the champion-matched train-seed-42 legacy-cache matrix
created by ``run_baseline_p0_matrix.py``.  Its metrics are rendered with a
dagger and labelled diagnostic-only whenever strict cache provenance keeps
``main_protocol_eligible`` false.  UMRCH is always rendered in a separate U2
panel.  The supervised CRH baseline is opt-in and always rendered separately
from U0/U2.  Exact DUH-EG remains a blocked row rather than a fabricated
number.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Mapping, Optional, Sequence

# Run directly as `python scripts/aggregate_baseline_p0_matrix.py`, which puts
# scripts/ on the path rather than the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dna_utils.bit_slice import resolve_bit_slice, selection_metric_by_bit  # noqa: E402
from dna_utils.flat_geometry import resolve_flat_geometry  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DATASETS = ("Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10")
# 30 added 2026-08-10 for the 5-slot / 15-base GroundedDNA variant.
# NOT length-generic: `bit//6` is 6.67 at 40 bits, and F06 established that 48
# bits is 6 slots of 4 bases rather than 8 of 3. Geometry comes from the declared
# panel in dna_utils.flat_geometry; this tuple only lists which budgets exist.
BITS = (30, 36, 40, 48)
DEFAULT_SEEDS = (42,)
U0_VARIANTS = (
    "cibhash",
    "cimon",
    "mls3rduh",
    "greedyhash",
    "bihalf",
    "sdc-paper",
    "oh",
    "hhch",
    "crovca",
)
U2_VARIANTS = ("umrch",)
U2_DATASETS = ("Flickr25k", "MSCOCO", "NUSWIDE")
SUPERVISED_VARIANTS = ("crh-supervised",)
SUPERVISED_DATASETS = DATASETS
DEFAULT_PANELS = ("u0", "u2")

DISPLAY = {
    "cibhash": "CIBHash",
    "cimon": "CIMON",
    "mls3rduh": "MLS³RDUH (paper-cache)",
    "greedyhash": "GreedyHash",
    "bihalf": "Bi-half",
    "sdc-paper": "SDC (paper)",
    "oh": "OH",
    "hhch": "HHCH",
    "crovca": "CroVCA",
    "umrch": "UMRCH",
    "crh-supervised": "CRH (supervised)",
}
METHOD = {
    "cibhash": "cibhash",
    "cimon": "cimon",
    "mls3rduh": "mls3rduh",
    "greedyhash": "greedyhash",
    "bihalf": "bihalf",
    "sdc-paper": "sdc",
    "oh": "oh",
    "hhch": "hhch",
    "crovca": "crovca",
    "umrch": "umrch",
    "crh-supervised": "crh",
}
HORIZON = {
    "cibhash": {dataset: 60 for dataset in DATASETS},
    "cimon": {dataset: 150 for dataset in DATASETS},
    "mls3rduh": {dataset: 150 for dataset in DATASETS},
    "greedyhash": {dataset: 60 for dataset in DATASETS},
    "bihalf": {
        "Flickr25k": 100,
        "MSCOCO": 150,
        "NUSWIDE": 100,
        "CIFAR10": 300,
    },
    "sdc-paper": {dataset: 100 for dataset in DATASETS},
    "oh": {dataset: 200 for dataset in DATASETS},
    "hhch": {dataset: 80 for dataset in DATASETS},
    "crovca": {dataset: 5 for dataset in DATASETS},
    "umrch": {dataset: 100 for dataset in U2_DATASETS},
    "crh-supervised": {
        "Flickr25k": 30,
        "MSCOCO": 30,
        "NUSWIDE": 30,
        "CIFAR10": 300,
    },
}

DUHEG_BLOCK = {
    "variant": "duheg",
    "status": "blocked_not_aggregated",
    "reason": (
        "Exact DUH-EG requires the authors' ordered selected-WordNet noun bank "
        "or an unambiguous selection specification; neither is available."
    ),
}

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

# This transition was reviewed when the persistent, stat-attested digest memo
# was added.  It changes only how immutable cache artifacts are hashed (a
# content SHA-256 is reused after a stable-stat check); it does not change the
# cache bytes, model, loss, training data, retrieval metric, or DNA projection.
# Keep this allow-list cryptographically exact: an unreviewed third digest must
# fall back to unknown/scientifically-relevant drift instead of inheriting the
# exemption by filename.
KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS = {
    "baseline/cache_provenance.py": {
        "before_sha256": (
            "ec761115371f09e6e3a00ab816888188b0df234805920b153ef06e08a598405c"
        ),
        "after_sha256": (
            "4fe40c862a23e265ecf0559232d8b6ac5b84b282df720670c3aed58e3daf588d"
        ),
        "classification": "non_scientific_performance_memo_only",
        "evidence": (
            "reviewed exact-hash transition adds a stat-attested persistent "
            "SHA-256 performance memo; scientific inputs and outputs remain "
            "content-bound by the same artifact digests"
        ),
    },
    "baseline/base_model.py": {
        "before_sha256": (
            "c9f39c05a27cca24ab0084bdd00462b634953499021a67c1cdc18dfc33837a47"
        ),
        "after_sha256": (
            "b9d7e1a0cfab613fd72ce3dc38fafeca82ca99f6121d20138bf22239ef34f3be"
        ),
        "non_scientific_variants_by_sha256": {
            # The old dispatcher remains scientifically equivalent for every
            # method that existed then, but it cannot have launched CRH.
            "c9f39c05a27cca24ab0084bdd00462b634953499021a67c1cdc18dfc33837a47": (
                U0_VARIANTS + U2_VARIANTS
            ),
        },
        "classification": "non_scientific_dispatch_extension_only",
        "evidence": (
            "reviewed exact-hash transition adds only CRH lazy dispatch aliases "
            "and CLI help; existing U0/U2 model construction is unchanged"
        ),
    },
    # 2026-08-10 bit-budget extension (30-bit input-validation branch only),
    # then the 2026-08-13 F06 geometry correction. The second edit replaced the
    # hard-coded `group_size = 6` with the declared panel, which is a NO-OP at
    # 30 and 36 bits (both panels are 6 bits per group: 5x3 and 6x3 codons) and
    # a genuine behaviour change at 48, where the declared panel is 6 slots of 4
    # bases (8 bits per group) rather than the 8 groups of 6 the old constant
    # produced. The 51 historical 48-bit cells were therefore extracted under a
    # geometry the current source no longer produces, so they are scoped OUT of
    # this exemption and stay blocked; only 30- and 36-bit cells are exempt.
    "scripts/extract_flat_baseline.py": {
        "before_sha256": (
            "433be227bf28131a3266e298e37b7a7fcab5e11d41005b75629d6904f81d990e"
        ),
        "after_sha256": (
            "bb05c68c7e613095d941a5b9131a01c52ddbd713d128b09de3d0977329bb6203"
        ),
        "reviewed_sha256": (
            "02c805b0ce914daf8d54469f9a9369cf1cd96cacba55367ad8da1adff8284c4f",
            "433be227bf28131a3266e298e37b7a7fcab5e11d41005b75629d6904f81d990e",
            # 2026-08-12 (21fbb21a): accepted bit budgets gain 40. One line.
            "4574ea770532635b251c802656c089223a8d40a416e74e7ab8b26fc7fbfbea68",
            "bb05c68c7e613095d941a5b9131a01c52ddbd713d128b09de3d0977329bb6203",
        ),
        # Which budgets each historical snapshot is still comparable at. A cell
        # is exempt only where the old code packed it identically to the current
        # code, so 48 never appears (the group size changed there) and neither
        # does a budget the snapshot could not accept at all.
        "non_scientific_bits_by_sha256": {
            # Predates the 30-bit acceptance branch, so it cannot have produced
            # a 30-bit cell; 6 bits per group at 36 matches current.
            "02c805b0ce914daf8d54469f9a9369cf1cd96cacba55367ad8da1adff8284c4f":
                (36,),
            "433be227bf28131a3266e298e37b7a7fcab5e11d41005b75629d6904f81d990e":
                (30, 36),
            # Accepts 40, but packs it as the constant 6 bits per group, which
            # does not divide 40; only its 30- and 36-bit cells are comparable.
            "4574ea770532635b251c802656c089223a8d40a416e74e7ab8b26fc7fbfbea68":
                (30, 36),
        },
        "classification": "non_scientific_bit_budget_extension",
        "evidence": (
            "reviewed exact-hash transitions: (1) widen the accepted checkpoint "
            "bit budget from (36, 48) to (30, 36, 48); (2) F06, take the group "
            "size from the declared panel instead of the constant 6. Identical "
            "packing at 30 and 36 bits; 48-bit cells are excluded by scope "
            "because their geometry did change"
        ),
    },
    "scripts/baseline_val_select_p0.py": {
        "before_sha256": (
            "88a811f65c0334a459d1f91555258641f3272ef6f17dae4dad4fda5af4500895"
        ),
        "after_sha256": (
            "6c66b838740a86c10451d97f3fe9c56de4e37acde889fdf552ecc51edb8311d2"
        ),
        "reviewed_sha256": (
            "88a811f65c0334a459d1f91555258641f3272ef6f17dae4dad4fda5af4500895",
            "770156da4e0dea34e01144acf68d4256e14ce3263c464e53e095e727b60d48cd",
            # 2026-08-12 (21fbb21a): SUPPORTED_BITS gains 40. One line; the
            # selection metric is unchanged at every other budget.
            "6c66b838740a86c10451d97f3fe9c56de4e37acde889fdf552ecc51edb8311d2",
        ),
        "classification": "non_scientific_bit_budget_extension",
        "evidence": (
            "reviewed exact-hash transitions widen SUPPORTED_BITS to include 30 "
            "and then 40, one tuple line each; selection metric is still "
            "raw_{base}base_base_hamming_mAP_at_R with base = bit // 2, so "
            "selection at every other budget is unchanged"
        ),
    },
    "scripts/run_modern_baseline_p0.py": {
        "before_sha256": (
            "9809a70fde66d473540fa11d10752a83453d60ac6bfc9b80d0490a1e7ce7eca5"
        ),
        "after_sha256": (
            "c333bf1aafcf809b57f5e3e46eb977b1666936c4ec962e927972a2804798ad4a"
        ),
        "reviewed_sha256": (
            "9809a70fde66d473540fa11d10752a83453d60ac6bfc9b80d0490a1e7ce7eca5",
            "2d7234a2a8f959f32a1f0fa5199cf00556289084f351359f7cb3cbcbdd4f5b16",
            "1dec886eaed08b4f01cc04c8ebaec81b01952d1bc6913696cdcc7a75830461e0",
            # 2026-08-10: SUPPORTED_BITS (36, 48) -> (30, 36, 48) plus the
            # matching --bit help string, to admit the 30-bit / 15-base budget
            # that matches the 5-slot GroundedDNA variant. Full diff is the
            # tuple and two comment lines; the 36- and 48-bit code paths are
            # byte-identical, so previously-run cells at those budgets remain
            # valid and comparable. Non-scientific for EVERY variant.
            "4e9959b28fd7118ecdbd794555e22ede53064df6fe447c2ab85d30c1c8541a6c",
            # 2026-08-12 (21fbb21a): SUPPORTED_BITS (30, 36, 48) -> adds 40 for
            # the 4-base codon panel. One tuple line; every other budget's code
            # path is byte-identical. Non-scientific for EVERY variant.
            "c333bf1aafcf809b57f5e3e46eb977b1666936c4ec962e927972a2804798ad4a",
        ),
        "non_scientific_variants_by_sha256": {
            # This snapshot predates CRH dispatch and has the stale CIBHash
            # horizon, so neither variant may inherit it.
            "9809a70fde66d473540fa11d10752a83453d60ac6bfc9b80d0490a1e7ce7eca5": tuple(
                variant for variant in U0_VARIANTS + U2_VARIANTS
                if variant != "cibhash"
            ),
            # CRH exists here.  The next edit changes only CIBHash's horizon,
            # making it non-scientific for every other existing variant.
            "2d7234a2a8f959f32a1f0fa5199cf00556289084f351359f7cb3cbcbdd4f5b16": tuple(
                variant
                for variant in U0_VARIANTS + U2_VARIANTS + SUPERVISED_VARIANTS
                if variant != "cibhash"
            ),
            # F15: these two had no scope entry at all, so any cell recorded at
            # either one blocked the whole aggregation as unknown drift -- even
            # though both differ from current only by a SUPPORTED_BITS line.
            # 2026-08-05 auditfix (2747352d) already carries the CIBHash horizon
            # correction, so this snapshot is comparable for every variant,
            # CIBHash included.
            "1dec886eaed08b4f01cc04c8ebaec81b01952d1bc6913696cdcc7a75830461e0":
                U0_VARIANTS + U2_VARIANTS + SUPERVISED_VARIANTS,
            # 2026-08-10 30-bit widening; differs from current only by adding 40
            # to SUPPORTED_BITS for the 4-base codon panel.
            "4e9959b28fd7118ecdbd794555e22ede53064df6fe447c2ab85d30c1c8541a6c":
                U0_VARIANTS + U2_VARIANTS + SUPERVISED_VARIANTS,
        },
        "classification": "non_scientific_for_reviewed_variants_only",
        "evidence": (
            "reviewed exact hashes cover the isolated CRH dispatch addition "
            "and the later CIBHash-only horizon correction; the latter is "
            "scientific for CIBHash and non-scientific only for the explicitly "
            "listed unaffected variants; the two later hashes widen "
            "SUPPORTED_BITS to 30 and then 40 by one tuple line each and are "
            "non-scientific for every variant"
        ),
    },
}
# The per-variant heterogeneity gate consults this set. It must list every
# classification the path audit can hand out for a REVIEWED transition,
# otherwise the two disagree: the path is cleared as non-scientific and the same
# path then blocks its variant as unknown drift. `non_scientific_bit_budget_
# extension` was missing, which is why a 30-bit matrix spanning the seed-42 and
# seed-43/44 roots reported 72 of 108 cells implementation-blocked while every
# path audited clean (F15).
NON_SCIENTIFIC_TRANSITION_CLASSIFICATIONS = frozenset({
    "non_scientific_performance_memo_only",
    "non_scientific_dispatch_extension_only",
    "non_scientific_for_reviewed_variants_only",
    "non_scientific_bit_budget_extension",
})

# Method-defining source profiles are filtered by exact content digest before
# duplicate handling, never by score or timestamp.  This both prevents a
# homogeneous stale implementation from being admitted merely because all of
# its cells agree with one another and lets an explicitly rerun canonical cell
# supersede an excluded historical profile.  In particular, the first
# common-P0 MLS3RDUH queue exposed a paper/release hybrid: release o=0.09N and
# default Linear initialization combined with the paper's SGD momentum.
CANONICAL_VARIANT_SOURCE_PROFILES = {
    "cibhash": {
        "path": "baseline/CIBHash.py",
        "sha256": (
            "ce1a1e3fde2c87eb9fe34644e11f63ca4c84fb757ac0eb17c126ccf0cabc3c8e"
        ),
        "profile": "cibhash-official-head-cache-adaptation-v2",
    },
    "cimon": {
        "path": "baseline/CIMON.py",
        # F05: the pre-fix digest was afd4696e2b36..., whose objective differs
        # from the official implementation (200-bin [0, 2) histogram, and
        # `weight_2` in {0, 2} rather than {0, 1}). It is NOT registered as a
        # non-scientific transition -- cells trained under it must be excluded
        # and retrained, which is what this digest bump enforces.
        "sha256": (
            "6f7a4864d3c245afe095a2d2c6dc3c155197a17e10ebd743f1466b3c1ff1136a"
        ),
        "profile": "cimon-official-objective-v2",
    },
    "mls3rduh": {
        "path": "baseline/MLS3RDUH.py",
        "sha256": (
            "98c8e29b488b71cf60e01835681d26e8231136b5ed30d0ccc1ab33de6998fc8f"
        ),
        "profile": "ijcai2020-paper-cache-v1",
    },
    "greedyhash": {
        "path": "baseline/GreedyHash.py",
        "sha256": (
            "5d80dca3034f1d830b32d80c3c27628d53b131333ead5b1f8ac7fa3e35f3bc89"
        ),
        "profile": "greedyhash-cache-adaptation-v1",
    },
    "bihalf": {
        "path": "baseline/BiHalf.py",
        # D6: the LR decay period is now per dataset, matching the upstream
        # per-dataset scripts. This CHANGES CIFAR-10 results (4 decays -> 2), so
        # it is not registered as a non-scientific transition: cells trained
        # under 4593e5c0... are excluded and must be re-run.
        "sha256": (
            "b41d9dc79ae55501dabca0a10790fb699702eff5bb0b0e89198543190516a2c2"
        ),
        "profile": "bihalf-author-schedule-v2",
    },
    "sdc-paper": {
        "path": "baseline/SDC.py",
        # D6: step_size is derived as int(0.8 * epochs), the upstream config's
        # own expression, instead of the constant 80. Numerically identical at
        # the author horizon of 100, so no stored result changes; the digest
        # moves because the file did.
        "sha256": (
            "bf2e13936866ffc80c163510321d69a5ba67bdc7bef6b5885eaf75edc166b1e6"
        ),
        "profile": "sdc-paper-horizon-derived-step-v2",
    },
    "oh": {
        "path": "baseline/OH.py",
        "sha256": (
            "53edb7e0081bfff56cd17b8f1baf9b559e5e91966ac03bb0732acdeae19f0ef1"
        ),
        "profile": "oh-cache-adaptation-v1",
    },
    "hhch": {
        "path": "baseline/HHCH.py",
        "sha256": (
            "72c1f745dc9b795b99012571204fae74e59cf23a02f9c729de9d39c758856e96"
        ),
        "profile": "hhch-release-schedule-cache-v1",
    },
    "crovca": {
        "path": "baseline/CroVCA.py",
        "sha256": (
            "58f71fb4411afae49c5117a2f476818b1736cfb8f78ea5e0585e53f66076cc4d"
        ),
        "profile": "crovca-cache2v-probe-v1",
    },
    "umrch": {
        "path": "baseline/UMRCH.py",
        "sha256": (
            "c747aca825d14a2e1eb31f5416208ee43446e57856c28753ada7ee15d7e01bf5"
        ),
        "profile": "umrch-fixed-clip-u2-v1",
    },
    "crh-supervised": {
        "path": "baseline/CRH.py",
        "sha256": (
            "ef09c8d29989dad4c98c466880133d52649975fc397a4b2a3a0c82619b37013d"
        ),
        "profile": "crh-supervised-cache-adaptation-v1",
    },
}

COMMON_REQUIRED_IMPLEMENTATION_PATHS = frozenset({
    "scripts/run_modern_baseline_p0.py",
    "baseline/base_model.py",
    "baseline/modern_unsupervised.py",
    "baseline/asset_provenance.py",
    "baseline/cache_provenance.py",
    "scripts/baseline_val_select_p0.py",
    "scripts/extract_flat_baseline.py",
    "scripts/apply_bio_projection.py",
    "dna_utils/bio_constraints.py",
    "val_split.py",
})


def _required_implementation_paths(variant: str) -> frozenset[str]:
    profile = CANONICAL_VARIANT_SOURCE_PROFILES.get(variant)
    if profile is None:
        return frozenset()
    return COMMON_REQUIRED_IMPLEMENTATION_PATHS | {str(profile["path"])}


@dataclass(frozen=True, order=True)
class Key:
    panel: str
    variant: str
    dataset: str
    bit: int
    seed: int

    @property
    def text(self) -> str:
        return (
            f"{self.panel}/{self.variant}/{self.dataset}/"
            f"{self.bit}b/seed{self.seed}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _implementation_fingerprint(snapshot: Mapping[str, str]) -> str:
    """Return a stable digest of an exact path-to-content implementation map."""
    canonical = json.dumps(
        dict(snapshot), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _protocol_identity_digest(identity: Mapping[str, object]) -> str:
    """Hash the complete protocol identity using the runner's canonical JSON."""
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _implementation_snapshot(
        errors: list[str], value: object) -> dict[str, str] | None:
    """Validate a manifest's immutable, repository-relative source snapshot."""
    raw = _mapping(value)
    if raw is None or not raw:
        errors.append("protocol_identity.implementation_sha256: missing/non-empty object")
        return None
    snapshot: dict[str, str] = {}
    for raw_path, raw_digest in raw.items():
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(
                "protocol_identity.implementation_sha256: source path must be "
                f"a non-empty string, found {raw_path!r}")
            continue
        source_path = Path(raw_path)
        try:
            resolved = (REPO / source_path).resolve()
            resolved.relative_to(REPO.resolve())
        except (OSError, ValueError):
            errors.append(
                "protocol_identity.implementation_sha256: source path must be "
                f"repository-relative, found {raw_path!r}")
            continue
        if source_path.is_absolute() or ".." in source_path.parts:
            errors.append(
                "protocol_identity.implementation_sha256: source path must be "
                f"repository-relative, found {raw_path!r}")
            continue
        normalized = source_path.as_posix()
        if not isinstance(raw_digest, str) or SHA256_PATTERN.fullmatch(
                raw_digest) is None:
            errors.append(
                "protocol_identity.implementation_sha256."
                f"{normalized}: missing/invalid SHA-256")
            continue
        if normalized in snapshot and snapshot[normalized] != raw_digest:
            errors.append(
                "protocol_identity.implementation_sha256: normalized source "
                f"path collision for {normalized!r}")
            continue
        snapshot[normalized] = raw_digest
    return snapshot or None


def _legacy_cache_blockers(blockers: object) -> list[str]:
    """Return explicit cache-provenance blockers that force diagnostic status."""
    if not isinstance(blockers, list):
        return []
    return sorted({
        blocker for blocker in blockers
        if isinstance(blocker, str) and blocker.startswith("cache_")
    })


def _manifest_bit(payload: Mapping[str, object]) -> int | None:
    raw = payload.get("bit_length")
    if raw is None:
        identity = _mapping(payload.get("protocol_identity"))
        if identity is not None:
            raw = identity.get("bit")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _panel_for_variant(variant: str) -> str | None:
    if variant in U0_VARIANTS:
        return "u0"
    if variant in U2_VARIANTS:
        return "u2"
    if variant in SUPERVISED_VARIANTS:
        return "supervised"
    return None


def _expected_information_tier(panel: str) -> str:
    try:
        return {
            "u0": "U0",
            "u2": "U2",
            "supervised": "S",
        }[panel]
    except KeyError as error:
        raise ValueError(f"unknown comparison panel: {panel!r}") from error


def _expected_semantic_condition(
        panel: str, dataset: str) -> dict[str, object]:
    if panel == "u0":
        return {
            "information_tier": "U0",
            "term_source_condition": "visual-only",
            "matched_official_taxonomy_dataset": "",
            "selection_verified": None,
        }
    if panel == "u2":
        return {
            "information_tier": "U2",
            "term_source_condition": (
                "target-benchmark-taxonomy-byte-exact"
            ),
            "matched_official_taxonomy_dataset": dataset,
            "selection_verified": None,
        }
    if panel == "supervised":
        return {
            "information_tier": "S",
            "term_source_condition": (
                "benchmark training labels and target class taxonomy"
            ),
            "matched_official_taxonomy_dataset": dataset,
            "selection_verified": None,
            "uses_training_labels_in_objective": True,
            "uses_training_labels_in_center_reassignment": True,
            "comparison_panel": "supervised",
        }
    raise ValueError(f"unknown comparison panel: {panel!r}")


def _manifest_key(payload: Mapping[str, object]) -> Key | None:
    variant = payload.get("variant")
    dataset = payload.get("dataset")
    bit = _manifest_bit(payload)
    try:
        seed = int(payload.get("seed"))
    except (TypeError, ValueError):
        return None
    if not isinstance(variant, str) or not isinstance(dataset, str):
        return None
    panel = _panel_for_variant(variant)
    if panel is None:
        return None
    # New supervised manifests always carry an explicit panel.  Requiring it
    # prevents a CRH result from being relabelled as target-label-free while
    # preserving historical U0/U2 manifests that predate this metadata field.
    if panel == "supervised" and payload.get("comparison_panel") != "supervised":
        return None
    if bit not in BITS:
        return None
    return Key(panel, variant, dataset, int(bit), seed)


def _source_profile_exclusion(
        payload: Mapping[str, object], key: Key) -> dict[str, str] | None:
    """Reject a source-bound method unless its exact declared profile matches."""
    profile = CANONICAL_VARIANT_SOURCE_PROFILES.get(key.variant)
    if profile is None:
        return None
    identity = _mapping(payload.get("protocol_identity"))
    snapshot = (
        _mapping(identity.get("implementation_sha256"))
        if identity is not None else None
    )
    source_path = str(profile["path"])
    expected = str(profile["sha256"])
    actual = snapshot.get(source_path) if snapshot is not None else None
    if actual == expected:
        return None
    return {
        "key": key.text,
        "variant": key.variant,
        "path": source_path,
        "expected_sha256": expected,
        "actual_sha256": str(actual) if actual is not None else "missing",
        "required_profile": str(profile["profile"]),
        "reason": (
            "manifest does not use the predeclared canonical method source; "
            "excluded before duplicate resolution"
        ),
    }


def _expected_keys(
        seeds: Sequence[int],
        panels: Sequence[str] = DEFAULT_PANELS,
        bits: Optional[Sequence[int]] = None) -> list[Key]:
    """F15: the expected matrix spans the REQUESTED budgets.

    With the module constant it always spanned all four, so a finished 30-bit
    panel could never be complete and the CRH fixtures observed exactly double
    their expected counts once the constant grew from two budgets to four.
    """
    bits = resolve_bit_slice(bits)
    selected = frozenset(panels)
    unknown = selected - frozenset({"u0", "u2", "supervised"})
    if unknown:
        raise ValueError(f"unknown comparison panels: {sorted(unknown)}")
    keys: list[Key] = []
    if "u0" in selected:
        keys.extend(
            Key("u0", variant, dataset, bit, seed)
            for variant in U0_VARIANTS
            for dataset in DATASETS
            for bit in bits
            for seed in seeds
        )
    if "u2" in selected:
        keys.extend(
            Key("u2", variant, dataset, bit, seed)
            for variant in U2_VARIANTS
            for dataset in U2_DATASETS
            for bit in bits
            for seed in seeds
        )
    if "supervised" in selected:
        keys.extend(
            Key("supervised", variant, dataset, bit, seed)
            for variant in SUPERVISED_VARIANTS
            for dataset in SUPERVISED_DATASETS
            for bit in bits
            for seed in seeds
        )
    return keys


def _append_equal(errors: list[str], label: str,
                  actual: object, expected: object) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, found {actual!r}")


def _number(errors: list[str], label: str, value: object,
            *, minimum: float = 0.0, maximum: float = 1.0) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label}: expected finite number, found {value!r}")
        return None
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        errors.append(
            f"{label}: expected finite value in [{minimum}, {maximum}], "
            f"found {value!r}")
        return None
    return result


def _declared_path(errors: list[str], label: str, value: object,
                   *, sibling: Path | None = None) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}: missing path")
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and sibling is not None:
        path = sibling.parent / path
    path = path.resolve()
    if sibling is not None and path != sibling.resolve():
        errors.append(
            f"{label}: declared {path}, expected sibling {sibling.resolve()}")
    if not path.is_file():
        errors.append(f"{label}: file does not exist: {path}")
        return None
    return path


def _verify_hash(errors: list[str], label: str, path: Path | None,
                 declared: object, *, enabled: bool) -> str | None:
    if not isinstance(declared, str) or len(declared) != 64:
        errors.append(f"{label}: missing/invalid declared SHA-256")
        return None
    if path is None or not enabled:
        return declared
    actual = _sha256(path)
    if actual != declared:
        errors.append(
            f"{label}: SHA-256 mismatch, declared {declared}, actual {actual}")
    return actual


def _verify_record_file(errors: list[str], label: str, value: object,
                        *, parent: Path, expected_sha: object | None,
                        verify_hashes: bool) -> dict[str, str] | None:
    record = _mapping(value)
    if record is None:
        errors.append(f"{label}: missing artifact record")
        return None
    path = _declared_path(errors, f"{label}.path", record.get("path"))
    if path is not None and path.parent != parent.resolve():
        errors.append(
            f"{label}.path: artifact is outside extraction directory: {path}")
    declared = record.get("sha256")
    _verify_hash(
        errors, f"{label}.sha256", path, declared, enabled=verify_hashes)
    if expected_sha is not None and declared != expected_sha:
        errors.append(
            f"{label}.sha256: bio/run manifests disagree "
            f"({declared!r} != {expected_sha!r})")
    if path is None or not isinstance(declared, str):
        return None
    return {"path": str(path), "sha256": declared}


def _validate_manifest(path: Path, key: Key,
                       *, verify_hashes: bool) -> dict[str, object]:
    errors: list[str] = []
    try:
        raw = _load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "key": key.text,
            "manifest": str(path),
            "status": "invalid",
            "validation_errors": [f"cannot load manifest: {error}"],
        }
    manifest = _mapping(raw)
    if manifest is None:
        return {
            "key": key.text,
            "manifest": str(path),
            "status": "invalid",
            "validation_errors": ["run manifest root must be an object"],
        }

    expected_panel = _panel_for_variant(key.variant)
    if expected_panel != key.panel:
        errors.append(
            "panel/variant mismatch: "
            f"{key.variant!r} belongs to {expected_panel!r}, not {key.panel!r}"
        )
    expected_tier = _expected_information_tier(key.panel)
    expected_semantic = _expected_semantic_condition(
        key.panel, key.dataset)
    expected_base = key.bit // 2
    _append_equal(errors, "variant", manifest.get("variant"), key.variant)
    _append_equal(errors, "method", manifest.get("method"), METHOD[key.variant])
    _append_equal(errors, "dataset", manifest.get("dataset"), key.dataset)
    _append_equal(errors, "bit_length", _manifest_bit(manifest), key.bit)
    if "base_length" in manifest:
        _append_equal(errors, "base_length", manifest.get("base_length"), expected_base)
    _append_equal(errors, "seed", manifest.get("seed"), key.seed)
    _append_equal(errors, "val_seed", manifest.get("val_seed"), 42)
    _append_equal(errors, "information_tier", manifest.get("information_tier"), expected_tier)
    _append_equal(
        errors, "semantic_information_condition",
        manifest.get("semantic_information_condition"), expected_semantic)
    if key.panel == "supervised":
        _append_equal(
            errors, "comparison_panel",
            manifest.get("comparison_panel"), "supervised")
    _append_equal(
        errors, "selection_metric", manifest.get("selection_metric"),
        f"raw_{expected_base}base_base_hamming_mAP_at_R")
    _append_equal(errors, "test_used_for_selection",
                  manifest.get("test_used_for_selection"), False)
    _append_equal(errors, "final_checkpoint_count",
                  manifest.get("final_checkpoint_count"), 1)
    _append_equal(errors, "nominal_schedule_horizon",
                  manifest.get("nominal_schedule_horizon"),
                  HORIZON[key.variant][key.dataset])
    _append_equal(errors, "run_manifest_phase",
                  manifest.get("run_manifest_phase"), "bio_projection_completed")

    deviations = manifest.get("protocol_deviations")
    if deviations != []:
        errors.append(f"protocol_deviations: expected [], found {deviations!r}")
    blockers = manifest.get("main_eligibility_blockers")
    if not isinstance(blockers, list):
        errors.append("main_eligibility_blockers: expected list")
        blockers = []
    declared_main_eligible = manifest.get("main_protocol_eligible") is True
    legacy_cache_blockers = _legacy_cache_blockers(blockers)
    # Aggregation is deliberately one-way: it may downgrade a declared result,
    # but can never promote a legacy-cache result into the paper-main pool.
    main_eligible = declared_main_eligible and not legacy_cache_blockers
    if declared_main_eligible and blockers:
        errors.append(
            "eligibility contradiction: main_protocol_eligible=true with blockers")
    if not declared_main_eligible and not blockers:
        errors.append(
            "eligibility contradiction: ineligible manifest has no blocker")

    best_epoch = manifest.get("best_epoch_zero_based")
    refit_epochs = manifest.get("refit_epochs")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) \
            or best_epoch < 0:
        errors.append(
            f"best_epoch_zero_based: expected non-negative integer, found {best_epoch!r}")
    elif refit_epochs != best_epoch + 1:
        errors.append(
            f"refit_epochs: expected E*+1={best_epoch + 1}, found {refit_epochs!r}")

    implementation_snapshot: dict[str, str] | None = None
    implementation_fingerprint: str | None = None
    identity = _mapping(manifest.get("protocol_identity"))
    if identity is None:
        errors.append("protocol_identity: missing object")
    else:
        _append_equal(errors, "protocol_identity.variant",
                      identity.get("variant"), key.variant)
        _append_equal(errors, "protocol_identity.dataset",
                      identity.get("dataset"), key.dataset)
        _append_equal(errors, "protocol_identity.bit",
                      identity.get("bit"), key.bit)
        _append_equal(errors, "protocol_identity.seed",
                      identity.get("seed"), key.seed)
        _append_equal(errors, "protocol_identity.val_seed",
                      identity.get("val_seed"), 42)
        _append_equal(errors, "protocol_identity.val_ratio",
                      identity.get("val_ratio"), 0.1)
        _append_equal(errors, "protocol_identity.eval_period",
                      identity.get("eval_period"), 5)
        _append_equal(errors, "protocol_identity.horizon",
                      identity.get("horizon"), HORIZON[key.variant][key.dataset])
        _append_equal(errors, "protocol_identity.batch_size_override",
                      identity.get("batch_size_override"), None)
        semantic = _mapping(identity.get("semantic_information_condition"))
        if semantic is None:
            errors.append(
                "protocol_identity.semantic_information_condition: missing object")
        else:
            _append_equal(
                errors, "protocol_identity.semantic_information_condition",
                dict(semantic), expected_semantic)
        implementation_snapshot = _implementation_snapshot(
            errors, identity.get("implementation_sha256"))
        if implementation_snapshot is not None:
            required_paths = _required_implementation_paths(key.variant)
            actual_paths = frozenset(implementation_snapshot)
            if actual_paths != required_paths:
                errors.append(
                    "protocol_identity.implementation_sha256 path-set mismatch; "
                    f"missing={sorted(required_paths - actual_paths)}, "
                    f"unexpected={sorted(actual_paths - required_paths)}")
            implementation_fingerprint = _implementation_fingerprint(
                implementation_snapshot)

    protocol_digest = manifest.get("protocol_digest_sha256")
    if not isinstance(protocol_digest, str) or SHA256_PATTERN.fullmatch(
            protocol_digest) is None:
        errors.append("protocol_digest_sha256: missing/invalid SHA-256")
    elif identity is not None:
        recomputed_protocol_digest = _protocol_identity_digest(identity)
        if recomputed_protocol_digest != protocol_digest:
            errors.append(
                "protocol_digest_sha256: protocol_identity canonical digest "
                f"mismatch; expected {recomputed_protocol_digest}, "
                f"found {protocol_digest}")

    final_checkpoint = _declared_path(
        errors, "final_checkpoint", manifest.get("final_checkpoint"))
    _verify_hash(
        errors, "final_checkpoint_sha256", final_checkpoint,
        manifest.get("final_checkpoint_sha256"), enabled=verify_hashes)

    extraction_dir = path.parent.resolve()
    declared_extraction = manifest.get("extraction_dir")
    if not isinstance(declared_extraction, str) \
            or Path(declared_extraction).expanduser().resolve() != extraction_dir:
        errors.append(
            f"extraction_dir: expected {extraction_dir}, found {declared_extraction!r}")

    extraction_hashes = _mapping(manifest.get("extraction_artifact_sha256"))
    if extraction_hashes is None:
        errors.append("extraction_artifact_sha256: missing object")
        extraction_hashes = {}
    for filename in ("extract_db.npz", "extract_query.npz", "args_extract.txt"):
        artifact = extraction_dir / filename
        if not artifact.is_file():
            errors.append(f"extraction artifact missing: {artifact}")
            artifact_path: Path | None = None
        else:
            artifact_path = artifact
        _verify_hash(
            errors, f"extraction_artifact_sha256.{filename}", artifact_path,
            extraction_hashes.get(filename), enabled=verify_hashes)

    protocol_manifest = _declared_path(
        errors, "protocol_manifest", manifest.get("protocol_manifest"),
        sibling=extraction_dir / "p0_protocol_manifest.json")
    _verify_hash(
        errors, "protocol_manifest_sha256", protocol_manifest,
        manifest.get("protocol_manifest_sha256"), enabled=verify_hashes)

    selection_artifact = _declared_path(
        errors, "selection_artifact", manifest.get("selection_artifact"))
    _verify_hash(
        errors, "selection_artifact_sha256", selection_artifact,
        manifest.get("selection_artifact_sha256"), enabled=verify_hashes)

    bio_sibling = extraction_dir / "bio_projection.json"
    bio_path = _declared_path(
        errors, "bio_projection_manifest",
        manifest.get("bio_projection_manifest"), sibling=bio_sibling)
    bio_declared_sha = manifest.get("bio_projection_manifest_sha256")
    bio_actual_sha = _verify_hash(
        errors, "bio_projection_manifest_sha256", bio_path,
        bio_declared_sha, enabled=verify_hashes)
    sidecar = _declared_path(
        errors, "bio_projection_sidecar",
        manifest.get("bio_projection_sidecar"),
        sibling=extraction_dir / "bio_projection.json.sha256")
    _verify_hash(
        errors, "bio_projection_sidecar_sha256", sidecar,
        manifest.get("bio_projection_sidecar_sha256"), enabled=verify_hashes)
    if sidecar is not None and bio_path is not None:
        try:
            sidecar_digest = sidecar.read_text(encoding="utf-8").strip().split()[0]
        except (OSError, IndexError) as error:
            errors.append(f"bio_projection_sidecar content: {error}")
        else:
            expected_digest = bio_actual_sha or bio_declared_sha
            if sidecar_digest != expected_digest:
                errors.append(
                    "bio_projection_sidecar content does not bind bio manifest")

    map_pre: float | None = None
    map_post: float | None = None
    dna_unique_pre: float | None = None
    dna_unique_post: float | None = None
    bio_paper_eligible = False
    cell_payload: Mapping[str, object] | None = None
    if bio_path is not None:
        try:
            bio_raw = _load_json(bio_path)
            bio = _mapping(bio_raw)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot load bio projection manifest: {error}")
            bio = None
        if bio is None:
            errors.append("bio projection manifest root must be an object")
        else:
            config = _mapping(bio.get("config"))
            if config is None:
                errors.append("bio config: missing object")
            else:
                _append_equal(errors, "bio expected_length",
                              config.get("expected_length"), expected_base)
                _append_equal(errors, "bio max_homopolymer_run",
                              config.get("max_homopolymer_run"), 3)
                _append_equal(errors, "bio gc_min_frac",
                              config.get("gc_min_frac"), 0.4)
                _append_equal(errors, "bio gc_max_frac",
                              config.get("gc_max_frac"), 0.6)
            cells = bio.get("cells")
            if not isinstance(cells, list) or len(cells) != 1:
                errors.append("bio cells: expected exactly one cell")
            else:
                cell_payload = _mapping(cells[0])
                if cell_payload is None:
                    errors.append("bio cell must be an object")

    if cell_payload is not None:
        cell = cell_payload
        _append_equal(errors, "bio cell name", cell.get("name"), key.variant)
        _append_equal(errors, "bio cell dataset", cell.get("dataset"), key.dataset)
        _append_equal(errors, "bio cell L", cell.get("L"), expected_base)
        _append_equal(errors, "bio post_compliance_db",
                      cell.get("post_compliance_db"), 1.0)
        _append_equal(errors, "bio post_compliance_qy",
                      cell.get("post_compliance_qy"), 1.0)
        _append_equal(errors, "bio projection_failures_db",
                      cell.get("projection_failures_db"), 0)
        _append_equal(errors, "bio projection_failures_qy",
                      cell.get("projection_failures_qy"), 0)
        _append_equal(errors, "bio projection invariant",
                      cell.get("projection_invariant_satisfied"), True)
        _append_equal(
            errors, "bio gc_count_range", cell.get("gc_count_range"),
            [math.ceil(expected_base * 0.4), math.floor(expected_base * 0.6)])
        map_pre = _number(errors, "map_at_R_pre", cell.get("map_at_R_pre"))
        map_post = _number(errors, "map_at_R_post", cell.get("map_at_R_post"))
        dna_unique_pre = _number(
            errors, "dna_unique_pre", cell.get("dna_unique_pre"))
        dna_unique_post = _number(
            errors, "dna_unique_post", cell.get("dna_unique_post"))
        r_value = cell.get("R")
        if isinstance(r_value, bool) or not isinstance(r_value, int) or r_value <= 0:
            errors.append(f"R: expected positive integer, found {r_value!r}")
        bio_paper_eligible = cell.get("paper_result_eligible") is True
        _append_equal(errors, "bio/main eligibility agreement",
                      bio_paper_eligible, main_eligible)

        inputs = _mapping(cell.get("input_artifacts")) or {}
        outputs = _mapping(cell.get("output_artifacts")) or {}
        run_outputs = _mapping(manifest.get("bio_projected_artifacts")) or {}
        _verify_record_file(
            errors, "bio input db", inputs.get("db"), parent=extraction_dir,
            expected_sha=extraction_hashes.get("extract_db.npz"),
            verify_hashes=verify_hashes)
        _verify_record_file(
            errors, "bio input query", inputs.get("query"), parent=extraction_dir,
            expected_sha=extraction_hashes.get("extract_query.npz"),
            verify_hashes=verify_hashes)
        for name in ("db", "query"):
            run_record = _mapping(run_outputs.get(name)) or {}
            _verify_record_file(
                errors, f"bio output {name}", outputs.get(name),
                parent=extraction_dir, expected_sha=run_record.get("sha256"),
                verify_hashes=verify_hashes)

        provenance = _mapping(cell.get("training_provenance"))
        if provenance is None:
            errors.append("bio training_provenance: missing object")
        else:
            _append_equal(
                errors, "bio training protocol identity",
                provenance.get("protocol_identity_sha256"), protocol_digest)
            checkpoint_record = _mapping(provenance.get("checkpoint"))
            if checkpoint_record is None:
                errors.append("bio training checkpoint record: missing object")
            else:
                _append_equal(
                    errors, "bio checkpoint SHA",
                    checkpoint_record.get("sha256"),
                    manifest.get("final_checkpoint_sha256"))

    expected_status = (
        "paper_result_eligible" if main_eligible
        else "completed_not_paper_eligible")
    _append_equal(errors, "bio_projection_status",
                  manifest.get("bio_projection_status"), expected_status)

    if errors:
        status = "invalid"
    elif main_eligible and bio_paper_eligible:
        status = "complete_main_eligible"
    elif not deviations and blockers:
        status = "complete_diagnostic_only"
    else:
        status = "complete_ineligible"

    return {
        "key": key.text,
        "panel": key.panel,
        "variant": key.variant,
        "display_name": DISPLAY[key.variant],
        "dataset": key.dataset,
        "bit": key.bit,
        "base_length": expected_base,
        "seed": key.seed,
        "manifest": str(path.resolve()),
        "manifest_sha256": _sha256(path) if verify_hashes else None,
        "status": status,
        "declared_main_protocol_eligible": declared_main_eligible,
        "main_protocol_eligible": main_eligible,
        "main_eligibility_blockers": blockers,
        "legacy_cache_diagnostic_only": bool(legacy_cache_blockers),
        "legacy_cache_blockers": legacy_cache_blockers,
        "paper_table_eligible": (
            status == "complete_main_eligible" and not legacy_cache_blockers),
        "map_at_R_pre": map_pre,
        "map_at_R_post": map_post,
        "delta_projection": (
            None if map_pre is None or map_post is None else map_post - map_pre),
        "dna_unique_pre": dna_unique_pre,
        "dna_unique_post": dna_unique_post,
        "best_epoch_zero_based": manifest.get("best_epoch_zero_based"),
        "refit_epochs": manifest.get("refit_epochs"),
        "protocol_digest_sha256": protocol_digest,
        "implementation_sha256": implementation_snapshot,
        "implementation_fingerprint_sha256": implementation_fingerprint,
        "bio_projection_manifest": None if bio_path is None else str(bio_path),
        "bio_projection_manifest_sha256": bio_declared_sha,
        "validation_errors": errors,
    }


def _audit_implementation_fingerprints(
        records: Mapping[Key, dict[str, object]], *,
        repo: Path = REPO,
        current_source_sha256: Mapping[str, str | None] | None = None,
        ) -> dict[str, object]:
    """Audit exact implementation snapshots across cells and against source.

    Every scientific source snapshot must match the current audited source,
    even when all historical cells share the same stale digest.  Otherwise a
    homogeneous old implementation could be admitted simply because it is
    internally consistent.  Only the cryptographically exact, reviewed
    non-scientific transitions declared above are exempted.
    """
    complete = [
        record for record in records.values()
        if str(record.get("status", "")).startswith("complete_")
    ]
    paths: dict[str, dict[str, list[str]]] = {}
    variants: dict[str, list[dict[str, object]]] = {}
    cell_variant: dict[str, str] = {}
    cell_bit: dict[str, int] = {}
    for record in complete:
        snapshot = record.get("implementation_sha256")
        if not isinstance(snapshot, Mapping):
            # This is normally caught during manifest validation.  Retain an
            # explicit audit result for hand-constructed/older record objects.
            continue
        record_key = str(record["key"])
        record_variant = str(record["variant"])
        cell_variant[record_key] = record_variant
        try:
            cell_bit[record_key] = int(record["bit"])
        except (KeyError, TypeError, ValueError):
            cell_bit[record_key] = -1
        variants.setdefault(record_variant, []).append(record)
        for source_path, digest in snapshot.items():
            if isinstance(source_path, str) and isinstance(digest, str):
                paths.setdefault(source_path, {}).setdefault(digest, []).append(
                    record_key)

    path_audit: list[dict[str, object]] = []
    blocking_by_cell: dict[str, set[str]] = {}
    known_warnings: list[dict[str, object]] = []
    unknown_current_drift: list[str] = []
    missing_current_source: list[str] = []
    unknown_cross_cell_drift: list[str] = []
    path_classification: dict[str, str] = {}
    for source_path in sorted(paths):
        versions = paths[source_path]
        if current_source_sha256 is not None:
            current_digest = current_source_sha256.get(source_path)
        else:
            candidate = (repo / source_path).resolve()
            try:
                candidate.relative_to(repo.resolve())
            except ValueError:
                current_digest = None
            else:
                current_digest = _sha256(candidate) if candidate.is_file() else None

        recorded_digests = set(versions)
        cross_cell = len(recorded_digests) > 1
        current_mismatches = sorted(
            digest for digest in recorded_digests
            if current_digest is not None and digest != current_digest)
        transition = KNOWN_NON_SCIENTIFIC_IMPLEMENTATION_TRANSITIONS.get(
            source_path)
        known_transition = False
        if transition is not None and current_digest == transition["after_sha256"]:
            raw_reviewed = transition.get("reviewed_sha256")
            if isinstance(raw_reviewed, (tuple, list, set, frozenset)):
                reviewed = {str(digest) for digest in raw_reviewed}
            else:
                reviewed = {
                    str(transition["before_sha256"]),
                    str(transition["after_sha256"]),
                }
            observed = recorded_digests | {current_digest}
            scope_ok = True
            raw_scopes = transition.get(
                "non_scientific_variants_by_sha256")
            if raw_scopes is not None:
                if not isinstance(raw_scopes, Mapping):
                    scope_ok = False
                else:
                    for digest in current_mismatches:
                        allowed = raw_scopes.get(digest)
                        if not isinstance(
                            allowed, (tuple, list, set, frozenset)
                        ) or any(
                            cell_variant.get(cell) not in allowed
                            for cell in versions[digest]
                        ):
                            scope_ok = False
                            break
            # F15/F06: an edit can be a no-op at one bit budget and a real
            # behaviour change at another. Without this, the 48-bit cells --
            # extracted as 8 groups of 6 rather than the declared 6 slots of 4
            # -- would ride in on an exemption written for 30/36.
            raw_bit_scopes = transition.get("non_scientific_bits_by_sha256")
            if scope_ok and raw_bit_scopes is not None:
                if not isinstance(raw_bit_scopes, Mapping):
                    scope_ok = False
                else:
                    for digest in current_mismatches:
                        allowed_bits = raw_bit_scopes.get(digest)
                        if not isinstance(
                            allowed_bits, (tuple, list, set, frozenset)
                        ) or any(
                            cell_bit.get(cell) not in allowed_bits
                            for cell in versions[digest]
                        ):
                            scope_ok = False
                            break
            known_transition = observed.issubset(reviewed) and scope_ok

        if known_transition and (cross_cell or current_mismatches):
            classification = str(transition["classification"])
            warning = {
                "path": source_path,
                "classification": classification,
                "before_sha256": transition["before_sha256"],
                "after_sha256": transition["after_sha256"],
                "evidence": transition["evidence"],
                "affected_cells": sorted({
                    cell for digest in current_mismatches
                    for cell in versions[digest]
                }),
            }
            known_warnings.append(warning)
        elif cross_cell:
            classification = "unknown_cross_cell_implementation_drift"
            unknown_cross_cell_drift.append(source_path)
            for cells in versions.values():
                for cell in cells:
                    blocking_by_cell.setdefault(cell, set()).add(
                        f"unknown implementation SHA drift: {source_path}")
        elif current_digest is None:
            classification = "current_source_missing"
            missing_current_source.append(source_path)
            for cells in versions.values():
                for cell in cells:
                    blocking_by_cell.setdefault(cell, set()).add(
                        f"current implementation source missing: {source_path}")
        elif current_mismatches:
            classification = "historical_snapshot_differs_from_current_source"
            unknown_current_drift.append(source_path)
            for digest in current_mismatches:
                for cell in versions[digest]:
                    blocking_by_cell.setdefault(cell, set()).add(
                        f"implementation SHA differs from current source: "
                        f"{source_path}")
        else:
            classification = "consistent_with_current_source"
        path_classification[source_path] = classification
        path_audit.append({
            "path": source_path,
            "classification": classification,
            "cross_cell_heterogeneous": cross_cell,
            "recorded_versions": [
                {"sha256": digest, "cells": sorted(cells)}
                for digest, cells in sorted(versions.items())
            ],
            "current_sha256": current_digest,
            "recorded_hashes_matching_current": (
                [] if current_digest is None
                else sorted(digest for digest in recorded_digests
                            if digest == current_digest)
            ),
            "recorded_hashes_differing_from_current": current_mismatches,
        })

    variant_audit: list[dict[str, object]] = []
    heterogeneous_variants: list[str] = []
    for variant in sorted(variants):
        variant_records = variants[variant]
        groups: dict[str, list[str]] = {}
        snapshots: list[tuple[str, Mapping[str, object]]] = []
        for record in variant_records:
            fingerprint = str(record.get("implementation_fingerprint_sha256"))
            groups.setdefault(fingerprint, []).append(str(record["key"]))
            snapshot = record.get("implementation_sha256")
            if isinstance(snapshot, Mapping):
                snapshots.append((str(record["key"]), snapshot))
        heterogeneous = len(groups) > 1
        differing_paths: set[str] = set()
        presence_drift_paths: set[str] = set()
        if heterogeneous:
            union_paths = {
                path for _, snapshot in snapshots for path in snapshot
                if isinstance(path, str)
            }
            for source_path in union_paths:
                observed = {snapshot.get(source_path, "<missing>")
                            for _, snapshot in snapshots}
                if len(observed) > 1:
                    differing_paths.add(source_path)
                if "<missing>" in observed:
                    presence_drift_paths.add(source_path)
        unknown_paths = sorted(
            source_path for source_path in differing_paths
            if source_path in presence_drift_paths
            or path_classification.get(source_path)
            not in NON_SCIENTIFIC_TRANSITION_CLASSIFICATIONS
        )
        comparison_safe = not unknown_paths
        if heterogeneous and not comparison_safe:
            heterogeneous_variants.append(variant)
            for record in variant_records:
                blocking_by_cell.setdefault(str(record["key"]), set()).add(
                    f"heterogeneous implementation fingerprint within {variant}")
        variant_audit.append({
            "variant": variant,
            "heterogeneous": heterogeneous,
            "comparison_safe": comparison_safe,
            "differing_paths": sorted(differing_paths),
            "presence_drift_paths": sorted(presence_drift_paths),
            "unknown_differing_paths": unknown_paths,
            "fingerprints": [
                {"sha256": digest, "cells": sorted(cells)}
                for digest, cells in sorted(groups.items())
            ],
        })

    blocked_cells = {
        cell: sorted(reasons)
        for cell, reasons in sorted(blocking_by_cell.items())
    }
    if blocked_cells:
        status = "blocked_unverified_implementation_source"
    elif known_warnings and (unknown_current_drift or missing_current_source):
        status = "warning_known_non_scientific_and_current_source_drift"
    elif known_warnings:
        status = "warning_known_non_scientific_performance_memo_transition"
    elif unknown_current_drift or missing_current_source:
        status = "warning_historical_snapshot_differs_from_current_source"
    elif complete:
        status = "consistent_with_current_source"
    else:
        status = "no_completed_cells_to_audit"
    return {
        "status": status,
        "comparison_safe": not blocked_cells,
        "completed_records_audited": len(complete),
        "blocked_cells": blocked_cells,
        "unknown_cross_cell_drift_paths": sorted(unknown_cross_cell_drift),
        "heterogeneous_variants_with_unknown_drift": heterogeneous_variants,
        "historical_snapshot_current_drift_paths": sorted(unknown_current_drift),
        "missing_current_source_paths": sorted(missing_current_source),
        "known_non_scientific_warnings": known_warnings,
        "variant_fingerprints": variant_audit,
        "path_audit": path_audit,
    }


def _apply_implementation_admission(
        records: Mapping[Key, dict[str, object]],
        audit: Mapping[str, object]) -> None:
    """Annotate aggregation eligibility without mutating source manifests."""
    raw_blocked = _mapping(audit.get("blocked_cells")) or {}
    for record in records.values():
        if not str(record.get("status", "")).startswith("complete_"):
            continue
        reasons = raw_blocked.get(str(record["key"]), [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        comparison_eligible = not reasons
        record["implementation_comparison_eligible"] = comparison_eligible
        record["implementation_audit_blockers"] = list(reasons)
        record["paper_table_eligible"] = bool(
            record.get("main_protocol_eligible") is True
            and record.get("legacy_cache_diagnostic_only") is not True
            and comparison_eligible
        )


def _record_admitted(record: Mapping[str, object] | None) -> bool:
    return bool(
        record is not None
        and str(record.get("status", "")).startswith("complete_")
        and record.get("implementation_comparison_eligible") is not False
    )


def _aggregate_cell(records: Mapping[Key, dict[str, object]], *, panel: str,
                    variant: str, dataset: str, bit: int,
                    seeds: Sequence[int]) -> dict[str, object]:
    selected = [records.get(Key(panel, variant, dataset, bit, seed))
                for seed in seeds]
    valid = [record for record in selected if _record_admitted(record)]
    values = [float(record["map_at_R_post"]) for record in valid
              if record.get("map_at_R_post") is not None]
    complete = len(values) == len(seeds)
    comparison_mean = statistics.mean(values) if values else None
    comparison_std = statistics.stdev(values) if len(values) >= 2 else None
    all_main_protocol_eligible = bool(
        complete and all(
            record.get("main_protocol_eligible") is True for record in valid))
    all_paper_table_eligible = bool(
        complete and all(
            record.get("paper_table_eligible") is True for record in valid))
    diagnostic_only = bool(complete and not all_paper_table_eligible)
    if not valid:
        aggregate_status = "missing"
    elif not complete:
        aggregate_status = "partial"
    elif all_paper_table_eligible:
        aggregate_status = "complete_paper_table_eligible"
    else:
        aggregate_status = "complete_diagnostic_only"
    return {
        "panel": panel,
        "variant": variant,
        "dataset": dataset,
        "bit": bit,
        "seeds_expected": list(seeds),
        "seeds_complete": [int(record["seed"]) for record in valid],
        "all_seeds_complete": complete,
        "aggregate_status": aggregate_status,
        "all_records_main_protocol_eligible": all_main_protocol_eligible,
        "all_records_paper_table_eligible": all_paper_table_eligible,
        "paper_table_eligible": all_paper_table_eligible,
        "diagnostic_only": diagnostic_only,
        # Deliberately reserve the unqualified mean/std fields for metrics that
        # may be copied into a paper table.  Diagnostic numbers remain
        # available under explicitly qualified names and in the records.
        "mean_map_at_R_post": (
            comparison_mean if all_paper_table_eligible else None),
        "sample_std_map_at_R_post": (
            comparison_std if all_paper_table_eligible else None),
        "admitted_comparison_mean_map_at_R_post": comparison_mean,
        "admitted_comparison_sample_std_map_at_R_post": comparison_std,
        "diagnostic_mean_map_at_R_post": (
            comparison_mean if diagnostic_only else None),
        "diagnostic_sample_std_map_at_R_post": (
            comparison_std if diagnostic_only else None),
        "records": [None if record is None else record["key"] for record in selected],
    }


def _cell_text(records: Mapping[Key, dict[str, object]], *, panel: str,
               variant: str, dataset: str, bit: int,
               seeds: Sequence[int]) -> str:
    selected = [records.get(Key(panel, variant, dataset, bit, seed))
                for seed in seeds]
    if any(record is not None and record.get("status") == "duplicate"
           for record in selected):
        return "DUP"
    if any(record is not None and record.get("status") == "invalid"
           for record in selected):
        return "ERR"
    if any(record is not None
           and str(record.get("status", "")).startswith("complete_")
           and record.get("implementation_comparison_eligible") is False
           for record in selected):
        return "SHA-DRIFT"
    valid = [record for record in selected if _record_admitted(record)]
    if not valid:
        return "-"
    if len(valid) != len(seeds):
        return f"PARTIAL {len(valid)}/{len(seeds)}"
    values = [float(record["map_at_R_post"]) for record in valid]
    diagnostic = any(record.get("status") != "complete_main_eligible"
                     for record in valid)
    suffix = "†" if diagnostic else ""
    if len(values) == 1:
        return f"{values[0]:.4f}{suffix}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}{suffix}"


def _markdown(payload: Mapping[str, object],
              records: Mapping[Key, dict[str, object]],
              seeds: Sequence[int]) -> str:
    summary = _mapping(payload["summary"]) or {}
    # F15: sections follow the slice this payload was aggregated over. Reading
    # it back off the payload keeps the prose and the numbers in agreement even
    # when an old JSON predates the field.
    _proto = _mapping(payload.get("protocol")) or {}
    _raw_slice = _proto.get("requested_bit_slice")
    report_bits = resolve_bit_slice(
        _raw_slice if isinstance(_raw_slice, (list, tuple)) else None)
    paper_admission = _mapping(payload.get("paper_table_admission")) or {}
    implementation = _mapping(payload.get("implementation_audit")) or {}
    protocol = _mapping(payload.get("protocol")) or {}
    raw_panels = protocol.get("comparison_panels", list(DEFAULT_PANELS))
    selected_panels = frozenset(
        str(panel) for panel in raw_panels
    ) if isinstance(raw_panels, (list, tuple)) else frozenset(DEFAULT_PANELS)
    raw_source_excluded = payload.get("source_profile_excluded_manifests")
    source_excluded = (
        raw_source_excluded if isinstance(raw_source_excluded, list) else [])
    lines = [
        "# Common-P0 baseline matrix aggregation",
        "",
        "> **Eligibility warning:** `†` values completed the requested raw-code → DNA "
        "conversion and common biological post-processing, but are diagnostic-only "
        "when legacy cache provenance keeps `main_protocol_eligible=false`. They must "
        "not be copied into the paper main table as eligible results.",
        "",
        f"- Train seed(s): `{', '.join(map(str, seeds))}`",
        "- Validation split: ratio `0.1`, seed `42`; E* chosen by raw base-Hamming mAP@R",
        # F15: was a hard-coded 36/48 string, which described the wrong panel
        # for every 30-bit aggregation ever emitted.
        "- Budgets: " + ", ".join(
            f"`{b} bit \u2192 {resolve_flat_geometry(b).total_bases} bases "
            f"({resolve_flat_geometry(b).num_codebooks}x{resolve_flat_geometry(b).bases_per_codebook})`"
            for b in report_bits),
        "- Post-processing: exact common DNA projection on both query and database",
        f"- Complete: {summary.get('complete', 0)} / {summary.get('expected', 0)}; "
        f"missing: {summary.get('missing', 0)}; invalid: {summary.get('invalid', 0)}; "
        f"duplicate: {summary.get('duplicate', 0)}; implementation-blocked: "
        f"{summary.get('implementation_comparison_blocked', 0)}",
        "- Noncanonical source-profile manifests excluded before duplicate "
        f"handling: {summary.get('source_profile_excluded', 0)}",
        f"- Implementation audit: `{implementation.get('status', 'unavailable')}`",
        f"- Strict paper-table admission: `{paper_admission.get('eligible', False)}`; "
        f"reasons: `{paper_admission.get('reason_codes', [])}`",
        "- Three-seed mean±std: " + (
            "candidate aggregate shown below" if len(seeds) >= 3
            else "pending (the current matrix is single-seed)"),
        "",
    ]
    if "u0" in selected_panels:
        for bit in report_bits:
            lines.extend([
                f"## U0 visual-only — {bit} bits / {bit // 2} bases",
                "",
                "| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |",
                "|---|---:|---:|---:|---:|",
            ])
            for variant in U0_VARIANTS:
                values = [
                    _cell_text(
                        records, panel="u0", variant=variant, dataset=dataset,
                        bit=bit, seeds=seeds)
                    for dataset in DATASETS
                ]
                lines.append(
                    f"| {DISPLAY[variant]} | " + " | ".join(values) + " |")
            lines.append("")

    if "u2" in selected_panels:
        for bit in report_bits:
            lines.extend([
                f"## U2 taxonomy-assisted — {bit} bits / {bit // 2} bases",
                "",
                "UMRCH consumes the exact target benchmark taxonomy and is not part of "
                "the strict U0 headline comparison.",
                "",
                "| Method | Flickr25K | MSCOCO | NUS-WIDE |",
                "|---|---:|---:|---:|",
            ])
            values = [
                _cell_text(
                    records, panel="u2", variant="umrch", dataset=dataset,
                    bit=bit, seeds=seeds)
                for dataset in U2_DATASETS
            ]
            lines.append("| UMRCH | " + " | ".join(values) + " |")
            lines.append("")

    if "supervised" in selected_panels:
        for bit in report_bits:
            lines.extend([
                f"## Supervised baseline — {bit} bits / {bit // 2} bases",
                "",
                "CRH consumes benchmark training labels and the target class taxonomy. "
                "It is reported only as a supervised baseline and must not be merged "
                "into either the U0 or U2 ranking.",
                "",
                "| Method | Flickr25K | MSCOCO | NUS-WIDE | CIFAR-10 |",
                "|---|---:|---:|---:|---:|",
            ])
            for variant in SUPERVISED_VARIANTS:
                values = [
                    _cell_text(
                        records, panel="supervised", variant=variant,
                        dataset=dataset, bit=bit, seeds=seeds)
                    for dataset in SUPERVISED_DATASETS
                ]
                lines.append(
                    f"| {DISPLAY[variant]} | " + " | ".join(values) + " |")
            lines.append("")

    lines.extend([
        "## Implementation fingerprint audit",
        "",
        "Old run manifests were not rewritten. Their exact recorded source "
        "fingerprints are compared across cells and against the current "
        "worktree; only unknown cross-cell drift blocks aggregation.",
        "",
        f"- Status: **{implementation.get('status', 'unavailable')}**",
        "- Cross-cell comparison safe: "
        f"`{str(implementation.get('comparison_safe', False)).lower()}`",
        "",
    ])
    raw_path_audit = implementation.get("path_audit")
    path_differences = [
        item for item in raw_path_audit
        if isinstance(item, Mapping)
        and item.get("classification") != "consistent_with_current_source"
    ] if isinstance(raw_path_audit, list) else []
    if path_differences:
        lines.extend([
            "| Source path | Recorded SHA-256 version(s) | Current SHA-256 | Classification |",
            "|---|---|---|---|",
        ])
        for item in path_differences:
            raw_versions = item.get("recorded_versions")
            versions = raw_versions if isinstance(raw_versions, list) else []
            recorded = ", ".join(
                f"`{version.get('sha256')}`"
                for version in versions if isinstance(version, Mapping)
            ) or "-"
            current = item.get("current_sha256")
            current_text = f"`{current}`" if isinstance(current, str) else "missing"
            lines.append(
                f"| `{item.get('path')}` | {recorded} | {current_text} | "
                f"`{item.get('classification')}` |")
        lines.append("")
    raw_known = implementation.get("known_non_scientific_warnings")
    known = raw_known if isinstance(raw_known, list) else []
    for warning in known:
        if not isinstance(warning, Mapping):
            continue
        lines.append(
            "- **Non-scientific implementation warning:** "
            f"`{warning.get('path')}` — {warning.get('evidence')}."
        )
    if known:
        lines.append("")

    if source_excluded:
        lines.extend([
            "## Canonical source-profile exclusions",
            "",
            "These manifests were rejected by exact source digest before "
            "duplicate resolution; no score or timestamp was consulted.",
            "",
            "| Cell | Required profile | Actual source SHA-256 | Manifest |",
            "|---|---|---|---|",
        ])
        for item in source_excluded:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"| `{item.get('key')}` | `{item.get('required_profile')}` | "
                f"`{item.get('actual_sha256')}` | `{item.get('manifest')}` |"
            )
        lines.append("")

    lines.extend([
        "## Exact DUH-EG",
        "",
        f"- Status: **{DUHEG_BLOCK['status']}**",
        f"- Reason: {DUHEG_BLOCK['reason']}",
        "",
        "## Audit legend",
        "",
        "- `-`: no completed manifest found.",
        "- `ERR`: artifact or protocol validation failed; no metric was admitted.",
        "- `DUP`: multiple completed manifests exist for one cell; no automatic "
        "test-dependent choice was made.",
        "- `SHA-DRIFT`: unknown cross-cell implementation drift; metric was not admitted.",
        "- `PARTIAL`: only some requested seeds completed.",
        "- `†`: completed but not main-table eligible (typically legacy cache provenance).",
        "",
    ])
    invalid_keys = payload.get("invalid_cells", [])
    duplicate_keys = payload.get("duplicate_cells", [])
    missing_keys = payload.get("missing_cells", [])
    if invalid_keys:
        lines.extend(["### Invalid cells", ""])
        lines.extend(f"- `{key}`" for key in invalid_keys)
        lines.append("")
    if duplicate_keys:
        lines.extend(["### Duplicate cells", ""])
        lines.extend(f"- `{key}`" for key in duplicate_keys)
        lines.append("")
    if missing_keys:
        lines.extend(["### Missing cells", ""])
        lines.extend(f"- `{key}`" for key in missing_keys)
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    environment_seeds = os.environ.get("SEEDS", "42").replace(",", " ").split()
    try:
        default_seeds = [int(value) for value in environment_seeds]
    except ValueError as error:
        raise ValueError(
            "SEEDS must be a comma- or space-separated integer list") from error
    if not default_seeds:
        default_seeds = list(DEFAULT_SEEDS)

    parser = argparse.ArgumentParser(
        description="Validate P0 manifests and emit JSON plus Markdown tables.")
    parser.add_argument(
        "--result-root", action="append", default=None,
        help=("Result root to scan; repeat for separately launched queues. "
              "Defaults to the seed-derived matrix root."))
    parser.add_argument("--seeds", nargs="+", type=int, default=default_seeds)
    parser.add_argument(
        "--panels", nargs="+",
        choices=("u0", "u2", "supervised"),
        default=list(DEFAULT_PANELS),
        help=(
            "Panels to validate. The historical default remains `u0 u2`; "
            "request `supervised` explicitly for CRH."
        ),
    )
    # F15: without this the expected matrix always spans every declared budget,
    # so a finished 30-bit panel can never be strict-complete and
    # --require-paper-eligible reports 0 of an expected 432 -- a number that
    # describes four budgets rather than anything that was run.
    parser.add_argument(
        "--bits", nargs="+", type=int, default=None,
        help=("Bit budgets to aggregate. Defaults to the paper's main panel "
              "(30). Pass e.g. `--bits 30 40` for the 15- and 20-base panels; "
              "the expected matrix, strict completion and the protocol "
              "metadata are all scoped to what is requested."))
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-markdown", default=None)
    parser.add_argument(
        "--verify-artifact-hashes", action=argparse.BooleanOptionalAction,
        default=True,
        help="Hash checkpoints, extraction NPZs, and projected artifacts (default true).")
    parser.add_argument(
        "--require-complete", action="store_true",
        help="Exit nonzero unless every requested cell is unique and validly complete.")
    parser.add_argument(
        "--require-paper-eligible", action="store_true",
        help=("Exit nonzero unless every requested cell is paper-table eligible "
              "and at least three unique train seeds were aggregated."))
    args = parser.parse_args()

    requested_bits = resolve_bit_slice(args.bits)

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain at least one unique integer")
    if not args.panels or len(set(args.panels)) != len(args.panels):
        parser.error("--panels must contain at least one unique panel")
    seeds = tuple(args.seeds)
    panels = tuple(args.panels)
    seed_slug = "seeds" + "-".join(map(str, seeds))
    panel_slug = "-".join(panels)
    default_matrix_name = (
        f"p0_matrix_{seed_slug}_legacy_cache"
        if panels == DEFAULT_PANELS
        else f"p0_matrix_{seed_slug}_{panel_slug}_legacy_cache"
    )
    result_roots = (
        [_absolute(path) for path in args.result_root]
        if args.result_root
        else [_absolute(f"result_baseline/{default_matrix_name}")]
    )
    expected = _expected_keys(seeds, panels, requested_bits)
    expected_set = set(expected)

    discovered: dict[Key, list[Path]] = {}
    malformed: list[dict[str, str]] = []
    extraneous: list[dict[str, str]] = []
    source_profile_excluded: list[dict[str, str]] = []
    discovered_paths = sorted({
        path.resolve()
        for result_root in result_roots if result_root.exists()
        for path in result_root.rglob("p0_run_manifest.json")
    })
    for path in discovered_paths:
        try:
            raw = _load_json(path)
            manifest = _mapping(raw)
        except (OSError, json.JSONDecodeError) as error:
            malformed.append({"path": str(path.resolve()), "error": str(error)})
            continue
        if manifest is None:
            malformed.append({
                "path": str(path.resolve()),
                "error": "manifest root is not an object",
            })
            continue
        key = _manifest_key(manifest)
        if key is None or key not in expected_set:
            extraneous.append({
                "path": str(path.resolve()),
                "identity": repr({
                    name: manifest.get(name)
                    for name in ("variant", "dataset", "seed", "bit_length")
                }),
            })
            continue
        exclusion = _source_profile_exclusion(manifest, key)
        if exclusion is not None:
            source_profile_excluded.append({
                "manifest": str(path.resolve()),
                **exclusion,
            })
            continue
        discovered.setdefault(key, []).append(path.resolve())

    records: dict[Key, dict[str, object]] = {}
    missing: list[str] = []
    invalid: list[str] = []
    duplicates: list[str] = []
    duplicate_manifests: dict[str, list[str]] = {}
    for key in expected:
        paths = discovered.get(key, [])
        if not paths:
            missing.append(key.text)
            continue
        if len(paths) > 1:
            duplicates.append(key.text)
            duplicate_manifests[key.text] = [str(path) for path in paths]
            records[key] = {
                "key": key.text,
                "panel": key.panel,
                "variant": key.variant,
                "dataset": key.dataset,
                "bit": key.bit,
                "seed": key.seed,
                "status": "duplicate",
                "manifests": [str(path) for path in paths],
                "validation_errors": [
                    "multiple manifests found; automatic score/timestamp "
                    "selection is forbidden"
                ],
            }
            continue
        record = _validate_manifest(
            paths[0], key, verify_hashes=args.verify_artifact_hashes)
        records[key] = record
        if record["status"] == "invalid":
            invalid.append(key.text)

    implementation_audit = _audit_implementation_fingerprints(records)
    _apply_implementation_admission(records, implementation_audit)

    aggregates: list[dict[str, object]] = []
    if "u0" in panels:
        aggregates.extend(
            _aggregate_cell(
                records, panel="u0", variant=variant, dataset=dataset,
                bit=bit, seeds=seeds)
            for variant in U0_VARIANTS
            for dataset in DATASETS
            for bit in requested_bits
        )
    if "u2" in panels:
        aggregates.extend(
            _aggregate_cell(
                records, panel="u2", variant="umrch", dataset=dataset,
                bit=bit, seeds=seeds)
            for dataset in U2_DATASETS
            for bit in requested_bits
        )
    if "supervised" in panels:
        aggregates.extend(
            _aggregate_cell(
                records, panel="supervised", variant=variant,
                dataset=dataset, bit=bit, seeds=seeds)
            for variant in SUPERVISED_VARIANTS
            for dataset in SUPERVISED_DATASETS
            for bit in requested_bits
        )
    complete = sum(
        1 for record in records.values()
        if str(record.get("status", "")).startswith("complete_"))
    main_eligible = sum(
        1 for record in records.values()
        if record.get("status") == "complete_main_eligible")
    diagnostic = sum(
        1 for record in records.values()
        if record.get("status") == "complete_diagnostic_only")
    implementation_blocked = sum(
        1 for record in records.values()
        if record.get("implementation_comparison_eligible") is False)
    paper_table_eligible = sum(
        1 for record in records.values()
        if record.get("paper_table_eligible") is True)
    integrity_incomplete = bool(
        missing or invalid or duplicates or malformed or implementation_blocked)
    paper_admission_reasons: list[str] = []
    if integrity_incomplete:
        paper_admission_reasons.append("matrix_incomplete_or_invalid")
    if paper_table_eligible != len(expected):
        paper_admission_reasons.append("not_all_records_paper_table_eligible")
    if len(seeds) < 3:
        paper_admission_reasons.append("fewer_than_three_train_seeds")
    strict_paper_eligible = not paper_admission_reasons
    payload: dict[str, object] = {
        "schema_version": 4,
        "generated_at_utc": _utc_now(),
        "result_roots": [str(path) for path in result_roots],
        "protocol": {
            "train_seeds": list(seeds),
            "comparison_panels": list(panels),
            "validation_seed": 42,
            "validation_ratio": 0.1,
            # F15: scoped to the requested slice, and 30/40 were missing
            # entirely -- a 30-bit run had no declared selection metric.
            "selection_metric_by_bit": selection_metric_by_bit(requested_bits),
            "requested_bit_slice": list(requested_bits),
            "mandatory_bio_projection": True,
            "three_seed_mean_std_status": (
                "pending" if len(seeds) < 3 else "candidate_aggregate"
            ),
        },
        "eligibility_warning": (
            "Metrics with complete_diagnostic_only status came from the explicit "
            "legacy-cache opt-in and must not be copied into the eligible paper "
            "main table until strict cache provenance is regenerated."
        ),
        "paper_table_admission": {
            "eligible": strict_paper_eligible,
            "reason_codes": paper_admission_reasons,
            "required_unique_train_seeds": 3,
            "observed_unique_train_seeds": len(seeds),
            "eligible_records": paper_table_eligible,
            "expected_records": len(expected),
        },
        "exact_duheg": DUHEG_BLOCK,
        "implementation_audit": implementation_audit,
        "summary": {
            "expected": len(expected),
            "complete": complete,
            "complete_main_eligible": main_eligible,
            "complete_diagnostic_only": diagnostic,
            "implementation_comparison_blocked": implementation_blocked,
            "paper_table_eligible": paper_table_eligible,
            "missing": len(missing),
            "invalid": len(invalid),
            "duplicate": len(duplicates),
            "malformed": len(malformed),
            "extraneous": len(extraneous),
            "source_profile_excluded": len(source_profile_excluded),
        },
        "records": [records[key] for key in expected if key in records],
        "aggregates": aggregates,
        "missing_cells": missing,
        "invalid_cells": invalid,
        "duplicate_cells": duplicates,
        "duplicate_manifests": duplicate_manifests,
        "malformed_manifests": malformed,
        "extraneous_manifests": extraneous,
        "source_profile_excluded_manifests": source_profile_excluded,
    }

    default_output_stem = (
        f"baseline_p0_matrix_{seed_slug}_legacy_cache"
        if panels == DEFAULT_PANELS
        else f"baseline_p0_matrix_{seed_slug}_{panel_slug}_legacy_cache"
    )
    out_json = _absolute(
        args.out_json or f"docs/{default_output_stem}.json")
    out_markdown = _absolute(
        args.out_markdown or f"docs/{default_output_stem}.md")
    _atomic_json(out_json, payload)
    _atomic_text(out_markdown, _markdown(payload, records, seeds))
    print(json.dumps({
        "json": str(out_json),
        "markdown": str(out_markdown),
        "summary": payload["summary"],
        "exact_duheg": DUHEG_BLOCK,
    }, indent=2, ensure_ascii=False))

    if args.require_complete and integrity_incomplete:
        return 2
    if args.require_paper_eligible and not strict_paper_eligible:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
