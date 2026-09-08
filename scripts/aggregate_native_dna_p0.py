#!/usr/bin/env python3
"""Fail-closed aggregation for native-DNA common-P0 experiments.

The aggregator never chooses between duplicate runs and never promotes a
diagnostic-only run into the strict paper table.  A record contributes to the
diagnostic aggregate only when its manifest, checkpoint, evaluation SHA-256,
metrics, and post-projection compliance all validate.  It contributes to the
strict aggregate only when the same manifest additionally declares
``main_protocol_eligible=true`` with no blockers.

Expected experiment matrix:

* methods: DNA24/PRIMO/Koike-2024/Koike-2026;
* datasets: Flickr25k, MSCOCO, NUSWIDE, CIFAR10;
* train seeds: 42, 43, 44;
* matched capacity: 18 DNA bases.

The input contract is a recursively discovered ``native_p0_run_manifest.json``::

    {
      "method": "bee2018",
      "dataset": "Flickr25k",
      "seed": 42,
      "best_epoch_zero_based": 9,
      "refit_epochs": 10,
      "run_manifest_phase": "completed",
      "main_protocol_eligible": false,
      "main_eligibility_blockers": ["cache_provenance_incomplete"],
      "evaluation_native_dna": {"path": "...", "sha256": "..."},
      "final_checkpoint": {"path": "...", "sha256": "..."}
    }

Paths may be absolute or relative to the manifest.  The evaluation file uses
the schema emitted by ``scripts/train_native_dna_baseline.py``.
"""

from __future__ import annotations

import argparse
import copy
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
from typing import Iterable, Mapping, Sequence

# Run directly as `python scripts/aggregate_native_dna_p0.py`, which puts
# scripts/ on the path rather than the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# F18: the length belongs to the MANIFEST being validated, not to this
# process. These module constants remain the CLI default; every validator below
# takes a protocol argument so an 18-base cell can be judged in a process
# configured for 15 instead of failing with "expected 15, found 18".
from dna_utils.native_protocol import (  # noqa: E402
    NativeProtocol, coerce_protocol, resolve_native_protocol)

DEFAULT_PROTOCOL = resolve_native_protocol()
_PROTOCOL_LABEL = DEFAULT_PROTOCOL.protocol_label



REPO = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "native_p0_run_manifest.json"
DATASETS = ("Flickr25k", "MSCOCO", "NUSWIDE", "CIFAR10")
DEFAULT_SEEDS = (42, 43, 44)
# 18 was the 6-slot / 36-bit budget; the paper is 15 bases since 2026-08-11.
# Must agree with run_native_dna_p0{,_matrix}.py or every cell is rejected.
LENGTH = DEFAULT_PROTOCOL.length_bases
MAP_AT_R = {
    "Flickr25k": 5000,
    "MSCOCO": 5000,
    "NUSWIDE": 5000,
    "CIFAR10": 1000,
}
PANELS = {
    "feature_distance_pair": ("bee2018", "bee2021"),
    "supervised_direct_prior": ("koike2024", "koike2026"),
}
METHODS = tuple(method for methods in PANELS.values() for method in methods)
PANEL_BY_METHOD = {
    method: panel for panel, methods in PANELS.items() for method in methods
}
def display_names_for(protocol) -> dict:
    """Method display names for a length.

    `DISPLAY` is built at import from the process default, so aggregating an
    18-base root inside a 15-base process labelled every row `PRIMO-15` while
    the payload said `length_bases=18`. The length is a property of the
    artefact, so the label has to follow the protocol that was requested.
    """
    length = coerce_protocol(protocol).length_bases
    names = dict(_DISPLAY_BASE)
    names["bee2021"] = (
        f"PRIMO-{length} frozen-predictor length-transfer")
    return names


_DISPLAY_BASE = {
    "bee2018": "DNA24 (Stewart et al., 2018)",
    "bee2021": "PRIMO frozen-predictor length-transfer",
    "koike2024": "Koike et al. (DATE/DAC 2024)",
    "koike2026": "Koike et al. (TCBB 2026)",
}

#: The process-default view, kept so existing readers keep working. Anything
#: that knows its protocol should call `display_names_for()` instead.
DISPLAY = {**_DISPLAY_BASE,
           "bee2021": (f"PRIMO-{DEFAULT_PROTOCOL.length_bases} "
                       f"frozen-predictor length-transfer")}

SUPERVISION = {
    "bee2018": "feature-distance pairs",
    "bee2021": "feature-distance pairs",
    "koike2024": "ground-truth labels",
    "koike2026": "ground-truth labels",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TEST_ACCESS_CONTRACT = {
    "stage1_test_evaluated": False,
    "refit_from_scratch": True,
    "refit_weights_argument": None,
    "terminal_trainer_invocations": 1,
    "terminal_official_test_extraction_passes": 1,
    "post_run_validation_reopens_official_test": False,
    "raw_and_post_dp_share_terminal_extraction": True,
    "full_train_extraction_saved_after_terminal_test": True,
}
COMMON_BIO_PROJECTION = {
    "algorithm": "minimum_hamming_dynamic_programming",
    "applied_to": ["query", "database"],
    "gc_min": 0.4,
    "gc_max": 0.6,
    "max_run": 3,
}


def common_bio_projection_for(protocol) -> dict:
    """The sealed common-DP contract at a given length.

    The 15/18-base panels record the fractional form above; the 24-base panel
    also records the integer window and the sequence length. Comparing every
    manifest against the module-level dict meant a canonical
    `aggregate(sealed24, protocol=24)` marked all 48 cells invalid on
    `common_bio_projection` -- the length-aware schema only existed inside the
    24 wrapper's monkey patch.
    """
    from dna_utils.gc_policy import resolve_gc_policy

    protocol = coerce_protocol(protocol)
    contract = dict(COMMON_BIO_PROJECTION)
    if protocol.length_bases == 24:
        policy = resolve_gc_policy(protocol.length_bases)
        contract.update({
            "gc_count_min_inclusive": policy.gc_min_count,
            "gc_count_max_inclusive": policy.gc_max_count,
            "sequence_length_bases": protocol.length_bases,
        })
    return contract

# A protocol digest proves only that a manifest is internally self-consistent:
# a caller could otherwise change a source profile and recompute that outer
# digest.  These versioned locks additionally bind the stable, method-specific
# training contract admitted by this aggregator.  They intentionally hash the
# *declared run-era* implementation SHA mapping rather than re-hashing the live
# working tree.  The latter would retroactively invalidate the authoritative
# 18-base runs whenever an unrelated method is added to a shared source file.
#
# The implicit variant name is used only while normalizing the historical
# 18-base schema, whose manifests predate an explicit ``pipeline_variant``.
IMPLICIT_PIPELINE_VARIANT = "legacy_implicit_18_v1"
# Keyed by matched length, because the lock covers `matched_length_bases` and
# the implementation SHAs, both of which change with the budget. Registering the
# 15-base values here is a REVIEWED transition, not a silencing of the guard:
#
#   * A real 15-base cell was run (bee2018 x CIFAR10 x seed 42) and its manifest
#     read back. Its protocol_identity differs from the sealed 18-base manifest
#     in exactly two of 23 fields: `matched_length_bases` (18 -> 15) and
#     `implementation_artifacts`, where four files moved -- run_native_dna_p0.py
#     and train_native_dna_baseline.py (this budget migration), dataloaders.py
#     (the 6 -> 5 slot caption truncation) and baseline/base_model.py.
#   * All eight artifacts the 15-base manifest records match the working tree
#     AND are committed and clean, so these runs are reproducible from git.
#     The 18-base manifest's base_model.py SHA matches NO commit in history,
#     i.e. that matrix was executed from an uncommitted tree -- the same
#     provenance defect already recorded for baseline_val_select_p0.py.
#   * The values below were recomputed analytically from _protocol_identity and
#     verified against the measured digest of the real run: bee2018 computes to
#     bc3c5eb8... and the executed cell produced bc3c5eb8..., byte-identical.
#
# Recompute with docs/newmodel_analysis/native_dna_lock_sha256_15base.json;
# execution contract is num_workers=4, extract_batch_size=512, query_chunk=64.
_METHOD_PROTOCOL_LOCK_BY_LENGTH = {
    18: {
        "bee2018": "533c7fe2fe7c2da593fc2b061ada18ea1e55feccf3f73a024ca6cbcda28935a4",
        "bee2021": "d5033a4f7e45d01395aa0aec0760845a2fd6f18f0984d940d2ca2de4ad8d2d66",
        "koike2024": "c8ebf0e6fe60ad2c159e41325dc57883de8126ccc7d316d1436a7a716d97effc",
        "koike2026": "830ce044c5b36d8be0863c1303c599eab0c6266d3e1684c4d9fa1ec1836e158b",
    },
    15: {
        "bee2018": "bc3c5eb824adcce1adbc62809e18cc4ecc4971390263cad29c658360f28ac8a2",
        "bee2021": "3c3476e3293b87ab90c8e1012c5064c910e5cb4afd27f9a9716915138e8c3d6e",
        "koike2024": "6eac8ca3171ee0b6578db0ff306495019414b979ac2e501af19519d0d3a57831",
        "koike2026": "2f2d00441c141d320ee59980be3679043a566657d3d4214a24046710ae1241f6",
    },
    # F18: the sealed 24-base locks, previously reachable only by patching this
    # module's globals from `aggregate_native_dna_p0_24`. Registering them by
    # length is what lets a 24-base manifest be judged without mutating state
    # the 15- and 18-base paths read.
    24: {
        "bee2018": "f368fe4f366e61d1bea8dce1117cf9a191abb2312bd19bd7359e6a8e6e30186d",
        "bee2021": "79d5fb8afe255f71c730799eae946501a51f214b70be7beaf6739ca8d4318e7b",
        "koike2024": "4196f4dfdb287bb2ad95d2b83755f66bdb1254a486cdf2a71aacc976809fac58",
        "koike2026": "30403cd4b4ed4e4895f3a9db94f3a1ed05f0123d2ea32c34467c9cf6b49b569a",
    },
}
# F18 follow-up: `scripts/run_native_dna_p0.py` is itself in
# IMPLEMENTATION_PATHS, so threading NativeProtocol through the validators moved
# its SHA (7b354a21... -> eb394591...) and with it EVERY method-protocol lock.
# The pre-F18 digests above stay reviewed because the historical cells carry
# them and were produced under that source; the post-F18 digests below are the
# same contract recomputed from the current source. Both are accepted.
#
# The transition is numerically a no-op at every declared length: the protocol
# object reproduces the module constants exactly
# (gc_min_count == ceil(GC_MIN*L), gc_max_count == floor(GC_MAX*L),
# max_homopolymer_run == MAX_RUN) and no training, extraction or projection code
# changed. Regenerate with scripts/recompute_native_method_locks.py after
# READING the source diff -- never by pasting numbers to make a check pass.
_POST_F18_METHOD_PROTOCOL_LOCK_BY_LENGTH = {
    # 15 bases, computed in the canonical driver AFTER the final guard and
    # entrypoint edits. Regenerate with
    # scripts/recompute_native_method_locks.py --length 15.
    15: {
        "bee2018": "1809366fd49075916b27fef765740ce3bf6e8c280554ba8185f6ddd72ee515ed",
        "bee2021": "bc88ecf9c6549ddcd7312ce7f2d446d4f0930435f0dcd8c679de9e9cfac44fc6",
        "koike2024": "f132126d1f67103584f0a6eadcbdf3f38281167fcdbd4aed10c666287b525c0f",
        "koike2026": "ad7d2d2d7daceebf6e883c83a5050b22e435d35aab5f608507b3845bf4b0972f",
    },
    # 18 bases, computed in the canonical driver AFTER the final guard and
    # entrypoint edits. Regenerate with
    # scripts/recompute_native_method_locks.py --length 18.
    18: {
        "bee2018": "dc9f9b9064c690b38c1ea3a6781480a2cf66f22c0c845111b2fde601f9c41390",
        "bee2021": "116880d8de8a237b16f06b5f65bcf727a67f321bb14eb5925be927291651bf88",
        "koike2024": "2343f3923be6e4e007a24ee97315af52c6f937e68d7cf7d4b33ee2ad7b680f18",
        "koike2026": "1289256f16dc0b4adc754f6c26e06e65b09231b273216a2253519bd99ccf1116",
    },
    # 20 bases, computed in the canonical driver AFTER the final guard and
    # entrypoint edits. Regenerate with
    # scripts/recompute_native_method_locks.py --length 20.
    20: {
        "bee2018": "43ad8e27cb76b6f0ed6289fca16345b969841de34ed56e84977da17af5acc451",
        "bee2021": "063a6fd0edde6ee124314079dfe5b24e72edeeeb9e84b70acff559bdbaa566aa",
        "koike2024": "d34ebe7012ae12fa870082f0ad6f12aaa7d4d20e40ee162f50ea686bcd265d7e",
        "koike2026": "00a294490c6c9bfc5c10d9de410c7cc0f0ef34f9a8b47e9ec913c87d0ffde934",
    },
    # 24 bases, computed in the run_native_dna_p0_24 wrapper AFTER the final guard and
    # entrypoint edits. Regenerate with
    # scripts/recompute_native_method_locks.py --length 24.
    24: {
        "bee2018": "ab3d7a416e35f0ca9b7329acd161cbe8380401e0a0b9e48e6225011e9089441a",
        "bee2021": "c2fa6fd4802837780a65693fb38b6dc7c0d45ff8fac575dc8c9b1d1b6b3280c6",
        "koike2024": "1729d38c682fa0c33d9627868b9c5d24ae747cb471d101b8e42ff6bcaf812150",
        "koike2026": "de1179ef80bb94965f4868c19bb7de8e008a09cf78dcff78732c9130007511f6",
    },
}


# A THIRD generation, from the cache-provenance gate added to
# `_SigLIP2FeatureCache.__init__` (2026-08-15). `dataloaders.py` is one of
# `IMPLEMENTATION_PATHS`, so its file digest enters every method identity even
# when the change cannot reach the method.
#
# Reviewed before registering, per the rule above. The native-DNA baselines
# operate on raw sequences: `train_native_dna_baseline.py` imports only
# `carve_val_indices` from `val_split`, and `_SigLIP2FeatureCache` is reached
# solely through `val_split.cache_rows_for_dataset`, which no native path calls.
# The new check therefore cannot execute on this path -- a strict no-op, and the
# digests moved only because the file hash is deliberately a conservative input.
# Regenerate with scripts/recompute_native_method_locks.py --length <L> after
# READING the source diff -- never by pasting numbers to make a check pass.
_POST_CACHE_GATE_METHOD_PROTOCOL_LOCK_BY_LENGTH = {
    15: {
        "bee2018": "bdb2f1a04f0358d11072e58366ca2f3e2e3e2c1cfbff2fee202e000eb67e3cee",
        "bee2021": "52696b7abbb02271d9472653a41b2e487d3178a11231e9977c89b9e2acc522eb",
        "koike2024": "e66495a18349d817de0aa9cc73958004afedc1b0265c622239ad2855d340ea9e",
        "koike2026": "2efaf496f405c51aa5532df9d1462b3ddda5ffd94590b990096a14337a557152",
    },
    18: {
        "bee2018": "ae76b57c561ccfa19f0538037fcca282f1da76ec48c6c80929181324615323db",
        "bee2021": "2f3526172e3d826ccb29a032fba7f601f6a16a7055aca84c64d10984a942444c",
        "koike2024": "a2fdd8d57bfe49cfbcf7329198d1a490751d13d24d1012109c1ae3505430488d",
        "koike2026": "a62fd811f3e9295c44c31dff7cae92baad86d28e09285af2ac8652ae6b2f26c4",
    },
    20: {
        "bee2018": "59014f3e0585989ff94e4bb46a1bc3c00a9bbfebc0f3f89b7bd970dd56bf3db8",
        "bee2021": "58b9195bb41ff8609cc868386ba660205afbb5c9526e570e5099c41174cf6a96",
        "koike2024": "559e5a9941f26e97ae52c9dd0a66dcd6157a49c81edf1472abfd057624d4d2d4",
        "koike2026": "3b467fdb54ebfb63aa71a63d065ecc22c3b663dabe34a18fd6ea2512f7b503fa",
    },
    24: {
        "bee2018": "0bc7273388831698ea490f52f3ff88369ca3248b6e5c4950489633cd4c731373",
        "bee2021": "979144f4127d1dc441725a366caf6be6488fb15c3a9026710d29798f45887ffb",
        "koike2024": "6ec9de8472e2b5e681705ff64e0a34c01e03d24743fa95ba56df70037cb2fe48",
        "koike2026": "6f15488bb414932465b6940ff6364f8a01dc9ad114461160749de24c74bc22c8",
    },
}


# A FOURTH generation for the clean Phase-3 source closure.  ``dataloaders.py``
# added fail-closed shape/dtype/sidecar validation inside
# ``_SigLIP2FeatureCache`` and slices foil tensors to the configured semantic
# slot count.  Native-DNA training uses
# ``baseline.base_model.CachedFeatureDataset`` and imports from dataloaders only
# the unchanged CIFAR row-ID/sampling helpers; it never constructs
# ``_SigLIP2FeatureCache``.  The edit therefore cannot reach a native model,
# objective, optimizer, split, extraction, retrieval metric, or BIO projection.
#
# These values deliberately use the clean-release versions of the two D6-only
# files in the conservative lock closure (``baseline/base_model.py`` at
# b9d7e1a0... and ``baseline/cache_provenance.py`` at 4fe40c86...), not the
# concurrent dirty D6 implementations.  Registering the combined dirty-tree
# digests would claim review of a different release.  Keep every earlier
# reviewed generation for historical manifests and append only this exact
# Phase-3 transition after the reachability review above.
# Regenerate with scripts/recompute_native_method_locks.py --length <L> after
# READING the source diff -- never by weakening the validator.
_POST_PHASE3_CACHE_VALIDATION_METHOD_PROTOCOL_LOCK_BY_LENGTH = {
    15: {
        "bee2018": "27ca7fc6154b4cd1f532316971395d85f49dea6a502802848f937273bbeefd60",
        "bee2021": "0a966eb5788f5750bc65826b457d63361e29462654e9c127d4a0733045803a73",
        "koike2024": "7f2029d3a031aad4065e8e09949047edc3779355614eace645714ccdcd05701e",
        "koike2026": "297b75d0c3522184a6dddbcd08750a2e1b971028dafdda16d2cfb0049a0a8a1d",
    },
    18: {
        "bee2018": "01ab2e67aef1fd5098d27ce1936e2f4bd27ecd599442a5b7aecb3e16be19e106",
        "bee2021": "cde6e77c7e1b0da5275c67eabd9578fc88eca3e8201cf5fbf753b9d7bb40d39e",
        "koike2024": "4cd4820df3d1bf630e7c6b9869f47fa2dd77aa3572eb7c8c7cf26ac7c171a636",
        "koike2026": "0e67c9c30fee78e057a5d17c02e89b217899c06e0ca0da29980763fc1a343a18",
    },
    20: {
        "bee2018": "2e8d2f367002345b5763fa3dc8e158e2942e0e7f3c273349d873c94778b3cc64",
        "bee2021": "78f0fe944a73ec2d6846b9139c8598971c7d5f11ee2e1f3ece68b2545f6f2f28",
        "koike2024": "32962a4aeec6666b96d4e11ebba54fbb0a0f3e125cefa8ff13aedc630a5f61bb",
        "koike2026": "9111e995b04f06c6b246249edd881b235d724e7ae3e1db01aed8ec6cb2c1d620",
    },
    24: {
        "bee2018": "ce35f1a9ffac67b439543b3a7fc189487cec177d21b0da089a3ef7dc505fce4c",
        "bee2021": "ececfbe66621d5b29311dc9e92f082c3c8fd69fb0989a81a5cbb4740f7d0353f",
        "koike2024": "b66e44cf8285dfc1f41662a3130013812436459359b2de3ba49455ce8b6d9180",
        "koike2026": "16ef1db2ee956630f564593dc1a4e0fc46646abf4a96f7cfd29a12fa02f6080f",
    },
}


def reviewed_method_locks(protocol) -> dict:
    """{method: frozenset(reviewed digests)} for a length.

    A lock is an allow-list, so it holds every digest that has been reviewed for
    that length -- not just the newest. Registering only the newest would orphan
    the historical cells; registering without review would defeat the check.
    """
    length = coerce_protocol(protocol).length_bases
    pre = _METHOD_PROTOCOL_LOCK_BY_LENGTH.get(length, {})
    post = _POST_F18_METHOD_PROTOCOL_LOCK_BY_LENGTH.get(length, {})
    gated = _POST_CACHE_GATE_METHOD_PROTOCOL_LOCK_BY_LENGTH.get(length, {})
    phase3 = (
        _POST_PHASE3_CACHE_VALIDATION_METHOD_PROTOCOL_LOCK_BY_LENGTH.get(
            length, {})
    )
    if not pre and not post and not gated and not phase3:
        raise SystemExit(
            f"no reviewed method-protocol lock registered for {length} bases; "
            f"known: {sorted(_REGISTERED_LOCK_LENGTHS)}. "
            f"Run scripts/recompute_native_method_locks.py, read the source "
            f"diff, and register the digest rather than disabling the check.")
    return {
        method: frozenset(
            d for d in (
                pre.get(method),
                post.get(method),
                gated.get(method),
                phase3.get(method),
            )
            if d is not None)
        for method in sorted(
            set(pre) | set(post) | set(gated) | set(phase3)
        )
    }


#: Every length that has ANY reviewed lock. The import guard and the
#: compatibility accessor below both key off this union rather than the pre-F18
#: table: registering 20 bases only in the post-F18 table left
#: `GDNA_NATIVE_DNA_BASES=20 python -c "import ..."` exiting 1, so the env20
#: aggregator and matrix could not even print `--help`.
_REGISTERED_LOCK_LENGTHS = sorted(
    set(_METHOD_PROTOCOL_LOCK_BY_LENGTH)
    | set(_POST_F18_METHOD_PROTOCOL_LOCK_BY_LENGTH)
    | set(_POST_CACHE_GATE_METHOD_PROTOCOL_LOCK_BY_LENGTH)
    | set(_POST_PHASE3_CACHE_VALIDATION_METHOD_PROTOCOL_LOCK_BY_LENGTH))

if LENGTH not in _REGISTERED_LOCK_LENGTHS:
    raise SystemExit(
        f"no reviewed method-protocol lock registered for LENGTH={LENGTH}; "
        f"known: {_REGISTERED_LOCK_LENGTHS}. Run one cell, read its manifest, "
        f"diff the identity against a known budget, and register the digest "
        f"rather than disabling the check."
    )


def method_protocol_lock_for(protocol: "NativeProtocol | int | None") -> dict:
    """One reviewed digest per method, for compatibility with older callers.

    Prefer `reviewed_method_locks()`, which returns the full accepted set. This
    keeps returning a single value per method so existing readers do not change
    shape, and prefers the current-source digest.
    """
    protocol = coerce_protocol(protocol)
    length = protocol.length_bases
    if length not in _REGISTERED_LOCK_LENGTHS:
        raise SystemExit(
            f"no reviewed method-protocol lock registered for {length} bases; "
            f"known: {_REGISTERED_LOCK_LENGTHS}. Run "
            f"scripts/recompute_native_method_locks.py, read the source diff, "
            f"and register the digest rather than disabling the check.")
    current = (
        _POST_PHASE3_CACHE_VALIDATION_METHOD_PROTOCOL_LOCK_BY_LENGTH.get(
            length, {})
    )
    gated = _POST_CACHE_GATE_METHOD_PROTOCOL_LOCK_BY_LENGTH.get(length, {})
    post = _POST_F18_METHOD_PROTOCOL_LOCK_BY_LENGTH.get(length, {})
    pre = _METHOD_PROTOCOL_LOCK_BY_LENGTH.get(length, {})
    methods = set(pre) | set(post) | set(gated) | set(current)
    return {
        method: current.get(
            method, gated.get(method, post.get(method, pre.get(method))))
        for method in sorted(methods)
    }


METHOD_PROTOCOL_LOCK_SHA256 = method_protocol_lock_for(LENGTH)


@dataclass(frozen=True, order=True)
class Key:
    method: str
    dataset: str
    seed: int

    @property
    def panel(self) -> str:
        return PANEL_BY_METHOD[self.method]

    @property
    def text(self) -> str:
        return f"{self.panel}/{self.method}/{self.dataset}/seed{self.seed}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _protocol_family_digest(identity: Mapping[str, object]) -> str:
    """Bind all protocol inputs except the two explicitly allowed run variants."""
    family = copy.deepcopy(dict(identity))
    family.pop("seed", None)
    execution = family.get("execution")
    if isinstance(execution, dict):
        execution.pop("device_request", None)
    return _canonical_digest(family)


def _artifact_mapping_is_bound(value: object) -> bool:
    records = _mapping(value)
    if not records:
        return False
    for record in records.values():
        item = _mapping(record)
        if item is None:
            return False
        if not isinstance(item.get("path"), str) or not item["path"]:
            return False
        digest = item.get("sha256")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            return False
    return True


def _method_protocol_lock_payload(
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Return the path-independent, stable method-protocol identity.

    Dataset/cache/split absolute paths, train seed, requested GPU, and runtime
    package versions are deliberately excluded.  They remain bound by the full
    protocol digest and the cross-seed protocol-family check.  This smaller lock
    is the immutable allow-list for source equations, training stages,
    implementation revisions, and the frozen PRIMO model.
    """

    implementation = _mapping(identity.get("implementation_artifacts"))
    implementation_sha256: dict[str, object] = {}
    if implementation is not None:
        for relative, raw_record in sorted(
            implementation.items(), key=lambda item: str(item[0])
        ):
            record = _mapping(raw_record)
            implementation_sha256[str(relative)] = (
                None if record is None else record.get("sha256")
            )

    execution = _mapping(identity.get("execution"))
    execution_contract = {
        field: None if execution is None else execution.get(field)
        for field in ("num_workers", "extract_batch_size", "query_chunk")
    }
    predictor = _mapping(identity.get("primo_predictor"))
    return {
        "schema_version": identity.get("schema_version"),
        "method": identity.get("method"),
        "matched_length_bases": identity.get("matched_length_bases"),
        "pipeline_variant": identity.get(
            "pipeline_variant", IMPLICIT_PIPELINE_VARIANT
        ),
        "val_split_ratio": identity.get("val_split_ratio"),
        "val_split_seed": identity.get("val_split_seed"),
        "selection_metric": identity.get("selection_metric"),
        "selection_distance": identity.get("selection_distance"),
        "source_profile": identity.get("source_profile"),
        "source_reproduction_audit": identity.get(
            "source_reproduction_audit"
        ),
        "candidate_eval_period": identity.get("candidate_eval_period"),
        "candidate_epochs_zero_based": identity.get(
            "candidate_epochs_zero_based"
        ),
        "stage_contract": identity.get("stage_contract"),
        "common_bio_projection": identity.get("common_bio_projection"),
        "implementation_sha256": implementation_sha256,
        "execution_contract": execution_contract,
        "primo_predictor_sha256": (
            None if predictor is None else predictor.get("sha256")
        ),
        "primo_frozen_predictor_length_transfer": identity.get(
            "primo_frozen_predictor_length_transfer"
        ),
    }


def _method_protocol_lock_digest(identity: Mapping[str, object]) -> str:
    return _canonical_digest(_method_protocol_lock_payload(identity))


def _validate_method_protocol_lock(
    identity: Mapping[str, object],
    key: "Key",
    errors: list[str],
    protocol: "NativeProtocol | int | None" = None,
) -> str:
    actual = _method_protocol_lock_digest(identity)
    reviewed = reviewed_method_locks(protocol).get(key.method, frozenset())
    if actual not in reviewed:
        errors.append(
            "protocol_identity.method_protocol_lock_sha256: stable source/"
            "stage/bio/implementation/execution contract mismatch "
            f"(reviewed {sorted(reviewed)}, found {actual})"
        )
    return actual


def _validate_protocol_identity(
    manifest: Mapping[str, object],
    key: "Key",
    errors: list[str],
    protocol: "NativeProtocol | int | None" = None,
) -> tuple[str | None, str | None]:
    protocol = coerce_protocol(protocol)
    identity = _mapping(manifest.get("protocol_identity"))
    declared_digest = manifest.get("protocol_digest_sha256")
    if identity is None:
        errors.append("protocol_identity: expected an object")
        return None, None
    if (
        not isinstance(declared_digest, str)
        or SHA256_RE.fullmatch(declared_digest) is None
    ):
        errors.append(
            "protocol_digest_sha256: expected 64 lowercase hexadecimal digits"
        )
        declared_digest = None
    actual_digest = _canonical_digest(identity)
    if declared_digest is not None and declared_digest != actual_digest:
        errors.append(
            "protocol_digest_sha256: does not bind protocol_identity "
            f"(declared {declared_digest}, actual {actual_digest})"
        )

    expected_scalars = {
        "schema_version": 1,
        "method": key.method,
        "dataset": key.dataset,
        "setting": "setting1",
        "seed": key.seed,
        "matched_length_bases": protocol.length_bases,
        "val_split_ratio": 0.1,
        "val_split_seed": 42,
        "selection_metric": "val_neural_raw_mAP_at_R",
        "selection_distance": "base_hamming",
        "candidate_eval_period": 5,
        "map_at_R": MAP_AT_R[key.dataset],
    }
    for field, expected in expected_scalars.items():
        if identity.get(field) != expected:
            errors.append(
                f"protocol_identity.{field}: expected {expected!r}, "
                f"found {identity.get(field)!r}"
            )
    expected_bio = common_bio_projection_for(protocol)
    if identity.get("common_bio_projection") != expected_bio:
        errors.append(
            "protocol_identity.common_bio_projection: does not match the "
            f"sealed common-DP contract for {protocol.length_bases} bases"
        )
    for field in ("source_profile", "stage_contract", "cache", "execution"):
        if _mapping(identity.get(field)) is None:
            errors.append(f"protocol_identity.{field}: expected an object")
    cache = _mapping(identity.get("cache"))
    if cache is not None and not _artifact_mapping_is_bound(cache.get("artifacts")):
        errors.append(
            "protocol_identity.cache.artifacts: expected path/SHA-bound inputs"
        )
    for field in ("split_artifacts", "implementation_artifacts"):
        if not _artifact_mapping_is_bound(identity.get(field)):
            errors.append(
                f"protocol_identity.{field}: expected non-empty path/SHA-bound inputs"
            )
    if not isinstance(identity.get("dataset_root"), str) or not identity["dataset_root"]:
        errors.append("protocol_identity.dataset_root: expected a non-empty path")
    candidates = identity.get("candidate_epochs_zero_based")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(isinstance(epoch, bool) or not isinstance(epoch, int) for epoch in candidates)
    ):
        errors.append(
            "protocol_identity.candidate_epochs_zero_based: expected a "
            "non-empty integer list"
        )
    predictor = identity.get("primo_predictor")
    if key.method == "bee2021":
        if not _artifact_mapping_is_bound({"predictor": predictor}):
            errors.append(
                "protocol_identity.primo_predictor: PRIMO requires a "
                "path/SHA-bound frozen predictor"
            )
    elif predictor is not None:
        errors.append(
            "protocol_identity.primo_predictor: must be null outside PRIMO"
        )
    _validate_method_protocol_lock(identity, key, errors, protocol)
    return declared_digest, _protocol_family_digest(identity)


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


def _number(errors: list[str], label: str, value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label}: expected a finite number in [0,1], found {value!r}")
        return None
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        errors.append(f"{label}: expected a finite number in [0,1], found {value!r}")
        return None
    return result


def _path_from_record(
    errors: list[str],
    label: str,
    value: object,
    *,
    manifest_path: Path,
    expected_name: str | None = None,
) -> tuple[Path | None, str | None]:
    record = _mapping(value)
    if record is None:
        errors.append(f"{label}: expected {{path,sha256}} object")
        return None, None
    raw_path = record.get("path")
    declared_sha = record.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        errors.append(f"{label}.path: missing/non-empty string required")
        path = None
    else:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        if not path.is_file():
            errors.append(f"{label}.path: file does not exist: {path}")
            path = None
        elif expected_name is not None and path.name != expected_name:
            errors.append(
                f"{label}.path: expected filename {expected_name!r}, found {path.name!r}"
            )
    if not isinstance(declared_sha, str) or SHA256_RE.fullmatch(declared_sha) is None:
        errors.append(f"{label}.sha256: missing/invalid lowercase SHA-256")
        declared_sha = None
    return path, declared_sha


def _verify_record(
    errors: list[str],
    label: str,
    value: object,
    *,
    manifest_path: Path,
    verify_hashes: bool,
    expected_name: str | None = None,
) -> tuple[Path | None, str | None]:
    path, declared_sha = _path_from_record(
        errors,
        label,
        value,
        manifest_path=manifest_path,
        expected_name=expected_name,
    )
    if path is not None and declared_sha is not None and verify_hashes:
        actual_sha = _sha256(path)
        if actual_sha != declared_sha:
            errors.append(
                f"{label}.sha256: mismatch, declared {declared_sha}, actual {actual_sha}"
            )
    return path, declared_sha


def _manifest_key(payload: Mapping[str, object]) -> Key | None:
    method = payload.get("method")
    dataset = payload.get("dataset")
    raw_seed = payload.get("seed")
    if method not in METHODS or dataset not in DATASETS:
        return None
    if isinstance(raw_seed, bool):
        return None
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError):
        return None
    return Key(str(method), str(dataset), seed)


def _validate_evaluation(
    evaluation: Mapping[str, object],
    key: Key,
    errors: list[str],
    protocol: "NativeProtocol | int | None" = None,
) -> dict[str, float | int | str | None]:
    protocol = coerce_protocol(protocol)
    if evaluation.get("method") != key.method:
        errors.append(
            f"evaluation.method: expected {key.method!r}, "
            f"found {evaluation.get('method')!r}"
        )
    if evaluation.get("dataset") != key.dataset:
        errors.append(
            f"evaluation.dataset: expected {key.dataset!r}, "
            f"found {evaluation.get('dataset')!r}"
        )
    if evaluation.get("length") != protocol.length_bases:
        errors.append(
            f"evaluation.length: expected {protocol.length_bases}, "
            f"found {evaluation.get('length')!r}"
        )
    if evaluation.get("protocol") != protocol.protocol_label:
        errors.append(
            f"evaluation.protocol: expected {protocol.protocol_label!r}, "
            f"found {evaluation.get('protocol')!r}"
        )
    if evaluation.get("supervision") != SUPERVISION[key.method]:
        errors.append(
            f"evaluation.supervision: expected {SUPERVISION[key.method]!r}, "
            f"found {evaluation.get('supervision')!r}"
        )
    if key.method == "bee2021" and evaluation.get("method_variant") != (
        "frozen_predictor_length_transfer"
    ):
        errors.append(
            "evaluation.method_variant: PRIMO-18 requires "
            "'frozen_predictor_length_transfer'"
        )

    neural = _mapping(evaluation.get("neural_raw"))
    projected = _mapping(evaluation.get("projected"))
    query = _mapping(evaluation.get("query"))
    database = _mapping(evaluation.get("database"))
    for label, value in (
        ("evaluation.neural_raw", neural),
        ("evaluation.projected", projected),
        ("evaluation.query", query),
        ("evaluation.database", database),
    ):
        if value is None:
            errors.append(f"{label}: missing object")

    raw_map = _number(
        errors,
        "evaluation.neural_raw.mAP_at_R",
        None if neural is None else neural.get("mAP_at_R"),
    )
    post_map = _number(
        errors,
        "evaluation.projected.mAP_at_R",
        None if projected is None else projected.get("mAP_at_R"),
    )
    db_unique = _number(
        errors,
        "evaluation.database.unique_ratio_projected",
        None if database is None else database.get("unique_ratio_projected"),
    )
    expected_r = MAP_AT_R[key.dataset]
    for label, metric in (
        ("evaluation.neural_raw", neural),
        ("evaluation.projected", projected),
    ):
        if metric is not None and metric.get("mAP_R_cutoff") != expected_r:
            errors.append(
                f"{label}.mAP_R_cutoff: expected {expected_r}, "
                f"found {metric.get('mAP_R_cutoff')!r}"
            )
    query_compliance = _number(
        errors,
        "evaluation.query.post_compliance",
        None if query is None else query.get("post_compliance"),
    )
    db_compliance = _number(
        errors,
        "evaluation.database.post_compliance",
        None if database is None else database.get("post_compliance"),
    )
    if query_compliance is not None and query_compliance < 1.0:
        errors.append(
            "evaluation.query.post_compliance: exact value 1.0 required"
        )
    if db_compliance is not None and db_compliance < 1.0:
        errors.append(
            "evaluation.database.post_compliance: exact value 1.0 required"
        )
    return {
        "post_dp_map_at_R": post_map,
        "raw_map_at_R": raw_map,
        "db_unique_post": db_unique,
        "query_post_compliance": query_compliance,
        "db_post_compliance": db_compliance,
    }


def _validate_manifest(
    path: Path,
    key: Key,
    *,
    verify_hashes: bool = True,
    protocol: "NativeProtocol | int | None" = None,
) -> dict[str, object]:
    protocol = coerce_protocol(protocol)
    errors: list[str] = []
    try:
        loaded = _load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "key": key.text,
            "method": key.method,
            "dataset": key.dataset,
            "seed": key.seed,
            "manifest": str(path.resolve()),
            "status": "invalid",
            "validation_errors": [f"cannot load manifest: {error}"],
        }
    manifest = _mapping(loaded)
    if manifest is None:
        return {
            "key": key.text,
            "method": key.method,
            "dataset": key.dataset,
            "seed": key.seed,
            "manifest": str(path.resolve()),
            "status": "invalid",
            "validation_errors": ["manifest root must be an object"],
        }

    if manifest.get("schema_version") != 1:
        errors.append(
            f"schema_version: expected 1, found {manifest.get('schema_version')!r}"
        )
    if manifest.get("method") != key.method:
        errors.append(
            f"method: expected {key.method!r}, found {manifest.get('method')!r}"
        )
    if manifest.get("dataset") != key.dataset:
        errors.append(
            f"dataset: expected {key.dataset!r}, found {manifest.get('dataset')!r}"
        )
    if manifest.get("seed") != key.seed:
        errors.append(
            f"seed: expected {key.seed}, found {manifest.get('seed')!r}"
        )
    if manifest.get("setting") != "setting1":
        errors.append(
            f"setting: expected 'setting1', found {manifest.get('setting')!r}"
        )
    if manifest.get("run_manifest_phase") != "completed":
        errors.append(
            "run_manifest_phase: expected 'completed', "
            f"found {manifest.get('run_manifest_phase')!r}"
        )
    if manifest.get("length") != protocol.length_bases:
        errors.append(
            f"length: expected {protocol.length_bases}, "
            f"found {manifest.get('length')!r}"
        )
    if manifest.get("base_length") != protocol.length_bases:
        errors.append(
            f"base_length: expected {protocol.length_bases}, "
            f"found {manifest.get('base_length')!r}"
        )
    if manifest.get("test_used_for_selection") is not False:
        errors.append("test_used_for_selection: expected false")
    if manifest.get("test_access_contract") != TEST_ACCESS_CONTRACT:
        errors.append(
            "test_access_contract: does not match the sealed exact-once contract"
        )
    protocol_digest, protocol_family = _validate_protocol_identity(
        manifest, key, errors, protocol
    )
    protocol_identity = _mapping(manifest.get("protocol_identity"))
    method_protocol_lock = (
        None
        if protocol_identity is None
        else _method_protocol_lock_digest(protocol_identity)
    )

    best_epoch = manifest.get("best_epoch_zero_based")
    refit_epochs = manifest.get("refit_epochs")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch < 0:
        errors.append(
            "best_epoch_zero_based: expected non-negative integer, "
            f"found {best_epoch!r}"
        )
    elif refit_epochs != best_epoch + 1:
        errors.append(
            f"refit_epochs: expected E*+1={best_epoch + 1}, found {refit_epochs!r}"
        )

    main_eligible = manifest.get("main_protocol_eligible")
    blockers = manifest.get("main_eligibility_blockers")
    if not isinstance(main_eligible, bool):
        errors.append("main_protocol_eligible: expected boolean")
        main_eligible = False
    if not isinstance(blockers, list) or not all(
        isinstance(item, str) and item for item in blockers
    ):
        errors.append(
            "main_eligibility_blockers: expected list of non-empty strings"
        )
        blockers = []
    if main_eligible and blockers:
        errors.append(
            "eligibility contradiction: eligible manifest has blockers"
        )
    if not main_eligible and not blockers:
        errors.append(
            "eligibility contradiction: ineligible manifest requires blocker"
        )

    checkpoint_path, checkpoint_sha = _verify_record(
        errors,
        "final_checkpoint",
        manifest.get("final_checkpoint"),
        manifest_path=path,
        verify_hashes=verify_hashes,
    )
    evaluation_path, evaluation_sha = _verify_record(
        errors,
        "evaluation_native_dna",
        manifest.get("evaluation_native_dna"),
        manifest_path=path,
        verify_hashes=verify_hashes,
        expected_name="evaluation_native_dna.json",
    )

    metrics: dict[str, float | int | str | None] = {
        "post_dp_map_at_R": None,
        "raw_map_at_R": None,
        "db_unique_post": None,
        "query_post_compliance": None,
        "db_post_compliance": None,
    }
    if evaluation_path is not None:
        try:
            evaluation_loaded = _load_json(evaluation_path)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"evaluation_native_dna: cannot load JSON: {error}")
        else:
            evaluation = _mapping(evaluation_loaded)
            if evaluation is None:
                errors.append("evaluation_native_dna: root must be an object")
            else:
                metrics = _validate_evaluation(evaluation, key, errors, protocol)

    if errors:
        status = "invalid"
    elif main_eligible:
        status = "complete_main_eligible"
    else:
        status = "complete_diagnostic_only"
    return {
        "key": key.text,
        "panel": key.panel,
        "method": key.method,
        "display_name": display_names_for(protocol)[key.method],
        "dataset": key.dataset,
        "seed": key.seed,
        "length": protocol.length_bases,
        "manifest": str(path.resolve()),
        "manifest_sha256": _sha256(path) if verify_hashes else None,
        "status": status,
        "main_protocol_eligible": bool(main_eligible) and not errors,
        "main_eligibility_blockers": list(blockers),
        "best_epoch_zero_based": best_epoch,
        "refit_epochs": refit_epochs,
        "final_checkpoint": (
            None if checkpoint_path is None else str(checkpoint_path)
        ),
        "final_checkpoint_sha256": checkpoint_sha,
        "evaluation_native_dna": (
            None if evaluation_path is None else str(evaluation_path)
        ),
        "evaluation_native_dna_sha256": evaluation_sha,
        "protocol_digest_sha256": protocol_digest,
        "protocol_family_sha256": protocol_family,
        "method_protocol_lock_sha256": method_protocol_lock,
        **metrics,
        "validation_errors": errors,
    }


def _expected_keys(seeds: Sequence[int]) -> list[Key]:
    return [
        Key(method, dataset, seed)
        for method in METHODS
        for dataset in DATASETS
        for seed in seeds
    ]


def _discover(
    roots: Iterable[Path],
) -> tuple[dict[Key, list[Path]], list[dict[str, str]]]:
    candidates: dict[Key, list[Path]] = {}
    unkeyed: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        paths = (
            [resolved]
            if resolved.is_file() and resolved.name == MANIFEST_NAME
            else sorted(resolved.rglob(MANIFEST_NAME)) if resolved.is_dir()
            else []
        )
        for path in paths:
            path = path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                loaded = _load_json(path)
            except (OSError, json.JSONDecodeError) as error:
                unkeyed.append({"manifest": str(path), "error": str(error)})
                continue
            payload = _mapping(loaded)
            key = None if payload is None else _manifest_key(payload)
            if key is None:
                unkeyed.append({
                    "manifest": str(path),
                    "error": "cannot derive recognized method/dataset/seed key",
                })
                continue
            candidates.setdefault(key, []).append(path)
    return candidates, unkeyed


def _placeholder(key: Key, status: str, *,
                 protocol: "NativeProtocol | int | None" = None,
                 **extra: object) -> dict[str, object]:
    protocol = coerce_protocol(protocol)
    return {
        "key": key.text,
        "panel": key.panel,
        "method": key.method,
        "display_name": display_names_for(protocol)[key.method],
        "dataset": key.dataset,
        "seed": key.seed,
        "length": protocol.length_bases,
        "status": status,
        "main_protocol_eligible": False,
        "main_eligibility_blockers": [],
        "post_dp_map_at_R": None,
        "raw_map_at_R": None,
        "db_unique_post": None,
        "best_epoch_zero_based": None,
        "protocol_digest_sha256": None,
        "protocol_family_sha256": None,
        "method_protocol_lock_sha256": None,
        "validation_errors": [],
        **extra,
    }


def _invalidate_mixed_protocol_families(
    records: Mapping[Key, dict[str, object]],
    seeds: Sequence[int],
) -> None:
    """Reject a 3-seed cell if any invariant protocol input differs.

    The family digest deliberately removes only ``seed`` and
    ``execution.device_request``. Cache/source/code/projection/data split and all
    other execution choices must therefore be identical.
    """
    complete_statuses = {
        "complete_main_eligible",
        "complete_diagnostic_only",
    }
    for method in METHODS:
        for dataset in DATASETS:
            selected = [
                records[Key(method, dataset, seed)]
                for seed in seeds
            ]
            complete = [
                record
                for record in selected
                if record.get("status") in complete_statuses
            ]
            families = {
                str(record["protocol_family_sha256"])
                for record in complete
                if isinstance(record.get("protocol_family_sha256"), str)
            }
            if len(families) <= 1:
                continue
            detail = ", ".join(
                f"seed{record['seed']}={record.get('protocol_family_sha256')}"
                for record in complete
            )
            message = (
                "mixed protocol families across seeds; only seed and requested "
                f"device may differ ({detail})"
            )
            for record in complete:
                errors = record.setdefault("validation_errors", [])
                if isinstance(errors, list):
                    errors.append(message)
                record["status"] = "invalid"
                record["main_protocol_eligible"] = False


def _resolve_records(
    candidates: Mapping[Key, Sequence[Path]],
    *,
    seeds: Sequence[int],
    blocked_methods: Mapping[str, str],
    verify_hashes: bool,
    protocol: "NativeProtocol | int | None" = None,
) -> dict[Key, dict[str, object]]:
    protocol = coerce_protocol(protocol)
    records: dict[Key, dict[str, object]] = {}
    for key in _expected_keys(seeds):
        paths = list(candidates.get(key, ()))
        if len(paths) > 1:
            records[key] = _placeholder(
                key,
                "duplicate",
                protocol=protocol,
                duplicate_manifests=[str(path.resolve()) for path in paths],
                validation_errors=[
                    "multiple manifests declare the same method/dataset/seed; "
                    "no timestamp or score based resolution is permitted"
                ],
            )
        elif len(paths) == 1:
            records[key] = _validate_manifest(
                paths[0], key, verify_hashes=verify_hashes, protocol=protocol
            )
        elif key.method in blocked_methods:
            reason = blocked_methods[key.method]
            records[key] = _placeholder(
                key,
                "blocked",
                protocol=protocol,
                block_reason=reason,
                main_eligibility_blockers=[reason],
            )
        else:
            records[key] = _placeholder(key, "missing", protocol=protocol)
    _invalidate_mixed_protocol_families(records, seeds)
    return records


def _admitted(record: Mapping[str, object], mode: str) -> bool:
    if mode == "diagnostic":
        return record.get("status") in (
            "complete_main_eligible",
            "complete_diagnostic_only",
        )
    if mode == "strict_main":
        return record.get("status") == "complete_main_eligible"
    raise ValueError(mode)


def _aggregate_cell(
    records: Mapping[Key, Mapping[str, object]],
    *,
    method: str,
    dataset: str,
    seeds: Sequence[int],
    mode: str,
    protocol: "NativeProtocol | int | None" = None,
) -> dict[str, object]:
    selected = [records[Key(method, dataset, seed)] for seed in seeds]
    admitted = [record for record in selected if _admitted(record, mode)]
    complete = len(admitted) == len(seeds)
    all_blocked = bool(selected) and all(
        record.get("status") == "blocked" for record in selected
    )
    if complete:
        status = "complete"
    elif admitted:
        status = "partial"
    elif all_blocked:
        status = "blocked"
    else:
        status = "missing"

    def values(field: str) -> list[float]:
        return [
            float(record[field])
            for record in admitted
            if record.get(field) is not None
        ]

    def complete_mean(field: str) -> float | None:
        data = values(field)
        return statistics.mean(data) if complete and len(data) == len(seeds) else None

    def complete_std(field: str) -> float | None:
        data = values(field)
        return statistics.stdev(data) if complete and len(data) >= 2 else None

    blockers = sorted({
        blocker
        for record in selected
        for blocker in record.get("main_eligibility_blockers", [])
        if isinstance(blocker, str)
    })
    protocol = coerce_protocol(protocol)
    epochs = {
        str(record["seed"]): int(record["best_epoch_zero_based"])
        for record in admitted
        if isinstance(record.get("best_epoch_zero_based"), int)
    }
    return {
        "mode": mode,
        "panel": PANEL_BY_METHOD[method],
        "method": method,
        "display_name": display_names_for(protocol)[method],
        "dataset": dataset,
        "seeds_expected": list(seeds),
        "seeds_admitted": [int(record["seed"]) for record in admitted],
        "aggregate_status": status,
        "all_seeds_complete": complete,
        "diagnostic_only": bool(
            mode == "diagnostic"
            and complete
            and any(
                record.get("status") == "complete_diagnostic_only"
                for record in admitted
            )
        ),
        "mean_post_dp_map_at_R": complete_mean("post_dp_map_at_R"),
        "sample_std_post_dp_map_at_R": complete_std("post_dp_map_at_R"),
        "mean_raw_map_at_R": complete_mean("raw_map_at_R"),
        "sample_std_raw_map_at_R": complete_std("raw_map_at_R"),
        "mean_db_unique_post": complete_mean("db_unique_post"),
        "sample_std_db_unique_post": complete_std("db_unique_post"),
        "best_epoch_zero_based_by_seed": epochs,
        "main_eligibility_blockers": blockers,
        "record_status_by_seed": {
            str(record["seed"]): record["status"] for record in selected
        },
    }


def _all_aggregates(
    records: Mapping[Key, Mapping[str, object]],
    seeds: Sequence[int],
    mode: str,
    protocol: "NativeProtocol | int | None" = None,
) -> list[dict[str, object]]:
    return [
        _aggregate_cell(
            records,
            method=method,
            dataset=dataset,
            seeds=seeds,
            mode=mode,
            protocol=protocol,
        )
        for method in METHODS
        for dataset in DATASETS
    ]


def aggregate(
    roots: Sequence[Path],
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    blocked_methods: Mapping[str, str] | None = None,
    verify_hashes: bool = True,
    protocol: "NativeProtocol | int | None" = None,
) -> tuple[dict[str, object], dict[Key, dict[str, object]]]:
    protocol = coerce_protocol(protocol)
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be a non-empty sequence of unique integers")
    blocked = dict(blocked_methods or {})
    unknown_blocked = set(blocked) - set(METHODS)
    if unknown_blocked:
        raise ValueError(f"unknown blocked methods: {sorted(unknown_blocked)}")
    candidates, unkeyed = _discover(roots)
    records = _resolve_records(
        candidates,
        seeds=normalized_seeds,
        blocked_methods=blocked,
        verify_hashes=verify_hashes,
        protocol=protocol,
    )
    strict = _all_aggregates(records, normalized_seeds, "strict_main", protocol)
    diagnostic = _all_aggregates(records, normalized_seeds, "diagnostic", protocol)
    counts = {
        status: sum(record["status"] == status for record in records.values())
        for status in (
            "complete_main_eligible",
            "complete_diagnostic_only",
            "missing",
            "blocked",
            "invalid",
            "duplicate",
        )
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "manifest_name": MANIFEST_NAME,
        "roots": [str(path.expanduser().resolve()) for path in roots],
        "verify_hashes": bool(verify_hashes),
        "length_bases": protocol.length_bases,
        "protocol_label": protocol.protocol_label,
        "datasets": list(DATASETS),
        "seeds": list(normalized_seeds),
        "panels": {panel: list(methods) for panel, methods in PANELS.items()},
        "display_names": display_names_for(protocol),
        # Report the set the validator accepts, not the pre-F18 table:
        # advertising a narrower allow-list than the gate enforces makes the
        # provenance describe a check that is not the one being run.
        "method_protocol_lock_sha256": {
            method: sorted(digests)
            for method, digests in reviewed_method_locks(protocol).items()
        },
        "blocked_methods": blocked,
        "summary": {
            "expected_records": len(records),
            **counts,
            "unkeyed_manifests": len(unkeyed),
            "strict_complete_cells": sum(
                item["aggregate_status"] == "complete" for item in strict
            ),
            "strict_partial_cells": sum(
                item["aggregate_status"] == "partial" for item in strict
            ),
            "diagnostic_complete_cells": sum(
                item["aggregate_status"] == "complete" for item in diagnostic
            ),
            "diagnostic_partial_cells": sum(
                item["aggregate_status"] == "partial" for item in diagnostic
            ),
        },
        "strict_main_aggregate": strict,
        "diagnostic_aggregate": diagnostic,
        "records": [records[key] for key in sorted(records)],
        "unkeyed_manifests": unkeyed,
    }
    return payload, records


def _aggregate_lookup(
    payload: Mapping[str, object], field: str
) -> dict[tuple[str, str], Mapping[str, object]]:
    raw = payload.get(field)
    items = raw if isinstance(raw, list) else []
    return {
        (str(item["method"]), str(item["dataset"])): item
        for item in items
        if isinstance(item, Mapping)
    }


def _cell_text(item: Mapping[str, object], *, strict: bool) -> str:
    status = item.get("aggregate_status")
    if status == "blocked":
        return "BLOCKED"
    if status == "partial":
        return (
            f"PARTIAL {len(item.get('seeds_admitted', []))}/"
            f"{len(item.get('seeds_expected', []))}"
        )
    if status != "complete":
        statuses = set(
            value for value in (_mapping(item.get("record_status_by_seed")) or {}).values()
            if isinstance(value, str)
        )
        if "duplicate" in statuses:
            return "DUP"
        if "invalid" in statuses:
            return "ERR"
        return "-"
    post = float(item["mean_post_dp_map_at_R"])
    raw_post_std = item.get("sample_std_post_dp_map_at_R")
    raw = float(item["mean_raw_map_at_R"])
    unique = float(item["mean_db_unique_post"])
    epochs = _mapping(item.get("best_epoch_zero_based_by_seed")) or {}
    epoch_text = "/".join(f"{seed}:{epochs.get(seed, '-')}" for seed in sorted(epochs))
    suffix = "†" if not strict and item.get("diagnostic_only") is True else ""
    post_text = (
        f"{post:.4f}"
        if raw_post_std is None
        else f"{post:.4f} ± {float(raw_post_std):.4f}"
    )
    return (
        f"{post_text}{suffix} "
        f"(raw {raw:.4f}, u {unique:.4f}, E* {epoch_text})"
    )


def _markdown(payload: Mapping[str, object]) -> str:
    # Names come from the payload being rendered, not from the process default:
    # rendering an 18-base aggregate inside a 15-base process labelled every row
    # PRIMO-15 while the header said 18 bases.
    _display = _mapping(payload.get("display_names")) or DISPLAY
    seeds = payload.get("seeds", [])
    summary = _mapping(payload.get("summary")) or {}
    strict = _aggregate_lookup(payload, "strict_main_aggregate")
    diagnostic = _aggregate_lookup(payload, "diagnostic_aggregate")
    lines = [
        "# Native-DNA common-P0 aggregation",
        "",
        f"- Expected train seeds: `{', '.join(map(str, seeds))}`",
        f"- Capacity: `{payload.get('length_bases', LENGTH)}` DNA bases",
        "- `U0-FD` and `S` are repository-local information-condition tags, "
        "not names claimed verbatim by every source paper.",
        "- Supervision regimes: `U0-FD` = unsupervised with respect to the "
        "target benchmark: target ground-truth labels, taxonomy, and captions "
        "do not enter the encoder objective, and pseudo-pair targets come from "
        "frozen optimization-train feature distances; `S` = target "
        "ground-truth train labels enter the objective.",
        "- As for the other common-P0 baselines, held-out labels are used only "
        "to score validation retrieval and select E*. Thus `U0-FD` describes "
        "the encoder objective, not a label-blind end-to-end evaluation "
        "protocol.",
        "- PRIMO remains `U0-FD`: its frozen external hybridization-yield "
        "predictor is non-semantic and does not consume target labels.",
        "- Every numeric cell requires matching checkpoint/evaluation SHA-256 "
        "and query/database post-compliance `1.0`.",
        "- `†` is a complete diagnostic-only aggregate; it must not be copied "
        "to the strict paper main table.",
        f"- Records: expected {summary.get('expected_records', 0)}, strict "
        f"{summary.get('complete_main_eligible', 0)}, diagnostic-only "
        f"{summary.get('complete_diagnostic_only', 0)}, missing "
        f"{summary.get('missing', 0)}, blocked {summary.get('blocked', 0)}, "
        f"invalid {summary.get('invalid', 0)}, duplicate "
        f"{summary.get('duplicate', 0)}.",
        "",
    ]
    panel_titles = {
        "feature_distance_pair": (
            "Unsupervised (`U0-FD`; target-label-free encoder objective) "
            "feature-distance-pair direct predecessors"
        ),
        "supervised_direct_prior": "Supervised (`S`) direct-prior baselines",
    }
    for panel, methods in PANELS.items():
        lines.extend([
            f"## {panel_titles[panel]} — strict main",
            "",
            "| Method | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |",
            "|---|---:|---:|---:|---:|",
        ])
        for method in methods:
            values = [
                _cell_text(strict[(method, dataset)], strict=True)
                for dataset in DATASETS
            ]
            lines.append(f"| {_display[method]} | " + " | ".join(values) + " |")
        lines.extend([
            "",
            f"## {panel_titles[panel]} — diagnostic",
            "",
            "| Method | Flickr25k | MSCOCO | NUS-WIDE | CIFAR10 |",
            "|---|---:|---:|---:|---:|",
        ])
        for method in methods:
            values = [
                _cell_text(diagnostic[(method, dataset)], strict=False)
                for dataset in DATASETS
            ]
            lines.append(f"| {_display[method]} | " + " | ".join(values) + " |")
        lines.append("")

    records = payload.get("records")
    blockers = [
        record for record in records
        if isinstance(record, Mapping) and record.get("main_eligibility_blockers")
    ] if isinstance(records, list) else []
    if blockers:
        lines.extend([
            "## Strict-main blockers",
            "",
            "| Cell | Status | Blocker(s) |",
            "|---|---|---|",
        ])
        for record in blockers:
            reason = "; ".join(map(str, record["main_eligibility_blockers"]))
            lines.append(
                f"| `{record['key']}` | `{record['status']}` | {reason} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return seeds


def _parse_blocked(values: Sequence[str]) -> dict[str, str]:
    blocked: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"blocked method must use METHOD=REASON syntax, found {value!r}"
            )
        method, reason = value.split("=", 1)
        method = method.strip()
        reason = reason.strip()
        if method not in METHODS or not reason:
            raise ValueError(f"invalid blocked method declaration: {value!r}")
        if method in blocked:
            raise ValueError(f"duplicate blocked method declaration: {method}")
        blocked[method] = reason
    return blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help=(
            "Root to scan recursively for native_p0_run_manifest.json; repeat "
            "for multiple roots"
        ),
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--length", type=int, default=None,
        help=(
            "Matched DNA length in bases. Defaults to $GDNA_NATIVE_DNA_BASES, "
            "then 15. Naming it here is how an 18-base result root is "
            "aggregated without exporting an environment variable that also "
            "reconfigures every other native script in the shell."))
    parser.add_argument(
        "--blocked-method",
        action="append",
        default=[],
        metavar="METHOD=REASON",
        help="Represent missing cells for a genuinely blocked method explicitly",
    )
    parser.add_argument(
        "--out-json",
        default=str(REPO / "docs" / "native_dna_p0_aggregate.json"),
    )
    parser.add_argument(
        "--out-md",
        default=str(REPO / "docs" / "native_dna_p0_aggregate.md"),
    )
    args = parser.parse_args()
    roots = (
        [Path(value) for value in args.root]
        if args.root
        else [
            REPO / "result_baseline" / "native_dna_p0",
            REPO / "result_native_dna",
        ]
    )
    try:
        blocked = _parse_blocked(args.blocked_method)
        payload, _ = aggregate(
            roots,
            seeds=args.seeds,
            blocked_methods=blocked,
            verify_hashes=True,
            protocol=args.length,
        )
    except ValueError as error:
        parser.error(str(error))
    _atomic_json(Path(args.out_json).expanduser().resolve(), payload)
    _atomic_text(
        Path(args.out_md).expanduser().resolve(),
        _markdown(payload),
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
