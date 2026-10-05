"""Checkpoint runtime state: the epoch a checkpoint must be inferred at (F01),
and the three training horizons a single `-e` used to conflate (D2).

WHY THIS EXISTS. `SigLIP2SemanticOTModel._current_epoch` is a plain Python int,
not a buffer, so it is not in the state dict. `extraction_siglip2.extract_code`
constructs a fresh model and loads raw weights, which leaves the epoch at 0 and
runs the Sinkhorn router at its INITIAL epsilon rather than the annealed value
the weights were trained with. Nothing raises; the codes are simply produced by
a router the training never used. Measured on the CIFAR final checkpoint, 24 of
the first 32 rows change `base_indices` between epoch 0 and epoch 4.

So resolution is fail-closed. When epsilon annealing is configured and the epoch
cannot be established, extraction ABORTS. Defaulting to 0 is exactly the bug.

D2 separates three things a single `-e` used to set at once:

    training_stop_epoch      when optimisation stops
    lr_schedule_horizon      the CosineAnnealingLR T_max (fixed at 60)
    sinkhorn_schedule_horizon the epsilon anneal horizon (N + 1)

Keeping the LR horizon fixed means every N candidate shares one learning-rate
prefix, so comparing N compares training length rather than three coupled knobs.
Unset values fall back to `args.epoch`, reproducing the historical behaviour
exactly so an old config does not silently change model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Optional

_SIDECAR_SUFFIX = ".runtime.json"
_SCHEMA_VERSION = 1

#: Extraction manifests only. Bumped to 2 when `sinkhorn_annealing_enabled`
#: became required: a manifest without it cannot say whether its null horizon
#: means "static epsilon" or "field forgotten" (§20.9). The checkpoint sidecar
#: keeps version 1 -- its layout did not change.
_MANIFEST_SCHEMA_VERSION = 2


class InferenceEpochUnresolved(RuntimeError):
    """Raised instead of silently inferring at epoch 0."""


# ------------------------------------------------------------------ horizons

@dataclass(frozen=True)
class Horizons:
    training_stop_epoch: int          # zero-based epoch training stops AT
    lr_schedule_horizon: int          # CosineAnnealingLR T_max
    sinkhorn_schedule_horizon: int    # epsilon anneal spans this many epochs


def resolve_horizons(args: Any) -> Horizons:
    budget = int(getattr(args, "epoch", 60) or 60)
    stop = getattr(args, "stop_after_epoch", None)
    stop = budget - 1 if stop is None else int(stop)
    lr_h = getattr(args, "lr_schedule_horizon", None)
    sk_h = getattr(args, "sinkhorn_schedule_horizon", None)
    return Horizons(
        training_stop_epoch=stop,
        lr_schedule_horizon=int(lr_h) if lr_h else budget,
        sinkhorn_schedule_horizon=int(sk_h) if sk_h else budget,
    )


def annealed_epsilon(epoch: int, horizon: int,
                     eps_init: Optional[float],
                     eps_final: Optional[float]) -> Optional[float]:
    """The cosine schedule used by the router, evaluated outside the model so
    manifests and tests do not have to instantiate one."""
    if eps_init is None or eps_final is None:
        return None
    t_max = max(int(horizon) - 1, 1)
    t = min(max(int(epoch), 0), t_max) / t_max
    cos_t = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(eps_final) + (float(eps_init) - float(eps_final)) * cos_t


# ------------------------------------------------------------------ metadata

def sha256_file(path: str) -> str:
    """Public digest helper. Callers record artefact SHAs with this so the
    manifest describes files that exist rather than a remembered value."""
    return _sha256(path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_epoch_zero_based: int
    training_epoch_budget: int
    stop_after_epoch: Optional[int]
    lr_schedule_horizon: int
    sinkhorn_schedule_horizon: int
    sinkhorn_epsilon_init: Optional[float]
    sinkhorn_epsilon_final: Optional[float]
    effective_sinkhorn_epsilon: Optional[float]
    lr_scheduler: Optional[str]
    extra: Mapping[str, Any]

    @staticmethod
    def sidecar_path(checkpoint_path: str) -> str:
        return str(checkpoint_path) + _SIDECAR_SUFFIX

    @classmethod
    def load(cls, checkpoint_path: str) -> Optional["CheckpointMetadata"]:
        p = cls.sidecar_path(checkpoint_path)
        if not os.path.exists(p):
            return None
        try:
            d = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            return None
        if d.get("schema_version") != _SCHEMA_VERSION:
            return None
        try:
            return cls(**d)
        except TypeError:
            return None


class RuntimeBindingRefused(RuntimeError):
    """A stage-T consumer's checkpoint or runtime witness is not the admitted pair."""


@dataclass(frozen=True)
class VerifiedRuntime:
    """Stage T (audits 759, 760): the terminal checkpoint's bytes and its runtime witness as one
    consumer read them ONCE and verified them against the admitted cell's pins. The consumer loads
    the weights from ``checkpoint_bytes`` and resolves the epoch from ``metadata`` (parsed from the
    verified witness bytes), so neither file is reopened by path before encoding and a later
    replacement cannot supply what is consumed."""
    checkpoint_path: str
    checkpoint_bytes: bytes
    checkpoint_sha256: str
    metadata: CheckpointMetadata
    terminal_epoch: int


def verified_runtime(checkpoint_path: str, checkpoint_bytes: bytes, witness_bytes: bytes, *,
                     checkpoint_sha256: str, witness_sha256: str,
                     terminal_epoch: int) -> VerifiedRuntime:
    """Bind one consumer's verified reads: both digests are the pins, the witness parses as this
    schema from those bytes, describes exactly that checkpoint and is at the admitted terminal epoch."""
    if hashlib.sha256(checkpoint_bytes).hexdigest() != checkpoint_sha256:
        raise RuntimeBindingRefused(f"{checkpoint_path} is not the admitted checkpoint bytes")
    if hashlib.sha256(witness_bytes).hexdigest() != witness_sha256:
        raise RuntimeBindingRefused(f"{CheckpointMetadata.sidecar_path(checkpoint_path)} is not the "
                                    f"admitted runtime witness bytes")
    try:
        d = json.loads(witness_bytes)
        md = (CheckpointMetadata(**d)
              if isinstance(d, dict) and d.get("schema_version") == _SCHEMA_VERSION else None)
    except (ValueError, TypeError):
        md = None
    if md is None:
        raise RuntimeBindingRefused("the admitted runtime witness is not a checkpoint sidecar of "
                                    f"schema {_SCHEMA_VERSION}")
    if md.checkpoint_sha256 != checkpoint_sha256:
        raise RuntimeBindingRefused("the admitted runtime witness describes another checkpoint")
    if md.checkpoint_epoch_zero_based != int(terminal_epoch):
        raise RuntimeBindingRefused(f"the admitted runtime witness is at epoch "
                                    f"{md.checkpoint_epoch_zero_based}, not the admitted terminal "
                                    f"epoch {int(terminal_epoch)}")
    return VerifiedRuntime(checkpoint_path=str(checkpoint_path), checkpoint_bytes=bytes(checkpoint_bytes),
                           checkpoint_sha256=checkpoint_sha256, metadata=md,
                           terminal_epoch=int(terminal_epoch))


def write_checkpoint_metadata(
    checkpoint_path: str, *,
    checkpoint_epoch_zero_based: int,
    training_epoch_budget: int,
    stop_after_epoch: Optional[int],
    lr_schedule_horizon: int,
    sinkhorn_schedule_horizon: int,
    sinkhorn_epsilon_init: Optional[float],
    sinkhorn_epsilon_final: Optional[float],
    lr_scheduler: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Write the sidecar next to the checkpoint, hashing the checkpoint so a
    later reload can tell whether the two still belong together."""
    md = CheckpointMetadata(
        schema_version=_SCHEMA_VERSION,
        checkpoint_path=os.path.abspath(checkpoint_path),
        checkpoint_sha256=_sha256(checkpoint_path),
        checkpoint_epoch_zero_based=int(checkpoint_epoch_zero_based),
        training_epoch_budget=int(training_epoch_budget),
        stop_after_epoch=None if stop_after_epoch is None else int(stop_after_epoch),
        lr_schedule_horizon=int(lr_schedule_horizon),
        sinkhorn_schedule_horizon=int(sinkhorn_schedule_horizon),
        sinkhorn_epsilon_init=sinkhorn_epsilon_init,
        sinkhorn_epsilon_final=sinkhorn_epsilon_final,
        effective_sinkhorn_epsilon=annealed_epsilon(
            checkpoint_epoch_zero_based, sinkhorn_schedule_horizon,
            sinkhorn_epsilon_init, sinkhorn_epsilon_final),
        lr_scheduler=lr_scheduler,
        extra=dict(extra or {}),
    )
    out = CheckpointMetadata.sidecar_path(checkpoint_path)
    tmp = out + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(asdict(md), fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    return out


# ---------------------------------------------------------------- resolution

#: The router's epsilon when no schedule is in force. It is a real operating
#: point, not an absence: `SemanticSinkhornRouter(epsilon=args.sinkhorn_
#: temperature)`. Recording null for it confuses the internal
#: `epsilon_override=None` sentinel with the epsilon the forward pass used.
DEFAULT_STATIC_SINKHORN_EPSILON = 0.05


def static_sinkhorn_epsilon(args: Any) -> float:
    return float(getattr(args, "sinkhorn_temperature",
                         DEFAULT_STATIC_SINKHORN_EPSILON))


@dataclass(frozen=True)
class ResolvedEpoch:
    epoch: int
    source: str                       # explicit_flag | checkpoint_metadata | no_annealing
    effective_sinkhorn_epsilon: Optional[float]
    sinkhorn_schedule_horizon: Optional[int]
    checkpoint_sha256: Optional[str]
    #: Whether an epsilon SCHEDULE was in force. This is a different question
    #: from `source`, which says only where the epoch was recovered from. A
    #: fresh no-anneal run writes a sidecar, so it resolves as
    #: `checkpoint_metadata` with no schedule at all -- and a schema that
    #: branched on `source == "no_annealing"` rejected it (§20.9).
    sinkhorn_annealing_enabled: bool = True


def resolve_inference_epoch(checkpoint_path: str, args: Any, *,
                            verified: Optional[VerifiedRuntime] = None) -> ResolvedEpoch:
    """Priority: explicit flag -> checkpoint metadata -> fail.

    An explicit flag still has to agree with metadata when metadata exists; a
    disagreement means one of the two is describing a different run and guessing
    which would defeat the purpose.

    With ``verified`` (stage T, audits 759-760) the metadata is the consumer's
    verified witness object and neither file is reopened; without it, unchanged.
    """
    eps_i = getattr(args, "sinkhorn_epsilon_init", None)
    eps_f = getattr(args, "sinkhorn_epsilon_final", None)
    if verified is None:
        md = CheckpointMetadata.load(checkpoint_path)

        if md is not None and os.path.exists(checkpoint_path):
            if _sha256(checkpoint_path) != md.checkpoint_sha256:
                raise InferenceEpochUnresolved(
                    f"the sidecar at {CheckpointMetadata.sidecar_path(checkpoint_path)} "
                    f"describes a different checkpoint (sha mismatch). It is stale -- "
                    f"most likely a previous run wrote into this directory. Re-save "
                    f"the checkpoint with metadata rather than trusting it.")
    else:
        if os.path.realpath(checkpoint_path) != os.path.realpath(verified.checkpoint_path):
            raise RuntimeBindingRefused(f"{checkpoint_path} is not the verified checkpoint "
                                        f"{verified.checkpoint_path}")
        md = verified.metadata
        if md.checkpoint_sha256 != verified.checkpoint_sha256:
            raise RuntimeBindingRefused("the verified runtime witness describes another checkpoint")

    # Whether a SCHEDULE is in force. The sidecar wins when it exists, because
    # it describes the run that produced these weights; `args` describes the
    # process about to load them.
    if md is not None:
        annealing = (md.sinkhorn_epsilon_init is not None
                     and md.sinkhorn_epsilon_final is not None)
        # The ENDPOINTS come from the sidecar too, not from the loader's args.
        # Reading the horizon from the sidecar and the endpoints from `args`
        # recorded an epsilon belonging to neither: a checkpoint annealed
        # 1.0 -> 0.1 loaded by a process configured 0.5 -> 0.2 was recorded at
        # 0.2, the value of a schedule that never ran (§23.4).
        eps_i, eps_f = md.sinkhorn_epsilon_init, md.sinkhorn_epsilon_final
    else:
        annealing = eps_i is not None and eps_f is not None
    static_epsilon = static_sinkhorn_epsilon(args)

    explicit = getattr(args, "inference_epoch", None)
    if explicit is not None:
        if md is not None and int(explicit) != md.checkpoint_epoch_zero_based:
            raise InferenceEpochUnresolved(
                f"--inference_epoch={int(explicit)} disagrees with the checkpoint "
                f"metadata epoch {md.checkpoint_epoch_zero_based}. Resolve which "
                f"run this checkpoint belongs to instead of overriding.")
        h = resolve_horizons(args)
        horizon = (md.sinkhorn_schedule_horizon if md
                   else h.sinkhorn_schedule_horizon)
        return ResolvedEpoch(
            epoch=int(explicit), source="explicit_flag",
            effective_sinkhorn_epsilon=(
                annealed_epsilon(int(explicit), horizon, eps_i, eps_f)
                if annealing else static_epsilon),
            sinkhorn_schedule_horizon=horizon if annealing else None,
            # A legacy checkpoint has no sidecar, but the file is present and
            # hashable: returning None here left the extraction manifest unable
            # to name what it loaded.
            checkpoint_sha256=(
                md.checkpoint_sha256 if md
                else (_sha256(checkpoint_path)
                      if os.path.isfile(checkpoint_path) else None)),
            sinkhorn_annealing_enabled=annealing)

    if md is not None:
        return ResolvedEpoch(
            epoch=md.checkpoint_epoch_zero_based, source="checkpoint_metadata",
            effective_sinkhorn_epsilon=(
                md.effective_sinkhorn_epsilon if annealing else static_epsilon),
            sinkhorn_schedule_horizon=(
                md.sinkhorn_schedule_horizon if annealing else None),
            checkpoint_sha256=md.checkpoint_sha256,
            sinkhorn_annealing_enabled=annealing)

    if eps_i is None or eps_f is None:
        # No annealing: the router uses its static epsilon and the epoch is
        # irrelevant, so an unknown epoch must not block extraction.
        # The epoch is irrelevant here, but the checkpoint identity never is:
        # returning None made static-epsilon extraction unable to name its own
        # weights, and the fail-closed manifest writer then refused it outright.
        return ResolvedEpoch(epoch=0, source="no_annealing",
                             # The static epsilon the router actually runs at.
                             # Recording null here made the operating point
                             # unrecoverable from the manifest.
                             effective_sinkhorn_epsilon=static_epsilon,
                             sinkhorn_schedule_horizon=None,
                             checkpoint_sha256=(
                                 _sha256(checkpoint_path)
                                 if os.path.isfile(checkpoint_path) else None),
                             sinkhorn_annealing_enabled=False)

    raise InferenceEpochUnresolved(
        f"Sinkhorn epsilon annealing is active ({eps_i} -> {eps_f}) but the "
        f"inference epoch for {checkpoint_path} cannot be established: no "
        f"--inference_epoch and no metadata sidecar. Proceeding would run the "
        f"router at the INITIAL epsilon, which is defect F01. Re-save the "
        f"checkpoint with write_checkpoint_metadata, or pass --inference_epoch "
        f"if the training epoch is known from the run log.")


def apply_inference_epoch(model: Any, checkpoint_path: str, args: Any, *,
                          verified: Optional[VerifiedRuntime] = None) -> ResolvedEpoch:
    """Resolve and push the epoch into the model. Call after load_state_dict and
    BEFORE the first forward. With ``verified`` the effective epoch and checkpoint
    must be the admitted ones before this returns (before any dataset access)."""
    if verified is None:
        resolved = resolve_inference_epoch(checkpoint_path, args)
    else:
        resolved = resolve_inference_epoch(checkpoint_path, args, verified=verified)
    setter = getattr(model, "set_current_epoch", None)
    if callable(setter):
        setter(resolved.epoch)
    if verified is not None and (resolved.epoch != verified.terminal_epoch
                                 or resolved.checkpoint_sha256 != verified.checkpoint_sha256):
        raise RuntimeBindingRefused(
            f"the effective runtime (epoch {resolved.epoch}, checkpoint "
            f"{str(resolved.checkpoint_sha256)[:12]}) is not the admitted terminal epoch "
            f"{verified.terminal_epoch} of {verified.checkpoint_sha256[:12]}")
    return resolved


def write_extraction_manifest(
    out_path: str, *, checkpoint_path: str, resolved: ResolvedEpoch,
    num_slots: int, bases_per_slot: int, split: str, n_rows: int,
    lr_schedule_horizon: Optional[int] = None,
    training_epoch_budget: Optional[int] = None,
    training_stop_epoch: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Record the runtime state an extraction actually ran under, so a table
    cannot be traced back to an unknown operating point."""
    d = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "split": split,
        "n_rows": int(n_rows),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": resolved.checkpoint_sha256,
        "inference_epoch": resolved.epoch,
        "inference_epoch_source": resolved.source,
        "effective_sinkhorn_epsilon": resolved.effective_sinkhorn_epsilon,
        "sinkhorn_schedule_horizon": resolved.sinkhorn_schedule_horizon,
        # Stated, not inferred from which other field happens to be null.
        "sinkhorn_annealing_enabled": bool(resolved.sinkhorn_annealing_enabled),
        # D2 asked for the whole schedule to be checkable from the artefact.
        # With only the Sinkhorn horizon recorded, the LR horizon and the stop
        # epoch could be recovered only by re-opening the config, so the
        # `-e`-coupling this decision separated was not verifiable downstream.
        "lr_schedule_horizon": lr_schedule_horizon,
        "training_epoch_budget": training_epoch_budget,
        "training_stop_epoch": training_stop_epoch,
        "num_slots": int(num_slots),
        "bases_per_slot": int(bases_per_slot),
        "total_bases": int(num_slots) * int(bases_per_slot),
        "total_bits": 2 * int(num_slots) * int(bases_per_slot),
    }
    d.update(dict(extra or {}))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    return out_path


def swap_best_into_final(best_path: str, final_path: str, *,
                         best_epoch_zero_based: int) -> str:
    """Promote the best checkpoint and re-stamp the sidecar together.

    Training wrote the sidecar for the FINAL weights, then overwrote those
    weights with the best ones without touching the sidecar. The recorded SHA
    then matched no file on disk, and the recorded epoch belonged to a different
    checkpoint -- which is exactly the input F01's epoch resolver trusts.
    """
    import shutil

    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"no best checkpoint at {best_path}")
    side = final_path + _SIDECAR_SUFFIX
    previous: dict = {}
    if os.path.exists(side):
        with open(side) as fh:
            previous = json.load(fh)
    shutil.copyfile(best_path, final_path)
    write_checkpoint_metadata(
        final_path,
        checkpoint_epoch_zero_based=int(best_epoch_zero_based),
        training_epoch_budget=previous.get("training_epoch_budget"),
        stop_after_epoch=previous.get("stop_after_epoch"),
        lr_schedule_horizon=previous.get("lr_schedule_horizon"),
        sinkhorn_schedule_horizon=previous.get("sinkhorn_schedule_horizon"),
        sinkhorn_epsilon_init=previous.get("sinkhorn_epsilon_init"),
        sinkhorn_epsilon_final=previous.get("sinkhorn_epsilon_final"),
        lr_scheduler=previous.get("lr_scheduler"),
        extra={"promoted_from": os.path.abspath(best_path),
               "promoted_reason": "best_val_checkpoint_swapped_into_final"},
    )
    return final_path
