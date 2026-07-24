"""Small, dependency-free guards for the leakage-free P0 protocol."""

from __future__ import annotations


def is_p0_stage1(*, val_split_ratio: float) -> bool:
    """Return whether the run is the validation-selection stage of P0."""

    return float(val_split_ratio) > 0.0


def is_p0_refit(
    *, stop_after_epoch: int | None, final_epoch_eval: bool
) -> bool:
    """Identify the full-train, fixed-E* second stage of the P0 protocol."""

    return stop_after_epoch is not None and bool(final_epoch_eval)


def should_load_official_test_split(
    *, val_protocol_active: bool, p0_refit_active: bool = False
) -> bool:
    """Return whether constructing a test dataset during training is allowed."""

    return not (bool(val_protocol_active) or bool(p0_refit_active))


def should_run_official_test_evaluation(
    *, evaluation_requested: bool, val_protocol_active: bool
) -> bool:
    """Return whether the current run may evaluate the official test split.

    A P0 stage-1 run uses a held-out subset of the designated training split
    to select E*.  Even when the generic ``--eval`` flag is present in a
    dataset launcher, that run must not extract or score the official test
    split.  Stage 2 has no validation split and therefore remains eligible for
    the single final official-test evaluation.
    """

    return bool(evaluation_requested) and not bool(val_protocol_active)


def should_run_training_time_evaluation(*, p0_refit_active: bool) -> bool:
    """Block all per-epoch evaluation after stage 1 has already fixed E*."""

    return not bool(p0_refit_active)


def should_run_test_consuming_visualization(
    *,
    visualization_requested: bool,
    val_protocol_active: bool,
    p0_refit_active: bool = False,
) -> bool:
    """Guard visualization paths that encode examples from the test split."""

    return (
        bool(visualization_requested)
        and not bool(val_protocol_active)
        and not bool(p0_refit_active)
    )


def resolve_eval_text_whiten_path(
    *,
    training_text_whiten_path: str | None,
    explicit_eval_text_whiten_path: str | None,
) -> str | None:
    """Resolve final-extraction whitening without silently changing semantics.

    Switching feature-cache directories does not authorize switching the
    learned/precomputed text transform. In particular, a local-slot-only
    train-fit matrix must remain active during P0 stage-2 extraction. Callers
    can request a different evaluation matrix only through the explicit
    override.
    """

    if explicit_eval_text_whiten_path:
        return str(explicit_eval_text_whiten_path)
    return training_text_whiten_path
