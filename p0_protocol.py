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


#: The campaign cell-id stage of an anchor-model scratch refit (generation v9, stage R).
ANCHOR_REFIT_CELL_STAGE = "refit"


def campaign_cell_stage(cell_id: str | None) -> str | None:
    """The ``stage=`` field of a Phase-3 campaign cell id, or None without a cell id."""

    if not cell_id:
        return None
    stages = [part[len("stage="):] for part in str(cell_id).split("|")
              if part.startswith("stage=")]
    if len(stages) != 1 or not stages[0]:
        raise ValueError(f"campaign cell id {cell_id!r} names no single stage")
    return stages[0]


def anchor_refit_withholds_official_test(
    *,
    axis_center: str | None,
    p0_refit_active: bool,
    cell_id: str | None,
    sealed_recipe: bool,
) -> bool:
    """Stage R of the anchor model ends at its terminal checkpoint (audits 743-744).

    Returns True when the trainer must WITHHOLD the official-test extraction and
    evaluation: the run is an anchor-model P0 refit inside a sealed stage-R
    campaign cell. The official test is then reachable only through the
    separately approved stage-T entry. Returns False for every model without
    axis-centred anchors, whose behaviour is unchanged.

    Fail-closed: an anchor-model P0 refit that is not a sealed stage-R campaign
    cell (no cell id, another stage, or no sealed recipe) raises, so a missing,
    malformed or altered stage marker cannot fall through to the automatic
    official-test path; and a stage-R cell that is not a P0 refit raises.
    """

    anchor_model = axis_center not in (None, "none")
    if not anchor_model:
        # unchanged for every model without anchors; only an anchor-confirmation cell id (which
        # always names its stage) is read, so a stage-R cell cannot run the pre-revision model
        if cell_id and "|axis_center=" in str(cell_id) \
                and campaign_cell_stage(cell_id) == ANCHOR_REFIT_CELL_STAGE:
            raise ValueError("an anchor stage-R cell must train the anchor model")
        return False
    stage = campaign_cell_stage(cell_id) if cell_id else None
    if stage == ANCHOR_REFIT_CELL_STAGE:
        if not bool(p0_refit_active):
            raise ValueError(
                "an anchor stage-R cell must be a P0 refit "
                "(--stop_after_epoch with --final_epoch_eval)")
        if not bool(sealed_recipe):
            raise ValueError("an anchor stage-R cell must carry its sealed scientific recipe")
        return True
    if bool(p0_refit_active):
        raise ValueError(
            "an anchor-model P0 refit runs only as a sealed stage-R campaign cell "
            f"(cell stage {stage!r}); it never reaches the official test by itself")
    return False
