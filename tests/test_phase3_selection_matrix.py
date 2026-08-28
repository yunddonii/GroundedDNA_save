"""The Phase 3 selection matrix must be the protocol, not an approximation.

D1 fixes the grid ({4, 9, 19, 39} for every dataset), the metric (raw
base-Hamming mAP@R on held-out train), the tie-break (smallest N) and the seed.
D2 fixes the search-stage horizons: `-e 60`, LR horizon 60 for every candidate,
Sinkhorn horizon N+1. The existing wrappers satisfy none of that -- one selects
on the official test split by design, the other hands both stages one argument
set and always chains a refit.

The failure that made this file necessary: the 2026-08-27 smoke ran at M=6,
18 bases, and was committed as a 15-base pass. `--num_semantic_parts`,
`--num_codebooks` and `GDNA_NUM_SEMANTIC_PARTS` all default to 6 and nothing
asserted otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.phase3_selection_matrix import (  # noqa: E402
    A_FLAGS,
    BASES_PER_SLOT,
    CANDIDATE_N,
    CellRefused,
    LR_HORIZON,
    S5_FLAGS,
    SEED,
    SLOTS,
    TOTAL_BASES,
    TOTAL_BITS,
    assert_geometry,
    build_command,
    cell_keys,
    read_selection,
    tag_for,
)


# ------------------------------------------------------------- the grid

def test_the_matrix_is_exactly_sixteen_keys():
    keys = cell_keys()
    assert len(keys) == 16
    assert len(set(keys)) == 16
    assert sorted({d for d, _ in keys}) == [
        "cifar10", "flickr25k", "mscoco", "nuswide"]
    assert sorted({n for _, n in keys}) == sorted(CANDIDATE_N) == [4, 9, 19, 39]


def test_every_cell_has_its_own_tag():
    tags = {tag_for(d, n) for d, n in cell_keys()}
    assert len(tags) == 16


# --------------------------------------------------- what the child gets

@pytest.mark.parametrize("dataset,n", cell_keys())
def test_the_child_is_told_the_paper_geometry(dataset, n):
    """Both the environment and the CLI, because either alone is not enough.

    The slot count is baked in at import time, so the flag without the
    environment variable aborts; the environment variable without the flag
    would leave `args.txt` describing something else.
    """
    _, env, _ = build_command(dataset, n, gpu=0)
    assert env["GDNA_NUM_SEMANTIC_PARTS"] == str(SLOTS)
    extra = env["EXTRA_ARGS"].split()
    assert extra[extra.index("--num_semantic_parts") + 1] == str(SLOTS)
    assert extra[extra.index("--num_codebooks") + 1] == str(SLOTS)
    assert env["NUM_CODONS"] == str(BASES_PER_SLOT)
    assert SLOTS * BASES_PER_SLOT == TOTAL_BASES == 15
    assert TOTAL_BITS == 30


@pytest.mark.parametrize("dataset,n", cell_keys())
def test_the_search_horizons_follow_d2(dataset, n):
    """LR horizon fixed at 60 for every N; Sinkhorn horizon N+1."""
    _, env, _ = build_command(dataset, n, gpu=0)
    extra = env["EXTRA_ARGS"].split()
    assert extra[extra.index("-e") + 1] == "60"
    assert extra[extra.index("--lr_schedule_horizon") + 1] == str(LR_HORIZON)
    assert extra[extra.index("--sinkhorn_schedule_horizon") + 1] == str(n + 1)
    assert env["STOP_EP"] == str(n)


@pytest.mark.parametrize("dataset,n", cell_keys())
def test_stage_one_is_a_selection_run_on_held_out_train(dataset, n):
    _, env, _ = build_command(dataset, n, gpu=0)
    assert env["VAL_RATIO"] == "0.1"
    assert env["VAL_SEED"] == str(SEED)
    extra = env["EXTRA_ARGS"].split()
    # Left at its default the identity would call the search a refit.
    assert extra[extra.index("--selection_mode") + 1] == "select"
    # No final-epoch evaluation: the official test split is reserved.
    assert "FINAL_EPOCH" not in env
    assert "--final_epoch_eval" not in extra


@pytest.mark.parametrize("dataset,n", cell_keys())
def test_the_s5_recipe_is_passed(dataset, n):
    """No gumbel, codon-Sinkhorn off -- the 18-base smoke used neither."""
    _, env, _ = build_command(dataset, n, gpu=0)
    extra = env["EXTRA_ARGS"].split()
    assert "--no_gumbel_softmax" in extra
    assert extra[extra.index("--lambda_codeword_codon_sinkhorn") + 1] == "0.0"
    for flag in A_FLAGS:
        assert flag in extra


@pytest.mark.parametrize("dataset,n", cell_keys())
def test_the_child_reads_the_provenance_caches(dataset, n):
    _, env, _ = build_command(dataset, n, gpu=0)
    for key in ("CACHE", "EVAL_CACHE", "WHITEN_NPZ"):
        assert "groundeddna_cache_v6prov" in env[key], key
    # Stage 1 fits whitening on the optimization-train rows only.
    assert env["WHITEN_NPZ"].endswith("text_whiten_optTrain_localOnly.npz")


def test_the_selection_metric_is_raw_base_hamming():
    _, env, _ = build_command("cifar10", 4, gpu=0)
    extra = env["EXTRA_ARGS"].split()
    assert extra[extra.index("--dna_distance_mode") + 1] == "base"


# ------------------------------------------------ the geometry assertion

def _fake_run(tmp_path: Path, *, slots=SLOTS, codons=BASES_PER_SLOT,
              dataset="CIFAR10", n=4, k=64) -> Path:
    from types import SimpleNamespace
    from dna_utils.run_identity import RunIdentity, write_run_manifest

    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    fields = {"num_semantic_parts": slots, "num_codebooks": slots,
              "num_codons_per_codebook": codons}
    (run / "args.txt").write_text(
        "\n".join(f"{k_}{'-' * 20}{v}" for k_, v in fields.items()) + "\n")
    write_run_manifest(str(run), RunIdentity.from_args(SimpleNamespace(
        dataset=dataset, setting="setting1", random_seed=SEED, epoch=60,
        stop_after_epoch=n, num_semantic_parts=slots,
        num_codons_per_codebook=codons, codebook_size=k,
        selection_mode="select", val_split_ratio=0.1, val_split_seed=SEED)))
    return run


def test_a_correct_cell_passes_the_geometry_assertion(tmp_path):
    run = _fake_run(tmp_path)
    record = assert_geometry(run, dataset="cifar10", n=4)
    assert record["num_semantic_parts"] == SLOTS


def test_the_eighteen_base_configuration_is_refused(tmp_path):
    """The exact shape of the 2026-08-27 run."""
    run = _fake_run(tmp_path, slots=6)
    with pytest.raises(CellRefused) as excinfo:
        assert_geometry(run, dataset="cifar10", n=4)
    assert "18-base failure" in str(excinfo.value)


@pytest.mark.parametrize("field,value", [
    ("dataset", "NUSWIDE"), ("n", 9), ("k", 128)])
def test_a_cell_that_is_not_the_one_asked_for_is_refused(tmp_path, field, value):
    kwargs = {field: value} if field != "n" else {"n": value}
    run = _fake_run(tmp_path, **kwargs)
    with pytest.raises(CellRefused):
        assert_geometry(run, dataset="cifar10", n=4)


def test_args_and_manifest_must_agree(tmp_path):
    """One can be right while the other is not."""
    run = _fake_run(tmp_path)
    (run / "args.txt").write_text(
        "num_semantic_parts" + "-" * 20 + "6\n"
        "num_codebooks" + "-" * 20 + "6\n"
        "num_codons_per_codebook" + "-" * 20 + "3\n")
    with pytest.raises(CellRefused):
        assert_geometry(run, dataset="cifar10", n=4)


# ------------------------------------------------ the selection value

def _sidecar(tmp_path: Path, **extra) -> Path:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "model_state_dict_best.pth.runtime.json").write_text(json.dumps({
        "checkpoint_epoch_zero_based": 3, "checkpoint_sha256": "a" * 64,
        "effective_sinkhorn_epsilon": 0.5, "lr_schedule_horizon": 60,
        "sinkhorn_schedule_horizon": 5,
        "extra": {"checkpoint_role": "best_mid_eval", **extra}}))
    return run


def test_the_value_comes_from_the_checkpoint_that_scored_it(tmp_path):
    run = _sidecar(tmp_path, selection_metric="eval_mAP_at_R",
                   selection_value=0.5775)
    record = read_selection(run)
    assert record["selection_value"] == pytest.approx(0.5775)
    assert record["checkpoint_sha256"] == "a" * 64


def test_a_cell_with_no_best_checkpoint_is_refused(tmp_path):
    with pytest.raises(CellRefused) as excinfo:
        read_selection(tmp_path)
    assert "selected nothing" in str(excinfo.value)


def test_a_different_selection_metric_is_refused(tmp_path):
    run = _sidecar(tmp_path, selection_metric="eval_mAP", selection_value=0.5)
    with pytest.raises(CellRefused) as excinfo:
        read_selection(run)
    assert "base-Hamming mAP@R" in str(excinfo.value)


@pytest.mark.parametrize("value", [999, -0.1, None, "0.5", True])
def test_a_value_that_is_not_a_proportion_is_refused(tmp_path, value):
    run = _sidecar(tmp_path, selection_metric="eval_mAP_at_R",
                   selection_value=value)
    with pytest.raises(CellRefused):
        read_selection(run)
