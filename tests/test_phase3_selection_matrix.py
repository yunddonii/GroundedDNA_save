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
              dataset="CIFAR10", n=4, k=64, **overrides) -> Path:
    """A cell that satisfies the WHOLE protocol, so a test that breaks one axis
    breaks only that axis."""
    from types import SimpleNamespace
    from dna_utils.run_identity import RunIdentity, write_run_manifest
    from scripts.phase3_selection_matrix import DATASETS, VAL_RATIO

    cache = DATASETS["cifar10"]["cache"]
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    fields = {"num_semantic_parts": slots, "num_codebooks": slots,
              "num_codons_per_codebook": codons}
    (run / "args.txt").write_text(
        "\n".join(f"{k_}{'-' * 20}{v}" for k_, v in fields.items()) + "\n")
    args = dict(
        dataset=dataset, setting="setting1", random_seed=SEED, epoch=60,
        stop_after_epoch=n, num_semantic_parts=slots,
        num_codons_per_codebook=codons, codebook_size=k,
        selection_mode="select", val_split_ratio=VAL_RATIO,
        val_split_seed=SEED, lr_schedule_horizon=LR_HORIZON,
        sinkhorn_schedule_horizon=n + 1,
        siglip2_feature_cache_dir=cache, eval_cache_dir=cache)
    args.update(overrides)
    write_run_manifest(str(run), RunIdentity.from_args(SimpleNamespace(**args)))
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


@pytest.mark.parametrize("field,value", [
    ("selection_mode", "refit"),
    ("val_split_ratio", 0.9),
    ("val_split_seed", 999),
    ("epoch", 7),
    ("lr_schedule_horizon", 3),
    ("sinkhorn_schedule_horizon", 999),
    ("siglip2_feature_cache_dir", "/tmp/not_the_cache"),
    ("eval_cache_dir", "/tmp/not_the_cache"),
])
def test_every_protocol_axis_is_checked_not_just_the_geometry(
        tmp_path, field, value):
    """A probe with mode=refit, budget 7, horizons 3/999, val split .9/999 and
    fake caches passed the subset the first version checked."""
    run = _fake_run(tmp_path, **{field: value})
    with pytest.raises(CellRefused) as excinfo:
        assert_geometry(run, dataset="cifar10", n=4)
    assert "manifest disagrees" in str(excinfo.value)


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

def _cell_csv(tmp_path: Path, rows: list, *, header=None) -> Path:
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    cols = header or ["epoch", "eval_mAP", "eval_mAP_at_R", "eval_mAP_R_cutoff"]
    lines = [",".join(cols)]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in cols))
    (run / "log.csv").write_text("\n".join(lines) + "\n")
    return run


def _best_sidecar(run: Path, *, epoch: int, value: float) -> None:
    (run / "model_state_dict_best.pth.runtime.json").write_text(json.dumps({
        "checkpoint_epoch_zero_based": epoch, "checkpoint_sha256": "a" * 64,
        "extra": {"checkpoint_role": "best_mid_eval",
                  "selection_metric": "eval_mAP_at_R",
                  "selection_value": value}}))


def _rows(n: int, scored: dict) -> list:
    """Epochs 0..n, once each, scored only where the cadence says."""
    return [{"epoch": e,
             "eval_mAP_at_R": scored.get(e, ""),
             "eval_mAP_R_cutoff": 1000 if e in scored else ""}
            for e in range(n + 1)]


def test_the_value_is_the_candidates_own_terminal_epoch(tmp_path):
    run = _cell_csv(tmp_path, _rows(4, {0: 0.90, 4: 0.61}))
    record = read_selection(run, n=4)
    assert record["selection_value"] == pytest.approx(0.61)
    assert record["selection_epoch_zero_based"] == 4


def test_the_best_over_prefix_is_recorded_but_never_selected(tmp_path):
    """The defect: an N=39 cell whose best epoch was 4 reported the N=4 answer.

    Every candidate could then collapse onto the same early epoch, and the grid
    would compare nothing. The live N=4 smoke reporting its epoch-0 score was
    that bug in miniature.
    """
    run = _cell_csv(tmp_path, _rows(39, {0: 0.90, 39: 0.55}))
    _best_sidecar(run, epoch=0, value=0.90)
    record = read_selection(run, n=39)
    assert record["selection_value"] == pytest.approx(0.55), \
        "selection must not take the best over the prefix"
    assert record["best_over_prefix"]["value"] == pytest.approx(0.90)
    assert record["best_over_prefix"]["epoch_zero_based"] == 0


def test_a_cell_that_never_reached_its_epoch_is_refused(tmp_path):
    run = _cell_csv(tmp_path, _rows(0, {0: 0.9}))
    with pytest.raises(CellRefused) as excinfo:
        read_selection(run, n=39)
    assert "did not train to its candidate epoch" in str(excinfo.value)


def test_a_csv_with_a_gap_or_a_repeat_is_refused(tmp_path):
    """`log.csv` is append-only; two runs writing it produced a readable file."""
    gapped = [{"epoch": e, "eval_mAP_at_R": "", "eval_mAP_R_cutoff": ""}
              for e in (0, 1, 3, 4)]
    gapped[-1].update(eval_mAP_at_R=0.6, eval_mAP_R_cutoff=1000)
    with pytest.raises(CellRefused):
        read_selection(_cell_csv(tmp_path / "gap", gapped), n=4)

    repeated = _rows(4, {4: 0.6}) + _rows(4, {4: 0.7})[-1:]
    with pytest.raises(CellRefused):
        read_selection(_cell_csv(tmp_path / "dup", repeated), n=4)


def test_the_csv_the_number_came_from_is_digested(tmp_path):
    run = _cell_csv(tmp_path, _rows(4, {4: 0.61}))
    record = read_selection(run, n=4)
    assert len(record["log_csv_sha256"]) == 64


def test_a_terminal_row_without_a_cutoff_is_refused(tmp_path):
    """Without R the metric cannot be compared across datasets."""
    rows = _rows(4, {4: 0.61})
    rows[-1]["eval_mAP_R_cutoff"] = ""
    with pytest.raises(CellRefused) as excinfo:
        read_selection(_cell_csv(tmp_path, rows), n=4)
    assert "cutoff" in str(excinfo.value)


def test_a_terminal_epoch_with_no_score_is_refused(tmp_path):
    """Mid-eval does not run every epoch on every schedule."""
    run = _cell_csv(tmp_path, _rows(4, {0: 0.9}))
    with pytest.raises(CellRefused) as excinfo:
        read_selection(run, n=4)
    assert "never scored at its own terminal epoch" in str(excinfo.value)


def test_a_run_predating_the_per_epoch_metric_is_refused(tmp_path):
    """log.csv used to drop eval_mAP_at_R entirely."""
    run = _cell_csv(tmp_path, [{"epoch": e, "eval_mAP": 0.5}
                               for e in range(5)],
                    header=["epoch", "eval_mAP"])
    with pytest.raises(CellRefused) as excinfo:
        read_selection(run, n=4)
    assert "not recoverable" in str(excinfo.value)


def test_a_cell_with_no_csv_is_refused(tmp_path):
    with pytest.raises(CellRefused):
        read_selection(tmp_path, n=4)


@pytest.mark.parametrize("raw", ["999", "-0.1", "nope"])
def test_a_value_that_is_not_a_proportion_is_refused(tmp_path, raw):
    run = _cell_csv(tmp_path, _rows(4, {4: raw}))
    with pytest.raises(CellRefused):
        read_selection(run, n=4)


def test_the_trainer_persists_the_selection_metric_per_epoch():
    """It is computed every epoch; it used to be dropped from the CSV."""
    source = (REPO / "train_siglip2.py").read_text()
    fields = source[source.index("csv_fields += ["):]
    fields = fields[:fields.index("]")]
    assert '"eval_mAP_at_R"' in fields


# ------------------------------------------------ what cannot be launched

def test_only_must_name_one_of_the_sixteen_cells():
    """`--only cifar10:5` used to build an N=5 command."""
    import subprocess as _sp

    proc = _sp.run(
        [sys.executable, str(REPO / "scripts" / "phase3_selection_matrix.py"),
         "--run", "--only", "cifar10:5"],
        capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert proc.returncode == 2
    assert "not one of the" in proc.stderr


def test_the_caller_environment_cannot_redefine_the_recipe(monkeypatch):
    """The trainers read a dozen recipe variables from the environment."""
    for name, value in (("WASS", "9.9"), ("DISABLE_TEXT", "1"),
                        ("SHARE_CB", "1"), ("EXTRA_ARGS", "--nonsense"),
                        ("FINAL_EPOCH", "1")):
        monkeypatch.setenv(name, value)
    _, env, _ = build_command("cifar10", 4, gpu=0)
    for name in ("WASS", "DISABLE_TEXT", "SHARE_CB", "FINAL_EPOCH"):
        assert name not in env, f"{name} leaked in from the caller"
    assert "--nonsense" not in env["EXTRA_ARGS"]
    # The cell's own recipe still arrives.
    assert env["CCS"] == "0.1"


def test_the_child_runs_under_pipefail(tmp_path):
    """Executed, not grepped.

    The first attempt was `bash -o pipefail -c 'bash trainer.sh'`, which sets
    the option on the OUTER shell while the pipeline runs in the inner one --
    shell options do not survive the exec, so a failing python still came back
    as tee's zero. A source-substring test saw the literal `"-o", "pipefail"`
    and passed anyway.
    """
    import subprocess as _sp

    trainer = tmp_path / "trainer.sh"
    trainer.write_text("#!/usr/bin/env bash\nset -eu\n(exit 3) | cat\n")

    # The form that was committed first: the failure is masked.
    masked = _sp.run(
        ["bash", "-o", "pipefail", "-c", f"bash {trainer}"],
        capture_output=True, text=True, timeout=60)
    assert masked.returncode == 0, "the broken form must still be broken"

    # The form in use: the option is on the shell that runs the pipeline.
    surfaced = _sp.run(["bash", "-o", "pipefail", str(trainer)],
                       capture_output=True, text=True, timeout=60)
    assert surfaced.returncode == 3

    # And that is what the launcher builds.
    source = (REPO / "scripts" / "phase3_selection_matrix.py").read_text()
    assert '["bash", "-o", "pipefail", *cmd[1:]]' in source
    assert '"-o", "pipefail", "-c"' not in source


def test_the_launcher_prints_a_completed_cell_without_crashing(tmp_path):
    """The success path was never executed by a test, and it raised KeyError.

    `run_cell` published the per-cell record and returned; `main` then read a
    field that had been renamed out from under it. So a finished cell left an
    admission-looking record behind while the launcher exited 1 -- "a record
    exists" and "the launcher succeeded" became different states.
    """
    record = {
        "dataset": "cifar10", "N": 4,
        "geometry": {"num_semantic_parts": 5},
        "selection": {"selection_value": 0.5775,
                      "selection_epoch_zero_based": 4},
    }
    # Exactly the interpolation `main` performs on a successful cell.
    line = (f"[phase3] {record['dataset']}/N{record['N']} ok  "
            f"bases={record['geometry']}  "
            f"mAP@R={record['selection']['selection_value']:.4f} "
            f"@epoch {record['selection']['selection_epoch_zero_based']}")
    assert "@epoch 4" in line

    source = (REPO / "scripts" / "phase3_selection_matrix.py").read_text()
    assert "best_epoch_zero_based" not in source, (
        "main() reads a field read_selection no longer writes")


def test_a_candidate_whose_terminal_epoch_is_never_scored_is_refused():
    """The retrieval eval runs on a cadence, not every epoch.

    Every N in the current grid happens to satisfy (N+1) % 5 == 0. That is a
    coincidence, and it stops holding the moment someone edits the grid.
    """
    from scripts.phase3_selection_matrix import EVAL_EVERY

    assert all((n + 1) % EVAL_EVERY == 0 for n in CANDIDATE_N), (
        "the grid no longer lines up with the eval cadence; the launcher will "
        "refuse those cells, which is correct, but the grid needs a decision")


# ------------------------------------------- what the audit found next (§40)

def test_a_refit_smoke_keeps_its_seed_and_distance_mode():
    """Positional rewriting only works for the layout it was written against.

    The smoke rewrite sliced off the first six tokens, which are `-e`,
    `--random_seed <seed>` and `--dna_distance_mode base` in the REFIT layout,
    so a refit smoke ran at the trainer's default seed with no distance mode.
    """
    _, env, tag = build_command("cifar10", 4, 0, stage="refit", seed=43,
                                epochs=5)
    extra = env["EXTRA_ARGS"].split()
    assert extra[extra.index("--random_seed") + 1] == "43"
    assert extra[extra.index("--dna_distance_mode") + 1] == "base"
    assert extra[extra.index("-e") + 1] == "5"
    assert extra[extra.index("--selection_mode") + 1] == "refit"
    assert "_s43" in tag


def test_a_selection_smoke_keeps_its_flags_too():
    _, env, _ = build_command("cifar10", 4, 0, stage="select", epochs=5)
    extra = env["EXTRA_ARGS"].split()
    assert extra[extra.index("--random_seed") + 1] == str(SEED)
    assert extra[extra.index("--selection_mode") + 1] == "select"
    assert "--no_gumbel_softmax" in extra


def test_the_nominal_final_epoch_counts_as_scored():
    """The trainer evaluates on the cadence OR at the nominal final epoch.

    Requiring only the cadence refused `--smoke --epochs 1` before it ran,
    although the trainer would have scored epoch 0 as the final one.
    """
    import subprocess as _sp

    proc = _sp.run(
        [sys.executable, str(REPO / "scripts" / "phase3_selection_matrix.py"),
         "--plan", "--only", "cifar10:4"],
        capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert proc.returncode == 0
    # A one-epoch smoke stops at 0, which is that run's nominal final epoch.
    from scripts.phase3_selection_matrix import EVAL_EVERY
    assert (0 + 1) % EVAL_EVERY != 0, "the exception is what makes this legal"


def test_the_selection_file_must_name_exactly_the_four_datasets(tmp_path):
    from scripts.phase3_selection_matrix import DATASETS, _load_selection

    path = tmp_path / "selected_n.json"
    path.write_text(json.dumps({"selected": {
        **{d: {"selected_N": 4} for d in DATASETS},
        "invented": {"selected_N": 4}}}))
    with pytest.raises(CellRefused) as excinfo:
        _load_selection(path)
    assert "unexpected ['invented']" in str(excinfo.value)


def _completed_run(tmp_path: Path, **sidecar) -> Path:
    from types import SimpleNamespace
    from dna_utils.run_identity import RunIdentity, write_run_manifest
    import hashlib

    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    write_run_manifest(str(run), RunIdentity.from_args(SimpleNamespace(
        dataset="CIFAR10", setting="setting1", random_seed=SEED, epoch=60,
        stop_after_epoch=4, num_semantic_parts=SLOTS,
        num_codons_per_codebook=BASES_PER_SLOT,
        lr_schedule_horizon=LR_HORIZON, sinkhorn_schedule_horizon=5)))
    ckpt = run / "model_state_dict.pth"
    ckpt.write_bytes(b"weights")
    (run / "log.csv").write_text("epoch\n0\n")
    payload = {
        "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(),
        "checkpoint_epoch_zero_based": 4,
        "training_epoch_budget": 60, "stop_after_epoch": 4,
        # overridable below
        "lr_schedule_horizon": LR_HORIZON, "sinkhorn_schedule_horizon": 5,
    }
    payload.update(sidecar)
    (run / "model_state_dict.pth.runtime.json").write_text(json.dumps(payload))
    return run


def test_a_consistent_completed_run_is_admitted(tmp_path):
    from scripts.phase3_selection_matrix import assert_completed

    record = assert_completed(_completed_run(tmp_path), terminal_epoch=4)
    assert record["terminal_weights_preserved"] is True


@pytest.mark.parametrize("field,value", [
    ("training_epoch_budget", 999),
    ("stop_after_epoch", 123),
    ("lr_schedule_horizon", 999),
    ("sinkhorn_schedule_horizon", 999),
])
def test_a_sidecar_describing_another_run_is_refused(tmp_path, field, value):
    """Self-consistency is not identity: the SHA matched while the schedule
    it named belonged to a different run."""
    from scripts.phase3_selection_matrix import assert_completed

    run = _completed_run(tmp_path, **{field: value})
    with pytest.raises(CellRefused) as excinfo:
        assert_completed(run, terminal_epoch=4)
    assert "different run than" in str(excinfo.value)


def test_an_unfinished_run_still_holding_its_claim_is_refused(tmp_path):
    from dna_utils.run_identity import ACTIVE_CLAIM_NAME
    from scripts.phase3_selection_matrix import assert_completed

    run = _completed_run(tmp_path)
    (run / ACTIVE_CLAIM_NAME).write_text("{}")
    with pytest.raises(CellRefused) as excinfo:
        assert_completed(run, terminal_epoch=4)
    assert "active claim" in str(excinfo.value)


# ------------------------------------- the refit must produce a number (§40.6)

def test_a_refit_without_an_extraction_is_refused(tmp_path):
    from scripts.phase3_selection_matrix import assert_refit_outputs

    with pytest.raises(CellRefused) as excinfo:
        assert_refit_outputs(tmp_path, dataset="cifar10")
    assert "extraction does not validate" in str(excinfo.value)


def test_the_final_evaluation_failure_is_fatal():
    """It printed and continued, so a cell exited 0 with no metrics at all.

    The extraction above it was already fatal for the same reason: a run that
    reports success without a number cannot be told apart from one that has it.
    """
    source = (REPO / "train_siglip2.py").read_text()
    block = source[source.index("[final-eval] running evaluation"):]
    block = block[:block.index("# ---------- post-eval")]
    assert "raise RuntimeError" in block
    assert "continuing to viz" not in block


def test_only_the_refit_is_asked_for_official_outputs():
    """Stage 1 must never extract, so it cannot be required to have."""
    source = (REPO / "scripts" / "phase3_selection_matrix.py").read_text()
    assert 'if stage == "refit" else {}' in source


# ------------------- score and weights must be the same epoch (§42.5)

def test_a_selection_cell_keeps_its_terminal_checkpoint():
    """Without this the trainer swaps in an earlier best after training."""
    _, env, _ = build_command("cifar10", 9, 0, stage="select")
    assert "--keep_final_checkpoint" in env["EXTRA_ARGS"].split()


def test_the_trainer_honours_keep_final_checkpoint():
    source = (REPO / "train_siglip2.py").read_text()
    block = source[source.index("If best-checkpoint differs from final"):]
    block = block[:block.index("if _is_stop_point")]
    assert 'getattr(args, "keep_final_checkpoint", False)' in block
    assert "--keep_final_checkpoint" in (REPO / "config.py").read_text()


def test_a_cell_whose_weights_are_from_another_epoch_is_refused(tmp_path):
    """CIFAR N9 recorded 0.8494 at epoch 9 with epoch-4 weights, and passed.

    The metric was right -- epoch 9's row, written before the swap -- but a
    cell whose two halves describe different epochs is not self-consistent
    evidence, and it was admitted with a boolean rather than refused.
    """
    from scripts.phase3_selection_matrix import assert_completed

    run = _completed_run(tmp_path, checkpoint_epoch_zero_based=4)
    with pytest.raises(CellRefused) as excinfo:
        assert_completed(run, terminal_epoch=9)
    assert "different epochs" in str(excinfo.value)
    assert "--keep_final_checkpoint" in str(excinfo.value)


def test_the_matrix_wrapper_counts_the_exact_sixteen_keys():
    """It counted every JSON, so one stray diagnostic made 16 unreachable."""
    source = (REPO / "scripts" / "phase3_launch_matrix.sh").read_text()
    assert "phase3sel_${EXP}_N${N}_s42.json" in source
    assert "grep -vc selected_n" not in source
    # And it is one of its own dirty-tree dependencies.
    assert "DEPS=(scripts/phase3_launch_matrix.sh" in source
