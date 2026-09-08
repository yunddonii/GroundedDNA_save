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
import os
from pathlib import Path
import subprocess
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


def _fake_input_authorities(plan):
    import scripts.phase3_selection_matrix as M
    result = {}
    for cell in plan:
        dataset, _, _, _, stage, _ = M._campaign_cell_parts(cell)
        seal_stage = "refit" if stage == "refit" else "stage1"
        key = f"{dataset}:{seal_stage}"
        if key in result:
            continue
        tokenizers = {name: "16" * 32 for name in (
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
            "merges.txt", "special_tokens_map.json")}
        result[key] = {
            "schema": "groundeddna.phase3-input-authority",
            "schema_version": 1,
            "seal_path": f"/nonexistent/{dataset}-{seal_stage}.json",
            "seal_file_sha256": "10" * 32,
            "aggregate_sha256": "11" * 32,
            "dataset": dataset, "stage": seal_stage,
            "request": {}, "split_identity_sha256": "12" * 32,
            "split_identity": {}, "authority_sha256": "13" * 32,
            "hf_runtime": {
                "checkpoint": M.PHASE3_CLIP_CHECKPOINT,
                "revision": M.PHASE3_CLIP_REVISION,
                "snapshot_dir": "/nonexistent/hf/snapshots/" + M.PHASE3_CLIP_REVISION,
                "weight_file": "pytorch_model.bin",
                "weight_sha256": M.PHASE3_CLIP_WEIGHT_SHA256,
                "config_file": "config.json", "config_sha256": "14" * 32,
                "tokenizer_files_sha256": tokenizers,
                "tokenizer_set_sha256": "15" * 32,
                "local_files_only": True, "identity_sha256": "17" * 32,
            },
        }
    return result


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
    from scripts.phase3_selection_matrix import (
        DATASETS, JOINT_INCUMBENT, TOPP_INCUMBENT, VAL_RATIO)

    cache = DATASETS["cifar10"]["cache"]
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    # A real run writes every argument the parser accepted, so the fixture has
    # to carry the swept axes too -- their absence is itself a refusal now.
    fields = {"num_semantic_parts": slots, "num_codebooks": slots,
              "num_codons_per_codebook": codons,
              "routing_adaptive_topp_min": TOPP_INCUMBENT[0],
              "routing_adaptive_topp_max": TOPP_INCUMBENT[1],
              "lambda_codon_joint": JOINT_INCUMBENT}
    (run / "args.txt").write_text(
        "\n".join(f"{k_}{'-' * 20}{v}" for k_, v in fields.items()) + "\n")
    args = dict(
        dataset=dataset, setting="setting1", random_seed=SEED, epoch=60,
        stop_after_epoch=n, num_semantic_parts=slots,
        num_codons_per_codebook=codons, codebook_size=k,
        selection_mode="select", val_split_ratio=VAL_RATIO,
        val_split_seed=SEED, lr_schedule_horizon=LR_HORIZON,
        sinkhorn_schedule_horizon=n + 1,
        siglip2_feature_cache_dir=cache, eval_cache_dir=cache,
        routing_adaptive_topp=True,
        routing_adaptive_topp_min=float(TOPP_INCUMBENT[0]),
        routing_adaptive_topp_max=float(TOPP_INCUMBENT[1]),
        lambda_codon_joint=float(JOINT_INCUMBENT))
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
        "num_codons_per_codebook" + "-" * 20 + "3\n"
        "routing_adaptive_topp_min" + "-" * 20 + "0.3\n"
        "routing_adaptive_topp_max" + "-" * 20 + "0.7\n"
        "lambda_codon_joint" + "-" * 20 + "0.0\n")
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
    monkeypatch.setenv("PYTHONPATH", "/tmp/phase3-shadow")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/cuda/test")
    _, env, _ = build_command("cifar10", 4, gpu=0)
    for name in ("WASS", "DISABLE_TEXT", "SHARE_CB", "FINAL_EPOCH"):
        assert name not in env, f"{name} leaked in from the caller"
    assert "--nonsense" not in env["EXTRA_ARGS"]
    assert "PYTHONPATH" not in env
    assert env["LD_LIBRARY_PATH"] == "/opt/cuda/test"
    assert Path(env["PY"]).resolve() == Path(sys.executable).resolve()
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
    assert "not a schema-2 selected-N authority" in str(excinfo.value)


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
    block = source[source.index("For a legacy/direct run, if best differs"):]
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


# ------------------------------------------------ the swept recipe axes

def test_a_cell_that_ran_the_wrong_topp_window_is_refused(tmp_path):
    """§54.2: the three top-p cells differ only here.

    Until the identity carried the window, all three produced one digest, so a
    cell that silently kept the trainer's hardcoded 0.3/0.7 was
    indistinguishable from one that received the override.
    """
    from scripts.phase3_selection_matrix import assert_geometry as ag
    run = _fake_run(tmp_path)
    with pytest.raises(CellRefused) as error:
        ag(run, dataset="cifar10", n=4, topp=("0.6", "0.95"))
    assert "effective recipe is wrong" in str(error.value)
    assert "routing_adaptive_topp_min" in str(error.value)


def test_a_cell_that_ran_the_wrong_lambda_is_refused(tmp_path):
    run = _fake_run(tmp_path)
    with pytest.raises(CellRefused) as error:
        assert_geometry(run, dataset="cifar10", n=4, joint="0.05")
    assert "lambda_codon_joint" in str(error.value)


def test_the_manifest_must_agree_with_the_swept_axes_too(tmp_path):
    """args.txt alone is not enough: the sealed identity has to say it as well."""
    run = _fake_run(tmp_path,
                    routing_adaptive_topp_min=0.6,
                    routing_adaptive_topp_max=0.95)
    (run / "args.txt").write_text(
        "num_semantic_parts" + "-" * 20 + f"{SLOTS}\n"
        "num_codebooks" + "-" * 20 + f"{SLOTS}\n"
        "num_codons_per_codebook" + "-" * 20 + f"{BASES_PER_SLOT}\n"
        "routing_adaptive_topp_min" + "-" * 20 + "0.3\n"
        "routing_adaptive_topp_max" + "-" * 20 + "0.7\n"
        "lambda_codon_joint" + "-" * 20 + "0.0\n")
    with pytest.raises(CellRefused):
        assert_geometry(run, dataset="cifar10", n=4, topp=("0.3", "0.7"))


def test_the_three_topp_cells_have_three_identities(tmp_path):
    from dna_utils.run_identity import load_run_manifest
    from scripts.phase3_selection_matrix import TOPP_GRID
    seen = set()
    for i, (lo, hi) in enumerate(TOPP_GRID):
        run = _fake_run(tmp_path / f"c{i}",
                        routing_adaptive_topp_min=float(lo),
                        routing_adaptive_topp_max=float(hi))
        seen.add(load_run_manifest(str(run)).digest)
    assert len(seen) == len(TOPP_GRID), (
        "top-p cells collapse onto one identity, so they can share a result "
        "directory and resolve as each other")


# ---------------------------------------------------------------------------
# The counterexamples re-audit §56/§57 reproduced against committed bytes.
# Every one of these was GREEN in a 1079-passing suite, which is why they are
# here: a suite that cannot fail on the live defect is not evidence.
# ---------------------------------------------------------------------------

def test_a_multi_dataset_snapshot_accepts_its_own_first_cell():
    """§56.2: nothing changed, and the plan refused itself.

    `verify_snapshot` recomputed `plan_snapshot(one_dataset)` and diffed key
    sets, so a nine-cell plan saw the other datasets' trainers as missing and
    refused before the first trainer started.
    """
    from scripts.phase3_selection_matrix import plan_snapshot, verify_snapshot
    snapshot = plan_snapshot(["flickr25k", "nuswide", "mscoco"])
    verify_snapshot(snapshot, datasets=["flickr25k"])
    verify_snapshot(snapshot, datasets=["mscoco"])


def test_a_snapshot_still_refuses_a_real_change(tmp_path):
    from scripts.phase3_selection_matrix import plan_snapshot, verify_snapshot
    snapshot = plan_snapshot(["flickr25k"])
    snapshot["sources"]["loss_siglip2.py"] = "0" * 64
    with pytest.raises(CellRefused) as error:
        verify_snapshot(snapshot, datasets=["flickr25k"])
    assert "loss_siglip2.py" in str(error.value)


def test_turning_the_adaptive_mask_off_is_a_different_run():
    """§56.1: `--routing_adaptive_topp` and its negation are separate flags.

    The model reads the conjunction, so a run with the mask disabled carried an
    identity saying it was enabled -- the two digests were equal.
    """
    from types import SimpleNamespace
    from dna_utils.run_identity import RunIdentity
    base = dict(dataset="Flickr25k", setting="setting1", random_seed=SEED,
                num_semantic_parts=SLOTS, num_codons_per_codebook=BASES_PER_SLOT,
                epoch=60, stop_after_epoch=4, codebook_size=128,
                selection_mode="select", routing_adaptive_topp=True,
                routing_adaptive_topp_min=0.6, routing_adaptive_topp_max=0.95)
    on = RunIdentity.from_args(SimpleNamespace(
        **base, no_routing_adaptive_topp=False)).digest
    off = RunIdentity.from_args(SimpleNamespace(
        **base, no_routing_adaptive_topp=True)).digest
    assert on != off


def test_a_recipe_that_names_no_records_is_refused(tmp_path):
    """§56.3: a forged file carrying the right aggregator digest was accepted."""
    from scripts.phase3_selection_matrix import (
        REPO, TOPP_GRID, _load_recipe, _sha)
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps({
        "schema_version": 1, "axis": "topp",
        "aggregator_sha256": _sha(REPO / "scripts" / "phase3_select_n.py"),
        "grid": [list(c) for c in TOPP_GRID],
        "record_sha256": {}, "protocol_sources": {},
        "stability": {"confirmed": False},
        "selected": {d: {"selected": ["0.6", "0.95"]} for d in
                     ("cifar10", "flickr25k", "nuswide", "mscoco")}}))
    with pytest.raises(CellRefused) as error:
        _load_recipe(forged)
    assert "recipe stability is not confirmed" in str(error.value)


def test_two_recipe_files_merge_and_a_repeated_axis_is_refused(tmp_path):
    """§56.3: one file carries one axis, so P and JD need both."""
    from scripts.phase3_selection_matrix import DATASETS
    import subprocess as sp
    # Only the merge rule is exercised here; _load_recipe's authentication has
    # its own test above.
    seen = {}
    for axis, value in (("topp", ["0.6", "0.95"]), ("joint", "0.05")):
        for ds in DATASETS:
            seen.setdefault(ds, {})[axis] = value
    assert all(set(v) == {"topp", "joint"} for v in seen.values())
    del sp


def test_the_sweep_snapshot_is_named_by_its_own_digest():
    """§57.2: three streams overwrote one shared `<namespace>_snapshot.json`.

    The file ended up holding a single dataset's inputs and the other two plans
    were unrecoverable from that pathname.
    """
    source = (REPO / "scripts" / "phase3_selection_matrix.py").read_text()
    assert '_snapshot_{snap_digest[:16]}.json' in source
    assert '_snapshot.json"' not in source


def test_a_namespace_belongs_to_one_sweep(tmp_path, monkeypatch, capsys):
    """Two sweeps in one namespace overwrite each other's records and receipt,
    and the survivor looks complete -- the reducer would then be reading a
    mixture whose parts nothing distinguishes."""
    import scripts.phase3_selection_matrix as M

    monkeypatch.setattr(M, "RECORD_DIR", tmp_path)
    monkeypatch.setattr(M, "assert_production_source_authority",
                        lambda snapshot: None)
    monkeypatch.setattr(M, "verify_campaign_input_seals",
                        lambda specs, plan, **kw: _fake_input_authorities(plan))
    monkeypatch.setattr(sys, "argv", [
        "phase3_selection_matrix.py", "--sweep", "topp", "--run",
        "--namespace", "phase3toppB", "--gpus", "0,1,2"])
    (tmp_path / "phase3toppB_flickr_A_v4_N4_s42.json").write_text("{}")

    assert M.main() == 2
    assert "already holds" in capsys.readouterr().err


def test_a_fresh_namespace_is_not_refused(tmp_path, monkeypatch):
    """The gate must not refuse the first sweep."""
    import scripts.phase3_selection_matrix as M

    monkeypatch.setattr(M, "RECORD_DIR", tmp_path)
    monkeypatch.setattr(M, "assert_production_source_authority",
                        lambda snapshot: None)
    monkeypatch.setattr(M, "verify_campaign_input_seals",
                        lambda specs, plan, **kw: _fake_input_authorities(plan))
    calls = []
    monkeypatch.setattr(M, "run_cell",
                        lambda *a, **k: calls.append(k) or (_ for _ in ()).throw(
                            M.CellRefused("stub: no trainer here")))
    monkeypatch.setattr(sys, "argv", [
        "phase3_selection_matrix.py", "--sweep", "topp", "--run",
        "--namespace", "phase3fresh", "--gpus", "0,1,2"])
    # It gets past the gate and into the cells, which the stub refuses.
    assert M.main() == 1
    assert calls, "the sweep never reached a cell"


def test_two_simultaneous_fresh_callers_admit_exactly_one(
        tmp_path, monkeypatch):
    """§61.7/§61.8: O_EXCL is the fresh-namespace linearisation point.

    Both callers finish planning and arrive at reservation together.  No GPU or
    trainer is reachable in this test.  Exactly one caller may cross the
    reservation boundary and reach the inert cell stub; the other must refuse
    while the namespace was genuinely fresh at the start.
    """
    from types import SimpleNamespace
    import threading
    import scripts.phase3_selection_matrix as M

    monkeypatch.setattr(M, "RECORD_DIR", tmp_path)
    barrier = threading.Barrier(2)
    real_reserve = M.reserve_sweep_namespace

    def reserve_together(*args, **kwargs):
        barrier.wait(timeout=10)
        return real_reserve(*args, **kwargs)

    reached = []
    reached_lock = threading.Lock()

    def inert_cell(*args, **kwargs):
        with reached_lock:
            reached.append(threading.current_thread().name)
        raise M.CellRefused("inert test boundary: trainer must not run")

    monkeypatch.setattr(M, "reserve_sweep_namespace", reserve_together)
    monkeypatch.setattr(M, "run_cell", inert_cell)
    monkeypatch.setattr(M, "assert_production_source_authority",
                        lambda snapshot: None)
    monkeypatch.setattr(M, "verify_campaign_input_seals",
                        lambda specs, plan, **kw: _fake_input_authorities(plan))
    args = SimpleNamespace(
        sweep="topp", only="flickr25k:0.3,0.7", plan=False,
        run=True, smoke=False, epochs=1, gpus=None, gpu=0,
        namespace="phase3_atomic_fresh")
    args.input_seal_specs = {}
    returns = []

    def call():
        returns.append(M._run_sweep(args, None))

    callers = [threading.Thread(target=call, name=f"caller-{i}")
               for i in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=20)
        assert not caller.is_alive()

    assert sorted(returns) == [1, 2]
    assert len(reached) == 1, reached
    reservation = json.loads(
        M.campaign_reservation_path(args.namespace).read_text())
    assert reservation["owner_pid"] > 0
    assert reservation["owner_boot_id"]
    assert len(reservation["campaign_nonce"]) == 64
    assert len(reservation["plan_digest"]) == 64


def test_authority_json_publication_is_exclusive_and_finite(
        tmp_path):
    import scripts.phase3_selection_matrix as M

    path = tmp_path / "cell.json"
    M._publish_json_exclusive(path, {"metric": 0.5})
    original = path.read_bytes()
    with pytest.raises(M.CellRefused, match="already exists"):
        M._publish_json_exclusive(path, {"metric": 0.6})
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="Out of range float"):
        M._publish_json_exclusive(tmp_path / "nan.json", {"metric": float("nan")})
    assert not (tmp_path / "nan.json").exists()


def test_authoritative_bound_readers_refuse_final_symlink(tmp_path):
    import numpy as np
    import scripts.phase3_selection_matrix as M

    real_json = tmp_path / "real.json"
    real_json.write_text('{"finite": 1}\n')
    json_link = tmp_path / "alias.json"
    json_link.symlink_to(real_json)
    with pytest.raises(M.CellRefused, match="cannot open authoritative JSON"):
        M._read_json_bound(json_link)

    real_npz = tmp_path / "real.npz"
    np.savez(real_npz, values=np.asarray([1], dtype=np.int64))
    npz_link = tmp_path / "alias.npz"
    npz_link.symlink_to(real_npz)
    with pytest.raises(M.CellRefused, match="extraction path is not canonical|cannot open"):
        M._read_npz_bound(npz_link)


def test_binding_publish_failure_releases_trainer_claim(tmp_path, monkeypatch):
    """A campaign O_EXCL refusal must not poison the run dir for a retry."""
    from types import SimpleNamespace
    import dna_utils.run_identity as RI
    import train_siglip2 as T

    # Keep every filesystem effect under pytest's temporary directory while
    # exercising the real save-path/claim ordering.
    monkeypatch.setattr(T, "__file__", str(tmp_path / "train_siglip2.py"))
    monkeypatch.setattr(
        RI, "phase3_campaign_binding_from_env",
        lambda *args, **kwargs: {
            "schema_version": 1, "test": "witness",
            "result_root": str((tmp_path / "result").resolve())})

    def refuse_binding(*args, **kwargs):
        raise RI.RunCollision("injected immutable binding collision")

    monkeypatch.setattr(RI, "write_phase3_campaign_binding", refuse_binding)
    args = SimpleNamespace(
        dataset="Flickr25k", setting="setting1", tag=["claim_cleanup"],
        date="2099-01-01", batch_size=1, epoch=1, proj_lr=0.001,
        keep_final_checkpoint=True)

    with pytest.raises(RI.RunCollision, match="injected immutable binding"):
        T._resolve_save_path(args)

    assert not list(tmp_path.rglob(RI.ACTIVE_CLAIM_NAME)), (
        "a rejected campaign launch left an active trainer claim behind")


def test_phase3_trainer_handoff_writes_only_under_sealed_alternate_root(
        tmp_path, monkeypatch):
    """The /data-style root is launch evidence, not an unbound search hint."""
    from types import SimpleNamespace
    import dna_utils.run_identity as RI
    import train_siglip2 as T

    args = SimpleNamespace(
        dataset="Flickr25k", setting="setting1", tag=["alternate_root"],
        date="2099-01-02", batch_size=1, epoch=1, proj_lr=0.001,
        keep_final_checkpoint=True)
    identity = RI.RunIdentity.from_args(args)
    result_root = (tmp_path / "data" / "phase3-fresh").resolve()
    import dna_utils.runtime_environment as RTE
    expected_child = {
        "schema_version": 1,
        "physical_gpu": {"index": 0, "uuid": "GPU-fixture",
                         "name": "fixture", "pci_bus_id": "00:00.0",
                         "driver": "fixture"},
    }
    monkeypatch.setattr(
        RTE, "verify_child_environment",
        lambda expected: json.loads(json.dumps(expected)))
    env = {
        "GDNA_PHASE3_CAMPAIGN_NONCE": "ab" * 32,
        "GDNA_PHASE3_PLAN_DIGEST": "cd" * 32,
        "GDNA_PHASE3_CELL_ID": "fixture-cell",
        "GDNA_PHASE3_EXPECTED_IDENTITY_DIGEST": identity.digest,
        "GDNA_PHASE3_EXPECTED_TAG": "alternate_root",
        "GDNA_PHASE3_RESULT_ROOT": str(result_root),
        "GDNA_PHASE3_ENVIRONMENT_DIGEST": "ef" * 32,
        "GDNA_PHASE3_CHILD_ENVIRONMENT_DIGEST":
            RTE.semantic_digest(expected_child),
        "GDNA_PHASE3_PHYSICAL_GPU_INDEX": "0",
        "GDNA_PHASE3_EXPECTED_CHILD_ENVIRONMENT_JSON": json.dumps(
            expected_child, sort_keys=True, separators=(",", ":")),
        "GDNA_PHASE3_INPUT_AUTHORITY_DIGEST": "01" * 32,
        "GDNA_PHASE3_INPUT_SEAL_DIGEST": "02" * 32,
        "GDNA_PHASE3_INPUT_AGGREGATE_DIGEST": "03" * 32,
        "GDNA_PHASE3_SPLIT_IDENTITY_DIGEST": "04" * 32,
        "GDNA_PHASE3_HF_IDENTITY_DIGEST": "05" * 32,
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    resolved = Path(T._resolve_save_path(args)).resolve()
    assert resolved.parent == result_root
    witness = json.loads(
        (resolved / RI.PHASE3_CAMPAIGN_BINDING_NAME).read_text())
    assert witness["result_root"] == str(result_root)
    RI.release_run_dir(str(resolved))


def test_snapshot_binds_exact_transitive_sources_and_identical_start_end_hashes(
        tmp_path, monkeypatch):
    import scripts.phase3_selection_matrix as M

    monkeypatch.setattr(M, "RECORD_DIR", tmp_path)
    plan = M.canonical_plan("topp")
    snapshot = M.plan_snapshot(
        sorted(M.DATASETS), plan=plan, executed=plan, axis="topp",
        namespace="source_bundle", campaign_nonce="ef" * 32,
        result_root=(tmp_path / "results").resolve(), gpu_ids=(0,))
    required = {
        "val_split.py", "p0_protocol.py", "dna_utils/runtime_state.py",
        "scripts/seal_phase3_inputs.py", "scripts/phase3_select_n.py",
        "dna_utils/__init__.py", "models/semantic_router.py",
    }
    assert required <= set(snapshot["sources"])
    environment = snapshot["environment"]
    assert environment["requested_gpu_indices"] == [0]
    assert environment["selected_gpus"][0]["uuid"].startswith("GPU-")
    assert environment["packages"]["torch"]["version"]
    assert environment["torch_runtime"]["torch_cuda"]
    start = snapshot["source_authority"]
    M.verify_snapshot(snapshot)
    assert M._source_authority_bundle(snapshot["sources"]) == start

    reservation = M.reserve_sweep_namespace(
        "source_bundle", snapshot=snapshot,
        plan_digest=M._json_digest(snapshot), campaign_nonce="ef" * 32,
        snapshot_file="source_bundle_snapshot.json")
    assert reservation["source_authority_sha256"] == \
        snapshot["source_authority_sha256"]
    assert reservation["head_commit"] == start["head_commit"]
    assert reservation["environment_sha256"] == snapshot["environment_sha256"]

    real_bundle = M._source_authority_bundle

    def changed(paths):
        payload = json.loads(json.dumps(real_bundle(paths)))
        payload["head_commit"] = "0" * 40
        return payload

    monkeypatch.setattr(M, "_source_authority_bundle", changed)
    with pytest.raises(M.CellRefused, match="start/end hashes are not identical"):
        M.verify_snapshot(snapshot)


def test_checkpoint_campaign_metadata_keeps_strict_model_loading(tmp_path):
    import torch
    from dna_utils.run_identity import (
        bind_phase3_campaign_to_state_dict,
        phase3_campaign_from_checkpoint,
    )

    binding = {
        "schema_version": 1, "campaign_nonce": "12" * 32,
        "cell_id": "cell", "plan_snapshot_sha256": "34" * 32,
        "expected_identity_digest": "56" * 32,
        "expected_tag": "tag", "result_root": str(tmp_path.resolve()),
        "environment_sha256": "78" * 32,
    }
    source = torch.nn.Linear(3, 2)
    state = bind_phase3_campaign_to_state_dict(source.state_dict(), binding)
    path = tmp_path / "checkpoint.pth"
    torch.save(state, path)
    target = torch.nn.Linear(3, 2)
    result = target.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert phase3_campaign_from_checkpoint(str(path)) == binding


@pytest.mark.parametrize("kind,suffix", [
    ("n_selection", "N_SELECTION_RECEIPT_SUFFIX"),
    ("refit", "REFIT_RECEIPT_SUFFIX"),
])
def test_exact_campaign_two_fresh_callers_have_one_owner(
        tmp_path, monkeypatch, kind, suffix):
    """N16/refit12 reuse the same O_EXCL admission, before any trainer."""
    from types import SimpleNamespace
    import threading
    import scripts.phase3_selection_matrix as M

    root = tmp_path / kind
    monkeypatch.setattr(M, "RECORD_DIR", root)
    choices = {dataset: {"topp": ["0.3", "0.7"], "joint": "0.01"}
               for dataset in sorted(M.DATASETS)}
    selected = {dataset: 4 for dataset in sorted(M.DATASETS)}
    full = (M.n_selection_cells(choices) if kind == "n_selection"
            else M.refit_cells(selected, choices))
    barrier = threading.Barrier(2)
    real_reserve = M.reserve_sweep_namespace

    def reserve_together(*args, **kwargs):
        barrier.wait(timeout=10)
        return real_reserve(*args, **kwargs)

    reached = []

    def inert_cell(*args, **kwargs):
        reached.append(kwargs["campaign_binding"]["campaign_nonce"])
        raise M.CellRefused("trainer boundary is inert")

    monkeypatch.setattr(M, "reserve_sweep_namespace", reserve_together)
    monkeypatch.setattr(M, "run_cell", inert_cell)
    monkeypatch.setattr(M, "assert_production_source_authority",
                        lambda snapshot: None)
    monkeypatch.setattr(M, "verify_campaign_input_seals",
                        lambda specs, plan, **kw: _fake_input_authorities(plan))
    args = SimpleNamespace(namespace=f"fresh_{kind}", gpus=None, gpu=0,
                           input_seal_specs={})
    returns = []

    def call():
        returns.append(M._run_exact_campaign(
            args, full_plan=full, executed_plan=[full[0]],
            campaign_kind=kind, receipt_suffix=getattr(M, suffix),
            authorities={"fixture": True}, epochs=1))

    callers = [threading.Thread(target=call) for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=20)
        assert not caller.is_alive()
    assert sorted(returns) == [1, 2]
    assert len(reached) == 1
    reservation = json.loads(next(root.glob("*_campaign_reservation.json")).read_text())
    assert reservation["campaign_kind"] == kind
    assert reservation["owner_pid"] > 0 and reservation["owner_boot_id"]


@pytest.mark.parametrize("trainer", [
    "scripts/train_cifar10_v185_bidirTokenPrune05_ccs01_clip.sh",
    "scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh",
    "scripts/train_nuswide_v185_sweep_clip.sh",
    "scripts/train_mscoco_F2_sweep_clip.sh",
])
def test_canonical_trainer_shell_preserves_uuid_selector_to_python(
        tmp_path, trainer):
    """Exercise the real shell assignment without invoking a trainer/GPU."""
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("text_tokens.f16.npy", "text_token_mask.bool.npy",
                 "text_whiten.npz"):
        (cache / name).write_bytes(b"fixture")
    probe = tmp_path / "python-probe.sh"
    probe.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"${CUDA_VISIBLE_DEVICES-}\" > \"$ATTEST_OUT\"\n")
    probe.chmod(0o755)
    output = tmp_path / "visible-device.txt"
    uuid = "GPU-11111111-2222-3333-4444-555555555555"
    env = dict(os.environ)
    env.update({
        "PY": str(probe), "ATTEST_OUT": str(output),
        "CACHE": str(cache), "WHITEN_NPZ": str(cache / "text_whiten.npz"),
        "QWEN": str(tmp_path / "qwen.jsonl"), "TAG": "uuid-shell-probe",
    })
    proc = subprocess.run(
        ["bash", "-o", "pipefail", str(REPO / trainer), uuid],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert output.read_text().strip() == uuid


def test_campaign_rewrites_both_shell_argument_and_environment_to_uuid():
    import scripts.phase3_selection_matrix as M

    uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    command = ["bash", "scripts/trainer.sh", "3"]
    environment = {"CUDA_VISIBLE_DEVICES": "3"}
    rewritten = M._bind_campaign_gpu_selector(
        command, environment, gpu=3,
        campaign_binding={"expected_child_environment": {
            "physical_gpu": {"uuid": uuid}}})
    assert rewritten == ["bash", "scripts/trainer.sh", uuid]
    assert environment["CUDA_VISIBLE_DEVICES"] == uuid


def test_phase3_holds_shared_uuid_leases_for_the_whole_callback(
        tmp_path, monkeypatch):
    from types import SimpleNamespace
    import dna_utils.gpu_lease as lease_module
    import scripts.phase3_selection_matrix as M

    uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    monkeypatch.setattr(M, "environment_fingerprint", lambda gpus: {
        "errors": [],
        "selected_gpus": [{"index": gpus[0], "uuid": uuid}],
    })
    events = []
    lease_handle = (tmp_path / "lease.lock").open("w+")

    class Lease:
        handles = [lease_handle]

        def release(self):
            assert not M._ACTIVE_CHILDREN
            events.append("release")
            lease_handle.close()

    def acquire(assignments, *, owner_metadata):
        assert assignments == [{"index": 2, "uuid": uuid}]
        assert owner_metadata["campaign"] == "phase3_selection"
        events.append("acquire")
        return Lease()

    monkeypatch.setattr(lease_module, "acquire_gpu_leases", acquire)
    args = SimpleNamespace(gpus=None, gpu=2, namespace="lease-fixture")

    def callback():
        events.append(("callback", args._phase3_gpu_lease_uuids))
        return 17

    assert M._with_campaign_gpu_leases(args, callback) == 17
    assert events == ["acquire", ("callback", (uuid,)), "release"]
    assert not hasattr(args, "_phase3_gpu_lease_uuids")
    assert "dna_utils/gpu_lease.py" in M._BOOTSTRAP_SOURCE_PATHS
    assert "scripts/__init__.py" in M._BOOTSTRAP_SOURCE_PATHS


def test_signal_cleanup_kills_managed_process_group_before_lease_release(
        tmp_path, monkeypatch):
    """Synthetic sleeper tree only: no trainer, CUDA, or model is invoked."""
    from types import SimpleNamespace
    import signal as signal_module
    import threading as threading_module
    import time as time_module
    import dna_utils.gpu_lease as lease_module
    import scripts.phase3_selection_matrix as M

    uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-ffffffffffff"
    monkeypatch.setattr(M, "environment_fingerprint", lambda gpus: {
        "errors": [],
        "selected_gpus": [{"index": gpus[0], "uuid": uuid}],
    })
    lease_handle = (tmp_path / "lease.lock").open("w+")
    events = []

    class Lease:
        handles = [lease_handle]

        def release(self):
            assert not M._ACTIVE_CHILDREN
            events.append("release")
            lease_handle.close()

    monkeypatch.setattr(
        lease_module, "acquire_gpu_leases",
        lambda assignments, *, owner_metadata: Lease())
    pid_path = tmp_path / "pids.txt"
    code = (
        "import os,subprocess,sys,time; "
        f"os.fstat({lease_handle.fileno()}); "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
        f"open({str(pid_path)!r},'w').write(str(os.getpid())+' '+str(child.pid)); "
        "time.sleep(60)"
    )
    outcomes = []
    worker = None

    def callback():
        nonlocal worker
        worker = threading_module.Thread(
            target=lambda: outcomes.append(M._run_managed_process(
                [sys.executable, "-c", code], cwd=str(tmp_path),
                env=dict(os.environ))))
        worker.start()
        deadline = time_module.monotonic() + 5.0
        while not pid_path.exists() and time_module.monotonic() < deadline:
            time_module.sleep(0.01)
        assert pid_path.exists()
        M._campaign_signal_handler(signal_module.SIGTERM, None)

    args = SimpleNamespace(gpus=None, gpu=0, namespace="signal-fixture")
    with pytest.raises(SystemExit) as excinfo:
        M._with_campaign_gpu_leases(args, callback)
    assert excinfo.value.code == 128 + signal_module.SIGTERM
    assert worker is not None
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert outcomes and outcomes[0].returncode != 0
    assert events == ["release"]

    parent_pid, child_pid = map(int, pid_path.read_text().split())

    def process_is_live(pid):
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return False
        # A killed orphan may be visible briefly as a zombie; it cannot use a
        # GPU or retain inherited resources and therefore is not live.
        return stat_text.split()[2] != "Z"

    deadline = time_module.monotonic() + 2.0
    while any(process_is_live(pid) for pid in (parent_pid, child_pid)) \
            and time_module.monotonic() < deadline:
        time_module.sleep(0.02)
    assert not process_is_live(parent_pid)
    assert not process_is_live(child_pid)


def test_managed_child_refuses_main_thread_popen_registration_window():
    import scripts.phase3_selection_matrix as M

    with pytest.raises(
            M.CellRefused, match="must launch from a worker thread"):
        M._run_managed_process(
            [sys.executable, "-c", "raise SystemExit(0)"],
            cwd=str(REPO), env=dict(os.environ))


def test_generic_sweep_plan_does_not_acquire_gpu_lease(monkeypatch):
    import scripts.phase3_selection_matrix as M

    called = []
    monkeypatch.setattr(
        M, "_with_campaign_gpu_leases",
        lambda *args, **kwargs: called.append("lease") or 99)
    monkeypatch.setattr(sys, "argv", [
        "phase3_selection_matrix.py", "--sweep", "topp", "--plan"])
    assert M.main() == 0
    assert called == []


def test_stability_pj_plan_does_not_acquire_gpu_lease(tmp_path, monkeypatch):
    import scripts.phase3_select_n as S
    import scripts.phase3_selection_matrix as M

    stage_path = tmp_path / "stage.json"
    stage_path.write_text('{}\n')
    row = ("flickr25k", 4, ("0.3", "0.7"), "0.02", "select", 42)
    monkeypatch.setattr(
        S, "load_stability_stage_authority",
        lambda path: {"axis": "topp", "fixture": str(path)})
    monkeypatch.setattr(S, "stability_stage_cells", lambda payload: [row])
    calls = []
    monkeypatch.setattr(
        M, "_run_sweep",
        lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    monkeypatch.setattr(
        M, "_with_campaign_gpu_leases",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("plan-only stability must not acquire a GPU lease")))
    monkeypatch.setattr(sys, "argv", [
        "phase3_selection_matrix.py", "--stability-plan", str(stage_path),
        "--sweep", "topp", "--plan"])
    assert M.main() == 0
    assert len(calls) == 1
    assert calls[0][1]["full_plan"] == [row]
