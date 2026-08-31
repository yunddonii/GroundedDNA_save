"""Reducing the matrix to a choice, and refusing to when it cannot be done.

D1: argmax of the raw base-Hamming mAP@R at each candidate's own terminal
epoch, ties to the SMALLEST N, and the chosen N is reused unchanged for seeds
42/43/44. The failure this guards against is a partial matrix choosing N from
whichever cells happened to finish.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.phase3_select_n import (  # noqa: E402
    _recipe_matrix,
    choose_recipe,
    SelectionRefused,
    choose,
    load_matrix,
)
from scripts.phase3_selection_matrix import (  # noqa: E402
    BASES_PER_SLOT,
    CANDIDATE_N,
    LR_HORIZON,
    RECORD_SCHEMA,
    REFIT_SEEDS,
    SEED,
    SLOTS,
    VAL_RATIO,
    VAL_SEED,
    build_command,
    cell_keys,
    protocol_digests,
    refit_tag_for,
)

# The fixture has to write what `run_cell` writes, including the per-dataset
# trainer digest. A fixture that omits it encodes the very gap §54.4 named.
_PROTOCOL = protocol_digests()


def _real_run(root: Path, dataset: str, n: int, value: float, *,
              topp=("0.3", "0.7"), joint="0.0") -> dict:
    """A run directory the aggregator can actually reopen.

    The old fixture wrote `run_dir: "/result/cifar10_N4"` and completion
    digests of repeated characters, so it encoded exactly the gap §54.5 named:
    sixteen records naming directories that do not exist were accepted. The
    aggregator re-hashes the checkpoint and log.csv and re-reads the metric
    from the CSV now, so the fixture has to produce all three.
    """
    import csv as _csv
    from types import SimpleNamespace
    from dna_utils.run_identity import RunIdentity, write_run_manifest
    from scripts.phase3_selection_matrix import DATASETS, _sha

    run = root / f"{dataset}_N{n}_{topp[0]}_{topp[1]}_{joint}"
    run.mkdir(parents=True, exist_ok=True)
    # `args.txt` is what the trainer parsed, and the reducer reads the swept
    # coordinate back out of it -- a record cannot label a run with someone
    # else's coordinate.
    (run / "args.txt").write_text(
        f"routing_adaptive_topp_min{'-' * 20}{topp[0]}\n"
        f"routing_adaptive_topp_max{'-' * 20}{topp[1]}\n"
        f"lambda_codon_joint{'-' * 20}{joint}\n")
    (run / "model_state_dict.pth").write_bytes(f"weights-{dataset}-{n}".encode())
    with open(run / "log.csv", "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(
            handle, fieldnames=["epoch", "eval_mAP_at_R", "eval_mAP_R_cutoff"])
        writer.writeheader()
        for e in range(n + 1):
            writer.writerow({"epoch": e,
                             "eval_mAP_at_R": f"{value if e == n else 0.1:.17g}",
                             "eval_mAP_R_cutoff": "1000"})
    spec = DATASETS[dataset]
    identity = RunIdentity.from_args(SimpleNamespace(
        dataset=spec["canon"], setting="setting1", random_seed=SEED, epoch=60,
        stop_after_epoch=n, num_semantic_parts=SLOTS,
        num_codons_per_codebook=BASES_PER_SLOT, codebook_size=spec["K"],
        selection_mode="select", val_split_ratio=VAL_RATIO,
        val_split_seed=VAL_SEED, lr_schedule_horizon=LR_HORIZON,
        sinkhorn_schedule_horizon=n + 1,
        siglip2_feature_cache_dir=spec["cache"], eval_cache_dir=spec["cache"],
        routing_adaptive_topp=True,
        routing_adaptive_topp_min=float(topp[0]),
        routing_adaptive_topp_max=float(topp[1]),
        lambda_codon_joint=float(joint)))
    write_run_manifest(str(run), identity)
    return {
        "run_dir": str(run),
        "topp": topp, "joint": joint,
        "identity_digest": identity.digest,
        "checkpoint_sha256": _sha(run / "model_state_dict.pth"),
        "log_csv_sha256": _sha(run / "log.csv"),
    }


def _record(dataset: str, n: int, value: float, *, run=None,
            **overrides) -> dict:
    run = run or {}
    base = {
        "schema_version": RECORD_SCHEMA,
        "namespace": "phase3sel",
        "protocol_sources": dict(protocol_digests(dataset)),
        "stage": "select",
        # A record has to carry the evidence a completed cell produces. Sixteen
        # JSONs with none of this, and `protocol_sources: null` throughout,
        # passed the first version of the aggregator.
        "epoch_budget": 60,
        "identity_digest": run.get("identity_digest", "d" * 64),
        "matrix": {"cells": 16},
        "inputs": {"codebook_size": 64},
        "completion": {
            "final_checkpoint": "model_state_dict.pth",
            "final_checkpoint_sha256": run.get("checkpoint_sha256", "a" * 64),
            "log_csv_sha256": run.get("log_csv_sha256", "b" * 64),
            "terminal_weights_preserved": True},
        "dataset": dataset, "N": n,
        "tag": f"phase3sel_{dataset}_N{n}_s42",
        "run_dir": run.get("run_dir", f"/result/{dataset}_N{n}"),
        "seed": SEED, "val_split_ratio": VAL_RATIO, "val_split_seed": VAL_SEED,
        "lr_schedule_horizon": LR_HORIZON,
        "sinkhorn_schedule_horizon": n + 1,
        "geometry": {"num_semantic_parts": SLOTS, "num_codebooks": SLOTS,
                     "num_codons_per_codebook": BASES_PER_SLOT},
        "selection": {"selection_metric": "eval_mAP_at_R",
                      "selection_value": value,
                      "selection_epoch_zero_based": n,
                      "map_r_cutoff": "1000",
                      "log_csv_sha256": run.get("log_csv_sha256", "b" * 64)},
        "is_candidate_cell": True,
    }
    base.update(overrides)
    return base


def _matrix(tmp_path: Path, values=None, per_cell=None) -> Path:
    out = tmp_path / "records"
    out.mkdir(exist_ok=True)
    runs = tmp_path / "runs"
    for dataset, n in cell_keys():
        value = (values or {}).get((dataset, n), 0.5 + 0.001 * n)
        record = _record(dataset, n, value,
                         run=_real_run(runs, dataset, n, value),
                         **(per_cell or {}).get((dataset, n), {}))
        (out / f"{dataset}_N{n}.json").write_text(json.dumps(record))
    return out


def test_a_complete_matrix_yields_one_n_per_dataset(tmp_path):
    matrix = load_matrix(_matrix(tmp_path))
    chosen = choose(matrix["cells"])
    assert sorted(chosen) == ["cifar10", "flickr25k", "mscoco", "nuswide"]
    for entry in chosen.values():
        assert entry["selected_N"] in CANDIDATE_N


def test_argmax_picks_the_best_terminal_value(tmp_path):
    values = {("cifar10", n): v for n, v in zip(CANDIDATE_N,
                                                [0.10, 0.90, 0.20, 0.30])}
    chosen = choose(load_matrix(_matrix(tmp_path, values))["cells"])
    assert chosen["cifar10"]["selected_N"] == 9
    assert chosen["cifar10"]["selection_value"] == pytest.approx(0.90)
    assert chosen["cifar10"]["tie_broken_by_smallest_N"] is False


def test_a_tie_goes_to_the_smallest_n(tmp_path):
    values = {("cifar10", n): 0.77 for n in CANDIDATE_N}
    chosen = choose(load_matrix(_matrix(tmp_path, values))["cells"])
    assert chosen["cifar10"]["selected_N"] == min(CANDIDATE_N) == 4
    assert chosen["cifar10"]["tied_with"] == sorted(CANDIDATE_N)
    assert chosen["cifar10"]["tie_broken_by_smallest_N"] is True


def test_every_candidate_is_reported_not_only_the_winner(tmp_path):
    chosen = choose(load_matrix(_matrix(tmp_path))["cells"])
    assert sorted(int(k) for k in chosen["cifar10"]["all_candidates"]) == \
        sorted(CANDIDATE_N)


# ------------------------------------------------------- what is refused

def test_a_partial_matrix_is_refused(tmp_path):
    """The failure this exists to prevent."""
    records = _matrix(tmp_path)
    (records / "cifar10_N39.json").unlink()
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert "15 of 16" in str(excinfo.value)
    assert "cifar10/N39" in str(excinfo.value)


def test_a_duplicated_cell_is_refused(tmp_path):
    records = _matrix(tmp_path)
    # The duplicate has to be re-derivable too, or it is refused for the wrong
    # reason and the duplicate check is never reached.
    (records / "cifar10_N4_again.json").write_text(json.dumps(_record(
        "cifar10", 4, 0.99,
        run=_real_run(tmp_path / "dup", "cifar10", 4, 0.99))))
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert "recorded twice" in str(excinfo.value)


def test_a_smoke_record_is_refused(tmp_path):
    records = _matrix(tmp_path, None,
                      {("cifar10", 4): {"is_candidate_cell": False}})
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert "not a candidate cell" in str(excinfo.value)


def test_a_value_from_the_wrong_epoch_is_refused(tmp_path):
    """An N=39 cell reporting its epoch-4 score is the collapse this catches."""
    bad = {"selection": {"selection_metric": "eval_mAP_at_R",
                         "selection_value": 0.9,
                         "selection_epoch_zero_based": 4,
                         "map_r_cutoff": "1000",
                         "log_csv_sha256": "b" * 64}}
    records = _matrix(tmp_path, None, {("cifar10", 39): bad})
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert "terminal epoch 39" in str(excinfo.value)


@pytest.mark.parametrize("field,value", [
    ("seed", 43), ("val_split_ratio", 0.2), ("lr_schedule_horizon", 5),
    ("sinkhorn_schedule_horizon", 999)])
def test_a_cell_run_under_a_different_protocol_is_refused(tmp_path, field, value):
    records = _matrix(tmp_path, None, {("cifar10", 4): {field: value}})
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert "protocol differs" in str(excinfo.value)


def test_the_eighteen_base_geometry_is_refused(tmp_path):
    bad = {"geometry": {"num_semantic_parts": 6, "num_codebooks": 6,
                        "num_codons_per_codebook": 3}}
    records = _matrix(tmp_path, None, {("cifar10", 4): bad})
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert "geometry differs" in str(excinfo.value)


def test_two_namespaces_are_refused(tmp_path):
    records = _matrix(tmp_path, None, {("cifar10", 4): {"namespace": "somewhere_else"}})
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert "one matrix, one namespace" in str(excinfo.value)


def test_records_from_different_protocol_sources_are_refused(tmp_path):
    other = dict(_PROTOCOL)
    other["train_siglip2.py"] = "0" * 64
    records = _matrix(tmp_path, None, {("cifar10", 4): {"protocol_sources": other}})
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert "not made the same way" in str(excinfo.value)


# ------------------------------------------------------------- the refit

def test_the_refit_follows_d2s_final_stage():
    """`-e N+1 --stop N`, so all three horizons collapse onto N+1 together."""
    _, env, tag = build_command("cifar10", 4, 0, stage="refit", seed=43)
    extra = env["EXTRA_ARGS"].split()
    assert extra[extra.index("-e") + 1] == "5"
    assert env["STOP_EP"] == "4"
    # Unset, so both fall back to --epoch. That IS the decision.
    assert "--lr_schedule_horizon" not in extra
    assert "--sinkhorn_schedule_horizon" not in extra
    assert extra[extra.index("--random_seed") + 1] == "43"
    assert extra[extra.index("--selection_mode") + 1] == "refit"
    assert tag == refit_tag_for("cifar10", 4, 43)


def test_the_refit_trains_on_the_full_train_and_evaluates_test_once():
    _, env, _ = build_command("cifar10", 4, 0, stage="refit", seed=42)
    assert env["VAL_RATIO"] == "0.0", "nothing is held out once N is chosen"
    assert env["FINAL_EPOCH"] == "1"
    assert env["WHITEN_NPZ"].endswith("text_whiten_trainOnly_localOnly.npz")


def test_the_selection_stage_still_holds_out_and_skips_test():
    _, env, _ = build_command("cifar10", 4, 0, stage="select")
    assert env["VAL_RATIO"] == "0.1"
    assert "FINAL_EPOCH" not in env
    assert env["WHITEN_NPZ"].endswith("text_whiten_optTrain_localOnly.npz")


def test_the_refit_is_twelve_cells():
    assert sorted(REFIT_SEEDS) == [42, 43, 44]
    plan = {(d, s) for d in ("cifar10", "flickr25k", "nuswide", "mscoco")
            for s in REFIT_SEEDS}
    assert len(plan) == 12


def test_one_n_per_dataset_is_reused_across_seeds():
    """Re-deriving N per seed is what the old multiseed queue did."""
    tags = {refit_tag_for("cifar10", 9, seed) for seed in REFIT_SEEDS}
    assert len(tags) == 3
    assert all("_N9_" in t for t in tags), "the same N for every seed"


def test_the_refit_refuses_without_a_selection(tmp_path):
    from scripts.phase3_selection_matrix import CellRefused, _load_selection

    with pytest.raises(CellRefused) as excinfo:
        _load_selection(tmp_path / "nothing.json")
    assert "run scripts/phase3_select_n.py" in str(excinfo.value)


def test_the_refit_refuses_an_out_of_grid_selection(tmp_path):
    from scripts.phase3_selection_matrix import CellRefused, _load_selection

    path = tmp_path / "selected_n.json"
    path.write_text(json.dumps({"selected": {
        d: {"selected_N": 7} for d in
        ("cifar10", "flickr25k", "nuswide", "mscoco")}}))
    with pytest.raises(CellRefused) as excinfo:
        _load_selection(path)
    assert "not one of" in str(excinfo.value)


# ------------------------------------------------ evidence, not shape (§39.2)

def test_a_fabricated_matrix_with_no_evidence_is_refused(tmp_path):
    """Sixteen hand-written JSONs used to be accepted as a completed matrix."""
    records = tmp_path / "records"
    records.mkdir()
    for dataset, n in cell_keys():
        (records / f"{dataset}_N{n}.json").write_text(json.dumps({
            "schema_version": RECORD_SCHEMA, "namespace": "phase3sel",
            "protocol_sources": None, "is_candidate_cell": True,
            "dataset": dataset, "N": n, "seed": SEED,
            "val_split_ratio": VAL_RATIO, "val_split_seed": VAL_SEED,
            "lr_schedule_horizon": LR_HORIZON,
            "sinkhorn_schedule_horizon": n + 1,
            "geometry": {"num_semantic_parts": SLOTS, "num_codebooks": SLOTS,
                         "num_codons_per_codebook": BASES_PER_SLOT},
            "selection": {"selection_metric": "eval_mAP_at_R",
                          "selection_value": 0.9,
                          "selection_epoch_zero_based": n}}))
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert "not a record a completed cell wrote" in str(excinfo.value)


@pytest.mark.parametrize("field", [
    "completion", "inputs", "matrix", "epoch_budget", "run_dir",
    "identity_digest", "protocol_sources"])
def test_a_record_missing_its_evidence_is_refused(tmp_path, field):
    records = _matrix(tmp_path, None, {("cifar10", 4): {field: None}})
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert field in str(excinfo.value)


@pytest.mark.parametrize("field", ["final_checkpoint_sha256", "log_csv_sha256"])
def test_a_completion_without_real_digests_is_refused(tmp_path, field):
    broken = {"final_checkpoint_sha256": "a" * 64, "log_csv_sha256": "b" * 64}
    broken[field] = "not-a-digest"
    records = _matrix(tmp_path, None, {("cifar10", 4): {"completion": broken}})
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert field in str(excinfo.value)


def test_records_agreeing_on_null_protocol_sources_are_refused(tmp_path):
    """Sixteen `null`s are also "one protocol"."""
    records = _matrix(tmp_path)
    for path in records.glob("*.json"):
        payload = json.loads(path.read_text())
        payload["protocol_sources"] = {"scripts/x.py": "0" * 64}
        path.write_text(json.dumps(payload))
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert "different protocol than the one reading them" in str(excinfo.value)


def test_a_refit_record_cannot_be_read_as_a_selection(tmp_path):
    records = _matrix(tmp_path, None, {("cifar10", 4): {"stage": "refit"}})
    with pytest.raises(SelectionRefused) as excinfo:
        load_matrix(records)
    assert "not a selection cell" in str(excinfo.value)


# ------------------------------------------------- the refit CLI (§39.3)

def test_refit_alone_does_not_silently_print_a_plan():
    """`--refit` fell into the default branch and exited 0."""
    import subprocess as _sp

    proc = _sp.run(
        [sys.executable, str(REPO / "scripts" / "phase3_selection_matrix.py"),
         "--refit"],
        capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert proc.returncode == 2
    assert "pass --run" in proc.stderr


def test_only_cannot_narrow_a_production_refit():
    """Three of twelve reported as success would be a matrix nobody completed."""
    import subprocess as _sp

    proc = _sp.run(
        [sys.executable, str(REPO / "scripts" / "phase3_selection_matrix.py"),
         "--run", "--refit", "--only", "cifar10:4"],
        capture_output=True, text=True, cwd=str(REPO), timeout=120)
    assert proc.returncode == 2
    assert "cannot produce the twelve cells" in proc.stderr


def test_the_refit_record_names_the_whitening_it_actually_used():
    """It recorded the optTrain path while the command used trainOnly."""
    from scripts.phase3_selection_matrix import DATASETS, _whitening

    assert _whitening(DATASETS["cifar10"], stage="refit").endswith(
        "text_whiten_trainOnly_localOnly.npz")
    assert _whitening(DATASETS["cifar10"], stage="select").endswith(
        "text_whiten_optTrain_localOnly.npz")


# ---------------------------------------------------------------------------
# §54.4: the trainer shell is the file that hardcodes the top-p window this
# stage sweeps, and it was not in PROTOCOL_SOURCES -- so the one source whose
# edit silently changes what a cell ran was the one the record did not name.
# ---------------------------------------------------------------------------

def test_a_record_naming_a_stale_trainer_is_refused(tmp_path):
    from scripts.phase3_selection_matrix import DATASETS
    records = _matrix(tmp_path)
    victim = next(p for p in records.glob("*.json"))
    payload = json.loads(victim.read_text())
    trainer = DATASETS["cifar10"]["trainer"]
    payload["protocol_sources"][trainer] = "0" * 64
    victim.write_text(json.dumps(payload, indent=2, sort_keys=True))
    with pytest.raises(SelectionRefused) as error:
        load_matrix(records)
    assert "trainer shell" in str(error.value)


def test_records_from_two_trees_are_still_refused(tmp_path):
    """The shared sources must be identical across the whole matrix."""
    records = _matrix(tmp_path)
    victim = next(p for p in records.glob("*.json"))
    payload = json.loads(victim.read_text())
    payload["protocol_sources"]["model_siglip2.py"] = "1" * 64
    victim.write_text(json.dumps(payload, indent=2, sort_keys=True))
    with pytest.raises(SelectionRefused):
        load_matrix(records)


# ---------------------------------------------------------------------------
# §54.5: the record is a report written by the process now being trusted, so
# the aggregator reopens the run and re-derives every claim from the bytes.
# ---------------------------------------------------------------------------

def test_a_record_naming_a_directory_that_does_not_exist_is_refused(tmp_path):
    records = _matrix(tmp_path)
    victim = next(records.glob("*.json"))
    payload = json.loads(victim.read_text())
    payload["run_dir"] = "/result/never_existed"
    victim.write_text(json.dumps(payload))
    with pytest.raises(SelectionRefused) as error:
        load_matrix(records)
    assert "does not exist" in str(error.value)


def test_a_checkpoint_changed_after_the_cell_is_refused(tmp_path):
    records = _matrix(tmp_path)
    victim = next(records.glob("*.json"))
    run = Path(json.loads(victim.read_text())["run_dir"])
    (run / "model_state_dict.pth").write_bytes(b"different weights")
    with pytest.raises(SelectionRefused) as error:
        load_matrix(records)
    assert "does not hash to the digest" in str(error.value)


def test_a_metric_that_disagrees_with_log_csv_is_refused(tmp_path):
    """The number has to be IN the bytes, not merely reported beside them."""
    records = _matrix(tmp_path)
    victim = next(records.glob("*.json"))
    payload = json.loads(victim.read_text())
    payload["selection"]["selection_value"] = 0.999
    victim.write_text(json.dumps(payload))
    with pytest.raises(SelectionRefused) as error:
        load_matrix(records)
    assert "log.csv epoch" in str(error.value)


def test_a_forged_identity_digest_is_refused(tmp_path):
    records = _matrix(tmp_path)
    victim = next(records.glob("*.json"))
    payload = json.loads(victim.read_text())
    payload["identity_digest"] = "f" * 64
    victim.write_text(json.dumps(payload))
    with pytest.raises(SelectionRefused) as error:
        load_matrix(records)
    assert "the run's identity is" in str(error.value)


# ---------------------------------------------------------------------------
# The recipe reduction: declared before the cells run, so the rule cannot be
# chosen after the numbers are seen.
# ---------------------------------------------------------------------------

def _grid():
    from scripts.phase3_selection_matrix import TOPP_GRID, TOPP_INCUMBENT
    return [tuple(c) for c in TOPP_GRID], tuple(TOPP_INCUMBENT)


def test_the_recipe_winner_is_the_argmax(tmp_path):
    grid, inc = _grid()
    cells = dict(zip(((("flickr25k"), c) for c in grid),
                     (0.7576, 0.7608, 0.7590, 0.7605)))
    got = choose_recipe(cells, axis="topp", grid=grid, incumbent=inc, datasets=sorted({d for d, _ in cells}))
    assert got["flickr25k"]["selected"] == list(grid[1])
    assert got["flickr25k"]["delta_vs_incumbent"] == pytest.approx(0.0032)
    assert got["flickr25k"]["tie_broken"] is False


def test_a_tie_keeps_the_incumbent(tmp_path):
    """Not-moving wins a tie, so a coin flip cannot rewrite the recipe."""
    grid, inc = _grid()
    got = choose_recipe({("f", c): 0.5 for c in grid}, axis="topp", grid=grid, incumbent=inc, datasets=sorted({d for d, _ in {("f", c): 0.5 for c in grid}}))
    assert got["f"]["selected"] == list(inc)
    assert got["f"]["tie_broken"] is True


def test_a_tie_without_the_incumbent_takes_the_earlier_coordinate():
    grid, inc = _grid()
    cells = {("f", c): 0.9 for c in grid[1:]}
    cells[("f", grid[0])] = 0.1
    got = choose_recipe(cells, axis="topp", grid=grid, incumbent=inc, datasets=sorted({d for d, _ in cells}))
    assert got["f"]["selected"] == list(grid[1])


def test_a_dataset_missing_one_cell_refuses_the_dataset():
    grid, inc = _grid()
    with pytest.raises(SelectionRefused) as error:
        choose_recipe({("f", grid[0]): 0.5, ("f", grid[1]): 0.6}, axis="topp", grid=grid, incumbent=inc, datasets=sorted({d for d, _ in {("f", grid[0]): 0.5, ("f", grid[1]): 0.6}}))
    assert "partial matrix" in str(error.value)


def test_a_coordinate_off_the_declared_grid_is_refused():
    grid, inc = _grid()
    cells = {("f", c): 0.5 for c in grid}
    cells[("f", ("0.9", "0.99"))] = 0.99
    with pytest.raises(SelectionRefused) as error:
        choose_recipe(cells, axis="topp", grid=grid, incumbent=inc, datasets=sorted({d for d, _ in cells}))
    assert "not in the declared" in str(error.value)


@pytest.mark.parametrize("bad", [float("nan"), 1.5, -0.1, True, "0.5", None])
def test_a_nonfinite_or_out_of_range_value_refuses(bad):
    grid, inc = _grid()
    cells = {("f", c): 0.5 for c in grid}
    cells[("f", grid[0])] = bad
    with pytest.raises(SelectionRefused):
        choose_recipe(cells, axis="topp", grid=grid, incumbent=inc, datasets=sorted({d for d, _ in cells}))


def test_the_lambda_grid_reduces_without_its_incumbent(tmp_path):
    """§56.3: the lambda grid is the five positive values.

    Its zero control comes from the top-p stage, so `values[incumbent]` raised
    KeyError('0.0') and the whole lambda reduction was unrunnable.
    """
    from scripts.phase3_selection_matrix import JOINT_GRID, JOINT_INCUMBENT
    grid = list(JOINT_GRID)
    cells = {("f", j): 0.5 + 0.01 * i for i, j in enumerate(grid)}
    got = choose_recipe(cells, axis="joint", grid=grid, incumbent=JOINT_INCUMBENT, datasets=sorted({d for d, _ in cells}))["f"]
    assert got["selected"] == grid[-1]
    assert got["incumbent_on_grid"] is False
    assert got["delta_vs_incumbent"] is None


def test_cells_from_several_plan_snapshots_are_not_one_experiment(tmp_path):
    """§57.2: nine `--only` processes each took their own snapshot."""
    from scripts.phase3_select_n import _sha

    def rebaseline(receipt, records_dir):
        """One cell ran against its own snapshot, and the receipt says so.

        Editing the record alone is caught earlier, by the receipt's own
        digest -- so the receipt is kept honest about the edited bytes, which
        is exactly what nine independent processes would have produced.
        """
        key = sorted(receipt["cells"])[1]
        entry = receipt["cells"][key]
        path = records_dir / entry["record"]
        payload = json.loads(path.read_text())
        payload["plan_snapshot_sha256"] = "9" * 64
        path.write_text(json.dumps(payload))
        entry["record_sha256"] = _sha(path)

    records = _sweep(tmp_path, seal=rebaseline)
    with pytest.raises(SelectionRefused) as error:
        _recipe_matrix(records, axis="topp", namespace="phase3sweep")
    assert "different plan snapshots" in str(error.value)


def test_the_recipe_reducer_ignores_the_n_matrix_records(tmp_path):
    """§57.3: reading every candidate record made the sixteen N cells fatal."""
    records = _sweep(tmp_path)
    for dataset, n in cell_keys():          # the sixteen N-selection records
        (records / f"phase3sel_{dataset}_N{n}.json").write_text(json.dumps(
            _record(dataset, n, 0.5, run=_real_run(tmp_path / "nm", dataset, n, 0.5))))
    from scripts.phase3_selection_matrix import TOPP_GRID
    matrix = _recipe_matrix(records, axis="topp", namespace="phase3sweep")
    from scripts.phase3_selection_matrix import TOPP_SWEEP_DATASETS
    assert len(matrix["cells"]) == len(TOPP_GRID) * len(TOPP_SWEEP_DATASETS)
    assert matrix["datasets"] == sorted(TOPP_SWEEP_DATASETS)


# ---------------------------------------------------------------------------
# A sweep as production writes it: records, a content-addressed snapshot whose
# `plan` names the exact cells, and a receipt sealing every record by name and
# by digest. The reducer reopens all three.
# ---------------------------------------------------------------------------

def _sweep(tmp_path: Path, *, namespace="phase3sweep", axis="topp",
           values=None, seal=None, plan=None):
    """A sweep exactly as production writes it, over the CANONICAL plan.

    The first version of this helper built a Flickr-only sweep, which sealed the
    very defect it was meant to guard: a snapshot naming one dataset was
    accepted as a complete matrix. The plan comes from the source now, so a test
    that wants an incomplete sweep has to break it on purpose.
    """
    import hashlib as _h
    from scripts.phase3_select_n import _sha
    from scripts.phase3_selection_matrix import canonical_plan, plan_snapshot

    plan = plan if plan is not None else canonical_plan(axis)
    datasets = sorted({c[0] for c in plan})
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    snapshot = plan_snapshot(datasets, plan=plan, executed=plan, axis=axis,
                             namespace=namespace)
    digest = _h.sha256(json.dumps(snapshot, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()
    snap_name = f"{namespace}_snapshot_{digest[:16]}.json"
    (records / snap_name).write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    sealed = {}
    for i, (ds, n, c, jd) in enumerate(plan):
        value = (values or {}).get((ds, tuple(c)), 0.50 + 0.001 * i)
        run = _real_run(tmp_path / "runs", ds, n, value, topp=tuple(c),
                        joint=jd)
        rec = _record(ds, n, value, run=run)
        rec["namespace"] = namespace
        rec["recipe"] = {"routing_adaptive_topp_min": c[0],
                         "routing_adaptive_topp_max": c[1],
                         "lambda_codon_joint": jd}
        rec["geometry"] = {"identity_digest": run["identity_digest"]}
        rec["plan_snapshot_sha256"] = digest
        name = f"{namespace}_{ds}_{c[0]}_{c[1]}_{jd}.json"
        (records / name).write_text(json.dumps(rec))
        sealed[f"{ds}/{tuple(c)}/{jd}"] = {
            "tag": rec["tag"], "run_dir": run["run_dir"],
            "identity_digest": run["identity_digest"],
            "record": name, "record_sha256": _sha(records / name),
            "recipe": rec["recipe"]}
    receipt = {"schema_version": 2, "axis": axis, "namespace": namespace,
               "plan_snapshot_sha256": digest, "plan_snapshot_file": snap_name,
               "expected_cells": len(plan), "declared_cells": len(plan),
               "cell_count": len(sealed), "cells": sealed}
    if seal:
        seal(receipt, records)
    (records / f"{namespace}_sweep_complete.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return records


# ---------------------------------------------------------------------------
# The four counterexamples re-audit §61 ran against the committed bytes of
# 00ec75d. Every one of them was accepted there, in a 1096-passing suite.
# ---------------------------------------------------------------------------

def _reseal(records, receipt=None):
    """Recompute the digest chain, so a forgery is self-consistent."""
    import hashlib as _h
    from scripts.phase3_select_n import _sha
    snap = next(records.glob("*_snapshot_*.json"))
    payload = json.loads(snap.read_text())
    digest = _h.sha256(json.dumps(payload, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()
    new_snap = records / f"{snap.name.split('_snapshot_')[0]}_snapshot_{digest[:16]}.json"
    snap.rename(new_snap)
    r_path = next(records.glob("*_sweep_complete.json"))
    r = receipt if receipt is not None else json.loads(r_path.read_text())
    r["plan_snapshot_sha256"] = digest
    r["plan_snapshot_file"] = new_snap.name
    for entry in r["cells"].values():
        rec_path = records / entry["record"]
        rec = json.loads(rec_path.read_text())
        rec["plan_snapshot_sha256"] = digest
        rec_path.write_text(json.dumps(rec))
        entry["record_sha256"] = _sha(rec_path)
    r_path.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")


def test_a_snapshot_that_declares_a_smaller_sweep_is_refused(tmp_path):
    """§61-1: a Flickr-only snapshot called itself a complete matrix."""
    from scripts.phase3_selection_matrix import canonical_plan
    plan = [c for c in canonical_plan("topp") if c[0] == "flickr25k"]
    records = _sweep(tmp_path, plan=plan)
    _reseal(records)
    with pytest.raises(SelectionRefused) as error:
        _recipe_matrix(records, axis="topp", namespace="phase3sweep")
    assert "declares a plan this tree does not" in str(error.value)


def test_a_snapshot_whose_coordinates_were_rewritten_is_refused(tmp_path):
    """§61-2: all four windows rewritten to 0.3/0.7, digest chain re-sealed."""
    records = _sweep(tmp_path)
    snap = next(records.glob("*_snapshot_*.json"))
    payload = json.loads(snap.read_text())
    for cell in payload["plan"]["declared_cells"]:
        cell["topp"] = ["0.3", "0.7"]
    snap.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _reseal(records)
    with pytest.raises(SelectionRefused) as error:
        _recipe_matrix(records, axis="topp", namespace="phase3sweep")
    assert "declares a plan this tree does not" in str(error.value)


def test_a_forged_receipt_entry_is_refused(tmp_path):
    """§61-3: tag, run_dir, identity and recipe replaced by foreign values."""
    def forge(receipt, records_dir):
        entry = receipt["cells"][sorted(receipt["cells"])[0]]
        entry["tag"] = "FOREIGN"
        entry["run_dir"] = "/tmp/foreign"
        entry["identity_digest"] = "f" * 64
        entry["recipe"] = {"routing_adaptive_topp_min": "0.9",
                           "routing_adaptive_topp_max": "0.99",
                           "lambda_codon_joint": "9.0"}
    records = _sweep(tmp_path, seal=forge)
    with pytest.raises(SelectionRefused) as error:
        _recipe_matrix(records, axis="topp", namespace="phase3sweep")
    assert "the record says" in str(error.value)


def test_rotating_args_and_labels_together_is_refused(tmp_path):
    """§61-4, the one that survived the previous fix.

    `args.txt` is a plain text file that nothing hashes, so rotating it in step
    with the record labels defeated a check that compared only those two. The
    run manifest's digest is verified against its own fields, and schema 4
    carries the recipe axes, so the manifest is the witness that cannot be
    rotated without breaking its own seal.
    """
    from scripts.phase3_select_n import _sha
    from scripts.phase3_selection_matrix import TOPP_GRID

    records = _sweep(tmp_path)
    grid = [tuple(c) for c in TOPP_GRID]
    receipt_path = next(records.glob("*_sweep_complete.json"))
    receipt = json.loads(receipt_path.read_text())
    for key, entry in receipt["cells"].items():
        if not key.startswith("flickr25k/"):
            continue
        path = records / entry["record"]
        rec = json.loads(path.read_text())
        here = (rec["recipe"]["routing_adaptive_topp_min"],
                rec["recipe"]["routing_adaptive_topp_max"])
        nxt = grid[(grid.index(here) + 1) % len(grid)]
        rec["recipe"]["routing_adaptive_topp_min"] = nxt[0]
        rec["recipe"]["routing_adaptive_topp_max"] = nxt[1]
        path.write_text(json.dumps(rec))
        entry["recipe"] = rec["recipe"]
        entry["record_sha256"] = _sha(path)
        # ...and args.txt rotated to agree, which is what defeated the old check
        args = Path(rec["run_dir"]) / "args.txt"
        args.write_text(
            f"routing_adaptive_topp_min{'-' * 20}{nxt[0]}\n"
            f"routing_adaptive_topp_max{'-' * 20}{nxt[1]}\n"
            f"lambda_codon_joint{'-' * 20}0.0\n")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    with pytest.raises(SelectionRefused) as error:
        _recipe_matrix(records, axis="topp", namespace="phase3sweep")
    assert "sealed identity says" in str(error.value)


def test_a_snapshot_declaring_more_than_it_executed_is_refused(tmp_path):
    """The smoke's own inconsistency: snapshot said 12, receipt said 1."""
    from scripts.phase3_selection_matrix import canonical_plan
    records = _sweep(tmp_path)
    snap = next(records.glob("*_snapshot_*.json"))
    payload = json.loads(snap.read_text())
    payload["plan"]["executed_cells"] = payload["plan"]["declared_cells"][:1]
    payload["plan"]["executed_count"] = 1
    snap.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _reseal(records)
    with pytest.raises(SelectionRefused) as error:
        _recipe_matrix(records, axis="topp", namespace="phase3sweep")
    assert "of its" in str(error.value) and "declared cells" in str(error.value)
