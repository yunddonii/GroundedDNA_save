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
    load_matrix as load_campaign_matrix,
    load_unsealed_record_matrix as load_matrix,
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


@pytest.fixture(autouse=True)
def _production_source_authority_fixture(monkeypatch):
    """Reducer fixtures model a clean committed production checkout.

    The shared development worktree is intentionally dirty while these tests
    run.  Preserve the real path set/digests while making the fixture's
    HEAD/worktree relationship clean and its pre-import handshake explicit.
    Dedicated tests below still exercise both refusal branches directly.
    """
    import copy
    import scripts.phase3_selection_matrix as M

    real_bundle = M._source_authority_bundle

    def clean_bundle(paths=M._BOOTSTRAP_SOURCE_PATHS):
        bundle = real_bundle(paths)
        for state in bundle["entries"].values():
            state["tracked"] = True
            state["head_sha256"] = state["worktree_sha256"]
            state["head_blob_oid"] = "fixture-clean-head"
            state["clean"] = True
        return bundle

    bootstrap = clean_bundle(M._BOOTSTRAP_SOURCE_PATHS)
    monkeypatch.setattr(M, "_source_authority_bundle", clean_bundle)
    monkeypatch.setattr(M, "_BOOTSTRAP_PREIMPORT_BUNDLE",
                        copy.deepcopy(bootstrap))
    monkeypatch.setattr(M, "_BOOTSTRAP_PREIMPORT_VERIFIED", True)

# The fixture has to write what `run_cell` writes, including the per-dataset
# trainer digest. A fixture that omits it encodes the very gap §54.4 named.
_PROTOCOL = protocol_digests()


def _stability_decisions(*, topp=("0.4", "0.8"), joint="0.03", n=9):
    from scripts.phase3_selection_matrix import DATASETS, TOPP_SWEEP_DATASETS
    return {
        "topp": {dataset: list(topp) for dataset in TOPP_SWEEP_DATASETS},
        "joint": {dataset: joint for dataset in DATASETS},
        "n": {dataset: n for dataset in DATASETS},
    }


def test_recipe_stability_honest_fixed_point_and_explicit_cifar_pin():
    from scripts.phase3_select_n import (
        PINNED_CIFAR_TOPP_POLICY, run_recipe_stability)
    boot = _stability_decisions()
    result = run_recipe_stability(
        boot, [{"topp": boot["topp"], "joint": boot["joint"]}],
        max_update_rounds=3)
    assert result["confirmed"] is True
    assert result["update_rounds"] == 0
    assert result["final_state"]["cifar10"]["topp"] == ["0.6", "0.95"]
    assert result["pinned_policy"] == PINNED_CIFAR_TOPP_POLICY


def test_recipe_stability_p_or_j_drift_requires_n_rerun():
    from scripts.phase3_select_n import run_recipe_stability
    boot = _stability_decisions()
    drift_p = {**boot["topp"], "flickr25k": ["0.5", "0.9"]}
    with pytest.raises(SelectionRefused, match="exact-16 N matrix must rerun"):
        run_recipe_stability(
            boot, [{"topp": drift_p, "joint": boot["joint"]}],
            max_update_rounds=3)

    n2 = {dataset: 19 for dataset in boot["n"]}
    result = run_recipe_stability(
        boot,
        [{"topp": drift_p, "joint": boot["joint"], "n": n2},
         {"topp": drift_p, "joint": boot["joint"]}],
        max_update_rounds=3)
    assert result["confirmed"] is True
    assert result["update_rounds"] == 1
    assert all(entry["N"] == 19 for entry in result["final_state"].values())


def test_recipe_stability_cycle_and_max_update_never_publish_winner():
    from scripts.phase3_select_n import run_recipe_stability
    boot = _stability_decisions()
    p2 = {dataset: ["0.5", "0.9"] for dataset in boot["topp"]}
    j2 = {dataset: "0.05" for dataset in boot["joint"]}
    n2 = {dataset: 19 for dataset in boot["n"]}
    with pytest.raises(SelectionRefused, match="cycles back"):
        run_recipe_stability(
            boot,
            [{"topp": p2, "joint": j2, "n": n2},
             {"topp": boot["topp"], "joint": boot["joint"],
              "n": boot["n"]}],
            max_update_rounds=3)

    p3 = {dataset: ["0.6", "0.95"] for dataset in boot["topp"]}
    j3 = {dataset: "0.07" for dataset in boot["joint"]}
    with pytest.raises(SelectionRefused, match="exceeds max_update_rounds=1"):
        run_recipe_stability(
            boot,
            [{"topp": p2, "joint": j2, "n": n2},
             {"topp": p3, "joint": j3, "n": boot["n"]}],
            max_update_rounds=1)


def test_recipe_stability_refuses_a_fake_cifar_p_candidate_and_jd_zero():
    from scripts.phase3_select_n import run_recipe_stability
    boot = _stability_decisions()
    boot["topp"]["cifar10"] = ["0.6", "0.95"]
    with pytest.raises(SelectionRefused, match="topp decisions must cover exactly"):
        run_recipe_stability(boot, [], max_update_rounds=3)
    boot = _stability_decisions()
    boot["joint"]["cifar10"] = "0.0"
    with pytest.raises(SelectionRefused, match="positive grid"):
        run_recipe_stability(boot, [], max_update_rounds=3)


def test_confirmation_plans_use_each_datasets_current_n_p_and_positive_jd():
    from scripts.phase3_select_n import (
        initial_recipe_state, stability_joint_cells, stability_topp_cells)
    from scripts.phase3_selection_matrix import JOINT_GRID, TOPP_GRID
    state = initial_recipe_state()
    for i, dataset in enumerate(sorted(state)):
        state[dataset]["N"] = CANDIDATE_N[i]
    state["flickr25k"]["joint"] = "0.07"
    p_rows = stability_topp_cells(state)
    assert all(n == state[dataset]["N"] and joint == state[dataset]["joint"]
               for dataset, n, _, joint in p_rows)
    assert {tuple(topp) for _, _, topp, _ in p_rows} == set(TOPP_GRID)
    j_rows = stability_joint_cells(state)
    assert all(n == state[dataset]["N"] and list(topp) == state[dataset]["topp"]
               for dataset, n, topp, _ in j_rows)
    assert {joint for _, _, _, joint in j_rows} == set(JOINT_GRID)
    assert "0.0" not in {joint for _, _, _, joint in j_rows}


def test_honest_stability_producer_is_reachable_by_recipe_consumer(
        tmp_path, monkeypatch):
    """GPU-free E2E: producer bytes are exactly what N/refit reopens."""
    import scripts.phase3_select_n as S
    import scripts.phase3_selection_matrix as M

    boot = _stability_decisions(topp=("0.6", "0.95"), joint="0.10", n=19)

    def fake_ref(ref, *, phase, round_index, state, max_update_rounds):
        del ref, max_update_rounds
        if phase.endswith("topp"):
            decisions = boot["topp"]
        elif phase.endswith("joint"):
            decisions = boot["joint"]
        else:
            decisions = boot["n"]
        return decisions, {
            "phase": phase, "round_index": round_index,
            "receipt_sha256": "1" * 64,
            "plan_snapshot_sha256": "2" * 64,
            "record_sha256": {"fixture.json": "3" * 64},
            "input_seals": {"fixture": {"hf_runtime_sha256": "4" * 64}},
            "environment_sha256": "5" * 64,
            "source_authority_sha256": "6" * 64,
            "decisions": decisions,
        }

    monkeypatch.setattr(S, "_stability_ref", fake_ref)
    ref = {"records_dir": "/sealed", "namespace": "fixture",
           "stage_plan": "/sealed/stage.json"}
    spec = {
        "schema_version": S.RECIPE_STABILITY_SCHEMA,
        "artifact_kind": "phase3_recipe_stability_evidence",
        "max_update_rounds": 3,
        "bootstrap": {"topp": ref, "joint": ref, "n": ref},
        "rounds": [{"topp": ref, "joint": ref}],
    }
    artifact = S.build_recipe_stability_artifact(spec)
    path = tmp_path / "stable_recipe.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    authority = M.load_recipe_authority([path])
    choices = M.verify_recipe_authority(authority)
    assert choices == artifact["selected"]
    assert choices["cifar10"]["topp"] == ["0.6", "0.95"]


@pytest.mark.parametrize("tamper", ("dataset", "seal", "hf", "environment"))
def test_stable_recipe_refuses_permutation_or_stale_runtime_authority(
        tmp_path, monkeypatch, tamper):
    import copy
    import scripts.phase3_select_n as S
    import scripts.phase3_selection_matrix as M

    boot = _stability_decisions()

    def fake_ref(ref, *, phase, round_index, state, max_update_rounds):
        del ref, max_update_rounds
        decisions = (boot["topp"] if phase.endswith("topp") else
                     boot["joint"] if phase.endswith("joint") else boot["n"])
        return decisions, {
            "phase": phase, "round_index": round_index,
            "receipt_sha256": "1" * 64,
            "plan_snapshot_sha256": "2" * 64,
            "record_sha256": {"fixture.json": "3" * 64},
            "input_seals": {"fixture": {
                "seal_sha256": "4" * 64,
                "hf_runtime_sha256": "5" * 64}},
            "environment_sha256": "6" * 64,
            "source_authority_sha256": "7" * 64,
            "decisions": decisions,
        }

    monkeypatch.setattr(S, "_stability_ref", fake_ref)
    ref = {"records_dir": "/sealed", "namespace": "fixture",
           "stage_plan": "/sealed/stage.json"}
    spec = {"schema_version": 1,
            "artifact_kind": "phase3_recipe_stability_evidence",
            "max_update_rounds": 3,
            "bootstrap": {"topp": ref, "joint": ref, "n": ref},
            "rounds": [{"topp": ref, "joint": ref}]}
    artifact = S.build_recipe_stability_artifact(spec)
    forged = copy.deepcopy(artifact)
    if tamper == "dataset":
        forged["selected"]["flickr25k"], forged["selected"]["nuswide"] = (
            forged["selected"]["nuswide"], forged["selected"]["flickr25k"])
        # Make the permutation observable even when the honest winners match.
        forged["selected"]["flickr25k"]["joint"] = "0.07"
    elif tamper == "seal":
        forged["resolved_evidence"]["bootstrap"]["topp"][
            "input_seals"]["fixture"]["seal_sha256"] = "8" * 64
    elif tamper == "hf":
        forged["resolved_evidence"]["bootstrap"]["topp"][
            "input_seals"]["fixture"]["hf_runtime_sha256"] = "8" * 64
    else:
        forged["resolved_evidence"]["bootstrap"]["topp"][
            "environment_sha256"] = "8" * 64
    path = tmp_path / f"forged_{tamper}.json"
    path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n")
    with pytest.raises(M.CellRefused, match="differs from reopened"):
        M._load_recipe(path)


def test_stage_plan_is_exclusive_current_state_authority(tmp_path):
    import scripts.phase3_select_n as S
    state = S.initial_recipe_state()
    payload = S.build_stability_stage_plan(
        phase="bootstrap_topp", state=state, round_index=0,
        max_update_rounds=3)
    path = tmp_path / "p0.json"
    S._publish_json_exclusive(path, payload)
    authority = S.load_stability_stage_authority(
        path, expected_phase="bootstrap_topp", expected_state=state,
        expected_round=0, expected_max_update_rounds=3)
    assert authority["sha256"] == S._sha(path)
    assert len(S.stability_stage_cells(payload)) == 12
    with pytest.raises(SelectionRefused, match="will not be replaced"):
        S._publish_json_exclusive(path, payload)


@pytest.fixture(autouse=True)
def _synthetic_input_seal_stats(monkeypatch):
    """Synthetic authority tests do not touch the production 400 GB caches."""
    import scripts.phase3_selection_matrix as M
    monkeypatch.setattr(M, "verify_snapshot_input_seals",
                        lambda snapshot, **kwargs: None)


def _fake_input_authorities(plan):
    import scripts.phase3_selection_matrix as M
    result = {}
    for cell in plan:
        dataset, _, _, _, stage, _ = M._campaign_cell_parts(cell)
        seal_stage = "refit" if stage == "refit" else "stage1"
        key = f"{dataset}:{seal_stage}"
        if key in result:
            continue
        tokens = {name: "26" * 32 for name in (
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
            "merges.txt", "special_tokens_map.json")}
        result[key] = {
            "schema": "groundeddna.phase3-input-authority", "schema_version": 1,
            "seal_path": f"/fixture/{dataset}-{seal_stage}.seal.json",
            "seal_file_sha256": "20" * 32, "aggregate_sha256": "21" * 32,
            "dataset": dataset, "stage": seal_stage, "request": {},
            "split_identity_sha256": "22" * 32, "split_identity": {},
            "authority_sha256": "23" * 32,
            "hf_runtime": {
                "checkpoint": M.PHASE3_CLIP_CHECKPOINT,
                "revision": M.PHASE3_CLIP_REVISION,
                "snapshot_dir": "/fixture/hf/snapshots/" + M.PHASE3_CLIP_REVISION,
                "weight_file": "pytorch_model.bin",
                "weight_sha256": M.PHASE3_CLIP_WEIGHT_SHA256,
                "config_file": "config.json", "config_sha256": "24" * 32,
                "tokenizer_files_sha256": tokens,
                "tokenizer_set_sha256": "25" * 32,
                "local_files_only": True, "identity_sha256": "27" * 32,
            },
        }
    return result


def _real_run(root: Path, dataset: str, n: int, value: float, *,
              topp=("0.3", "0.7"), joint="0.0", tag=None,
              campaign=None, stage="select", seed=SEED,
              input_authority=None) -> dict:
    """A run directory the aggregator can actually reopen.

    The old fixture wrote `run_dir: "/result/cifar10_N4"` and completion
    digests of repeated characters, so it encoded exactly the gap §54.5 named:
    sixteen records naming directories that do not exist were accepted. The
    aggregator re-hashes the checkpoint and log.csv and re-reads the metric
    from the CSV now, so the fixture has to produce all three.
    """
    import csv as _csv
    from dna_utils.run_identity import RunIdentity, write_run_manifest
    from scripts.phase3_selection_matrix import (
        DATASETS, _sha, expected_run_identity)

    # The directory carries its own tag, exactly as the trainer names it: the
    # reducer requires that, so a rotated coordinate cannot keep its directory.
    run = root / (f"260831+{dataset}_setting1_{tag}+bs+64"
                  if tag else f"{dataset}_N{n}_{topp[0]}_{topp[1]}_{joint}")
    run.mkdir(parents=True, exist_ok=True)
    # `args.txt` is what the trainer parsed, and the reducer reads the swept
    # coordinate back out of it -- a record cannot label a run with someone
    # else's coordinate.
    (run / "args.txt").write_text(
        f"routing_adaptive_topp_min{'-' * 20}{topp[0]}\n"
        f"routing_adaptive_topp_max{'-' * 20}{topp[1]}\n"
        f"lambda_codon_joint{'-' * 20}{joint}\n")
    (run / "model_state_dict.pth").write_bytes(
        f"weights-{dataset}-{n}-{stage}-{seed}".encode())
    with open(run / "log.csv", "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(
            handle, fieldnames=["epoch", "eval_mAP_at_R", "eval_mAP_R_cutoff"])
        writer.writeheader()
        for e in range(n + 1):
            writer.writerow({"epoch": e,
                             "eval_mAP_at_R": f"{value if e == n else 0.1:.17g}",
                             "eval_mAP_R_cutoff": "1000"})
    identity = expected_run_identity(
        dataset, n, topp=topp, joint=joint, stage=stage, seed=seed,
        input_authority=input_authority)
    write_run_manifest(str(run), identity)
    out = {
        "run_dir": str(run),
        "topp": topp, "joint": joint,
        "identity_digest": identity.digest,
        "run_identity_sha256": _sha(run / "run_identity.json"),
        "checkpoint_sha256": _sha(run / "model_state_dict.pth"),
        "log_csv_sha256": _sha(run / "log.csv"),
    }
    if campaign is not None:
        import torch
        from dna_utils.run_identity import (
            PHASE3_CAMPAIGN_BINDING_NAME, bind_phase3_campaign_to_state_dict)
        from dna_utils.runtime_state import write_checkpoint_metadata

        evidence = {
            "schema_version": 1, **campaign,
            "actual_identity_digest": identity.digest,
            "actual_child_environment": campaign[
                "expected_child_environment"],
            "trainer_pid": 4242, "trainer_boot_id": "fixture-boot-id",
        }
        evidence_path = run / PHASE3_CAMPAIGN_BINDING_NAME
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        # The checkpoint itself, not only mutable JSON beside it, commits to
        # the cell binding. A one-parameter state dict is sufficient here and
        # keeps the test GPU/trainer-free.
        state = torch.nn.Linear(1, 1).state_dict()
        bind_phase3_campaign_to_state_dict(state, evidence)
        torch.save(state, run / "model_state_dict.pth")
        out["checkpoint_sha256"] = _sha(run / "model_state_dict.pth")
        write_checkpoint_metadata(
            str(run / "model_state_dict.pth"),
            checkpoint_epoch_zero_based=n,
            training_epoch_budget=identity.epoch_budget,
            stop_after_epoch=n,
            lr_schedule_horizon=identity.lr_schedule_horizon,
            sinkhorn_schedule_horizon=identity.sinkhorn_schedule_horizon,
            sinkhorn_epsilon_init=identity.sinkhorn_epsilon_init,
            sinkhorn_epsilon_final=identity.sinkhorn_epsilon_final,
            lr_scheduler="cosine",
            extra={
                "tag": campaign["expected_tag"],
                "dataset": identity.dataset,
                "random_seed": identity.seed,
                "num_semantic_parts": identity.num_slots,
                "num_codons_per_codebook": identity.bases_per_slot,
                "phase3_campaign": evidence,
            })
        out.update(
            checkpoint_runtime_sha256=_sha(
                Path(str(run / "model_state_dict.pth") + ".runtime.json")),
            phase3_campaign_evidence_sha256=_sha(evidence_path),
            args_txt_sha256=_sha(run / "args.txt"),
            phase3_campaign=evidence,
        )
    return out


def _write_refit_outputs(run_dir: Path, *, dataset: str, n: int,
                         seed: int) -> dict:
    """Write tiny production-schema extraction/evaluation evidence."""
    import numpy as np
    from dna_utils.extraction_validation import base_indices_to_2bit
    from dna_utils.runtime_state import (
        ResolvedEpoch, sha256_file, write_extraction_manifest)
    from scripts.phase3_selection_matrix import DATASETS, _sha

    checkpoint = run_dir / "model_state_dict.pth"
    config = run_dir / "config.pt"
    config.write_bytes(b"fixture-config")
    manifests = {}
    for i, split in enumerate(("db", "query", "train")):
        bases = np.tile(
            np.asarray([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]],
                       dtype=np.int64), (2, 1))
        bases[1] = np.roll(bases[1], i + 1)
        npz = run_dir / f"extract_{split}.npz"
        np.savez(
            npz, base_indices=bases,
            hash_2bit=base_indices_to_2bit(bases),
            codebook_indices=np.asarray([[0, 1, 0, 1, 0],
                                         [1, 0, 1, 0, 1]], dtype=np.int64),
            labels=np.asarray([0, 1], dtype=np.int64))
        manifest = run_dir / f"extraction_manifest_{split}.json"
        write_extraction_manifest(
            str(manifest), checkpoint_path=str(checkpoint),
            resolved=ResolvedEpoch(
                epoch=n, source="checkpoint_metadata",
                effective_sinkhorn_epsilon=0.1,
                sinkhorn_schedule_horizon=n + 1,
                checkpoint_sha256=sha256_file(str(checkpoint)),
                sinkhorn_annealing_enabled=True),
            num_slots=SLOTS, bases_per_slot=BASES_PER_SLOT,
            split=split, n_rows=2, lr_schedule_horizon=n + 1,
            training_epoch_budget=n + 1, training_stop_epoch=n,
            extra={
                "config_path": str(config),
                "config_sha256": sha256_file(str(config)),
                "dataset": DATASETS[dataset]["canon"],
                "random_seed": seed,
                "codebook_size": DATASETS[dataset]["K"],
                "backfilled": False, "npz_path": str(npz),
                "npz_sha256": sha256_file(str(npz)),
            })
        manifests[split] = manifest
    marker = {
        "schema_version": 1, "splits": sorted(manifests),
        "manifest_sha256": {
            split: _sha(path) for split, path in sorted(manifests.items())},
    }
    marker_path = run_dir / "extraction_complete.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    from evaluation_siglip2 import evaluation, resolve_map_at_r
    canonical = DATASETS[dataset]["canon"]
    cutoff = resolve_map_at_r(canonical)
    evaluation(str(run_dir), distance_mode="base",
               codebook_size=DATASETS[dataset]["K"], map_at_r=cutoff,
               dataset_name=canonical)
    post = evaluation(
        str(run_dir), distance_mode="base",
        codebook_size=DATASETS[dataset]["K"], map_at_r=cutoff,
        dataset_name=canonical, bio_project=True,
        bio_gc_min_frac=0.4, bio_gc_max_frac=0.6,
        bio_max_homopolymer_run=3)
    from scripts.eval_cell_bioproj import _read_json_artifact
    _, raw_artifact = _read_json_artifact(
        str(run_dir / "evaluation_siglip2_base.json"))
    _, post_artifact = _read_json_artifact(
        str(run_dir / "evaluation_siglip2_base_bioproj.json"))
    from dna_utils.extraction_validation import metric_input_binding
    binding = metric_input_binding(
        str(run_dir), required_splits=("db", "query", "train"),
        allow_backfilled=False)
    from dna_utils.gc_policy import resolve_gc_policy
    policy = resolve_gc_policy(SLOTS * BASES_PER_SLOT)
    cell_result = {
        "input_binding": binding, "dataset": canonical,
        "evaluation_artifacts": {
            "raw": raw_artifact, "bio_projected": post_artifact},
        "evaluator_input_artifacts": post["input_artifacts"],
        "bio_stats": post["bio_stats"],
        "K": DATASETS[dataset]["K"],
        "mAP_at_R_bioproj": post["mAP_at_R"],
        "full_mAP_bioproj": post["mAP"],
        "full_mAP_pre_projection": post["bio_stats"]["mAP_pre_projection"],
        "DNA_unique_DB": post["unique_code_ratio"],
        "gc_min_frac": 0.4, "gc_max_frac": 0.6,
        "total_bases": SLOTS * BASES_PER_SLOT,
        "gc_count_min_inclusive": policy.gc_min_count,
        "gc_count_max_inclusive": policy.gc_max_count,
        "bio_max_homopolymer_run": policy.max_run,
        "gc_policy_version": policy.policy_version,
        "map_r_cutoff": cutoff,
    }
    (run_dir / "cell_result.json").write_text(
        json.dumps(cell_result, indent=2, sort_keys=True) + "\n")
    from scripts.pairwise_nmi import _record_for_result
    nmi_record, _, _ = _record_for_result(
        str(run_dir), allow_backfilled=False,
        required_splits=("db", "query", "train"))
    (run_dir / "pairwise_nmi.json").write_text(
        json.dumps(nmi_record, indent=2, sort_keys=True) + "\n")
    from scripts.seal_cell_analysis import seal
    seal(run_dir, allow_backfilled=False,
         required_splits=("db", "query", "train"))
    from scripts.phase3_selection_matrix import assert_refit_outputs
    return assert_refit_outputs(run_dir, dataset=dataset)


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
            "final_checkpoint_epoch_zero_based": n,
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


def _n_campaign(tmp_path: Path, monkeypatch, *, values=None,
                namespace="phase3n") -> tuple[Path, dict]:
    """Exact-16 latest-schema N authority with no trainer/GPU invocation."""
    from scripts.phase3_select_n import _sha
    import scripts.phase3_selection_matrix as M

    choices = {dataset: {"topp": ["0.3", "0.7"], "joint": "0.01"}
               for dataset in sorted(M.DATASETS)}
    # Stability is a separate blocked protocol. These tests exercise the N
    # campaign/reducer below that gate with an injected already-verified value.
    monkeypatch.setattr(M, "verify_recipe_authority", lambda authority: choices)
    authority = {"schema_version": 999, "fixture": "already-verified"}
    plan = M.n_selection_cells(choices)
    input_seals = _fake_input_authorities(plan)
    nonce = "cd" * 32
    records = tmp_path / "n-records"
    records.mkdir()
    result_root = (tmp_path / "n-runs").resolve()
    snapshot = M.plan_snapshot(
        sorted(M.DATASETS), plan=plan, executed=plan, axis="n",
        namespace=namespace, campaign_nonce=nonce,
        campaign_kind="n_selection", authorities={"recipe": authority},
        result_root=result_root, input_seals=input_seals)
    digest = M._json_digest(snapshot)
    snap_name = f"{namespace}_snapshot_{digest[:16]}.json"
    (records / snap_name).write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    sealed = {}
    for dataset, n, topp, joint, stage, seed in plan:
        value = (values or {}).get((dataset, n), 0.5 + 0.001 * n)
        cell_id = M.campaign_cell_id(
            dataset, n, topp=topp, joint=joint, stage=stage, seed=seed)
        binding = snapshot["plan"]["cell_bindings"][cell_id]
        campaign = M.launch_binding_from_expected(binding, digest)
        input_authority = input_seals[f"{dataset}:stage1"]
        run = _real_run(
            result_root, dataset, n, value, topp=tuple(topp),
            joint=joint, tag=binding["expected_tag"], campaign=campaign,
            input_authority=input_authority)
        rec = _record(dataset, n, value, run=run)
        rec.update(namespace=namespace, tag=binding["expected_tag"],
                   plan_snapshot_sha256=digest, campaign=campaign,
                   result_root=str(result_root),
                   environment_sha256=snapshot["environment_sha256"],
                   input_authority=input_authority)
        rec["recipe"] = {
            "routing_adaptive_topp_min": topp[0],
            "routing_adaptive_topp_max": topp[1],
            "lambda_codon_joint": joint,
        }
        rec["geometry"]["identity_digest"] = run["identity_digest"]
        rec["completion"].update({
            "run_identity_sha256": run["run_identity_sha256"],
            "args_txt_sha256": run["args_txt_sha256"],
            "checkpoint_runtime_sha256": run["checkpoint_runtime_sha256"],
            "phase3_campaign_evidence_sha256":
                run["phase3_campaign_evidence_sha256"],
            "phase3_campaign": run["phase3_campaign"],
        })
        name = f"{rec['tag']}.json"
        path = records / name
        path.write_text(json.dumps(rec))
        sealed[cell_id] = {
            "dataset": dataset, "N": n, "seed": seed, "stage": stage,
            "tag": rec["tag"], "run_dir": rec["run_dir"],
            "identity_digest": run["identity_digest"], "record": name,
            "record_sha256": _sha(path), "recipe": rec["recipe"],
            "campaign": campaign, "completion": rec["completion"],
        }

    reservation_name = f"{namespace}{M.CAMPAIGN_RESERVATION_SUFFIX}"
    reservation = {
        "schema_version": M.CAMPAIGN_RESERVATION_SCHEMA,
        "namespace": namespace, "owner_pid": 4242,
        "owner_boot_id": "fixture-boot", "owner_host": "fixture",
        "campaign_nonce": nonce, "plan_digest": digest,
        "plan_snapshot_file": snap_name, "campaign_kind": "n_selection",
        "result_root": str(result_root),
        "qwen_root": snapshot["qwen_root"],
        "source_authority_sha256": snapshot["source_authority_sha256"],
        "environment_sha256": snapshot["environment_sha256"],
        "input_seals_sha256": M._json_digest(input_seals),
        "head_commit": snapshot["source_authority"]["head_commit"],
        "declared_cells": 16, "executed_cells": 16,
    }
    reservation_path = records / reservation_name
    reservation_path.write_text(
        json.dumps(reservation, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": M.SWEEP_RECEIPT_SCHEMA, "axis": "n",
        "campaign_kind": "n_selection", "namespace": namespace,
        "campaign_nonce": nonce,
        "result_root": str(result_root),
        "qwen_root": snapshot["qwen_root"],
        "source_authority_sha256": snapshot["source_authority_sha256"],
        "environment_sha256": snapshot["environment_sha256"],
        "input_seals": input_seals,
        "input_seals_sha256": M._json_digest(input_seals),
        "campaign_reservation_file": reservation_name,
        "campaign_reservation_sha256": _sha(reservation_path),
        "plan_snapshot_sha256": digest, "plan_snapshot_file": snap_name,
        "expected_cells": 16, "declared_cells": 16, "cell_count": 16,
        "cells": sealed,
    }
    receipt_path = records / f"{namespace}{M.N_SELECTION_RECEIPT_SUFFIX}"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return records, choices


def test_exact_16_campaign_recomputes_n_from_receipt_only(
        tmp_path, monkeypatch):
    records, _ = _n_campaign(tmp_path, monkeypatch)
    # An unrelated candidate-looking JSON is outside exact receipt membership
    # and therefore cannot contaminate the reduction.
    (records / "unrelated.json").write_text(json.dumps({
        "is_candidate_cell": True, "dataset": "invented", "N": 999}))
    matrix = load_campaign_matrix(records, namespace="phase3n")
    selected = choose(matrix["cells"])
    assert len(matrix["cells"]) == 16
    assert all(entry["selected_N"] == 39 for entry in selected.values())


@pytest.mark.parametrize("tamper", ("preimport", "dirty", "untracked"))
def test_reducer_requires_preimport_and_clean_tracked_source_authority(tamper):
    import copy
    import scripts.phase3_selection_matrix as M
    from scripts.phase3_select_n import _verify_production_source_admission

    authority = M._source_authority_bundle(M._BOOTSTRAP_SOURCE_PATHS)
    digest = M._semantic_digest(authority)
    snapshot = {
        "source_authority": authority,
        "source_authority_sha256": digest,
        "preimport_source_authority_sha256": digest,
    }
    broken = copy.deepcopy(snapshot)
    if tamper == "preimport":
        broken["preimport_source_authority_sha256"] = None
    else:
        state = next(iter(broken["source_authority"]["entries"].values()))
        state["clean" if tamper == "dirty" else "tracked"] = False
        broken["source_authority_sha256"] = M._semantic_digest(
            broken["source_authority"])
        broken["preimport_source_authority_sha256"] = \
            broken["source_authority_sha256"]
    with pytest.raises(SelectionRefused, match=(
            "pre-import" if tamper == "preimport" else "untracked or dirty")):
        _verify_production_source_admission(broken, "fixture-snapshot")


@pytest.mark.parametrize("mutation", ("missing", "extra", "duplicate"))
def test_exact_16_receipt_refuses_missing_extra_or_duplicate_membership(
        tmp_path, monkeypatch, mutation):
    import scripts.phase3_selection_matrix as M

    records, _ = _n_campaign(tmp_path, monkeypatch)
    receipt_path = records / f"phase3n{M.N_SELECTION_RECEIPT_SUFFIX}"
    receipt = json.loads(receipt_path.read_text())
    keys = sorted(receipt["cells"])
    if mutation == "missing":
        receipt["cells"].pop(keys[0])
    elif mutation == "extra":
        receipt["cells"]["invented|N=999"] = dict(receipt["cells"][keys[0]])
    else:
        receipt["cells"][keys[1]]["record"] = \
            receipt["cells"][keys[0]]["record"]
        receipt["cells"][keys[1]]["record_sha256"] = \
            receipt["cells"][keys[0]]["record_sha256"]
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SelectionRefused):
        load_campaign_matrix(records, namespace="phase3n")


def _selected_n_artifact(records: Path, out: Path) -> Path:
    from scripts.phase3_select_n import (
        N_SELECTION_REDUCTION, SELECTED_N_SCHEMA, _sha)

    matrix = load_campaign_matrix(records, namespace="phase3n")
    payload = {
        "schema_version": SELECTED_N_SCHEMA,
        "artifact_kind": "phase3_selected_n",
        "reduction": N_SELECTION_REDUCTION,
        "namespace": matrix["namespace"],
        "records_dir": matrix["records_dir"],
        "protocol_sources": matrix["protocol_sources"],
        "aggregator_sha256": _sha(REPO / "scripts" / "phase3_select_n.py"),
        "receipt_file": matrix["receipt_file"],
        "receipt_sha256": matrix["receipt_sha256"],
        "plan_snapshot_file": matrix["plan_snapshot_file"],
        "plan_snapshot_sha256": matrix["plan_snapshot_sha256"],
        "recipe_authority": matrix["recipe_authority"],
        "record_sha256": matrix["record_sha256"],
        "selected": choose(matrix["cells"]),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def test_selected_n_reopens_all_16_and_rejects_forged_or_stale_choice(
        tmp_path, monkeypatch):
    from scripts.phase3_select_n import verify_selection_artifact

    records, _ = _n_campaign(tmp_path, monkeypatch)
    selected = _selected_n_artifact(records, tmp_path / "selected_n.json")
    assert verify_selection_artifact(selected) == {
        dataset: 39 for dataset in
        ("cifar10", "flickr25k", "mscoco", "nuswide")}

    payload = json.loads(selected.read_text())
    payload["selected"]["cifar10"]["selected_N"] = 4
    selected.write_text(json.dumps(payload))
    with pytest.raises(SelectionRefused, match="selected N is not"):
        verify_selection_artifact(selected)


def test_selected_n_rejects_stale_source_seal(tmp_path, monkeypatch):
    from scripts.phase3_select_n import verify_selection_artifact

    records, _ = _n_campaign(tmp_path, monkeypatch)
    selected = _selected_n_artifact(records, tmp_path / "selected_n.json")
    payload = json.loads(selected.read_text())
    payload["protocol_sources"]["train_siglip2.py"] = "0" * 64
    selected.write_text(json.dumps(payload))
    with pytest.raises(SelectionRefused, match="source seal is stale"):
        verify_selection_artifact(selected)


def test_n_full_16_bundle_permutation_is_refused_by_checkpoint_binding(
        tmp_path, monkeypatch):
    """All mutable evidence is re-signed; checkpoint metadata stays physical."""
    from dna_utils.run_identity import (
        PHASE3_CAMPAIGN_BINDING_NAME, write_run_manifest)
    from scripts.phase3_select_n import _sha
    import scripts.phase3_selection_matrix as M

    records, choices = _n_campaign(tmp_path, monkeypatch)
    receipt_path = records / f"phase3n{M.N_SELECTION_RECEIPT_SUFFIX}"
    receipt = json.loads(receipt_path.read_text())
    snapshot = json.loads(next(records.glob("*_snapshot_*.json")).read_text())
    datasets = sorted(M.DATASETS)
    bundles = []
    for old_key, entry in list(receipt["cells"].items()):
        path = records / entry["record"]
        rec = json.loads(path.read_text())
        target_ds = datasets[(datasets.index(rec["dataset"]) + 1) % len(datasets)]
        bundles.append((old_key, entry, path, rec, target_ds))
    assert len(bundles) == 16
    for i, (_, _, path, rec, _) in enumerate(bundles):
        run = Path(rec["run_dir"])
        tmp_run = run.parent / f"n-cycle-run-{i}"
        tmp_record = records / f"n-cycle-record-{i}.json"
        run.rename(tmp_run)
        path.rename(tmp_record)
        rec["_tmp_run"], rec["_tmp_record"] = str(tmp_run), str(tmp_record)

    new_entries = {}
    for _, old_entry, _, rec, target_ds in bundles:
        tmp_run = Path(rec.pop("_tmp_run"))
        tmp_record = Path(rec.pop("_tmp_record"))
        n = rec["N"]
        topp = tuple(choices[target_ds]["topp"])
        joint = choices[target_ds]["joint"]
        target_cell = M.campaign_cell_id(
            target_ds, n, topp=topp, joint=joint, stage="select", seed=SEED)
        planned = snapshot["plan"]["cell_bindings"][target_cell]
        target_tag = planned["expected_tag"]
        target_run = tmp_run.parent / (
            f"260831+{target_ds}_setting1_{target_tag}+bs+64")
        tmp_run.rename(target_run)
        identity = M.expected_run_identity(
            target_ds, n, topp=topp, joint=joint, stage="select",
            input_authority=snapshot["input_seals"][f"{target_ds}:stage1"])
        assert identity.digest == planned["expected_identity_digest"]
        write_run_manifest(str(target_run), identity)
        campaign = M.launch_binding_from_expected(
            planned, receipt["plan_snapshot_sha256"])
        evidence_path = target_run / PHASE3_CAMPAIGN_BINDING_NAME
        evidence = json.loads(evidence_path.read_text())
        evidence.update(campaign)
        evidence["actual_identity_digest"] = identity.digest
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        runtime_path = Path(str(target_run / "model_state_dict.pth") +
                            ".runtime.json")
        runtime = json.loads(runtime_path.read_text())
        runtime["checkpoint_path"] = str(
            (target_run / "model_state_dict.pth").resolve())
        runtime["extra"].update({
            "tag": target_tag,
            "dataset": identity.dataset,
            "random_seed": identity.seed,
            "num_semantic_parts": identity.num_slots,
            "num_codons_per_codebook": identity.bases_per_slot,
        })
        runtime.setdefault("extra", {})["phase3_campaign"] = evidence
        runtime_path.write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        rec.update(dataset=target_ds, tag=target_tag, run_dir=str(target_run),
                   identity_digest=identity.digest, campaign=campaign,
                   protocol_sources=M.protocol_digests(target_ds),
                   input_authority=snapshot["input_seals"][
                       f"{target_ds}:stage1"])
        rec["geometry"]["identity_digest"] = identity.digest
        rec["completion"].update({
            "run_identity_sha256": _sha(target_run / "run_identity.json"),
            "phase3_campaign_evidence_sha256": _sha(evidence_path),
            "checkpoint_runtime_sha256": _sha(runtime_path),
        })
        target_name = f"{target_tag}.json"
        target_path = records / target_name
        tmp_record.write_text(json.dumps(rec))
        tmp_record.rename(target_path)
        new_entries[target_cell] = {
            **old_entry, "dataset": target_ds, "N": n, "seed": SEED,
            "stage": "select", "tag": target_tag,
            "run_dir": str(target_run), "identity_digest": identity.digest,
            "record": target_name, "record_sha256": _sha(target_path),
            "recipe": rec["recipe"], "campaign": campaign,
            "completion": rec["completion"],
        }
    receipt["cells"] = new_entries
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SelectionRefused,
                       match=("checkpoint serialization commits another|"
                              "runtime metadata is not the exact terminal")):
        load_campaign_matrix(records, namespace="phase3n")


def _refit_campaign(tmp_path: Path, monkeypatch,
                    *, namespace="phase3refit") -> tuple[Path, dict, Path]:
    """Exact-12 latest-schema refit authority, with tiny valid outputs."""
    from scripts.phase3_select_n import _sha
    import scripts.phase3_select_n as S
    import scripts.phase3_selection_matrix as M

    monkeypatch.setattr(
        M, "PAPER_SPLIT_ROWS",
        {dataset: {"db": 2, "query": 2, "train": 2}
         for dataset in M.DATASETS})

    choices = {dataset: {"topp": ["0.3", "0.7"], "joint": "0.01"}
               for dataset in sorted(M.DATASETS)}
    selected = {dataset: 4 for dataset in sorted(M.DATASETS)}
    monkeypatch.setattr(M, "verify_recipe_authority", lambda authority: choices)
    selected_path = tmp_path / "selected-authority.json"
    selected_path.write_text("fixture selected authority\n")
    monkeypatch.setattr(S, "verify_selection_artifact",
                        lambda path: dict(selected))
    recipe_authority = {"schema_version": 999, "fixture": "verified-recipe"}
    selection_authority = {
        "schema_version": 1, "path": str(selected_path),
        "sha256": _sha(selected_path), "selected_n": selected,
    }
    plan = M.refit_cells(selected, choices)
    input_seals = _fake_input_authorities(plan)
    nonce = "ef" * 32
    records = tmp_path / "refit-records"
    records.mkdir()
    result_root = (tmp_path / "refit-runs").resolve()
    snapshot = M.plan_snapshot(
        sorted(M.DATASETS), plan=plan, executed=plan, axis="refit",
        namespace=namespace, campaign_nonce=nonce, campaign_kind="refit",
        authorities={"recipe": recipe_authority,
                     "selected_n": selection_authority},
        result_root=result_root, input_seals=input_seals)
    digest = M._json_digest(snapshot)
    snap_name = f"{namespace}_snapshot_{digest[:16]}.json"
    (records / snap_name).write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    sealed = {}
    for dataset, n, topp, joint, stage, seed in plan:
        cell_id = M.campaign_cell_id(
            dataset, n, topp=topp, joint=joint, stage=stage, seed=seed)
        binding = snapshot["plan"]["cell_bindings"][cell_id]
        campaign = M.launch_binding_from_expected(binding, digest)
        input_authority = input_seals[f"{dataset}:refit"]
        run = _real_run(
            result_root, dataset, n, 0.5,
            topp=tuple(topp), joint=joint, tag=binding["expected_tag"],
            campaign=campaign, stage="refit", seed=seed,
            input_authority=input_authority)
        outputs = _write_refit_outputs(
            Path(run["run_dir"]), dataset=dataset, n=n, seed=seed)
        rec = _record(dataset, n, 0.5, run=run)
        rec.update(
            namespace=namespace, tag=binding["expected_tag"], stage="refit",
            seed=seed, val_split_ratio=0.0, selection_mode="refit",
            epoch_budget=n + 1, lr_schedule_horizon=n + 1,
            sinkhorn_schedule_horizon=n + 1,
            plan_snapshot_sha256=digest, campaign=campaign,
            result_root=str(result_root),
            environment_sha256=snapshot["environment_sha256"],
            input_authority=input_authority,
            protocol_sources=M.protocol_digests(dataset),
            selection={"selection_metric": None,
                       "note": "scratch refit has no selection"})
        rec["recipe"] = {
            "routing_adaptive_topp_min": topp[0],
            "routing_adaptive_topp_max": topp[1],
            "lambda_codon_joint": joint,
        }
        rec["geometry"]["identity_digest"] = run["identity_digest"]
        rec["identity_digest"] = run["identity_digest"]
        rec["completion"].update({
            "run_identity_sha256": run["run_identity_sha256"],
            "args_txt_sha256": run["args_txt_sha256"],
            "checkpoint_runtime_sha256": run["checkpoint_runtime_sha256"],
            "phase3_campaign_evidence_sha256":
                run["phase3_campaign_evidence_sha256"],
            "phase3_campaign": run["phase3_campaign"], **outputs,
        })
        name = f"{rec['tag']}.json"
        path = records / name
        path.write_text(json.dumps(rec))
        sealed[cell_id] = {
            "dataset": dataset, "N": n, "seed": seed, "stage": stage,
            "tag": rec["tag"], "run_dir": rec["run_dir"],
            "identity_digest": run["identity_digest"], "record": name,
            "record_sha256": _sha(path), "recipe": rec["recipe"],
            "campaign": campaign, "completion": rec["completion"],
        }
    reservation_name = f"{namespace}{M.CAMPAIGN_RESERVATION_SUFFIX}"
    reservation = {
        "schema_version": M.CAMPAIGN_RESERVATION_SCHEMA,
        "namespace": namespace, "owner_pid": 4242,
        "owner_boot_id": "fixture-boot", "owner_host": "fixture",
        "campaign_nonce": nonce, "plan_digest": digest,
        "plan_snapshot_file": snap_name, "campaign_kind": "refit",
        "result_root": str(result_root),
        "qwen_root": snapshot["qwen_root"],
        "source_authority_sha256": snapshot["source_authority_sha256"],
        "environment_sha256": snapshot["environment_sha256"],
        "input_seals_sha256": M._json_digest(input_seals),
        "head_commit": snapshot["source_authority"]["head_commit"],
        "declared_cells": 12, "executed_cells": 12,
    }
    reservation_path = records / reservation_name
    reservation_path.write_text(
        json.dumps(reservation, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": M.SWEEP_RECEIPT_SCHEMA, "axis": "refit",
        "campaign_kind": "refit", "namespace": namespace,
        "campaign_nonce": nonce,
        "result_root": str(result_root),
        "qwen_root": snapshot["qwen_root"],
        "source_authority_sha256": snapshot["source_authority_sha256"],
        "environment_sha256": snapshot["environment_sha256"],
        "input_seals": input_seals,
        "input_seals_sha256": M._json_digest(input_seals),
        "campaign_reservation_file": reservation_name,
        "campaign_reservation_sha256": _sha(reservation_path),
        "plan_snapshot_sha256": digest, "plan_snapshot_file": snap_name,
        "expected_cells": 12, "declared_cells": 12, "cell_count": 12,
        "cells": sealed,
    }
    (records / f"{namespace}{M.REFIT_RECEIPT_SUFFIX}").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return records, selected, selected_path


def test_exact_12_refit_reopens_checkpoint_and_final_output_identity(
        tmp_path, monkeypatch):
    from scripts.phase3_select_n import verify_refit_campaign

    records, _, _ = _refit_campaign(tmp_path, monkeypatch)
    verified = verify_refit_campaign(records, namespace="phase3refit")
    assert len(verified["record_sha256"]) == 12
    for evidence in verified["datasets"].values():
        assert len(set(evidence["final_checkpoint_sha256"].values())) == 3


def test_refit_aggregate_refuses_identical_checkpoint_sha_across_seeds(
        tmp_path, monkeypatch):
    import scripts.phase3_select_n as S
    import scripts.phase3_selection_matrix as M

    records, _, _ = _refit_campaign(tmp_path, monkeypatch)
    receipt_path = records / f"phase3refit{M.REFIT_RECEIPT_SUFFIX}"
    receipt = json.loads(receipt_path.read_text())
    cifar_entries = sorted(
        (entry for entry in receipt["cells"].values()
         if entry["dataset"] == "cifar10"),
        key=lambda entry: entry["seed"])
    duplicated = cifar_entries[0]["completion"]["final_checkpoint_sha256"]
    victim = cifar_entries[1]
    record_path = records / victim["record"]
    record = json.loads(record_path.read_text())
    record["completion"]["final_checkpoint_sha256"] = duplicated
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    victim["completion"]["final_checkpoint_sha256"] = duplicated
    victim["record_sha256"] = S._sha(record_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # Isolate the aggregate-level acceptance gate.  Normal verification would
    # already reject the copied digest against the actual checkpoint bytes.
    monkeypatch.setattr(S, "_reverify_run", lambda *args, **kwargs: None)
    with pytest.raises(SelectionRefused, match="distinct terminal checkpoint"):
        S.verify_refit_campaign(records, namespace="phase3refit")


def test_refit_refuses_selected_n_authority_changed_after_launch(
        tmp_path, monkeypatch):
    from scripts.phase3_select_n import verify_refit_campaign

    records, _, selected_path = _refit_campaign(tmp_path, monkeypatch)
    selected_path.write_text("stale selected authority bytes\n")
    with pytest.raises(SelectionRefused,
                       match="selected-N authority is absent or changed"):
        verify_refit_campaign(records, namespace="phase3refit")


def test_refit_full_12_mutable_bundle_permutation_is_refused(
        tmp_path, monkeypatch):
    """Rotate seeds and re-sign every JSON/output manifest around the weights."""
    from dna_utils.run_identity import (
        PHASE3_CAMPAIGN_BINDING_NAME, write_run_manifest)
    from scripts.phase3_select_n import _sha, verify_refit_campaign
    import scripts.phase3_selection_matrix as M

    records, _, _ = _refit_campaign(tmp_path, monkeypatch)
    receipt_path = records / f"phase3refit{M.REFIT_RECEIPT_SUFFIX}"
    receipt = json.loads(receipt_path.read_text())
    snapshot = json.loads(next(records.glob("*_snapshot_*.json")).read_text())
    seeds = list(M.REFIT_SEEDS)
    bundles = []
    for key, entry in list(receipt["cells"].items()):
        path = records / entry["record"]
        rec = json.loads(path.read_text())
        target_seed = seeds[(seeds.index(rec["seed"]) + 1) % len(seeds)]
        bundles.append((key, entry, path, rec, target_seed))
    assert len(bundles) == 12
    for i, (_, _, path, rec, _) in enumerate(bundles):
        run = Path(rec["run_dir"])
        tmp_run = run.parent / f"refit-cycle-run-{i}"
        tmp_record = records / f"refit-cycle-record-{i}.json"
        run.rename(tmp_run)
        path.rename(tmp_record)
        rec["_tmp_run"], rec["_tmp_record"] = str(tmp_run), str(tmp_record)

    new_entries = {}
    for _, old_entry, _, rec, target_seed in bundles:
        tmp_run = Path(rec.pop("_tmp_run"))
        tmp_record = Path(rec.pop("_tmp_record"))
        dataset, n = rec["dataset"], rec["N"]
        topp = (rec["recipe"]["routing_adaptive_topp_min"],
                rec["recipe"]["routing_adaptive_topp_max"])
        joint = rec["recipe"]["lambda_codon_joint"]
        target_cell = M.campaign_cell_id(
            dataset, n, topp=topp, joint=joint,
            stage="refit", seed=target_seed)
        planned = snapshot["plan"]["cell_bindings"][target_cell]
        target_tag = planned["expected_tag"]
        target_run = tmp_run.parent / (
            f"260831+{dataset}_setting1_{target_tag}+bs+64")
        tmp_run.rename(target_run)
        identity = M.expected_run_identity(
            dataset, n, topp=topp, joint=joint,
            stage="refit", seed=target_seed,
            input_authority=snapshot["input_seals"][f"{dataset}:refit"])
        write_run_manifest(str(target_run), identity)
        (target_run / "args.txt").write_text(
            f"routing_adaptive_topp_min{'-' * 20}{topp[0]}\n"
            f"routing_adaptive_topp_max{'-' * 20}{topp[1]}\n"
            f"lambda_codon_joint{'-' * 20}{joint}\n"
            f"random_seed{'-' * 20}{target_seed}\n")
        campaign = M.launch_binding_from_expected(
            planned, receipt["plan_snapshot_sha256"])
        evidence_path = target_run / PHASE3_CAMPAIGN_BINDING_NAME
        evidence = json.loads(evidence_path.read_text())
        evidence.update(campaign)
        evidence["actual_identity_digest"] = identity.digest
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        runtime_path = Path(str(target_run / "model_state_dict.pth") +
                            ".runtime.json")
        runtime = json.loads(runtime_path.read_text())
        runtime["checkpoint_path"] = str(
            (target_run / "model_state_dict.pth").resolve())
        runtime["extra"].update({
            "tag": target_tag,
            "dataset": identity.dataset,
            "random_seed": target_seed,
            "num_semantic_parts": identity.num_slots,
            "num_codons_per_codebook": identity.bases_per_slot,
        })
        runtime.setdefault("extra", {})["phase3_campaign"] = evidence
        runtime_path.write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n")

        manifest_hashes = {}
        for split in ("db", "query", "train"):
            path = target_run / f"extraction_manifest_{split}.json"
            manifest = json.loads(path.read_text())
            manifest.update(
                random_seed=target_seed,
                checkpoint_path=str(target_run / "model_state_dict.pth"),
                config_path=str(target_run / "config.pt"),
                npz_path=str(target_run / f"extract_{split}.npz"))
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            manifest_hashes[split] = _sha(path)
        marker_path = target_run / "extraction_complete.json"
        marker = json.loads(marker_path.read_text())
        marker["manifest_sha256"] = manifest_hashes
        marker_path.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n")

        rec.update(seed=target_seed, tag=target_tag, run_dir=str(target_run),
                   identity_digest=identity.digest, campaign=campaign,
                   input_authority=snapshot["input_seals"][
                       f"{dataset}:refit"])
        rec["geometry"]["identity_digest"] = identity.digest
        rec["completion"].update({
            "run_identity_sha256": _sha(target_run / "run_identity.json"),
            "args_txt_sha256": _sha(target_run / "args.txt"),
            "phase3_campaign_evidence_sha256": _sha(evidence_path),
            "checkpoint_runtime_sha256": _sha(runtime_path),
            "extraction_manifest_sha256": manifest_hashes,
            "extraction_complete_sha256": _sha(marker_path),
        })
        target_name = f"{target_tag}.json"
        target_path = records / target_name
        tmp_record.write_text(json.dumps(rec))
        tmp_record.rename(target_path)
        new_entries[target_cell] = {
            **old_entry, "seed": target_seed, "tag": target_tag,
            "run_dir": str(target_run), "identity_digest": identity.digest,
            "record": target_name, "record_sha256": _sha(target_path),
            "campaign": campaign, "completion": rec["completion"],
        }
    receipt["cells"] = new_entries
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SelectionRefused,
                       match="checkpoint serialization does not commit"):
        verify_refit_campaign(records, namespace="phase3refit")


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
    assert "protocol sources differ from this tree" in str(excinfo.value)


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
    assert "not a schema-2 selected-N authority" in str(excinfo.value)


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
    assert "protocol sources differ from this tree" in str(excinfo.value)


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
    assert "protocol sources differ from this tree" in str(error.value)


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
    from scripts.phase3_selection_matrix import (
        CAMPAIGN_RESERVATION_SCHEMA, CAMPAIGN_RESERVATION_SUFFIX,
        SWEEP_RECEIPT_SCHEMA, _json_digest, campaign_cell_id, canonical_plan,
        launch_binding_from_expected, plan_snapshot)

    plan = plan if plan is not None else canonical_plan(axis)
    input_seals = _fake_input_authorities(plan)
    nonce = "ab" * 32
    datasets = sorted({c[0] for c in plan})
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    result_root = (tmp_path / "runs").resolve()
    snapshot = plan_snapshot(
        datasets, plan=plan, executed=plan, axis=axis, namespace=namespace,
        campaign_nonce=nonce, result_root=result_root,
        input_seals=input_seals)
    digest = _h.sha256(json.dumps(snapshot, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()
    snap_name = f"{namespace}_snapshot_{digest[:16]}.json"
    (records / snap_name).write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    sealed = {}
    from scripts.phase3_selection_matrix import tag_for
    for i, (ds, n, c, jd) in enumerate(plan):
        value = (values or {}).get((ds, tuple(c)), 0.50 + 0.001 * i)
        tag = tag_for(ds, n, namespace=namespace, topp=tuple(c), joint=jd)
        cell_id = campaign_cell_id(ds, n, topp=tuple(c), joint=jd)
        planned = snapshot["plan"]["cell_bindings"][cell_id]
        campaign = launch_binding_from_expected(planned, digest)
        input_authority = input_seals[f"{ds}:stage1"]
        run = _real_run(result_root, ds, n, value, topp=tuple(c),
                        joint=jd, tag=tag, campaign=campaign,
                        input_authority=input_authority)
        assert run["identity_digest"] == planned["expected_identity_digest"]
        rec = _record(ds, n, value, run=run)
        rec["tag"] = tag
        rec["namespace"] = namespace
        rec["recipe"] = {"routing_adaptive_topp_min": c[0],
                         "routing_adaptive_topp_max": c[1],
                         "lambda_codon_joint": jd}
        rec["geometry"] = {"identity_digest": run["identity_digest"]}
        rec["plan_snapshot_sha256"] = digest
        rec["campaign"] = campaign
        rec["result_root"] = str(result_root)
        rec["environment_sha256"] = snapshot["environment_sha256"]
        rec["input_authority"] = input_authority
        rec["completion"].update({
            "run_identity_sha256": run["run_identity_sha256"],
            "args_txt_sha256": run["args_txt_sha256"],
            "checkpoint_runtime_sha256": run["checkpoint_runtime_sha256"],
            "phase3_campaign_evidence_sha256":
                run["phase3_campaign_evidence_sha256"],
            "phase3_campaign": run["phase3_campaign"],
        })
        name = f"{namespace}_{ds}_{c[0]}_{c[1]}_{jd}.json"
        (records / name).write_text(json.dumps(rec))
        sealed[cell_id] = {
            "tag": rec["tag"], "run_dir": run["run_dir"],
            "identity_digest": run["identity_digest"],
            "record": name, "record_sha256": _sha(records / name),
            "recipe": rec["recipe"], "campaign": campaign,
            "completion": {
                field: rec["completion"].get(field) for field in (
                    "final_checkpoint_sha256", "log_csv_sha256",
                    "run_identity_sha256",
                    "args_txt_sha256", "checkpoint_runtime_sha256",
                    "phase3_campaign_evidence_sha256")}}
    reservation_name = f"{namespace}{CAMPAIGN_RESERVATION_SUFFIX}"
    reservation = {
        "schema_version": CAMPAIGN_RESERVATION_SCHEMA,
        "namespace": namespace, "owner_pid": 12345,
        "owner_boot_id": "fixture-owner-boot-id", "owner_host": "fixture",
        "campaign_nonce": nonce, "plan_digest": digest,
        "plan_snapshot_file": snap_name,
        "campaign_kind": f"recipe_{axis}",
        "result_root": str(result_root),
        "qwen_root": snapshot["qwen_root"],
        "source_authority_sha256": snapshot["source_authority_sha256"],
        "environment_sha256": snapshot["environment_sha256"],
        "input_seals_sha256": _json_digest(input_seals),
        "head_commit": snapshot["source_authority"]["head_commit"],
        "declared_cells": len(plan), "executed_cells": len(plan),
    }
    reservation_path = records / reservation_name
    reservation_path.write_text(
        json.dumps(reservation, indent=2, sort_keys=True) + "\n")
    receipt = {"schema_version": SWEEP_RECEIPT_SCHEMA,
               "axis": axis, "campaign_kind": f"recipe_{axis}",
               "namespace": namespace,
               "campaign_nonce": nonce,
               "result_root": str(result_root),
               "qwen_root": snapshot["qwen_root"],
               "source_authority_sha256": snapshot["source_authority_sha256"],
               "environment_sha256": snapshot["environment_sha256"],
               "input_seals": input_seals,
               "input_seals_sha256": _json_digest(input_seals),
               "campaign_reservation_file": reservation_name,
               "campaign_reservation_sha256": _sha(reservation_path),
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
    reservation_path = records / r["campaign_reservation_file"]
    reservation = json.loads(reservation_path.read_text())
    reservation["plan_digest"] = digest
    reservation["plan_snapshot_file"] = new_snap.name
    reservation_path.write_text(
        json.dumps(reservation, indent=2, sort_keys=True) + "\n")
    r["campaign_reservation_sha256"] = _sha(reservation_path)
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

    The current completion evidence also hashes ``args.txt``.  Rotating it with
    the record must therefore fail before (or at) the manifest comparison.
    """
    from scripts.phase3_select_n import _sha
    from scripts.phase3_selection_matrix import TOPP_GRID

    records = _sweep(tmp_path)
    grid = [tuple(c) for c in TOPP_GRID]
    receipt_path = next(records.glob("*_sweep_complete.json"))
    receipt = json.loads(receipt_path.read_text())
    for key, entry in receipt["cells"].items():
        if not key.startswith("flickr25k|"):
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
    assert "args.txt does not hash" in str(error.value) or \
        "sealed identity says" in str(error.value) or \
        "completion differs from reopened" in str(error.value)


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


def test_a_re_signed_manifest_cannot_move_a_cell(tmp_path):
    """§61.3: `load_run_manifest` only checks a manifest against ITSELF.

    Rotating the coordinate and calling `write_run_manifest` again produces a
    new, internally valid manifest carrying the wrong window -- verified
    directly: the same directory reads back 0.6/0.95 and then 0.3/0.7, each
    time with a digest that matches its own fields.

    The anchor is therefore the plan the SOURCE declares, which the snapshot has
    already been compared against. Every declared cell implies exactly one tag,
    the receipt seals which file carries it, and the run directory has to carry
    it too -- so a rotated coordinate needs a tag that the honest cell already
    holds.
    """
    import dataclasses
    from dna_utils.run_identity import load_run_manifest, write_run_manifest
    from scripts.phase3_select_n import _sha

    records = _sweep(tmp_path)
    receipt_path = next(records.glob("*_sweep_complete.json"))
    receipt = json.loads(receipt_path.read_text())
    # The 0.6/0.95 cell specifically: picking the first sorted Flickr key gives
    # the 0.3/0.7 one, and "rotating" it to 0.3/0.7 changes nothing.
    key = next(k for k in sorted(receipt["cells"])
               if k.startswith("flickr25k|") and "0.6" in k)
    entry = receipt["cells"][key]
    path = records / entry["record"]
    rec = json.loads(path.read_text())
    run = Path(rec["run_dir"])

    # Re-sign the manifest under a different window, and make every other
    # witness agree with it.
    identity = load_run_manifest(str(run))
    # An UNOCCUPIED window, so the duplicate-coordinate check does not fire
    # first and this test exercises the tag gate it is named for. Rotating onto
    # an occupied one is refused too, as a duplicate.
    forged = dataclasses.replace(identity, routing_adaptive_topp_min=0.2,
                                 routing_adaptive_topp_max=0.6)
    write_run_manifest(str(run), forged)
    assert load_run_manifest(str(run)) is not None, (
        "the re-signed manifest must be internally valid, or this test proves "
        "nothing")
    (run / "args.txt").write_text(
        f"routing_adaptive_topp_min{'-' * 20}0.2\n"
        f"routing_adaptive_topp_max{'-' * 20}0.6\n"
        f"lambda_codon_joint{'-' * 20}0.0\n")
    rec["recipe"]["routing_adaptive_topp_min"] = "0.2"
    rec["recipe"]["routing_adaptive_topp_max"] = "0.6"
    rec["identity_digest"] = forged.digest
    rec["geometry"]["identity_digest"] = forged.digest
    path.write_text(json.dumps(rec))
    entry["recipe"] = rec["recipe"]
    entry["identity_digest"] = forged.digest
    entry["record_sha256"] = _sha(path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    with pytest.raises(SelectionRefused) as error:
        _recipe_matrix(records, axis="topp", namespace="phase3sweep")
    assert "args.txt does not hash" in str(error.value) or \
        "which the plan says is" in str(error.value) or \
        "run manifest does not hash" in str(error.value)


def test_rotating_onto_an_occupied_window_is_refused_as_a_duplicate(tmp_path):
    """The other half: two cells cannot both claim one coordinate."""
    import dataclasses
    from dna_utils.run_identity import load_run_manifest, write_run_manifest
    from scripts.phase3_select_n import _sha

    records = _sweep(tmp_path)
    receipt_path = next(records.glob("*_sweep_complete.json"))
    receipt = json.loads(receipt_path.read_text())
    key = next(k for k in sorted(receipt["cells"])
               if k.startswith("flickr25k|") and "0.6" in k)
    entry = receipt["cells"][key]
    path = records / entry["record"]
    rec = json.loads(path.read_text())
    run = Path(rec["run_dir"])
    forged = dataclasses.replace(load_run_manifest(str(run)),
                                 routing_adaptive_topp_min=0.3,
                                 routing_adaptive_topp_max=0.7)
    write_run_manifest(str(run), forged)
    (run / "args.txt").write_text(
        f"routing_adaptive_topp_min{'-' * 20}0.3\n"
        f"routing_adaptive_topp_max{'-' * 20}0.7\n"
        f"lambda_codon_joint{'-' * 20}0.0\n")
    rec["recipe"]["routing_adaptive_topp_min"] = "0.3"
    rec["recipe"]["routing_adaptive_topp_max"] = "0.7"
    rec["identity_digest"] = forged.digest
    rec["geometry"]["identity_digest"] = forged.digest
    path.write_text(json.dumps(rec))
    entry["recipe"] = rec["recipe"]
    entry["identity_digest"] = forged.digest
    entry["record_sha256"] = _sha(path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SelectionRefused) as error:
        _recipe_matrix(records, axis="topp", namespace="phase3sweep")
    assert "args.txt does not hash" in str(error.value) or \
        "recorded twice" in str(error.value) or \
        "run manifest does not hash" in str(error.value)


def test_a_run_directory_that_does_not_carry_its_tag_is_refused(tmp_path):
    records = _sweep(tmp_path)
    victim = next(records.glob("phase3sweep_flickr25k_*.json"))
    rec = json.loads(victim.read_text())
    moved = Path(rec["run_dir"]).parent / "260831+flickr25k_setting1_elsewhere+bs+64"
    Path(rec["run_dir"]).rename(moved)
    rec["run_dir"] = str(moved)
    victim.write_text(json.dumps(rec))
    from scripts.phase3_select_n import _sha
    receipt_path = next(records.glob("*_sweep_complete.json"))
    receipt = json.loads(receipt_path.read_text())
    for entry in receipt["cells"].values():
        if entry["record"] == victim.name:
            entry["run_dir"] = str(moved)
            entry["record_sha256"] = _sha(victim)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SelectionRefused) as error:
        _recipe_matrix(records, axis="topp", namespace="phase3sweep")
    assert "does not carry its own" in str(error.value) or \
        "terminal checkpoint/runtime refusal" in str(error.value)


def test_full_bundle_cyclic_relabel_is_refused_by_trainer_evidence(tmp_path):
    """§61.5/§61.8: rotate every mutable label and preserve the exact set.

    This is the stronger attack the tag/manifest gate missed.  All twelve result
    directories (four windows in each of three datasets), checkpoints, logs,
    manifests, args, standalone launch witnesses, runtime sidecars, records,
    receipt keys and receipt entries are moved one canonical window forward.
    Every mutable JSON witness is rewritten and re-signed for the target cell.
    Only the binding serialized *inside* the unchanged checkpoint still names
    the producing cell, so its SHA-committed metadata must refuse the cycle.
    """
    import dataclasses
    from dna_utils.run_identity import load_run_manifest, write_run_manifest
    from scripts.phase3_select_n import _sha
    from scripts.phase3_selection_matrix import (
        TOPP_GRID, campaign_cell_id, launch_binding_from_expected, tag_for)

    namespace = "phase3sweep"
    records = _sweep(tmp_path, namespace=namespace)
    receipt_path = records / f"{namespace}_sweep_complete.json"
    receipt = json.loads(receipt_path.read_text())
    snapshot = json.loads(next(records.glob("*_snapshot_*.json")).read_text())
    digest = receipt["plan_snapshot_sha256"]
    grid = [tuple(c) for c in TOPP_GRID]

    bundles = []
    for key, entry in list(receipt["cells"].items()):
        record_path = records / entry["record"]
        rec = json.loads(record_path.read_text())
        here = (rec["recipe"]["routing_adaptive_topp_min"],
                rec["recipe"]["routing_adaptive_topp_max"])
        target = grid[(grid.index(here) + 1) % len(grid)]
        bundles.append((key, entry, record_path, rec, here, target))
    assert len(bundles) == 3 * len(grid) == 12

    # Move every occupied path through a temporary name so the permutation is
    # collision-free; this is exactly why the old "target is occupied" argument
    # did not cover a full cycle.
    for i, (_, _, record_path, rec, _, _) in enumerate(bundles):
        run = Path(rec["run_dir"])
        tmp_run = run.parent / f"cyclic-run-tmp-{i}"
        tmp_record = records / f"cyclic-record-tmp-{i}.json"
        run.rename(tmp_run)
        record_path.rename(tmp_record)
        rec["_attack_tmp_run"] = str(tmp_run)
        rec["_attack_tmp_record"] = str(tmp_record)

    new_entries = {}
    for old_key, old_entry, old_record_path, rec, _, target in bundles:
        tmp_run = Path(rec.pop("_attack_tmp_run"))
        tmp_record = Path(rec.pop("_attack_tmp_record"))
        n = rec["N"]
        dataset = rec["dataset"]
        target_tag = tag_for(
            dataset, n, namespace=namespace, topp=target, joint="0.0")
        target_cell = campaign_cell_id(
            dataset, n, topp=target, joint="0.0")
        target_run = tmp_run.parent / (
            f"260831+{dataset}_setting1_{target_tag}+bs+64")
        tmp_run.rename(target_run)

        identity = load_run_manifest(str(target_run))
        forged = dataclasses.replace(
            identity, routing_adaptive_topp_min=float(target[0]),
            routing_adaptive_topp_max=float(target[1]))
        write_run_manifest(str(target_run), forged)
        (target_run / "args.txt").write_text(
            f"routing_adaptive_topp_min{'-' * 20}{target[0]}\n"
            f"routing_adaptive_topp_max{'-' * 20}{target[1]}\n"
            f"lambda_codon_joint{'-' * 20}0.0\n")

        planned = snapshot["plan"]["cell_bindings"][target_cell]
        target_campaign = launch_binding_from_expected(planned, digest)
        assert forged.digest == planned["expected_identity_digest"]
        from dna_utils.run_identity import PHASE3_CAMPAIGN_BINDING_NAME
        evidence_path = target_run / PHASE3_CAMPAIGN_BINDING_NAME
        evidence = json.loads(evidence_path.read_text())
        evidence.update(target_campaign)
        evidence["actual_identity_digest"] = forged.digest
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        runtime_path = Path(str(target_run / "model_state_dict.pth") +
                            ".runtime.json")
        runtime = json.loads(runtime_path.read_text())
        runtime["checkpoint_path"] = str(
            (target_run / "model_state_dict.pth").resolve())
        runtime["extra"].update({
            "tag": target_tag,
            "dataset": forged.dataset,
            "random_seed": forged.seed,
            "num_semantic_parts": forged.num_slots,
            "num_codons_per_codebook": forged.bases_per_slot,
        })
        runtime.setdefault("extra", {})["phase3_campaign"] = evidence
        runtime_path.write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        rec.update(
            tag=target_tag, run_dir=str(target_run),
            identity_digest=forged.digest, campaign=target_campaign)
        rec["recipe"].update(
            routing_adaptive_topp_min=target[0],
            routing_adaptive_topp_max=target[1])
        rec["geometry"]["identity_digest"] = forged.digest
        rec["completion"]["args_txt_sha256"] = _sha(target_run / "args.txt")
        rec["completion"]["run_identity_sha256"] = _sha(
            target_run / "run_identity.json")
        rec["completion"]["phase3_campaign_evidence_sha256"] = _sha(
            evidence_path)
        rec["completion"]["checkpoint_runtime_sha256"] = _sha(runtime_path)

        target_name = (
            f"{namespace}_{dataset}_{target[0]}_{target[1]}_0.0.json")
        target_record = records / target_name
        tmp_record.write_text(json.dumps(rec))
        tmp_record.rename(target_record)
        new_entries[target_cell] = {
            **old_entry,
            "tag": target_tag, "run_dir": str(target_run),
            "identity_digest": forged.digest, "record": target_name,
            "record_sha256": _sha(target_record),
            "recipe": rec["recipe"], "campaign": target_campaign,
            "completion": {
                    field: rec["completion"].get(field) for field in (
                        "final_checkpoint_sha256", "log_csv_sha256",
                        "run_identity_sha256",
                        "args_txt_sha256", "checkpoint_runtime_sha256",
                    "phase3_campaign_evidence_sha256")},
        }
        del old_key, old_record_path

    receipt["cells"] = new_entries
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    with pytest.raises(SelectionRefused) as error:
        _recipe_matrix(records, axis="topp", namespace=namespace)
    assert "checkpoint serialization does not commit" \
        in str(error.value)
