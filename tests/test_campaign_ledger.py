"""Adversarial tests for the immutable Phase-5 campaign ledger."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.campaign_ledger as ledger_module  # noqa: E402
from scripts.campaign_ledger import (  # noqa: E402
    CampaignRefused,
    PLAN_SNAPSHOT_NAME,
    RECEIPT_NAME,
    open_campaign,
    record_cell,
    require_sealed,
    seal_campaign,
)
from scripts._ablation_exec import CANONICAL_RUNNER  # noqa: E402
from dna_utils.run_identity import RunIdentity, write_run_manifest  # noqa: E402
from dna_utils.runtime_state import write_checkpoint_metadata  # noqa: E402
from tests.test_extraction_run_validation import _cell as extraction_cell  # noqa: E402
from tests.test_seal_cell_analysis import _halves, _seal  # noqa: E402

TOKEN = "a" * 64
OTHER_TOKEN = "b" * 64
EXPERIMENTS = {
    "cifar_A_v4": "CIFAR10",
    "flickr_A_v4": "Flickr25k",
    "nuswide_A_v4": "NUSWIDE",
    "mscoco_A_v5b": "MSCOCO",
}
ARMS = (
    "A2_no_text", "A4_shared_codebook", "A5_none", "A5_joint",
    "A5_nogumbel", "A5_both")
_REAL_CANONICAL_RECIPE_ASSERT = ledger_module._assert_canonical_recipe


@pytest.fixture(autouse=True)
def _synthetic_plan_authority(monkeypatch):
    """Unit fixtures are deliberately tiny/fake; production has its own test."""
    monkeypatch.setattr(
        ledger_module, "_assert_canonical_recipe", lambda _plan: None)


def _plan(tmp_path: Path, *, n: int = 4, runner: str = CANONICAL_RUNNER,
          drop_last: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pinned = [
        "GDNA_NUM_SEMANTIC_PARTS", "NUM_CODONS", "K", "FIXED_N",
        "WHITEN_VARIANT", "CACHE_OVERRIDE", "WDIR_OVERRIDE",
        "QWEN_OVERRIDE", "TAG_SUFFIX"]
    cells = []
    for exp, dataset in EXPERIMENTS.items():
        for arm in ARMS:
            tag = f"promptAblA_{exp}_{arm}"
            flags = [
                "--routing_adaptive_topp_min", "0.6",
                "--routing_adaptive_topp_max", "0.95",
                "--lambda_codeword_codon_sinkhorn", "0.0",
                "--lambda_codon_joint", "0.03", "--selection_mode", arm,
                "-e", str(n + 1)]
            if arm == "A2_no_text":
                flags += ["--disable_text_supervision"]
            if arm == "A4_shared_codebook":
                flags += ["--share_codebook"]
            if arm in ("A5_nogumbel", "A5_both"):
                flags += ["--no_gumbel_softmax"]
            cells.append({
                "exp": exp, "dataset": dataset, "cell": arm, "N": n,
                "tag": tag, "runner": runner, "flags": flags,
                "env": {
                    "GDNA_NUM_SEMANTIC_PARTS": "5", "NUM_CODONS": "3",
                    "K": "64" if dataset == "CIFAR10" else "128",
                    "FIXED_N": str(n), "WHITEN_VARIANT": "_localOnly",
                    "CACHE_OVERRIDE": str(tmp_path / "cache" / exp),
                    "WDIR_OVERRIDE": str(tmp_path / "foils" / exp),
                    "QWEN_OVERRIDE": str(tmp_path / "qwen" / f"{exp}.jsonl"),
                    "TAG_SUFFIX": f"_{arm}",
                },
            })
    if drop_last:
        cells.pop()
    plan = {
        "schema_version": 2, "expected_cells": len(cells),
        "env_passthrough": ["PATH", "PY"],
        "env_passthrough_values": {
            "PATH": os.environ["PATH"], "PY": sys.executable},
        "env_pinned": pinned, "runners": [runner], "cells": cells,
        "selected_n": {exp: n for exp in EXPERIMENTS},
        "selection_sha256": "1" * 64,
    }
    plan["plan_digest"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return path


def _open(tmp_path: Path, *, token: str = TOKEN, n: int = 4):
    plan = _plan(tmp_path, n=n)
    ledger = tmp_path / "ledger"
    held = open_campaign(
        ledger, plan, owner_pid=os.getpid(), campaign_token=token,
        expected_plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest())
    return ledger, plan, held


def _mock_success_validation(monkeypatch):
    def evidence(_held, _cell, run_dir):
        return {
            "run_dir": run_dir, "run_identity_digest": "2" * 64,
            "artifacts": {"analysis_complete.json": {
                "sha256": "3" * 64, "size_bytes": 1}},
            "analysis_metrics": {"map_at_R_bioproj": 0.5},
            "analysis_authority": {
                "scientific_authority": False,
                "paper_result_eligible": False,
                "requires_downstream_deterministic_recomputation": True,
            },
        }

    monkeypatch.setattr(ledger_module, "_validate_cell_output", evidence)
    monkeypatch.setattr(
        ledger_module, "_validate_outcome",
        lambda *args, **kwargs: ({"valid": True}, "4" * 64))
    return evidence


def _finish_mocked(ledger: Path, held: dict):
    (ledger / "outcomes").mkdir(exist_ok=True)
    for tag in held["tags"]:
        run_dir = ledger.parent / "result" / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        record_cell(
            ledger, tag, "ok", campaign_token=held["campaign_token"],
            run_dir=str(run_dir),
            outcome_path=str(ledger / "outcomes" / f"{tag}.json"))


def test_open_freezes_exact_plan_and_generation(tmp_path):
    ledger, plan, held = _open(tmp_path)
    snapshot = ledger / PLAN_SNAPSHOT_NAME
    assert snapshot.read_bytes() == plan.read_bytes()
    assert held["campaign_token"] == TOKEN
    assert held["launcher_pid"] == os.getpid()
    assert held["expected_cells"] == 24
    assert len(held["source_sha256"]) >= 20


def test_noncanonical_count_and_foreign_runner_are_refused(tmp_path):
    with pytest.raises(CampaignRefused, match="exactly 24"):
        open_campaign(
            tmp_path / "short-ledger", _plan(tmp_path / "short", drop_last=True),
            owner_pid=os.getpid(), campaign_token=TOKEN)


def test_fresh_planner_authority_refuses_science_and_env_relabel(
        tmp_path, monkeypatch):
    path = _plan(tmp_path)
    expected = json.loads(path.read_text())
    monkeypatch.setattr(
        ledger_module, "_fresh_canonical_plan", lambda: expected)
    _REAL_CANONICAL_RECIPE_ASSERT(expected)

    forged = json.loads(json.dumps(expected))
    forged["cells"][0]["env"]["NUM_CODONS"] = "4"
    with pytest.raises(CampaignRefused, match="canonical Phase-5 recipe"):
        _REAL_CANONICAL_RECIPE_ASSERT(forged)

    poisoned = json.loads(json.dumps(expected))
    poisoned["env_passthrough"].append("BASH_ENV")
    poisoned["env_passthrough_values"]["BASH_ENV"] = str(
        tmp_path / "startup.sh")
    with pytest.raises(CampaignRefused, match="canonical Phase-5 recipe"):
        _REAL_CANONICAL_RECIPE_ASSERT(poisoned)

    loader_poisoned = json.loads(json.dumps(expected))
    loader_poisoned["env_passthrough_values"]["HOME"] = "/tmp/adversarial-loader"
    with pytest.raises(CampaignRefused, match="canonical Phase-5 recipe"):
        _REAL_CANONICAL_RECIPE_ASSERT(loader_poisoned)


def test_mscoco_identity_uses_existing_trainer_loss_defaults(tmp_path):
    plan = json.loads(_plan(tmp_path).read_text())
    cells = [cell for cell in plan["cells"]
             if cell["dataset"] == "MSCOCO"]
    a2 = next(cell for cell in cells if cell["cell"] == "A2_no_text")
    # The compact synthetic plan omits explicit A2 zero flags; inject the
    # canonical flags to isolate the default-vs-override behavior.
    a2["flags"] += ["--lambda_text_code_kl", "0.0",
                    "--lambda_text_hash_ntxent", "0.0",
                    "--lambda_xmodal_commit", "0.0"]
    assert ledger_module._expected_identity_fields(a2)[
        "lambda_text_code_kl"] == 0.0
    for cell in cells:
        if cell["cell"] == "A2_no_text":
            continue
        identity = ledger_module._expected_identity_fields(cell)
        assert identity["lambda_text_code_kl"] == 0.10
        assert identity["lambda_text_hash_ntxent"] == 0.10
        assert identity["lambda_xmodal_commit"] == 0.10
    with pytest.raises(CampaignRefused, match="runner must be"):
        open_campaign(
            tmp_path / "foreign-ledger",
            _plan(tmp_path / "foreign", runner="foreign.sh"),
            owner_pid=os.getpid(), campaign_token=TOKEN)


def test_second_generation_cannot_overwrite_reservation(tmp_path):
    ledger, plan, _ = _open(tmp_path)
    original = (ledger / "campaign_reservation.json").read_bytes()
    with pytest.raises(CampaignRefused, match="already exists"):
        open_campaign(
            ledger, plan, owner_pid=os.getpid(), campaign_token=OTHER_TOKEN)
    assert (ledger / "campaign_reservation.json").read_bytes() == original


def test_wrong_generation_token_cannot_record(tmp_path):
    ledger, _, held = _open(tmp_path)
    with pytest.raises(CampaignRefused, match="token"):
        record_cell(
            ledger, held["tags"][0], "failed", campaign_token=OTHER_TOKEN)


def test_success_requires_executor_outcome_and_scientific_artifacts(tmp_path):
    ledger, _, held = _open(tmp_path)
    tag = held["tags"][0]
    empty = Path(held["result_root"]) / (
        f"20990101+ds_{tag}_P0refit_e4+fixture")
    # Never create test material under the real result root. A caller pointing
    # elsewhere is refused before any manifest inspection.
    foreign = tmp_path / "result" / empty.name
    foreign.mkdir(parents=True)
    with pytest.raises(CampaignRefused, match="both run_dir and executor"):
        record_cell(
            ledger, tag, "ok", campaign_token=TOKEN, run_dir=str(foreign))
    with pytest.raises(CampaignRefused, match="outside reserved result root"):
        ledger_module._validate_cell_output(
            held, ledger_module._cell_for_tag(held, tag)[1], str(foreign))


def test_real_output_validator_reopens_identity_extraction_bio_nmi_and_sidecar(
        tmp_path):
    plan_path = _plan(tmp_path / "authority")
    plan = json.loads(plan_path.read_text())
    cell = plan["cells"][0]
    result_root = tmp_path / "result"
    result_root.mkdir()
    run = result_root / (
        f"20990101+cifar10_setting1_{cell['tag']}_P0refit_e4+fixture")
    run.mkdir()
    extraction_cell(run)

    expected = ledger_module._expected_identity_fields(cell)
    identity = RunIdentity(**expected)
    write_run_manifest(str(run), identity)
    (run / "args.txt").write_text("synthetic strict-bound fixture\n")
    write_checkpoint_metadata(
        str(run / "model_state_dict.pth"),
        checkpoint_epoch_zero_based=identity.stop_after_epoch,
        training_epoch_budget=identity.epoch_budget,
        stop_after_epoch=identity.stop_after_epoch,
        lr_schedule_horizon=identity.lr_schedule_horizon,
        sinkhorn_schedule_horizon=identity.sinkhorn_schedule_horizon,
        sinkhorn_epsilon_init=identity.sinkhorn_epsilon_init,
        sinkhorn_epsilon_final=identity.sinkhorn_epsilon_final,
        lr_scheduler="synthetic")
    _halves(run)
    sealed = _seal(run)
    assert sealed.returncode == 0, sealed.stdout + sealed.stderr

    held = {"result_root": str(result_root)}
    evidence = ledger_module._validate_cell_output(held, cell, str(run))
    assert evidence["run_identity_digest"] == identity.digest
    assert evidence["analysis_authority"]["scientific_authority"] is False
    assert set(evidence["artifacts"]) >= {
        "model_state_dict.pth", "extraction_complete.json",
        "analysis_complete.json"}

    sidecar = run / "model_state_dict.pth.runtime.json"
    payload = json.loads(sidecar.read_text())
    payload["checkpoint_epoch_zero_based"] = 3
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(CampaignRefused, match="checkpoint runtime differs"):
        ledger_module._validate_cell_output(held, cell, str(run))


def test_cell_record_is_create_once_and_failure_cannot_be_relabelled(tmp_path):
    ledger, _, held = _open(tmp_path)
    tag = held["tags"][0]
    record_cell(ledger, tag, "failed", campaign_token=TOKEN, detail="rc=23")
    original = (ledger / f"cell_{tag}.json").read_bytes()
    with pytest.raises(CampaignRefused, match="already exists"):
        record_cell(ledger, tag, "failed", campaign_token=TOKEN, detail="rc=0")
    assert (ledger / f"cell_{tag}.json").read_bytes() == original


def test_exact_24_mocked_integrity_cells_seal_but_not_as_paper_authority(
        tmp_path, monkeypatch):
    _mock_success_validation(monkeypatch)
    ledger, plan, held = _open(tmp_path)
    _finish_mocked(ledger, held)
    receipt = seal_campaign(ledger, campaign_token=TOKEN)
    assert receipt["cell_count"] == 24
    assert receipt["scientific_authority"] is False
    assert receipt["paper_result_eligible"] is False
    assert receipt["requires_downstream_deterministic_recomputation"] is True
    reopened = require_sealed(
        ledger, plan_file_sha256=hashlib.sha256(plan.read_bytes()).hexdigest())
    assert reopened["cell_count"] == 24


def test_twenty_three_cells_or_one_failed_cell_never_seals(tmp_path, monkeypatch):
    _mock_success_validation(monkeypatch)
    ledger, _, held = _open(tmp_path)
    (ledger / "outcomes").mkdir(exist_ok=True)
    for tag in held["tags"][:-1]:
        run_dir = tmp_path / "result" / tag
        run_dir.mkdir(parents=True)
        record_cell(
            ledger, tag, "ok", campaign_token=TOKEN, run_dir=str(run_dir),
            outcome_path=str(ledger / "outcomes" / f"{tag}.json"))
    record_cell(
        ledger, held["tags"][-1], "failed", campaign_token=TOKEN,
        detail="trainer rc=23")
    with pytest.raises(CampaignRefused, match="23 of 24"):
        seal_campaign(ledger, campaign_token=TOKEN)
    assert not (ledger / RECEIPT_NAME).exists()


def test_output_change_between_record_and_seal_is_refused(tmp_path, monkeypatch):
    evidence = _mock_success_validation(monkeypatch)
    ledger, _, held = _open(tmp_path)
    _finish_mocked(ledger, held)

    changed_tag = held["tags"][0]
    real = ledger_module._validate_cell_output

    def changed(held_arg, cell, run_dir):
        value = real(held_arg, cell, run_dir)
        if cell["tag"] == changed_tag:
            value = json.loads(json.dumps(value))
            value["artifacts"]["analysis_complete.json"]["sha256"] = "9" * 64
        return value

    monkeypatch.setattr(ledger_module, "_validate_cell_output", changed)
    with pytest.raises(CampaignRefused, match="cannot seal"):
        seal_campaign(ledger, campaign_token=TOKEN)
    assert not (ledger / RECEIPT_NAME).exists()
    assert callable(evidence)


def test_plan_snapshot_and_source_drift_fail_closed(tmp_path, monkeypatch):
    ledger, _, held = _open(tmp_path)
    snapshot = ledger / PLAN_SNAPSHOT_NAME
    snapshot.chmod(0o644)
    snapshot.write_bytes(snapshot.read_bytes() + b" ")
    with pytest.raises(CampaignRefused, match="snapshot is missing or changed"):
        record_cell(
            ledger, held["tags"][0], "failed", campaign_token=TOKEN)

    other_ledger, _, other_held = _open(
        tmp_path / "other", token=OTHER_TOKEN)
    original = dict(other_held["source_sha256"])
    monkeypatch.setattr(
        ledger_module, "source_digests",
        lambda root=REPO: {**original, "config.py": "0" * 64})
    with pytest.raises(CampaignRefused, match="source changed"):
        record_cell(
            other_ledger, other_held["tags"][0], "failed",
            campaign_token=OTHER_TOKEN)


def test_reservation_fields_are_reconstructed_from_snapshot_and_repo(tmp_path):
    ledger, _, held = _open(tmp_path)
    path = ledger / "campaign_reservation.json"
    forged = json.loads(path.read_text())
    forged.update({
        "selected_n": {key: 999 for key in forged["selected_n"]},
        "selection_sha256": "f" * 64,
        "plan_digest": "e" * 64,
        "tag_set_sha256": "d" * 64,
        "result_root": str((tmp_path / "forged-result").resolve()),
    })
    path.chmod(0o644)
    path.write_text(json.dumps(forged, sort_keys=True) + "\n")
    with pytest.raises(CampaignRefused, match="immutable authorities"):
        record_cell(
            ledger, held["tags"][0], "failed", campaign_token=TOKEN)


def test_stale_entry_from_another_generation_is_not_replayed(tmp_path):
    first, _, first_held = _open(tmp_path / "first", token=TOKEN)
    tag = first_held["tags"][0]
    record_cell(first, tag, "failed", campaign_token=TOKEN)
    second, _, _ = _open(tmp_path / "second", token=OTHER_TOKEN)
    shutil.copyfile(first / f"cell_{tag}.json", second / f"cell_{tag}.json")
    with pytest.raises(CampaignRefused, match="cannot seal"):
        seal_campaign(second, campaign_token=OTHER_TOKEN)


def test_post_seal_record_and_receipt_overwrite_are_refused(tmp_path, monkeypatch):
    _mock_success_validation(monkeypatch)
    ledger, _, held = _open(tmp_path)
    _finish_mocked(ledger, held)
    seal_campaign(ledger, campaign_token=TOKEN)
    with pytest.raises(CampaignRefused, match="sealed campaign"):
        record_cell(
            ledger, held["tags"][0], "failed", campaign_token=TOKEN)
    with pytest.raises(CampaignRefused, match="already exists"):
        seal_campaign(ledger, campaign_token=TOKEN)


def test_rewritten_receipt_selection_binding_is_refused(tmp_path, monkeypatch):
    _mock_success_validation(monkeypatch)
    ledger, _, held = _open(tmp_path)
    _finish_mocked(ledger, held)
    seal_campaign(ledger, campaign_token=TOKEN)
    path = ledger / RECEIPT_NAME
    forged = json.loads(path.read_text())
    forged["selected_n"] = {key: 999 for key in forged["selected_n"]}
    forged["selection_sha256"] = "f" * 64
    path.chmod(0o644)
    path.write_text(json.dumps(forged, sort_keys=True) + "\n")
    with pytest.raises(CampaignRefused, match="receipt contract"):
        require_sealed(ledger)


def test_cli_key_uses_exact_bytes_and_open_rejects_synthetic_recipe(tmp_path):
    plan = _plan(tmp_path)
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    command = [sys.executable, str(REPO / "scripts" / "campaign_ledger.py")]
    key = subprocess.run(
        [*command, "key", "--plan", str(plan),
         "--expect-plan-sha256", digest], capture_output=True, text=True)
    assert key.returncode == 0 and len(key.stdout.strip()) == 64
    opened = subprocess.run(
        [*command, "open", "--ledger", str(tmp_path / "cli-ledger"),
         "--plan", str(plan), "--expect-plan-sha256", digest,
         "--owner-pid", str(os.getpid()), "--campaign-token", TOKEN],
        capture_output=True, text=True)
    assert opened.returncode != 0
    assert "canonical Phase-5 plan authority" in opened.stderr
    assert not (tmp_path / "cli-ledger" / "campaign_reservation.json").exists()


def test_same_tags_different_n_share_global_collision_key(tmp_path):
    first = ledger_module.campaign_lock_key(_plan(tmp_path / "a", n=4))
    second = ledger_module.campaign_lock_key(_plan(tmp_path / "b", n=9))
    assert first == second


def test_record_refuses_foreign_outcome_even_if_validator_is_mocked(
        tmp_path, monkeypatch):
    _mock_success_validation(monkeypatch)
    ledger, _, held = _open(tmp_path)
    tag = held["tags"][0]
    run_dir = tmp_path / "result" / tag
    run_dir.mkdir(parents=True)
    foreign = tmp_path / "forged-outcome.json"
    foreign.write_text("{}\n")
    with pytest.raises(CampaignRefused, match="canonical ledger/outcomes"):
        record_cell(
            ledger, tag, "ok", campaign_token=TOKEN,
            run_dir=str(run_dir), outcome_path=str(foreign.resolve()))


def test_outcome_gpu_lease_path_is_bound_to_physical_uuid(tmp_path):
    ledger, _, held = _open(tmp_path)
    tag = held["tags"][0]
    index, _ = ledger_module._cell_for_tag(held, tag)
    outcome_dir = ledger / "outcomes"
    outcome_dir.mkdir()
    path = outcome_dir / f"{tag}.json"
    run_dir = str((REPO / "result" / f"fixture_{tag}").resolve())
    payload = {
        "schema": "groundeddna.ablation-cell-outcome", "schema_version": 1,
        "campaign_token": TOKEN,
        "plan_file_sha256": held["plan_file_sha256"],
        "cell_index": index, "tag": tag, "child_returncode": 0,
        "repo_root": held["repo_root"], "run_dir": run_dir,
        "source_sha256": held["source_sha256"],
        "gpu": {"index": 0, "uuid": "GPU-FORGED"},
        "gpu_lease_paths": ["/tmp/not-a-real-lease"],
        "executor_pid": os.getpid(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(CampaignRefused, match="not bound"):
        ledger_module._validate_outcome(
            held, path=path, index=index, tag=tag, run_dir=run_dir)


def test_outcome_rejects_bool_index_and_nonphysical_self_consistent_uuid(
        tmp_path, monkeypatch):
    from dna_utils.gpu_lease import GPU_LEASE_ROOT

    ledger, _, held = _open(tmp_path)
    tag = held["tags"][0]
    index, _ = ledger_module._cell_for_tag(held, tag)
    outcome_dir = ledger / "outcomes"
    outcome_dir.mkdir()
    path = outcome_dir / f"{tag}.json"
    run_dir = str((REPO / "result" / f"fixture_{tag}").resolve())
    payload = {
        "schema": "groundeddna.ablation-cell-outcome", "schema_version": 1,
        "campaign_token": TOKEN,
        "plan_file_sha256": held["plan_file_sha256"],
        "cell_index": index, "tag": tag, "child_returncode": 0,
        "repo_root": held["repo_root"], "run_dir": run_dir,
        "source_sha256": held["source_sha256"],
        "gpu": {"index": True, "uuid": "GPU-FORGED-NOT-PRESENT"},
        "gpu_lease_paths": [str(
            GPU_LEASE_ROOT / "GPU-FORGED-NOT-PRESENT.lock")],
        "executor_pid": os.getpid(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(CampaignRefused, match="not bound"):
        ledger_module._validate_outcome(
            held, path=path, index=index, tag=tag, run_dir=run_dir)


def test_outcome_requires_exact_executor_and_completion_provenance(
        tmp_path, monkeypatch):
    from dna_utils.gpu_lease import GPU_LEASE_ROOT

    ledger, _, held = _open(tmp_path)
    tag = held["tags"][0]
    index, _ = ledger_module._cell_for_tag(held, tag)
    outcome_dir = ledger / "outcomes"
    outcome_dir.mkdir()
    path = outcome_dir / f"{tag}.json"
    run_dir = str((REPO / "result" / f"fixture_{tag}").resolve())
    gpu = {
        "index": 0, "uuid": "GPU-REAL-PHYSICAL", "name": "Physical",
        "pci_bus_id": "0000:01:00.0", "driver": "real"}
    payload = {
        "schema": "groundeddna.ablation-cell-outcome", "schema_version": 1,
        "campaign_token": TOKEN,
        "plan_file_sha256": held["plan_file_sha256"],
        "cell_index": index, "tag": tag, "child_returncode": 0,
        "repo_root": held["repo_root"], "run_dir": run_dir,
        "source_sha256": held["source_sha256"], "gpu": gpu,
        "gpu_lease_paths": [str(
            GPU_LEASE_ROOT / "GPU-REAL-PHYSICAL.lock")],
    }
    monkeypatch.setattr(
        ledger_module, "_physical_gpu_assignment", lambda _index: gpu)
    path.write_text(json.dumps(payload))
    with pytest.raises(CampaignRefused, match="not bound"):
        ledger_module._validate_outcome(
            held, path=path, index=index, tag=tag, run_dir=run_dir)

    payload.update({
        "executor_pid": os.getpid(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    path.write_text(json.dumps(payload))
    outcome, _ = ledger_module._validate_outcome(
        held, path=path, index=index, tag=tag, run_dir=run_dir)
    assert outcome["executor_pid"] == os.getpid()

    payload["gpu"] = {
        "index": 0, "uuid": "GPU-FORGED-NOT-PRESENT",
        "name": "Forged", "pci_bus_id": "0000:00:00.0",
        "driver": "forged"}
    payload["gpu_lease_paths"] = [str(
        GPU_LEASE_ROOT / "GPU-FORGED-NOT-PRESENT.lock")]
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        ledger_module, "_physical_gpu_assignment",
        lambda _index: {
            "index": 0, "uuid": "GPU-REAL-PHYSICAL",
            "name": "Physical", "pci_bus_id": "0000:01:00.0",
            "driver": "real"})
    with pytest.raises(CampaignRefused, match="physical host assignment"):
        ledger_module._validate_outcome(
            held, path=path, index=index, tag=tag, run_dir=run_dir)


def test_world_writable_ledger_and_dead_owner_are_refused(tmp_path):
    plan = _plan(tmp_path / "plan")
    unsafe = tmp_path / "unsafe-ledger"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    with pytest.raises(CampaignRefused, match="owner-controlled"):
        open_campaign(
            unsafe, plan, owner_pid=os.getpid(), campaign_token=TOKEN)
    with pytest.raises(CampaignRefused, match="not live"):
        open_campaign(
            tmp_path / "dead-ledger", plan, owner_pid=999_999_999,
            campaign_token=TOKEN)


def test_source_closure_is_exact_and_contains_transitive_model_dependencies(
        monkeypatch):
    python_closure = set(ledger_module._python_source_closure())
    assert {path for path in ledger_module.SOURCE_PATHS
            if path.endswith(".py")} == python_closure
    assert {
        "dataloaders.py", "dna_utils/__init__.py",
        "dna_utils/bio_constraints.py", "dna_utils/gc_policy.py",
        "dna_utils/runtime_state.py", "models/pretrained_backbone.py",
        "models/pretrained_backbone_clip.py", "models/visual_encoder.py",
        "models/text_encoder.py", "models/adapters.py",
        "models/semantic_router.py", "models/__init__.py",
        "scripts/__init__.py", "scripts/_ablation_campaign_launch.py",
        "scripts/_ablation_group_guard.py",
    } <= python_closure
    assert "scripts/lib/result_dir.sh" in ledger_module.SOURCE_PATHS

    original = ledger_module._sha_file
    adapters = REPO / "models" / "adapters.py"
    monkeypatch.setattr(
        ledger_module, "_sha_file",
        lambda path: "0" * 64 if path == adapters else original(path))
    assert ledger_module.source_digests()["models/adapters.py"] == "0" * 64

    model_init = REPO / "models" / "__init__.py"
    monkeypatch.setattr(
        ledger_module, "_sha_file",
        lambda path: "1" * 64 if path == model_init else original(path))
    assert ledger_module.source_digests()["models/__init__.py"] == "1" * 64
