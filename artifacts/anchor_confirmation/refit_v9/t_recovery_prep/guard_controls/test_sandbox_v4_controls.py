"""Controls of the r8 OS boundary (sandboxed_pytest_r8.py; audits 802, 804, 805, 808), run INSIDE the sandbox.
Every case uses synthetic names; nothing here exists on the host side of the boundary. DENY cases must fail
because the target is absent, read-only, unreachable or has no device -- for this process AND for children of
any form (isolated interpreter, minimal environment, shell, copied script). ALLOW cases must succeed."""
import json, os, shutil, socket, subprocess, sys
from pathlib import Path
import pytest

TREE = os.getcwd()
PY = sys.executable
NOPE = "synthetic-never-opened"
LIVE = ["/home/yschoi/gdna_anchorRT_result/x/model_state_dict.pth", "/home/yschoi/gdna_anchorRT_ops/device_budget_ledger.jsonl",
        "/home/yschoi/gdna_anchorRTrec_ops/device_budget_ledger.jsonl", "/home/yschoi/gdna_anchorRT_recovery_claims/x.json",
        "/home/yschoi/GroundedDNA/docs/PHASE1_PHASE2_REAUDIT_2026-08-14.md",
        "/data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/ancR9_sweep_complete.json",
        "/data/yschoi/groundeddna_cache_v6prov/nuswide_clip_tokens/meta.json", "/data/yschoi/dataset/x.jpg",
        "/home/yschoi/.cache/huggingface/hub", "/tmp/groundeddna-d6-gpu-leases-1003"]


@pytest.mark.parametrize("path", LIVE)
def test_live_paths_do_not_exist_here(path):
    assert not os.path.lexists(path)


@pytest.mark.parametrize("rel", [f"cache/{NOPE}.npz", f"result/{NOPE}/config.pt", "artifacts/anchor_confirmation",
                                 "artifacts/umrch_flickr25k_embeddings.npy", f"dataset/{NOPE}"])
def test_tree_data_paths_are_empty_here(rel):
    with pytest.raises(OSError):
        open(os.path.join(TREE, rel), "rb")


def test_the_tree_is_read_only():
    with pytest.raises(OSError):
        open(os.path.join(TREE, "scripts", f"{NOPE}.py"), "w")
    with pytest.raises(OSError):
        os.mkdir(os.path.join(TREE, NOPE))


def test_a_bare_relative_payload_name_resolves_to_nothing():
    with pytest.raises(OSError):                       # audit 808: no inference from a listing history
        open("audit-never-opened.pt", "rb")


def test_an_alias_in_tmp_to_a_live_path_dangles(tmp_path):
    alias = tmp_path / "alias"
    os.symlink("/home/yschoi/gdna_anchorRT_result/x/model_state_dict.pth", alias)
    with pytest.raises(OSError):
        open(alias, "rb")


def test_no_network():
    with pytest.raises(OSError), socket.socket() as s:
        s.settimeout(2)
        s.connect(("1.1.1.1", 53))


def test_no_gpu_device_or_driver():
    assert not [d for d in os.listdir("/dev") if d.startswith("nvidia")]
    if shutil.which("nvidia-smi"):
        done = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        assert done.returncode != 0 or "Synthetic GPU" in done.stdout


CHILD_READ = "import sys; open(sys.argv[1], 'rb')"


@pytest.mark.parametrize("form", ["plain", "isolated", "minimal-env", "no-site"])
@pytest.mark.parametrize("target", [LIVE[0], f"result/{NOPE}/config.pt"])
def test_children_of_every_form_see_the_same_boundary(tmp_path, form, target):
    script = tmp_path / "reader.py"
    script.write_text(CHILD_READ)
    argv = {"plain": [PY, str(script), target], "isolated": [PY, "-I", "-B", str(script), target],
            "minimal-env": [PY, str(script), target], "no-site": [PY, "-S", "-E", str(script), target]}[form]
    env = {"PATH": "/usr/bin"} if form == "minimal-env" else None
    done = subprocess.run(argv, cwd=TREE, env=env, capture_output=True, text=True)
    assert done.returncode != 0 and ("FileNotFoundError" in done.stderr or "OSError" in done.stderr)


def test_a_copied_shell_script_sees_the_same_boundary(tmp_path):
    copy = tmp_path / "copied.sh"
    copy.write_text(f"cat {LIVE[0]} && exit 0\ncat {TREE}/result/{NOPE}/config.pt && exit 0\n"
                    f"touch {TREE}/scripts/{NOPE} && exit 0\nexit 9\n")
    done = subprocess.run(["bash", str(copy)], cwd=TREE, env={}, capture_output=True, text=True)
    assert done.returncode == 9


def test_git_serves_source_but_no_payload_blob():
    ok = subprocess.run(["git", "show", "HEAD:scripts/anchor_refit_stage.py"], capture_output=True)
    assert ok.returncode == 0 and b"Stages R and T" in ok.stdout
    payload = subprocess.run(["git", "show", "HEAD:artifacts/umrch_flickr25k_embeddings.npy"], capture_output=True)
    assert payload.returncode != 0
    record = subprocess.run(["git", "show", "HEAD:artifacts/anchor_confirmation/ancR9_sweep_complete.json"],
                            capture_output=True)
    assert record.returncode != 0
    assert subprocess.run(["git", "show", "3dd1c02:train_siglip2.py"], capture_output=True).returncode == 0


def test_writes_to_live_names_stay_inside_the_sandbox(tmp_path):
    """Creating a live-looking path here touches only the sandbox's own throwaway root (the host check after the
    run confirms the real path is unchanged)."""
    try:
        os.makedirs("/home/yschoi/gdna_anchorRT_recovery_claims", exist_ok=True)
        Path("/home/yschoi/gdna_anchorRT_recovery_claims/sandbox-control.json").write_text("{}")
        made = True
    except OSError:
        made = False
    (tmp_path / "made.json").write_text(json.dumps({"made_inside": made}))


def test_private_tmp_and_cleanup_work(tmp_path):
    tree = tmp_path / "a" / "b"
    tree.mkdir(parents=True)
    (tree / "x.npz").write_bytes(b"synthetic")
    assert (tree / "x.npz").read_bytes() == b"synthetic"
    shutil.rmtree(tmp_path / "a")
    assert not (tmp_path / "a").exists()
    assert subprocess.run([PY, "-c", "print(1)"], capture_output=True).returncode == 0
