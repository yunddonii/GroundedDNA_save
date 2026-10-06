"""Positive controls for the r8 test boundary v3 (audits 802.2, 804). Synthetic targets only: no named
path is ever opened and no scientific producer runs. DENY cases must be refused (in this process, or in
the guarded child -- which then exits nonzero); ALLOW cases must not be refused. The decision-level cases
call the policy functions directly with strings (non-executing), as the audit's AST checks do."""
import json, os, socket, subprocess, sys
from pathlib import Path
import pytest
import gdna_guard_policy as G

W = "/data/yschoi/gdna_anchor_refit_v9r8"
PY = sys.executable
NOPE = "synthetic-never-opened"

DENY_OPENS = [
    ("live-result-payload", f"/home/yschoi/gdna_anchorRT_result/{NOPE}/model_state_dict.pth", "rb"),
    ("worktree-cache-payload", f"{W}/cache/{NOPE}.npz", "rb"),
    ("worktree-result-config", f"{W}/result/{NOPE}/config.pt", "rb"),
    ("worktree-artifacts-json", f"{W}/artifacts/anchor_confirmation/{NOPE}.json", "r"),
    ("other-tmp", f"/tmp/{NOPE}-outside-private/x.json", "r"),
    ("live-recovery-ledger-append", f"/home/yschoi/gdna_anchorRTrec_ops/{NOPE}.jsonl", "a"),
    ("worktree-source-write", f"{W}/scripts/{NOPE}.py", "w"),
    ("gpu-device", f"/dev/nvidia-{NOPE}", "rb"),
    ("main-tree-ledger", f"/home/yschoi/GroundedDNA/docs/{NOPE}.md", "r"),
]


@pytest.mark.parametrize("name,path,mode", DENY_OPENS, ids=[c[0] for c in DENY_OPENS])
def test_deny_open(name, path, mode):
    with pytest.raises(PermissionError, match="boundary guard"):
        open(path, mode)


def test_deny_live_claim_exclusive_create():
    with pytest.raises(PermissionError, match="boundary guard"):
        os.open(f"/home/yschoi/gdna_anchorRT_recovery_claims/{NOPE}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY)


@pytest.mark.parametrize("target", [f"/home/yschoi/gdna_anchorRT_recovery_claims/{NOPE}.json",
                                    f"/data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/{NOPE}.json"],
                         ids=["alias-to-claim-root", "alias-to-r7-records"])
def test_deny_a_private_alias_to_a_live_path(tmp_path, target):
    alias = tmp_path / "alias.json"
    os.symlink(target, alias)
    with pytest.raises(PermissionError, match="boundary guard"):
        open(alias, "r")


def test_deny_a_relative_write_into_the_worktree(monkeypatch):
    monkeypatch.chdir(W)
    with pytest.raises(PermissionError, match="boundary guard"):
        open(f"artifacts/{NOPE}.json", "w")


def test_deny_a_dir_fd_relative_mkdir_in_the_worktree():
    fd = os.open(W, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(PermissionError, match="boundary guard"):
            os.mkdir(NOPE, dir_fd=fd)
    finally:
        os.close(fd)


def test_deny_a_rename_into_a_live_root(tmp_path):
    (tmp_path / "x").write_text("x")
    with pytest.raises(PermissionError, match="boundary guard"):
        os.rename(tmp_path / "x", f"/home/yschoi/gdna_anchorRT_recovery_claims/{NOPE}")


def test_deny_an_inet_connection():
    with socket.socket() as s, pytest.raises(PermissionError, match="boundary guard"):
        s.connect(("127.0.0.1", 9))


DENY_CHILDREN = [
    ("python-m-entry", [PY, "-m", "scripts.anchor_terminal_test", "--help"]),
    ("bash-wrapper-plain", ["bash", f"{W}/scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh"]),
    ("direct-extraction", [PY, f"{W}/extraction_siglip2.py", "--help"]),
    ("t-entry", [PY, "-B", f"{W}/scripts/anchor_terminal_test.py", "--help"]),
    ("train-extraction", [PY, f"{W}/scripts/extract_train_split.py", "--help"]),
    ("python-E", [PY, "-E", "-c", "pass"]),
    ("python-S", [PY, "-S", "-c", "pass"]),
    ("python-s", [PY, "-s", "-c", "pass"]),
    ("nvidia-smi", ["nvidia-smi"]),
    ("git-write", ["git", "commit", "--allow-empty", "-m", "x"]),
    ("git-show-payload-blob", ["git", "show", f"HEAD:cache/{NOPE}.npz"]),
    ("git-diff-output", ["git", "diff", f"--output={W}/{NOPE}.txt"]),
    ("git-C-other-repo", ["git", "-C", "/data/yschoi/gdna_anchor_refit_v9r6", "rev-parse", "HEAD"]),
]


@pytest.mark.parametrize("name,argv", DENY_CHILDREN, ids=[c[0] for c in DENY_CHILDREN])
def test_deny_child(name, argv):
    with pytest.raises(PermissionError, match="boundary guard"):
        subprocess.run(argv, capture_output=True, cwd=W)


def test_deny_an_unhooked_python_child(tmp_path):
    with pytest.raises(PermissionError, match="boundary guard"):
        subprocess.run([PY, "-c", "pass"], env={"PATH": os.environ["PATH"]}, capture_output=True)


def test_deny_an_isolated_fixture_that_names_a_project_module(tmp_path):
    fixture = tmp_path / "verifier.py"
    fixture.write_text("import scripts.phase3_selection_matrix\n")
    with pytest.raises(PermissionError, match="boundary guard"):
        subprocess.run([PY, "-I", "-B", str(fixture)], capture_output=True, cwd=str(tmp_path))


IN_CHILD = [
    ("python-c-reads-worktree-result", [PY, "-c", f"open('{W}/result/{NOPE}/config.pt','rb')"], W),
    ("analysis-relative-dir-in-worktree",
     [PY, f"{W}/scripts/eval_cell_bioproj.py", "--dir", f"result/{NOPE}", "--dataset", "CIFAR10", "--K", "64"], W),
    ("python-c-relative-write-in-worktree-cwd", [PY, "-c", f"open('{NOPE}-out.json','w')"], W),
    ("python-c-writes-live-root", [PY, "-c", f"open('/home/yschoi/gdna_anchorRTrec_ops/{NOPE}','w')"], W),
]


@pytest.mark.parametrize("name,argv,cwd", IN_CHILD, ids=[c[0] for c in IN_CHILD])
def test_deny_inside_the_guarded_child(tmp_path, name, argv, cwd):
    argv = [a.replace("PRIVATE", str(tmp_path)) for a in argv]
    before = set(Path(os.environ["GDNA_GUARD_LOGDIR"]).glob("violations.*.jsonl"))
    done = subprocess.run(argv, capture_output=True, text=True, cwd=cwd)
    assert done.returncode != 0
    assert "boundary guard refused" in done.stderr
    after = set(Path(os.environ["GDNA_GUARD_LOGDIR"]).glob("violations.*.jsonl"))
    assert after - before or any(f.stat().st_size for f in after)


def test_decisions_without_execution(tmp_path):
    # the capture-only render: allowed only with the reviewed shim under the private root
    shim = tmp_path / "capture_argv"
    shim.write_text(f"#!{os.path.realpath(PY)}\n{G._shim_text()}")
    out = tmp_path / "argv.json"
    wrapper = f"{W}/scripts/train_flickr25k_v185_bidirTokenPrune05_clip.sh"
    env = dict(os.environ, PY=str(shim), GDNA_RENDER_ARGV_OUT=str(out))
    assert G.child_verdict(["bash", wrapper], str(tmp_path), env) == "allow-capture-render"
    shim.write_text(f"#!{os.path.realpath(PY)}\nimport runpy\n")
    assert G.child_verdict(["bash", wrapper], str(tmp_path), env) == "deny"
    assert G.child_verdict(["bash", wrapper], W, dict(env, PY=str(tmp_path / "capture_argv"))) == "deny"
    # analysis scripts are admitted only hooked; their opens are then policed in the child
    assert G.child_verdict([PY, f"{W}/scripts/pairwise_nmi.py", "--help"], str(tmp_path), None) == \
        "allow-hooked-worktree-script"
    assert G.child_verdict([PY, f"{W}/scripts/pairwise_nmi.py", "--help"], str(tmp_path), {"PATH": "x"}) == "deny"


def test_allow_controls(tmp_path):
    subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=W, capture_output=True, check=True)
    subprocess.run(["git", "show", "HEAD:scripts/anchor_refit_stage.py"], cwd=W, capture_output=True, check=True)
    subprocess.run([PY, "-c", "print(1)"], capture_output=True, check=True)
    fixture = tmp_path / "verifier.py"
    fixture.write_text(f"open({str(tmp_path / 'marker')!r}, 'w').write('ok')\n")
    subprocess.run([PY, "-I", "-B", str(fixture)], capture_output=True, check=True, cwd=str(tmp_path))
    assert (tmp_path / "marker").read_text() == "ok"
    assert Path(f"{W}/scripts/anchor_refit_stage.py").read_bytes()
    (tmp_path / "x.npz").write_bytes(b"synthetic")
    assert (tmp_path / "x.npz").read_bytes() == b"synthetic"


def test_content_bound_forms_refuse_altered_fixtures(tmp_path):
    """Decision-level (non-executing): each content-bound child form is admitted only with its reviewed bytes."""
    probe = tmp_path / "entry_probe.py"
    probe.write_text(G._reviewed_entry_probe())
    argv = [PY, str(probe), W, str(tmp_path / "result"), "nothing-imported-first"]
    assert G.child_verdict(argv, str(tmp_path), {"PATH": "x"}) == "allow-reviewed-entry-probe"
    probe.write_text(G._reviewed_entry_probe() + "\nimport subprocess\n")
    assert G.child_verdict(argv, str(tmp_path), {"PATH": "x"}) == "deny"
    driver = tmp_path / "driver.py"
    driver.write_text(G._test_constant("tests/test_anchor_confirm_env_handoff.py", "DRIVER"))
    dargv = [PY, str(driver), W, str(tmp_path / "o.json"), "flickr25k", "[]", "0.02", "{}"]
    assert G.child_verdict(dargv, str(tmp_path), {"PATH": "x"}) == "allow-reviewed-handoff-driver"
    driver.write_text(G._test_constant("tests/test_anchor_confirm_env_handoff.py", "DRIVER") + "\n# changed\n")
    assert G.child_verdict(dargv, str(tmp_path), {"PATH": "x"}) == "deny"
    hist = tmp_path / "hist" / "scripts"
    hist.mkdir(parents=True)
    copy = hist / "seal_phase3_inputs.py"
    copy.write_bytes(Path(f"{W}/scripts/seal_phase3_inputs.py").read_bytes())
    sargv = [PY, "-I", "-B", str(copy), "verify", "--seal", str(tmp_path / "s.json")]
    assert G.child_verdict(sargv, str(tmp_path), None) == "allow-isolated-reviewed-copy"
    assert G.child_verdict(sargv[:-1] + ["/home/yschoi/gdna_anchorRT_result/s.json"], str(tmp_path), None) == "deny"
    copy.write_bytes(copy.read_bytes() + b"\n# changed\n")
    assert G.child_verdict(sargv, str(tmp_path), None) == "deny"
    stand_in = tmp_path / "verifier.py"
    stand_in.write_text("import json, os\nopen('/home/yschoi/gdna_anchorRT_result/x.json')\n")
    assert G.child_verdict([PY, "-I", "-B", str(stand_in)], str(tmp_path), None) == "deny"
    # the launcher's self-exec is admitted only with an environment that keeps the hook
    launcher = f"{W}/scripts/phase3_selection_matrix.py"
    assert G._exec_ok((PY, [PY, launcher, "--plan"], dict(os.environ))) is True
    assert G._exec_ok((PY, [PY, launcher, "--plan"], {"PATH": "x"})) is False
    assert G._exec_ok((PY, [PY, f"{W}/scripts/anchor_terminal_test.py"], dict(os.environ))) is False
