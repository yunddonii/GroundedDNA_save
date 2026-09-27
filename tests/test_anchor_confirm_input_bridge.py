"""Anchor confirmation: historical input verification across source generations (audit 715).

The stage-1 input seals were built in the historical tree and bind six of its sources (five caption-
foil producers and val_split.py) by path and stat. `verify_seal` rebuilds a seal with its own REPO,
so run from another worktree it refuses (anchor stage S, run 20260927T143532Z-4ada8a0d). These
tests build a REAL small seal with a byte copy of the verifier inside a temporary "historical" tree
and check it from this tree:

- the relocated-worktree refusal is reproduced in process;
- `verify_seal_historically` (the pinned historical verifier in an isolated child) admits it;
- altered or re-stamped historical sources refuse BEFORE any child starts (audit 716), and a
  change during the child refuses after it;
- a changed new-generation val_split.py, a substituted root or seal,
  an unpinned verifier, forged or stale reports, a seal or verifier changed across the handoff and
  a stale expected authority all refuse;
- the child dies with the launcher and with an interrupted wait;
- a stats-only or non-anchor check never takes the bridge.

Only fixtures in tmp_path are used; no real seal, cache or GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import scripts.phase3_selection_matrix as M                 # noqa: E402
from scripts.phase3_selection_matrix import CellRefused     # noqa: E402
from scripts import seal_phase3_inputs as S                  # noqa: E402
import test_seal_phase3_inputs as SEALFIX                   # noqa: E402

PY = sys.executable
SOURCES = sorted(M.HISTORICAL_PRODUCER_SOURCES)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_tree(root: Path) -> Path:
    """A historical tree: byte copies of the verifier and the six sources it binds."""
    for rel in [M.HISTORICAL_SEAL_VERIFIER, *SOURCES]:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, target)
    return root


def seal_in(root: Path, request: S.SealRequest, output: Path) -> Path:
    """Seal with the tree's OWN verifier copy, as the historical campaign did."""
    done = subprocess.run(
        [PY, "-I", "-B", str(root / M.HISTORICAL_SEAL_VERIFIER), "seal",
         "--dataset", request.dataset, "--stage", request.stage,
         "--dataset-root", request.dataset_root, "--feature-cache", request.feature_cache,
         "--foil-cache", request.foil_cache, "--whitening", request.whitening,
         "--qwen", request.qwen, "--split-rows", request.split_rows, "--output", str(output)],
        cwd=root, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return output


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A real seal built in tmp/hist; the launcher runs from REPO with the root pinned to it."""
    hist = copy_tree(tmp_path / "hist")
    fixture = SEALFIX._fixture(tmp_path / "inputs")
    path = seal_in(hist, fixture["request"], tmp_path / "flickr.stage1.seal.json")
    monkeypatch.setattr(M, "HISTORICAL_INPUT_ROOT", hist)
    return {"hist": hist, "seal": path, "request": fixture["request"], "tmp": tmp_path,
            "fixture": fixture}


def no_child(monkeypatch):
    """Any child start fails the test: a refusal must come before the rehash."""
    def refuse(*a, **k):
        raise AssertionError("the historical verifier was started")
    monkeypatch.setattr(M.subprocess, "Popen", refuse)


def fake_verifier(world, monkeypatch, body: str) -> Path:
    """Replace the tree's verifier with a stand-in and pin its bytes (the pin is tested alone)."""
    verifier = world["hist"] / M.HISTORICAL_SEAL_VERIFIER
    verifier.write_text("import json, os, sys\nseal = sys.argv[sys.argv.index('--seal') + 1]\n"
                        "agg = json.load(open(seal))['aggregate_digest']['sha256']\n" + body)
    monkeypatch.setattr(M, "HISTORICAL_SEAL_VERIFIER_SHA256", sha(verifier))
    return verifier


# ---- the relocated worktree: the failure, and its repair -------------------------------------------
def test_the_in_process_full_check_refuses_a_seal_built_in_another_tree(world):
    """The audit-715 failure, reproduced: this tree's REPO is not the tree that sealed the inputs."""
    with pytest.raises(S.SealError, match=r"drifted: \$seal\.inputs\.foil_derivation\.producer_sources"):
        S.verify_seal(world["seal"])


def test_the_same_tree_full_check_still_passes(world):
    """Legacy behaviour is unchanged: the historical tree's verifier admits its own seal."""
    done = subprocess.run([PY, "-I", "-B", str(world["hist"] / M.HISTORICAL_SEAL_VERIFIER), "verify",
                           "--seal", str(world["seal"])], cwd=world["hist"], capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0 and done.stdout.startswith("verified "), done.stderr


def test_the_historical_verifier_admits_the_relocated_seal(world):
    evidence = {}
    authority = M.verify_seal_historically(world["seal"], evidence=evidence)
    loaded = json.loads(world["seal"].read_text())
    assert authority == S.authority_from_verified_seal(world["seal"], loaded)
    assert authority["seal_file_sha256"] == sha(world["seal"])
    row = evidence["flickr25k:stage1"]
    assert row["returncode"] == 0 and row["seal"]["sha256"] == sha(world["seal"])
    assert row["report"] == f"verified {world['seal']} {loaded['aggregate_digest']['sha256']}"
    assert row["argv"][1:3] == ["-I", "-B"] and row["root"] == str(world["hist"])
    assert row["verifier"]["sha256"] == M.HISTORICAL_SEAL_VERIFIER_SHA256
    inputs = loaded["inputs"]
    assert row["historical_sources"] == {**inputs["foil_derivation"]["producer_sources"],
                                         "val_split.py": inputs["protocol_sources"]["val_split"]}
    # the stats-only recheck that follows admission reads the recorded (historical) paths
    S.verify_seal_stats(world["seal"], expected_aggregate_sha256=authority["aggregate_sha256"])


def test_the_verifier_copy_is_the_pinned_historical_bytes():
    """The pin names the historical verifier; this tree's copy is byte-identical to it."""
    assert sha(REPO / M.HISTORICAL_SEAL_VERIFIER) == M.HISTORICAL_SEAL_VERIFIER_SHA256
    for rel, digest in M.HISTORICAL_PRODUCER_SOURCES.items():
        assert sha(REPO / rel) == digest, rel
    assert set(M.HISTORICAL_PRODUCER_SOURCES) == {*S._FOIL_DERIVATION_SOURCES, "val_split.py"}


# ---- refusals before any rehash --------------------------------------------------------------------
def test_an_unpinned_verifier_refuses_before_the_child(world, monkeypatch):
    no_child(monkeypatch)
    monkeypatch.setattr(M, "HISTORICAL_SEAL_VERIFIER_SHA256", "0" * 64)
    with pytest.raises(S.SealError, match="is not the pinned bytes"):
        M.verify_seal_historically(world["seal"])


@pytest.mark.parametrize("rel", SOURCES)
def test_a_seal_binding_another_producer_digest_refuses_before_the_child(world, monkeypatch, rel):
    """Every one of the six sources is checked, not only the first reported difference."""
    no_child(monkeypatch)
    monkeypatch.setitem(M.HISTORICAL_PRODUCER_SOURCES, rel, "f" * 64)
    with pytest.raises(S.SealError, match="not the pinned historical source|another protocol source"):
        M.verify_seal_historically(world["seal"])


def test_a_substituted_root_refuses_before_the_child(world, monkeypatch):
    """An identical copy of the tree elsewhere is not the root the seal names."""
    no_child(monkeypatch)
    other = copy_tree(world["tmp"] / "elsewhere")
    monkeypatch.setattr(M, "HISTORICAL_INPUT_ROOT", other)
    with pytest.raises(S.SealError, match="not the pinned historical source"):
        M.verify_seal_historically(world["seal"])


def test_a_seal_built_in_this_tree_refuses_before_the_child(world, monkeypatch):
    """A seal whose producers are this worktree's copies is not a historical seal."""
    no_child(monkeypatch)
    local = S.create_seal(world["request"], world["tmp"] / "local.json")
    assert local["inputs"]["protocol_sources"]["val_split"]["path"] == str(REPO / "val_split.py")
    with pytest.raises(S.SealError, match="not the pinned historical source"):
        M.verify_seal_historically(world["tmp"] / "local.json")


@pytest.mark.parametrize("edit", ["extra_producer", "missing_producer", "split_source"])
def test_a_seal_with_other_source_records_refuses_before_the_child(world, monkeypatch, edit):
    """Edited JSON with a consistent aggregate still refuses on the source records themselves."""
    no_child(monkeypatch)
    body = json.loads(world["seal"].read_text())
    producers = body["inputs"]["foil_derivation"]["producer_sources"]
    if edit == "extra_producer":
        producers["scripts/other.py"] = dict(producers[SOURCES[0]])
    elif edit == "missing_producer":
        producers.pop(SOURCES[0])
    else:
        body["inputs"]["split_identity"]["protocol_source"]["path"] = str(REPO / "val_split.py")
    body.pop("aggregate_digest")
    body = S.with_aggregate(body)
    path = world["tmp"] / "edited.json"
    path.write_text(json.dumps(body))
    with pytest.raises(S.SealError, match="producer sources|another protocol source"):
        M.verify_seal_historically(path)


def test_a_seal_whose_aggregate_does_not_hold_refuses_before_the_child(world, monkeypatch):
    no_child(monkeypatch)
    body = json.loads(world["seal"].read_text())
    body["inputs"]["qwen_jsonl"]["path"] = "/elsewhere.jsonl"
    path = world["tmp"] / "tampered.json"
    path.write_text(json.dumps(body))
    with pytest.raises(S.SealError, match="aggregate digest does not match"):
        M.verify_seal_historically(path)


def test_a_changed_new_generation_val_split_refuses_before_the_child(world, monkeypatch):
    """This generation's own split carve must be the sealed protocol source."""
    no_child(monkeypatch)
    here = world["tmp"] / "newgen"
    here.mkdir()
    (here / "val_split.py").write_bytes((REPO / "val_split.py").read_bytes() + b"# changed\n")
    monkeypatch.setattr(M, "REPO", here)
    with pytest.raises(S.SealError, match="is not the sealed protocol source"):
        M.verify_seal_historically(world["seal"])


# ---- the child's own refusal and a forged, stale or mutated handoff --------------------------------
@pytest.mark.parametrize("rel", SOURCES)
def test_an_altered_historical_source_refuses_before_the_child(world, monkeypatch, rel):
    """Audit 716: the pins still match the seal JSON, the physical file does not (content). No
    historical code may run -- the verifier executes val_split.py -- and no rehash is spent."""
    target = world["hist"] / rel
    target.write_bytes(target.read_bytes() + b"# altered\n")
    no_child(monkeypatch)
    with pytest.raises(S.SealError, match=r"^before the historical verifier: the historical source "
                                          r".* is not its sealed identity"):
        M.verify_seal_historically(world["seal"])


@pytest.mark.parametrize("rel", SOURCES)
def test_a_restamped_historical_source_refuses_before_the_child(world, monkeypatch, rel):
    """Same bytes, another mtime: the recorded stat identity is checked before the child too."""
    target = world["hist"] / rel
    stamp = target.stat()
    os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10**9))
    no_child(monkeypatch)
    with pytest.raises(S.SealError, match=r"^before the historical verifier: .* is not its sealed "
                                          r"identity: .*stat"):
        M.verify_seal_historically(world["seal"])


@pytest.mark.parametrize("rel", SOURCES)
def test_a_historical_source_changed_during_the_child_refuses_after_it(world, monkeypatch, rel):
    """Bound across the handoff: a change after the pre-child check is caught after the child."""
    target = world["hist"] / rel
    fake_verifier(world, monkeypatch,
                  f"st = os.stat({str(target)!r})\n"
                  f"os.utime({str(target)!r}, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))\n"
                  "print(f'verified {seal} {agg}')\n")
    with pytest.raises(S.SealError, match=r"^after the historical verifier: the historical source"):
        M.verify_seal_historically(world["seal"])


def test_altered_payload_bytes_are_refused_by_the_rebuild(world):
    """Full-verification semantics are the historical verifier's: a changed input byte refuses."""
    victim = world["fixture"]["qwen"]
    victim.write_bytes(victim.read_bytes() + b"\n")
    with pytest.raises(S.SealError, match=r"historical verifier refused \(rc 2\)"):
        M.verify_seal_historically(world["seal"])


@pytest.mark.parametrize("body,reason", [
    ("print(f'verified {seal} {\"0\" * 64}')\n", "report is not"),                      # another aggregate
    ("print(f'verified {seal}.other {agg}')\n", "report is not"),                       # another seal
    ("pass\n", "report is not"),                                                         # silent success
    ("print(f'verified {seal} {agg}')\nprint('extra')\n", "report is not"),             # extra output
    ("print(f'verified {seal} {agg}')\nsys.exit(2)\n", r"refused \(rc 2\)"),           # nonzero exit
    ("os.chmod(seal, 0o644)\nopen(seal, 'a').write(' ')\nprint(f'verified {seal} {agg}')\n",
     "changed while the historical verifier ran"),                                       # seal mutated
    ("open(__file__, 'a').write('#')\nprint(f'verified {seal} {agg}')\n",
     "changed while it ran"),                                                            # verifier mutated
])
def test_a_forged_or_mutated_handoff_refuses(world, monkeypatch, body, reason):
    fake_verifier(world, monkeypatch, body)
    with pytest.raises(S.SealError, match=reason):
        M.verify_seal_historically(world["seal"])


def test_an_honest_stand_in_is_admitted_positive_control(world, monkeypatch):
    """The forged cases above differ from this one only in the line under test."""
    fake_verifier(world, monkeypatch, "print(f'verified {seal} {agg}')\n")
    assert M.verify_seal_historically(world["seal"])["seal_file_sha256"] == sha(world["seal"])


def test_a_stale_expected_authority_refuses(world):
    authority = M.verify_seal_historically(world["seal"])
    stale = {**authority, "aggregate_sha256": "0" * 64}
    with pytest.raises(S.SealError, match="authority changed across the historical verification"):
        M.verify_seal_historically(world["seal"], expected=stale)
    assert M.verify_seal_historically(world["seal"], expected=authority) == authority


# ---- lifecycle: the child never outlives the launcher -----------------------------------------------
def alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().split(") ")[1][0] not in "ZX"
    except (FileNotFoundError, IndexError):
        return False


def wait_for(path: Path, seconds=60.0) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def test_the_child_dies_with_a_killed_launcher(world, monkeypatch):
    """Before any lease the launcher has no signal handler: a supervisor stop kills it outright."""
    marker = world["tmp"] / "child.pid"
    verifier = fake_verifier(world, monkeypatch,
                             f"open({str(marker)!r}, 'w').write(str(os.getpid()))\nimport time\n"
                             "time.sleep(120)\n")
    driver = world["tmp"] / "driver.py"
    driver.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "import scripts.phase3_selection_matrix as M\n"
        f"M.HISTORICAL_INPUT_ROOT = Path({str(world['hist'])!r})\n"
        f"M.HISTORICAL_SEAL_VERIFIER_SHA256 = {sha(verifier)!r}\n"
        f"M.verify_seal_historically({str(world['seal'])!r})\n")
    parent = subprocess.Popen([PY, str(driver)], env={**os.environ, "GDNA_NUM_SEMANTIC_PARTS": "5"})
    child = None
    try:
        child = wait_for(marker)
        assert alive(child)
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=30)
        deadline = time.monotonic() + 10
        while alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not alive(child), "the historical verifier outlived its launcher"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child is not None and alive(child):
            os.kill(child, signal.SIGKILL)


def test_an_interrupted_wait_kills_the_child(world, monkeypatch):
    marker = world["tmp"] / "child.pid"
    fake_verifier(world, monkeypatch, f"open({str(marker)!r}, 'w').write(str(os.getpid()))\n"
                                      "import time\ntime.sleep(120)\n")
    real = subprocess.Popen.communicate

    def interrupted(self, *a, **k):
        wait_for(marker)
        raise KeyboardInterrupt
    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        M.verify_seal_historically(world["seal"])
    child = int(marker.read_text())
    monkeypatch.setattr(subprocess.Popen, "communicate", real)
    try:
        assert not alive(child), "an interrupted wait left the historical verifier running"
    finally:
        if alive(child):                    # never leave a stray, even when the assertion fails
            os.kill(child, signal.SIGKILL)


# ---- which checks take the bridge --------------------------------------------------------------------
CELL = ("flickr25k", 4, ("0.6", "0.95"), "0.02", "select", 42, (), "anchors")


def canonical_clip(monkeypatch, authority):
    hf = authority["hf_runtime"]
    monkeypatch.setattr(M, "PHASE3_CLIP_CHECKPOINT", hf["checkpoint"])
    monkeypatch.setattr(M, "PHASE3_CLIP_REVISION", hf["revision"])
    monkeypatch.setattr(M, "PHASE3_CLIP_WEIGHT_SHA256", hf["weight_sha256"])


def test_the_campaign_check_takes_the_bridge_only_for_a_full_anchor_admission(world, monkeypatch):
    specs = {("flickr25k", "stage1"): str(world["seal"])}
    authority = M.verify_seal_historically(world["seal"])
    canonical_clip(monkeypatch, authority)
    evidence = {}
    admitted = M.verify_campaign_input_seals(specs, [CELL], full=True, historical=True, evidence=evidence)
    assert admitted == {"flickr25k:stage1": authority} and set(evidence) == {"flickr25k:stage1"}
    calls = []
    monkeypatch.setattr(M, "verify_seal_historically", lambda *a, **k: calls.append("bridge"))
    # stats-only after admission, and every non-anchor full check, keep the in-process verifier
    assert M.verify_campaign_input_seals(specs, [CELL], full=False, historical=True,
                                         expected=admitted) == admitted
    with pytest.raises(CellRefused, match=r"refused: .*drifted"):
        M.verify_campaign_input_seals(specs, [CELL], full=True)
    assert calls == []


def test_a_bridge_refusal_is_a_campaign_refusal(world, monkeypatch):
    monkeypatch.setattr(M, "HISTORICAL_SEAL_VERIFIER_SHA256", "0" * 64)
    with pytest.raises(CellRefused, match="input seal flickr25k:stage1 refused: .*not the pinned bytes"):
        M.verify_campaign_input_seals({("flickr25k", "stage1"): str(world["seal"])}, [CELL],
                                      full=True, historical=True)
