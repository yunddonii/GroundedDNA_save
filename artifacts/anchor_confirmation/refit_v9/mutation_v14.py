"""Mutation battery v14 (generation v9 r8: the stage-T recovery; audits 797-808).

v13's bounded method with three changes. (1) Every declared test runs inside the r8 operating-system
boundary (t_recovery_prep/sandboxed_pytest_r8.py): a bubblewrap sandbox in which only the interpreter, the
system runtime, the mutated sparse SANDBOX copy of the tree (read-only, data directories empty), a source-only
git store and a private /tmp exist -- no live data, result, cache, ledger, claim or lease path, no network, no
GPU. The tree under test is the sandbox itself, so the imported launcher, stage and mutated modules resolve
inside it (audit 807); the runner records the tree HEAD and its own digest. (2) The mutants are the recovery's
(t_recovery_prep/mutants_v14.py, RY1-RY47). (3) The report pins the final sources, the runner and the
mutant list. As before: content hashes ONLY for the reviewed inventory (the manifest closure plus the declared
test files), every other tracked file by git index object id and stat; the harness under its own open() guard;
a sparse sandbox without artifacts/; each mutant replaces exactly one pattern (each found exactly once before
any run) and disables a whole condition or step; every declared test must first PASS unmutated (baseline),
and a mutant counts as detected only if every declared test fails (rc 1) with its declared text in the output
and the sandbox probe confirmed the boundary. Not mutated, and why: admit_stopped_campaign's explicit
pre-execution absence check (redundant by design with the stage-R admission's own withheld check for the
non-carried cell: a mutant there is equivalent); the T entry's scope expression (`RECOVERY_SCOPE if recovery
else f"stage-T-{mode}"` yields the same string for mode recovery: equivalent).
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

WT = Path("/data/yschoi/gdna_anchor_refit_v9r8")
COMMIT, OUT = sys.argv[1], Path(sys.argv[2])
PY = "/home/yschoi/.conda/envs/dna_hashing/bin/python"

sys.path.insert(0, str(Path(__file__).resolve().parent / "t_recovery_prep"))
from mutants_v14 import MUTANTS  # noqa: E402

GUARD = WT / "artifacts/anchor_confirmation/refit_v9/t_recovery_prep/sandboxed_pytest_r8.py"
BOUNDED = WT / "artifacts/anchor_confirmation/refit_v9/bounded_tree.py"
MANIFEST = WT / "artifacts/anchor_confirmation/authority_manifest_v9r8.json"
MUTANTS_FILE = WT / "artifacts/anchor_confirmation/refit_v9/t_recovery_prep/mutants_v14.py"
EXTRAS = sorted({t[0] for *_x, d in MUTANTS for t, _m in d})
BINARY = (".npz", ".npy", ".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".pkl")
REFUSED, CHILDREN = [], []


def _harness_hook(event, args):
    if event == "os.exec":
        REFUSED.append(f"exec:{args[0]}")
        raise PermissionError("battery guard refused a process image replacement")
    if event == "subprocess.Popen":
        CHILDREN.append([os.fsdecode(a) for a in (args[1] or [])][:4])
        return
    if event != "open" or not args or isinstance(args[0], int):
        return
    try:
        path = os.path.abspath(os.fsdecode(args[0]))
    except (TypeError, ValueError):
        return
    safe = path.startswith(("/tmp/", sys.prefix + "/", sys.base_prefix + "/"))
    in_wt = path.startswith(str(WT) + "/")
    allowed_wt = in_wt and (not path.startswith(str(WT / "artifacts") + "/")
                            or path in (str(MANIFEST), str(BOUNDED), str(GUARD), str(MUTANTS_FILE), os.path.abspath(__file__))
                                  )
    deny = (path.endswith(BINARY) and not safe) or \
        (path.startswith(("/data/", "/home/")) and not safe and not allowed_wt)
    if deny:
        REFUSED.append(path)
        raise PermissionError(f"battery guard refused an open: {path}")


import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("bounded_tree", BOUNDED)
BT = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(BT)


def tree_state(root):
    return BT.state(root, BT.inventory(root, MANIFEST, EXTRAS))


def tracked_state():
    return tree_state(WT), BT.head_clean(WT, BT.inventory(WT, MANIFEST, EXTRAS))


RUN = [0]
COMMIT_SHA = [None]


def run_test(box, test, timeout=600):
    RUN[0] += 1
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GDNA_NUM_SEMANTIC_PARTS="5")
    env.pop("PYTHONPATH", None)
    env.pop("GDNA_ALLOW_REAL_ARTIFACT_TESTS", None)
    guard_dir = OUT / "guard" / str(RUN[0])
    began = time.monotonic()
    try:
        done = subprocess.run([PY, str(GUARD), str(guard_dir), str(box), "--", "-q", f"{test[0]}::{test[1]}"],
                              cwd=box, env=env, capture_output=True, text=True, timeout=timeout)
        result = json.loads((guard_dir / "result.json").read_text())
        log = (guard_dir / "pytest.log").read_text(errors="replace") if (guard_dir / "pytest.log").exists() else ""
        refused = [] if "pytest_rc" in result else [result.get("refused", "no result")]
        if result.get("tree_head") != COMMIT_SHA[0]:
            refused.append(f"sandbox tree HEAD {result.get('tree_head')} is not the battery commit")
        identities = (result.get("probe") or {}).get("module_identities") or {}
        outside = sorted(n for n, v in identities.items() if not str(v.get("file", "")).startswith(str(box) + "/"))
        if not identities or outside:
            refused.append(f"modules not imported from the mutated sandbox: {outside or 'no identities'}")
        return done.returncode, done.stdout + done.stderr + log, time.monotonic() - began, refused
    except subprocess.TimeoutExpired as error:
        return None, f"TIMEOUT\n{error.stdout or ''}", time.monotonic() - began, ["timeout"]


def main():
    sys.addaudithook(_harness_hook)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "logs").mkdir()
    (OUT / "tmp").mkdir()
    (OUT / "guard").mkdir()
    before = tracked_state()
    box = OUT / "sandbox"
    # A sparse sandbox without artifacts/: a full checkout would materialise the unrelated tracked
    # binaries (v10 and v11 sandboxes did) although no declared test needs them.
    subprocess.run(["git", "-C", str(WT), "worktree", "add", "--no-checkout", "--detach", str(box), COMMIT],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(box), "sparse-checkout", "set", "--no-cone", "/*", "!/artifacts/"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(box), "checkout", "--detach", COMMIT], check=True, capture_output=True)
    report_sparse = sorted(p.name for p in box.iterdir())
    COMMIT_SHA[0] = subprocess.run(["git", "-C", str(box), "rev-parse", "HEAD"], check=True, capture_output=True,
                                   text=True).stdout.strip()
    report = {"commit": COMMIT, "baseline": [], "mutants": [], "sandbox_top_level": report_sparse,
              "sandbox_has_artifacts": (box / "artifacts").exists()}
    try:
        assert not before[1], f"the worktree is not clean at HEAD for the inventory: {before[1][:6]}"
        assert tree_state(box)["content"] == before[0]["content"], "the sandbox is not the worktree's inventory bytes"
        for name, rel, old, new, declared in MUTANTS:     # every pattern exactly once before any run
            assert (box / rel).read_text().count(old) == 1, f"{name}: pattern count is not 1"
        tests = sorted({t for *_x, declared in MUTANTS for t, _m in declared})
        for test in tests:
            rc, text, seconds, refused = run_test(box, test)
            ran = "1 passed" in text and not refused
            report["baseline"].append({"test": test[1], "rc": rc, "one_passed": ran, "refused_opens": refused,
                                       "seconds": round(seconds, 1)})
            print(f"baseline {'ok ' if rc == 0 and ran else 'BAD'} {test[1]} ({seconds:.0f}s)", flush=True)
            if rc != 0 or not ran:
                (OUT / "logs" / f"baseline_{len(report['baseline']):02d}.log").write_text(text)
        if any(row["rc"] != 0 or not row["one_passed"] for row in report["baseline"]):
            raise SystemExit("an unmutated declared test failed or did not run; no mutant is interpretable")
        for number, (name, rel, old, new, declared) in enumerate(MUTANTS):
            path = box / rel
            original = path.read_bytes()
            row = {"mutant": name, "file": rel, "tests": []}
            path.write_text(original.decode().replace(old, new))
            try:
                for test, marker in declared:
                    rc, output, seconds, refused = run_test(box, test)
                    intended = rc == 1 and marker in output and not refused
                    (OUT / "logs" / f"{number:02d}_{len(row['tests'])}.log").write_text(output)
                    row["tests"].append({"test": test[1], "rc": rc, "declared_text": marker, "refused_opens": refused,
                                         "intended": intended, "seconds": round(seconds, 1)})
            finally:
                path.write_bytes(original)
            assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(original).digest()
            row["result"] = ("detected" if row["tests"] and all(t["intended"] for t in row["tests"])
                             else "NOT DETECTED AS DECLARED")
            report["mutants"].append(row)
            print(f"{row['result']:26s} {name}", flush=True)
        report["sandbox_restored"] = tree_state(box)["content"] == before[0]["content"]
    finally:
        subprocess.run(["git", "-C", str(WT), "worktree", "remove", "--force", str(box)],
                       check=True, capture_output=True)
        stray = subprocess.run(["pgrep", "-af", str(OUT / "tmp")], capture_output=True,
                               text=True).stdout.strip()
        after = tracked_state()
        report.update({"inventory_sha256_equal": before[0]["content"] == after[0]["content"],
                       "other_tracked_identity_and_stat_equal": before[0]["other"] == after[0]["other"],
                       "head_clean_before": not before[1], "head_clean_after": not after[1],
                       "inventory": sorted(before[0]["content"]), "harness_refused_opens": REFUSED,
                       "harness_children": CHILDREN[:400], "stray_processes": stray})
        (OUT / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    detected = sum(1 for m in report["mutants"] if m["result"] == "detected")
    unchanged = before[0] == after[0] and before[1] == after[1] == []
    print(f"detected {detected}/{len(MUTANTS)}; inventory and checkout unchanged {unchanged}; "
          f"harness refused {len(REFUSED)}; stray {bool(stray)}")
    return 0 if detected == len(MUTANTS) and unchanged and not REFUSED and not stray else 1


if __name__ == "__main__":
    raise SystemExit(main())
