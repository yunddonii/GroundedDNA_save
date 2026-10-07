"""Preserve the per-execution boundary records of the battery v14 run at 3fd0117 (audit 822).

The battery (mutation_v14.py) ran each declared test through sandboxed_pytest_r8.py and kept one record directory
per execution, <battery-root>/guard/<counter>/, in a temporary root. This copies, byte for byte, exactly the five
named files of each of the 113 executions -- result.json, probe.json, bwrap.json, git_store.json, pytest.log -- and
nothing else: the sandbox git object store (store.git), the private /tmp (tmp) and the worktree git directory
(worktree_gitdir) are left out, by name, and any other entry refuses.

Counter order is fixed by mutation_v14.main(): counters 1..54 are report["baseline"] in order; counters 55..113 are
each mutant's declared tests in report["mutants"] order, whose case log is logs/<mutant number:02d>_<case>.log.
This cross-checks that order against the run's own bytes: each mutant case's archived log must end with that
counter's pytest.log, every record must name the tested HEAD and runner, the rc in result.json must equal the
report's rc, and the module digests must equal the 3fd0117 source except the one mutated file of a mutant case.
Limit: a baseline record's content cannot tell which of the 54 tests it ran (each says "1 passed"); its test name
comes from the counter order alone, supported by non-decreasing record times.

Usage: archive_battery_guard_records.py <battery-root> <archived-battery-dir> <destination-dir>
Writes <destination>/<counter>/<five files>, <destination>/INDEX.json and <destination>/SHA256SUMS; refuses an
existing destination.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

NAMED = ("result.json", "probe.json", "bwrap.json", "git_store.json", "pytest.log")
EXCLUDED = ("store.git", "tmp", "worktree_gitdir")
TESTED = "3fd0117ae06642283f4ee185f073dd37800a6a70"
RUNNER = "a684299cde9c27bf7cd717a6d50bd3ce26e70fa9678596af710a5351f9a86732"
TREE = Path(__file__).resolve().parents[4]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tested_digest(rel):
    data = subprocess.run(["git", "-C", str(TREE), "show", f"{TESTED}:{rel}"], check=True, capture_output=True).stdout
    return hashlib.sha256(data).hexdigest()


def main(root, archived, dest):
    root, archived, dest = Path(root).resolve(), Path(archived).resolve(), Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    report_bytes = (root / "report.json").read_bytes()
    assert report_bytes == (archived / "report.json").read_bytes(), "the temporary report is not the archived one"
    report = json.loads(report_bytes)
    assert report["commit"] == "3fd0117", report["commit"]
    order = [{"phase": "baseline", "test": row["test"], "report_rc": row["rc"], "mutated_file": None,
              "mutant": None, "archived_log": None} for row in report["baseline"]]
    for number, mutant in enumerate(report["mutants"]):
        for case, row in enumerate(mutant["tests"]):
            order.append({"phase": "mutant", "test": row["test"], "report_rc": row["rc"],
                          "mutant": mutant["mutant"], "mutant_number": number, "case": case,
                          "mutated_file": mutant["file"], "archived_log": f"logs/{number:02d}_{case}.log"})
    counters = sorted(int(p.name) for p in (root / "guard").iterdir())
    assert counters == list(range(1, len(order) + 1)), f"counters {counters[:3]}..{counters[-3:]} vs {len(order)}"
    pins, problems, index, mtimes = {}, [], [], []
    for counter, row in zip(counters, order):
        src = root / "guard" / str(counter)
        entries = {p.name: p for p in src.iterdir()}
        extra = sorted(set(entries) - set(NAMED) - set(EXCLUDED))
        missing = [n for n in NAMED if n not in entries or entries[n].is_symlink() or not entries[n].is_file()]
        if extra or missing:
            problems.append(f"{counter}: extra {extra} missing/non-regular {missing}")
            continue
        result = json.loads((src / "result.json").read_text())
        log = (src / "pytest.log").read_text(errors="replace")
        # a mutant case runs with its one mutated file in place, so its tree is clean exactly when it is a baseline
        checks = {"tree_head": result.get("tree_head") == TESTED,
                  "tree_clean_iff_baseline": result.get("tree_status_clean") is (row["phase"] == "baseline"),
                  "runner": result.get("runner_sha256") == RUNNER, "rc": result.get("pytest_rc") == row["report_rc"],
                  "pytest_log_nonempty": bool(log.strip())}
        probe = result.get("probe") or {}
        checks["boundary"] = (probe.get("network") is False and probe.get("nvidia_devices") == []
                              and probe.get("tree_writable") is False and probe.get("visible_data_entries") == []
                              and all(probe.get("absent", {}).values()) and bool(probe.get("absent")))
        deviating = []
        for module, ident in (probe.get("module_identities") or {}).items():
            rel = ident["file"].split("/sandbox/", 1)[-1]
            pins.setdefault(rel, tested_digest(rel))
            if not ident.get("inside_tree") or "/sandbox/" not in ident["file"]:
                problems.append(f"{counter}: {module} imported outside the sandbox")
            if ident["sha256"] != pins[rel]:
                deviating.append(rel)
        # every v14 mutant edits one of the five recorded modules: exactly that one must be off its 3fd0117 bytes
        checks["modules"] = len(probe.get("module_identities") or {}) == 5 and deviating == (
            [] if row["phase"] == "baseline" else [row["mutated_file"]])
        if row["phase"] == "baseline":
            checks["order"] = "1 passed" in log
        else:
            text = (archived / row["archived_log"]).read_text(errors="replace")
            checks["order"] = text.endswith(log)
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            problems.append(f"{counter}: failed {bad}")
        out = dest / str(counter)
        out.mkdir()
        files = {}
        for name in NAMED:
            shutil.copyfile(src / name, out / name)
            files[name] = sha(out / name)
            assert files[name] == sha(src / name), f"{counter}/{name} changed in the copy"
        mtimes.append(os.stat(src / "result.json").st_mtime)
        index.append({"counter": counter, **row, "pytest_rc": result.get("pytest_rc"), "checks": checks,
                      "module_digests_off_3fd0117": deviating, "files": files,
                      "excluded_entries": sorted(set(entries) & set(EXCLUDED))})
    checks_order = all(a <= b for a, b in zip(mtimes, mtimes[1:]))
    if not checks_order:
        problems.append("result.json mtimes are not non-decreasing in counter order")
    summary = {"source_root": str(root), "tested_commit": TESTED, "runner_sha256": RUNNER,
               "report_sha256": hashlib.sha256(report_bytes).hexdigest(), "executions": len(index),
               "baseline": sum(r["phase"] == "baseline" for r in index),
               "mutant_cases": sum(r["phase"] == "mutant" for r in index),
               "named_files": list(NAMED), "excluded_by_name": list(EXCLUDED),
               "pinned_module_digests_at_3fd0117": pins, "mtime_order_non_decreasing": checks_order,
               "script_sha256": sha(__file__), "problems": problems, "records": index}
    (dest / "INDEX.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    lines = [f"{sha(p)}  ./{p.relative_to(dest)}" for p in sorted(dest.rglob("*"))
             if p.is_file() and p.name != "SHA256SUMS"]
    (dest / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"executions {len(index)} (baseline {summary['baseline']}, mutant cases {summary['mutant_cases']}); "
          f"files {len(lines)}; problems {len(problems)}: {problems[:5]}")
    return 0 if not problems and len(index) == 113 else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
