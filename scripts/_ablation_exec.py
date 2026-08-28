"""Run one planned ablation cell, with exactly the plan's environment (F09).

The child is `prompt_ablation_A_cell_fixedN.sh`, which reads a dozen variables
from its environment -- and several of them change the experiment rather than
the logging. An inherited `NUM_CODONS=4` turns the paper's 15 bases into 20;
`CURVE=1` drops FINAL_EPOCH and swaps the isolated refit for the test-monitored
path; `K` changes the capacity of every non-A4 cell; `WHITEN_VARIANT` changes
the transform. So the child gets `env -i` plus the plan's allow-list, filled
from the values the plan RECORDED, and nothing else.

THE PLAN IS INPUT, NOT AUTHORITY. The first version read whatever JSON it was
handed and ran it: a forged plan with a negative index (which Python resolves
from the end of the list), a foreign `runner`, and arbitrary live environment
all executed at rc 0, and a 24-entry plan holding one cell twice passed the
chain's `len == expected` check while covering only 23 configurations. Every one
of those is refused here, and the plan's own digest is re-derived and compared
against the digest the chain computed when it built the file -- so a plan edited
between `--out` and the twenty-fourth cell does not run.

It also runs the child under `bash -o pipefail`. That is necessary and not
sufficient: shell options are not exported, so the wrapper has to set its own
boundary before starting the trainer, which it now does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = 2


class PlanRejected(RuntimeError):
    """The plan cannot be trusted to say what should run."""


def _digest_of(plan: dict) -> str:
    """The plan's digest is over the plan WITHOUT its digest field."""
    body = {k: v for k, v in plan.items() if k != "plan_digest"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_plan(path: Path, *, expect_sha256: str | None) -> dict:
    raw = path.read_bytes()
    if expect_sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expect_sha256:
            raise PlanRejected(
                f"{path} hashes to {actual[:16]}..., but the campaign was "
                f"started against {expect_sha256[:16]}...; the plan changed "
                f"after the run began")
    try:
        plan = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise PlanRejected(f"{path}: {error}") from None
    if not isinstance(plan, dict):
        raise PlanRejected(f"{path} is not an object")

    if plan.get("schema_version") != SCHEMA_VERSION:
        raise PlanRejected(
            f"{path}: schema_version {plan.get('schema_version')!r}, not "
            f"{SCHEMA_VERSION}")
    stored = plan.get("plan_digest")
    if not isinstance(stored, str) or stored != _digest_of(plan):
        raise PlanRejected(
            f"{path}: plan_digest does not match the plan it is in; the file "
            f"was edited after it was written")

    cells = plan.get("cells")
    expected = plan.get("expected_cells")
    if not isinstance(cells, list) or not isinstance(expected, int) \
            or isinstance(expected, bool):
        raise PlanRejected(f"{path}: no cell list, or no expected count")
    if len(cells) != expected:
        raise PlanRejected(
            f"{path}: {len(cells)} cells, expected {expected}")
    # Uniqueness is checked over the tag, which is the result directory the cell
    # will claim. `len(cells) == expected` says nothing about coverage: a plan
    # holding one cell twice has the right length and one fewer configuration.
    for key in ("env_passthrough", "env_pinned", "runners"):
        if not isinstance(plan.get(key), list):
            raise PlanRejected(f"{path}: {key} is not a list")
    tags = [c.get("tag") for c in cells if isinstance(c, dict)]
    if len(tags) != len(cells):
        raise PlanRejected(f"{path}: a cell is not an object")
    if len(set(tags)) != len(tags):
        duplicated = sorted({t for t in tags if tags.count(t) > 1})
        raise PlanRejected(
            f"{path}: {duplicated} appear more than once; two cells cannot "
            f"claim one result directory")
    return plan


def cell_at(plan: dict, index: int) -> dict:
    """Bounds-checked, and non-negative: `cells[-1]` is a real element."""
    cells = plan["cells"]
    if index < 0 or index >= len(cells):
        raise PlanRejected(
            f"index {index} is not in 0..{len(cells) - 1}"
            + ("; a negative index silently selects from the end of the list"
               if index < 0 else ""))
    return cells[index]


def child_env(plan: dict, cell: dict) -> dict:
    """`env -i` plus the plan's allow-list at the plan's recorded values."""
    allowed = set(plan["env_passthrough"])
    pinned = set(plan["env_pinned"])
    values = plan.get("env_passthrough_values")
    if not isinstance(values, dict):
        raise PlanRejected("the plan records no env_passthrough_values")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise PlanRejected(
            f"env_passthrough_values carries {unknown}, which the plan's own "
            f"allow-list does not permit")

    cell_env = cell.get("env")
    if not isinstance(cell_env, dict):
        raise PlanRejected(f"{cell.get('tag')!r}: no environment")
    stray = sorted(set(cell_env) - pinned)
    if stray:
        raise PlanRejected(
            f"{cell.get('tag')!r} sets {stray}, which the plan does not list "
            f"as pinned; a plan cannot introduce a variable the review did not "
            f"see")

    env = dict(values)
    env.update(cell_env)
    return env


def child_argv(plan: dict, cell: dict, gpu: str) -> list:
    runner = cell.get("runner")
    if runner not in set(plan.get("runners") or ()):
        raise PlanRejected(
            f"{cell.get('tag')!r} names runner {runner!r}, which is not one of "
            f"the plan's {sorted(plan.get('runners') or ())}")
    exp = cell.get("exp")
    if not isinstance(exp, str) or not exp:
        raise PlanRejected(f"{cell.get('tag')!r}: no experiment name")
    return ["bash", "-o", "pipefail", runner, str(gpu), exp]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--expect-plan-sha256", default=None,
                        help=("the digest the campaign was started against; "
                              "the chain computes it once and passes it to "
                              "every cell, so the plan cannot change mid-run"))
    parser.add_argument("--repo", default=str(REPO),
                        help="working directory for the child")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        plan = load_plan(Path(args.plan), expect_sha256=args.expect_plan_sha256)
        cell = cell_at(plan, args.index)
        env = child_env(plan, cell)
        cmd = child_argv(plan, cell, args.gpu)
    except PlanRejected as error:
        print(f"[ablation-exec] REFUSED: {error}", file=sys.stderr)
        return 1

    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.dry_run:
        json.dump({"cmd": cmd, "env": env}, sys.stdout, indent=2,
                  sort_keys=True)
        print()
        return 0
    return subprocess.run(cmd, cwd=args.repo, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
