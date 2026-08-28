"""Run one planned ablation cell, with exactly the plan's environment (F09).

The child is `prompt_ablation_A_cell_fixedN.sh`, which reads a dozen variables
from its environment -- and several of them change the experiment rather than
the logging. An inherited `NUM_CODONS=4` turns the paper's 15 bases into 20;
`CURVE=1` drops FINAL_EPOCH and swaps the isolated refit for the test-monitored
path; `K` changes the capacity of every non-A4 cell; `WHITEN_VARIANT` changes
the transform. So the child gets `env -i` plus the plan's allow-list plus the
plan's pinned values, and nothing else.

It also runs the child under `bash -o pipefail`: the wrapper's trainer ends in
`python ... | tee`, and without that the pipeline's status is tee's.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    cell = plan["cells"][args.index]
    allowed = set(plan["env_passthrough"])

    env = {k: v for k, v in os.environ.items() if k in allowed}
    env.update(cell["env"])
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = ["bash", "-o", "pipefail", cell["runner"], str(args.gpu),
           cell["exp"]]
    if args.dry_run:
        json.dump({"cmd": cmd, "env": env}, sys.stdout, indent=2,
                  sort_keys=True)
        print()
        return 0
    return subprocess.run(cmd, cwd=str(REPO), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
